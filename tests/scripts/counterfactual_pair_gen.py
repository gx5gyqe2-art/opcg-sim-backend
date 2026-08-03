"""反実仮想ペア教師の生成（v24・`docs/reports/cpu_v23_value_blind_cause_20260729.md` §4 の本命）。

v23 の結論: value は「勝者局面の相貌」（体が並ぶ・手札が減る・フラグが立つ）を学んでおり、
「この局面でその行動を取るべきか」という**行動の因果**を教える教師がコーパスに無い。
密ラベル（状態相関）の z は交絡を含み（m2@64: 在場 dz=+0.31）、q_root はエコーする
（m4@2: dz=+0.02 なのに dq=+0.15）。

本スクリプトはその欠落を直接埋める: 固定対面の自己対戦から
**「化粧系（PLAY / ATTACH_DON / ACTIVATE_MAIN）と進行系（ATTACK / TURN_END）が同時に合法」**な
決定点を採取し、各行動種の代表手（現ネットの1手先 value 最良＝ネットが誤りがちな側の最良）を
**同一の決定化世界（CRN）**から終局までロールアウトして、**子盤面そのもの**に測定勝率
z = 2·wr − 1 をラベル付けする。同一親からの子同士が同じ世界線で対照されるので、
状態相関でなく**行動の因果差**がラベルに入る。

出力は `dense_finetune.py` 互換シャード（scalars/field/card_idx/value/q_root/turns_left）。
**q_root=NaN で吐く**＝`build_labels` が勝敗単独ラベルへ退化させる（エコー遮断が構造的に入る）。

設計上の選択:
  - ロールアウトは `counterfactual_referee.rollout`（sticky世界線・temp0）を再利用＝v8 以来の
    レフェリーと同じ因果測定機構。プレイヤーは現既定（gen8）＝密コーパスと同じ生成主体
    （v23 実測で z 側は正直＝エコーは q_root 固有なので、gen8 ロールアウトの z は使える）
  - 決定点の採掘は自己対戦を **copy-apply**（`game.apply`）で進めて親盤面を保持（間引きは
    ゲームあたり上限のランダム抽出＝コスト制御）
  - 枝の列挙は**枝刈り無し**（referee と同じ・化粧系の手が serve 枝刈りで消えないように）
  - 閉ループにしない（生成ネット固定・1回きり・判定は外部の coach/arena）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/counterfactual_pair_gen.py \
    --games 96 --workers 4 --worlds 4 --out /tmp/cfpair_v24
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")

COSMETIC = ("PLAY", "ATTACH_DON", "ACTIVATE_MAIN")   # 相貌が即時に良く見える側（v23）
PROGRESS = ("ATTACK", "TURN_END")                     # コスト先行/別レジームで不利に見える側
BRANCH_TYPES = COSMETIC + PROGRESS


def qualifies(types):
    """決定点の採掘条件（pure）: 化粧系と進行系が同時に合法＝因果対照が意味を持つ点。

    TURN_END はメイン決定に常在なので、実質「メイン決定かつ化粧系の選択肢がある」に一致する
    （防御窓 SELECT_COUNTER 等は TURN_END を含まないため自然に除外される）。"""
    ts = set(types)
    return bool(ts & set(COSMETIC)) and bool(ts & set(PROGRESS))


def pick_branches(descs, values):
    """行動種ごとの代表手 index を返す（pure）。代表＝その種で1手先 value 最良
    （＝現ネットが選びがちな側の最良を対照する）。value 無し（適用失敗）は除外。"""
    best = {}
    for k, (d, v) in enumerate(zip(descs, values)):
        at = (d or {}).get("action_type")
        if at not in BRANCH_TYPES or v is None:
            continue
        if at not in best or v > best[at][1]:
            best[at] = (k, v)
    return sorted(k for k, _ in best.values())


def causal_z(wins, worlds):
    """勝ち数 → z ∈ [-1, 1]（pure）。"""
    return 2.0 * wins / max(worlds, 1) - 1.0


def sample_points(turns, k, rng):
    """採掘候補（各点のターン番号列）から上限 k 点をターン分散で抽出（pure・昇順）。

    メイン決定は1ターン内に連続して現れる（1手打つごとに再判定）ため一様抽出は同一ターンに
    固まる。ラウンドロビン＝まず各ターンから1点ずつ（ターン順・ターン内はランダム）、
    足りなければ2点目…と埋める＝カバレッジ優先のコスト制御。"""
    n = len(turns)
    if n <= k:
        return list(range(n))
    by_turn = {}
    for i, t in enumerate(turns):
        by_turn.setdefault(t, []).append(i)
    for idxs in by_turn.values():
        rng.shuffle(idxs)
    out = []
    r = 0
    while len(out) < k:
        added = False
        for t in sorted(by_turn):
            if r < len(by_turn[t]) and len(out) < k:
                out.append(by_turn[t][r]); added = True
        if not added:
            break
        r += 1
    return sorted(out)


# --- worker ----------------------------------------------------------------------
_G = {}


def _init_worker(matchup, decks_json, rollout_sims):
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(sims=rollout_sims, true_board=False)
    db = _load_db()
    eng = LearnedEngine()
    a, b = matchup.split(":")
    _G.update(
        db=db, eng=eng,
        vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
        pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
        game_gen=_make_fixed_matchup_game(decks_json, a, b),   # 対局生成（GEN 枝刈り設定）
        game_root=OPCGGame(prune_futile=False),                # 枝列挙は枝刈り無し（referee と同じ）
        game_serve=OPCGGame(),                                 # ロールアウト（serve 同等）
        CR=CR)


def _decide(game, m, name, sims, eps, rng, world_seed):
    """自己対戦の1手（rollout と同じ sticky 世界線・任意の Dirichlet ノイズ＝局面多様性）。"""
    from az_mcts_tree import TreeMCTS
    key = (int(getattr(m, "turn_count", 0) or 0), name)
    ds = int((world_seed * 1000003 + key[0] * 131 + (0 if name == "p1" else 7)) % (2 ** 63 - 1))
    mcts = TreeMCTS(game, value_fn=_G["vf"], priors_fn=_G["pf"], c_puct=1.5, n_sims=sims,
                    dirichlet_eps=eps,
                    determinize_fn=lambda s, r, _d=ds, _n=name:
                        game.determinize(s, _n, np.random.default_rng(_d)),
                    rng=rng)
    mv, _n, _legal = mcts.run(m)
    return mv


def process_game(task):
    """1局: 自己対戦（copy-apply）→ 採掘 → 各点の反実仮想ファン → 行データを返す。"""
    seed, cfg = task
    CR = _G["CR"]
    import rl_encoder as E
    from opcg_sim.src.core import cpu_ai
    game, groot, gserve = _G["game_gen"], _G["game_root"], _G["game_serve"]
    eng, vf = _G["eng"], _G["vf"]
    rng = np.random.default_rng(seed)
    m = game.new_game(_G["db"], seed)
    snaps = []                                    # (盤面, 手番) — copy-apply なので参照を保持できる
    steps = 0
    while m.winner is None and not gserve.is_terminal(m) and steps < CR.MAX_STEPS:
        name = gserve.current_player(m)
        if name is None:
            break
        try:
            legal = groot.legal_actions(m)
            types = [(cpu_ai._describe_move(m, mv) or {}).get("action_type") for mv in legal]
        except Exception:
            types = []
        if qualifies(types):
            snaps.append((m, name))
        mv = _decide(game, m, name, cfg["gen_sims"], cfg["eps"], rng, world_seed=seed)
        if mv is None:
            break
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps += 1

    picked = sample_points([int(getattr(s, "turn_count", 0) or 0) for s, _ in snaps],
                           cfg["points_per_game"], rng)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root", "turns_left")}
    diag = []
    for pi in picked:
        m0, name = snaps[pi]
        try:
            legal = groot.legal_actions(m0)
        except Exception:
            continue
        descs, vals, childs = [], [], []
        for mv in legal:
            try:
                d = cpu_ai._describe_move(m0, mv) or {}
                c = groot.apply(m0, mv, name)
                v = None if c is None else float(vf(c, name))
            except Exception:
                d, c, v = {}, None, None
            descs.append(d); childs.append(c); vals.append(v)
        branch = pick_branches(descs, vals)
        if len(branch) < 2:
            continue                              # 対照が組めない点は捨てる（因果差が取れない）
        wins = {k: 0 for k in branch}
        ends = {k: [] for k in branch}
        ok_worlds = 0
        for w in range(cfg["worlds"]):
            wseed = seed * 1009 + pi * 101 + w * 97
            try:
                world = gserve.determinize(m0, name, np.random.default_rng(wseed))
            except Exception:
                continue
            ok_worlds += 1
            for k in branch:
                cw = gserve.apply(world, legal[k], name)
                if cw is None:
                    continue
                winner, _ld, et = CR.rollout(gserve, _G["vf"], _G["pf"], cw, name,
                                             world_seed=wseed, rng_seed=wseed * 31 + k)
                if winner == name:
                    wins[k] += 1
                ends[k].append(et)
        if ok_worlds == 0:
            continue
        for k in branch:
            c = childs[k]
            enc = E.encode(c, name, eng.vocab, version=eng.enc_version)
            rows["scalars"].append(enc["scalars"])
            rows["field"].append(enc["field"])
            rows["card_idx"].append(enc["card_idx"])
            rows["value"].append(causal_z(wins[k], ok_worlds))
            rows["q_root"].append(np.nan)          # z 単独ラベル＝エコー遮断（build_labels 契約）
            tl = (np.mean(ends[k]) - float(getattr(c, "turn_count", 0) or 0)) if ends[k] else np.nan
            rows["turns_left"].append(max(0.0, float(tl)) if np.isfinite(tl) else np.nan)
        diag.append({"turn": int(getattr(m0, "turn_count", 0) or 0),
                     "branches": {descs[k].get("action_type"): round(causal_z(wins[k], ok_worlds), 3)
                                  for k in branch}})
    return rows, diag, len(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=96)
    ap.add_argument("--shard-games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=4, help="CRN 世界数（z の分解能=2/worlds）")
    ap.add_argument("--rollout-sims", type=int, default=96, help="ロールアウト decide の sims")
    ap.add_argument("--gen-sims", type=int, default=160, help="採掘用自己対戦の sims（serve 同等）")
    ap.add_argument("--eps", type=float, default=0.15, help="採掘用自己対戦の Dirichlet ノイズ")
    ap.add_argument("--points-per-game", type=int, default=4)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--seed-base", type=int, default=910000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "cfpair_*.npz")))   # 再開時は既存分をスキップ
    cfg = {k: getattr(args, k) for k in ("worlds", "rollout_sims", "gen_sims", "eps",
                                         "points_per_game")}
    cfg["gen_sims"] = args.gen_sims
    print(f"=== 反実仮想ペア生成 matchup={args.matchup} worlds={args.worlds} "
          f"rollout_sims={args.rollout_sims} 既存シャード={done} ===", flush=True)

    t_all = time.time()
    tot_rows = tot_games = tot_points = 0
    contrast = []                                 # 化粧系 − 進行系 の z 差（生成中の一次診断）
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.matchup, args.decks_json, args.rollout_sims)) as pool:
        shard = done
        games_done = done * args.shard_games
        while games_done < args.games:
            n = min(args.shard_games, args.games - games_done)
            t0 = time.time()
            tasks = [(args.seed_base + games_done + g, cfg) for g in range(n)]
            outs = pool.map(process_game, tasks)
            parts = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root",
                                     "turns_left")}
            diags, snap_total = [], 0
            for rows, diag, n_snaps in outs:
                for k in parts:
                    parts[k].extend(rows[k])
                diags.extend(diag); snap_total += n_snaps
            nrows = len(parts["value"])
            if nrows:
                arrays = {
                    "scalars": np.stack(parts["scalars"]).astype(np.float32),
                    "field": np.stack(parts["field"]).astype(np.float32),
                    "card_idx": np.stack(parts["card_idx"]).astype(np.int32),
                    "value": np.array(parts["value"], dtype=np.float32),
                    "q_root": np.array(parts["q_root"], dtype=np.float32),
                    "turns_left": np.array(parts["turns_left"], dtype=np.float32),
                    "kind": np.array(["cfpr"] * nrows),
                }
                np.savez_compressed(os.path.join(args.out, f"cfpair_{shard:05d}.npz"), **arrays)
                with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                    json.dump({"source": "counterfactual_pair", "games": n, "rows": nrows,
                               "points": len(diags), "worlds": args.worlds,
                               "rollout_sims": args.rollout_sims, "matchup": args.matchup,
                               "schema_version": 2, "diag": diags}, f, ensure_ascii=False)
            for d in diags:
                b = d["branches"]
                cos = [z for t, z in b.items() if t in COSMETIC]
                prg = [z for t, z in b.items() if t in PROGRESS]
                if cos and prg:
                    contrast.append(max(cos) - max(prg))
            tot_rows += nrows; tot_games += n; tot_points += len(diags)
            games_done += n
            cmean = float(np.mean(contrast)) if contrast else None
            print(f"shard{shard}: {n}局 → {len(diags)}点/{nrows}行（候補{snap_total}） "
                  f"{time.time() - t0:.0f}s  累計 {games_done}局/{tot_points}点/{tot_rows}行"
                  f"  化粧−進行 z差 {cmean if cmean is None else round(cmean, 3)}"
                  f"（{(time.time() - t_all) / 60:.0f}分経過）", flush=True)
            shard += 1
    print(f"CFPAIR_RESULT {json.dumps({'games': tot_games, 'points': tot_points, 'rows': tot_rows, 'out': args.out, 'contrast_mean': None if not contrast else round(float(np.mean(contrast)), 4), 'min': round((time.time() - t_all) / 60, 1)}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
