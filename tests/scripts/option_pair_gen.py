"""登場時オプションのカード単位ペア教師の生成（v31・`docs/reports/cpu_v30_option_feature_20260802.md` §4-B）。

v30 の結論: v7 特徴（登場時オプションの実測）は m4@2 型の value を初めて正しい方向へ動かした
（子盤面差 -0.140→-0.106）が、**value 回帰では符号反転に届かない**（拮抗する2子の勾配が弱い）。
v26 原因分析 §3 が予告した通り、順位を確定するには**条件付き対比の順位損失**（v12.1 の
`build_rank_pairs`/`rank_finetune`）が要る。

**v24 のペア生成器（`counterfactual_pair_gen.py`）ではダメな理由**: あれは**行動種**（PLAY/ATTACK…）で
分岐し種ごとに代表1手へ潰す。m4@2 は「イワンコフ(PLAY) vs A&S&L(PLAY)」＝**同じ PLAY 種の別カード**
なので対照が作れない。本スクリプトは**カード単位**で PLAY 枝を列挙し、さらに「今出さず温存」枝
（TURN_END 相当＝現局面そのもの）も対照に含める＝「どのカードを今出すか／出さないか」を教える。

出力は v12.1 の順位学習が読む形式: scalars/field/card_idx/**value(=因果z)**/**group**（同一決定点の
子が共有）/q_root=NaN/turns_left。`build_rank_pairs(child)` が同一 group 内の z 差 > δ を順位ペアに
する。符号化は出荷ネットの版（gen10=v7）＝m4@2 の差を表せる表現の上で順位を教える。

3つが初めて揃う: v24 のデータ（局面ごとの因果対照・カード単位に拡張）× v12.1 の順位ヒンジ
（回帰でなく v(勝)>v(負) を直接強制）× v7 の表現（gen10 で差を表せることは実証済み）。

**閉ループにしない**（生成ネット固定・判定は外部の coach/arena）。マーク局面シード
（`--mark-frac`）で m4@2 型の点を分布へ直接注入する。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/option_pair_gen.py \
    --matchup nami:shanks --mark-frac 0.3 --games 128 --workers 4 --worlds 6 --out /tmp/optpair
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

_G = {}


def causal_z(wins, worlds):
    """測定勝率 → 因果 z = 2·wr − 1 ∈ [-1,1]（pure・v24 と同一規約）。worlds=0 は 0。"""
    return (2.0 * wins / worlds - 1.0) if worlds > 0 else 0.0


def spread(zs):
    """子 z の最大差（pure）: 0 なら「どのカードを出しても結果が同じ」＝順位教師として無情報。"""
    return (max(zs) - min(zs)) if zs else 0.0


def option_branches(descs):
    """決定点の合法手記述子列 → 対照する枝の index 列（pure・**カード単位**）。

    - PLAY 枝は card_id ごとに1つ（同名複製は代表1つ＝順位教師は札の種類を区別すれば十分）
    - 「今出さず温存」＝ TURN_END（現局面へ進める枝）を1つ含める＝「出す vs 温存」を対照
    m4@2 は {PLAY イワンコフ, PLAY A&S&L, …, TURN_END} が枝になる。2枝未満は呼び出し側で捨てる。
    """
    out, seen_play = [], set()
    end_idx = None
    for i, d in enumerate(descs):
        at = (d or {}).get("action_type")
        if at == "PLAY":
            cid = d.get("card")
            if cid is not None and cid not in seen_play:
                seen_play.add(cid)
                out.append(i)
        elif at == "TURN_END" and end_idx is None:
            end_idx = i
    if end_idx is not None:
        out.append(end_idx)
    return out


def qualifies(descs):
    """採掘条件（pure）: **ON_PLAY 持ちの PLAY が2枚以上**合法＝『どの登場時カードを今出すか』の点。

    m4@2 パターンそのもの。1枚だけなら「出す vs 温存」しか対照できず順位の情報が薄いので
    2枚以上に絞る（TURN_END は別途 option_branches が足す）。"""
    play_cards = {(d or {}).get("card") for d in descs
                  if (d or {}).get("action_type") == "PLAY"
                  and (d or {}).get("onplay")}
    play_cards.discard(None)
    return len(play_cards) >= 2


def sample_points(turns, k, rng):
    """採掘点をターン分散で k 個抽出（pure・v24/defense_cf と同流儀）。"""
    if len(turns) <= k:
        return list(range(len(turns)))
    import collections
    by = collections.defaultdict(list)
    for i, t in enumerate(turns):
        by[t].append(i)
    picked, keys = [], sorted(by)
    ri = 0
    while len(picked) < k:
        added = False
        for t in keys:
            if by[t]:
                picked.append(by[t].pop(int(rng.integers(len(by[t]))) if len(by[t]) > 1 else 0))
                added = True
                if len(picked) >= k:
                    break
        if not added:
            break
    return sorted(picked)


def _describe_onplay(m, mv, cpu_ai):
    """記述子に onplay フラグを足す（そのカードが ON_PLAY 能力を持つか）。"""
    d = cpu_ai._describe_move(m, mv) or {}
    uuid = (mv.get("payload") or {}).get("uuid") or mv.get("card_uuid")
    ci = cpu_ai._find_card(m, uuid) if uuid else None
    abil = (getattr(getattr(ci, "master", None), "abilities", None) or ()) if ci else ()
    d["onplay"] = any(getattr(getattr(ab, "trigger", None), "name", "") == "ON_PLAY" for ab in abil)
    return d


def _init_worker(matchup, decks_json, rollout_sims, mark_frac):
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(sims=rollout_sims, true_board=False)
    db = _load_db()
    eng = LearnedEngine()                          # 出荷既定（現 gen10=v7）
    a, b = matchup.split(":")
    seed_boards = None
    if mark_frac > 0.0:
        from mark_seeds import load_mark_boards
        seed_boards = load_mark_boards(db)
    _G.update(
        db=db, eng=eng, enc_version=eng.enc_version,
        vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
        pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
        game_gen=_make_fixed_matchup_game(decks_json, a, b),
        game_root=OPCGGame(prune_futile=False),    # 枝列挙は枝刈り無し（referee と同じ）
        game_serve=OPCGGame(),
        CR=CR, seed_boards=seed_boards, mark_frac=mark_frac)


def _decide(m, name, sims, eps, rng, world_seed):
    from az_mcts_tree import TreeMCTS
    game = _G["game_gen"]
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
    """1局: 自己対戦で採掘 → 各点でカード単位ファン → CRN ロールアウトで因果z → 子行を返す。

    group は (shard_seed, 点index) から**グローバル一意**に作る（呼び出し側の shard で衝突しない）。
    """
    import rl_encoder as E
    from opcg_sim.src.core import cpu_ai
    seed, cfg, gbase = task
    CR = _G["CR"]
    groot, gserve, game = _G["game_root"], _G["game_serve"], _G["game_gen"]
    ev = _G["enc_version"]
    rng = np.random.default_rng(seed)

    # マーク局面シード: 確率 mark_frac でマーク盤面のクローンから開始（m4@2 型を注入）。
    sb = _G.get("seed_boards")
    if sb and _G["mark_frac"] > 0.0 and float(rng.random()) < _G["mark_frac"]:
        m = sb[int(rng.integers(len(sb)))].clone()
    else:
        m = game.new_game(_G["db"], seed)

    snaps = []
    steps = 0
    max_mine = cfg.get("max_mine_steps", CR.MAX_STEPS)
    while m.winner is None and not gserve.is_terminal(m) and steps < max_mine:
        name = gserve.current_player(m)
        if name is None:
            break
        try:
            legal = groot.legal_actions(m)
            descs = [_describe_onplay(m, mv, cpu_ai) for mv in legal]
        except Exception:
            descs = []
        if descs and qualifies(descs):
            snaps.append((m, name))
        mv = _decide(m, name, cfg["gen_sims"], cfg["eps"], rng, world_seed=seed)
        if mv is None:
            break
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps += 1

    picked = sample_points([int(getattr(s, "turn_count", 0) or 0) for s, _ in snaps],
                           cfg["points_per_game"], rng)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root", "turns_left", "group")}
    diag = []
    for gi, pi in enumerate(picked):
        m0, name = snaps[pi]
        try:
            legal = groot.legal_actions(m0)
            descs = [_describe_onplay(m0, mv, cpu_ai) for mv in legal]
        except Exception:
            continue
        branch = option_branches(descs)
        if len(branch) < 2:
            continue
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
        group_id = gbase + gi                       # グローバル一意（呼び出し側で gbase を割当）
        zs = []
        for k in branch:
            c = gserve.apply(m0, legal[k], name)    # ラベル対象の子（実世界＝手札公開のまま）
            if c is None:
                continue
            enc = E.encode(c, name, _G["eng"].vocab, version=ev)
            z = causal_z(wins[k], ok_worlds)
            zs.append(z)
            rows["scalars"].append(enc["scalars"])
            rows["field"].append(enc["field"])
            rows["card_idx"].append(enc["card_idx"])
            rows["value"].append(z)
            rows["q_root"].append(np.nan)
            tl = (np.mean(ends[k]) - float(getattr(c, "turn_count", 0) or 0)) if ends[k] else np.nan
            rows["turns_left"].append(max(0.0, float(tl)) if np.isfinite(tl) else np.nan)
            rows["group"].append(group_id)
        diag.append({"turn": int(getattr(m0, "turn_count", 0) or 0),
                     "n_branch": len(branch), "spread": round(spread(zs), 3),
                     "cards": [descs[k].get("card") or "TURN_END" for k in branch]})
    return rows, diag, len(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--shard-games", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=6, help="CRN 世界数（z の分解能=2/worlds）")
    ap.add_argument("--rollout-sims", type=int, default=48)
    ap.add_argument("--gen-sims", type=int, default=160, help="採掘用自己対戦の sims")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--points-per-game", type=int, default=4)
    ap.add_argument("--max-mine-steps", type=int, default=24,
                    help="採掘自己対戦の打ち切りステップ（狙う点は序盤＝turn1-4 なので全局回さない）")
    ap.add_argument("--mark-frac", type=float, default=0.0, help="マーク局面から開始する比率")
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--seed-base", type=int, default=970000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "optpair_*.npz")))
    cfg = {k: getattr(args, k) for k in ("worlds", "rollout_sims", "gen_sims", "eps",
                                         "points_per_game", "max_mine_steps")}
    print(f"=== オプションペア生成 matchup={args.matchup} worlds={args.worlds} "
          f"mark_frac={args.mark_frac} 既存シャード={done} ===", flush=True)

    t_all = time.time()
    tot_rows = tot_games = tot_pts = tot_info = 0
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.matchup, args.decks_json, args.rollout_sims, args.mark_frac)) as pool:
        shard = done
        games_done = done * args.shard_games
        while games_done < args.games:
            n = min(args.shard_games, args.games - games_done)
            # group の gbase は (shard, game) から一意に割当＝再開しても衝突しない。
            tasks = [(args.seed_base + games_done + g, cfg, (games_done + g) * 100)
                     for g in range(n)]
            outs = pool.map(process_game, tasks)
            parts = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root",
                                     "turns_left", "group")}
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
                    "group": np.array(parts["group"], dtype=np.int64),
                    "kind": np.array(["optpr"] * nrows),
                }
                np.savez_compressed(os.path.join(args.out, f"optpair_{shard:05d}.npz"), **arrays)
                info = sum(1 for d in diags if d["spread"] > 0)
                with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                    json.dump({"source": "option_pair", "games": n, "rows": nrows,
                               "points": len(diags), "informative": info, "worlds": args.worlds,
                               "mark_frac": args.mark_frac, "matchup": args.matchup,
                               "enc_version": _G.get("enc_version"), "schema_version": 2,
                               "diag": diags[:20]}, f, ensure_ascii=False)
                tot_info += info
            tot_rows += nrows
            tot_pts += len(diags)
            tot_games += n
            games_done += n
            print(f"shard{shard}: {n}局 → {nrows}行 / {len(diags)}点（有情報 "
                  f"{sum(1 for d in diags if d['spread'] > 0)}）"
                  f" 累計 {tot_games}局/{tot_pts}点（{(time.time() - t_all) / 60:.0f}分）", flush=True)
            shard += 1
    print(f"OPTPAIR_GEN_RESULT {json.dumps({'games': tot_games, 'rows': tot_rows, 'points': tot_pts, 'informative': tot_info, 'out': args.out})}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
