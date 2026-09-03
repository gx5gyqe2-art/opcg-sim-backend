"""防御窓の反実仮想コーパス生成（①防御応答矯正フェーズ2・2026-07-30）。

フェーズ1（`defense_cf_probe.py`・`docs/reports/defense_cf_phase1_20260730.md`）で
「防御温度つき反実仮想測定は人間判断と band 内 12/13 で整合」を確認したので、同じ測定を
**自己対戦から量産**して value の教師にする。

狙い（v23/m1@3 の機序）: 手札（カウンター）の価値は「将来それが使われる世界」でしか勝敗に
現れない。argmax ロールアウトでは温存の価値が測定から消え、ネットは「手札が減った＝勝者の相貌」
を学ぶ。**def_temp で防御応答をサンプリング**したロールアウトなら温存が結果に現れる。

各防御窓で:
  1. 選択肢（素通し PASS / 各カウンター / 各ブロッカー・card_id で重複排除）を列挙
  2. **同一決定化世界 × 枝間共有のロールアウト乱数**（フェーズ1で発見した CRN 破れの修正後）で
     終局まで打ち、選択肢ごとの勝率 → 因果 z = 2·wr − 1
  3. 各選択肢の**子盤面**に **margin_blend ラベル**（v34・z ＋ 0.25·clip(平均残ライフ差/4)＝
     gen11 採用で実証済みの「勝ち方の質」タイブレーク。防御窓は「守った/守らなかった」が
     勝敗を覆さず残ライフに現れる窓が多く、二値 z だけでは拮抗して順位ペアが立たない）と
     **group ID**（同一窓の子盤面束＝順位学習 `build_rank_pairs` が読む）を付けて出力
     （dense_finetune 互換・q_root=NaN＝勝敗単独ラベル）。

出力は `counterfactual_pair_gen.py`（v24・メイン決定の展開/攻撃）と同じスキーマだが、
**採掘する窓が防御専用**で、あちらが持たない「守る/守らない」の因果対照を作る（1トピック=1ファイル）。
v34 の学習側は `option_pair_finetune`（蒸留アンカー付き順位学習）＝ defcf_*.npz も読める。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/defense_cf_gen.py \
    --games 24 --workers 4 --worlds 8 --out /tmp/defcorpus
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
from defense_cf_probe import dedupe_branches   # 選択肢の同一視は probe と共有（1定義）
from opcg_sim.src.learned.config import BOX_RESOLVE_DEPTH
from opcg_sim.src.learned.mcts import resolve_battle_inplace   # 解決規約は探索と共有（1定義）
from option_pair_gen import margin_blend       # v34: ラベル式は option_pair と共有（1定義）

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
DEF_WINDOWS = ("SELECT_COUNTER", "SELECT_BLOCKER")


def causal_z(wins, worlds):
    """勝ち数 → z ∈ [-1, 1]（pure）。v24 と同一規約。"""
    return 2.0 * wins / max(worlds, 1) - 1.0


def spread(zs):
    """選択肢間の z 幅（pure）。0＝どの防御選択も同じ＝教師として情報が無い窓。"""
    return (max(zs) - min(zs)) if zs else 0.0


def pick_windows(turns, k, rng):
    """採掘候補（各窓の**層キー**列）から上限 k 窓を層分散で抽出（pure・昇順）。

    防御窓は攻撃連打の1ターンに固まるため、一様抽出だと同一ターンばかりになる
    （v24 で同型の偏りを踏んだ）。各層から1窓ずつのラウンドロビンで埋める。

    **層キーは v39 で「ターン番号」から「守る側の残ライフ」へ変更**（2026-08-07）。
    ターン分散だと序盤から順に埋まって k 窓で打ち切られ、**低ライフ帯（リーサル圏）の窓が
    ほぼ採れていなかった**（実測 69群: ライフ5=12/4=33/3=21/**2=3**/1=0）。守るか通すかの
    交換レートは残ライフで符号が変わる（序盤は素通しが得・終盤は守るのが必須）ので、
    低ライフ帯が欠けた教師で学習すると「守るな」に倒れる危険がある。完走した対局には必ず
    低ライフ局面が含まれる＝データは既に在り、**選び方だけで捨てていた**（生成コストは不変）。"""
    n = len(turns)
    if n <= k:
        return list(range(n))
    by_turn = {}
    for i, t in enumerate(turns):
        by_turn.setdefault(t, []).append(i)   # t＝層キー（v39: 守る側の残ライフ）
    for idxs in by_turn.values():
        rng.shuffle(idxs)
    out, r = [], 0
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


def _init_worker(matchup, decks_json, rollout_sims, enc_version):
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(sims=rollout_sims, true_board=False)
    eng = LearnedEngine()
    if matchup == "random":
        # ランダムリーダー×生成デッキ（N2・2026-08-25）: 教師も実デッキを使わない
        # （CLAUDE.md「開発判断の前提」＝実デッキは完全ホールドアウトの検証帯）。
        # g15_gen --matchup random / promotion_gate._leader_pair と同規約。
        from opcg_game import OPCGGame as _OG
        from opcg_sim.src.learned.config import GEN_PRUNE_FUTILE as _GPF

        class _RandomGame(_OG):
            def new_game(self, db, seed, leaders=None):
                import random as _r
                from deck_synth import synth_deck
                from opcg_sim.src.core.gamestate import GameManager, Player
                _r.seed(seed)
                pool = getattr(self, "_pool", None)
                if pool is None:
                    pool = sorted(cid for cid, _ in db.raw_db.items()
                                  if (db.get_card(cid) is not None
                                      and getattr(db.get_card(cid).type, "name", "")
                                      == "LEADER"))
                    self._pool = pool
                rl = _r.Random(seed * 7919 + 13)
                la, lb = rl.choice(pool), rl.choice(pool)
                l1, c1 = synth_deck(db, la, seed=seed, owner="p1")
                l2, c2 = synth_deck(db, lb, seed=seed + 1, owner="p2")
                m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
                m.start_game()
                return m

        _G.update(db=_load_db(), CR=CR, eng=eng, enc_version=enc_version,
                  vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
                  pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
                  game_gen=_RandomGame(prune_futile=_GPF),
                  gserve=OPCGGame())
        return
    a, b = matchup.split(":")
    # ラベル（勝敗）は出荷ネットのロールアウトで測る一方、**行の符号化は enc_version**
    # （既定 v6＝手札資源集約つき）。ネットが v5 でも盤面の記述として新特徴を教師行に載せる
    # ＝学習側が温スタート拡張して新特徴を使える（v5 のまま出すと新5列が全ゼロで学習不能）。
    _G.update(db=_load_db(), CR=CR, eng=eng, enc_version=enc_version,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
              game_gen=_make_fixed_matchup_game(decks_json, a, b),
              gserve=OPCGGame())


def _decide(game, m, name, sims, eps, rng, world_seed):
    """自己対戦の1手（rollout と同じ sticky 世界線・Dirichlet ノイズで局面多様化）。"""
    from az_mcts_tree import TreeMCTS
    key = (int(getattr(m, "turn_count", 0) or 0), name)
    ds = int((world_seed * 1000003 + key[0] * 131 + (0 if name == "p1" else 7)) % (2 ** 63 - 1))
    mcts = TreeMCTS(game, value_fn=_G["vf"], priors_fn=_G["pf"], c_puct=1.5, n_sims=sims,
                    dirichlet_eps=eps,
                    determinize_fn=lambda s, r, _n=name:
                        game.determinize(s, _n, np.random.default_rng(ds)),
                    rng=rng)
    mv, _n, _legal = mcts.run(m)
    return mv


def process_game(task):
    """1局: 自己対戦（copy-apply で盤面保持）→ 防御窓採掘 → 反実仮想測定 → 行データ。"""
    seed, cfg, gbase = task
    CR = _G["CR"]
    import rl_encoder as E
    from opcg_sim.src.core import cpu_ai
    game, gserve, eng = _G["game_gen"], _G["gserve"], _G["eng"]
    rng = np.random.default_rng(seed)
    m = game.new_game(_G["db"], seed)
    snaps, steps = [], 0
    while m.winner is None and not gserve.is_terminal(m) and steps < CR.MAX_STEPS:
        pa = m.pending_actor_action()
        if pa and pa[1] in DEF_WINDOWS:
            snaps.append((m, pa[0]))               # 防御窓＝採掘候補
        name = gserve.current_player(m)
        if name is None:
            break
        mv = _decide(game, m, name, cfg["gen_sims"], cfg["eps"], rng, world_seed=seed)
        if mv is None:
            break
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps += 1

    # 層キー＝守る側の残ライフ（v39）。ターン番号だと序盤に偏り低ライフ帯が採れない。
    picked = pick_windows([len((s.p1 if s.p1.name == who else s.p2).life) for s, who in snaps],
                          cfg["windows_per_game"], rng)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root", "turns_left",
                            "group")}
    diag = []
    for gi, pi in enumerate(picked):
        m0, name = snaps[pi]
        actor = m0.p1 if m0.p1.name == name else m0.p2
        legal = m0.get_legal_actions(actor) or []
        descs = []
        for mv in legal:
            try:
                descs.append(cpu_ai._describe_move(m0, mv) or {})
            except Exception:
                descs.append({})
        branches = dedupe_branches(descs)
        if len(branches) < 2:
            continue                               # 選択肢1つ＝対照が組めない
        wins = {k: 0 for k, _ in branches}
        ends = {k: [] for k, _ in branches}
        lds = {k: [] for k, _ in branches}          # v34: mover 視点の残ライフ差（勝ち方の質）
        childs = {}
        ok_worlds = 0
        for w in range(cfg["worlds"]):
            wseed = seed * 1009 + pi * 101 + w * 97
            try:
                world = gserve.determinize(m0, name, np.random.default_rng(wseed))
            except Exception:
                continue
            ok_worlds += 1
            for key, i in branches:
                cw = gserve.apply(world, legal[i], name)
                if cw is None:
                    continue
                # rng_seed は枝に依存させない（フェーズ1で修正した CRN 規約）
                winner, ld, et = CR.rollout(gserve, _G["vf"], _G["pf"], cw, name,
                                            world_seed=wseed, rng_seed=wseed * 131,
                                            def_temp=cfg["def_temp"])
                if winner == name:
                    wins[key] += 1
                ends[key].append(et)
                lds[key].append(ld)
        if ok_worlds == 0:
            continue
        group_id = gbase + gi                       # グローバル一意（呼び出し側で gbase を割当）
        zs = []
        for key, i in branches:
            child = gserve.apply(m0, legal[i], name)   # ラベル対象は実盤面の子（世界は決定化前）
            if child is None:
                continue
            # **戦闘を解決してから符号化する**（v35・train/serve skew の解消）。
            # 戦闘途中の子は「1000 を切った子」と「2000 を切った子」が手札-1・ライフ不変で
            # **ほぼ同一の入力**になり、そこへ異なるラベル（z=-0.875 と -0.562）を付けていた
            # ＝学習不可能な教師だった（2026-08-04 実測。v34 で教えても動かず、強く押すと
            # 他点が壊れた原因の疑い）。探索側（静止探索）と**同一の解決関数**を使い、
            # ネットが実際に見る盤面＝解決時の手札と盤面にラベルを付ける。
            # v39: 解決規約も serve と揃える（value_fn＋BOX_RESOLVE_DEPTH）。深さ0のままだと
            # 「policy 任せで途中終了した出口」＝**実対局では到達しない盤面**を教えることになる
            # （深さ1の serve は最善 continuation まで読み切って別の出口へ至る）。
            resolve_battle_inplace(gserve, child, _G["pf"],
                                   value_fn=_G["vf"], box_depth=BOX_RESOLVE_DEPTH)
            childs[key] = child
            z = margin_blend(causal_z(wins[key], ok_worlds),
                             float(np.mean(lds[key])) if lds[key] else None)
            zs.append(z)
            enc = E.encode(child, name, eng.vocab, version=_G["enc_version"])
            rows["scalars"].append(enc["scalars"])
            rows["field"].append(enc["field"])
            rows["card_idx"].append(enc["card_idx"])
            rows["value"].append(z)
            rows["q_root"].append(np.nan)          # 勝敗単独ラベル（エコー遮断）
            tl = (np.mean(ends[key]) - float(getattr(child, "turn_count", 0) or 0)) if ends[key] else np.nan
            rows["turns_left"].append(max(0.0, float(tl)) if np.isfinite(tl) else np.nan)
            rows["group"].append(group_id)
        if zs:
            diag.append({"turn": int(getattr(m0, "turn_count", 0) or 0),
                         "window": m0.pending_actor_action()[1],
                         "spread": round(spread(zs), 3),
                         "branches": {f"{k[0]}:{k[1] or ''}": round(causal_z(wins[k], ok_worlds), 3)
                                      for k, _ in branches if k in childs}})
    return rows, diag, len(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--shard-games", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=8, help="CRN 世界数（z 分解能=2/worlds）")
    ap.add_argument("--rollout-sims", type=int, default=48)
    ap.add_argument("--gen-sims", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--def-temp", type=float, default=0.7,
                    help="ロールアウト内の防御応答サンプリング温度（0=argmax＝旧来の盲点）")
    # 1局あたりの採掘窓数。コスト構造（v39 実測・worlds4/rollout-sims24）:
    #   1局の生成（自己対戦）≈3分は局ごとに1回だけ／1窓のラベル付け ≈2分（枝×世界×ロールアウト）
    #   → 群あたり = 生成3分/窓数 + 2分。6窓なら 2.5分、12窓なら 2.25分（**約1割の短縮**）。
    # 窓数を増やすほど局生成が償却される一方、同一対局由来の窓は盤面が相関する（多様性は
    # 件数ほど増えない）。既定 6 は v34 期のヒューリスティックで根拠の記録が無い。
    # v39 の再生成では 12 を明示指定した（残ライフ層で採るため、帯あたり2窓以上を確保する
    # 狙いも兼ねる。ユーザ判断 2026-08-07＝時間短縮を優先）。
    ap.add_argument("--windows-per-game", type=int, default=6)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--enc-version", type=int, default=8,
                    help="教師行の符号化版（既定 8＝gen11 の現行版。ロールアウトは出荷ネット）")
    ap.add_argument("--seed-base", type=int, default=920000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "defcf_*.npz")))
    cfg = {k: getattr(args, k) for k in ("worlds", "rollout_sims", "gen_sims", "eps",
                                         "def_temp", "windows_per_game")}
    print(f"=== 防御窓CFコーパス生成 matchup={args.matchup} worlds={args.worlds} "
          f"def_temp={args.def_temp} ev={args.enc_version} 既存シャード={done} ===", flush=True)

    t_all = time.time()
    tot_rows = tot_win = tot_games = 0
    spreads = []
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.matchup, args.decks_json, args.rollout_sims,
                           args.enc_version)) as pool:
        shard = done
        games_done = done * args.shard_games
        while games_done < args.games:
            n = min(args.shard_games, args.games - games_done)
            t0 = time.time()
            # group の gbase は **seed_base 込み**で割り当てる（2026-08-04 実害: 別ランの
            # コーパスを --dirs で連結すると group ID が衝突し、**無関係な窓の子盤面同士が
            # 順位ペアにされる**＝教師が黙って壊れる。実測 119/121 群が衝突していた）。
            outs = pool.map(process_game, [(args.seed_base + games_done + g, cfg,
                                            (args.seed_base + games_done + g) * 100)
                                           for g in range(n)])
            parts = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root",
                                     "turns_left", "group")}
            diags, cand = [], 0
            for rows, diag, n_snaps in outs:
                for k in parts:
                    parts[k].extend(rows[k])
                diags.extend(diag); cand += n_snaps
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
                    "kind": np.array(["defcf"] * nrows),
                }
                np.savez_compressed(os.path.join(args.out, f"defcf_{shard:05d}.npz"), **arrays)
                with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                    json.dump({"source": "defense_cf", "games": n, "rows": nrows,
                               "windows": len(diags), "worlds": args.worlds,
                               "def_temp": args.def_temp, "rollout_sims": args.rollout_sims,
                               "matchup": args.matchup, "enc_version": args.enc_version,
                               "schema_version": 2, "diag": diags}, f,
                              ensure_ascii=False)
            spreads += [d["spread"] for d in diags]
            tot_rows += nrows; tot_win += len(diags); tot_games += n
            games_done += n
            sp = float(np.mean(spreads)) if spreads else 0.0
            info = float(np.mean([s > 0 for s in spreads])) if spreads else 0.0
            print(f"shard{shard}: {n}局 → {len(diags)}窓/{nrows}行（候補{cand}） "
                  f"{time.time() - t0:.0f}s  累計 {games_done}局/{tot_win}窓/{tot_rows}行 "
                  f"z幅平均 {sp:.3f}・有情報率 {info:.2f}"
                  f"（{(time.time() - t_all) / 60:.0f}分経過）", flush=True)
            shard += 1
    print(f"DEFENSE_CF_GEN_RESULT {json.dumps({'games': tot_games, 'windows': tot_win, 'rows': tot_rows, 'out': args.out, 'spread_mean': round(float(np.mean(spreads)), 4) if spreads else None, 'min': round((time.time() - t_all) / 60, 1)}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
