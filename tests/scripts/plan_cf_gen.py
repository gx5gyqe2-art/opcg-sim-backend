"""ターン出口教師の反実仮想コーパス生成（v38・2026-08-06・`defense_cf_gen` の兄弟）。

**なぜ要るか（v37 の負の結果から）**: 箱の階層を1つ上げるたびに、その箱の出口盤面で value を
較正し直す必要がある。戦闘箱に「解決後盤面の教師」（v35 `defense_cf_gen`）が要ったのと同型で、
**ターン箱には「ターン末盤面の教師」が要る**。v37② の実測: ターン末 value は不発イワンコフ線を
一貫して高評価（−0.138 vs ウタ線 −0.178）＝分散でなく**バイアス**で、K世界平均では消えない。
gen11 で教えた矯正は決定点近傍の盤面に住んでおり、ターン末の符号化までは転移していなかった。

**自己対戦の相関データでは学べない**: 選ばれなかったプランのターン末盤面は存在しない。同じ
ターン開始局面から**両方のプランを強制実行**して両方のターン末を作り、レフェリーで測って初めて
「実際にはこちらの方が勝つ」という反実仮想の比較がデータになる（v24/v34 と同じ因果対照の設計）。

各ターン開始局面（自ターン・非戦闘・最初のメイン判断）で:
  1. 候補プランを列挙（`plan.rollout_plan`＝policy 温度サンプル・1本目 argmax・重複除去）
  2. **同一決定化世界 × 枝間共有のロールアウト乱数**（CRN）でプランを箱実行→ターン末→終局まで
     打ち、プランごとの勝率 → 因果 z = 2·wr − 1（＋ margin_blend で勝ち方の質をタイブレーク）
  3. 各プランの**ターン末盤面**（実盤面上で実行したもの）に z と group ID を付けて出力

実行規約は serve のプラン読み出しと**同一関数**（`plan.execute_plan`）＝ train/serve skew の予防。
出力スキーマは defcf/optpair と同一（`option_pair_finetune` がそのまま読む）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/plan_cf_gen.py \
    --games 24 --workers 4 --worlds 8 --out /tmp/plancf
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
import defense_cf_gen as DG                                        # 規約は防御CFと共有（1定義）
from defense_cf_gen import causal_z, spread, _decide
from option_pair_gen import margin_blend                            # ラベル式も共有（1定義）
from opcg_sim.src.learned import plan as PL                         # 実行規約は serve と共有

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")

# ワーカー状態は防御CFと**同一の dict オブジェクト**を共有する。`_decide`（自己対戦の1手）を
# あちらから import しており、それは defense_cf_gen._G を読むため（別 dict にすると空を引く）。
_G = DG._G


def _init_worker(matchup, decks_json, rollout_sims, enc_version):
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(sims=rollout_sims, true_board=False)
    eng = LearnedEngine()
    a, b = matchup.split(":")
    _G.update(db=_load_db(), CR=CR, eng=eng, enc_version=enc_version,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
              game_gen=_make_fixed_matchup_game(decks_json, a, b),
              gserve=OPCGGame())


def pick_turn_starts(n, k, rng):
    """候補ターン開始（各1件）から k 件を**一様抽出**して昇順で返す（pure・rng 消費）。

    防御CFの `pick_windows`（ターンごとのラウンドロビン）は使えない: 防御窓は攻撃連打の
    特定ターンに固まるため round-robin が正しいが、**ターン開始は毎ターン1件ずつ存在する**ので
    `sorted(by_turn)` 順に埋めると常に最小ターンから k 件になる（スモーク実測: turn 0,0,1 ばかり）。
    プランの本質的な判断（ドン10枚の配分・展開と攻撃の両立）は中盤以降に現れるため、
    ターン全域から一様に採る。"""
    if n <= k:
        return list(range(n))
    return sorted(int(i) for i in rng.choice(n, size=k, replace=False))


def is_turn_start(mgr, game, name):
    """`name` の自ターンの**最初のメイン判断**か（pure 判定・非戦闘・手番一致）。

    ドン配分も展開も攻撃もまだ何も起きていない時点＝プラン全体が意味を持つ唯一の決定点。
    「そのターンで初めて見た自分のメイン局面」かどうかは呼び出し側が (turn, name) で管理する。"""
    from opcg_sim.src.learned.mcts import in_battle, _turn_owner
    return (not in_battle(mgr) and _turn_owner(mgr) == name
            and game.current_player(mgr) == name)


def propose_plans(game, mgr, name, rng, n_proposals, n_worlds):
    """候補プランを列挙（rng 消費）。

    **先頭手ごとに1本**（＋policy 最良の継続）で構造的に多様性を確保する。温度サンプリングだけ
    だと policy が尖った局面で全提案が argmax に潰れ、対照が組めず窓が捨てられる（スモーク実測:
    27候補中1窓しか行にならなかった）。手動プローブ（4配置プラン×32世界）が「最後の1ドンの
    配分だけ」の対照を作れたのと同じ構造＝先頭の資源配分で枝を張り、以降は現行方策に委ねる。
    先頭手は (action_type, card) で重複排除（`dedupe_branches` と同じ同一視）。"""
    from defense_cf_probe import dedupe_branches
    from opcg_sim.src.core import cpu_ai
    worlds = []
    for _ in range(max(1, min(n_proposals, n_worlds))):
        try:
            worlds.append(game.determinize(mgr, name, rng))
        except Exception:
            break
    if not worlds:
        return []
    plans = []
    base = PL.rollout_plan(game, worlds[0], name, _G["vf"], _G["pf"], rng, temp=0.0)
    if base:
        plans.append(base)
    w0 = worlds[0]
    legal = game.legal_actions(w0)
    descs = []
    for mv in legal:
        try:
            descs.append(cpu_ai._describe_move(w0, mv) or {})
        except Exception:
            descs.append({})
    for key, i in dedupe_branches(descs):
        if len(plans) >= n_proposals:
            break
        if (key[0] or "") == "TURN_END":
            continue                       # 「何もしない」は plans 空＝TURN_END で自然に表現
        nxt = game.apply(w0, legal[i], name)
        if nxt is None:
            continue
        tail = PL.rollout_plan(game, nxt, name, _G["vf"], _G["pf"], rng, temp=0.0)
        steps = (PL.move_sig(legal[i]),) + tuple(tail)
        if steps not in plans:
            plans.append(steps)
    return plans


def process_game(task):
    """1局: 自己対戦 → ターン開始局面を採掘 → プラン反実仮想測定 → 行データ。"""
    seed, cfg, gbase = task
    CR = _G["CR"]
    import rl_encoder as E
    game, gserve, eng = _G["game_gen"], _G["gserve"], _G["eng"]
    rng = np.random.default_rng(seed)
    m = game.new_game(_G["db"], seed)
    snaps, seen_turns, steps_n = [], set(), 0
    while m.winner is None and not gserve.is_terminal(m) and steps_n < CR.MAX_STEPS:
        name = gserve.current_player(m)
        if name is None:
            break
        key = (int(getattr(m, "turn_count", 0) or 0), name)
        if key not in seen_turns and is_turn_start(m, gserve, name):
            seen_turns.add(key)
            snaps.append((m, name))               # ターン開始＝採掘候補
        mv = _decide(game, m, name, cfg["gen_sims"], cfg["eps"], rng, world_seed=seed)
        if mv is None:
            break
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps_n += 1

    picked = pick_turn_starts(len(snaps), cfg["windows_per_game"], rng)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root", "turns_left",
                            "group", "value_a", "value_b")}
    diag = []
    for gi, pi in enumerate(picked):
        m0, name = snaps[pi]
        plans = propose_plans(gserve, m0, name, rng, cfg["proposals"], cfg["worlds"])
        if len(plans) < 2:
            continue                              # 候補1つ＝対照が組めない
        wins = {j: 0 for j in range(len(plans))}
        ends = {j: [] for j in range(len(plans))}
        lds = {j: [] for j in range(len(plans))}
        ok_worlds = 0
        # 分割半ラベル（v44・`--split-halves`）: 世界を偶奇で2組に割り、**同じ決定点の
        # ラベルを独立2回引いた**ものを別列（value_a/value_b）で出す。両者の順位が一致
        # するかが「そのラベルは信号か引きの当たり外れか」の直接の測定になる（worlds を
        # 増やす価値を、本生成に着手する前に安価に判定するため）。列は学習側の
        # `load_pairs_corpus` が読まないので、既存の学習経路には一切影響しない。
        wins_h = {j: [0, 0] for j in range(len(plans))}
        lds_h = {j: [[], []] for j in range(len(plans))}
        ok_h = [0, 0]
        for w in range(cfg["worlds"]):
            wseed = seed * 1009 + pi * 101 + w * 97
            try:
                world = gserve.determinize(m0, name, np.random.default_rng(wseed))
            except Exception:
                continue
            ok_worlds += 1
            h = w % 2                             # 分割半（偶奇で独立2組へ振り分け）
            ok_h[h] += 1
            for j, plan_steps in enumerate(plans):
                end = PL.execute_plan(gserve, world, name, list(plan_steps), _G["vf"], _G["pf"])
                if end is None:
                    continue
                if gserve.is_terminal(end):       # ターン内で決着＝ロールアウト不要
                    won = (getattr(end, "winner", None) == name)
                    wins[j] += 1 if won else 0
                    wins_h[j][h] += 1 if won else 0
                    ends[j].append(float(getattr(end, "turn_count", 0) or 0))
                    lds[j].append(1.0 if won else -1.0)
                    lds_h[j][h].append(1.0 if won else -1.0)
                    continue
                # rng_seed は枝に依存させない（CRN 規約・防御CFと同一）
                winner, ld, et = CR.rollout(gserve, _G["vf"], _G["pf"], end, name,
                                            world_seed=wseed, rng_seed=wseed * 131,
                                            def_temp=cfg["def_temp"])
                if winner == name:
                    wins[j] += 1
                    wins_h[j][h] += 1
                ends[j].append(et)
                lds[j].append(ld)
                lds_h[j][h].append(ld)
        if ok_worlds == 0:
            continue
        group_id = gbase + gi
        zs = []
        for j, plan_steps in enumerate(plans):
            # ラベル対象は**実盤面上でプランを実行したターン末**（世界は決定化前）。
            # 符号化は serve が実際に到達する盤面と同一の実行規約（`plan.execute_plan`）で作る。
            end = PL.execute_plan(gserve, m0, name, list(plan_steps), _G["vf"], _G["pf"])
            if end is None:
                continue
            z = margin_blend(causal_z(wins[j], ok_worlds),
                             float(np.mean(lds[j])) if lds[j] else None)
            zs.append(z)
            enc = E.encode(end, name, eng.vocab, version=_G["enc_version"])
            rows["scalars"].append(enc["scalars"])
            rows["field"].append(enc["field"])
            rows["card_idx"].append(enc["card_idx"])
            rows["value"].append(z)
            for h, col in ((0, "value_a"), (1, "value_b")):
                rows[col].append(margin_blend(causal_z(wins_h[j][h], ok_h[h]),
                                              float(np.mean(lds_h[j][h])) if lds_h[j][h] else None)
                                 if ok_h[h] else np.nan)
            rows["q_root"].append(np.nan)          # 勝敗単独ラベル（エコー遮断）
            tl = (np.mean(ends[j]) - float(getattr(end, "turn_count", 0) or 0)) if ends[j] else np.nan
            rows["turns_left"].append(max(0.0, float(tl)) if np.isfinite(tl) else np.nan)
            rows["group"].append(group_id)
        if zs:
            diag.append({"turn": int(getattr(m0, "turn_count", 0) or 0),
                         "plans": len(plans), "spread": round(spread(zs), 3),
                         "z": [round(z, 3) for z in zs],
                         "len": [len(p) for p in plans]})
    return rows, diag, len(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--shard-games", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=8, help="CRN 世界数（z 分解能=2/worlds）")
    ap.add_argument("--proposals", type=int, default=4, help="候補プラン数（1本目 argmax）")
    ap.add_argument("--rollout-sims", type=int, default=48)
    ap.add_argument("--gen-sims", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--def-temp", type=float, default=0.7)
    ap.add_argument("--windows-per-game", type=int, default=4)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=940000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "plancf_*.npz")))
    cfg = {k: getattr(args, k) for k in ("worlds", "proposals", "rollout_sims", "gen_sims",
                                         "eps", "def_temp", "windows_per_game")}
    print(f"=== ターン出口CFコーパス生成 matchup={args.matchup} worlds={args.worlds} "
          f"proposals={args.proposals} def_temp={args.def_temp} ev={args.enc_version} "
          f"既存シャード={done} ===", flush=True)

    t_all = time.time()
    tot_rows = tot_games = 0
    spreads = []
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.matchup, args.decks_json, args.rollout_sims,
                           args.enc_version)) as pool:
        shard = done
        games_done = done * args.shard_games
        while games_done < args.games:
            n = min(args.shard_games, args.games - games_done)
            t0 = time.time()
            # group の gbase は **seed_base 込み**（別ランのコーパスをマージしても衝突しない）
            outs = pool.map(process_game, [(args.seed_base + games_done + g, cfg,
                                            (args.seed_base + games_done + g) * 100)
                                           for g in range(n)])
            parts = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root",
                                     "turns_left", "group", "value_a", "value_b")}
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
                    "kind": np.array(["plancf"] * nrows),
                    # 分割半ラベル（v44・信頼度測定用。学習側の loader は読まない）
                    "value_a": np.array(parts["value_a"], dtype=np.float32),
                    "value_b": np.array(parts["value_b"], dtype=np.float32),
                }
                np.savez_compressed(os.path.join(args.out, f"plancf_{shard:05d}.npz"), **arrays)
                with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                    json.dump({"source": "plan_cf", "games": n, "rows": nrows,
                               "windows": len(diags), "worlds": args.worlds,
                               "proposals": args.proposals, "def_temp": args.def_temp,
                               "rollout_sims": args.rollout_sims, "matchup": args.matchup,
                               "enc_version": args.enc_version, "schema_version": 1,
                               "diag": diags}, f, ensure_ascii=False)
            spreads += [d["spread"] for d in diags]
            tot_rows += nrows; tot_games += n
            games_done += n; shard += 1
            info = (f"  シャード{shard - 1}: {n}局 {nrows}行 {len(diags)}窓 "
                    f"（候補{cand}）{time.time() - t0:.0f}s")
            if spreads:
                info += f" 有情報窓率={np.mean([s > 0.2 for s in spreads]):.2f}"
            print(info, flush=True)

    print(f"PLANCF_GEN_DONE 局={tot_games} 行={tot_rows} "
          f"{time.time() - t_all:.0f}s out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
