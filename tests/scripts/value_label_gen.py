"""一般盤面の**純粋勝率ラベル**コーパス生成（v47 手順1・2026-08-09）。

**何のためか**: v47 手順0（`value_calibration_audit.py`）で本体 value の系統誤差を検出したが、
その測定には2つの疑いが残っていた——(a) ラベルが `margin_blend` 混合（ライフ差タイブレーク
w=0.25 込み）、(b) 盤面が**出口分布**（ターン末・戦闘後）に偏る。本生成器は両方を消す:
**勝敗だけの純粋ラベル**を、**木の葉が実際に見る一般盤面**（ターン全域・自ターン/相手ターン・
戦闘中も含む任意の決定点）に付ける。監査の物差しであると同時に、較正修正の教師にもなる。

**既存生成器との違い**（1トピック=1ファイルの棲み分け）:
  - `plan_cf_gen` / `defense_cf_gen` は**同一決定点の兄弟枝を対照する順位ペア**教師で、
    盤面は「プラン実行後のターン末」「戦闘解決後の出口」に限定される。
  - 本器は枝を作らない。**1盤面 = 1行**で、その盤面からの勝率そのものを測る（水準の較正用）。
    group は盤面ごとに一意＝順位学習には使えない（`load_pairs_corpus` は 2行以上の群を要求）。

**ラベル**: 各盤面から K 世界（CRN）を決定化してロールアウトし、世界ごとの勝敗を `win_w`、
残ライフ差を `life_w` に**生のまま**保存する（v44 と同じ規約）。集計 z を焼き込まないので、
予算 K'≤K の再集計も blend の有無も後から選べる。`value` 列には純粋 z（=2*勝率-1）を入れる
＝**混合しない**（本器の目的が blend 汚染の除去なので）。

**採取の層化**: 自己対戦の全決定点から一様抽出するとターン数の分布が対局長に引きずられ、
序盤に偏る（実測: 平均対局長のほぼ半分が turn≤6）。ターン帯（0-6 / 7-10 / 11+）ごとに
クォータを切って抽出する＝監査の層別セルに最低数を確保する。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_label_gen.py \\
    --games 40 --boards-per-game 6 --worlds 8 --workers 4 --out /tmp/vlabel
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

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
import defense_cf_gen as DG                       # ワーカー状態と自己対戦1手は共有（1定義）
from defense_cf_gen import _decide                # noqa: E402
from plan_cf_gen import DECKS_JSON, _init_worker  # 初期化も共有（同一の _G を埋める）

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_G = DG._G

# ターン帯のクォータ（監査の層別セルに最低数を確保する）。合計が boards_per_game を超える
# 場合は先頭から詰める。序盤に偏るのを防ぐのが目的なので中盤以降を厚くする。
TURN_BANDS = ((0, 7), (7, 11), (11, 999))


def pick_stratified(snaps, k, rng):
    """(turn, ...) を持つ候補列から、ターン帯クォータで k 件を抽出（pure・rng 消費）。

    各帯へ k を等分し、足りない帯のぶんは他帯から補う（総数は必ず min(k, len(snaps))）。
    帯内は一様抽出＝同一対局の連続局面が固まらないようにする。"""
    if len(snaps) <= k:
        return list(range(len(snaps)))
    by_band = {b: [] for b in range(len(TURN_BANDS))}
    for i, s in enumerate(snaps):
        t = s[2]
        for bi, (lo, hi) in enumerate(TURN_BANDS):
            if lo <= t < hi:
                by_band[bi].append(i)
                break
    quota = [k // len(TURN_BANDS)] * len(TURN_BANDS)
    for i in range(k - sum(quota)):
        quota[i % len(TURN_BANDS)] += 1
    out, short = [], 0
    for bi in range(len(TURN_BANDS)):
        pool, q = by_band[bi], quota[bi]
        take = min(q, len(pool))
        if take:
            out += [int(x) for x in rng.choice(pool, size=take, replace=False)]
        short += q - take
    if short:                                     # 足りない帯のぶんを残りから補充
        rest = [i for i in range(len(snaps)) if i not in set(out)]
        if rest:
            out += [int(x) for x in rng.choice(rest, size=min(short, len(rest)), replace=False)]
    return sorted(out)


def process_game(task):
    """1局: 自己対戦 → 決定点を層化抽出 → 各盤面を K 世界でロールアウト → 行データ。"""
    seed, cfg, gbase = task
    CR = _G["CR"]
    import rl_encoder as E
    from opcg_sim.src.learned.mcts import in_battle
    game, gserve, eng = _G["game_gen"], _G["gserve"], _G["eng"]
    rng = np.random.default_rng(seed)
    m = game.new_game(_G["db"], seed)
    snaps, steps_n = [], 0
    while m.winner is None and not gserve.is_terminal(m) and steps_n < CR.MAX_STEPS:
        name = gserve.current_player(m)
        if name is None:
            break
        # **葉が見る盤面**をそのまま採る（自ターン/相手ターン・戦闘中も含む任意の決定点）。
        snaps.append((m, name, int(getattr(m, "turn_count", 0) or 0), bool(in_battle(m))))
        mv = _decide(game, m, name, cfg["gen_sims"], cfg["eps"], rng, world_seed=seed)
        if mv is None:
            break
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps_n += 1

    picked = pick_stratified(snaps, cfg["boards_per_game"], rng)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root", "turns_left",
                            "group", "win_w", "life_w")}
    diag = []
    for bi, pi in enumerate(picked):
        m0, name, turn, battling = snaps[pi]
        win_w = [np.nan] * cfg["worlds"]
        life_w = [np.nan] * cfg["worlds"]
        ends = []
        for w in range(cfg["worlds"]):
            wseed = seed * 1009 + pi * 101 + w * 97
            try:
                world = gserve.determinize(m0, name, np.random.default_rng(wseed))
            except Exception:
                continue
            if gserve.is_terminal(world):
                won = (getattr(world, "winner", None) == name)
                win_w[w] = 1.0 if won else 0.0
                life_w[w] = 1.0 if won else -1.0
                continue
            winner, ld, et = CR.rollout(gserve, _G["vf"], _G["pf"], world, name,
                                        world_seed=wseed, rng_seed=wseed * 131,
                                        def_temp=cfg["def_temp"])
            win_w[w] = 1.0 if winner == name else 0.0
            life_w[w] = ld
            ends.append(et)
        ok = int(np.isfinite(win_w).sum())
        if ok == 0:
            continue
        z = 2.0 * float(np.nansum(win_w)) / ok - 1.0     # **純粋勝率 z（混合しない）**
        enc = E.encode(m0, name, eng.vocab, version=_G["enc_version"])
        rows["scalars"].append(enc["scalars"])
        rows["field"].append(enc["field"])
        rows["card_idx"].append(enc["card_idx"])
        rows["value"].append(z)
        rows["win_w"].append(list(win_w))
        rows["life_w"].append(list(life_w))
        rows["q_root"].append(np.nan)                    # 勝敗単独ラベル（エコー遮断）
        tl = (np.mean(ends) - turn) if ends else np.nan
        rows["turns_left"].append(max(0.0, float(tl)) if np.isfinite(tl) else np.nan)
        rows["group"].append(gbase + bi)                 # 1盤面=1群（順位学習には使えない）
        diag.append({"turn": turn, "battle": battling, "z": round(z, 3), "ok": ok})
    return rows, diag, len(snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--shard-games", type=int, default=4,
                    help="1シャードあたりの局数。**既存 out に対して変えると resume が壊れる**"
                         "（消化済み推定が 既存シャード数×本値 のため）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--worlds", type=int, default=8, help="CRN 世界数（ラベル1σ≈0.35@K=8）")
    ap.add_argument("--boards-per-game", type=int, default=6)
    ap.add_argument("--rollout-sims", type=int, default=24)
    ap.add_argument("--gen-sims", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--def-temp", type=float, default=0.7)
    ap.add_argument("--matchup", default="nami:shanks")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=970000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = len(glob.glob(os.path.join(args.out, "vlabel_*.npz")))
    cfg = {k: getattr(args, k) for k in ("worlds", "boards_per_game", "rollout_sims",
                                         "gen_sims", "eps", "def_temp")}
    print(f"=== 純粋ラベル盤面コーパス matchup={args.matchup} worlds={args.worlds} "
          f"boards/局={args.boards_per_game} ev={args.enc_version} 既存シャード={done} ===",
          flush=True)

    t_all = time.time()
    tot_rows = 0
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.matchup, args.decks_json, args.rollout_sims,
                           args.enc_version)) as pool:
        shard = done
        games_done = done * args.shard_games
        while games_done < args.games:
            n = min(args.shard_games, args.games - games_done)
            t0 = time.time()
            outs = pool.map(process_game, [(args.seed_base + games_done + g, cfg,
                                            (args.seed_base + games_done + g) * 100)
                                           for g in range(n)])
            parts = {k: [] for k in ("scalars", "field", "card_idx", "value", "q_root",
                                     "turns_left", "group", "win_w", "life_w")}
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
                    "value": np.asarray(parts["value"], dtype=np.float32),
                    "q_root": np.asarray(parts["q_root"], dtype=np.float32),
                    "turns_left": np.asarray(parts["turns_left"], dtype=np.float32),
                    "group": np.asarray(parts["group"], dtype=np.int64),
                    "win_w": np.asarray(parts["win_w"], dtype=np.float32),
                    "life_w": np.asarray(parts["life_w"], dtype=np.float32),
                }
                np.savez_compressed(os.path.join(args.out, f"vlabel_{shard:05d}.npz"), **arrays)
                with open(os.path.join(args.out, f"meta_{shard:05d}.json"), "w") as f:
                    json.dump({"source": "value_label", "games": n, "rows": nrows,
                               "worlds": args.worlds, "rollout_sims": args.rollout_sims,
                               "matchup": args.matchup, "enc_version": args.enc_version,
                               "diag": diags[:40]}, f, ensure_ascii=False)
            dt = time.time() - t0
            tb = {}
            for d in diags:
                for lo, hi in TURN_BANDS:
                    if lo <= d["turn"] < hi:
                        tb[f"{lo}-{hi if hi < 999 else '+'}"] = tb.get(
                            f"{lo}-{hi if hi < 999 else '+'}", 0) + 1
            print(f"  シャード{shard}: {n}局 {nrows}行 （候補{cand}）{dt:.0f}s "
                  f"ターン帯 {tb}", flush=True)
            tot_rows += nrows
            games_done += n
            shard += 1
    print(f"VALUE_LABEL_GEN_DONE 局={games_done} 行={tot_rows} "
          f"{time.time() - t_all:.0f}s out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
