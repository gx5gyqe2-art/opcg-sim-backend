"""防御窓の反実仮想測定と人間整合（①防御応答矯正フェーズ1・2026-07-30・読み取り専用）。

背景: 手札（特にカウンター）の価値は「将来の防御窓でそれが使われる世界」でしか勝敗に現れない。
現行ネットは守らなさすぎ（defense_rate_probe 実測）なので、argmax ロールアウトの測定・ラベルは
手札温存の価値を構造的に取り逃す（m1@3 のクロス測定・v23 相貌学習と同根）。L1 を教師にせず
この循環を切る第一歩として、**防御探索つきロールアウト**（`counterfactual_referee.rollout` の
`def_temp`）で防御窓の各選択肢の因果差を測り、**人間（ユーザ実対局）の実選択との整合**を検証する。

各防御窓（人間側の SELECT_COUNTER / SELECT_BLOCKER / 防御 PASS）で:
  1. 真盤面復元（`state_at_action`）→ 窓であることを pending で照合（違えば skip）
  2. 選択肢（素通し PASS / 各カウンター / 各ブロッカー・card_id で重複排除）を列挙
  3. 同一CRN世界 × 防御温度つき終局ロールアウト → 選択肢ごとの勝ち数
  4. 人間の実選択と測定順位の整合（top一致 / band内=勝ち数差<しきい）を記録

このスクリプトは**測定系の健全性検査**（フェーズ2の量産に進む前の関門）。人間選択は
ラベルに使わず、測定が人間とどれだけ割れるかの監査に使う（割れた窓はユーザレビューへ）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/defense_cf_probe.py \
      --worlds 8 --def-temp 0.7 --out /tmp/defcf.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

_FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fixtures", "replays", "gen7_marks_20260728")
REPLAYS = {
    "m1": "opcg_replay_2057134394987494995.json.gz",
    "m2": "opcg_replay_3806796710697874793.json.gz",
    "m3": "opcg_replay_5936431651544026420.json.gz",
    "m4": "opcg_replay_6563214359889287880.json.gz",
    "m5": "opcg_replay_9195490382040907274.json.gz",
}
DEF_WINDOWS = ("SELECT_COUNTER", "SELECT_BLOCKER")


def branch_key(desc):
    """選択肢の同一視キー（pure）: 行動種 × card_id（同名カウンターの複製は等価）。"""
    d = desc or {}
    return (d.get("action_type"), d.get("card"))


def dedupe_branches(descs):
    """記述子列 → 重複排除した (key, 元index) 列（pure・列挙順維持）。"""
    seen, out = set(), []
    for i, d in enumerate(descs):
        k = branch_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append((k, i))
    return out


def agreement(human_key, results, band=3):
    """人間の選択と測定結果の整合（pure）。results={key: wins}。

    returns: {human_wins, best_key, best_wins, agree_top, agree_band}
      - agree_top: 人間の選択が最多勝ち（同数タイ含む）
      - agree_band: 最多との勝ち数差 < band（レフェリーの断定則と同じ許容）
    人間の選択が results に無い（列挙漏れ）は agree_* = None＝計器側の欠陥として表面化させる。"""
    if human_key not in results:
        return {"human_wins": None, "best_key": max(results, key=results.get) if results else None,
                "best_wins": max(results.values()) if results else None,
                "agree_top": None, "agree_band": None}
    hw = results[human_key]
    bw = max(results.values())
    bk = max(results, key=results.get)
    return {"human_wins": hw, "best_key": bk, "best_wins": bw,
            "agree_top": hw >= bw, "agree_band": (bw - hw) < band}


# --- worker ----------------------------------------------------------------------
_G = {}


def _init_worker(sims):
    import counterfactual_referee as CR
    import p3_loop as P
    import replay_reeval as RE
    import replay_runner as RR
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(sims=sims, true_board=True)
    eng = LearnedEngine()
    _G.update(db=_load_db(), CR=CR, RE=RE, RR=RR,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
              gserve=OPCGGame(), recs={})


def _rec_of(tag):
    if tag not in _G["recs"]:
        raw = _G["RE"].load_replay_json(os.path.join(_FIX, REPLAYS[tag]))
        _G["recs"][tag] = raw.get("replay", raw)
    return _G["recs"][tag]


def probe_window(task):
    """1窓: 復元 → 選択肢列挙 → CRN×防御温度ロールアウト → 人間整合。"""
    tag, idx, cfg = task
    from opcg_sim.src.core import cpu_ai
    CR, gserve = _G["CR"], _G["gserve"]
    rec = _rec_of(tag)
    recorded = rec["actions"][idx]
    human_key = (recorded.get("action_type"), recorded.get("card"))
    m0, who = _G["RR"].state_at_action(_G["db"], rec, idx)
    if m0 is None:
        return {"tag": tag, "idx": idx, "skip": "restore_fail"}
    name = who if isinstance(who, str) else who.name
    pa = m0.pending_actor_action()
    if not pa or pa[0] != name or pa[1] not in DEF_WINDOWS:
        return {"tag": tag, "idx": idx, "skip": f"not_window:{pa[1] if pa else None}"}
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
        return {"tag": tag, "idx": idx, "skip": "single_choice"}
    results, ok_worlds = {}, 0
    for w in range(cfg["worlds"]):
        wseed = 70000 + w * 97
        try:
            world = gserve.determinize(m0, name, np.random.default_rng(wseed))
        except Exception:
            continue
        ok_worlds += 1
        for key, i in branches:
            child = gserve.apply(world, legal[i], name)
            if child is None:
                continue
            # rng_seed は**枝に依存させない**（真のCRN）。枝別乱数は対照性を壊し、m2@48 で
            # 「素通し4勝 vs カウンター8勝」という偽の差を作った（共有乱数では 10 vs 9=同値）。
            winner, _ld, _et = CR.rollout(gserve, _G["vf"], _G["pf"], child, name,
                                          world_seed=wseed, rng_seed=wseed * 131,
                                          def_temp=cfg["def_temp"])
            results[key] = results.get(key, 0) + (1 if winner == name else 0)
    ag = agreement(human_key, results, band=cfg["band"])
    return {"tag": tag, "idx": idx, "turn": rec["actions"][idx].get("turn"),
            "window": pa[1], "human": list(human_key), "worlds": ok_worlds,
            "results": {f"{k[0]}:{k[1] or ''}": v for k, v in results.items()}, **ag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="m1,m2,m3,m4,m5")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--sims", type=int, default=64, help="ロールアウト decide sims")
    ap.add_argument("--def-temp", type=float, default=0.7,
                    help="ロールアウト内の防御窓サンプリング温度（0=従来argmax）")
    ap.add_argument("--band", type=int, default=3, help="agree_band の勝ち数差しきい")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=0, help=">0 で1局あたり上限")
    ap.add_argument("--out", default=None, help="JSONL 追記先（再開可能）")
    args = ap.parse_args()

    import replay_reeval as RE
    done = set()
    if args.out and os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["tag"], r["idx"]))
    tasks = []
    cfg = {"worlds": args.worlds, "def_temp": args.def_temp, "band": args.band}
    for tag in args.tags.split(","):
        raw = RE.load_replay_json(os.path.join(_FIX, REPLAYS[tag]))
        rec = raw.get("replay", raw)
        cpu = rec.get("cpu_player_id")
        idxs = [i for i, a in enumerate(rec["actions"])
                if a.get("player") != cpu
                and a.get("action_type") in ("SELECT_COUNTER", "SELECT_BLOCKER", "PASS")
                and (tag, i) not in done]
        if args.max_windows > 0:
            idxs = idxs[:args.max_windows]
        tasks += [(tag, i, cfg) for i in idxs]
    print(f"=== 防御窓の反実仮想×人間整合 windows={len(tasks)} worlds={args.worlds} "
          f"def_temp={args.def_temp} ===", flush=True)

    t0 = time.time()
    n_ok = n_top = n_band = 0
    disagrees = []
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(args.sims,)) as pool:
        for r in pool.imap_unordered(probe_window, tasks):
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r.get("skip"):
                print(f"  {r['tag']}@{r['idx']} skip:{r['skip']}", flush=True)
                continue
            n_ok += 1
            n_top += bool(r["agree_top"])
            n_band += bool(r["agree_band"])
            mark = "○" if r["agree_top"] else ("△" if r["agree_band"] else "✗")
            print(f"  {mark} {r['tag']}@{r['idx']:<4} turn{r['turn']} [{r['window']}] "
                  f"人間={r['human'][0]}:{r['human'][1] or ''} ({r['human_wins']}勝) "
                  f"最良={r['best_key'][0]}:{r['best_key'][1] or ''} ({r['best_wins']}勝) "
                  f"{time.time() - t0:.0f}s", flush=True)
            if not r["agree_band"]:
                disagrees.append(r)
    print(f"\n=== 整合サマリ: 測定可 {n_ok}窓 / top一致 {n_top} / band内 {n_band} "
          f"（band外の不一致 {len(disagrees)}） ===")
    for r in disagrees:
        print(f"  ✗ {r['tag']}@{r['idx']} 人間={r['human']} {r['results']}")
    print(f"DEFENSE_CF_RESULT {json.dumps({'windows': n_ok, 'agree_top': n_top, 'agree_band': n_band}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
