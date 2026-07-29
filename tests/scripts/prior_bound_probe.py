"""VERIFIED v2 未解決点の prior/value 分解プローブ（v20・読み取り専用）。

問い: コーチゲート v2 の改善ターゲットが **value 較正2周（gen7・gen8）でも 0.00 のまま**なのは
「policy prior が正着を候補から沈めている」からか、「value が正着と誤着の差を感じない」からか。
v13/v15/v17/v19 の実測は value 側の打ち手を出し尽くしつつあり、次に何を教えるかの照準を決めるには
この分解が要る（`docs/reports/gen8_adoption_20260729.md` §5）。

`mark_deep_probe.py`（v6）と役割が違う: あちらは**フレーム復元**＋人間述語で「深探索で解けるか」を
3分類する計器。本プローブは**真盤面復元**（`state_at_action`）＋レフェリー検証済み accept 集合で、
さらに **prior を直読みする**（探索を介さない一次証拠）。1トピック=1ファイルにつき分離する。

各点で4列を測る（すべて現既定エンジン・製品コード無改変）:
  1. `prior`   : 合法手上の policy prior のうち **accept 集合が占める確率質量**と、accept 最良手の順位。
                 探索を介さない直接証拠＝「候補として見えているか」
  2. `dv`      : accept 最良手と CPU 実選択手の**着手後 value 差**（value_fn で子盤面を評価）。
                 正なら value は正着を上に見ている＝選べないのは prior/探索側の問題
  3. `deep`    : sims を上げた decide の accept 率（探索の浅さが原因か）
  4. `flat`    : prior 一様化での深探索 accept 率（policy 起因の分離・mark_deep_probe と同じ機構）

分類（pure・`classify`）:
  - `OK`          : 基準 sims で既に accept（対象外）
  - `EXPLORABLE`  : 深探索で立ち上がる＝探索の浅さ
  - `PRIOR_BOUND` : prior 質量が薄い、かつ flat（一様prior）で立ち上がる＝policy が読ませていない
  - `VALUE_BLIND` : dv ≤ 0＝value が正着を上に見ていない＝value/表現の問題
  - `SEARCH_AVERSE`: prior が正着を1位に置き（rank=1）1手先 value も支持する（dv>0）のに、
                 どれだけ深く探索しても選ばない＝**多手先の読みが正着から離れる**。2026-07-29 の
                 初回測定で3件見つかった第3の機序（prior でも 1手先 value でもない）
  - `UNRESOLVED`  : いずれにも当てはまらない（要個別調査）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/prior_bound_probe.py --out /tmp/prior.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import coach_gate as CG
import p3_loop as P
import rl_encoder as E
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_game import OPCGGame
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.core.cpu_learned import LearnedEngine

PRIOR_THIN = 0.15      # accept 集合の prior 質量がこれ未満＝「候補として沈んでいる」
DEEP_PASS = 2 / 3      # 深探索/flat の accept 率がこれ以上＝立ち上がった（mark_deep_probe と同基準）
_EPS = 1e-9


def classify(base_rate, deep_rate, flat_rate, prior_mass, dv, prior_rank=None):
    """4列 → 分類名（pure）。優先順は「探索で解ける → prior が原因 → value が原因」。"""
    if base_rate is None:
        return "UNRESTORABLE"
    if base_rate >= 0.5:
        return "OK"
    if deep_rate is not None and deep_rate >= DEEP_PASS - _EPS:
        return "EXPLORABLE"
    if (prior_mass is not None and prior_mass < PRIOR_THIN
            and flat_rate is not None and flat_rate >= DEEP_PASS - _EPS):
        return "PRIOR_BOUND"
    if dv is not None and dv <= 0.0:
        return "VALUE_BLIND"
    if prior_rank == 1 and dv is not None and dv > 0.0:
        return "SEARCH_AVERSE"
    return "UNRESOLVED"


@contextlib.contextmanager
def _flat_prior():
    """policy prior を一様化（`TreeMCTS` は priors_fn=None で一様＝製品コード無改変の診断パッチ）。
    `mark_deep_probe._flat_prior` と同じ機構（あちらはフレーム復元経路で使う）。"""
    orig = CL._priors_fn
    CL._priors_fn = lambda pnet, vocab, enc_version=1: None
    try:
        yield
    finally:
        CL._priors_fn = orig


def prior_readout(eng, m0, actor, accept):
    """合法手上の policy prior → (accept 質量, accept 最良手の順位, top1 の記述子)。

    探索を介さない一次証拠。prior が引けない（policy 無し等）場合は (None, None, None)。"""
    legal = m0.get_legal_actions(actor) or []
    if not legal:
        return None, None, None
    pf = P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version)
    if pf is None:
        return None, None, None
    p = pf(m0, legal)
    if p is None:
        return None, None, None
    p = np.asarray(p, dtype=np.float64)
    descs = []
    for mv in legal:
        try:
            descs.append(cpu_ai._describe_move(m0, mv) or {})
        except Exception:
            descs.append({"action_type": (mv or {}).get("action_type")})
    ok = np.array([CG.hit(d, accept) for d in descs], dtype=bool)
    mass = float(p[ok].sum()) if ok.any() else 0.0
    order = np.argsort(-p)                      # prior 降順
    rank = next((r + 1 for r, j in enumerate(order) if ok[j]), None)
    top1 = descs[int(order[0])]
    return mass, rank, {"action_type": top1.get("action_type"), "card": top1.get("card")}


def value_gap(eng, m0, actor, accept):
    """accept 最良手と「prior top1（＝CPU が実際に選びがちな手）」の**着手後 value** 差（pure寄り）。

    value_fn は葉評価そのもの＝「value が正着を上に見ているか」を探索抜きで測る。
    着手は `game.apply` のコピー経路で行い、元盤面は壊さない。"""
    legal = m0.get_legal_actions(actor) or []
    if not legal:
        return None
    game = OPCGGame()
    vf = P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version)
    name = actor if isinstance(actor, str) else actor.name
    best_acc, best_other = None, None
    for mv in legal:
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
            child = game.apply(m0, mv, name)
            if child is None:
                continue
            v = float(vf(child, name))
        except Exception:
            continue
        if CG.hit(d, accept):
            best_acc = v if best_acc is None else max(best_acc, v)
        else:
            best_other = v if best_other is None else max(best_other, v)
    if best_acc is None or best_other is None:
        return None
    return best_acc - best_other


def decide_rate(eng, m0, actor, accept, seeds, sims):
    """coach_gate.decide_rate と同じ（accept への命中率）。sims を変えて呼ぶ。"""
    return CG.decide_rate(eng, m0, actor, accept, seeds, sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="value.npz[,policy.npz]（未指定＝出荷既定＝現 gen8）")
    ap.add_argument("--sims-base", type=int, default=160)
    ap.add_argument("--sims-deep", type=int, default=1600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seeds-deep", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="基準 sims で accept の点も深掘りする（既定は誤る点のみ＝コスト抑制）")
    ap.add_argument("--out", default=None, help="JSON 保存先")
    args = ap.parse_args()

    CR.ARGS = argparse.Namespace(true_board=True)     # 真盤面復元（coach_gate と同条件）
    db = _load_db()
    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)
        label = parts[0].split("/")[-1]
    else:
        eng = LearnedEngine()
        label = "既定(gen8)"

    replays = {**__import__("mark_gate").REPLAYS, **CG.REPLAYS_V2}
    CR.GAMES = {}
    rows, t0 = [], time.time()
    print(f"=== prior/value 分解プローブ net={label} base={args.sims_base} deep={args.sims_deep} ===",
          flush=True)
    for tag, i, accept in CG.VERIFIED_V2:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            rows.append({"tag": tag, "i": i, "cls": "UNRESTORABLE", "reason": built})
            print(f"  {tag}@{i:<4} 復元不可: {built}", flush=True)
            continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        base = decide_rate(eng, m0, actor, accept, args.seeds, args.sims_base)
        mass, rank, top1 = prior_readout(eng, m0, actor, accept)
        dv = value_gap(eng, m0, actor, accept)
        deep = flat = None
        if args.all or base < 0.5:
            deep = decide_rate(eng, m0, actor, accept, args.seeds_deep, args.sims_deep)
            with _flat_prior():
                flat = decide_rate(eng, m0, actor, accept, args.seeds_deep, args.sims_deep)
        cls = classify(base, deep, flat, mass, dv, rank)
        rows.append({"tag": tag, "i": i, "base": base, "deep": deep, "flat": flat,
                     "prior_mass": mass, "prior_rank": rank, "prior_top1": top1,
                     "dv": dv, "cls": cls})
        f = lambda x: " ---" if x is None else f"{x:4.2f}"
        print(f"  {tag}@{i:<4} base={f(base)} deep={f(deep)} flat={f(flat)} "
              f"prior質量={f(mass)} 順位={rank if rank else '-':<3} "
              f"dv={' ----' if dv is None else f'{dv:+5.2f}'} → {cls}"
              f"  (top1={top1.get('action_type') if top1 else '-'}"
              f"/{top1.get('card') if top1 else '-'}) {time.time() - t0:.0f}s", flush=True)

    print("\n=== 分類サマリ（次に何を教えるかの照準）===")
    for cls, dest in (("EXPLORABLE", "探索の浅さ＝深探索再ラベルの射程"),
                      ("PRIOR_BOUND", "policy が候補を沈めている＝policy 側の手当てが要る"),
                      ("VALUE_BLIND", "value が正着を上に見ていない＝特徴/表現の問題"),
                      ("SEARCH_AVERSE", "prior も1手先valueも支持するのに深探索が離れる＝読み出し/多手先の問題"),
                      ("UNRESOLVED", "個別調査"),
                      ("OK", "現既定で正着"),
                      ("UNRESTORABLE", "盤面復元不可")):
        ks = [f"{r['tag']}@{r['i']}" for r in rows if r["cls"] == cls]
        if ks:
            print(f"  {cls:<13} {len(ks)}件: {', '.join(ks)}  ＝ {dest}")
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"PRIOR_PROBE_RESULT {json.dumps({c: sum(r['cls'] == c for r in rows) for c in set(r['cls'] for r in rows)}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
