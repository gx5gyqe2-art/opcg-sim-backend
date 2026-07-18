"""コーチゲート（v9 §4・mark_gate の後継）: レフェリー検証済みバンドで候補ネットを判定する。

mark_gate（v4/v5 の人間マーク述語）は真実源の登場で部分的に古くなった——実測で
g3@115「無意味な守りをしない」は**守るのが唯一の勝ち筋**（レフェリー: カウンター1/8勝ち・
素通しは捲り32世界でも0勝）、@33 は「どの攻撃でも勝つ」同価値圏だった。本ゲートは
人間述語でなく**真盤面レフェリーの同価値バンド（band-top プランの初手集合）**への所属で
判定する。人間マークはレフェリーで裏取りされた形で引き継がれる（ユーザ承認 2026-07-18）。

判定（mark_gate と同型・gen5 と候補を同条件で比較）:
  - 非退行: base が確実に打てていた点（base≥0.8）で chall が大きく落ちない（chall > base−0.4）
  - 改善: ヒット率合計が base 以上（レフェリー正解へ近づいたか＝進歩検出）
  PASS = 非退行 かつ 改善。

VERIFIED の各点は真盤面レフェリー実測（worlds/sims/日付を出典に明記）から採録。
`--regen` での自動再検証は将来項（現状は採録値が正・変更時はレフェリーを回して更新する）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/coach_gate.py \
    --challenger cand_value.npz,cand_policy.npz --seeds 5
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import mark_gate as MG
import replay_reeval as RE
from opcg_sim.src.core import cpu_ai

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# レフェリー検証済み決定点（真盤面・出典は各行コメント: 世界数/sims/実測日）。
# accept = 同価値バンド（band-top）プランの**初手**の (action_type, card) 集合。
# card=None は action_type のみで判定（PASS/TURN_END 等）。
VERIFIED = [
    # @33: 全攻撃系が 8/8 勝ち・バンド= bare Marco / 付与→リーダー / OP15-119 / 付与Zeus系
    #      （8世界 sims32 auto 2026-07-18）
    ("g3", 33, {("ATTACK", "PRB02-008"), ("ATTACH_DON", "OP11-041"),
                ("ATTACK", "OP15-119"), ("ATTACH_DON", "OP11-106")}),
    # @64: 素攻撃 ≈ 攻撃者へ付与→攻撃（12世界 sims32 正味1・2026-07-17）
    ("g3", 64, {("ATTACK", "PRB02-008"), ("ATTACH_DON", "PRB02-008")}),
    # @68: 付与→ゼウスで攻撃が断定勝ち（16世界 正味+3・素攻撃/リーダー付与はバンド外・2026-07-17）
    ("g3", 68, {("ATTACH_DON", "OP11-106")}),
    # @82（防御窓）: 素通し PASS が最良・EB03/105切りはライフ差でバンド外
    #      （プランスイープ 4世界＋root 6世界・2026-07-17）
    ("g3", 82, {("PASS", None)}),
    # @93: 展開（唯一の勝ち筋系）。root 6世界=OP16-056 1/6・sweep 4世界=OP15-119 系＝
    #      展開2種を許容・付与/攻撃はバンド外（2026-07-16/17）
    ("g3", 93, {("PLAY", "OP16-056"), ("PLAY", "OP15-119")}),
    # @115（防御窓）: OP16-056 カウンターが唯一の勝ち筋（8世界 1/8・捲り32世界でも守り側のみ勝ち・
    #      素通しは最下位・2026-07-18）＝旧 mark_gate「無意味な守りをしない」を反転
    ("g3", 115, {("SELECT_COUNTER", "OP16-056")}),
    # @137: 捲り筋はゼウス付与→ゼウス攻撃のみ（捲り16世界 1/16・他0・2026-07-17）
    ("g3", 137, {("ATTACH_DON", "OP11-106")}),
]


def hit(desc, accept):
    """decide の記述（action_type/card）が合格集合に入るか（pure）。"""
    at = desc.get("action_type")
    card = desc.get("card")
    return (at, card) in accept or (at, None) in accept


def decide_rate(eng, m0, actor, accept, seeds, sims):
    n = 0
    for s in range(seeds):
        eng._world_seeds = {}
        mv = eng.decide(m0, actor, sims=sims, rng=np.random.default_rng(9100 + 97 * s))
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        if hit(d, accept):
            n += 1
    return n / max(seeds, 1)


def judge(rows, regress_base=0.8, regress_drop=0.4):
    """点別 (base, chall) → (非退行OK, 改善OK, 退行リスト)（pure・mark_gate と同型の判定）。"""
    regressions = [(tag, i, b, c) for (tag, i, b, c) in rows
                   if b >= regress_base and c <= b - regress_drop]
    improve = sum(c for _t, _i, _b, c in rows) >= sum(b for _t, _i, b, _c in rows)
    return (not regressions), improve, regressions


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--baseline", default=None,
                    help="value.npz[,policy.npz]（既定=出荷 gen5）")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sims", type=int, default=160)
    ARGS = ap.parse_args()
    CR.ARGS = argparse.Namespace(true_board=True)

    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()

    def _eng(spec):
        if not spec:
            return LearnedEngine()
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0],
                             policy_path=parts[1] if len(parts) > 1 else None)

    base_eng = _eng(ARGS.baseline)
    chall_eng = _eng(ARGS.challenger)

    CR.GAMES = {}
    rows = []
    for tag, i, accept in VERIFIED:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(MG.REPLAYS[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            print(f"{tag}@{i}: 復元不可（スキップ）: {built}")
            continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        b = decide_rate(base_eng, m0, actor, accept, ARGS.seeds, ARGS.sims)
        c = decide_rate(chall_eng, m0, actor, accept, ARGS.seeds, ARGS.sims)
        rows.append((tag, i, b, c))
        print(f"  {tag}@{i:<4} base={b:.2f} chall={c:.2f}  合格手={sorted(accept)}")
    ok_nr, ok_imp, regs = judge(rows)
    print(f"\n改善: {'OK' if ok_imp else 'NG'}"
          f"（chall計 {sum(c for *_ , c in rows):.1f} vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {[(t, i, b, c) for t, i, b, c in regs]}")
    verdict = "PASS" if (ok_nr and ok_imp) else "FAIL"
    print(f"COACH_GATE_RESULT {json.dumps({'verdict': verdict, 'points': len(rows)})}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
