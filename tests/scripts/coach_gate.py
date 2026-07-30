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

# --- VERIFIED v2（gen7 実対局マーク 2026-07-28・`mark_referee_verify.py` worlds=8 実測） ---
# 出典: tests/fixtures/replays/gen7_marks_20260728/（5局34マーク→真盤面復元＋ターン一致 17点→
# 裁定14点。捲り1/32勝ちの極薄点 @94 は不採録＝旧@137型の反省）。バンド外=改善ターゲット・
# バンド内=非退行ガードの両方を採録する（判定則 judge は共通＝base≥0.8 の点が退行しないこと）。
# 旧 g3（単一対局・gen4期）は --profile g3 で存続（診断用）。
_FIX2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fixtures", "replays", "gen7_marks_20260728")
REPLAYS_V2 = {
    "m1": os.path.join(_FIX2, "opcg_replay_2057134394987494995.json.gz"),   # ナミ(人) vs シャンクス(CPU)
    "m2": os.path.join(_FIX2, "opcg_replay_3806796710697874793.json.gz"),   # シャンクス(人) vs ナミ(CPU)
    "m4": os.path.join(_FIX2, "opcg_replay_6563214359889287880.json.gz"),   # ナミ(人) vs シャンクス(CPU)
    "m5": os.path.join(_FIX2, "opcg_replay_9195490382040907274.json.gz"),   # シャンクス(人) vs ナミ(CPU)
}
# **VERIFIED v3**（2026-07-30 再裁定・/tmp/mark_verify3.jsonl・worlds16）。旧 v2（13点・worlds8）は
# 効果対話の既定解決欠陥（`docs/reports/default_interaction_fix_20260730.md`＝捨て札が公開札を
# 捨てる／up-to 獲得を常時見送る）で PLAY 系プランの測定が汚染されていたため、修正後エンジンで
# 34マークを全点再裁定した（11/34 で裁定/accept が変化・旧表は同レポートと v18 レポートに保存）。
# 変数名は既存プローブ（prior_bound_probe / value_blind_probe）互換のため VERIFIED_V2 のまま。
VERIFIED_V2 = [
    # m1: CPU=シャンクス
    ("m1", 3,  {("PLAY", "OP09-002"), ("PLAY", "ST30-004")}),              # ガード（1コスト展開は正・ウタも band）
    ("m1", 14, {("SELECT_COUNTER", "OP09-002"), ("SELECT_COUNTER", "OP10-011")}),  # 反転: カウンターは band 内（旧: PASS のみ）
    ("m1", 15, {("SELECT_COUNTER", "OP10-011")}),                          # チョッパーで守る（PASS は band 外へ）
    ("m1", 42, {("ATTACH_DON", "OP09-002"), ("ATTACH_DON", "OP12-008"),
                ("ATTACK", "ST30-004")}),                                  # ガード
    ("m1", 94, {("ATTACK", "OP09-001"), ("ATTACK", "OP09-002"),
                ("ATTACK", "ST30-004")}),                                  # 新規: 攻撃すべき（耐久でなく）
    # m2: CPU=ナミ
    ("m2", 12, {("PASS", None)}),                                          # 反転: 素通しが正（旧: カウンター）
    ("m2", 44, {("ATTACH_DON", "OP11-041")}),                              # リーダーへ付与（守り）
    ("m2", 58, {("PASS", None)}),                                          # ガード（accept は素通しのみに縮小）
    ("m2", 64, {("ATTACK", "OP16-056")}),                                  # クマシー出しでなく攻撃
    ("m2", 66, {("ATTACK", "EB03-055")}),                                  # ロビンで攻撃（accept 縮小）
    # m4: CPU=シャンクス
    ("m4", 2,  {("PLAY", "OP13-007")}),                                    # 正解変化: TURN_END でなくエース&サボ&ルフィを出す
    ("m4", 8,  {("ATTACH_DON", "ST30-004"), ("ATTACK", "OP09-001"),
                ("ATTACK", "ST30-004"), ("PLAY", "OP13-007")}),            # ガード（band 拡大）
    ("m4", 12, {("ATTACK", "ST30-004"), ("PLAY", "OP09-002")}),            # 正解変化: イワンコフで攻撃 or ウタ展開
    # m5: CPU=ナミ
    ("m5", 7,  {("ATTACH_DON", "OP11-041"), ("PLAY", "OP11-106")}),        # ナミ3ドン付与
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


def min_reliable_delta(seeds):
    """点別の命中率差が『測定ノイズでない』と言える最小幅（pure・2σ・最悪ケース p=0.5）。

    命中率は seeds 回のベルヌーイ試行＝SE ≤ 0.5/√n。2条件の差の SE は √2 倍なので
    2σ ≈ 1.414/√n。v22 実測（`docs/reports/coach_gate_variance_20260729.md`）で
    5seed（bar 0.63）では m4@8 の 0.60→0.20 が『退行』に見えたが、16seed（bar 0.35）では
    両者 0.38 で差が無かった。**この bar 未満の点別増減を『治った/壊れた』と書かない**。"""
    return 1.4142135623730951 / (seeds ** 0.5) if seeds > 0 else float("inf")


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
                    help="value.npz[,policy.npz]（既定=出荷既定＝現 gen7）")
    ap.add_argument("--seeds", type=int, default=16,
                    help="点ごとの decide 回数。**5 は分散が大きすぎる**（v22 実測: 5seed で "
                         "『退行』に見えた m4@8 が 16seed では差なし）。`min_reliable_delta` 参照")
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--profile", default="v2", choices=("v2", "g3", "all"),
                    help="v2=gen7実対局13点（既定）／g3=旧7点（gen4期・診断用）／all=両方")
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

    points = {"v2": VERIFIED_V2, "g3": VERIFIED,
              "all": VERIFIED + VERIFIED_V2}[ARGS.profile]
    replays = {**MG.REPLAYS, **REPLAYS_V2}
    CR.GAMES = {}
    rows = []
    for tag, i, accept in points:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
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
    bar = min_reliable_delta(ARGS.seeds)
    sig = [(t, i, b, c) for t, i, b, c in rows if abs(c - b) >= bar]
    print(f"\n測定ノイズでないと言える差の下限（2σ・seeds={ARGS.seeds}）= {bar:.2f}")
    print(f"  この bar を超えた点: "
          + (", ".join(f"{t}@{i}({b:.2f}→{c:.2f})" for t, i, b, c in sig) if sig else "なし")
          + "  ← これ未満の増減は『治った/壊れた』と読まない")
    ok_nr, ok_imp, regs = judge(rows)
    print(f"\n改善: {'OK' if ok_imp else 'NG'}"
          f"（chall計 {sum(c for *_ , c in rows):.1f} vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {[(t, i, b, c) for t, i, b, c in regs]}")
    verdict = "PASS" if (ok_nr and ok_imp) else "FAIL"
    print(f"COACH_GATE_RESULT {json.dumps({'verdict': verdict, 'points': len(rows)})}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
