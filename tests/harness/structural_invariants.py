"""構造不変条件スキャン群（カテゴリH ポストモーテムの横展開4スキャンの回帰ツール化）。

`docs/reports/quality_postmortem_categoryH.md` §5/§6 の「ベースラインが凍結し oracle/classify が
盲点になる」構造的バグ候補を、AST 構造不変条件として機械検出する。ベースライン（挙動の安定性）と
オラクル（パーサのフォールバック量）が測れない *条件スコープ／期間／選択者／全体性* の死角を埋める。

検出器（すべて現状 0 件＝ラチェット上限0。新規混入を機械的に検出する）:

  - H_LEADING_GATE_LEAK : 先頭ゲート条件が「。その後、」をまたいで後続を無条件化（カテゴリH 本体）
  - DURATION_WRITEOFF   : 「このターン中/まで」等の時限テキストがあるのに付与系が duration=INSTANT
  - CHOOSER_MISSING     : 「相手は自身の…」で対象選択者が controller のまま（相手の隠匿札を自分が選ぶ）
  - SUBETE_COUNT_DEGRADE: 「すべて」なのに対象 count≥1 かつ select_mode が全体でない（数量詞の退化）

使い方:
    OPCG_LOG_SILENT=1 python tests/structural_invariants.py            # 件数サマリ
    OPCG_LOG_SILENT=1 python tests/structural_invariants.py --show     # 該当一覧
"""
import argparse
import os
import re

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.src.models.effect_types import Sequence, Branch, GameAction, Choice
from opcg_sim.src.models.enums import ActionType, Player
# H 検出は parser.py のH是正と同一定義を共有する（ゲートと修正の一貫性）。
from opcg_sim.src.core.effects.parser import _H_SETUP_ACTIONS, _h_is_genuine

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "opcg_sim", "data", "opcg_cards.json")

CATEGORIES = (
    "H_LEADING_GATE_LEAK", "DURATION_WRITEOFF", "CHOOSER_MISSING", "SUBETE_COUNT_DEGRADE",
)

# 時限テキスト（期間付き効果を意図する表現）。
_DURATION_TEXT = re.compile(r'(このターン中|このバトル中|次の(相手|自分)のターン終了時まで|まで、)')
# duration を持つべき付与系アクション。
_DURATION_ACTIONS = frozenset({
    ActionType.BUFF, ActionType.GRANT_KEYWORD, ActionType.GRANT_EFFECT,
    ActionType.KEYWORD, ActionType.COST_BUFF, ActionType.SET_BASE_POWER,
})


def _walk(node):
    """ノード木中の全 GameAction を深く列挙する。"""
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk(a)
    elif isinstance(node, Branch):
        yield from _walk(node.if_true)
        yield from _walk(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk(o)


def _scan_h(ab):
    """カテゴリH 本体（`EffectParser._lift_h_gate` と同一構造定義）。"""
    eff = ab.effect
    if not isinstance(eff, Sequence):
        return None
    acts = eff.actions
    branch_idx = None
    for i, a in enumerate(acts):
        if isinstance(a, Branch) and a.if_false is None and a.condition is not None:
            branch_idx = i
            break
        if isinstance(a, GameAction) and a.type in _H_SETUP_ACTIONS:
            continue
        return None
    if branch_idx is None:
        return None
    tail = acts[branch_idx + 1:]
    if any(isinstance(t, Branch) for t in tail):
        return None
    if any(_h_is_genuine(t) for t in tail):
        return " / ".join((getattr(t, "raw_text", "") or "").strip()[:30] for t in tail)
    return None


def scan(db):
    """全カードを走査し、カテゴリ別に (card_id, trigger, detail) を返す。"""
    findings = {c: [] for c in CATEGORIES}
    for cid in sorted(db.raw_db.keys()):
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        for ab in master.abilities:
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            h = _scan_h(ab)
            if h:
                findings["H_LEADING_GATE_LEAK"].append((cid, trig, h))
            for nd in (ab.cost, ab.effect):
                for a in _walk(nd):
                    rt = a.raw_text or ""
                    if (a.type in _DURATION_ACTIONS and a.duration == "INSTANT"
                            and _DURATION_TEXT.search(rt)):
                        findings["DURATION_WRITEOFF"].append((cid, trig, rt[:40]))
                    if (re.search(r'相手は自身の', rt) and a.target is not None
                            and a.target.chooser != Player.OPPONENT):
                        findings["CHOOSER_MISSING"].append((cid, trig, rt[:40]))
                    # 「すべて」の数量詞退化。SOURCE（全体パッシブ）／「見て」（参照範囲）／
                    # 「ずつ」（対象ごと配分）は count が別意味のため除外する。
                    if (("すべて" in rt or "全て" in rt) and a.target is not None
                            and a.target.count >= 1
                            and a.target.select_mode not in ("ALL", "SOURCE")
                            and "見て" not in rt and "ずつ" not in rt):
                        findings["SUBETE_COUNT_DEGRADE"].append((cid, trig, rt[:40]))
    return findings


def run(show: bool):
    db = CardLoader(DATA)
    db.load()
    findings = scan(db)
    print("=== 構造不変条件スキャン群（横展開4スキャン）===")
    for c in CATEGORIES:
        print(f"  {c:<22}: {len(findings[c]):4d}")
    if show:
        for c in CATEGORIES:
            for cid, trig, detail in findings[c]:
                print(f"  [{c}] {cid} {trig}  {detail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    run(show=args.show)
