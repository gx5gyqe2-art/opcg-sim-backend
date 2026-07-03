"""条件“偽”パスの被覆（カテゴリH 再発防止 §6.2）。

カテゴリH の見逃しの本丸は「ベースラインは*条件が真*の出力を正本として凍結し、*条件が偽*のとき
B が漏れて走る挙動を一切検証しない」ことだった（`docs/reports/quality_postmortem_categoryH.md` §3.2）。
本ハーネスは死角の逆側＝**条件を偽にして発動し、ゲートされた効果が一切走らない（状態変化ゼロ）**
ことを動的に確認する。

方式:
  - **ability レベルのゲート条件を持つ能力**（先頭条件が能力全体を支配＝H 是正後の正しい形）を対象。
  - その ability.condition のみを **強制 False**（同一オブジェクト識別）にし、内部 Branch の else や
    無条件アクションには触らない（それらは H ではなく、誤検知の元になるため）。
  - 汎用盤面で発動し、`_smart_drain` 後に **ソース自身の登場移動以外の実盤面変化
    （キャラ増減/KO/ドン/ライフ/パワー/キーワード等）** があれば FALSE_PATH_LEAK。

先頭ゲートが能力全体を支配していれば、条件偽で一切動かない。もし B が漏れて走れば
（ability.condition が effect を覆い切れていない＝H 退行）ここで顕在化する。
静的な構造ゲート（test_structural_gate）と相補的なランタイム検証。

使い方:
    OPCG_LOG_SILENT=1 python tests/false_path_coverage.py
    OPCG_LOG_SILENT=1 python tests/false_path_coverage.py --show
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import effect_coverage as cov
from opcg_sim.src.models.effect_types import Branch, Choice, Sequence  # noqa: F401
from opcg_sim.src.models.enums import ConditionType
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "opcg_sim", "data")


def _is_gated(ab) -> bool:
    """ability レベルのゲート条件（先頭条件の引き上げ結果）を持つか。

    TURN_LIMIT / NONE のみの能力は実質ゲート無しとして除外する（偽にしても意味がない）。
    """
    c = getattr(ab, "condition", None)
    if c is None:
        return False
    t = getattr(c, "type", None)
    if t in (None, ConditionType.NONE, ConditionType.TURN_LIMIT):
        return False
    return True


def classify_false_path(master, ability, trig: str) -> str:
    """ability.condition のみを強制 False にして発動し、漏れ（実盤面変化）があるかを分類する。"""
    try:
        gm, p1, p2, source = cov._build_test_state(
            master, source_in_hand=(trig == "ON_PLAY"))
    except Exception as e:
        return f"SETUP_ERROR:{str(e)[:30]}"

    # この能力の top-level 条件だけを False に固定する（内部 Branch・else は通常評価）。
    gate = ability.condition
    orig = EffectResolver._check_condition

    def patched(self, player, condition, source_card, host_card=None):
        if condition is gate:
            return False
        return orig(self, player, condition, source_card, host_card)

    EffectResolver._check_condition = patched
    sb = cov._stat_snap(p1, p2)
    fb = cov._zone_fingerprint(p1, p2)
    try:
        if trig == "ON_PLAY":
            gm.play_card_action(p1, source)
        else:
            gm.resolve_ability(p1, ability, source)
        cov._smart_drain(gm)
    except Exception:
        return "ERROR"
    finally:
        EffectResolver._check_condition = orig

    if gm.active_interaction:
        return "INTERACTIVE"  # 偽パスでも対話に入る＝ゲート外の選択（別途扱い）
    sa = cov._stat_snap(p1, p2)
    fa = cov._zone_fingerprint(p1, p2)
    # ソース自身の登場移動（ON_PLAY のプレイ行為）は実行とみなさない。
    ignore = frozenset({source.uuid}) if trig == "ON_PLAY" else frozenset()
    fb2 = {k: v for k, v in fb.items() if not k.endswith(source.uuid)}
    fa2 = {k: v for k, v in fa.items() if not k.endswith(source.uuid)}
    if cov._moved(fb2, fa2) or cov._stat_changed(sb, sa, ignore):
        return "FALSE_PATH_LEAK"
    return "GATED_OK"


def collect(card_filter: Optional[str] = None):
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    buckets = defaultdict(list)
    total = len(card_ids)
    for i, cid in enumerate(card_ids, 1):
        if i % 300 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        for ab in master.abilities:
            if not _is_gated(ab):
                continue
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            if trig in ("PASSIVE", "YOUR_TURN", "OPPONENT_TURN"):
                continue  # 継続/静的系は別ハーネス
            status = classify_false_path(master, ab, trig)
            buckets[status].append(
                (cid, master.name, trig,
                 (getattr(ab.effect, "raw_text", "") or "")[:48]))
    sys.stderr.write(f"\r完了: {total} カード処理済み\n")
    return buckets


def run(show: bool, card_filter: Optional[str] = None):
    buckets = collect(card_filter)
    print("=== 条件“偽”パスの被覆（カテゴリH 再発防止）===")
    for k in ("GATED_OK", "FALSE_PATH_LEAK", "INTERACTIVE", "ERROR"):
        mark = "  ★漏れ（要修正）" if k == "FALSE_PATH_LEAK" else ""
        print(f"  {k:<18}: {len(buckets.get(k, [])):4d}{mark}")
    if show:
        for cid, name, trig, raw in buckets.get("FALSE_PATH_LEAK", []):
            print(f"  [LEAK] {cid:<12} {trig:<14} {name}  | {raw}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--card", default=None)
    args = ap.parse_args()
    run(show=args.show, card_filter=args.card)
