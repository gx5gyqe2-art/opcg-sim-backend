"""隠れミスターゲット／lift 不具合の検出ツール（OTHER指標に出ない不具合の可視化）。

`effect_diagnostics.py`（OTHER カウント）や `compare_parsers.py`（新規OTHER検知）では
**捕捉できない**意味的バグ（実行はされるが盤面操作が誤る／前段アクションが消失する）を、
全カード(opcg_cards.json)の生成 AST を走査して検出する。

検出する疑わしいパターン（detector）:
  A. PLAY_CARD で対象 zone=FIELD            … 場からは登場できない＝ほぼ誤ターゲット
  B. REVEALED_CARD_TRAIT がアビリティ条件へ lift … 公開(LOOK)消失＋条件評価の順序矛盾
  C. デッキ「公開/見て」を含むのに LOOK 無し  … 公開句がレガシーに落ちて LOOK が生成されていない
  D. TEMP を消費する操作なのに LOOK 無し     … TEMP が空のまま＝no-op（残り処理等が機能しない）

いずれも OTHER には現れないため、本ツールで件数とカード番号を棚卸しし、
PHoSv ブランチと同型（公開句の分割＋TEMP化、インライン Branch 化、対象 zone の正規化）で
順次是正する。回帰防止として件数のベースラインも出力する。

実行:
    OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py            # サマリ＋各 detector 上位
    OPCG_LOG_SILENT=1 python tests/mistarget_diagnostics.py --top 40   # 各 detector を40件表示
"""
import json
import os
import sys
import unicodedata
from collections import defaultdict

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.models.enums import ActionType, ConditionType, Zone

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "opcg_sim", "data"
)


def _nfc(s):
    return unicodedata.normalize("NFC", s or "")


def _walk_actions(node):
    """AST を走査して全 GameAction を yield する（sub_effect も含む）。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
        if getattr(node, "sub_effect", None) is not None:
            yield from _walk_actions(node.sub_effect)
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk_actions(a)
    elif isinstance(node, Branch):
        yield from _walk_actions(node.if_true)
        yield from _walk_actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk_actions(o)


def _condition_types(cond):
    """Condition（AND/OR は args を再帰）に含まれる ConditionType を集合で返す。"""
    if cond is None:
        return set()
    out = {cond.type}
    for a in getattr(cond, "args", None) or []:
        out |= _condition_types(a)
    return out


def _target_zones(action):
    """action.target.zone を Zone のリストで返す（zone は単体 or リスト）。"""
    tq = getattr(action, "target", None)
    if tq is None:
        return []
    z = getattr(tq, "zone", None)
    if z is None:
        return []
    return list(z) if isinstance(z, list) else [z]


# 「TEMP を populate する」アクション（これが無いと TEMP 消費は no-op）。
# LOOK（デッキ上→TEMP）と LOOK_LIFE（ライフ上→TEMP）の両方が TEMP を満たす。
_TEMP_POPULATORS = {ActionType.LOOK, ActionType.LOOK_LIFE}
# 「TEMP を消費する」操作で、消費元が TEMP のもの。
_TEMP_CONSUMER_ZONE = Zone.TEMP


# detector のキー（出力順）。
KEY_A = "A. PLAY_CARD zone=FIELD"
KEY_B = "B. REVEALED_CARD_TRAIT lift(公開消失)"
KEY_C = "C. デッキトップ公開/見て あるのに LOOK無し"
KEY_D = "D. TEMP消費だが LOOK無し(no-op)"


def scan():
    """全カードを走査し、detector 名 -> list[(number, name, snippet)] の dict を返す。"""
    path = os.path.join(DATA_DIR, "opcg_cards.json")
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    parser = EffectParserV2()

    # detector 名 -> list[(number, name, snippet)]
    hits = defaultdict(list)
    cards_scanned = 0

    for c in cards:
        number = c.get("number") or c.get("id") or "?"
        name = c.get("name") or ""
        text = c.get("効果(テキスト)") or ""
        trig = c.get("効果(トリガー)") or ""
        if (not text or text.strip() in ("なし", "None", "")) and not trig:
            continue
        cards_scanned += 1
        ntext = _nfc(text)

        abilities = []
        if text:
            abilities += parser.parse_card_text(text)
        if trig:
            abilities += parser.parse_card_text(trig, as_trigger=True)

        card_has_look = False
        for ab in abilities:
            actions = list(_walk_actions(ab.effect)) + list(_walk_actions(ab.cost))
            if any(a.type in _TEMP_POPULATORS for a in actions):
                card_has_look = True

        for ab in abilities:
            actions = list(_walk_actions(ab.effect)) + list(_walk_actions(ab.cost))
            ab_has_look = any(a.type in _TEMP_POPULATORS for a in actions)
            cond_types = _condition_types(getattr(ab, "condition", None))

            # A. PLAY_CARD で対象 zone=FIELD（場からは登場できない）。
            #    ただし「このカードを登場させる」（自己登場 play_self, ref_id=self / SOURCE）は正当
            #    なので除外する（自身を場に出す＝ライフ/手札からの自己登場）。
            for a in actions:
                if a.type != ActionType.PLAY_CARD or Zone.FIELD not in _target_zones(a):
                    continue
                tq = getattr(a, "target", None)
                if getattr(tq, "ref_id", None) == "self" or getattr(tq, "select_mode", None) == "SOURCE":
                    continue  # play_self（自己登場）は正当
                hits[KEY_A].append((number, name, _nfc(a.raw_text)))
                break

            # B. REVEALED_CARD_TRAIT がアビリティ条件へ lift（公開 LOOK が同一能力に無い）
            if ConditionType.REVEALED_CARD_TRAIT in cond_types and not ab_has_look:
                hits[KEY_B].append(
                    (number, name, _nfc(getattr(ab.condition, "raw_text", "")))
                )

            # D. TEMP を消費する操作なのに同一能力に LOOK 無し（TEMP 空＝no-op）
            for a in actions:
                if _TEMP_CONSUMER_ZONE in _target_zones(a) and not ab_has_look:
                    hits[KEY_D].append(
                        (number, name, _nfc(a.raw_text))
                    )
                    break

        # C. デッキトップを「公開/見て」るのに LOOK 無し（カード単位）。
        #    「デッキの上からN枚／一番上を 公開/見て」に限定（手札公開・任意コスト宣言の
        #    相手デッキ公開などは別系統なので除外）。
        import re as _re
        deck_top_look = bool(
            _re.search(r"デッキの(?:上から\d+枚(?:まで)?|一番上)を(?:公開|見て)", ntext)
        )
        # 任意コスト宣言（DECLARE_COST）は、相手デッキトップの公開を AST の LOOK ではなく
        # 宣言インタラクションの resume フック（gamestate.resolve_interaction）で行う
        # 設計のため、LOOK 不在は no-op ではない（test_declare_cost_reveal_and_match で
        # エンドツーエンド検証済み）。
        card_has_declare = any(
            a.type == ActionType.DECLARE_COST
            for ab in abilities
            for a in list(_walk_actions(ab.effect)) + list(_walk_actions(ab.cost))
        )
        if deck_top_look and not card_has_look and not card_has_declare:
            hits[KEY_C].append((number, name, ntext[:60]))

    return cards_scanned, hits


def run(top: int = 25):
    cards_scanned, hits = scan()

    # ---- 出力 ----
    print("=== 隠れミスターゲット／lift 不具合 検出 (opcg_cards.json) ===")
    print(f"走査カード数 : {cards_scanned}")
    print()
    print("--- detector 別 件数（カード単位）---")
    order = [KEY_A, KEY_B, KEY_C, KEY_D]
    for key in order:
        uniq = {h[0] for h in hits.get(key, [])}
        print(f"  {len(uniq):4d}  {key}")
    print()

    for key in order:
        rows = hits.get(key, [])
        # カード番号で重複排除（同一カードで複数句ヒットは1行に）
        seen = set()
        uniq_rows = []
        for num, nm, snip in rows:
            if num in seen:
                continue
            seen.add(num)
            uniq_rows.append((num, nm, snip))
        print(f"--- {key} : {len(uniq_rows)}枚 上位{top} ---")
        for num, nm, snip in uniq_rows[:top]:
            print(f"  {num:10s} {nm[:14]:14s} {snip[:54]}")
        print()


if __name__ == "__main__":
    top = 25
    if "--top" in sys.argv:
        try:
            top = int(sys.argv[sys.argv.index("--top") + 1])
        except (ValueError, IndexError):
            pass
    run(top)
