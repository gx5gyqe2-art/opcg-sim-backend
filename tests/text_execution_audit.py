"""テキスト↔実行 不一致 監査ハーネス。

カードの日本語テキスト（およびパース後の IR/AST）を解析し、「テキストが要求する効果」と
「エンジンが実際に行う動作」の不一致を **自動検出** してフラグを立てる。
1枚ずつ直す whack-a-mole を、全体像の見える burn down に変えるための検出エンジン。

フラグ:
  FLAG_OTHER         : AST に ActionType.OTHER が残る（未実装句）
  FLAG_HIDDEN_LEAK   : デッキ/ライフ(隠しゾーン)の「上から」を対話選択させてしまう（中身が見える）
  FLAG_DURATION      : 「このターン中/このバトル中」なのに duration が不一致 or INSTANT
  FLAG_COST_LIMIT    : 「〜以下のコスト/枚数以下」なのにコスト/枚数制限が IR に無い
  FLAG_TARGET_SIDE   : 「相手の/自分の」とターゲット player が逆
  FLAG_MISSING_ACTION: テキストの動詞キーワードに対応するアクションが AST に無い
  FLAG_SUSPEND_LEAK  : 発動後に未解決の interaction / temp_zone リークが残る（実行時）

実行:
  OPCG_LOG_SILENT=1 python tests/text_execution_audit.py                 # 全カード集計
  OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --flag HIDDEN_LEAK
  OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --deck imu nami # 実2デッキのみ
  OPCG_LOG_SILENT=1 python tests/text_execution_audit.py --card OP11-041
"""
import os
import re
import sys
import json
from collections import Counter, defaultdict

import conftest  # noqa: F401

from opcg_sim.src.models.enums import ActionType, Zone, Player
from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")


def _nfc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", s or "")


def walk(node):
    """AST を GameAction 単位で走査（cost/effect 兼用）。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from walk(a)
    elif isinstance(node, Branch):
        yield from walk(node.if_true)
        if node.if_false:
            yield from walk(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from walk(o)


# 動詞キーワード → そのテキストが要求する ActionType 群（いずれか実行されるべき）
_VERB_ACTIONS = {
    "KOする": {"KO"},
    "を引く": {"DRAW"},
    "ドロー": {"DRAW"},
    "手札に加え": {"MOVE_TO_HAND", "BOUNCE", "MOVE_CARD", "SEARCH", "DRAW"},
    "手札に戻": {"BOUNCE", "MOVE_TO_HAND", "MOVE_CARD"},
    "トラッシュに置": {"TRASH", "DISCARD", "TRASH_FROM_DECK", "KO", "MOVE", "MOVE_CARD"},
    "捨てる": {"DISCARD", "TRASH"},
    "レストにする": {"REST"},
    "アクティブにする": {"ACTIVE", "ACTIVE_DON"},
    "登場させ": {"PLAY_CARD"},
    "ライフの上に加え": {"MOVE_CARD", "HEAL", "MOVE"},
    "ダメージを与え": {"DEAL_DAMAGE"},
    "デッキの下に置": {"DECK_BOTTOM", "MOVE_CARD", "MOVE"},
    "公開": {"REVEAL", "LOOK", "MOVE_TO_HAND", "SEARCH"},
}


def audit_ability(text, ability):
    """1能力をテキストと突き合わせてフラグのリストを返す。"""
    flags = []
    actions = list(walk(ability.cost)) + list(walk(ability.effect))
    action_types = {a.type.name for a in actions if a and hasattr(a.type, "name")}
    # 能力ローカルのテキスト（無ければカード全文）でキーワード判定の誤検出を抑える
    ab_text = _nfc(getattr(ability, "raw_text", "") or "") or _nfc(text)

    # FLAG_OTHER
    if any(a.type == ActionType.OTHER for a in actions if a):
        flags.append(("FLAG_OTHER", ""))

    for a in actions:
        if not a:
            continue
        raw = _nfc(a.raw_text or "")
        tq = a.target

        # FLAG_HIDDEN_LEAK: デッキ/ライフの「上から/下から」(位置指定=不可視)を対話選択させる。
        #   「デッキから」(山札サーチ)や「見て/公開/選ぶ/上か下」は対象外（正当な対話）。
        if tq and getattr(tq, "zone", None) in (Zone.DECK, Zone.LIFE):
            mode = getattr(tq, "select_mode", "CHOOSE")
            positional = ("上から" in raw or "下から" in raw)
            interactive_words = ("見て" in raw or "公開" in raw or "選ぶ" in raw
                                 or "選び" in raw or "か下" in raw)
            if positional and mode == "CHOOSE" and not interactive_words and a.type != ActionType.LOOK:
                flags.append(("FLAG_HIDDEN_LEAK", f"{a.type.name} {tq.zone.name} '{raw[:30]}'"))

        # FLAG_DURATION
        dur = getattr(a, "duration", "INSTANT")
        if "このターン中" in raw and dur != "THIS_TURN":
            flags.append(("FLAG_DURATION", f"このターン中→{dur} '{raw[:30]}'"))
        elif "このバトル中" in raw and dur != "THIS_BATTLE":
            flags.append(("FLAG_DURATION", f"このバトル中→{dur} '{raw[:30]}'"))

        # FLAG_COST_LIMIT
        if tq and ("以下のコスト" in raw or "枚数以下" in raw or "枚数分以下" in raw):
            if getattr(tq, "cost_max", None) is None and getattr(tq, "cost_max_dynamic", None) is None:
                flags.append(("FLAG_COST_LIMIT", f"'{raw[:34]}'"))

        # FLAG_TARGET_SIDE: 「相手の(キャラ/リーダー)」を対象にするのに player=SELF。
        #   「相手の効果で」等の非対象節は除外。
        if tq is not None and getattr(tq, "player", None) == Player.SELF \
                and getattr(tq, "zone", None) == Zone.FIELD \
                and re.search(r"相手の(?!効果)[^。]*?(キャラ|リーダー)", raw):
            flags.append(("FLAG_TARGET_SIDE", f"相手の→SELF '{raw[:30]}'"))

    # FLAG_MISSING_ACTION: 能力テキストの動詞があるのに対応アクションが無い（OTHER 時は除外）
    if not any(a.type == ActionType.OTHER for a in actions if a):
        for kw, expected in _VERB_ACTIONS.items():
            if kw in ab_text and not (expected & action_types):
                flags.append(("FLAG_MISSING_ACTION", f"'{kw}' 期待{expected} 実際{sorted(action_types)}"))

    return flags


def run(flag_filter=None, card_filter=None, deck_filter=None):
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()

    card_ids = sorted(db.raw_db.keys())
    if deck_filter:
        ids = set()
        for d in deck_filter:
            j = json.load(open(os.path.join(DATA, f"{d}.json")))
            ids.add(j["leader"]["number"])
            for c in j["cards"]:
                ids.add(c["number"])
        card_ids = [c for c in card_ids if c in ids]
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    flag_counts = Counter()
    per_card = defaultdict(list)
    for cid in card_ids:
        m = db.get_card(cid)
        if not m or not m.abilities:
            continue
        text = getattr(m, "effect_text", "") or ""
        for ab in m.abilities:
            for fname, detail in audit_ability(text, ab):
                if flag_filter and fname != f"FLAG_{flag_filter}":
                    continue
                flag_counts[fname] += 1
                per_card[cid].append((fname, ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger), detail))

    print("=== テキスト↔実行 監査サマリ ===")
    print(f"  走査カード数: {len(card_ids)}")
    for fname in ("FLAG_OTHER", "FLAG_HIDDEN_LEAK", "FLAG_DURATION", "FLAG_COST_LIMIT",
                  "FLAG_TARGET_SIDE", "FLAG_MISSING_ACTION"):
        if flag_filter and fname != f"FLAG_{flag_filter}":
            continue
        print(f"  {fname:<22}: {flag_counts.get(fname, 0)}")
    print(f"  フラグの立ったカード数: {len(per_card)}")
    print()

    show = card_filter or deck_filter or flag_filter
    if show:
        for cid in sorted(per_card):
            m = db.get_card(cid)
            print(f"--- {cid} {m.name}")
            print(f"    TEXT: {(getattr(m,'effect_text','') or '')[:110]}")
            for fname, trig, detail in per_card[cid]:
                print(f"    [{fname}] ({trig}) {detail}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    flag = card = None
    deck = None
    i = 0
    while i < len(args):
        if args[i] == "--flag" and i + 1 < len(args):
            flag = args[i + 1]; i += 2
        elif args[i] == "--card" and i + 1 < len(args):
            card = args[i + 1]; i += 2
        elif args[i] == "--deck":
            deck = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                deck.append(args[i]); i += 1
        else:
            i += 1
    run(flag_filter=flag, card_filter=card, deck_filter=deck)
