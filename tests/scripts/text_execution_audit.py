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

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.models.enums import ActionType, Zone, Player, CardType
from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
from opcg_sim.src.utils.loader import CardLoader
from effect_coverage import _build_test_state

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "opcg_sim", "data")


def _nfc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", s or "")


def walk(node):
    """AST を GameAction 単位で走査（cost/effect/sub_effect 再帰）。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
        if node.sub_effect is not None:
            yield from walk(node.sub_effect)
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
    # REST_DON を含む: 「相手のドン!!1枚をレストにする」は REST_DON が正解
    "レストにする": {"REST", "REST_DON", "ATTACH_DON"},
    "アクティブにする": {"ACTIVE", "ACTIVE_DON"},
    # 「登場させた時」はトリガー条件なので別途除外ロジックで対処
    "登場させ": {"PLAY_CARD"},
    "ライフの上に加え": {"MOVE_CARD", "HEAL", "MOVE"},
    # 「ダメージを与えた時」はトリガー条件なので別途除外ロジックで対処
    "ダメージを与え": {"DEAL_DAMAGE"},
    "デッキの下に置": {"DECK_BOTTOM", "MOVE_CARD", "MOVE"},
    # 公開: FACE_UP_LIFE はライフを表向きで公開する動作、DECLARE_COST は公開+宣言の複合
    # MOVE_CARD: 「公開し手札に加える」はサーチ（MOVE_CARD）として実装されるケースも含む
    # LOOK_LIFE: 「ライフの上から1枚を公開」は engine が LOOK_LIFE で実装する（OP10-022/ST13-007 等）
    "公開": {"REVEAL", "LOOK", "LOOK_LIFE", "MOVE_TO_HAND", "SEARCH", "FACE_UP_LIFE", "DECLARE_COST", "MOVE_CARD"},
}

# 動詞がトリガー条件（〜した時）としてのみ現れる場合はチェックをスキップする
# 対: "登場させた時" / "ダメージを与えた時" など
_TRIGGER_CONDITION_SKIP = {
    "登場させ": (re.compile(_nfc(r"登場させた時")), re.compile(_nfc(r"登場させ(る|てもよい|ることができる)"))),
    "ダメージを与え": (re.compile(_nfc(r"ダメージを与えた時")), re.compile(_nfc(r"ダメージを与え(る|てもよい|ることができる)"))),
    "手札に戻": (
        re.compile(_nfc(r"手札に戻った時")),
        re.compile(_nfc(r"手札に戻(す|せる|すことができる)"))
    ),
}

# 動詞がテキスト中に現れるが、実アクションではない特定パターン
# (禁止節・受け身ルール・コスト修飾語など)
_VERB_EXTRA_SKIP = {
    "手札に加え":   re.compile(_nfc(r"手札に加えられない")),       # 禁止節
    "デッキの下に置": re.compile(_nfc(r"デッキの下に置かれる")),  # 受け身ルール
    "ドロー":       re.compile(_nfc(r"ドローフェイズ")),          # フェーズ名
    "登場させ":     re.compile(_nfc(r"登場させる[^。\n]*コスト")), # コスト低減の名詞節
    "を引く":       re.compile(_nfc(r"引くことができない")),       # ドロー禁止節
}


def runtime_hidden_leak(master, ability):
    """能力を発動し、デッキ/ライフ(隠しゾーン)の対象選択で中断する＝中身が見える、を検出。
    発動できない(コスト未達等)場合や例外は「リークなし」として扱う（保守的）。"""
    try:
        on_play = (ability.trigger.name if hasattr(ability.trigger, "name") else "") == "ON_PLAY"
        gm, p1, p2, src = _build_test_state(master, source_in_hand=on_play)
        if on_play and master.type != CardType.EVENT:
            gm.play_card_action(p1, src)
        else:
            gm.resolve_ability(p1, ability, src)
        n = 0
        while gm.active_interaction and n < 30:
            ia = gm.active_interaction
            q = (ia.get("continuation") or {}).get("query")
            if (q is not None and getattr(q, "zone", None) in (Zone.DECK, Zone.LIFE)
                    and "REVEAL_SELECT" not in getattr(q, "flags", set())):
                # REVEAL_SELECT は「自分のライフ／デッキを明示公開して選ぶ」正当な対話で、
                # 情報リークではない（resolver も同 flag で対話選択を許可する）。
                return True
            cand = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            try:
                gm.resolve_interaction(p1 if p1.name == ia.get("player_id") else p2,
                                       {"selected_uuids": cand[:1], "index": 0})
            except Exception:
                break
            n += 1
    except Exception:
        return False
    return False


def audit_ability(text, ability):
    """1能力をテキストと突き合わせてフラグのリストを返す。"""
    flags = []
    cost_actions = list(walk(ability.cost))
    effect_actions = list(walk(ability.effect))
    actions = cost_actions + effect_actions
    action_types = {a.type.name for a in actions if a and hasattr(a.type, "name")}
    # 能力ローカルのテキスト（無ければ内部アクションノードの raw_text、最終フォールバックはカード全文）
    _ab_rt = _nfc(getattr(ability, "raw_text", "") or "")
    if not _ab_rt:
        # Catalog entries: collect raw_text from inner action nodes
        inner_texts = [_nfc(a.raw_text or "") for a in actions if a and getattr(a, "raw_text", None)]
        _ab_rt = " ".join(inner_texts)
    if not _ab_rt:
        _ab_rt = _nfc(text)
    ab_text = _ab_rt

    # FLAG_OTHER
    if any(a.type == ActionType.OTHER for a in actions if a):
        flags.append(("FLAG_OTHER", ""))

    # FLAG_DURATION はコストアクションではなくエフェクトアクションのみチェック
    # （コスト句に「このターン中」が混入する誤検出を防ぐ）
    for a in effect_actions:
        if not a:
            continue
        raw = _nfc(a.raw_text or "")
        tq = a.target

        # REPLACE_EFFECT コンテナ自体の duration は不問（sub_effect が担う）
        if a.type == ActionType.REPLACE_EFFECT:
            continue
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
        #   除外:
        #    - 「相手の効果で」等の非対象節。
        #    - 対象が明示的に SOURCE（このキャラ自身）の場合（C9 同値パワー等）。
        #    - 直前の選択を参照する ref_id（「選んだキャラ」=保存済み対象を使うため player は無関係）。
        #    - 期間/タイミング句「(次の)相手の(ターン/エンドフェイズ)(終了時)(まで/中)」（matcher も
        #      player 判定からこれを除去する。残すと「このキャラのパワー+N」等で誤検出する）。
        raw_no_dur = re.sub(
            _nfc(r"(?:次の)?相手の(?:ターン|エンドフェイズ)(?:終了時)?(?:まで|中)"), "", raw)
        if tq is not None and getattr(tq, "player", None) == Player.SELF \
                and getattr(tq, "zone", None) == Zone.FIELD \
                and getattr(tq, "select_mode", None) != "SOURCE" \
                and not getattr(tq, "ref_id", None) \
                and re.search(r"相手の(?!効果)[^。]*?(キャラ|リーダー)", raw_no_dur):
            flags.append(("FLAG_TARGET_SIDE", f"相手の→SELF '{raw[:30]}'"))

    # FLAG_MISSING_ACTION: 能力テキストの動詞があるのに対応アクションが無い（OTHER 時は除外）
    if not any(a.type == ActionType.OTHER for a in actions if a):
        for kw, expected in _VERB_ACTIONS.items():
            if kw not in ab_text:
                continue
            # トリガー条件（〜した時）としてのみ現れる場合はスキップ
            if kw in _TRIGGER_CONDITION_SKIP:
                trig_pat, eff_pat = _TRIGGER_CONDITION_SKIP[kw]
                if trig_pat.search(ab_text) and not eff_pat.search(ab_text):
                    continue
            # 否定節・受け身ルール・コスト名詞節によるスキップ
            if kw in _VERB_EXTRA_SKIP:
                skip_pat = _VERB_EXTRA_SKIP[kw]
                # positive action formを確認する正規表現
                positive_forms = {
                    "手札に加え": _nfc(r"手札に加え(る|よい|られる|ることができる|た)"),
                    "デッキの下に置": _nfc(r"デッキの下に置(く|き[、。]|いた)"),
                    "ドロー": _nfc(r"ドロー(する|し[、。]|した|させる)"),
                    "登場させ": _nfc(r"登場させ(てもよい|ることができる|る[。、]|る$|た)"),
                    "を引く": _nfc(r"[^きれ]を引(く|いた|いて)"),
                }
                has_skip = skip_pat.search(ab_text)
                has_positive = bool(re.search(positive_forms.get(kw, r"$^"), ab_text)) if kw in positive_forms else False
                if has_skip and not has_positive:
                    continue
            if not (expected & action_types):
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
            ab_flags = audit_ability(text, ab)
            # FLAG_HIDDEN_LEAK は実行時に判定（隠しゾーンで実際に選択中断するか）
            if (not flag_filter or flag_filter == "HIDDEN_LEAK") and runtime_hidden_leak(m, ab):
                ab_flags.append(("FLAG_HIDDEN_LEAK", "deck/life 選択中断(中身が見える)"))
            for fname, detail in ab_flags:
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
