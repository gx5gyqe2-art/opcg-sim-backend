import re
from typing import List, Optional
from ...models.effect_types import (
    Ability, EffectNode, GameAction, Sequence, Branch, Choice, ValueSource, TargetQuery, Condition, _nfc
)
from ...models.enums import ActionType, TriggerType, ConditionType, Zone, CompareOperator, Player
from .matcher import parse_target
from ...utils.logger_config import log_event

class EffectParser:
    def __init__(self):
        pass

    # キーワード能力タグのみのセグメントパターン（Ability オブジェクト生成不要）
    # 括弧は半角・全角どちらも受け付ける
    _KEYWORD_ONLY_RE = re.compile(
        r'^【(?:ブロッカー|速攻(?:[：:キャラ]*)?|ダブルアタック|バニッシュ|ブロック不可|フィルム|貫通|シフト|ルール)】(?:[（(][^）)]*[）)])?$'
    )
    # キーワード説明の括弧書き（全角・半角）
    _PAREN_ONLY_RE = re.compile(r'^[（(].+[）)]$')
    # コスト・制限タグ
    _DON_TURN_TAG_RE = re.compile(r'【(?:ターン1回|ドン[ ]*(?:!!|‼)[ ]*[××][ ]*\d+)】')
    # 既知のトリガータグ（これ以外の【X】はトリガーではない）
    _TRIGGER_TAG_RE = re.compile(
        r'【(?:登場時|起動メイン|メイン|アタック時|ブロック時|KO時|自分のターン終了時|'
        r'相手のターン終了時|相手のアタック時|自分のターン中|相手のターン中|'
        r'カウンター|トリガー|ゲーム開始時)】'
    )

    # テキスト埋め込みトリガー「〈timing〉時、発動できる」のうち、エンジンが実際に
    # ディスパッチする timing → TriggerType。これに該当すれば自動発動するよう上書きする。
    # （非ディスパッチの timing は既存トリガーを維持し、PASSIVE のみ手動発動へ退避する。）
    _DISPATCHED_TEXT_TRIGGERS = (
        ("相手のキャラがアタックした時", "ON_OPP_ATTACK"),
        ("相手がアタックした時", "ON_OPP_ATTACK"),
        ("ライフが離れた時", "ON_LIFE_DECREASE"),
    )

    def _strip_text_trigger(self, trigger, effect_text: str):
        """効果文先頭の埋め込みトリガー宣言「〈timing〉時、発動できる」/「〈cond〉場合、発動できる」/
        単独「発動できる」を取り除く。

        従来これらは「発動できる」が動詞なしの原子句 → ActionType.OTHER（サイレント失敗）に
        落ち、後続の本体効果のみが効いていた。トリガー宣言を解消して:
          - 「〈cond〉場合、発動できる」→ 条件を残し「発動できる」のみ除去（Branch-lift に委ねる）
          - 「〈timing〉時、発動できる」→ 節ごと除去。ディスパッチ対象 timing はトリガー上書き。
            非ディスパッチ timing は既存トリガー維持（PASSIVE のみ ACTIVATE_MAIN に退避＝
            常時誤発動を避けつつ手動発動可能にする）。
        戻り値: (新トリガー, 残りの効果テキスト, 発動任意フラグ)。
        発動任意フラグは、自動誘発トリガー（ディスパッチ対象 timing）の「発動できる」を
        上書きした場合のみ True（任意発動＝使用確認が必要。OP11-041 等）。
        """
        t = effect_text
        # 「〈X〉(時|場合)、発動できる」または 先頭「発動できる」のみにマッチ（誤検知防止に
        # 「を発動できる」等の効果動詞形は対象外＝直前は「時、」「場合、」か文頭に限定）。
        m = re.match(_nfc(r"^((?:[^。]*?(?:時|場合))、)?発動できる(?:ことができる)?[。、]?"), t)
        if not m:
            return trigger, t, False
        prefix = m.group(1) or ""
        remainder = t[m.end():].strip()
        # 「…場合、発動できる」→ 条件部を残して Branch-lift に任せる
        if _nfc("場合") in prefix:
            return trigger, (prefix + remainder).strip(), False
        new_trigger = trigger
        activation_optional = False
        from ...models.enums import TriggerType as _TT
        for key, tt_name in self._DISPATCHED_TEXT_TRIGGERS:
            if _nfc(key) in prefix:
                new_trigger = getattr(_TT, tt_name)
                activation_optional = True
                break
        else:
            if trigger == _TT.PASSIVE:
                # 非ディスパッチ timing × PASSIVE は常時誤発動の温床 → 手動発動へ退避
                new_trigger = _TT.ACTIVATE_MAIN
        return new_trigger, remainder, activation_optional

    def parse_card_text(self, text: str, as_trigger: bool = False) -> List[Ability]:
        norm = _nfc(text)
        if not norm or norm.strip() in ['なし', 'None', '']:
            return []

        segments = re.split(r'\s*/\s*|\n', norm)
        segments = [s.strip() for s in segments if s.strip()]

        # 「以下から…選ぶ」の選択肢項目（` / ` で別セグメントに分かれた「・」項目や
        # 条件付き項目）を、Choice を導入する親セグメントへ `\n` で再結合する。
        # 従来は別 Ability として分割→破棄され、options が空の Choice になっていた（難所）。
        # 新しい 【...】 タグで始まるセグメントは別能力なので再結合を打ち切る。
        merged: List[str] = []
        absorbing = False
        for seg in segments:
            opens_new_ability = seg.startswith(_nfc('【'))
            if absorbing and not opens_new_ability:
                merged[-1] = merged[-1] + '\n' + seg
                continue
            merged.append(seg)
            # 「以下から…選ぶ」(Choice) と「…によって以下の効果をそれぞれ適用する」
            # (段階効果 Sequence-of-Branch) の双方で後続の「・」項目を本体へ吸収する。
            absorbing = bool(re.search(_nfc(r'以下から.{0,6}?選ぶ'), seg)
                             or re.search(_nfc(r'以下の効果を.{0,4}?適用する'), seg))
        segments = merged

        # 「【メイン】/【カウンター】<効果>」のように、本体を持たない先頭トリガータグだけの
        # セグメントは、次セグメント（同一の効果本体）を共有する *別トリガーの能力*。
        # `/` 分割で「【メイン】」が本体なしの別セグメントに割れ、ACTIVATE_MAIN の effect が
        # None になっていた（焔裂き 等）。次セグメントの本体を借りて展開する。
        # キーワードのみタグ（【ブロッカー】等）は効果共有ではないため対象外。
        expanded: List[str] = []
        for i, seg in enumerate(segments):
            # 「【自分のターン中】【登場時】」のような複数タグのみのセグメントも本体共有の
            # 対象にする（従来は単一タグ限定で、OP08-007 の ON_PLAY 側が effect=None になった）。
            is_lone_tag = bool(re.fullmatch(_nfc(r'(?:【[^】]+】)+'), seg)) and not self._KEYWORD_ONLY_RE.match(seg)
            if (is_lone_tag and i + 1 < len(segments)
                    and segments[i + 1].startswith(_nfc('【'))
                    and not self._KEYWORD_ONLY_RE.match(segments[i + 1])):
                body = re.sub(_nfc(r'^【[^】]+】'), '', segments[i + 1])
                expanded.append(seg + body)
            else:
                expanded.append(seg)
        segments = expanded

        abilities = []
        for seg in segments:
            # キーワード能力宣言 / キーワード説明括弧書きはスキップ（Ability 不要）
            if self._KEYWORD_ONLY_RE.match(seg):
                continue
            if self._PAREN_ONLY_RE.match(seg) and not re.search(r'【[^】]+】', seg):
                continue
            # 「・」で始まる選択肢セグメント（再結合できなかった孤立項目）はスキップ
            if seg.startswith(_nfc('・')):
                continue
            try:
                ab = self.parse_ability(seg)
                if ab.trigger != TriggerType.UNKNOWN or ab.effect is not None:
                    abilities.append(ab)
            except Exception as e:
                log_event("WARNING", "parser.segment_skip", f"Skipped segment: {seg[:30]} | {e}")

        if as_trigger:
            for ab in abilities:
                ab.trigger = TriggerType.TRIGGER

        return abilities

    def parse_ability(self, text: str) -> Ability:
        log_event("DEBUG", "parser.input", f"Input text: {text[:50]}")
        try:
            norm_text = _nfc(text)

            # スコープ付き効果無効「相手の【登場時】効果は無効になる」の【登場時】を非タグ化して
            # 保全する（後段の clean_text がタグを除去するとスコープが失われ、全効果無効と
            # 区別できなくなるため）。同時に、この【登場時】がトリガー誤検出（ON_PLAY）の原因に
            # なっていたのを解消する（このセグメントの真のトリガーは【起動メイン】等）。OP09-081。
            norm_text = re.sub(_nfc(r'(相手の)【(登場時)】(効果)'), r'\1\2\3', norm_text)

            # 参照発動「このカードの【登場時】/【KO時】効果を発動する」の参照タグも
            # 非タグ化して保全する（clean_text のタグ除去で参照先が消え、常に
            # ACTIVATE_MAIN を展開して no-op になっていた。OP16-102 等 15 枚）。
            norm_text = re.sub(
                _nfc(r'(この(?:カード|キャラ)の)【(登場時|KO時|アタック時|起動メイン|メイン)】(効果)'),
                r'\1\2\3', norm_text)

            # 「ルール上、…できず、ゲーム開始時、…」（OP13-079）: 前半はデッキ構築制限で
            # ランタイム効果ではない。残すと parse_target が後半の対象解析を汚染する
            # （EVENT/cost_min が混入）ため、ゲーム開始時節が続く場合のみ除去する。
            # 単独の「ルール上、…」節（他18枚）は rule_processing ルールが担当するため不変。
            norm_text = re.sub(_nfc(r'^ルール上、[^。]*?できず、(?=ゲーム開始時)'), '', norm_text)

            # トリガー検出は前処理前のテキストで行う（コスト/制限タグ除去前に判定）
            trigger = self._detect_trigger(norm_text)
            log_event("INFO", "parser.trigger", f"Detected trigger: {trigger.name}")

            # 【ターン1回】を前処理で検出し条件として記憶
            turn_limit_cond = None
            if _nfc("【ターン1回】") in norm_text or _nfc("(ターンに1回)") in norm_text or _nfc("（ターンに1回）") in norm_text:
                turn_limit_cond = Condition(type=ConditionType.TURN_LIMIT, value=1)
                norm_text = re.sub(_nfc(r'【ターン1回】|\(ターンに1回\)|（ターンに1回）'), '', norm_text).strip()

            # 【ドン!!×N】コストタグを前処理で検出（スペース有無・‼ 混在に対応）
            don_cost_value = 0
            don_match = re.search(_nfc(r'【ドン[ ]*(?:!!|‼)[ ]*[××][ ]*(\d+)】'), norm_text)
            if don_match:
                don_cost_value = int(don_match.group(1))
                norm_text = norm_text.replace(don_match.group(0), '').strip()

            # ターン文脈タグ＋イベントトリガーの二重タグ（例【相手のターン中】【KO時】）は、
            # イベント側がトリガーになりターン文脈が脱落していた。タグ除去前に有無を記録し、
            # 最終トリガーがターン常時型（YOUR_TURN/OPPONENT_TURN）でなければ CONTEXT 条件
            # として ability.condition に統合する（resolver が turn_player で評価する）。
            has_self_turn_tag = _nfc("【自分のターン中】") in norm_text
            has_opp_turn_tag = _nfc("【相手のターン中】") in norm_text

            # トリガー/注釈タグの除去。
            # ただしキーワード能力タグ（【ブロッカー】等）は効果本体で「付与」を
            # 表すため保持する。脱落すると原子句が「このキャラはを得る」になり、
            # GRANT_KEYWORD（付与すべきキーワード）を復元できなくなる（既知バグ）。
            # 「【X】を持つカード」（プロパティ参照: 例 OP16-080「【トリガー】を持つカード」）の
            # タグも保持する（脱落すると対象の絞り込みが消える）。トリガー検出は前処理前の
            # テキストで済んでおり、トリガー宣言の【トリガー】prefix は「を持つ」が続かないため
            # ここでは除去される。
            clean_text = re.sub(
                _nfc(r'【(?!ブロッカー|速攻|ダブルアタック|バニッシュ|ブロック不可|貫通|シフト)[^】]*?】(?!を持つ|効果を持たない)'),
                '', norm_text
            ).strip()

            cost_node = None
            effect_text = clean_text
            cost_optional = False

            # コストの分離（全角コロン優先、なければ半角コロン）。
            # ただし【速攻：キャラ】のようにキーワード能力タグ内部の「：」は
            # コスト区切りではないため、タグ内部をマスクした上で判定する。
            masked = re.sub(r'【[^】]*】', lambda m: '〇' * len(m.group(0)), clean_text)
            colon = _nfc("：") if _nfc("：") in masked else (_nfc(":") if _nfc(":") in masked else None)
            cost_gate_cond = None
            if colon:
                idx = masked.index(colon)
                cost_text = clean_text[:idx]
                # コスト節先頭のゲート条件「〜の場合、」を ability 条件へ引き上げる（OP11-103 等）。
                cost_gate_cond, cost_text = self._extract_leading_condition(cost_text)
                cost_node = self._parse_cost_node(cost_text)
                effect_text = clean_text[idx + 1:]
                # 「〜できる：」「〜してもよい：」のコスト句は任意（払うかをプレイヤーが選ぶ）。
                # 自動誘発トリガーでは発動前に使用確認を挟むためのフラグ（resolver が参照）。
                if _nfc("できる") in cost_text or _nfc("してもよい") in cost_text:
                    cost_optional = True

            # 【ドン!!×N】は「このカードにドン!!がN枚以上付与されている」発動条件であり、
            # コストエリアのドンをレストにする支払いではない（従来は REST_DON コストに誤変換し、
            # 自分のターンは active ドンを消費し、相手ターンは active 0 で常に不発になっていた）。
            # → HAS_DON 条件（付与ドン≧N）として ability.condition に統合する（resolver が
            #    source_card.attached_don を参照して評価する）。
            don_cond = None
            if don_cost_value > 0:
                don_cond = Condition(
                    type=ConditionType.HAS_DON,
                    value=don_cost_value,
                    operator=CompareOperator.GE,
                    raw_text=_nfc(f"【ドン!!×{don_cost_value}】"),
                )

            # 効果文先頭の「〈イベント〉した時」（ドン返却/退場/捨て/トリガー登場 等）を EVENT_THIS_TURN
            # 条件として引き上げ、効果本体から除去する。エンジンが該当イベントを記録していなければ
            # 発動しない（OP06-042/OP07-038/OP12-040/OP13-100/OP09-061 の「イベント無しでは不発」）。
            event_cond, effect_text = self._extract_event_condition(effect_text)

            # テキスト埋め込みトリガー宣言「〈timing〉時、発動できる」を解消（OTHER 化を防ぐ）。
            trigger, effect_text, activation_optional = self._strip_text_trigger(trigger, effect_text)

            # テキスト埋め込み「ゲーム開始時、」のトリガー宣言を効果本体から除去する
            # （残すと parse_target が宣言部まで対象解析してしまう）。
            if trigger == TriggerType.GAME_START:
                effect_text = re.sub(_nfc(r'^ゲーム開始時、'), '', effect_text).strip()

            # 効果本体の解析
            # 埋め込み反応型トリガー（本文の「〜時、」）が認識された能力では、効果文先頭に残る
            # そのトリガー句を除去する。残すと parse_target が KO 条件のフィルタ（特徴/パワー/枚数）を
            # 効果の対象に取り込んでしまう（OP14-041: 「…パワー5000以上の…キャラがKOされた時、相手の
            # ライフ1枚を手札に加える」で MOVE_CARD 対象にアマゾン・リリー/九蛇/power5000 が混入）。
            if trigger in (TriggerType.ON_KO, TriggerType.ON_DAMAGE_DEALT_TO_LIFE,
                           TriggerType.ON_LEAVE, TriggerType.ON_EVENT_PLAY, TriggerType.ON_OPP_PLAY):
                effect_text = re.sub(
                    _nfc(r'^[^。：:]*?(?:された|なった|与えた|離れた|発動した|登場させた)時、'),
                    '', effect_text).strip()

            # 効果先頭のゲート条件「〜の場合、」は、後続が単一文（内部に「。」が無く、連用形で
            # 連なる複数アクション）のとき ability.condition へ引き上げる。従来は先頭アクションのみ
            # 条件付きの内部 Branch になり、後続アクションが無条件化していた
            # （ST29-001「ライフ2枚以下なら引いて捨てる」の DISCARD が無条件化）。
            # 純粋なゲーム状態条件（ターン数 等）は能力全体に掛かるため、複数文でも引き上げる。
            _GLOBAL_GATE = (ConditionType.TURN_COUNT,)
            effect_gate_cond = None
            _lead_cond, _lead_rest = self._extract_leading_condition(effect_text)
            if _lead_cond is not None and (
                    _lead_cond.type in _GLOBAL_GATE
                    or (_nfc("。") not in _lead_rest.rstrip(_nfc("。"))
                        and re.search(_nfc(r'(し|き|り|め|ち|に|べ|ね|げ| じ)[、，]'), _lead_rest))):
                effect_gate_cond = _lead_cond
                effect_text = _lead_rest

            effect_node = self._parse_to_node(effect_text)

            # ゲーム開始時のデッキサーチはルール上シャッフルを伴う（OP13-079）。
            if (trigger == TriggerType.GAME_START and effect_node is not None
                    and _nfc("デッキから") in effect_text):
                effect_node = Sequence(actions=[
                    effect_node,
                    GameAction(type=ActionType.SHUFFLE,
                               raw_text=_nfc("デッキをシャッフルする")),
                ])

            # 先頭のゲート条件（「〜の場合、」）を ability.condition に引き上げる
            final_condition = turn_limit_cond
            ctx_cond = None
            if trigger not in (TriggerType.YOUR_TURN, TriggerType.OPPONENT_TURN):
                if has_opp_turn_tag:
                    ctx_cond = Condition(type=ConditionType.CONTEXT, value="OPPONENT_TURN",
                                         raw_text=_nfc("【相手のターン中】"))
                elif has_self_turn_tag:
                    ctx_cond = Condition(type=ConditionType.CONTEXT, value="SELF_TURN",
                                         raw_text=_nfc("【自分のターン中】"))
            if ctx_cond is not None:  # ターン文脈タグの CONTEXT 条件を統合
                final_condition = ctx_cond if final_condition is None else Condition(
                    type=ConditionType.AND, args=[final_condition, ctx_cond])
            if don_cond is not None:  # 【ドン!!×N】の付与ドン条件を統合
                final_condition = don_cond if final_condition is None else Condition(
                    type=ConditionType.AND, args=[final_condition, don_cond])
            if event_cond is not None:  # 効果節先頭から引き上げたターン内イベント条件
                final_condition = event_cond if final_condition is None else Condition(
                    type=ConditionType.AND, args=[final_condition, event_cond])
            if effect_gate_cond is not None:  # 効果節先頭から引き上げたゲート条件（単一文・複数アクション）
                final_condition = effect_gate_cond if final_condition is None else Condition(
                    type=ConditionType.AND, args=[final_condition, effect_gate_cond])
            if cost_gate_cond is not None:  # コスト節から引き上げた条件
                final_condition = cost_gate_cond if final_condition is None else Condition(
                    type=ConditionType.AND, args=[final_condition, cost_gate_cond])
            if isinstance(effect_node, Branch) and effect_node.if_false is None and effect_node.condition is not None:
                if final_condition is None:
                    final_condition = effect_node.condition
                else:
                    final_condition = Condition(
                        type=ConditionType.AND,
                        args=[final_condition, effect_node.condition]
                    )
                effect_node = effect_node.if_true

            # 自動誘発の「〈timing〉時、発動できる」は任意発動（resolver が CONFIRM_OPTIONAL で
            # 使用確認を挟む）。複合効果の部分拒否の曖昧さを避けるため単一アクションに限定する。
            if (activation_optional and isinstance(effect_node, GameAction)
                    and effect_node.type != ActionType.OTHER):
                effect_node.is_optional = True

            # 置換効果（「(このキャラ/他のキャラが)KOされる/場を離れる場合、代わりに〜」）。
            # 「…される場合」はゲート条件ではなくトリガー文脈なので、REPLACE_EFFECT で
            # 包み（置換アクションは sub_effect に保持）、PASSIVE 能力として扱う。
            # 自身の置換（このキャラ）は条件が status に包含されるので ab.condition は不要。
            # 他のキャラを守る型は OPPONENT_REMOVAL 条件を保持し、_active_replacement で評価。
            repl_status = self._replacement_status(_nfc(text))
            if repl_status and effect_node is not None:
                effect_node = GameAction(
                    type=ActionType.REPLACE_EFFECT,
                    status=repl_status,
                    sub_effect=effect_node,
                    raw_text=_nfc(text),
                )
                if _nfc("このキャラ") in _nfc(text):
                    final_condition = None
                trigger = TriggerType.PASSIVE

            return Ability(
                trigger=trigger,
                condition=final_condition,
                cost=cost_node,
                effect=effect_node,
                cost_optional=cost_optional,
                raw_text=_nfc(text)
            )
        except Exception as e:
            log_event(level_key="ERROR", action="parser.parse_ability_error", msg=f"Failed to parse: {text[:20]} | Error: {str(e)}")
            return Ability(trigger=TriggerType.UNKNOWN, effect=None, raw_text=_nfc(text))

    def _replacement_status(self, norm_text: str) -> Optional[str]:
        """置換効果（「代わりに〜」）の対象除去種別を返す。

        - 「バトルでKOされる」→ "BATTLE_KO"（戦闘KOの置換）
        - 「(相手の効果で)場を離れる / KOされる」→ "LEAVE"（相手効果による除去の置換）
        該当しなければ None。
        """
        if _nfc("代わりに") not in norm_text:
            return None
        if _nfc("場を離れる") not in norm_text and _nfc("KOされ") not in norm_text:
            return None
        return "BATTLE_KO" if _nfc("バトル") in norm_text else "LEAVE"

    # 効果文先頭の「〈イベント〉した時、」をターン内イベント条件へ写像する定義。
    # (正規表現, イベント名, 最小回数)。先頭マッチのみ採用し、本文から除去する。
    _EVENT_CLAUSE_PATTERNS = (
        (r'^自分の場のドン[ 　]*(?:!!|‼)?が(\d+)枚以上[^。]*?ドン[ 　]*(?:!!|‼)?デッキに戻された時、', "DON_RETURNED", None),
        (r'^自分の場のドン[ 　]*(?:!!|‼)?が[^。]*?ドン[ 　]*(?:!!|‼)?デッキに戻された時、', "DON_RETURNED", 1),
        (r'^[^。]*?キャラが自分の効果で場を離れた時、', "CHAR_LEFT_BY_OWN_EFFECT", 1),
        (r'^自分の特徴《海軍》を持つカードの効果で[^。]*?捨てられた時、', "NAVY_DISCARD", 1),
        (r'^自分の【トリガー】を持つキャラが登場した時、', "TRIGGER_CHAR_PLAYED", 1),
    )

    def _extract_event_condition(self, effect_text: str):
        """効果文先頭の「〈イベント〉した時、」を (EVENT_THIS_TURN 条件, 残りテキスト) に分離する。
        該当しなければ (None, 元テキスト)。「発動できる」が続く形（OP07-038/OP13-100）にも対応。"""
        t = _nfc(effect_text)
        for pat, ev_name, fixed_min in self._EVENT_CLAUSE_PATTERNS:
            m = re.match(_nfc(pat), t)
            if not m:
                continue
            ev_min = fixed_min if fixed_min is not None else int(m.group(1))
            rest = t[m.end():].strip()
            # 「〜時、発動できる。」形は「発動できる。」を取り除いて本体を残す。
            rest = re.sub(_nfc(r'^発動できる[。、]?'), '', rest).strip()
            cond = Condition(type=ConditionType.EVENT_THIS_TURN, value=(ev_name, ev_min),
                             player=Player.SELF, raw_text=t)
            return cond, rest
        return None, effect_text

    def _extract_leading_condition(self, text: str):
        """テキスト先頭のゲート条件「〜の場合、/〜なら、」を (Condition, 残りテキスト) に分離する。

        条件が解釈できない（GENERIC）場合は引き上げず (None, 元テキスト) を返す（誤抽出防止）。
        コスト節先頭の条件（例:「自分のリーダーが「しらほし」の場合、…できる」）を
        ability.condition へ引き上げるために使う。
        """
        norm = _nfc(text)
        m = re.match(_nfc(r'^(.+?)(?:の場合|なら)、(.+)$'), norm, re.DOTALL)
        if not m:
            return None, text
        cond = self._parse_condition_obj(m.group(1))
        if cond is None or cond.type == ConditionType.GENERIC:
            return None, text
        return cond, m.group(2)

    def _parse_cost_node(self, cost_text: str) -> Optional[EffectNode]:
        """
        コストテキストを解析する。
        「このキャラをレストにできる」「このリーダーをレストにできる」パターンを
        ref_id="self" の REST アクションとして処理する。
        「ドン!!-N,<追加コスト>」パターンを RETURN_DON + 追加コストの Sequence として処理する。
        「N(レスト説明文),追加コスト」パターンを REST_DON + 追加コストの Sequence として処理する。
        """
        norm = _nfc(cost_text)

        # 「N(コストエリアの説明文),追加コスト」: REST_DON + 追加コストを Sequence に分割
        # 例: 3(コストエリアのドン!!を指定の数レストにできる),自分の手札1枚を捨てることができる
        # 例: ①(コストエリアのドン!!を指定の数レストにできる),このキャラを手札に戻すことができる
        num_paren_m = re.match(
            _nfc(r'([①-⑨⑩➀-➉]|\d+)(\([^)]*\))[、，,　 ]+(.+)'),
            norm, re.DOTALL
        )
        if num_paren_m and _nfc('レスト') in num_paren_m.group(2):
            num_paren_part = num_paren_m.group(1) + num_paren_m.group(2)
            add_cost_part = num_paren_m.group(3).strip()
            num_node = self._parse_to_node(num_paren_part, is_cost=True)
            add_node = self._parse_to_node(add_cost_part, is_cost=True)
            if num_node is not None and add_node is not None:
                num_acts = num_node.actions if isinstance(num_node, Sequence) else [num_node]
                add_acts = add_node.actions if isinstance(add_node, Sequence) else [add_node]
                return Sequence(actions=num_acts + add_acts)
            return num_node if num_node is not None else add_node

        # 「ドン!!-N,<追加コスト>」: ドン!!返却＋追加コストを Sequence に分割
        # 「ドン!!-N(説明文),追加コスト」のように括弧付き説明が挟まる表記も対応
        don_prefix_m = re.match(
            _nfc(r'(ドン[ 　]*(?:!!|‼)[ 　]*[-－−‐][ 　]*(\d+))(?:\([^)]*\))?[、，,　 ]+(.+)'),
            norm, re.DOTALL
        )
        if don_prefix_m:
            don_count = int(don_prefix_m.group(2))
            rest_part = don_prefix_m.group(3).strip()
            don_action = GameAction(
                type=ActionType.RETURN_DON,
                value=ValueSource(base=don_count),
                raw_text=don_prefix_m.group(1),
            )
            rest_node = self._parse_to_node(rest_part, is_cost=True)
            if rest_node is not None:
                if isinstance(rest_node, Sequence):
                    return Sequence(actions=[don_action] + rest_node.actions)
                return Sequence(actions=[don_action, rest_node])
            return don_action

        m = re.search(_nfc(r'この(?:キャラ|リーダー)をレストに(し[、，]|できる|する)'), norm)
        if m:
            rest_action = GameAction(
                type=ActionType.REST,
                target=TargetQuery(
                    player=Player.SELF,
                    zone=Zone.FIELD,
                    count=1,
                    is_strict_count=True,
                    ref_id="self"
                ),
                raw_text=norm
            )
            before_part = norm[:m.start()].strip('、，。 ')
            remainder = norm[m.end():].strip('、，。 ')

            actions = []
            if before_part:
                before_node = self._parse_to_node(before_part, is_cost=True)
                if isinstance(before_node, Sequence):
                    actions.extend(before_node.actions)
                elif before_node is not None:
                    actions.append(before_node)
            actions.append(rest_action)
            if remainder and remainder not in ('ことができる', 'できる'):
                rest_node = self._parse_to_node(remainder, is_cost=True)
                if isinstance(rest_node, Sequence):
                    actions.extend(rest_node.actions)
                elif rest_node is not None:
                    actions.append(rest_node)
            return Sequence(actions=actions) if len(actions) > 1 else (actions[0] if actions else rest_action)

        return self._parse_to_node(norm, is_cost=True)

    def _detect_trigger(self, text: str) -> TriggerType:
        norm_text = _nfc(text)
        # 「【登場時】効果を持たない」は対象修飾（トリガーではない）。能力トリガーの【登場時】のみ拾う
        # （PRB01-001「【起動メイン】…【登場時】効果を持たないキャラ…」が ON_PLAY 化するのを防ぐ）。
        if re.search(_nfc(r'【登場時】(?!効果を持たない)'), norm_text): return TriggerType.ON_PLAY
        if _nfc("【起動メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【アタック時】") in norm_text: return TriggerType.ON_ATTACK
        if _nfc("【ブロック時】") in norm_text: return TriggerType.ON_BLOCK
        if _nfc("【KO時】") in norm_text: return TriggerType.ON_KO
        if _nfc("【自分のターン終了時】") in norm_text: return TriggerType.TURN_END
        if _nfc("【相手のターン終了時】") in norm_text: return TriggerType.OPP_TURN_END
        if _nfc("【相手のアタック時】") in norm_text: return TriggerType.OPPONENT_ATTACK
        # 埋め込み反応型トリガー（本文の「〜された/与えた時」）は ACTIVATE_MAIN/PASSIVE/UNKNOWN
        # フォールバックのみ上書きする。明示イベントタグ（上記）と timing タグ（【ターン中】）には
        # 劣後する——【自分のターン中】系のカードは既存の挙動テストが YOUR_TURN ロックを前提に
        # するため、ここでは触らない（誘発化は別途、条件ゲート方式で扱うべき領域）。
        embedded = self._detect_embedded_reactive(norm_text)
        # 【ターン中】タグ + 本文の KO/ライフダメージ誘発は、本来その イベント誘発
        # （ON_KO/ON_DAMAGE_DEALT_TO_LIFE）でありターン中は CONTEXT 条件として後段で保全される。
        # コスト節の後（「手札2枚を捨てる：相手のキャラがKOされた時、〜」OP03-076）も含めて上書きする。
        primary_reactive = embedded if embedded in (
            TriggerType.ON_KO, TriggerType.ON_DAMAGE_DEALT_TO_LIFE) else None
        if _nfc("【自分のターン中】") in norm_text: return primary_reactive or TriggerType.YOUR_TURN
        if _nfc("【相手のターン中】") in norm_text: return primary_reactive or TriggerType.OPPONENT_TURN
        if _nfc("【カウンター】") in norm_text: return TriggerType.COUNTER
        if _nfc("【トリガー】") in norm_text: return TriggerType.TRIGGER
        if _nfc("【ゲーム開始時】") in norm_text: return TriggerType.GAME_START

        # 【ドン!!×N】 または 【ターン1回】 のみ（既知トリガータグなし）。
        # 【ターン1回】（起動回数制限）を含む、または【ドン!!×N】を含まない → 起動メイン（活性化能力）。
        # 【ドン!!×N】のみ（活性化タグも【ターン1回】も無い）→ 常在（PASSIVE）条件付き効果。付与ドンが
        # N枚ある間つねに適用される継続効果で、_apply_passive_effects が再計算レイヤ(passive_power 等)へ
        # 適用する（OP13-004「コスト8以上がいれば全体+1000」/ ST14-001「全キャラ コスト+1」等）。
        if self._DON_TURN_TAG_RE.search(norm_text) and not self._TRIGGER_TAG_RE.search(norm_text):
            has_don_tag = re.search(_nfc(r'【ドン[ ]*(?:!!|‼)[ ]*[××][ ]*\d+】'), norm_text)
            has_turn1 = _nfc("【ターン1回】") in norm_text
            # 反応型イベント句「〜した/された/終了 時」を含む【ドン!!×N】は継続効果ではなく誘発（KOした時/
            # バトル終了時 等。OP04-086/ST08-013）。PASSIVE 化すると再計算で毎回発動するため除外する。
            reactive_time = re.search(
                _nfc(r'(?:した|された|受けた|なった|与えた|戻された|離れた|登場した|終了)時'), norm_text)
            if has_don_tag and not has_turn1 and not reactive_time:
                return embedded or TriggerType.PASSIVE
            return embedded or TriggerType.ACTIVATE_MAIN

        # 無タグの反応型「この(キャラ/カード)が…KOされた時、」等は PASSIVE ではなく
        # 対応するトリガーへ写像する。PASSIVE のままだと _apply_passive_effects の
        # 再計算のたびに本体効果が実行され、さらに対話中断が後続の解決を飲み込む
        # （OP11-035/OP11-051 等）。
        if re.match(_nfc(r'^この(キャラ|カード)が[^。【】]*KOされた時'), norm_text):
            return TriggerType.ON_KO
        if re.match(_nfc(r'^この(リーダー|キャラ|カード)が[^。【】]*アタック(した|された)時'), norm_text):
            return TriggerType.ON_ATTACK
        # テキスト埋め込み「ゲーム開始時、」（タグ無し。OP13-079 等のリーダー初期セットアップ）。
        # PASSIVE 判定だと常時再計算のたびに登場効果が走るため GAME_START へ写像する。
        if re.match(_nfc(r'^ゲーム開始時、'), norm_text):
            return TriggerType.GAME_START

        # 既知トリガータグがなければ → 埋め込み反応型 or PASSIVE（常時・条件付き効果・特殊タイミング等）
        # キーワードタグ（【ブロッカー】等）は既に _TRIGGER_TAG_RE に含まれておらず
        # この時点で明示的なトリガーが判別できないため PASSIVE として扱う
        if not self._TRIGGER_TAG_RE.search(norm_text):
            return embedded or TriggerType.PASSIVE

        return embedded or TriggerType.UNKNOWN

    def _detect_embedded_reactive(self, norm_text: str):
        """タグではなく本文に埋め込まれた反応型トリガー（「〜された/与えた時」）を判定する。

        timing タグ（【ターン中】）や ACTIVATE_MAIN/PASSIVE フォールバックより優先し、本来の誘発種別を
        返す。明示イベントタグ（【登場時】【KO時】等）は呼び出し側で先に判定済みのため、ここは
        無タグ／timing タグ／【ドン!!×N】のみのケースで効く。該当が無ければ None。
        置換条件「KOされる時/場合」は『された』(過去) と区別され誤検出しない。
        「登場した時」は OP13-100/OP14-041 等が YOUR_TURN ロック挙動を前提にするため対象外。
        """
        # 相手ライフへのダメージ誘発（「このリーダーのアタックによって、相手のライフにダメージを与えた時」）
        if re.search(_nfc(r'ライフに.{0,8}ダメージを与えた時'), norm_text):
            return TriggerType.ON_DAMAGE_DEALT_TO_LIFE
        # KO 誘発（主語不問:「相手のキャラがKOされた時」「…キャラがKOされた時」）
        if re.search(_nfc(r'KOされた時'), norm_text):
            return TriggerType.ON_KO
        # 場を離れた誘発（「自分の…キャラが場を離れた時」）
        if re.search(_nfc(r'場を離れた時'), norm_text):
            return TriggerType.ON_LEAVE
        # イベント発動誘発（「自分がイベントを発動した時」）
        if re.search(_nfc(r'イベントを発動した時'), norm_text):
            return TriggerType.ON_EVENT_PLAY
        # 相手の登場誘発（「相手が…登場させた時」）
        if re.search(_nfc(r'相手が[^。]{0,30}登場させた時'), norm_text):
            return TriggerType.ON_OPP_PLAY
        return None

    def _parse_to_node(self, text: str, is_cost: bool = False) -> EffectNode:
        norm_text = _nfc(text)

        # 選択肢「以下から…選ぶ」: 「・」項目（または改行区切りの各文）を options として
        # Choice を生成する。後続の「。」分割より前に処理しないと選択肢構造が壊れるため、
        # ここで最優先に捌く。「…の場合、以下から…選ぶ」の先頭条件ゲートは Branch でラップ。
        if re.search(_nfc(r'以下から.{0,6}?選ぶ'), norm_text):
            choice = self._parse_choice(norm_text, is_cost)
            if choice is not None:
                return choice

        # 段階効果「（自分の）<ゾーン>の枚数によって以下の効果をそれぞれ適用する。\n・N枚以上…」:
        # 「それぞれ適用」= 該当する全ティアを累積適用する（択一ではない）。各「・」項目を
        # Branch[<ゾーン>_COUNT >= N] → 効果 に変換した Sequence にする（従来 OTHER で全不発）。
        if re.search(_nfc(r'以下の効果を.{0,4}?適用する'), norm_text):
            applied = self._parse_apply_each(norm_text, is_cost)
            if applied is not None:
                return applied

        # 二択「AするかB、する」: 「以下から1つを選ぶ」を介さない 〜するか〜する 形式の択一。
        # 動詞終止形(u段かな)＋「か、」を境界に2アクションへ分割して Choice 化する
        # （従来 MISSING_ACTION。名詞の「か」=「自分か相手」「リーダーかキャラ」とは語尾で区別）。
        suruka = self._parse_suruka_choice(norm_text, is_cost)
        if suruka is not None:
            return suruka

        # 対象択一「<Aの対象>か、<Bの対象><枚数>を、<動詞>」: 制約の異なる2つの対象が
        # 同じ動詞・枚数を共有する択一（例 OP03-096「相手のコスト0のキャラか、相手のコスト
        # 3以下のステージ1枚までを、KOする」）。1アクションに融合して片方の制約が落ちるのを防ぐ。
        alt = self._parse_target_alternative_choice(norm_text, is_cost)
        if alt is not None:
            return alt

        # 共有対象の二択「<X>を、<A>か<B>」（か の後に読点なし）: 1つの対象 X に対する
        # 2アクションの択一（例:「…キャラ1枚までを、ライフの上に表向きで加えるか登場させる」）。
        shared = self._parse_shared_target_choice(norm_text, is_cost)
        if shared is not None:
            return shared

        # 連用形「引き、」を「引く、」に正規化してから分割
        norm_text = re.sub(_nfc(r'引き、'), _nfc('引く、'), norm_text)
        # 「デッキに戻しシャッフルする」は〈戻す〉と〈シャッフル〉の連用接続。区切らないと
        # shuffle ルールが全体を丸呑みし、手札→デッキの移動(DECK_BOTTOM)が消失する（OP06-047）。
        # 主語が「相手は」の場合はシャッフル対象が相手デッキになるよう主語を引き継ぐ。
        # コスト形「戻しシャッフルできる」は別構文のため対象外。
        norm_text = re.sub(
            _nfc(r'(相手は[^。]*?)デッキに戻しシャッフルする'),
            _nfc(r'\1デッキに戻す。相手のデッキをシャッフルする'), norm_text
        )
        norm_text = re.sub(
            _nfc(r'デッキに戻しシャッフルする'),
            _nfc('デッキに戻す。デッキをシャッフルする'), norm_text
        )
        # デッキの上を見るサーチ/並べ替えは「見て、」で区切り、LOOK を独立アクション化する。
        # （区切らないと「デッキの上から4枚を見て、…1枚までを公開し手札に加える」が
        #  1原子句化し、parse_target が「4枚」を count に誤取得して誤った対象になる。）
        # ライフ等の他の「見て、」には影響させないよう、デッキ文脈のみ「。」へ置換する。
        norm_text = re.sub(
            _nfc(r'(デッキの上から\d+枚(?:まで)?を見て)、'), r'\1。', norm_text
        )
        # 「デッキの上からN枚を公開し、…1枚までを登場させる」も同様に分割し、LOOK を独立化する。
        # （区切らないと公開→登場が1原子句化し、レガシーフォールバックが PLAY_CARD の対象を
        #  FIELD/DECK に誤推定する。分割後は look_deck(LOOK→TEMP)＋play_from_temp(TEMP→FIELD)＋
        #  remaining_*（残り→デッキ）が正しく連携する。）「公開する」（句点付き別構文）は対象外。
        norm_text = re.sub(
            _nfc(r'(デッキの(?:上から\d+枚(?:まで)?|一番上)を公開し)、'), r'\1。', norm_text
        )
        # 「引く、」「捨て、」は lookbehind で分割（動詞を前の部分に残す）
        # 「捨て、」を ON で消費すると「自分の手札1枚を」が動詞なしの断片になるため
        # (?<=捨て)、 に変更して「捨て」を前クローズに残す。
        # 「相手の…をKOし、このカードを手札に加える」のような〈相手への除去＋自己バウンス〉は
        # KOし／レストにし／戻し（連用形＋読点）が逐次接続。これを区切らないと self_to_hand 等が
        # 全体を丸呑みし、前段の相手への除去アクションが消失する（TRIGGER カードで多発）。
        # 動詞を前クローズに残すため lookbehind で「、」のみ分割する。
        # 「相手のキャラを、このターン中、パワー-4000し、自分のライフの上から1枚を手札に
        # 加える」のように 数値+「し、」で別アクションが連結される句も分割する
        # （区切らないと後続の手札/ライフ操作ルールが全体を丸呑みし、前段のバフ/デバフが
        # 消失する）。数値直後の「し、」に限定し、公開し/レストにし等の他語尾には影響しない。
        # 「ドン‼…をレストで追加し、自分の手札から…を登場させる」のように「追加し、」で
        # ドン操作(RAMP_DON)と後続アクションが連結される句も分割する（区切らないと don_add が
        # 全体を丸呑みし、後段の登場/サーチが消失する＝MISSING_ACTION。OP09-022 リム 等）。
        # 「…をアクティブにし、このキャラは…パワー＋N」のようにドン/自己のアクティブ化と
        # 後続バフが連用接続される句も分割する（区切らないと power_buff が全体を丸呑みし、
        # 前段のアクティブ化が消失する。OP06-028/029 等）。
        split_pattern = _nfc(
            r'。|その後、|(?<=置き)、|(?<=加え)、|(?<=引く)、|(?<=捨て)、|発動できる、|させ、'
            r'|(?<=KOし)、|(?<=レストにし)、|(?<=戻し)、|(?<=\d)し、|(?<=付与し)、|(?<=追加し)、'
            r'|(?<=アクティブにし)、'
        )

        parts = re.split(split_pattern, norm_text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            return Sequence(actions=[self._parse_logic_block(p, is_cost) for p in parts])
        elif parts:
            return self._parse_logic_block(parts[0], is_cost)
        return None

    def _parse_logic_block(self, text: str, is_cost: bool) -> EffectNode:
        norm_text = _nfc(text)

        # 条件分岐
        match = re.search(_nfc(r'^(.+?)(?:場合|なら|することで)、(.+)$'), norm_text)
        if match:
            cond_text, rest_text = match.groups()
            # 「ライフの上から1枚を公開し、」が cond_text の先頭に埋め込まれている場合は
            # FACE_UP_LIFE アクションとして先行実行する（条件節ではなくアクション節）
            life_reveal_m = re.match(
                _nfc(r'自分のライフの上から(\d+)枚を公開し、(.+)'), cond_text
            )
            if life_reveal_m:
                n = int(life_reveal_m.group(1))
                remaining_cond = life_reveal_m.group(2)
                # LOOK_LIFE でライフ上 n 枚を temp へ公開する。後続の「登場させてもよい」
                # (play_from_temp) が temp から消費し、不発時は resolver の temp 回収が
                # ライフ上へ戻す。従来は FACE_UP_LIFE（その場で表向き）で temp に載らず、
                # 消費側が no-op だった（OP10-022/ST13-007 等）。
                look_action = GameAction(
                    type=ActionType.LOOK_LIFE,
                    value=ValueSource(base=n),
                    raw_text=life_reveal_m.group(0),
                )
                branch = Branch(
                    condition=self._parse_condition_obj(remaining_cond),
                    if_true=self._parse_to_node(rest_text, is_cost)
                )
                return Sequence(actions=[look_action, branch])
            # cond_text 先頭にトリガー句「〜時、」が残っている場合（トリガー未認識のフォールバックで
            # 「自分が…登場させた時、自分のキャラが3枚以下の場合」のように混入）、ゲート条件は最後の
            # 「時、」以降。残すと「手札から登場」の 手札 が HAND_COUNT を誤誘発する（OP02-026）。
            cond_part = cond_text
            if _nfc("時、") in cond_part:
                cond_part = cond_part.rsplit(_nfc("時、"), 1)[-1]
            return Branch(
                condition=self._parse_condition_obj(cond_part),
                if_true=self._parse_to_node(rest_text, is_cost)
            )

        # 選択肢
        if _nfc("以下から1つを選ぶ") in norm_text:
            options = self._extract_options(norm_text)
            return Choice(
                message=_nfc("効果を選択してください"),
                options=[self._parse_to_node(opt, is_cost) for opt in options],
                option_labels=options
            )

        node = self._parse_atomic_action(norm_text, is_cost)

        # 任意効果マーカー: 効果文脈で「〜してもよい／てもよい」で終わる句は、発動するかを
        # プレイヤーが選べる（resolver が yes/no 確認へ中断）。コストは ":" で既に任意のため対象外。
        # 「できる」は注釈/コスト/キーワード/トリガー宣言で多義のため、ここでは明示マーカーのみ拾う。
        if (not is_cost and isinstance(node, GameAction)
                and node.type not in (ActionType.REPLACE_EFFECT, ActionType.DECLARE_COST, ActionType.OTHER)
                and re.search(_nfc(r"(してもよい|てもよい)"), norm_text)):
            node.is_optional = True

        return node

    def _parse_atomic_action(self, text: str, is_cost: bool) -> GameAction:
        norm_text = _nfc(text)
        act_type = self._detect_action_type(norm_text)
        value_src = self._parse_value(norm_text, act_type)

        if _nfc("残りを") in norm_text:
            target_query = TargetQuery(select_mode="ALL", save_id=None, zone=Zone.TEMP, player=Player.SELF)
        else:
            target_query = parse_target(norm_text)

        # destination の推定
        destination = None
        if act_type == ActionType.MOVE_CARD:
            # ライフを先に判定（「手札...ライフの上に加える」の誤検知を防ぐ）
            if _nfc("ライフの上に加える") in norm_text or _nfc("ライフの上から") not in norm_text and _nfc("ライフの上") in norm_text and _nfc("加える") in norm_text:
                destination = Zone.LIFE
            elif _nfc("ライフの下") in norm_text:
                destination = Zone.LIFE
            elif _nfc("手札に加える") in norm_text or _nfc("手札に戻す") in norm_text or (_nfc("手札") in norm_text and _nfc("加える") in norm_text):
                destination = Zone.HAND
            elif _nfc("トラッシュ") in norm_text:
                destination = Zone.TRASH
            elif _nfc("デッキの上") in norm_text:
                destination = Zone.DECK
                target_query.select_mode = "TOP"

        status = None
        keyword_match = re.search(r'『(.*?)』', norm_text)
        if keyword_match:
            status = keyword_match.group(1)

        if act_type == ActionType.GRANT_KEYWORD:
            kw_match = re.search(_nfc(r'【(ブロッカー|速攻[^】]*|ダブルアタック|バニッシュ|ブロック不可|貫通|シフト)】'), norm_text)
            if kw_match:
                status = kw_match.group(1)

        if _nfc("選び") in norm_text:
            target_query.save_id = "selected_card"
        if _nfc("そのカード") in norm_text or _nfc("そのキャラ") in norm_text:
            target_query.ref_id = "selected_card"

        return GameAction(
            type=act_type,
            target=target_query,
            value=value_src,
            status=status,
            destination=destination,
            raw_text=norm_text
        )

    def _parse_value(self, text: str, act_type: ActionType) -> ValueSource:
        norm_text = _nfc(text)
        if _nfc("枚につき") in norm_text or _nfc("枚数につき") in norm_text:
            nums = re.findall(r'[+-]?\d+', norm_text)
            base_val = int(nums[0]) if nums else 1
            return ValueSource(base=0, dynamic_source="COUNT_REFERENCE", multiplier=base_val)
        # BUFF: パワー±N から値を取り出す（1枚などの数量と混在するため専用パターン使用）
        if act_type == ActionType.BUFF:
            m = re.search(_nfc(r'パワー([+-]?\d+)'), norm_text)
            if m:
                return ValueSource(base=int(m.group(1)))
        nums = re.findall(r'[+-]?\d+', norm_text)
        base_val = int(nums[0]) if nums else 0
        return ValueSource(base=base_val)

    def _detect_action_type(self, text: str) -> ActionType:
        t = _nfc(text)

        # 引く より先に明確なアクションを判定する（順序が重要）

        # ドン!!付与系（「付与する」が含まれる）
        if (_nfc("付与する") in t or _nfc("付与できる") in t) and (_nfc("ドン!!") in t or _nfc("ドン‼") in t):
            return ActionType.ATTACH_DON

        # ドン!!をアクティブで追加（RAMP_DON）
        if _nfc("アクティブで追加") in t or _nfc("アクティブで加える") in t:
            return ActionType.RAMP_DON

        # カード引く（連用形「引き」も対応）
        if _nfc("引く") in t or re.search(_nfc(r'カード\d*枚?を?引き'), t):
            return ActionType.DRAW

        # KO
        if _nfc("KOする") in t:
            return ActionType.KO

        # ライフへの追加（HEAL = デッキ→ライフ）
        if re.search(_nfc(r'デッキの上から.{0,15}ライフ.{0,5}(加える|置く)'), t):
            return ActionType.HEAL
        if re.search(_nfc(r'デッキの上から.{0,5}(加える|置く)'), t) and _nfc("ライフ") in t:
            return ActionType.HEAL

        # ライフ→手札 または 手札→ライフ などカード移動
        if (_nfc("ライフの上") in t or _nfc("ライフの下") in t) and (_nfc("手札に加える") in t or _nfc("手札に戻す") in t):
            return ActionType.MOVE_CARD
        if (_nfc("手札") in t) and (_nfc("ライフの上に加える") in t or _nfc("ライフの下に加える") in t):
            return ActionType.MOVE_CARD

        # 登場させる
        if _nfc("登場させる") in t:
            return ActionType.PLAY_CARD

        # デッキの上に置く（DECK_TOP）
        if re.search(_nfc(r'デッキの上に置く'), t):
            return ActionType.DECK_TOP
        if re.search(_nfc(r'デッキの上から.{0,3}(目)?に置く'), t):
            return ActionType.DECK_TOP

        # デッキの下に置く（DECK_BOTTOM）
        if _nfc("デッキの下") in t and _nfc("置く") in t:
            return ActionType.DECK_BOTTOM

        # トラッシュに置く / 捨てる
        if _nfc("トラッシュに置く") in t:
            return ActionType.DISCARD
        if _nfc("捨てる") in t:
            return ActionType.DISCARD

        # 手札に戻す / 手札に加える
        if _nfc("手札に戻す") in t or _nfc("手札に加える") in t:
            return ActionType.BOUNCE

        # レストにする（連用形「レストにし」も対応）
        if re.search(_nfc(r'レストに(する|できる|し[、。]|して)'), t):
            return ActionType.REST

        # アクティブにする（DON!!以外）
        if _nfc("アクティブにする") in t:
            return ActionType.ACTIVE

        # キーワード能力の付与（ブロッカーを得る、速攻を得る等）
        if _nfc("を得る") in t and any(_nfc(kw) in t for kw in [
            "ブロッカー", "速攻", "ダブルアタック", "バニッシュ", "ブロック不可", "貫通"
        ]):
            return ActionType.GRANT_KEYWORD

        # パワー / 得る → BUFF
        if _nfc("パワー") in t or _nfc("得る") in t:
            return ActionType.BUFF

        return ActionType.OTHER

    def _parse_condition_obj(self, text: str) -> Condition:
        norm_text = _nfc(text)

        # 置換の対象指定「（自分の）「X」がKOされる/場を離れる（場合）」: 離れるカードが名前 X か
        # （OP12-061「自分の「トラファルガー・ロー」がKOされる場合」）。離脱カードを source_card として
        # 評価する SOURCE_STATE("NAME", X) にする。従来は GENERIC で名称限定が脱落していた。
        repl_name_m = re.search(_nfc(r'「([^」]+)」が(?:KOされる|場を離れる)'), norm_text)
        if repl_name_m:
            return Condition(type=ConditionType.SOURCE_STATE,
                             value=("NAME", repl_name_m.group(1)),
                             player=Player.SELF, raw_text=norm_text)
        # 置換の対象指定「（自分の）（元々の）パワーN以上/以下のキャラがKOされる/場を離れる（場合）」:
        # 離脱カードのパワーで限定（OP05-001「自分のパワー5000以上のキャラがKOされる」）。
        repl_pow_m = re.search(_nfc(r'パワー(\d+)(以上|以下)のキャラが(?:KOされる|場を離れる)'), norm_text)
        if repl_pow_m:
            op = CompareOperator.GE if repl_pow_m.group(2) == _nfc('以上') else CompareOperator.LE
            return Condition(type=ConditionType.SOURCE_STATE,
                             value=("POWER", int(repl_pow_m.group(1))), operator=op,
                             player=Player.SELF, raw_text=norm_text)
        # 置換の対象指定「（自分の）（元々の）コストN以上/以下のキャラがKOされる/場を離れる」（EB03-001）。
        repl_cost_m = re.search(_nfc(r'コスト(\d+)(以上|以下)のキャラが(?:KOされる|場を離れる)'), norm_text)
        if repl_cost_m:
            op = CompareOperator.GE if repl_cost_m.group(2) == _nfc('以上') else CompareOperator.LE
            return Condition(type=ConditionType.SOURCE_STATE,
                             value=("COST", int(repl_cost_m.group(1))), operator=op,
                             player=Player.SELF, raw_text=norm_text)

        # 選言条件「A、または B」「Aか、B」= A または B。同一資源の二値しきい値
        # （例「ドン!!が0枚、または8枚以上」ST10-002 /「ドン!!が0枚か、3枚以上」OP05-060）。
        # 双方が数量しきい値（N枚 / 以上 / 以下）を含む場合のみ分割し、対象/選択肢の「AかB」誤爆を避ける。
        or_split = re.search(_nfc(r'^(?P<a>.+?(?:\d+枚|以上|以下))(?:、?または|か、)(?P<b>.+)$'), norm_text)
        if or_split and re.search(_nfc(r'\d+枚|以上|以下'), or_split.group("b")):
            a_txt = or_split.group("a")
            b_txt = or_split.group("b")
            # 第2項で資源主語が省略されている場合（「…か、3枚以上ある場合」）は第1項の主語「…が」を補う。
            resource_kw = ['ドン', '手札', 'ライフ', 'トラッシュ', 'デッキ', 'キャラ']
            if not any(_nfc(k) in b_txt for k in resource_kw):
                prefix = re.match(_nfc(r'(.*?が)'), a_txt)
                if prefix:
                    b_txt = prefix.group(1) + b_txt
            sub_a = self._parse_condition_obj(a_txt)
            sub_b = self._parse_condition_obj(b_txt)
            valid = [c for c in (sub_a, sub_b)
                     if c and c.type not in (ConditionType.NONE, ConditionType.OTHER)]
            if len(valid) == 2:
                return Condition(type=ConditionType.OR, args=valid, raw_text=norm_text)
            if len(valid) == 1:
                return valid[0]

        # 複合条件「Aがいて、Bの場合」「Aが…以下で、Bの…」= A かつ B。
        # 連結部（がいて/がい(ない)て/があり/以上で/以下で 等）の直後の読点で2分割し、
        # 各半を再帰解析して AND にする。従来は全文を1条件として最初の数値で誤分類していた
        # （例「コスト8以上のキャラがいて、手札6枚以下」→ HAND_COUNT>=8 と誤読）。
        split_m = re.search(
            _nfc(r'^(?P<a>.+?(?:がい(?:て|る)|枚以上いて|枚以下いて|がいなくて|があり|がある|'
                 r'以上で|以下で|以上であり|以下であり|を持ち))、(?P<b>.+)$'),
            norm_text)
        if split_m:
            a_txt = split_m.group("a")
            b_txt = split_m.group("b")
            # 「がいて」連結は存在条件なので「いる」に正規化して再帰解析する
            a_norm = re.sub(_nfc(r'いて$'), _nfc('いる'), a_txt)
            a_norm = re.sub(_nfc(r'いなくて$'), _nfc('いない'), a_norm)
            a_norm = re.sub(_nfc(r'(以上|以下)で$'), r'\1', a_norm)
            # 「を持ち」連結（例「リーダーが特徴《X》を持ち、…の場合」）は終止形に正規化
            a_norm = re.sub(_nfc(r'を持ち$'), _nfc('を持つ'), a_norm)
            sub_a = self._parse_condition_obj(a_norm)
            sub_b = self._parse_condition_obj(b_txt)
            valid = [c for c in (sub_a, sub_b)
                     if c and c.type not in (ConditionType.NONE, ConditionType.OTHER)]
            if len(valid) == 2:
                return Condition(type=ConditionType.AND, args=valid, raw_text=norm_text)
            if len(valid) == 1:
                return valid[0]

        # C8「公開したカードが宣言したコストと同じ場合」: 宣言コスト＝公開カードのコスト。
        # 他の数値/特徴条件より先に判定する（「コスト」を含むため誤分類を避ける）。
        if _nfc("宣言したコスト") in norm_text and _nfc("同じ") in norm_text:
            return Condition(type=ConditionType.DECLARED_COST_MATCH, raw_text=norm_text)

        # 比較演算子と数値を抽出
        operator = CompareOperator.EQ
        value = 0
        nums = re.findall(_nfc(r'(\d+)枚?(以上|以下|より多い|未満)?'), norm_text)
        if nums:
            value = int(nums[0][0])
            op_str = nums[0][1]
            op_map = {
                _nfc('以上'): CompareOperator.GE,
                _nfc('以下'): CompareOperator.LE,
                _nfc('より多い'): CompareOperator.GT,
                _nfc('未満'): CompareOperator.LT,
            }
            operator = op_map.get(op_str, CompareOperator.EQ)

        # 対象プレイヤーの判定
        p = Player.OPPONENT if _nfc("相手") in norm_text else Player.SELF

        if re.search(_nfc(r'ドン[ 　]*(?:!!|‼)'), norm_text):
            # 「自分のドン!!が相手より多い」等の相互比較条件
            if (re.search(_nfc(r'相手.{0,20}ドン[ 　]*(?:!!|‼)'), norm_text)
                    or re.search(_nfc(r'ドン[ 　]*(?:!!|‼).{0,20}(?:より|以上|以下)'), norm_text)
                    and _nfc("相手") in norm_text):
                op_m = re.search(_nfc(r'(以下|以上|より多い|未満|より少ない)'), norm_text)
                cmp_op = {
                    _nfc('以上'): CompareOperator.GE,
                    _nfc('以下'): CompareOperator.LE,
                    _nfc('より多い'): CompareOperator.GT,
                    _nfc('未満'): CompareOperator.LT,
                    _nfc('より少ない'): CompareOperator.LT,
                }.get(op_m.group(1) if op_m else '', CompareOperator.GE)
                return Condition(type=ConditionType.DON_COUNT_COMPARE, operator=cmp_op, player=Player.SELF, raw_text=norm_text)
            # 明示の数値が無い存在条件「（場の）ドン‼がある場合」=1枚以上 /「ない場合」=0枚。
            # 既定（nums 空）の EQ 0 は「0枚と等しい」になり「ある場合」を逆転させていた（OP13-003）。
            # 「付与されているドン‼がある場合」は場のドン枚数ではなく付与ドン（attached）の条件なので
            # ここでは触らない（既存挙動を維持し、別途 HAS_DON 系で扱うべき領域）。
            if not nums and _nfc("付与") not in norm_text:
                if _nfc("ない") in norm_text:
                    return Condition(type=ConditionType.DON_COUNT, operator=CompareOperator.EQ, value=0, player=p, raw_text=norm_text)
                if _nfc("ある") in norm_text:
                    return Condition(type=ConditionType.DON_COUNT, operator=CompareOperator.GE, value=1, player=p, raw_text=norm_text)
            return Condition(type=ConditionType.DON_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        # イベント発動済み条件（「このターン中、自分が元々のコスト3以上のイベントを発動している場合」
        # OP15-002）。このターン中にコストN以上のイベントを発動したか（EVENT_THIS_TURN）。未発動なら不発。
        ev_played_m = re.search(_nfc(r'コスト(\d+)以上のイベントを発動している'), norm_text)
        if ev_played_m:
            return Condition(type=ConditionType.EVENT_THIS_TURN,
                             value=(f"EVENT_PLAYED_COST_GE_{ev_played_m.group(1)}", 1),
                             player=Player.SELF, raw_text=norm_text)

        # ターン数条件（「自分の第2ターン以降の場合」OP15-058）。turn_count >= N。
        turn_m = re.search(_nfc(r'第(\d+)ターン以降'), norm_text)
        if turn_m:
            return Condition(type=ConditionType.TURN_COUNT, operator=CompareOperator.GE,
                             value=int(turn_m.group(1)), player=p, raw_text=norm_text)

        # ライフ＋手札の合計（「自分のライフと手札の合計枚数が4枚以下の場合」OP04-040）。
        # LIFE_COUNT より先に判定する（「ライフ」を含むため誤分類を避ける）。
        if _nfc("ライフと手札の合計") in norm_text or _nfc("手札とライフの合計") in norm_text:
            return Condition(type=ConditionType.LIFE_HAND_SUM, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("ライフ") in norm_text:
            return Condition(type=ConditionType.LIFE_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("手札") in norm_text:
            return Condition(type=ConditionType.HAND_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if _nfc("トラッシュ") in norm_text:
            return Condition(type=ConditionType.TRASH_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        # デッキ枚数（「自分のデッキが20枚以下の場合」等）。"デッキの上から…" は除外。
        if _nfc("デッキが") in norm_text:
            return Condition(type=ConditionType.DECK_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        if (_nfc("リーダーが") in norm_text or _nfc("リーダーの特徴") in norm_text
                or _nfc("リーダーのパワー") in norm_text):
            # 特徴は《X》だけでなく『X』（『白ひげ海賊団』『B・W』等の名称系特徴）でも書かれる。
            trait_match = re.search(_nfc(r'[《<『]([^》>』]+)[》>』]'), norm_text)
            name_match = re.search(_nfc(r'「([^」]+)」'), norm_text)
            if trait_match:
                return Condition(type=ConditionType.LEADER_TRAIT, value=trait_match.group(1), player=p, raw_text=norm_text)
            if name_match:
                return Condition(type=ConditionType.LEADER_NAME, value=name_match.group(1), player=p, raw_text=norm_text)
            if _nfc("多色") in norm_text:
                return Condition(type=ConditionType.LEADER_COLOR, value=_nfc("多色"), player=p, raw_text=norm_text)
            # 単色リーダー条件（「自分のリーダーが青を含む」等）
            color_m = re.search(_nfc(r'(赤|青|緑|黄|黒|紫)'), norm_text)
            if color_m:
                return Condition(type=ConditionType.LEADER_COLOR, value=color_m.group(1), player=p, raw_text=norm_text)
            # 属性条件（斬/打/射/特/知）。括弧は半角・全角どちらも対応。
            attr_m = re.search(_nfc(r'属性[（(]([^）)]+)[）)]'), norm_text)
            if attr_m:
                return Condition(type=ConditionType.LEADER_ATTRIBUTE, value=attr_m.group(1), player=p, raw_text=norm_text)
            # リーダーの状態条件（アクティブ/レスト/パワー）
            if _nfc("アクティブ") in norm_text:
                return Condition(type=ConditionType.LEADER_STATE, value="IS_ACTIVE", player=p, raw_text=norm_text)
            if _nfc("レスト") in norm_text:
                return Condition(type=ConditionType.LEADER_STATE, value="IS_RESTED", player=p, raw_text=norm_text)
            pow_leader_m = re.search(_nfc(r'パワーが?(\d+)(以上|以下)'), norm_text)
            if pow_leader_m:
                thr = int(pow_leader_m.group(1))
                op = CompareOperator.GE if pow_leader_m.group(2) == _nfc('以上') else CompareOperator.LE
                return Condition(type=ConditionType.LEADER_STATE, value=("POWER", thr), operator=op, player=p, raw_text=norm_text)

        # キャラのコスト合計（「自分のキャラのコストの合計が N 以上/以下の場合」OP10-022）。
        # 枚数(FIELD_COUNT)より先に判定する（「合計」を含むため誤分類を避ける）。
        cost_sum_m = re.search(_nfc(r'キャラのコストの合計が(\d+)(以上|以下)'), norm_text)
        if cost_sum_m:
            thr = int(cost_sum_m.group(1))
            op = CompareOperator.GE if cost_sum_m.group(2) == _nfc('以上') else CompareOperator.LE
            return Condition(type=ConditionType.FIELD_COST_SUM, operator=op, value=thr, player=p, raw_text=norm_text)

        # 盤面のキャラ枚数（「自分の（レストの／特徴《X》の／コストN以上の）キャラがM枚以上いる」
        # 「…キャラがいる」）。数値が「フィルタ(コストN以上)」と「枚数(M枚)」で混在し得るため、
        # 閾値は必ず「M枚」側から取り、フィルタは parse_target に委ねる（保守的な分類）。
        # 「このキャラが…される/場を離れる/登場した」等の単体状態・置換条件は対象外。
        if (_nfc("キャラ") in norm_text and _nfc("このキャラ") not in norm_text
                and (_nfc("いる") in norm_text or _nfc("いない") in norm_text or re.search(_nfc(r"\d+枚(以上|以下)"), norm_text))
                and not re.search(_nfc(r"(される|場を離れる|登場した|公開)"), norm_text)
                and _nfc("のみ") not in norm_text):
            tq = parse_target(norm_text)
            mc = re.search(_nfc(r"(\d+)枚(以上|以下|より多い|未満)?"), norm_text)
            if mc:
                thr = int(mc.group(1))
                cnt_op = {
                    _nfc('以上'): CompareOperator.GE,
                    _nfc('以下'): CompareOperator.LE,
                    _nfc('より多い'): CompareOperator.GT,
                    _nfc('未満'): CompareOperator.LT,
                }.get(mc.group(2), CompareOperator.GE)
                # 「N枚いない」→ 実質 N枚未満
                if _nfc("いない") in norm_text and mc.group(2) is None:
                    cnt_op = CompareOperator.LT
            elif _nfc("いない") in norm_text:
                thr, cnt_op = 1, CompareOperator.LT  # 1枚もいない
            else:
                thr, cnt_op = 1, CompareOperator.GE  # 「いる」=1枚以上
            return Condition(type=ConditionType.FIELD_COUNT, target=tq,
                             operator=cnt_op, value=thr, player=tq.player, raw_text=norm_text)

        # SOURCE_STATE: このキャラ自身の状態条件（レスト/アクティブ/パワー/登場ターン）
        if _nfc("このキャラ") in norm_text:
            if _nfc("登場したターン") in norm_text:
                return Condition(type=ConditionType.SOURCE_STATE, value="ENTERED_THIS_TURN", player=p, raw_text=norm_text)
            if _nfc("アクティブ") in norm_text:
                return Condition(type=ConditionType.SOURCE_STATE, value="IS_ACTIVE", player=p, raw_text=norm_text)
            if _nfc("レスト") in norm_text:
                return Condition(type=ConditionType.SOURCE_STATE, value="IS_RESTED", player=p, raw_text=norm_text)
            pow_m = re.search(_nfc(r'パワーが?(\d+)(以上|以下)'), norm_text)
            if pow_m:
                thr = int(pow_m.group(1))
                op = CompareOperator.GE if pow_m.group(2) == _nfc('以上') else CompareOperator.LE
                return Condition(type=ConditionType.SOURCE_STATE, value=("POWER", thr), operator=op, player=p, raw_text=norm_text)

        # FIELD_ALL_TRAIT: 場のキャラ全員が特定の特徴を持つ（「のみ」条件）
        if _nfc("のみ") in norm_text and _nfc("キャラ") in norm_text:
            # 特徴《X》（完全一致）
            trait_m = re.search(_nfc(r'[《<]([^》>]+)[》>]'), norm_text)
            if trait_m:
                return Condition(type=ConditionType.FIELD_ALL_TRAIT, value=(trait_m.group(1), False), player=p, raw_text=norm_text)
            # 「X」/『X』を含む特徴（部分一致）
            contains_m = re.search(_nfc(r'[『「]([^』」]+)[』」]を含む特徴'), norm_text)
            if contains_m:
                return Condition(type=ConditionType.FIELD_ALL_TRAIT, value=(contains_m.group(1), True), player=p, raw_text=norm_text)

        # HAS_CHARACTER: 特定の名前のキャラが場にいる/いない（枚数指定・状態指定含む）
        # 状態付き: 「X」がレストの / 「X」がアクティブの
        has_char_state_m = re.search(_nfc(r'「([^」]+)」が(レスト|アクティブ)'), norm_text)
        if has_char_state_m:
            char_name = has_char_state_m.group(1)
            state = "IS_RESTED" if has_char_state_m.group(2) == _nfc('レスト') else "IS_ACTIVE"
            return Condition(type=ConditionType.HAS_CHARACTER, value=(char_name, state), operator=CompareOperator.GE, player=p, raw_text=norm_text)
        # 枚数指定
        has_char_count_m = re.search(_nfc(r'「([^」]+)」が(\d+)枚(以上|以下)?い'), norm_text)
        if has_char_count_m:
            char_name = has_char_count_m.group(1)
            count_thr = int(has_char_count_m.group(2))
            cnt_op = {
                _nfc('以上'): CompareOperator.GE,
                _nfc('以下'): CompareOperator.LE,
            }.get(has_char_count_m.group(3) or '', CompareOperator.GE)
            return Condition(type=ConditionType.HAS_CHARACTER, value=(char_name, count_thr), operator=cnt_op, player=p, raw_text=norm_text)
        # 存在/不在
        has_char_m = re.search(_nfc(r'「([^」]+)」が(?:い(る|ない)|あ(る|ない))'), norm_text)
        if has_char_m:
            char_name = has_char_m.group(1)
            present_part = has_char_m.group(2) or has_char_m.group(3)
            present = present_part == _nfc('る')
            op = CompareOperator.GE if present else CompareOperator.EQ
            return Condition(type=ConditionType.HAS_CHARACTER, value=char_name, operator=op, player=p, raw_text=norm_text)

        # RESTED_COUNT: レスト状態のカード総数（フィールド＋ドン!!）
        if _nfc("レストのカード") in norm_text:
            return Condition(type=ConditionType.RESTED_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

        # OPPONENT_REMOVAL: 相手の効果/バトルで場を離れる/KOされる置換条件
        if ((_nfc("相手の効果で") in norm_text or _nfc("相手によって") in norm_text)
                and (_nfc("場を離れる") in norm_text or _nfc("KOされる") in norm_text)):
            val: dict = {"trigger": "KO" if _nfc("KOされる") in norm_text else "LEAVE"}
            pow_max_m = re.search(_nfc(r'元々のパワー(\d+)以下'), norm_text)
            pow_min_m = re.search(_nfc(r'元々のパワー(\d+)以上'), norm_text)
            cost_max_m = re.search(_nfc(r'元々のコスト(\d+)以下'), norm_text)
            trait_m = re.search(_nfc(r'[《<]([^》>]+)[》>]'), norm_text)
            if pow_max_m: val["power_max"] = int(pow_max_m.group(1))
            if pow_min_m: val["power_min"] = int(pow_min_m.group(1))
            if cost_max_m: val["cost_max"] = int(cost_max_m.group(1))
            if trait_m: val["trait"] = trait_m.group(1)
            return Condition(type=ConditionType.OPPONENT_REMOVAL, value=val, player=p, raw_text=norm_text)

        # FIELD_COUNT_COMPARE: 自分と相手の場キャラ数の相対比較
        fc_cmp_m = re.search(_nfc(r'キャラが相手のキャラより(少ない|多い)'), norm_text)
        if fc_cmp_m:
            op = CompareOperator.LT if fc_cmp_m.group(1) == _nfc('少ない') else CompareOperator.GT
            return Condition(type=ConditionType.FIELD_COUNT_COMPARE, operator=op, player=Player.SELF, raw_text=norm_text)

        # REVEALED_CARD_TRAIT: 公開したカードの特徴/コスト/タイプ条件。
        # 公開(LOOK)が独立クローズに分割される場合、条件側には「公開し」が残らないため
        # 「そのカード」を主たる手掛かりとする（filter が1つも取れなければ下へフォールスルー）。
        if _nfc("そのカード") in norm_text:
            val: dict = {}
            # 含む特徴: 『X』を含む特徴
            contains_m = re.search(_nfc(r'[『「]([^』」]+)[』」]を含む特徴'), norm_text)
            if contains_m:
                val["trait"] = contains_m.group(1)
                val["trait_contains"] = True
            # 完全一致特徴: 《X》
            exact_m = re.search(_nfc(r'[《<]([^》>]+)[》>]'), norm_text)
            if exact_m and "trait" not in val:
                val["trait"] = exact_m.group(1)
                val["trait_contains"] = False
            # コスト条件（「コスト5以下」「コスト5の」= 完全一致）
            cost_m = re.search(_nfc(r'コスト(\d+)(以上|以下)?'), norm_text)
            if cost_m:
                val["cost"] = int(cost_m.group(1))
                if cost_m.group(2) == _nfc('以上'):
                    val["cost_op"] = CompareOperator.GE
                elif cost_m.group(2) == _nfc('以下'):
                    val["cost_op"] = CompareOperator.LE
                else:
                    val["cost_op"] = CompareOperator.EQ
            # カード名条件（「サボ」等の完全一致。『X』を含む特徴 とは別）
            name_m = re.search(_nfc(r'「([^」]+)」'), norm_text)
            if name_m:
                val["name"] = name_m.group(1)
            # カードタイプ
            if _nfc("キャラカード") in norm_text:
                val["card_type"] = "キャラ"
            elif _nfc("イベントカード") in norm_text:
                val["card_type"] = "イベント"
            if val:
                return Condition(type=ConditionType.REVEALED_CARD_TRAIT, value=val, player=p, raw_text=norm_text)

        # PREV_ACTION: 直前アクションの成否（「そうした場合」「そうしなかった場合」「登場させた場合」）
        if _nfc("そうしなかった") in norm_text:
            return Condition(type=ConditionType.PREV_ACTION, value="SKIPPED", player=p, raw_text=norm_text)
        if _nfc("そうした") in norm_text:
            return Condition(type=ConditionType.PREV_ACTION, value="SUCCEEDED", player=p, raw_text=norm_text)
        # "場合" は _parse_logic_block の区切り正規表現で削除済みのため単独でチェック
        if _nfc("登場させた") in norm_text:
            return Condition(type=ConditionType.PREV_ACTION, value="PLAYED_CARD", player=p, raw_text=norm_text)

        return Condition(type=ConditionType.GENERIC, raw_text=norm_text)

    def _parse_apply_each(self, text: str, is_cost: bool) -> Optional[EffectNode]:
        """「<ゾーン>の枚数によって以下の効果をそれぞれ適用する。\n・N枚以上…」を
        Sequence[Branch[ZONE_COUNT>=N] → 効果, …] に変換する（該当ティアを累積適用）。

        OP15-092 のような段階パッシブ。「それぞれ適用」のため択一(Choice)ではなく、
        条件を満たす全ティアを順に適用する。閾値は各項目の「N枚以上」から取る。
        項目が割れない/参照ゾーンが取れない場合は None（呼び出し側が従来解析へフォールバック）。
        """
        norm = _nfc(text)
        # 参照ゾーン（枚数の基準）と対象プレイヤーを head から判定。
        head_m = re.search(_nfc(r'(自分|相手|お互い)?の?(トラッシュ|ライフ|手札|デッキ)の枚数によって'), norm)
        if not head_m:
            return None
        zone_word = head_m.group(2)
        ctype = {
            _nfc("トラッシュ"): ConditionType.TRASH_COUNT,
            _nfc("ライフ"): ConditionType.LIFE_COUNT,
            _nfc("手札"): ConditionType.HAND_COUNT,
            _nfc("デッキ"): ConditionType.DECK_COUNT,
        }.get(_nfc(zone_word))
        if ctype is None:
            return None
        cplayer = Player.OPPONENT if head_m.group(1) == _nfc("相手") else Player.SELF
        # 本体（適用する。以降）の「・」項目を抽出。
        m_end = re.search(_nfc(r'以下の効果を.{0,4}?適用する'), norm)
        tail = norm[m_end.end():].lstrip(_nfc('。\n 　')) if m_end else ""
        options = self._extract_options(tail)
        branches: List[EffectNode] = []
        for opt in options:
            cm = re.match(_nfc(r'\s*(\d+)枚以上(?:ある)?(?:の)?(?:場合)?[、,]?\s*(.+)$'), opt, re.DOTALL)
            if not cm:
                continue
            thr = int(cm.group(1))
            eff_text = cm.group(2).strip().rstrip(_nfc('。'))
            # 「<主語>は…になり、コスト+M」の連用中止は文境界として正規化し、後段フラグメントに
            # 主語を伝播する（区切ると「コスト+M」が主語を失い対象が曖昧化＝PASSIVE で対象選択
            # 中断に陥るため）。主語が「この(キャラ/リーダー/カード)は」のときのみ伝播する。
            subj_m = re.match(_nfc(r'(この(?:キャラ|リーダー|カード)は)'), eff_text)
            subj = subj_m.group(1) if subj_m else ''
            eff_text = re.sub(_nfc(r'になり、'), _nfc('になる。') + subj, eff_text)
            eff_node = self._parse_to_node(eff_text, is_cost)
            if eff_node is None:
                continue
            cond = Condition(type=ctype, operator=CompareOperator.GE, value=thr,
                             player=cplayer, raw_text=opt)
            branches.append(Branch(condition=cond, if_true=eff_node))
        if not branches:
            return None
        return Sequence(actions=branches) if len(branches) > 1 else branches[0]

    def _parse_choice(self, text: str, is_cost: bool) -> Optional[EffectNode]:
        """「（条件、）以下から…選ぶ。\n・項目…」を Choice（必要なら条件 Branch）に変換する。

        options が抽出できない場合は None を返し、呼び出し側が通常解析へフォールバックする。
        """
        norm = _nfc(text)
        m = re.search(_nfc(r'以下から.{0,6}?選ぶ'), norm)
        if not m:
            return None
        head = norm[:m.end()]            # 「（条件、）…以下から1つを選ぶ」
        tail = norm[m.end():]            # 「。\n・A。\n・B。」（選択肢本体）
        tail = tail.lstrip(_nfc('。\n 　'))
        options = self._extract_options(tail)
        if len(options) < 2:
            return None  # 選択肢が割れない場合は Choice 化しない（誤検知防止）
        # 「相手は以下から…選ぶ」は相手が選択する（IR に記録。既定は自分）。
        chooser = Player.OPPONENT if re.search(_nfc(r'相手は\s*以下から'), head) else Player.SELF
        choice = Choice(
            message=_nfc("効果を選択してください"),
            options=[self._parse_to_node(opt, is_cost) for opt in options],
            option_labels=options,
            player=chooser,
        )
        # 「…の場合／なら、以下から…選ぶ」: 先頭の条件ゲートを Branch でラップする。
        cond_m = re.search(_nfc(r'^(.+?)(?:場合|なら)、\s*以下から'), head)
        if cond_m:
            return Branch(condition=self._parse_condition_obj(cond_m.group(1)), if_true=choice)
        return choice

    def _node_has_real_action(self, node) -> bool:
        """node が「実行系のあるアクション」を含むか（OTHER だけの空振りでないか）を判定。"""
        if node is None:
            return False
        if isinstance(node, GameAction):
            return node.type != ActionType.OTHER
        if isinstance(node, Sequence):
            return any(self._node_has_real_action(a) for a in node.actions)
        if isinstance(node, Branch):
            return self._node_has_real_action(node.if_true) or self._node_has_real_action(node.if_false)
        if isinstance(node, Choice):
            return any(self._node_has_real_action(o) for o in node.options)
        return False

    def _parse_suruka_choice(self, text: str, is_cost: bool) -> Optional[EffectNode]:
        """「AするかB、する」形式の二択を Choice に変換する（無ければ None）。

        動詞終止形(u 段かな)の直後に来る「か、」だけを境界にすることで、名詞の並列
        （「自分か相手」「リーダーかキャラ」「イベントか【ブロッカー】」）を誤って割らない。
        左右がともに実行系アクションに解釈できる場合のみ Choice 化し、過検知を避ける。
        """
        norm = _nfc(text)
        if _nfc("以下から") in norm:
            return None  # モーダル選択は _parse_choice が担当
        m = re.search(_nfc(r'[るくすつぶむうぐ]か、'), norm)
        if not m:
            return None
        boundary = m.start() + 1  # 動詞末尾の次（「か」の位置）
        left = norm[:boundary].strip()
        right = norm[m.end():].strip().rstrip(_nfc('。')).strip()
        if not left or not right:
            return None
        chooser = Player.OPPONENT if re.search(_nfc(r'相手は'), norm[:boundary]) else Player.SELF
        opt_a = self._parse_to_node(left, is_cost)
        opt_b = self._parse_to_node(right, is_cost)
        if not (self._node_has_real_action(opt_a) and self._node_has_real_action(opt_b)):
            return None  # どちらかが空振りなら択一にしない（レガシー解釈へ委ねる）
        return Choice(
            message=_nfc("効果を選択してください"),
            options=[opt_a, opt_b],
            option_labels=[left, right],
            player=chooser,
        )

    def _parse_target_alternative_choice(self, text: str, is_cost: bool) -> Optional[EffectNode]:
        """「<A対象>か、<B対象><枚数>を、<動詞>」形式の対象択一を Choice 化する（無ければ None）。

        A と B は制約（コスト/特徴/種別/側）が異なる対象記述で、末尾の「<枚数>を、<動詞>」を
        共有する。共有サフィックスを A 側にも補って各オプションを解釈する。
        例 OP03-096「相手のコスト0のキャラか、相手のコスト3以下のステージ1枚までを、KOする」。

        名詞並列の「自分か相手／リーダーかキャラ」等を誤って割らないよう、A・B の両方が
        対象修飾語（コスト/特徴/枚数/種別）を含み、かつ両側が実行系アクションに解釈できる
        場合のみ Choice 化する。
        """
        norm = _nfc(text)
        if _nfc("以下から") in norm:
            return None
        # 「〜場合」を含む句は条件節（「ドンが0枚か、3枚以上ある場合」OP05-060）であって
        # 対象択一ではない。条件分岐は専用経路に委ねる。
        if _nfc("場合") in norm:
            return None
        m = re.search(_nfc(r'か、'), norm)
        if not m:
            return None
        left = norm[:m.start()].strip()
        right = norm[m.end():].strip().rstrip(_nfc('。')).strip()
        if not left or not right:
            return None
        # 動詞終止形＋「か、」の場合は _parse_suruka_choice（別対象・動詞二択）に委ねる。
        if re.search(_nfc(r'[るくすつぶむうぐ]$'), left):
            return None
        # 左側が「が\d+枚」等の数量条件記述（条件節の名残）なら対象ではない。
        if re.search(_nfc(r'が\d+枚'), left):
            return None
        # 共有サフィックス「<枚数>(まで)?を、<動詞>」を right から抽出する。
        suf = re.search(_nfc(r'(\d+枚(?:まで)?を、.+)$'), right)
        if not suf:
            return None
        shared_suffix = suf.group(1)
        # A 側は対象記述のみ（種別 キャラ/ステージ/リーダー 等を含むこと）。
        if not re.search(_nfc(r'(キャラ|ステージ|リーダー|カード|ドン)'), left):
            return None
        # B 側（right の先頭〜枚数の手前）も対象記述であること。
        b_desc = right[:suf.start()]
        if not re.search(_nfc(r'(キャラ|ステージ|リーダー|カード|ドン)'), b_desc):
            return None
        opt_a = self._parse_to_node(left + shared_suffix, is_cost)
        opt_b = self._parse_to_node(right, is_cost)
        if not (self._node_has_real_action(opt_a) and self._node_has_real_action(opt_b)):
            return None
        chooser = Player.OPPONENT if re.search(_nfc(r'相手は'), norm[:m.start()]) else Player.SELF
        return Choice(
            message=_nfc("対象を選択してください"),
            options=[opt_a, opt_b],
            option_labels=[left, b_desc],
            player=chooser,
        )

    def _parse_shared_target_choice(self, text: str, is_cost: bool) -> Optional[EffectNode]:
        """「<X>を、<A>か<B>」形式の共有対象二択を Choice に変換する（無ければ None）。

        対象 X を両オプションの先頭に補って解釈する点が _parse_suruka_choice（別対象）と異なる。
        「か」の後に読点が無い（"加えるか登場させる"）ことで読点付き二択(「するか、」)と区別する。
        左右がともに実行系アクションに解釈できる場合のみ Choice 化する（過検知防止）。
        """
        norm = _nfc(text)
        if _nfc("以下から") in norm or _nfc("か、") in norm:
            return None  # モーダル選択 / 読点付き二択は別経路
        sep = norm.rfind(_nfc("を、"))
        if sep < 0:
            return None
        target_part = norm[:sep]
        actions = norm[sep + len(_nfc("を、")):]
        # アクション部の動詞終止形(u段かな)直後の「か」(読点なし)を境界にする。
        m = re.search(_nfc(r'[るくすつぶむうぐ]か(?![、。])'), actions)
        if not m:
            return None
        a = actions[:m.start() + 1].strip()
        b = actions[m.end():].strip().rstrip(_nfc('。')).strip()
        if not a or not b or not target_part:
            return None
        opt_a = self._parse_to_node(f"{target_part}を、{a}", is_cost)
        opt_b = self._parse_to_node(f"{target_part}を、{b}", is_cost)
        if not (self._node_has_real_action(opt_a) and self._node_has_real_action(opt_b)):
            return None
        return Choice(
            message=_nfc("効果を選択してください"),
            options=[opt_a, opt_b],
            option_labels=[a, b],
            player=Player.SELF,
        )

    def _extract_options(self, text: str) -> List[str]:
        norm_text = _nfc(text)
        lines = [l.strip() for l in norm_text.split('\n') if l.strip()]
        # 「・」始まりの行を選択肢として優先抽出（末尾の「。」は除去）。
        bullets = [
            re.sub(_nfc(r'^[・\-]\s*'), '', l).rstrip(_nfc('。')).strip()
            for l in lines if l.startswith((_nfc('・'), _nfc('-')))
        ]
        if bullets:
            return bullets
        # 「・」が無い場合: 改行区切りの各文（2件以上）を選択肢とみなす。
        if len(lines) > 1:
            return [l.rstrip(_nfc('。')).strip() for l in lines]
        # 単一行のみ: 従来の「、」分割フォールバック（後方互換）。
        parts = re.split(_nfc(r'、'), norm_text)
        return [p.strip() for p in parts if _nfc("選ぶ") not in p and _nfc("以下から") not in p]
