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

    def parse_card_text(self, text: str, as_trigger: bool = False) -> List[Ability]:
        norm = _nfc(text)
        if not norm or norm.strip() in ['なし', 'None', '']:
            return []

        segments = re.split(r'\s*/\s*|\n', norm)
        segments = [s.strip() for s in segments if s.strip()]

        abilities = []
        for seg in segments:
            # キーワード能力宣言 / キーワード説明括弧書きはスキップ（Ability 不要）
            if self._KEYWORD_ONLY_RE.match(seg):
                continue
            if self._PAREN_ONLY_RE.match(seg) and not re.search(r'【[^】]+】', seg):
                continue
            # 「・」で始まる選択肢セグメントは Choice の option として親セグメントで処理済み
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

            # トリガー/注釈タグの除去。
            # ただしキーワード能力タグ（【ブロッカー】等）は効果本体で「付与」を
            # 表すため保持する。脱落すると原子句が「このキャラはを得る」になり、
            # GRANT_KEYWORD（付与すべきキーワード）を復元できなくなる（既知バグ）。
            clean_text = re.sub(
                _nfc(r'【(?!ブロッカー|速攻|ダブルアタック|バニッシュ|ブロック不可|貫通|シフト)[^】]*?】'),
                '', norm_text
            ).strip()

            cost_node = None
            effect_text = clean_text

            # コストの分離（全角コロン優先、なければ半角コロン）。
            # ただし【速攻：キャラ】のようにキーワード能力タグ内部の「：」は
            # コスト区切りではないため、タグ内部をマスクした上で判定する。
            masked = re.sub(r'【[^】]*】', lambda m: '〇' * len(m.group(0)), clean_text)
            colon = _nfc("：") if _nfc("：") in masked else (_nfc(":") if _nfc(":") in masked else None)
            if colon:
                idx = masked.index(colon)
                cost_node = self._parse_cost_node(clean_text[:idx])
                effect_text = clean_text[idx + 1:]

            # ドン!!コストタグを cost_node に統合
            if don_cost_value > 0:
                don_cost_action = GameAction(
                    type=ActionType.REST_DON,
                    value=ValueSource(base=don_cost_value),
                    target=TargetQuery(zone=Zone.COST_AREA, player=Player.SELF, count=don_cost_value),
                    raw_text=_nfc(f"ドン!!{don_cost_value}枚をレストにする")
                )
                if cost_node is not None:
                    cost_node = Sequence(actions=[don_cost_action, cost_node])
                else:
                    cost_node = don_cost_action

            # 効果本体の解析
            effect_node = self._parse_to_node(effect_text)

            # 先頭のゲート条件（「〜の場合、」）を ability.condition に引き上げる
            final_condition = turn_limit_cond
            if isinstance(effect_node, Branch) and effect_node.if_false is None and effect_node.condition is not None:
                if final_condition is None:
                    final_condition = effect_node.condition
                else:
                    final_condition = Condition(
                        type=ConditionType.AND,
                        args=[final_condition, effect_node.condition]
                    )
                effect_node = effect_node.if_true

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

    def _parse_cost_node(self, cost_text: str) -> Optional[EffectNode]:
        """
        コストテキストを解析する。
        「このキャラをレストにできる」「このリーダーをレストにできる」パターンを
        ref_id="self" の REST アクションとして処理する。
        """
        norm = _nfc(cost_text)

        if re.search(_nfc(r'このキャラをレスト|このリーダーをレスト|このキャラをレストにできる'), norm):
            return GameAction(
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

        return self._parse_to_node(norm, is_cost=True)

    def _detect_trigger(self, text: str) -> TriggerType:
        norm_text = _nfc(text)
        if _nfc("【登場時】") in norm_text: return TriggerType.ON_PLAY
        if _nfc("【起動メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【メイン】") in norm_text: return TriggerType.ACTIVATE_MAIN
        if _nfc("【アタック時】") in norm_text: return TriggerType.ON_ATTACK
        if _nfc("【ブロック時】") in norm_text: return TriggerType.ON_BLOCK
        if _nfc("【KO時】") in norm_text: return TriggerType.ON_KO
        if _nfc("【自分のターン終了時】") in norm_text: return TriggerType.TURN_END
        if _nfc("【相手のターン終了時】") in norm_text: return TriggerType.OPP_TURN_END
        if _nfc("【相手のアタック時】") in norm_text: return TriggerType.OPPONENT_ATTACK
        if _nfc("【自分のターン中】") in norm_text: return TriggerType.YOUR_TURN
        if _nfc("【相手のターン中】") in norm_text: return TriggerType.OPPONENT_TURN
        if _nfc("【カウンター】") in norm_text: return TriggerType.COUNTER
        if _nfc("【トリガー】") in norm_text: return TriggerType.TRIGGER
        if _nfc("【ゲーム開始時】") in norm_text: return TriggerType.GAME_START

        # 【ドン!!×N】 または 【ターン1回】 のみ（既知トリガータグなし）→ 起動メイン
        # 【ブロッカー】等を得る効果も正しく ACTIVATE_MAIN と判定できるよう、
        # 既知トリガータグ (_TRIGGER_TAG_RE) の有無のみを判断基準にする
        if self._DON_TURN_TAG_RE.search(norm_text) and not self._TRIGGER_TAG_RE.search(norm_text):
            return TriggerType.ACTIVATE_MAIN

        # 既知トリガータグがなければ → PASSIVE（常時・条件付き効果・特殊タイミング等）
        # キーワードタグ（【ブロッカー】等）は既に _TRIGGER_TAG_RE に含まれておらず
        # この時点で明示的なトリガーが判別できないため PASSIVE として扱う
        if not self._TRIGGER_TAG_RE.search(norm_text):
            return TriggerType.PASSIVE

        return TriggerType.UNKNOWN

    def _parse_to_node(self, text: str, is_cost: bool = False) -> EffectNode:
        norm_text = _nfc(text)

        # 連用形「引き、」を「引く、」に正規化してから分割
        norm_text = re.sub(_nfc(r'引き、'), _nfc('引く、'), norm_text)
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
        split_pattern = _nfc(r'。|その後、|置き、|加え、|(?<=引く)、|(?<=捨て)、|発動できる、|させ、')

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
            return Branch(
                condition=self._parse_condition_obj(cond_text),
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

        return self._parse_atomic_action(norm_text, is_cost)

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
            return Condition(type=ConditionType.DON_COUNT, operator=operator, value=value, player=p, raw_text=norm_text)

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
            # コスト条件
            cost_m = re.search(_nfc(r'コスト(\d+)(以上|以下)'), norm_text)
            if cost_m:
                val["cost"] = int(cost_m.group(1))
                val["cost_op"] = CompareOperator.GE if cost_m.group(2) == _nfc('以上') else CompareOperator.LE
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

    def _extract_options(self, text: str) -> List[str]:
        norm_text = _nfc(text)
        lines = norm_text.split('\n')
        options = [re.sub(_nfc(r'^[・\-]\s*'), '', l).strip() for l in lines if l.strip().startswith((_nfc('・'), _nfc('-')))]
        if not options:
            parts = re.split(_nfc(r'、'), norm_text)
            options = [p.strip() for p in parts if _nfc("選ぶ") not in p and _nfc("以下から") not in p]
        return options
