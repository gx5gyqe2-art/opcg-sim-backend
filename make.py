import os
import re

# File Paths
path_effect_types = os.path.join("opcg_sim", "src", "models", "effect_types.py")
path_matcher = os.path.join("opcg_sim", "src", "core", "effects", "matcher.py")
path_parser = os.path.join("opcg_sim", "src", "core", "effects", "parser.py")
path_resolver = os.path.join("opcg_sim", "src", "core", "effects", "resolver.py")

# ---------------------------------------------------------
# 1. Update effect_types.py (Inject fields to TargetQuery)
# ---------------------------------------------------------
def update_effect_types():
    if not os.path.exists(path_effect_types):
        print(f"⚠️ {path_effect_types} not found. Skipping.")
        return

    with open(path_effect_types, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if fields already exist
    if "is_partial_match" in content:
        print(f"ℹ️ {path_effect_types} already has new fields.")
        return

    # Regex to find the TargetQuery class definition end or fields
    # We look for the last field definition in TargetQuery and append new ones
    pattern = r"(class TargetQuery.*?:[\s\S]*?)(    [a-zA-Z0-9_]+:.*? = .*?\n)(?!\s+[a-zA-Z0-9_]+:)"
    
    # Simple insertion: find "select_mode: str" (one of the fields) and append after it if strict regex fails
    # Or just replace the class end.
    # Let's try to append at the end of the dataclass fields
    
    new_fields = "    is_partial_match: bool = False\n    check_original_power: bool = False\n"
    
    if "tag: Optional[str] = None" in content:
        content = content.replace("tag: Optional[str] = None", "tag: Optional[str] = None\n" + new_fields)
        with open(path_effect_types, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ {path_effect_types} updated.")
    else:
        print(f"⚠️ Could not find insertion point in {path_effect_types}.")

# ---------------------------------------------------------
# 2. Matcher Code
# ---------------------------------------------------------
matcher_code = """import re
import logging
import unicodedata
from ...models.effect_types  import TargetQuery, _nfc
from ...models.enums import Player, Zone, ParserKeyword
from ...utils.logger_config import log_event

def parse_target(tgt_text: str, default_player: Player = Player.SELF) -> TargetQuery:
    tq = TargetQuery(raw_text=tgt_text, player=default_player)
    
    log_event("DEBUG", "matcher.parse_start", f"Parsing target text: {tgt_text}")

    if tgt_text == _nfc(ParserKeyword.THIS_CARD) or (tgt_text == _nfc(ParserKeyword.SELF_REF) and _nfc(ParserKeyword.SELF_REF + "の") not in tgt_text):
        tq.select_mode = "SOURCE"
        return tq

    if _nfc(ParserKeyword.REMAINING) in tgt_text:
        tq.select_mode = "REMAINING"
        tq.count = -1
        tq.zone = Zone.TEMP
        return tq

    # --- Player Detection ---
    if _nfc(ParserKeyword.EACH_OTHER) in tgt_text: tq.player = Player.ALL
    elif _nfc(ParserKeyword.OPPONENT) in tgt_text: tq.player = Player.OPPONENT
    elif _nfc(ParserKeyword.OWNER) in tgt_text: 
        is_dest = False
        for suffix in ["の手札", "のデッキ", "のライフ", "のトラッシュ"]:
            if _nfc(ParserKeyword.OWNER + suffix) in tgt_text:
                is_dest = True
                break
        if not is_dest:
            tq.player = Player.OWNER
        elif _nfc(ParserKeyword.OPPONENT) in tgt_text:
            tq.player = Player.OPPONENT
        else:
            tq.player = default_player
    elif _nfc(ParserKeyword.SELF) in tgt_text or _nfc(ParserKeyword.SELF_REF) in tgt_text: tq.player = Player.SELF

    # --- Zone Detection ---
    zone_map = {
        _nfc("手札"): Zone.HAND,
        _nfc("トラッシュ"): Zone.TRASH,
        _nfc("ライフ"): Zone.LIFE,
        _nfc("デッキ"): Zone.DECK,
        _nfc("コストエリア"): Zone.COST_AREA,
        _nfc("場"): Zone.FIELD
    }
    
    found_zone = None
    pattern = re.compile(r'(手札|トラッシュ|ライフ|デッキ|場|コストエリア)(?:.{0,5})(?:を|から|の)')
    matches = pattern.finditer(tgt_text)
    
    for m in matches:
        z_name = _nfc(m.group(1))
        post_match = tgt_text[m.end():]
        if z_name == _nfc("デッキ") and (_nfc("下") in post_match or _nfc("上") in post_match):
             if _nfc("から") not in post_match[:5]: 
                 continue
        if z_name in zone_map:
            found_zone = zone_map[z_name]
            break
    
    if not found_zone:
        if _nfc(ParserKeyword.LEADER) in tgt_text or _nfc(ParserKeyword.CHARACTER) in tgt_text:
            found_zone = Zone.FIELD
        elif _nfc("ドン") in tgt_text:
            found_zone = Zone.COST_AREA

    if found_zone:
        tq.zone = found_zone
    else:
        tq.zone = Zone.FIELD

    # --- Card Type ---
    if _nfc(ParserKeyword.LEADER) in tgt_text: tq.card_type.append("LEADER")
    if _nfc(ParserKeyword.CHARACTER) in tgt_text: tq.card_type.append("CHARACTER")
    if _nfc(ParserKeyword.EVENT) in tgt_text: tq.card_type.append("EVENT")
    if _nfc(ParserKeyword.STAGE) in tgt_text: tq.card_type.append("STAGE")
    
    # --- Filters ---
    # Name Logic
    m_name = re.search(r'「([^」]+)」', tgt_text)
    if m_name:
        if (m_name.group(0) + _nfc(ParserKeyword.EXCEPT)) not in tgt_text:
            tq.names.append(m_name.group(1))
    
    if _nfc("含む") in tgt_text:
        tq.is_partial_match = True

    # Traits
    traits = re.findall(_nfc(ParserKeyword.TRAIT + r'[《<]([^》>]+)[》>]'), tgt_text)
    tq.traits.extend(traits)
    
    # Attributes (Allow 《》, <>, (), [])
    attrs = re.findall(_nfc(ParserKeyword.ATTRIBUTE + r'[《<(\[]([^》>\])]+)[》>\])]'), tgt_text)
    tq.attributes.extend(attrs)
    
    for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
        if f"{c}の" in tgt_text: tq.colors.append(c)

    # --- Cost Filter ---
    m_c = re.search(_nfc(ParserKeyword.COST + r'[^+\-\d]?(\d+)(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_c:
        start_idx = m_c.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        
        end_idx = m_c.end()
        post_match = tgt_text[end_idx:]
        is_set_action = _nfc("にする") in post_match[:5]

        if prefix_context not in ['+', '-', '\\u2212', '\\u2010'] and not is_set_action:
            val = int(m_c.group(1))
            if m_c.group(2) == _nfc(ParserKeyword.ABOVE): tq.cost_min = val
            else: tq.cost_max = val

    # --- Power Filter ---
    if _nfc("元々") in tgt_text:
        tq.check_original_power = True

    m_p = re.search(_nfc(ParserKeyword.POWER + r'[^+\-\d]?(\d+)(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_p:
        start_idx = m_p.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        # Ignore "+3000" etc (Buffs)
        if prefix_context not in ['+', '-', '\\u2212', '\\u2010']:
            val = int(m_p.group(1))
            if m_p.group(2) == _nfc(ParserKeyword.ABOVE): tq.power_min = val
            else: tq.power_max = val
    
    # --- Status Filter ---
    # Don't treat "レストにする" or "レストにできる" as filtering for Rested cards
    if _nfc("にする") not in tgt_text and _nfc("ならない") not in tgt_text and _nfc("できる") not in tgt_text:
        if _nfc(ParserKeyword.REST) in tgt_text or _nfc("レスト") in tgt_text: tq.is_rest = True
        elif _nfc("アクティブ") in tgt_text: tq.is_rest = False
    
    if re.search(r'(\d+|枚)まで', tgt_text): tq.is_up_to = True 

    if _nfc(ParserKeyword.ALL_HIRAGANA) in tgt_text or _nfc(ParserKeyword.ALL) in tgt_text:
        tq.count = -1
        tq.select_mode = "ALL"
    else:
        m_cnt = re.search(r'(\d+)' + _nfc(ParserKeyword.COUNT_SUFFIX), tgt_text)
        tq.count = int(m_cnt.group(1)) if m_cnt else 1
    
    # --- Vanilla Check ---
    if _nfc("効果のない") in tgt_text or _nfc("効果がない") in tgt_text:
        tq.is_vanilla = True

    log_event("DEBUG", "matcher.parse_result", f"Parsed: player={tq.player.name}, count={tq.count}, up_to={tq.is_up_to}, zone={tq.zone}")
    return tq

def get_target_cards(game_manager, query: TargetQuery, source_card) -> list:
    if query.select_mode == "SOURCE": return [source_card]

    owner_player = game_manager.p1 if game_manager.p1.name == source_card.owner_id else game_manager.p2
    opponent_player = game_manager.p2 if owner_player == game_manager.p1 else game_manager.p1

    target_players = []
    if query.player == Player.SELF: target_players = [owner_player]
    elif query.player == Player.OPPONENT: target_players = [opponent_player]
    elif query.player == Player.ALL: target_players = [game_manager.p1, game_manager.p2]
    elif query.player == Player.OWNER: target_players = [owner_player]

    candidates = []
    for p in target_players:
        if not p: continue
        if query.zone == Zone.FIELD:
            candidates.extend(p.field)
            if not query.card_type or "LEADER" in query.card_type:
                if p.leader: candidates.append(p.leader)
            if p.stage: candidates.append(p.stage)
        elif query.zone == Zone.HAND: candidates.extend(p.hand)
        elif query.zone == Zone.TRASH: candidates.extend(p.trash)
        elif query.zone == Zone.LIFE: candidates.extend(p.life)
        elif query.zone == Zone.DECK: candidates.extend(p.deck)
        elif query.zone == Zone.COST_AREA:
            # Return active dons usually for costs, but if returning don, maybe any?
            # Default to active + rested for general search, logic filters status
            candidates.extend(p.don_active)
            candidates.extend(p.don_rested)
        elif query.zone == Zone.TEMP: candidates.extend(p.temp_zone)

    results = []
    for card in candidates:
        if not card: continue
        
        # Don checks
        if query.zone == Zone.COST_AREA:
            # Don usually doesn't have traits/cost/power in the same way, skip those checks if checking Don
            if query.is_rest is not None and card.is_rest != query.is_rest: continue
            results.append(card)
            continue

        if query.colors and not any(c in card.master.color.value for c in query.colors): continue
        if query.attributes and card.master.attribute.value not in query.attributes: continue
        
        if query.cost_max is not None and card.current_cost > query.cost_max: continue
        if query.cost_min is not None and card.current_cost < query.cost_min: continue
        
        # Power Check
        pwr = card.master.power if getattr(query, 'check_original_power', False) else card.get_power(True)
        if query.power_max is not None and pwr > query.power_max: continue
        if query.power_min is not None and pwr < query.power_min: continue
        
        # Vanilla Filter
        if getattr(query, 'is_vanilla', False):
            txt = card.master.effect_text
            if txt and txt.strip() not in ["", "なし", "-"]: continue

        # Name Check
        if query.names:
            if getattr(query, 'is_partial_match', False):
                # Partial: if ANY query name is in card name
                if not any(n in card.master.name for n in query.names): continue
            else:
                # Exact: if name not in query names
                if card.master.name not in query.names: continue

        if query.traits and not any(t in card.master.traits for t in query.traits): continue
        if query.is_rest is not None and card.is_rest != query.is_rest: continue
        results.append(card)

    if not results:
        log_level = "WARNING"
        if query.select_mode in ["ALL", "REMAINING"] or query.is_up_to: log_level = "INFO"
        log_event(level_key=log_level, action="matcher.no_target", msg=f"No targets found for query: {query.raw_text}", player="system", payload={"query_raw": query.raw_text, "zone": query.zone.name, "target_player": query.player.name, "real_target_names": [p.name for p in target_players], "candidates_scanned": len(candidates)})

    return results
"""

# ---------------------------------------------------------
# 3. Parser Code
# ---------------------------------------------------------
parser_code = """from __future__ import annotations
import re
import unicodedata
from typing import List, Optional, Tuple
from ...models.effect_types import Ability, EffectAction, TargetQuery, Condition, _nfc
from ...models.enums import (
    Phase, Player, Zone, ActionType, TriggerType, 
    CompareOperator, ConditionType, ParserKeyword
)
from .matcher import parse_target

class Effect:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text
        self.abilities: List[Ability] = []
        self._parse()

    def _normalize(self, text: str) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        
        # Map circled numbers to Don cost text
        circled = {'①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
                   '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10'}
        for k, v in circled.items():
             text = text.replace(k, f"ドン!!{v}枚をレストにできる。")

        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'（.*?）', '', text)
        
        replacements = {
            '[': '『', ']': '』', '<': '《', '>': '》', 
            '(': '(', ')': ')', '【': '『', '】': '』',
            '：': ':', '。': '。', '、': '、',
            '−': '-', '‒': '-', '–': '-',
            '＋': '+', '➕': '+',
            '／': '/',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'ドン!!', 'ドン', text)
        text = re.sub(r'DON!!', 'ドン', text)
        
        # Handle "Don and Character" compound cost
        # "自分のドン1枚とこのキャラをレストにできる" -> "自分のドン1枚をレストにできる。このキャラをレストにできる"
        text = re.sub(r'(ドン\d+枚)と(.+?をレストにできる)', r'\\1をレストにできる。\\2', text)
        
        return text

    def _parse(self):
        if not self.raw_text: return
        normalized = self._normalize(self.raw_text)
        parts = [p for p in normalized.split('/') if p.strip()]
        for part in parts:
            trigger = self._detect_trigger(part)
            body_text = re.sub(r'『[^』]+』', '', part)
            costs = []
            actions = []
            if ':' in body_text:
                cost_text, effect_text = body_text.rsplit(':', 1)
                costs = self._parse_recursive(cost_text, is_cost=True)
                actions = self._parse_recursive(effect_text)
            else:
                actions = self._parse_recursive(body_text)
            if actions or costs:
                self.abilities.append(Ability(trigger=trigger, costs=costs, actions=actions, raw_text=part))

    def _detect_trigger(self, text: str) -> TriggerType:
        if '『登場時』' in text: return TriggerType.ON_PLAY
        if '『起動メイン』' in text: return TriggerType.ACTIVATE_MAIN
        if '『相手のアタック時』' in text: return TriggerType.ON_OPP_ATTACK
        if '『アタック時』' in text: return TriggerType.ON_ATTACK
        if '『ブロック時』' in text: return TriggerType.ON_BLOCK
        if '『KO時』' in text: return TriggerType.ON_KO
        if '『相手のターン終了時』' in text: return TriggerType.OPP_TURN_END
        if '『ターン終了時』' in text: return TriggerType.TURN_END
        if '『自分のターン中』' in text: return TriggerType.PASSIVE
        if '『相手のターン中』' in text: return TriggerType.PASSIVE
        if '『カウンター』' in text: return TriggerType.COUNTER
        if '『トリガー』' in text: return TriggerType.TRIGGER
        return TriggerType.UNKNOWN

    def _parse_recursive(self, text: str, is_cost: bool = False) -> List[EffectAction]:
        if not text: return []
        sentences = [s for s in text.split('。') if s]
        root_actions = []
        last_action = None

        for sentence in sentences:
            parts = re.split(r'その後、|、その後', sentence)
            for part in parts:
                current_actions = self._parse_logic_block(part, is_cost)
                for act in current_actions:
                    if last_action:
                        last_action.then_actions.append(act)
                    else:
                        root_actions.append(act)
                    last_action = self._get_deepest_action(act)
        return root_actions

    def _get_deepest_action(self, action: EffectAction) -> EffectAction:
        if not action.then_actions:
            return action
        return self._get_deepest_action(action.then_actions[-1])

    def _parse_logic_block(self, text: str, is_cost: bool) -> List[EffectAction]:
        or_action = self._parse_or_split(text, is_cost)
        if or_action:
            return [or_action]

        match = re.search(r'^(.+?)(場合|なら|することで)、(.+)$', text)
        if match:
            condition_text, _, result_text = match.groups()
            condition = self._parse_condition(condition_text)
            then_actions = self._parse_recursive(result_text, is_cost)
            return [EffectAction(
                type=ActionType.OTHER,
                condition=condition,
                then_actions=then_actions,
                raw_text=text
            )]
        return self._parse_atomic_action(text, is_cost)

    def _parse_or_split(self, text: str, is_cost: bool) -> Optional[EffectAction]:
        if 'か、' in text:
            parts = text.split('か、', 1)
            if len(parts) == 2:
                part_a_raw = parts[0].strip()
                part_b_raw = parts[1].strip()
                
                match = re.search(r'(を|に)(.+)$', part_b_raw)
                if match:
                    connector = match.group(1)
                    verb = match.group(2)
                    text_a = f"{part_a_raw}{connector}{verb}"
                    text_b = part_b_raw
                    actions_a = self._parse_recursive(text_a, is_cost)
                    actions_b = self._parse_recursive(text_b, is_cost)
                    
                    if actions_a and actions_b:
                        return EffectAction(
                            type=ActionType.SELECT_OPTION,
                            details={
                                "resolvable_options": [actions_a[0], actions_b[0]],
                                "option_labels": [text_a, text_b]
                            },
                            raw_text=text
                        )
        return None

    def _parse_atomic_action(self, text: str, is_cost: bool) -> List[EffectAction]:
        if '見て' in text or '公開' in text or '見る' in text:
            return self._handle_look_action(text)

        act_type = self._detect_action_type(text)
        val = self._extract_value_for_action(text, act_type)

        target = None
        NO_TARGET_ACTIONS = [
            ActionType.DRAW, 
            ActionType.RAMP_DON, 
            ActionType.SHUFFLE, 
            ActionType.LIFE_RECOVER,
            ActionType.VICTORY,
            ActionType.RULE_PROCESSING,
            ActionType.SELECT_OPTION,
            ActionType.REPLACE_EFFECT,
            ActionType.MODIFY_DON_PHASE,
            ActionType.PASSIVE_EFFECT,
            ActionType.ACTIVE_DON,
            ActionType.OTHER
        ]
        
        # Don returning is usually ActionType.RETURN_DON, which doesn't select targets in current logic?
        # But if the text says "Don 1 to deck", it's RETURN_DON.
        
        is_calculation_or_rule = any(kw in text for kw in ["につき", "できない", "されない", "いる"])
        
        if act_type not in NO_TARGET_ACTIONS and not is_calculation_or_rule:
            if any(kw in text for kw in ['それ', 'そのカード', 'そのキャラ']):
                target = TargetQuery(select_mode="REFERENCE", raw_text="last_target")
                if not target.tag: target.tag = "last_target"
            else:
                default_p = Player.SELF
                if act_type in [ActionType.KO, ActionType.DEAL_DAMAGE, ActionType.REST, ActionType.ATTACK_DISABLE, ActionType.FREEZE, ActionType.MOVE_TO_HAND]:
                    if "自分" not in text:
                        default_p = Player.OPPONENT
                
                target = parse_target(text, default_player=default_p)
                
                if any(kw in text for kw in ['選び', '対象とし']):
                    target.tag = "last_target"
        
        return [EffectAction(
            type=act_type,
            target=target,
            value=val,
            raw_text=text
        )]

    def _extract_value_for_action(self, text: str, act_type: ActionType) -> int:
        num_pattern = r'([+\-＋−]?\s*\d+)'
        
        if act_type == ActionType.BUFF:
            match = re.search(r'パワー' + num_pattern, text)
            if match: return int(self._normalize_number_str(match.group(1)))
        
        if act_type == ActionType.COST_CHANGE:
            match = re.search(r'コスト' + num_pattern, text)
            if match: return int(self._normalize_number_str(match.group(1)))
        
        if act_type == ActionType.SET_COST:
            match = re.search(r'コスト(?:を)?(\d+)にする', text)
            if match: return int(match.group(1))

        return self._extract_number(text)

    def _normalize_number_str(self, s: str) -> str:
        s = unicodedata.normalize('NFKC', s)
        s = s.replace(' ', '').replace('　', '')
        return s

    def _extract_number(self, text: str) -> int:
        match = re.search(r'([-\u2212\u2010\u2011\u2012\u2013\u2014\u2015\uff0d+]?)(\d+)', text)
        if match:
            sign = match.group(1)
            num = int(match.group(2))
            if sign in ['-', '\u2212', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\uff0d']:
                return -num
            return num
        return 0

    def _detect_action_type(self, text: str) -> ActionType:
        if 'ドン' in text:
            if ('戻す' in text or 'ドンデッキ' in text or '-' in text or '−' in text):
                return ActionType.RETURN_DON
            if '付与されているドン' in text and '付与する' in text:
                return ActionType.MOVE_ATTACHED_DON
            if '付与' in text or '付ける' in text:
                return ActionType.ATTACH_DON
            if 'ドンフェイズ' in text:
                return ActionType.MODIFY_DON_PHASE
            if '追加' in text:
                return ActionType.RAMP_DON
            if 'アクティブ' in text:
                return ActionType.ACTIVE_DON
            # If "Don ... Rest", it's ActionType.REST
            if 'レスト' in text:
                return ActionType.REST

        if 'ライフ' in text:
            if any(k in text for k in ['加える', '置く', '向き', '手札', 'トラッシュ']):
                return ActionType.LIFE_MANIPULATE

        if 'アタック' in text and '対象' in text and '変更' in text:
            return ActionType.REDIRECT_ATTACK

        if 'ダメージ' in text and ('与え' in text or '受ける' in text):
            return ActionType.DEAL_DAMAGE
            
        if 'アクティブにならない' in text:
            return ActionType.FREEZE

        if '代わりに' in text: return ActionType.REPLACE_EFFECT
        if '選ぶ' in text and ('つ' in text or 'から' in text): return ActionType.SELECT_OPTION
        if 'シャッフル' in text: return ActionType.SHUFFLE
        if 'コスト' in text and 'にする' in text: return ActionType.SET_COST
        if '場を離れない' in text: return ActionType.PREVENT_LEAVE
        if 'できない' in text or '不可' in text or '加えられない' in text: return ActionType.RESTRICTION
        if '発動する' in text and ('効果' in text or 'イベント' in text): return ActionType.EXECUTE_MAIN_EFFECT
        if '勝利する' in text and ('ゲーム' in text or '敗北' in text): return ActionType.VICTORY
        if 'としても扱う' in text or '何枚でも' in text or 'カウンター' in text: return ActionType.RULE_PROCESSING
        if 'アタック' in text and ('できない' in text or '不可' in text): return ActionType.ATTACK_DISABLE
        if '無効' in text: return ActionType.NEGATE_EFFECT
            
        if 'デッキ' in text and '上' in text and ('置く' in text or '戻す' in text or '加える' in text): return ActionType.DECK_TOP

        if 'コスト' in text and ('-' in text or '下げる' in text or '+' in text or '上げる' in text):
             return ActionType.COST_CHANGE
        
        if '得る' in text: return ActionType.GRANT_KEYWORD
        if '引く' in text: return ActionType.DRAW
        if '登場' in text: return ActionType.PLAY_CARD
        if 'KO' in text: return ActionType.KO
        if '手札' in text and ('戻す' in text or '加える' in text): return ActionType.MOVE_TO_HAND
        if 'トラッシュ' in text or '捨てる' in text: return ActionType.TRASH
        if 'デッキ' in text and '下' in text: return ActionType.DECK_BOTTOM
        if 'パワー' in text: return ActionType.BUFF
        if 'レスト' in text: return ActionType.REST
        if 'アクティブ' in text: return ActionType.ACTIVE
        
        return ActionType.OTHER

    def _parse_condition(self, text: str) -> Optional[Condition]:
        type_ = ConditionType.NONE
        op = CompareOperator.EQ
        val = 0
        target_in_condition = None

        if '公開したカード' in text:
            type_ = ConditionType.CONTEXT
            if 'イベント' in text: val = "TYPE_EVENT"
            elif 'キャラ' in text: val = "TYPE_CHARACTER"
            elif '特徴' in text:
                val = "HAS_TRAIT"
                m = re.search(r'[《<]([^》>]+)[》>]', text)
                if m: target_in_condition = TargetQuery(raw_text=m.group(0), traits=[m.group(1)])
            elif 'コスト' in text:
                val = "COST_CHECK"
                nums = re.findall(r'(\d+)', text)
                if nums: target_in_condition = TargetQuery(raw_text=text, cost_min=int(nums[0]))

        elif 'そうしなかった' in text:
            type_ = ConditionType.CONTEXT
            val = "LAST_ACTION_FAILURE"

        elif 'そうした' in text or '登場させた' in text:
            type_ = ConditionType.CONTEXT
            val = "LAST_ACTION_SUCCESS"
        
        elif 'ライフ' in text: type_ = ConditionType.LIFE_COUNT
        elif 'ドン' in text: type_ = ConditionType.DON_COUNT
        elif '手札' in text: type_ = ConditionType.HAND_COUNT
        elif 'トラッシュ' in text: type_ = ConditionType.TRASH_COUNT
        elif 'デッキ' in text: type_ = ConditionType.DECK_COUNT
        elif '特徴' in text: type_ = ConditionType.HAS_TRAIT
        elif 'リーダー' in text: type_ = ConditionType.LEADER_NAME
        elif 'キャラ' in text or '持つ' in text: type_ = ConditionType.HAS_UNIT

        if type_ not in [ConditionType.CONTEXT, ConditionType.NONE]:
             if type_ in [ConditionType.HAS_TRAIT, ConditionType.HAS_UNIT]:
                 target_in_condition = parse_target(text)
             
             nums = re.findall(r'(\d+)', text)
             if nums: val = int(nums[0])

        if '以上' in text: op = CompareOperator.GE
        elif '以下' in text: op = CompareOperator.LE
        
        str_val = ""
        m_name = re.search(r'[「『]([^」』]+)[」』]', text)
        if m_name: 
            str_val = m_name.group(1)
            if type_ == ConditionType.NONE: type_ = ConditionType.LEADER_NAME
        
        m_trait = re.search(r'[《<]([^》>]+)[》>]', text)
        if m_trait:
            str_val = m_trait.group(1)
            if type_ != ConditionType.CONTEXT:
                type_ = ConditionType.HAS_TRAIT
                op = CompareOperator.HAS

        if type_ == ConditionType.LEADER_NAME: 
            val = str_val
            op = CompareOperator.EQ
        elif type_ == ConditionType.HAS_TRAIT and type_ != ConditionType.CONTEXT:
            val = str_val
            op = CompareOperator.HAS

        return Condition(
            type=type_, 
            operator=op, 
            value=val, 
            target=target_in_condition,
            raw_text=text
        )

    def _handle_look_action(self, text: str) -> List[EffectAction]:
        val = self._extract_number(text)
        if val <= 0: val = 1
        
        look = EffectAction(
            type=ActionType.LOOK, 
            value=val, 
            source_zone=Zone.DECK, 
            dest_zone=Zone.TEMP, 
            raw_text=f"デッキの上から{val}枚を見る"
        )
        
        if '加える' in text or '公開' in text:
            move_target = parse_target(text)
            move_target.zone = Zone.TEMP
            move_target.tag = "last_target"
            
            move = EffectAction(
                type=ActionType.MOVE_TO_HAND, 
                target=move_target, 
                source_zone=Zone.TEMP, 
                dest_zone=Zone.HAND,
                raw_text="選択して手札に加える"
            )
            look.then_actions.append(move)
            
        if '残り' in text or '下' in text:
            rem_target = TargetQuery(zone=Zone.TEMP, select_mode="ALL", player=Player.SELF)
            
            dest_z = Zone.DECK
            dest_pos = "BOTTOM"
            act_t = ActionType.DECK_BOTTOM
            raw_t = "残りをデッキの下に置く"

            if 'トラッシュ' in text:
                dest_z = Zone.TRASH
                dest_pos = None
                act_t = ActionType.TRASH
                raw_t = "残りをトラッシュに置く"

            remainder_action = EffectAction(
                type=act_t, 
                target=rem_target, 
                source_zone=Zone.TEMP, 
                dest_zone=dest_z, 
                dest_position=dest_pos,
                raw_text=raw_t
            )
            if look.then_actions:
                look.then_actions[-1].then_actions.append(remainder_action)
            else:
                look.then_actions.append(remainder_action)

        return [look]
"""

# ---------------------------------------------------------
# 4. Resolver Code
# ---------------------------------------------------------
resolver_code = """from typing import Optional, List, Any, Dict
import random
import copy
from ...models.enums import ActionType, Zone, ConditionType, CompareOperator, TriggerType
from ...models.effect_types import EffectAction, Condition
from ...models.models import CardType, DonInstance
from ...utils.logger_config import log_event

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..gamestate import GameManager, Player, CardInstance

def check_condition(game_manager: 'GameManager', player: 'Player', condition: Optional[Condition], source_card: 'CardInstance', effect_context: Optional[Dict[str, Any]] = None) -> bool:
    if not condition: return True
    from .matcher import get_target_cards
    
    if effect_context is None: effect_context = {}

    res = False
    
    if condition.type == ConditionType.CONTEXT:
        if condition.value == "LAST_ACTION_SUCCESS":
            res = effect_context.get("last_action_success", False)
            log_event("DEBUG", "resolver.ctx_check", f"Checking LAST_ACTION_SUCCESS: {res}", player=player.name)
        
        elif str(condition.value).startswith("TYPE_") or condition.value in ["HAS_TRAIT", "COST_CHECK"]:
            revealed = effect_context.get("revealed_cards", [])
            if not revealed:
                res = False
            else:
                target_card = revealed[0]
                
                if condition.value == "TYPE_EVENT":
                    res = (target_card.master.type == CardType.EVENT)
                elif condition.value == "TYPE_CHARACTER":
                    res = (target_card.master.type == CardType.CHARACTER)
                elif condition.value == "HAS_TRAIT" and condition.target:
                    res = any(t in target_card.master.traits for t in condition.target.traits)
                elif condition.value == "COST_CHECK" and condition.target:
                    res = True
                    if condition.target.cost_min is not None and target_card.current_cost < condition.target.cost_min:
                        res = False

    elif condition.target:
        matches = get_target_cards(game_manager, condition.target, source_card)
        res = len(matches) > 0
        log_event("DEBUG", "resolver.check_condition_target", f"Target condition: {len(matches)} matches", player=player.name)
    elif condition.type == ConditionType.LIFE_COUNT:
        val = len(player.life)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.HAND_COUNT:
        val = len(player.hand)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.DON_COUNT:
        val = len(player.don_active) + len(player.don_rested)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.TRASH_COUNT:
        val = len(player.trash)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.DECK_COUNT:
        val = len(player.deck)
        if condition.operator == CompareOperator.GE: res = val >= condition.value
        elif condition.operator == CompareOperator.LE: res = val <= condition.value
        else: res = val == condition.value
    elif condition.type == ConditionType.HAS_TRAIT:
        has_in_field = any(condition.value in c.master.traits for c in player.field)
        has_in_source = (source_card and condition.value in source_card.master.traits)
        res = has_in_field or has_in_source
    elif condition.type == ConditionType.LEADER_NAME:
        res = player.leader and player.leader.master.name == condition.value
    
    log_event("INFO", "resolver.condition_result", f"Condition [{condition.raw_text}]: {res}", player=player.name)
    return res

def execute_action(
    game_manager: 'GameManager', 
    player: 'Player', 
    action: EffectAction, 
    source_card: 'CardInstance', 
    effect_context: Optional[Dict[str, Any]] = None
) -> bool:
    from .matcher import get_target_cards
    if effect_context is None: effect_context = {}

    if not check_condition(game_manager, player, action.condition, source_card, effect_context):
        return True

    targets = []
    selected_uuids = effect_context.get("selected_uuids")
    
    if action.type == ActionType.SELECT_OPTION:
        selected_option = effect_context.get("selected_option_index")
        
        if selected_option is None:
            labels = []
            if action.details and "option_labels" in action.details:
                labels = [{"label": l, "value": i} for i, l in enumerate(action.details["option_labels"])]
            else:
                labels = [{"label": "選択肢1", "value": 0}, {"label": "選択肢2", "value": 1}]

            game_manager.active_interaction = {
                "player_id": player.name,
                "action_type": "SELECT_OPTION", 
                "message": action.raw_text or "効果を選択してください",
                "options": labels,
                "can_skip": False,
                "continuation": {
                    "action": action,
                    "source_card_uuid": source_card.uuid,
                    "effect_context": effect_context
                }
            }
            log_event("INFO", "resolver.select_option_suspend", "Suspended for Option Selection", player=player.name)
            return False
            
        else:
            log_event("INFO", "resolver.select_option_resume", f"Option {selected_option} selected", player=player.name)
            
            if "selected_option_index" in effect_context:
                del effect_context["selected_option_index"]

            if action.details and "resolvable_options" in action.details:
                options = action.details["resolvable_options"]
                if 0 <= selected_option < len(options):
                    chosen_action = options[selected_option]
                    log_event("INFO", "resolver.execute_option", f"Executing option {selected_option}", player=player.name)
                    
                    effective_action = copy.copy(action)
                    effective_action.type = chosen_action.type
                    effective_action.target = chosen_action.target
                    effective_action.value = chosen_action.value
                    effective_action.condition = chosen_action.condition
                    effective_action.raw_text = chosen_action.raw_text
                    effective_action.details = chosen_action.details
                    effective_action.source_zone = chosen_action.source_zone
                    effective_action.dest_zone = chosen_action.dest_zone
                    
                    effective_action.then_actions = list(action.then_actions)
                    if chosen_action.then_actions:
                        effective_action.then_actions[0:0] = chosen_action.then_actions

                    sub_success = execute_action(game_manager, player, effective_action, source_card, effect_context)
                    if not sub_success: return False

    if action.target:
        if action.target.select_mode == "REFERENCE":
            last_uuid = effect_context.get("last_target_uuid")
            if last_uuid:
                ref_card = game_manager._find_card_by_uuid(last_uuid)
                if ref_card: targets = [ref_card]
            log_event("DEBUG", "resolver.resolve_reference", f"Resolved reference to: {[t.name for t in targets]}", player=player.name)
        else:
            candidates = get_target_cards(game_manager, action.target, source_card)
            
            is_search = (action.target.zone == Zone.TEMP) or (action.source_zone == Zone.TEMP)
            
            should_interact = action.target.select_mode not in ["ALL", "SOURCE", "SELF", "REMAINING"] and (len(candidates) > 0 or is_search)

            if should_interact:
                if selected_uuids is None:
                    log_event("INFO", "resolver.suspend", f"Selection required for {action.type}. Candidates: {len(candidates)}", player=player.name)
                    
                    display_candidates = candidates
                    if is_search:
                        display_candidates = player.temp_zone
                    
                    game_manager.active_interaction = {
                        "player_id": player.name,
                        "action_type": "SEARCH_AND_SELECT",
                        "message": action.raw_text or "対象を選択してください",
                        "candidates": display_candidates, 
                        "selectable_uuids": [c.uuid for c in candidates],
                        "can_skip": True,
                        "continuation": {
                            "action": action,
                            "source_card_uuid": source_card.uuid,
                            "effect_context": effect_context
                        }
                    }
                    return False
                
                targets = [c for c in candidates if c.uuid in selected_uuids]
                
                if "selected_uuids" in effect_context:
                    del effect_context["selected_uuids"]
            else:
                targets = candidates

    if targets and action.target and action.target.tag == "last_target":
        effect_context["last_target_uuid"] = targets[0].uuid

    action_success = self_execute(game_manager, player, action, targets, source_card=source_card, effect_context=effect_context)
    
    effect_context["last_action_success"] = action_success
    
    if not action_success:
        return False

    if action.then_actions:
        for sub in action.then_actions:
            if not execute_action(game_manager, player, sub, source_card, effect_context):
                return False
                
    return True

def self_execute(game_manager, player, action, targets, source_card=None, effect_context=None) -> bool:
    if effect_context is None: effect_context = {}
    is_success = True

    TARGET_REQUIRED_ACTIONS = [
        ActionType.KO, ActionType.MOVE_TO_HAND, ActionType.TRASH,
        ActionType.DECK_BOTTOM, ActionType.DECK_TOP, 
        ActionType.REST, ActionType.ACTIVE, ActionType.BUFF, 
        ActionType.COST_CHANGE, ActionType.GRANT_KEYWORD, 
        ActionType.ATTACK_DISABLE, ActionType.NEGATE_EFFECT, 
        ActionType.PREVENT_LEAVE, ActionType.ATTACH_DON, 
        ActionType.MOVE_ATTACHED_DON, ActionType.REDIRECT_ATTACK,
        ActionType.FREEZE, ActionType.PLAY_CARD, ActionType.SET_COST
    ]
    
    if action.type in TARGET_REQUIRED_ACTIONS:
        if action.target and not action.target.is_up_to:
            if not targets:
                log_event("DEBUG", "resolver.fail_no_target", f"Action {action.type} failed: Target required but not found.", player=player.name)
                return False

    if action.type == ActionType.DRAW:
        game_manager.draw_card(player, action.value)
    elif action.type == ActionType.RAMP_DON:
        for _ in range(action.value):
            if player.don_deck:
                don = player.don_deck.pop(0)
                if 'レスト' in action.raw_text:
                    don.is_rest = True
                    player.don_rested.append(don)
                else:
                    don.is_rest = False
                    player.don_active.append(don)
        log_event("INFO", "resolver.ramp_don", f"Ramped {action.value} Don", player=player.name)
    
    elif action.type == ActionType.LOOK:
        target_p = game_manager.opponent if '相手' in action.raw_text else player
        moved_count = 0
        moved_cards = []
        for _ in range(action.value):
            if target_p.deck:
                card = target_p.deck.pop(0)
                player.temp_zone.append(card)
                moved_cards.append(card)
                moved_count += 1
        
        effect_context["revealed_cards"] = moved_cards
        log_event("INFO", "resolver.look", f"Moved {moved_count} cards from {target_p.name}'s deck to temp_zone", player=player.name)
        if not moved_cards: is_success = False
    
    elif action.type == ActionType.KO:
        for t in targets:
            if "PREVENT_LEAVE" in t.flags:
                log_event("INFO", "resolver.prevent_leave", f"{t.master.name} is protected from leaving field", player=player.name)
                continue
            owner, _ = game_manager._find_card_location(t)
            if owner: game_manager.move_card(t, Zone.TRASH, owner)
    elif action.type == ActionType.MOVE_TO_HAND:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            real_owner = game_manager.p1 if t.owner_id == game_manager.p1.name else game_manager.p2
            if real_owner: game_manager.move_card(t, Zone.HAND, real_owner)

    elif action.type == ActionType.TRASH:
        if not targets and "できる" in action.raw_text:
             is_success = False
        
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            real_owner = game_manager.p1 if t.owner_id == game_manager.p1.name else game_manager.p2
            if real_owner: game_manager.move_card(t, Zone.TRASH, real_owner)

    elif action.type == ActionType.DECK_BOTTOM:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            real_owner = game_manager.p1 if t.owner_id == game_manager.p1.name else game_manager.p2
            if real_owner: game_manager.move_card(t, Zone.DECK, real_owner, dest_position="BOTTOM")

    elif action.type == ActionType.BUFF:
        for t in targets: t.power_buff += action.value
    elif action.type == ActionType.REST:
        for t in targets: 
            t.is_rest = True
            # Handle Don Cost Payment (Active -> Rested)
            if isinstance(t, DonInstance) or (hasattr(t, 'owner_id') and hasattr(t, 'attached_to')):
                if t in player.don_active:
                    player.don_active.remove(t)
                    player.don_rested.append(t)
                    log_event("INFO", "resolver.rest_don", "Don moved to rested state", player=player.name)

    elif action.type == ActionType.ACTIVE:
        for t in targets: t.is_rest = False
    
    elif action.type == ActionType.ACTIVE_DON:
        count = action.value
        if count <= 0: count = 1
        reactivated = 0
        
        # attached don -> active
        if player.don_attached_cards:
            while reactivated < count and player.don_attached_cards:
                don = player.don_attached_cards.pop()
                if don.attached_to:
                    attached_card = game_manager._find_card_by_uuid(don.attached_to)
                    if attached_card:
                        attached_card.attached_don = max(0, attached_card.attached_don - 1)
                don.attached_to = None
                don.is_rest = False
                player.don_active.append(don)
                reactivated += 1
                
        # rested don -> active
        if reactivated < count and player.don_rested:
            while reactivated < count and player.don_rested:
                don = player.don_rested.pop()
                don.is_rest = False
                player.don_active.append(don)
                reactivated += 1
                
        log_event("INFO", "resolver.active_don", f"Activated {reactivated} Don", player=player.name)

    elif action.type == ActionType.ATTACH_DON:
        if targets and player.don_active:
            don = player.don_active.pop(0)
            target_card = targets[0]
            don.attached_to = target_card.uuid
            player.don_attached_cards.append(don)
            target_card.attached_don += 1
            
    elif action.type == ActionType.COST_CHANGE:
        for t in targets:
            t.cost_buff += action.value
            log_event("INFO", "effect.cost_change", f"{t.master.name} cost buffed by {action.value}", player=player.name)

    elif action.type == ActionType.LIFE_MANIPULATE:
        moved_any = False
        if targets:
            for t in targets:
                owner, current_zone = game_manager._find_card_location(t)
                if not owner: continue
                
                if current_zone == owner.life:
                    dest = Zone.HAND
                    if "トラッシュ" in action.raw_text or "捨てる" in action.raw_text:
                         dest = Zone.TRASH
                    
                    game_manager.move_card(t, dest, owner)
                    log_event("INFO", "effect.life_move", f"Moved {t.master.name} from Life to {dest.name}", player=player.name)
                    moved_any = True
                else:
                    game_manager.move_card(t, Zone.LIFE, owner, dest_position="TOP")
                    log_event("INFO", "effect.life_recover", f"Added {t.master.name} to Life", player=player.name)
                    moved_any = True
        
        if not moved_any and not targets:
            if "加える" in action.raw_text or "回復" in action.raw_text or "デッキ" in action.raw_text:
                source_list = player.deck
                if source_list:
                    card = source_list.pop(0)
                    player.life.append(card)
                    log_event("INFO", "effect.life_recover", "Recovered 1 Life from Deck", player=player.name)
            elif "トラッシュ" in action.raw_text or "手札" in action.raw_text:
                is_success = False
                log_event("DEBUG", "resolver.life_manipulate_fail", "Target required for Life manipulation but not found", player=player.name)

    elif action.type == ActionType.GRANT_KEYWORD:
        keywords_map = {
            "速攻": "速攻",
            "ブロッカー": "ブロッカー",
            "バニッシュ": "バニッシュ",
            "ダブルアタック": "ダブルアタック",
            "突進": "突進",
            "再起動": "再起動"
        }
        found_kw = None
        for kw_jp, kw_internal in keywords_map.items():
            if kw_jp in action.raw_text:
                found_kw = kw_internal
                for t in targets:
                    t.current_keywords.add(found_kw)
                    log_event("INFO", "effect.grant_keyword", f"Granted [{found_kw}] to {t.master.name}", player=player.name)
                break 

    elif action.type == ActionType.ATTACK_DISABLE:
        for t in targets:
            t.flags.add("ATTACK_DISABLE")
            log_event("INFO", "effect.attack_disable", f"{t.master.name} cannot attack", player=player.name)

    elif action.type == ActionType.NEGATE_EFFECT:
        for t in targets:
            t.negated = True
            log_event("INFO", "effect.negate", f"{t.master.name} effects negated", player=player.name)

    elif action.type == ActionType.EXECUTE_MAIN_EFFECT:
        exec_targets = targets if targets else [source_card] if source_card else []
        for t in exec_targets:
             if not t or not t.master.abilities: continue
             target_ability = next((a for a in t.master.abilities if a.trigger not in [TriggerType.TRIGGER, TriggerType.COUNTER]), None)
             if target_ability:
                 log_event("INFO", "effect.execute_main", f"Executing nested ability of {t.master.name}", player=player.name)
                 game_manager.resolve_ability(player, target_ability, source_card=t)

    elif action.type == ActionType.VICTORY:
        game_manager.winner = player.name
        log_event("INFO", "game.victory_special", f"{player.name} wins by effect!", player=player.name)

    elif action.type == ActionType.RULE_PROCESSING:
        log_event("INFO", "effect.rule", f"Static rule processed: {action.raw_text}", player=player.name)

    elif action.type == ActionType.RESTRICTION:
        log_event("INFO", "effect.restriction", f"Restriction applied: {action.raw_text}", player=player.name)

    elif action.type == ActionType.DECK_TOP:
        for t in targets:
            owner, _ = game_manager._find_card_location(t)
            real_owner = game_manager.p1 if t.owner_id == game_manager.p1.name else game_manager.p2
            if real_owner:
                game_manager.move_card(t, Zone.DECK, real_owner, dest_position="TOP")
                log_event("INFO", "effect.deck_top", f"Moved {t.master.name} to Deck Top", player=player.name)

    elif action.type == ActionType.DEAL_DAMAGE:
        target_p = player if '自分' in action.raw_text or '受ける' in action.raw_text else game_manager.opponent
        if target_p.life:
            life_card = target_p.life.pop(0)
            target_p.hand.append(life_card)
            log_event("INFO", "effect.damage", f"{target_p.name} took 1 damage from effect", player=player.name)

    elif action.type == ActionType.SELECT_OPTION:
        log_event("INFO", "effect.select_option", "Option processed (Branch logic not implemented in Parser yet)", player=player.name)

    elif action.type == ActionType.SET_COST:
        for t in targets:
            new_buff = action.value - t.master.cost
            t.cost_buff = new_buff
            log_event("INFO", "effect.set_cost", f"Set {t.master.name} cost to {action.value}", player=player.name)

    elif action.type == ActionType.SHUFFLE:
        random.shuffle(player.deck)
        log_event("INFO", "effect.shuffle", "Deck shuffled", player=player.name)

    elif action.type == ActionType.PREVENT_LEAVE:
        for t in targets:
            t.flags.add("PREVENT_LEAVE")
            log_event("INFO", "effect.prevent_leave", f"{t.master.name} gained PREVENT_LEAVE", player=player.name)

    elif action.type == ActionType.REPLACE_EFFECT:
        log_event("INFO", "effect.replace", f"Replacement Effect: {action.raw_text}", player=player.name)

    elif action.type == ActionType.MOVE_ATTACHED_DON:
        don_source = source_card if source_card and source_card.attached_don > 0 else None
        
        if don_source and targets:
            target_card = targets[0]
            moving_don = next((d for d in player.don_attached_cards if d.attached_to == don_source.uuid), None)
            
            if moving_don:
                moving_don.attached_to = target_card.uuid
                don_source.attached_don -= 1
                target_card.attached_don += 1
                log_event("INFO", "effect.move_don", f"Moved Don from {don_source.master.name} to {target_card.master.name}", player=player.name)

    elif action.type == ActionType.MODIFY_DON_PHASE:
        log_event("INFO", "effect.modify_don_phase", f"Don Phase Modified: {action.raw_text}", player=player.name)

    elif action.type == ActionType.REDIRECT_ATTACK:
        if game_manager.active_battle and targets:
            new_target = targets[0]
            game_manager.active_battle["target"] = new_target
            log_event("INFO", "effect.redirect", f"Attack target changed to {new_target.master.name}", player=player.name)

    elif action.type == ActionType.RETURN_DON:
        target_player = game_manager.opponent if "相手" in action.raw_text else player
        count = abs(action.value) if action.value != 0 else 1
        
        returned_count = 0
        
        while returned_count < count and target_player.don_active:
            don = target_player.don_active.pop()
            target_player.don_deck.append(don)
            returned_count += 1
            
        while returned_count < count and target_player.don_rested:
            don = target_player.don_rested.pop()
            target_player.don_deck.append(don)
            returned_count += 1
            
        if returned_count < count and target_player.don_attached_cards:
            while returned_count < count and target_player.don_attached_cards:
                don = target_player.don_attached_cards.pop()
                if don.attached_to:
                    attached_card = game_manager._find_card_by_uuid(don.attached_to)
                    if attached_card:
                        attached_card.attached_don = max(0, attached_card.attached_don - 1)
                
                don.attached_to = None
                target_player.don_deck.append(don)
                returned_count += 1

        log_event("INFO", "effect.return_don", f"{target_player.name} returned {returned_count} Don to deck", player=player.name)
    
    elif action.type == ActionType.FREEZE:
        for t in targets:
            t.flags.add("FREEZE")
            log_event("INFO", "effect.freeze", f"{t.master.name} frozen", player=player.name)

    elif action.type == ActionType.PLAY_CARD:
        for t in targets:
            owner, current_zone = game_manager._find_card_location(t)
            if owner:
                game_manager.move_card(t, Zone.FIELD, owner)
                t.is_newly_played = True
                t.attached_don = 0
                log_event("INFO", "effect.play_card", f"Played {t.master.name} by effect", player=player.name)

                if not t.ability_disabled:
                    for ability in t.master.abilities:
                        if ability.trigger == TriggerType.ON_PLAY:
                            game_manager.resolve_ability(player, ability, source_card=t)

    return is_success
"""

def update_files():
    update_effect_types()
    
    with open(path_matcher, "w", encoding="utf-8") as f:
        f.write(matcher_code)
    print(f"✅ {path_matcher} updated.")

    with open(path_parser, "w", encoding="utf-8") as f:
        f.write(parser_code)
    print(f"✅ {path_parser} updated.")

    with open(path_resolver, "w", encoding="utf-8") as f:
        f.write(resolver_code)
    print(f"✅ {path_resolver} updated.")

if __name__ == "__main__":
    update_files()
