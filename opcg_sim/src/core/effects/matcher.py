import re
import logging
import unicodedata
from ...models.effect_types  import TargetQuery, _nfc
from ...models.enums import Player, Zone, ParserKeyword, Attribute
from ...utils.logger_config import log_event

def parse_target(tgt_text: str, default_player: Player = Player.SELF) -> TargetQuery:
    tq = TargetQuery(raw_text=tgt_text, player=default_player)

    if tgt_text == _nfc(ParserKeyword.THIS_CARD) or (tgt_text == _nfc(ParserKeyword.SELF_REF) and _nfc(ParserKeyword.SELF_REF + "の") not in tgt_text):
        tq.select_mode = "SOURCE"
        return tq

    if _nfc(ParserKeyword.REMAINING) in tgt_text:
        tq.select_mode = "REMAINING"
        tq.count = -1
        tq.zone = Zone.TEMP
        return tq

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
        elif _nfc(ParserKeyword.DON) in tgt_text:
            found_zone = Zone.COST_AREA

    if found_zone:
        tq.zone = found_zone
    else:
        tq.zone = Zone.FIELD

    if _nfc(ParserKeyword.LEADER) in tgt_text: tq.card_type.append("LEADER")
    if _nfc(ParserKeyword.CHARACTER) in tgt_text: tq.card_type.append("CHARACTER")
    if _nfc(ParserKeyword.EVENT) in tgt_text: tq.card_type.append("EVENT")
    if _nfc(ParserKeyword.STAGE) in tgt_text: tq.card_type.append("STAGE")
    
    m_name = re.search(r'「([^」]+)」', tgt_text)
    if m_name:
        if (m_name.group(0) + _nfc(ParserKeyword.EXCEPT)) not in tgt_text:
            tq.names.append(m_name.group(1))
    
    if _nfc("含む") in tgt_text:
        tq.flags.add("NAME_PARTIAL")
    
    raw_traits = re.findall(r'[《<]([^》>]+)[》>]', tgt_text)
    attr_values = [a.value for a in Attribute if a != Attribute.NONE]
    final_traits = []
    
    for t in raw_traits:
        if t in attr_values:
            tq.attributes.append(t)
        else:
            final_traits.append(t)
            
    tq.traits.extend(final_traits)

    attrs = re.findall(_nfc(ParserKeyword.ATTRIBUTE + r'[((]([^))]+)[))]'), tgt_text)
    tq.attributes.extend(attrs)
    
    for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
        if f"{c}の" in tgt_text: tq.colors.append(c)

    m_c = re.search(_nfc(ParserKeyword.COST + r'[^+\-\d]?(\d+)(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_c:
        start_idx = m_c.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        
        end_idx = m_c.end()
        post_match = tgt_text[end_idx:]
        is_set_action = _nfc("にする") in post_match[:5]

        if prefix_context not in ['+', '-', '\u2212', '\u2010'] and not is_set_action:
            val = int(m_c.group(1))
            if m_c.group(2) == _nfc(ParserKeyword.ABOVE): tq.cost_min = val
            else: tq.cost_max = val

    m_p = re.search(_nfc(ParserKeyword.POWER + r'[^+\-\d]?(\d+)\D?(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_p:
        start_idx = m_p.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        if prefix_context not in ['+', '-', '\u2212', '\u2010']:
            val = int(m_p.group(1))
            if m_p.group(2) == _nfc(ParserKeyword.ABOVE): tq.power_min = val
            else: tq.power_max = val
    
    if _nfc("にする") not in tgt_text and _nfc("ならない") not in tgt_text and _nfc("にできる") not in tgt_text:
        if _nfc(ParserKeyword.REST) in tgt_text: tq.is_rest = True
        elif _nfc("レスト") in tgt_text: tq.is_rest = True
        elif _nfc("アクティブ") in tgt_text: tq.is_rest = False
    
    if re.search(r'(\d+|枚)まで', tgt_text): tq.is_up_to = True 

    if _nfc(ParserKeyword.ALL_HIRAGANA) in tgt_text or _nfc(ParserKeyword.ALL) in tgt_text:
        tq.count = -1
        tq.select_mode = "ALL"
    else:
        m_cnt = re.search(r'(\d+)' + _nfc(ParserKeyword.COUNT_SUFFIX), tgt_text)
        tq.count = int(m_cnt.group(1)) if m_cnt else 1
    
    if _nfc("効果のない") in tgt_text or _nfc("効果がない") in tgt_text:
        tq.is_vanilla = True

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
        elif query.zone == Zone.TEMP: candidates.extend(p.temp_zone)
        elif query.zone == Zone.DECK: candidates.extend(p.deck)
        elif query.zone == Zone.COST_AREA:
            candidates.extend(p.don_active)
            candidates.extend(p.don_rested)

    dynamic_cost_max = None
    if query.cost_max_dynamic == "DON_COUNT_FIELD":
        p = owner_player 
        dynamic_cost_max = len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)

    results = []
    seen_names = set()
    for card in candidates:
        if not card: continue
        
        if not hasattr(card, "master"):
            if query.is_rest is not None and card.is_rest != query.is_rest: continue
            if query.card_type: continue
            if query.traits: continue
            if query.colors: continue
            if query.attributes: continue
            if query.names: continue
            if query.cost_min is not None or query.cost_max is not None: continue
            if query.power_min is not None or query.power_max is not None: continue
            results.append(card)
            continue
        
        if query.card_type and card.master.type.name not in query.card_type:
            continue

        if query.colors and not any(c in card.master.color.value for c in query.colors): continue
        if query.attributes and card.master.attribute.value not in query.attributes: continue
        
        if query.cost_max is not None and card.current_cost > query.cost_max: continue
        if query.cost_min is not None and card.current_cost < query.cost_min: continue
        
        if dynamic_cost_max is not None and card.current_cost > dynamic_cost_max: continue

        if query.power_max is not None and card.get_power(True) > query.power_max: continue
        if query.power_min is not None and card.get_power(True) < query.power_min: continue
        
        if query.is_vanilla:
            txt = card.master.effect_text
            if txt and txt.strip() not in ["", "なし", "-"]: continue

        if query.names:
            if "NAME_PARTIAL" in query.flags:
                if not any(n in card.master.name for n in query.names): continue
            else:
                if card.master.name not in query.names: continue

        if query.is_unique_name:
            if card.master.name in seen_names: continue
            seen_names.add(card.master.name)

        if query.traits and not any(t in card.master.traits for t in query.traits): continue
        if query.is_rest is not None and card.is_rest != query.is_rest: continue
        results.append(card)

    if not results:
        log_level = "WARNING"
        if query.select_mode in ["ALL", "REMAINING"] or query.is_up_to: log_level = "INFO"
        log_event(level_key=log_level, action="matcher.no_target", msg=f"No targets found for query: {query.raw_text}", player="system", payload={"query_raw": query.raw_text, "zone": query.zone.name, "target_player": query.player.name, "real_target_names": [p.name for p in target_players], "candidates_scanned": len(candidates)})

    return results
