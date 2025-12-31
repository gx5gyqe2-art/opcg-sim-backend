import re
import logging
from ...models.effect_types  import TargetQuery, _nfc
from ...models.enums import Player, Zone, ParserKeyword

logger = logging.getLogger("opcg_sim")

def parse_target(tgt_text: str, default_player: Player = Player.SELF) -> TargetQuery:
    tq = TargetQuery(raw_text=tgt_text, player=default_player)
    
    # 定数化
    if tgt_text == _nfc(ParserKeyword.THIS_CARD) or (tgt_text == _nfc(ParserKeyword.SELF_REF) and _nfc(ParserKeyword.SELF_REF + "の") not in tgt_text):
        tq.select_mode = "SOURCE"
        return tq

    if _nfc(ParserKeyword.REMAINING) in tgt_text:
        tq.select_mode = "REMAINING"
        tq.count = -1
        tq.zone = Zone.TEMP
        return tq

    if _nfc(ParserKeyword.EACH_OTHER) in tgt_text: 
        tq.player = Player.ALL
    elif _nfc(ParserKeyword.OWNER) in tgt_text: 
        tq.player = Player.OWNER
    elif _nfc(ParserKeyword.OPPONENT) in tgt_text:
        if default_player == Player.OPPONENT:
            tq.player = Player.SELF
        else:
            tq.player = Player.OPPONENT
    elif _nfc(ParserKeyword.SELF) in tgt_text or _nfc(ParserKeyword.SELF_REF) in tgt_text:
        tq.player = Player.SELF

    # ゾーン判定
    if _nfc(ParserKeyword.HAND) in tgt_text: tq.zone = Zone.HAND
    elif _nfc(ParserKeyword.TRASH) in tgt_text: tq.zone = Zone.TRASH
    elif _nfc(ParserKeyword.LIFE) in tgt_text: tq.zone = Zone.LIFE
    elif _nfc(ParserKeyword.DECK) in tgt_text: tq.zone = Zone.DECK
    elif _nfc(ParserKeyword.DON) in tgt_text: tq.zone = Zone.COST_AREA 
    else: tq.zone = Zone.FIELD

    # カードタイプ判定
    if _nfc(ParserKeyword.LEADER) in tgt_text: tq.card_type.append("LEADER")
    if _nfc(ParserKeyword.CHARACTER) in tgt_text: tq.card_type.append("CHARACTER")
    if _nfc(ParserKeyword.EVENT) in tgt_text: tq.card_type.append("EVENT")
    if _nfc(ParserKeyword.STAGE) in tgt_text: tq.card_type.append("STAGE")
    
    # 名称指定
    m_name = re.search(r'「([^」]+)」', tgt_text)
    if m_name:
        name_val = m_name.group(1)
        full_match = m_name.group(0)
        exclusion_marker = _nfc(ParserKeyword.EXCEPT)
        if (full_match + exclusion_marker) not in tgt_text:
            tq.names.append(name_val)
    
    # 特徴・属性
    traits = re.findall(_nfc(ParserKeyword.TRAIT + r'[《<]([^》>]+)[》>]'), tgt_text)
    tq.traits.extend(traits)
    attrs = re.findall(_nfc(ParserKeyword.ATTRIBUTE + r'[((]([^))]+)[))]'), tgt_text)
    tq.attributes.extend(attrs)
    
    # 色（色はEnum化も可能だが、現状はリストで維持）
    for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
        if f"{c}の" in tgt_text: tq.colors.append(c)

    # コスト
    m_c = re.search(_nfc(ParserKeyword.COST + r'\D?(\d+)\D?(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_c:
        val = int(m_c.group(1))
        if m_c.group(2) == _nfc(ParserKeyword.ABOVE): tq.cost_min = val
        else: tq.cost_max = val

    # パワー
    m_p = re.search(_nfc(ParserKeyword.POWER + r'\D?(\d+)\D?(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_p:
        val = int(m_p.group(1))
        if m_p.group(2) == _nfc(ParserKeyword.ABOVE): tq.power_min = val
        else: tq.power_max = val
    
    # 状態
    if _nfc(ParserKeyword.REST) in tgt_text: tq.is_rest = True # "レストにする" の "レスト" 部分マッチに依存
    elif _nfc("レスト") in tgt_text: tq.is_rest = True # キーワードが "レストにする" なので単体 "レスト" も補足
    elif _nfc("アクティブ") in tgt_text: tq.is_rest = False
    
    # 枚数
    if _nfc(ParserKeyword.ALL_HIRAGANA) in tgt_text or _nfc(ParserKeyword.ALL) in tgt_text:
        tq.count = -1
        tq.select_mode = "ALL"
    else:
        m_cnt = re.search(r'(\d+)' + _nfc(ParserKeyword.COUNT_SUFFIX), tgt_text)
        tq.count = int(m_cnt.group(1)) if m_cnt else 1
    
    return tq

def get_target_cards(game_manager, query: TargetQuery, source_card) -> list:
    # ... (既存ロジック変更なし) ...
    # 1. 自己参照モード
    if query.select_mode == "SOURCE":
        return [source_card]

    # 2. 対象プレイヤーの決定
    target_players = []
    if query.player == Player.SELF:
        target_players = [game_manager.turn_player]
    elif query.player == Player.OPPONENT:
        target_players = [game_manager.opponent]
    elif query.player == Player.ALL:
        target_players = [game_manager.p1, game_manager.p2]
    elif query.player == Player.OWNER:
        owner, _ = game_manager._find_card_location(source_card)
        target_players = [owner] if owner else []

    candidates = []
    # 3. 指定ゾーンからの抽出
    for p in target_players:
        if query.zone == Zone.FIELD:
            candidates.extend(p.field)
            if not query.card_type or "LEADER" in query.card_type:
                if p.leader: candidates.append(p.leader)
            if p.stage: candidates.append(p.stage)
        elif query.zone == Zone.HAND:
            candidates.extend(p.hand)
        elif query.zone == Zone.TRASH:
            candidates.extend(p.trash)
        elif query.zone == Zone.LIFE:
            candidates.extend(p.life)
        elif query.zone == Zone.TEMP:
            candidates.extend(p.temp_zone)

    # 4. フィルタリング
    results = []
    for card in candidates:
        if not card: continue
        if query.colors and not any(c in card.master.color.value for c in query.colors): continue
        if query.attributes and card.master.attribute.value not in query.attributes: continue
        if query.cost_max is not None and card.current_cost > query.cost_max: continue
        if query.cost_min is not None and card.current_cost < query.cost_min: continue
        if query.power_max is not None and card.get_power(True) > query.power_max: continue
        if query.power_min is not None and card.get_power(True) < query.power_min: continue
        if query.names and card.master.name not in query.names: continue
        if query.traits and not any(t in card.master.traits for t in query.traits): continue
        if query.is_rest is not None and card.is_rest != query.is_rest: continue
        
        results.append(card)

    # 5. 枚数制限の適用
    if query.count == -1 or query.select_mode in ["ALL", "REMAINING"]:
        return results
    return results[:query.count]
