import re
import logging
import unicodedata
from ...models.effect_types  import TargetQuery, _nfc
from ...models.enums import Player, Zone, ParserKeyword, Attribute, TriggerType
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

    # プレイヤー判定では「(お互い/相手/自分)のライフの枚数以下のコストを持つ」のような
    # コスト上限修飾句を除去する。この「お互いの/相手の」はコスト基準であって対象側ではなく
    # （実際の対象は後続の「相手のキャラ」等）、player を誤って ALL/OPPONENT にしてしまうため。
    player_text = re.sub(
        _nfc(r'(?:お互いの|相手の|自分の)?ライフの(?:合計)?枚数(?:分)?以下のコストを持つ'),
        '', tgt_text)
    # 同様に「(相手の/自分の)場のドン‼の枚数以下のコストを持つ」もコスト基準であって
    # 対象側ではない（「自分の手札から…相手の場のドン‼の枚数以下のコストを持つ『X』」で
    # player を OPPONENT に誤判定し相手手札を見てしまう: OP08-062 カタクリ）。
    player_text = re.sub(
        _nfc(r'(?:相手の|自分の|お互いの)?場のドン(?:!!|‼)?の枚数(?:分)?以下のコストを持つ'),
        '', player_text)
    # 期間/タイミング句の「相手の」は対象側ではないため除去する
    # （「自分のリーダーを、次の相手のターン終了時まで、パワー+2000」で OPPONENT 誤判定を防ぐ）。
    # 「相手のキャラ」等の実対象修飾は残すよう、ターン/エンドフェイズ＋まで/中 に限定する。
    player_text = re.sub(
        _nfc(r'(?:次の)?相手の(?:ターン|エンドフェイズ)(?:終了時)?(?:まで|中)'),
        '', player_text)
    # 選択者句「相手が選び/選ぶ/選んで」は対象側ではなく「誰が選ぶか」の指定
    # （「自分の手札1枚を相手が選び、捨てる」= 対象は自分の手札、選ぶのは相手）。
    # player 判定から除去し、chooser として保持する。
    chooser = None
    if re.search(_nfc(r'相手が選(?:び|ぶ|んで)'), player_text):
        chooser = Player.OPPONENT
        player_text = re.sub(_nfc(r'相手が選(?:び|ぶ|んで)'), '', player_text)
    # トリガー条件句「相手が…した時、」も対象側判定を汚すため除去する
    player_text = re.sub(_nfc(r'相手が[^、。]*した時、?'), '', player_text)

    if _nfc(ParserKeyword.EACH_OTHER) in player_text: tq.player = Player.ALL
    elif _nfc(ParserKeyword.OPPONENT) in player_text: tq.player = Player.OPPONENT
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
            
    elif _nfc(ParserKeyword.SELF) in player_text or _nfc(ParserKeyword.SELF_REF) in tgt_text: tq.player = Player.SELF

    zone_map = {
        _nfc("手札"): Zone.HAND,
        _nfc("トラッシュ"): Zone.TRASH,
        _nfc("ライフ"): Zone.LIFE,
        _nfc("デッキ"): Zone.DECK,
        _nfc("コストエリア"): Zone.COST_AREA,
        _nfc("場"): Zone.FIELD
    }
    
    found_zone = None

    # ゾーン検出も修飾句除去後のテキストで行う。「お互いのライフの合計枚数以下の
    # コストを持つ相手のキャラをKOする」で zone=LIFE と誤検出し、フィールドの
    # キャラではなくライフ札を動かしていた（雷迎/ロブ・ルッチ等のトリガー）。
    pattern = re.compile(r'(手札|トラッシュ|ライフ|デッキ|場|コストエリア)(?:.{0,5})(?:を|から|の)')
    matches = pattern.finditer(player_text)

    for m in matches:
        z_name = _nfc(m.group(1))
        post_match = player_text[m.end():]
        
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

    # 複数ゾーン「手札かトラッシュから」「場か手札の」: 「Zか(ら)Z」並列で
    # 候補ゾーンを併記する（EB03-049 手札かトラッシュ / OP13-079 場か手札）。
    # 「場」は「登場/場合」と紛れるため後置詞（の/から/に/か）を要求する。
    multi_zones = []
    if re.search(_nfc(r'(手札|トラッシュ|デッキ|ライフ|場)(?:から|の|に)?か[、,]?(?:[^。]{0,4})?(手札|トラッシュ|デッキ|ライフ|場)'), player_text):
        zone_markers = [
            (_nfc("手札"), Zone.HAND), (_nfc("トラッシュ"), Zone.TRASH),
            (_nfc("ライフ"), Zone.LIFE), (_nfc("デッキ"), Zone.DECK),
        ]
        for zk, zv in zone_markers:
            if zk in player_text and zv not in multi_zones:
                multi_zones.append(zv)
        # 「場」はゾーン後置詞付きのときだけ（登場/場合を除外）
        if re.search(_nfc(r'場(?:の|から|に|か)'), player_text) and Zone.FIELD not in multi_zones:
            multi_zones.append(Zone.FIELD)

    if len(multi_zones) >= 2:
        tq.zone = multi_zones
    elif found_zone:
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
        else:
            # 「「◯◯」以外のキャラ」: その名前を除外対象にする（従来は無視され、
            # 当該カード自身も対象に含めてしまっていた）。
            tq.exclude_names.append(m_name.group(1))
    
    if _nfc("含む") in tgt_text:
        tq.flags.add("NAME_PARTIAL")

    # 「（このキャラ）他の」「このキャラ以外」: ソース自身を候補から除外する。
    # 例: EB02-018「自分のキャラの他の『バギー』がいない場合」（自分自身を数えない）、
    # OP04-111「このキャラ以外の自分の特徴《ホーミーズ》を持つキャラ」（自身をコストに使わない）。
    if _nfc("他の") in tgt_text or _nfc("このキャラ以外") in tgt_text or _nfc("以外の自分") in tgt_text:
        tq.flags.add("EXCLUDE_SOURCE")

    # 「【トリガー】を持つ（キャラ/カード）」対象フィルタ: トリガー能力所持に限定（matcher が絞り込む）。
    # 全対象種別で効くよう parse_target に置く（従来は discard ルールのみで、PLAY_CARD 等に
    # 適用されず「【トリガー】を持つキャラを登場」の絞り込みが脱落していた: OP03-022）。
    if re.search(_nfc(r'【トリガー】を持つ'), tgt_text):
        tq.flags.add("HAS_TRIGGER")
        # 「《特徴》か【トリガー】を持つ（キャラ）」= 特徴 OR トリガー所持（OP05-002）。
        if re.search(_nfc(r'か【トリガー】を持つ'), tgt_text):
            tq.flags.add("TRAIT_OR_TRIGGER")

    # 特徴は《X》/<X> に加え 『X』（例: 『白ひげ海賊団』を含む特徴を持つ）でも表記される。
    # 名前は「X」を使うため 『』 と衝突しない（condition 側も 『X』 を特徴として扱う）。
    raw_traits = re.findall(r'[《<『]([^》>』]+)[》>』]', tgt_text)
    attr_values = [a.value for a in Attribute if a != Attribute.NONE]
    final_traits = []
    
    for t in raw_traits:
        if t in attr_values:
            tq.attributes.append(t)
        else:
            final_traits.append(t)
            
    tq.traits.extend(final_traits)

    # 「《特徴》（を持つキャラカード）か「名前」」= 特徴 OR 名前。「か」が名前の開き括弧へ
    # 直接かかる場合に OR とみなす（OP11-022「《海王類》を持つキャラカードか「メガロ」」）。
    if tq.traits and tq.names and re.search(_nfc(r'か[「『《]'), tgt_text):
        tq.flags.add("TRAIT_OR_NAME")

    attrs = re.findall(_nfc(ParserKeyword.ATTRIBUTE + r'[((]([^))]+)[))]'), tgt_text)
    tq.attributes.extend(attrs)
    
    for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
        if f"{c}の" in tgt_text: tq.colors.append(c)

    # コスト範囲「コストNからM」（N以上M以下）。範囲表記は単一しきい値より先に判定する
    #   （従来は「コスト3」だけを拾い cost_max=3 に縮退していた: OP10-099）。
    m_crange = re.search(_nfc(ParserKeyword.COST + r'(\d+)から(\d+)'), tgt_text)
    if m_crange:
        tq.cost_min = int(m_crange.group(1))
        tq.cost_max = int(m_crange.group(2))
    m_c = None if m_crange else re.search(_nfc(ParserKeyword.COST + r'[^+＋\-－−‐\d]?(\d+)(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_c:
        start_idx = m_c.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        
        end_idx = m_c.end()
        post_match = tgt_text[end_idx:]
        is_set_action = _nfc("にする") in post_match[:5]

        if prefix_context not in ['+', '-', '\u2212', '\u2010', '\uff0b', '\uff0d'] and not is_set_action:
            val = int(m_c.group(1))
            if m_c.group(2) == _nfc(ParserKeyword.ABOVE): tq.cost_min = val
            else: tq.cost_max = val

    # \u52d5\u7684\u30b3\u30b9\u30c8\u4e0a\u9650: \u300c\uff08\u81ea\u5206\u306e\uff09\u5834\u306e\u30c9\u30f3!!\u306e\u679a\u6570\uff08\u5206\uff09\u4ee5\u4e0b\u306e\u30b3\u30b9\u30c8\u300d\u2192 DON_COUNT_FIELD\u3002
    #   \u6570\u5024\u3067\u306f\u306a\u304f\u5834\u306e\u30c9\u30f3!!\u679a\u6570\u3067\u30b3\u30b9\u30c8\u4e0a\u9650\u304c\u6c7a\u307e\u308b\uff08\u865a\u306e\u7389\u5ea7 OP13-099 \u7b49\uff09\u3002
    #   \u30a8\u30f3\u30b8\u30f3 get_target_cards \u304c DON_COUNT_FIELD \u3092\u8a55\u4fa1\u3059\u308b\u3002
    m_don_cap = re.search(_nfc(r'(\u76f8\u624b\u306e|\u81ea\u5206\u306e)?(?:\u5834\u306e)?\u30c9\u30f3(?:!!|\u203c)?\u306e\u679a\u6570(?:\u5206)?\u4ee5\u4e0b\u306e\u30b3\u30b9\u30c8'), tgt_text)
    if m_don_cap:
        # \u300c\u76f8\u624b\u306e\u5834\u306e\u30c9\u30f3\u203c\u306e\u679a\u6570\u4ee5\u4e0b\u306e\u30b3\u30b9\u30c8\u300d\u306f\u76f8\u624b\u306e\u30c9\u30f3\u679a\u6570\uff08OP08-062\uff09\u3001
        # \u65e2\u5b9a\uff08\u660e\u793a\u306a\u3057/\u81ea\u5206\u306e\uff09\u306f\u81ea\u5206\u306e\u30c9\u30f3\u679a\u6570\uff08\u865a\u306e\u7389\u5ea7 OP13-099 \u7b49\uff09\u3002
        if m_don_cap.group(1) == _nfc("\u76f8\u624b\u306e"):
            tq.cost_max_dynamic = "DON_COUNT_FIELD_OPPONENT"
        else:
            tq.cost_max_dynamic = "DON_COUNT_FIELD"

    # \u52d5\u7684\u30b3\u30b9\u30c8\u4e0a\u9650: \u300c(\u76f8\u624b\u306e/\u81ea\u5206\u306e/\u304a\u4e92\u3044\u306e) \u30e9\u30a4\u30d5\u306e(\u5408\u8a08)?\u679a\u6570(\u5206)?\u4ee5\u4e0b\u306e\u30b3\u30b9\u30c8\u300d\u3002
    #   \u30e9\u30a4\u30d5\u679a\u6570\u3067\u30b3\u30b9\u30c8\u4e0a\u9650\u304c\u6c7a\u307e\u308b\uff08OP04-112 \u30e4\u30de\u30c8 / OP05-102 \u30b2\u30c0\u30c4 \u7b49\uff09\u3002
    #   \u300c\u304a\u4e92\u3044\u306e\u2026\u5408\u8a08\u300d\u306f\u4e21\u8005\u30e9\u30a4\u30d5\u5408\u8a08\u3001\u305d\u308c\u4ee5\u5916\u306f\u6240\u6709\u8005\u57fa\u6e96\u3067 \u76f8\u624b/\u81ea\u5206 \u3092\u5224\u5b9a\u3002
    #   \u30a8\u30f3\u30b8\u30f3 get_target_cards \u304c LIFE_COUNT_* \u3092\u8a55\u4fa1\u3059\u308b\u3002
    if re.search(_nfc(r'\u30e9\u30a4\u30d5\u306e(?:\u5408\u8a08)?\u679a\u6570(?:\u5206)?\u4ee5\u4e0b\u306e\u30b3\u30b9\u30c8'), tgt_text):
        m_life = re.search(_nfc(r'(\u304a\u4e92\u3044\u306e|\u76f8\u624b\u306e|\u81ea\u5206\u306e)?\u30e9\u30a4\u30d5\u306e(?:\u5408\u8a08)?\u679a\u6570'), tgt_text)
        prefix = m_life.group(1) if m_life and m_life.group(1) else ""
        if prefix == _nfc("\u304a\u4e92\u3044\u306e"):
            tq.cost_max_dynamic = "LIFE_COUNT_BOTH"
        elif prefix == _nfc("\u81ea\u5206\u306e"):
            tq.cost_max_dynamic = "LIFE_COUNT_SELF"
        else:
            # \u65e2\u5b9a\u306f\u76f8\u624b\u306e\u30e9\u30a4\u30d5\uff08\u300c\u76f8\u624b\u306e\u300d\u660e\u793a\uff0f\u7701\u7565\u6642\u3068\u3082\u76f8\u624b\u57fa\u6e96\u304c\u5927\u534a\uff09
            tq.cost_max_dynamic = "LIFE_COUNT_OPPONENT"

    m_p = re.search(_nfc(ParserKeyword.POWER + r'[^+\uff0b\-\uff0d\u2212\u2010\d]?(\d+)(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_p:
        start_idx = m_p.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        if prefix_context not in ['+', '-', '\u2212', '\u2010', '\uff0b', '\uff0d']:
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

    # 「任意の枚数」: プレイヤーが 0..N 枚を任意に選べる可変選択。is_up_to=True かつ
    # 大きめの count（フィールド/手札の実上限を超える）で対象選択中断（_suspend_for_target_selection,
    # min=0/max=count）に乗せる。select_mode は CHOOSE のまま（ALL=自動全選択にしない）。
    if _nfc("任意の枚数") in tgt_text:
        tq.is_up_to = True
        tq.count = 50
    
    if _nfc("効果のない") in tgt_text or _nfc("効果がない") in tgt_text:
        tq.is_vanilla = True

    # 「選んだ／その（カード/キャラ/リーダー）」は直前の選択（SELECT, save_id="selected_card"）で
    # 保存した対象を参照する。resolver._resolve_targets は ref_id が saved_targets に
    # 無ければ通常マッチへフォールバックするため、選択が先行しない場合も安全。
    if re.search(_nfc(r"(選んだ|その)(カード|キャラ|リーダー)"), tgt_text):
        tq.ref_id = "selected_card"

    if chooser is not None:
        tq.chooser = chooser

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

    # zone はリスト（「手札かトラッシュから」EB03-049 / 「場か手札」OP13-079）も取り得る。
    zones = query.zone if isinstance(query.zone, list) else [query.zone]

    candidates = []
    for p in target_players:
        if not p: continue
        for z in zones:
            if z == Zone.FIELD:
                candidates.extend(p.field)
                if not query.card_type or "LEADER" in query.card_type:
                    if p.leader: candidates.append(p.leader)
                if p.stage: candidates.append(p.stage)
            elif z == Zone.HAND: candidates.extend(p.hand)
            elif z == Zone.TRASH: candidates.extend(p.trash)
            elif z == Zone.LIFE: candidates.extend(p.life)
            elif z == Zone.TEMP: candidates.extend(p.temp_zone)
            elif z == Zone.DECK: candidates.extend(p.deck)
            elif z == Zone.COST_AREA:
                candidates.extend(p.don_active)
                candidates.extend(p.don_rested)

    dynamic_cost_max = None
    if query.cost_max_dynamic == "DON_COUNT_FIELD":
        p = owner_player
        dynamic_cost_max = len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)
    elif query.cost_max_dynamic == "DON_COUNT_FIELD_OPPONENT":
        p = opponent_player
        dynamic_cost_max = len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)
    elif query.cost_max_dynamic == "LIFE_COUNT_OPPONENT":
        dynamic_cost_max = len(opponent_player.life)
    elif query.cost_max_dynamic == "LIFE_COUNT_SELF":
        dynamic_cost_max = len(owner_player.life)
    elif query.cost_max_dynamic == "LIFE_COUNT_BOTH":
        dynamic_cost_max = len(owner_player.life) + len(opponent_player.life)

    exclude_source = "EXCLUDE_SOURCE" in query.flags

    results = []
    seen_names = set()
    for card in candidates:
        if not card: continue

        if exclude_source and source_card is not None and card is source_card: continue

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

        # 【修正】card.master.colors (List[Color]) の各値を確認するように変更
        if query.colors:
            card_colors = [c.value for c in card.master.colors] if card.master.colors else []
            if not any(qc in card_colors for qc in query.colors): continue

        if query.attributes and card.master.attribute.value not in query.attributes: continue
        
        if query.cost_max is not None and card.current_cost > query.cost_max: continue
        if query.cost_min is not None and card.current_cost < query.cost_min: continue
        
        if dynamic_cost_max is not None and card.current_cost > dynamic_cost_max: continue

        if query.power_max is not None and card.get_power(True) > query.power_max: continue
        if query.power_min is not None and card.get_power(True) < query.power_min: continue
        
        if query.is_vanilla:
            txt = card.master.effect_text
            if txt and txt.strip() not in ["", "なし", "-"]: continue

        # 「《特徴》か「名前」」= 特徴 OR 名前（両者の AND ではない）。OP11-022「《海王類》かメガロ」が
        # trait∧name の AND になり対象が常に空になっていた。フラグ時は OR で照合する。
        if "TRAIT_OR_NAME" in query.flags and (query.names or query.traits):
            if "NAME_PARTIAL" in query.flags:
                name_ok = bool(query.names) and any(n in card.master.name for n in query.names)
            else:
                name_ok = bool(query.names) and card.master.name in query.names
            trait_ok = bool(query.traits) and any(t in card.master.traits for t in query.traits)
            if not (name_ok or trait_ok): continue
            if query.exclude_names and card.master.name in query.exclude_names: continue
        else:
            if query.names:
                if "NAME_PARTIAL" in query.flags:
                    if not any(n in card.master.name for n in query.names): continue
                else:
                    if card.master.name not in query.names: continue

            if query.exclude_names and card.master.name in query.exclude_names: continue

            if query.traits and not any(t in card.master.traits for t in query.traits):
                # 「《特徴》か【トリガー】を持つ」は特徴 OR トリガー所持。特徴不一致でも
                # トリガー所持なら通す（OP05-002）。それ以外は従来どおり除外。
                if "TRAIT_OR_TRIGGER" not in query.flags:
                    continue
                _trig = bool(getattr(card.master, "trigger_text", "")) or any(
                    ab.trigger == TriggerType.TRIGGER for ab in getattr(card.master, "abilities", ()))
                if not _trig:
                    continue
        if query.is_rest is not None and card.is_rest != query.is_rest: continue

        # 「【トリガー】を持つカード」フィルタ: トリガー能力（master.trigger_text 非空、または
        # TriggerType.TRIGGER 能力）を持つカードのみに限定する（OP16-080 等）。
        # TRAIT_OR_TRIGGER（特徴 OR トリガー）の場合は上の特徴フィルタで OR 判定済みのため除外。
        if "HAS_TRIGGER" in query.flags and "TRAIT_OR_TRIGGER" not in query.flags:
            has_trig = bool(getattr(card.master, "trigger_text", "")) or any(
                ab.trigger == TriggerType.TRIGGER for ab in getattr(card.master, "abilities", ()))
            if not has_trig:
                continue
        
        # 名前重複排除は全てのフィルタを通過した後に実施
        if query.is_unique_name:
            if card.master.name in seen_names: continue
            seen_names.add(card.master.name)
            
        results.append(card)

    if not results:
        log_level = "WARNING"
        if query.select_mode in ["ALL", "REMAINING"] or query.is_up_to: log_level = "INFO"
        zone_name = ",".join(z.name for z in zones)
        log_event(level_key=log_level, action="matcher.no_target", msg=f"No targets found for query: {query.raw_text}", player="system", payload={"query_raw": query.raw_text, "zone": zone_name, "target_player": query.player.name, "real_target_names": [p.name for p in target_players], "candidates_scanned": len(candidates)})

    return results
