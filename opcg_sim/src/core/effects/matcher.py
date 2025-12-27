import re  # 追加が必要です
from ..effects import TargetQuery, _nfc
from ...models.enums import Player, Zone

def parse_target(tgt_text: str, default_player: Player = Player.SELF) -> TargetQuery:
    # インデントを左に寄せて、単独の関数として定義します
    tq = TargetQuery(raw_text=tgt_text, player=default_player)
    
    # [追加] "このキャラ" 等の自己参照キーワードの場合、モードをSOURCEに設定して即返す
    if tgt_text == _nfc("このキャラ") or (tgt_text == _nfc("自身") and _nfc("自身の") not in tgt_text):
        tq.select_mode = "SOURCE"
        return tq

    # --- 追加: 残りカード判定 ---
    if _nfc("残り") in tgt_text:
        tq.select_mode = "REMAINING"
        tq.count = -1
        tq.zone = Zone.TEMP
        return tq

    # --- 視点ベース(カメラ切り替え)によるプレイヤー判定ロジック ---
    if _nfc("お互い") in tgt_text: 
        tq.player = Player.ALL
    elif _nfc("持ち主") in tgt_text: 
        tq.player = Player.OWNER
    elif _nfc("相手") in tgt_text:
        if default_player == Player.OPPONENT:
            tq.player = Player.SELF
        else:
            tq.player = Player.OPPONENT
    elif _nfc("自分") in tgt_text or _nfc("自身") in tgt_text:
        tq.player = Player.SELF

    # ゾーン判定
    if _nfc("手札") in tgt_text: tq.zone = Zone.HAND
    elif _nfc("トラッシュ") in tgt_text: tq.zone = Zone.TRASH
    elif _nfc("ライフ") in tgt_text: tq.zone = Zone.LIFE
    elif _nfc("デッキ") in tgt_text: tq.zone = Zone.DECK
    elif _nfc("ドン") in tgt_text: tq.zone = Zone.COST_AREA 
    else: tq.zone = Zone.FIELD

    # カードタイプ判定
    if _nfc("リーダー") in tgt_text: tq.card_type.append("LEADER")
    if _nfc("キャラ") in tgt_text: tq.card_type.append("CHARACTER")
    if _nfc("イベント") in tgt_text: tq.card_type.append("EVENT")
    if _nfc("ステージ") in tgt_text: tq.card_type.append("STAGE")
    
    # 名称判定
    m_name = re.search(r'「([^」]+)」', tgt_text)
    if m_name:
        name_val = m_name.group(1)
        full_match = m_name.group(0)
        exclusion_marker = _nfc("以外の")
        if (full_match + exclusion_marker) not in tgt_text:
            tq.names.append(name_val)
    
    # 特徴・属性・色
    traits = re.findall(_nfc(r'特徴[《<]([^》>]+)[》>]'), tgt_text)
    tq.traits.extend(traits)
    attrs = re.findall(_nfc(r'属性[((]([^))]+)[))]'), tgt_text)
    tq.attributes.extend(attrs)
    for c in [_nfc("赤"), _nfc("緑"), _nfc("青"), _nfc("紫"), _nfc("黒"), _nfc("黄")]:
        if f"{c}の" in tgt_text: tq.colors.append(c)

    # コスト・パワー
    m_c = re.search(_nfc(r'コスト\D?(\d+)\D?(以下|以上)?'), tgt_text)
    if m_c:
        val = int(m_c.group(1))
        if m_c.group(2) == _nfc("以上"): tq.cost_min = val
        else: tq.cost_max = val

    m_p = re.search(_nfc(r'パワー\D?(\d+)\D?(以下|以上)?'), tgt_text)
    if m_p:
        val = int(m_p.group(1))
        if m_p.group(2) == _nfc("以上"): tq.power_min = val
        else: tq.power_max = val
    
    # 状態（レスト/アクティブ）
    if _nfc("レスト") in tgt_text: tq.is_rest = True
    elif _nfc("アクティブ") in tgt_text: tq.is_rest = False
    
    # 枚数判定
    if _nfc("すべて") in tgt_text or _nfc("全て") in tgt_text:
        tq.count = -1
        tq.select_mode = "ALL"
    else:
        m_cnt = re.search(r'(\d+)枚', tgt_text)
        tq.count = int(m_cnt.group(1)) if m_cnt else 1
    
    return tq
