"""プレイヤーレベル・アクションのハンドラ（対象ループ前に完結する 22 種）。

いずれも旧 `apply_action_to_engine` 冒頭の `if act_name == "X": ...; return` 分岐を逐語移設したもの。
`self` を第1引数 `gm`（GameManager）へ読み替えただけで、挙動・戻り値は不変。
"""
import random
from ...models.enums import ActionType, Zone, TriggerType
from ..rules_constants import SELF_RESTRICTION_KEYS
from .registry import game_handler


@game_handler(ActionType.RULE_PROCESSING,
              when=lambda a: getattr(a, "status", None) in SELF_RESTRICTION_KEYS)
def rule_processing_self_restriction(gm, player, action, targets, value, source_card):
    # 自己制限（「自分は、このターン中、…できない」= self_cannot）の登録。
    # parser が RULE_PROCESSING + status=制限キーで生成する。対象を持たないため、
    # 通常の `for target in targets` ループ前にここで処理して player に記録する。
    rec = {"expire": gm.turn_count}  # 「このターン中」: 現ターン内のみ有効
    if action.value and getattr(action.value, "base", None):
        rec["min_cost"] = action.value.base
    player.restrictions[action.status] = rec
    return True


@game_handler(ActionType.DRAW)
def draw(gm, player, action, targets, value, source_card):
    target_player = player
    if action.target and getattr(action.target, 'player', None) is not None:
        if getattr(action.target.player, 'name', '') == 'OPPONENT':
            target_player = gm.p2 if player == gm.p1 else gm.p1
    # 「自分の効果でカードを引くことができない」: 効果解決による DRAW を抑止する。
    if gm._active_restriction(target_player, "CANNOT_DRAW_BY_EFFECT"):
        return True
    gm.draw_card(target_player, value)
    return True


@game_handler(ActionType.DEAL_DAMAGE)  # DAMAGE は DEAL_DAMAGE のエイリアス（同一 enum メンバー）
def deal_damage(gm, player, action, targets, value, source_card):
    # 「相手に N ダメージを与える」: 相手リーダーへ N ダメージ。ライフ上から N 枚を
    # 手札へ移し（【トリガー】発動・ON_LIFE_DECREASE 発火）、ライフが尽きれば勝利。
    # 従来 DEAL_DAMAGE は未実装で no-op だった（ニコ・ロビン等のダメージ効果が不発）。
    damaged = gm.p2 if player == gm.p1 else gm.p1
    if action.target and getattr(getattr(action.target, 'player', None), 'name', '') == 'SELF':
        damaged = player
    n = value if value and value > 0 else 1
    life_lost = 0
    for _ in range(n):
        if damaged.life:
            life_card = damaged.life.pop(0)
            trig = next((a for a in life_card.master.abilities if a.trigger == TriggerType.TRIGGER), None)
            gm.move_card(life_card, Zone.HAND, damaged)
            life_lost += 1
            # 【トリガー】は任意。確認付きで待ち行列へ積む（即時解決しない）。
            if trig:
                gm._enqueue_trigger(damaged, trig, life_card, optional=True)
        else:
            gm.winner = player.name
            break
    # ON_LIFE_DECREASE を積み、【トリガー】と共にこの場で消化する。
    if life_lost and not gm.winner:
        gm._enqueue_life_decrease(damaged, life_lost)
    gm._advance_pending_triggers()
    return True


@game_handler(ActionType.SHUFFLE)
def shuffle(gm, player, action, targets, value, source_card):
    target_player = player
    if action.target and getattr(action.target, 'player', None) is not None:
        if getattr(action.target.player, 'name', '') == 'OPPONENT':
            target_player = gm.p2 if player == gm.p1 else gm.p1
    random.shuffle(target_player.deck)
    return True


@game_handler(ActionType.LOOK)
def look(gm, player, action, targets, value, source_card):
    if getattr(action, "status", None) == "OPPONENT":
        # 「相手のデッキの上から N 枚を見る」: 公開のみで盤面は不変（並びも変えない）。
        # 後続消費が無いため temp_zone には載せない（TEMP リーク防止）。
        opp = gm.p2 if player == gm.p1 else gm.p1
        count = min(value if value else 1, len(opp.deck))
        return True
    count = value
    deck = player.deck
    if len(deck) < count: count = len(deck)
    for _ in range(count):
        card = deck.pop(0)
        player.temp_zone.append(card)
    return True


@game_handler(ActionType.LOOK_LIFE)
def look_life(gm, player, action, targets, value, source_card):
    # 「（自分か相手の）ライフの上から N枚を見る」→ 対象プレイヤーのライフ上 value 枚を
    # 同プレイヤーの temp_zone へ移して公開する。後続の Choice が temp→ライフ上/下に戻す。
    # status=="OPPONENT" で相手のライフを対象（相手の temp_zone に載るため、戻し先も相手）。
    target_player = player
    if getattr(action, "status", None) == "OPPONENT":
        target_player = gm.p2 if player == gm.p1 else gm.p1
    count = value if value else 1
    moved = 0
    for _ in range(count):
        if not target_player.life:
            break
        card_ = target_player.life.pop(0)
        # 不発時の回収先を記録する（temp 回収はデッキトップではなくライフ上へ戻す）
        card_._temp_origin = "LIFE"
        target_player.temp_zone.append(card_)
        moved += 1
    return True


@game_handler(ActionType.MOVE_ATTACHED_DON)
def move_attached_don(gm, player, action, targets, value, source_card):
    # 「付与されているドン‼N枚をコストエリアにレストで戻す」: 付与中のドンを N 枚外し、
    # レスト状態で don_rested（コストエリア）へ。付与先キャラの attached_don も減算する。
    n = value if value and value > 0 else 1
    moved = 0
    for don in list(player.don_attached_cards):
        if moved >= n:
            break
        tgt_uuid = getattr(don, "attached_to", None)
        player.don_attached_cards.remove(don)
        don.attached_to = None
        don.is_rest = True
        player.don_rested.append(don)
        if tgt_uuid:
            tgt = next((c for c in ([player.leader] + player.field) if c and c.uuid == tgt_uuid), None)
            if tgt is not None and getattr(tgt, "attached_don", 0) > 0:
                tgt.attached_don -= 1
        moved += 1
    # コストとして使われるため、要求枚数を戻せたかを成否で返す（付与ドン不足なら不成立）。
    return moved >= n


@game_handler(ActionType.REDIRECT_ATTACK)
def redirect_attack(gm, player, action, targets, value, source_card):
    # 「（選んだキャラ/このリーダー等）にアタックの対象を変更する」: 進行中バトルの
    # 対象を差し替える。targets[0] が新しい対象（多くはコントローラー側のキャラ/リーダー）。
    if gm.active_battle and targets:
        new_target = targets[0]
        gm.active_battle["target"] = new_target
        gm.active_battle["target_owner"] = gm.p1 if gm.p1.name == new_target.owner_id else gm.p2
    return True


@game_handler(ActionType.DISABLE_ABILITY,
              when=lambda a: getattr(a, "status", None) == "OPP_ONPLAY")
def disable_opp_onplay(gm, player, action, targets, value, source_card):
    # 「（次の相手のターン終了時まで、）相手の登場時効果は無効になる」: 相手プレイヤーに
    # ON_PLAY 無効化の期限(turn_count)を設定する。次の相手ターン(=turn_count+1)を覆う。
    opp = gm.p2 if player == gm.p1 else gm.p1
    dur = getattr(action, "duration", "INSTANT")
    opp.negate_onplay_until = gm.turn_count + (1 if dur == "UNTIL_NEXT_TURN_END" else 0)
    return True


@game_handler(ActionType.EXTRA_TURN)
def extra_turn(gm, player, action, targets, value, source_card):
    # 「このターンの後に自分のターンを追加で得る」: switch_turn が消費する
    gm.pending_extra_turn = player.name
    return True


@game_handler(ActionType.VICTORY)
def victory(gm, player, action, targets, value, source_card):
    # 「（自分は）ゲームに勝利する」: 能動勝利。即座に winner を設定する。
    # status="REPLACE_DECKOUT_LOSS" はデッキアウト敗北の置換マーカー(PASSIVE)で、
    # 直接実行されない（_has_deckout_win_replace で走査）。万一実行された場合は無視。
    if getattr(action, "status", None) == "REPLACE_DECKOUT_LOSS":
        return True
    gm.winner = player.name
    return True


@game_handler(ActionType.ORDER_LIFE)
def order_life(gm, player, action, targets, value, source_card):
    # 「（自分/相手の）ライフすべてを見て、好きな順番で置く」: ライフを任意順に並べ替える。
    # ライフ2枚以上のときは resolver が ARRANGE_DECK 対話(dest_kind=LIFE)で先に中断し、
    # プレイヤーが順序を選ぶ。ここに来るのはライフ1枚以下（並べ替え不要）の場合で、
    # 並びを保持する（枚数不変・カード消失なし・TEMP 非汚染）。
    target_player = player
    if getattr(action, "status", None) == "OPPONENT":
        target_player = gm.p2 if player == gm.p1 else gm.p1
    return True


@game_handler(ActionType.EXECUTE_EVENT)
def execute_event(gm, player, action, targets, value, source_card):
    # 「自分の手札から（条件）イベント1枚までを、発動する」: 手札のイベントの効果を
    # 解決し、発動後にトラッシュへ送る。効果解決は DEAL_DAMAGE のライフトリガー解決
    # （上記）と同じく resolve_ability の再入で行う（新規実行コンテキストを生成する
    # ため既存スタックを汚さない）。targets は matcher が手札のイベントを解決済み。
    _main_trigs = (TriggerType.ACTIVATE_MAIN, TriggerType.COUNTER, TriggerType.ON_PLAY)
    for ev in targets:
        gm._record_event_played(ev)   # 「このターン中…イベントを発動」条件用（OP15-002）
        ev_ability = next((a for a in ev.master.abilities
                           if a.effect is not None and a.trigger in _main_trigs), None)
        if ev_ability is None:
            ev_ability = next((a for a in ev.master.abilities if a.effect is not None), None)
        if ev_ability is not None:
            gm.resolve_ability(player, ev_ability, source_card=ev)
        gm.move_card(ev, Zone.TRASH, player)
    return True


@game_handler(ActionType.SELECT)
def select(gm, player, action, targets, value, source_card):
    return True


@game_handler(ActionType.HEAL, ActionType.LIFE_RECOVER)
def heal(gm, player, action, targets, value, source_card):
    for _ in range(value):
        if player.deck:
            player.life.append(player.deck.pop(0))
    return True


@game_handler(ActionType.TRASH_FROM_DECK)
def trash_from_deck(gm, player, action, targets, value, source_card):
    # 「（自分／相手の）デッキの上からN枚をトラッシュに置く」（mill）。
    # デッキは並びが意味を持つため対象選択させず、上から value 枚を送る。
    target_player = player
    if getattr(action, "status", None) == "OPPONENT":
        target_player = gm.p2 if player == gm.p1 else gm.p1
    milled = 0
    for _ in range(value):
        if not target_player.deck:
            break
        target_player.trash.append(target_player.deck.pop(0))
        milled += 1
    return True


@game_handler(ActionType.SWAP_POWER)
def swap_power(gm, player, action, targets, value, source_card):
    # 「選んだキャラそれぞれの元々のパワーを、このターン中、入れ替える」（OP14-001）。
    # 2体の元々パワー(master.power)を相互に base_power_override へ上書きする
    # （絶対値の base 上書きで reset_turn_status により失効＝このターン中）。
    valid = [t for t in targets if t is not None]
    if len(valid) >= 2:
        a, b = valid[0], valid[1]
        pa = a.master.power or 0
        pb = b.master.power or 0
        a.base_power_override = pb
        b.base_power_override = pa
    return True


@game_handler(ActionType.RAMP_DON)
def ramp_don(gm, player, action, targets, value, source_card):
    # status=="RESTED" の場合はレスト状態でコストエリアへ（「レストで追加」）。
    add_rested = getattr(action, "status", None) == "RESTED"
    for _ in range(value):
        if player.don_deck:
            don = player.don_deck.pop(0)
            don.is_rest = add_rested
            if add_rested:
                player.don_rested.append(don)
            else:
                player.don_active.append(don)
    return True


@game_handler(ActionType.RETURN_DON)
def return_don(gm, player, action, targets, value, source_card):
    # 「ドン‼-N」/「ドン!!デッキに戻す」: 場のドン!!を N 枚ドン!!デッキへ戻す。
    # resolver が対象ドン!!を選ばせた場合は _return_don_selection の uuid を戻す。
    # 選択が無い（直接呼び出し/ヘッドレス）場合は影響の小さい順（レスト→アクティブ
    # →付与中）に自動で戻す。
    tp = gm._don_pool_player(player, action)
    selection = getattr(gm, "_return_don_selection", None)
    gm._return_don_selection = None
    returned = 0
    if selection:
        by_uuid = {d.uuid: d for d in
                   (tp.don_active + tp.don_rested + tp.don_attached_cards)}
        for uid in selection:
            don = by_uuid.get(uid)
            if don is not None and gm._return_one_don(tp, don):
                returned += 1
    else:
        for _ in range(value):
            if tp.don_rested:
                don = tp.don_rested[-1]
            elif tp.don_active:
                don = tp.don_active[-1]
            elif tp.don_attached_cards:
                don = tp.don_attached_cards[-1]
            else:
                break
            if gm._return_one_don(tp, don):
                returned += 1
    if returned > 0:
        gm.record_turn_event("DON_RETURNED", returned)
    return True


@game_handler(ActionType.REST_DON)
def rest_don(gm, player, action, targets, value, source_card):
    # 「ドン!!N枚をレストにする」/【ドン!!×N】コスト: アクティブ→レスト。
    # ドンは均質なため枚数(value)ベースで処理する。
    tp = gm._don_pool_player(player, action)
    rested = 0
    for _ in range(value):
        if not tp.don_active:
            break
        don = tp.don_active.pop(0)
        don.is_rest = True
        tp.don_rested.append(don)
        rested += 1
    # 「レストにしたドン!!1枚につき…」(§7-5) 用に実レスト枚数を記録する。ドンは targets を
    # 介さず枚数処理するため、resolver の len(targets) では 0 になる（OP13-001）。
    gm._last_resource_count = rested
    return True


@game_handler(ActionType.FREEZE_DON)
def freeze_don(gm, player, action, targets, value, source_card):
    # 「（相手の）ドン!!N枚までは、次の相手のリフレッシュフェイズでアクティブにならない」
    # (OP07-026 ドン側)。レストのドン!!を value 枚まで is_frozen にする。refresh_all が
    # フリーズ中のドン!!を1回スキップする（キャラの flags["FREEZE"] と同じ1回限り）。
    tp = gm._don_pool_player(player, action)
    frozen = 0
    for don in tp.don_rested:
        if frozen >= value:
            break
        if not don.is_frozen:
            don.is_frozen = True
            frozen += 1
    gm._last_resource_count = frozen
    return True


@game_handler(ActionType.ACTIVE_DON, when=lambda a: not getattr(a, 'target', None))
def active_don_by_count(gm, player, action, targets, value, source_card):
    # 「ドン!!N枚をアクティブにする」: レスト→アクティブ（枚数ベース）。
    tp = gm._don_pool_player(player, action)
    # 「キャラの効果でドン‼をアクティブにできない」: 効果によるアクティブ化を抑止。
    if gm._active_restriction(tp, "CANNOT_ACTIVATE_DON"):
        return True
    activated = 0
    for _ in range(value):
        if not tp.don_rested:
            break
        don = tp.don_rested.pop()
        don.is_rest = False
        tp.don_active.append(don)
        activated += 1
    gm._last_resource_count = activated
    return True
