"""対象ループ・ハンドラ（1対象への適用・21 種）。

旧 `apply_action_to_engine` 後半の `for target in targets: ... elif act_name == "X": <body>` を
逐語移設したもの。`self`→`gm`、対象スキップの `continue`→`return`、冗長な `success = True`
（ランナー側が常に True 初期化するため no-op）を除去しただけで、挙動は不変。

除去保護・置換ゲート・success 規約は `target_loop.run_target_loop` が一元管理する。
"""
import re
from ...models.enums import ActionType, Zone, TriggerType
from ...models.models import DonInstance
from .registry import target_handler


@target_handler(ActionType.PREVENT_LEAVE)
def prevent_leave(gm, player, action, target, owner, source_list, value, source_card):
    # PASSIVE 由来(INSTANT)はマーカーのみ（除去時に _active_protection が走査）。
    # トリガー効果の期間付き保護は継続効果フラグとして対象に付与する
    # （従来は no-op で「次の…まで、バトルでKOされない」が機能しなかった）。
    dur = getattr(action, "duration", "INSTANT")
    if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END"):
        flag = f"PREVENT_{action.status or 'LEAVE'}"
        expire_turn = gm.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
        gm.continuous.apply(target, "FLAG", dur, flag=flag, expire_turn=expire_turn)


@target_handler(ActionType.KO)
def ko(gm, player, action, target, owner, source_list, value, source_card):
    gm.move_card(target, Zone.TRASH, owner)
    gm._resolve_on_ko(target, owner, cause="EFFECT", effect_controller=player)


@target_handler(ActionType.DISCARD, ActionType.TRASH)
def discard(gm, player, action, target, owner, source_list, value, source_card):
    gm.move_card(target, Zone.TRASH, owner)


@target_handler(ActionType.REVEAL)
def reveal(gm, player, action, target, owner, source_list, value, source_card):
    # 公開のみ（盤面不変）。旧 no-op 分岐。
    pass


@target_handler(ActionType.BOUNCE, ActionType.MOVE_TO_HAND)
def bounce(gm, player, action, target, owner, source_list, value, source_card):
    gm.move_card(target, Zone.HAND, owner)


@target_handler(ActionType.MOVE)
def move(gm, player, action, target, owner, source_list, value, source_card):
    dest_zone = action.destination or Zone.TRASH
    gm.move_card(target, dest_zone, owner)


@target_handler(ActionType.BUFF)
def buff(gm, player, action, target, owner, source_list, value, source_card):
    if action.status == "POWER_OVERRIDE":
        # PASSIVE 再計算由来は再計算レイヤへ（即時効果の上書きを消さない）
        if getattr(gm, "_in_passive_recalc", False):
            target.passive_power_override = value
        else:
            target.base_power_override = value
    elif action.status == "COST_OVERRIDE":
        # コスト絶対値セット（「このターン中、コスト0にする」等）。base_power_override
        # と同様に reset_turn_status で失効する（passive 再計算では消えない）。
        target.base_cost_override = value
    elif action.status == "COST_REDUCTION":
        # 期間付き（このターン中／このバトル中 等）は継続効果(timed_cost)へ。
        # cost_buff は _apply_passive_effects で毎回リセットされ消えるため。
        # 期間指定なし(INSTANT)は従来どおり cost_buff（PASSIVE 再計算で再適用される）。
        dur = getattr(action, "duration", "INSTANT")
        if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END"):
            expire_turn = gm.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
            gm.continuous.apply(target, "COST", dur, amount=value, expire_turn=expire_turn)
        elif hasattr(target, 'cost_buff'):
            target.cost_buff += value
    elif action.status == "COUNTER":
        # 「カウンター+Nになる」: 手札カードのカウンター値修正。
        # PASSIVE 再計算レイヤ（passive_counter）に載せる。
        if getattr(gm, "_in_passive_recalc", False):
            target.passive_counter += value
        else:
            target.passive_counter += value  # 即時付与も同レイヤ（手札は recalc でリセット）
    elif action.status == "BLOCKER_DISABLE":
        target.flags.add("BLOCKER_DISABLED")
        target.current_keywords.discard("ブロッカー")
        target.timed_keywords.discard("ブロッカー")  # 効果付与分の【ブロッカー】も無効化
    else:
        # 期間付きパワー増減は継続効果(timed_power)として管理する。
        #  - THIS_BATTLE: バトル終了で失効（同一ターンの後続バトルへ持ち越さない）。
        #  - THIS_TURN / UNTIL_NEXT_TURN_END: ターン境界の reset_turn_status で
        #    消えると困る（例: 被攻撃リーダーの「このターン中+N」が resolve_attack の
        #    target.reset_turn_status で battle 終了時に即消える）。継続効果に載せて存続させる。
        dur = getattr(action, "duration", "INSTANT")
        if dur in ("THIS_BATTLE", "THIS_TURN", "UNTIL_NEXT_TURN_END"):
            expire_turn = gm.turn_count + 1 if dur == "UNTIL_NEXT_TURN_END" else 0
            gm.continuous.apply(target, "POWER", dur, amount=value, expire_turn=expire_turn)
        elif getattr(gm, "_in_passive_recalc", False):
            # PASSIVE/YOUR_TURN 再計算中: 再計算レイヤに載せる（累積防止）
            target.passive_power += value
        elif hasattr(target, 'power_buff'):
            target.power_buff += value


@target_handler(ActionType.ATTACK_DISABLE, ActionType.RESTRICTION)
def attack_disable(gm, player, action, target, owner, source_list, value, source_card):
    # 「（このターン中／次の相手のターン終了時まで）アタックできない」。
    # アタック税（status=ATTACK_TAX_DISCARD_N）はアタック「不可」ではなく、
    # アタック時に手札N枚の支払いを要求する継続フラグとして付与する（declare_attack で強制）。
    dur = getattr(action, "duration", "INSTANT")
    flag = getattr(action, "status", None) or "ATTACK_DISABLE"
    if not (isinstance(flag, str) and flag.startswith("ATTACK_TAX_")):
        flag = "ATTACK_DISABLE"
    if dur == "UNTIL_NEXT_TURN_END":
        gm.continuous.apply(target, "FLAG", "UNTIL_NEXT_TURN_END", flag=flag, expire_turn=gm.turn_count + 1)
    else:
        gm.continuous.apply(target, "FLAG", "THIS_TURN", flag=flag)


@target_handler(ActionType.PREVENT_REST)
def prevent_rest(gm, player, action, target, owner, source_list, value, source_card):
    # 「（相手の）キャラは…までレストにできない」: レスト不可＝そのキャラは
    # 自身をレストできない＝アタックもブロックもできない（どちらも本体をレストにする）。
    # ATTACK_DISABLE と同様、継続効果の timed_flags に "CANNOT_REST" を載せ、
    # declare_attack / has_blocker でこのフラグを弾く。
    dur = getattr(action, "duration", "INSTANT")
    if dur == "UNTIL_NEXT_TURN_END":
        gm.continuous.apply(target, "FLAG", "UNTIL_NEXT_TURN_END", flag="CANNOT_REST", expire_turn=gm.turn_count + 1)
    else:
        gm.continuous.apply(target, "FLAG", "THIS_TURN", flag="CANNOT_REST")


@target_handler(ActionType.FREEZE)
def freeze(gm, player, action, target, owner, source_list, value, source_card):
    # 「次の相手のリフレッシュフェイズでアクティブにならない」
    # refresh_all が flags["FREEZE"] を確認してからリセットするため、
    # ターン境界を跨ぐ flags に直接書き込む（timed_flags でなく flags）。
    target.flags.add("FREEZE")


@target_handler(ActionType.NEGATE_EFFECT)
def negate_effect(gm, player, action, target, owner, source_list, value, source_card):
    # 「（このターン中／次の相手のターン終了時まで、）効果を無効にする」。
    # 継続効果フラグ "EFFECTS_DISABLED" として付与し、reset_turn_status で
    # 途中解除されないようにする（is_effect_negated が timed_flags も見る）。
    # 期間指定が無い場合は当該ターン終了まで（THIS_TURN）として扱う。
    dur = getattr(action, "duration", "INSTANT")
    cdur = dur if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END") else "THIS_TURN"
    expire_turn = gm.turn_count + 1 if cdur == "UNTIL_NEXT_TURN_END" else 0
    gm.continuous.apply(target, "FLAG", cdur, flag="EFFECTS_DISABLED", expire_turn=expire_turn)
    target._refresh_keywords()


@target_handler(ActionType.RULE_PROCESSING)
def rule_processing(gm, player, action, target, owner, source_list, value, source_card):
    # ルール上の注記（カード名 alias、デッキ枚数ルール等）→ エンジン no-op
    pass


@target_handler(ActionType.REST)
def rest(gm, player, action, target, owner, source_list, value, source_card):
    _was_rested = target.is_rest
    target.is_rest = True
    if isinstance(target, DonInstance) and source_list is not None:
        if source_list is not owner.don_rested:
            if target in source_list:
                source_list.remove(target)
                owner.don_rested.append(target)
                if hasattr(target, 'attached_to'): target.attached_to = None
    # アクティブ→レスト遷移で ON_REST（キャラがレストになった時）を誘発する。
    # 要因＝効果（effect_controller=player）。ドン!!は対象外。既にレストなら不発。
    if not _was_rested and not isinstance(target, DonInstance):
        gm._fire_on_rest_triggers(target, by_attack=False, effect_controller=player,
                                  cause_source=source_card)


@target_handler(ActionType.PLAY_CARD)
def play_card(gm, player, action, target, owner, source_list, value, source_card):
    # 「手札のこのカードは効果で登場できない」: 手札源かつ当該 PASSIVE を持つ対象は
    # 効果による登場をスキップする（NO_EFFECT_PLAY）。
    if source_list is getattr(owner, "hand", None) and gm._blocks_effect_play(target):
        return
    gm.move_card(target, Zone.FIELD, owner)
    target.is_newly_played = True
    # 「レストで登場させる」: フィールドに出た瞬間レスト状態にする。
    # 効果の明示 RESTED、または owner の「キャラはレストで登場する」PASSIVE のいずれか。
    if getattr(action, "status", None) == "RESTED" or gm._has_rested_play(owner):
        target.is_rest = True
    if not target.is_effect_negated:
        for ability in target.master.abilities:
            if ability.trigger == TriggerType.ON_PLAY:
                gm.resolve_ability(owner, ability, source_card=target)
    gm._apply_passive_effects(owner)
    # 効果による登場でも場のキャラ上限超過なら強制トラッシュ（ガード付き）。
    gm._enforce_field_limit(owner)


@target_handler(ActionType.DECK_BOTTOM)
def deck_bottom(gm, player, action, target, owner, source_list, value, source_card):
    # 並び替え/上下選択を要する場合は resolver が ARRANGE_DECK で先に中断するため、
    # ここに来るのは位置確定（TOP/BOTTOM/未指定=BOTTOM）の配置のみ。
    _pos = "TOP" if getattr(action, "dest_position", None) == "TOP" else "BOTTOM"
    gm.move_card(target, Zone.DECK, owner, dest_position=_pos)


@target_handler(ActionType.ACTIVE, ActionType.ACTIVE_DON)
def active(gm, player, action, target, owner, source_list, value, source_card):
    target.is_rest = False
    if isinstance(target, DonInstance):
        if target in owner.don_rested:
            owner.don_rested.remove(target)
            owner.don_active.append(target)


@target_handler(ActionType.ATTACH_DON)
def attach_don(gm, player, action, target, owner, source_list, value, source_card):
    # value 枚のドン!!を付与する。status に "OPP" を含めば相手のドンプールから
    # 付与する（OP15-015「相手のレストのドン‼を付与」）。
    #
    # status に "RESTED" を含む（テキスト「レストのドン‼N枚まで付与」。全98枚がこの
    # 一文言で、「(ドンを)レストにして付与」型は0枚）効果は、コストエリアの“既にレスト
    # 状態のドン”だけを付与する。アクティブのドンは絶対にレストにしない・巻き込まない。
    # 「N枚まで」＝あるだけで可なので、レストが足りなければ少なく付与する。
    #   理由: アクティブのドンを「レストにして付与」するのは基本アクションの『ドン付与』
    #   （action_api ATTACH_DON＝don_active 由来。get_legal_actions も don_active>0 で提示）の
    #   役割で、これらカード効果はそれとは別物＝既にレストのドン（コスト支払いで生じた／
    #   エネルが ramp で追加した等）を再活用する。従来はレスト不足時にアクティブへ
    #   フォールバックし、アクティブのドンまでレスト化して吸っていた＝全カード共通のバグ。
    # status に "RESTED" を含まない汎用「ドン付与」は従来どおり active 優先・尽きたら rested。
    st = action.status or ""
    from_rested = ("RESTED" in st)
    from_opp = ("OPP" in st)
    don_owner = (gm.p2 if player == gm.p1 else gm.p1) if from_opp else player
    n = value if value and value > 0 else 1
    attached = 0
    for _ in range(n):
        if from_rested:
            pool = don_owner.don_rested
        else:
            pool = don_owner.don_active or don_owner.don_rested
        if not pool:
            break
        don = pool.pop(0)
        don.attached_to = target.uuid
        don.is_rest = from_rested
        don_owner.don_attached_cards.append(don)
        target.attached_don += 1
        attached += 1


@target_handler(ActionType.MOVE_CARD)
def move_card(gm, player, action, target, owner, source_list, value, source_card):
    dest = action.destination if action.destination else Zone.HAND
    # 自己制限（self_cannot）:「自分の効果でライフを手札に加えられない」。
    # 自分のライフ→自分の手札の移動のみ抑止する（相手への移動・他ゾーンは対象外）。
    if (dest == Zone.HAND and source_list is owner.life and owner is player
            and gm._active_restriction(player, "CANNOT_LIFE_TO_HAND")):
        return
    dest_pos = getattr(action, 'dest_position', 'BOTTOM') or 'BOTTOM'
    gm.move_card(target, dest, owner, dest_position=dest_pos)
    # 「ライフの上に表向きで加える」等: face_up が指定されていればライフでの向きを反映。
    # ライフは既定で裏向き(is_face_up=False)なので、表向き指定を明示的に立てる。
    if dest == Zone.LIFE and getattr(action, "face_up", None) is not None:
        target.is_face_up = bool(action.face_up)


@target_handler(ActionType.DECK_TOP)
def deck_top(gm, player, action, target, owner, source_list, value, source_card):
    gm.move_card(target, Zone.DECK, owner, dest_position="TOP")


@target_handler(ActionType.FACE_UP_LIFE)
def face_up_life(gm, player, action, target, owner, source_list, value, source_card):
    # 「ライフを表向き／裏向きにする」: status="DOWN" のみ裏向き、他は表向き。
    target.is_face_up = (action.status != "DOWN")


@target_handler(ActionType.GRANT_KEYWORD)
def grant_keyword(gm, player, action, target, owner, source_list, value, source_card):
    keyword = action.status
    if not keyword and getattr(action, 'raw_text', ''):
        import unicodedata as _ud
        _kw = re.search(r'【([^】]+)】', _ud.normalize('NFC', action.raw_text))
        if _kw:
            keyword = _kw.group(1)
    if keyword:
        # 継続効果として付与する（timed_keywords）。current_keywords へ直接
        # 加えると _apply_passive_effects のリセットで消えてしまうため。
        dur = getattr(action, "duration", "INSTANT")
        cdur = dur if dur in ("THIS_TURN", "THIS_BATTLE", "UNTIL_NEXT_TURN_END") else "PERMANENT"
        expire_turn = gm.turn_count + 1 if cdur == "UNTIL_NEXT_TURN_END" else 0
        gm.continuous.apply(target, "KEYWORD", cdur, keyword=keyword, expire_turn=expire_turn)
