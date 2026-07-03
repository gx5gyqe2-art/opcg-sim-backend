"""アタック宣言・バトル解決・勝敗判定（GameManager からの移管・第1引数 gm）。"""
from __future__ import annotations

from ..journal import JournaledDict, JournaledList
from ...models.enums import CardType, TriggerType, Zone, ActionType, Phase


def declare_attack(gm, attacker: Card, target: Card):
    attacker_owner, _ = gm._find_card_location(attacker)
    target_owner, _ = gm._find_card_location(target)
    gm._validate_action(attacker_owner, "MAIN_ACTION")
    # 先攻・後攻ともに「自分の最初のターン」はリーダー・キャラのいずれもアタックできない（公式準拠）。
    # ターンは先攻=turn_count 1、後攻=turn_count 2 と交互に進むため、turn_count <= 2 が
    # 両プレイヤーの最初のターンを覆う。
    if gm.turn_count <= 2:
        raise ValueError("最初のターンはアタックできません。")
    if "ATTACK_DISABLE" in attacker.flags or "ATTACK_DISABLE" in attacker.timed_flags: raise ValueError("このカードは効果によりアタックできません。")
    if "CANNOT_REST" in attacker.timed_flags: raise ValueError("このカードは効果によりレストにできないためアタックできません。")
    if attacker.is_rest: raise ValueError("アタックするカードはアクティブ状態でなければなりません。")
    # 召喚酔い: 登場したターンのキャラは攻撃できない。ただし「速攻」を持てば可。
    # リーダーは is_newly_played=False のため影響を受けない。
    if (attacker.master.type == CardType.CHARACTER
            and attacker.is_newly_played
            and not attacker.has_keyword("速攻")):
        raise ValueError("登場したターンのキャラクターは攻撃できません（速攻を除く）。")
    # 自己制限（self_cannot）:「リーダーにアタックできない」。相手リーダーへの攻撃宣言を弾く。
    if (target.master.type == CardType.LEADER
            and attacker_owner is not None
            and gm._active_restriction(attacker_owner, "CANNOT_ATTACK_LEADER")):
        raise ValueError("効果により、このターンはリーダーにアタックできません。")
    if (target.master.type == CardType.CHARACTER and not target.is_rest
            and not attacker.has_keyword("ATTACK_ACTIVE")):
        raise ValueError("レスト状態のキャラクターのみ攻撃可能です。")
    # アタック税（OP08-043「アタックする際、自身の手札N枚を捨てなければアタックできない」）。
    # 付与された ATTACK_TAX_DISCARD_N フラグがあれば、手札N枚を支払えるときのみアタック可。
    tax_flags = [f for f in (attacker.flags | attacker.timed_flags)
                 if isinstance(f, str) and f.startswith("ATTACK_TAX_DISCARD_")]
    if tax_flags:
        need = max(int(f.rsplit("_", 1)[1]) for f in tax_flags)
        if len(attacker_owner.hand) < need:
            raise ValueError(f"アタックするには手札{need}枚を捨てる必要があり、手札が足りません。")
        # コスト支払い: 手札N枚を捨てる。どの札を捨てるかは本来プレイヤー選択だが、宣言経路を
        # 中断させないため先頭からN枚を捨てる（捨て札選択の対話化は今後の課題）。
        for _ in range(need):
            attacker_owner.trash.append(attacker_owner.hand.pop(0))
    attacker.is_rest = True
    gm.active_battle = JournaledDict({"attacker": attacker, "target": target, "attacker_owner": attacker_owner, "target_owner": target_owner, "counter_buff": 0})

    # アタック時/相手のアタック時トリガーを順に解決する。途中でいずれかが対象選択や
    # 選択(Choice)で中断した場合、解決前にブロッカー/カウンター段階へ進むと、未解決の
    # interaction とカウンター操作が衝突する（"期待:CHOICE" エラー）。トリガーを待ち行列に
    # 積み、_advance_battle_triggers で1つずつ解決し、全て片付いてからフェイズ遷移する。
    triggers = []
    if attacker.master.abilities:
        for ability in attacker.master.abilities:
            if ability.trigger == TriggerType.ON_ATTACK:
                triggers.append((attacker_owner, ability, attacker))
            # 「このキャラがレストになった時」(ON_REST) はアタック宣言で自身がレストになった
            # 瞬間に誘発する（要因＝アタックなので「効果で」限定の能力は対象外）。
            # OP14-119/027/028/032/035 等。CONTEXT/ターン1回条件は resolve_ability が評価。
            elif (ability.trigger == TriggerType.ON_REST
                  and gm._rest_subject_matches(ability, attacker, attacker,
                                                 attacker_owner, by_attack=True)):
                triggers.append((attacker_owner, ability, attacker))
    opp_cards = ([target_owner.leader] if target_owner.leader else []) + target_owner.field
    for card in opp_cards:
        for ability in card.master.abilities:
            if ability.trigger == TriggerType.ON_OPP_ATTACK:
                triggers.append((target_owner, ability, card))
    gm._battle_triggers = JournaledList(triggers)
    gm._advance_battle_triggers()

def _advance_battle_triggers(gm):
    """積んだバトルトリガーを順に解決し、全て解決後に防御フェイズへ遷移する。
    途中で interaction が立ったら中断（resolve_interaction が解決後に再度呼ぶ）。"""
    if not gm.active_battle:
        gm._battle_triggers = JournaledList()
        return
    while getattr(gm, "_battle_triggers", None):
        player, ability, card = gm._battle_triggers.pop(0)
        gm.resolve_ability(player, ability, source_card=card)
        if gm.active_interaction:
            return  # 中断: 解決後に resolve_interaction から再開される
    # 全トリガー解決 → ブロッカー/カウンター段階へ
    target_owner = gm.active_battle["target_owner"]
    if gm.has_blocker(target_owner):
        gm.phase = Phase.BLOCK_STEP
    else:
        gm.phase = Phase.BATTLE_COUNTER

def handle_block(gm, blocker: Optional[Card] = None):
    if not gm.active_battle: return
    target_owner = gm.active_battle["target_owner"]; gm._validate_action(target_owner, "SELECT_BLOCKER")
    if blocker:
        blocker.is_rest = True
        gm.active_battle["target"] = blocker
        # 【ブロック時】効果を発動する（従来は未発火＝14枚が no-op だった）。
        if blocker.master.abilities and not blocker.is_effect_negated and not blocker.negated:
            for ability in blocker.master.abilities:
                if ability.trigger == TriggerType.ON_BLOCK:
                    gm.resolve_ability(target_owner, ability, source_card=blocker)
        if gm.active_interaction:
            # ブロック時効果が対象選択等で中断した場合はここで返す（resume が継続）。
            return
    gm.phase = Phase.BATTLE_COUNTER;

def apply_counter(gm, player: Player, counter_card: Optional[Card] = None, don_list: Optional[List[DonInstance]] = None):
    if not gm.active_battle: return
    if counter_card is None: gm.resolve_attack(); return
    gm._validate_action(player, "SELECT_COUNTER")
    if counter_card.master.type == CardType.EVENT:
        gm.pay_cost(player, counter_card.master.cost, don_list)
        for ability in counter_card.master.abilities:
            if ability.trigger == TriggerType.COUNTER: gm.resolve_ability(player, ability, source_card=counter_card)
        # 「自分のキャラすべては、このターン中、…代わりに〜できる」(EB02-030) のような
        # 継続付与型の置換を登録する。イベントは即トラッシュで場に残らないため、
        # _find_replacement の場上 protector 走査では拾えない。player へ this-turn 付与する。
        gm._register_granted_replacements(player, counter_card)
        gm.move_card(counter_card, Zone.TRASH, player)
    else:
        counter_value = getattr(counter_card, "current_counter", counter_card.master.counter or 0); gm.active_battle["counter_buff"] += counter_value
        gm.move_card(counter_card, Zone.TRASH, player)

def resolve_attack(gm):
    if not gm.active_battle: return
    attacker = gm.active_battle["attacker"]; target = gm.active_battle["target"]
    attacker_owner = gm.active_battle["attacker_owner"]; target_owner = gm.active_battle["target_owner"]
    counter_buff = gm.active_battle.get("counter_buff", 0)
    is_my_turn = (attacker_owner == gm.turn_player); is_target_turn = (target_owner == gm.turn_player)
    attacker_pwr = attacker.get_power(is_my_turn); target_pwr = target.get_power(is_target_turn) + counter_buff
    life_lost = 0
    if target == target_owner.leader:
        if attacker_pwr >= target_pwr:
            damage_amount = 2 if attacker.has_keyword("ダブルアタック") else 1; is_banish = attacker.has_keyword("バニッシュ")
            for _ in range(damage_amount):
                if target_owner.life:
                    life_card = target_owner.life.pop(0)
                    dest_zone = Zone.TRASH if is_banish else Zone.HAND
                    trigger_ability = None if is_banish else next(
                        (a for a in life_card.master.abilities if a.trigger == TriggerType.TRIGGER), None
                    )
                    gm.move_card(life_card, dest_zone, target_owner)
                    life_lost += 1
                    # 【トリガー】は「発動できる」（任意）。即時解決せず確認付きで待ち行列へ。
                    # 複数枚（ダブルアタック等）でも確認/解決が中断を跨いで消失しない。
                    if trigger_ability:
                        gm._enqueue_trigger(target_owner, trigger_ability, life_card, optional=True)
                else: gm.winner = attacker_owner.name; break
    else:
        if attacker_pwr >= target_pwr:
            if gm._active_protection(target, ("BATTLE_KO",), attacker=attacker):
                pass
            else:
                repl = gm._find_replacement(target, ("BATTLE_KO",))
                if repl is not None and getattr(repl[3], "is_optional", False):
                    # 任意のバトルKO置換（「代わりに〜してもよい/できる」OP10-034 等）は、
                    # 被KO側に「代わりの効果を使うか」を確認するため戦闘を中断する。
                    # accept→置換実行（本来のKOをスキップ）、decline→本来のKOを実行。
                    # どちらの分岐も resume 時に _finish_attack で戦闘後処理を行う。
                    gm._suspend_for_battle_ko_replacement(target, target_owner, life_lost)
                    return
                elif repl is not None:
                    # 任意でない置換は従来どおり即時実行（内側選択はヘッドレス自動解決）。
                    gm._active_replacement(target, ("BATTLE_KO",))
                else:
                    gm.move_card(target, Zone.TRASH, target_owner)
                    gm._resolve_on_ko(target, target_owner, cause="BATTLE")

    gm._finish_attack(target, target_owner, life_lost)

def _finish_attack(gm, target: Card, target_owner: Player, life_lost: int):
    """戦闘解決後の共通後処理。インラインのバトルKO判定からも、任意バトルKO置換の
    確認(CONFIRM_OPTIONAL)からの resume からも呼ばれる。"""
    target.reset_turn_status(keep_don=True); gm.active_battle = None; gm.phase = Phase.MAIN; gm.check_victory()
    gm.continuous.expire("BATTLE_END", gm.turn_count)
    if not gm.winner:
        gm._apply_passive_effects(gm.turn_player)
    # ライフが離れた回数ぶん ON_LIFE_DECREASE を待ち行列へ積み、【トリガー】と共に消化する。
    if life_lost and not gm.winner:
        gm._enqueue_life_decrease(target_owner, life_lost)
    gm._advance_pending_triggers()

def _suspend_for_battle_ko_replacement(gm, target: Card, target_owner: Player, life_lost: int):
    """任意のバトルKO置換を被KO側へ確認するため戦闘を中断する（CONFIRM_OPTIONAL）。
    resume 時: accept→置換実行（KOスキップ）／decline→本来のKO、その後 _finish_attack。
    ヘッドレス/CPU の既定応答(index0=accept)は従来の自動採用と一致する。"""
    gm.active_interaction = {
        "player_id": target_owner.name,
        "action_type": "CONFIRM_OPTIONAL",
        "source_card_name": target.master.name,
        "source_card_uuid": target.uuid,
        "message": f"「{target.master.name}」がバトルでKOされます。代わりの効果を使用しますか？",
        "can_skip": True,
        "continuation": {
            "kind": "BATTLE_KO_REPLACE",
            "source_card_uuid": target.uuid,
            "target_owner_name": target_owner.name,
            "life_lost": life_lost,
        },
    }

def has_blocker(gm, player: Player) -> bool:
    for card in player.field:
        if (not card.is_rest and card.has_keyword("ブロッカー")
                and "BLOCKER_DISABLED" not in card.flags
                and "CANNOT_REST" not in card.timed_flags):
            return True
    return False

def check_victory(gm):
    # デッキアウト: 通常は本人の敗北（相手の勝利）。ただし C10「自分のデッキが0枚に
    # なった場合、敗北する代わりに勝利する」(VICTORY/REPLACE_DECKOUT_LOSS) を持つ場合は
    # 本人の勝利へ置換する（OP03-040 ナミ等）。
    if not gm.p1.deck:
        gm.winner = gm.p1.name if gm._has_deckout_win_replace(gm.p1) else gm.p2.name
    elif not gm.p2.deck:
        gm.winner = gm.p2.name if gm._has_deckout_win_replace(gm.p2) else gm.p1.name

def _has_deckout_win_replace(gm, player) -> bool:
    """player がデッキアウト時の敗北→勝利の置換能力(PASSIVE)を持つか。"""
    units = [player.leader] + list(player.field)
    for card in units:
        if not card or not getattr(card, "master", None) or getattr(card, "negated", False):
            continue
        if getattr(card, "is_effect_negated", False):
            continue
        for ab in card.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            eff = gm._find_action(ab.effect, ActionType.VICTORY)
            if eff is not None and eff.status == "REPLACE_DECKOUT_LOSS":
                return True
    return False
