"""誘発能力キュー・KO/レスト/離脱/ライフ減少トリガー（GameManager からの移管・第1引数 gm）。"""
from __future__ import annotations

import re

from ..journal import JournaledDict
from ...models.enums import TriggerType, Zone, CardType
from ._helpers import _nfc


def _enqueue_trigger(gm, player: Player, ability: Ability, card: CardInstance,
                     optional: bool = False) -> None:
    """誘発能力を待ち行列へ積む。optional=True は発動可否（使う/使わない）を確認する。"""
    # JournaledDict 化＝中断中の item["_confirmed"]=True 等の in-place 変更を巻き戻せるように
    # する（parked resolver の make/unmake 化・round-trip ゲートで照合）。
    gm._pending_triggers.append(JournaledDict({
        "player": player, "ability": ability, "card": card,
        "optional": optional, "_confirmed": False,
    }))

def _advance_pending_triggers(gm) -> None:
    """積んだ誘発能力を順に消化する。中断（対話）が立ったら return し、
    resolve_interaction が解決後に再度呼ぶ。optional は未確認なら確認対話へ中断する。"""
    while gm._pending_triggers:
        if gm.active_interaction:
            return  # 別の対話が進行中: 解決後に再開される
        item = gm._pending_triggers[0]
        if item.get("optional") and not item.get("_confirmed"):
            gm._suspend_for_trigger_confirm(item)
            return
        gm._pending_triggers.pop(0)
        gm._relocate_activated_trigger_card(item)
        gm.resolve_ability(item["player"], item["ability"], source_card=item["card"])
        if gm.active_interaction:
            return  # 効果解決が対象選択等で中断 → resolve_interaction が再開

def _relocate_activated_trigger_card(gm, item: Dict[str, Any]) -> None:
    """発動が確定したライフ公開【トリガー】のカードを、効果解決前に手札からトラッシュへ移す。

    ライフのカードはダメージ時に一旦手札へ置かれるが、【トリガー】を「発動する」場合は
    手札に残らず、効果解決後にトラッシュへ置かれる（OPCG ルール）。効果が自身を登場させる
    /手札に加える場合は、その効果の move_card が（トラッシュにある）当該カードを再配置する
    ため、最終的な置き場所は効果に従う。発動しない（パス）場合はこの関数を通らず手札に残る。
    """
    ability = item.get("ability")
    card = item.get("card")
    player = item.get("player")
    if ability is None or card is None or player is None:
        return
    if getattr(ability, "trigger", None) != TriggerType.TRIGGER:
        return  # ON_LIFE_DECREASE 等（場のカードの誘発）は対象外
    if card in player.hand:
        gm.move_card(card, Zone.TRASH, player)

def _suspend_for_trigger_confirm(gm, item: Dict[str, Any]) -> None:
    """【トリガー】等の発動可否を yes/no で確認するため中断する。
    resolve_interaction が CONFIRM_TRIGGER を処理して同じ item を再投入/破棄する。"""
    player = item["player"]
    card = item["card"]
    # 【トリガー】接頭辞はライフ公開トリガーのみ。任意の誘発（ターン開始時/場を離れた時
    # 等の「発動できる」）では誘発名を出さず汎用文言にする。
    ability = item.get("ability")
    is_life_trigger = getattr(ability, "trigger", None) == TriggerType.TRIGGER
    prefix = "【トリガー】" if is_life_trigger else ""
    gm.active_interaction = {
        "player_id": player.name,
        "action_type": "CONFIRM_TRIGGER",
        "source_card_name": card.master.name,
        "source_card_uuid": card.uuid,
        "message": f"{prefix}「{card.master.name}」の効果を発動しますか？",
        "can_skip": True,
        "continuation": {"trigger_item": item},
    }

def _ko_trigger_matches(gm, ability: Ability, owner: Player,
                        cause: str, effect_controller: Player = None) -> bool:
    """このキャラ自身の【KO時】誘発の要因・タイミング修飾を判定する。

    書き下し形「…KOされた時」の前段に出る修飾を raw_text から解釈し、修飾が
    無ければ常に発火（従来挙動）。ブラケット【KO時】（"KOされた時" を含まない）は
    要因を問わず発火する。
    - cause: "BATTLE"（戦闘KO）/ "EFFECT"（効果KO）。
    - effect_controller: 効果KOを引き起こした側（戦闘KOは None）。
    修飾:
    - 「相手の(キャラの)効果で」: 相手の効果KOのみ（戦闘KO・自分の効果KOを除外）。
    - 「自分の効果で」: 自分の効果KOのみ。
    - 「(単に)効果で」: 効果KOのみ（戦闘KOを除外）。
    - 【相手のターン中】: 相手ターン中のみ。【自分のターン中】: 自分ターン中のみ。
    """
    raw = _nfc(getattr(ability, "raw_text", "") or "")
    if _nfc("KOされた時") not in raw:
        return True  # ブラケット【KO時】等：要因を問わず発火
    # タイミングスコープ
    if _nfc("相手のターン中") in raw and gm.turn_player is owner:
        return False
    if _nfc("自分のターン中") in raw and gm.turn_player is not owner:
        return False
    # 要因（「KOされた時」の前段の修飾を見る）
    pre = raw.split(_nfc("KOされた時"))[0]
    opp = gm.p1 if owner is gm.p2 else gm.p2
    if _nfc("相手の") in pre and _nfc("効果で") in pre:
        return cause == "EFFECT" and effect_controller is opp
    if _nfc("自分の効果で") in pre:
        return cause == "EFFECT" and effect_controller is owner
    if _nfc("効果で") in pre:
        return cause == "EFFECT"
    return True

def _resolve_on_ko(gm, card: Card, owner: Player,
                   cause: str = "EFFECT", effect_controller: Player = None):
    # このターンに当該プレイヤーのキャラが KO された事実を記録する
    # （「このターン中、相手のキャラがKOされている場合」OP16-100 の判定用）。
    gm.record_turn_event(f"CHAR_KOED_{owner.name}", 1)
    # 他カードの「…キャラがKOされた時」リスナーを積む（自身の【KO時】とは独立）。
    gm._enqueue_ko_listeners(card, owner)
    if not card.master.abilities: return
    for ability in card.master.abilities:
        if ability.trigger == TriggerType.ON_KO:
            if not gm._ko_trigger_matches(ability, owner, cause, effect_controller):
                continue
            gm.resolve_ability(owner, ability, source_card=card)

def _rest_subject_matches(gm, ability: Ability, rested_card: Card, host: Card,
                          host_owner: Player, by_attack: bool,
                          effect_controller: Player = None, cause_source: Card = None) -> bool:
    """ON_REST 誘発（「（この）キャラが（自分の/相手の効果で）レストになった時」）の
    主語・要因フィルタ。条件には載らない修飾を raw_text から解釈する。

    - 主語: 「このキャラ」＝能力保持カード(host)自身がレストになった場合のみ。
      「キャラ」(この無し)＝任意のキャラがレストになった場合（host 識別は問わない）。
    - 要因: 「自分の効果で」＝host_owner の効果による効果レスト（アタック不可）。
      「相手の(キャラの)効果で」＝host_owner の相手の効果による効果レスト（アタック不可）。
      修飾無し＝アタック/効果どちらでも可。
    """
    raw = _nfc(getattr(ability, "raw_text", "") or "")
    pre = raw.split(_nfc("レストになった時"))[0]
    # 主語フィルタ
    if _nfc("このキャラ") in pre and rested_card is not host:
        return False
    # 要因フィルタ
    if _nfc("自分の効果で") in pre:
        if by_attack or effect_controller is not host_owner:
            return False
    elif _nfc("相手の") in pre and _nfc("効果で") in pre:
        if by_attack or effect_controller is None or effect_controller is host_owner:
            return False
        # 「相手のキャラの効果で」は発生源がキャラに限定（リーダーの効果では発火しない。OP14-070）。
        # 発生源が判明している場合のみ厳密化する（不明＝従来どおり発火を許容）。
        if _nfc("相手のキャラの効果で") in pre and cause_source is not None:
            src_master = getattr(cause_source, "master", None)
            if src_master is None or src_master.type != CardType.CHARACTER:
                return False
    return True

def _fire_on_rest_triggers(gm, rested_card: Card, by_attack: bool,
                           effect_controller: Player = None, cause_source: Card = None):
    """キャラがレストになった時(ON_REST)の誘発を、両プレイヤーのリーダー/場から探して解決する。

    要因（by_attack=アタック宣言 / effect_controller=効果でレストにした側 /
    cause_source=効果の発生源カード）を主語・要因フィルタ（_rest_subject_matches）へ渡す。
    発動可否・文脈（自分のターン中 等）・ターン1回は resolve_ability/_check_condition が評価する。"""
    pending = []
    for p in (gm.p1, gm.p2):
        hosts = ([p.leader] if p.leader else []) + list(p.field)
        for host in hosts:
            if host is None or not host.master.abilities:
                continue
            for ability in host.master.abilities:
                if ability.trigger != TriggerType.ON_REST:
                    continue
                if gm._rest_subject_matches(ability, rested_card, host, p,
                                              by_attack=by_attack,
                                              effect_controller=effect_controller,
                                              cause_source=cause_source):
                    pending.append((p, ability, host))
    for owner, ability, host in pending:
        gm.resolve_ability(owner, ability, source_card=host)
        if gm.active_interaction:
            return

def _leave_subject_matches(gm, ability: Ability, leaving_card: Card,
                           ability_owner: Player, leaving_owner: Player) -> bool:
    """ON_LEAVE 誘発の主語フィルタ（「自分の特徴《X》を持つキャラが（相手の効果で）場を
    離れた時」）が、実際に離れたカードに一致するか。条件には載らない主語修飾を raw_text
    から解釈する（側＝自分/相手、特徴、カード名、「相手の効果で」限定）。"""
    raw = _nfc(getattr(ability, "raw_text", "") or "")
    pre = raw.split(_nfc("場を離れ"))[0]
    # 側（自分/相手）: 既定は自分。
    if _nfc("相手の") in pre and _nfc("自分の") not in pre:
        if leaving_owner is ability_owner:
            return False
    else:
        if leaving_owner is not ability_owner:
            return False
    # 「相手の効果で場を離れた時」限定: 相手のターン中（＝相手の効果による除去）でなければ不発。
    # 自分のターン中の自己バウンス等では誘発しない（OP13-078）。
    if _nfc("相手の効果で") in pre and gm.turn_player is leaving_owner:
        return False
    # 特徴フィルタ（《X》/『X』。「X を含む特徴」も部分一致で拾う）。
    traits = re.findall(r'[《<『]([^》>』]+)[》>』]', pre)
    if traits and not any(any(t in ct for ct in leaving_card.master.traits) for t in traits):
        return False
    # カード名フィルタ（「X」）。
    names = re.findall(r'「([^」]+)」', pre)
    if names and not any(leaving_card.master.matches_name(n, partial=True) for n in names):
        return False
    return True

def _enqueue_on_leave(gm, leaving_card: Card, leaving_owner: Player) -> None:
    """キャラが場を離れた時(ON_LEAVE)の誘発を、両プレイヤーのリーダー/場から探して積む。
    主語フィルタ（側・特徴・名前）に一致する能力のみを対象とする（バギー OP16-041 等）。"""
    for owner in (gm.p1, gm.p2):
        holders = ([owner.leader] if owner.leader else []) + list(owner.field)
        for holder in holders:
            if holder is leaving_card:
                continue
            for ability in getattr(holder.master, "abilities", ()):
                if ability.trigger != TriggerType.ON_LEAVE:
                    continue
                if not gm._leave_subject_matches(ability, leaving_card, owner, leaving_owner):
                    continue
                optional = _nfc("発動できる") in _nfc(getattr(ability, "raw_text", "") or "")
                gm._enqueue_trigger(owner, ability, holder, optional=optional)

# ---------------------------------------------------------------------------
# キャラ登場イベントのリスナー（「…が登場した時」を持つ他カードの誘発）
# パーサはタイミングタグから PASSIVE/YOUR_TURN/OPPONENT_TURN に写像するが、これらは
# 継続効果の再計算ループでは反応型（_is_reactive_passive）としてスキップされるため、
# 登場イベントの発生地点からここで発火させる（OP14-041/OP13-100/OP16-079）。
# ---------------------------------------------------------------------------

_CHAR_PLAYED_LISTENER_TRIGGERS = (TriggerType.PASSIVE, TriggerType.YOUR_TURN,
                                  TriggerType.OPPONENT_TURN)

def _played_subject_matches(gm, ability: Ability, holder_owner: Player,
                            played_card: Card, played_owner: Player,
                            from_zone: str = None) -> bool:
    """「…が登場した時」リスナーの主語・タイミングフィルタ。

    - タイミング: YOUR_TURN（【自分のターン中】）は保持者のターンのみ、
      OPPONENT_TURN（【相手のターン中】）は保持者の相手のターンのみ。PASSIVE は常時。
    - 主語の側: 「相手の…」＝保持者の相手のキャラ登場のみ。既定（「自分の」/無指定）＝自分。
    - 出所ゾーン: 「トラッシュから」等は from_zone（登場元）と一致した時のみ。
    - 特徴《X》・カード名「X」・「【トリガー】を持つ」を登場カードに適用する。
    """
    raw = _nfc(getattr(ability, "raw_text", "") or "")
    if _nfc("登場した時") not in raw or _nfc("このキャラが登場した時") in raw:
        return False
    pre = raw.split(_nfc("登場した時"))[0]
    # 【相手のターン中】等のタイミングタグ内の「相手の/自分の」を主語判定に混ぜない。
    pre = re.sub(r'【[^】]*】', '', pre)
    if ability.trigger == TriggerType.YOUR_TURN and gm.turn_player is not holder_owner:
        return False
    if ability.trigger == TriggerType.OPPONENT_TURN and gm.turn_player is holder_owner:
        return False
    if _nfc("相手の") in pre:
        if played_owner is holder_owner:
            return False
    elif played_owner is not holder_owner:
        return False
    for key, zone in ((_nfc("トラッシュから"), "TRASH"), (_nfc("手札から"), "HAND"),
                      (_nfc("デッキから"), "DECK"), (_nfc("ライフから"), "LIFE")):
        if key in pre and from_zone != zone:
            return False
    if _nfc("【トリガー】を持つ") in pre:
        has_trig = bool(getattr(played_card.master, "trigger_text", "")) or any(
            ab.trigger == TriggerType.TRIGGER for ab in (played_card.master.abilities or ()))
        if not has_trig:
            return False
    traits = re.findall(r'[《<『]([^》>』]+)[》>』]', pre)
    if traits and not any(any(t in ct for ct in played_card.master.traits) for t in traits):
        return False
    names = re.findall(r'「([^」]+)」', pre)
    if names and not any(played_card.master.matches_name(n, partial=True) for n in names):
        return False
    return True

def _enqueue_char_played_listeners(gm, played_card: Card, played_owner: Player,
                                   from_zone: str = None) -> None:
    """キャラ登場時に、両プレイヤーのリーダー/場/ステージから「…が登場した時」
    リスナーを探して誘発待ち行列へ積む。登場カード自身は対象外（自身の【登場時】は
    ON_PLAY 経路が担う）。"""
    if played_card.master.type != CardType.CHARACTER:
        return
    for owner in (gm.p1, gm.p2):
        holders = ([owner.leader] if owner.leader else []) + list(owner.field)
        if owner.stage:
            holders.append(owner.stage)
        for holder in holders:
            if holder is None or holder is played_card:
                continue
            for ability in (holder.master.abilities or ()):
                if ability.trigger not in _CHAR_PLAYED_LISTENER_TRIGGERS:
                    continue
                if not gm._played_subject_matches(ability, owner, played_card,
                                                  played_owner, from_zone):
                    continue
                optional = _nfc("発動できる") in _nfc(getattr(ability, "raw_text", "") or "")
                gm._enqueue_trigger(owner, ability, holder, optional=optional)

# ---------------------------------------------------------------------------
# 第三者KOリスナー（「自分の/相手の…キャラがKOされた時」を持つ他カードの誘発）
# _resolve_on_ko はKOされたカード自身の【KO時】しか解決しないため、リーダー等が持つ
# 「…キャラがKOされた時」はここで走査して積む（OP14-041/OP01-061/OP13-002 等）。
# ---------------------------------------------------------------------------

def _ko_listener_matches(gm, ability: Ability, holder_owner: Player,
                         koed_card: Card, koed_owner: Player) -> bool:
    """他カードのKOを監視するリスナーの主語フィルタ。

    主語が「このキャラ」（＝自身KO・既存の self 経路が担う）のものは対象外。
    側（自分の/相手の。無指定=両方）・特徴《X》・カード名「X」・
    「（元々の）パワーN以上」をKOされたカードに適用する。タイミング
    （【自分のターン中】等）・【ドン!!×N】・【ターン1回】は条件として
    resolve_ability が評価する。"""
    raw = _nfc(getattr(ability, "raw_text", "") or "")
    if _nfc("KOされた時") not in raw:
        return False
    pre = raw.split(_nfc("KOされた時"))[0]
    # 【自分のターン中】等のタイミングタグ内の「自分の/相手の」を主語判定に混ぜない。
    pre = re.sub(r'【[^】]*】', '', pre)
    if _nfc("このキャラが") in pre or _nfc("キャラが") not in pre:
        return False
    if _nfc("相手の") in pre and _nfc("自分の") not in pre:
        if koed_owner is holder_owner:
            return False
    elif _nfc("自分の") in pre:
        if koed_owner is not holder_owner:
            return False
    m = re.search(_nfc(r'(?:元々の)?パワー(\d+)以上'), pre)
    if m and (koed_card.master.power or 0) < int(m.group(1)):
        return False
    traits = re.findall(r'[《<『]([^》>』]+)[》>』]', pre)
    if traits and not any(any(t in ct for ct in koed_card.master.traits) for t in traits):
        return False
    names = re.findall(r'「([^」]+)」', pre)
    if names and not any(koed_card.master.matches_name(n, partial=True) for n in names):
        return False
    return True

def _enqueue_ko_listeners(gm, koed_card: Card, koed_owner: Player) -> None:
    """KO発生時に、両プレイヤーのリーダー/場/ステージから第三者KOリスナーを探して
    誘発待ち行列へ積む（KOされたカード自身の【KO時】は _resolve_on_ko の self 経路）。"""
    for owner in (gm.p1, gm.p2):
        holders = ([owner.leader] if owner.leader else []) + list(owner.field)
        if owner.stage:
            holders.append(owner.stage)
        for holder in holders:
            if holder is None or holder is koed_card:
                continue
            for ability in (holder.master.abilities or ()):
                if ability.trigger != TriggerType.ON_KO:
                    continue
                if not gm._ko_listener_matches(ability, owner, koed_card, koed_owner):
                    continue
                optional = _nfc("発動できる") in _nfc(getattr(ability, "raw_text", "") or "")
                gm._enqueue_trigger(owner, ability, holder, optional=optional)

def _enqueue_life_decrease(gm, player: Player, count: int = 1) -> None:
    """「ライフが離れた時」(ON_LIFE_DECREASE) 能力を、離れた枚数ぶん待ち行列へ積む。

    公式裁定: 「ライフが離れた時」は自分／相手どちらのライフが離れても、ダメージ／効果の
    いずれでも条件成立する（例: OP11-041 ナミ）。よって離れたライフの持ち主（引数 player）に
    依らず、両プレイヤーの場・リーダーを走査して積む。実際に発動するかは各能力の条件
    （【自分のターン中】=CONTEXT SELF_TURN ／【ターン1回】=TURN_LIMIT ／ 手札枚数 等）が評価し、
    結果としてターンプレイヤー側の能力のみが発動する。発動可否（引く/引かない）は各能力の
    効果内の Choice でプレイヤーが選ぶ。"""
    for _ in range(max(1, count)):
        for owner in (gm.p1, gm.p2):
            cards = ([owner.leader] if owner.leader else []) + owner.field
            for card in cards:
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.ON_LIFE_DECREASE:
                        gm._enqueue_trigger(owner, ability, card, optional=False)

def _fire_on_life_decrease(gm, player: Player, count: int = 1):
    """ライフ離脱の誘発を積んで即座に消化する（効果ダメージ等の単発経路用）。
    戦闘ダメージ経路は resolve_attack 末尾でまとめて積む。"""
    gm._enqueue_life_decrease(player, count)
    gm._advance_pending_triggers()
