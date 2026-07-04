"""ゲーム開始・マリガン・ターン進行/フェイズ遷移（GameManager からの移管・第1引数 gm）。"""
from __future__ import annotations

import random
import logging

from ..journal import JournaledDict, JournaledList, JournaledSet
from ...models.enums import Phase, TriggerType
from ..effects.resolver import EffectResolver

_logger = logging.getLogger("opcg.engine")


def start_game(gm, first_player: Optional[Player] = None):
    
    gm.p1.shuffle_deck()
    gm.p2.shuffle_deck()
    
    for p in [gm.p1, gm.p2]:
        if p.leader:
            for ability in p.leader.master.abilities:
                if ability.trigger == TriggerType.GAME_START:
                    gm.resolve_ability(p, ability, source_card=p.leader)
                    
                    if gm.active_interaction:
                        gm.setup_phase_pending = True
                        if first_player: gm.turn_player = first_player; gm.opponent = gm.p2 if first_player == gm.p1 else gm.p1
                        else: gm.turn_player = gm.p1; gm.opponent = gm.p2
                        return

    gm.finish_setup()

    if first_player: gm.turn_player = first_player; gm.opponent = gm.p2 if first_player == gm.p1 else gm.p1
    else: gm.turn_player = gm.p1; gm.opponent = gm.p2
    # マリガンフェーズへ移行（両プレイヤーの確定後にゲーム開始）
    gm.phase = Phase.MULLIGAN
    gm.mulligan_done = JournaledSet()

def do_mulligan(gm, player: 'Player') -> None:
    """手札5枚全てをデッキ底に戻してシャッフル→5枚引き直す（全交換・1回限り）"""
    if gm.phase != Phase.MULLIGAN:
        raise ValueError("マリガンフェーズではありません。")
    if player.name in gm.mulligan_done:
        raise ValueError("既にマリガンを実施済みです。")
    # 手札を全てデッキ底に戻す
    hand_count = len(player.hand)
    player.deck.extend(player.hand)
    player.hand.clear()
    random.shuffle(player.deck)
    for _ in range(5):
        if player.deck:
            player.hand.append(player.deck.pop(0))
    gm.mulligan_done.add(player.name)
    gm._check_mulligan_complete()

def keep_hand(gm, player: 'Player') -> None:
    """手札をキープしてマリガンをスキップ"""
    if gm.phase != Phase.MULLIGAN:
        raise ValueError("マリガンフェーズではありません。")
    if player.name in gm.mulligan_done:
        raise ValueError("既にマリガンを実施済みです。")
    gm.mulligan_done.add(player.name)
    gm._check_mulligan_complete()

def _check_mulligan_complete(gm) -> None:
    """両プレイヤーのマリガン確定後にゲーム開始"""
    if gm.p1.name in gm.mulligan_done and gm.p2.name in gm.mulligan_done:
        gm.turn_count = 1
        gm.refresh_phase()

def finish_setup(gm):
    gm.p1.place_life()
    gm.p1.draw_initial_hand()
    gm.p2.place_life()
    gm.p2.draw_initial_hand()

def end_turn(gm):
    gm._validate_action(gm.turn_player, "MAIN_ACTION")
    gm.phase = Phase.END
    gm._fire_turn_end_triggers()
    # 「このターン終了時、〜」で予約された遅延アクションを解決する。
    gm._flush_pending_end_of_turn()
    gm.continuous.expire("TURN_END", gm.turn_count)
    gm.switch_turn()

def _fire_turn_end_triggers(gm):
    """ターン終了時トリガーを発火する。ターンプレイヤーの【自分のターン終了時】
    (TURN_END) と、非ターンプレイヤーの【相手のターン終了時】(OPP_TURN_END)。"""
    def _units(pl):
        us = [pl.leader] + pl.field
        if pl.stage: us.append(pl.stage)
        return us
    for pl, trig in ((gm.turn_player, TriggerType.TURN_END),
                     (gm.opponent, TriggerType.OPP_TURN_END)):
        for card in _units(pl):
            if card and card.master.abilities:
                for ability in card.master.abilities:
                    if ability.trigger == trig:
                        # 先行トリガーが確認/選択で中断中は即時解決できない（resolver は
                        # 中断中1ステップも実行せず return し、能力が無言で消える）。
                        # コスト付きターン終了時は使用確認(CONFIRM_OPTIONAL)で中断するのが
                        # 常態のため、中断中は誘発待ち行列へ積み、対話完了時に消化する。
                        if gm.active_interaction:
                            gm._enqueue_trigger(pl, ability, card, optional=False)
                        else:
                            gm.resolve_ability(pl, ability, source_card=card)

def _flush_pending_end_of_turn(gm):
    """end_turn フックで、予約された遅延アクション（このターン終了時、〜）を解決する。"""
    if not gm.pending_end_of_turn:
        return
    pending = gm.pending_end_of_turn
    gm.pending_end_of_turn = JournaledList()
    for player, node, source_card in pending:
        # ターン終了時トリガーの確認等で中断中は直接実行できない（resolver が中断中は
        # 1ステップも実行せず return し、遅延アクションが無言で消える）。deferred 継続へ
        # 退避し、中断解決後に resolve_interaction 末尾が再開する。
        if gm.active_interaction:
            gm._defer_resolver_stack(player, source_card, [node],
                                     {"_flushing_delayed": True})
            continue
        # 場を離れたカードのソース由来でも、トラッシュ送り等は対象解決時に弾かれる。
        resolver = EffectResolver(gm)
        resolver.context["_flushing_delayed"] = True
        resolver.execution_stack = [node]
        try:
            resolver._process_stack(player, source_card)
        except Exception as e:
            # 遅延効果（ターン終了時フラッシュ）の1件が失敗しても、残りの pending の
            # フラッシュは続行する（1件の破綻で全体を止めない）。診断のみ残す。
            _logger.debug("遅延効果のフラッシュで1件失敗（続行）: %r", e, exc_info=True)
        for ev in resolver.action_history:
            gm.action_events.append({
                "type": "EFFECT", "player": player.name,
                "card_name": source_card.master.name,
                "action": ev.get("action", ""), "targets": ev.get("targets", []),
                "value": ev.get("value"), "success": ev.get("success", True),
            })

def switch_turn(gm):
    # ターンが切り替わる/追加ターンに入る = 新しいターン。ターン内イベント記録をクリアする。
    gm._turn_events = JournaledDict()
    # 追加ターン（EXTRA_TURN）: 予約したプレイヤーがターンプレイヤーのまま継続する
    if getattr(gm, "pending_extra_turn", None) == gm.turn_player.name:
        gm.pending_extra_turn = None
        gm.turn_count += 1
        gm.refresh_phase()
        return
    gm.turn_player, gm.opponent = gm.opponent, gm.turn_player
    gm.turn_count += 1
    gm.refresh_phase()

def refresh_phase(gm):
    gm._reset_player_status(gm.opponent); gm.refresh_all(gm.turn_player); gm.draw_phase()

def _reset_player_status(gm, player: Player):
    # 相手ターン開始時に直前のターンプレイヤー(=現opponent)の一時効果を解除するが、
    # 付与ドン!!は剥がさない（持ち主の次のリフレッシュフェイズまでカードに残る）。
    all_units = [player.leader] + player.field
    if player.stage: all_units.append(player.stage)
    for card in all_units:
        # ターン境界のリセット。【ターン1回】の使用回数もここで戻す。
        if card: card.reset_turn_status(keep_don=True, clear_usage=True)

def refresh_all(gm, player: Player):
    all_units = [player.leader] + player.field
    if player.stage: all_units.append(player.stage)
    for card in all_units:
        if card:
            is_frozen = "FREEZE" in card.flags
            # ターン境界のリセット。【ターン1回】の使用回数もここで戻す。
            card.reset_turn_status(clear_usage=True)
            if not is_frozen: card.is_rest = False
    
    # フリーズ中のドン!!（FREEZE_DON / OP07-026）は今回のリフレッシュではアクティブに
    # 戻さず、レストのまま据え置いてフラグを下ろす（1回限りのフリーズ）。
    still_frozen, to_activate = [], []
    for don in player.don_rested:
        if don.is_frozen:
            don.is_frozen = False
            still_frozen.append(don)
        else:
            don.is_rest = False
            to_activate.append(don)
    player.don_active.extend(to_activate)
    player.don_rested = JournaledList(still_frozen)
    
    for don in player.don_attached_cards:
        don.is_rest = False
        don.attached_to = None
        player.don_active.append(don)
    player.don_attached_cards = JournaledList()

def draw_phase(gm):
    if gm.turn_count > 1: gm.draw_card(gm.turn_player)
    gm.don_phase()

def don_phase(gm):
    cards_to_add = 1 if gm.turn_count == 1 else 2
    for _ in range(cards_to_add):
        if gm.turn_player.don_deck:
            don = gm.turn_player.don_deck.pop(0); gm.turn_player.don_active.append(don)
    gm.main_phase()

def main_phase(gm): 
    gm.phase = Phase.MAIN
    gm._apply_passive_effects(gm.turn_player)
