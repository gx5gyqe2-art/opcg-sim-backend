"""常在効果（PASSIVE）の再計算・手札自己コスト（GameManager からの移管・ステートレス。第1引数 gm）。"""
from __future__ import annotations

import unicodedata

from .. import journal
from ...models.enums import TriggerType
from ...models.effect_types import GameAction, Sequence, Branch, Choice
from ..effects.resolver import EffectResolver


def refresh_passive_state(gm) -> None:
    """盤面依存の常在効果（パワー/コスト/キーワード）を現在の盤面で再計算する。
    API のアクション境界や対話完了時に呼び、トラッシュ枚数等の変化を即時反映する
    （「自分のトラッシュN枚につき+1000」OP09-086 等のリアルタイム反映）。"""
    if gm.active_interaction or getattr(gm, "_in_passive_recalc", False):
        return
    if gm.turn_player is not None:
        gm._apply_passive_effects(gm.turn_player)

def _is_reactive_passive(gm, ability) -> bool:
    """無タグの反応型（「…が登場した時、」「…が戻された時、」等）でトリガー写像が
    まだ無い PASSIVE 能力か。常時効果ではないため再計算ループで実行してはならない
    （実行すると盤面操作のたびに本体効果が発動し、対話中断が他の解決を飲み込む）。"""
    first = gm._find_first_action(ability.effect)
    raw = getattr(first, "raw_text", "") if first is not None else ""
    # 能力本体の raw_text も見る。先頭アクションの文（例「自分はゲームに勝利する」）に
    # 「…した時、」が無くても、能力全体（例「相手が【ブロッカー】を発動した時、…」OP09-118）
    # が反応型なら再計算ループで実行してはならない（PASSIVE+VICTORY が相手ライフ0だけで
    # 誤って自動勝利するのを防ぐ。本来は相手のブロッカー発動が必要）。
    ab_raw = getattr(ability, "raw_text", "") or ""
    combined = unicodedata.normalize("NFC", (raw or "") + " " + ab_raw)
    return bool(gm._REACTIVE_RE.search(combined))

def _find_first_action(gm, node):
    if node is None:
        return None
    if isinstance(node, GameAction):
        return node
    if isinstance(node, Sequence):
        for a in node.actions:
            found = gm._find_first_action(a)
            if found is not None:
                return found
    elif isinstance(node, Branch):
        return gm._find_first_action(node.if_true) or gm._find_first_action(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            found = gm._find_first_action(o)
            if found is not None:
                return found
    return None

def _apply_passive_effects(gm, player: Player):
    # 対話中断中は再計算しない。Step1 のリセットは無条件に走る一方、Step2/3 の
    # resolve_ability は active_interaction ガードで何も実行できず、リセットだけが
    # 残って PASSIVE/YOUR_TURN バフが消えてしまうため（クザンのコスト-5 等）。
    if gm.active_interaction:
        return
    # Phase2 dirty-flag（探索中のみ）: 前回再計算から journal._mut_count が不変＝盤面入力が
    # 不変なら、継続効果（cost_buff/passive_power/current_keywords）は前回値のまま正しい＝再計算
    # を省く。_mut_count は journaled な全変更＋探索開始で増えるため取り残し無し。正常プレイ
    # （_active is None）では作動せず常に再計算＝従来挙動と完全同値。ロールバックはバフを正しく
    # 復元する＋_mut_count >= _passive_mc を保つため、省略しても必要時は必ず再計算され安全。
    if journal._TL.active is not None and journal._TL.mut_count == gm._passive_mc:
        return
    # YOUR_TURN 効果は常にターンプレイヤー基準で適用する（呼び出し元が owner を
    # 渡しても誤適用しない）。
    if gm.turn_player is not None:
        player = gm.turn_player
    opponent = gm.p2 if player == gm.p1 else gm.p1

    # Step 1: 両プレイヤーのバフ・一時キーワードをリセット
    # 値が変わるときだけ書く（無条件代入は make/unmake の journaled 書き込みを毎ノード大量に生む＝
    # 探索の支配コスト。大半のカードはバフ 0 ＝ no-op 代入なので、ガードで journaling 量を削る。
    # 最終状態は無条件代入と完全同一＝挙動・方策不変）。
    for p in [player, opponent]:
        for c in ([p.leader] if p.leader else []) + p.field + ([p.stage] if p.stage else []):
            if c:
                if c.cost_buff:
                    c.cost_buff = 0
                if c.passive_power:
                    c.passive_power = 0
                if c.passive_power_override is not None:
                    c.passive_power_override = None
                if c.current_keywords != c.master.keywords:
                    c.current_keywords = c.master.keywords.copy()
        for c in p.hand:
            if c:
                if c.cost_buff:
                    c.cost_buff = 0
                if c.passive_counter:
                    c.passive_counter = 0

    # Step 2/3 で適用される INSTANT パワーバフは passive_power（再計算レイヤ）に
    # 載せる。power_buff に加えると _apply_passive_effects が呼ばれるたびに
    # 累積し、PASSIVE「パワー+1000」が盤面操作のたびに際限なく増えていた。
    gm._in_passive_recalc = True
    try:
        # Step 2: YOUR_TURN 効果（アクティブプレイヤーのカードのみ）
        #   ステージ（player.stage）も対象に含める。聖地マリージョア(コスト軽減)・
        #   虚の玉座(リーダー+1000) 等の STAGE の YOUR_TURN 効果が従来発動していなかった。
        for card in ([player.leader] if player.leader else []) + player.field + ([player.stage] if player.stage else []):
            if not card or not card.master.abilities: continue
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.YOUR_TURN:
                    if gm._is_reactive_passive(ability):
                        continue  # 「【自分のターン中】…された時」型はイベント誘発（EB02-035 等）
                    gm.resolve_ability(player, ability, source_card=card)

        # Step 2': OPPONENT_TURN 効果（非アクティブプレイヤーのカードのみ）。
        #   「【相手のターン中】自分のキャラすべてをコスト+1」(OP16-080) 等の継続効果。
        #   コントローラから見て「相手のターン」＝非ターンプレイヤーのカードが該当する。
        #   YOUR_TURN と同じく再計算レイヤ（cost_buff/passive_power）へ載るため、
        #   ターンが替われば自然に消える。
        for card in ([opponent.leader] if opponent.leader else []) + opponent.field + ([opponent.stage] if opponent.stage else []):
            if not card or not card.master.abilities: continue
            for ability in card.master.abilities:
                if ability.trigger == TriggerType.OPPONENT_TURN:
                    if gm._is_reactive_passive(ability):
                        continue  # 「【相手のターン中】…された時」型はイベント誘発
                    gm.resolve_ability(opponent, ability, source_card=card)

        # Step 3: PASSIVE 効果（両プレイヤーのカードを評価）。ステージも含める。
        for p in [player, opponent]:
            for card in ([p.leader] if p.leader else []) + p.field + ([p.stage] if p.stage else []):
                if not card or not card.master.abilities: continue
                for ability in card.master.abilities:
                    if ability.trigger == TriggerType.PASSIVE:
                        if gm._is_reactive_passive(ability):
                            continue  # 「…された時」型はイベント誘発であり再計算で実行しない
                        gm.resolve_ability(p, ability, source_card=card)
    finally:
        gm._in_passive_recalc = False

    # Step 4: 手札カードの自己コスト増減 PASSIVE（「手札のこのカードは、〈条件〉、コスト±N」）。
    #   手札の PASSIVE は Step2/3 の場走査では評価されないため、ここで個別に評価する。
    #   対象は手札のこのカード自身（target.flags に "SELF_IN_HAND"）。条件成立時のみ
    #   cost_buff を加算する（Step1 で 0 にリセット済み）。ウタ ST23-001/サッチ OP16-005 等。
    gm._apply_hand_self_cost(player, opponent)

    # Phase2 dirty-flag: 再計算完了時点の _mut_count を記録（探索中のみ）。自身の書き込みを
    # 含んだ後の値を object.__setattr__ で保持する（journaled せず _mut_count も増やさない＝
    # 次回呼び出しで外部変更が無ければ一致してスキップできる）。
    if journal._TL.active is not None:
        object.__setattr__(gm, "_passive_mc", journal._TL.mut_count)

def _apply_hand_self_cost(gm, player: Player, opponent: Player):
    resolver = None
    for p in [player, opponent]:
        for card in p.hand:
            if not card or not card.master.abilities:
                continue
            for ability in card.master.abilities:
                if ability.trigger != TriggerType.PASSIVE:
                    continue
                eff = ability.effect
                tq = getattr(eff, "target", None)
                if (eff is None or getattr(eff, "status", None) != "COST_REDUCTION"
                        or tq is None or "SELF_IN_HAND" not in getattr(tq, "flags", set())):
                    continue
                if resolver is None:
                    resolver = EffectResolver(gm)
                if ability.condition is not None and not resolver._check_condition(
                        p, ability.condition, card):
                    continue
                card.cost_buff += eff.value.base
