"""除去保護・置換効果・自己制限のガード判定（GameManager からの移管・ステートレス。第1引数 gm）。"""
from __future__ import annotations

import re

from ...models.enums import TriggerType, ActionType, CardType
from ..effects.resolver import EffectResolver
from ..effects.matcher import get_target_cards
from ._helpers import _nfc, _ability_turn_limit, _ability_index


def _has_rested_play(gm, player: Player) -> bool:
    """player が「自分のキャラはレストで登場する」PASSIVE を持つか（RESTED_PLAY マーカー）。"""
    cards = ([player.leader] if player.leader else []) + list(player.field)
    for c in cards:
        if not c or getattr(c, "is_effect_negated", False) or not getattr(c, "master", None):
            continue
        for ab in c.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            act = gm._find_action(ab.effect, ActionType.RESTRICTION)
            if act is not None and getattr(act, "status", None) == "RESTED_PLAY":
                return True
    return False

def _active_restriction(gm, player: Player, key: str) -> Optional[Dict[str, Any]]:
    """player に有効な自己制限（self_cannot）があれば、そのパラメータ dict を返す。
    turn_count <= expire の間だけ有効。期限切れエントリは掃除して None を返す。"""
    rec = getattr(player, "restrictions", {}).get(key)
    if not rec:
        return None
    if gm.turn_count <= rec.get("expire", -1):
        return rec
    # 期限切れは破棄
    player.restrictions.pop(key, None)
    return None

def _blocks_effect_play(gm, card: CardInstance) -> bool:
    """card が「手札のこのカードは効果で登場できない」PASSIVE を持つか（NO_EFFECT_PLAY）。"""
    if not card or not getattr(card, "master", None):
        return False
    for ab in card.master.abilities:
        if ab.trigger != TriggerType.PASSIVE:
            continue
        act = gm._find_action(ab.effect, ActionType.RESTRICTION)
        if act is not None and getattr(act, "status", None) == "NO_EFFECT_PLAY":
            return True
    return False

def _active_protection(gm, card: CardInstance, status_values: Tuple[str, ...], actor: Optional[Player] = None, attacker: Optional[CardInstance] = None) -> bool:
    if not card or not getattr(card, "master", None) or card.negated:
        return False
    owner = gm.p1 if gm.p1.name == card.owner_id else gm.p2

    # トリガー効果が継続効果として付与した期間付き保護（timed_flags）。
    # 例: 「このキャラは、次の自分のターン開始時まで、バトルでKOされない」(ON_ATTACK)
    for s in status_values:
        if f"PREVENT_{s}" in (card.flags | card.timed_flags):
            return True

    resolver = None
    # 走査対象: 自身に加え、オーナーのリーダー/フィールド/ステージの範囲保護
    # （「自分の特徴《X》を持つキャラすべては…場を離れない」等。従来は自身のみ走査で
    #   他カードを守る保護が機能しなかった）。
    protectors = [card]
    if owner.leader and owner.leader is not card:
        protectors.append(owner.leader)
    protectors.extend(fc for fc in owner.field if fc is not card)
    if getattr(owner, "stage", None) and owner.stage is not card:
        protectors.append(owner.stage)
    # 除去を行うプレイヤー(actor)側の範囲保護も走査する。「相手のキャラすべては、自分の効果で
    # 場を離れない」(OP14-079) のように、保護能力の持ち主(actor)と被保護カード(owner=相手)が
    # 別プレイヤーのケースに対応する。range クエリの照合で actor の相手側＝owner 側のカードのみ
    # 一致するため、自分側を守る保護は誤適用されない。
    if actor is not None and actor is not owner:
        if actor.leader and actor.leader is not card:
            protectors.append(actor.leader)
        protectors.extend(fc for fc in actor.field if fc is not card)
        if getattr(actor, "stage", None) and actor.stage is not card:
            protectors.append(actor.stage)

    for protector in protectors:
        if getattr(protector, "is_effect_negated", False) or getattr(protector, "negated", False):
            continue
        for ab in protector.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            eff = gm._find_action(ab.effect, ActionType.PREVENT_LEAVE)
            if eff is None:
                continue
            if eff.status not in status_values:
                continue
            # 保護対象クエリの照合: SOURCE は protector 自身のみを守る。
            # 範囲クエリは card が範囲に含まれるかを実体化して確認する。
            tgt = getattr(eff, "target", None)
            if tgt is None or getattr(tgt, "select_mode", "SOURCE") == "SOURCE":
                if protector is not card:
                    continue
            else:
                if card not in get_target_cards(gm, tgt, protector):
                    continue
            if ab.condition is not None:
                if resolver is None:
                    resolver = EffectResolver(gm)
                src = card if protector is card else protector
                if not resolver._check_condition(owner, ab.condition, src):
                    continue
            # 属性限定のバトルKO耐性（「属性《斬》を持つカードとのバトルでKOされず」OP08-114）。
            # 保護はバトル相手(attacker)が指定属性を持つ場合のみ有効。属性が不明（非バトル経路）や
            # 不一致なら適用しない。
            attr_m = re.search(_nfc(r'属性[(（《]([斬打射特知])[)）》]を持つ(?:カード|キャラ)?との(?:バトル|戦闘)'),
                               getattr(eff, "raw_text", "") or "")
            if attr_m:
                req_attr = attr_m.group(1)
                if attacker is None or getattr(attacker.master, "attribute", None) is None \
                        or attacker.master.attribute.value != req_attr:
                    continue
            # 【ターン1回】保護（例:「このキャラはターンに1回、相手の効果でKOされない」）は
            # 1ターンに1回まで。_check_condition の TURN_LIMIT は常時 True を返すため、ここで
            # 使用回数を直接 enforce する（resolve_ability を経由しない保護経路のため）。
            limit = _ability_turn_limit(ab)
            if limit is not None:
                key = _ability_index(protector, ab)
                if protector.ability_used_this_turn.get(key, 0) >= limit:
                    continue
                protector.ability_used_this_turn[key] = protector.ability_used_this_turn.get(key, 0) + 1
            return True
    return False

def _find_replacement(gm, card: CardInstance, status_values: Tuple[str, ...]):
    if not card or not getattr(card, "master", None) or card.negated:
        return None
    owner = gm.p1 if gm.p1.name == card.owner_id else gm.p2

    # 走査対象: 除去されるカード自身 → オーナーのリーダー → フィールドの他キャラ
    # （自身の置換効果と、他キャラを守る OPPONENT_REMOVAL 型置換効果の両方をカバー）
    candidates = [card]
    if owner.leader and owner.leader is not card:
        candidates.append(owner.leader)
    for fc in owner.field:
        if fc is not card:
            candidates.append(fc)

    for protector in candidates:
        if getattr(protector, 'is_effect_negated', False):
            continue
        for ab in protector.master.abilities:
            if ab.trigger != TriggerType.PASSIVE:
                continue
            eff = gm._find_action(ab.effect, ActionType.REPLACE_EFFECT)
            if eff is None:
                continue
            if eff.status not in status_values:
                continue
            # 自己無効化（「キャラの「X」がいる場合、この効果は無効になる」OP05-100）。
            # 指定名のキャラがいずれかの場にいれば、この置換は発動しない。
            neg_m = re.search(_nfc(r'「([^」]+)」がい[るて][^。]*?この効果は無効'),
                              getattr(eff, "raw_text", "") or "")
            if neg_m:
                neg_name = neg_m.group(1)
                if any(c.master.matches_name(neg_name, partial=True)
                       for pl in (gm.p1, gm.p2) for c in pl.field):
                    continue
            sub = getattr(eff, "sub_effect", None)
            if sub is None:
                continue
            resolver = EffectResolver(gm)
            # 条件チェック: source_card=除去されるカード（OPPONENT_REMOVAL/名称フィルタ評価用）、
            # host_card=保護者（HAS_DON 等の「能力保持カードの付与ドン」評価用。OP05-001 はリーダー）。
            if ab.condition is not None and not resolver._check_condition(owner, ab.condition, card, host_card=protector):
                continue
            # 代わりの行動が取れない場合は置換不成立。
            # sub_effect の source は「離れるカード」(card) とする。置換文の「代わりに
            # （そのカードを）〜」は離れるカード自身を対象に取り得るため（OP11-101 の
            # 「代わりに自分のライフの上に裏向きで加える」= 離れるカードをライフへ）。
            if not resolver._can_satisfy_node(owner, sub, card):
                continue
            # 【ターン1回】置換は1ターンに1回まで enforce する（parser が自己置換の TURN_LIMIT を
            # 落とすため raw_text 併用。resolve_ability を経由しない置換経路のため直接管理）。
            _limit = _ability_turn_limit(ab)
            if _limit is not None and protector.ability_used_this_turn.get(_ability_index(protector, ab), 0) >= _limit:
                continue
            return (protector, ab, eff, sub)

    # 継続付与型の置換（EB02-030「自分のキャラすべては、このターン中、…代わりに〜できる」）。
    # 場に残らないイベント由来のため owner.granted_replacements を参照する。付与対象は
    # 自分のキャラ（除去されるカード自身が自分のキャラであること）。ab/eff は持たないので
    # ターン制限・条件は付与時に消化済みとして None を返す。
    if getattr(card, "master", None) and card.master.type == CardType.CHARACTER:
        for g in getattr(owner, "granted_replacements", []):
            if g.get("status") not in status_values:
                continue
            if gm.turn_count > g.get("expire_turn", 0):
                continue
            sub = g.get("sub_effect")
            if sub is None:
                continue
            resolver = EffectResolver(gm)
            if not resolver._can_satisfy_node(owner, sub, card):
                continue
            return (card, None, None, sub)
    return None

def _register_granted_replacements(gm, player: Player, source_card: Card) -> None:
    """カウンターイベント等が持つ REPLACE_EFFECT を「このターン中」付与の置換として登録する。
    イベントは場に残らないため、被除去キャラ側から参照できるよう player へ退避する
    （EB02-030「自分のキャラすべては、このターン中、バトルでKOされる場合、代わりに〜できる」）。"""
    import copy
    for ability in source_card.master.abilities:
        eff = gm._find_action(ability.effect, ActionType.REPLACE_EFFECT)
        if eff is None:
            continue
        sub = getattr(eff, "sub_effect", None)
        if sub is None:
            continue
        # 「〜できる／〜してもよい」は任意。parser が sub.is_optional に載せ切れない場合に
        # 備え raw_text からも判定し、付与する sub のコピーへ反映する（共有ノードを汚さない）。
        raw = _nfc(getattr(eff, "raw_text", "") or getattr(ability, "raw_text", "") or "")
        is_optional = bool(getattr(sub, "is_optional", False)) or ("できる" in raw) or ("てもよい" in raw)
        sub_copy = copy.copy(sub)
        sub_copy.is_optional = is_optional
        player.granted_replacements.append({
            "status": eff.status,
            "sub_effect": sub_copy,
            "is_optional": is_optional,
            "expire_turn": gm.turn_count,
        })
    return None

def _active_replacement(gm, card: CardInstance, status_values: Tuple[str, ...],
                        can_suspend: bool = False) -> bool:
    found = gm._find_replacement(card, status_values)
    if found is None:
        return False
    owner = gm.p1 if gm.p1.name == card.owner_id else gm.p2
    protector, ab, eff, sub = found
    resolver = EffectResolver(gm)
    _limit = _ability_turn_limit(ab)
    # 置換は除去解決の最中に発生する「入れ子の中断」。失われる外側継続が無い
    # （can_suspend=除去アクションの後続が空・単一対象）場合は、内側の中断
    # （対象選択／任意確認）を**そのまま UI へ提示**し、被保護側に選ばせる
    # （interaction はスタックに残し、resume で sub_effect を完了させる）。
    # 外側継続が残るケースでは従来どおり保守的に自動解決して同期完了させる
    # （任意=採用、対象=有効候補先頭）。どちらも置換は成立扱い（本来の除去はスキップ）。
    outer_interaction = gm.active_interaction
    gm.active_interaction = None
    resolver.execution_stack = [sub]
    resolver._process_stack(owner, card)
    suspended = gm.active_interaction is not None
    if _limit is not None:  # 発動成立 → 【ターン1回】の使用回数を消費
        k = _ability_index(protector, ab)
        protector.ability_used_this_turn[k] = protector.ability_used_this_turn.get(k, 0) + 1
    if suspended and can_suspend:
        # 内側中断を UI へ提示（自動解決しない）。除去はスキップ＝置換成立。
        # 外側に後続継続（このシーケンスの残アクション／除去ループの残対象）があれば、
        # 呼び出し側がそれを deferred フレームへ退避できるようシグナルを立てる（B）。
        gm._replacement_suspended = True
        return True
    # 自動解決パス（外側継続あり、または中断なし）。
    gm._auto_resolve_replacement(owner)
    gm.active_interaction = outer_interaction
    return True

def _auto_resolve_replacement(gm, owner: Player, limit: int = 16) -> None:
    """置換 sub_effect が残した中断（任意確認／対象選択）を保守的に同期解決する。

    単一 continuation 設計ではネストした中断を UI へ伝播できないため、置換は headless で
    完結させる: 任意確認(CONFIRM_OPTIONAL)は accept（保護を実行）、対象選択(SELECT_TARGET)は
    有効候補から必要数を自動選択する。選択 UI のフロント連携は E14/E15 の将来課題。"""
    n = 0
    while gm.active_interaction and n < limit:
        ia = gm.active_interaction
        atype = ia.get("action_type")
        pid = ia.get("player_id")
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        if atype == "SELECT_TARGET":
            cand = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            mx = (ia.get("constraints") or {}).get("max", 1) or 1
            payload = {"selected_uuids": cand[:mx], "index": 0}
        elif atype == "CONFIRM_OPTIONAL":
            payload = {"accepted": True}
        elif atype == "CHOICE":
            payload = {"index": 0}
        else:
            # 想定外の中断種別は安全側に倒して打ち切る（宙吊り防止のため interaction を解除）。
            gm.active_interaction = None
            break
        try:
            gm.resolve_interaction(actor, payload)
        except Exception as e:
            gm.active_interaction = None
            break
        n += 1
