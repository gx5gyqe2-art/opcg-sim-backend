"""対話（interaction）解決・pending 要求生成・deferred 継続（GameManager からの移管・第1引数 gm）。"""
from __future__ import annotations

import json
import hashlib
import logging

from ..journal import JournaledList, JournaledSet
from ..rules_constants import FIELD_LIMIT
from ...models.models import CONST
from ...models.enums import Phase, Zone, CardType, TriggerType, PendingMessage
from ..effects.resolver import EffectResolver

_logger = logging.getLogger("opcg.engine")


def resolve_interaction(gm, player: Player, payload: Dict[str, Any]):
    if not gm.active_interaction:
        return
        
    continuation = gm.active_interaction.get("continuation")
    if not continuation:
        gm.active_interaction = None
        return

    action_type = gm.active_interaction.get("action_type")

    # 誘発能力の発動確認（CONFIRM_TRIGGER）: continuation は trigger_item のみで
    # source_card_uuid を持たないため、汎用 source 解決より先に処理する。
    if action_type == "CONFIRM_TRIGGER":
        item = continuation.get("trigger_item")
        accepted = payload.get("accepted")
        if accepted is None:
            if payload.get("skip") is True or payload.get("declined") is True:
                accepted = False
            else:
                accepted = payload.get("index", 0) == 0
        gm.active_interaction = None
        if item is not None:
            if accepted:
                item["_confirmed"] = True  # 先頭のまま再投入 → 解決へ
            elif item in gm._pending_triggers:
                gm._pending_triggers.remove(item)
        gm._advance_pending_triggers()
        # ターン開始時誘発の解決で保留していたリフレッシュフェイズ以降を再開する。
        if (not gm.active_interaction and not gm._pending_triggers
                and getattr(gm, "turn_start_pending", False)):
            gm.turn_start_pending = False
            gm.refresh_phase()
        if not gm.active_interaction and gm.active_battle \
                and gm.phase not in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER):
            gm._advance_battle_triggers()
        return

    # 場のキャラ上限超過の強制トラッシュ。発生源カードを持たない（ルール処理）ため、
    # 汎用 source 解決より先に処理する。選んだキャラをトラッシュ（KOではないので
    # 「KO時」誘発は起こさない）。
    if action_type == "FIELD_OVERFLOW_TRASH":
        owner = gm.p1 if gm.p1.name == continuation.get("owner_name") else gm.p2
        selected = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
        gm.active_interaction = None
        for uid in selected:
            card = next((c for c in owner.field if c.uuid == uid), None)
            if card:
                gm.move_card(card, Zone.TRASH, owner)
        gm.refresh_passive_state()
        # 複数体同時超過などでまだ超過していれば再度要求する（保険）。
        if len(owner.field) > FIELD_LIMIT:
            gm._suspend_for_field_overflow(owner)
        # 超過対話の背後に積まれた誘発（効果登場の【登場時】等）を消化する。この分岐は
        # 共通末尾（下の _advance_pending_triggers）を通らず return するため、ここで
        # 消化しないと次のアクション境界まで滞留する。
        if not gm.active_interaction and gm._pending_triggers:
            gm._advance_pending_triggers()
        return

    source_uuid = continuation["source_card_uuid"]
    source_card = gm._find_card_by_uuid(source_uuid)
    if not source_card:
        gm.active_interaction = None
        return

    resolver = EffectResolver(gm)
    
    if action_type == "SELECT_TARGET":
        selected_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
        
        selected_cards = []
        candidates = gm.active_interaction.get("candidates", [])
        for uid in selected_uuids:
            card = next((c for c in candidates if c.uuid == uid), None)
            if card: selected_cards.append(card)
        
        query = continuation.get("query")

        # ▼▼▼ 修正: save_idがなくても、一時的に選択結果を渡せるようにする ▼▼▼
        if "effect_context" in continuation:
            continuation["effect_context"]["temp_resolved_targets"] = selected_cards

        if query and getattr(query, 'save_id', None):
             continuation["effect_context"]["saved_targets"][query.save_id] = selected_cards
        
        gm.active_interaction = None
        resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

    elif action_type == "SELECT_RESOURCE":
        # ドン!!返却(RETURN_DON)の対象ドン!!選択。選んだ uuid を context に載せて再開すると、
        # RETURN_DON 再実行時に当該ドン!!を戻す。
        selected_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
        effect_context = continuation.get("effect_context", {})
        effect_context["_return_don_uuids"] = selected_uuids
        gm.active_interaction = None
        # RETURN_DON は効果の責任者（source_card の持ち主）視点で再実行する。
        # 「相手は自身の場のドン!!を戻す」（status=OPPONENT）では選択者＝相手だが、
        # _don_pool_player は player を基準に相手プールを引くため、応答者(相手)で再開すると
        # 相手の相手=自分のプールを指して空振りする。責任者基準なら選んだ相手ドンが正しく戻る。
        controller = gm.p1 if gm.p1.name == source_card.owner_id else gm.p2
        resolver.resume_execution(controller, source_card, continuation.get("execution_stack", []), effect_context)

    elif action_type == "CHOICE":
        selected_index = payload.get("index", payload.get("selected_option_index", 0))

        resolver.resume_choice(player, source_card, selected_index, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

    elif action_type == "CONFIRM_OPTIONAL":
        # 任意効果（「〜してもよい」）/ 任意コスト能力（「〜できる：」）の発動可否。
        # accepted=False（パス/拒否）ならスキップ。
        accepted = payload.get("accepted")
        if accepted is None:
            # selected_uuids 非空 / index>0 / skip フラグ等から推定（既定は発動=True）
            if payload.get("skip") is True or payload.get("declined") is True:
                accepted = False
            else:
                accepted = payload.get("index", 0) == 0
        # 任意バトルKO置換（A）の確認: accept→置換実行（本来のKOをスキップ）、
        # decline→本来のKOを実行。どちらも _finish_attack で戦闘後処理して return。
        if continuation.get("kind") == "BATTLE_KO_REPLACE":
            gm.active_interaction = None
            target = source_card
            target_owner = gm.p1 if gm.p1.name == continuation.get("target_owner_name") else gm.p2
            life_lost = continuation.get("life_lost", 0)
            if accepted and gm._active_replacement(target, ("BATTLE_KO",)):
                pass
            else:
                # 拒否、または置換が成立しなくなった場合は本来の KO を進める。
                gm.move_card(target, Zone.TRASH, target_owner)
                gm._resolve_on_ko(target, target_owner, cause="BATTLE")
            gm._finish_attack(target, target_owner, life_lost)
            return
        gm.active_interaction = None
        confirm_ability = continuation.get("confirm_ability")
        if confirm_ability is not None:
            # 任意コスト能力（A-3）の使用確認: accept で cost_confirmed=True で再入。
            # decline は何もしない（使用回数も未消費）。gamestate 経由で action_events を記録。
            if accepted:
                gm.resolve_ability(player, confirm_ability, source_card, cost_confirmed=True)
        else:
            optional_node = continuation.get("optional_node")
            resolver.resume_optional(player, source_card, bool(accepted), optional_node,
                                     continuation.get("execution_stack", []), continuation.get("effect_context", {}))

    elif action_type == "ARRANGE_DECK":
        # (2a)(2b) 並び替え/上下選択の確定。selected_uuids が配置順、position が上下。
        # ヘッドレス(drain)は selected_uuids=[] / position 無し → 現状順・fixed_position。
        ordered_uuids = payload.get("selected_uuids") or payload.get("extra", {}).get("selected_uuids", [])
        position = (payload.get("position") or payload.get("extra", {}).get("position")
                    or continuation.get("fixed_position", "BOTTOM"))
        position = "TOP" if str(position).upper() == "TOP" else "BOTTOM"
        cards = continuation.get("arrange_targets", [])
        if ordered_uuids:
            by_uuid = {c.uuid: c for c in cards}
            ordered = [by_uuid[u] for u in ordered_uuids if u in by_uuid]
            for c in cards:  # 指定漏れは元の順序で末尾に補う
                if c not in ordered:
                    ordered.append(c)
        else:
            ordered = list(cards)
        dest_kind = continuation.get("dest_kind", "DECK")
        gm.active_interaction = None
        if dest_kind == "LIFE":
            # ライフ並べ替え: ordered を新しいライフ順とする（life[0]=一番上）。
            owner_name = continuation.get("dest_owner")
            tp = gm.p1 if (owner_name and gm.p1.name == owner_name) else (gm.p2 if owner_name else player)
            rest = [c for c in tp.life if c not in ordered]
            tp.life = JournaledList(ordered + rest)
        else:
            # デッキ配置: BOTTOM は順に append（先頭が上）、TOP は逆順 insert(0) で
            # ordered[0] が最上面になるようにする。
            seq = ordered if position == "BOTTOM" else list(reversed(ordered))
            for c in seq:
                owner, _ = gm._find_card_location(c)
                if owner:
                    gm.move_card(c, Zone.DECK, owner, dest_position=position)
        resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), continuation.get("effect_context", {}))

    elif action_type == "DECLARE_COST":
        # C8: 宣言コストを記録し、相手デッキトップを公開して context に保存してから再開。
        declared = payload.get("declared_value", payload.get("index", 0))
        try:
            declared = int(declared)
        except (TypeError, ValueError):
            declared = 0
        effect_context = continuation.get("effect_context", {})
        effect_context["declared_cost"] = declared
        opponent = gm.p2 if player == gm.p1 else gm.p1
        revealed = opponent.deck[0] if opponent.deck else None
        if revealed is not None:
            effect_context["last_revealed_card"] = revealed
        else:
            pass
        gm.active_interaction = None
        resolver.resume_execution(player, source_card, continuation.get("execution_stack", []), effect_context)

    # 再開経路（resume_execution/resume_choice/resume_optional）で実行された
    # アクションも action_events へ記録する（resolve_ability 経由と同じ扱い。
    # 記録しないと中断を挟んだ効果が「何も実行していない」ように見える）。
    for ev in resolver.action_history:
        gm.action_events.append({
            "type": "EFFECT",
            "player": player.name,
            "card_name": source_card.master.name,
            "action": ev.get("action", ""),
            "targets": ev.get("targets", []),
            "value": ev.get("value"),
            "success": ev.get("success", True),
            **({"dest": ev["dest"]} if ev.get("dest") else {}),   # 移動系の行き先（additive）
        })

    if not gm.active_interaction and gm.setup_phase_pending:
        gm.finish_setup()
        gm.setup_phase_pending = False
        gm.phase = Phase.MULLIGAN
        gm.mulligan_done = JournaledSet()

    # ライフ公開【トリガー】/ON_LIFE_DECREASE 等のペンディング誘発が残っていれば消化する。
    if not gm.active_interaction and gm._pending_triggers:
        gm._advance_pending_triggers()

    # バトルトリガー(ON_ATTACK/ON_OPP_ATTACK)解決中の中断から復帰した場合:
    # バトルが進行中(active_battle あり)でまだ防御フェイズへ遷移していなければ、
    # 残りトリガーの解決＋フェイズ遷移を再開する（カウンター衝突エラーの防止）。
    if (not gm.active_interaction and gm.active_battle
            and gm.phase not in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER)):
        gm._advance_battle_triggers()

    # ターン開始時誘発の解決（対象選択/並び替え等の共通経路）で保留していた
    # リフレッシュフェイズ以降を再開する。
    if (not gm.active_interaction and not gm._pending_triggers
            and getattr(gm, "turn_start_pending", False)):
        gm.turn_start_pending = False
        gm.refresh_phase()

    # 入れ子の除去置換が中断したことで退避された外側継続（後続シーケンス／残対象）を、
    # 中断が解消された後に再開する（accepted limitation B = 多段継続の対話化）。
    # フィールド上限超過の処理より前に置き、継続完了後の最終盤面で超過判定する。
    if not gm.active_interaction and gm._deferred_continuations:
        gm._resume_deferred_continuations()

    # ON_PLAY 等の対話が片付いた後で場のキャラ上限超過が残っていれば強制トラッシュ。
    # 誘発/バトル進行（上記）を横取りしないよう最後に置き、1プレイヤーずつ逐次化する。
    if not gm.active_interaction:
        for pl in (gm.p1, gm.p2):
            if len(pl.field) > FIELD_LIMIT:
                gm._enforce_field_limit(pl)
                break

def get_pending_request(gm, with_request_id: bool = True) -> Optional[Dict[str, Any]]:
    # with_request_id=False: CPU 探索/自己対戦のドレイン経路など request_id を読まない呼び出し用の
    # 高速パス。request_id は**フロント専用**（入力側で未使用・下記 _rid コメント参照）なので、
    # 使わない側では正規化 JSON + sha1（候補 to_dict を含む要求全体のハッシュ）を丸ごと省く。
    # MCTS は _simulate ごとに _drain_own_interactions→get_pending_request を大量に呼ぶため、
    # 候補が多い盤面ではこのハッシュが CPU を占有し 1 バッチが病的に遅くなる（w3 実測: 通常 ~300s の
    # バッチが >13min 停滞）。既定 True で従来挙動・API 契約（test_api_contract の request_id 安定性）は不変。
    pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
    battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    KEY_ACTION = pending_props.get('ACTION', 'action')
    KEY_MSG = pending_props.get('MESSAGE', 'message')
    KEY_UUIDS = pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids')
    KEY_SKIP = pending_props.get('CAN_SKIP', 'can_skip')
    KEY_CANDIDATES = pending_props.get('CANDIDATES', 'candidates')
    KEY_CONSTRAINTS = pending_props.get('CONSTRAINTS', 'constraints')
    KEY_OPTIONS = pending_props.get('OPTIONS', 'options')

    def _rid(d: Dict[str, Any]) -> str:
        # request_id は「同一の要求なら安定・要求が変われば変化」する決定的ハッシュにする。
        # 従来は get のたびに uuid4 を再生成しており、フロントの『request_id 変化＝新要求』検知が
        # 毎ポーリング/WS更新で誤発火していた（機能バグ）。入力側で request_id は未使用＝安全に変更可。
        #
        # ハッシュは**要求の全内容（request_id 自身を除く）＋turn_count**を正規化 JSON で取る。
        # player_id/action/message/selectable_uuids だけでは、同一ターン内に連続する
        # 別要求（同名カード2枚が各々出す同文の確認、options だけ異なる CHOICE、
        # revealed view だけ異なる探索など）が衝突し、フロントが 2 件目を新要求と認識できず
        # モーダル再マウント/選択リセットが起きない。source_card_uuid・options・constraints・
        # candidates など識別に効く全フィールドを取り込むことでこれを防ぐ。
        # 盤面不変なら d は各 to_dict()/スカラが決定的なので同一 → rid も安定（＝元の修正意図を維持）。
        if not with_request_id:
            return ""  # 高速パス: 呼び出し側が request_id を読まない（フロント専用フィールド）ときは省略。
        payload = {k: v for k, v in d.items() if k != "request_id"}
        key = json.dumps([gm.turn_count, payload],
                         sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

    # マリガンは先行プレイヤー(turn_player)から順に要求する。
    if gm.phase == Phase.MULLIGAN:
        mulligan_order = ([gm.turn_player, gm.opponent]
                          if gm.turn_player and gm.opponent else [gm.p1, gm.p2])
        for player in mulligan_order:
            if player.name not in gm.mulligan_done:
                hand_candidates = [c.to_dict() for c in player.hand]
                _mreq = {
                    KEY_PID: player.name,
                    KEY_ACTION: "MULLIGAN",
                    KEY_MSG: "マリガンするカードを選んでください（交換なし＝キープ）",
                    KEY_CANDIDATES: hand_candidates,
                    KEY_UUIDS: [c.uuid for c in player.hand],
                    KEY_CONSTRAINTS: {"min": 0, "max": len(player.hand)},
                    KEY_SKIP: True,
                }
                _mreq["request_id"] = _rid(_mreq)
                return _mreq
        return None

    if gm.active_interaction:
        action_type = gm.active_interaction.get("action_type")
        fe_action = "SEARCH_AND_SELECT" if action_type in ("SELECT_TARGET", "FIELD_OVERFLOW_TRASH") else action_type
        
        candidates = gm.active_interaction.get("candidates", [])
        candidate_uuids = [c.uuid for c in candidates] if candidates else []
        # candidate_dicts（各候補の to_dict）は**フロント表示専用**（既定解決＝default_interaction_payload
        # は selectable_uuids/constraints しか読まない）。候補が多い盤面では c.to_dict() のリスト構築が
        # MCTS のドレイン経路で CPU を占有するため、request_id 不要の高速パスでは丸ごと省く。
        candidate_dicts = ([c.to_dict() for c in candidates] if candidates else []) if with_request_id else []
        
        req = {
            KEY_PID: gm.active_interaction.get("player_id"),
            KEY_ACTION: fe_action,
            KEY_MSG: gm.active_interaction.get("message", "選択してください"),
            KEY_UUIDS: gm.active_interaction.get("selectable_uuids", candidate_uuids),
            KEY_SKIP: gm.active_interaction.get("can_skip", False),
            KEY_CANDIDATES: candidate_dicts,
            KEY_CONSTRAINTS: gm.active_interaction.get("constraints"),
            "options": gm.active_interaction.get("options"),
        }
        # 効果の発生源カードを UI で表示できるよう uuid を併せて渡す。
        src_uuid = gm.active_interaction.get("source_card_uuid")
        if src_uuid:
            req[pending_props.get('SOURCE_CARD_UUID', 'source_card_uuid')] = src_uuid
        # ARRANGE_DECK(並び替え/上下選択)はフロントの UI 切替フラグを併せて渡す。
        if action_type == "ARRANGE_DECK":
            req["allow_position"] = gm.active_interaction.get("allow_position", False)
            req["allow_reorder"] = gm.active_interaction.get("allow_reorder", False)
        req["request_id"] = _rid(req)
        return req

    if not gm.active_battle and gm.phase in [Phase.BLOCK_STEP, Phase.BATTLE_COUNTER]:
        gm.phase = Phase.MAIN
        
    request = None
    ACT_BLOCKER = battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER')
    ACT_COUNTER = battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER')
    
    if gm.phase == Phase.BLOCK_STEP and gm.active_battle:
        target_owner = gm.active_battle["target_owner"]
        blockers = [c.uuid for c in target_owner.field if not c.is_rest and c.has_keyword("ブロッカー") and "CANNOT_REST" not in c.timed_flags]
        request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_BLOCKER, KEY_MSG: PendingMessage.SELECT_BLOCKER.value, KEY_UUIDS: blockers, KEY_SKIP: True}
    elif gm.phase == Phase.BATTLE_COUNTER and gm.active_battle:
        target_owner = gm.active_battle["target_owner"]
        # カウンター候補: (a) counter 値を持つカード（手札から捨てるだけ＝コスト不要）／
        # (b) COUNTER トリガのイベント。ただしイベントは発動コストを active ドン!! で払える分だけ
        # 提示する（MAIN_ACTION の PLAY と同じ支払可能性フィルタ）。払えないイベントを出すと
        # apply_counter → pay_cost で「ドン!!が不足」例外になる（合法手生成のバグ・要調査で発見）。
        _don_active = len(target_owner.don_active)
        counters = [c.uuid for c in target_owner.hand
                    if c.current_counter > 0
                    or (c.master.type == CardType.EVENT
                        and any(abil.trigger == TriggerType.COUNTER for abil in c.master.abilities)
                        and (c.master.cost or 0) <= _don_active)]
        request = {KEY_PID: target_owner.name, KEY_ACTION: ACT_COUNTER, KEY_MSG: PendingMessage.SELECT_COUNTER.value, KEY_UUIDS: counters, KEY_SKIP: True}
    elif gm.phase == Phase.MAIN:
        selectable = [c.uuid for c in gm.turn_player.hand]
        selectable += [c.uuid for c in gm.turn_player.field if not c.is_rest]
        if gm.turn_player.leader and not gm.turn_player.leader.is_rest:
            selectable.append(gm.turn_player.leader.uuid)
        request = {KEY_PID: gm.turn_player.name, KEY_ACTION: "MAIN_ACTION", KEY_MSG: PendingMessage.MAIN_ACTION.value, KEY_UUIDS: selectable, KEY_SKIP: True}
    if request is not None:
        request["request_id"] = _rid(request)
    return request

def default_interaction_payload(gm, pending: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """効果対話に対する「妥当な既定解決」のペイロードを構築する。

    本番（自己対戦/CPU）でも使える機械的な既定選択:
      - 必要最小数 (constraints.min) を満たすよう候補の先頭から選ぶ
      - can_skip なら 0 件選択（スキップ）も可だが、min>0 のときは min 件選ぶ
      - CHOICE/CONFIRM 系は index=0（最初の選択肢/発動する）
    AI（PR2）は本メソッドを評価関数で上書きして最良選択を選ぶ。
    """
    if pending is None:
        pending = gm.get_pending_request() or {}
    pending_props = CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_UUIDS = pending_props.get('SELECTABLE_UUIDS', 'selectable_uuids')
    KEY_CONSTRAINTS = pending_props.get('CONSTRAINTS', 'constraints')
    uuids = list(pending.get(KEY_UUIDS, []) or [])
    constraints = pending.get(KEY_CONSTRAINTS) or {}
    try:
        min_n = int(constraints.get("min", 0))
    except (TypeError, ValueError):
        min_n = 0
    try:
        max_n = int(constraints.get("max", len(uuids)))
    except (TypeError, ValueError):
        max_n = len(uuids)
    take = max(min_n, 0)
    take = min(take, max_n, len(uuids))
    selected = uuids[:take]
    return {
        "selected_uuids": selected,
        "index": 0,
        "accepted": True,
        "position": "BOTTOM",
        "declared_value": 0,
    }

def pending_actor_action(gm) -> Optional[Tuple[str, str]]:
    """`get_pending_request()` の (player_id, action) **だけ**を安価に返す（CPU 探索の葉/手番判定用）。

    探索は各ノードでこの 2 値しか見ない（手は `get_legal_actions` から得る）一方、
    `get_pending_request` は毎回 selectable 構築・候補 to_dict・request_id ハッシュ（要求全体の
    正規化 JSON を sha1）を作るため重い（探索コストの ~12%）。本メソッドは**判定ロジックと
    副作用（BLOCK_STEP/BATTLE_COUNTER で
    active_battle が無いときの phase→MAIN 正規化）を get_pending_request と一致**させたうえで、
    重い payload を作らない。一致は `tests/test_cpu_make_unmake.py` で機械照合する。
    """
    if gm.phase == Phase.MULLIGAN:
        order = ([gm.turn_player, gm.opponent]
                 if gm.turn_player and gm.opponent else [gm.p1, gm.p2])
        for p in order:
            if p.name not in gm.mulligan_done:
                return (p.name, "MULLIGAN")
        return None
    if gm.active_interaction:
        at = gm.active_interaction.get("action_type")
        fe = "SEARCH_AND_SELECT" if at in ("SELECT_TARGET", "FIELD_OVERFLOW_TRASH") else at
        return (gm.active_interaction.get("player_id"), fe)
    if not gm.active_battle and gm.phase in (Phase.BLOCK_STEP, Phase.BATTLE_COUNTER):
        gm.phase = Phase.MAIN  # get_pending_request と同じ副作用
    battle_actions = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    if gm.phase == Phase.BLOCK_STEP and gm.active_battle:
        return (gm.active_battle["target_owner"].name, battle_actions.get('SELECT_BLOCKER', 'SELECT_BLOCKER'))
    if gm.phase == Phase.BATTLE_COUNTER and gm.active_battle:
        return (gm.active_battle["target_owner"].name, battle_actions.get('SELECT_COUNTER', 'SELECT_COUNTER'))
    if gm.phase == Phase.MAIN:
        return (gm.turn_player.name, "MAIN_ACTION")
    return None

def _defer_resolver_stack(gm, player: Player, source_card, execution_stack, context) -> None:
    """除去置換の中断で失われる外側リゾルバの後続（execution_stack）を退避する（B1）。"""
    gm._deferred_continuations.insert(0, {
        "kind": "RESOLVER_STACK",
        "player_name": player.name,
        "source_card_uuid": source_card.uuid if source_card else None,
        "execution_stack": JournaledList(execution_stack),
        "effect_context": context,
    })

def _defer_removal_targets(gm, player: Player, action, remaining_targets, value) -> None:
    """複数対象除去で置換中断したとき、未処理の残対象を退避する（B2）。
    再開時に apply_action_to_engine を残対象で再実行する（uuid で解決し直す）。"""
    gm._deferred_continuations.append({
        "kind": "REMOVAL_TARGETS",
        "player_name": player.name,
        "action": action,
        "remaining_target_uuids": [t.uuid for t in remaining_targets],
        "value": value,
    })

def _resume_deferred_continuations(gm, limit: int = 64) -> None:
    """中断が無くなった後、退避した外側継続を LIFO で再開する。
    再開した継続が新たな中断を生んだら、active_interaction が立ってループは止まる
    （残りは次の解決後に再びここで処理される）。"""
    n = 0
    while not gm.active_interaction and gm._deferred_continuations and n < limit:
        frame = gm._deferred_continuations.pop()
        kind = frame.get("kind")
        player = gm.p1 if gm.p1.name == frame.get("player_name") else gm.p2
        try:
            if kind == "RESOLVER_STACK":
                src = frame.get("source_card_uuid")
                source_card = gm._find_card_by_uuid(src) if src else None
                resolver = EffectResolver(gm)
                resolver.resume_execution(player, source_card,
                                          frame.get("execution_stack", []),
                                          frame.get("effect_context", {}))
            elif kind == "REMOVAL_TARGETS":
                remaining = [c for c in (gm._find_card_by_uuid(u) for u in frame.get("remaining_target_uuids", [])) if c]
                if remaining:
                    gm.apply_action_to_engine(player, frame.get("action"), remaining, frame.get("value"))
        except Exception as e:
            # 退避した継続フレームの1件が再開に失敗しても、残りのフレームの再開は続行する
            # （置換中断の解決後に外側継続をまとめて再開する経路。1件破綻で全体を止めない）。診断のみ残す。
            _logger.debug("deferred 継続フレームの再開で1件失敗（続行）: %r", e, exc_info=True)
        n += 1
