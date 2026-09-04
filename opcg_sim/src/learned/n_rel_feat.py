"""n_rel_feat: NRel（N系 v3「対と手順」本体）の符号化（P0・2026-09-04・`docs/n_attention_plan.md` §2）。

現行の符号化（`encoder.encode` v12＝スカラー 94・card_idx 22 枠）に対して、次の 3 つを**足す**
（既存の列は 1 bit も変えない＝append-only・v12 の教材と温スタートを壊さない）:

  1. **トークン状態 S**（22 枠 × S_DIM）: 盤面依存の「今の値」。現在パワー/コスト・付与ドン・
     攻撃可否・ブロッカー可否・カウンター値・今出せるか・戻すドン数・**能力ごとの条件の充足
     （エンジンの `_check_condition` で評価した真偽）**・KO 時/アタック時/相手アタック時の能力・
     相手の未見プールからの脅威（次ターンに届く除去の割合）。
  2. **関係 R**（自 16 枠 × 相手 6 枠、自 16 × 自 16 × R_DIM）: 「i の効果が j に**今届くか・
     あと何点か**」をエンジンが計算して与える（ネットに引き算を学ばせない）。自×自は
     「i の減算で k のしきい値が届く」＝**組**（例: ガンマナイフ × 神の裁き）。
  3. **グローバル追加列**（EXTRA_DIM）: 起動の未使用・加速可能枚数・残りの攻撃可能数・速攻・
     ライフ圧・自デッキ残の役割別枚数・**相手の未見プール**（デッキリスト − 見えたカード＝
     手札∪山札∪伏せライフ。人間が「環境デッキの中身」として知っている範囲）の役割と脅威・
     リーダーパワーの現在/見込み・次ターンのドン・次ターンに出せる最大の札・守りの単価。
     `encoder.encode(version=13)` はこの列を v12 の末尾に付ける。

設計原則（`docs/n_attention_plan.md` §1）: 関係は符号化側で計算・条件の充足はエンジンの真偽・
ID 非依存（カード ID・名前・特性を列にしない）・エンジンと木は変えない。

公平性: 相手の手札の**中身**は使わない。未見プール（手札∪山札∪伏せライフの合計）は人間が
デッキリストから知りうる範囲（ユーザ決定 2026-09-04）。相手の山札の順・実際の手札は与えない。

コスト: カードごとの**効果プロファイル**（しきい値・減算量・戻すドン・加速・トリガー）は
マスター単位でキャッシュする（盤面に依らない）。盤面依存の計算は 22 枠の小さなループと
16×6／16×16 の関係だけ。条件評価（`_check_condition`）は条件を持つ能力にだけ呼ぶ。
"""
from __future__ import annotations

import numpy as np

from opcg_sim.src.models.effect_types import (
    ActionType, TriggerType, Branch, Choice, GameAction, Sequence)
from opcg_sim.src.models.effect_types import Player as _EPlayer

MAX_FIELD = 5
MAX_HAND = 10
N_OWN = 1 + MAX_FIELD + MAX_HAND          # 16: 自L・自場5・手札10
N_OPP = 1 + MAX_FIELD                     # 6: 相手L・相手場5
N_TOK = N_OWN + N_OPP                     # 22（card_idx の先頭 22 枠と同じ並び: 自L,相L,自場5,相場5,手札10）

# ---- トークン状態 S の列 ----
S_COLS = ("power_now", "cost_now", "attached_don", "is_rest", "is_sick", "can_attack_now",
          "is_blocker_active", "counter_value", "playable_now", "don_return_cost",
          "cond_ok0", "cond_ok1", "cond_ok2", "cond_ok3",
          "trig_ko", "trig_attack", "trig_opp_attack", "threat_next", "is_char", "is_event")
S_DIM = len(S_COLS)                        # 20
MAX_AB = 4

# ---- 関係 R の列 ----
R_COLS = ("atk_margin", "ko_gap", "cost_gap", "red_amount", "feasible")
R_DIM = len(R_COLS)                        # 5
GAP_SAT = 1.5                              # 「届かない/該当なし」の飽和値

# ---- グローバル追加列（v13 = v12 + EXTRA_DIM） ----
ROLES = ("removal", "reduction", "lock", "draw", "counter", "blocker", "big")
EXTRA_COLS = (("leader_act_avail", "don_addable", "attackers_left", "rush_in_hand",
               "opp_counters_in_trash", "life_pressure")
              + tuple(f"deck_{r}" for r in ROLES)
              + tuple(f"opp_pool_{r}" for r in ROLES)
              + ("opp_pool_max_power", "opp_pool_big", "opp_pool_counter_total", "opp_pool_blockers",
                 "leader_power_now", "leader_power_max", "don_next_turn", "max_play_next_turn",
                 "guard_per_card"))
EXTRA_DIM = len(EXTRA_COLS)                # 29

_REMOVAL_OPS = {ActionType.KO, ActionType.DECK_BOTTOM, ActionType.MOVE_TO_HAND, ActionType.TRASH,
                ActionType.MOVE_CARD, ActionType.DECK_TOP}
_LOCK_OPS = {ActionType.REST, ActionType.FREEZE, ActionType.LOCK, ActionType.ATTACK_DISABLE,
             ActionType.PREVENT_REST}
_RED_OPS = {ActionType.BP_BUFF, ActionType.DEBUFF, getattr(ActionType, "BUFF", ActionType.BP_BUFF)}
_RAMP_OPS = {ActionType.RAMP_DON, ActionType.ACTIVE_DON}
_BLOCKER_KW = "ブロッカー"
_RUSH_KW = "速攻"
_BIG_POWER = 7000
_BIG_COST = 7


# ---------------------------------------------------------------------------
# 効果プロファイル（マスター単位・キャッシュ）
# ---------------------------------------------------------------------------
def _walk(node, out):
    if node is None:
        return
    if isinstance(node, GameAction):
        out.append(node)
        _walk(getattr(node, "sub_effect", None), out)
    elif isinstance(node, (Sequence, Branch, Choice)):
        for ch in (getattr(node, "actions", None) or getattr(node, "options", None) or []):
            _walk(ch, out)
        for at in ("if_true", "if_false"):
            _walk(getattr(node, at, None), out)


def _is_opp(tgt):
    return tgt is not None and getattr(getattr(tgt, "player", None), "name", "SELF") == "OPPONENT"


def _base(v):
    try:
        return float(getattr(v, "base", 0) or 0)
    except Exception:
        return 0.0


_PROFILES: dict = {}


def profile(master):
    """マスター → 効果プロファイル（盤面非依存・キャッシュ）。

    thr: [(power_max|None, cost_max|None, needs_rest, kind)] kind∈{'removal','lock'}・相手対象の
         しきい値効果（しきい値が無い「1枚を KO」は (None, None)＝全てに届く）
    red: 相手キャラへの減算量の最大（正の数・パワー単位）
    ret_don: 能力コストで戻すドン枚数の最大（1c 登場ドロー・0c イベントの「ドン−1」）
    ramp: 自分のドン追加量（登場時/起動）
    trig: トリガー名の集合・counter_event: カウンター時の +パワー（両用イベント）
    """
    key = getattr(master, "card_id", None) or id(master)
    hit = _PROFILES.get(key)
    if hit is not None:
        return hit
    thr, red, ret_don, ramp, cev = [], 0.0, 0.0, 0.0, 0.0
    trig = set()
    for ab in (getattr(master, "abilities", ()) or ()):
        tname = getattr(getattr(ab, "trigger", None), "name", "UNKNOWN")
        trig.add(tname)
        acts = []
        _walk(getattr(ab, "effect", None), acts)
        for a in acts:
            t = getattr(a, "type", None)
            tgt = getattr(a, "target", None)
            if t in _REMOVAL_OPS or t in _LOCK_OPS:
                if _is_opp(tgt):
                    ctypes = [str(x).upper() for x in (getattr(tgt, "card_type", None) or [])]
                    allow_leader = (not ctypes) or any("LEADER" in x for x in ctypes)
                    thr.append((getattr(tgt, "power_max", None), getattr(tgt, "cost_max", None),
                                bool(getattr(tgt, "is_rest", None)),
                                "removal" if t in _REMOVAL_OPS else "lock", allow_leader))
            elif t in _RED_OPS:
                amt = _base(getattr(a, "value", None))
                if _is_opp(tgt) and amt < 0:
                    red = max(red, -amt)
                elif tname == "COUNTER" and amt > 0 and not _is_opp(tgt):
                    cev = max(cev, amt)
            elif t in _RAMP_OPS and not _is_opp(tgt):
                ramp += max(1.0, _base(getattr(a, "value", None)))
        costs = []
        _walk(getattr(ab, "cost", None), costs)
        for c in costs:
            if getattr(c, "type", None) == ActionType.RETURN_DON:
                ret_don = max(ret_don, max(1.0, _base(getattr(c, "value", None))))
    kws = set(getattr(master, "keywords", ()) or ())
    prof = {"thr": tuple(thr), "red": red, "ret_don": ret_don, "ramp": ramp, "trig": frozenset(trig),
            "counter_event": cev, "blocker": _BLOCKER_KW in kws, "rush": _RUSH_KW in kws,
            "has_draw": any(getattr(a, "type", None) == ActionType.DRAW
                            for ab in (getattr(master, "abilities", ()) or ())
                            for a in _walk_list(getattr(ab, "effect", None)))}
    _PROFILES[key] = prof
    return prof


def _walk_list(node):
    out = []
    _walk(node, out)
    return out


def roles_of(master):
    """役割ベクトル（ROLES の順・0/1）。デッキ残・未見プールの集計に使う。"""
    p = profile(master)
    t = getattr(getattr(master, "type", None), "name", "")
    return np.array([
        1.0 if any(th[3] == "removal" for th in p["thr"]) else 0.0,
        1.0 if p["red"] > 0 else 0.0,
        1.0 if any(th[3] == "lock" for th in p["thr"]) else 0.0,
        1.0 if p["has_draw"] else 0.0,
        1.0 if ((getattr(master, "counter", 0) or 0) > 0 or p["counter_event"] > 0) else 0.0,
        1.0 if p["blocker"] else 0.0,
        1.0 if (t == "CHARACTER" and ((getattr(master, "power", 0) or 0) >= _BIG_POWER
                                      or (getattr(master, "cost", 0) or 0) >= _BIG_COST)) else 0.0,
    ], np.float32)


def _reach(thr, power, cost, is_rest, is_leader=False):
    """しきい値効果 thr が (power, cost, is_rest) の対象に届くか。届く=True と「差」を返す。

    差 = max(power−P, (cost−C)×1000) を 1e4 で割った量（≤0 なら届く）。しきい値が無い効果は −1。
    対象がリーダーで、効果の対象種別がキャラ限定なら届かない（KO/レストはリーダーに撃てない）。"""
    pm, cm, needs_rest, _kind, allow_leader = thr
    if is_leader and not allow_leader:
        return False, GAP_SAT
    if needs_rest and not is_rest:
        return False, GAP_SAT
    gaps = []
    if pm is not None:
        gaps.append((float(power) - float(pm)) / 10000.0)
    if cm is not None:
        gaps.append((float(cost) - float(cm)) * 0.1)
    if not gaps:
        return True, -1.0
    g = max(gaps)
    return g <= 0.0, g


# ---------------------------------------------------------------------------
# 盤面依存の計算
# ---------------------------------------------------------------------------
def _pl(manager, name):
    return manager.p1 if manager.p1.name == name else manager.p2


def _slots(me, opp):
    """22 枠のカード列（None＝空枠）。並びは card_idx と同じ。"""
    own_field = (list(me.field)[:MAX_FIELD] + [None] * MAX_FIELD)[:MAX_FIELD]
    opp_field = (list(opp.field)[:MAX_FIELD] + [None] * MAX_FIELD)[:MAX_FIELD]
    hand = (list(me.hand)[:MAX_HAND] + [None] * MAX_HAND)[:MAX_HAND]
    return [me.leader, opp.leader] + own_field + opp_field + hand


def _zone(i):
    if i == 0:
        return "own_leader"
    if i == 1:
        return "opp_leader"
    if i < 2 + MAX_FIELD:
        return "own_field"
    if i < 2 + 2 * MAX_FIELD:
        return "opp_field"
    return "hand"


def _own_index(i):
    """22 枠 index → 自トークン index（0..15）。自L=0・自場=1..5・手札=6..15。"""
    if i == 0:
        return 0
    if 2 <= i < 2 + MAX_FIELD:
        return 1 + (i - 2)
    return 6 + (i - (2 + 2 * MAX_FIELD))


def _opp_index(i):
    """22 枠 index → 相手トークン index（0..5）。相L=0・相場=1..5。"""
    return 0 if i == 1 else 1 + (i - (2 + MAX_FIELD))


def _power(c, my_turn):
    try:
        return int(c.get_power(my_turn))
    except Exception:
        return int(getattr(getattr(c, "master", None), "power", 0) or 0)


def _cost(c):
    try:
        return int(c.current_cost)
    except Exception:
        return int(getattr(getattr(c, "master", None), "cost", 0) or 0)


def _tname(c):
    return getattr(getattr(getattr(c, "master", None), "type", None), "name", "")


def _has_kw(c, kw):
    try:
        return bool(c.has_keyword(kw))
    except Exception:
        return kw in (getattr(getattr(c, "master", None), "keywords", ()) or ())


def _cond_flags(manager, player, card):
    """能力 k（≤4）の条件が今真か（条件無し＝1）。評価はエンジンの `_check_condition`。"""
    out = np.ones(MAX_AB, np.float32)
    abs_ = list(getattr(getattr(card, "master", None), "abilities", ()) or ())[:MAX_AB]
    if not any(getattr(ab, "condition", None) is not None for ab in abs_):
        return out
    try:
        from opcg_sim.src.core.effects.resolver import EffectResolver
        res = EffectResolver(manager)
    except Exception:
        return out
    for k, ab in enumerate(abs_):
        cond = getattr(ab, "condition", None)
        if cond is None:
            continue
        try:
            out[k] = 1.0 if res._check_condition(player, cond, card, card) else 0.0
        except Exception:
            out[k] = 1.0
    return out


def _leader_act_avail(manager, me):
    """リーダー起動（ACTIVATE_MAIN）が今の合法手にあるか＝未使用かつ条件/コストを満たす。"""
    if me.leader is None or getattr(manager, "turn_player", None) is not me:
        return 0.0
    try:
        pa = manager.pending_actor_action()
        if not pa or pa[0] != me.name or pa[1] != "MAIN_ACTION":
            return 0.0
        lu = getattr(me.leader, "uuid", None)
        for a in manager.get_legal_actions(me):
            if a.get("action_type") == "ACTIVATE_MAIN":
                p = a.get("payload") or {}
                if (a.get("card_uuid") or p.get("uuid") or p.get("card_uuid")) == lu:
                    return 1.0
    except Exception:
        return 0.0
    return 0.0


def _unseen_pool(opp):
    pool = list(getattr(opp, "hand", ()) or ()) + list(getattr(opp, "deck", ()) or ())
    pool += [c for c in (getattr(opp, "life", ()) or ()) if not getattr(c, "is_face_up", False)]
    return pool


def encode_rel(manager, me_name):
    """盤面 → {"tokens": [22,S_DIM], "rel_om": [16,6,R_DIM], "rel_oo": [16,16,R_DIM], "extra": [EXTRA_DIM]}。"""
    me, opp = _pl(manager, me_name), (manager.p2 if manager.p1.name == me_name else manager.p1)
    my_turn = getattr(manager, "turn_player", me) is me
    slots = _slots(me, opp)
    tok = np.zeros((N_TOK, S_DIM), np.float32)
    n_active = len(getattr(me, "don_active", ()) or ())
    don_total_opp = (len(getattr(opp, "don_active", ()) or ()) + len(getattr(opp, "don_rested", ()) or ())
                     + len(getattr(opp, "don_attached_cards", ()) or ()))
    don_next_opp = min(10, don_total_opp + min(1, len(getattr(opp, "don_deck", ()) or ())))
    pool = _unseen_pool(opp)
    pool_thr = []          # (thr, card_cost) — 相手プールの除去/減算が次ターン届くかの判定に使う
    for c in pool:
        m = getattr(c, "master", None)
        if m is None:
            continue
        p = profile(m)
        cc = int(getattr(m, "cost", 0) or 0)
        for th in p["thr"]:
            pool_thr.append((th, cc, p["red"]))
    n_pool = max(1, len(pool))

    pw = [0] * N_TOK
    cs = [0] * N_TOK
    for i, c in enumerate(slots):
        if c is None:
            continue
        z = _zone(i)
        owner_turn = my_turn if z in ("own_leader", "own_field", "hand") else (not my_turn)
        m = getattr(c, "master", None)
        p = profile(m) if m is not None else None
        t = _tname(c)
        pw[i] = _power(c, owner_turn) if t in ("LEADER", "CHARACTER") else 0
        cs[i] = _cost(c)
        on_board = z in ("own_leader", "own_field", "opp_leader", "opp_field")
        sick = bool(getattr(c, "is_newly_played", False)) and not _has_kw(c, _RUSH_KW)
        rest = bool(getattr(c, "is_rest", False))
        tok[i, 0] = pw[i] / 10000.0
        tok[i, 1] = cs[i] / 10.0
        tok[i, 2] = float(getattr(c, "attached_don", 0) or 0) / 5.0
        tok[i, 3] = 1.0 if rest else 0.0
        tok[i, 4] = 1.0 if (on_board and sick) else 0.0
        tok[i, 5] = 1.0 if (on_board and owner_turn and not rest and not sick
                            and t in ("LEADER", "CHARACTER")) else 0.0
        tok[i, 6] = 1.0 if (on_board and t == "CHARACTER" and not rest and _has_kw(c, _BLOCKER_KW)) else 0.0
        if z == "hand":
            cv = 0.0
            try:
                cv = float(getattr(c, "current_counter", 0) or 0)
            except Exception:
                cv = float(getattr(m, "counter", 0) or 0)
            if p is not None and t == "EVENT":
                cv = max(cv, p["counter_event"])
            tok[i, 7] = min(cv / 2000.0, 2.5)
            if t == "CHARACTER":
                tok[i, 8] = 1.0 if cs[i] <= n_active else 0.0
            elif t in ("EVENT", "STAGE"):
                tok[i, 8] = 1.0 if cs[i] <= n_active else 0.0
        if p is not None:
            tok[i, 9] = min(p["ret_don"], 3.0) / 3.0
            tok[i, 14] = 1.0 if "ON_KO" in p["trig"] else 0.0
            tok[i, 15] = 1.0 if "ON_ATTACK" in p["trig"] else 0.0
            tok[i, 16] = 1.0 if ("ON_OPP_ATTACK" in p["trig"] or "OPPONENT_ATTACK" in p["trig"]) else 0.0
        owner = me if z in ("own_leader", "own_field", "hand") else opp
        tok[i, 10:14] = _cond_flags(manager, owner, c)
        if z in ("own_leader", "own_field") and pool_thr:
            hit = 0
            for th, cc, red in pool_thr:
                if cc > don_next_opp:
                    continue
                ok, _g = _reach(th, pw[i] - red, cs[i], rest, is_leader=(_zone(i) == 'own_leader'))
                if ok:
                    hit += 1
            tok[i, 17] = min(hit / n_pool, 1.0)
        tok[i, 18] = 1.0 if t == "CHARACTER" else 0.0
        tok[i, 19] = 1.0 if t == "EVENT" else 0.0

    # ---- 関係 R ----
    own_ids = [i for i in range(N_TOK) if _zone(i) in ("own_leader", "own_field", "hand")]
    opp_ids = [i for i in range(N_TOK) if _zone(i) in ("opp_leader", "opp_field")]
    rel_om = np.zeros((N_OWN, N_OPP, R_DIM), np.float32)
    rel_om[:, :, 1] = GAP_SAT
    rel_om[:, :, 2] = GAP_SAT
    rel_oo = np.zeros((N_OWN, N_OWN, R_DIM), np.float32)
    rel_oo[:, :, 1] = GAP_SAT
    rel_oo[:, :, 2] = GAP_SAT
    profs = {}
    for i in own_ids:
        c = slots[i]
        if c is None or getattr(c, "master", None) is None:
            continue
        profs[i] = profile(c.master)
    for i in own_ids:
        c = slots[i]
        if c is None:
            continue
        pi = profs.get(i)
        oi = _own_index(i)
        usable = tok[i, 8] > 0 or _zone(i) != "hand"      # 手札は今出せる/使えるとき、場は常に
        cond_all = float(tok[i, 10:14].min())
        for j in opp_ids:
            d = slots[j]
            if d is None:
                continue
            oj = _opp_index(j)
            if _zone(i) != "hand" and tok[i, 5] > 0:
                rel_om[oi, oj, 0] = (pw[i] - pw[j]) / 10000.0
            if pi is None:
                continue
            best_gap, best_cgap, feas = GAP_SAT, GAP_SAT, 0.0
            j_leader = _zone(j) == "opp_leader"
            for th in pi["thr"]:
                pm, cm, needs_rest, _k, _al = th
                ok, g = _reach(th, pw[j], cs[j], bool(getattr(d, "is_rest", False)), is_leader=j_leader)
                if pm is not None or cm is None:
                    best_gap = min(best_gap, g)
                if cm is not None:
                    best_cgap = min(best_cgap, (cs[j] - cm) * 0.1)
                if ok and usable and cond_all > 0:
                    feas = 1.0
            rel_om[oi, oj, 1] = best_gap
            rel_om[oi, oj, 2] = best_cgap
            rel_om[oi, oj, 3] = min(pi["red"], 15000.0) / 10000.0 if pi["red"] > 0 else 0.0
            rel_om[oi, oj, 4] = feas
    # 自×自: i の減算で k のしきい値が相手の誰かに届くか（組）
    for i in own_ids:
        pi = profs.get(i)
        if pi is None or pi["red"] <= 0:
            continue
        oi = _own_index(i)
        for k in own_ids:
            if k == i:
                continue
            pk = profs.get(k)
            if pk is None or not pk["thr"]:
                continue
            ok_ = _own_index(k)
            best, feas = GAP_SAT, 0.0
            for j in opp_ids:
                d = slots[j]
                if d is None:
                    continue
                for th in pk["thr"]:
                    ok, g = _reach(th, pw[j] - pi["red"], cs[j], bool(getattr(d, "is_rest", False)),
                                   is_leader=(_zone(j) == "opp_leader"))
                    best = min(best, g)
                    if ok and (tok[i, 8] > 0 or _zone(i) != "hand") and (tok[k, 8] > 0 or _zone(k) != "hand"):
                        feas = 1.0
            rel_oo[oi, ok_, 1] = best
            rel_oo[oi, ok_, 3] = min(pi["red"], 15000.0) / 10000.0
            rel_oo[oi, ok_, 4] = feas

    # ---- グローバル追加列 ----
    ex = np.zeros(EXTRA_DIM, np.float32)
    ex[0] = _leader_act_avail(manager, me) if my_turn else 0.0
    add = 0.0
    rush = 0
    for c in (getattr(me, "hand", ()) or ()):
        m = getattr(c, "master", None)
        if m is None:
            continue
        p = profile(m)
        if p["ramp"] > 0 and int(getattr(m, "cost", 0) or 0) <= n_active:
            add += p["ramp"]
        if p["rush"] and getattr(m, "type", None) is not None and _tname(c) == "CHARACTER":
            rush += 1
    ex[1] = min(add, 5.0) / 5.0
    ex[2] = sum(1 for i in own_ids if tok[i, 5] > 0) / 6.0
    ex[3] = min(rush, 5) / 5.0
    ex[4] = min(sum(1 for c in (getattr(opp, "trash", ()) or ())
                    if (getattr(getattr(c, "master", None), "counter", 0) or 0) > 0), 10) / 10.0
    opp_attack = sum(pw[j] for j in opp_ids if slots[j] is not None and not getattr(slots[j], "is_rest", False))
    my_guard = sum(tok[i, 7] * 2000.0 for i in own_ids if _zone(i) == "hand") + 1000.0 * sum(
        1 for i in own_ids if tok[i, 6] > 0)
    ex[5] = float(np.clip((opp_attack - my_guard) / 20000.0, -1.5, 1.5))
    dr = np.zeros(len(ROLES), np.float32)
    for c in (getattr(me, "deck", ()) or ()):
        m = getattr(c, "master", None)
        if m is not None:
            dr += roles_of(m)
    ex[6:6 + len(ROLES)] = np.minimum(dr, 10.0) / 10.0
    pr = np.zeros(len(ROLES), np.float32)
    pmax, pbig, pctr, pblk = 0.0, 0, 0.0, 0
    for c in pool:
        m = getattr(c, "master", None)
        if m is None:
            continue
        pr += roles_of(m)
        pp = float(getattr(m, "power", 0) or 0)
        if _tname(c) == "CHARACTER":
            pmax = max(pmax, pp)
            if pp >= _BIG_POWER:
                pbig += 1
        pctr += float(getattr(m, "counter", 0) or 0)
        if profile(m)["blocker"]:
            pblk += 1
    b = 6 + len(ROLES)
    ex[b:b + len(ROLES)] = np.minimum(pr, 10.0) / 10.0
    b += len(ROLES)
    ex[b + 0] = pmax / 10000.0
    ex[b + 1] = min(pbig, 10) / 10.0
    ex[b + 2] = min(pctr, 20000.0) / 20000.0
    ex[b + 3] = min(pblk, 10) / 10.0
    don_total_me = (n_active + len(getattr(me, "don_rested", ()) or ())
                    + len(getattr(me, "don_attached_cards", ()) or ()))
    don_next_me = min(10, don_total_me + min(1, len(getattr(me, "don_deck", ()) or ())))
    lp_now = _power(me.leader, my_turn) if me.leader is not None else 0
    lp_max = (_power(me.leader, False) + 1000 * don_next_me) if me.leader is not None else 0
    ex[b + 4] = lp_now / 10000.0
    ex[b + 5] = lp_max / 10000.0
    ex[b + 6] = don_next_me / 10.0
    mp = 0
    for c in (getattr(me, "hand", ()) or ()):
        m = getattr(c, "master", None)
        if m is not None and _tname(c) == "CHARACTER":
            cc = int(getattr(m, "cost", 0) or 0)
            if cc <= don_next_me:
                mp = max(mp, cc)
    ex[b + 7] = mp / 10.0
    n_opp_attackers = max(1, sum(1 for j in opp_ids if slots[j] is not None))
    ex[b + 8] = min((my_guard / 2000.0) / n_opp_attackers, 5.0) / 5.0
    return {"tokens": tok, "rel_om": rel_om, "rel_oo": rel_oo, "extra": ex}


def extra_scalars(manager, me_name):
    """`encoder.encode(version=13)` が v12 の末尾へ付けるグローバル追加列（EXTRA_DIM）。"""
    return encode_rel(manager, me_name)["extra"]
