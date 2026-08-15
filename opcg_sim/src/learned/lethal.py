"""リーサル距離の実測スキャン（符号化 v10 のΔ特徴・2026-08-12 v52b 採用判定）。

台本レース（MCTS なし・決定論・効果はエンジンが解決＝静的解釈ゼロ）で
「この盤面からあと何ターンで詰むか」を測る。研究計器 `tests/scripts/lethal_distance_probe.py`
（v52/v52b で 58 乖離点＋一般 60 点に対し検証済み）の v1/v2 測定の本番移植で、
per-step clone を **1回 clone＋in-place 適用**に置き換えた高速版。

**公平性契約**（encoder.py と同じ）: 特徴に相手の非公開情報を使わない。
  - d_me   (v1 無抵抗)   : 自分の全力レース。相手は TURN_END のみ＝相手手札を読まない。
  - d_opp  (v1 無抵抗)   : 相手の全力レース。台本は手札から何もプレイしない＝場とリーダー
                           （公開情報）駆動。
  - d_opp_def (v2 防御込み): 相手の全力レース vs **自分の**カウンター防御台本（自分の手札
                           印字値のみ＝自己情報）。v52 が特定した乖離族の正体
                           「最初の詰め手を守り切れるか」を直接測る成分。
  - d_me_def は**入れない**: 相手の防御台本が相手手札のカウンター値を読む＝契約違反。
    クリーン3成分の説明力は検証済み（v52b 追補: 乖離58 LOO 0.69・一般60 LOO 0.60、
    4成分版との差は LOO ±0.03 以内）。

台本（probe v2 と同一・変更するときは probe との等価性テストを更新すること）:
  - 手番側: 効果対話は既定解決（accept 側）→ リーダー起動能力 → 相手リーダーへ攻撃
    （枝刈り後の列挙順先頭）→ TURN_END。
  - 非手番側: defend=False は素通し（TURN_END/PASS/decline）。defend=True は
    カウンター防御台本＝攻撃が通り、手持ちカウンター（印字値）合計で止め切れる時だけ
    最大値から切る。
"""
from typing import Optional

from opcg_sim.src.core import cpu_ai

MAX_TURNS = 12          # 打ち切り（v52 検証と同値）。詰まない＝ MAX_TURNS+1
MAX_STEPS = 240


def _desc(m, mv):
    try:
        return cpu_ai._describe_move(m, mv) or {}
    except Exception:
        return {}


def _counter_value(owner, card_id):
    for c in owner.hand:
        if getattr(c.master, "card_id", None) == card_id:
            v = int(getattr(c, "current_counter", 0) or 0)
            if v > 0:
                return v
    return 0


def _script_move(gs, m, name, defend):
    """probe v2 の台本方策と同一の1手選択（docstring 参照）。"""
    legal = gs.legal_actions(m)
    if not legal:
        return None
    cur = gs.current_player(m)
    descs = [(_desc(m, mv), mv) for mv in legal]
    if cur != name:
        if defend:
            ctr = [(d, mv) for d, mv in descs if d.get("action_type") == "SELECT_COUNTER"]
            ab = getattr(m, "active_battle", None)
            if ctr and ab:
                try:
                    atk = int(ab["attacker"].get_power(True))
                    tgt = int(ab["target"].get_power(False)) + int(ab.get("counter_buff", 0) or 0)
                except Exception:
                    atk, tgt = 0, 0
                need = atk - tgt + 1000 if atk >= tgt else 0   # 攻撃は atk>=def で通る
                if need > 0:
                    owner = ab["target_owner"]
                    vals = sorted(((_counter_value(owner, d.get("card")), d, mv)
                                   for d, mv in ctr), key=lambda x: -x[0])
                    total = sum(v for v, _d, _m in vals if v > 0)
                    if total >= need and vals and vals[0][0] > 0:
                        return vals[0][2]          # 止め切れる時だけ最大値から切る
        for d, mv in descs:                        # 相手の自ターンは何もせず END＝無抵抗
            if d.get("action_type") == "TURN_END":
                return mv
        for d, mv in descs:
            if d.get("action_type") in ("PASS",):
                return mv
        for d, mv in descs:
            if d.get("action_type") == "RESOLVE_EFFECT_SELECTION" and d.get("accepted") is False:
                return mv
        return legal[0]
    resolves = [(d, mv) for d, mv in descs if d.get("action_type") == "RESOLVE_EFFECT_SELECTION"]
    if resolves:
        for d, mv in resolves:
            if d.get("accepted") is not False:
                return mv
        return resolves[0][1]
    for d, mv in descs:                            # リーダー起動能力（再装填等の経済）
        if d.get("action_type") == "ACTIVATE_MAIN":
            me = m.p1 if m.p1.name == name else m.p2
            if me.leader is not None and d.get("card") == me.leader.master.card_id:
                return mv
    opp = m.p2 if m.p1.name == name else m.p1
    lid = opp.leader.master.card_id if opp.leader else None
    atks = [(d, mv) for d, mv in descs
            if d.get("action_type") == "ATTACK" and (d.get("targets") or [None])[0] == lid]
    if atks:
        return atks[0][1]
    for d, mv in descs:
        if d.get("action_type") == "TURN_END":
            return mv
    return legal[0]


def lethal_distance(gs, m0, name, max_turns=MAX_TURNS, defend=False) -> int:
    """name 視点の台本レース距離（自ターン数・詰まねば max_turns+1）。m0 は変更しない。

    probe 版との差は適用経路のみ: per-step clone → **1回 clone＋in-place**
    （`cpu_ai._apply_move_inplace`）。適用例外＝ probe の apply→None と同じく測定打ち切り。

    **乱数状態ガード**: クローン上のレースでも効果解決（デッキサーチ後のシャッフル等）が
    グローバル `random`/`np.random` を消費する。符号化は観測であり世界を進めてはならない
    （消費すると同一シード対局の軌道が変わる＝CRN/決定論再生が壊れる。bb2 実測 2026-08-12:
    行数 8652→8553 の乖離・教師再生 35/50 不一致の原因）。測定前後で状態を退避/復元する。
    """
    import random as _random
    import numpy as _np
    _st_py = _random.getstate()
    _st_np = _np.random.get_state()
    try:
        return _lethal_distance_inner(gs, m0, name, max_turns, defend)
    finally:
        _random.setstate(_st_py)
        _np.random.set_state(_st_np)


def _lethal_distance_inner(gs, m0, name, max_turns, defend) -> int:
    m = m0.clone()
    m.action_events = []
    my_turns = 0
    steps = 0
    while steps < MAX_STEPS:
        if m.winner is not None:
            return my_turns if m.winner == name else max_turns + 1
        cur = gs.current_player(m)
        if cur is None:
            return max_turns + 1
        mv = _script_move(gs, m, name, defend)
        if mv is None:
            return max_turns + 1
        d = _desc(m, mv)
        is_end = (cur == name and d.get("action_type") == "TURN_END")
        try:
            cpu_ai._apply_move_inplace(m, cur, mv, stop_at_select=True)
        except Exception:
            return max_turns + 1
        steps += 1
        if is_end:
            my_turns += 1
            if my_turns >= max_turns:
                return max_turns + 1
    return max_turns + 1


_GS = None


def _game():
    global _GS
    if _GS is None:
        from opcg_sim.src.learned.adapter import OPCGGame
        # probe v2 と同じ枝刈りあり列挙（prune_futile=True 明示＝config 変更に依存しない）
        _GS = OPCGGame(prune_futile=True)
    return _GS


def lethal_scan(manager, me_name: str, max_turns: int = MAX_TURNS):
    """符号化 v10 の3値: (d_me, d_opp, d_opp_def)。それぞれ 0..max_turns+1 の整数。

    例外時は全て max_turns+1（「測れない＝詰みは見えない」の中立側）。
    """
    gs = _game()
    opp = manager.p2.name if manager.p1.name == me_name else manager.p1.name
    try:
        d_me = lethal_distance(gs, manager, me_name, max_turns=max_turns)
        d_opp = lethal_distance(gs, manager, opp, max_turns=max_turns)
        d_opp_def = lethal_distance(gs, manager, opp, max_turns=max_turns, defend=True)
    except Exception:
        cap = max_turns + 1
        return cap, cap, cap
    return d_me, d_opp, d_opp_def
