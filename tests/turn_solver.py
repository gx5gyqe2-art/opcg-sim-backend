"""① ターンソルバ（オラクル・dev・docs/reports/cpu_correctness_instruments_20260628.md §2）。

**eval を一切使わず**、与えられた完全情報の盤面から「**このターンに攻撃側がリーサルを強制できるか**」を
**エンジンを総当たり駆動する有界 minimax** で客観判定する（ルールは再実装せず既存エンジンの遷移を使う＝
エンジンの正しさを継承）。これが Regret 計器(①)の Ground Truth 源。

minimax の構造（終端=WIN/LOSS のみ。評価関数なし）:
- 攻撃側(attacker)のノード = MAX：いずれかの手で強制勝ちに至れば lethal。
- 守備側(defender)のノード = MIN：いずれかの応手で生存できれば not-lethal（防御側は生存を選ぶ）。
- 攻撃側ターンが相手 MAIN へ渡った（=このターン中に倒せなかった）= 葉 False。
- winner 確定 = 葉（winner==attacker なら True）。

隠れ情報は扱わない：**呼び出し側が相手手札を決定化した完全情報盤面**を渡す（世界平均/層化は上位層）。
node_budget 超過時は None（=この局面はソルバの予算内で解けない＝別扱い）。

検証は test_turn_solver.py（紙で解ける極小局面の解析解と一致）。次段で愚直全探索との二重実装 fuzz を足す。
"""
from opcg_sim.src.core import action_api


class _BudgetExceeded(Exception):
    pass


def _player_by_name(m, name):
    return m.p1 if m.p1.name == name else m.p2


def _apply(m, actor, mv):
    if mv["kind"] == "battle":
        action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
    else:
        action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))


def _solve(m, attacker, defender, budget):
    # 葉: 勝敗確定。
    if m.winner is not None:
        return m.winner == attacker
    pa = m.pending_actor_action()
    if pa is None:
        return False  # 決定待ちが無く勝敗未確定（ターン外）＝このターンでは倒せていない
    pid, action = pa
    # 攻撃側ターンが相手の手番（MAIN開始）へ渡った＝このターン中に倒せなかった。
    if pid == defender and action == "MAIN_ACTION":
        return False
    budget[0] -= 1
    if budget[0] <= 0:
        raise _BudgetExceeded()
    actor = _player_by_name(m, pid)
    moves = m.get_legal_actions(actor)
    if not moves:
        return False
    is_max = (pid == attacker)   # 攻撃側=MAX(勝ちを探す)／守備側=MIN(生存を探す)
    saw = False
    for mv in moves:
        c = m.clone()
        ca = _player_by_name(c, pid)
        try:
            _apply(c, ca, mv)
        except Exception:
            continue
        r = _solve(c, attacker, defender, budget)
        saw = True
        if is_max and r:
            return True      # 攻撃側: 強制勝ち手を1つ見つけた＝lethal
        if (not is_max) and (not r):
            return False     # 守備側: 生存できる応手を1つ見つけた＝not-lethal
    if not saw:
        return False
    # MAX: どの手も勝てなかった→False／MIN: どの応手も負け→True（防御不能＝lethal）。
    return not is_max


def is_lethal(manager, attacker_name, node_budget=200000):
    """完全情報盤面 `manager` で、`attacker_name` が**このターンにリーサルを強制できる**なら True。

    予算 `node_budget` を超えたら None（=解けない局面・上位で別集計）。盤面は破壊しない（毎ノード clone）。
    """
    other = manager.p2.name if manager.p1.name == attacker_name else manager.p1.name
    budget = [node_budget]
    try:
        return _solve(manager.clone(), attacker_name, other, budget)
    except _BudgetExceeded:
        return None
