"""① Lethal Regret 計器（dev・docs/reports/cpu_correctness_instruments_20260628.md §2）。

「正しい選択ができるCPUか」を**自己対戦Eloでなく Average Regret（連続値）**で測る本体。
- ある決定局面 S（攻撃側の手番）で、攻撃側の各合法手 i について
  **P_true(i) = E_world[ is_lethal( apply(i, 決定化world), 攻撃側 ) ]**（ターンソルバ＝客観オラクル）。
- **Regret = max_i P_true(i) − P_true(CPUの手)**。0 なら最善（リーサルを取りこぼしていない）。
- 分散低減：**同一局面の全手 i を同じ W 世界で採点（CRN）**＝Regret は相関ペア差で低分散
  （個々の P_true は高分散でも差はくっきり）。層化抽出は次段の精緻化。

採点は**客観 P_true（決定化世界の期待値）**で行う＝CPU が見た世界では採点しない（循環回避）。
fair CPU は隠れ情報を読まないので、worst-case でなく**期待値最大の手**と比べるのが正当（レビュー合意）。

Greedy/Random エージェントの Regret を**較正基準**に併走（Greedy≈CPU≈0 ならデータが易しすぎ＝ゴミ）。

実行（スモーク）: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/lethal_regret.py --positions 10 --worlds 40
"""
import argparse
import random
import statistics

import conftest  # noqa: F401
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.enums import CardType
from cpu_selfplay import build_deck, _load_db
from engine_helpers import make_master, make_instance
from turn_solver import is_lethal

LETHAL_MIN = 0.05   # max P_true がこの未満の局面はリーサル機会なし＝採点対象外。


def _apply(m, actor, mv):
    if mv["kind"] == "battle":
        action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
    else:
        action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))


def _move_sig(mv):
    return cpu_ai._move_sig(mv)


def _p_true_table(S, attacker, moves, worlds, budget):
    """各手 i の P_true(i)=mean_world is_lethal(apply(i,world))。同一 worlds で全手採点（CRN）。

    予算超過(None)が混じった手は除外（measurable でない）。返り値 {sig: P_true}（採点できた手のみ）。
    """
    other = S.p2.name if S.p1.name == attacker else S.p1.name
    table = {}
    for mv in moves:
        vals = []
        ok = True
        for w in worlds:
            c = w.clone()
            actor = c.p1 if c.p1.name == attacker else c.p2
            try:
                _apply(c, actor, mv)
            except Exception:
                ok = False
                break
            r = is_lethal(c, attacker, node_budget=budget)
            if r is None:
                ok = False
                break
            vals.append(1.0 if r else 0.0)
        if ok and vals:
            table[_move_sig(mv)] = sum(vals) / len(vals)
    return table


def _greedy_move(S, attacker, moves):
    """較正用の素朴エージェント: リーダーへの ATTACK を最優先、無ければ最初の手。"""
    atk_leader = [m for m in moves if m.get("action_type") == "ATTACK"]
    if atk_leader:
        return atk_leader[0]
    return moves[0]


def score_position(S, attacker, agents, worlds, budget):
    """局面 S の P_true テーブルを**1回だけ**作り、全エージェントの Regret を返す。

    返り値 {agent名: regret or None}。リーサル機会なし(best<LETHAL_MIN)/手不足なら全 None。
    """
    actor = S.p1 if S.p1.name == attacker else S.p2
    moves = S.get_legal_actions(actor)
    if not moves or len(moves) < 2:
        return {k: None for k in agents}
    table = _p_true_table(S, attacker, moves, worlds, budget)
    if not table:
        return {k: None for k in agents}
    best = max(table.values())
    if best < LETHAL_MIN:
        return {k: None for k in agents}   # リーサル機会のない局面＝対象外
    out = {}
    for name, choose in agents.items():
        mv = choose(S, attacker, moves)
        sig = _move_sig(mv)
        out[name] = (best - table[sig]) if sig in table else None
    return out


def _gen_positions(db, n_games, max_plies, seed0):
    """自己対戦を進め、p1 の MAIN_ACTION 開始（ターン境界）でクローンを採取（決定局面）。"""
    snaps = []
    cpu_ai.set_budget_override(40)
    try:
        for g in range(n_games):
            random.seed(seed0 + g)
            l1, c1 = build_deck(db, "p1")
            l2, c2 = build_deck(db, "p2")
            m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
            m.start_game()
            rng = random.Random((seed0 + g) * 9 + 1)
            for _ in range(max_plies):
                if m.winner is not None:
                    break
                pa = m.pending_actor_action()
                if not pa:
                    break
                pid, action = pa
                actor = m.p1 if m.p1.name == pid else m.p2
                if pid == "p1" and action == "MAIN_ACTION":
                    snaps.append(m.clone())
                try:
                    mv = cpu_ai.decide_guarded(m, actor, "hard", rng, pimc_worlds=1)
                    if mv is None:
                        break
                    _apply(m, actor, mv)
                except Exception:
                    break
    finally:
        cpu_ai.set_budget_override(None)
    return snaps


def _make_worlds(S, attacker, W, rng):
    return [cpu_ai._determinize_opponent(S, attacker, rng) for _ in range(W)]


def _vanilla(cid, owner, power, keywords=None, rest=False):
    m = make_master(card_id=cid, name=cid, type=CardType.CHARACTER,
                    cost=2, power=power, counter=1000, abilities=(), effect_text="")
    if keywords:
        object.__setattr__(m, "keywords", set(keywords))
    inst = make_instance(m, owner=owner)
    inst.is_rest = rest
    inst.is_newly_played = False
    return inst


def _build_validation_positions(db, n):
    """**選択が結果を分ける**構成済みリーサル局面を作る（計器が良手/悪手を弁別するか検証）。

    構成: 相手ライフ1・相手場に「囮のレスト キャラ」1体・p1=リーダー+バニラ1体(総打点2)。
    正解=2体ともリーダーへ攻撃(=リーサル)／悪手=囮キャラを攻撃して打点を浪費(=非リーサル)。
    → 正しく選べば Regret 0、囮に吸われると Regret>0。決定論的に再現。
    """
    from test_turn_solver import _gm_at_p1_main  # 既存の p1 メイン到達ヘルパを再利用
    out = []
    for i in range(n):
        gm = _gm_at_p1_main(db, seed=i)
        gm.p2.hand.clear()
        gm.p2.life.clear()
        gm.p2.life.append(gm.p2.deck.pop(0))           # ライフ1
        gm.p2.field[:] = [_vanilla(f"DECOY{i}", "p2", 3000, rest=True)]  # 囮（レスト＝攻撃対象になる）
        gm.p1.field[:] = [_vanilla(f"ATK{i}", "p1", 6000)]              # +リーダーで総打点2
        gm.p1.hand.clear()
        gm.p1.don_active.clear(); gm.p1.don_rested.clear()
        out.append(gm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=10, help="採点する決定局面数の上限")
    ap.add_argument("--worlds", type=int, default=40, help="P_true 推定の決定化世界数 W（CRN共有）")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--budget", type=int, default=30000, help="ターンソルバの node 予算")
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--validate", action="store_true",
                    help="構成済みの『選択が結果を分ける』リーサル局面で計器の弁別力を検証（高速）")
    args = ap.parse_args()
    db = _load_db()
    rng = random.Random(99)

    if args.validate:
        snaps = _build_validation_positions(db, args.positions)
        print(f"検証局面（構成済み・囮あり）: {len(snaps)}", flush=True)
    else:
        snaps = _gen_positions(db, args.games, args.max_plies, args.seed0)
        print(f"採取した p1 決定局面: {len(snaps)}（{args.games}局）", flush=True)

    def cpu_choose(S, atk, moves):
        return cpu_ai.decide(S, S.p1 if S.p1.name == atk else S.p2, "hard",
                             random.Random(0), moves=moves, pimc_worlds=args.pimc)

    agents = {"CPU(pimc%d)" % args.pimc: cpu_choose,
              "Greedy": _greedy_move,
              "Random": lambda S, atk, moves: rng.choice(moves)}

    regrets = {k: [] for k in agents}
    scored = 0
    for S in snaps:
        worlds = _make_worlds(S, "p1", args.worlds, random.Random(12345))  # 全エージェント同一世界(CRN)
        res = score_position(S, "p1", agents, worlds, args.budget)          # テーブルは局面ごと1回
        if any(v is not None for v in res.values()):
            scored += 1
            for name, r in res.items():
                if r is not None:
                    regrets[name].append(r)
            print(f"  [scored {scored}] " + " / ".join(
                f"{k}={('%.3f' % v) if v is not None else 'NA'}" for k, v in res.items()), flush=True)
        if scored >= args.positions:
            break

    print(f"\n=== Lethal Regret（リーサル機会のある {scored} 局面・W={args.worlds}・budget={args.budget}） ===")
    for name in agents:
        rs = regrets[name]
        if rs:
            avg = statistics.mean(rs)
            mx = max(rs)
            print(f"  {name:14s}: n={len(rs):3d}  Average Regret={avg:.4f}  max={mx:.3f}")
        else:
            print(f"  {name:14s}: n=0（採点局面なし）")
    print("\n解釈: CPU の Average Regret ≈0 ＝戦術リーサルを取りこぼしていない。"
          "Greedy も≈0 ならデータが易しすぎ（要・荒らし局面）。CPU>0 かつ Greedy≫CPU なら計器が機能。")


if __name__ == "__main__":
    main()
