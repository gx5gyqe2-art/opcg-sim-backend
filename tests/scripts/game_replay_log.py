"""1局まるごとの人間可読プレイログ（v48・2026-08-09・読み取り専用）。

**問い**: あるデッキを CPU は**まともに回せているか**。統計ではなく打ち回しそのものを見る。

v47b で `nami:p_enel` の実測勝率がエネル席 5.3%（訓練対面は実測 67.6% 対 予測 69.3% と
一致するので計器の問題ではない）と出た。ユーザの実プレイ知見では「エネルはナミに不利」
だが 5% は不利の範囲を超える。**value の較正を論じる前に、そもそもエンジンがこのデッキを
回せているのかを目で確かめる**——それが本器の用途。

**既存器との棲み分け**:
  - `turn_exhaust_probe.py`（v45）は**1ターン**を打ち切って探索の中身（visit%/Q/prior）を出す。
    「なぜその手を選んだか」を見る器で、対象は検証点の復元盤面1つ。
  - `divergence_probe.py` は乖離カタログの**集計**。
  - 本器は**1局を通し**で、各決定点の「盤面 → 合法手数 → 選んだ手」だけを平たく出す。
    探索の内部は出さない代わりに、**デッキが機能しているか**（キーカードが出るか・
    リーダー能力が撃たれるか・付与先は妥当か）を人間が読める形にする。

**読み方**: `--focus` で見たい側のリーダーを指定すると、その席の決定だけ詳細表示になる。
末尾のサマリで「行動種別の回数」「プレイされたカード」「リーダー能力の発動回数」が出るので、
まずそこを見て、気になったターンを本文で追う。

**注意**: 既定は `--eps 0`（Dirichlet ノイズなし＝serve と同じ決定的なプレイ）。
コーパス生成が見ていた play を再現したいときは `--eps 0.15` にする（`value_label_gen` の既定）。
ラベル自体は探索ではなく `CR.rollout`（より安い方策）で付くので、**本器で見えるのは
生成時の自己対戦の質であって、ラベルの質そのものではない**。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/game_replay_log.py \\
    --matchup nami:p_enel --seed 991001 --focus OP15-058
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
import defense_cf_gen as DG                                  # noqa: E402
from plan_cf_gen import DECKS_JSON, _init_worker             # noqa: E402
from opcg_sim.src.core import cpu_ai                         # noqa: E402

_G = DG._G


def _decide(game, m, name, sims, eps, rng, world_seed):
    """自己対戦の1手。`defense_cf_gen._decide` と同一設計だが eps=0 を既定にできる。

    生成器の `_decide` をそのまま呼ばないのは、あちらが「盤面を散らす」ために
    Dirichlet ノイズ込みで設計されているため。**打ち回しを見る**用途ではノイズは邪魔なので
    既定を 0 にし、生成時の再現が要るときだけ 0.15 を渡す。
    """
    from az_mcts_tree import TreeMCTS
    ds = int((world_seed * 1000003 + int(getattr(m, "turn_count", 0) or 0) * 131
              + (0 if name == "p1" else 7)) % (2 ** 63 - 1))
    mcts = TreeMCTS(game, value_fn=_G["vf"], priors_fn=_G["pf"], c_puct=1.5, n_sims=sims,
                    dirichlet_eps=eps,
                    determinize_fn=lambda s, r, _n=name:
                        game.determinize(s, _n, np.random.default_rng(ds)),
                    rng=rng)
    mv, _n, legal = mcts.run(m)
    return mv, legal


def _leader_id(pl):
    ld = getattr(pl, "leader", None)
    if ld is None:
        return "?"
    return getattr(ld.master, "card_id", None) or getattr(ld.master, "name", "?")


def _char_str(c):
    """場キャラ1体を「card_id(P power,+N don,rest)」に整形する。"""
    cid = getattr(c.master, "card_id", None) or getattr(c.master, "name", "?")
    try:
        pw = int(c.get_power(False))
    except Exception:
        pw = int(getattr(c.master, "power", 0) or 0)
    bits = [f"P{pw}"]
    ad = int(getattr(c, "attached_don", 0) or 0)
    if ad:
        bits.append(f"+{ad}don")
    if getattr(c, "is_rest", False):
        bits.append("rest")
    return f"{cid}({','.join(bits)})"


def _hand_str(pl):
    """自手札を「card_id(cコスト,Cカウンター)」で並べる。

    **自分の手札のみ**に使う（相手手札は encoder も枚数しか見ない＝公平性契約。
    ログでも出さないことで、読む側が「CPU に見えていない情報」で判断しないようにする）。
    ドンをリーダーに注いだ手が妥当かは、そのとき出せた手札を見ないと判定できないので、
    コストと**カウンター値**（＝出さずに守りへ残す価値）の両方を出す。
    """
    out = []
    for c in pl.hand:
        m = getattr(c, "master", None)
        if m is None:
            continue
        cid = getattr(m, "card_id", None) or getattr(m, "name", "?")
        cost = int(getattr(m, "cost", 0) or 0)
        try:
            cv = int(float(getattr(c, "current_counter", None) or 0)
                     or float(getattr(m, "counter", 0) or 0))
        except Exception:
            cv = 0
        out.append(f"{cid}(c{cost}{f',C{cv}' if cv else ''})")
    return " ".join(out) or "（手札なし）"


def _side_str(pl, show_hand=False):
    da, dr = len(pl.don_active), len(pl.don_rested)
    ld_don = int(getattr(getattr(pl, "leader", None), "attached_don", 0) or 0)
    field = " / ".join(_char_str(c) for c in pl.field) or "（場は空）"
    s = (f"ライフ{len(pl.life)} ドン{da + dr}(活{da}/レ{dr}) 手札{len(pl.hand)}"
         f"{f' L+{ld_don}don' if ld_don else ''}\n        場: {field}")
    if show_hand:
        s += f"\n        手: {_hand_str(pl)}"
    return s


def _fmt_move(m, mv):
    d = cpu_ai._describe_move(m, mv) or {}
    s = str(d.get("action_type"))
    if d.get("card"):
        s += f" {d['card']}"
    if d.get("targets"):
        s += f" → {','.join(str(t) for t in d['targets'])}"
    if d.get("selected"):
        s += f" [選択: {','.join(str(t) for t in d['selected'])}]"
    if d.get("accepted") is False:
        s += " （見送り）"
    return s, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", default="nami:p_enel")
    ap.add_argument("--seed", type=int, default=991001)
    ap.add_argument("--sims", type=int, default=128, help="決定の探索数（生成器の gen_sims 相当）")
    ap.add_argument("--eps", type=float, default=0.0, help="Dirichlet ノイズ（生成再現は 0.15）")
    ap.add_argument("--focus", default="", help="詳細表示するリーダー card_id（空＝両席）")
    ap.add_argument("--no-hand", action="store_true",
                    help="手番側の手札内訳を出さない（既定は出す。相手手札は常に非表示）")
    ap.add_argument("--max-steps", type=int, default=0, help="0=CR.MAX_STEPS")
    ap.add_argument("--decks-json", default=DECKS_JSON)
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--rollout-sims", type=int, default=24)
    ap.add_argument("--out", default="", help="要約 JSON の保存先（空=保存しない）")
    args = ap.parse_args()

    _init_worker(args.matchup, args.decks_json, args.rollout_sims, args.enc_version)
    CR, game, gserve = _G["CR"], _G["game_gen"], _G["gserve"]
    max_steps = args.max_steps or CR.MAX_STEPS
    rng = np.random.default_rng(args.seed)
    m = game.new_game(_G["db"], args.seed)

    lead = {p.name: _leader_id(p) for p in (m.p1, m.p2)}
    print(f"=== {args.matchup} seed={args.seed} sims={args.sims} eps={args.eps}")
    print(f"    p1={lead['p1']}  p2={lead['p2']}\n")

    acts = collections.defaultdict(collections.Counter)   # leader -> action_type
    played = collections.defaultdict(collections.Counter)  # leader -> card played
    steps, last_turn = 0, -1
    while m.winner is None and not gserve.is_terminal(m) and steps < max_steps:
        name = gserve.current_player(m)
        if name is None:
            break
        me = m.p1 if m.p1.name == name else m.p2
        opp = m.p2 if m.p1.name == name else m.p1
        lid = _leader_id(me)
        show = (not args.focus) or (lid == args.focus)
        turn = int(getattr(m, "turn_count", 0) or 0)
        if show and turn != last_turn:
            print(f"\n---------- ターン {turn} ----------")
            last_turn = turn
        mv, legal = _decide(game, m, name, args.sims, args.eps, rng, world_seed=args.seed)
        if mv is None:
            break
        txt, d = _fmt_move(m, mv)
        at = str(d.get("action_type"))
        acts[lid][at] += 1
        if at in ("PLAY", "PLAY_CARD", "ACTIVATE_MAIN") and d.get("card"):
            played[lid][d["card"]] += 1
        if show:
            print(f"  [{lid}] {_side_str(me, show_hand=not args.no_hand)}")
            print(f"        相手: {_side_str(opp)}")   # 相手手札は出さない（公平性契約と同じ扱い）
            print(f"     合法{len(legal):3d}手 → {txt}")
        child = gserve.apply(m, mv, name)
        if child is None:
            break
        m = child
        steps += 1

    win = getattr(m, "winner", None)
    wl = lead.get(win, "?") if win else "（未決着＝打ち切り）"
    print(f"\n=== 決着: {wl}  （{steps}手・turn {int(getattr(m, 'turn_count', 0) or 0)}）")
    for lid in sorted(acts):
        print(f"\n[{lid}] 行動種別: " + " ".join(f"{k}×{v}" for k, v in acts[lid].most_common()))
        if played[lid]:
            print(f"    プレイ/起動したカード: "
                  + " ".join(f"{k}×{v}" for k, v in played[lid].most_common()))
    summary = {"matchup": args.matchup, "seed": args.seed, "winner_leader": wl, "steps": steps,
               "actions": {k: dict(v) for k, v in acts.items()},
               "played": {k: dict(v) for k, v in played.items()}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    print("GAME_REPLAY_LOG_DONE " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
