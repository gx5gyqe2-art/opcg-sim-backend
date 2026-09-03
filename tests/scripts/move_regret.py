"""手の監査 段2: 容疑者だけを反実仮想で測る（regret 実測・2026-08-17）。

段1（`move_audit.py`）が挙げた容疑者の判断点について、**選択肢を同一世界・共通乱数で終局まで
打ち分け**、「打った手」と「最良の選択肢」の勝率差＝**regret** を出す。これが「その手は
正しかったか」の定量的な答えになる。

局面の復元: 段1 の行は `(seed, decision)` を持ち、対局は seed で決定論再生できる
（トレースは乱数を消費しない＝`_fill_trace` の乱数状態ガード）。同じ設定で対局を再生し、
`on_decision_point` で**その判断点の直前の manager を複製**して起点にする。1つの seed に
複数の容疑者があっても**再生は1回**で全部拾う。

世界の作り方（`serve_referee` と同じ規約）: 手札・ライフ・場は復元した真値のまま、
**山札の並びだけ**を世界 w の共有 seed で決める。全選択肢が同じ世界を共有する（CRN）ので、
差し引きで残るのは選択の効果だけ。ロールアウトは両席とも本番 decide（つまみ無し）。

**飽和の明示**（v49 の教訓）: 全選択肢が同率（勝率の分散ゼロ）の判断点は「差が無い」のではなく
**判別不能**として `saturated` を立て、regret の集計から外す。世界数を増やすか L1 駆動で
クロスチェックする対象になる。

出力は判断点ごとの regret（＋選択肢ごとの勝率）。段3 は `regret × 出現頻度` の上位を人が裁定し、
ゲート点＋注入教師にする。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/move_regret.py \\
    --suspects /tmp/move_audit.jsonl --worlds 4 --max-suspects 24 --workers 4 \\
    --out /tmp/move_regret.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json
import multiprocessing as mp

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import numpy as np  # noqa: E402

ROLLOUT_MAX_STEPS = 600
_G = {}


def load_suspects(path, max_suspects=0, categories=None, per_category=0):
    """段1 の jsonl から容疑者を読む（優先度降順・純関数寄りの I/O）。

    `categories` 指定時はそのカテゴリだけ（「防御だけ深く測る」等の絞り込み）。
    `per_category` 指定時は**カテゴリごとに上位N点**を取る（層化抽出）。優先度順に素で取ると
    three_way の多いカテゴリ（ドン付与・攻撃）に偏り、**カテゴリ別の平均 regret が作れない**
    ため。段2 は高価なので、限られた点数を「どこへ物量を投じるかの判断」に使える形で配る。
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("suspect"):
                continue
            if categories and r.get("category") not in categories:
                continue
            rows.append(r)
    rows.sort(key=lambda r: (-(r.get("priority") or 0), r["seed"], r["decision"]))
    if per_category:
        taken = collections.defaultdict(int)
        picked = []
        for r in rows:
            c = r.get("category") or "その他"
            if taken[c] >= per_category:
                continue
            taken[c] += 1
            picked.append(r)
        rows = picked
    return rows[:max_suspects] if max_suspects else rows


def plan_replays(suspects):
    """seed ごとに「復元したい決定番号の集合」へ畳む（純関数）。

    同じ seed の容疑者は**1回の再生**でまとめて復元する（再生は容疑者数に比例させない）。
    """
    by = collections.defaultdict(list)
    for r in suspects:
        by[r["seed"]].append(r)
    return {seed: sorted(rows, key=lambda r: r["decision"]) for seed, rows in by.items()}


def regret_of(options, chosen_key):
    """選択肢の勝率表から regret を出す（純関数）。

    返り値 (regret, best_key, saturated):
      regret    … 最良の勝率 − 打った手の勝率（0 以上。打った手が最良なら 0）
      saturated … 全選択肢が同率＝**判別不能**（差が無いのではない）
    """
    if not options:
        return None, None, True
    wrs = {k: v["wr"] for k, v in options.items()}
    best_key = max(wrs, key=lambda k: (wrs[k], k))
    saturated = (max(wrs.values()) == min(wrs.values()))
    if chosen_key not in wrs:
        return None, best_key, saturated
    return round(wrs[best_key] - wrs[chosen_key], 4), best_key, saturated


def _shuffle_decks(m, world):
    """世界 w: 山札の並びだけを共有 seed で決める（手札・ライフ・場は復元した真値のまま）。"""
    for i, pl in enumerate((m.p1, m.p2)):
        r = np.random.default_rng(90000 + world * 101 + i)
        order = r.permutation(len(pl.deck))
        pl.deck[:] = [pl.deck[int(i2)] for i2 in order]


def _init(engine_spec, leaders_mode, decks, sims, worlds, max_options):
    """sims はロールアウト用。**再生**は各行の `audit_sims`（監査時の条件）を使う。"""
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_game import OPCGGame
    _G["db"] = _load_db()
    _G["leaders"], _G["decks"], _G["sims"] = leaders_mode, decks, sims
    _G["worlds"], _G["max_options"] = worlds, max_options
    # sims はエンジンへ渡す＝**ロールアウトにも効く**（既定 160＝本番仕様。開発時に下げると
    # 段2 全体が軽くなる）。serve_referee は本番既定固定だが、本器は容疑者を大量に回すので
    # 費用のつまみを1つ持たせる（測定条件は出力に載せる）。
    if engine_spec:
        v, _, p = engine_spec.partition(",")
        _G["engine"] = LearnedEngine(value_path=v, policy_path=p or None, sims=sims)
    else:
        _G["engine"] = LearnedEngine(sims=sims)
    _G["gr"] = OPCGGame(prune_futile=False)   # 候補列挙は無枝刈り（実際に選べた手を全部見る）
    _G["gs"] = OPCGGame()                     # 進行は serve 同等
    _G["replay_engines"] = {}


def _replay_engine(spec, sims):
    """再生用エンジン（監査時と同じネット・同じ sims）をキャッシュして返す。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    key = (spec or "", sims)
    if key not in _G["replay_engines"]:
        if spec:
            v, _, p = spec.partition(",")
            _G["replay_engines"][key] = LearnedEngine(value_path=v, policy_path=p or None, sims=sims)
        else:
            _G["replay_engines"][key] = LearnedEngine(sims=sims)
    return _G["replay_engines"][key]


class _Capture:
    """observer: 指定した決定番号の**直前**の manager を複製して溜める。"""

    def __init__(self, wanted):
        self.wanted = set(wanted)
        self.n = 0
        self.frames = {}

    def on_decision_point(self, ctx):
        if (self.n + 1) in self.wanted:
            self.frames[self.n + 1] = (ctx.manager.clone(), ctx.actor.name)

    def on_decision(self, ctx, move):
        self.n += 1


def _rollout(m, mover, rng_seed):
    """両席とも本番 decide で終局まで。返り値 (winner, mover視点のライフ差)。"""
    eng, gs = _G["engine"], _G["gs"]
    eng._world_seeds = {}
    rng = np.random.default_rng(rng_seed)
    steps = 0
    while m.winner is None and not gs.is_terminal(m) and steps < ROLLOUT_MAX_STEPS:
        name = gs.current_player(m)
        if name is None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        mv = eng.decide(m, actor, rng=rng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            break
        m, steps = m2, steps + 1
    me = m.p1 if m.p1.name == mover else m.p2
    opp = m.p2 if m.p1.name == mover else m.p1
    return m.winner, len(me.life or []) - len(opp.life or [])


def _measure(frame, actor_name, row):
    """1つの判断点を測る: 選択肢 × 世界で打ち分けて勝率表を返す。"""
    from opcg_sim.src.core import cpu_ai
    gr, worlds = _G["gr"], _G["worlds"]
    legal = gr.legal_actions(frame)
    if not legal:
        return None
    # 等価手（同一 card_id 等）はマージ＝同じ手を何度も測らない（探索の等価規約と同じ鍵）。
    uniq, seen = [], set()
    for mv in legal:
        k = cpu_ai._move_equiv_key(frame, mv)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((k, mv))
    chosen_desc = row.get("chosen")
    # 打った手は必ず測る。残りは列挙順で上限まで（上限は費用の直接の制御）。
    def _is_chosen(mv):
        d = cpu_ai._describe_move(frame, mv) or {}
        return chosen_desc is not None and d == chosen_desc
    ordered = sorted(uniq, key=lambda km: (0 if _is_chosen(km[1]) else 1))
    ordered = ordered[:max(2, _G["max_options"])]

    out = {}
    for k, mv in ordered:
        wins = life = 0.0
        for w in range(worlds):
            m = frame.clone()
            _shuffle_decks(m, w)
            m2 = _G["gs"].apply(m, mv, actor_name)
            if m2 is None:
                continue
            winner, ld = _rollout(m2, actor_name, 90000 + w * 7 + 1)
            wins += 1.0 if winner == actor_name else 0.0
            life += ld
        desc = cpu_ai._describe_move(frame, mv) or {}
        out[json.dumps(desc, ensure_ascii=False, sort_keys=True)] = {
            "wr": round(wins / worlds, 4), "life": round(life / worlds, 3),
            "chosen": bool(_is_chosen(mv)), "move": desc,
        }
    return out


def _run_seed(job):
    """1 seed 分: 対局を1回再生して容疑者の局面を復元し、各点を測る（子プロセス）。"""
    seed, rows = job
    from game_driver import run_game, make_seat, leader_deck_builder
    from promotion_gate import _leader_pair
    db = _G["db"]
    # **再生は監査時の条件で**（sims/対面/デッキ/ネット）。違うと対局が分岐して別の判断点に
    # 着地する（席までずれる）。行に埋めた audit_* を正とし、無ければ CLI の指定へ落とす。
    head = rows[0]
    a_sims = head.get("audit_sims") or _G["sims"]
    a_leaders = head.get("audit_leaders") or _G["leaders"]
    a_decks = head.get("audit_decks") or _G["decks"]
    a_engine = head.get("audit_engine") or ""
    la, lb = _leader_pair(db, seed, a_leaders)
    if a_decks == "synth" and la:
        from deck_synth import synth_deck_builder
        builder = synth_deck_builder(la, lb, seed=seed)
    else:
        builder = leader_deck_builder(la, lb) if la else None
    cap = _Capture([r["decision"] for r in rows])
    seat = make_seat(kind="learned", want_trace=False, sims=a_sims,
                     engine=_replay_engine(a_engine, a_sims))
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat}, deck_builder=builder,
                 observers=(cap,), max_steps=1500, legal_moves="skip", invariants="raise",
                 stop_after_decisions=max(r["decision"] for r in rows))
    except BaseException as e:
        return [{"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}"}]

    out = []
    for row in rows:
        got = cap.frames.get(row["decision"])
        if got is None:
            out.append({**row, "error": "局面を復元できなかった"})
            continue
        frame, actor_name = got
        if actor_name != row.get("seat"):
            out.append({**row, "error": f"席がずれた（{actor_name} != {row.get('seat')}）"})
            continue
        options = _measure(frame, actor_name, row)
        if not options:
            out.append({**row, "error": "選択肢なし"})
            continue
        chosen_key = next((k for k, v in options.items() if v["chosen"]), None)
        regret, best_key, saturated = regret_of(options, chosen_key)
        out.append({**row, "regret": regret, "saturated": saturated,
                    "n_options": len(options), "worlds": _G["worlds"],
                    "best": options[best_key]["move"] if best_key else None,
                    "chosen_matched": chosen_key is not None,
                    "options": list(options.values())})
    return out


def summarize(rows):
    """カテゴリ別の平均 regret（純関数）。飽和・未測定は母数から外し件数を明示する。"""
    by = collections.defaultdict(lambda: {"n": 0, "sum": 0.0, "saturated": 0, "skipped": 0})
    for r in rows:
        b = by[r.get("category") or "その他"]
        if r.get("regret") is None:
            b["skipped"] += 1
            continue
        if r.get("saturated"):
            b["saturated"] += 1
            continue
        b["n"] += 1
        b["sum"] += r["regret"]
    return {k: {"n": v["n"], "mean_regret": round(v["sum"] / v["n"], 4) if v["n"] else None,
                "saturated": v["saturated"], "skipped": v["skipped"]}
            for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspects", required=True, help="段1（move_audit）の jsonl")
    ap.add_argument("--worlds", type=int, default=4, help="世界数（山札の並びだけを振る）")
    ap.add_argument("--max-options", type=int, default=5, help="1判断点で測る選択肢の上限")
    ap.add_argument("--max-suspects", type=int, default=24)
    ap.add_argument("--per-category", type=int, default=0,
                    help="カテゴリごとに上位N点だけ測る（層化抽出＝カテゴリ別平均を作れる）")
    ap.add_argument("--categories", default="", help="カンマ区切りで絞る（例: 防御,ドン付与）")
    ap.add_argument("--sims", type=int, default=160,
                    help="ロールアウトの探索数（既定=本番）。再生は監査時の sims を使う")
    ap.add_argument("--engine", default="")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real"))
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cats = [c for c in args.categories.split(",") if c] or None
    suspects = load_suspects(args.suspects, args.max_suspects, cats, args.per_category)
    jobs = list(plan_replays(suspects).items())
    print(f"手の監査 段2: 容疑者{len(suspects)}点／{len(jobs)}局・世界{args.worlds}・"
          f"選択肢上限{args.max_options}・sims={args.sims}", flush=True)

    # 結果は**seed 単位で即座に書き出す**（1判断点で十数ロールアウト＝打ち切りが現実的に起きる。
    # 最後にまとめて書くと打ち切りで全損する＝アリーナ台帳と同じ教訓）。
    out_f = open(args.out, "w") if args.out else None
    rows = []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.engine, args.leaders, args.decks, args.sims,
                           args.worlds, args.max_options)) as pool:
        for res in pool.imap_unordered(_run_seed, jobs):
            for r in res:
                rows.append(r)
                if out_f:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_f.flush()
                if r.get("error"):
                    print(f"  seed {r['seed']}@{r.get('decision')}: {r['error']}", flush=True)
                else:
                    tag = ("飽和" if r.get("saturated")
                           else (f"regret {r['regret']:+.3f}" if r.get("regret") is not None
                                 else "打った手が候補に無い"))
                    print(f"  seed {r['seed']}@{r['decision']} {r['category']}: {tag} "
                          f"（選択肢{r['n_options']}）", flush=True)

    print("\nカテゴリ別の平均 regret（n / 飽和 / 未測定）:")
    for k, v in sorted(summarize(rows).items(), key=lambda kv: -(kv[1]["mean_regret"] or -1)):
        print(f"  {k:<10} regret={v['mean_regret']}  n={v['n']}  "
              f"飽和={v['saturated']}  未測定={v['skipped']}")
    worst = sorted((r for r in rows if r.get("regret")), key=lambda r: -r["regret"])[:10]
    if worst:
        print("\nregret 上位（段3の裁定候補）:")
        for r in worst:
            print(f"  {r['regret']:+.3f} seed {r['seed']}@{r['decision']} {r['category']} "
                  f"打={r['chosen']} 最良={r['best']}")
    if out_f:
        out_f.close()
        print(f"\n結果 -> {args.out}（seed ごとに逐次書き出し済み）")
    print("MOVE_REGRET_DONE " + json.dumps(
        {"measured": sum(1 for r in rows if r.get("regret") is not None),
         "saturated": sum(1 for r in rows if r.get("saturated"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
