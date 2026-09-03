"""手の監査 段1: 安い一次フィルタ（ロールアウト無し・2026-08-17）。

**問い**: CPU が打った手は正しかったか。全判断点を反実仮想で測るのは高すぎるので、
3段ファネルの1段目として**ロールアウト無しの信号だけで容疑者を絞る**。

各判断点で次を突き合わせる（すべて `cpu_learned` のトレース＝観測専用）:
  - CPU が実際に打った手（`chosen`）
  - **L1（古典CPU）の手**（`l1_move`）＝全く別の評価軸の第二意見
  - **policy の事前分布での順位**（`policy_rank`）＝1 なら policy の第一候補
  - **箱の出口評価での「打った手 vs 最良手」の価値差**（`q_gap`）

容疑者の条件（`classify_suspect`・純関数＝テストで固定）:
  - `three_way`     … CPU / L1 / policy の三者が食い違う（最も情報量が多い）
  - `toss_up`       … **1位と2位の Q 差**（q_margin）がほぼ 0＝どちらでもよいと読んでいる
  - `policy_low`    … policy が低く見ていた手を探索が選んだ（policy_rank ≥ 3）
  - `off_top_q`     … 読み出しが Q 最良でない手を選んだ（root 乗り換え・箱の出口）

`q_gap`（打った手 vs 最良手）は CPU がほぼ常に最良 Q を選ぶため中央値 0 で、**迷いの指標に
ならない**（実測）。迷いは 1位と2位の差＝`q_margin` で見る。L1 との不一致は単独では 4割に出て
絞り込みにならないので、容疑者条件としては three_way（policy とも食い違う）でのみ使う。

**L1 は効果の対話では第二意見にならない**（実測: 効果選択カテゴリの L1 不一致率 0%）。
対話は選択肢の意味づけが評価関数の外にあるため、そのカテゴリで three_way は立たない
＝効果選択の容疑者は toss_up / policy_low / off_top_q で拾う。

**段2以降との接続**: 各行は `(seed, decision)` を持つ。対局は seed で決定論再生できるので、
`run_game(..., stop_after_decisions=decision)` でその判断点の直前まで再生すれば局面を厳密に
復元できる（トレースは乱数を消費しない＝`_fill_trace` の乱数状態ガード）。段2はそこから
選択肢ごとに同一世界・共通乱数で終局まで打ち分け、regret を測る。

**この監査の値打ちは集計にある**: 個々の手の正誤より「攻撃判断の平均 q_gap 0.03・防御 0.11」の
ようにカテゴリ別の損失が出ることで、次にどこへ物量を投じるかがデータで決まる。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/move_audit.py \\
    --games 8 --workers 4 --leaders random --decks synth --out /tmp/move_audit.jsonl
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

# 容疑者判定のしきい値（純関数の引数既定＝テストで固定する）。
TOSS_UP_MARGIN = 0.01     # 1位と2位の Q 差。枝間マージン実測（0.02〜0.03）の半分＝実質同着
POLICY_LOW_RANK = 3       # policy の3番手以下を探索が選んだ
OFF_TOP_Q = 0.001         # 打った手が最良 Q でない（読み出しが Q 以外の理由で選んだ）
DECIDED_ABS = 0.9         # |Q| がこれ以上＝勝敗がほぼ決している（何を選んでも同じ）

_G = {}


def category_of(row):
    """判断点のカテゴリ（展開/攻撃/ドン付与/効果選択/防御/終了）。

    集計の単位。行動種別だけでは「効果の対話」と「メインの選択」が混ざるので、
    対話種別（dialog）があればそちらを優先する。
    """
    # dialog は MAIN_ACTION（＝メイン手番そのもの）にも付くので、**効果の対話種別**だけを見る。
    dlg = (row.get("dialog") or "").upper()
    if dlg in ("SEARCH_AND_SELECT", "CONFIRM_OPTIONAL", "ARRANGE_DECK", "SELECT_RESOURCE",
               "CHOICE", "DECLARE_COST"):
        return "効果選択"
    if dlg == "MULLIGAN":
        return "マリガン"
    at = (row.get("action_type") or "").upper()
    if at == "RESOLVE_EFFECT_SELECTION":
        return "効果選択"      # dialog が採れなかった対話（トレース欠測）も効果選択に寄せる
    if at in ("SELECT_COUNTER", "SELECT_BLOCKER") or (row.get("kind") == "battle"):
        return "防御"
    if at == "ATTACK":
        return "攻撃"
    if at == "ATTACH_DON":
        return "ドン付与"
    if at in ("PLAY", "PLAY_CARD"):
        return "展開"
    if at == "ACTIVATE_MAIN":
        return "起動メイン"
    if at == "TURN_END":
        return "ターン終了"
    return at or "その他"


def classify_suspect(row, toss_up_margin=TOSS_UP_MARGIN, policy_low_rank=POLICY_LOW_RANK,
                     off_top_q=OFF_TOP_Q, decided_abs=DECIDED_ABS):
    """容疑者フラグの集合を返す（純関数・ロールアウト無しの信号だけで決める）。

    `row` は監査行（chosen/l1_move/l1_disagrees/policy_rank/policy_top/q_gap）。
    情報が欠けている項目は「その条件では容疑者にしない」（欠測を疑いに数えない）。
    """
    # **勝敗がほぼ決している点は測らない**（段2 の実測: 飽和した判断点は全選択肢 wr=1.000＝
    # 何を選んでも勝つ局面だった）。高価なロールアウトを18本使っても「判別不能」しか返らない。
    val = row.get("value")
    if val is not None and abs(val) >= decided_abs:
        return set()

    flags = set()
    gap, margin = row.get("q_gap"), row.get("q_margin")
    rank = row.get("policy_rank")
    l1_diff = bool(row.get("l1_disagrees"))
    chosen, ptop = row.get("chosen"), row.get("policy_top")
    policy_diff = bool(ptop and chosen and ptop != chosen)

    if l1_diff and policy_diff:
        flags.add("three_way")
    if margin is not None and margin <= toss_up_margin:
        flags.add("toss_up")
    if rank is not None and rank >= policy_low_rank:
        flags.add("policy_low")
    if gap is not None and gap > off_top_q:
        flags.add("off_top_q")
    return flags


# 容疑者の優先度（段2は高価なので上位だけ回す）。三者食い違いが最も情報量が多い。
SUSPECT_WEIGHT = {"three_way": 3.0, "policy_low": 2.0, "off_top_q": 1.5, "toss_up": 1.0}


def priority(row):
    """容疑者の優先度スコア（純関数・降順に段2へ回す）。"""
    return round(sum(SUSPECT_WEIGHT.get(f, 0.0) for f in (row.get("suspect") or [])), 3)


class _Collector:
    """observer: 判断点ごとにトレースを1行へ畳む（manager は触らない＝決定論契約）。"""

    def __init__(self, seed, cond=None):
        self.seed = seed
        self.cond = cond or {}     # 測定条件（段2 が**同じ条件で再生**するために必要）
        self.rows = []

    def on_decision(self, ctx, move):
        tr = dict(ctx.trace or {})
        row = {
            "seed": self.seed,
            "decision": len(self.rows) + 1,        # 1始まり＝run_game(stop_after_decisions=n) と対応
            "turn": tr.get("turn", getattr(ctx.manager, "turn_count", None)),
            "seat": getattr(ctx.actor, "name", None),
            "kind": move.get("kind"),
            "action_type": move.get("action_type"),
            "dialog": tr.get("dialog"),
            "chosen": tr.get("chosen"),
            "value": tr.get("value"),
            "q_gap": tr.get("q_gap"),
            "q_margin": tr.get("q_margin"),
            "policy_rank": tr.get("policy_rank"),
            "policy_top": tr.get("policy_top"),
            "l1_move": tr.get("l1_move"),
            "l1_disagrees": tr.get("l1_disagrees"),
            "n_candidates": len(tr.get("candidates") or []),
            "readout": tr.get("readout"),
            # 測定条件。段2 は `(seed, decision)` で局面を復元するが、**再生は同じ条件**
            # （同じ sims・同じ対面/デッキ・同じネット）でないと対局が分岐して別の判断点に
            # 着地する（実測: sims を変えたら席までずれた）。行に埋めて持ち回る。
            **self.cond,
        }
        row["category"] = category_of(row)
        row["suspect"] = sorted(classify_suspect(row))
        row["priority"] = priority(row)
        self.rows.append(row)


def _init(sims, engine_spec, leaders_mode, decks, max_steps):
    from cpu_arena import _load_db
    _G["db"] = _load_db()
    _G["sims"] = sims
    _G["leaders"] = leaders_mode
    _G["decks"] = decks
    _G["max_steps"] = max_steps
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    _G["engine_spec"] = engine_spec
    if engine_spec:
        v, _, p = engine_spec.partition(",")
        _G["engine"] = LearnedEngine(value_path=v, policy_path=p or None)
    else:
        _G["engine"] = LearnedEngine()          # 出荷既定


def _audit_one(seed):
    """1局を本番仕様で打ち、全判断点の監査行を返す（子プロセス）。"""
    from game_driver import run_game, make_seat, leader_deck_builder
    from promotion_gate import _leader_pair
    db, sims = _G["db"], _G["sims"]
    la, lb = _leader_pair(db, seed, _G["leaders"])
    if _G["decks"] == "synth" and la:
        from deck_synth import synth_deck_builder
        builder = synth_deck_builder(la, lb, seed=seed)
    else:
        builder = leader_deck_builder(la, lb) if la else None
    col = _Collector(seed, cond={"audit_sims": sims, "audit_leaders": _G["leaders"],
                                "audit_decks": _G["decks"], "audit_engine": _G["engine_spec"]})
    seat = make_seat(kind="learned", want_trace=True, sims=sims, engine=_G["engine"])
    try:
        res = run_game(seed, db, seats={"p1": seat, "p2": seat}, deck_builder=builder,
                       observers=(col,), max_steps=_G["max_steps"],
                       legal_moves="skip", invariants="raise")
        winner = getattr(res, "winner", None)
    except BaseException as e:                   # 1局の失敗で監査全体を止めない
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": col.rows,
                "leaders": [la, lb]}
    for r in col.rows:
        r["winner"] = winner
    return {"seed": seed, "rows": col.rows, "leaders": [la, lb], "winner": winner}


def summarize(rows):
    """カテゴリ別の集計（純関数）: 判断点数・平均 |q_gap|・容疑者率・L1不一致率。"""
    by = collections.defaultdict(lambda: {"n": 0, "gap_sum": 0.0, "gap_n": 0,
                                          "suspects": 0, "l1_diff": 0})
    for r in rows:
        b = by[r.get("category") or "その他"]
        b["n"] += 1
        if r.get("q_margin") is not None:
            b["gap_sum"] += abs(r["q_margin"])
            b["gap_n"] += 1
        if r.get("suspect"):
            b["suspects"] += 1
        if r.get("l1_disagrees"):
            b["l1_diff"] += 1
    out = {}
    for k, b in by.items():
        out[k] = {"n": b["n"],
                  "mean_margin": round(b["gap_sum"] / b["gap_n"], 4) if b["gap_n"] else None,
                  "suspect_rate": round(b["suspects"] / b["n"], 3) if b["n"] else None,
                  "l1_diff_rate": round(b["l1_diff"] / b["n"], 3) if b["n"] else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--sims", type=int, default=160, help="本番既定（監査は本番仕様で測る）")
    ap.add_argument("--engine", default="", help="value.npz[,policy.npz]（既定=出荷既定ネット）")
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real"))
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth"))
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="監査行の jsonl（段2の入力）")
    ap.add_argument("--top", type=int, default=0,
                    help="出力を優先度上位N件の容疑者に絞る（0=全判断点を書く）")
    args = ap.parse_args()

    seeds = [args.seed_base + i for i in range(args.games)]
    print(f"手の監査 段1: {len(seeds)}局・sims={args.sims}・{args.leaders}/{args.decks}", flush=True)
    rows, errors = [], []
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.sims, args.engine, args.leaders, args.decks, args.max_steps)) as pool:
        for r in pool.imap_unordered(_audit_one, seeds):
            rows.extend(r["rows"])
            if r.get("error"):
                errors.append((r["seed"], r["error"]))
                print(f"  seed {r['seed']}: {r['error']}", flush=True)
            print(f"  seed {r['seed']} {r.get('leaders')}: 判断{len(r['rows'])}点 "
                  f"容疑者{sum(1 for x in r['rows'] if x['suspect'])}", flush=True)

    kinds = collections.Counter(f for r in rows for f in r["suspect"])
    print(f"\n判断点 {len(rows)} / 容疑者 {sum(1 for r in rows if r['suspect'])}"
          f"（{100.0 * sum(1 for r in rows if r['suspect']) / max(1, len(rows)):.1f}%）")
    print("容疑者の内訳:", dict(kinds))
    print("\nカテゴリ別（n / 平均マージン / 容疑者率 / L1不一致率）:")
    for k, v in sorted(summarize(rows).items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {k:<10} {v['n']:>5}  margin={v['mean_margin']}  "
              f"suspect={v['suspect_rate']}  l1diff={v['l1_diff_rate']}")
    if errors:
        print(f"\n成立しなかった対局 {len(errors)} 件: {errors[:3]}")
    if args.out:
        out_rows = rows
        if args.top:
            out_rows = sorted((r for r in rows if r["suspect"]),
                              key=lambda r: (-r["priority"], r["seed"], r["decision"]))[:args.top]
        with open(args.out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n監査行 {len(out_rows)}件 -> {args.out}")
    print("MOVE_AUDIT_DONE " + json.dumps(
        {"rows": len(rows), "suspects": sum(1 for r in rows if r["suspect"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
