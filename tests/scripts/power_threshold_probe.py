"""打点しきい値の未達スキャン（v48・2026-08-10・読み取り専用）。

**問い**: そのターン、**しきい値以上の打点を作れたのに作らなかった**か。

勝率での裁定裏取りは、飽和負け局面では 12世界でも ±1勝に揺れて識別できない（v48 実測: エネル
turn 3 のプラン比較で人間の線と CPU の線が同価値バンド内）。一方「作れた打点」は**盤面と
ドン枚数から決定論的に決まる**ので、ノイズゼロで数えられる。

ユーザ裁定（2026-08-10）: 「エネルがナミに勝つには 7000 以上で殴る必要がある。キャラクターの
エネル(OP15-118・8000)を出すのもパワー10000 が重要だから」。相手リーダー 5000 に対し、
カウンター1000 を1枚使われると 6000 で止まる。7000 あればカウンター2枚を要求できるし、
相手場の 6000 ブロッカー（EB03-053 / OP15-113 等）も抜ける。**打点のしきい値が勝敗を分ける
対面**では、勝率の平均より「しきい値を跨げたか」のほうが信号が濃い。

**計算**（リプレイのフレームのみ・エンジン不要）:
  - 到達可能打点 = max( 攻撃可能なカードの素パワー + アクティブドン枚数 × 1000 )
    攻撃可能 = ターン開始時に場にいて（＝召喚酔いでない）レストでないキャラ、およびレストでない
    リーダー。付与ドンはターン開始時 0（リフレッシュで戻る）なので素パワーで数える。
  - 実際の打点 = そのターンの ATTACK 行動の攻撃側パワーの最大値
    （フレームのパワーは手番側のみ付与ドンを加算する presenter 規約なので、そのまま読める）
  - **未達** = 到達可能 ≥ しきい値 かつ 実際 < しきい値

**射程の限定**: リーダー能力やイベントによる一時的なパワー増（エネルの「レストのドン4枚まで
付与」で**レスト**のドンも打点になる等）は数えない＝到達可能打点は**過小評価**。つまり本器が
「未達」と言う点は確実に未達だが、未達を全部拾えているわけではない（偽陽性を出さない側に倒す）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/power_threshold_probe.py \\
    --replays e1,e2 --leader OP15-058 --threshold 7000
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
import coach_gate as CG  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402


def _side_at(fr, pid):
    return (fr.get("players") or {}).get(pid) or {}


def scan(raw, focus_leader, threshold):
    """1リプレイを走査し、自ターンごとの (到達可能, 実際, 未達) を返す。"""
    rec = raw.get("replay", raw)
    acts = rec["actions"]
    frames = raw.get("frames") or []
    byidx = {f.get("action_index"): f for f in frames if f.get("action_index") is not None}
    init = next((f for f in frames if f.get("action_index") is None), None)
    leaders = rec.get("leaders") or {}
    pid = next((p for p in ("p1", "p2") if leaders.get(p) == focus_leader), None)
    if pid is None:
        return None

    def before(k):
        return byidx.get(k - 1, init)

    turns = {}
    for i, a in enumerate(acts):
        if a.get("player") == pid:
            turns.setdefault(int(a.get("turn", 0) or 0), []).append(i)

    rows = []
    for t in sorted(turns):
        idxs = turns[t]
        start = before(idxs[0])
        if start is None:
            continue
        s = _side_at(start, pid)
        don = int(s.get("don_active", 0) or 0)
        cands = []
        ld = s.get("leader") or {}
        if ld and not ld.get("is_rest"):
            cands.append((ld.get("card_id"), int(ld.get("power", 0) or 0)))
        for c in s.get("field") or []:
            if not c.get("is_rest"):
                cands.append((c.get("card_id"), int(c.get("power", 0) or 0)))
        if not cands:
            continue
        reach_card, base = max(cands, key=lambda x: x[1])
        reachable = base + don * 1000

        actual, actual_card = 0, None
        for i in idxs:
            if acts[i].get("action_type") != "ATTACK":
                continue
            fr = before(i)
            if fr is None:
                continue
            sb = _side_at(fr, pid)
            cid = acts[i].get("card")
            pw = 0
            if (sb.get("leader") or {}).get("card_id") == cid:
                pw = int((sb.get("leader") or {}).get("power", 0) or 0)
            for c in sb.get("field") or []:
                if c.get("card_id") == cid:
                    pw = max(pw, int(c.get("power", 0) or 0))
            if pw > actual:
                actual, actual_card = pw, cid
        rows.append({"turn": t, "don": don, "reachable": reachable, "reach_card": reach_card,
                     "actual": actual, "actual_card": actual_card,
                     "gap": reachable >= threshold and actual < threshold})
    return {"pid": pid, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="e1,e2", help="coach_gate のリプレイタグ（カンマ区切り）")
    ap.add_argument("--leader", default="OP15-058", help="走査対象のリーダー card_id")
    ap.add_argument("--threshold", type=int, default=7000)
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48}
    tot_turns = tot_gap = 0
    for tag in [t.strip() for t in args.replays.split(",") if t.strip()]:
        raw = RE.load_replay_json(table[tag])
        r = scan(raw, args.leader, args.threshold)
        if r is None:
            print(f"{tag}: リーダー {args.leader} が居ない（skip）")
            continue
        print(f"\n=== {tag}（{args.leader} = {r['pid']}・しきい値 {args.threshold}）")
        print(f"  {'ターン':>4} {'活ドン':>5} {'到達可能':>8} {'実際':>7}   最大打点候補→実際の攻撃")
        for x in r["rows"]:
            tot_turns += 1
            tot_gap += 1 if x["gap"] else 0
            mark = "  ← 未達（作れたのに作っていない）" if x["gap"] else ""
            print(f"  {x['turn']:>4} {x['don']:>5} {x['reachable']:>8} {x['actual']:>7}"
                  f"   {x['reach_card']}→{x['actual_card']}{mark}")
    print(f"\n=== 合計: 自ターン {tot_turns} 件中 **{tot_gap} 件が未達**"
          f"（到達可能 ≥ {args.threshold} なのに実際 < {args.threshold}）")
    print("POWER_THRESHOLD_PROBE " + json.dumps(
        {"threshold": args.threshold, "turns": tot_turns, "gaps": tot_gap}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
