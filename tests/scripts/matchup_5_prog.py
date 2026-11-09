"""対面記事 6 対面 720 局（`matchup_5.py play`／`trace` の記録）の進行を、勝ち負けで分けて文面に起こす
（WP `matchup-5` の補助・読み取りだけ・1 回きりの計測＝テスト無し）。`matchup_kr_prog.py` を 6 対面に一般化したもの。

    python tests/scripts/matchup_5_prog.py > out.md    # 6 対面の集計と代表局（標準出力に Markdown）

`trace_<対面>.jsonl` は `matchup_5.py trace`（`--trace-dir` を変えたときは `M5_TRACE_DIR` で指す）。
視点の側は常に p1（表の「勝ち局・負け局」は視点の側から見た勝ち負け）。
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matchup_5 as M5  # noqa: E402

TR_DIR = os.environ.get("M5_TRACE_DIR", M5.OUT_DIR)
MIN_N = 15     # 型ごとの勝率の表に出す最小の局数
MAX_TURN = 8
#: 型の表で見る「自分の何ターン目までに」の上限（遅いターンほど「負けているから使った」の逆向きの因果が混ざるので序盤に絞る）
FEAT_TURN = 4
_NAMES = {}


def nm(cid):
    """カード DB の名前（コスト）。同名・同コストの別の札はカード番号で分ける。"""
    if not _NAMES:
        db = json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))
        seen = collections.defaultdict(set)
        for x in db:
            seen[(x.get("name"), x.get("コスト"))].add(x["number"].split("_")[0])
        for x in db:
            k = (x.get("name"), x.get("コスト"))
            base = f"{k[0]}({k[1]})" if k[1] is not None else str(k[0])
            _NAMES[x["number"]] = base if len(seen[k]) == 1 else f"{base}[{x['number']}]"
    return _NAMES.get(cid, str(cid))


def side_name(mu, seat):
    a, b = M5.MU[mu][3].split(" 対 ") if " 対 " in M5.MU[mu][3] else ("視点の" + M5.MU[mu][3].split()[0], "相手の" + M5.MU[mu][3].split()[0])
    return a if seat == "p1" else b


def turns_of(t, ig):
    """ターンごとの要約（手番の席・出した札・起動・リーダー効果・アタック・ターン末のライフ・手札・場）。"""
    mf = t["me_first"]
    lname = {s: (nm(M5.deck(M5.MU[t["mu"]][0 if s == "p1" else 1])[0]).split("(")[0]) for s in ("p1", "p2")}
    out = {}
    for r in t["rows"]:
        tn = r["turn"]
        if tn == 0:
            continue
        tp = r["tp"]
        d = out.setdefault(tn, {"turn": tn, "tp": tp, "own_turn": M5.own_turn(tn, tp, mf), "plays": [], "acts": [],
                                "leader": {"p1": 0, "p2": 0}, "attacks": []})
        mv = r["mv"]
        seat = r["seat"]
        if mv:
            is_leader = mv.get("uuid") and mv.get("uuid") == t["luid"].get(seat)
            if mv["at"] == "ACTIVATE_MAIN" and is_leader:
                d["leader"][seat] += 1
            elif (r["act"] == "CONFIRM_OPTIONAL" and mv.get("acc") and lname[seat]
                  and r["msg"].startswith(f"「{lname[seat]}」の効果")):
                d["leader"][seat] += 1
        if seat == tp and mv:
            if mv["at"] == "PLAY":
                d["plays"].append(nm(mv.get("card")))
            elif mv["at"] == "ACTIVATE_MAIN" and not (mv.get("uuid") and mv.get("uuid") == t["luid"].get(seat)):
                d["acts"].append("起動:" + nm(mv.get("card")))
            elif mv["at"] == "ATTACK":
                src = "リーダー" if mv.get("uuid") == t["luid"].get(seat) else nm(mv.get("card"))
                tg = (mv.get("tgt") or ["?"])[0]
                other = "p2" if seat == "p1" else "p1"
                tg_name = "リーダー" if tg == M5.deck(M5.MU[t["mu"]][0 if other == "p1" else 1])[0] else nm(tg)
                d["attacks"].append(f"{src}→{tg_name}")
        st = r["st"]
        d["life"] = (st["p1"]["life"], st["p2"]["life"])
        d["hand"] = (len(st["p1"]["hand"]), len(st["p2"]["hand"]))
        d["field"] = ([nm(c[0]) for c in st["p1"]["field"]], [nm(c[0]) for c in st["p2"]["field"]])
        d["hand_p1"] = [nm(c) for c in st["p1"]["hand"]]
    vals = ig["values"]
    for tn, d in out.items():
        vs = [v for s_, t_, v in vals if t_ == tn]
        d["v"] = round(vs[-1], 2) if vs else None
    return [out[k] for k in sorted(out)]


def narrate(mu, t, ig):
    ts = turns_of(t, ig)
    me, opp = side_name(mu, "p1"), side_name(mu, "p2")
    head = (f"#### 局 {t['idx']}（{me}{'先攻' if t['me_first'] else '後攻'}・"
            f"{me + 'の勝ち' if t['me_won'] else opp + 'の勝ち'}・{t['turns']} ターン）")
    lines = [head, "", f"| ターン | 手番 | 出した札・起動 | リーダー効果（視点／相手） | アタック | ターン末 ライフ 視点/相手 | 手札 視点/相手 | 視点の評価 | 視点の手札（視点の手番のみ） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in ts:
        who = me if d["tp"] == "p1" else opp
        plays = "・".join(d["plays"] + d["acts"]) or "—"
        att = "、".join(d["attacks"]) or "—"
        ld = f"{d['leader']['p1'] or '—'}／{d['leader']['p2'] or '—'}"
        lines.append(f"| {d['turn']}（{who}{d['own_turn']}） | {who} | {plays} | {ld} | {att} | {d['life'][0]}/{d['life'][1]} | "
                     f"{d['hand'][0]}/{d['hand'][1]} | {d['v']} | {'・'.join(d['hand_p1']) if d['tp'] == 'p1' else ''} |")
    last = ts[-1]
    lines += ["", f"終局の場: 視点 {'・'.join(last['field'][0]) or '（無し）'} ／ 相手 {'・'.join(last['field'][1]) or '（無し）'}", ""]
    return "\n".join(lines)


def analyse(mu):
    games = M5.load_games(M5.OUT_DIR, mu)
    per = {r["idx"]: r for r in M5.read_jsonl(os.path.join(M5.OUT_DIR, f"items_{mu}.jsonl"))}
    tr = {t["idx"]: t for t in M5.read_jsonl(os.path.join(TR_DIR, f"trace_{mu}.jsonl"))}
    idxs = sorted(i for i in tr if not games[(mu, i)]["void"])
    T = {i: turns_of(tr[i], per[i]) for i in idxs}
    won = [i for i in idxs if tr[i]["me_won"]]
    lost = [i for i in idxs if not tr[i]["me_won"]]
    me, opp = side_name(mu, "p1"), side_name(mu, "p2")
    out = [f"## {M5.MU[mu][3]}（{me}視点・{len(idxs)} 局・{me}の勝ち {len(won)}・負け {len(lost)}）", ""]
    fw = [i for i in idxs if tr[i]["me_first"]]
    sw = [i for i in idxs if not tr[i]["me_first"]]
    out.append(f"{me}の勝率: 全体 {len(won) / len(idxs):.2f}・先攻 {sum(tr[i]['me_won'] for i in fw) / max(len(fw), 1):.2f}"
               f"・後攻 {sum(tr[i]['me_won'] for i in sw) / max(len(sw), 1):.2f}。"
               f"局の長さ 勝ち局 {sum(tr[i]['turns'] for i in won) / max(len(won), 1):.1f}・負け局 {sum(tr[i]['turns'] for i in lost) / max(len(lost), 1):.1f} ターン。")
    out.append("")

    def own(i, ot, seat):
        for d in T[i]:
            if d["tp"] == seat and d["own_turn"] == ot:
                return d
        return None

    for seat in ("p1", "p2"):
        sname = me if seat == "p1" else opp
        out.append(f"### {sname}が自分の何ターン目に何を出したか（{me}の勝ち局 vs 負け局）")
        out.append("")
        out.append(f"| 何ターン目 | {me}の勝ち局でよく出した札 | {me}の負け局でよく出した札 | 勝ち局: {sname}のリーダー効果を使った局 / アタック平均 | 負け局: 同 |")
        out.append("|---|---|---|---|---|")
        for ot in range(1, MAX_TURN + 1):
            cells = []
            for grp in (won, lost):
                c = collections.Counter()
                ld = at = n = 0
                for i in grp:
                    d = own(i, ot, seat)
                    if not d:
                        continue
                    n += 1
                    for p in set(d["plays"]):
                        c[p] += 1
                    ld += d["leader"][seat] > 0
                    at += len(d["attacks"])
                cells.append((c, ld, at, n))
            (cw, lw, aw, nw), (cl, ll, al, nl) = cells
            f = lambda c, n: ("、".join(f"{k} {v}" for k, v in c.most_common(5)) or "—") + f"（局 {n}）"
            out.append(f"| {ot} | {f(cw, nw)} | {f(cl, nl)} | {lw}/{nw} / {aw / max(nw, 1):.1f} | {ll}/{nl} / {al / max(nl, 1):.1f} |")
        out.append("")
    out.append(f"### ライフと手札の推移（{me}の自分の何ターン目の末・平均）")
    out.append("")
    out.append(f"| 何ターン目 | 勝ち局: {me}ライフ / {opp}ライフ / {me}手札 / {opp}手札 / {me}の場 / {opp}の場 | 負け局: 同 |")
    out.append("|---|---|---|")
    for ot in range(1, MAX_TURN + 1):
        cells = []
        for grp in (won, lost):
            ds = [d for d in (own(i, ot, "p1") for i in grp) if d]
            if not ds:
                cells.append("—")
                continue
            m = lambda f: sum(f(d) for d in ds) / len(ds)
            cells.append(f"{m(lambda d: d['life'][0]):.1f} / {m(lambda d: d['life'][1]):.1f} / {m(lambda d: d['hand'][0]):.1f} / "
                         f"{m(lambda d: d['hand'][1]):.1f} / {m(lambda d: len(d['field'][0])):.1f} / {m(lambda d: len(d['field'][1])):.1f}（局 {len(ds)}）")
        out.append(f"| {ot} | {cells[0]} | {cells[1]} |")
    out.append("")
    # 型ごとの勝率: 「札 c を自分の N ターン目に出した（その札を初めて出したのが N）」と「出さなかった」
    base = len(won) / len(idxs)
    feats = []
    for seat in ("p1", "p2"):
        first = {}
        for i in idxs:
            f = {}
            for d in T[i]:
                if d["tp"] != seat:
                    continue
                for p in d["plays"]:
                    f.setdefault(p, d["own_turn"])
            first[i] = f
        cards = collections.Counter(c for i in idxs for c in first[i])
        for c, _n in cards.items():
            for ot in range(1, FEAT_TURN + 1):
                s = [i for i in idxs if first[i].get(c, 99) <= ot]
                if MIN_N <= len(s) <= len(idxs) - MIN_N:
                    r = sum(tr[i]["me_won"] for i in s) / len(s)
                    rest = [i for i in idxs if i not in set(s)]
                    rr = sum(tr[i]["me_won"] for i in rest) / len(rest)
                    feats.append((r - rr, seat, c, ot, len(s), r, len(rest), rr))
        # リーダー効果を自分の N ターン目までに使った
        for ot in range(1, FEAT_TURN + 1):
            s = [i for i in idxs if any(d["leader"][seat] for d in T[i]
                                        if M5.own_turn(d["turn"], seat, tr[i]["me_first"]) <= ot)]
            if MIN_N <= len(s) <= len(idxs) - MIN_N:
                r = sum(tr[i]["me_won"] for i in s) / len(s)
                rest = [i for i in idxs if i not in set(s)]
                rr = sum(tr[i]["me_won"] for i in rest) / len(rest)
                feats.append((r - rr, seat, "（リーダー効果）", ot, len(s), r, len(rest), rr))
    # 同じ札で一番差の大きいターンだけ残す
    best = {}
    for f in feats:
        k = (f[1], f[2])
        if k not in best or abs(f[0]) > abs(best[k][0]):
            best[k] = f
    top = sorted(best.values(), key=lambda f: -abs(f[0]))[:14]
    out.append(f"### 勝ち負けを分けた進行（{me}の勝率の差が大きい順・局数 {MIN_N} 以上・全体の勝率 {base:.2f}）")
    out.append("")
    out.append(f"| 進行 | した局 | その局の{me}の勝率 | しなかった局 | その局の{me}の勝率 | 差 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for dv, seat, c, ot, n, r, nr, rr in top:
        who = me if seat == "p1" else opp
        what = f"{who}が自分の {ot} ターン目までに{c}を使った" if c == "（リーダー効果）" else f"{who}が {c} を自分の {ot} ターン目までに出した"
        out.append(f"| {what} | {n} | {r:.2f} | {nr} | {rr:.2f} | {dv:+.2f} |")
    out.append("")
    # 代表局: 勝ち局で長さが中ほどのもの・負け局で一度は優勢だったのに負けたもの・負け局で長さが中ほどのもの
    def swing(i):
        vs = [v for _, _, v in per[i]["values"]]
        return max(vs) if vs else 0
    picks = []
    if won:
        mlen = sorted(tr[i]["turns"] for i in won)[len(won) // 2]
        picks.append(("勝ち局の典型（長さが中ほど）", min(won, key=lambda i: (abs(tr[i]["turns"] - mlen), i))))
    if lost:
        picks.append(("一度は優勢だったのに負けた局", max(lost, key=lambda i: (swing(i), -i))))
        mlen = sorted(tr[i]["turns"] for i in lost)[len(lost) // 2]
        rest = [i for i in lost if i != picks[-1][1]]
        if rest:
            picks.append(("負け局の典型（長さが中ほど）", min(rest, key=lambda i: (abs(tr[i]["turns"] - mlen), i))))
    out.append("### 代表局")
    out.append("")
    for tag, i in picks:
        out.append(f"- {tag}: 局 {i}")
    out.append("")
    for tag, i in picks:
        out.append(narrate(mu, tr[i], per[i]))
    return "\n".join(out)


def main():
    sel = [x for x in os.environ.get("M5_MU", "").split(",") if x] or list(M5.MU)
    for mu in sel:
        print(analyse(mu))
        print()


if __name__ == "__main__":
    main()
