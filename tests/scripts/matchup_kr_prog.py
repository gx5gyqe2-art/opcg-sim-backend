"""紫カイドウ対青ロックス 200 局（`matchup_kr.py play`／`trace` の記録）の進行を、勝ち負けで分けて文面に起こす
（WP `matchup-kr` の補助・読み取りだけ・1 回きりの計測＝テスト無し）。

    python tests/scripts/matchup_kr_prog.py             # 集計（標準出力に Markdown）
    NARR=1,103 python tests/scripts/matchup_kr_prog.py  # 局 1・103 のターンごとの文面も付ける

`trace.jsonl` は `matchup_kr.py trace`（`--trace-dir` を変えたときは `KR_TRACE` で指す）。"""
import collections
import json
import os
import sys

sys.path.insert(0, "tests/scripts")
import matchup_kr as MK  # noqa: E402

TR = os.environ.get("KR_TRACE", os.path.join(MK.OUT_DIR, "trace.jsonl"))
NICK = {"EB04-031": "6キング", "EB04-032": "1クイーン", "OP17-074": "3ヤマト", "OP17-073": "ホーキンス",
        "OP17-062": "ブロッカーカイドウ", "OP17-063": "除去カイドウ", "OP17-061": "大看板", "OP17-065": "9クイーン",
        "OP17-075": "2ドレーク", "OP17-069": "9ジャック", "OP17-060": "うるティ&ページワン", "OP17-068": "3ササキ",
        "OP17-076": "ウォロロロ(カウンター)", "OP17-078": "酒龍八卦",
        "OP17-049": "5リンリン", "OP17-048": "シキ", "OP17-056": "ロックス海賊団(イベント)", "OP17-045": "凶",
        "OP17-044": "キャプテン・ジョン", "OP17-118": "10ロックス", "OP17-050": "シュトロイゼン", "OP17-052": "マーロン",
        "OP17-054": "ステューシー", "OP17-042": "4カイドウ(ブロッカー)", "OP17-040": "ニューゲート", "OP17-055": "ハチノス",
        "OP17-046": "グロリオーサ"}


def nm(cid):
    if cid in NICK:
        return NICK[cid]
    n, c, k = MK.cname(cid)
    return f"{n}({c})" if c is not None else str(n)


def games():
    g = MK.load_games(MK.OUT_DIR)
    per = {r["idx"]: r for r in MK.read_jsonl(os.path.join(MK.OUT_DIR, "items_games.jsonl"))}
    tr = {t["idx"]: t for t in MK.read_jsonl(TR)}
    return g, per, tr


def turns_of(t, ig):
    """ターンごとの要約（手番の席・出した札・起動・リーダー効果・アタック数・ターン末のライフ・手札・場）。"""
    kf = t["kaido_first"]
    out = {}
    for r in t["rows"]:
        tn = r["turn"]
        if tn == 0:
            continue
        d = out.setdefault(tn, {"turn": tn, "tp": r["tp"], "who": "紫" if r["tp"] == "p1" else "青",
                                "own_turn": MK.own_turn(tn, r["tp"], kf), "plays": [], "acts": [], "leader": 0,
                                "attacks": [], "events": []})
        mv = r["mv"]
        if r["seat"] == "p1" and r["act"] == "CONFIRM_OPTIONAL" and r["battle"] and r["msg"].startswith("「カイドウ」の効果を使用") and mv and mv.get("acc"):
            d["leader"] += 1
        if r["seat"] == r["tp"] and mv:
            if mv["at"] == "PLAY":
                d["plays"].append(nm(mv.get("card")))
            elif mv["at"] == "ACTIVATE_MAIN":
                c = mv.get("card")
                d["acts"].append("起動:" + nm(c))
            elif mv["at"] == "ATTACK":
                src = nm(mv.get("card")) if mv.get("card") and not mv.get("card", "").startswith(("OP17-058", "OP17-039")) else "リーダー"
                tgt = mv.get("tgt") or []
                tg = "リーダー" if (not tgt or tgt[0] in ("OP17-058", "OP17-039")) else nm(tgt[0])
                d["attacks"].append(f"{src}→{tg}")
        st = r["st"]
        d["life"] = (st["p1"]["life"], st["p2"]["life"])
        d["hand"] = (len(st["p1"]["hand"]), len(st["p2"]["hand"]))
        d["field"] = ([nm(c[0]) for c in st["p1"]["field"]], [nm(c[0]) for c in st["p2"]["field"]])
        d["don"] = (st["p1"]["don"], st["p2"]["don"])
        # 自分（紫）の手札の中身は紫の手番だけ出す
        d["hand_p1"] = [nm(c) for c in st["p1"]["hand"]]
    vals = ig["values"]
    for tn, d in out.items():
        vs = [v for s_, t_, v in vals if t_ == tn]
        d["v"] = round(vs[-1], 2) if vs else None
    return [out[k] for k in sorted(out)]


def narrate(t, ig, per_g):
    ts = turns_of(t, ig)
    kf = t["kaido_first"]
    head = f"### 局 {t['idx']}（紫カイドウ{'先攻' if kf else '後攻'}・{'紫の勝ち' if t['kaido_won'] else '青の勝ち'}・{t['turns']} ターン）"
    lines = [head, "", "| ターン | 手番 | 出した札・起動 | 紫のリーダー効果 | アタック | ターン末 ライフ 紫/青 | 手札 紫/青 | 紫の評価 | 紫の手札（ターン末・紫の手番のみ） |", "|---|---|---|---|---|---|---|---|---|"]
    for d in ts:
        plays = "・".join(d["plays"] + d["acts"]) or "—"
        att = "、".join(d["attacks"]) or "—"
        lines.append(f"| {d['turn']}（{d['who']}{d['own_turn']}） | {d['who']} | {plays} | {'使' * d['leader'] or '—'} | {att} | {d['life'][0]}/{d['life'][1]} | {d['hand'][0]}/{d['hand'][1]} | {d['v']} | {'・'.join(d['hand_p1']) if d['who'] == '紫' else ''} |")
    last = ts[-1]
    lines.append("")
    lines.append(f"終局の場: 紫 {last['field'][0] or '（無し）'} ／ 青 {last['field'][1] or '（無し）'}")
    return "\n".join(lines)


def main():
    g, per, tr = games()
    idxs = sorted(i for i in tr if not g[i]["void"])
    won = [i for i in idxs if g[i]["kaido_won"]]
    lost = [i for i in idxs if not g[i]["kaido_won"]]
    T = {i: turns_of(tr[i], per[i]) for i in idxs}

    def own(i, ot, who="紫"):
        for d in T[i]:
            if d["who"] == who and d["own_turn"] == ot:
                return d
        return None

    out = []
    out.append(f"局 {len(idxs)}（紫の勝ち {len(won)}・負け {len(lost)}）")
    # 1. ターンごとの紫の出し札（勝ち／負け）
    for who, label in (("紫", "紫カイドウ"), ("青", "青ロックス")):
        out.append(f"\n## {label}の自分の何ターン目に何を出したか（勝ち局 vs 負け局・紫視点の勝ち負け）\n")
        out.append("| 自分の何ターン目 | 勝ち局でよく出した札（局数） | 負け局でよく出した札（局数） | 勝ち局: 紫のリーダー効果をこのターンに使った局 / アタック平均 | 負け局: 同 |")
        out.append("|---|---|---|---|---|")
        for ot in range(1, 9):
            row = []
            stats = {}
            for name, grp in (("w", won), ("l", lost)):
                c = collections.Counter()
                ld, at, n = 0, 0, 0
                for i in grp:
                    d = own(i, ot, who)
                    if not d:
                        continue
                    n += 1
                    for p in set(d["plays"]):
                        c[p] += 1
                    ld += d["leader"] > 0 if who == "紫" else 0
                    at += len(d["attacks"])
                stats[name] = (c, ld, at, n)
            def fmt(c, n):
                return "、".join(f"{k} {v}" for k, v in c.most_common(5)) + f"（局 {n}）"
            cw, lw, aw, nw = stats["w"]
            cl, ll, al, nl = stats["l"]
            out.append(f"| {ot} | {fmt(cw, nw)} | {fmt(cl, nl)} | {lw}/{nw} / {aw / max(nw, 1):.1f} | {ll}/{nl} / {al / max(nl, 1):.1f} |")
    # 2. ライフの推移（紫の何ターン目の末）
    out.append("\n## ライフと手札の推移（紫カイドウの自分の何ターン目の末・平均）\n")
    out.append("| 紫の何ターン目 | 勝ち局: 紫ライフ / 青ライフ / 紫手札 / 青手札 / 紫の場の数 / 青の場の数 | 負け局: 同 |")
    out.append("|---|---|---|")
    for ot in range(1, 9):
        cells = []
        for grp in (won, lost):
            ds = [own(i, ot) for i in grp]
            ds = [d for d in ds if d]
            if not ds:
                cells.append("—")
                continue
            m = lambda f: sum(f(d) for d in ds) / len(ds)
            cells.append(f"{m(lambda d: d['life'][0]):.1f} / {m(lambda d: d['life'][1]):.1f} / {m(lambda d: d['hand'][0]):.1f} / {m(lambda d: d['hand'][1]):.1f} / {m(lambda d: len(d['field'][0])):.1f} / {m(lambda d: len(d['field'][1])):.1f}（局 {len(ds)}）")
        out.append(f"| {ot} | {cells[0]} | {cells[1]} |")
    # 3. 型ごとの勝率
    out.append("\n## 進行の型ごとの勝率（紫カイドウ）\n")
    out.append("| 型 | 局数 | 紫の勝率 |")
    out.append("|---|---:|---:|")

    def rate(sel, label):
        s = [i for i in idxs if sel(i)]
        if s:
            w = sum(g[i]["kaido_won"] for i in s)
            out.append(f"| {label} | {len(s)} | {w / len(s):.2f} |")

    def played(i, ot, card, who="紫"):
        d = own(i, ot, who)
        return bool(d and card in d["plays"])

    def first_turn(i, cards, who="紫"):
        for d in T[i]:
            if d["who"] == who and any(c in d["plays"] for c in cards):
                return d["own_turn"]
        return None

    BIG = ["ブロッカーカイドウ", "除去カイドウ", "大看板", "9クイーン", "9ジャック"]
    rate(lambda i: played(i, 3, "6キング"), "3 ターン目に 6キング")
    rate(lambda i: not played(i, 3, "6キング"), "3 ターン目に 6キングなし")
    for ot in (4, 5, 6, 7):
        rate(lambda i, ot=ot: first_turn(i, BIG) == ot, f"最初の大型（9〜10 コスト）が {ot} ターン目")
    rate(lambda i: first_turn(i, BIG) is None, "大型を一度も出さず")
    for c in BIG:
        rate(lambda i, c=c: any(c in d["plays"] for d in T[i] if d["who"] == "紫"), f"{c} を出した局")
    for n in (0, 1, 2, 3):
        rate(lambda i, n=n: min(3, sum(d["plays"].count("ブロッカーカイドウ") + d["plays"].count("除去カイドウ") for d in T[i] if d["who"] == "紫")) == n, f"10 コストのカイドウを {n}{'+' if n == 3 else ''} 枚出した")
    for ot in (2, 3, 4):
        rate(lambda i, ot=ot: any(d["leader"] for d in T[i] if d["who"] == "紫" and d["own_turn"] <= ot and not any(("6キング" in e["plays"]) for e in T[i] if e["who"] == "紫" and e["own_turn"] < d["own_turn"])), f"6キングより前（〜{ot} ターン目）にリーダー効果を使った")
    rate(lambda i: not any(d["leader"] for d in T[i] if d["who"] == "紫" and d["own_turn"] <= 3), "3 ターン目までリーダー効果を使わず")
    # 青側
    for ot in (3, 4, 5, 6):
        rate(lambda i, ot=ot: first_turn(i, ["シキ"], "青") == ot, f"青がシキを {ot} ターン目に出した")
    rate(lambda i: first_turn(i, ["シキ"], "青") is None, "青がシキを出さず")
    rate(lambda i: any("10ロックス" in d["plays"] for d in T[i] if d["who"] == "青"), "青が 10ロックスを出した")
    rate(lambda i: any("ロックス海賊団(イベント)" in d["plays"] for d in T[i] if d["who"] == "青"), "青がロックス海賊団（イベント）を使った")
    rate(lambda i: any("5リンリン" in d["plays"] for d in T[i] if d["who"] == "青"), "青が 5リンリンを出した")
    rate(lambda i: any("ニューゲート" in d["plays"] for d in T[i] if d["who"] == "青" and d["own_turn"] <= 3), "青が 3 ターン目までにニューゲート")
    # 4. 決着の形
    out.append("\n## 決着の形\n")
    turns_w = [g[i]["turns"] for i in won]
    turns_l = [g[i]["turns"] for i in lost]
    out.append(f"- 勝ち局の長さ 平均 {sum(turns_w)/len(turns_w):.1f} ターン・負け局 {sum(turns_l)/len(turns_l):.1f} ターン")
    # ライフ 0 にしながら取り切れなかった（負け局）
    print("\n".join(out))
    # 5. 代表局
    json.dump({"won": won, "lost": lost}, open(os.path.join(MK.OUT_DIR, "prog_split.json"), "w"))
    sel = os.environ.get("NARR", "")
    if sel:
        for i in [int(x) for x in sel.split(",")]:
            print("\n" + narrate(tr[i], per[i], g[i]))


if __name__ == "__main__":
    main()
