"""残ドン掘りの方針対照（腕A vs 腕B）の層化集計（2026-09-02・読み取り専用）。

なぜ要るか: 掘り1回の効果は数%で1局の勝敗ノイズに埋もれる。方針レベルで対照した台帳
（`arena_resume --cand-residual-dig`・候補席＝腕A・基準席＝素の同一ネット）を、

  1. **リーダー層**（ドン追加効果を持つリーダー／持たない）で割る＝「戻したドンがいつ帰るか」
     の差が効くかを、力学の理屈ではなく勝敗で見る。層はカードテキストの構造語
     「ドン‼デッキから」で機械判定（リーダー名のハードコード無し）。候補席がそのリーダーを
     握った局だけをその層に数える（対面は seed で固定・席入替 CRN）
  2. **掘りイベントの区分**（ターン帯・場のドン・ドンデッキ残・カード）で、発火した局の
     勝率を割り直す＝「ドン4・残2で掘ったとき」「シュラを終了前に出した局」が損か得かを
     事後に測る（規則には入れない・仮説として検証する）

判定規約はペア水準 CI（`arena_parallel._pair_level_ci`）。層内はペアではなく局単位に
なるので**素朴 Bernoulli の Wilson 区間**で出し、その旨を明示する（層化すると席入替の対が
崩れるため）。全体はペア水準。

実行例:
  PYTHONPATH=tests python tests/scripts/dig_cf_breakdown.py --in "/home/user/digcf/*.jsonl"
"""
import argparse
import glob
import json
import math
import os
import sys as _sys

import os as _os  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

DON_RAMP_MARK = "ドン!!デッキから"     # DB 側は「‼」・パーサの raw_text は「!!」＝正規化して照合


def leader_has_don_ramp(db, cid) -> bool:
    """リーダーがドンデッキからドンを追加する効果を持つか（テキスト構造語・pure）。"""
    try:
        c = db.get_card(cid)
        blob = " ".join(getattr(ab, "raw_text", "") or "" for ab in (getattr(c, "abilities", ()) or ()))
        if not blob:
            blob = getattr(c, "effect_text", "") or getattr(c, "text", "") or ""
        return DON_RAMP_MARK in blob.replace("‼", "!!")
    except Exception:
        return False


def wilson(w, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def game_records(rows):
    """台帳行 → 局単位 [(leader_of_cand, won, dig_events)]（void 行は除外・pure）。"""
    out = []
    for r in rows:
        if r.get("score") is None or not r.get("games"):
            continue
        la, lb = (r.get("leaders") or [None, None])[:2]
        wa, wb = r["games"]
        dig = r.get("dig") or [[], []]
        out.append((la, float(wa), dig[0] or []))   # game a: 候補が la
        out.append((lb, float(wb), dig[1] or []))   # game b: 候補が lb
    return out


def bucket_of(ev):
    t = int(ev.get("turn", 0) or 0)
    tb = "t1-2" if t <= 2 else "t3-4" if t <= 4 else "t5-6" if t <= 6 else "t7+"
    return tb, f"don{int(ev.get('don_active', 0)) + int(ev.get('don_rested', 0))}", f"dd{int(ev.get('don_deck', 0))}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="台帳 jsonl（glob 可）")
    ap.add_argument("--no-db", action="store_true", help="リーダー層の判定を省く（DB 無し環境）")
    args = ap.parse_args()
    paths = []
    for pat in args.src:
        paths += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("台帳が無い", flush=True)
        return 2
    rows = read_rows(paths)
    from arena_parallel import _pair_level_ci
    valid = [r for r in rows if r.get("score") is not None]
    if valid:
        ci = _pair_level_ci([r["score"] / 2.0 for r in valid])
        print(f"全体（ペア水準）: {len(valid)}ペア wr {ci['win_rate']:.4f} "
              f"[{ci['lo']:.3f}, {ci['hi']:.3f}] Elo {ci['elo']:+.1f}  void {len(rows) - len(valid)}",
              flush=True)
    games = game_records(rows)
    if not games:
        print("有効な局が無い", flush=True)
        return 1

    # 1) リーダー層（候補席が握ったリーダーで割る・局単位 Wilson）
    if not args.no_db:
        from cpu_selfplay import _load_db
        db = _load_db()
        strata = {"ドン追加リーダー": [], "その他": []}
        for lid, won, _ev in games:
            key = "ドン追加リーダー" if (lid and leader_has_don_ramp(db, lid)) else "その他"
            strata[key].append(won)
        print("\nリーダー層（局単位・Wilson95%・席入替の対は崩れる）:", flush=True)
        for k, v in strata.items():
            p, lo, hi = wilson(sum(v), len(v))
            print(f"  {k:10s} {int(sum(v))}/{len(v)} = {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)

    # 2) 掘り発火の有無・区分別（局単位）
    fired = [(w, ev) for _l, w, ev in games if ev]
    quiet = [w for _l, w, ev in games if not ev]
    print(f"\n掘り発火: {len(fired)}局（発火なし {len(quiet)}局）", flush=True)
    if fired:
        p, lo, hi = wilson(sum(w for w, _ in fired), len(fired))
        print(f"  発火あり局の勝率 {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    if quiet:
        p, lo, hi = wilson(sum(quiet), len(quiet))
        print(f"  発火なし局の勝率 {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    buckets = {}
    cards = {}
    for w, evs in fired:
        seen = set()
        for ev in evs:
            for b in bucket_of(ev):
                seen.add(b)
            cards.setdefault(ev.get("card"), []).append(w)
        for b in seen:
            buckets.setdefault(b, []).append(w)
    print("\n区分別（その区分の掘りを1回以上した局の勝率・局は重複して数える）:", flush=True)
    for b in sorted(buckets):
        v = buckets[b]
        p, lo, hi = wilson(sum(v), len(v))
        print(f"  {b:8s} {int(sum(v))}/{len(v)} = {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    print("\nカード別:", flush=True)
    for c, v in sorted(cards.items(), key=lambda kv: -len(kv[1])):
        p, lo, hi = wilson(sum(v), len(v))
        print(f"  {str(c):10s} {int(sum(v))}/{len(v)} = {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    n_ev = sum(len(ev) for _l, _w, ev in games)
    print(f"\n発火イベント総数 {n_ev}（局あたり {n_ev / len(games):.2f}）", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
