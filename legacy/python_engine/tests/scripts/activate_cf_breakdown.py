"""残り起動（腕A2）の方針対照の層化集計（2026-09-02・読み取り専用）。

`arena_resume --cand-residual-activate low|high`（候補席＝腕A2「木が TURN_END を選んだ時に、
未使用のドン追加起動効果を起動し、付与対話を方針で解く」／基準席＝素の同一ネット）の台帳を、
`dig_cf_breakdown` と同じ規約で割り直す:

  1. 全体はペア水準 CI（`arena_parallel._pair_level_ci`）
  2. リーダー層（ドン追加効果あり／なし・局単位 Wilson）——なしの層は腕A2 が発火しないので
     「対照の対照」（両腕同一の打ち回し＝0.5 付近に収まるはず）
  3. 発火の有無・起動回数・付与の区分（付与先のパワー帯／攻撃可否／候補数）別の勝率

台帳行の `act`（`promotion_gate._play_pair_detail` が候補席の events を game a/b 別に記録。
kind="activate"＝起動・kind="attach"＝付与）を読む。人間裁定は入れない＝勝敗だけで良否を決める。

実行例:
  PYTHONPATH=tests python tests/scripts/activate_cf_breakdown.py --in "/home/user/actcf/*.jsonl"
"""
import argparse
import glob
import os
import sys as _sys

import os as _os  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import dig_cf_breakdown as D  # noqa: E402  共通の読み取り/Wilson/層判定を再利用（規約の二重化なし）


def game_records(rows):
    """台帳行 → 局単位 [(leader_of_cand, won, act_events)]（void 行は除外・pure）。"""
    out = []
    for r in rows:
        if r.get("score") is None or not r.get("games"):
            continue
        la, lb = (r.get("leaders") or [None, None])[:2]
        wa, wb = r["games"]
        act = r.get("act") or [[], []]
        # 候補が各局で握ったリーダーは `cand_leaders`（2026-09-03〜）。無い古い台帳は [la, la]
        # （promotion_gate は game b でも候補に la を渡していた・従来の「game b は lb」は誤帰属）。
        cl = r.get("cand_leaders") or [la, la]
        out.append((cl[0], float(wa), act[0] or []))
        out.append((cl[1], float(wb), act[1] or []))
    return out


def _power_band(p):
    return "pw<=2000" if p <= 2000 else "pw3-5k" if p <= 5000 else "pw6k+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="台帳 jsonl（glob 可）")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    paths = []
    for pat in args.src:
        paths += sorted(glob.glob(pat)) if any(ch in pat for ch in "*?[") else [pat]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("台帳が無い", flush=True)
        return 2
    rows = D.read_rows(paths)
    from arena_parallel import _pair_level_ci
    valid = [r for r in rows if r.get("score") is not None]
    if valid:
        ci = _pair_level_ci([r["score"] / 2.0 for r in valid])
        print(f"全体（ペア水準）: {len(valid)}ペア wr {ci['win_rate']:.4f} "
              f"[{ci['lo']:.3f}, {ci['hi']:.3f}] Elo {ci['elo']:+.1f}  void {len(rows) - len(valid)}",
              flush=True)
    fired_pairs = [r for r in valid if any((r.get("act") or [[], []])[i] for i in (0, 1))]
    quiet_pairs = [r for r in valid if r not in fired_pairs]
    for name, rs in (("発火ペア", fired_pairs), ("無発火ペア", quiet_pairs)):
        if rs:
            ci = _pair_level_ci([r["score"] / 2.0 for r in rs])
            print(f"  {name}: {len(rs)}ペア wr {ci['win_rate']:.4f} [{ci['lo']:.3f}, {ci['hi']:.3f}]",
                  flush=True)
    games = game_records(rows)
    if not games:
        print("有効な局が無い", flush=True)
        return 1
    if not args.no_db:
        from cpu_selfplay import _load_db
        db = _load_db()
        strata = {"ドン追加リーダー": [], "その他": []}
        for lid, won, _ev in games:
            key = "ドン追加リーダー" if (lid and D.leader_has_don_ramp(db, lid)) else "その他"
            strata[key].append(won)
        print("\nリーダー層（局単位・Wilson95%）:", flush=True)
        for k, v in strata.items():
            p, lo, hi = D.wilson(sum(v), len(v))
            print(f"  {k:10s} {int(sum(v))}/{len(v)} = {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    fired = [(w, ev) for _l, w, ev in games if ev]
    quiet = [w for _l, w, ev in games if not ev]
    print(f"\n発火（起動）局: {len(fired)}（発火なし {len(quiet)}）", flush=True)
    for name, v in (("発火あり局", [w for w, _ in fired]), ("発火なし局", quiet)):
        if v:
            p, lo, hi = D.wilson(sum(v), len(v))
            print(f"  {name}の勝率 {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    buckets = {}
    for w, evs in fired:
        n_act = sum(1 for e in evs if e.get("kind") == "activate")
        n_att = sum(1 for e in evs if e.get("kind") == "attach")
        seen = {f"起動{min(n_act, 3)}{'+' if n_act >= 3 else ''}回",
                "付与あり" if n_att else "付与なし"}
        for e in evs:
            if e.get("kind") == "activate":
                t = int(e.get("turn", 0) or 0)
                seen.add("t1-2" if t <= 2 else "t3-4" if t <= 4 else "t5-6" if t <= 6 else "t7+")
                seen.add(f"dd{int(e.get('don_deck', 0))}")
            elif e.get("kind") == "attach":
                seen.add(_power_band(int(e.get("power", 0) or 0)))
                seen.add("付与先:攻撃可" if e.get("can_attack") else "付与先:攻撃不可")
        for b in seen:
            buckets.setdefault(b, []).append(w)
    print("\n区分別（その区分を含む局の勝率・局は重複して数える）:", flush=True)
    for b in sorted(buckets):
        v = buckets[b]
        p, lo, hi = D.wilson(sum(v), len(v))
        print(f"  {b:12s} {int(sum(v))}/{len(v)} = {p:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    n_ev = sum(len(ev) for _l, _w, ev in games)
    print(f"\n発火イベント総数 {n_ev}（局あたり {n_ev / len(games):.2f}）", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
