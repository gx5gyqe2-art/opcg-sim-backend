"""VALUE_BLIND 4点の原因分析プローブ（v23・読み取り専用）。

v20（`docs/reports/cpu_v20_prior_value_20260729.md`）が分離した VALUE_BLIND 4点
（m2@64 / m2@66 / m4@2 / m4@8）は「低コストキャラの展開・そこへの付与を過大評価」で一貫する。
密ラベル2周（gen7・gen8）でも符号が反転しなかった理由＝**なぜ value が過大評価するのか**を、
特徴設計（次の打ち手）の照準を決めるために2つの測定で切り分ける:

  A. **遮蔽帰属（occlusion attribution）**: 正着の子盤面（accept 最良）と誤着の子盤面
     （非accept 最良＝CPU が選ぶ側）の符号化を特徴グループ単位で入れ替え、
     「誤着側の value を押し上げているのはどの特徴群か」を直接測る。
     ネットは既定 vnet をブラックボックスとして使う（製品コード無改変）。
  B. **コーパス対照走査**: 密ラベルコーパス（/tmp/dense_v19 等）から「同リーダー・同ターン帯で
     当該カードが自場に居る行/居ない行」を取り、z（勝敗）と q_root（探索自己評価）の群間差を比較。
     - 対照（居ない行）が僅少 → **反実仮想の欠如**（教師が「出さない」場合を知らない）
     - z は差を付けないのに q_root が持ち上げる → **q_root エコー**（混合ラベル y=0.5z+0.5q の
       q 側が旧癖を再生産し、密ラベルでも矯正されない）

分類の含意:
  - 帰属が「自場ID/自場特徴」（＝出したカードそのもの）に集中 → Embedding が当該カードを
    過大評価＝カード評価の教師不足（B の結果と合わせて読む）
  - 帰属が「展開余力/手札枚数」等の集約 → 集約特徴の較正問題＝特徴設計で分離可能
  - B でエコー陽性 → ラベル側の手当て（q 混合比・矯正局面の z 純化）が特徴設計より先

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_blind_probe.py \
      --corpus /tmp/dense_v19 --corpus /tmp/dense_v17 --out /tmp/vb.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import coach_gate as CG
import rl_encoder as E
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_game import OPCGGame
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine

DEFAULT_POINTS = "m4@2"     # v20 の VALUE_BLIND 4点のうち VERIFIED_V2 に残るのは m4@2 のみ
# （m2@64/m4@8 は 2026-08-04・m2@66 は 2026-08-08 に取り下げ。指定しても accept が引けず skip される）

# --- 特徴グループ（SCALARS_V5=55 の並び・encoder.py encode() と1対1） --------------------
# 遮蔽の単位。scalars はインデックス列、field は行（キャラ枠）、card_idx は枠位置。
# 3キーの全インデックスを**重複なく被覆**する分割であること（test_value_blind_probe.py が固定）。
GROUPS = {
    "ライフ":            ("scalars", [0, 1]),
    "ドン枚数":          ("scalars", [2, 3, 4, 5]),
    "手札枚数":          ("scalars", [6, 7]),
    "場の枚数":          ("scalars", [8, 9]),
    "ターン/手番":        ("scalars", [10, 11]),
    "リーダー打点":       ("scalars", [12, 13]),
    "リーダー付与ドン":    ("scalars", [14, 15]),
    "山札/トラッシュ/KO": ("scalars", list(range(16, 22))),
    "能力使用済フラグ":    ("scalars", list(range(22, 34))),
    "召喚酔いフラグ":     ("scalars", list(range(34, 46))),
    "自デッキ残集約":     ("scalars", list(range(46, 51))),
    "相手脅威集約":       ("scalars", [51, 52, 53]),
    "展開余力":          ("scalars", [54]),
    "自場特徴":          ("field", [0, 1, 2, 3, 4]),
    "相手場特徴":         ("field", [5, 6, 7, 8, 9]),
    "リーダーID":         ("card_idx", [0, 1]),
    "自場ID":            ("card_idx", [2, 3, 4, 5, 6]),
    "相手場ID":          ("card_idx", [7, 8, 9, 10, 11]),
    "手札ID":            ("card_idx", list(range(12, 22))),
    "ステージID":         ("card_idx", [22, 23]),
}


def swap_group(dst, src, key, idxs):
    """dst 符号化のグループ (key, idxs) を src の値に入れ替えた**新しい**符号化 dict（pure）。

    scalars/card_idx は要素、field は行（キャラ枠）単位。dst/src は変更しない。"""
    out = {k: v.copy() for k, v in dst.items()}
    out[key][np.asarray(idxs)] = src[key][np.asarray(idxs)]
    return out


def attribution(vf, enc_bad, enc_good):
    """遮蔽帰属: 誤着子盤面の value 優位 gap=v(bad)−v(good) を特徴グループへ分解（pure）。

    fwd = v(bad) − v(bad←good[g]) ＝「bad の優位のうちグループ g に乗っている分」
    rev = v(good←bad[g]) − v(good) ＝「bad の g を good に移植したときの押し上げ」
    非線形ネットでは総和は gap に一致しないが、fwd/rev の両向きで大きい群は頑健な犯人。
    returns: (gap, rows)  rows=[{group, fwd, rev, mean}] を |mean| 降順。"""
    v_bad, v_good = vf(enc_bad), vf(enc_good)
    rows = []
    for g, (key, idxs) in GROUPS.items():
        fwd = v_bad - vf(swap_group(enc_bad, enc_good, key, idxs))
        rev = vf(swap_group(enc_good, enc_bad, key, idxs)) - v_good
        rows.append({"group": g, "fwd": float(fwd), "rev": float(rev),
                     "mean": float((fwd + rev) / 2.0)})
    rows.sort(key=lambda r: -abs(r["mean"]))
    return float(v_bad - v_good), rows


def scan_target(enc_parent, enc_child):
    """親→子の符号化差分から走査対象カードを特定（pure）。

    展開（自場IDに新出）→ (idx, "deploy")。付与（自場特徴の attached_don 列=3 が増加した枠）→
    (その枠のID, "attach")。どちらでもない（攻撃・PASS 等）→ (None, None)。"""
    p_own = list(enc_parent["card_idx"][2:7])
    c_own = list(enc_child["card_idx"][2:7])
    new = [i for i in c_own if i != 0 and c_own.count(i) > p_own.count(i)]
    if new:
        return int(new[0]), "deploy"
    for r in range(5):
        if (enc_child["card_idx"][2 + r] != 0
                and enc_child["field"][r, 3] > enc_parent["field"][r, 3] + 1e-9):
            return int(enc_child["card_idx"][2 + r]), "attach"
    return None, None


def contrast_stats(z, q, present):
    """コーパス対照統計（pure）。z=勝敗ラベル、q=q_root（NaN許容）、present=対象カード自場在否。

    returns dict: 群別 n / mean_z / mean_q（有限のみ）と、
      dz = mean_z(在) − mean_z(不在)   … 勝敗が展開を支持する差
      dq = mean_q(在) − mean_q(不在)   … 探索自己評価が展開を支持する差
      echo = dq − dz                   … 正に大きいほど「q_root が z の裏付け無く持ち上げている」
    """
    z = np.asarray(z, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    out = {}
    for name, m in (("present", present), ("absent", ~present)):
        zz, qq = z[m], q[m]
        qf = qq[np.isfinite(qq)]
        out[name] = {"n": int(m.sum()),
                     "mean_z": float(zz.mean()) if zz.size else None,
                     "mean_q": float(qf.mean()) if qf.size else None}
    p, a = out["present"], out["absent"]
    dz = (p["mean_z"] - a["mean_z"]) if (p["mean_z"] is not None and a["mean_z"] is not None) else None
    dq = (p["mean_q"] - a["mean_q"]) if (p["mean_q"] is not None and a["mean_q"] is not None) else None
    out["dz"] = dz
    out["dq"] = dq
    out["echo"] = (dq - dz) if (dz is not None and dq is not None) else None
    return out


def corpus_scan(corpus_dir, leader_idx, target_idx, turn_max):
    """密シャード群から「同リーダー・turn≤turn_max・手番」の行を集め対照統計を返す。"""
    z_all, q_all, pr_all = [], [], []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "dense_*.npz"))):
        d = np.load(path)
        s, ci = d["scalars"], d["card_idx"]
        m = (s[:, 11] == 1.0) & (s[:, 10] <= turn_max) & (ci[:, 0] == leader_idx)
        if not m.any():
            continue
        z_all.append(d["value"][m])
        q_all.append(d["q_root"][m])
        pr_all.append((ci[m, 2:7] == target_idx).any(axis=1))
    if not z_all:
        return None
    return contrast_stats(np.concatenate(z_all), np.concatenate(q_all), np.concatenate(pr_all))


def best_children(eng, m0, actor, accept):
    """accept 最良/非accept 最良の (value, 記述, 符号化) を返す（v20 value_gap の子盤面保持版）。"""
    legal = m0.get_legal_actions(actor) or []
    game = OPCGGame()
    name = actor if isinstance(actor, str) else actor.name

    def val_of(enc):
        return float(eng.vnet.predict({k: enc[k][None, ...] for k in
                                       ("scalars", "field", "card_idx")})[0])

    best = {"acc": None, "other": None}
    for mv in legal:
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
            child = game.apply(m0, mv, name)
            if child is None:
                continue
            enc = E.encode(child, name, eng.vocab, version=eng.enc_version)
            v = val_of(enc)
        except Exception:
            continue
        slot = "acc" if CG.hit(d, accept) else "other"
        if best[slot] is None or v > best[slot][0]:
            best[slot] = (v, {"action_type": d.get("action_type"), "card": d.get("card")}, enc)
    return best["acc"], best["other"], val_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="value.npz[,policy.npz]（未指定＝出荷既定＝現 gen8）")
    ap.add_argument("--points", default=DEFAULT_POINTS, help="tag@i をカンマ区切り")
    ap.add_argument("--corpus", action="append", default=[],
                    help="密シャードのディレクトリ（複数可・例 /tmp/dense_v19）")
    ap.add_argument("--out", default=None, help="JSON 保存先")
    args = ap.parse_args()

    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()

    want = [(t, int(i)) for t, i in (p.split("@") for p in args.points.split(","))]
    accept_of = {(t, i): a for t, i, a in CG.VERIFIED_V2}
    replays = {**__import__("mark_gate").REPLAYS, **CG.REPLAYS_V2}
    CR.GAMES = {}
    results, t0 = [], time.time()
    print(f"=== VALUE_BLIND 原因分析 net={'指定' if args.net else '既定(gen8)'} "
          f"points={args.points} ===", flush=True)
    for tag, i in want:
        accept = accept_of.get((tag, i))
        if accept is None:
            print(f"  {tag}@{i}: VERIFIED_V2 に無い（スキップ）", flush=True)
            continue
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            print(f"  {tag}@{i}: 復元不可 {built}", flush=True)
            continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        acc, other, vf = best_children(eng, m0, actor, accept)
        if acc is None or other is None:
            print(f"  {tag}@{i}: accept/非accept の子盤面が揃わない（スキップ）", flush=True)
            continue
        gap, rows = attribution(vf, other[2], acc[2])
        enc_parent = E.encode(m0, name, eng.vocab, version=eng.enc_version)
        tidx, mode = scan_target(enc_parent, other[2])
        turn = float(enc_parent["scalars"][10])
        rec = {"tag": tag, "i": i, "gap": gap,
               "acc": {"v": acc[0], **acc[1]}, "bad": {"v": other[0], **other[1]},
               "attribution": rows, "scan": {"target_idx": tidx, "mode": mode,
                                             "leader_idx": int(enc_parent["card_idx"][0]),
                                             "turn": turn}}
        print(f"  {tag}@{i:<4} gap={gap:+.3f}  bad={other[1]['action_type']}/{other[1]['card']}"
              f"(v={other[0]:+.3f})  acc={acc[1]['action_type']}/{acc[1]['card']}(v={acc[0]:+.3f})"
              f"  [{time.time() - t0:.0f}s]", flush=True)
        for r in rows[:5]:
            print(f"      {r['group']:<12} fwd={r['fwd']:+.3f} rev={r['rev']:+.3f} "
                  f"mean={r['mean']:+.3f}", flush=True)
        for cdir in args.corpus:
            if tidx is None:
                break
            st = corpus_scan(cdir, rec["scan"]["leader_idx"], tidx, turn + 1.0)
            if st is None:
                continue
            rec.setdefault("corpus", {})[cdir] = st
            fmt = lambda x: "----" if x is None else f"{x:+.3f}"
            print(f"      corpus[{os.path.basename(cdir)}] {mode} idx={tidx}: "
                  f"在n={st['present']['n']} 不在n={st['absent']['n']} "
                  f"dz={fmt(st['dz'])} dq={fmt(st['dq'])} echo={fmt(st['echo'])}", flush=True)
        results.append(rec)

    # 集計: 全点で符号が揃う帰属グループ（gap と同方向＝bad を押し上げる群）
    print("\n=== 帰属サマリ（全点で mean>0＝誤着側を一貫して押し上げる群） ===")
    if results:
        names = [r["group"] for r in results[0]["attribution"]]
        for g in sorted(names, key=lambda g: -min(
                next(x["mean"] for x in r["attribution"] if x["group"] == g) for r in results)):
            means = [next(x["mean"] for x in r["attribution"] if x["group"] == g) for r in results]
            if min(means) > 0.0:
                print(f"  {g:<12} " + " ".join(f"{m:+.3f}" for m in means))
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False)
    print(f"VALUE_BLIND_PROBE_RESULT {json.dumps({'points': len(results)}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
