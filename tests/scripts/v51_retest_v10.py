"""v51 再試験 @ 符号化 v10（G系・2026-08-12・v52b 採用判定の最終 Go/No-Go）。

**問い**: v51 の負の結果（乖離族教師50点は暗記のみ・転移ゼロ＝representation-bound）は、
リーサル距離Δ3値（v10・`lethal.py`）が入ると解消するか。解消すれば「探索＋value＋Δ要約」
の特徴言語拡張が正しい処方だったことの実証（G/B 合流路線の勝ち筋）。

v51_finetune と同じ学習規約（教師 MSE＋蒸留アンカー交互・v33規約）で、入力だけ v10:
  - 教師50: meta の (seed, turn, who) から決定論再生 → v10 符号化（value=実測EV）
  - L45: `lethal_calibration_probe` のスキャン段を同引数で再走 → 盤面復元 →
    **v9 符号化の内容一致**で fixture 行と照合（順序仮定なし）→ v10 行に EV を移植
  - 一般60: `holdout60_boards.json`（保全済み盤面表）→ v10 符号化
  - アンカー: 全リプレイの一般盤面を新規採取（評価点 (tag,i) は除外・教師は gen14 の
    v9 予測＝恒等拡張前の挙動へ引き戻す）
ベースは gen14 を **恒等温スタート**（warm_start_value 9→10・新3列ゼロ＝学習前の全指標が
v51 の「前」と一致することが再現チェック）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/v51_retest_v10.py --out /tmp/cand_v10
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import rl_encoder as E  # noqa: E402
import rl_net as RN  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")
FIXT = os.path.join(REPO, "tests", "fixtures", "candidates", "v51_teacher")


def _rows_append(rows, enc, value):
    rows["scalars"].append(enc["scalars"])
    rows["field"].append(enc["field"])
    rows["card_idx"].append(enc["card_idx"])
    rows["value"].append(np.float32(value))


def _stack(rows):
    return {k: np.stack(v) if k != "value" else np.array(v, np.float32)
            for k, v in rows.items()}


def _metrics(net, X):
    p = net.predict({k: X[k] for k in ("scalars", "field", "card_idx")})
    e = p - X["value"]
    return p, {"MAE": round(float(np.mean(np.abs(e))), 3),
               "r": round(float(np.corrcoef(p, X["value"])[0, 1]), 3),
               "sign": round(float(np.mean(np.sign(p) == np.sign(X["value"]))), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--anchor-rows", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    import mark_gate as MG
    import replay_reeval as RE
    import coach_gate as CG
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_sim.src.core.cpu_learned import LearnedEngine, warm_start_value

    db = _load_db()
    vocab = E.build_vocab(db)
    gs = OPCGGame()
    eng = LearnedEngine()
    t0 = time.time()

    # --- 1) 教師50: 決定論再生（lethal_distance_probe と同じ seed 規約）→ v10 行
    game = _make_fixed_matchup_game(
        os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json"),
        "nami", "shanks")
    T = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    for meta_f in ("meta_r1.json", "meta_r2.json"):
        meta = json.load(open(os.path.join(FIXT, meta_f)))
        wanted = {}
        for d in meta["diag"]:
            if d.get("teach"):
                wanted.setdefault(d["seed"], []).append((d["turn"], d["who"], d["ev"]))
        for seed, pts in sorted(wanted.items()):
            m = game.new_game(db, seed)
            drng = np.random.default_rng(seed * 17 + 3)
            left = {(t, w): ev for t, w, ev in pts}
            steps = 0
            while left and m.winner is None and not gs.is_terminal(m) and steps < 400:
                name = gs.current_player(m)
                if name is None:
                    break
                t = int(getattr(m, "turn_count", 0) or 0)
                if (t, name) in left:
                    ev = left.pop((t, name))
                    _rows_append(T, E.encode(m, name, vocab, version=10), ev)
                actor = m.p1 if m.p1.name == name else m.p2
                eng._world_seeds = {}
                mv = eng.decide(m, actor, sims=32, rng=drng)
                if mv is None:
                    break
                m2 = gs.apply(m, mv, name)
                if m2 is None:
                    break
                m = m2
                steps += 1
            print(f"  教師 {meta_f} seed{seed}: 累計{len(T['value'])} {time.time()-t0:.0f}s",
                  flush=True)
    T = _stack(T)
    print(f"教師 {len(T['value'])} 行（v10）")

    # --- 2) L45: スキャン再走 → v9 内容一致で fixture とペアリング → v10 行
    L45fix = {k: np.load(os.path.join(FIXT, "lethal45_v50.npz"))[k]
              for k in ("scalars", "value")}
    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    L45 = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    eval_points = set()
    n_matched = 0
    for tag in table:
        raw = RE.load_replay_json(table[tag])
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        seen, n_tag = set(), 0
        for i, a in enumerate(acts):
            if n_tag >= 6:
                break
            fr = fbi.get(i - 1)
            if fr is None:
                continue
            ps = fr.get("players") or {}
            lifes = {p: len((ps.get(p) or {}).get("life") or []) for p in ("p1", "p2")}
            if min(lifes.values()) > 1:                     # 元引数 --min-life 1
                continue
            key = (a.get("player"), int(a.get("turn", 0) or 0))
            if key in seen:
                continue
            built = MG._restore(db, rec, fbi, acts, i)
            if isinstance(built, str) or built is None:
                continue
            m0, actor = built
            if m0.winner is not None:
                continue
            name = actor.name if hasattr(actor, "name") else actor
            seen.add(key)
            n_tag += 1
            e9 = E.encode(m0, name, vocab, version=9)
            hit = np.where((L45fix["scalars"] == e9["scalars"][None, :]).all(axis=1))[0]
            if len(hit) != 1:
                continue                                    # fixture でラベル化されなかった候補
            _rows_append(L45, E.encode(m0, name, vocab, version=10),
                         float(L45fix["value"][hit[0]]))
            eval_points.add((tag, i))
            n_matched += 1
    L45 = _stack(L45)
    print(f"L45 照合 {n_matched}/{len(L45fix['value'])} 行（v9 内容一致） {time.time()-t0:.0f}s")

    # --- 3) 一般60: 保全済み盤面表 → v10 行
    spec = json.load(open(os.path.join(FIXT, "holdout60_boards.json")))
    H60 = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    cache = {}
    for r in spec["rows"]:
        tag, i = r["tag"], r["i"]
        if tag not in cache:
            raw = RE.load_replay_json(table[tag])
            rec = raw.get("replay", raw)
            cache[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                          rec["actions"])
        rec, fbi, acts = cache[tag]
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str) or built is None:
            continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        _rows_append(H60, E.encode(m0, name, vocab, version=10), r["ev"])
        eval_points.add((tag, i))
    H60 = _stack(H60)
    print(f"一般60 → {len(H60['value'])} 行 {time.time()-t0:.0f}s")

    # --- 4) アンカー: 全リプレイの一般盤面（評価点除外）→ v9 行（gen14 教師）＋ v10 行（生徒）
    A9 = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    A10 = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    for tag in table:
        raw = RE.load_replay_json(table[tag])
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        seen, n_tag = set(), 0
        for i, a in enumerate(acts):
            if n_tag >= 40 or len(A9["value"]) >= args.anchor_rows:
                break
            if (tag, i) in eval_points:
                continue
            if int(a.get("turn", 0) or 0) < 2:
                continue
            fr = fbi.get(i - 1)
            if fr is None:
                continue
            key = (a.get("player"), int(a.get("turn", 0) or 0))
            if key in seen:
                continue
            built = MG._restore(db, rec, fbi, acts, i)
            if isinstance(built, str) or built is None:
                continue
            m0, actor = built
            if m0.winner is not None:
                continue
            name = actor.name if hasattr(actor, "name") else actor
            seen.add(key)
            n_tag += 1
            _rows_append(A9, E.encode(m0, name, vocab, version=9), 0.0)
            _rows_append(A10, E.encode(m0, name, vocab, version=10), 0.0)
    A9, A10 = _stack(A9), _stack(A10)
    print(f"アンカー {len(A9['value'])} 盤面 {time.time()-t0:.0f}s")

    # --- 5) 学習（v51 と同一規約・入力のみ v10）
    base9 = RN.ValueNet.load(os.path.join(MODELS, "gen14_value.npz"))
    yA = base9.predict({k: A9[k] for k in ("scalars", "field", "card_idx")})
    net = warm_start_value(RN.ValueNet.load(os.path.join(MODELS, "gen14_value.npz")), 9, 10)

    p0_45, m0_45 = _metrics(net, L45)
    _p0_60, m0_60 = _metrics(net, H60)
    dec_mask = np.abs(p0_45 - L45["value"]) >= 0.85     # v50 の乖離族（学習外・転移の的）
    print(f"学習前（恒等＝gen14 再現チェック）: 教師MAE {_metrics(net, T)[1]['MAE']}"
          f" / L45 {m0_45} / 乖離族{int(dec_mask.sum())}点"
          f" MAE {np.mean(np.abs((p0_45 - L45['value'])[dec_mask])):.3f} / H60 {m0_60}",
          flush=True)

    rng = np.random.default_rng(17)
    nT = len(T["value"])
    for ep in range(args.epochs):
        order = rng.permutation(nT)
        for k in range(0, nT, 32):
            selb = order[k:k + 32]
            b = {kk: T[kk][selb] for kk in ("scalars", "field", "card_idx")}
            _p, cache_ = net.forward(b)
            net.step(net.backward(cache_, T["value"][selb]), lr=args.lr)
            asel = rng.integers(0, len(yA), 192)
            ab = {kk: A10[kk][asel] for kk in ("scalars", "field", "card_idx")}
            _pa, cachea = net.forward(ab)
            net.step(net.backward(cachea, yA[asel]), lr=args.lr)
        if (ep + 1) % 30 == 0:
            print(f"  ep{ep+1}: 教師MAE {_metrics(net, T)[1]['MAE']}", flush=True)

    # --- 6) 前後測定
    p1_45, m1_45 = _metrics(net, L45)
    _p1_60, m1_60 = _metrics(net, H60)
    drift = net.predict({k: A10[k] for k in ("scalars", "field", "card_idx")}) - yA
    res = {"teacher_mae": _metrics(net, T)[1]["MAE"],
           "L45": {"before": m0_45, "after": m1_45},
           "deceptive_heldout": {"n": int(dec_mask.sum()),
                                 "before_mae": round(float(np.mean(np.abs((p0_45 - L45["value"])[dec_mask]))), 3),
                                 "after_mae": round(float(np.mean(np.abs((p1_45 - L45["value"])[dec_mask]))), 3)},
           "H60": {"before": m0_60, "after": m1_60},
           "anchor_drift_std": round(float(drift.std()), 4),
           "n_teacher": int(nT), "n_L45": int(len(L45["value"])),
           "n_H60": int(len(H60["value"])), "n_anchor": int(len(yA))}
    os.makedirs(args.out, exist_ok=True)
    net.save(os.path.join(args.out, "value.npz"))
    with open(os.path.join(args.out, "meta_v10_retest.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(args.out, "rows_T.npz"), **T)
    np.savez_compressed(os.path.join(args.out, "rows_L45.npz"), **L45)
    np.savez_compressed(os.path.join(args.out, "rows_H60.npz"), **H60)
    print("V51_RETEST_V10 " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
