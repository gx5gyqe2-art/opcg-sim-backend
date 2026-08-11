"""リーサル近接の較正プローブ（v50・2026-08-11・読み取り専用＋ラベル出力）。

**問い**: 「相手（または自分）のライフが 0〜1 ＝あと一撃」の盤面で、本体 value は
実際の勝率をどれだけ誤るか。

**背景（v49 A2 実測 2026-08-10）**: h1 turn9 の人間の実ターン末（相手ライフ0・相手手札1・
自分はドン使い切り主力レスト）を gen13 本体 value は **−0.181** と悲観したが、本番仕様
ロールアウトの実測は **5/6 勝（EV≈+0.67）**＝誤差 ≈0.85（枝間マージンの約30倍）。機序は
素材・テンポの劣勢が「リーサル目前」を上書きすること。CPU 自己対戦でエネルが「1本の
過剰打点」に沈む（回復するナミに競走で負ける）行動系欠陥の価値側の根と目される。

**A1 と違い勝率ラベルが機能する**: 終盤盤面からはエンジンが数ターンで打ち切れるため
ラベルは飽和しない（5/6・4/6 と信号が出ることを v49 で実測済み）。よって本器は
較正誤差の測定器であると同時に**教師ラベルの採掘器**（--out で npz 行を吐く）。

**測り方**:
  - 対象: 各リプレイの復元可能な決定点のうち min(自ライフ, 相手ライフ) ≤ 1 の盤面
    （1ターンにつき手番側の最初の1点・終局済みは除外）
  - 予測: 本体 value（手番プレイヤ視点・探索の葉と同じ規約・aux 減衰なし）
  - 実測: serve_referee と同じ真値世界（山札シャッフルのみ CRN・--worlds 既定6）で
    両席とも**出荷既定 decide** により終局まで → 手番視点 EV = 2·wr − 1
  - 誤差 = 予測 − 実測EV（負＝悲観・正＝楽観）。ライフ帯（敵0/敵1/自0/自1）で層別

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/lethal_calibration_probe.py \\
    --replays h1,e1,e2 --worlds 6 --out /tmp/lethal_rows
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
import rl_encoder as E  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_game import OPCGGame  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn  # noqa: E402
from serve_referee import rollout_serve, _shuffle_decks  # noqa: E402


def bucket(me_life, opp_life):
    if opp_life <= 0:
        return "敵0"
    if me_life <= 0:
        return "自0"
    if opp_life == 1:
        return "敵1"
    return "自1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="h1,e1,e2",
                    help="リプレイタグ or JSON パス（カンマ区切り）。'all'＝全フィクスチャ表")
    ap.add_argument("--worlds", type=int, default=6)
    ap.add_argument("--min-life", type=int, default=1,
                    help="min(自ライフ,相手ライフ) がこの値以下の盤面だけ採る（既定1＝リーサル圏）。"
                         "99 で全盤面＝一般ホールドアウトのラベル化（bb1 の実盤面評価用）")
    ap.add_argument("--min-turn", type=int, default=0,
                    help="このターン以降の盤面だけ採る（序盤盤面のロールアウトは長い＝コスト制御）")
    ap.add_argument("--max-per-replay", type=int, default=8, help="1リプレイあたりの採取上限")
    ap.add_argument("--net", default="", help="value.npz[,policy.npz]（空＝出荷既定・測定/ロールアウト共通）")
    ap.add_argument("--out", default="", help="ラベル行（npz）を書くディレクトリ（空＝測定のみ）")
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    tags = list(table) if args.replays == "all" else \
        [t.strip() for t in args.replays.split(",") if t.strip()]
    db = _load_db()
    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0],
                            policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()
    gs = OPCGGame()
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version, aux_tiebreak=False)

    rows = []
    out_rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "group")}
    for tag in tags:
        raw = RE.load_replay_json(table.get(tag, tag))
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        seen, n_tag = set(), 0
        for i, a in enumerate(acts):
            if n_tag >= args.max_per_replay:
                break
            fr = fbi.get(i - 1)
            if fr is None:
                continue
            ps = fr.get("players") or {}
            lifes = {p: len((ps.get(p) or {}).get("life") or []) for p in ("p1", "p2")}
            if min(lifes.values()) > args.min_life:
                continue
            if int(a.get("turn", 0) or 0) < args.min_turn:
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
            me = m0.p1 if m0.p1.name == name else m0.p2
            opp = m0.p2 if m0.p1.name == name else m0.p1
            pred = vf(m0, name)
            wins, ok = 0, 0
            for w in range(args.worlds):
                mb = MG._restore(db, rec, fbi, acts, i)
                if isinstance(mb, str) or mb is None:
                    continue
                mw, _ = mb
                _shuffle_decks(mw, w)
                eng._world_seeds = {}
                wn, _ld = rollout_serve(eng, gs, mw, name, rng_seed=52000 + w * 7919)
                ok += 1
                wins += 1 if wn == name else 0
            if ok == 0:
                continue
            ev = 2.0 * wins / ok - 1.0
            b = bucket(len(me.life or []), len(opp.life or []))
            rows.append({"tag": tag, "i": i, "turn": key[1], "who": name, "bucket": b,
                         "me_life": len(me.life or []), "opp_life": len(opp.life or []),
                         "pred": round(pred, 3), "wr": f"{wins}/{ok}", "ev": round(ev, 3),
                         "err": round(pred - ev, 3)})
            n_tag += 1
            if args.out:
                enc = E.encode(m0, name, eng.vocab, version=eng.enc_version)
                for k in ("scalars", "field", "card_idx"):
                    out_rows[k].append(enc[k])
                out_rows["value"].append(ev)
                out_rows["group"].append(len(rows) - 1)
            print(f"  {tag}@{i} T{key[1]} {name} [{b}] 予測{pred:+.3f} 実測{wins}/{ok}"
                  f"(EV{ev:+.2f}) 誤差{pred - ev:+.3f}", flush=True)

    print(f"\n=== ライフ帯別の較正誤差（予測−実測EV・負＝悲観）")
    for b in ("敵0", "敵1", "自0", "自1"):
        sub = [r["err"] for r in rows if r["bucket"] == b]
        if sub:
            print(f"  {b}: n={len(sub)}  平均 {np.mean(sub):+.3f}  最悪 {min(sub):+.3f}/{max(sub):+.3f}")
    if args.out and out_rows["value"]:
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "lethal_00000.npz"),
                            scalars=np.array(out_rows["scalars"], np.float32),
                            field=np.array(out_rows["field"], np.float32),
                            card_idx=np.array(out_rows["card_idx"], np.int64),
                            value=np.array(out_rows["value"], np.float32),
                            group=np.array(out_rows["group"], np.int64))
        print(f"ラベル行 {len(out_rows['value'])} 件を {args.out} へ保存")
    print("LETHAL_CALIB " + json.dumps({"n": len(rows), "worlds": args.worlds,
                                        "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
