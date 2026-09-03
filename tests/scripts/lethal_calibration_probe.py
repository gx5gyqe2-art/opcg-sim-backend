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


_G = {}


def _init_worker(net, label_sims):
    """盤面並列ワーカー（--workers>1）。label_sims>0 なら教師正本（CR.rollout sims=N）で
    ラベル化する——本番 serve（160sims・逐次）はターン6以降の盤面で1盤面15〜30分かかり
    ホールドアウト量産に不適（2026-08-12 実測・7盤面/数時間）。CR canon は約3倍/手 安く、
    盤面並列と合わせ10〜20倍速い。"""
    import argparse as _ap
    import counterfactual_referee as CR
    import p3_loop as P
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn
    CR.ARGS = _ap.Namespace(sims=label_sims or 48, true_board=False)
    db = _load_db()
    if net:
        parts = net.split(",")
        eng = LearnedEngine(value_path=parts[0],
                            policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()
    _G.update(CR=CR, db=db, eng=eng, gs=OPCGGame(), label_sims=label_sims,
              vf=P.value_fn_of(eng.vnet, eng.vocab, eng.enc_version),
              pf=P.priors_fn_of(eng.pnet, eng.vocab, eng.enc_version),
              vpred=_value_fn(eng.vnet, eng.vocab, eng.enc_version))


def label_one(task):
    """1盤面: 復元→予測→worlds ロールアウト→行。(row, enc) を返す（enc は --out 用）。"""
    import mark_gate as MGw
    import replay_reeval as REw
    import rl_encoder as Ew
    path, tag, i, worlds, want_enc = task
    raw = REw.load_replay_json(path)
    rec = raw.get("replay", raw)
    acts = rec["actions"]
    fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
    built = MGw._restore(_G["db"], rec, fbi, acts, i)
    if isinstance(built, str) or built is None:
        return None
    m0, actor = built
    if m0.winner is not None:
        return None
    name = actor.name if hasattr(actor, "name") else actor
    me = m0.p1 if m0.p1.name == name else m0.p2
    opp = m0.p2 if m0.p1.name == name else m0.p1
    pred = _G["vpred"](m0, name)
    wins = ok = 0
    for w in range(worlds):
        mb = MGw._restore(_G["db"], rec, fbi, acts, i)
        if isinstance(mb, str) or mb is None:
            continue
        mw, _ = mb
        _shuffle_decks(mw, w)
        try:
            if _G["label_sims"]:
                wn, _ld, _et = _G["CR"].rollout(_G["gs"], _G["vf"], _G["pf"], mw, name,
                                                world_seed=52000 + w,
                                                rng_seed=(52000 + w) * 131, def_temp=0.7)
            else:
                _G["eng"]._world_seeds = {}
                wn, _ld = rollout_serve(_G["eng"], _G["gs"], mw, name,
                                        rng_seed=52000 + w * 7919)
        except Exception:
            continue
        ok += 1
        wins += 1 if wn == name else 0
    if ok == 0:
        return None
    ev = 2.0 * wins / ok - 1.0
    b = bucket(len(me.life or []), len(opp.life or []))
    row = {"tag": tag, "i": i, "turn": int(acts[i].get("turn", 0) or 0), "who": name,
           "bucket": b, "me_life": len(me.life or []), "opp_life": len(opp.life or []),
           "pred": round(pred, 3), "wr": f"{wins}/{ok}", "ev": round(ev, 3),
           "err": round(pred - ev, 3)}
    enc = None
    if want_enc:
        e = Ew.encode(m0, name, _G["eng"].vocab, version=_G["eng"].enc_version)
        enc = (e["scalars"], e["field"], e["card_idx"], ev)
    return row, enc


def bucket(me_life, opp_life):
    if opp_life <= 0:
        return "敵0"
    if me_life <= 0:
        return "自0"
    if opp_life == 1:
        return "敵1"
    if me_life == 1:
        return "自1"
    return "一般"


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
    ap.add_argument("--workers", type=int, default=1, help=">1 で盤面並列（候補走査は逐次）")
    ap.add_argument("--label-sims", type=int, default=0,
                    help="0=本番 serve（160sims・高価）・N>0=教師正本 CR.rollout（例 48）")
    # --- 接戦帯 v2（2026-08-13 監査後）: 帯を**状態で事前定義**する（測定EVでの事後定義は
    # 「真は決着済みだが世界数が少なくたまたま割れた盤面」を混入させ帯の信号を薄める）。
    ap.add_argument("--max-life-diff", type=int, default=-1,
                    help="|自ライフ−相手ライフ| がこの値以下の盤面だけ採る（-1=無効。接戦帯 v2=1）")
    ap.add_argument("--min-both-life", type=int, default=0,
                    help="min(両ライフ) がこの値以上の盤面だけ採る（リーサル圏の除外・接戦帯 v2=1）")
    # --- 分散実行（ワーカー間で盤面をストライプ分割）と逐次シャード出力（走行中 push 可） ---
    ap.add_argument("--board-offset", type=int, default=0, help="候補列の開始オフセット（分散用）")
    ap.add_argument("--board-stride", type=int, default=1, help="候補列のストライド（分散用・ワーカー数）")
    ap.add_argument("--shard-rows", type=int, default=0,
                    help=">0: この行数ごとに --out へ逐次シャード（lethal_%%05d.npz＋meta.jsonl＋"
                         "provenance.json）を書く。0=従来の一括 lethal_00000.npz")
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    tags = list(table) if args.replays == "all" else \
        [t.strip() for t in args.replays.split(",") if t.strip()]

    # フェーズ1: 候補走査（軽い・逐次）。フェーズ2: ラベル化（重い・盤面並列可）。
    db = _load_db()
    tasks = []
    for tag in tags:
        path = table.get(tag, tag)
        raw = RE.load_replay_json(path)
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
            if args.max_life_diff >= 0 and abs(lifes["p1"] - lifes["p2"]) > args.max_life_diff:
                continue
            if min(lifes.values()) < args.min_both_life:
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
            tasks.append((path, tag, i, args.worlds, bool(args.out)))
            n_tag += 1
    if args.board_stride > 1 or args.board_offset:
        tasks = tasks[args.board_offset::args.board_stride]
    print(f"候補 {len(tasks)} 盤面（workers={args.workers}"
          f"・offset/stride={args.board_offset}/{args.board_stride}・"
          f"ラベル器={'CR canon sims' + str(args.label_sims) if args.label_sims else '本番serve'}）",
          flush=True)

    if args.out and args.shard_rows:
        os.makedirs(args.out, exist_ok=True)
        import subprocess
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        except Exception:
            rev = "?"
        from opcg_sim.src.learned import config as _cfg
        with open(os.path.join(args.out, "provenance.json"), "w") as f:
            json.dump({"git_rev": rev, "worlds": args.worlds, "label_sims": args.label_sims,
                       "band": {"min_life": args.min_life, "min_turn": args.min_turn,
                                "max_life_diff": args.max_life_diff,
                                "min_both_life": args.min_both_life,
                                "max_per_replay": args.max_per_replay},
                       "offset_stride": [args.board_offset, args.board_stride],
                       "serve_don_box": False,   # 旧ドン箱は純正AZ化（2026-08-25）で削除
                       "don_margin_env": os.environ.get("OPCG_DON_MARGIN", ""),
                       "replays": args.replays}, f, ensure_ascii=False)

    rows = []
    out_rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "group")}
    shard_buf, shard_no = [], 0

    def _flush_shard():
        nonlocal shard_buf, shard_no
        if not shard_buf:
            return
        L = max(len(e[0][2]) for e in shard_buf)
        ci = np.zeros((len(shard_buf), L), np.int64)
        for k, (e, _r) in enumerate(shard_buf):
            ci[k, :len(e[2])] = e[2]
        path = os.path.join(args.out, f"lethal_{shard_no:05d}.npz")
        tmp = os.path.join(args.out, f".lethal_{shard_no:05d}.tmp.npz")
        np.savez_compressed(tmp, scalars=np.stack([e[0] for e, _ in shard_buf]),
                            field=np.stack([e[1] for e, _ in shard_buf]), card_idx=ci,
                            value=np.array([e[3] for e, _ in shard_buf], np.float32))
        os.replace(tmp, path)
        with open(os.path.join(args.out, "meta.jsonl"), "a") as f:
            for _e, r in shard_buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shard_no += 1
        shard_buf = []

    def _consume(res):
        if res is None:
            return
        row, enc = res
        rows.append(row)
        if enc is not None:
            if args.shard_rows:
                shard_buf.append((enc, row))
                if len(shard_buf) >= args.shard_rows:
                    _flush_shard()
            else:
                out_rows["scalars"].append(enc[0])
                out_rows["field"].append(enc[1])
                out_rows["card_idx"].append(enc[2])
                out_rows["value"].append(enc[3])
                out_rows["group"].append(len(rows) - 1)
        print(f"  {row['tag']}@{row['i']} T{row['turn']} {row['who']} [{row['bucket']}]"
              f" 予測{row['pred']:+.3f} 実測{row['wr']}(EV{row['ev']:+.2f})"
              f" 誤差{row['err']:+.3f}", flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                          initargs=(args.net, args.label_sims)) as pool:
            for res in pool.imap_unordered(label_one, tasks):
                _consume(res)
    else:
        _init_worker(args.net, args.label_sims)
        for t in tasks:
            _consume(label_one(t))

    print(f"\n=== ライフ帯別の較正誤差（予測−実測EV・負＝悲観）")
    for b in ("敵0", "敵1", "自0", "自1"):
        sub = [r["err"] for r in rows if r["bucket"] == b]
        if sub:
            print(f"  {b}: n={len(sub)}  平均 {np.mean(sub):+.3f}  最悪 {min(sub):+.3f}/{max(sub):+.3f}")
    if args.shard_rows:
        _flush_shard()
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
