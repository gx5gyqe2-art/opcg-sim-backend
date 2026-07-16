"""バッチ式ラーナー（learner・単独writer）: data枝群から新鮮バッチを集めて1ラウンド学習→net枝push。

docs/reports/batched_selfplay_design_20260710.md。1本だけ走らせる（net枝の単独writer）。

ループ: 各 data枝の meta を fetch→鮮度フィルタ（未消費 かつ against_round>=round-staleness）で採用→
        リプレイバッファに連結→value/policy を低LRで1ラウンド学習→round++、consumed 更新→net枝へpush。

実行例（司令塔セッションで1本）:
  OPCG_PD_NET_BRANCH=claude/p3-pd-net \
  OPCG_PD_DATA_BRANCHES=claude/p3-pd-data-w1,claude/p3-pd-data-w2 \
  OPCG_PD_WT=/tmp/pd-learn OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_learn.py \
    --enc-version 3 --lr 2e-4 --min-new 200 --max-staleness 3
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer, train_policy
from cpu_selfplay import _load_db
from opcg_sim.src.learned.config import V4_LABEL_ALPHA, V4_AUX_TURNS_WEIGHT, V4_TURNS_SCALE
import p3_run as R
import pd_batch_common as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_BR = os.environ.get("OPCG_PD_NET_BRANCH", "claude/p3-pd-net")
DATA_BRS = [b for b in os.environ.get("OPCG_PD_DATA_BRANCHES", "").split(",") if b]
NET_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-learn") + "/net"
DATA_WT = os.environ.get("OPCG_PD_WT", "/tmp/pd-learn") + "/data"


def _git(wt, *a):
    return subprocess.run(["git", "-C", wt] + list(a), capture_output=True, text=True)


def _ensure_wt(wt, br):
    if not os.path.exists(wt + "/.git"):
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", br], capture_output=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", wt, "origin/" + br],
                       capture_output=True)
    _git(wt, "fetch", "origin", br)
    _git(wt, "reset", "--hard", "origin/" + br)


def _read_data_branch(br):
    """data枝を fetch し (meta, value配列dict, policy教師list) を返す（無ければ None）。

    batch スキーマ v2（q_root/turns_left・docs/cpu_v4_plan.md §4-1/4-2）は追加列も読む。
    旧形式（v1）は q_root←value（混合が勝敗単独に退化）・turns_left←NaN（補助損失から除外）で
    埋めて後方互換＝バッファのキー集合を一定に保つ。
    """
    _ensure_wt(DATA_WT, br)
    d = DATA_WT + "/p3data"
    if not os.path.exists(d + "/meta.json") or not os.path.exists(d + "/batch.npz"):
        return None
    meta = json.load(open(d + "/meta.json"))
    z = np.load(d + "/batch.npz")
    arrays = {k: z[k] for k in ("scalars", "field", "card_idx", "value")}
    for k in ("q_root", "turns_left"):
        if k in z.files:
            arrays[k] = z[k]
    arrays = C.normalize_batch_v2(arrays)
    pol = C.unpack_policy(z)   # 旧形式（policy無し）は [] ＝後方互換
    return meta, arrays, pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2, 3, 4))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--buffer", type=int, default=120000)
    ap.add_argument("--min-new", type=int, default=200,
                    help="（旧）小バッチ対応後、consume は即時・学習は --games-per-update 駆動＝本引数は未使用")
    ap.add_argument("--max-staleness", type=int, default=3, help="round-これ より古い against_round は捨てる")
    ap.add_argument("--games-per-update", type=int, default=128,
                    help="この games 数ごとに学習1ラウンド＝並列でも1局あたりの勾配露出を一定に保つ"
                         "（薄まり防止・既定=1バッチの games＝K=1で従来と同一）")
    ap.add_argument("--max-updates-per-round", type=int, default=16, help="1波で回す最大学習ラウンド（暴発防止）")
    ap.add_argument("--target-round", type=int, default=10 ** 9)
    ap.add_argument("--poll", type=int, default=30, help="新データが無いときの待機秒")
    # v4 学習仕様（docs/cpu_v4_plan.md §4-2）。--label-alpha 1.0 --aux-weight 0 で従来（勝敗単独）と一致。
    ap.add_argument("--label-alpha", type=float, default=V4_LABEL_ALPHA,
                    help="value 混合ラベル y = α·勝敗 + (1−α)·q_root の α")
    ap.add_argument("--aux-weight", type=float, default=V4_AUX_TURNS_WEIGHT,
                    help="残りターン補助損失の重み（0で無効）")
    ap.add_argument("--turns-scale", type=float, default=V4_TURNS_SCALE,
                    help="turns_left の正規化スケール（clip 上限）")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="忘却抑制: 凍結v4教師への value アンカー MSE の重み（v5 §4-4b・0で無効）")
    # v6 柱①（昇格ゲート・docs/reports/v5_adoption_20260715.md §4-1）: 最新ネットは candidate
    # （p3ckpt）に留め、この games 間隔で promotion_gate に挑戦→勝った場合のみ best（p3best）を更新。
    # 生成側（pd_gen）は p3best があればそこから生成する＝run をいつ止めても best が残る。
    ap.add_argument("--promote-every", type=int, default=0,
                    help="この cum_games 間隔で昇格ゲートを実行（0=無効＝v5 以前の挙動）")
    ap.add_argument("--gate-pairs1", type=int, default=12)
    ap.add_argument("--gate-pairs2", type=int, default=38)
    ap.add_argument("--gate-workers", type=int, default=3)
    ap.add_argument("--policy-smooth", type=float, default=0.0,
                    help="v7 案E: policy 教師のラベル平滑化 α（t'=(1−α)t+α/K・0=従来。"
                         "prior が 0 に沈む盲点の不可逆化を防ぐ床。推奨 0.03）")
    args = ap.parse_args()
    assert DATA_BRS, "OPCG_PD_DATA_BRANCHES が空"
    ev = args.enc_version
    vocab = E.build_vocab(_load_db())

    _ensure_wt(NET_WT, NET_BR)
    ck = NET_WT + "/p3ckpt"; R.CK = ck
    man = json.load(open(ck + "/manifest.json"))
    vnet, pnet = R.load_nets(vocab, enc_version=ev)
    if pnet is None:
        pnet = PolicyScorer(ctx_dim=E.feature_dim(ev), hidden=vnet.W1.shape[1], seed=0)
    consumed = {k: int(v) for k, v in man.get("consumed", {}).items()}
    # 忘却抑制（v5 §4-4b）: 凍結 v4 教師（gen4→ev 温スタート・恒等＝gen4 の value 意見そのもの）。
    # 学習しない参照。buf の value 予測をアンカー先にして「v4 の知識から離れ過ぎない」正則化を掛ける。
    teacher = None
    if args.distill_weight > 0.0:
        _tv = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen4_value.npz"))
        teacher = R.warm_start_value(_tv, R._net_enc_version(_tv), ev)
    print(f"learner: net枝={NET_BR} round={man.get('round',0)} data枝={len(DATA_BRS)}本 "
          f"consumed={consumed} distill={args.distill_weight}", flush=True)

    def _push_ckpt(msg):
        vnet.save(ck + "/value.npz"); pnet.save(ck + "/policy.npz")
        json.dump(man, open(ck + "/manifest.json", "w"))
        _git(NET_WT, "add", "p3ckpt")
        if os.path.exists(NET_WT + "/p3best"):
            _git(NET_WT, "add", "p3best")
        _git(NET_WT, "commit", "--amend", "-m", msg)
        return _git(NET_WT, "push", "--force", "origin", "HEAD:refs/heads/" + NET_BR).returncode == 0

    def _run_gate():
        """昇格ゲート（v6 柱①）: candidate(p3ckpt) vs best(p3best・無ければ出荷既定gen5)。

        subprocess 実行＝arena の multiprocessing/メモリを learner 本体から隔離。判定・履歴は
        manifest に記録し、昇格時は p3best を candidate で上書きする（次の _push_ckpt が拾う）。"""
        import shutil
        best_dir = NET_WT + "/p3best"
        best = (best_dir + "/value.npz," + best_dir + "/policy.npz"
                if os.path.exists(best_dir + "/value.npz") else "")
        cmd = [_sys.executable, os.path.join(REPO, "tests", "scripts", "promotion_gate.py"),
               "--candidate", ck + "/value.npz," + ck + "/policy.npz",
               "--pairs1", str(args.gate_pairs1), "--pairs2", str(args.gate_pairs2),
               "--workers", str(args.gate_workers),
               "--seed-base", str(21000 + man.get("round", 0) * 1009)]
        if best:
            cmd += ["--best", best]
        env = dict(os.environ, OPCG_LOG_SILENT="1", PYTHONPATH=os.path.join(REPO, "tests"))
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in reversed((r.stdout or "").splitlines())
                     if l.startswith("GATE_RESULT ")), None)
        if line is None:   # ゲート自体の失敗は昇格失敗として扱う（学習は止めない）
            print(f"  [gate] 実行失敗 rc={r.returncode}: {(r.stderr or '')[-300:]}", flush=True)
            res = {"promoted": False, "error": True}
        else:
            res = json.loads(line[len("GATE_RESULT "):])
        if res.get("promoted"):
            os.makedirs(best_dir, exist_ok=True)
            shutil.copy(ck + "/value.npz", best_dir + "/value.npz")
            shutil.copy(ck + "/policy.npz", best_dir + "/policy.npz")
            man["gate_best_round"] = man.get("round", 0)
        hist = man.setdefault("gate_history", [])
        hist.append({"round": man.get("round", 0), "cum": man.get("cum_games", 0), **res})
        man["gate_history"] = hist[-30:]
        man["gate_last_cum"] = man.get("cum_games", 0)
        print(f"  [gate] round{man.get('round',0)} cum{man.get('cum_games',0)}: "
              f"{'昇格' if res.get('promoted') else '棄却'} {res}", flush=True)

    # consume（バッファ連結・consumed 更新）と学習を分離する（小バッチ運用・v4）:
    # 生成側が --games を小さくしてもバッチ到着ごとに即 consume（amend 上書きによる全損を防ぐ）し、
    # 学習は pending_games が games_per_update 貯まってから＝games:updates 比を粒度に依らず一定に保つ。
    buf_v, buf_p = None, []
    pending_games = int(man.get("pending_games", 0))   # manifest 永続化＝learner 再起動で未学習分を失わない
    while man.get("round", 0) < args.target_round:
        metas = []
        cache, cache_p = {}, {}
        for br in DATA_BRS:
            r = _read_data_branch(br)
            if r is not None:
                metas.append(r[0]); cache[r[0]["worker"]] = r[1]; cache_p[r[0]["worker"]] = r[2]
        accepted, skipped = C.plan_consumption(metas, consumed, man.get("round", 0), args.max_staleness)
        if skipped:
            stale = [w for w, why in skipped.items() if why == "stale"]
            if stale:
                print(f"  [round {man.get('round',0)}] stale破棄: {stale}", flush=True)
        if accepted:
            new_states = sum(cache[m["worker"]]["value"].shape[0] for m in accepted)
            if new_states > args.buffer:
                print(f"  [warn] 1波の新規{new_states}局面 > buffer{args.buffer}＝一部が学習前に溢れる。"
                      f"buffer を上げるか generator 数/バッチを下げること。", flush=True)
            for m in accepted:
                buf_v = C.ring_append(buf_v, cache[m["worker"]], args.buffer)
                buf_p.extend(cache_p[m["worker"]])
            buf_p[:] = buf_p[-args.buffer:]
            consumed = C.update_consumed(consumed, accepted)
            pending_games += sum(m["games"] for m in accepted)
            man["cum_games"] = man.get("cum_games", 0) + sum(m["games"] for m in accepted)
            man["consumed"] = consumed
            man["pending_games"] = pending_games
            # consume-only push: round は進めない（バックプレッシャ解除＝generator を止めない）。
            ok = _push_ckpt(f"pd-net round{man.get('round',0)} cum{man['cum_games']} "
                            f"pend{pending_games} (consume)")
            print(f"  consume {len(accepted)}枝/{new_states}局面 pend={pending_games}g "
                  f"buf{len(buf_v['value'])} push={'OK' if ok else 'FAIL'}", flush=True)

        n_up, pending_games = C.wave_plan(pending_games, args.games_per_update,
                                          args.max_updates_per_round)
        if n_up == 0:
            time.sleep(args.poll)
            _ensure_wt(NET_WT, NET_BR)   # 他は無いが将来の安全のため再同期
            continue

        # v4: 混合ラベル（学習時合成＝α変更にデータ再生成不要）＋残りターン補助ターゲット。
        data_eff = dict(buf_v)
        data_eff["value"] = C.mixed_value_label(buf_v["value"], buf_v["q_root"], args.label_alpha)
        with np.errstate(invalid="ignore"):
            data_eff["aux"] = np.clip(buf_v["turns_left"], 0, args.turns_scale) / args.turns_scale
        if teacher is not None:
            # 教師アンカー先＝凍結 v4 の現バッファ上の value 予測（波ごとに再計算＝バッファ更新に追従）。
            data_eff["distill"] = RN._predict_chunked(teacher, buf_v)
        tm, vm = RN.train(vnet, data_eff, epochs=args.epochs * n_up, lr=args.lr, batch=256,
                          val_frac=0.05, aux_weight=args.aux_weight,
                          distill_weight=args.distill_weight)
        # 残りターン補助ヘッドの検証誤差（±ターン換算・時計学習の直接指標＝run 監視の主要計器）。
        aux_txt = ""
        fin = np.flatnonzero(np.isfinite(buf_v["turns_left"]))[-4000:]
        if args.aux_weight > 0 and fin.size:
            sub = {k: buf_v[k][fin] for k in ("scalars", "field", "card_idx")}
            pred_t = vnet.predict_aux(sub) * args.turns_scale
            true_t = np.clip(buf_v["turns_left"][fin], 0, args.turns_scale)
            aux_txt = f" aux±{float(np.abs(pred_t - true_t).mean()):.2f}T"
        # policy も直列(p3_run)と同じく毎ラウンド学習（凍結だと直列と挙動が乖離＝比較が汚れる）。
        if buf_p:
            train_policy(pnet, buf_p, epochs=args.epochs * n_up, lr=args.lr,
                         smooth=args.policy_smooth)
        man["round"] = man.get("round", 0) + 1   # round=netバージョン数（staleness基準・1push=1版）
        man["updates"] = man.get("updates", 0) + args.epochs * n_up   # 累積勾配パス（学習量の真の指標）
        man["pending_games"] = pending_games
        if (args.promote_every > 0 and
                man.get("cum_games", 0) - man.get("gate_last_cum", 0) >= args.promote_every):
            vnet.save(ck + "/value.npz"); pnet.save(ck + "/policy.npz")   # gate は保存済み candidate を読む
            _run_gate()
        ok = _push_ckpt(f"pd-net round{man['round']} cum{man['cum_games']} upd{man['updates']} "
                        f"buf{len(buf_v['value'])}")
        print(f"  round{man['round']} x{n_up}回学習 vmse={tm:.4f}/{vm:.4f}{aux_txt} "
              f"buf{len(buf_v['value'])} cum{man['cum_games']} pend={pending_games}g "
              f"push={'OK' if ok else 'FAIL'}", flush=True)
    print("LEARN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
