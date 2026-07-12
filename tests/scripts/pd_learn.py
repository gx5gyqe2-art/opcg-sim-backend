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
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2, 3))
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
    print(f"learner: net枝={NET_BR} round={man.get('round',0)} data枝={len(DATA_BRS)}本 "
          f"consumed={consumed}", flush=True)

    def _push_ckpt(msg):
        vnet.save(ck + "/value.npz"); pnet.save(ck + "/policy.npz")
        json.dump(man, open(ck + "/manifest.json", "w"))
        _git(NET_WT, "add", "p3ckpt")
        _git(NET_WT, "commit", "--amend", "-m", msg)
        return _git(NET_WT, "push", "--force", "origin", "HEAD:refs/heads/" + NET_BR).returncode == 0

    # consume（バッファ連結・consumed 更新）と学習を分離する（小バッチ運用・v4）:
    # 生成側が --games を小さくしてもバッチ到着ごとに即 consume（amend 上書きによる全損を防ぐ）し、
    # 学習は pending_games が games_per_update 貯まってから＝games:updates 比を粒度に依らず一定に保つ。
    buf_v, buf_p = None, []
    pending_games = 0
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
        RN.train(vnet, data_eff, epochs=args.epochs * n_up, lr=args.lr, batch=256, val_frac=0.05,
                 aux_weight=args.aux_weight)
        # policy も直列(p3_run)と同じく毎ラウンド学習（凍結だと直列と挙動が乖離＝比較が汚れる）。
        if buf_p:
            train_policy(pnet, buf_p, epochs=args.epochs * n_up, lr=args.lr)
        man["round"] = man.get("round", 0) + 1   # round=netバージョン数（staleness基準・1push=1版）
        man["updates"] = man.get("updates", 0) + args.epochs * n_up   # 累積勾配パス（学習量の真の指標）
        ok = _push_ckpt(f"pd-net round{man['round']} cum{man['cum_games']} upd{man['updates']} "
                        f"buf{len(buf_v['value'])}")
        print(f"  round{man['round']} x{n_up}回学習 buf{len(buf_v['value'])} "
              f"cum{man['cum_games']} pend={pending_games}g push={'OK' if ok else 'FAIL'}", flush=True)
    print("LEARN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
