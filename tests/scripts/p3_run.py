"""P3 シャード積み上げドライバ（揮発性環境で忠実な本走を完遂・docs/.../cpu_rl_pilot_*）。

設計（レビュー収束）:
- **小シャード単位**で自己対戦→**オンライン更新(低LR)**→**net-only checkpoint を専用ブランチへ force-push**。
  回収で失うのは直近1シャードのみ（netは2MBで永続・生データはscratch使い捨て）。
- **4コア並列**自己対戦（BLASスレッドは 1 に固定＝スラッシング回避・レビュー指摘）。
- **セッション内リプレイバッファ**で相関緩和（忘却対策）＋ルートDirichletノイズ。
- 1世代分(target局)で frozen スナップショット＋status=AWAITING_GATEで**停止**（世代跨ぎは人間ゲート）。
- 再開: 起動時に checkpoint ブランチへ同期し最新netからレジューム。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_run.py --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --max-shards 4 --workers 4
     （--enc-version は必須。版はこの引数のみで決まる）
"""
import os
# numpy/BLAS インポート前にスレッドを1に固定（4プロセス×各全コア=スラッシングを防ぐ）。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import subprocess
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import warm_start_value, warm_start_policy, _net_enc_version
from opcg_sim.src.learned.config import SELFPLAY_SIMS, SELFPLAY_DIRICHLET_EPS, SELFPLAY_TEMP_MOVES
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer, state_context, train_policy
from az_mcts_tree import TreeMCTS   # make/unmake版（唯一の探索実装。旧clone版は削除済み）
from opcg_action import legal_action_matrix, ACTION_DIM
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import p3_loop as P

# checkpoint の worktree/枝は環境変数で上書き可（案B: クラスタ別に別 checkpoint 枝へ隔離し、
# 案A の /tmp/p3ckpt-wt・claude/p3-checkpoints と衝突させない）。未設定なら従来の既定。
WT = os.environ.get("OPCG_P3_WT", "/tmp/p3ckpt-wt")
CK = WT + "/p3ckpt"
BR = os.environ.get("OPCG_P3_BRANCH", "claude/p3-checkpoints")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_DEFAULT = 10000


# ---- git checkpoint ----
def _git(*a, cwd=WT):
    return subprocess.run(["git", "-C", cwd] + list(a), capture_output=True, text=True)


def ensure_wt():
    if not os.path.exists(WT + "/.git"):
        _git("worktree", "prune", cwd=REPO)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", BR], capture_output=True, text=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", WT, BR], capture_output=True, text=True)
    _git("fetch", "origin", BR)
    _git("reset", "--hard", "origin/" + BR)
    os.makedirs(CK, exist_ok=True)


def push_ckpt(msg):
    _git("add", "p3ckpt")
    _git("commit", "--amend", "-m", msg)
    return _git("push", "--force", "origin", BR).returncode == 0


def read_manifest():
    p = CK + "/manifest.json"
    return json.load(open(p)) if os.path.exists(p) else {"gen": 0, "cum_games": 0, "shards": 0, "status": "INIT"}


def write_manifest(m):
    json.dump(m, open(CK + "/manifest.json", "w"))


def load_nets(vocab, enc_version):
    """符号化世代 `enc_version`（＝**引数が唯一の版指定**）のネットをロード/新規作成する。

    版はロード先の重み次元ではなく `enc_version` が決める。既存チェックポイントの入力次元が
    `enc_version` と食い違う場合は**黙って引き継がず即エラー**にする（共有チェックポイント
    ブランチに別版の残骸があると、v2 実行で v1 ネットを拾ってエンコード次元不一致で落ちる
    footgun を、明示エラーへ変える）。異版を混ぜたいときはチェックポイントを掃除して再実行。
    """
    vp, pp, g0 = CK + "/value.npz", CK + "/policy.npz", CK + "/gen0_value.npz"
    want = E.feature_dim(enc_version)
    # OPCG_P3_LEAD_SLOTS: リーダー条件付け(LC)実験のガード。セットすると「読んだ/作った net の lead_slots が
    # この値でなければ黙って走らずエラー停止」＝legacy net を LCパイロットで訓練する事故（サイレント汚染）を
    # 構造的に封じる。未設定＝制約なし＝従来（legacy=lead_slots0）の全runは無影響。
    _wl = os.environ.get("OPCG_P3_LEAD_SLOTS")
    want_lead = int(_wl) if _wl not in (None, "") else None

    def _vguard(vnet, src):
        feat = vnet.feat_dim
        if feat != want:
            raise SystemExit(
                f"ERROR: {src} の入力次元(feat_dim={feat}) が --enc-version {enc_version}"
                f"(feat_dim={want}) と不一致。版は引数で決まります＝別版のチェックポイントは"
                f"掃除してから再実行してください（origin/{BR} をクリーンにする）。")
        return vnet

    prod_v = os.path.join(REPO, "opcg_sim", "data", "learned", "gen2_value.npz")
    prod_p = os.path.join(REPO, "opcg_sim", "data", "learned", "gen2_policy.npz")
    if os.path.exists(vp):
        vnet = _vguard(RN.ValueNet.load(vp), vp)
        pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None
    elif os.path.exists(g0):
        vnet = _vguard(RN.ValueNet.load(g0), g0)
        pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None
    elif enc_version >= 2:
        # v2 Gen0 は出荷 v1 Gen2 から**温スタート**する（乱数より圧倒的に筋が良い：出荷の
        # 実力を引き継ぎ、増えた特徴の使い方だけを学ぶ）。append-only 拡張＝拡張直後の出力は
        # 出荷 v1 と恒等。旧版(v1)出荷から現行(ev)への差分を warm_start_* が自動計算する。
        base_v = _net_enc_version(RN.ValueNet.load(prod_v))   # 出荷ネットの版（＝1）
        vnet = warm_start_value(RN.ValueNet.load(prod_v), base_v, enc_version)
        if want_lead == 2:
            # 種なし起動でも LC で始める（防御的＝空 checkpoint から legacy を作らない）。
            vnet = vnet.to_leader_conditioned()
        pnet = warm_start_policy(PolicyScorer.load(prod_p), base_v, enc_version) \
            if os.path.exists(prod_p) else None
        vnet.save(g0); vnet.save(vp)
        if pnet is not None:
            pnet.save(pp)
    else:
        src = os.path.join(REPO, "tests", "p2_sl_net.npz")
        vnet = _vguard(RN.ValueNet.load(src), src); vnet.save(g0); vnet.save(vp)
        pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None

    # LC ガード（全経路の最終チェック）: 期待 lead_slots と実物が食い違えば停止＝サイレント汚染防止。
    if want_lead is not None and int(getattr(vnet, "lead_slots", 0)) != want_lead:
        raise SystemExit(
            f"ERROR: 読み込んだ value net の lead_slots={getattr(vnet, 'lead_slots', 0)} が "
            f"OPCG_P3_LEAD_SLOTS={want_lead} と不一致。legacy net を LC 実験で訓練する事故を防ぐため停止。"
            f"→ checkpoint枝({BR})に正しいLC種があるか／このセッションがLC対応コード枝(lead_slots実装入り)"
            f"に居るかを確認してください。")

    # EffFeat ガード（v3 実験用・LCガードと同型）: OPCG_P3_EFF_DIM=期待する eff_dim（例 116）。
    _we = os.environ.get("OPCG_P3_EFF_DIM")
    if _we not in (None, ""):
        want_eff = int(_we)
        have_eff = int(getattr(vnet, "eff_dim", 0))
        if have_eff != want_eff:
            raise SystemExit(
                f"ERROR: 読み込んだ value net の eff_dim={have_eff} が OPCG_P3_EFF_DIM={want_eff} と"
                f"不一致。EffFeat 無しの net を v3 実験で訓練する事故を防ぐため停止。"
                f"→ checkpoint枝({BR})に正しい v3 種があるか／v3 対応コード枝に居るかを確認してください。")

    if pnet is not None and int(pnet.in_dim) - ACTION_DIM != want:
        raise SystemExit(
            f"ERROR: {pp} の policy ctx 次元 が --enc-version {enc_version}(feat_dim={want}) と"
            f"不一致。別版のチェックポイントを掃除してから再実行してください。")
    return vnet, pnet


# ---- 並列自己対戦ワーカー ----
_W = {}


def _init_worker():
    """ワーカー1プロセスにつき1回 db/vocab/game をロード（タスク毎の再ロードを避ける）。"""
    db = _load_db()
    _W["db"] = db
    _W["vocab"] = E.build_vocab(db)
    _W["game"] = OPCGGame()


def _gen_task(payload):
    seed, n_games, sims, eps, vpath, ppath, ev, leaders = payload
    db, vocab, game = _W["db"], _W["vocab"], _W["game"]
    vnet = RN.ValueNet.load(vpath)
    pnet = PolicyScorer.load(ppath) if ppath else None
    vf = P.value_fn_of(vnet, vocab, ev); pf = P.priors_fn_of(pnet, vocab, ev)
    rng = np.random.default_rng(seed)
    S, F, I, Y, pol = [], [], [], [], []
    for _ in range(n_games):
        m = game.new_game(db, int(rng.integers(1 << 30)), leaders=leaders)
        rv, rp, steps = [], [], 0
        while game.winner(m) is None and not game.is_terminal(m) and steps < 400:
            name = game.current_player(m)
            if name is None:
                break
            mc = TreeMCTS(game, value_fn=vf, priors_fn=pf, n_sims=sims,
                          determinize_fn=lambda s, r: game.determinize(s, name, r),
                          rng=rng, dirichlet_eps=eps)
            move, N, legal = mc.run(m)
            if move is None or N is None or N.sum() == 0:
                break
            enc = E.encode(m, name, vocab, version=ev)
            rv.append((enc, name))
            rp.append((state_context(m, name, vocab, version=ev), legal_action_matrix(m, legal, name), N / N.sum()))
            a = int(np.argmax(N)) if steps >= SELFPLAY_TEMP_MOVES else int(rng.choice(len(N), p=(N / N.sum())))
            try:
                cpu_ai._apply_move_inplace(m, name, legal[a])
            except Exception:
                break
            steps += 1
        w = game.winner(m)
        if w is None:
            continue
        for enc, who in rv:
            S.append(enc["scalars"]); F.append(enc["field"]); I.append(enc["card_idx"])
            Y.append(1.0 if who == w else -1.0)
        pol.extend(rp)
    if not S:
        return None
    return (np.stack(S), np.stack(F), np.stack(I), np.array(Y, dtype=np.float32), pol)


def selfplay_shard(pool, workers, n_games, sims, eps, vpath, ppath, base_seed, ev=1, leaders=None):
    """n_games を workers 個に分割して並列生成→マージ。"""
    per = max(1, n_games // workers)
    tasks = [(base_seed * 131 + w * 977 + 1, per, sims, eps, vpath, ppath, ev, leaders)
             for w in range(workers)]
    parts = pool.map(_gen_task, tasks)
    S, F, I, Y, pol = [], [], [], [], []
    for p in parts:
        if p is None:
            continue
        S.append(p[0]); F.append(p[1]); I.append(p[2]); Y.append(p[3]); pol.extend(p[4])
    if not S:
        return None, None
    vdata = {"scalars": np.concatenate(S), "field": np.concatenate(F),
             "card_idx": np.concatenate(I), "value": np.concatenate(Y)}
    return vdata, pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=SELFPLAY_SIMS)
    ap.add_argument("--max-shards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--buffer", type=int, default=30000)
    ap.add_argument("--dirichlet-eps", type=float, default=SELFPLAY_DIRICHLET_EPS)
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2),
                    help="符号化世代（必須・版はこの引数のみで決まる。1=出荷Gen2互換／"
                         "2=リーダー付与ドン特徴）")
    ap.add_argument("--rotate-leaders", action="store_true",
                    help="自己対戦のリーダーを全リーダーから抽選＋リアルデッキ化（穴B）")
    args = ap.parse_args()
    ev = args.enc_version

    _db0 = _load_db()
    vocab = E.build_vocab(_db0)
    leaders = None
    if args.rotate_leaders:
        from deckgen import all_leader_ids
        leaders = all_leader_ids(_db0)
        print(f"リーダーローテーション ON: {len(leaders)} 種", flush=True)
    ensure_wt()
    man = read_manifest()
    if man.get("status") == "AWAITING_GATE":
        print(f"⛔ AWAITING_GATE: Gen{man['gen']} 完成。人間クロス評価ゲート待ち"
              f"（Gen{man['gen']} vs Gen{man['gen']-1} を評価し GO なら status を解除して再開）。")
        return 0
    vnet, pnet = load_nets(vocab, enc_version=ev)
    print(f"再開: gen={man['gen']} cum_games={man['cum_games']} shards={man['shards']} enc=v{ev} "
          f"target={args.target} workers={args.workers}", flush=True)

    buf_v = {"scalars": [], "field": [], "card_idx": [], "value": []}
    buf_p = []
    pool = mp.Pool(args.workers, initializer=_init_worker)
    try:
        for _ in range(args.max_shards):
            t0 = time.perf_counter()
            vnet.save(CK + "/_cur_v.npz")
            ppath = None
            if pnet is not None:
                pnet.save(CK + "/_cur_p.npz"); ppath = CK + "/_cur_p.npz"
            vdata, pol = selfplay_shard(pool, args.workers, args.shard_games, args.sims,
                                        args.dirichlet_eps, CK + "/_cur_v.npz", ppath, man["shards"],
                                        ev=ev, leaders=leaders)
            if vdata is None:
                print("  採取0スキップ"); continue
            for k in buf_v:
                buf_v[k].append(vdata[k])
            buf_p.extend(pol)
            vb = {k: np.concatenate(buf_v[k])[-args.buffer:] for k in buf_v}
            buf_p[:] = buf_p[-args.buffer:]
            RN.train(vnet, vb, epochs=2, lr=args.lr, batch=256, val_frac=0.05)
            if pnet is None:
                pnet = PolicyScorer(ctx_dim=E.feature_dim(ev), hidden=128, seed=0)
            ce = train_policy(pnet, buf_p, epochs=2, lr=args.lr)
            man["cum_games"] += args.shard_games; man["shards"] += 1
            vnet.save(CK + "/value.npz"); pnet.save(CK + "/policy.npz")
            gate = man["cum_games"] >= args.target
            if gate:
                ng = man["gen"] + 1
                vnet.save(CK + f"/gen{ng}_value.npz"); pnet.save(CK + f"/gen{ng}_policy.npz")
                man["gen"] = ng; man["cum_games"] = 0; man["status"] = "AWAITING_GATE"
            write_manifest(man)
            ok = push_ckpt(f"p3: gen{man['gen']} cum{man['cum_games']} shard{man['shards']}")
            dt = time.perf_counter() - t0
            print(f"  shard{man['shards']} {len(vdata['value'])}局面 buf{len(vb['value'])} "
                  f"ce={ce:.3f} push={'OK' if ok else 'FAIL'} {dt:.0f}s "
                  f"({args.shard_games/dt:.2f} g/s)", flush=True)
            if gate:
                print(f"\n🏁 Gen{man['gen']} 完成。AWAITING_GATE で停止＝人間クロス評価ゲートへ。"); break
    finally:
        pool.close(); pool.join()
    print(f"\n完了: gen={man['gen']} cum_games={man['cum_games']} shards={man['shards']}。再実行で続きから。")
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
