"""P3: AZ自己対戦RLループ（OPCG・value+pointer policy）＋世代間クロス評価。

docs/.../cpu_rl_pilot_plan_20260629.md P3。Gen0(=P2のSL価値net＋uniform prior)で自己対戦し、
(局面, MCTS訪問分布, 最終勝敗) を採取→ value net(outcome)＋policy(訪問分布) を学習して Gen1…と進める。
policy は uniform から **RL で育てる**（P2でL1模倣policyを足さない＝模倣の天井回避・レビュー確定）。

判定はクロス評価（gen N+1 vs gen N／対Gen0）。損切り（レビュー確定）:
  Gen1 vs Gen0 ≥0.55 で続行・以降の対前世代は 0.51〜0.52(後退でないこと)・
  Gen3までに「Gen_k vs Gen0」が0.55未達なら停止。**本走は N=400 CRN・常設CPU VM**。
  本環境＝ハーネス＋インフラ試走（疎通のみ・勝率は無視）。

スモーク: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_loop.py --smoke --enc-version 1
        （--enc-version は必須。版はこの引数のみで決まる＝2 でリーダー付与ドン特徴の v2）
"""
import argparse
import random
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.config import SELFPLAY_TEMP_MOVES
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS   # make/unmake版（唯一の探索実装。旧clone版は削除済み）
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer, state_context, train_policy
from opcg_action import legal_action_matrix
from cpu_selfplay import _load_db


# ---- ネットを MCTS の value_fn / priors_fn に変換 ----
def value_fn_of(net, vocab, enc_version=1):
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab, version=enc_version)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(net.predict(batch)[0])
    return value


def priors_fn_of(policy, vocab, enc_version=1):
    if policy is None:
        return None
    def priors(state, legal):
        me = state.pending_actor_action()[0]
        ctx = state_context(state, me, vocab, version=enc_version)
        am = legal_action_matrix(state, legal, me)
        p = policy.priors(ctx, am)
        return p if p.shape[0] == len(legal) else None
    return priors


def _sample(counts, rng, temp):
    if temp <= 1e-6:
        return int(np.argmax(counts))
    p = counts.astype(np.float64) ** (1.0 / temp)
    s = p.sum()
    return int(rng.choice(len(p), p=p / s)) if s > 0 else int(np.argmax(counts))


# ---- 自己対戦でデータ採取 ----
_BATTLE_RESPONSES = ("SELECT_BLOCKER", "SELECT_COUNTER")


def selfplay_game(game, value_fn, priors_fn, vocab, sims, c_puct, rng, temp_moves=SELFPLAY_TEMP_MOVES,
                  max_steps=400, enc_version=1, leaders=None, dirichlet_eps=0.0, db=None,
                  l1_seat=None, seed_boards=None, seed_frac=0.0):
    """1局の自己対戦データを採取する（直列/pd並列生成の共通コア・v4計画 §4-1）。

    v4 での拡張（docs/cpu_v4_plan.md）:
    - **(a) sticky 世界線**: PIMC 決定化 seed を (turn, 手番) 単位で固定＝ターン内の連続 decide が
      同一世界でプランを評価する（本番 `cpu_learned._world_rng` と同じ規約）。dirichlet ノイズは
      従来どおり毎 decide 新鮮（探索多様性は維持）。
    - **(c) 防御応答の温度延長**: SELECT_BLOCKER/SELECT_COUNTER は steps に依らず温度サンプリング＝
      「カウンターを切って延命した対局」を分布に入れる（prior が PASS 寄りでも実際に切られる）。
    - **(d) 対戦相手の混合**: `l1_seat`（"p1"/"p2"）を指定すると、その席は L1-hard
      （`cpu_ai.decide_guarded`＝時計を手書きで持ち実際に防御する）が指す。**ゲームを伸ばす
      プレイヤーがいる長さ分散のある対局**を供給する。policy 教師は net 席の決定のみ・
      value/turns_left は対局全体から採る（L1 席の q_root は NaN→merge で勝敗ラベルへ退化）。
    - **q_root/turns_left の記録**: 混合ラベル（§4-2）と残りターン補助ターゲット用。
      q_root = 探索後 root の ΣN·Q/ΣN（to-move 視点・終局減衰込み）。

    v5 での拡張（cpu_v5_plan.md §4-2）:
    - **(e) マーク局面シード**: `seed_boards`（復元済み失敗局面 GameManager のリスト）を渡し、
      各局を確率 `seed_frac` でそのプールから開始する（残りは通常の turn1 new_game）。観測された
      失敗モードそのものを in-distribution 化する。中盤開始でも軌跡・ラベル採取は通常経路と同一
      （turns_left は終局ターンから逆算＝開始ターンに依らず正しい）。`seed_frac=0`（既定）で挙動不変
      （rng 消費順も従来どおり＝seed_boards 未指定 or frac=0 のとき seed 判定の乱数を引かない）。

    返り値: (val_recs, pol_recs, winner)。val_recs は (enc, who, q_root, turns_left)。
    """
    if seed_boards and seed_frac > 0.0 and float(rng.random()) < float(seed_frac):
        # マーク局面から開始（プールから決定論抽選＝rng で再現可能）。開始盤面は毎局クローン
        # （selfplay は m を破壊的に進めるため・プール本体は不変で使い回す）。
        m = seed_boards[int(rng.integers(len(seed_boards)))].clone()
    else:
        m = game.new_game(db=(db if db is not None else _DB), seed=int(rng.integers(1 << 30)),
                          leaders=leaders)
    val_recs, pol_recs = [], []   # (enc, who, q_root, turn_no) / (ctx, am, visit, who)
    steps = 0
    world_seeds = {}   # (turn, name) -> 決定化 seed。dict＝戦闘応答で手番が交互に挟まっても sticky
    l1_rng = random.Random(int(rng.integers(1 << 30))) if l1_seat else None
    l1_mem = {}
    while game.winner(m) is None and not game.is_terminal(m) and steps < max_steps:
        name = game.current_player(m)
        if name is None:
            break
        turn_no = int(getattr(m, "turn_count", 0) or 0)
        enc = E.encode(m, name, vocab, version=enc_version)

        if name == l1_seat:
            # (d) L1-hard 席: 探索統計が無いので policy 教師は採らず、value は q_root=NaN で記録
            #（merge_val_recs が勝敗ラベルへ退化させる）。
            actor = m.p1 if m.p1.name == name else m.p2
            move = cpu_ai.decide_guarded(m, actor, "hard", l1_rng, l1_mem)
            if move is None:
                break
            val_recs.append((enc, name, float("nan"), turn_no))
            try:
                cpu_ai._apply_move_inplace(m, name, move)
            except Exception:
                break
            steps += 1
            continue

        key = (turn_no, name)
        det_seed = world_seeds.get(key)           # (a) sticky: 同一 (turn, 手番) は同一世界
        if det_seed is None:
            det_seed = world_seeds[key] = int(rng.integers(2 ** 63 - 1))
        mcts = TreeMCTS(game, value_fn=value_fn, priors_fn=priors_fn, c_puct=c_puct,
                        n_sims=sims, dirichlet_eps=dirichlet_eps,
                        determinize_fn=lambda s, r, _sd=det_seed:
                            game.determinize(s, name, np.random.default_rng(_sd)),
                        rng=rng)
        move, N, legal = mcts.run(m)
        if move is None or N is None or N.sum() == 0:
            break
        stats = mcts.last_stats
        q_root = float((stats["N"] * stats["Q"]).sum() / max(float(stats["N"].sum()), 1.0))
        visit = N.astype(np.float64) / N.sum()
        val_recs.append((enc, name, q_root, turn_no))
        ctx = state_context(m, name, vocab, version=enc_version)
        am = legal_action_matrix(m, legal, name)
        pol_recs.append((ctx, am, visit, name))
        pend = m.get_pending_request() or {}
        is_battle_resp = pend.get("action") in _BATTLE_RESPONSES
        a = _sample(N, rng, temp=1.0 if (steps < temp_moves or is_battle_resp) else 0.0)
        try:
            cpu_ai._apply_move_inplace(m, name, legal[a])
        except Exception:
            break
        steps += 1
    winner = game.winner(m)
    end_turn = int(getattr(m, "turn_count", 0) or 0)
    # turns_left = 終局ターン − 記録時ターン（対局終了後に逆算・v4 §4-2 の補助ターゲット）。
    val_recs = [(enc, who, q, max(0, end_turn - t)) for enc, who, q, t in val_recs]
    return val_recs, pol_recs, winner


def merge_val_recs(val_recs, winner, sinks):
    """val_recs（selfplay_game の返り値）を配列バッファ dict へ展開する（直列/並列生成の共通処理）。

    sinks: {"S":[], "F":[], "I":[], "Y":[], "Q":[], "T":[]}（呼び出し側で用意・複数局で使い回し）。
    q_root=NaN（L1 席＝探索統計なし）は勝敗ラベルへ退化させる（混合ラベルが z 単独になる）。
    """
    import math as _math
    for enc, who, q_root, turns_left in val_recs:
        z = 1.0 if who == winner else -1.0
        sinks["S"].append(enc["scalars"]); sinks["F"].append(enc["field"])
        sinks["I"].append(enc["card_idx"])
        sinks["Y"].append(z)
        sinks["Q"].append(q_root if _math.isfinite(q_root) else z)
        sinks["T"].append(turns_left)


def pack_vdata(sinks):
    """sinks（merge_val_recs で充填）→ vdata dict（batch.npz スキーマ v2 のvalue側）。空なら None。"""
    if not sinks["S"]:
        return None
    return {"scalars": np.stack(sinks["S"]), "field": np.stack(sinks["F"]),
            "card_idx": np.stack(sinks["I"]), "value": np.array(sinks["Y"], dtype=np.float32),
            "q_root": np.array(sinks["Q"], dtype=np.float32),
            "turns_left": np.array(sinks["T"], dtype=np.float32)}


def generate(game, value_fn, priors_fn, vocab, n_games, sims, c_puct, rng, log=print, enc_version=1,
             leaders=None):
    sinks = {"S": [], "F": [], "I": [], "Y": [], "Q": [], "T": []}
    pol = []
    for g in range(n_games):
        vr, pr, w = selfplay_game(game, value_fn, priors_fn, vocab, sims, c_puct, rng,
                                  enc_version=enc_version, leaders=leaders)
        if w is None:
            continue
        merge_val_recs(vr, w, sinks)
        for ctx, am, visit, who in pr:
            pol.append((ctx, am, visit))
        if (g + 1) % 5 == 0:
            log(f"  selfplay {g+1}/{n_games}（value局面{len(sinks['Y'])} policy{len(pol)}）", flush=True)
    return pack_vdata(sinks), (pol if sinks["S"] else None)


def train_generation(vocab, vdata, pol, d_emb=24, hidden=128, v_epochs=20, p_epochs=4, seed=0, log=print,
                     enc_version=1):
    vnet = RN.ValueNet(len(vocab), d_emb=d_emb, hidden=hidden, feat_dim=E.feature_dim(enc_version), seed=seed)
    tm, vm = RN.train(vnet, vdata, epochs=v_epochs, lr=2e-3, batch=256, val_frac=0.05)
    pnet = PolicyScorer(ctx_dim=E.feature_dim(enc_version), hidden=hidden, seed=seed)
    ce = train_policy(pnet, pol, epochs=p_epochs, lr=2e-3)
    log(f"  train: value mse={tm:.3f}/{vm:.3f}  policy ce={ce:.3f}", flush=True)
    return vnet, pnet


# ---- クロス評価（CRN・先後交互） ----
def _agent(game, vnet, pnet, vocab, sims, c_puct, enc_version=1):
    vf = value_fn_of(vnet, vocab, enc_version); pf = priors_fn_of(pnet, vocab, enc_version)
    def act(state, name, rng):
        mcts = TreeMCTS(game, value_fn=vf, priors_fn=pf, c_puct=c_puct, n_sims=sims,
                        determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
        move, _, _ = mcts.run(state)
        if move is None:
            legal = game.legal_actions(state)
            move = legal[0] if legal else None
        return move
    return act


def cross_eval(game, agentA, agentB, pairs, seed0=3000, leaders=None):
    res = {"a_win": 0, "draw": 0, "a_loss": 0}
    for i in range(pairs):
        for a_is_p1 in (True, False):
            m = game.new_game(_DB, seed0 + i, leaders=leaders)
            rng = np.random.default_rng((seed0 + i) * 7 + (0 if a_is_p1 else 1))
            steps = 0
            while game.winner(m) is None and not game.is_terminal(m) and steps < 400:
                name = game.current_player(m)
                if name is None:
                    break
                ag = agentA if (name == "p1") == a_is_p1 else agentB
                mv = ag(m, name, rng)
                if mv is None:
                    break
                try:
                    cpu_ai._apply_move_inplace(m, name, mv)
                except Exception:
                    break
                steps += 1
            w = game.winner(m)
            if w is None:
                res["draw"] += 1
            else:
                res["a_win" if ((w == "p1") == a_is_p1) else "a_loss"] += 1
    res["games"] = pairs * 2
    return res


_DB = None


def main():
    global _DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="疎通のみ（勝率は無視・dev用）")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--gens", type=int, default=1)
    ap.add_argument("--eval-pairs", type=int, default=3)
    ap.add_argument("--sl-net", default=None, help="Gen0 value net（無ければ乱数初期化）")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--enc-version", type=int, required=True, choices=(1, 2),
                    help="符号化世代（必須・版はこの引数のみで決まる。2=リーダー付与ドン特徴。"
                         "v2 は Gen0 から学習し直す）")
    ap.add_argument("--rotate-leaders", action="store_true",
                    help="自己対戦のリーダーを全リーダーから抽選＋リアルデッキ化（穴B: 分布多様化）")
    args = ap.parse_args()

    _DB = _load_db()
    vocab = E.build_vocab(_DB)
    game = OPCGGame()
    rng = np.random.default_rng(0)

    leaders = None
    if args.rotate_leaders:
        from deckgen import all_leader_ids
        leaders = all_leader_ids(_DB)
        print(f"リーダーローテーション ON: {len(leaders)} 種から抽選", flush=True)

    # Gen0: value=SL net(or 乱数)・policy=uniform(None)。
    if args.sl_net:
        v0 = RN.ValueNet.load(args.sl_net); print(f"Gen0 value net: {args.sl_net}", flush=True)
        loaded_feat = v0.feat_dim
        if loaded_feat != E.feature_dim(args.enc_version):
            print(f"ERROR: --sl-net の入力次元 {loaded_feat} が enc-version={args.enc_version} "
                  f"(feat_dim={E.feature_dim(args.enc_version)}) と不一致", flush=True)
            return 1
    else:
        v0 = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(args.enc_version), seed=0)
    gens = [(v0, None)]   # (value_net, policy or None)

    print(f"=== P3 {'SMOKE' if args.smoke else 'RUN'}: gens={args.gens} games/gen={args.games} "
          f"sims={args.sims} ===", flush=True)
    for g in range(args.gens):
        vnet, pnet = gens[-1]
        t0 = time.perf_counter()
        vdata, pol = generate(game, value_fn_of(vnet, vocab, args.enc_version),
                              priors_fn_of(pnet, vocab, args.enc_version),
                              vocab, args.games, args.sims, args.c_puct, rng,
                              enc_version=args.enc_version, leaders=leaders)
        if vdata is None:
            print("データ0（全局未決着）"); return 1
        nv, npnet = train_generation(vocab, vdata, pol, seed=g, enc_version=args.enc_version)
        gens.append((nv, npnet))
        # クロス評価: 新世代 vs 直前世代。
        a_new = _agent(game, nv, npnet, vocab, args.sims, args.c_puct, args.enc_version)
        a_old = _agent(game, vnet, pnet, vocab, args.sims, args.c_puct, args.enc_version)
        r = cross_eval(game, a_new, a_old, args.eval_pairs, leaders=leaders)
        wr = (r["a_win"] + 0.5 * r["draw"]) / r["games"]
        print(f"Gen{g+1} vs Gen{g}: 勝率={wr:.3f} {r}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    if args.smoke:
        print("\nSMOKE: ループ疎通OK（自己対戦→value/policy学習→クロス評価が例外なく完走）。"
              "※勝率は乱数＝判定に使わない（レビュー確定）。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
