"""n_record: 純正Nループの棋譜ダンプ生成器（2026-08-26・ユーザ決定「純正に準じてやりましょう」）。

**設計**: AlphaZero の自己対戦棋譜に相当する生データを1本で採る——
  ①全判断点（main=木探索／window=窓の根畳み／commit=箱コミット機械実行）の v12 符号化＋勝敗 z
  ②main 窓は**候補と訪問分布**（`_merge_root_stats` と同一集計＝decide の選択と同じ等価マージ後）
採掘器（z 密教師・方策ターゲット）は別計器＝ダンプは生のまま保存し、教師の取り方を後から
変えられるようにする（スナップショット別生成器を教師パターンごとに増やさない）。

**規約**: ランダムリーダー×生成デッキ（`promotion_gate._leader_pair`/`deck_synth` と同規約
seed*7919+13）。候補生成は prune_futile=GEN_PRUNE_FUTILE（生成は枝刈りを外す＝v6 柱⑤）。
探索プロファイルは serve 既定（箱化一式＋箱コミット ON）＝実対局と同じ行動列。
seed 決定論（global random / numpy drng とも seed 導出）。

実行例（分散: 子が --seed-base を分担）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_record_gen.py \\
    --games 60 --seed-base 200000 --workers 4 --sims 64 --out n_records/part1

シャード npz の列（D=判断点行・K=main候補の flatten）:
  scalars(D,94) field(D,·) card_idx(D,24)   … v12 符号化（判断点の状態・手番視点）
  z(D)                                        … 勝敗 ±1（手番視点・純正 AZ の素の z）
  who(D) kind(D: 0=main/1=window/2=commit) turn(D) step(D) seed(D)
  sig(D str)                                  … 選択手の move_sig（JSON・箱レベル）
  pol_len(D) pol_chosen(D)                    … main の候補数／選択候補の slice 内 index（無ければ -1/0）
  pol_n(K) pol_q(K) pol_k(K) pol_sig(K str)   … 訪問合算 n・行動価値 q・don_k（-1=無し・配分箱の
                                                k 違いは同 sig のため必須）・候補 move_sig（JSON）
  pol_cid(K str) pol_tcid(K str)              … 候補の主体/第1対象の**カードID**（uuid は対局固有で
                                                事後解決できないためダンプ時に解決・無ければ ""）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

MAX_STEPS = 400
MAX_CI = 24            # card_idx の PAD 長（G系符号化の既定枠）
ENC_VERSION = 12       # 現行 G 系の符号化世代（v12）
_G = {}

_KIND = {"main": 0, "window": 1, "commit": 2}


def _init_worker(sims, value_path, policy_path, n1_net=None,
                 dirichlet_eps=0.0, temp_turns=0, neff_net=None, dump_v2=False):
    import rl_encoder as E
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned.adapter import OPCGGame as _AG
    from opcg_sim.src.learned.config import GEN_PRUNE_FUTILE
    db = _load_db()
    # 生成は枝刈りを外す（GEN_PRUNE_FUTILE=False・v6 柱⑤: 刈った枝は学習データに現れない）。
    # それ以外（箱化・箱コミット・quiesce 等）は serve 既定のまま＝実対局と同じ行動列。
    ex = dict(dirichlet_eps=(dirichlet_eps or None), temp_turns=(temp_turns or None))
    if neff_net:
        # 効果構造符号化ネットで生成（NEff が生成役ゲートを通過した後の本流）
        from n_eff_gate import neff_engine
        eng = neff_engine(neff_net, **ex)
        eng.game = _AG(prune_futile=GEN_PRUNE_FUTILE)
    elif n1_net:
        # 第2波以降（純正AZ: 最新のNネットで生成する）。value+方策とも N1 を注入。
        from n1_gate import n1_engine
        eng = n1_engine(n1_net, **ex)
        eng.game = _AG(prune_futile=GEN_PRUNE_FUTILE)
    else:
        eng = LearnedEngine(value_path=value_path, policy_path=policy_path,
                            game=_AG(prune_futile=GEN_PRUNE_FUTILE), **ex)
    leaders = sorted(cid for cid, _ in db.raw_db.items()
                     if (db.get_card(cid) is not None
                         and getattr(db.get_card(cid).type, "name", "") == "LEADER"))
    _G.update(E=E, db=db, gs=OPCGGame(), eng=eng, vocab=eng.vocab,
              sims=sims, leaders=leaders, v2=bool(dump_v2))
    if dump_v2:
        # dump v2（NRel P1・2026-09-04）: 符号化は本番 encoder の v13（v12＋グローバル追加 29）、
        # トークン状態 S（float32・`n_rel_feat.encode_rel`）と候補の主体/対象の 22 枠 index を足す。
        # 関係 R は保存しない（訓練時に `relations_from_dump` で再計算＝ユーザ決定）。
        from opcg_sim.src.learned import encoder as PE
        from opcg_sim.src.learned import n_rel_feat as NR
        _G.update(PE=PE, NR=NR)


def _uuid_cids(m):
    """uuid → カードID（master.id）の写像（両者の leader/hand/field/stage/trash/life）。

    方策ターゲットの候補（PLAY/ATTACK/DON_BOX 等の主体・対象）はこの範囲で足りる
    （デッキ内サーチ等の選択は対話窓＝main の候補に uuid が出ない）。"""
    out = {}
    for p in (m.p1, m.p2):
        cards = list(p.hand) + list(p.field) + list(p.trash) + list(p.life)
        if p.leader is not None:
            cards.append(p.leader)
        if p.stage is not None:
            cards.append(p.stage)
        for c in cards:
            u = getattr(c, "uuid", None)
            cid = getattr(getattr(c, "master", None), "card_id", None)
            if u and cid:
                out[u] = str(cid)
    return out


def play_one(seed):
    E, gs, eng = _G["E"], _G["gs"], _G["eng"]
    from opcg_sim.src.core.gamestate import GameManager, Player
    random.seed(seed)
    try:
        from deck_synth import synth_deck
        rl = random.Random(seed * 7919 + 13)          # promotion_gate._leader_pair と同規約
        la, lb = rl.choice(_G["leaders"]), rl.choice(_G["leaders"])
        l1, c1 = synth_deck(_G["db"], la, seed=seed, owner="p1")
        l2, c2 = synth_deck(_G["db"], lb, seed=seed + 1, owner="p2")
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        m.start_game()
    except Exception:
        return None
    rows = {k: [] for k in ("scalars", "field", "card_idx", "who", "kind", "turn",
                            "step", "sig", "pol_len", "pol_chosen", "tokens")}
    pol = {"n": [], "q": [], "k": [], "sig": [], "cid": [], "tcid": [], "si": [], "ti": []}
    acts = {"p1": 0, "p2": 0}
    steps = 0
    drng = np.random.default_rng(seed * 31 + 7)
    try:
        while m.winner is None and not gs.is_terminal(m) and steps < MAX_STEPS:
            name = gs.current_player(m)
            if name is None:
                break
            actor = m.p1 if m.p1.name == name else m.p2
            if _G.get("v2"):
                enc = _G["PE"].encode(m, name, _G["vocab"], version=ENC_VERSION_V2)
            else:
                enc = E.encode(m, name, _G["vocab"], version=ENC_VERSION)
            rec = {}
            eng._world_seeds = {}
            mv = eng.decide(m, actor, sims=_G["sims"], rng=drng, record=rec)
            if mv is None:
                break
            # 判断点の行（符号化は decide 前の状態＝この判断が見た盤面）
            rows["scalars"].append(enc["scalars"])
            rows["field"].append(enc["field"])
            ci = np.zeros(MAX_CI, np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            ci[:len(src)] = src
            rows["card_idx"].append(ci)
            rows["who"].append(name)
            rows["kind"].append(_KIND.get(rec.get("kind"), 0))
            rows["turn"].append(int(getattr(m, "turn_count", 0) or 0))
            rows["step"].append(steps)
            rows["sig"].append(json.dumps(rec.get("sig"), ensure_ascii=False))
            groups = rec.get("groups") or []
            rows["pol_len"].append(len(groups))
            chosen = -1
            for gi, g in enumerate(groups):
                # 配分箱は k 違いが同 sig（move_sig は don_k 非含有）＝ (sig, k) で照合
                if g["sig"] == rec.get("sig") and g.get("k") == rec.get("k"):
                    chosen = gi
                    break
            rows["pol_chosen"].append(chosen)
            cids = _uuid_cids(m) if groups else {}
            smap = {}
            if _G.get("v2"):
                NR = _G["NR"]
                rows["tokens"].append(NR.encode_rel(m, name)["tokens"])
                _me = m.p1 if m.p1.name == name else m.p2
                _opp = m.p2 if _me is m.p1 else m.p1
                smap = {getattr(c, "uuid", None): i
                        for i, c in enumerate(NR._slots(_me, _opp)) if c is not None}
            for g in groups:
                pol["n"].append(float(g["n"]))
                pol["q"].append(float(g["q"]))
                pol["k"].append(-1 if g.get("k") is None else int(g["k"]))
                pol["sig"].append(json.dumps(g["sig"], ensure_ascii=False))
                sg = g["sig"]
                pol["cid"].append(cids.get(sg[1]) or "")
                tg = list(sg[2] or ())
                pol["tcid"].append((cids.get(tg[0]) or "") if tg else "")
                if _G.get("v2"):
                    pol["si"].append(smap.get(sg[1], -1))
                    pol["ti"].append(smap.get(tg[0], -1) if tg else -1)
            m2 = gs.apply(m, mv, name)
            if m2 is None:
                return None
            m = m2
            steps += 1
            d = mv.get("action_type") if isinstance(mv, dict) else None
            if d not in (None, "TURN_END", "PASS", "KEEP_HAND", "MULLIGAN"):
                acts[name] = acts.get(name, 0) + 1
    except Exception:
        return None
    turn = int(getattr(m, "turn_count", 0) or 0)
    if (m.winner is None or (m.winner is not None and turn < 4)
            or steps >= MAX_STEPS or min(acts.values()) == 0):
        return None                                  # 純正 z＝勝敗が付いた対局のみ採用
    z = {"p1": 1.0 if m.winner == "p1" else -1.0}
    z["p2"] = -z["p1"]
    out_v2 = {}
    if _G.get("v2"):
        out_v2 = {"tokens": np.array(rows["tokens"], np.float32),
                  "pol_si": np.array(pol["si"], np.int16), "pol_ti": np.array(pol["ti"], np.int16)}
    return {**out_v2, "scalars": np.array(rows["scalars"], np.float32),
            "field": np.array(rows["field"], np.float32),
            "card_idx": np.array(rows["card_idx"], np.int64),
            "z": np.array([z[w] for w in rows["who"]], np.float32),
            "who": np.array([0 if w == "p1" else 1 for w in rows["who"]], np.int8),
            "kind": np.array(rows["kind"], np.int8),
            "turn": np.array(rows["turn"], np.int16),
            "step": np.array(rows["step"], np.int32),
            "seed": np.full(len(rows["who"]), seed, np.int64),
            "sig": np.array(rows["sig"]),
            "pol_len": np.array(rows["pol_len"], np.int32),
            "pol_chosen": np.array(rows["pol_chosen"], np.int16),
            "pol_n": np.array(pol["n"], np.float32),
            "pol_q": np.array(pol["q"], np.float32),
            "pol_k": np.array(pol["k"], np.int16),
            "pol_sig": np.array(pol["sig"]) if pol["sig"] else np.array([], dtype="U1"),
            "pol_cid": np.array(pol["cid"]) if pol["cid"] else np.array([], dtype="U1"),
            "pol_tcid": np.array(pol["tcid"]) if pol["tcid"] else np.array([], dtype="U1")}


_ROW_KEYS = ("scalars", "field", "card_idx", "z", "who", "kind", "turn", "step",
             "seed", "sig", "pol_len", "pol_chosen")
_POL_KEYS = ("pol_n", "pol_q", "pol_k", "pol_sig", "pol_cid", "pol_tcid")
_V2_KEYS = ("tokens", "pol_si", "pol_ti")
ENC_VERSION_V2 = 13    # dump v2 の符号化世代（v12 + n_rel_feat のグローバル追加 29・append-only）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=64,
                    help="生成 sims（純正Nループは 32→64+ に引き上げ・2026-08-26 設計）")
    ap.add_argument("--value", default=None, help="価値ネット npz（既定=出荷 G15）")
    ap.add_argument("--policy", default=None, help="方策ネット npz（既定=出荷 G15）")
    ap.add_argument("--n1-net", default=None,
                    help="N系ネット npz（指定時は value/policy を無視して N1 エンジンで生成）")
    ap.add_argument("--dirichlet-eps", type=float, default=0.25,
                    help="root priors への Dirichlet ノイズ（純正AZ の自己対戦既定 0.25・0=無効）")
    ap.add_argument("--neff-net", default=None,
                    help="効果構造符号化ネット npz（指定時は NEff エンジンで生成）")
    ap.add_argument("--temp-turns", type=int, default=4,
                    help="この turn まではメイン窓を訪問分布 τ=1 でサンプリング（0=無効）")
    ap.add_argument("--shard-games", type=int, default=10)
    ap.add_argument("--dump-v2", action="store_true",
                    help="NRel 用 dump v2（符号化 v13＋トークン状態 S float32＋候補の枠 index・2026-09-04）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    keys = _ROW_KEYS + _POL_KEYS + (_V2_KEYS if args.dump_v2 else ())
    buf = {k: [] for k in keys}
    shard = n_rows = n_drop = n_main = 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=(args.sims, args.value,
                                                args.policy, args.n1_net,
                                                args.dirichlet_eps, args.temp_turns,
                                                args.neff_net, args.dump_v2)) as pool:
        done = 0
        for r in pool.imap_unordered(play_one,
                                     [args.seed_base + i for i in range(args.games)]):
            done += 1
            if r is None:
                n_drop += 1
            else:
                for k in buf:
                    buf[k].append(r[k])
                n_rows += len(r["z"])
                n_main += int((r["kind"] == 0).sum())
            if done % args.shard_games == 0 or done == args.games:
                if buf["z"]:
                    path = os.path.join(args.out, f"n_record_{shard:05d}.npz")
                    tmp = os.path.join(args.out, f".n_record_{shard:05d}.tmp.npz")
                    np.savez_compressed(tmp, **{k: np.concatenate(buf[k]) for k in buf})
                    os.replace(tmp, path)
                    shard += 1
                    buf = {k: [] for k in keys}
                print(f"  {done}/{args.games}局 行{n_rows}（main {n_main}） 棄却{n_drop}"
                      f" {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out, "meta_n_record.json"), "w") as f:
        json.dump({"games": args.games, "rows": n_rows, "main_rows": n_main,
                   "dropped": n_drop, "sims": args.sims,
                   "enc_version": ENC_VERSION_V2 if args.dump_v2 else ENC_VERSION,
                   "dump_version": 2 if args.dump_v2 else 1, "seed_base": args.seed_base,
                   "value": args.value, "policy": args.policy,
                   "n1_net": args.n1_net, "neff_net": args.neff_net,
                   "dirichlet_eps": args.dirichlet_eps,
                   "temp_turns": args.temp_turns}, f, ensure_ascii=False)
    print("N_RECORD_DONE " + json.dumps({"rows": n_rows, "main_rows": n_main,
                                         "dropped": n_drop}))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
