"""n1_gate: N1ネット（value+方策チャネル）の serve 接続とゲート（純正Nループ④ 2026-08-26）。

`n1_train.py` が保存した N1 を LearnedEngine へ**両輪で**注入する:
  - value: `N1ValueAdapter`（vnet ダックタイプ・出口ヘッドは持たない＝has_exit_head False
    →戦闘箱/対話箱の物差しは本体 value に自動フォールバック＝純正の単一価値関数）
  - policy: `n1_priors`（`LearnedEngine.priors_override` seam・候補素性は訓練と同一の
    49次元＝action_type onehot＋主体/対象カード物理＋don_k/対象有無・点内 softmax）

サブコマンド:
  gate  … coach 13点（既定 gen15 vs N1エンジン）＝行動ゲート（`n0_spike.gate` と同じ判定）
  smoke … 実対局1局を N1 エンジン同士で回して完走確認（配線の煙試験）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n1_gate.py gate \\
    --net /home/user/n1_wave1/n1_net.npz --seeds 5
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

import n0_spike as N0
from n1_train import N1Net, ATYPES, NA, D_PHYS, F_CAND

MAX_CI = N0.MAX_CI


class N1ValueAdapter:
    """N1Net → vnet ダックタイプ。出口ヘッド無し＝単一価値関数（純正 AZ）。"""

    def __init__(self, net, tab):
        self.net = net
        self.tab = tab

    def _fwd(self, batch):
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])
        if ci.shape[1] < MAX_CI:
            ci = np.concatenate([ci, np.zeros((len(ci), MAX_CI - ci.shape[1]), ci.dtype)], 1)
        return self.net.value(sc, N0.card_channel(ci[:, :MAX_CI], self.tab))

    def predict(self, batch):
        return self._fwd(batch)

    def predict_with_aux(self, batch):
        v = self._fwd(batch)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return False                       # 出口ヘッド無し→本体 value にフォールバック

    def predict_exit(self, batch, kind):
        return self._fwd(batch)


def _uuid_card(manager, uuid):
    if not uuid:
        return None
    for pl in (manager.p1, manager.p2):
        cards = ([pl.leader] if pl.leader is not None else []) + list(pl.field) \
            + list(pl.hand) + list(pl.trash) + list(pl.life) \
            + ([pl.stage] if getattr(pl, "stage", None) is not None else [])
        for c in cards:
            if c is not None and getattr(c, "uuid", None) == uuid:
                return c
    return None


def _cand_row(manager, mv, vocab, tab):
    """1候補 → 訓練と同一の素性49次元（`n1_train.cand_feats` の serve 版）。"""
    x = np.zeros(F_CAND, np.float32)
    at = mv.get("action_type") or ""
    try:
        x[ATYPES.index(at)] = 1.0
    except ValueError:
        x[NA - 1] = 1.0
    p = mv.get("payload") or {}
    c = _uuid_card(manager, mv.get("card_uuid") or p.get("uuid"))
    if c is not None:
        idx = vocab.get(getattr(getattr(c, "master", None), "card_id", None), 0)
        x[NA:NA + D_PHYS] = tab[idx, :D_PHYS]
    tids = p.get("target_ids") or []
    if tids:
        t = _uuid_card(manager, tids[0])
        if t is not None:
            tidx = vocab.get(getattr(getattr(t, "master", None), "card_id", None), 0)
            x[NA + D_PHYS:NA + 2 * D_PHYS] = tab[tidx, :D_PHYS]
        x[-1] = 1.0
    k = p.get("don_k")
    x[-2] = (float(k) / 5.0) if (at == "DON_BOX" and k is not None) else 0.0
    return x


def n1_priors(net, tab, vocab, enc_version=12):
    """`LearnedEngine.priors_override` 用の priors(state, legal)→np.array。"""
    from opcg_sim.src.learned import encoder as E

    def priors(state, legal):
        try:
            pa = state.pending_actor_action()
            if not pa or not legal:
                return None
            me = pa[0]
            enc = E.encode(state, me, vocab, version=enc_version)
            sc = np.asarray(enc["scalars"], np.float32)[None, :]
            ci = np.zeros((1, MAX_CI), np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            ci[0, :len(src)] = src
            feats = np.stack([_cand_row(state, mv, vocab, tab) for mv in legal])
            seg = np.zeros(len(legal), np.int64)
            lo = net.policy_logits(sc, N0.card_channel(ci, tab), seg, feats)
            return N1Net._seg_softmax(lo, seg, 1)
        except Exception:
            return None                    # priors 失敗は一様（呼び出し側の既定）へ
    return priors


def n1_engine(net_path, **engine_kw):
    """N1 を value+policy 両輪で積んだ LearnedEngine（engine_kw はそのまま渡す＝生成の
    探索多様性 dirichlet_eps/temp_turns 等）。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    tab, vocab = N0.build_card_table()
    net = N1Net.load(net_path)
    eng = LearnedEngine(**engine_kw)
    eng.vnet = N1ValueAdapter(net, tab)
    eng.priors_override = n1_priors(net, tab, vocab, eng.enc_version)
    return eng


def gate(args):
    """coach 13点（既定 gen15 vs N1）＝`n0_spike.gate` と同じ判定基準。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import mark_gate as MG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    # --base-net: 前世代比（N系同士・純正Nループの主ゲート）。未指定=gen15 既定（外部参照）。
    base = n1_engine(args.base_net) if args.base_net else LearnedEngine()
    chall = n1_engine(args.net)
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    rows = []
    for tag, i, accept in CG.VERIFIED_V2:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            rec, fbi, actions = CR.GAMES[tag]
            built = MG._restore(db, rec, fbi, actions, i)
            if isinstance(built, str) or built is None:
                print(f"{tag}@{i}: 復元不可"); continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        b = CG.decide_rate(base, m0, actor, accept, args.seeds, 160)
        c = CG.decide_rate(chall, m0, actor, accept, args.seeds, 160)
        rows.append((tag, i, b, c))
        print(f"  {tag}@{i:<4} base={b:.2f} n1={c:.2f}", flush=True)
    ok_nr, ok_imp, regs = CG.judge(rows)
    print(f"改善: {'OK' if ok_imp else 'NG'}（n1計 {sum(c for *_, c in rows):.1f}"
          f" vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {regs}")
    print("N1_GATE_RESULT", json.dumps({"verdict": "PASS" if (ok_nr and ok_imp) else "FAIL"}))
    return 0


def smoke(args):
    """N1 エンジン同士で1局完走（配線の煙試験・void/例外/優先度の欠陥検出）。"""
    import random
    from opcg_game import OPCGGame
    from cpu_selfplay import _load_db
    from deck_synth import synth_deck
    from opcg_sim.src.core.gamestate import GameManager, Player
    db = _load_db()
    eng = n1_engine(args.net)
    gs = OPCGGame()
    random.seed(args.seed)
    leaders = sorted(cid for cid, _ in db.raw_db.items()
                     if (db.get_card(cid) is not None
                         and getattr(db.get_card(cid).type, "name", "") == "LEADER"))
    rl = random.Random(args.seed * 7919 + 13)
    la, lb = rl.choice(leaders), rl.choice(leaders)
    l1, c1 = synth_deck(db, la, seed=args.seed, owner="p1")
    l2, c2 = synth_deck(db, lb, seed=args.seed + 1, owner="p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    drng = np.random.default_rng(args.seed)
    steps = 0
    while m.winner is None and not gs.is_terminal(m) and steps < 400:
        name = gs.current_player(m)
        if name is None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=args.sims, rng=drng)
        if mv is None:
            break
        m2 = gs.apply(m, mv, name)
        if m2 is None:
            print("N1_SMOKE_RESULT", json.dumps({"verdict": "FAIL", "step": steps}))
            return 1
        m = m2
        steps += 1
    ok = m.winner is not None
    print("N1_SMOKE_RESULT", json.dumps(
        {"verdict": "PASS" if ok else "FAIL", "winner": m.winner, "steps": steps,
         "turns": int(getattr(m, "turn_count", 0) or 0)}))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--net", required=True)
    g.add_argument("--seeds", type=int, default=5)
    g.add_argument("--base-net", default=None,
                   help="基準側も N 系ネットにする（前世代比）。未指定=出荷 gen15")
    s = sub.add_parser("smoke")
    s.add_argument("--net", required=True)
    s.add_argument("--seed", type=int, default=424242)
    s.add_argument("--sims", type=int, default=32)
    args = ap.parse_args()
    return gate(args) if args.cmd == "gate" else smoke(args)


if __name__ == "__main__":
    _sys.exit(main())
