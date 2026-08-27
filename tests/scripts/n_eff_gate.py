"""n_eff_gate: 効果構造符号化ネットの serve 接続とゲート（`n_eff_train` の対・2026-08-27）。

`n1_gate` と同じ構図（value=vnet ダックタイプ・policy=priors_override seam）で、
カード表現だけ効果埋め込み64次元に置換。serve では重みが凍結なので**カード表は
アダプタ初期化時に1回だけ前計算**する（毎 decide の再計算をしない）。
候補素性は訓練と同一の139次元（printed パワーマージン含む＝train/serve 一致）。

サブコマンド:
  gate  … coach 13点（--base-net 指定で N 系前世代比・未指定=gen15）
  smoke … NEff 同士の実対局1局完走（配線の煙試験）
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

from n1_gate import _uuid_card
from n1_train import ATYPES
from n_eff_feat import build_eff_tables
from n_eff_train import NEffNet, D_CARD_FEAT, F_CAND, MAX_CI, NA


class NEffValueAdapter:
    """NEffNet → vnet ダックタイプ（表は前計算・出口ヘッド無し＝単一価値関数）。"""

    def __init__(self, net):
        self.net = net
        self.tab = net.card_table()          # serve は重み凍結＝1回だけ計算

    def _fwd(self, batch):
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])
        if ci.shape[1] < MAX_CI:
            ci = np.concatenate([ci, np.zeros((len(ci), MAX_CI - ci.shape[1]), ci.dtype)], 1)
        r2 = self.net.body(sc, self.net.cards_in(ci[:, :MAX_CI], self.tab))
        return np.tanh((r2 @ self.net.Wv + self.net.bv)[:, 0])

    def predict(self, batch):
        return self._fwd(batch)

    def predict_with_aux(self, batch):
        v = self._fwd(batch)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return False

    def predict_exit(self, batch, kind):
        return self._fwd(batch)


def _cand_row(net, tab, manager, mv, vocab):
    """1候補 → 訓練と同一の素性139次元（serve 版）。"""
    x = np.zeros(F_CAND, np.float32)
    at = mv.get("action_type") or ""
    try:
        x[ATYPES.index(at)] = 1.0
    except ValueError:
        x[NA - 1] = 1.0
    p = mv.get("payload") or {}
    ci = ti = 0
    c = _uuid_card(manager, mv.get("card_uuid") or p.get("uuid"))
    if c is not None:
        ci = vocab.get(getattr(getattr(c, "master", None), "card_id", None), 0)
    x[NA:NA + D_CARD_FEAT] = tab[ci]
    tids = p.get("target_ids") or []
    if tids:
        t = _uuid_card(manager, tids[0])
        if t is not None:
            ti = vocab.get(getattr(getattr(t, "master", None), "card_id", None), 0)
        x[NA + D_CARD_FEAT:NA + 2 * D_CARD_FEAT] = tab[ti]
    k = p.get("don_k")
    kk = float(k) if (at == "DON_BOX" and k is not None) else 0.0
    has_t = 1.0 if tids else 0.0
    x[-4] = kk / 5.0
    x[-3] = has_t
    x[-2] = float(np.clip((net.PWR[ci] + kk * 1000.0 - net.PWR[ti]) / 10000.0, -1, 1)) * has_t
    x[-1] = float(net.ISL[ti]) * has_t
    return x


def neff_priors(net, tab, vocab, enc_version=12):
    from opcg_sim.src.learned import encoder as E

    def priors(state, legal):
        try:
            pa = state.pending_actor_action()
            if not pa or not legal:
                return None
            enc = E.encode(state, pa[0], vocab, version=enc_version)
            sc = np.asarray(enc["scalars"], np.float32)[None, :]
            ci = np.zeros((1, MAX_CI), np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            ci[0, :len(src)] = src
            feats = np.stack([_cand_row(net, tab, state, mv, vocab) for mv in legal])
            r2 = net.body(sc, net.cards_in(ci, tab))
            seg = np.zeros(len(legal), np.int64)
            u = np.concatenate([r2[seg], feats], 1)
            rp = np.maximum(u @ net.Wp1 + net.bp1, 0.0)
            lo = (rp @ net.Wp2 + net.bp2)[:, 0]
            return NEffNet._seg_softmax(lo, seg, 1)
        except Exception:
            return None
    return priors


def neff_engine(net_path, **engine_kw):
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    net = NEffNet.load(net_path, tables=(stats, ab, abm, pwr, isl))
    eng = LearnedEngine(**engine_kw)
    adapter = NEffValueAdapter(net)
    eng.vnet = adapter
    eng.priors_override = neff_priors(net, adapter.tab, vocab, eng.enc_version)
    return eng


def gate(args):
    import counterfactual_referee as CR
    import coach_gate as CG
    import mark_gate as MG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    if args.base_net:
        from n1_gate import n1_engine
        base = n1_engine(args.base_net)
    else:
        base = LearnedEngine()
    chall = neff_engine(args.net)
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
        print(f"  {tag}@{i:<4} base={b:.2f} neff={c:.2f}", flush=True)
    ok_nr, ok_imp, regs = CG.judge(rows)
    print(f"改善: {'OK' if ok_imp else 'NG'}（neff計 {sum(c for *_, c in rows):.1f}"
          f" vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {regs}")
    print("N_EFF_GATE_RESULT", json.dumps({"verdict": "PASS" if (ok_nr and ok_imp) else "FAIL"}))
    return 0


def smoke(args):
    import random
    from opcg_game import OPCGGame
    from cpu_selfplay import _load_db
    from deck_synth import synth_deck
    from opcg_sim.src.core.gamestate import GameManager, Player
    db = _load_db()
    eng = neff_engine(args.net)
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
            print("N_EFF_SMOKE_RESULT", json.dumps({"verdict": "FAIL", "step": steps}))
            return 1
        m = m2
        steps += 1
    ok = m.winner is not None
    print("N_EFF_SMOKE_RESULT", json.dumps(
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
                   help="基準側を N 系ネットに（前世代比）。未指定=出荷 gen15")
    s = sub.add_parser("smoke")
    s.add_argument("--net", required=True)
    s.add_argument("--seed", type=int, default=434343)
    s.add_argument("--sims", type=int, default=32)
    args = ap.parse_args()
    return gate(args) if args.cmd == "gate" else smoke(args)


if __name__ == "__main__":
    _sys.exit(main())
