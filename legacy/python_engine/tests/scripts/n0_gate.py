"""N0 スパイクの計器（`probe`／`gate`）——**Python エンジンを要る**ので legacy 側に置く。

訓練とカード表（`build_card_table`／`N0Net`）は `opcg_sim/learned/train/n0_spike.py` に残っている
（エンジン非依存）。ここにあるのは旧 coach 13 点ゲートと枝別出口の採点で、
`counterfactual_referee`／`coach_gate`／`mark_gate`／`replay_reeval`／`cpu_selfplay`（いずれも
Python エンジンの計器）と `LearnedEngine` を使う。

2026-09-07・第 2 段 `rs-archive-cutover` で分離（計画 §16.3-6: `opcg_sim/` から `legacy/` を
import する箇所を 0 にする）。回し方は tag `py-engine-final` を checkout するのが正本。
"""
import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.train.n0_spike import (  # noqa: E402
    CTX, D_IN, MAX_CI, N0Net, build_card_table, card_channel)


def probe(args):
    """m1@14 の枝別出口を N0 で採点（枝順位の N0 版・素通しが1位であるべき）。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from legacy.python_engine.core import cpu_ai
    from legacy.python_engine.core.cpu_learned import LearnedEngine
    from opcg_sim.learned import encoder as ENC
    from legacy.python_engine.learned.mcts import resolve_battle_inplace
    tab, vocab = build_card_table()
    net = N0Net.load(args.net)
    CR.ARGS = argparse.Namespace(true_board=True); CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m1"]); rec = raw.get("replay", raw)
    CR.GAMES["m1"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m1", 14)
    name = who if isinstance(who, str) else who.name
    eng = LearnedEngine(macro_moves=False, defense_box=False)   # 原始手空間で枝を出す
    legal = eng.game.legal_actions(m)
    print("=== m1@14 N0(battle出口文脈) ===")
    scored = []
    for mv in legal:
        c = m.clone(); c.action_events = []
        try:
            cpu_ai._apply_move_inplace(c, name, mv, stop_at_select=True)
            resolve_battle_inplace(eng.game, c)
            enc = ENC.encode(c, name, eng.vocab, version=12)
            sc = np.asarray(enc["scalars"], np.float32)[None]
            cix = np.zeros((1, MAX_CI), np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            cix[0, :len(src)] = src
            ctx = np.zeros((1, CTX), np.float32); ctx[0, 1] = 1.0
            v = float(net.forward(sc, card_channel(cix, tab), ctx)[0])
        except Exception as e:
            v = None
        d = cpu_ai._describe_move(m, mv) or {}
        scored.append((v, f"{d.get('action_type')}:{d.get('card')}"))
    for v, s in sorted(scored, key=lambda t: -(t[0] if t[0] is not None else -9)):
        print(f"  {v:+.4f}  {s}" if v is not None else f"  None  {s}")


class N0ValueAdapter:
    """N0Net を ValueNet のダックタイプ（predict/predict_exit/has_exit_head）に適合させ、
    LearnedEngine の葉評価・戦闘出口・ターン出口の物差しを**単一胴体の文脈切替**で供給する
    （N1＝エンジン統合・2026-08-25）。aux は持たない（本体の aux 粘り項も純正AZ化で削除済み）。"""

    def __init__(self, net, tab):
        self.net = net
        self.tab = tab

    def _fwd(self, batch, ctx_id):
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])
        if ci.shape[1] < MAX_CI:
            ci = np.concatenate([ci, np.zeros((len(ci), MAX_CI - ci.shape[1]), ci.dtype)], 1)
        ctx = np.zeros((len(sc), CTX), np.float32)
        ctx[:, ctx_id] = 1.0
        return self.net.forward(sc, card_channel(ci[:, :MAX_CI], self.tab), ctx)

    def predict(self, batch):
        return self._fwd(batch, 0)

    def predict_with_aux(self, batch):
        v = self._fwd(batch, 0)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return kind in ("battle", "turn")

    def predict_exit(self, batch, kind):
        return self._fwd(batch, 1 if kind == "battle" else 2)


def n0_engine(net_path):
    """N0 を積んだ LearnedEngine（符号化 v12 は共通・policy は現行 gen15 のまま）。"""
    from legacy.python_engine.core.cpu_learned import LearnedEngine
    tab, _vocab = build_card_table()
    eng = LearnedEngine()
    eng.vnet = N0ValueAdapter(N0Net.load(net_path), tab)
    return eng


def gate(args):
    """coach 13点（既定 vs N0エンジン）＝N1 の行動ゲート。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import mark_gate as MG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from legacy.python_engine.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    base = LearnedEngine()
    chall = n0_engine(args.net)
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
        print(f"  {tag}@{i:<4} base={b:.2f} n0={c:.2f}", flush=True)
    ok_nr, ok_imp, regs = CG.judge(rows)
    print(f"改善: {'OK' if ok_imp else 'NG'}（n0計 {sum(c for *_, c in rows):.1f}"
          f" vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {regs}")
    print("N0_GATE_RESULT", json.dumps({"verdict": "PASS" if (ok_nr and ok_imp) else "FAIL"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--net", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--net", required=True)
    args = ap.parse_args()
    return probe(args) if args.cmd == "probe" else gate(args)


if __name__ == "__main__":
    _sys.exit(main())
