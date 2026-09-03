"""マクロ手化の等価性検証（P1/P2 の正しさ監査・2026-08-24）。

箱（DON_BOX の配分形/アタック形）が**意味を保存しているか**を実盤面で全数検査する:
  A) 適用等価性 … 箱を1手で適用した終状態 ＝ 等価な原始手列を逐次適用した終状態
     （乱数は両経路で同一 seed に固定＝効果のシャッフル等で偽陽性を出さない）
  B) 被覆性     … マクロONの候補から到達できる先頭原始手の集合 ⊇ OFFの原始手集合
     （付与先・攻撃(攻撃者,対象)の欠落がない。多枚付与の並びの縮約は設計どおり）
  C) 出力合法性 … 各箱の先頭原始手（don_box_first_primitive）が OFF の合法手に含まれる

盤面は自己対戦（gen15 既定）の全メイン窓からサンプル。終状態の指紋は「勝敗・両者の
ライフ/手札/場/アクティブドン枚数・場の(カードID,レスト,付与数)集合・保留アクション」。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/macro_equiv_probe.py \
    --games 2 --seed-base 770000
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import random

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai


def fingerprint(m):
    """終状態の指紋（pure）。エンコーダ非依存＝符号化のバグと独立に盤面そのものを比べる。"""
    def side(p):
        field = tuple(sorted(
            (getattr(c.master, "card_id", None), bool(getattr(c, "is_rest", False)),
             (c.attached_don if isinstance(getattr(c, "attached_don", None), int)
              else len(getattr(c, "attached_don", None) or [])))
            for c in (getattr(p, "field", None) or [])))
        lead = getattr(p, "leader", None)
        lead_fp = (bool(getattr(lead, "is_rest", False)),
                   (lead.attached_don if isinstance(getattr(lead, "attached_don", None), int)
                    else len(getattr(lead, "attached_don", None) or []))) if lead else None
        return (len(getattr(p, "life", None) or []), len(getattr(p, "hand", None) or []),
                len(getattr(p, "don_active", None) or []), lead_fp, field)
    pa = m.pending_actor_action()
    return (m.winner, side(m.p1), side(m.p2), (pa[0], str(pa[1])[:40]) if pa else None)


def primitive_seq(box):
    """箱と等価な原始手列（pure）。"""
    pl = box.get("payload") or {}
    seq = [{"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": pl.get("uuid")}}
           for _ in range(int(pl.get("don_k", 0) or 0))]
    if pl.get("target_ids"):
        seq.append({"kind": "game", "action_type": "ATTACK",
                    "payload": {"uuid": pl.get("uuid"),
                                "target_ids": list(pl.get("target_ids") or [])}})
    return seq


def _seeded_apply(frame, name, moves):
    """乱数を固定して原始手列を逐次適用（両経路の乱数消費を同一に保つ）。"""
    random.seed(1234); np.random.seed(1234)
    cur = frame
    for mv in moves:
        cur = cpu_ai._apply_clone(cur, name, mv, stop_at_select=True)
        if cur is None:
            return None
    return cur


class _Done(BaseException):
    pass


class _Cap:
    def __init__(self, limit):
        self.limit = limit
        self.n = 0
        self.frames = []
        self._keys = cpu_ai._pending_keys()

    def on_decision_point(self, ctx):
        _kp, ka = self._keys
        if (ctx.pending or {}).get(ka) != "MAIN_ACTION":
            return
        self.frames.append((getattr(ctx.actor, "name", None), ctx.manager.clone()))

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.limit:
            raise _Done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=770000)
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core import cpu_learned as CL
    from opcg_sim.src.learned.adapter import OPCGGame

    db = _load_db()
    eng = CL.LearnedEngine()
    g_on = OPCGGame(macro_moves=True)
    g_off = OPCGGame(macro_moves=False)
    frames = []
    for i in range(args.games):
        seed = args.seed_base + i
        cap = _Cap(limit=160)
        seat = make_seat(kind="learned", want_trace=False, sims=60, engine=eng)
        la, lb = _leader_pair(db, seed, "random")
        try:
            run_game(seed, db, seats={"p1": seat, "p2": seat},
                     deck_builder=synth_deck_builder(la, lb, seed=seed),
                     observers=(cap,), max_steps=1500, legal_moves="skip",
                     invariants="raise", stop_after_decisions=160)
        except _Done:
            pass
        frames += cap.frames
    frames = frames[: args.max_frames]

    n_box = n_equiv_ok = n_equiv_ng = n_legal_ng = 0
    n_cov_ok = n_cov_ng = 0
    examples = []
    for name, m in frames:
        on = g_on.legal_actions(m)
        off = g_off.legal_actions(m)
        off_prims = {(x.get("action_type"), (x.get("payload") or {}).get("uuid"),
                      tuple((x.get("payload") or {}).get("target_ids") or ()))
                     for x in off if x.get("action_type") in ("ATTACH_DON", "ATTACK")}
        # C) 先頭原始手の合法性 と B) 被覆性
        reach = set()
        for b in on:
            if b.get("action_type") != "DON_BOX":
                continue
            n_box += 1
            fp_mv = cpu_ai.don_box_first_primitive(b)
            key = (fp_mv.get("action_type"), (fp_mv.get("payload") or {}).get("uuid"),
                   tuple((fp_mv.get("payload") or {}).get("target_ids") or ()))
            if key not in off_prims:
                n_legal_ng += 1
                if len(examples) < 5:
                    examples.append(("先頭原始手が非合法", key))
            pl = b.get("payload") or {}
            reach.add(("ATTACH_DON", pl.get("uuid"), ())) if int(pl.get("don_k", 0) or 0) > 0 \
                else None
            if pl.get("target_ids"):
                reach.add(("ATTACK", pl.get("uuid"), tuple(pl.get("target_ids"))))
            # A) 適用等価性
            box_end = _seeded_apply(m, name, [b])
            seq_end = _seeded_apply(m, name, primitive_seq(b))
            if box_end is None and seq_end is None:
                n_equiv_ok += 1
            elif box_end is None or seq_end is None or \
                    fingerprint(box_end) != fingerprint(seq_end):
                n_equiv_ng += 1
                if len(examples) < 5:
                    examples.append(("適用不一致", (pl.get("uuid"), pl.get("don_k"),
                                                    tuple(pl.get("target_ids") or ()))))
            else:
                n_equiv_ok += 1
        for key in off_prims:
            if key in reach:
                n_cov_ok += 1
            else:
                n_cov_ng += 1
                if len(examples) < 5:
                    examples.append(("被覆漏れ", key))
    print(f"frames={len(frames)} 箱={n_box}")
    print(f"A 適用等価: ok={n_equiv_ok} ng={n_equiv_ng}")
    print(f"B 被覆:     ok={n_cov_ok} 漏れ={n_cov_ng}")
    print(f"C 出力合法: ng={n_legal_ng}")
    for e in examples:
        print("  例:", e)
    ok = n_equiv_ng == 0 and n_cov_ng == 0 and n_legal_ng == 0
    print(f"MACRO_EQUIV_{'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
