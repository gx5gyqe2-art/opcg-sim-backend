"""V7 リーサル族のターン出口教師（系統3・π非依存・2026-08-21）。

段3裁定 #15（監査条件 505000: 台本リーサルを差し置いてドン全付与→部分攻撃）を一般化した
支配ペア。決定点で **防御込み台本リーサル**（`lethal.lethal_distance(..., defend=True) == 0`＝
相手が実カウンターで最大防御しても今ターンに勝ち切る）が存在するとき、

  good … リーサル台本を最後まで実行した出口（勝利済み盤面・z=+0.5）
  bad  … リーサルを取らない出口（z=−0.5）×2種:
         pass = 何もせず閉幕 ／ dump = 浮ドンをリーダーへ全付与して閉幕（#15 の実挙動）

順序は「確実な勝ち ≥ 勝っていない出口」の論理支配なのでロールアウト審判は不要（π 非依存）。
防御込み検証は生成側が相手手札を読むが、これはラベルの保守化（勝ちの確認）であり、
符号化特徴には相手非公開情報を一切入れない（公平性契約は encoder 側で維持）。

B2a/B2b/B2c 切り分け（2026-08-21）で系統2（自己対戦結果ラベル）はアンカーと系統的に
衝突すると確定したため、リーサル族も系統3方式（ルールラベル）で教える。

出力: `planlet_*.npz`（plandom と同スキーマ）。学習は
  exit_head_finetune.py --head turn --globs "plandom_*.npz,planlet_*.npz" ...

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/plan_lethal_gen.py \
    --games 24 --seed-base 730000 --workers 6 --out /tmp/planlet
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import lethal as LT
from opcg_sim.src.learned import plan as PL

MAX_MAIN_PER_TURN = 2     # リーサル判定はメイン冒頭が本体。ターン後半の重複採取を抑える
K_DUMP = 4                # dump 出口で振る浮ドンの上限


class _Done(BaseException):
    pass


class _MainCap:
    """自席メイン判断（pending が MAIN_ACTION）の直前 manager を、ターンごとに上限つきで複製。"""

    def __init__(self, limit_decisions):
        self.limit = limit_decisions
        self.n = 0
        self.frames = []      # (decision_no, turn, seat, manager)
        self._per_turn = {}
        self._keys = cpu_ai._pending_keys()

    def on_decision_point(self, ctx):
        m = ctx.manager
        name = getattr(ctx.actor, "name", None)
        _kp, k_action = self._keys
        if (ctx.pending or {}).get(k_action) != "MAIN_ACTION":
            return
        if getattr(getattr(m, "turn_player", None), "name", None) != name:
            return
        turn = int(getattr(m, "turn_count", 0) or 0)
        key = (name, turn)
        if self._per_turn.get(key, 0) >= MAX_MAIN_PER_TURN:
            return
        self._per_turn[key] = self._per_turn.get(key, 0) + 1
        self.frames.append((self.n + 1, turn, name, m.clone()))

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.limit:
            raise _Done()


def lethal_exit(gs, frame, name, max_steps=LT.MAX_STEPS):
    """防御込みリーサル台本を勝利まで実行した manager を返す（勝てなければ None）。

    `lethal._lethal_distance_inner` と同じ台本・同じ in-place 適用。defend=True＝
    非手番側（相手）が実カウンターで防御する（docstring 参照）。"""
    m = frame.clone()
    m.action_events = []
    steps = 0
    while steps < max_steps:
        if m.winner is not None:
            return m if m.winner == name else None
        cur = gs.current_player(m)
        if cur is None:
            return None
        mv = LT._script_move(gs, m, name, defend=True)
        if mv is None:
            return None
        d = LT._desc(m, mv)
        if cur == name and d.get("action_type") == "TURN_END":
            return None          # ターンを跨ぐ＝「今ターンの詰み」ではない
        try:
            cpu_ai._apply_move_inplace(m, cur, mv, stop_at_select=True)
        except Exception:
            return None
        steps += 1
    return None


def non_lethal_exits(frame, name, vf, pf, game):
    """リーサルを取らない出口 [(variant, manager)]。pass=即閉幕 / dump=リーダー全付与→閉幕。"""
    out = []
    try:
        out.append(("pass", PL.execute_plan(game, frame.clone(), name, [],
                                            vf, pf, battle_value_fn=None)))
    except BaseException:
        pass
    p = frame.p1 if getattr(frame.p1, "name", None) == name else frame.p2
    spare = len(getattr(p, "don_active", []) or [])
    lead = getattr(p, "leader", None)
    if spare > 0 and lead is not None:
        try:
            sig = PL.scripted_plan(game, frame.clone(), name,
                                   [("ATTACH", lead.uuid)] * min(spare, K_DUMP), vf, pf)
            if sig:
                out.append(("dump", PL.execute_plan(game, frame.clone(), name, list(sig),
                                                    vf, pf, battle_value_fn=None)))
        except BaseException:
            pass
    return out


def cut_at_group(buf, size):
    """size 行たまった buf を group 境界で切る（(チャンク, 残り)・pure）。

    組は可変行数（win 1 + 負例1〜2）なのでシャード境界で組が割れると
    build_rank_pairs の対が欠ける。境界の group が続く限り伸ばして切る。"""
    cut = size
    gcut = buf[cut - 1][2]
    while cut < len(buf) and buf[cut][2] == gcut:
        cut += 1
    return buf[:cut], buf[cut:]


_G = {}


def _init(sims, enc_version):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    eng = CL.LearnedEngine(sims=sims,
                           # 教師生成は**原始手の全空間**が対象（レフェリー/計器と同じ原則・
                           # 2026-08-25 既定 ON 化に伴う明示化）: 箱化された候補では
                           # V1/V4/リーサル対の原始手探索が成立しない（実測 0 対）。
                           macro_moves=False, defense_box=False, box_dialog=False)
    _G["eng"] = eng
    _G["vf"] = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    _G["pf"] = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    _G["enc_version"] = enc_version or eng.enc_version


def _run_game(job):
    seed, sims = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.learned import encoder as E
    db, eng = _G["db"], _G["eng"]
    la, lb = _leader_pair(db, seed, "random")
    cap = _MainCap(limit_decisions=200)
    seat = make_seat(kind="learned", want_trace=False, sims=sims, engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=200)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": []}

    gs = LT._game()
    rows = []      # (enc, z, group_local, variant)
    gi = 0
    hits = 0
    for dec, turn, name, frame in cap.frames:
        try:
            if LT.lethal_distance(gs, frame, name, defend=True) != 0:
                continue
            hits += 1
            win = lethal_exit(gs, frame, name)
            if win is None:
                continue      # 距離0なのに勝てない＝台本の揺れ。対を立てない
            ew = E.encode(win, name, eng.vocab, version=_G["enc_version"])
            alts = non_lethal_exits(frame, name, _G["vf"], _G["pf"], eng.game)
        except BaseException:
            continue
        made = False
        for variant, alt in alts:
            try:
                ea = E.encode(alt, name, eng.vocab, version=_G["enc_version"])
            except BaseException:
                continue
            if np.array_equal(ew["scalars"], ea["scalars"]) and \
                    np.array_equal(ew["field"], ea["field"]):
                continue      # 出口が同一＝順位を教えられない（alt でも勝ってしまった等）
            if getattr(alt, "winner", None) == name:
                continue      # 取らなくても勝つ盤面＝支配が立たない
            rows.append((ea, -0.5, gi, variant))
            made = True
        if made:
            rows.append((ew, +0.5, gi, "win"))
            gi += 1
    return {"seed": seed, "error": None, "rows": rows,
            "frames": len(cap.frames), "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=730000)
    ap.add_argument("--sims", type=int, default=160, help="局面採取の自己対戦（分布=本番仕様）")
    ap.add_argument("--enc-version", type=int, default=0, help="0=エンジンの符号化世代に従う")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.seed_base + i, args.sims) for i in range(args.games)]
    buf, stats = [], {"groups": 0, "pass": 0, "dump": 0, "hits": 0, "errors": 0}
    shard = [0]

    def _flush(chunk):
        if not chunk:
            return
        arrays = {
            "scalars": np.stack([r[0]["scalars"] for r in chunk]).astype(np.float32),
            "field": np.stack([r[0]["field"] for r in chunk]).astype(np.float32),
            "card_idx": np.stack([r[0]["card_idx"] for r in chunk]).astype(np.int32),
            "value": np.array([r[1] for r in chunk], dtype=np.float32),
            "group": np.array([r[2] for r in chunk], dtype=np.int64),
        }
        np.savez_compressed(os.path.join(args.out, f"planlet_{shard[0]:05d}.npz"), **arrays)
        shard[0] += 1

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.sims, args.enc_version)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                stats["errors"] += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            gbase = res["seed"] * 1000
            groups = set()
            for enc, z, g, variant in res["rows"]:
                buf.append((enc, z, gbase + g))
                groups.add(g)
                if variant in stats:
                    stats[variant] += 1
            stats["groups"] += len(groups)
            stats["hits"] += res.get("hits", 0)
            print(f"  seed {res['seed']}: 判断点{res.get('frames', 0)}・リーサル点{res.get('hits', 0)}"
                  f"・組{len(groups)}", flush=True)
            # 組（可変行数）が割れないよう group 境界で切る
            while len(buf) >= args.shard_size:
                chunk, buf = cut_at_group(buf, args.shard_size)
                _flush(chunk)
    _flush(buf)
    print(f"PLAN_LETHAL_DONE groups={stats['groups']} pass={stats['pass']} dump={stats['dump']} "
          f"lethal_frames={stats['hits']} errors={stats['errors']} shards={shard[0]} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
