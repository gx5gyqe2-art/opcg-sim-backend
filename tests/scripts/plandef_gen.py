"""D族＝防御の支配ペア教師（π非依存・マクロ手化 P4-b・2026-08-24）。

防御監査（defense_audit_probe・2026-08-24）で「現行 value は防御選択の差をほぼ感じない
（守る価値の中央値 +0.000）のに過剰防御が出る」ことを確認した。防御には厳密支配が作れる:

  D1 守れないのに払う … 手札の印字カウンター総量 < 必要値なら、何を払っても止まらない
     ＝素通し（手札温存）が任意の支払いを支配する（裁定 m2@58「守れないから捨てる」の一般化）
  D2 必要以上に払う   … 最小で守り切れる札組 S に余分な1枚を足しても戦闘結果は不変
     ＝S が S+1枚 を支配する（余分な札の分だけ純損）

各防御窓（被攻撃時の SELECT_COUNTER/SELECT_BLOCKER 入口）で成立する族の対を作り、
**戦闘を解決し切った出口盤面**を符号化して ±0.5 の順位ラベルで出力する（`plandef_*.npz`・
defcf/plandom 互換スキーマ）。学習は battle 出口ヘッドの再訓練:
  exit_head_finetune.py --head battle --replace-head --base gen15 --enc-version 12 \
    --dirs <既存defcf郡>,<plandef> --globs "defcf_*.npz,plandef_*.npz"
（既存 defcf を混ぜて元の較正を保持する＝v43 レシピの上に D族を足す）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/plandef_gen.py \
    --games 24 --seed-base 790000 --workers 4 --out /tmp/plandef
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import random

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai

DEF_WINDOWS = ("SELECT_COUNTER", "SELECT_BLOCKER")


def counter_cards(hand):
    """印字カウンター値のあるカード [(値, uuid)]（降順・pure）。"""
    vals = [(int(getattr(c, "current_counter", 0) or 0), c.uuid)
            for c in (hand or [])]
    return sorted([(v, u) for v, u in vals if v > 0], key=lambda x: -x[0])


def battle_need(manager, name):
    """この戦闘を止めるのに必要なカウンター値（止まっている/戦闘外は 0・pure）。"""
    bat = getattr(manager, "active_battle", None)
    if bat is None:
        return 0
    try:
        atk = int(bat["attacker"].get_power(True))
        tgt = int(bat["target"].get_power(False)) + int(bat.get("counter_buff", 0) or 0)
    except Exception:
        return 0
    return atk - tgt + 1000 if atk >= tgt else 0


def d_family(manager, name):
    """この窓で成立する族 [(tag, good_uuids, bad_uuids)]（uuid=切るカウンター札の列・pure）。

    D1: 総量不足 → good=[]（素通し）/ bad=[最大の1枚]
    D2: 最小で守れて余りがある → good=最小札組 / bad=最小札組+余りの最大1枚"""
    me = manager.p1 if manager.p1.name == name else manager.p2
    need = battle_need(manager, name)
    if need <= 0:
        return []
    cs = counter_cards(getattr(me, "hand", []) or [])
    total = sum(v for v, _ in cs)
    if not cs:
        return []
    if total < need:
        return [("D1", [], [cs[0][1]])]
    acc, picks = 0, []
    for v, u in cs:
        if acc >= need:
            break
        acc += v
        picks.append(u)
    rest = [u for _, u in cs if u not in picks]
    if rest:
        return [("D2", picks, picks + [rest[0]])]
    return []


_G = {}


def _init(sims, enc_version):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    eng = CL.LearnedEngine(sims=sims)
    _G["eng"] = eng
    _G["enc_version"] = enc_version or eng.enc_version


def battle_exit(manager, name, pay_uuids):
    """カウンター列を払い、残りの戦闘を**エンジンと同じ解決規約**で閉じた盤面を返す（失敗は None）。

    2026-08-25 規約一致修正（N1 ゲートの発見）: 旧実装は「残る自窓を素通しで流す」簡略規約で、
    serve（resolve_battle_inplace の出口価値最良継続・box_depth=config）と**教師の出口分布が
    ずれていた**。defense_cf_gen と同じく解決規約を探索と共有する（1定義）。"""
    m = manager.clone()
    m.action_events = []
    random.seed(4242)
    np.random.seed(4242)
    try:
        for u in pay_uuids:
            cpu_ai._apply_move_inplace(
                m, name, {"kind": "battle", "action_type": "SELECT_COUNTER",
                          "card_uuid": u}, stop_at_select=True)
        from opcg_sim.src.learned.config import BOX_RESOLVE_DEPTH
        from opcg_sim.src.learned.mcts import resolve_battle_inplace
        from opcg_sim.src.core import cpu_learned as CL
        eng = _G["eng"]
        pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
        resolve_battle_inplace(eng.game, m, pf, value_fn=eng._battle_value_fn(),
                               box_depth=BOX_RESOLVE_DEPTH)
        return m if getattr(m, "active_battle", None) is None else None
    except Exception:
        return None


class _Done(BaseException):
    pass


def _run_game(job):
    seed, sims = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.learned import encoder as E
    db, eng = _G["db"], _G["eng"]
    la, lb = _leader_pair(db, seed, "random")
    windows = []

    class Cap:
        def __init__(self):
            self.n = 0
            self._keys = cpu_ai._pending_keys()
            self._seen = set()

        def on_decision_point(self, ctx):
            _kp, ka = self._keys
            if (ctx.pending or {}).get(ka) not in DEF_WINDOWS:
                return
            m = ctx.manager
            bat = getattr(m, "active_battle", None)
            if bat is None or id(bat) in self._seen:
                return
            self._seen.add(id(bat))
            windows.append((getattr(ctx.actor, "name", None), m.clone()))

        def on_decision(self, ctx, move):
            self.n += 1
            if self.n > 300:
                raise _Done()

    seat = make_seat(kind="learned", want_trace=False, sims=sims, engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(Cap(),), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=300)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": []}

    rows = []
    gi = 0
    tags = {"D1": 0, "D2": 0}
    for name, m in windows:
        for tag, good_u, bad_u in d_family(m, name):
            mg = battle_exit(m, name, good_u)
            mb = battle_exit(m, name, bad_u)
            if mg is None or mb is None:
                continue
            try:
                eg = E.encode(mg, name, _G["eng"].vocab, version=_G["enc_version"])
                eb = E.encode(mb, name, _G["eng"].vocab, version=_G["enc_version"])
            except Exception:
                continue
            if np.array_equal(eg["scalars"], eb["scalars"]) and \
                    np.array_equal(eg["field"], eb["field"]):
                continue
            rows.append((eg, +0.5, gi))
            rows.append((eb, -0.5, gi))
            tags[tag] += 1
            gi += 1
    return {"seed": seed, "error": None, "rows": rows,
            "windows": len(windows), "tags": tags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=790000)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--enc-version", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.seed_base + i, args.sims) for i in range(args.games)]
    buf, stats = [], {"pairs": 0, "D1": 0, "D2": 0, "errors": 0}
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
        np.savez_compressed(os.path.join(args.out, f"plandef_{shard[0]:05d}.npz"), **arrays)
        shard[0] += 1

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.sims, args.enc_version)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                stats["errors"] += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            gbase = res["seed"] * 1000
            for enc, z, g in res["rows"]:
                buf.append((enc, z, gbase + g))
            stats["pairs"] += len(res["rows"]) // 2
            for k in ("D1", "D2"):
                stats[k] += res["tags"].get(k, 0)
            print(f"  seed {res['seed']}: 窓{res.get('windows', 0)}"
                  f"・対 {len(res['rows']) // 2} {res['tags']}", flush=True)
            while len(buf) >= args.shard_size:
                _flush(buf[: args.shard_size])
                buf = buf[args.shard_size:]
    _flush(buf)
    print(f"PLAN_DEF_DONE pairs={stats['pairs']} D1={stats['D1']} D2={stats['D2']} "
          f"errors={stats['errors']} shards={shard[0]} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
