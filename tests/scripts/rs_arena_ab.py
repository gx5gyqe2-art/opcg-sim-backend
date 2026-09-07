"""切替の受け入れ測定: **Rust decide 対 Python decide**（計画 `docs/rust_engine_plan.md` §3 P5）。

第 2 段 `rs-archive-cutover` は「CPU の思考を Python から Rust へ差し替える」変更なので、
受け入れは**同じ盤面・同じネットで打たせて互角か**で見る。ルールの一致は golden 2 本
（監査 3,386 件・再生 200 局）と決定オラクル（9,849/9,855）が既に担保しているので、ここで
測るのは**強さ**だけ。

**構成**（両席とも出荷既定 a1・盤面は Rust）:
  - Rust 席 … `opcg_engine.Game.decide`（切替後の本番経路）
  - Python 席 … `rs_bridge.manager_from_hidden` で `GameManager` を組んで
    `cpu_learned.LearnedEngine.decide` に読ませる。

**2 つのモード**（`--mode`）:

  `fair`（既定・**受け入れの本番**）… 「同じアルゴリズムの 2 実装が同じ強さか」を測る。
    橋渡しに由来する**構造的な不利**を 3 つ取り除く:
      1. 期間付き効果の一覧（`ContinuousEffectManager.effects`）を `hidden` から戻す
         （`rs_bridge` は復元しない＝「このターン中 −5000」がターンを跨いで残る・§8.17 の実バグ）
      2. 箱コミットの残り手順を `(turn, seat)` で持ち越す（Python は `id(manager)` を鍵にするので、
         決定のたびに manager を組み直す経路では**一度も当たらない**）
      3. ターン内 sticky 世界線を同上の鍵で持ち越す（同じ理由で失われていた）
    残る差は「対話の途中では `hidden` に継続が無いので Python 席の読みが浅くなる」1 点だけ。

  `legacy` … **切替前の serve の経路そのまま**（上の 3 つを取り除かない）。切替が serve の
    強さに与えた影響を測る＝「互角」ではなく「どれだけ変わったか」を見るためのモード。

Python 席が出した手が Rust の合法手に無い局面は **void** として母数から外し、件数を必ず載せる。

規約はアリーナと同じ: 同 seed で席を入れ替えた 2 局＝1 ペア、勝率はペア水準（0/0.5/1）、
95% CI は `opcg_sim.loop.arena.pair_level_ci`。

実行例（主条件＝ランダム対面×生成デッキ／副条件＝固定ミラー）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_arena_ab.py \\
    --pairs 200 --leaders random --decks synth --seed-base 6000000 --out /tmp/ab_main.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import multiprocessing as mp                                           # noqa: E402
import random                                                          # noqa: E402
import time                                                            # noqa: E402

import os as _os, sys as _sys                                          # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                      # noqa: E402,F401

from opcg_sim.loop import arena as A                                   # noqa: E402
from opcg_sim.loop import decks as D                                   # noqa: E402
from opcg_sim.loop import driver as DR                                 # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402

_G = {}


def _fair_engine():
    """`(turn, seat)` を鍵に持ち越す `LearnedEngine`（`fair` モード用）。

    素の `LearnedEngine` は箱コミット（`_commits`）とターン内 sticky 世界線（`_world_seeds`）を
    `id(manager)` で鍵付けし、`weakref` で同一性を確かめる。決定のたびに `hidden` から manager を
    組み直す経路では**一度も当たらない**＝機構が丸ごと死ぬ。実装の優劣ではなく橋渡しの都合なので、
    鍵を差し替えて両席に同じ機構を持たせる（判定規約・探索の中身は一切変えない）。
    """
    import weakref
    from legacy.python_engine.core.cpu_learned import LearnedEngine

    class _Keyed(LearnedEngine):
        def _commit_key(self, manager, name):
            return (int(getattr(manager, "turn_count", 0) or 0), name)

        def _store_commit(self, manager, name, steps):
            # 親は `weakref.ref(manager)` を添えて同一性を確かめるが、manager は決定のたびに
            # 別物なので鍵 (turn, seat) だけで持つ（手順そのものは親と同じ list）。
            if steps:
                self._commits[self._commit_key(manager, name)] = list(steps)

        def _commit_step(self, manager, player, name, trace):
            # 親の実装は `(weakref, steps)` の組を想定するので、その場で組んで渡す
            # （`weakref.ref(manager)` は今の manager＝検証を必ず通る）。
            key = self._commit_key(manager, name)
            steps = self._commits.get(key)
            if steps is None:
                return None
            self._commits[key] = (weakref.ref(manager), steps)
            try:
                return super()._commit_step(manager, player, name, trace)
            finally:
                hit = self._commits.get(key)
                # 親は消化後に list を書き換える／契約違反なら key ごと消す。
                self._commits[key] = hit[1] if isinstance(hit, tuple) else hit
                if not self._commits.get(key):
                    self._commits.pop(key, None)

        def _world_rng(self, manager, name, rng):
            import numpy as _np
            key = (int(getattr(manager, "turn_count", 0) or 0), name)
            seed = self._world_seeds.get(key)
            if seed is None:
                seed = int(rng.integers(0, 2 ** 63 - 1))
                self._world_seeds[key] = seed
            return _np.random.default_rng(seed)

    return _Keyed()


def _init(sims, mode="fair"):
    E.engine()
    _G["sims"] = sims
    _G["mode"] = mode
    _G["spec"] = E.SeatSpec(None, sims=sims)
    _G["db"] = D.load_db()
    if mode == "fair":
        _G["py"] = _fair_engine()
    else:
        from legacy.python_engine.core.cpu_learned import LearnedEngine
        _G["py"] = LearnedEngine()


def _py_move(game, name):
    """Python 席の 1 手（`hidden` から `GameManager` を組んで読ませる）。"""
    from legacy.python_engine.core import rs_bridge
    hidden = json.loads(game.hidden_json())
    pending = json.loads(game.pending_json())
    manager = rs_bridge.manager_from_hidden(_G["db"], hidden, suppress_pending=False,
                                            names={"p1": "p1", "p2": "p2"})
    rs_bridge.attach_shallow_interaction(manager, pending)
    if _G.get("mode") == "fair":
        # 期間付き効果の**一覧**を戻す（`rs_bridge` はカード側の値しか戻さない＝失効させる側が
        # 効果を知らず「このターン中 −5000」が残る・§8.17 で実測した記録の穴）。
        from legacy.python_engine.tests.harness import rs_record
        rs_record._restore_continuous(manager, hidden)
    player = manager.p1 if manager.p1.name == name else manager.p2
    return _G["py"].decide(manager, player, sims=_G["sims"])


def _play(seed, py_seat, la, lb, decks, max_steps=DR.DEFAULT_MAX_STEPS):
    """1 局。`py_seat` が Python decide、もう一方が Rust decide。"""
    db = _G["db"]
    p1, p2 = D.build_pair(db, la, lb, seed, decks)
    game = DR.new_game(seed, p1, p2)
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    # 箱コミット／sticky 世界線は (turn, seat) 鍵なので、**局をまたいで持ち越さない**。
    _G["py"]._commits.clear()
    _G["py"]._world_seeds.clear()
    steps = 0
    # Python 席の探索は numpy の global 乱数を引く＝seed を張って決定論にする。
    random.seed(seed)
    while game.winner() is None and steps < max_steps:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        if name == py_seat:
            move = _py_move(game, name)
        else:
            turn = int(game.turn_count or 0)
            c = carry[name].get(turn, name)
            out = json.loads(game.decide(name, _G["spec"].decide_opts(seed, turn, name, c)))
            carry[name].put(out)
            move = out.get("move")
        if move is None:
            raise DR.GameAborted(f"seed={seed} step={steps} 手が出ない（{name}）")
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"],
                                   json.dumps(move.get("payload") or {}, ensure_ascii=False))
        steps += 1
    if game.winner() is None:
        raise DR.GameAborted(f"seed={seed} 決着せず（{steps} 手）")
    return game.winner(), steps


def play_pair(spec):
    """1 ペア＝同 seed・席入替の 2 局。**Rust 席の**勝ち数 0..2 を返す。"""
    seed, leaders_mode, decks = spec
    db = _G["db"]
    la, lb = D.leader_pair(db, seed, leaders_mode)
    try:
        # game a: Rust=p1（la）／Python=p2  ・ game b: Python=p1（lb）／Rust=p2（la）
        wa, sa = _play(seed, "p2", la, lb, decks)
        wb, sb = _play(seed, "p1", lb, la, decks)
    except Exception as e:                                    # noqa: BLE001
        return {"seed": seed, "score": None, "leaders": [la, lb],
                "void": f"{type(e).__name__}: {str(e)[:140]}"}
    return {"seed": seed, "score": (1.0 if wa == "p1" else 0.0) + (1.0 if wb == "p2" else 0.0),
            "leaders": [la, lb], "steps": [sa, sb]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=int, default=200, help="ペア数（局数は×2）")
    ap.add_argument("--seed-base", type=int, default=6000000)
    ap.add_argument("--max-pairs", type=int, default=0, help="この実行で回す上限（0=全部）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=E.SERVE_SIMS)
    ap.add_argument("--leaders", default="random", choices=("fixed", "random", "real", "purple"))
    ap.add_argument("--decks", default="synth", choices=("singleton", "synth", "synth_dig"))
    ap.add_argument("--mode", default="fair", choices=("fair", "legacy"),
                    help="fair=橋渡しの構造的な不利を外す（受け入れの本番）／"
                         "legacy=切替前の serve の経路そのまま")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記台帳・再開の正）")
    args = ap.parse_args(argv)

    planned = list(range(args.seed_base, args.seed_base + args.pairs))
    done = A.load_ledger(args.out)
    todo = A.remaining_seeds(planned, done)
    print(f"消化済み {len(done)}/{args.pairs} ペア・残り {len(todo)}", flush=True)
    if todo:
        batch = todo[: args.max_pairs] if args.max_pairs else todo
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers, initializer=_init,
                                          initargs=(args.sims, args.mode)) as pool:
            with open(args.out, "a") as f:
                specs = [(s, args.leaders, args.decks) for s in batch]
                for i, row in enumerate(pool.imap_unordered(play_pair, specs), 1):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    if i % 10 == 0:
                        print(f"  {i}/{len(batch)} ペア {time.time()-t0:.0f}s", flush=True)
        done = A.load_ledger(args.out)
        print(f"今回 {len(batch)} ペア（{time.time() - t0:.0f}s）・累計 {len(done)}/{args.pairs}",
              flush=True)
    res = A.final_result(planned, done, frac=0.55)
    if res is None:
        # 未消化がある間も途中経過は出す（判定は出さない）。
        valid = [s for s in planned if done.get(s) is not None]
        if valid:
            ci = A.pair_level_ci([done[s] / 2.0 for s in valid])
            print(f"（途中）{len(valid)} ペア wr={ci['win_rate']:.4f} "
                  f"CI[{ci['lo']:.4f},{ci['hi']:.4f}]", flush=True)
        return 0
    # 「互角」の判定（§3 P5 の受け入れ）: void ≤2% かつ 勝率 0.5±0.05。
    void_rate = res["void"] / max(len(planned), 1)
    res.update({"label": args.label, "mode": args.mode, "leaders": args.leaders,
                "decks": args.decks,
                "sims": args.sims, "void_rate": round(void_rate, 4),
                "even": bool(void_rate <= 0.02 and abs(res["wr"] - 0.5) <= 0.05)})
    res.pop("promoted", None)
    print("RS_AB_FINAL " + json.dumps(res, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
