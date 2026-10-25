"""世界を同時に回すスレッドの上限（`docs/search_efficiency_plan.md` §7.55・WP `rs-world-parallel`）。

`Game.decide` の opts `world_threads`（既定 0＝世界 1 本に 1 スレッド・今までどおり）は
**1 手の時間とピークのコア数だけ**を変え、打つ手と出力は 1 bit も変えない。ここでは出荷のネット
（実盤面・実デッキ・ネットの forward を複数スレッドから呼ぶ）で、1 局の決定点ごとに
世界 2 本・4 本 × 上限 0・1・2・4 の `decide` の出力 JSON が丸ごと一致することを見る。
束ねの算術の正本は Rust の単体テスト（`search/decide.rs` の `world_threads_do_not_change_the_output`）。

なぜ基盤健全性（`cpu_infra`）か: 探索の内部機構（世界の並べ方）の健全性しか見ておらず、
ゲームプレイの正しさ（効果解決・カード消失・API 契約）には触れない。

Rust の wheel が無い環境では **skip せず fail** する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_rs_world_threads.py -q -s -p no:cacheprovider
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

pytestmark = pytest.mark.cpu_infra  # 基盤健全性（探索の内部機構）

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

#: 並べ方を見るだけなので探索は軽くする（決定点ごとに 2×4 回 decide する）。
SIMS = 16
STEPS = 40


def test_world_threads_keep_decide_output_bit_identical():
    E.engine()
    db = D.load_db()
    seed = 3100001
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    spec = E.SeatSpec(None, sims=SIMS, budget=0, worlds=2, lethal_proof="root")
    checked = [0]

    def obs(game, name, turn, step, out, move):
        if step >= STEPS:
            return
        carry = DR.Carry()  # 持ち越しは比べる全部で同じ（空）にする
        base_opts = json.loads(spec.decide_opts(seed, turn, name, carry.get(turn, name)))
        for worlds in (2, 4):
            outs = {}
            for threads in (0, 1, 2, 4):
                opts = dict(base_opts, worlds=worlds, world_threads=threads)
                got = json.loads(game.decide(name, json.dumps(opts)))
                for k in ("lethal",):        # 根の詰みの証明は秒を載せる（時間だけ違う）
                    if isinstance(got.get(k), dict):
                        got[k] = {kk: vv for kk, vv in got[k].items() if "sec" not in kk}
                outs[threads] = json.dumps(got, sort_keys=True)
            for threads, s in outs.items():
                assert s == outs[0], f"step={step} worlds={worlds} world_threads={threads}"
        checked[0] += 1

    DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=STEPS, observer=obs)
    assert checked[0] >= 20, checked[0]
