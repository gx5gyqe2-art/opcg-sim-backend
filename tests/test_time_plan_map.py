"""時間 × 方針の地図（`tests/scripts/time_plan_map.py`）の読み方が反転していないことの契約。

基盤健全性（`cpu_infra`）: 計器の集計の健全性だけを見る（ゲームプレイには触れない）。
**この計器は符号を間違えると結論が丸ごと逆になる**（「速いときに殴る」が「遅いときに殴る」になる）
ので、帯の切り方と視点（どちらのライフを引くか）をここで固定する。

守る性質:
  1. 帯の境界（`margin_band`／`clock_band`）が仕様どおり。
  2. `_cells` の share／winrate／best_plan が手計算と一致し、n が閾値未満の方針は best にしない。
  3. `_spread` は n の少ない帯を混ぜない。
  4. **視点**: `collect` の `true_margin` は**その行の席から見た（自分 − 相手）**＝
     自分が有利なら正（手作りの 1 局で固定）。
"""
import json
import os
import sys

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned.train import plan_labels as PL
from opcg_sim.learned.train import time_labels as TL

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import time_plan_map as TPM  # noqa: E402

pytestmark = pytest.mark.cpu_infra

SC_DIM = 127
S_DIM = 22          # 符号化 v14
N_TOK = 22


def test_bands():
    assert TPM.margin_band(-3.4) == "m<=-2"
    assert TPM.margin_band(-2.0) == "m<=-2"
    assert TPM.margin_band(-1.2) == "m-1"
    assert TPM.margin_band(0.4) == "m0"
    assert TPM.margin_band(-0.4) == "m0"
    assert TPM.margin_band(1.0) == "m+1"
    assert TPM.margin_band(2.6) == "m>=+2"
    assert [TPM.clock_band(x) for x in (0.0, 2.4, 2.6, 4.4, 4.6, 12.0)] == \
        ["t<=2", "t<=2", "t3-4", "t3-4", "t5+", "t5+"]


def _rec(plan, z, margin=0.0, clock=3.0):
    return {"played": PL.PLAN_CLASSES.index(plan), "z": z,
            "true_margin": margin, "pred_margin": margin,
            "true_clock": clock, "pred_clock": clock}


def test_cells_share_winrate_and_best():
    recs = ([_rec("face", 1.0, 2.0) for _ in range(30)] +
            [_rec("face", -1.0, 2.0) for _ in range(10)] +      # face 40 本・勝率 0.75
            [_rec("board", 1.0, 2.0) for _ in range(10)] +
            [_rec("board", -1.0, 2.0) for _ in range(30)] +     # board 40 本・勝率 0.25
            [_rec("mixed", 1.0, 2.0) for _ in range(5)])        # n<30 → best の候補外
    cells = TPM._cells(recs, TPM.ATTACK, lambda r: TPM.margin_band(r["true_margin"]),
                       TPM.MARGIN_BANDS)
    assert list(cells) == ["m>=+2"]
    c = cells["m>=+2"]
    assert c["n"] == 85
    assert c["share"]["face"] == pytest.approx(40 / 85)
    assert c["winrate"]["face"] == pytest.approx(0.75)
    assert c["winrate"]["board"] == pytest.approx(0.25)
    assert c["winrate"]["mixed"] == pytest.approx(1.0)      # 5 本すべて勝ち
    assert c["best_plan"] == "face"                          # mixed は n<30 なので選ばない
    assert c["winrate_all"] == pytest.approx(45 / 85)


def test_spread_ignores_thin_bands():
    recs = ([_rec("face", 1.0, 2.0) for _ in range(80)] + [_rec("board", 1.0, 2.0) for _ in range(20)] +
            [_rec("face", 1.0, -2.0) for _ in range(20)] + [_rec("board", 1.0, -2.0) for _ in range(80)] +
            [_rec("face", 1.0, 0.0) for _ in range(3)])       # n<100 の帯は spread に入れない
    cells = TPM._cells(recs, TPM.ATTACK, lambda r: TPM.margin_band(r["true_margin"]),
                       TPM.MARGIN_BANDS)
    sp = TPM._spread(cells, TPM.ATTACK)
    assert sp["face"] == pytest.approx(0.6)                   # 0.8 − 0.2
    assert sp["board"] == pytest.approx(0.6)
    assert sp["mixed"] == pytest.approx(0.0)


class _StubNet:
    """forward を持たない代わりに 0 を返すだけのネット（`--ablate rel` と同じ扱い）。"""
    ablate = {"rel"}
    time = True

    def value_with_time(self, sc, ci, tok, rom, roo):
        n = len(sc)
        return np.zeros(n, np.float32), np.zeros((n, TL.D_TIME), np.float32)


def _sig(at, uuid=None, target=None):
    return json.dumps([at, uuid, [target] if target else [], [], None])


def test_collect_margin_is_from_the_row_seat(tmp_path):
    """p1 が押している 1 局: p1 の行は正の margin・p2 の行は同じ大きさの負。"""
    from opcg_sim.loop import decks as D
    db = D.load_db()
    leader = next(cid for cid in sorted(db.raw_db)
                  if getattr(getattr(db.get_card(cid), "type", None), "name", "") == "LEADER")
    # (who, turn, kind, my_life, opp_life)・p1 の自席は 1/3/5/7・p2 は 2/4/6
    steps = [(0, 1, 0, 5, 5), (1, 2, 0, 5, 4), (0, 3, 0, 5, 3), (1, 4, 0, 3, 5),
             (0, 5, 0, 5, 1), (1, 6, 0, 1, 5), (0, 7, 0, 5, 1)]
    n = len(steps)
    lives = np.array([[s[3], s[4]] for s in steps], np.float32)
    sc = np.zeros((n, SC_DIM), np.float32)
    sc[:, 0] = lives[:, 0]; sc[:, 1] = lives[:, 1]
    shard = {
        "scalars": sc.astype(np.float16),
        "tokens": np.zeros((n, N_TOK, S_DIM), np.float16),
        "card_idx": np.ones((n, 24), np.int16),
        "z": np.array([1.0 if s[0] == 0 else -1.0 for s in steps], np.float32),
        "who": np.array([s[0] for s in steps], np.int8),
        "turn": np.array([s[1] for s in steps], np.int16),
        "kind": np.array([s[2] for s in steps], np.int8),
        "step": np.arange(n, dtype=np.int32),
        "seed": np.full(n, 7, np.int64),                       # holdout（7%7==0）に入る
        # 自席ターンはリーダー攻撃＝face・相手ターンはライフが減れば take
        "sig": np.array([_sig("DON_BOX", "u-l", "u-opp") for _ in steps]),
        "pol_len": np.ones(n, np.int32), "pol_chosen": np.zeros(n, np.int16),
        "pol_sig": np.array([_sig("DON_BOX", "u-l", "u-opp") for _ in steps]),
        "pol_cid": np.array([leader] * n), "pol_tcid": np.array([leader] * n),
        "pol_n": np.ones(n, np.float32), "pol_q": np.zeros(n, np.float32),
        "pol_k": np.zeros(n, np.int16),
        "pol_si": np.zeros(n, np.int16), "pol_ti": np.zeros(n, np.int16),
    }
    d = tmp_path / "n_records"
    d.mkdir()
    np.savez_compressed(d / "n_record_00000.npz", **shard)
    att, dfn, games, _no_true = TPM.collect(_StubNet(), None, [str(d)], holdout_mod=7)
    assert games == 1
    assert att, "自席ターンの行が 1 つも取れていない"
    # p1（押している側）の行は正・p2 の行は負で、絶対値は同じ組み合わせから来る
    p1 = [r for r in att if r["turn"] % 2 == 1]
    p2 = [r for r in att if r["turn"] % 2 == 0]
    assert p1 and all(r["true_margin"] > 0 for r in p1), [r["true_margin"] for r in p1]
    assert p2 and all(r["true_margin"] < 0 for r in p2), [r["true_margin"] for r in p2]
    # 残りターン数は「この先の自席ターンの数」＝p1 のターン 1 では 4
    assert p1[0]["true_clock"] == pytest.approx(4.0)
    b = TPM.block(att, TPM.ATTACK)
    assert b["rows"] == len(att) and b["by_true_margin"]
