"""v10 符号化世代（**リーサル距離Δ3値**・append-only・2026-08-12・v52b）の検証。

v10 は v9（scalars 70）末尾に [d_me/13, d_opp/13, d_opp_def/13]（台本レース実測・
`opcg_sim/src/learned/lethal.py`）の 3 を足す（73）。

なぜ要るか（v51/v52/v52b 実測）:
  - 乖離族（見かけと実質が乖離・ライフ差が逆向きに壊れる盤面）は現行特徴で表現不能
    （representation-bound・v51 転移ゼロ）。リーサル距離は同族で唯一正の説明力
    （乖離58点 r+0.35・一般60点 r+0.52）。
  - d_me_def（自攻撃 vs 相手の実防御）は相手手札のカウンター値を読む＝公平性契約違反のため
    **入れない**（クリーン3成分の検証は v52b 追補・LOO 差 ±0.03 以内）。

固定する性質:
  - 版マップ: scalars_dim(10) = 70+3 = 73（append-only）
  - 配線: v10 末尾 3 値が `lethal_scan` の直接呼び出しと一致（offset ズレの検出）
  - 決定論: 同一盤面の再符号化で全特徴が一致（台本レースに乱数なし）
  - 接頭辞不変: encode(v9) == encode(v10)[:70]
  - 恒等温スタート（v9→v10 拡張で出力不変）
"""
import os

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
import rl_encoder as E
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import warm_start_value
from opcg_sim.src.learned.lethal import lethal_scan, MAX_TURNS
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
OFF = E.SCALARS_V9   # v10 追加ブロックの先頭 offset（70）


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def vocab(db):
    return E.build_vocab(db)


def _mark_state(db, fn, idx):
    raw = RE.load_replay_json(os.path.join(FIX, fn))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, idx)
    return m, (who if isinstance(who, str) else who.name)


def test_version_map_appends_three():
    assert E.scalars_dim(10) == E.scalars_dim(9) + 3 == 73
    assert 10 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_v10_tail_wiring_and_determinism(db, vocab):
    """v10 末尾 3 値の配線（offset ズレ検出）＋台本レースの決定論。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    enc = E.encode(m, name, vocab, version=10)
    cap = float(MAX_TURNS + 1)
    want = [v / cap for v in lethal_scan(m, name)]
    assert np.allclose(enc["scalars"][OFF:OFF + 3], want)
    assert all(0.0 < v <= 1.0 for v in enc["scalars"][OFF:OFF + 3])
    enc2 = E.encode(m, name, vocab, version=10)
    assert np.array_equal(enc["scalars"], enc2["scalars"])


def test_v10_scan_does_not_mutate_board(db, vocab):
    """測定は clone 上で行われ、元盤面を変更しない（v9 符号化が前後で一致）。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    before = E.encode(m, name, vocab, version=9)
    lethal_scan(m, name)
    after = E.encode(m, name, vocab, version=9)
    assert np.array_equal(before["scalars"], after["scalars"])
    assert np.array_equal(before["field"], after["field"])
    assert np.array_equal(before["card_idx"], after["card_idx"])


def test_v10_scan_does_not_consume_global_rng(db, vocab):
    """符号化は観測＝グローバル乱数状態を消費しない（CRN/決定論再生の契約）。

    台本レースはクローン上でも効果解決（シャッフル等）が `random`/`np.random` を
    消費し得る。ガード無しだと同一シード対局の軌道が符号化の有無で変わる
    （bb2 実測: 同一シード 400 局で行数 8652→8553 に乖離・教師再生 35/50 不一致）。
    """
    import random
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    random.seed(12345)
    np.random.seed(6789)
    st_py = random.getstate()
    st_np = np.random.get_state()
    E.encode(m, name, vocab, version=10)
    assert random.getstate() == st_py, "encode(v10) がグローバル random を消費した"
    after = np.random.get_state()
    assert after[0] == st_np[0] and np.array_equal(after[1], st_np[1]) \
        and after[2:] == st_np[2:], "encode(v10) がグローバル np.random を消費した"


def test_prefix_invariant_v9(db, vocab):
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    e9 = E.encode(m, name, vocab, version=9)
    e10 = E.encode(m, name, vocab, version=10)
    assert np.allclose(e9["scalars"], e10["scalars"][:E.SCALARS_V9])


def test_warm_start_v9_to_v10_is_identity(db, vocab):
    """v9 ネットを v10 へ拡張しても同一盤面の予測が完全一致（新3行ゼロ＝恒等）。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    net9 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(9), seed=3)
    net10 = warm_start_value(net9, 9, 10)
    assert net10.feat_dim == E.feature_dim(10)
    e9 = E.encode(m, name, vocab, version=9)
    e10 = E.encode(m, name, vocab, version=10)
    b9 = {k: e9[k][None, ...] for k in ("scalars", "field", "card_idx")}
    b10 = {k: e10[k][None, ...] for k in ("scalars", "field", "card_idx")}
    assert float(net9.predict(b9)[0]) == pytest.approx(float(net10.predict(b10)[0]), abs=1e-9)
