"""v8 符号化世代（**自場集約＝相手v5との純対称化**・append-only・2026-08-02/03・v32）の検証。

v8 は v7（scalars 63）末尾に [自場総火力, 自場高パワー(≥7000)数, 自場ブロッカー数] の 3 を
足す（66）＝v5 が相手場に与えた集約（`_opp_field_aggregate`）と**同じ関数・同じ正規化**。

なぜ要るか（ユーザ指摘 2026-08-02「パワー2000以下のキャラの盤面価値は低い」）: v5 で相手場に
集約を与えた一方、自場はキャラ数の生カウントのみ＝2000 も 10000 も同じ「1体」。gen10 の
反実仮想実測（power_value_probe）でバニラ2000追加が 6000 体の 2/3 の加点を得ており、
自場の総火力を見せて「体の数」と「体の質」を分離する。**しきい値つきの弱ボディ特徴は
設けない**（汎用性のためのユーザ方針 2026-08-03）: 平均パワーは総火力÷キャラ数として
ネットが既存特徴から導出できる。

固定する性質:
  - 版マップ: scalars_dim(8) = 63+3 = 66（append-only）
  - 配線: v8 末尾 3 値が _opp_field_aggregate(自場) と一致（offset ズレの検出）
  - 対称性: 自分視点の自場集約 == 相手視点の相手場集約（同一盤面・同じ関数）
  - 接頭辞不変: encode(v7) == encode(v8)[:63]
  - 恒等温スタート（v7→v8 拡張で出力不変）
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
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化拡張の機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
OFF = E.SCALARS_V7   # v8 追加ブロックの先頭 offset（63）
V5_OPP_OFF = E.SCALARS_V4  # v5 相手場集約の先頭 offset（51）


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
    assert E.scalars_dim(8) == E.scalars_dim(7) + 3 == 66
    assert 8 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_v8_tail_wiring_on_real_board(db, vocab):
    """v8 末尾 3 値の配線（offset ズレ検出）: 盤面から直接計算した自場集約と一致する。

    m4@8＝中盤・両場にキャラがいる点を使う（自場集約が非ゼロで初めて配線ミスが見える）。
    """
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    me = m.p1 if m.p1.name == name else m.p2
    assert len(me.field) > 0, "自場が空だと配線検証にならない（別 idx を選ぶ）"
    enc = E.encode(m, name, vocab, version=8)
    assert np.allclose(enc["scalars"][OFF:OFF + 3], E._opp_field_aggregate(me.field))


def test_v8_is_symmetric_with_v5_opp_aggregate(db, vocab):
    """純対称性: 自分視点の「自場集約」（v8）は、相手視点の「相手場集約」（v5）と同一。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    other = m.p2.name if m.p1.name == name else m.p1.name
    e_me = E.encode(m, name, vocab, version=8)
    e_opp = E.encode(m, other, vocab, version=8)
    assert np.allclose(e_me["scalars"][OFF:OFF + 3],
                       e_opp["scalars"][V5_OPP_OFF:V5_OPP_OFF + 3])


def test_prefix_invariant_v7(db, vocab):
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    e7 = E.encode(m, name, vocab, version=7)
    e8 = E.encode(m, name, vocab, version=8)
    assert np.allclose(e7["scalars"], e8["scalars"][:E.SCALARS_V7])


def test_warm_start_v7_to_v8_is_identity(db, vocab):
    """v7 ネットを v8 へ拡張しても同一盤面の予測が完全一致（新3行ゼロ＝恒等）。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    net7 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(7), seed=3)
    net8 = warm_start_value(net7, 7, 8)
    assert net8.feat_dim == E.feature_dim(8)
    e7 = E.encode(m, name, vocab, version=7)
    e8 = E.encode(m, name, vocab, version=8)
    b7 = {k: e7[k][None, ...] for k in ("scalars", "field", "card_idx")}
    b8 = {k: e8[k][None, ...] for k in ("scalars", "field", "card_idx")}
    assert float(net7.predict(b7)[0]) == pytest.approx(float(net8.predict(b8)[0]), abs=1e-9)
