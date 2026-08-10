"""v9 符号化世代（**ドンデッキ残＋自デッキ残キャラ頂点**・append-only・2026-08-10・v49）の検証。

v9 は v8（scalars 66）末尾に [自ドンデッキ残/10, 相手ドンデッキ残/10,
自デッキ残キャラ最大パワー/10000, 同最大コスト/10] の 4 を足す（70）。

なぜ要るか（v48/v49 実測 2026-08-10）:
  - **ドンデッキ残**: リーダー固有のドン上限（紫エネル OP15-058 はドンデッキ6）と
    「don!!-X で山へ戻したドンがリーダー効果で再装填される」経済が、この量なしには
    **原理的に**見えない。h1@2（turn1 サトリで掘る/無行動）のターン末比較で value
    Δ=+0.011＝無差別だった根因。両者のドンデッキ残は 6 vs 5 で違うのに符号化上は同一盤面
    だった。公開情報＝公平性契約に適合。
  - **自デッキ残キャラ頂点（最大パワー/最大コスト）**: 「山に何が眠っているか」を連続量で
    載せる。v4 の高コストキャラ残は cost≥7 のしきい値カウントで OP15-118（cost6/8000）を
    落とすが、既存特徴の意味は append-only 契約のため変えず、**しきい値特徴も新設しない**
    （ユーザ方針 2026-08-03）＝頂点（max）という連続量で足す。

固定する性質:
  - 版マップ: scalars_dim(9) = 66+4 = 70（append-only）
  - 配線: v9 末尾 4 値が盤面から直接計算した値と一致（offset ズレの検出）
  - 感度: ドンデッキ残・デッキ内容を変えると該当特徴**だけ**が動く（A1 案件の識別可能性）
  - 接頭辞不変: encode(v8) == encode(v9)[:66]
  - 恒等温スタート（v8→v9 拡張で出力不変）
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
OFF = E.SCALARS_V8   # v9 追加ブロックの先頭 offset（66）


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


def test_version_map_appends_four():
    assert E.scalars_dim(9) == E.scalars_dim(8) + 4 == 70
    assert 9 in E.known_versions() and E.known_versions() == sorted(E.known_versions())


def test_v9_tail_wiring_on_real_board(db, vocab):
    """v9 末尾 4 値の配線（offset ズレ検出）: 盤面から直接計算した値と一致する。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    me = m.p1 if m.p1.name == name else m.p2
    opp = m.p2 if m.p1.name == name else m.p1
    enc = E.encode(m, name, vocab, version=9)
    want = [len(me.don_deck) / 10.0, len(opp.don_deck) / 10.0] + E._deck_apex(me.deck)
    assert np.allclose(enc["scalars"][OFF:OFF + 4], want)
    assert want[3] > 0, "実盤面のデッキにキャラが残っていない（別 idx を選ぶ）"


def test_v9_don_deck_sensitivity(db, vocab):
    """A1 案件の識別可能性: ドンデッキ残が 1 枚違う盤面は v9 特徴で区別できる（v8 では同一）。

    h1@2 の「掘る線 vs 無行動」がまさにドンデッキ残 6 vs 5 の差＝v8 まで不可視だった。
    """
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    me = m.p1 if m.p1.name == name else m.p2
    e0 = E.encode(m, name, vocab, version=9)
    assert len(me.don_deck) > 0
    popped = me.don_deck.pop()
    try:
        e1 = E.encode(m, name, vocab, version=9)
    finally:
        me.don_deck.append(popped)
    diff = np.nonzero(e0["scalars"] != e1["scalars"])[0]
    assert list(diff) == [OFF], f"自ドンデッキ残の特徴だけが動くべき（動いた index: {list(diff)}）"
    assert e0["scalars"][OFF] - e1["scalars"][OFF] == pytest.approx(0.1)


def test_v9_deck_apex_sees_cost6_8000(db, vocab):
    """頂点量は cost6/8000（OP15-118 型＝v4 の cost≥7 カウントが落とす帯）を見る。"""
    class _T:
        name = "CHARACTER"

    class _M:
        def __init__(self, cost, power):
            self.type, self.cost, self.power = _T(), cost, power

    class _C:
        def __init__(self, cost, power):
            self.master = _M(cost, power)

    deck = [_C(1, 2000), _C(6, 8000), _C(2, 3000)]
    assert E._deck_apex(deck) == [8000 / 10000.0, 6 / 10.0]
    assert E._deck_apex([]) == [0.0, 0.0]


def test_prefix_invariant_v8(db, vocab):
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    e8 = E.encode(m, name, vocab, version=8)
    e9 = E.encode(m, name, vocab, version=9)
    assert np.allclose(e8["scalars"], e9["scalars"][:E.SCALARS_V8])


def test_warm_start_v8_to_v9_is_identity(db, vocab):
    """v8 ネットを v9 へ拡張しても同一盤面の予測が完全一致（新4行ゼロ＝恒等）。"""
    m, name = _mark_state(db, "opcg_replay_6563214359889287880.json.gz", 8)
    net8 = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                    feat_dim=E.feature_dim(8), seed=3)
    net9 = warm_start_value(net8, 8, 9)
    assert net9.feat_dim == E.feature_dim(9)
    e8 = E.encode(m, name, vocab, version=8)
    e9 = E.encode(m, name, vocab, version=9)
    b8 = {k: e8[k][None, ...] for k in ("scalars", "field", "card_idx")}
    b9 = {k: e9[k][None, ...] for k in ("scalars", "field", "card_idx")}
    assert float(net8.predict(b8)[0]) == pytest.approx(float(net9.predict(b9)[0]), abs=1e-9)
