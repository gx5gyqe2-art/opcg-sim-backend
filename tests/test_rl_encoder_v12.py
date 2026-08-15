"""符号化 v12（= v9 + リーダー物理要約24・**リーサル距離Δ抜き**・2026-08-15）の検証。

なぜ分岐したか（実測・`docs/reports/gen15_adoption_20260815.md` §3）: v10 のリーサル距離Δは
エンジンで台本を再生する実測特徴で **~25ms/盤面**。探索は1手で数百回符号化するため、
decide が **0.47s（v9）→13.5s（v11）** と本番予算1秒を28倍超過していた。一方リーダー要約は
カードID キャッシュで**実質ゼロコスト**かつ gen15 系の改善の本体だったので、
**出荷実績のある v9 系譜に無料の24列だけを継ぐ**のが v12。

固定する性質:
  - 版マップに登録され次元は 94（= v9 70 + 24）
  - `encode(version=12)` は **v11 の列 [0:70]+[73:97] と bit 一致**（＝コーパスは
    `corpus_v11_to_v12` の切り出しで作れる＝**対局の再生成が不要**であることの根拠）
  - 前半70列は **v9 と一致**（G14 からの温スタートが末尾ゼロ追加の恒等拡張になる）
  - **Δのエンジン台本再生を通らない**（`lethal_scan` を呼ばない＝コストが v9 並み）
  - `battle_resource_cols(12)` は範囲内で、末尾24列がリーダー要約を指す
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import rl_encoder as E
import replay_reeval as RE
import replay_runner as RR
from cpu_selfplay import _load_db

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化世代）

import os

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
M1 = "opcg_replay_2057134394987494995.json.gz"


@pytest.fixture(scope="module")
def board():
    db = _load_db()
    raw = RE.load_replay_json(os.path.join(FIX, M1))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, 3)
    return m, (who if isinstance(who, str) else who.name)


@pytest.fixture(scope="module")
def vocab():
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    return LearnedEngine().vocab


def test_version_registered_with_94_dims():
    assert 12 in E.known_versions()
    assert E.scalars_dim(12) == 94 == E.scalars_dim(9) + 24
    dims = [E.scalars_dim(v) for v in E.known_versions()]
    assert len(set(dims)) == len(dims), "次元→版の逆引きが一意でなくなった"


def test_v12_is_v11_minus_lethal_columns(board, vocab):
    """v12 == v11 の [0:70]+[73:97]（コーパス切り出しの正当性そのもの）。"""
    m, name = board
    s11 = E.encode(m, name, vocab, version=11)["scalars"]
    s12 = E.encode(m, name, vocab, version=12)["scalars"]
    sliced = np.concatenate([s11[:E.scalars_dim(9)],
                             s11[E.scalars_dim(10):E.scalars_dim(11)]])
    assert np.array_equal(s12, sliced)


def test_v12_prefix_equals_v9(board, vocab):
    """前半70列は v9 と一致＝G14 からの温スタートが恒等（末尾ゼロ追加）になる。"""
    m, name = board
    s9 = E.encode(m, name, vocab, version=9)["scalars"]
    s12 = E.encode(m, name, vocab, version=12)["scalars"]
    assert np.array_equal(s12[:len(s9)], s9)


def test_v12_does_not_run_lethal_scan(board, vocab, monkeypatch):
    """Δのエンジン台本再生を通らない（コストが v9 並みである理由）。"""
    import opcg_sim.src.learned.lethal as L
    calls = {"n": 0}
    orig = L.lethal_scan

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(L, "lethal_scan", counting)
    m, name = board
    E.encode(m, name, vocab, version=12)
    assert calls["n"] == 0, "v12 が lethal_scan を呼んでいる（安価版の前提が崩れる）"
    E.encode(m, name, vocab, version=11)
    assert calls["n"] == 1, "v11 は従来どおり lethal_scan を呼ぶはず"


def test_battle_resource_cols_v12_within_bounds():
    cols = E.battle_resource_cols(12)
    assert len(cols) == len(set(cols))
    assert min(cols) >= 0 and max(cols) < E.scalars_dim(12)
    tail = [c for c in cols if c >= E.scalars_dim(9)]
    assert tail == list(range(E.scalars_dim(9), E.scalars_dim(12))), \
        "末尾24列（リーダー要約）を指していない"
