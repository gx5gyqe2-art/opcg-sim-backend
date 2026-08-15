"""符号化 v12（= v9 + リーダー物理要約24・**リーサル距離Δ抜き**・2026-08-15）の検証。

なぜ分岐したか（実測 2026-08-15）: v10 のリーサル距離Δはエンジンで台本を再生する実測特徴で
**~25ms/盤面**。探索は1手で数百回符号化するため、候補ネットの decide が **0.47s（v9）→
13.5s（v11）** と本番予算1秒を28倍超過した（アリーナが 10分/ペアで頭打ちになって発覚）。
一方リーダー要約は能力木の走査をカードIDでキャッシュ＝**実質ゼロコスト**で、gen15 系の
改善はこちらの寄与だった（v10 のΔは v53 で両系とも転移せず効果未実証）。よって
**出荷実績のある v9 系譜に無料の24列だけを継ぐ**のが v12。

固定する性質:
  - 次元 94・列レイアウトは [v9 70 | リーダー 24]
  - **v11 行からの列切り出しと bit 一致**（既存コーパスを再生成せず教師にできる根拠）
  - 前半70列は v9 と bit 一致（G14 からの温スタートが恒等である根拠）
  - **リーサルスキャンを呼ばない**（コスト削減の実体・呼べば失敗する細工で証明）
  - warm_start_value(9→12) が恒等（新24列がゼロの盤面で出力不変）
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import rl_encoder as E
import rl_net as RN
import replay_reeval as RE
import replay_runner as RR
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine, warm_start_value

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（符号化の版レイアウト）

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
    return LearnedEngine().vocab


def test_dim_and_layout():
    """94 = v9(70) + リーダー24。版マップに登録済み（次元は一意＝逆引きが成立する）。"""
    assert E.scalars_dim(12) == 94 == E.scalars_dim(9) + 24
    dims = [E.scalars_dim(v) for v in E.known_versions()]
    assert len(dims) == len(set(dims)), "版→列数が一意でないと次元からの版判別が壊れる"


def test_v12_equals_v11_slice_and_v9_prefix(board, vocab):
    """v11 行の列切り出し（[0:70]+[73:97]）と bit 一致・前半は v9 と bit 一致。"""
    m, name = board
    e9 = E.encode(m, name, vocab, version=9)["scalars"]
    e11 = E.encode(m, name, vocab, version=11)["scalars"]
    e12 = E.encode(m, name, vocab, version=12)["scalars"]
    sliced = np.concatenate([e11[:70], e11[73:97]])
    assert np.array_equal(e12, sliced), "v11 からの切り出しで教師を作れる前提が崩れている"
    assert np.array_equal(e12[:70], e9), "v9 からの append-only（温スタート恒等）が崩れている"


def test_v12_does_not_run_lethal_scan(board, vocab, monkeypatch):
    """v12 はリーサルスキャンを呼ばない（＝コスト削減の実体）。v11 は呼ぶ。"""
    import opcg_sim.src.learned.lethal as L

    def boom(*a, **k):
        raise AssertionError("v12 でリーサルスキャンが呼ばれた（安価版の意味がない）")

    monkeypatch.setattr(L, "lethal_scan", boom)
    m, name = board
    E.encode(m, name, vocab, version=12)          # 呼ばない＝成功する
    with pytest.raises(AssertionError):
        E.encode(m, name, vocab, version=11)      # 呼ぶ＝細工に当たる


def test_warm_start_9_to_12_is_identity(board, vocab):
    """G14 系譜（v9）→ v12 の温スタートは恒等（増えた24列がゼロなら出力不変）。"""
    m, name = board
    e9 = E.encode(m, name, vocab, version=9)
    net9 = RN.ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16,
                       feat_dim=E.feature_dim(9), seed=3)
    net12 = warm_start_value(net9, 9, 12)
    b9 = {"scalars": e9["scalars"][None, :], "field": e9["field"][None, ...],
          "card_idx": np.asarray(e9["card_idx"])[None, :]}
    b12 = dict(b9, scalars=np.concatenate(
        [e9["scalars"], np.zeros(24, np.float32)])[None, :])
    assert np.allclose(net9.predict(b9), net12.predict(b12))


def test_battle_resource_cols_v12_in_bounds():
    """リソース束の列（棚上げ中の評価器用）も v12 の次元内に収まる。"""
    cols = E.battle_resource_cols(12)
    assert len(cols) == len(set(cols))
    assert min(cols) >= 0 and max(cols) < E.scalars_dim(12)
    assert set(E.battle_resource_cols(9)) <= set(cols), "v9 の束は v12 に含まれるはず"
