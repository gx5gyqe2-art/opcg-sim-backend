"""ε強制防御（v26・`tests/harness/p3_loop._forced_defense_index`）の検証。

**なぜこの機構が要るか**（`docs/reports/cpu_v25_dense_regen_20260731.md` §5）: v4(c) の
防御応答温度は「訪問分布からサンプル」であり、その訪問分布を作るのは現行 value。守りに
価値を認めないネットでは訪問が PASS に集中し、温度1.0でも守る対局はほぼ生成されない
（v25 コーパスは温度延長込みで 2048局生成したが、守り採択率 0.281 ＝ gen8 0.289 と同値）。
「価値を知らないから守らない→守った対局が無いから価値を学べない」の循環を、訪問分布を
**無視した**強制抽選で断つ＝密生成に「守った世界の勝敗」を大量に入れる。

固定する性質:
  - eps=0（既定）で **None を返し乱数も引かない**＝既存生成の軌跡と rng 消費順が不変
    （seed_frac/relabel_frac と同じ作法。ここが崩れると全既存コーパスと非互換になる）
  - eps=1 で必ず「守る手」を選ぶ・PASS は決して選ばない
  - 守る手が無い窓（PASS のみ）では None＝呼び出し側の通常サンプルへ委ねる
  - 選択は守る手の**一様分布**（訪問分布に依存しない＝これが分布の新規性の源）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

import p3_loop as PL

pytestmark = pytest.mark.cpu_infra


def _win(n_def=3, with_pass=True):
    """防御窓の合法手列を模す（守る手 n_def 個 ++ PASS）。"""
    legal = [{"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": f"c{i}"}
             for i in range(n_def)]
    if with_pass:
        legal.append({"kind": "battle", "action_type": "PASS", "card_uuid": None})
    return legal


class _NoDrawRng:
    """引かれたら失敗する rng（乱数消費ゼロの契約を機械的に検証する）。"""

    def random(self):
        raise AssertionError("eps=0 で乱数を引いた（既存コーパスとの rng 消費順が壊れる）")

    def integers(self, *a, **k):
        raise AssertionError("eps=0 で乱数を引いた")


def test_eps_zero_is_inert_and_draws_no_randomness():
    assert PL._forced_defense_index(_win(), _NoDrawRng(), 0.0) is None


def test_no_defense_option_returns_none_without_drawing():
    """PASS しか無い窓は強制のしようがない＝乱数も引かず通常経路へ。"""
    only_pass = [{"kind": "battle", "action_type": "PASS", "card_uuid": None}]
    assert PL._forced_defense_index(only_pass, _NoDrawRng(), 1.0) is None


def test_eps_one_always_picks_a_defending_move():
    legal = _win(n_def=3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i = PL._forced_defense_index(legal, rng, 1.0)
        assert i is not None and legal[i]["action_type"] != "PASS"


def test_selection_is_uniform_over_defending_moves():
    """守る手の一様抽選（訪問分布に依存しない）＝分布の新規性の源。全枝が現れることを見る。"""
    legal = _win(n_def=4)
    rng = np.random.default_rng(7)
    picked = {PL._forced_defense_index(legal, rng, 1.0) for _ in range(200)}
    assert picked == {0, 1, 2, 3}          # PASS(index 4) は決して選ばれない


def test_eps_controls_force_rate():
    """ε は強制率そのもの（ε=0.5 で概ね半分）。残りは呼び出し側の通常サンプルへ落ちる。"""
    legal = _win(n_def=2)
    rng = np.random.default_rng(3)
    forced = sum(PL._forced_defense_index(legal, rng, 0.5) is not None for _ in range(400))
    assert 150 <= forced <= 250            # 二項(400,0.5) の十分広い帯


def test_selfplay_game_accepts_the_knob_with_identity_default():
    """生成コアが引数を受け、既定値が従来と同一シグネチャで通ること（配線の存在確認）。"""
    import inspect
    sig = inspect.signature(PL.selfplay_game)
    assert sig.parameters["def_force_eps"].default == 0.0
