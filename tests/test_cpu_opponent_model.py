"""リーダー推測の相手モデル（`cpu_opponent_model.build_profile`）の純関数テスト。

docs/SPEC.md §2.5.4。`build_profile` はテンプレートデッキから静的プロファイルを作る純関数で、
評価器への配線（profile 補正・opponent-model）は撤去済み（CPU 評価は L1 単一系統）。本テストは
`build_profile` の集計ロジックだけを固定する。
"""
import types

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core import cpu_opponent_model


# ---------------------------------------------------------------------------
# build_profile（純関数）
# ---------------------------------------------------------------------------

def _master(counter=0, cost=3, keywords=None, text=""):
    return types.SimpleNamespace(counter=counter, cost=cost,
                                 keywords=set(keywords or []), effect_text=text)


def test_build_profile_empty_is_neutral():
    p = cpu_opponent_model.build_profile([])
    assert p.n_cards == 0
    assert p.defense_factor == 1.0
    assert p.aggro_lean == 0.5


def test_build_profile_counter_heavy_control_is_defensive():
    """高カウンター・ブロッカー多・除去ありの構築 → defense_factor 高・aggro_lean 低。"""
    cards = ([_master(counter=2000, cost=5, keywords=["ブロッカー"], text="このキャラをKOする")] * 8
             + [_master(counter=1000, cost=4)] * 12)
    p = cpu_opponent_model.build_profile(cards)
    assert p.counter_card_ratio == 1.0
    assert p.blocker_ratio > 0
    assert p.defense_factor > 1.2
    assert p.aggro_lean < 0.4


def test_build_profile_low_cost_no_counter_is_aggro():
    """低コスト・カウンター無し・除去無しの構築 → aggro_lean 高・defense_factor 低。"""
    cards = [_master(counter=0, cost=1)] * 10 + [_master(counter=0, cost=2)] * 10
    p = cpu_opponent_model.build_profile(cards)
    assert p.counter_card_ratio == 0.0
    assert p.aggro_lean > 0.7
    assert p.defense_factor < 1.0


def test_removal_cue_excludes_self_ko_triggers():
    """除去判定（removal_ratio）は「【KO時】＝自分が KO された時」の自軍トリガーを除去と数えない。

    2026-06-19 修正: 旧来は素の "KO" キューが【KO時】（ホグバック/ペローナ/マルコ等の防御・リソース札）を
    除去と誤カウントし、グラインド寄りデッキを過剰にコントロール分類していた。除去動詞 "KOする"/"KOできる" と
    相手をデッキへ戻すバウンス "持ち主のデッキ" のみを除去として捕捉する。"""
    self_ko = [_master(text="【KO時】カード2枚を引く。")] * 10            # 防御/リソース＝除去ではない
    self_dig = [_master(text="自分のデッキの上から3枚を見て、残りをデッキの下に置く。")] * 10  # 自己ディグ
    assert cpu_opponent_model.build_profile(self_ko).removal_ratio == 0.0
    assert cpu_opponent_model.build_profile(self_dig).removal_ratio == 0.0
    real_removal = ([_master(text="相手のコスト5以下のキャラ1枚までを、KOする。")] * 5
                    + [_master(text="コスト6以下のキャラ2枚までを、持ち主のデッキの下に置く。")] * 5)
    assert cpu_opponent_model.build_profile(real_removal).removal_ratio == 1.0
