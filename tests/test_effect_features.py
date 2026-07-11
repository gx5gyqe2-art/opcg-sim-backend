"""EffFeat（効果セマンティクス特徴テーブル）の検証（docs/reports/effect_semantics_v3_plan_20260708.md §1）。

決定性・PAD規約・次元、および実カードのスポットチェック（設計書のゼロショット根拠カード:
OP03ナミ=VICTORY／OP11ナミ=ON_OPP_ATTACK+2kバフ+HAS_DON+手札コスト／コスト操作とパワーバフの分離／
ATTACH_DON全体センチネル／印刷キーワード）。
"""
import numpy as np

import conftest  # noqa: F401
from cpu_selfplay import _load_db
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import effect_features as EF

_DB = _load_db()
_VOCAB = E.build_vocab(_DB)
_TAB = EF.build_efffeat(_DB, _VOCAB)

A = EF.ABILITY_DIM
_BUFF_OFF = EF.N_TRIG + EF.N_ACT           # 能力スロット内の BUFF細分ブロック先頭
_MISC_OFF = _BUFF_OFF + 6
_COND_OFF = _MISC_OFF + 2
_COST_OFF = _COND_OFF + 6 + 1 + 2


def _slot(cid, s):
    return _TAB[_VOCAB[cid]][s * A:(s + 1) * A]


def _static(cid):
    return _TAB[_VOCAB[cid]][2 * A:]


def test_table_shape_pad_and_determinism():
    assert _TAB.shape == (len(_VOCAB) + 1, EF.FEATURE_DIM)
    assert not _TAB[0].any(), "PAD行(idx=0)は全ゼロのはず"
    again = EF.build_efffeat(_DB, _VOCAB)
    assert np.array_equal(_TAB, again), "EffFeat は決定的（2回構築で完全一致）のはず"


def test_coverage_all_cards_nonzero_and_ability_blocks():
    assert int(_TAB[1:].any(axis=1).sum()) == len(_VOCAB), "全カードが非ゼロ（静的ブロックがあるため）"
    n_ab = 0
    for cid, i in _VOCAB.items():
        c = _DB.get_card(cid)
        if c is not None and (getattr(c, "abilities", None) or []):
            n_ab += 1
            assert _TAB[i][:A].any(), f"{cid}: 効果持ちなのに能力slot1が全ゼロ"
    assert n_ab > 2000, "効果持ちカードが期待数（~2327）に達しない＝DB/走査の破損疑い"


def test_op03_nami_victory_condition_inversion():
    """OP03-040 ナミ＝「デッキ0枚で勝利」: VICTORY独立枠＋PASSIVE＋資源条件（DECK_COUNT）が立つ。"""
    s1 = _slot("OP03-040", 0)
    assert s1[EF.TRIGGERS.index("PASSIVE")] == 1.0
    assert s1[EF.N_TRIG + EF.ACTIONS.index("VICTORY")] == 1.0, "勝利条件変更が独立枠に立たない"
    assert s1[_COND_OFF + 2] == 1.0, "資源閾値条件（DECK_COUNT）が立たない"
    st = _static("OP03-040")
    assert st[0] == 1.0, "LEADER 種別ビット"


def test_op11_nami_defensive_leader():
    """OP11-041 ナミ＝ドン×1・相手アタック時・手札1枚捨てて+2000: slot2 に各特徴が立つ。"""
    s2 = _slot("OP11-041", 1)
    assert s2[EF.TRIGGERS.index("ON_OPP_ATTACK")] == 1.0
    assert s2[_BUFF_OFF + 1] == 1.0, "パワーバフ2000バケット"
    assert s2[_COND_OFF + 0] == 1.0, "HAS_DON 条件"
    assert s2[_COND_OFF + 1] == 1.0, "TURN_LIMIT 条件"
    assert s2[_COST_OFF + 1] == 1.0, "手札系コスト（DISCARD）"


def test_cost_buff_separated_from_power_buff():
    """OP01-067「コスト-1」: コスト操作枠が立ち、パワーバフのバケットは立たない（status×値スケール判別）。"""
    s1 = _slot("OP01-067", 0)
    assert s1[_BUFF_OFF + 4] == 1.0, "コスト操作枠"
    assert not s1[_BUFF_OFF:_BUFF_OFF + 4].any(), "パワーバフ/デバフのバケットに漏れている"


def test_attach_don_all_sentinel():
    """OP04-004「すべてに1枚ずつ」= base=99 センチネル → 全体付与ビット。"""
    s1 = _slot("OP04-004", 0)
    assert s1[EF.N_TRIG + EF.ACTIONS.index("ATTACH_DON")] == 1.0
    assert s1[_MISC_OFF + 1] == 1.0, "ATTACH_DON 全体センチネル(99)のビット"


def test_static_block_printed_keyword_counter_cost():
    """OP01-025 ゾロ: 印刷キーワード「速攻」・コスト帯・種別 CHARACTER。"""
    st = _static("OP01-025")
    assert st[1] == 1.0, "CHARACTER 種別"
    assert st[10 + 1] == 1.0, "印刷キーワード「速攻」"
    assert _static("OP01-067")[4] == 1.0, "カウンター1000ビット（OP01-067 クロコダイル）"
    assert _static("OP01-006")[5] == 1.0, "カウンター2000ビット（OP01-006 お玉）"
