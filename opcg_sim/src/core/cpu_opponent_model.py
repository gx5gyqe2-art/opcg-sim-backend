"""リーダー推測の相手モデル（`normal` 難易度・docs/SPEC.md §2.5.4）。

相手（人間）のリーダーに紐づく**テンプレートデッキ**（`cpu_templates`）から、相手が
「どんなデッキをどう回すか」の**静的プロファイル**を作る。隠れ情報（実手札・実デッキ）は
一切読まず、リーダーに紐づくメタ知識（テンプレ構成）の集計のみを使う＝フェア。

プロファイルは `cpu_ai.evaluate` に渡され、相手側の評価を次のように補正する:
  - `defense_factor` : 相手手札の防御価値の倍率（カウンター/ブロッカーが厚い構築ほど > 1）。
                       公開情報のみ（normal）で相手手札の中身を読まない代わりに、構築から
                       「この手札はどれくらい守りに使えるか」を推測して織り込む。
  - `aggro_lean`     : 相手の攻め寄り度 0..1（低コスト・低カウンター・除去少＝高）。CPU は値が
                       高いほど自分のライフを厚く見る（レースされる前提の守り）。

静的（観測ベリーフ更新・Monte Carlo 決定化は行わない）。
"""
from dataclasses import dataclass
from typing import List

# 集計の基準値（重み調整の起点）。
_BASE_COUNTER = 750.0          # カウンター平均値の基準（典型 ~0/1000/2000 の混在）
_DEF_FACTOR_MIN = 0.6
_DEF_FACTOR_MAX = 1.8
_BLOCKER_DEF_BONUS = 0.8       # ブロッカー比率 1.0 で defense_factor に最大 +0.8

# 除去（コントロール寄り）を示すテキスト手掛かり（粗い検出。攻め/受け傾向の判定用）。
# 注意（2026-06-19 修正）: 旧来は素の "ＫＯ"/"KO" と "デッキの下" を含めていたが、
#   - "KO" は **【KO時】＝このカードが KO された時** の自軍トリガー（防御/リソース札＝ホグバック/ペローナ/
#     マルコ等）を除去と誤カウント → 除去動詞の "KOする"/"KOできる" に限定。
#   - "デッキの下" は **自分のデッキを掘る**自己ディグ（残りをデッキの下へ）も拾う → 相手をデッキへ戻す
#     バウンス除去は "持ち主のデッキ" で捕捉できるため素の "デッキの下" は除外。
# これにより removal_ratio の過大評価（≒コントロール過剰分類・理想ライン傾きの過大）を是正する。
_REMOVAL_CUES = ("ＫＯする", "KOする", "ＫＯできる", "KOできる", "手札に戻", "持ち主のデッキ")


@dataclass(frozen=True)
class OpponentProfile:
    """テンプレートデッキ由来の静的な相手プロファイル。"""
    n_cards: int
    counter_avg: float          # 全カードのカウンター値平均
    counter_card_ratio: float   # カウンター値を持つカードの比率
    blocker_ratio: float        # ブロッカーの比率
    removal_ratio: float        # 除去テキストを持つカードの比率
    avg_cost: float
    defense_factor: float       # 相手手札防御価値の倍率（>1=守り厚い）
    aggro_lean: float           # 0..1（高=攻め寄り）


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_profile(masters: List) -> OpponentProfile:
    """テンプレートのカード定義（CardMaster の列・キャラ/イベント想定）から静的プロファイルを作る。

    リーダー/ドンは呼び出し側で除外して渡す想定だが、空入力でも安全な既定値を返す。
    """
    cards = [m for m in masters if m is not None]
    n = len(cards)
    if n == 0:
        # 中立既定（テンプレ未供給時のフォールバックと同義）。
        return OpponentProfile(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5)

    counters = [float(getattr(m, "counter", 0) or 0) for m in cards]
    counter_avg = sum(counters) / n
    counter_card_ratio = sum(1 for c in counters if c > 0) / n
    blocker_ratio = sum(1 for m in cards if "ブロッカー" in (getattr(m, "keywords", set()) or set())) / n
    removal_ratio = sum(1 for m in cards
                        if any(cue in (getattr(m, "effect_text", "") or "") for cue in _REMOVAL_CUES)) / n
    costs = [float(getattr(m, "cost", 0) or 0) for m in cards]
    avg_cost = sum(costs) / n

    # 防御倍率: カウンター平均の厚み ＋ ブロッカー比率のボーナス。
    defense_factor = _clamp(counter_avg / _BASE_COUNTER + _BLOCKER_DEF_BONUS * blocker_ratio,
                            _DEF_FACTOR_MIN, _DEF_FACTOR_MAX)

    # 攻め寄り度: 低コスト・低カウンター密度・除去少 ほど高い。
    cost_sig = _clamp((3.5 - avg_cost) / 1.5, 0.0, 1.0)
    counter_sig = _clamp((0.5 - counter_card_ratio) / 0.5, 0.0, 1.0)
    control_sig = _clamp(removal_ratio / 0.3, 0.0, 1.0)
    aggro_lean = _clamp(0.5 * cost_sig + 0.4 * counter_sig + 0.1 * (1.0 - control_sig), 0.0, 1.0)

    return OpponentProfile(
        n_cards=n,
        counter_avg=counter_avg,
        counter_card_ratio=counter_card_ratio,
        blocker_ratio=blocker_ratio,
        removal_ratio=removal_ratio,
        avg_cost=avg_cost,
        defense_factor=defense_factor,
        aggro_lean=aggro_lean,
    )
