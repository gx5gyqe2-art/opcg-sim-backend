"""CPU 自デッキの勝ち筋プラン（自動分類・docs/SPEC.md §2.5.5）。

CPU は **自分のデッキ構成は完全情報**で知っているので、構成から「このデッキはどう勝つか」の
静的プラン（アグロ／ミッドレンジ／コントロール）を逆算的に作る。プランは `cpu_ai.evaluate` に
渡され、**勝ち筋に沿うよう自分側の評価重みを補正**する＝デッキによって最善手が変わる。

「逆算」の実装は次の3層（フル探索の backward induction は非現実的なので近似する）:
  1. プラン分類      : 構成（平均コスト/カウンター密度/除去率/パワーカーブ）から攻め寄り度を出し分類。
  2. 動的重み        : アーキタイプごとに評価重みを切替（effect なし低パワーの存在価値・カウンター温存・
                       ライフ重視・攻め圧）。これが「効果なし5000未満を出すか」をデッキ依存で決める。
  3. 逆算項          : 「相手を削り切れる盤面」を加点（最短リーサルへ誘導）＋クロック先行/リソース差の
                       マイルストーン採点（勝利状態からの逆算サブゴール）。`cpu_ai._plan_progress` が担う。

フェア性: 参照するのは **自分のデッキ構成のみ**（隠れ情報＝相手の実手札・実デッキは読まない）。
静的（観測ベリーフ更新・決定化は行わない）＝相手モデル（cpu_opponent_model）と同方針。
"""
from dataclasses import dataclass
from typing import List, Optional

from . import cpu_opponent_model

# アーキタイプ分類の境界（aggro_lean: 0=受け寄り .. 1=攻め寄り）。
_AGGRO_THRESHOLD = 0.6
_CONTROL_THRESHOLD = 0.4

# アーキタイプ別の評価重み乗数（プリセット）。midrange は全て中立（≒現行挙動）。
#   vanilla_body_mult : 「効果なし・低パワー・関連キーワード無し」のキャラの“場にいるだけ”価値の倍率。
#                       control は置物を強く割り引き（カウンターを手放す損が上回る＝手札温存）。
#   attacker_mult     : 攻め圧（実際に攻撃できる体）の倍率。aggro で増し、control で減らす。
#   life_mult         : 自分ライフ価値の倍率。control はライフ温存を重視。
#   counter_mult      : 自分の手札カウンター価値の倍率。control は防御札の温存を重視（>1=出し渋る）。
#   lethal_mult       : 逆算リーサル項（削り切れる盤面加点）の倍率。aggro で攻めの止めを優先。
#   milestone_mult    : マイルストーン項（クロック先行/リソース差）の倍率。
#   clock_rate        : 想定ダメージクロック（相手ライフ/ターン）。aggro ほど速い。
@dataclass(frozen=True)
class PlanProfile:
    n_cards: int
    archetype: str            # "aggro" | "midrange" | "control"
    aggro_lean: float         # 0..1（高=攻め寄り）
    avg_cost: float
    vanilla_body_mult: float
    attacker_mult: float
    life_mult: float
    counter_mult: float
    lethal_mult: float
    milestone_mult: float
    clock_rate: float


# 中立プラン（テンプレ/構成が無いときのフォールバック＝ほぼ現行挙動）。
NEUTRAL = PlanProfile(
    n_cards=0, archetype="midrange", aggro_lean=0.5, avg_cost=0.0,
    vanilla_body_mult=1.0, attacker_mult=1.0, life_mult=1.0, counter_mult=1.0,
    lethal_mult=1.0, milestone_mult=1.0, clock_rate=0.8,
)

_PRESETS = {
    "aggro": dict(vanilla_body_mult=1.10, attacker_mult=1.25, life_mult=1.0,
                  counter_mult=0.85, lethal_mult=1.4, milestone_mult=1.2, clock_rate=1.2),
    "midrange": dict(vanilla_body_mult=1.0, attacker_mult=1.0, life_mult=1.0,
                     counter_mult=1.0, lethal_mult=1.0, milestone_mult=1.0, clock_rate=0.8),
    "control": dict(vanilla_body_mult=0.45, attacker_mult=0.9, life_mult=1.15,
                    counter_mult=1.4, lethal_mult=0.8, milestone_mult=1.0, clock_rate=0.5),
}


def _classify(aggro_lean: float) -> str:
    if aggro_lean >= _AGGRO_THRESHOLD:
        return "aggro"
    if aggro_lean <= _CONTROL_THRESHOLD:
        return "control"
    return "midrange"


def build_plan(masters: List, leader: Optional[object] = None) -> PlanProfile:
    """自デッキのカード定義（CardMaster の列・リーダー/ドンは除外して渡す想定）からプランを作る。

    空入力・分類不能でも安全に NEUTRAL（≒現行挙動）を返す。`leader` は将来のリーダー別補正用に
    受け取るが現状は未使用（構成からの自動分類のみ）。
    """
    cards = [m for m in masters if m is not None]
    if not cards:
        return NEUTRAL
    # aggro_lean / avg_cost は相手モデルと同じ集計器を流用（DRY・一貫性）。
    prof = cpu_opponent_model.build_profile(cards)
    archetype = _classify(prof.aggro_lean)
    p = _PRESETS[archetype]
    return PlanProfile(
        n_cards=len(cards), archetype=archetype, aggro_lean=prof.aggro_lean,
        avg_cost=prof.avg_cost, **p,
    )
