"""CPU 自デッキの勝ち筋プラン（自動分類・docs/SPEC.md §2.5.5）。

CPU は **自分のデッキ構成は完全情報**で知っているので、構成から「このデッキはどう勝つか」の
静的プラン（アグロ／ミッドレンジ／コントロール）を逆算的に作る。プランは `cpu_ai.evaluate` に
渡され、**勝ち筋に沿うよう自分側の評価重みを補正**する＝デッキによって最善手が変わる。

「逆算」の実装は次の3層（フル探索の backward induction は非現実的なので近似する）:
  1. プラン分類      : 構成（平均コスト/カウンター密度/除去率/パワーカーブ）から攻め寄り度を出し分類。
  2. 動的重み        : アーキタイプごとに評価重みを切替（effect なし低パワーの存在価値・カウンター温存・
                       ライフ重視・攻め圧）。これが「効果なし5000未満を出すか」をデッキ依存で決める。
  3. 逆算項          : 「相手を削り切れる盤面」を加点（最短リーサルへ誘導）＋マイルストーン採点
                       ＝クロック先行（aggro）／**J値スケジュール遵守**（control＝実測 (相手J値−自分J値) が
                       構成から導いた理想差 `delta_schedule[t]` を上回る分を加点）。`cpu_ai._plan_progress` が担う。

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
#   idle_don_mult     : 葉（自分の手番でない静止点）での**余剰アクティブドン**価値の倍率（B-1・§2.5.3）。
#                       OPCG は防御にドンを付与できないので、ターン終了後に浮いたアクティブドンの保持
#                       価値は本来低い。これを 1.0 のままにすると「両枝でクロック同値→ドンの床(200/枚)で
#                       タイブレーク→握る」になり余剰ドン温存を招く。カウンターの薄い攻め寄りデッキほど
#                       強く減価し、浮かせ得を消して攻めへ向ける（<1.0=減価）。`is_turn=False` の自分側のみ作動。
#   threat_atk_mult   : 攻撃的キーワード資産（ダブルアタック/速攻/バニッシュ/アンブロッカブル）の倍率（A-2・
#                       §2.5.6）。aggro は攻め札を高く・control は低く見る。両側対称（自分の攻め札＝資産／
#                       相手の攻め札＝除去すべき脅威 を同じレンズで評価）。
#   threat_def_mult   : 防御的キーワード資産（効果耐性「KOされない」）の倍率（A-2）。control は高く・aggro は低く。
#   act_margin_mult   : 「何もしない（ターンを畳む）」判定マージン `_ACT_MARGIN` の倍率（A-2）。aggro は小さく
#                       （テンポ攻めを通す）・control は大きく（曖昧な展開は畳んで守りを残す）。
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
    idle_don_mult: float
    threat_atk_mult: float = 1.0
    threat_def_mult: float = 1.0
    act_margin_mult: float = 1.0
    # 理想ライン（J値スケジュール・§2.5.5／設計メモ 20260616）: ターン別に開くべき理想の
    #   (相手J値 − 自分J値)。`_plan_progress` が実測 J値差との乖離（スケジュール遵守度）を採点する。
    #   空 `()` は未導出（NEUTRAL・単体テストの _PRESETS 構築）＝従来のリソース差採点へフォールバック。
    delta_schedule: tuple = ()


# 中立プラン（テンプレ/構成が無いときのフォールバック＝ほぼ現行挙動）。
NEUTRAL = PlanProfile(
    n_cards=0, archetype="midrange", aggro_lean=0.5, avg_cost=0.0,
    vanilla_body_mult=1.0, attacker_mult=1.0, life_mult=1.0, counter_mult=1.0,
    lethal_mult=1.0, milestone_mult=1.0, clock_rate=0.8, idle_don_mult=1.0,
)

_PRESETS = {
    "aggro": dict(vanilla_body_mult=1.10, attacker_mult=1.25, life_mult=1.0,
                  counter_mult=0.85, lethal_mult=1.4, milestone_mult=1.2, clock_rate=1.2,
                  idle_don_mult=0.4, threat_atk_mult=1.30, threat_def_mult=0.85, act_margin_mult=0.6),
    "midrange": dict(vanilla_body_mult=1.0, attacker_mult=1.0, life_mult=1.0,
                     counter_mult=1.0, lethal_mult=1.0, milestone_mult=1.0, clock_rate=0.8,
                     idle_don_mult=0.7, threat_atk_mult=1.0, threat_def_mult=1.0, act_margin_mult=1.0),
    "control": dict(vanilla_body_mult=0.45, attacker_mult=0.9, life_mult=1.15,
                    counter_mult=1.4, lethal_mult=0.8, milestone_mult=1.0, clock_rate=0.5,
                    idle_don_mult=0.85, threat_atk_mult=0.85, threat_def_mult=1.25, act_margin_mult=1.5),
}


def _classify(aggro_lean: float) -> str:
    if aggro_lean >= _AGGRO_THRESHOLD:
        return "aggro"
    if aggro_lean <= _CONTROL_THRESHOLD:
        return "control"
    return "midrange"


# 理想ライン（J値スケジュール）導出パラメータ（§2.5.5・設計メモ 20260616）。
_SCHED_TURNS = 8                # 理想スケジュールを持つターン数（以降は末尾値でクランプ）
_SCHED_SLOPE_BASE = 0.25       # J値差/ターンの下限傾き
_SCHED_SLOPE_AGGRO = 0.9       # 攻め寄り度の寄与（速く差を開く理想＝アグロ）
_SCHED_SLOPE_REMOVAL = 0.6     # 除去密度の寄与（トレード強要で相手 J値を押し上げる）
_SCHED_SLOPE_MAX = 2.0


def _derive_delta_schedule(aggro_lean: float, removal_ratio: float) -> tuple:
    """構成から理想 J値差スケジュール（ターン別に開くべき (相手J値 − 自分J値)）を線形近似で導出。

    攻め寄り・除去多いほど傾きが急＝早ターンで差を開く理想。`cpu_ai._plan_progress` が
    実測 J値差との乖離（スケジュール遵守度）を採点する。フェア性: 参照は自デッキ構成のみ。
    """
    slope = _SCHED_SLOPE_BASE + _SCHED_SLOPE_AGGRO * aggro_lean + _SCHED_SLOPE_REMOVAL * removal_ratio
    slope = max(_SCHED_SLOPE_BASE, min(_SCHED_SLOPE_MAX, slope))
    return tuple(round(slope * t, 3) for t in range(_SCHED_TURNS + 1))


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
        avg_cost=prof.avg_cost,
        delta_schedule=_derive_delta_schedule(prof.aggro_lean, prof.removal_ratio),
        **p,
    )
