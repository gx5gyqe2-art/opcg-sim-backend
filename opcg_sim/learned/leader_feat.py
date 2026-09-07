"""リーダー物理要約特徴（v11・2026-08-14・`backbone_leader_feature_desk_20260813.md` §5）。

**問い**: 接戦盤面の帰趨を支配する「リーダーの再帰効果」（毎ターン積み上がるドン経済・
回復・ミル・常在修正）が、現行符号化に**1ビットも映っていない**（消去はしご 2.6σ・
重み直し天井 r=0.16 で確定）。その帰結を **ID ではなく物理量**（毎ターンの期待レート）
として符号化へ渡す。ID 非依存＝パースできる新リーダーへ即汎化（本プロジェクトの根本制約）。

**導出**: パース済み能力木（`effect_types.Ability`）を ActionType で歩き、
トリガー種別の「発火頻度重み」を掛けて毎ターン率へ換算する。
  - TURN_END/TURN_START/ACTIVATE_MAIN(ターン1回) ≈ 1回/ターン
  - ON_ATTACK ≈ 1回/ターン（リーダーは概ね毎ターン1回アタック）
  - PASSIVE/YOUR_TURN/OPPONENT_TURN ＝ 常在（重み1・持続量として読む）
  - その他（ON_PLAY 等リーダーで発火しない/稀）＝ 0.3 の保守重み
  - コスト句（ドン!!−N・RETURN_DON 等）は**負のドン率**として算入
**未対応の明示**: 条件（特徴《…》要求等）の動的充足ゲートは v1 では掛けない
（実デッキはリーダー軸に寄せてあるため実盤面では概ね充足＝静的近似で開始。
条件付き能力の存在は cond_frac 次元として渡す）。ルール改変（デッキ0勝利・
ドンデッキN枚）は専用次元（rule_flag / don_deck_delta）。

キャッシュ: (card_id, effect_text) キー。符号化は観測＝ global random を消費しない
（純粋な木walk・エンジン実行なし）。
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import numpy as np

from opcg_sim.src.models.effect_types import Ability, Branch, Choice, GameAction, Sequence
from opcg_sim.src.models.enums import ActionType, TriggerType, Zone

# 次元定義（順序は append-only 契約: 変更禁止・追加は末尾のみ）
DIMS = ("don_rate",        # 0: ドン経済Δ/ターン（RAMP_DON − RETURN/REST_DON コスト。ATTACH は配分であり経済でないため対象外）
        "life_rate",       # 1: ライフ獲得率（LIFE_RECOVER・LIFE への MOVE_CARD）
        "mill_self",       # 2: 自デッキ削り率（TRASH_FROM_DECK/DECK_BOTTOM 自分側）
        "mill_opp",        # 3: 相手デッキ削り率
        "draw_rate",       # 4: ドロー/手札補充率
        "pow_own",         # 5: 自軍常在/毎ターンパワー修正（/1000）
        "pow_opp",         # 6: 敵軍パワー修正（デバフ・/1000・負値）
        "removal",         # 7: 除去率（KO/バウンス/FREEZE/LOCK）
        "deploy",          # 8: 展開率（PLAY_CARD・コスト軽減 COST_BUFF 込み）
        "atk_disable",     # 9: 自リーダーのアタック不可（ATTACK_DISABLE）
        "rule_flag",       # 10: ルール改変あり（VICTORY/RULE 系・デッキ0勝利等）
        "cond_frac")       # 11: 条件付き（特徴《…》等）能力の割合＝静的近似の告白次元
LEADER_FEAT_DIM = len(DIMS)

_RATE = {TriggerType.TURN_END: 1.0, TriggerType.TURN_START: 1.0,
         TriggerType.ACTIVATE_MAIN: 1.0, TriggerType.ON_ATTACK: 1.0,
         TriggerType.OPP_TURN_END: 1.0,
         TriggerType.PASSIVE: 1.0, TriggerType.YOUR_TURN: 1.0,
         TriggerType.OPPONENT_TURN: 1.0, TriggerType.RULE: 1.0,
         # 防御系（2026-08-14 修正: 相手は毎ターン攻撃し、接戦帯ではライフも毎ターン動く。
         # 0.3 の既定重みでは fixture ナミ（被弾ドロー＋防御パンプ）とシャンクス（アタック時
         # デバフ）の主力能力がほぼ消えていた——ナミ:シャンクス帯が bb6 で動かなかった主因）
         TriggerType.ON_OPP_ATTACK: 1.0, TriggerType.OPPONENT_ATTACK: 1.0,
         TriggerType.ON_LIFE_DECREASE: 0.7, TriggerType.ON_DAMAGE_DEALT_TO_LIFE: 0.7}
_RATE_DEFAULT = 0.3

_DON_DECK_RE = re.compile(r"ドン!!デッキは(\d+)枚")
_TRAIT_RE = re.compile(r"特徴《[^》]+》")


def _val(v) -> float:
    """ValueSource → 概算量（dynamic は base が無ければ 1 の保守値）。"""
    if v is None:
        return 1.0
    base = float(getattr(v, "base", 0) or 0)
    if base == 0 and getattr(v, "dynamic_source", None):
        base = 1.0
    mult = float(getattr(v, "multiplier", 1) or 1)
    return (base if base else 1.0) * (mult if mult else 1.0)


def _tgt_is_self(node: GameAction) -> Optional[bool]:
    t = getattr(node, "target", None)
    p = getattr(t, "player", None) if t is not None else None
    name = getattr(p, "name", None) or (str(p) if p is not None else None)
    if name is None:
        return None
    if "OPPONENT" in str(name).upper() or "OPP" in str(name).upper():
        return False
    return True


def _walk(node, out: List[GameAction]):
    if node is None:
        return
    if isinstance(node, GameAction):
        out.append(node)
        _walk(getattr(node, "sub_effect", None), out)
    elif isinstance(node, Sequence):
        for a in node.actions:
            _walk(a, out)
    elif isinstance(node, Branch):
        _walk(node.if_true, out)
        _walk(node.if_false, out)
    elif isinstance(node, Choice):
        for a in node.options:
            _walk(a, out)


def _accumulate(vec: np.ndarray, node: GameAction, w: float, sign: float = 1.0):
    at = getattr(node, "type", None)
    v = _val(getattr(node, "value", None))
    self_side = _tgt_is_self(node)
    if at in (ActionType.RAMP_DON, ActionType.ACTIVE_DON):
        # ACTIVE_DON（レスト起こし）＝再利用可能ドン＝経済（2026-08-14 修正: bg_luffy OP16-022）
        vec[0] += sign * w * max(v, 1.0)
    elif at in (ActionType.RETURN_DON, ActionType.REST_DON, ActionType.FREEZE_DON):
        vec[0] -= w * max(v, 1.0) * (1.0 if sign > 0 else 0.5)
    elif at in (ActionType.LIFE_RECOVER, ActionType.HEAL) or (
            at == ActionType.MOVE_CARD and getattr(node, "destination", None) == Zone.LIFE):
        vec[1] += sign * w * max(v, 1.0)
    elif at in (ActionType.TRASH_FROM_DECK, ActionType.DECK_BOTTOM):
        k = 2 if self_side in (True, None) else 3
        vec[k] += w * max(v, 1.0)
    elif at == ActionType.DRAW:
        vec[4] += sign * w * max(v, 1.0)
    elif at == ActionType.DISCARD:
        # 手札経済の負側（2026-08-14 被覆監査: 落ち26件の筆頭）。自分側＝コスト/自傷は
        # draw_rate から減算（手札Δ/ターンに統一）。相手側＝ハンデス＝資源攻撃として除去へ。
        if self_side is False:
            vec[7] += w * 0.5
        else:
            vec[4] -= w * max(v, 1.0)
    elif at in (ActionType.BP_BUFF, ActionType.SET_BASE_POWER, ActionType.BUFF,
                ActionType.SWAP_POWER):
        # BUFF は BP_BUFF と別列挙（2026-08-14 修正: fixture ナミ+2000/シャンクス−1000 は
        # BUFF で表現されており旧分岐では消えていた）。符号は base の符号×対象側で決める。
        amt = v / 1000.0
        if self_side is False:
            vec[6] += w * (-abs(amt))
        else:
            vec[5] += w * amt
    elif at in (ActionType.KO, ActionType.FREEZE, ActionType.LOCK,
                ActionType.PREVENT_REST, ActionType.DEAL_DAMAGE) or (
            at in (ActionType.BOUNCE, ActionType.MOVE_CARD, ActionType.MOVE_TO_HAND,
                   ActionType.TRASH, ActionType.MOVE) and self_side is False):
        # BOUNCE 等の移動系は相手側対象のみ除去（自分側はコスト/配置換え・ハンニャバル自己戻し等）
        vec[7] += w
    elif at == ActionType.REST and self_side is False:
        vec[7] += w * 0.5                          # 相手キャラのレスト＝擬似除去（半分重み）
    elif at == ActionType.ACTIVE and self_side is not False:
        vec[8] += w * 0.5                          # 自キャラ起こし＝テンポ（展開の半分重み）
    elif at in (ActionType.PLAY_CARD, ActionType.EXECUTE_EVENT):
        vec[8] += w
    elif at in (ActionType.COST_BUFF, ActionType.COST_CHANGE, ActionType.SET_COST):
        vec[8] += w * 0.5
    elif at == ActionType.ATTACK_DISABLE:
        vec[9] = 1.0
    elif at in (ActionType.VICTORY, ActionType.EXTRA_TURN, ActionType.RULE_PROCESSING):
        vec[10] = 1.0


_CACHE: dict = {}


def leader_static_vector(master) -> np.ndarray:
    """リーダー master → 物理要約ベクトル（LEADER_FEAT_DIM）。非リーダーにも安全（能力を同規約で読む）。"""
    key = (getattr(master, "card_id", None), getattr(master, "effect_text", "") or "")
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    vec = np.zeros(LEADER_FEAT_DIM, dtype=np.float32)
    abilities = list(getattr(master, "abilities", ()) or ())
    n_cond = 0
    for ab in abilities:
        trig = getattr(ab, "trigger", None)
        if getattr(trig, "name", "UNKNOWN") == "UNKNOWN":
            continue
        w = _RATE.get(trig, _RATE_DEFAULT)
        acts: List[GameAction] = []
        _walk(getattr(ab, "effect", None), acts)
        for node in acts:
            _accumulate(vec, node, w, sign=1.0)
        cost_acts: List[GameAction] = []
        _walk(getattr(ab, "cost", None), cost_acts)
        for node in cost_acts:
            _accumulate(vec, node, w, sign=-1.0)
        raw = getattr(ab, "raw_text", "") or ""
        if getattr(ab, "condition", None) is not None or _TRAIT_RE.search(raw):
            n_cond += 1
    text = getattr(master, "effect_text", "") or ""
    m = _DON_DECK_RE.search(text)
    if m:                                          # エネル型: ドンデッキN枚ルール＝経済上限Δ
        vec[0] += (int(m.group(1)) - 10) / 10.0
        vec[10] = 1.0
    if "敗北する代わりに勝利" in text or "勝利する" in text:
        vec[10] = 1.0
    if abilities:
        vec[11] = n_cond / len(abilities)
    vec = np.clip(vec, -5.0, 5.0)
    _CACHE[key] = vec
    return vec


def leader_pair_vectors(manager, me_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """(自リーダー, 相手リーダー) のベクトル。リーダー不在（理論上ない）はゼロ。"""
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = manager.p2 if manager.p1.name == me_name else manager.p1
    z = np.zeros(LEADER_FEAT_DIM, dtype=np.float32)
    mv = leader_static_vector(me.leader.master) if getattr(me, "leader", None) is not None else z
    ov = leader_static_vector(opp.leader.master) if getattr(opp, "leader", None) is not None else z
    return mv, ov
