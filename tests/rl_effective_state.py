"""encode_v3: 実効状態特徴（エンジン計算の“今の盤面で実際どうなるか”）を注入した state エンコーダ。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md §表現。encode_v2（fingerprint・静的な説明書）に、
エンジンがルール処理のため既に計算している**決定的な現在状態**を14次元追加する。全て公開情報＋
自分の手札から導出（隠匿情報の仮定なし＝v4b原則）。

追加の狙い（held-out ゲートで黒ひげ欠損に効く経路・OP16-080 実効果の検証済み理解に基づく）:
  - **実効コスト（current_cost）**: OP16-080 の【相手のターン中】コスト+1 等、印刷コストと違う
    「今のコスト」。従来は master.cost しか見えず防御パッシブが存在しないのと同じだった。
  - **場の動的状態の集約**: レスト数・付与ドン・現在パワー計（fingerprint は静的で持たない）。
  - **防御リソース**: 自手札のトリガー持ち枚数／リーダー防御（ON_OPP_ATTACK）の可用・相手の脅威。
    「トリガーを1枚払って守る」の払う/温存の採点に必要な入力。
  - **登場時無効フラグ**: negate_onplay_until（OP09-081 等・一般特徴）。
"""
import numpy as np

from rl_encoder_v2 import encode_v2, DIM as DIM_V2

EXTRA = 14
DIM_V3 = DIM_V2 + EXTRA


def _cur_cost(c):
    v = getattr(c, "current_cost", None)
    if v is None:
        v = getattr(c.master, "cost", 0)
    return float(v or 0)


def _cur_power(c):
    try:
        return float(c.current_power)
    except Exception:
        return float(getattr(c.master, "power", 0) or 0)


def _has_trigger(c):
    return bool(getattr(c.master, "trigger_text", None))


def _leader_has_opp_attack_ability(leader):
    if leader is None:
        return False
    for ab in (getattr(leader.master, "abilities", None) or []):
        if getattr(getattr(ab, "trigger", None), "name", "") in ("ON_OPP_ATTACK", "OPPONENT_ATTACK"):
            return True
    return False


def effective_state_feats(manager, me_name):
    """to-move 視点の実効状態 14 次元（float32）。"""
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = manager.p2 if manager.p1.name == me_name else manager.p1
    turn = getattr(manager, "turn_count", 0)

    def field_aggs(pl):
        fld = list(pl.field)
        n = max(len(fld), 1)
        mean_cost = sum(_cur_cost(c) for c in fld) / n
        tot_pow = sum(_cur_power(c) for c in fld)
        rest = sum(1 for c in fld if getattr(c, "is_rest", False))
        don = sum(float(getattr(c, "attached_don", 0) or 0) for c in fld)
        return mean_cost / 10.0, tot_pow / 50000.0, rest / 5.0, don / 10.0

    m0, m1, m2, m3 = field_aggs(me)
    o0, o1, o2, o3 = field_aggs(opp)
    trig_n = sum(1 for c in me.hand if _has_trigger(c))
    don_act = len(me.don_active)
    playable = sum(1 for c in me.hand if _cur_cost(c) <= don_act)
    my_def_ready = 1.0 if (_leader_has_opp_attack_ability(me.leader) and trig_n > 0) else 0.0
    opp_def_threat = 1.0 if (_leader_has_opp_attack_ability(opp.leader) and len(opp.hand) > 0) else 0.0
    # negate_onplay_until: 既定0。0>=turn_count(0) の偽陽性を防ぐため「設定済み(>0)かつ有効期間内」。
    def _negated(pl):
        v = getattr(pl, "negate_onplay_until", 0) or 0
        return 1.0 if (v > 0 and v >= turn) else 0.0
    my_onplay_negated = _negated(me)
    opp_onplay_negated = _negated(opp)

    return np.array([
        m0, m1, m2, m3, o0, o1, o2, o3,
        trig_n / 10.0, playable / 10.0,
        my_def_ready, opp_def_threat,
        my_onplay_negated, opp_onplay_negated,
    ], dtype=np.float32)


def encode_v3(manager, me_name, vocab, fps):
    """encode_v2 ＋ 実効状態14次元（DIM_V3）。"""
    return np.concatenate([
        encode_v2(manager, me_name, vocab, fps),
        effective_state_feats(manager, me_name),
    ]).astype(np.float32)


def make_value_fn_for(net, vocab, fps, encode_fn):
    """任意の encode で葉価値 [-1,1] を返す value_fn（TreeMCTS 用）。"""
    def vf(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        return float(np.tanh(net.value(encode_fn(state, to_move, vocab, fps))))
    return vf
