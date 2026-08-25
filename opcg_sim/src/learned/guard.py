"""受け方針箱（マクロ手化 P6-c・2026-08-25・`docs/cpu_macro_plan.md` §2 上位箱）。

相手ターンの入口で「このターンの受けの姿勢」を1つ選び、ターン中 sticky にする。
方針は**予算の層**＝「手札・盤面コストをどこまで払うか」だけを決め、効果の個別判断は
含まない（イベント/起動/ブロッカーの価値比較は下の層＝戦闘箱・対話箱の出口 value が
窓ごとに解決する。トリガー・強制誘発はコストが無いので方針の対象外）。

方針の語彙（見送り第一級・「局所判断」＝現行挙動が常に候補に残る）:
  local   … 方針を立てない＝戦闘ごとの現行入口コミット（既定・平坦時のフォールバック）
  pass    … 素通し（手札・盤面コストを払わない: SELECT_COUNTER/SELECT_BLOCKER を落とす）
  minimal … 最小防御（カウンター窓では最小の印字1枚ずつ＝貪欲最小・止まったら払わない）
  hold    … 死守（純カウンター窓で総量が足りるなら PASS を落とす＝守れる限り守る）

選び方は既存規約の再利用: 各方針で相手ターンを台本解決（自分の防御窓＝方針で候補整形→
出口 value 最良・相手の手＝policy 最良・対話窓＝対話箱）し、**ターン末盤面の value** で
比較する。差が min_spread 未満なら local（薄い差で現行の較正を上書きしない＝プランの
flat_exits と同じ原則）。
"""
from opcg_sim.src.core import cpu_ai, journal
from opcg_sim.src.core.journal import JournaledList
import random

from .config import PLAN_MIN_SPREAD, TURN_QUIESCE_MAX_PLIES
from .mcts import (in_battle, in_dialog, quiesce_choice, resolved_branch_values,
                   _turn_owner)

POLICIES = ("local", "pass", "minimal", "hold")
_DEF_TYPES = ("SELECT_COUNTER", "SELECT_BLOCKER")


def _counter_value(mgr, name, mv):
    """SELECT_COUNTER 候補の印字カウンター値（読めない形は None・pure）。"""
    actor = mgr.p1 if mgr.p1.name == name else mgr.p2
    uuid = (mv.get("payload") or {}).get("uuid") or mv.get("card_uuid")
    for c in (getattr(actor, "hand", None) or []):
        if getattr(c, "uuid", None) == uuid:
            v = int(getattr(c, "current_counter", 0) or 0)
            return v if v > 0 else None
    return None


def shape_moves(policy, mgr, name, moves):
    """防御窓の候補を方針で整形する（pure・空にはしない）。

    整形は SELECT_COUNTER/SELECT_BLOCKER にだけ作用し、それ以外の候補（イベント・起動・
    対話）は常に残す＝効果による受けは下の層が窓ごとに判断する。整形の結果が空になる
    場合は原候補を返す（自己修復＝方針が成立しない窓では局所判断に落ちる）。"""
    if policy in (None, "local"):
        return moves
    kinds = {m.get("action_type") for m in moves}
    if not (kinds & set(_DEF_TYPES)):
        return moves
    if policy == "pass":
        out = [m for m in moves if m.get("action_type") not in _DEF_TYPES]
        return out if out else moves
    need = cpu_ai.defense_battle_need(mgr)
    if need is None:
        return moves
    counters = [(m, _counter_value(mgr, name, m)) for m in moves
                if m.get("action_type") == "SELECT_COUNTER"]
    pure = all(v is not None for _, v in counters)     # 印字で閉じる窓か（イベント混在は触らない）
    if policy == "minimal":
        if need <= 0:
            # 止まっている: 印字カウンターを重ねない（D2' と同じ・ブロッカー/効果は残す）
            out = [m for m in moves if m.get("action_type") != "SELECT_COUNTER"]
            return out if out else moves
        if counters and pure:
            smallest = min(v for _, v in counters)
            out = [m for m in moves
                   if m.get("action_type") != "SELECT_COUNTER"
                   or _counter_value(mgr, name, m) == smallest]
            return out if out else moves
        return moves
    if policy == "hold":
        if need > 0 and counters and pure and \
                sum(v for _, v in counters) >= need:
            out = [m for m in moves if m.get("action_type") != "PASS"]
            return out if out else moves
        return moves
    return moves


def resolve_opp_turn(game, mgr, me, policy, value_fn, priors_fn,
                     battle_value_fn=None, dialog_box=False,
                     max_plies=TURN_QUIESCE_MAX_PLIES):
    """相手ターンが終わるまで mgr をその場で進める（方針の台本・巻き戻しは呼び出し側）。

    規約は `resolve_turn_inplace` と同族: 戦闘窓＝出口 value の箱（自分の窓は方針で候補
    整形してから）・対話窓＝対話箱（dialog_box 時）・それ以外＝policy 最良手。"""
    start_owner = _turn_owner(mgr)
    bvf = battle_value_fn or value_fn
    n = 0
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != start_owner:
            break
        name = game.current_player(mgr)
        legal = game.legal_actions(mgr) if name else None
        if not legal:
            break
        if name == me:
            legal = shape_moves(policy, mgr, me, legal)
        pick = None
        if len(legal) == 1:
            pick = 0
        elif in_battle(mgr):
            vals = resolved_branch_values(game, mgr, name, legal, bvf, priors_fn)
            ok = [i for i, v in enumerate(vals) if v is not None]
            pick = max(ok, key=lambda i: vals[i]) if ok else None
        elif dialog_box and in_dialog(mgr):
            vals = resolved_branch_values(game, mgr, name, legal, value_fn, priors_fn,
                                          window_pred=in_dialog)
            ok = [i for i, v in enumerate(vals) if v is not None]
            pick = max(ok, key=lambda i: vals[i]) if ok else None
        if pick is None:
            pick = quiesce_choice(mgr, legal, priors_fn)
        try:
            cpu_ai._apply_move_inplace(mgr, name, legal[pick], stop_at_select=True)
        except Exception:
            break
        n += 1
    return n


def select_guard_policy(game, world, me, value_fn, priors_fn,
                        battle_value_fn=None, dialog_box=False,
                        min_spread=PLAN_MIN_SPREAD):
    """方針を台本比較で1つ選ぶ（world は決定化済みクローン・不変）。返り値 (policy, 診断)。

    全方針を**同一の乱数列**（CRN）から解決し、ターン末盤面を `value_fn(・, me)` で比較。
    差が min_spread 未満なら "local"（薄い差で現行挙動を上書きしない）。"""
    base_rng_state = random.getstate()
    scores = {}
    for pol in POLICIES:
        random.setstate(base_rng_state)
        saved_events = world.action_events
        v = None
        try:
            with journal.transaction():
                world.action_events = JournaledList()
                resolve_opp_turn(game, world, me, pol, value_fn, priors_fn,
                                 battle_value_fn=battle_value_fn,
                                 dialog_box=dialog_box)
                v = value_fn(world, me)
        except Exception:
            v = None
        finally:
            world.action_events = saved_events
        scores[pol] = v
    random.setstate(base_rng_state)
    finite = {k: v for k, v in scores.items() if v is not None}
    diag = {"scores": {k: (round(v, 4) if v is not None else None)
                       for k, v in scores.items()}}
    if not finite:
        return "local", {**diag, "skipped": "no_exit"}
    best = max(finite, key=finite.get)
    spread = max(finite.values()) - min(finite.values())
    diag["spread"] = round(spread, 4)
    if len(finite) > 1 and spread < min_spread:
        return "local", {**diag, "skipped": "flat_exits"}
    return best, diag
