"""ターンプランの列挙・K世界期待値評価・逐次実行（v37②・2026-08-06）。

**ターンを箱として畳む第2段**（ユーザ方針 2026-08-05「ターンも箱とみなす事でゲーム全体としての
プランを立てられるように」）。手単位の木は「1つの決定化世界」で建つため、サンプルされた相手
手札がたまたま誤った手を良く見せる seed で誤る（v37① 実測: m2@44/m5@7 の 0.6 前後は正着が
1位の seed と外す seed の混合）。本モジュールは**予測を当てるのではなく分布で選ぶ**:

  1. 提案: policy の温度サンプリングで自ターンを何本かロールアウトし、候補プラン
     （メイン手の列）を作る（戦闘窓は出口 value の箱＝gen12 の確立規約）
  2. 評価: 各プランを **K 個の決定化世界（プラン間で共有＝CRN）** で箱実行し、
     **自ターン末の value の平均**で選ぶ（4プラン×32世界の手動プローブと同じ原理。
     22pt 差の検出実績＝seed 依存のブレを平均で消す）
  3. 実行: 選んだプランを1手ずつ返す（ターン内 sticky）。次の手が実盤面で非合法に
     なったら（想定外の応手＝計画が割れた）プランを捨てて再計画する

プランの手は (action_type, payload) の**実 uuid** で持つ。決定化は相手の伏せ手札だけを
再サンプルする（自分の手札・場・相手の公開盤面の uuid は全世界で共通）ため、同じ手が
全世界でそのまま適用できる。自ターンのメイン判断のみが対象＝防御窓・相手ターンは不変。
"""
import numpy as np

from opcg_sim.src.core import cpu_ai
from .config import (PLAN_MIN_SPREAD, PLAN_PROPOSALS, PLAN_TEMP, PLAN_WORLDS,
                     TURN_QUIESCE_MAX_PLIES)
from .mcts import in_battle, quiesce_choice, resolved_branch_values, _turn_owner


def move_sig(mv):
    """手の同一性キー（pure）: action_type ＋ payload の uuid/対象（世界に依存しない）。

    効果選択（RESOLVE_EFFECT_SELECTION）は uuid/target_ids を持たず `selected_uuids` と
    accepted で区別されるため、それも鍵に含める（含めないと「誰に -1000 を当てるか」等の
    全選択肢が同一キーに潰れ、プラン実行が別の選択肢を適用してしまう・v39）。"""
    p = mv.get("payload") or {}
    return (mv.get("action_type") or p.get("action_type"), p.get("uuid"),
            tuple(p.get("target_ids") or ()),
            tuple(p.get("selected_uuids") or ()), p.get("accepted"))


def _find_move(legal, sig):
    for mv in legal:
        if move_sig(mv) == sig:
            return mv
    return None


def _battle_box_step(game, mgr, name, value_fn, priors_fn):
    """戦闘窓を出口 value の箱で1手進める（gen12 の木の箱化と同一規約）。"""
    legal = game.legal_actions(mgr)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    vals = resolved_branch_values(game, mgr, name, legal, value_fn, priors_fn)
    ok = [i for i, v in enumerate(vals) if v is not None]
    return legal[max(ok, key=lambda i: vals[i])] if ok else legal[0]


def rollout_plan(game, world, name, value_fn, priors_fn, rng, temp=PLAN_TEMP,
                 max_plies=TURN_QUIESCE_MAX_PLIES):
    """1本のターンロールアウトから候補プラン（自分のメイン手の signature 列）を作る。

    メイン手＝priors の温度サンプリング（temp=0 で argmax）。戦闘窓＝箱。TURN_END で終了
    （TURN_END 自体はプランに含めない＝実行側は「プランが尽きたら」木/既定に委ねる）。"""
    steps = []
    mgr = world
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != name:
            break
        actor = game.current_player(mgr)
        if actor is None:
            break
        if in_battle(mgr):
            mv = _battle_box_step(game, mgr, actor, value_fn, priors_fn)
        elif actor != name:
            mv = _battle_box_step(game, mgr, actor, value_fn, priors_fn)  # 相手の割込は箱
        else:
            legal = game.legal_actions(mgr)
            if not legal:
                break
            if priors_fn is not None and len(legal) > 1:
                p = priors_fn(mgr, legal)
                if p is None:
                    i = 0
                elif temp <= 0:
                    i = int(np.argmax(p))
                else:
                    w = np.maximum(np.asarray(p, dtype=float), 1e-9) ** (1.0 / temp)
                    i = int(rng.choice(len(legal), p=w / w.sum()))
            else:
                i = 0
            mv = legal[i]
            if (cpu_ai._describe_move(mgr, mv) or {}).get("action_type") == "TURN_END":
                break
            steps.append(move_sig(mv))
        if mv is None:
            break
        nxt = game.apply(mgr, mv, actor)
        if nxt is None:
            break
        mgr = nxt
    return tuple(steps)


def execute_plan(game, world, name, steps, value_fn, priors_fn,
                 max_plies=TURN_QUIESCE_MAX_PLIES):
    """プランを world 上で箱実行し**ターン末の盤面**を返す（world は不変・clone-apply）。

    **serve（プラン読み出し）と教師（プランCF生成）が共有する実行規約の単一の正**（v38）。
    別々に実装すると必ずずれ、教師が「serve が実際に到達しない盤面」を教えることになる
    （v35 の train/serve skew と同型の予防）。

    手が非合法な世界では skip（相手の応手次第で対象が消える等＝その世界でのプランの自然な
    縮退）。プランが尽きたら TURN_END を適用してターンを閉じる（閉じるまでがプラン）。"""
    mgr = world
    idx = 0
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != name:
            return mgr                          # 終局／ターンが替わった＝そこが出口
        actor = game.current_player(mgr)
        if actor != name or in_battle(mgr):
            mv = _battle_box_step(game, mgr, actor, value_fn, priors_fn)
        else:
            legal = game.legal_actions(mgr)
            mv = None
            while idx < len(steps) and mv is None:
                mv = _find_move(legal, steps[idx])
                idx += 1                        # 非合法な手は skip（縮退）
            if mv is None:                      # プランが尽きた → ターンを閉じる
                for cand in legal:
                    if (cpu_ai._describe_move(mgr, cand) or {}).get("action_type") == "TURN_END":
                        mv = cand
                        break
                if mv is None and legal:
                    # TURN_END が出せない＝自分の効果選択が保留中（v39 でこれが探索の決定点に
                    # なった）。プランに無い選択は policy 最良手で埋める＝ターンを閉じられずに
                    # 途中の盤面を「ターン末」と誤って評価する事故を防ぐ。
                    mv = legal[quiesce_choice(mgr, legal, priors_fn)]
        if mv is None:
            return mgr
        nxt = game.apply(mgr, mv, actor)
        if nxt is None:
            return mgr
        mgr = nxt
    return mgr


def evaluate_plan(game, world, name, steps, value_fn, priors_fn,
                  max_plies=TURN_QUIESCE_MAX_PLIES, exit_value_fn=None):
    """プランを箱実行した**自ターン末の value**（name 視点）。実行は `execute_plan` が正。

    `exit_value_fn`（v39・ターン末専用ヘッド）を渡すと**出口盤面の評価だけ**をそちらで行う。
    実行途中の戦闘窓は従来どおり `value_fn`（戦闘出口の較正＝gen12 が持つ規約）で畳む＝
    「どの箱の出口か」と「どのヘッドで測るか」を1対1に保つ。None は従来と同値。"""
    exit_mgr = execute_plan(game, world, name, steps, value_fn, priors_fn, max_plies)
    return (exit_value_fn or value_fn)(exit_mgr, name)


def select_plan(game, manager, name, value_fn, priors_fn, rng,
                n_worlds=PLAN_WORLDS, n_proposals=PLAN_PROPOSALS, exit_value_fn=None,
                min_spread=PLAN_MIN_SPREAD):
    """プランを提案→K世界期待値で選ぶ。返り値 (steps, 診断 dict)。候補が無ければ (None, {})。

    世界はプラン間で共有（CRN）＝差はプランだけから生じる。提案の1本目は必ず argmax
    （現行 policy の最良線を常に候補に含める）。

    `min_spread`（v39・`config.PLAN_MIN_SPREAD`）: 候補の出口 value の幅がこれ未満なら
    **箱化を放棄して (None, 診断) を返す**＝呼び出し側は従来の探索に委ねる。平坦な窓の薄い差は
    プランの優劣ではなくノイズで、そこで箱に決めさせると決定点近傍の較正（gen11 で教えた
    m1@3 型の矯正など）を薄い差で上書きしてしまう。"""
    worlds = []
    for _ in range(n_worlds):
        try:
            worlds.append(game.determinize(manager, name, rng))
        except Exception:
            break
    if not worlds:
        return None, {}
    plans = []
    for k in range(n_proposals):
        t = 0.0 if k == 0 else PLAN_TEMP
        steps = rollout_plan(game, worlds[k % len(worlds)], name, value_fn, priors_fn,
                             rng, temp=t)
        if steps and steps not in plans:
            plans.append(steps)
    if not plans:
        return None, {}
    scores = []
    for steps in plans:
        vs = [evaluate_plan(game, w, name, steps, value_fn, priors_fn,
                            exit_value_fn=exit_value_fn) for w in worlds]
        vs = [v for v in vs if v is not None]
        scores.append(float(np.mean(vs)) if vs else float("-inf"))
    best = int(np.argmax(scores))
    finite = [s for s in scores if s > float("-inf")]
    spread = float(max(finite) - min(finite)) if len(finite) > 1 else 0.0
    diag = {"n_plans": len(plans), "n_worlds": len(worlds),
            "scores": [round(s, 4) for s in scores], "best": best,
            "spread": round(spread, 4)}
    if len(finite) > 1 and spread < min_spread:
        return None, {**diag, "skipped": "flat_exits"}
    return plans[best], diag
