"""ターンプランの列挙・K世界期待値評価・逐次実行（v37②・2026-08-06）。

**計器専用**（serve 配線は 2026-08-25 に削除・純正AZ化）: 本モジュールを import するのは
教師/計器（`plan_dom_gen.py`/`plan_lethal_gen.py`/`plan_cf2_gen.py` 等）のみで、
実対局の decide からは呼ばれない（例外: `move_sig`/`_find_move`＝手の同一性キーの単一定義は
箱コミット実行〔`cpu_learned`・2026-08-26〕も共有する。プラン機構自体は serve 非配線のまま）。

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
import itertools

import numpy as np

from opcg_sim.src.core import cpu_ai
from .mcts import (TURN_QUIESCE_MAX_PLIES, in_battle, in_dialog, quiesce_choice,
                   resolved_branch_values, _turn_owner)

# --- プラン計器のつまみ（旧 config.PLAN_*・純正AZ化 2026-08-25 でここへ移設）--------------
PLAN_WORLDS = 6        # 期待値を取る決定化世界の数（プラン間で共有＝CRN）
PLAN_PROPOSALS = 6     # 提案ロールアウト本数（1本目は argmax・残りは温度サンプル・重複除去）
PLAN_TEMP = 1.0        # 提案サンプリングの温度（0=argmax）
# **平坦な箱は箱にしない**（v39・2026-08-06）: 候補プランのターン末 value がほとんど割れない窓では、
# 出口の差はプランの優劣でなくノイズに近い＝箱化を放棄して呼び出し側（探索）に委ねる閾値。
PLAN_MIN_SPREAD = 0.15
# 構造化提案（2026-08-20 ユーザ設計「プレイするカードの組 × 浮ドンの使い途」）。
PLAN_STRUCT_PROPOSALS = True
PLAN_STRUCT_SETS = 4     # 「出す組」の候補数（空集合を含む）
PLAN_STRUCT_MAX = 8      # 構造化提案の上限本数（組×変種の展開後にこの数で打ち切り）


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


def _battle_box_step(game, mgr, name, value_fn, priors_fn, window_pred=None):
    """戦闘窓を出口 value の箱で1手進める（gen12 の木の箱化と同一規約）。

    `window_pred=in_dialog`（対話箱・P6-a・2026-08-25）で効果対話窓にも同じ規約を使う
    （物差しは呼び出し側が渡す＝対話窓は本体 value・戦闘窓は戦闘出口ヘッド）。"""
    legal = game.legal_actions(mgr)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    vals = resolved_branch_values(game, mgr, name, legal, value_fn, priors_fn,
                                  window_pred=window_pred)
    ok = [i for i, v in enumerate(vals) if v is not None]
    return legal[max(ok, key=lambda i: vals[i])] if ok else legal[0]


def rollout_plan(game, world, name, value_fn, priors_fn, rng, temp=PLAN_TEMP,
                 max_plies=TURN_QUIESCE_MAX_PLIES, battle_value_fn=None,
                 dialog_box=False):
    """1本のターンロールアウトから候補プラン（自分のメイン手の signature 列）を作る。

    メイン手＝priors の温度サンプリング（temp=0 で argmax）。戦闘窓＝箱。TURN_END で終了
    （TURN_END 自体はプランに含めない＝実行側は「プランが尽きたら」木/既定に委ねる）。

    `battle_value_fn`（v41）: 戦闘箱の物差し（None=`value_fn`）。箱の階層と物差しを1対1に保つ。"""
    steps = []
    mgr = world
    bvf = battle_value_fn or value_fn
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != name:
            break
        actor = game.current_player(mgr)
        if actor is None:
            break
        if in_battle(mgr):
            mv = _battle_box_step(game, mgr, actor, bvf, priors_fn)
        elif dialog_box and in_dialog(mgr):
            # 対話箱（P6-a）: 自他を問わず効果対話窓は出口 value 最良で埋め、その sig も
            # プランに含める（scripted_plan の v39 規約と同じ＝実行が別の選択肢を適用しない）
            mv = _battle_box_step(game, mgr, actor, value_fn, priors_fn,
                                  window_pred=in_dialog)
            if mv is None:
                break
            if actor == name:
                steps.append(move_sig(mv))
            nxt = game.apply(mgr, mv, actor)
            if nxt is None:
                break
            mgr = nxt
            continue
        elif actor != name:
            mv = _battle_box_step(game, mgr, actor, bvf, priors_fn)  # 相手の割込は箱
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


def _own_player(manager, name):
    return manager.p1 if getattr(manager.p1, "name", None) == name else manager.p2


def hand_play_sets(manager, name, max_sets=PLAN_STRUCT_SETS):
    """「このターン出すカードの組」の候補（構造化提案の第1軸・ユーザ設計 2026-08-20）。

    人間は「今の手札から、このターンはこれを出して余ったドンは振る」と考える。分岐の
    第1軸は**プレイするカードの組**で、予算はアクティブドン。同名カード（card_id）の
    組は1つに畳む。必ず含める: **空集合**（全ドンを圧力に使う線）と**コスト和最大の組**
    （盤面アドに全振りする線）。残り枠はコスト和の大きい順。

    返り値: [(uuids タプル（コスト降順＝出す順）, コスト和), ...]（先頭は空集合）。"""
    p = _own_player(manager, name)
    budget = len(getattr(p, "don_active", []) or [])
    cards = []
    for c in (getattr(p, "hand", None) or []):
        cost = getattr(getattr(c, "master", None), "cost", None)
        if cost is None or cost > budget:
            continue
        cards.append((c.uuid, int(cost), getattr(c.master, "card_id", c.uuid)))
    # 部分集合の列挙（手札≤10 → 高々1024）。同名の組は初出だけ残す。
    seen, feasible = set(), []
    for r in range(1, len(cards) + 1):
        for combo in itertools.combinations(cards, r):
            cost_sum = sum(c[1] for c in combo)
            if cost_sum > budget:
                continue
            key = tuple(sorted(c[2] for c in combo))
            if key in seen:
                continue
            seen.add(key)
            ordered = tuple(u for u, _, _ in sorted(combo, key=lambda x: -x[1]))
            feasible.append((ordered, cost_sum))
    feasible.sort(key=lambda fc: (-fc[1], len(fc[0])))
    out = [((), 0)]
    for fc in feasible:
        if len(out) >= max_sets:
            break
        out.append(fc)
    return out


def struct_intents(manager, name, max_sets=PLAN_STRUCT_SETS):
    """プレイ組×浮ドンの使い途を intent（抽象方針の列）に展開する（構造化提案の第2軸）。

    正準順序は **登場 → ドン付与 → アタック**（P1: 付与はアタックの前・段3裁定の原則）。
    付与/攻撃の対象は**今アクティブな既存ユニット**のみ＝このターン登場するカードと
    レスト済みには振らない（P1/P2 を生成側で守る。カード固有の例外—レスト時常在や
    相手ターン常在—は policy 提案側が拾う）。浮ドンの変種:
      spread … アクティブなアタッカーへ順繰りに振ってから総攻撃
      leader … リーダーへ全振りしてから総攻撃（圧力線）
      hold   … 振らずに攻撃だけ（温存線）
    返り値: [(label, intent), ...]。intent の要素は ("PLAY", uuid) / ("ATTACH", uuid) /
    ("ATTACK", uuid)。"""
    p = _own_player(manager, name)
    budget = len(getattr(p, "don_active", []) or [])
    attackers = []
    lead = getattr(p, "leader", None)
    if lead is not None and not getattr(lead, "is_rest", False):
        attackers.append(lead.uuid)
    for c in (getattr(p, "field", None) or []):
        if getattr(c, "is_rest", False) or getattr(c, "is_newly_played", False):
            continue
        attackers.append(c.uuid)
    out = []
    for uuids, cost_sum in hand_play_sets(manager, name, max_sets=max_sets):
        spare = budget - cost_sum
        plays = [("PLAY", u) for u in uuids]
        attacks = [("ATTACK", a) for a in attackers]
        variants = []
        if spare > 0 and attackers:
            spread = [("ATTACH", attackers[i % len(attackers)]) for i in range(spare)]
            variants.append(("spread", plays + spread + attacks))
            if lead is not None and len(attackers) > 1:
                to_lead = [("ATTACH", lead.uuid)] * spare
                variants.append(("leader", plays + to_lead + attacks))
        variants.append(("hold", plays + attacks))
        for vname, intent in variants:
            if not intent:
                continue
            out.append((f"struct:c{cost_sum}+don{spare}:{vname}", intent))
    return out


_MAIN_TYPES = {"PLAY", "ATTACK", "ATTACH_DON", "ACTIVATE_MAIN", "TURN_END"}


_ATTACH_TYPES = ("ATTACH_DON", "DON_BOX")


def canonicalize_steps(steps):
    """P1 正準化（pure）: 付与系 sig を**最初の ATTACK の直前**へブロック移動する。

    - 付与（ATTACH_DON/DON_BOX）はドン予算を消費しない手より後に置く理由がなく、
      アタック後の付与はそのターンの攻撃に乗らない（段3裁定 P1）。
    - 付与は効果対話を生まないため、PLAY→RESOLVE 等の隣接ペアを壊さずに移動できる。
    - ATTACK が無ければ原順のまま（動かす根拠がない）。相対順序は保存（安定移動）。"""
    steps = list(steps)
    kinds = [(s[0] if isinstance(s, (list, tuple)) else None) for s in steps]
    if "ATTACK" not in kinds:
        return tuple(steps)
    first_atk = kinds.index("ATTACK")
    attaches = [s for s, k in zip(steps, kinds) if k in _ATTACH_TYPES]
    late = [s for s, k in zip(steps[first_atk:], kinds[first_atk:]) if k in _ATTACH_TYPES]
    if not late:
        return tuple(steps)                  # 既に全付与が攻撃前＝正準
    rest = [s for s, k in zip(steps, kinds) if k not in _ATTACH_TYPES]
    head = [s for s, k in zip(steps[:first_atk], kinds[:first_atk]) if k not in _ATTACH_TYPES]
    tail = rest[len(head):]
    return tuple(head + attaches + tail)


def _attack_candidates(legal, uuid, tgt=None):
    """攻撃者 uuid（と任意の対象 tgt uuid）に合致する ATTACK 手を返す（pure）。"""
    cands = [m for m in legal
             if m.get("action_type") == "ATTACK"
             and (m.get("payload") or {}).get("uuid") == uuid]
    if tgt is not None:
        cands = [m for m in cands
                 if ((m.get("payload") or {}).get("target_ids") or [None])[0] == tgt]
    return cands


def scripted_plan(game, world, name, intent, value_fn, priors_fn,
                  max_plies=TURN_QUIESCE_MAX_PLIES, battle_value_fn=None):
    """intent（抽象方針）を world 上で実現して手の signature 列にする（構造化提案の実現器）。

    自分のメイン窓では intent を先頭から走査して**最初に合法化できる指示**を採る（正準順序を
    保ちつつ、実現できない指示は縮退）。ATTACK の対象は priors の argmax（不在なら先頭＝
    列挙規約により相手リーダー）。対話（効果選択等・メイン手が無い窓）は `quiesce_choice` で
    埋めて**その sig も列に含める**（プラン実行が別の選択肢を適用しない・v39 の規約）。
    メイン窓で何も実現できなくなったら終了（ターンを閉じるのは `execute_plan` 側の規約）。"""
    steps = []
    todo = list(intent)
    mgr = world
    bvf = battle_value_fn or value_fn
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != name:
            break
        actor = game.current_player(mgr)
        if actor is None:
            break
        if in_battle(mgr) or actor != name:
            mv = _battle_box_step(game, mgr, actor, bvf, priors_fn)
        else:
            legal = game.legal_actions(mgr)
            if not legal:
                break
            is_main = any((cpu_ai._describe_move(mgr, m) or {}).get("action_type")
                          in _MAIN_TYPES for m in legal)
            mv = None
            if is_main:
                for i, item in enumerate(todo):
                    kind, uuid = item[0], item[1]
                    if kind == "ATTACK":
                        # 任意の第3要素＝攻撃対象 uuid（V8 リーダー攻撃族・2026-08-22。
                        # 2要素の従来形は無指定＝priors argmax のまま）
                        tgt = item[2] if len(item) > 2 else None
                        cands = _attack_candidates(legal, uuid, tgt)
                        if cands:
                            j = 0
                            if priors_fn is not None and len(cands) > 1:
                                pr = priors_fn(mgr, cands)
                                if pr is not None:
                                    j = int(np.argmax(pr))
                            mv = cands[j]
                    else:
                        at = "PLAY" if kind == "PLAY" else "ATTACH_DON"
                        for m in legal:
                            if m.get("action_type") == at and \
                                    (m.get("payload") or {}).get("uuid") == uuid:
                                mv = m
                                break
                    if mv is not None:
                        del todo[i]
                        break
                if mv is None:
                    break                       # 方針を使い切った/実現不能＝提案はここまで
                steps.append(move_sig(mv))
            else:
                mv = legal[quiesce_choice(mgr, legal, priors_fn)]
                steps.append(move_sig(mv))      # 対話の選択もプランの一部（v39）
        if mv is None:
            break
        nxt = game.apply(mgr, mv, actor)
        if nxt is None:
            break
        mgr = nxt
    return tuple(steps)


def execute_plan(game, world, name, steps, value_fn, priors_fn,
                 max_plies=TURN_QUIESCE_MAX_PLIES, battle_value_fn=None,
                 dialog_box=False):
    """プランを world 上で箱実行し**ターン末の盤面**を返す（world は不変・clone-apply）。

    **serve（プラン読み出し）と教師（プランCF生成）が共有する実行規約の単一の正**（v38）。
    別々に実装すると必ずずれ、教師が「serve が実際に到達しない盤面」を教えることになる
    （v35 の train/serve skew と同型の予防）。

    手が非合法な世界では skip（相手の応手次第で対象が消える等＝その世界でのプランの自然な
    縮退）。プランが尽きたら TURN_END を適用してターンを閉じる（閉じるまでがプラン）。"""
    mgr = world
    idx = 0
    bvf = battle_value_fn or value_fn
    for _ in range(max_plies):
        if game.is_terminal(mgr) or _turn_owner(mgr) != name:
            return mgr                          # 終局／ターンが替わった＝そこが出口
        actor = game.current_player(mgr)
        if in_battle(mgr):
            mv = _battle_box_step(game, mgr, actor, bvf, priors_fn)
        elif dialog_box and in_dialog(mgr):
            # 対話箱（P6-a）: プランに sig があればそれを優先、無ければ出口 value で埋める
            legal = game.legal_actions(mgr)
            mv = _find_move(legal, steps[idx]) if idx < len(steps) else None
            if mv is not None:
                idx += 1
            else:
                mv = _battle_box_step(game, mgr, actor, value_fn, priors_fn,
                                      window_pred=in_dialog)
        elif actor != name:
            mv = _battle_box_step(game, mgr, actor, bvf, priors_fn)
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
                  max_plies=TURN_QUIESCE_MAX_PLIES, exit_value_fn=None,
                  battle_value_fn=None, dialog_box=False):
    """プランを箱実行した**自ターン末の value**（name 視点）。実行は `execute_plan` が正。

    「どの箱の出口か」と「どのヘッドで測るか」を1対1に保つ（v39/v41）:
      - ターン箱の出口＝`exit_value_fn`（ターン末専用ヘッド）
      - 実行途中の戦闘窓＝`battle_value_fn`（戦闘出口専用ヘッド）
      - どちらも None なら `value_fn`＝v39 以前と完全に同値。"""
    if dialog_box:
        exit_mgr = execute_plan(game, world, name, steps, value_fn, priors_fn, max_plies,
                                battle_value_fn=battle_value_fn, dialog_box=True)
    else:
        # 旧署名で呼ぶ＝execute_plan を差し替える既存テスト/計器（test_exit_heads の
        # モック等）と後方互換（dialog_box 既定 False では挙動も呼び出し形も従来どおり）
        exit_mgr = execute_plan(game, world, name, steps, value_fn, priors_fn, max_plies,
                                battle_value_fn=battle_value_fn)
    return (exit_value_fn or value_fn)(exit_mgr, name)


def select_plan(game, manager, name, value_fn, priors_fn, rng,
                n_worlds=PLAN_WORLDS, n_proposals=PLAN_PROPOSALS, exit_value_fn=None,
                min_spread=PLAN_MIN_SPREAD, battle_value_fn=None, dialog_box=False):
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
    plans, labels = [], []

    def _add(steps, label):
        if steps and steps not in plans:
            plans.append(steps)
            labels.append(label)

    for k in range(n_proposals):
        t = 0.0 if k == 0 else PLAN_TEMP
        steps = rollout_plan(game, worlds[k % len(worlds)], name, value_fn, priors_fn,
                             rng, temp=t, battle_value_fn=battle_value_fn,
                             dialog_box=dialog_box)
        # W5（2026-08-22）: policy 提案は現行方策の癖で「攻撃→付与」の誤順（P1違反・段3裁定
        # #1/#2/502006@130）を含みうる。付与を最初の攻撃の前へ正準化した版**も**候補に足す
        # （原順も残す＝正準化で壊れる世界があっても候補集合は狭まらない）。
        canon = canonicalize_steps(steps)
        if canon != steps:
            # 正準版を**先に**追加する: 出口が同一（順序だけの違い＝P8）なら同点になり、
            # argmax は先頭＝正準版を選ぶ（P1 のタイブレーク）。
            _add(canon, "policy:canon" if k == 0 else f"policy:canon:t{PLAN_TEMP:g}")
        _add(steps, "policy:argmax" if k == 0 else f"policy:t{PLAN_TEMP:g}")
    # 構造化提案（プレイ組×浮ドンの使い途・ユーザ設計 2026-08-20）: policy 提案は現行方策の
    # 癖（例: ドン付与への偏り）を引き継ぐため、**正解の型が候補に入らない**ことがある
    # （段3裁定 #1/#2 の実測）。人間の分岐構造で候補を別経路から供給する。
    if PLAN_STRUCT_PROPOSALS:
        for i, (label, intent) in enumerate(struct_intents(manager, name)[:PLAN_STRUCT_MAX]):
            steps = scripted_plan(game, worlds[i % len(worlds)], name, intent,
                                  value_fn, priors_fn, battle_value_fn=battle_value_fn)
            _add(steps, label)
    if not plans:
        return None, {}
    scores = []
    for steps in plans:
        vs = [evaluate_plan(game, w, name, steps, value_fn, priors_fn,
                            exit_value_fn=exit_value_fn,
                            battle_value_fn=battle_value_fn,
                            dialog_box=dialog_box) for w in worlds]
        vs = [v for v in vs if v is not None]
        scores.append(float(np.mean(vs)) if vs else float("-inf"))
    best = int(np.argmax(scores))
    finite = [s for s in scores if s > float("-inf")]
    spread = float(max(finite) - min(finite)) if len(finite) > 1 else 0.0
    diag = {"n_plans": len(plans), "n_worlds": len(worlds),
            "scores": [round(s, 4) for s in scores], "best": best,
            "kinds": labels, "spread": round(spread, 4)}
    if len(finite) > 1 and spread < min_spread:
        return None, {**diag, "skipped": "flat_exits"}
    return plans[best], diag
