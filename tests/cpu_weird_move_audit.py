"""CPU「変な手」監査ツール（Phase 0 物差し・dev 専用・docs/reports/cpu_weird_move_remediation_plan §4）。

決定論的 CPU 対 CPU 自己対戦を N 局回し、**各意思決定**について「変な手」を再現可能な数値へ落とす。
以降の Phase（評価変更・パッチ撤去・学習）の全合否判定の基準線（ベースライン）になる最重要成果物。

これは出荷物（`opcg_sim/`）の挙動を一切変えない**観測専用**ツール（`tests/` dev 専用）。手選択は本番と
同一の `cpu_ai.decide_guarded`（plan=None・hard）で行い、監査は「選ばれた手」を**そのトレースから**採点する
だけ（trace は RNG save/restore で囲われ手選択を変えない＝観測専用。同じ手が選ばれることは
test_cpu_replay.py で担保）。

== 2 つの regret 指標（食い違いが診断の核） ==
本ツールは「行動が do-nothing をどれだけ上回るか」を**異なる方法で 2 通り**測り、その差から症状を切り分ける。

  (A) **per-move settle regret**（null-move regret） = eval_settled(選択手) − eval_settled(TURN_END)。
      両辺とも `_settle_eval` 経由＝戦闘解決後（相手 MAIN の静止点）で採点。**注意（重要）**: `_settle_eval`
      は手の直後にターンを畳むため、**ATTACH_DON / PLAY のような「後続の同ターン攻撃で回収される準備手」を
      構造的に過小評価する**（付与しても攻撃せずターンが畳まれる＝付与価値が回収されない＝regret≈−W_DON_ACTIVE
      で偽陽性になる）。攻撃・カウンターは settle が自分でその手の結果を解決するため正しい。
      ＝**③無駄ドン/①差≤0 の絶対数は上限（over-count）であり、信頼できる頭出しは②自殺攻撃と④届かない
      カウンター。準備手（setup）は (B) AI探索 regret 軸で見るべき**。
  (B) **AI探索 regret**（search regret） = deep(選択手) − deep(TURN_END)。＝**探索本体**が多 ply 先読みで
      「その手は何もしないをどれだけ上回ると判断したか」。探索が準備手の同ターン回収まで読むため (A) の
      truncation を免れる。trace の未切り詰め deep スコアから純粋な引き算で回収＝追加コスト無し。

測る「変な手」カテゴリ（計画 §4 / §1 の3症状・(A) settle regret ベース）:
  ① null-move regret ≤ 0 なのに行動した（diff ≤ 0 なのに do-nothing より良くない手を選んだ）。
     TURN_END が合法に無い局面（防御応答中・対象選択中など）は regret 定義不能＝スキップ（誤検出しない）。
  ② 自殺攻撃: 攻撃で自軍をレストに晒すが、その手の null-move regret ≤ 0（KO もトレードも割に合わない）。
  ③ 無駄ドン: ドン付与/ドン起動効果（ATTACH_DON）だが null-move regret ≤ 0（正味の盤面改善が無い）。
     ※上記のとおり setup truncation で偽陽性を含む＝上限値。
  ④ 届かないカウンター: カウンターステップでカウンター札を切った（`counter_buff>0`）のに、最終的に
     PASS で攻撃が通った（`cpu_ai._counter_needed` が依然 >0＝必要パワーに届かず守れなかった）。
     SELECT_COUNTER は札を 1 枚ずつ積む途中手なので、**カウンターステップを締める PASS** で「積んだ札が
     結局無駄になった戦闘」を 1 件と数える（途中の単票では計上しない＝積み増し中を誤検出しない）。

第2軸（2 指標の食い違いで「変な手」候補をさらに切り分け＝(A)(B) の交差分類）:
  - `search_dispreferred`: **AI探索 regret ≤ 0**（探索自身が TURN_END 以下と評価したのに打った）。
    ＝探索/タイブレーク/畳み判定のバグの疑い（評価ではなく**探索の選び方**がおかしい）。
  - `eval_gap`: **AI探索 regret > 0 かつ settle regret ≤ 0**（探索は好むが settle が回収しない）。
    ＝value-realization gap か準備手 truncation（評価/ホライズンのギャップ）。
  この 2 軸は ①〜③ の候補（自分のメイン手で両 regret が取れた手）を母集団に集計する。

結果ラベル: フラグした手について、その手以降を**既定方策（同一 decide_guarded）で最後までプレイ**し、
実際にフラグした側が敗北へ繋がったかを記録する（`--label`）。コストが高いのでフラグ手のサンプリング率
（`--label-rate`）で制御し、出力にサンプル数を明記する。

出力は「カテゴリ別の件数／100局」＋「2 軸（search_dispreferred / eval_gap）の件数／100局」を中心に、
決定論で再現可能な集計レポート（seed 固定・件数・代表局面の特定情報＝turn/手番/手の説明・両 regret）。

実行例:
    OPCG_LOG_SILENT=1 python tests/cpu_weird_move_audit.py --games 100 --seed 0
    OPCG_LOG_SILENT=1 python tests/cpu_weird_move_audit.py --games 20 --seed 0 --label --label-rate 0.25
    OPCG_LOG_SILENT=1 python tests/cpu_weird_move_audit.py --games 100 --seed 0 --json /tmp/audit.json
"""
import argparse
import json
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary
from opcg_sim.src.models.enums import Phase

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS, InvariantError

# 「行動が do-nothing を上回る」と見なす最小マージン。`cpu_ai._EPS`（≈1.0）と同じく丸め誤差を無視するための
# わずかな閾値。regret <= _NEUTRAL_EPS を「差≤0（変な手）」と数える（ちょうど 0＝同値も含む）。
_NEUTRAL_EPS = 1.0

CATEGORIES = ("neutral_or_worse", "suicidal_attack", "wasted_don", "unreachable_counter")
# 第2軸（AI探索 regret で「変な手」候補をさらに切り分ける・(A)(B) の交差分類）。
SECONDARY_AXES = ("search_dispreferred", "eval_gap")


def _exposes_attacker(manager, actor_name: str, move: Dict[str, Any]) -> bool:
    """この ATTACK 手が自軍の攻撃体をレストに晒すか（リーダー攻撃でない＝キャラがレストになる）。

    OPCG ではキャラのアタックで攻撃体がレストになる（次の相手ターンに無防備で晒される）。リーダー自身の
    アタックはレストにならない（晒すリスクが小さい）ため自殺攻撃の対象外。攻撃体が見つからない/リーダー
    なら False。
    """
    payload = move.get("payload") or {}
    uuid = payload.get("uuid") or move.get("card_uuid")
    if not uuid:
        return False
    actor = cpu_ai._player_by_name(manager, actor_name)
    leader = getattr(actor, "leader", None)
    if leader is not None and getattr(leader, "uuid", None) == uuid:
        return False  # リーダー攻撃はレストに晒さない
    return any(getattr(c, "uuid", None) == uuid for c in actor.field)


def classify_decision(manager, actor_name: str, move: Dict[str, Any],
                      moves: List[Dict[str, Any]],
                      search_regret: Optional[float] = None) -> Dict[str, Any]:
    """1 意思決定を採点し、該当する「変な手」カテゴリ（複数可）と第2軸を返す（観測専用）。

    返り値: ``{"flags": set[str], "axes": set[str], "regret": float|None,
               "search_regret": float|None, "action_type": str, ...}``。
    - `regret`（= (A) per-move settle regret）は本番 decide と同一条件（plan=None・hard＝see_opp_hand=True）
      で算出する。①〜④はこの settle regret ベース。
    - `search_regret`（= (B) AI探索 regret = deep(選択手)−deep(TURN_END)）は呼び出し側が**選択手のトレース**
      から取り出して渡す（手選択を変えない観測値）。`None`（取れない局面/単一手/easy）なら第2軸は付かない。
      第2軸（settle regret が取れた ①②③ 候補にのみ付与）:
        search_dispreferred = AI探索 regret ≤ 0（探索自身が TURN_END 以下と評価したのに打った＝探索バグ疑い）
        eval_gap            = AI探索 regret > 0 かつ settle regret ≤ 0（探索は好むが settle が回収しない
                              ＝value-realization gap / 準備手 truncation）
    """
    action_type = move.get("action_type")
    rec: Dict[str, Any] = {"action_type": action_type, "flags": set(), "axes": set(),
                           "regret": None, "search_regret": search_regret, "has_turn_end": False}

    # ④ 届かないカウンター: カウンターステップを締める PASS で、カウンター札を積んだ（counter_buff>0）のに
    # 攻撃が依然通る（_counter_needed>0）＝積んだ札が無駄だった戦闘を 1 件と数える。SELECT_COUNTER 単体は
    # 札を 1 枚ずつ積む途中手なので計上しない（積み増し中の誤検出を防ぐ）。
    if action_type == "SELECT_COUNTER":
        return rec  # 積み増し途中＝判定保留（締めの PASS で判定する）
    if action_type == "PASS" and getattr(manager, "phase", None) == Phase.COUNTER_STEP:
        ab = getattr(manager, "active_battle", None)
        spent = float((ab or {}).get("counter_buff", 0) or 0)
        needed = cpu_ai._counter_needed(manager)
        if spent > 0 and needed is not None and needed > 0:
            rec["flags"].add("unreachable_counter")
        return rec

    # ①②③ は「自分のメイン手で TURN_END を基準に測れる」局面のみ。null-move regret を算出。
    nmr = cpu_ai.null_move_regret(manager, actor_name, move, moves=moves,
                                  see_opp_hand=True, profile=None, plan=None)
    if nmr is None:
        return rec  # TURN_END 基準が無い局面（防御応答中・対象選択中）＝regret 定義不能＝スキップ
    rec["has_turn_end"] = True
    regret = nmr["regret"]
    rec["regret"] = regret
    rec["chosen_settled"] = nmr["chosen_settled"]
    rec["end_settled"] = nmr["end_settled"]

    if action_type == "TURN_END":
        return rec  # do-nothing そのもの＝基準＝変な手ではない

    neutral = regret <= _NEUTRAL_EPS
    if neutral:
        rec["flags"].add("neutral_or_worse")          # ① 差≤0 なのに行動を選んだ
        if action_type == "ATTACK" and _exposes_attacker(manager, actor_name, move):
            rec["flags"].add("suicidal_attack")        # ② 自殺攻撃（晒すのに割に合わない）
        if action_type == "ATTACH_DON":
            rec["flags"].add("wasted_don")             # ③ 無駄ドン（正味改善が無い）

    # 第2軸（AI探索 regret で切り分け）: 行動手（非 TURN_END）で search_regret が取れたときのみ付与。
    # settle と探索の食い違いを「探索バグ（≤0 を打った）」と「評価ギャップ（探索は好むが settle 未回収）」に分ける。
    if search_regret is not None:
        if search_regret <= _NEUTRAL_EPS:
            rec["axes"].add("search_dispreferred")    # 探索自身が TURN_END 以下と評価したのに打った
        elif neutral:                                  # 探索 regret>0 だが settle regret≤0
            rec["axes"].add("eval_gap")               # value-realization gap / 準備手 truncation
    return rec


def _decide_traced(manager, actor, difficulty, rng, mem) -> tuple:
    """本番同一の `decide_guarded` で手を選びつつ、その**選択手のトレース**から AI探索 regret を回収する。

    手選択の決定論を変えないために `trace` 経路（RNG save/restore で囲われ観測専用）を使う。`trace_read_ahead`
    は重い読み筋（PV）クローンなので監査では不要＝False（候補スコア・regret はクローン少回で取れる）。
    返り値: ``(move, search_regret|None)``。search_regret は ``trace["search_regret"]``＝deep(選択手)−deep(TURN_END)
    （単一手/easy/TURN_END が候補に無い局面では None）。
    """
    trace: Dict[str, Any] = {}
    move = cpu_ai.decide_guarded(manager, actor, difficulty, rng, mem,
                                 trace=trace, trace_read_ahead=False)
    return move, trace.get("search_regret")


def _play_to_finish(manager, mem, difficulty_of, max_steps) -> Optional[str]:
    """`manager`（クローン）を既定方策（decide_guarded・本番同一）で最後まで進め勝者を返す（結果ラベル用）。

    決着しない/例外は None（不明扱い）。`mem` はクローン側の独立コピーを使う（live を汚さない）。
    """
    pending_props = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {})
    pid_key = pending_props.get("PLAYER_ID", "player_id")
    step = 0
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            return None
        req_pid = pending[pid_key]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        moves = manager.get_legal_actions(actor)
        if not moves:
            return None
        move = cpu_ai.decide_guarded(manager, actor, difficulty_of[req_pid], random,
                                     mem.setdefault(req_pid, {}))
        if move is None:
            return None
        manager.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        except Exception:
            return None
        step += 1
    return manager.winner


def audit_one_game(seed: int, db, p1_leader=None, p2_leader=None,
                   difficulty: str = "hard", max_steps: int = DEFAULT_MAX_STEPS,
                   label: bool = False, label_rate: float = 1.0,
                   label_max_steps: int = 2000, rng_label=None) -> Dict[str, Any]:
    """1 局を決定論再生し、各意思決定を採点して「変な手」を集計する（本番同一の手選択）。

    結果ラベル（`label=True`）: フラグした手について、その時点のクローンを既定方策で最後まで進め、
    フラグした側がその局で敗北したかを記録する（`label_rate` でサンプリング）。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1", p1_leader)
    l2, c2 = build_deck(db, "p2", p2_leader)
    if not l1 or not l2:
        raise RuntimeError("リーダーを含むデッキを構築できませんでした。")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()

    difficulty_of = {"p1": difficulty, "p2": difficulty}
    mem: Dict[str, Dict[str, Any]] = {"p1": {}, "p2": {}}

    counts = {cat: 0 for cat in CATEGORIES}
    axis_counts = {ax: 0 for ax in SECONDARY_AXES}
    # ②自殺攻撃の第2軸内訳（探索バグ search_dispreferred か評価ギャップ eval_gap か）。
    suicidal_axis = {ax: 0 for ax in SECONDARY_AXES}
    decisions = 0
    examples: List[Dict[str, Any]] = []
    label_stats = {cat: {"sampled": 0, "lost": 0} for cat in CATEGORIES}

    pending_props = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {})
    pid_key = pending_props.get("PLAYER_ID", "player_id")
    step = 0
    prev_turn = manager.turn_count

    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            raise InvariantError([("STUCK", "no pending request and no winner")], step, [])
        req_pid = pending[pid_key]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        moves = manager.get_legal_actions(actor)
        if not moves:
            raise InvariantError([("NO_LEGAL_MOVE", f"no legal moves for {req_pid}")], step, [])

        # 本番同一の手選択をしつつ、選択手のトレースから AI探索 regret を回収する（trace は観測専用＝
        # RNG save/restore で囲われ手選択を変えない。同じ手が選ばれることは test_cpu_replay.py で担保）。
        move, search_regret = _decide_traced(manager, actor, difficulty_of[req_pid], random,
                                             mem.setdefault(req_pid, {}))
        if move is None:
            raise InvariantError([("NO_DECISION", f"decide returned None for {req_pid}")], step, [])

        # --- 採点（観測専用・live を変えない） ---
        rec = classify_decision(manager, req_pid, move, moves, search_regret=search_regret)
        if rec["regret"] is not None:
            decisions += 1
        flags = rec["flags"]
        axes = rec["axes"]
        # 第2軸（search_dispreferred/eval_gap）は ①〜③ 候補（settle regret が取れた行動手）に付く。
        # フラグの有無に依らず、行動手で search_regret が取れた手を母集団に集計する。
        for ax in axes:
            axis_counts[ax] += 1
        if "suicidal_attack" in flags:
            for ax in axes:
                suicidal_axis[ax] += 1
        if flags:
            for cat in flags:
                counts[cat] += 1
            ex = {
                "seed": seed, "turn": manager.turn_count, "player": req_pid,
                "move": cpu_ai._describe_move(manager, move),
                "flags": sorted(flags),
                "axes": sorted(axes),
                "regret": None if rec["regret"] is None else round(rec["regret"], 1),
                "search_regret": None if rec["search_regret"] is None else round(rec["search_regret"], 1),
            }
            # 結果ラベル: クローンを既定方策で最後まで進め、フラグした側が敗北したかを記録（サンプリング）。
            if label and (rng_label is None or rng_label.random() < label_rate):
                lab_mem = {"p1": {}, "p2": {}}
                outcome = _play_to_finish(manager.clone(), lab_mem, difficulty_of, label_max_steps)
                lost = (outcome is not None and outcome != req_pid)
                ex["label_outcome"] = outcome
                ex["label_lost"] = lost
                for cat in flags:
                    label_stats[cat]["sampled"] += 1
                    if lost:
                        label_stats[cat]["lost"] += 1
            if len(examples) < 12:
                examples.append(ex)

        # --- 手を実際に適用して進行 ---
        manager.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        except Exception as e:
            raise InvariantError(
                [("ACTION_EXCEPTION", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")], step, [])

        violations = check_invariants(manager)
        if manager.turn_count != prev_turn:
            violations += check_turn_boundary(manager)
            prev_turn = manager.turn_count
        if violations:
            raise InvariantError(violations, step, [])
        step += 1

    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", f"game did not finish within {max_steps} steps")], step, [])

    return {
        "seed": seed, "winner": manager.winner, "turns": manager.turn_count,
        "decisions_scored": decisions, "counts": counts, "axis_counts": axis_counts,
        "suicidal_axis": suicidal_axis, "examples": examples,
        "label_stats": label_stats,
        "p1_leader": l1.master.card_id, "p2_leader": l2.master.card_id,
    }


def _build_summary(games, finished, seed, difficulty, total_decisions, total_counts,
                   total_axis_counts, total_suicidal_axis, label_stats, examples, failures) -> Dict[str, Any]:
    per100 = {cat: (100.0 * total_counts[cat] / finished if finished else 0.0) for cat in CATEGORIES}
    axis_per100 = {ax: (100.0 * total_axis_counts[ax] / finished if finished else 0.0)
                   for ax in SECONDARY_AXES}
    return {
        "games_requested": games, "games_finished": finished, "seed": seed,
        "difficulty": difficulty, "decisions_scored": total_decisions,
        "counts": total_counts, "per_100_games": per100,
        "axis_counts": total_axis_counts, "axis_per_100_games": axis_per100,
        "suicidal_axis": total_suicidal_axis,
        "label_stats": label_stats, "examples": examples,
        "failures": [(s, str(e.violations)) for s, e in failures],
    }


def run_audit(games: int, seed: int, db, difficulty="hard", max_steps=DEFAULT_MAX_STEPS,
              label=False, label_rate=1.0, label_max_steps=2000,
              progress=False, json_path=None) -> Dict[str, Any]:
    """N 局の監査を回し、カテゴリ別の総件数・件数/100局・代表局面・結果ラベルを集約する。

    `progress=True` で 1 局完了ごとに途中集計を1行出力し、`json_path` 指定時は**毎局フラッシュ**する
    （hard 自己対戦は重く 1 局 ~30-40s＝長時間ランがタイムアウトしても途中までの集計を回収できるようにする）。
    """
    rng_label = random.Random(seed ^ 0x5151) if label else None
    total_counts = {cat: 0 for cat in CATEGORIES}
    total_axis_counts = {ax: 0 for ax in SECONDARY_AXES}
    total_suicidal_axis = {ax: 0 for ax in SECONDARY_AXES}
    total_decisions = 0
    examples: List[Dict[str, Any]] = []
    label_stats = {cat: {"sampled": 0, "lost": 0} for cat in CATEGORIES}
    finished = 0
    failures: List[tuple] = []

    def _flush_json():
        if json_path:
            summ = _build_summary(games, finished, seed, difficulty, total_decisions, total_counts,
                                  total_axis_counts, total_suicidal_axis, label_stats, examples, failures)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summ, f, ensure_ascii=False, indent=2,
                          default=lambda o: sorted(o) if isinstance(o, set) else str(o))

    for i in range(games):
        gseed = seed + i
        try:
            res = audit_one_game(gseed, db, difficulty=difficulty, max_steps=max_steps,
                                 label=label, label_rate=label_rate,
                                 label_max_steps=label_max_steps, rng_label=rng_label)
        except InvariantError as e:
            failures.append((gseed, e))
            print(f"audit seed={gseed}: FAILED step={e.step} violations={e.violations}", flush=True)
            continue
        finished += 1
        total_decisions += res["decisions_scored"]
        for cat in CATEGORIES:
            total_counts[cat] += res["counts"][cat]
            label_stats[cat]["sampled"] += res["label_stats"][cat]["sampled"]
            label_stats[cat]["lost"] += res["label_stats"][cat]["lost"]
        for ax in SECONDARY_AXES:
            total_axis_counts[ax] += res["axis_counts"][ax]
            total_suicidal_axis[ax] += res["suicidal_axis"][ax]
        for ex in res["examples"]:
            if len(examples) < 30:
                examples.append(ex)
        if progress:
            print(f"[{finished}/{games}] seed={gseed} cumulative counts="
                  f"{[total_counts[c] for c in CATEGORIES]} (順: ①②③④) "
                  f"axes={[total_axis_counts[a] for a in SECONDARY_AXES]} (順: 探索dis/eval_gap)", flush=True)
        _flush_json()

    return _build_summary(games, finished, seed, difficulty, total_decisions, total_counts,
                          total_axis_counts, total_suicidal_axis, label_stats, examples, failures)


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 72)
    print("CPU 変な手 監査レポート（Phase 0 ベースライン）")
    print("=" * 72)
    print(f"seed={summary['seed']}  games_finished={summary['games_finished']}/"
          f"{summary['games_requested']}  difficulty={summary['difficulty']}")
    print(f"decisions_scored (null-move regret 算出できた意思決定数)={summary['decisions_scored']}")
    print("-" * 72)
    print(f"{'カテゴリ':<22}{'総件数':>8}{'件数/100局':>14}")
    labels = {
        "neutral_or_worse": "①差≤0で行動",
        "suicidal_attack": "②自殺攻撃",
        "wasted_don": "③無駄ドン",
        "unreachable_counter": "④届かないカウンター",
    }
    for cat in CATEGORIES:
        print(f"{labels[cat]:<22}{summary['counts'][cat]:>8}{summary['per_100_games'][cat]:>14.1f}")
    print("-" * 72)
    # 注意書き（per-move settle regret の構造的過小評価＝③/① の絶対数は上限）。
    print("注: (A) per-move settle regret は ATTACH_DON/PLAY 等の準備手を構造的に過小評価する")
    print("    （手の直後にターンを畳むため同ターンの攻撃での回収を読まない）。③無駄ドン/①差≤0 の")
    print("    絶対数は**上限（over-count）**であり、信頼できる頭出しは②自殺攻撃と④届かないカウンター。")
    print("    setup 手は下の (B) AI探索 regret 軸で見ること。")
    print("-" * 72)
    # 第2軸（AI探索 regret = deep(選択手)−deep(TURN_END)）。①〜③候補のうち探索バグ/評価ギャップを切り分ける。
    axis_labels = {
        "search_dispreferred": "AI探索 regret≤0 (探索バグ疑い)",
        "eval_gap": "AI探索>0 & settle≤0 (評価gap)",
    }
    ac = summary.get("axis_counts", {})
    ap100 = summary.get("axis_per_100_games", {})
    print(f"{'第2軸（AI探索 regret）':<34}{'総件数':>8}{'件数/100局':>14}")
    for ax in SECONDARY_AXES:
        print(f"{axis_labels[ax]:<34}{ac.get(ax, 0):>8}{ap100.get(ax, 0.0):>14.1f}")
    # ②自殺攻撃の第2軸内訳（探索バグか評価ギャップかの初期診断）。
    sa = summary.get("suicidal_axis", {})
    if summary["counts"].get("suicidal_attack", 0):
        print(f"  └ ②自殺攻撃の内訳: search_dispreferred={sa.get('search_dispreferred', 0)}  "
              f"eval_gap={sa.get('eval_gap', 0)}  (/ ②計 {summary['counts']['suicidal_attack']})")
    print("-" * 72)
    ls = summary["label_stats"]
    if any(ls[c]["sampled"] for c in CATEGORIES):
        print("結果ラベル（フラグ手以降を既定方策で完走・サンプリング）: 敗北/サンプル")
        for cat in CATEGORIES:
            s = ls[cat]
            if s["sampled"]:
                rate = 100.0 * s["lost"] / s["sampled"]
                print(f"  {labels[cat]:<22}{s['lost']:>4}/{s['sampled']:<4}  ({rate:.0f}% 敗北)")
        print("-" * 72)
    if summary["examples"]:
        print("代表局面（先頭 12 件・決定論で再現可能＝seed/turn/player/move）:")
        for ex in summary["examples"][:12]:
            tail = ""
            if "label_lost" in ex:
                tail = f"  -> {'LOST' if ex['label_lost'] else 'won/other'}"
            sr = ex.get("search_regret")
            axtxt = f" axes={ex['axes']}" if ex.get("axes") else ""
            print(f"  seed={ex['seed']} t{ex['turn']} {ex['player']} "
                  f"{ex['move']} settle_regret={ex['regret']} search_regret={sr} "
                  f"{ex['flags']}{axtxt}{tail}")
    if summary["failures"]:
        print("-" * 72)
        print(f"failures: {len(summary['failures'])} 局 (seed, violations):")
        for s, v in summary["failures"]:
            print(f"  seed={s}: {v}")
    print("=" * 72)


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU 変な手 監査ツール（Phase 0 物差し）")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--label", action="store_true",
                    help="フラグ手以降を既定方策で完走し敗北へ繋がったか記録（高コスト）")
    ap.add_argument("--label-rate", type=float, default=1.0,
                    help="結果ラベルを取るフラグ手のサンプリング率（0..1）")
    ap.add_argument("--label-max-steps", type=int, default=2000,
                    help="結果ラベルの完走 step 上限（超過は不明扱い）")
    ap.add_argument("--json", default=None, help="集計サマリ JSON の出力先（毎局フラッシュ＝途中回収可）")
    ap.add_argument("--progress", action="store_true",
                    help="1 局完了ごとに途中集計を1行出力（長時間ランの進捗確認）")
    args = ap.parse_args(argv)

    db = _load_db()
    summary = run_audit(args.games, args.seed, db, difficulty=args.difficulty,
                        max_steps=args.max_steps, label=args.label,
                        label_rate=args.label_rate, label_max_steps=args.label_max_steps,
                        progress=args.progress, json_path=args.json)
    print_report(summary)
    if args.json:
        print(f"wrote summary -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
