"""CPU「変な手」反実仮想（手の差し替え）測定ツール（dev 専用・観測専用）。

Phase 0 監査（`cpu_weird_move_audit.py`）は各「変な手」以降を既定方策で完走し「フラグ側が負けたか」を
**非ペア**でラベル付けした。結果は本物率≒基準線50%（②自殺56%/n9・①53%/n184・③50%/n135）＝ほぼ無相関
だった。本ツールはこれを**ペア比較**で確証する＝「変な手は実際に勝敗を損ねているか？」を因果的に判定する。

== 反実仮想の設計（差し替え基準・RNG 制御・公平性） ==
決定論自己対戦を seed 範囲で回し、各局で `classify_decision`（監査と同一）がフラグした変な手の**決定点 D**を
捕捉する。各 D で同一状態のクローンを 2 つ作り、片方には**実際に選ばれた手（＝変な手）**を、もう片方には
**「最善の非変な手」**を打ち、それぞれ既定方策（`decide_guarded`）で終局させ、**手番側 P が勝ったか**を測る。

  - baseline 枝: D のクローンに変な手を打って `_play_to_finish` で終局。P が勝てば win=1。
  - 反実仮想枝: D のクローンに「最善の非変な手」を打って終局。P が勝てば win=1。
  - 差し替え基準（「最善の非変な手」）: D の合法手のうち、その手を `classify_decision` に掛けても**いずれの
    変な手カテゴリにもフラグされない**手だけを候補にし、`_eval_move_settled`（相手 MAIN まで整流採点・監査の
    settle 評価と同一）が最大の手を選ぶ。do-nothing（TURN_END）も候補に含むため、「無理に動かない」も対抗手に
    なる。変な手しか非変な手として残らない（全候補がフラグされる）稀なケースはスキップ（差し替え対象なし）。
  - RNG 制御（公平性の担保）: クローンは deepcopy で完全独立（相互汚染なし）。`_play_to_finish` 内の
    `decide_guarded` は global `random` を消費する（タイブレークの `rng.shuffle`）。両枝の終局直前に
    **同一の固定 seed で global `random` を seed** するため、D 以降の downstream RNG 列は両枝で同一になる
    （同じ seed 起点・同じ既定方策）。差は「D で打った手」だけ＝手の差し替えの純効果を測る。

== 集計 ==
全体および**カテゴリ別（②自殺攻撃・③無駄ドン・①差≤0）**に:
  baseline 勝率（P 視点）／反実仮想 勝率／**Δ = CF − baseline**（差し替えで勝率がどれだけ上がるか）／標本数。
Δが有意に>0 なら「変な手は損＝直す価値あり」、Δ≒0 なら「無害＝強化目的では直す意味薄い（見栄え目的のみ）」。

== stall 防止 ==
前景でチャンク実行する（バックグラウンド化しない）。1 D あたり 2 終局ロールアウト＝重いので:
  - `--max-flags-per-game` で 1 局あたり捕捉する変な手を先頭 N 件に絞る（②自殺は希少なので優先サンプル）。
  - `--json` 指定時は **1 局ごとにフラッシュ**＝タイムアウトしても途中集計を回収できる。
  - 終わった分の数値＋標本数を必ず報告できる構造（部分でも可）。

実行例:
    OPCG_LOG_SILENT=1 python tests/cpu_counterfactual.py --games 20 --seed 0 \\
        --max-flags-per-game 3 --json /tmp/cf.json --progress
"""
import argparse
import json
import random
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS, InvariantError
from cpu_weird_move_audit import classify_decision, _play_to_finish

# 反実仮想で扱う「変な手」カテゴリ（④届かないカウンターは締めの PASS で判定する戦闘単位の指標で、
# 「同じ状態から別の手を打って終局」というペア差し替えの枠に乗りにくいため対象外＝本ツールは①②③に集中）。
CF_CATEGORIES = ("neutral_or_worse", "suicidal_attack", "wasted_don")

# Δ を測る母集団タグ。"all" は①②③いずれかでフラグされた全決定点。
AGG_TAGS = ("all",) + CF_CATEGORIES


def _apply_move(manager, actor, move) -> bool:
    """move を manager に実適用する（battle/game を振り分け）。例外は False。"""
    manager.action_events = []
    try:
        if move["kind"] == "battle":
            action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
    except Exception:
        return False
    return True


def _best_non_weird_move(manager, actor_name: str, chosen: Dict[str, Any],
                         moves: List[Dict[str, Any]], info_policy: str) -> Optional[Dict[str, Any]]:
    """D の合法手から「最善の非変な手」を返す（差し替え対抗手）。

    候補 = `classify_decision` でいずれの変な手カテゴリ（①②③）にもフラグされない手（TURN_END=do-nothing も
    含む）。その中で `_eval_move_settled`（相手 MAIN まで整流採点）が最大の手を選ぶ。非変な手が無い／採点不能
    のみなら None（差し替え対象なし＝スキップ）。観測専用（クローン上採点・live を変えない）。
    """
    see = (info_policy != "fair")
    best_move = None
    best_score = None
    for m in moves:
        rec = classify_decision(manager, actor_name, m, moves, search_regret=None, see_opp_hand=see)
        if rec["flags"] & set(CF_CATEGORIES):
            continue  # 変な手＝対抗手にしない（do-nothing 基準も含めて非変な手のみ）
        score = cpu_ai._eval_move_settled(manager, actor_name, m, see, None, None)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_move = m
    return best_move


def _rollout_after(state_clone, actor_name: str, move: Dict[str, Any],
                   difficulty_of: Dict[str, str], roll_seed: int,
                   max_steps: int, info_policy: str) -> Optional[str]:
    """D のクローンに `move` を打ってから既定方策で終局させ勝者名を返す（不明/未決着は None）。

    公平性: 終局直前に **同一の固定 seed (`roll_seed`) で global `random` を seed** する。両枝で同じ
    `roll_seed` を渡すため、`move` 適用後の downstream RNG（タイブレーク等）が両枝で一致する。
    """
    actor = cpu_ai._player_by_name(state_clone, actor_name)
    if not _apply_move(state_clone, actor, move):
        return None
    # downstream RNG を両枝で揃える（move 適用は決定論なので適用後に seed すれば D 以降が同一列になる）。
    random.seed(roll_seed)
    mem = {"p1": {}, "p2": {}}
    return _play_to_finish(state_clone, mem, difficulty_of, max_steps, info_policy=info_policy)


def counterfactual_one_game(seed: int, db, difficulty: str = "hard",
                            max_steps: int = DEFAULT_MAX_STEPS,
                            label_max_steps: int = 2000,
                            max_flags_per_game: int = 3,
                            prioritize_suicidal: bool = True,
                            info_policy: str = "hard") -> Dict[str, Any]:
    """1 局を決定論再生し、フラグされた変な手の決定点でペア反実仮想を測る。

    各決定点 D（先頭 `max_flags_per_game` 件・`prioritize_suicidal` で②を優先採取）で:
      baseline（変な手で終局・P 勝ち?）／CF（最善の非変な手で終局・P 勝ち?）を記録。
    返り値: 各 D の結果リスト + 集計用カウンタ（カテゴリ別 baseline_wins/cf_wins/n）。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1", None)
    l2, c2 = build_deck(db, "p2", None)
    if not l1 or not l2:
        raise RuntimeError("リーダーを含むデッキを構築できませんでした。")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()

    difficulty_of = {"p1": difficulty, "p2": difficulty}
    mem: Dict[str, Dict[str, Any]] = {"p1": {}, "p2": {}}
    see = (info_policy != "fair")

    pending_props = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {})
    pid_key = pending_props.get("PLAYER_ID", "player_id")

    # この局でフラグされた決定点（の捕捉済みデータ）。後でロールアウトする。
    captured: List[Dict[str, Any]] = []
    step = 0

    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            raise InvariantError([("STUCK", "no pending request and no winner")], step, [])
        req_pid = pending[pid_key]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        moves = manager.get_legal_actions(actor)
        if not moves:
            raise InvariantError([("NO_LEGAL_MOVE", f"no legal moves for {req_pid}")], step, [])

        # 本番同一の手選択（決定論・観測専用）。trace は不要（search_regret は本ツールでは使わない）。
        move = cpu_ai.decide_guarded(manager, actor, difficulty_of[req_pid], random,
                                     mem.setdefault(req_pid, {}), info_policy=info_policy)
        if move is None:
            raise InvariantError([("NO_DECISION", f"decide returned None for {req_pid}")], step, [])

        # 採点（観測専用）。①②③のいずれかでフラグされたら決定点 D として捕捉候補。
        rec = classify_decision(manager, req_pid, move, moves, search_regret=None, see_opp_hand=see)
        flags = rec["flags"] & set(CF_CATEGORIES)
        if flags:
            # CF 枝で必要な「最善の非変な手」と、両枝の状態クローンをこの時点で確保する
            # （D 以降に live を進めると同じ状態が作れないため、捕捉時にクローンを取る）。
            cf_move = _best_non_weird_move(manager, req_pid, move, moves, info_policy)
            if cf_move is not None:
                captured.append({
                    "seed": seed, "turn": manager.turn_count, "player": req_pid,
                    "flags": sorted(flags),
                    "regret": None if rec["regret"] is None else round(rec["regret"], 1),
                    "move_desc": cpu_ai._describe_move(manager, move),
                    "cf_desc": cpu_ai._describe_move(manager, cf_move),
                    "weird_move": move,
                    "cf_move": cf_move,
                    "base_clone": manager.clone(),
                    "cf_clone": manager.clone(),
                })

        # live を本物の手で進める。
        if not _apply_move(manager, actor, move):
            raise InvariantError([("ACTION_EXCEPTION", "apply failed")], step, [])
        step += 1

    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", f"game did not finish within {max_steps} steps")], step, [])

    # --- 捕捉した決定点のサンプリング（②自殺を優先・先頭 N 件） ---
    if prioritize_suicidal:
        captured.sort(key=lambda d: 0 if "suicidal_attack" in d["flags"] else 1)
    sampled = captured[:max_flags_per_game] if max_flags_per_game > 0 else captured

    # --- 各 D でペア反実仮想ロールアウト ---
    results: List[Dict[str, Any]] = []
    for idx, d in enumerate(sampled):
        # 公平性: この D の両枝で共有する固定 seed（局 seed と D 位置から決定論的に導出）。
        roll_seed = (seed * 1_000_003 + d["turn"] * 97 + idx) & 0x7FFFFFFF
        base_winner = _rollout_after(d["base_clone"], d["player"], d["weird_move"],
                                     difficulty_of, roll_seed, label_max_steps, info_policy)
        cf_winner = _rollout_after(d["cf_clone"], d["player"], d["cf_move"],
                                   difficulty_of, roll_seed, label_max_steps, info_policy)
        # 手番側 P 視点の勝敗。終局しなかった枝（None）はこの D を計上しない（ペアが揃わないと不公平）。
        if base_winner is None or cf_winner is None:
            results.append({**_strip_clones(d), "base_winner": base_winner, "cf_winner": cf_winner,
                            "counted": False})
            continue
        base_p_win = 1 if base_winner == d["player"] else 0
        cf_p_win = 1 if cf_winner == d["player"] else 0
        results.append({**_strip_clones(d), "base_winner": base_winner, "cf_winner": cf_winner,
                        "base_p_win": base_p_win, "cf_p_win": cf_p_win, "counted": True})

    return {"seed": seed, "winner": manager.winner, "turns": manager.turn_count,
            "n_flagged": len(captured), "results": results,
            "p1_leader": l1.master.card_id, "p2_leader": l2.master.card_id}


def _strip_clones(d: Dict[str, Any]) -> Dict[str, Any]:
    """JSON 出力用に重いクローン/手 dict を除いた記録を返す。"""
    return {k: v for k, v in d.items()
            if k not in ("base_clone", "cf_clone", "weird_move", "cf_move")}


def _empty_agg() -> Dict[str, Dict[str, int]]:
    return {tag: {"n": 0, "base_wins": 0, "cf_wins": 0} for tag in AGG_TAGS}


def _accumulate(agg: Dict[str, Dict[str, int]], res: Dict[str, Any]) -> None:
    """1 D のペア結果を集計へ加算（全体 + フラグ該当カテゴリ別）。終局しなかった枝は計上しない。"""
    if not res.get("counted"):
        return
    bw, cw = res["base_p_win"], res["cf_p_win"]
    tags = ["all"] + [f for f in res["flags"] if f in CF_CATEGORIES]
    for tag in tags:
        agg[tag]["n"] += 1
        agg[tag]["base_wins"] += bw
        agg[tag]["cf_wins"] += cw


def run_counterfactual(games: int, seed: int, db, difficulty="hard",
                       max_steps=DEFAULT_MAX_STEPS, label_max_steps=2000,
                       max_flags_per_game=3, prioritize_suicidal=True,
                       progress=False, json_path=None, info_policy="hard") -> Dict[str, Any]:
    """seed..seed+games-1 で反実仮想を回し、カテゴリ別 baseline/CF 勝率と Δ を集計する。

    `json_path` 指定時は **1 局ごとにフラッシュ**＝長時間ランがタイムアウトしても途中集計を回収できる。
    """
    agg = _empty_agg()
    examples: List[Dict[str, Any]] = []
    failures: List[Tuple[int, Any]] = []
    finished = 0
    pairs_total = 0
    pairs_skipped_nofinish = 0

    def _summary() -> Dict[str, Any]:
        rows = {}
        for tag in AGG_TAGS:
            a = agg[tag]
            n = a["n"]
            base_wr = (a["base_wins"] / n) if n else None
            cf_wr = (a["cf_wins"] / n) if n else None
            delta = (cf_wr - base_wr) if (n and base_wr is not None) else None
            rows[tag] = {"n": n, "base_wins": a["base_wins"], "cf_wins": a["cf_wins"],
                         "base_winrate": base_wr, "cf_winrate": cf_wr, "delta": delta}
        return {"games_requested": games, "games_finished": finished, "seed": seed,
                "difficulty": difficulty, "info_policy": info_policy,
                "max_flags_per_game": max_flags_per_game,
                "prioritize_suicidal": prioritize_suicidal,
                "pairs_total": pairs_total, "pairs_skipped_nofinish": pairs_skipped_nofinish,
                "agg": rows, "examples": examples,
                "failures": [(s, str(e)) for s, e in failures]}

    def _flush():
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(_summary(), f, ensure_ascii=False, indent=2)

    for i in range(games):
        gseed = seed + i
        try:
            res = counterfactual_one_game(
                gseed, db, difficulty=difficulty, max_steps=max_steps,
                label_max_steps=label_max_steps, max_flags_per_game=max_flags_per_game,
                prioritize_suicidal=prioritize_suicidal, info_policy=info_policy)
        except InvariantError as e:
            failures.append((gseed, e))
            print(f"cf seed={gseed}: FAILED {e}", flush=True)
            continue
        except Exception as e:  # noqa: BLE001 (観測専用ツール＝1局の失敗で全体を止めない)
            failures.append((gseed, f"{type(e).__name__}: {e}"))
            print(f"cf seed={gseed}: ERROR {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
            continue
        finished += 1
        for r in res["results"]:
            pairs_total += 1
            if not r.get("counted"):
                pairs_skipped_nofinish += 1
            _accumulate(agg, r)
            if r.get("counted") and len(examples) < 40:
                examples.append({k: r[k] for k in
                                 ("seed", "turn", "player", "flags", "regret",
                                  "move_desc", "cf_desc", "base_p_win", "cf_p_win")})
        if progress:
            a = agg["all"]
            print(f"[{finished}/{games}] seed={gseed} flagged={res['n_flagged']} "
                  f"pairs(all)={a['n']} base_wr={_wr(a)} cf_wr={_wr(a, 'cf_wins')} "
                  f"suicidal_n={agg['suicidal_attack']['n']}", flush=True)
        _flush()

    summ = _summary()
    _flush()
    return summ


def _wr(a: Dict[str, int], key: str = "base_wins") -> str:
    n = a["n"]
    return f"{(a[key] / n):.2f}" if n else "-"


def print_report(summary: Dict[str, Any]) -> None:
    print("=" * 78)
    print("CPU 変な手 反実仮想（手の差し替え）測定レポート")
    print("=" * 78)
    print(f"seed={summary['seed']}  games_finished={summary['games_finished']}/"
          f"{summary['games_requested']}  difficulty={summary['difficulty']}"
          f"  info_policy={summary['info_policy']}")
    print(f"max_flags_per_game={summary['max_flags_per_game']}  "
          f"prioritize_suicidal={summary['prioritize_suicidal']}")
    print(f"ペア総数={summary['pairs_total']}  "
          f"（うち終局せず除外={summary['pairs_skipped_nofinish']}）")
    print("-" * 78)
    labels = {
        "all": "全体（①②③いずれか）",
        "neutral_or_worse": "①差≤0で行動",
        "suicidal_attack": "②自殺攻撃",
        "wasted_don": "③無駄ドン",
    }
    print(f"{'母集団':<22}{'標本n':>7}{'baseline勝率':>14}{'CF勝率':>12}{'Δ=CF-base':>14}")
    for tag in AGG_TAGS:
        r = summary["agg"][tag]
        if r["base_winrate"] is None:
            print(f"{labels[tag]:<22}{r['n']:>7}{'-':>14}{'-':>12}{'-':>14}")
        else:
            print(f"{labels[tag]:<22}{r['n']:>7}{r['base_winrate']:>14.3f}"
                  f"{r['cf_winrate']:>12.3f}{r['delta']:>+14.3f}")
    print("-" * 78)
    print("Δ>0 = 変な手を最善の非変な手へ差し替えると手番側 P の勝率が上がる（＝変な手は損）。")
    print("Δ≒0 = 差し替えても勝率が動かない（＝その変な手は無害・強化目的では直す意味薄い）。")
    print("-" * 78)
    if summary["examples"]:
        print("代表ペア（先頭 15 件・seed/turn/player/手/差し替え手/勝敗 base->cf）:")
        for ex in summary["examples"][:15]:
            print(f"  seed={ex['seed']} t{ex['turn']} {ex['player']} {ex['flags']} "
                  f"regret={ex['regret']}")
            print(f"      weird={ex['move_desc']}  -> cf={ex['cf_desc']}  "
                  f"P_win: base={ex['base_p_win']} cf={ex['cf_p_win']}")
    if summary["failures"]:
        print("-" * 78)
        print(f"failures: {len(summary['failures'])} 局")
        for s, v in summary["failures"][:10]:
            print(f"  seed={s}: {v}")
    print("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU 変な手 反実仮想（手の差し替え）測定")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--label-max-steps", type=int, default=2000,
                    help="ロールアウト完走 step 上限（超過は不明扱い＝そのペアを除外）")
    ap.add_argument("--max-flags-per-game", type=int, default=3,
                    help="1 局あたり反実仮想する変な手の先頭件数（0=全件・重い）")
    ap.add_argument("--no-prioritize-suicidal", action="store_true",
                    help="②自殺攻撃を優先採取しない（既定は②優先＝希少サンプルを取りこぼさない）")
    ap.add_argument("--fair", action="store_true",
                    help="フェア hard で測定（info_policy=fair）。既定 OFF＝従来 cheat（hard）。")
    ap.add_argument("--json", default=None, help="集計サマリ JSON 出力先（1 局ごとフラッシュ＝途中回収可）")
    ap.add_argument("--progress", action="store_true", help="1 局完了ごとに途中集計を1行出力")
    args = ap.parse_args(argv)

    db = _load_db()
    info_policy = "fair" if args.fair else "hard"
    summary = run_counterfactual(
        args.games, args.seed, db, difficulty=args.difficulty, max_steps=args.max_steps,
        label_max_steps=args.label_max_steps, max_flags_per_game=args.max_flags_per_game,
        prioritize_suicidal=not args.no_prioritize_suicidal,
        progress=args.progress, json_path=args.json, info_policy=info_policy)
    print_report(summary)
    if args.json:
        print(f"wrote summary -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
