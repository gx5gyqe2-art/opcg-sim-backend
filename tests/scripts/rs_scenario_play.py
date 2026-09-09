"""分岐点シナリオ: 人間のプレイログの一部を CPU に打たせて比べる道具（`docs/rust_engine_plan.md` §20.1）。

旧計器（`legacy/` の `human_replay_divergence.py`／`coach_gate.py`）は使わない。**Rust の
`Game`（`from_hidden`＋`decide` の trace）だけで新規に作る**。API は無い（CLI のみ）。

サブコマンド:
  list <replay.json[.gz]>   … ターンごとの action_index 範囲・手番・その間の手を一覧（下見）
  add  --replay ... --seat ... --start-turn N --end-turn M --title ... --note-file ...
                             … シナリオ JSON（`tests/fixtures/scenarios/<name>.json`）を書く
  play --scenario <name|all> --net <npz> [--opp-net <npz>] --seeds N --sims N --out <dir>
       [--select-rule visits|q_min_n] [--q-min-frac F] [--root-prior-temp T]
                             … 分岐点を復元し、両席を（候補／相手）ネットで end_turn の
                               TURN_END（か決着）まで打つ。frames.json（ビューアの「ファイルを
                               開く」で読める）と .md（Claude が分析するための完全な事実の記録）
                               を書く。

判定は人・分析は Claude（コーディネータ）の役割（§20.1）。`.md` は要約や合否を書かない。
"""
import argparse
import gzip
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_ROOT, _TESTS_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.api.config import REPLAY_SCHEMA  # noqa: E402
from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.api.services.replay import _frame_side  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402
from opcg_sim.loop.engine import DEFAULT_NET  # noqa: E402

FIXTURES_DIR = os.path.join(_TESTS_DIR, "fixtures")
REPLAYS_DIR = os.path.join(FIXTURES_DIR, "replays")
SCENARIOS_DIR = os.path.join(FIXTURES_DIR, "scenarios")
CARDS_PATH = os.path.join(_ROOT, "opcg_sim", "data", "opcg_cards.json")
DEFAULT_OUT_DIR = "/tmp/scenario_out"

#: 1 シナリオの安全上限（無限ループ検出・`opcg_sim/loop/driver.py::DEFAULT_MAX_STEPS` と同じ）。
MAX_STEPS = 400


# --- 入出力ヘルパ ------------------------------------------------------------

def _load_json_maybe_gz(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _resolve_replay_source(source: str) -> str:
    """シナリオの `source`（`tests/fixtures/` からの相対パス）→ 実ファイルパス。"""
    return os.path.join(FIXTURES_DIR, source)


def _relpath_from_fixtures(path: str) -> str:
    path = os.path.abspath(path)
    return os.path.relpath(path, FIXTURES_DIR)


def _load_card_texts() -> dict:
    """`opcg_cards.json` → `{card_id: {"name","text","trigger_text","counter","cost","power"}}`。"""
    with open(CARDS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for c in raw:
        cid = c.get("number")
        if not cid:
            continue
        out[cid] = {
            "name": c.get("name") or "",
            "text": c.get("効果(テキスト)") or "",
            "trigger_text": c.get("効果(トリガー)") or "",
            "counter": c.get("カウンター") or "",
            "cost": c.get("コスト") or "",
            "power": c.get("パワー") or "",
        }
    return out


def _list_scenario_files() -> list:
    if not os.path.isdir(SCENARIOS_DIR):
        return []
    return sorted(f for f in os.listdir(SCENARIOS_DIR) if f.endswith(".json"))


def _load_scenario(name: str) -> dict:
    path = os.path.join(SCENARIOS_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _net_label(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return base[len("nrel_"):] if base.startswith("nrel_") else base


# --- 記述子の整形（人間の手・CPU の手・合法手・候補で共通）------------------

def _fmt_action(a: dict, card_texts: dict) -> str:
    if not a:
        return "(なし)"
    bits = [str(a.get("action_type") or "?")]
    card = a.get("card")
    if card:
        name = (card_texts.get(card) or {}).get("name", "")
        bits.append(f"{card}({name})" if name else str(card))
    if a.get("don_k") is not None:
        bits.append(f"don_k={a['don_k']}")
    if a.get("targets"):
        bits.append("targets=" + ",".join(str(t) for t in a["targets"]))
    if a.get("selected"):
        bits.append("selected=" + ",".join(str(s) for s in a["selected"]))
    if a.get("selected_slots") is not None:
        bits.append(f"slots={a['selected_slots']}")
    if a.get("index") is not None:
        bits.append(f"index={a['index']}")
    if a.get("position"):
        bits.append(f"position={a['position']}")
    if a.get("card_uuid"):
        bits.append(f"uuid={a['card_uuid']}")
    if a.get("accepted") is not None:
        bits.append(f"accepted={a['accepted']}")
    if a.get("payload"):
        bits.append(f"payload={json.dumps(a['payload'], ensure_ascii=False, default=str)}")
    return " ".join(bits)


def _fmt_card(c: dict, card_texts: dict, extra: str = "") -> str:
    info = card_texts.get(c.get("card_id"), {}) or {}
    bits = [f"{c.get('name') or info.get('name', '')}({c.get('card_id')})"]
    if "power" in c:
        bits.append(f"pow={c['power']}")
    if "cost" in c:
        bits.append(f"cost={c['cost']}")
    if info.get("counter"):
        bits.append(f"counter={info['counter']}")
    if c.get("is_rest"):
        bits.append("REST")
    if c.get("attached_don"):
        bits.append(f"don+{c['attached_don']}")
    if c.get("keywords"):
        bits.append("kw=" + ",".join(c["keywords"]))
    if c.get("ability_disabled"):
        bits.append("disabled")
    if c.get("is_frozen"):
        bits.append("frozen")
    if extra:
        bits.append(extra)
    return " ".join(bits)


# --- list --------------------------------------------------------------------

def cmd_list(args) -> int:
    payload = _load_json_maybe_gz(args.replay)
    card_texts = _load_card_texts()
    actions = (payload.get("replay") or {}).get("actions") or []
    turns: dict = {}
    for i, a in enumerate(actions):
        turns.setdefault(a.get("turn"), []).append((i, a))
    for turn in sorted(turns, key=lambda t: (t is None, t)):
        entries = turns[turn]
        first_i = entries[0][0]
        last_i = entries[-1][0]
        active = entries[0][1].get("player")
        print(f"turn={turn} action_index={first_i}..{last_i} active={active}")
        for i, a in entries:
            print(f"    [{i}] src={a.get('src')} {a.get('player')}: {_fmt_action(a, card_texts)}")
    return 0


# --- add -----------------------------------------------------------------

def cmd_add(args) -> int:
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    source_rel = _relpath_from_fixtures(args.replay)
    note = ""
    if args.note_file:
        with open(args.note_file, encoding="utf-8") as f:
            note = f.read().strip()
    name = args.name
    if not name:
        replay_dir = os.path.basename(os.path.dirname(os.path.abspath(args.replay)))
        name = f"{replay_dir}_t{args.start_turn}-{args.end_turn}"
    scenario = {
        "source": source_rel,
        "seat": args.seat,
        "start_turn": args.start_turn,
        "end_turn": args.end_turn,
        "title": args.title,
        "note": note,
        "tags": args.tags or [],
    }
    out_path = os.path.join(SCENARIOS_DIR, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[scenario] wrote {out_path}")
    return 0


# --- play ----------------------------------------------------------------

def _record_frame(game: RsGame, actions: list) -> dict:
    board = game.board()
    players = board.get("players") or {}
    decks = game.deck_counts()
    ab = board.get("active_battle")
    battle = None
    if ab:
        battle = {"attacker_uuid": ab.get("attacker_uuid"), "target_uuid": ab.get("target_uuid")}
    info = board.get("turn_info") or {}
    return {
        "action_index": (len(actions) - 1) if actions else None,
        "turn": info.get("turn_count"),
        "phase": info.get("current_phase"),
        "active": info.get("active_player_id"),
        "winner": info.get("winner"),
        "players": {"p1": _frame_side(players.get("p1") or {}, decks.get("p1", 0)),
                    "p2": _frame_side(players.get("p2") or {}, decks.get("p2", 0))},
        "pending": game.get_pending_request() or None,
        "battle": battle,
    }


def _play_one(name: str, scenario: dict, payload: dict, seat_net: str, opp_net: str,
             seed: int, sims: int, select_rule: str = None, q_min_frac: float = None,
             root_prior_temp: float = None):
    """1 seed ぶんの再生。戻り値 `(frames_json, steps_log, frames)`。

    `select_rule`／`q_min_frac`／`root_prior_temp`（省略可・§20.5）は**両席に同じ設定**で渡す
    （省略＝serve 既定）。各決定の実測時間は `decisions[i]["decide_ms"]` に入る。
    """
    start_idx = hb.turn_start_index(payload, scenario["start_turn"])
    hidden = hb.frame_to_hidden(payload, start_idx, seed)
    seat = scenario["seat"]
    opp = "p2" if seat == "p1" else "p1"
    net_for = {seat: seat_net, opp: opp_net}

    load_engine()
    # 探索乱数の基点（`RsGame.__init__` が global `random` から 1 度だけ引く）を
    # (シナリオ, seed) から決める＝**同じ (シナリオ, seed) なら設定を変えても同じ世界サンプル列**
    # を見る。§20.5 の掃引は「設定だけを変えた比較」なので、基点が毎回変わると差が設定由来か
    # 引き直し由来か分からない（2026-09-08 に判明・それまでの出力は基点が毎回別だった）。
    random.seed(f"{name}:{seed}")
    game = RsGame.from_hidden(hidden, seed=seed)

    actions: list = []
    frames: list = [_record_frame(game, actions)]
    decisions: list = []
    steps_log: list = []
    end_turn = scenario["end_turn"]

    for _ in range(MAX_STEPS):
        if game.winner is not None:
            break
        pending = game.get_pending_request()
        if not pending:
            break
        player_id = pending["player_id"]
        board_before = frames[-1]
        turn_before = game.turn_count
        action_index = len(actions)
        tr: dict = {}
        t0 = time.perf_counter()
        move = game.decide(player_id, trace=tr, net=net_for[player_id], sims=sims,
                           action_index=action_index, select_rule=select_rule,
                           q_min_frac=q_min_frac, root_prior_temp=root_prior_temp)
        decide_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if move is None:
            break
        # V の帰属（§20.4 の 3）: commit 消化（機械実行・探索していない）は計算しない。
        attribution = game.attribution(player_id) if tr.get("kind") != "commit" else None
        desc = game.describe_move(move)
        decisions.append({"turn": turn_before, "player": player_id,
                          "action_index": action_index, **tr, "attribution": attribution,
                          "decide_ms": decide_ms})
        actions.append({"src": "cpu", "turn": turn_before, "player": player_id, **desc})
        game.apply_move(player_id, move)
        events = list(game.action_events)
        frames.append(_record_frame(game, actions))
        steps_log.append({
            "turn": turn_before, "player": player_id, "board_before": board_before,
            "legal_stats": tr.get("legal_stats"), "candidates": tr.get("candidates"),
            "chosen": desc, "value": tr.get("value"), "events": events,
            "kind": tr.get("kind"), "commit_from": tr.get("commit_from"),
            "pv": tr.get("pv"), "attribution": attribution, "decide_ms": decide_ms,
        })
        if desc.get("action_type") == "TURN_END" and turn_before == end_turn:
            break

    replay_src = payload.get("replay") or {}
    result = {
        "success": True,
        "game_id": f"scenario:{name}:s{seed}",
        "replay": {
            "schema": REPLAY_SCHEMA,
            "seed": str(seed),
            "first_player": None,
            "difficulty": "learned",
            "cpu_player_id": None,
            "leaders": {s: ((hidden["players"][s]["leader"] or {}).get("card_id"))
                        for s in ("p1", "p2")},
            "decks": replay_src.get("decks") or {},
            "actions": actions,
        },
        "decisions": decisions,
        "frames": frames,
        "frames_truncated": False,
    }
    return result, steps_log, frames


def _field_entry_turns(*frame_lists) -> dict:
    """カード uuid → 場に初めて確認できたターン（複数のフレーム列をまたいで最小を取る）。"""
    first_turn: dict = {}
    for frames in frame_lists:
        for fr in frames:
            t = fr.get("turn")
            for seat in ("p1", "p2"):
                for c in ((fr.get("players") or {}).get(seat) or {}).get("field", []) or []:
                    uid = c.get("uuid")
                    if uid is None:
                        continue
                    if uid not in first_turn or (t is not None and t < first_turn[uid]):
                        first_turn[uid] = t
    return first_turn


def _human_reference(payload: dict, scenario: dict) -> list:
    """`同じ範囲で人間が打った手`（`replay.actions` から抜く・記述子＋直前のフレーム）。"""
    actions = (payload.get("replay") or {}).get("actions") or []
    frames = payload.get("frames") or []
    seat = scenario["seat"]
    start_t, end_t = scenario["start_turn"], scenario["end_turn"]
    out = []
    for i, a in enumerate(actions):
        if a.get("player") != seat:
            continue
        t = a.get("turn")
        if t is None or t < start_t or t > end_t:
            continue
        frame = frames[i] if i < len(frames) else None
        out.append({"turn": t, "player": seat, "action": a, "frame": frame})
    return out


def _render_board(frame: dict, entry_turns: dict, card_texts: dict) -> str:
    if not frame:
        return "(盤面なし)"
    lines = [f"- turn={frame.get('turn')} phase={frame.get('phase')} "
            f"active={frame.get('active')} winner={frame.get('winner')}"]
    if frame.get("battle"):
        lines.append(f"- battle: {frame['battle']}")
    for seat in ("p1", "p2"):
        s = (frame.get("players") or {}).get(seat) or {}
        lines.append(f"- {seat}:")
        if s.get("leader"):
            lines.append(f"  - leader: {_fmt_card(s['leader'], card_texts)}")
        if s.get("stage"):
            lines.append(f"  - stage: {_fmt_card(s['stage'], card_texts)}")
        field = s.get("field") or []
        lines.append(f"  - field ({len(field)}):")
        for c in field:
            t = entry_turns.get(c.get("uuid"))
            cur = frame.get("turn")
            if t is None:
                extra = "登場T不明"
            elif cur is not None and t < cur:
                extra = f"登場T{t}"
            else:
                extra = f"登場T{t}(このターン)"
            lines.append(f"    - {_fmt_card(c, card_texts, extra)}")
        hand = s.get("hand") or []
        lines.append(f"  - hand ({len(hand)}):")
        for c in hand:
            lines.append(f"    - {_fmt_card(c, card_texts)}")
        lines.append(f"  - life: {len(s.get('life') or [])} 枚")
        trash = s.get("trash") or []
        lines.append(f"  - trash ({len(trash)}): " + (
            ", ".join(_fmt_card(c, card_texts) for c in trash) if trash else "(なし)"))
        lines.append(f"  - don: active={s.get('don_active', 0)} rested={s.get('don_rested', 0)} "
                    f"deck={s.get('don_deck', 0)}")
        lines.append(f"  - deck_count={s.get('deck_count', 0)}")
    return "\n".join(lines)


def _render_pending(pending) -> str:
    if not pending:
        return "(pending なし)"
    return (f"action={pending.get('action')} message={pending.get('message')!r} "
           f"selectable_uuids={pending.get('selectable_uuids')} can_skip={pending.get('can_skip')}")


def _render_candidates(cands, card_texts) -> str:
    if not cands:
        return "(候補なし)"
    return "\n".join(
        f"  - visit%={c.get('visit_pct')} q={c.get('q')} move={_fmt_action(c.get('move') or {}, card_texts)}"
        for c in cands)


def _render_events(events) -> str:
    if not events:
        return "(イベントなし)"
    return "\n".join(f"  - {e}" for e in events)


def _render_legal_stats(stats, card_texts) -> str:
    """探索が見た全候補（P・N・Q）。訪問 0（考えなかった／読む前に切った）は × 印
    （§20.4: 「考えなかった」と「読んで捨てた」を区別する）。
    """
    if not stats:
        return "(候補なし＝window／commit の決定，または探索が空)"
    lines = []
    for c in stats:
        mark = "×" if not c.get("n") else " "
        lines.append(f"  - [{mark}] p={c.get('p')} n={c.get('n')} q={c.get('q')} "
                     f"move={_fmt_action(c.get('move') or {}, card_texts)}")
    return "\n".join(lines)


def _render_pv(pv, card_texts) -> str:
    """PV（主変化・§20.4 の 1）: 探索が想定した「この後の進行」。"""
    if not pv:
        return "(PV なし＝window／commit の決定，または木が空)"
    lines = []
    for i, p in enumerate(pv):
        lines.append(f"  {i + 1}. {p.get('seat')}: {_fmt_action(p.get('move') or {}, card_texts)} "
                     f"(n={p.get('n')} q={p.get('q')})")
    return "\n".join(lines)


def _render_attribution(attribution) -> str:
    """V の帰属（§20.4 の 3）上位 5: 枠を潰したときの ΔV。相関であって理由ではない。"""
    if not attribution:
        return "(なし＝commit の決定，または attribution 未計算)"
    return "\n".join(f"  - slot={a.get('slot')} {a.get('label')} dv={a.get('dv'):+.4f}"
                     for a in attribution)


def _collect_card_ids(*frame_lists) -> set:
    ids: set = set()

    def add_side(side):
        for key in ("leader", "stage"):
            c = (side or {}).get(key)
            if c:
                ids.add(c.get("card_id"))
        for key in ("field", "hand", "life", "trash"):
            for c in (side or {}).get(key) or []:
                ids.add(c.get("card_id"))

    for frames in frame_lists:
        for fr in frames:
            for seat in ("p1", "p2"):
                add_side((fr.get("players") or {}).get(seat))
    ids.discard(None)
    return ids


def _build_markdown(name: str, scenario: dict, seed: int, seat_net: str, opp_net: str,
                    steps_log: list, frames: list, human_ref: list,
                    original_frames: list, card_texts: dict) -> str:
    entry_turns = _field_entry_turns(original_frames, frames)
    lines = [f"# シナリオ `{name}`（seed={seed}）", ""]
    lines.append(f"- source: {scenario.get('source')}")
    lines.append(f"- seat（人間が打っていた席）: {scenario.get('seat')}")
    lines.append(f"- turn: {scenario.get('start_turn')} 〜 {scenario.get('end_turn')}")
    lines.append(f"- title: {scenario.get('title', '')}")
    lines.append(f"- tags: {', '.join(scenario.get('tags') or [])}")
    lines.append(f"- net: seat={seat_net} opp={opp_net}")
    lines.append("")
    lines.append("## シナリオの note（人間の方針・自由記述）")
    lines.append("")
    lines.append(scenario.get("note") or "(note なし)")
    lines.append("")
    lines.append("## 同じ範囲で人間が打った手（参考・このツールが再現するわけではない）")
    lines.append("")
    if not human_ref:
        lines.append("(該当なし)")
    for h in human_ref:
        lines.append(f"### T{h['turn']} {h['player']}: {_fmt_action(h['action'], card_texts)}")
        lines.append("")
        lines.append(_render_board(h["frame"], entry_turns, card_texts))
        lines.append("")
    lines.append("## CPU の決定（このツールが打った手・省略なし）")
    for i, step in enumerate(steps_log):
        lines.append("")
        lines.append(f"### 手 #{i} — turn={step['turn']} player={step['player']} "
                     f"kind={step.get('kind')}")
        if step.get("kind") == "commit":
            # 箱コミットの機械実行（探索していない）: §20.4 の指示どおり 1 行で書く。
            lines.append("")
            lines.append(f"- 決定 #{step.get('commit_from')} で焼き込まれた継続: "
                         f"{_fmt_action(step['chosen'], card_texts)}")
            continue
        lines.append("")
        lines.append("#### (1) 盤面（決定前・両席の全情報）")
        lines.append(_render_board(step["board_before"], entry_turns, card_texts))
        lines.append("")
        lines.append("#### (2) pending")
        lines.append(_render_pending((step["board_before"] or {}).get("pending")))
        lines.append("")
        legal_stats = step.get("legal_stats")
        lines.append(f"#### (3) 探索が見た候補（{len(legal_stats or [])} 個・P/N/Q・訪問0は×）")
        lines.append(_render_legal_stats(legal_stats, card_texts))
        lines.append("")
        lines.append("#### (4) 探索の候補（上位5・等価手マージ後）と選んだ手")
        lines.append(_render_candidates(step.get("candidates"), card_texts))
        lines.append(f"  - 選んだ手: {_fmt_action(step['chosen'], card_texts)} "
                    f"(value={step.get('value')})")
        lines.append("")
        lines.append("#### (6) PV（主変化・CPU が想定したこの後の進行）")
        lines.append(_render_pv(step.get("pv"), card_texts))
        lines.append("")
        lines.append("#### (7) V の帰属（上位5・枠を潰したときの ΔV）")
        lines.append(_render_attribution(step.get("attribution")))
        lines.append("")
        lines.append("#### (5) 適用後のイベント")
        lines.append(_render_events(step["events"]))
    lines.append("")
    lines.append("## 付録: 登場したカードの効果本文")
    all_ids = _collect_card_ids(frames, [h["frame"] for h in human_ref if h.get("frame")])
    for cid in sorted(i for i in all_ids if i):
        info = card_texts.get(cid, {})
        lines.append("")
        lines.append(f"### {cid} {info.get('name', '')}")
        if info.get("trigger_text"):
            lines.append(f"- トリガー: {info['trigger_text']}")
        lines.append(f"- 効果: {info.get('text') or '(なし)'}")
    return "\n".join(lines) + "\n"


def cmd_play(args) -> int:
    names = _list_names_for_play(args.scenario)
    card_texts = _load_card_texts()
    seat_net = args.net or DEFAULT_NET
    opp_net = args.opp_net or seat_net
    net_label = _net_label(seat_net)
    for name in names:
        scenario = _load_scenario(name)
        replay_path = _resolve_replay_source(scenario["source"])
        payload = _load_json_maybe_gz(replay_path)
        human_ref = _human_reference(payload, scenario)
        original_frames = payload.get("frames") or []
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        for seed in range(args.seeds):
            result, steps_log, frames = _play_one(
                name, scenario, payload, seat_net, opp_net, seed, args.sims,
                select_rule=args.select_rule, q_min_frac=args.q_min_frac,
                root_prior_temp=args.root_prior_temp)
            base = f"{net_label}_s{seed}"
            frames_path = os.path.join(out_dir, f"{base}.frames.json")
            with open(frames_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=1, default=str)
                f.write("\n")
            md = _build_markdown(name, scenario, seed, seat_net, opp_net, steps_log, frames,
                                 human_ref, original_frames, card_texts)
            md_path = os.path.join(out_dir, f"{base}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md)
            last = frames[-1] if frames else {}
            print(f"[play] {name} seed={seed}: turn={last.get('turn')} "
                 f"winner={last.get('winner')} steps={len(steps_log)} -> {frames_path}")
    return 0


def _list_names_for_play(spec: str) -> list:
    if spec == "all":
        return [os.path.splitext(f)[0] for f in _list_scenario_files()]
    return [spec]


# --- main ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="ターンごとの action_index・手番・手を一覧（下見）")
    p.add_argument("replay", help="リプレイ JSON（.json/.json.gz）")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("add", help="シナリオ JSON を書く")
    p.add_argument("--replay", required=True)
    p.add_argument("--seat", required=True, choices=("p1", "p2"))
    p.add_argument("--start-turn", type=int, required=True)
    p.add_argument("--end-turn", type=int, required=True)
    p.add_argument("--title", default="")
    p.add_argument("--note-file", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--name", default=None, help="省略時は <リプレイのディレクトリ名>_t<start>-<end>")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("play", help="分岐点を復元し CPU に打たせる")
    p.add_argument("--scenario", required=True, help="シナリオ名（拡張子なし）または 'all'")
    p.add_argument("--net", default=None, help="候補ネット（既定＝出荷既定）")
    p.add_argument("--opp-net", default=None, help="相手席のネット（既定＝--net と同じ）")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--sims", type=int, default=160)
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    # §20.5（WP `rs-search-a`）: 探索の設定。既定（None）＝serve 既定のまま＝1 bit も変わらない。
    p.add_argument("--select-rule", default=None, choices=("visits", "q_min_n"),
                  help="根で出す手の選び方（既定＝visits＝訪問数最多）")
    p.add_argument("--q-min-frac", type=float, default=None,
                  help="q_min_n の訪問下限の割合（既定 0.125＝sims/8）")
    p.add_argument("--root-prior-temp", type=float, default=None,
                  help="根の事前分布を P^(1/t) へ（既定 1.0＝そのまま・2.0 で平坦化）")
    p.set_defaults(func=cmd_play)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
