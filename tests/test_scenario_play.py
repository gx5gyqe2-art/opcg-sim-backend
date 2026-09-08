"""分岐点シナリオ（人間のプレイと比べる道具）のテスト（`docs/rust_engine_plan.md` §20.1・`rs-scenario`）。

対象:
  `opcg_sim/loop/hidden_build.py`（フレーム → 記録 v5 `hidden` の復元）
  `tests/scripts/rs_scenario_play.py`（`play` の出力＝frames／decisions／.md）

Rust の wheel が無い環境では **skip せず fail** する（`make test` のゲートが黙って通らないように・
`tests/harness/rs_golden.py::load_engine` と同じ方針）。
"""
import glob
import gzip
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ。tests/scripts も import 可能にする)

import pytest  # noqa: E402

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

import rs_scenario_play as scp  # noqa: E402  (tests/scripts のベア import)

FIXTURES_DIR = _bootstrap.FIXTURES_DIR
REPLAYS_DIR = _os.path.join(FIXTURES_DIR, "replays")
SCENARIOS_DIR = _os.path.join(FIXTURES_DIR, "scenarios")


def _replay_files() -> list:
    return sorted(glob.glob(_os.path.join(REPLAYS_DIR, "*", "*.json.gz")))


def _load_replay(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _scenario_names() -> list:
    if not _os.path.isdir(SCENARIOS_DIR):
        return []
    return sorted(_os.path.splitext(f)[0] for f in _os.listdir(SCENARIOS_DIR) if f.endswith(".json"))


@pytest.fixture(scope="module")
def engine():
    return load_engine()


# --- (a) frame_to_hidden → from_hidden → board() の往復 ----------------------

def test_frame_to_hidden_roundtrip_matches_frame(engine):
    """復元した `hidden` から作った対局の `board()` が、元のフレームと一致する
    （手札・場・ライフ・トラッシュ・ドン枚数・ターン・手番）。"""
    replay_files = _replay_files()
    assert replay_files, f"サンプルのリプレイが無い: {REPLAYS_DIR}"
    checked = 0
    for path in replay_files:
        payload = _load_replay(path)
        frames = payload.get("frames") or []
        turns = sorted({fr["turn"] for fr in frames if fr.get("turn")})
        for turn in turns:
            try:
                idx = hb.turn_start_index(payload, turn)
                hidden = hb.frame_to_hidden(payload, idx, seed=turn)
            except ValueError:
                # 決着後の残骸フレーム等、復元不能なターンはスキップ（このテストは
                # 「復元できるフレームが正しく往復するか」を見る。復元可否の判定自体は
                # `test_frame_to_hidden_rejects_mid_dialog_frame` 等が別に見る）。
                continue
            game = RsGame.from_hidden(hidden, seed=turn)
            board = game.board()
            frame = frames[idx]

            assert board["turn_info"]["turn_count"] == frame["turn"], (path, turn)
            assert board["turn_info"]["active_player_id"] == frame["active"], (path, turn)

            for seat in ("p1", "p2"):
                b = board["players"][seat]
                fr = frame["players"][seat]
                for zone in ("hand", "field", "life", "trash"):
                    got = [c["uuid"] for c in b["zones"][zone]]
                    want = [c["uuid"] for c in fr[zone]]
                    assert got == want, (path, turn, seat, zone, got, want)
                assert len(b["don_active"]) == fr["don_active"], (path, turn, seat, "don_active")
                assert len(b["don_rested"]) == fr["don_rested"], (path, turn, seat, "don_rested")
                assert b["don_deck_count"] == fr["don_deck"], (path, turn, seat, "don_deck")
                if fr.get("leader"):
                    assert b["leader"]["uuid"] == fr["leader"]["uuid"], (path, turn, seat, "leader uuid")
                    assert b["leader"]["card_id"] == fr["leader"]["card_id"]
                    assert b["leader"]["attached_don"] == fr["leader"]["attached_don"]
            checked += 1
    assert checked >= 10, f"検証できたターン開始フレームが少なすぎる（{checked}）"


# --- (c) 対話・戦闘の途中フレームは拒否 --------------------------------------

def test_frame_to_hidden_rejects_mid_dialog_frame():
    path = _replay_files()[0]
    payload = _load_replay(path)
    frames = payload.get("frames") or []
    mid_idx = next(
        i for i, fr in enumerate(frames)
        if (fr.get("pending") or {}).get("action") not in (None, "MAIN_ACTION"))
    with pytest.raises(ValueError, match="別のターンを"):
        hb.frame_to_hidden(payload, mid_idx, seed=1)


def test_frame_to_hidden_rejects_mid_battle_frame():
    path = _replay_files()[0]
    payload = _load_replay(path)
    frames = payload.get("frames") or []
    battle_idx = next((i for i, fr in enumerate(frames) if fr.get("battle")), None)
    if battle_idx is None:
        pytest.skip("このサンプルには戦闘中のフレームが無い")
    with pytest.raises(ValueError, match="別のターンを"):
        hb.frame_to_hidden(payload, battle_idx, seed=1)


# --- (b) play がフレーム／決定／.md を出す -----------------------------------

def test_play_scenario_produces_frames_decisions_and_markdown(engine):
    names = _scenario_names()
    assert names, f"シナリオが無い: {SCENARIOS_DIR}（`rs_scenario_play.py add` で作る）"
    name = names[0]
    scenario = scp._load_scenario(name)
    replay_path = scp._resolve_replay_source(scenario["source"])
    payload = scp._load_json_maybe_gz(replay_path)

    result, steps_log, frames = scp._play_one(
        name, scenario, payload, scp.DEFAULT_NET, scp.DEFAULT_NET, seed=0, sims=16)

    assert result["success"] is True
    assert result["replay"]["schema"]
    assert len(result["frames"]) >= 2
    assert steps_log, "1手も打たれていない"
    assert result["decisions"], "decisions が空"
    assert any(d.get("candidates") for d in result["decisions"]), "candidates を持つ決定が無い"

    last_frame = result["frames"][-1]
    actions = result["replay"]["actions"]
    ended_by_turn_end = bool(actions) and actions[-1].get("action_type") == "TURN_END" and \
        actions[-1].get("turn") == scenario["end_turn"]
    ended_by_win = last_frame.get("winner") is not None
    assert ended_by_turn_end or ended_by_win, (last_frame, actions[-1] if actions else None)

    human_ref = scp._human_reference(payload, scenario)
    original_frames = payload.get("frames") or []
    card_texts = scp._load_card_texts()
    md = scp._build_markdown(name, scenario, 0, scp.DEFAULT_NET, scp.DEFAULT_NET,
                             steps_log, frames, human_ref, original_frames, card_texts)
    for marker in ("(1) 盤面", "(2) pending", "(3) 合法手", "(4) 探索の候補", "(5) 適用後のイベント",
                  "付録: 登場したカードの効果本文", "シナリオの note"):
        assert marker in md, marker
