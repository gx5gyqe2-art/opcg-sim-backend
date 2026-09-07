"""実対局の再生 golden ゲート（Rust エンジンだけで回る・`docs/rust_engine_plan.md` §16.1）。

**問い**: Rust エンジンは、記録済みの局（初期盤面＋行動列）を再生したとき、記録した golden
（＝Python エンジンが同じ局を打って出した盤面・合法手・イベントログの sha1）と同じ列を出すか。

これが `tests/scripts/rs_diff_replay.py` の実行結果を毎回 Python エンジンで確認する運用の
置き換え。golden 1 件は「再生に要る最小の入力（`setup.hidden`＋行動列＋シャッフル再同期材料）」
と「期待する sha1」だけを持つ＝Python エンジンは要らない（`opcg_engine.replay()` だけで回る）。

対象は `tests/fixtures/rs_goldens/replay/`（random 150 局・L1 50 局・シード帯は
`docs/rust_engine_plan.md` §16.1 の払い出し）。

golden の作り直し（**挙動を意図的に変えたときだけ**）:

    make golden-replay   # = rs_diff_replay.py --golden-out（random と l1 の両方を打ち直す）

Rust の wheel が無い環境では **skip せず fail** する（ゲートが黙って通らないように）。
"""
import glob
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

from harness.rs_golden import GOLDEN_VERSION, REPLAY_GOLDEN_DIR, load_engine, replay_hashes  # noqa: E402

_MAX_REPORTED = 5


def _golden_files() -> list:
    return sorted(glob.glob(_os.path.join(REPLAY_GOLDEN_DIR, "*.json")))


@pytest.fixture(scope="module")
def golden_games() -> list:
    files = _golden_files()
    assert files, f"再生 golden が無い: {REPLAY_GOLDEN_DIR}（`make golden-replay` で生成する）"
    out = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            out.append((_os.path.basename(path), json.load(f)))
    return out


def test_golden_version_and_record_version_are_current(golden_games):
    engine = load_engine()
    for name, g in golden_games:
        assert g["version"] == GOLDEN_VERSION, f"{name}: golden の形式が古い（`make golden-replay`）"
        assert g["record_version"] == engine.record_version(), (
            f"{name}: 記録契約 RECORD_VERSION が変わっている（`make golden-replay` で作り直す）"
        )


def test_golden_covers_random_and_l1_policies(golden_games):
    """random 150 局・L1 50 局（計画 §16.1 の受け入れ）を下回っていないか。"""
    by_policy: dict = {}
    for _, g in golden_games:
        by_policy.setdefault(g["policy"], []).append(g["seed"])
    assert len(by_policy.get("random", [])) >= 150, (
        f"random の局数が減っている: {len(by_policy.get('random', []))}"
    )
    assert len(by_policy.get("l1", [])) >= 50, (
        f"l1 の局数が減っている: {len(by_policy.get('l1', []))}"
    )
    for policy, seeds in by_policy.items():
        assert len(set(seeds)) == len(seeds), f"{policy}: seed が重複している"


def test_rust_replay_matches_the_golden(golden_games):
    """各局で Rust の再生結果（盤面・合法手・イベント）の sha1 が golden と一致する。"""
    engine = load_engine()
    bad = []
    for name, g in golden_games:
        record = g["input"]
        try:
            out = json.loads(engine.replay(json.dumps(record)))
        except (ValueError, NotImplementedError) as e:
            bad.append(f"{name}: replay failed: {type(e).__name__}: {e}")
            continue
        shuffled = [s.get("shuffled") or [] for s in record["steps"]]
        got = replay_hashes(out.get("states"), out.get("legal"), out.get("events"), shuffled)
        if got == g["hashes"]:
            continue
        where = next((k for k in g["hashes"] if got.get(k) != g["hashes"][k]), "?")
        bad.append(f"{name} (seed={g['seed']} policy={g['policy']}): {where} mismatch")
        if len(bad) >= _MAX_REPORTED:
            break
    assert not bad, "再生 golden と食い違う局:\n  " + "\n  ".join(bad)
