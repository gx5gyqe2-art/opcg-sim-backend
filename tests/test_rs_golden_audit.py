"""全カード監査の golden ゲート（Rust エンジンだけで回る・`docs/rust_engine_plan.md` §16.1）。

**問い**: Rust エンジンは、全カードの全能力を汎用盤面で発動して既定応答で解決したとき、
記録した golden（＝Python エンジンが同じ手順で出した盤面とイベントの sha1）と同じ列を出すか。

これが `test_full_card_audit.py`／`test_full_card_baseline.py`（Python エンジンで全カードを
回す 2 本）の置き換え。Rust は `opcg_engine.golden_audit(card_id, trigger, ability_index)` で
盤面の生成（`audit::build_test_state`）から既定解決（`audit::drain_default`）まで自前で辿るので、
このテストに Python エンジンは要らない（3,386 能力で約 6 秒）。

golden の作り直し（**挙動を意図的に変えたときだけ**）:

    make golden-audit    # = rs_audit_replay.py --golden-out tests/fixtures/rs_goldens/audit.json

作り直すと Python 側の記録と Rust の `golden_audit` を 1 件ずつ突き合わせ直す（mismatch=0 が
生成の受け入れ条件）＝golden は「Python と Rust が一致した事実」を焼き付けたものになる。

Rust の wheel が無い環境では **skip せず fail** する（ゲートが黙って通らないように）。
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

from harness.rs_golden import AUDIT_GOLDEN, GOLDEN_VERSION, load_engine  # noqa: E402

# 最初の不一致だけでなく、まとめて数件は見せる（原因の広がりが分かるように）。
_MAX_REPORTED = 5


@pytest.fixture(scope="module")
def golden() -> dict:
    assert _os.path.exists(AUDIT_GOLDEN), (
        f"監査 golden が無い: {AUDIT_GOLDEN}（`make golden-audit` で生成する）"
    )
    with open(AUDIT_GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def test_golden_version_and_record_version_are_current(golden):
    """golden の形式と記録契約が今のコードと合っているか（古い golden で緑にしない）。"""
    engine = load_engine()
    assert golden["version"] == GOLDEN_VERSION, "golden の形式が古い（`make golden-audit`）"
    assert golden["record_version"] == engine.record_version(), (
        "記録契約 RECORD_VERSION が変わっている（`make golden-audit` で作り直す）"
    )


def test_golden_covers_every_ability(golden):
    """収録件数のラチェット（fixture が痩せたら落ちる）。"""
    entries = golden["entries"]
    assert len(entries) >= 3386, f"監査 golden の件数が減っている: {len(entries)}"
    keys = {(e["card_id"], e["ability_index"]) for e in entries}
    assert len(keys) == len(entries), "同じ (card_id, ability_index) が二重に入っている"


def test_rust_audit_matches_the_golden(golden):
    """3,386 能力すべてで Rust の段ごとの sha1 と要約が golden と一致する。"""
    engine = load_engine()
    assert hasattr(engine, "golden_audit"), (
        "opcg_engine に golden_audit が無い（wheel が古い＝`make rust-develop`）"
    )
    bad = []
    for entry in golden["entries"]:
        got = json.loads(engine.golden_audit(
            entry["card_id"], entry["trigger"], entry["ability_index"]))
        if got["hashes"] == entry["hashes"] and got["summary"] == entry["summary"]:
            continue
        where = "summary"
        for i, (exp, act) in enumerate(zip(entry["hashes"], got["hashes"])):
            if exp != act:
                where = f"hashes[{i}] {exp[:8]}!={act[:8]}"
                break
        else:
            if len(entry["hashes"]) != len(got["hashes"]):
                where = f"hashes[len {len(entry['hashes'])}!={len(got['hashes'])}]"
        bad.append(f"{entry['card_id']}#{entry['ability_index']}"
                   f"({entry['trigger']}): {where}")
        if len(bad) >= _MAX_REPORTED:
            break
    assert not bad, "監査 golden と食い違う能力:\n  " + "\n  ".join(bad)
