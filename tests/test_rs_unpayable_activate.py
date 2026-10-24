"""払えないコストの【起動メイン】を合法手に出さない（`docs/search_efficiency_plan.md` §7.46・WP `rs-unpayable-activate`・
`docs/reports/2026-10-24_unpayable_activate.md`）。

生成の局 3461302（生成役 r22・デッキ real）の実際の盤面（`tests/fixtures/unpayable_activate_3461302.json`＝p2 のメインの判断点で
場に光月モモの助 OP16-084「【起動メイン】コスト20以上のこのキャラをトラッシュに置くことができる：…」が居る最初の盤面）で:
  - コスト 5 のモモの助の起動は合法手に出ない（2026-10-24 までは出て、打つと何も起きず、p2 が同じ手番に繰り返して 400 手の上限に当たった）。
  - 同じ盤面でモモの助のコストを 20 にすれば合法手に出る（払えるときは出る）。

必須（ゲームプレイ＝決着しない局）。Rust の wheel が無い環境では skip せず fail する。カードごとの単体と DB 全体の掃き出しは
`rust/opcg_engine/src/rules/tests_unpayable_activate.rs`。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_rs_unpayable_activate.py -q -s -p no:cacheprovider
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.loop import engine as E  # noqa: E402

FIXTURE = _os.path.join(_bootstrap.FIXTURES_DIR, "unpayable_activate_3461302.json")


def _load():
    eng = E.engine()
    with open(FIXTURE, encoding="utf-8") as f:
        body = json.load(f)
    return eng, body


def _activations(game, uuid):
    legal = json.loads(game.legal_json("p2"))
    return [m for m in legal if m["action_type"] == "ACTIVATE_MAIN" and (m.get("payload") or {}).get("uuid") == uuid]


def test_3461302_unpayable_momonosuke_is_not_legal():
    eng, body = _load()
    game = eng.Game.from_hidden(json.dumps(body["hidden"], ensure_ascii=False))
    pend = json.loads(game.pending_json())
    assert pend["player_id"] == "p2" and pend["action"] == "MAIN_ACTION", pend
    assert _activations(game, body["momo_uuid"]) == []
    # 他の手は残っている（合法手の列挙そのものは壊れていない）。
    assert any(m["action_type"] == "TURN_END" for m in json.loads(game.legal_json("p2")))


def test_3461302_momonosuke_with_cost_20_is_legal():
    eng, body = _load()
    hidden = body["hidden"]
    for c in hidden["players"]["p2"]["field"]:
        if c["uuid"] == body["momo_uuid"]:
            c["cost_buff"] = 15  # 元々のコスト 5 → 20
    game = eng.Game.from_hidden(json.dumps(hidden, ensure_ascii=False))
    assert len(_activations(game, body["momo_uuid"])) == 1
