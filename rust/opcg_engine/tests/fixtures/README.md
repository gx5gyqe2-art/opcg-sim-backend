# `cargo test` の盤面 fixture（P1-model・`docs/rust_engine_plan.md` §9）

`src/model.rs` のテスト（`hidden_fixture_roundtrips_to_the_python_board_dict` ほか）が読む実データ。
**Python 版が正本**なので、期待値は Python の盤面 dict をそのまま持ってきたもので、Rust 側の出力から
作り直してはいけない。

| ファイル | 中身 |
|---|---|
| `hidden_v2.json` | 記録形式 v2 の `hidden` 1 行（§9.1）＝盤面を完全に再構成できる内部状態。約 51KB |
| `board_v2.json` | 同じ行の `state`（Python の `rs_diff_replay.py::board_dict`）から `pending_request` を除いたもの＝**期待値**。約 26KB |
| `masters_v2.json` | `opcg_effects.json` を「この盤面に出るカード（51 枚）」だけに絞った効果 JSON＝`MasterTable::from_effects_json` の入力。約 122KB |

選んだ行は seed=500000・`--policy random` の 113 行目（0 起点 112・`turn_count=16`・`BATTLE_COUNTER`）。
戦闘中・付与ドン!!あり・表向きライフあり・トラッシュありの行を選んである（退化した盤面だと照合が
素通りになるため、`hidden_fixture_has_the_state_the_board_dict_exercises` がこの性質を固定している）。

## 作り直し方

```bash
python -m opcg_sim.tools.export_effects_json          # opcg_sim/data/opcg_effects.json（生成物）
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py \
  --mode state --games 1 --seed-base 500000 --policy random --dump /tmp/rec.json
```

`/tmp/rec.json` の `[setup] + steps` を 1 本のリストにして 1 行を選び、その `hidden` を
`hidden_v2.json` へ、`state`（`pending_request` を除く）を `board_v2.json` へ、その盤面に出る
`card_id` の分だけ `opcg_effects.json` の `cards` を絞ったものを `masters_v2.json` へ書く
（いずれも `json.dump(..., ensure_ascii=False, sort_keys=True, separators=(",",":"))`）。
`hidden_v2.json` は 100KB 以下に収める。
