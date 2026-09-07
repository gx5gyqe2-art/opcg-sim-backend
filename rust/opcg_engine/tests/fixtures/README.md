# `cargo test` の盤面 fixture（P1-model・`docs/rust_engine_plan.md` §9）

`src/model.rs` のテスト（`hidden_fixture_roundtrips_to_the_python_board_dict` ほか）が読む実データ。
**Python 版が正本**なので、期待値は Python の盤面 dict をそのまま持ってきたもので、Rust 側の出力から
作り直してはいけない。

| ファイル | 中身 |
|---|---|
| `hidden_v2.json` | 記録形式 v2 の `hidden` 1 行（§9.1。v3 で `active_battle` に `attacker_owner`/`target_owner` を追記済み）＝盤面を完全に再構成できる内部状態。約 51KB |
| `board_v2.json` | 同じ行の `state`（Python の `rs_diff_replay.py::board_dict`）から `pending_request` を除いたもの＝**期待値**。約 26KB |
| `audit_eb01_049_v4.json` | **監査記録**（記録 v4 の `kind:"audit"`・§11.2）1 件＝EB01-049「相手のコスト2以下のキャラ1枚までを、KOする」の登場時（中断 1 回）。`fire.state` と `steps[].state` が**期待値＝Python の盤面 dict**で、`effects::tests_effects::audit_oracle_matches_python` が `state::replay_audit_with` の出力と段ごとに突き合わせる。Rust が読まない欄（`setup.state`／`fire.hidden`／`steps[].hidden`）は落としてある。約 78KB |
| `audit_masters_v4.json` | 上の記録に出る**実カード（EB01-049）だけ**に絞った効果 JSON＝`MasterTable::from_effects_json` の入力（`FILLER`・合成リーダーは記録の `extra_masters` が持つ）。約 2KB |
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

監査 fixture（`audit_eb01_049_v4.json`／`audit_masters_v4.json`）の作り直し方:

```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py \
  --card-ids EB01-049 --dump /tmp/audit.json
```

`/tmp/audit.json` から `setup.state`・`fire.hidden`・`steps[].hidden` を落としたものを
`audit_eb01_049_v4.json` へ、`opcg_effects.json` の `cards` を `EB01-049` だけに絞ったもの
（`counts` も数え直す）を `audit_masters_v4.json` へ書く（どちらも
`json.dump(..., ensure_ascii=False, sort_keys=True, separators=(",",":"))`）。
**期待値は Python の出力**で、Rust の出力から作り直してはいけない。
