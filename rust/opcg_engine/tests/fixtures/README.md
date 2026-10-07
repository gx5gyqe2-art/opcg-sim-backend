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

## 理論の全移植の golden（段 1〜6・2026-10-06／07）

| ファイル | 中身 |
|---|---|
| `theory_leaves_golden.jsonl.gz` | 葉（`src/theory/leaves_*.rs`・`cond.rs`）の**記録した呼び出し** 20,859 行（Python が本物の通し〔8 器 × `f_identity/rec`・実 w41・合成 w39 の各 5 局〕の中で実際に解いた入力・大域・核の答え・出力・浮動小数は 16 進のビット）。`theory::tests_leaves` が解き直してビットで比べる。約 1 MB |
| `theory_core_golden.jsonl.gz` | 値付けの核（段 3）の記録した呼び出し。`theory::core::tests_core` |
| `theory_outer_golden.jsonl.gz` | 守る側の外側と耐久（段 4）の記録した呼び出し。`theory::core::tests_outer` |
| `theory_rows_golden.jsonl.gz` | 局の駆動と行の関数（段 5／6）の記録した呼び出し（局の枠つき）。`theory::core::tests_rows` |
| `rd_dp_golden.jsonl`・`rd_solve_golden.jsonl.gz` | 守る側の計算（第 1〜3 段）の記録した解。`theory::tests` |
| `theory_cards.json.gz`・`theory_effects.json.gz` | 記録を取ったときのカード表と語彙（2803 枚）・効果の木 |

**段 7（2026-10-07）で Python の理論（記録を取った側）と、記録の器・間引きの道具（旧 `tests/scripts/theory_capture*.py`・
`theory_*_golden.py`・`rd_kernel_golden.py`・答え合わせの原文 `tests/harness/rule_don_ref.py`）を消した**——これらの golden は
**固定の正本**（Python の最後の出力の写し）で、Python 無しで `cargo test` が解き直す。**式を意図して変えたら**記録を取り直す道具は
もう無いので、変えた Rust の出力で作り直し（golden は「その時点の Rust の出力」になる＝正しさの独立した証拠ではない）、
差分を必ずレビューする。旧い Python は凍結ブランチ `claude/theory-switches-final` と履歴で辿れる。

### 覚え書きの鍵を正確にした作り直し（2026-10-07・`docs/reports/2026-10-07_memo_exact.md`）

値付けの核・外側・局の駆動の覚え書きの鍵を全部正確にした（丸めた鍵・文脈の抜け・先勝ちの共有をやめた）ので、
`theory_core_golden`・`theory_outer_golden`・`theory_rows_golden` の**戻りの欄**（`r`・`cs`・`ev`／`result`）を Rust の出力で作り直した
（入力の欄は 1 文字も変えていない・記録の再生で前もって入れる覚え書き `pre` は読まなくなったので全部の行から外した）。
`theory_leaves_golden`・`rd_*_golden` は覚え書きを通らないので変わらない。作り直しの器は Rust だけで回る:

```bash
cd rust/opcg_engine
OPCG_THEORY_GOLDEN_WRITE=1 cargo test --no-default-features tests_regen -- --include-ignored --nocapture
```

（`OPCG_THEORY_GOLDEN_WRITE` が無ければ書き換える行を数えるだけ。各行を試験と同じ道〔`tests_core`／`tests_outer`／`tests_rows` の
`run_one`〕で解き、戻りが違う行だけ書き換える。**golden は「その時点の Rust の出力」**＝作り直したら差分を必ずレビューする。）

