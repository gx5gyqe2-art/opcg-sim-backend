# tests/fixtures/

テストが読み込むデータ資産（凍結ベースライン・期待値マニフェスト・held-out デッキ集合）。

| ファイル | 生成/更新元 | 用途 |
|---|---|---|
| `full_card_baseline.json` | `python tests/harness/full_card_audit.py --regen` | 全カード挙動ベースライン（`test_full_card_baseline.py` / `test_verified_buckets.py` が照合） |
| `expected_effects.json` | `python tests/harness/expected_effects.py --regen` | 期待挙動マニフェスト（`effect_oracle` が突き合わせ） |
| `heldout_decks.json` | 手動（ユーザ実対局リプレイの凍結入力） | held-out 実デッキ集合（`test_heldout_decks.py` が凍結検証） |
| `replays/*.json.gz` | 手動（リプレイビューア「リプレイ保存」出力を gzip） | **人間マーク付き実対局リプレイ**（frames+marks 同梱・現2局×16マーク）。`replay_reeval.py`/`mark_gate.py` が盤面復元して人間フィードバック回帰に使う（`docs/cpu_v4_plan.md` §4-3/§6） |
| `harm_profile.json` | `make theory-table REAL=<w41> SYN="<w39> <w42>" WORK=<dir>`（`tests/scripts/theory_table.py`・2026-10-08） | 理論の器の表（損害の輪郭・`theory_slope`・`sigma_t`・`sigma_rel`・`sigma_rel_whole`・`w_bar`）。**順に依存する**（`crossing_bridge` → σ → それを入れた `theory_bridge` → `w_bar`）ので必ずこの 1 本で作り直す。`w_bar_provenance` が `w_bar` を測った時の測り方（`consumer`＝その `w̄` を使う行の分子と同じ `σ_T`・`σ_rel`・2026-10-08 決定 B）・`σ_T`・`σ_rel`・輪郭を覚え、`κ` を使う器（`crossing_bridge.kappa_clock`）は今の表と違えば落ちる（手で σ だけ書き換えない）。検算だけは `make theory-table-check` |

> 配置規約（`docs/refactoring_tests_and_errors.md` 参照）: テストが読むデータは本ディレクトリ、
> テストが import する基盤ライブラリは `tests/harness/`、単体実行の実験/計測 CLI は
> `tests/scripts/` に置く（harness/scripts の移設は E-2/E-3 で実施）。
