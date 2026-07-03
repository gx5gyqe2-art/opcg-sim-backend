# tests/fixtures/

テストが読み込むデータ資産（凍結ベースライン・期待値マニフェスト・held-out デッキ集合）。

| ファイル | 生成/更新元 | 用途 |
|---|---|---|
| `full_card_baseline.json` | `python tests/full_card_audit.py --regen` | 全カード挙動ベースライン（`test_full_card_baseline.py` / `test_verified_buckets.py` が照合） |
| `expected_effects.json` | `python tests/expected_effects.py --regen` | 期待挙動マニフェスト（`effect_oracle` が突き合わせ） |
| `heldout_decks.json` | 手動（ユーザ実対局リプレイの凍結入力） | held-out 実デッキ集合（`test_heldout_decks.py` が凍結検証） |

> 配置規約（`docs/refactoring_tests_and_errors.md` 参照）: テストが読むデータは本ディレクトリ、
> テストが import する基盤ライブラリは `tests/harness/`、単体実行の実験/計測 CLI は
> `tests/scripts/` に置く（harness/scripts の移設は E-2/E-3 で実施）。
