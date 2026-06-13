# ドキュメント索引（引き継ぎ資料 → 仕様書）

本リポジトリの引き継ぎ資料は、**システム仕様書**と**テスト仕様書**として再編した。
本書は各仕様書への索引（エントリポイント）である。

## 仕様書

| 文書 | 内容 |
|---|---|
| [`docs/SPEC.md`](SPEC.md) | **システム仕様書**。全体アーキテクチャ、コアゲームルール（ターン/戦闘/召喚酔い・速攻/場5体上限）、オンライン対戦（ルーム/WS）、カード効果システム、ファイルマップ、実装上の不変条件 |
| [`docs/TEST_SPEC.md`](TEST_SPEC.md) | **テスト仕様書**。テスト戦略、テストスイート一覧、診断/監査ツール、ルール追加・検証フロー、品質ゲート |
| [`docs/parser_v2.md`](parser_v2.md) | カード効果パーサ（EffectParserV2）の設計詳細・ルール一覧 |
| [`docs/leader_specs/`](leader_specs/README.md) | 全137リーダーのカード個別仕様（テキスト/期待挙動/テストケース）。作成ガイド [`_GUIDE.md`](leader_specs/_GUIDE.md)、テスト方針 [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、既知差異 [`ISSUES.md`](leader_specs/ISSUES.md) |

フロントエンドの仕様は `opcg-sim-frontend/docs/SPEC.md`。

## クイックスタート

```bash
# テスト（キャプチャ無効・ログ抑止が必須）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider

# 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
```

詳細な検証フローは [`docs/TEST_SPEC.md`](TEST_SPEC.md) を参照。
