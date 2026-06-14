# opcg-sim-backend

ワンピースカードゲーム シミュレータのバックエンド（FastAPI + 独自ルールエンジン）。
ルールモード（公式ルール自動進行：ソロ／オンライン対戦／CPU 対戦）とフリーモード（自由操作）を提供する。

## ドキュメント

文書は種別（仕様＝正本 / 報告＝時点記録）で分類している。索引は [`docs/README.md`](docs/README.md)。

- [`docs/SPEC.md`](docs/SPEC.md) — システム仕様（アーキテクチャ・コアルール・オンライン対戦・CPU 対戦・効果システム）
- [`docs/TEST_SPEC.md`](docs/TEST_SPEC.md) — テスト仕様（戦略・スイート・効果検証ハーネス・品質ゲート）
- [`docs/parser_v2.md`](docs/parser_v2.md) — カード効果パーサ設計
- [`docs/leader_specs/`](docs/leader_specs/README.md) — 全137リーダーの個別仕様

## クイックスタート

```bash
# テスト（ログ抑止・キャプチャ無効が必須）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider

# 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
```
