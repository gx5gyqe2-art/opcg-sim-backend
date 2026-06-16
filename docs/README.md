# ドキュメント索引

`opcg-sim-backend` のドキュメントは、**文書の種別（ライフサイクル）** で分類する。
種別を混在させず、それぞれの更新ルールに従って維持する。

| 種別 | 役割 | 更新ルール |
|---|---|---|
| **仕様（正本）** | システムの「現在のあるべき姿」 | 実装変更に追従して**常に最新**に保つ |
| **報告（点）** | 特定時点の調査・検証スナップショット | 追記・改変しない（履歴として残す） |

> 計画書（plan）は実装完了後に正本（SPEC / TEST_SPEC）へ吸収し、文書としては残さない。
> 設計の経緯は git 履歴を参照する。

## 仕様（正本）— `docs/` 直下

| 文書 | 内容 |
|---|---|
| [`SPEC.md`](SPEC.md) | **システム仕様書**。全体アーキテクチャ／コアゲームルール（ターン・戦闘・召喚酔い/速攻・場5体上限）／オンライン対戦（ルーム・WS）／**CPU 対戦・AI**（§2.5）／カード効果システム／ファイルマップ／既知のモデル化制約（§6.1） |
| [`TEST_SPEC.md`](TEST_SPEC.md) | **テスト仕様書**。テスト戦略／スイート一覧／診断・監査ツール／**効果検証ハーネス**（CPU 対 CPU 自己対戦・インバリアント検出, §3.1）／品質ゲート／デッキ単位の手動検証 |
| [`parser_v2.md`](parser_v2.md) | カード効果パーサ（EffectParserV2）の設計・ルール一覧・既知のパース制約 |
| [`leader_specs/`](leader_specs/README.md) | 全137リーダーのカード個別仕様（テキスト／期待挙動／テストケース）。作成ガイド [`_GUIDE.md`](leader_specs/_GUIDE.md)、テスト方針 [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、既知差異 [`ISSUES.md`](leader_specs/ISSUES.md) |

フロントエンドの仕様は `opcg-sim-frontend/docs/`。

> エージェント（Claude）の運用ルール（開発・CI・マージのリズム／品質ゲート／文書更新方針）は
> リポジトリ直下の [`../CLAUDE.md`](../CLAUDE.md) に定義する（毎セッション自動読込）。

## 報告（点）— `docs/reports/`

| 文書 | 内容 |
|---|---|
| [`reports/effect_verification_iter1.md`](reports/effect_verification_iter1.md) | 効果検証イテレーション1（2026-06）のトリアージ報告 |
| [`reports/effect_verification_iter2.md`](reports/effect_verification_iter2.md) | 効果検証イテレーション2（2026-06）の修正報告（EB01-001／お互い同時両側／検出器精度） |
| [`reports/quality_postmortem_categoryH.md`](reports/quality_postmortem_categoryH.md) | 品質ポストモーテム（2026-06）。カテゴリH（先頭条件が「。その後、」をまたぐ漏れ・~119能力/全弾）の見逃し原因分析と横展開調査（Duration/chooser/すべては健全） |
| [`reports/cpu_precision_batch_20260616.md`](reports/cpu_precision_batch_20260616.md) | CPU 精度向上（2026-06-16）。検証基盤フェーズ0（凍結ベースライン Elo＋regret ログ＋ドン→クロック decide パズル）＋バッチ C-1/B-3/C-3/C-2 の実装記録と、アリーナで観測した normal<easy（独立の既存課題）の所見・A/B 結果 |
| [`reports/cpu_plan_ideal_line_design_20260616.md`](reports/cpu_plan_ideal_line_design_20260616.md) | 設計メモ（2026-06-16）。自デッキ「理想ライン」自動導出プラン（A・構成からのヒューリスティック）＋J値（白＝デッキ残＋トラッシュ）差分スケジュールでの進捗採点・相手リーダー由来 `OpponentProfile` でのマッチアップ補正。`PlanProfile`/`_plan_progress` 拡張案・フェア性/回帰/段階導入計画 |

## クイックスタート

```bash
# テスト（キャプチャ無効・ログ抑止が必須）
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider

# 全カード構造不変条件・挙動ベースライン
OPCG_LOG_SILENT=1 python tests/full_card_audit.py
```

詳細な検証フローは [`TEST_SPEC.md`](TEST_SPEC.md) を参照。
