# プロジェクト運用ルール（エージェント向け）

このファイルは Claude（エージェント）が本リポジトリで作業する際の取り決め。毎セッション従う。

## 開発・CI・マージのリズム

1. **ブランチで開発**しコミットする（指定があればそのブランチ。無ければ作業用ブランチを作る）。
2. **push → PR 作成**する。PR を出して **CI が起動した瞬間に、チャットへ「CI投げました（約1分）」と一報**する
   （ユーザは iPhone アプリ利用。アプリを離れている間はこの一報が端末プッシュ通知になる）。
3. **CI 結果はユーザに聞かれたら確認**して返信する。この環境には GitHub API トークンが無く、
   裏での自動ポーリングはできない（`api.github.com` は未認証・共有IPでレート制限）。
   **失敗は Webhook で自動的にチャットへ届く**ので、届いたら調査・修正する。
4. **マージはユーザの明示の指示があるまで実行しない**（CI が緑でも勝手にマージしない）。
5. マージすると PR 購読は自動解除される。マージ済み PR は再オープンしない。

> 補足: 「CI 完了（成功）の瞬間に自動通知」だけは現状の権限では不可。実現するには `repo` スコープの
> PAT を環境変数（例 `GITHUB_TOKEN`）として環境に追加する必要がある。追加されれば `Monitor`+`curl`
> で 30 秒間隔ポーリング＋完了時 proactive 通知が組める。

## マージ前に緑であるべき品質ゲート

`OPCG_LOG_SILENT=1` と `-s`（キャプチャ無効）は本スイートの必須フラグ。API テストは `fastapi`/`httpx`
導入後に collection 可。

```bash
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider   # 全テスト
OPCG_LOG_SILENT=1 python tests/full_card_audit.py                     # 構造不変条件 = 0
```

- 全テスト pass（`test_full_card_baseline.py`＝挙動ベースライン一致、`test_effect_oracle_gate.py`＝
  HAS_OTHER/PER_TURN_LIMIT_GAP/UP_TO_GAP = 0 のラチェットを含む）
- 構造監査 `EXCEPTION/CARD_LOSS/TEMP_LEAK = 0`
- **挙動を意図的に変えた場合のみ** `python tests/full_card_audit.py --regen` でベースライン再生成し、
  差分をレビューする。検証済みデッキの挙動を直したら `tests/test_verified_decks.py` にアサート追記。

詳細は `docs/TEST_SPEC.md`（§4 検証フロー／§5 品質ゲート）、`docs/README.md`（文書索引）を参照。

## ドキュメントの更新

- 仕様（正本: `docs/SPEC.md` / `TEST_SPEC.md` / `parser_v2.md` / `leader_specs/`）は実装変更に追従して最新に保つ。
- 報告（`docs/reports/`）は特定時点のスナップショットで追記・改変しない。
