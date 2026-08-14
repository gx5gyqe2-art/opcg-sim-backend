# テスト実行ワーカー プロンプト（汎用・再利用可）

品質ゲート（`make test`）を**別セッション＝別コンテナ**に実行させ、結果を回収するためのプロンプト。
親セッションはテストの10分間ブロックされずに他の作業を続けられる。

起動と回収の仕組みは [`session_orchestration.md`](session_orchestration.md) を参照。
**結果の受け取りは git push 経由のみ**（`post_turn_summary` はデータ経路に使えない）。

## 使い方

`<TESTED_BRANCH>`（テスト対象のブランチ）と `<TAG>`（レポートを識別する短い名前）を置換して使う。

```
create_session(
  title:           "test-worker <TAG>",
  tags:            ["test-worker"],
  source_url:      "https://github.com/gx5gyqe2-art/opcg-sim-backend",
  source_revision: "refs/heads/<TESTED_BRANCH>",
  outcome_branch:  "claude/test-report-<TAG>",
  prompt:          "<下の本文>"
)
```

回収:

```bash
git fetch origin claude/test-report-<TAG>
git show FETCH_HEAD:test_report_<TAG>.md
```

---

## プロンプト本文（ここから貼る）

あなたはテスト実行専用のワーカーです。**品質ゲートを1回実行して結果を報告するだけ**が仕事です。
**テストが失敗しても、原因の修正やコードの変更は一切行わないでください**（報告のみ）。

### 1. 環境準備

このコンテナにはテスト依存が入っていません。最初にインストールします。

```bash
cd /home/user/opcg-sim-backend
python -m pip install -q pytest pytest-xdist fastapi httpx numpy pydantic uvicorn websockets requests python-multipart
```

`google-cloud-*` は不要です（`tests/_bootstrap.py` がスタブを提供します）。

インストール後、収集だけ先に通しておくと環境不備を早く検知できます（7秒程度）。

```bash
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -m "not slow" -p no:cacheprovider --collect-only 2>&1 | tail -3
```

`1585/1589 tests collected (4 deselected)` 付近の行が出れば環境は正常です
（2026-08-14 実測。件数はテスト追加に伴い増えます）。`ModuleNotFoundError` が出たら
インストールが失敗しているので、その旨を報告して終了してください。

### 2. 対象の確認

```bash
git rev-parse --abbrev-ref HEAD
git log --oneline -1
```

`<TESTED_BRANCH>` に居ることを確認してください。違っていたら以降を実行せず、その旨を報告して終了します。

### 3. テスト実行

`make test` は**約10分**かかります。Bash ツールの既定タイムアウトは2分・最大でも10分なので、
**必ずバックグラウンド実行（`run_in_background: true`）にして、完了通知を待ってください**。
フォアグラウンドで実行するとタイムアウトで打ち切られ、結果が取れません。

```bash
cd /home/user/opcg-sim-backend
make test 2>&1 | tee /home/user/test_run.log
```

`make test` の中身は `Makefile` が正本です（`OPCG_LOG_SILENT=1` と `-s` は必須フラグとして
Makefile に含まれています）。**コマンドを手で書き換えないでください。**

`-m "not slow"` により重テストは除外されます。これは意図された挙動です。

### 4. レポート作成

完了したら、リポジトリ直下に `test_report_<TAG>.md` を作り、次を記載します。

```
# テスト実行レポート <TAG>

- 対象ブランチ: <TESTED_BRANCH>
- HEAD: <git rev-parse HEAD>
- 実行日時(UTC): <date -u>
- コマンド: make test
- 終了コード: <make の exit code>
- 所要時間: <実測>

## 結果サマリ

<pytest 最終行をそのまま。例: "1234 passed, 5 skipped in 587.12s">

## 失敗したテスト

<失敗が無ければ「なし」。あれば失敗したテストの ID を全て列挙し、
それぞれの assertion エラー行を3〜5行だけ引用する>

## ログ末尾

<test_run.log の末尾40行>
```

**ログ全文はコミットしないでください**（巨大になります）。上記の抜粋だけにします。

### 5. push

```bash
git checkout -b claude/test-report-<TAG>
git add test_report_<TAG>.md
git commit -m "test report <TAG>: <passed/failed の要約>"
git push -u origin claude/test-report-<TAG>
```

network エラーで失敗した場合のみ 2s→4s→8s→16s のバックオフで最大4回リトライしてください。

### 6. 報告

最終メッセージで「green / red」と失敗件数を1行で報告してください。
（この報告は要約されて親に届きます。詳細は push したレポートが正本です。）

**やらないこと**: テストの修正、コードの変更、`make regen-baseline` の実行、PR の作成。

---

## 補足

- **`make test-slow` が要るとき**: make/unmake（journal）周辺を変更した場合のみ。その場合は §3 のコマンドを
  `make test-slow` に替えたワーカーをもう1つ立てる（現状 ~245秒のテストが1本）。
- **並列で複数ブランチを試験する場合**: `<TAG>` をブランチごとに変え、`outcome_branch` も分ける。
  同一ブランチへの同時 push は衝突する。
- **レート制限はアカウント共有**。テストワーカーを大量に並べても総スループットは頭打ちになる。
