# 別セッション（ワーカー）の起動と結果回収

長時間の処理（品質ゲート・自己対戦生成・学習ループ）を**別セッション＝別コンテナ**に投げ、
親セッションはブロックされずに作業を続けるための共通の仕組み。個々のワーカーの中身は
[`test_worker_prompt.md`](test_worker_prompt.md) など `*_worker_prompt.md` 側に書く。ここは**起動と回収の経路だけ**を扱う。

## 前提（この環境の性質）

- セッションは**独立した使い捨てコンテナ**で動く。リポジトリはセッション開始時に fresh clone され、
  一定時間の無操作でコンテナは回収される。
- したがって **残したいものは commit して push するまで存在しない**。`/home` に置いたファイル、
  実行ログ、`pip install` した依存は、そのセッションが消えると一緒に消える。
- 親子セッションでファイルシステムは共有されない。**共有されるのは git のリモートだけ**。

## 起動

`create_session`（claude-code-remote MCP）で子セッションを作る。テストワーカーの例:

```
create_session(
  title:           "test-worker <TAG>",          # 一覧で識別するための名前
  tags:            ["test-worker"],              # 後から list_sessions で絞り込む
  source_url:      "https://github.com/gx5gyqe2-art/opcg-sim-backend",
  source_revision: "refs/heads/<TESTED_BRANCH>", # 子が clone するブランチ
  outcome_branch:  "claude/test-report-<TAG>",   # 子が push するブランチ
  prompt:          "<ワーカープロンプト本文>"
)
```

- `environment_id` を省くと親と同じ環境を継承する（環境変数・ネットワークポリシーも同じ）。
- `source_revision` を省くと既定ブランチになる。**テスト対象を明示すること**。
- `outcome_branch` は子の push 先。ワーカーごとに変える（同一ブランチへの同時 push は衝突する）。
- `permission_mode: "plan"` は**使わない**。人間の承認待ちで止まり、誰も見ていないワーカーは
  そこで永久に停止する。

`prompt` は**それ単体で完結する指示**にする。子は親の会話を一切知らない。

## 結果の回収

**データ経路は git push だけ**。子セッションの最終メッセージは要約されて親に届くが、
これは「green / red」程度の短い通知であって、**データの受け渡しには使えない**
（`post_turn_summary` も同様＝報告経路であってデータ経路ではない）。

したがってワーカーには必ず

1. 成果物をリポジトリ内のファイルとして書く
2. `outcome_branch` に commit して push する

までをやらせる。親はそのブランチから読む:

```bash
git fetch origin claude/test-report-<TAG>
git show FETCH_HEAD:test_report_<TAG>.md
```

作業ツリーを汚さずに読めるので、親が別の作業中でも安全に回収できる。

## 監視・操作

| やること | 手段 |
|---|---|
| 状態を見る | `list_sessions`（`tags` で絞る）／`get_session` |
| 追加指示を出す | `send_message`（子の文脈は保持される） |
| 暴走を止める | `interrupt_session` → `send_message` で軌道修正 |
| 片付ける | `archive_session`（コンテナが解放される） |

子の完了通知を待つ間、親はポーリングのために `sleep` で待たない。

## 注意

- **レート制限はアカウント共有**。ワーカーを並べても総スループットは頭打ちになる。
  「10分の待ちを1本だけ肩代わりさせる」用途が最も割に合う。
- 子は依存が入っていない素のコンテナで始まる。**`pip install` から書く**。
- 子に「失敗したら直せ」と言わない限り、**報告だけさせる**方が安全（テストワーカーは報告専任）。
  修正は結果を見た親（または人間）が判断して行う。
