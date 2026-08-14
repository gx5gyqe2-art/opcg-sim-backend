# セッション並列運用（親セッションから子セッションを起動する）

Claude Code のリモートセッションから、**別のリモートセッション＝別コンテナ**を起動して並列に作業させるための
運用手順。CPU 律速の作業（自己対戦サンプリング、shard 分割した学習データ生成）を複数コンテナに分散する用途を想定する。

検証は 2026-08-13〜14 に実施。**実測で確認できた事実と、確認できなかった事実を明示的に分けて書く**。

## 1. 前提

- 実行環境は `anthropic_cloud`。環境は `list_environments` で取得する（本アカウントでは "post" の1つのみ）。
- コンテナはセッションごとに独立。**リポジトリはセッション開始時に fresh clone される**。
- 親のワーキングツリーの未コミット変更は子に引き継がれない。

## 2. ツール

MCP サーバ `claude-code-remote` が提供する。

| 目的 | ツール | 副作用 |
|---|---|---|
| 子セッション起動 | `create_session` | **あり**（コンテナ作成） |
| 状態確認 | `get_session` / `list_sessions` | なし |
| 環境一覧 | `list_environments` | なし |
| 起動済みセッションへの追加指示 | `create_trigger` + `fire_trigger` | あり |
| 中断 | `interrupt_session` | あり |
| 後片付け | `archive_session` | あり |

> **ツール名が2表記ある。** `mcp__Claude_Code_Remote__<tool>` と `mcp__<uuid>__<tool>` の両方が観測された
> （同一セッション内で前者から後者へ切り替わった実例あり。切り替わり後は前者が
> `No such tool available` になる）。`.claude/settings.json` の allowlist は**両方を登録**しないと
> 承認プロンプトを抑止できない。UUID が環境ごとに変わる場合は追記が必要。

## 3. 手順

### 3.1 起動

```
create_session(
  title:           "<識別しやすい名前>",
  tags:            ["<グループ名>"],          # 後で list_sessions のフィルタに使う
  source_url:      "https://github.com/gx5gyqe2-art/opcg-sim-backend",
  source_revision: "refs/heads/main",
  outcome_branch:  "claude/<作業ごとに一意な名前>",
  prompt:          "<完全に自己完結した指示>"
)
```

- **`source_url` / `source_revision` は明示する。** 省略すると `session_context.sources` が空のまま返り、
  リポジトリが利用可能かが不定になる。
- **`permission_mode` は指定しない。** 親のモードを継承する。親より強いモードを渡すと
  `exceeds parent session's "default"` で**起動そのものが失敗する**。子に強い権限が要るなら、
  親セッションをアプリ側で強いモードで開始しておく。
- **`prompt` は完全に自己完結させる。** 子は親の会話を一切知らない。
- 並列で起動するときは**同一メッセージ内で複数の `create_session` を発行**する。

### 3.2 進捗の確認

`get_session` の `session_status` は `PENDING → RUNNING → IDLE` と遷移する。
`task_summary`（実行中）と `post_turn_summary`（完了後）に短い要約が入る。

### 3.3 成果の受け取り

**子の成果を親へ渡す経路は git push のみ。**

`post_turn_summary` は**別途生成される要約であり、子の最終メッセージの転記ではない**。
プロンプトで出力形式を指定しても無視される（2回の実測で確認）。「進んだ／終わった」の粗い進捗表示専用で、
**データの受け渡しには使えない**。

したがって、親が読む必要のある値は必ずファイルに書かせて push させ、親は `git fetch` して読む。

```
git fetch origin <branch>
git show FETCH_HEAD:<path>
```

### 3.4 起動後の追加指示

```
create_trigger(name: "...", persistent_session_id: "<子のID>", prompt: "<追加指示>")
fire_trigger(trigger_id: "<trig_...>")
```

`cron_expression` / `run_once_at` を省略すると自走しない poke 専用トリガになる。
実測で `fire_trigger` の**1秒後**に対象セッションが反応した。

**会話コンテキストは継続する**（実測: 初回の `cache_write` 79,987 トークンと、poke 後の
`cache_read` 79,987 トークンが完全一致）。「さっきの続きで」という指示が通る。

> **制約**: trigger 経由で起こされたセッションは **MCP ツールを持たない**（作成時に警告が出る）。
> 標準ツール（Bash/Read/Write 等）のみ。**poke で起こした子に孫セッションを起動させることはできない。**
> 多段運用が要るなら `create_session` の初回 prompt で完結させる。

### 3.5 後片付け

- トリガ: `delete_trigger`
- セッション: `archive_session`（**archive すると会話内容が読めなくなる**。値を確認してから実行する）
- 作業ブランチ: 下記「既知の制約」を参照

## 4. 実測値（2026-08-13）

| 項目 | 実測 |
|---|---|
| 起動〜完了（repo なし・コマンド数本） | 19秒 |
| 起動〜clone〜commit〜push 完了 | 40秒 |
| `fire_trigger` 〜 対象セッションの反応 | 1秒 |
| poke 〜 完了 | 21秒 |
| コンテナ仕様（親・子とも） | nproc=4 / mem 15GB |

親と子で `/proc/sys/kernel/random/boot_id` が異なることを確認済み（親 `e5bb0c60…` / 子 `65ffd870…`）＝
**別コンテナであることは実証済み**。

## 5. 既知の制約

1. **成果の受け渡しは git push のみ**（§3.3）。
2. **レート制限はアカウント共有**（`five_hour` ウィンドウ）。並列数を増やしても総スループットは頭打ちになる。
3. **子から親へのメッセージ経路は無い。** `ListAgents` は "No reachable agents" を返す。
   親は `get_session` のポーリングか push されたコミットでしか状況を知れない。
4. **子のツール権限は親を超えられない。** `extra_allowed_tools` に親が持たないものを書いても落とされる。
5. **同一ブランチへの同時 push は衝突する。** シャードごとにブランチを分けるか、
   書き込むファイルを完全に分離する。
6. **リモートブランチの削除ができない。** `git push origin --delete <branch>` は
   `send-pack: unexpected disconnect` で失敗する（セッションの git proxy が拒否していると思われる）。
   テスト用ブランチは GitHub 側で削除する。
7. **`~/.claude/projects/` に過去セッションの transcript は残らない。** コンテナが毎回作り直されるため。
   履歴走査に依存する仕組み（`/fewer-permission-prompts` 等）はこの環境では機能しない。
8. **`.claude/settings.local.json` は永続しない**（gitignore 対象＝新コンテナに存在しない）。
   allowlist を永続させるには**リポジトリにコミットする `.claude/settings.json`** を使う。
   反映は**次のセッションから**（settings はセッション開始時に読まれる）。

## 6. 未検証

- **3件以上の同時起動時に CPU が本当に分離されるか。** 単体では親子が各 nproc=4 を持つことを確認済みだが、
  同時実行時にスループットが落ちないか（物理コアの取り合いが無いか）は**未測定**。
  測る場合の設計: 各子に固定 CPU 負荷（例: SHA-256 を30秒回して反復数を数える）を実行させ、
  `start_epoch` / `end_epoch` / 反復数をファイルに書いて push させる。実行区間の重なりで直列化の有無を、
  反復数の単独実行時との比較で競合の有無を判定する。単独実行時の基準値は 40,125,000 反復/30秒（nproc=4 のうち1コア使用）。
