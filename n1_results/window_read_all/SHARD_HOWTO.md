# WP `window-read-all` 段 1 のシャードの回し方（A＝読み〔eval〕・B＝打ち比べ〔judge〕）

## A. 読み（eval）のシャード 1〜3（先に回す）

- 点の表は `n1_results/window_read_all/points.jsonl`（350 点）。最初の 121 点は `evals.jsonl` に済み（飛ばされる）。
- **pid mod 4 == i** の残りの点がシャード i（各 55〜60 点）。シャード 0 は `claude/window-read-all` のセッションが回す。
- 所要の目安: 4 本並べで **1 シャード 40〜60 分**。

```bash
I=1   # ← シャード番号（1〜3）
cd /home/user/opcg-sim-backend
git fetch origin claude/window-read-all
git checkout -B claude/window-read-all-e$I origin/claude/window-read-all
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/window_read_all.py eval --workers 4 --shard $I/4 \
  > n1_results/window_read_all/eval_shard$I.log 2>&1 &
# 1 行目 `EVAL shard=I/4 todo 55〜60` を確かめ、9 分以内の待ちを繰り返して見張る。終わったら:
git add n1_results/window_read_all/evals_shard$I.jsonl n1_results/window_read_all/eval_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "window-read-all: stage 1 eval shard $I/4"
git push -u origin claude/window-read-all-e$I
```

- push 先は **`claude/window-read-all-e$I`**。コードと `RESULT.json` は変えない。

## B. 打ち比べ（judge）——A を回収した後に、別に知らせる


仕様: `docs/search_efficiency_plan.md` §7.67 の段 1（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/window_read_all.py`（`judge --shard i/8`）。
読み取りだけ（エンジン・ネットは変えない）＝**ゲートは不要**。

## 分け方
- 比べる点は、`n1_results/window_read_all/evals.jsonl` で「短い木（640 回・世界 2 本）の手が今の手と違う」点（`claude/window-read-all` に push 済み）。
- **点の番号（pid）mod 8 == i** の点がシャード i。シャード 0 は `claude/window-read-all` のセッションが回す。
- 1 点 16 通り（公平な配り直し・今の手と短い木の手の対・今のターンの残りは両席とも出荷の設定、その先は審判 r27・160 回）。
- 所要の目安: 4 本並べで 1 シャード（15〜20 点）**1〜1.5 時間**。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）

cd /home/user/opcg-sim-backend
git fetch origin claude/window-read-all
git checkout -B claude/window-read-all-j$I origin/claude/window-read-all

pip install -q maturin
make rust-develop

OPCG_LOG_SILENT=1 nohup python tests/scripts/window_read_all.py judge --k 16 --workers 4 --shard $I/8 \
  > n1_results/window_read_all/judge_shard$I.log 2>&1 &
```

- 1 行目に `JUDGE shard=I/8 todo 15〜20 ...` が出れば正しく始まっている（0 ならブランチの取り方が違う）。
- 点ごとに `[k/n] pid=... type=... diff=...` の行が増える。9 分以内の待ちを繰り返して見張る。止まっても同じコマンドで再開できる。

## 終わったら

```bash
git add n1_results/window_read_all/judge_shard$I.jsonl n1_results/window_read_all/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "window-read-all: stage 1 judge shard $I/8"
git push -u origin claude/window-read-all-j$I
```

- push 先は **`claude/window-read-all-j$I`**。コードは変えない。**`RESULT.json` は書き換えない**（`claude/window-read-all` のセッションが回収して集計する）。
