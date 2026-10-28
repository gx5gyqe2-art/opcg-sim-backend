# WP `main-read` のシャードの回し方（A＝読み〔eval〕・B＝打ち比べ〔judge〕）

仕様: `docs/search_efficiency_plan.md` §7.68（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/main_read.py`。
読み取りだけ（エンジン・ネットは変えない）＝**ゲートは不要**。コードと `RESULT.json` は変えない。

## A. 読み（eval）のシャード 1〜7

- 点の表は `n1_results/main_read/points.jsonl`（5 種類 × 50＝250 点）。4 点は `evals_pre.jsonl` に済み（飛ばされる）。
- **pid mod 8 == i** の点がシャード i（各 30〜32 点）。シャード 0 は `claude/main-read` のセッションが回す。
- 1 点＝4 倍の回数の木 1 回＋2 つの手それぞれを次の自分のターンの頭まで出荷の設定で打つ（世界 8 本）。4 本並べで 1 点 5〜9 分 → **1 シャード 40〜70 分**。

```bash
I=1   # ← シャード番号（1〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/main-read
git checkout -B claude/main-read-e$I origin/claude/main-read
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/main_read.py eval --workers 4 --shard $I/8 \
  > n1_results/main_read/eval_shard$I.log 2>&1 &
# 1 行目 `EVAL shard=I/8 todo 30〜32` を確かめ、9 分以内の待ちを繰り返して見張る（止まっても同じコマンドで再開できる）。終わったら:
git add n1_results/main_read/evals_shard$I.jsonl n1_results/main_read/eval_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "main-read: eval shard $I/8"
git push -u origin claude/main-read-e$I
```

## B. 打ち比べ（judge）——A を回収した後に、改めて知らせる

```bash
I=1   # ← シャード番号（1〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/main-read
git checkout -B claude/main-read-j$I origin/claude/main-read
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/main_read.py judge --k 16 --workers 4 --shard $I/8 \
  > n1_results/main_read/judge_shard$I.log 2>&1 &
# 終わったら:
git add n1_results/main_read/judge_shard$I.jsonl n1_results/main_read/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "main-read: judge shard $I/8"
git push -u origin claude/main-read-j$I
```
