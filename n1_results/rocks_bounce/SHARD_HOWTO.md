# WP `rocks-bounce` の打ち比べのシャード（0〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.70。道具: `tests/scripts/rocks_bounce.py`。エンジン・ネットは変えない（打ち比べるだけ）＝**ゲートは不要**。
コードと `RESULT.json` は変えない。点の表 `n1_results/rocks_bounce/points.jsonl` はコミット済み（作り直さない）。

## 手順（シャード I ＝ 0〜7 のどれか。下の `I` を置き換える）

```bash
I=0   # ← シャード番号（0〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/rocks-bounce
git checkout -B claude/rocks-bounce-j$I origin/claude/rocks-bounce
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/rocks_bounce.py judge --workers 4 --k 16 --turns 4-6 --shard $I/8 \
  > n1_results/rocks_bounce/judge_shard$I.log 2>&1 &
```

- 1 行目に `JUDGE shard=I/8 todo 26`（か 27）が出れば正しく始まっている。点ごとに `[k/26] ... ok=True diff=...` が増える（1 点 約 9 分）。
  9 分以内の待ちを繰り返して見張る。止まっても同じコマンドで再開できる（終えた点は飛ばす）。最後に `JUDGE_DONE` が出れば終わり（約 1 時間）。

## 終わったら

```bash
git add n1_results/rocks_bounce/judge_shard$I.jsonl n1_results/rocks_bounce/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "rocks-bounce: judge shard $I/8"
git push -u origin claude/rocks-bounce-j$I
```

- push 先は **`claude/rocks-bounce-j$I`**。
