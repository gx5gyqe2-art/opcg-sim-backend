# WP `don-read` 段 1 の読みのシャード（1〜3）の回し方

仕様: `docs/search_efficiency_plan.md` §7.66 の段 1（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/don_read.py`（`read --shard i/4`）。
読み取りだけ（エンジン・ネットは変えない）＝**ゲートは不要**。

## 分け方
- 点の表は `n1_results/don_read/points.jsonl`（72 点）。**番号（pid）mod 4 == i** の点がシャード i（18 点ずつ）。
- シャード 0 は `claude/don-read` のセッションが回す。
- 1 点＝配り直した世界 16 本 × 手 2 つ × 読み 3 つ（今の読み・窓を閉じてから短い木・次の自分のターンの頭まで出荷の設定で打つ）。
- 所要の目安: 4 本並べで 1 点 6〜12 分 → **1 シャード 1.5〜2.5 時間**。

## 手順（シャード i ＝ 1〜3 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜3）

cd /home/user/opcg-sim-backend
git fetch origin claude/don-read
git checkout -B claude/don-read-j$I origin/claude/don-read

pip install -q maturin
make rust-develop

OPCG_LOG_SILENT=1 nohup python tests/scripts/don_read.py read --workers 4 --k 16 --shard $I/4 \
  > n1_results/don_read/read_shard$I.log 2>&1 &
```

- 1 行目に `READ shard=I/4 todo 18（か 17） ...` が出れば正しく始まっている（0 ならブランチの取り方が違う）。
- 点ごとに `[k/n] pid=... ok=True sec=...` の行が増える。9 分以内の待ちを繰り返して見張る。止まっても同じコマンドで再開できる（終えた点は飛ばす）。

## 終わったら

```bash
git add n1_results/don_read/reads_shard$I.jsonl n1_results/don_read/read_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "don-read: stage 1 read shard $I/4"
git push -u origin claude/don-read-j$I
```

- push 先は **`claude/don-read-j$I`**。コードは変えない。`RESULT.json` は要らない（`claude/don-read` のセッションが回収して集計する）。
