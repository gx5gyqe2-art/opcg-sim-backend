# WP `hold-bias` の判定シャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.59。道具: `tests/scripts/hold_bias.py`（`judge --shard i/n`）。
局（`n1_results/hold_bias/games.jsonl`・150 局・seed 4171000〜4171149）は打ち終えている＝**シャードは判定だけ**。

## 分け方
- 判定する点の全体の並び（`hold_bias.job_order`・局の記録だけで決まる pure な並び・9 場面・計 1,450 点）の**番号 mod 8 == i** の点がシャード i。
- 既に済んだ点（`points.jsonl` の 115 点と、push 済みの `points_shard*.jsonl`）は飛ばす。各シャードの残りは 166〜167 点。
- シャード 0 はこのセッション（`claude/hold-bias`）が回している。
- 所要の目安: 4 コアで 1 点あたり約 60 秒 ÷ 4 本 ≈ **1 シャード 40〜50 分**（今のターンの残りを出荷の設定で打つので点ごとにばらつく）。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）

cd /home/user/opcg-sim-backend
git fetch origin claude/hold-bias
git checkout -B claude/hold-bias-j$I origin/claude/hold-bias

# Rust エンジンの wheel（maturin が無ければ入れる）
pip install -q maturin
make rust-develop

# 判定（K=8・4 本並べ・出力は n1_results/hold_bias/points_shard$I.jsonl）
nohup python tests/scripts/hold_bias.py judge --k 8 --workers 4 --shard $I/8 \
  > n1_results/hold_bias/judge_shard$I.log 2>&1 &
```

- 1 行目に `JUDGE shard=I/8 todo 166 前後 ...` が出れば正しく始まっている（`todo` が 0 や 1,000 を超えるならブランチの取り方が違う）。
- 終わると log の最後に `JUDGE_DONE` が出る。再開できる（止まったら同じコマンドを打ち直せば、済んだ点は飛ばす）。

## push（20〜30 分ごとの途中結果と、終わったとき）

```bash
git add -f n1_results/hold_bias/points_shard$I.jsonl n1_results/hold_bias/judge_shard$I.log
git commit -m "hold-bias: 判定シャード $I/8"
git push -u origin claude/hold-bias-j$I
```

終わったら `RESULT.json`（リポジトリ直下）を次の形で書いて、同じブランチに push する:

```json
{"wp": "hold-bias", "shard": "I/8", "status": "done",
 "branch": "claude/hold-bias-jI",
 "files": ["n1_results/hold_bias/points_shardI.jsonl", "n1_results/hold_bias/judge_shardI.log"],
 "points": <points_shardI.jsonl の行数>, "notes": ""}
```

## 決まり
- **コードは変えない**（道具もエンジンも。コードを変えないので `make test` は要らない）。PR は作らない・マージしない。
- `games.jsonl` や他のシャードのファイルは触らない（集めるのは `claude/hold-bias` のセッション）。
- 1 シャード分の点が判定できない（`why` が `replay` 等）行が多いときは、wheel が本線と違う疑い。`notes` に書いて完走する。
