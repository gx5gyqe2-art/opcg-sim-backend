# rollout-teach のシャードの回し方（30 本・1 本 ≒ 2 時間）

WP `rollout-teach`（`docs/search_efficiency_plan.md` §7.80）の段 1。シャード番号 `K`（1〜30）、2 桁の零詰めを `KK`（例 `07`）とする。
**コードは変えない**。道具は `tests/scripts/rollout_teach.py`、盤面の一覧は `n1_results/rollout_teach/positions.json.gz`（このブランチに同梱・作り直さない）。

## 1. 用意（約 10 分）

```bash
cd /home/user/opcg-sim-backend            # リポジトリの場所（違えば読み替える）
git fetch origin claude/rollout-teach
git checkout -B claude/rollout-teach-sKK FETCH_HEAD
pip install -q maturin && pip install -q numpy
make rust-develop                         # エンジンの wheel（数分）
python -c "import opcg_engine"            # 通ること
```

4 波の記録（128 ブランチ・合計約 2GB）を取る（約 5 分）:

```bash
for wv in 70f 71f 72f 73f; do
  mkdir -p ~/n$wv
  for w in $(seq -w 1 32); do
    [ -d ~/n$wv/n_records/n${wv}_w${w}_p0 ] && continue
    for t in 1 2 3 4; do git fetch -q --depth 1 origin claude/n$wv-w$w && break; sleep $((2**t)); done
    git archive FETCH_HEAD n_records | tar -x -C ~/n$wv
  done
done
ls -d ~/n7*f/n_records/*_p[0-3] | wc -l    # 512 であること
```

## 2. 回す（約 1.6〜2 時間・4 並列）

```bash
nohup env OPCG_LOG_SILENT=1 python tests/scripts/rollout_teach.py play --shard K/30 --workers 4 > ~/rt_sKK.log 2>&1 &
```

- 背景で回し、**9 分以内の待ちを繰り返して** `~/rt_sKK.log` を見張る（`PLAY_DONE` が出たら終わり）。ターンを閉じない。
- 進み具合は 10 局ごとに `[n/834 局] 盤面 …・失敗 …・残り見込み … 分` と出る。
- 落ちたら同じコマンドを打ち直せば、`n1_results/rollout_teach/rows_sKK.jsonl` に書けた盤面を飛ばして続きから回る。

## 3. 確かめる

```bash
wc -l n1_results/rollout_teach/rows_sKK.jsonl                           # 3332 か 3336（833／834 局 × 4）
grep -c '"ok": false' n1_results/rollout_teach/rows_sKK.jsonl           # 失敗の数（0 が普通・数十までは可）
```

失敗が 1 割を超えたら push の前にコーディネータへ知らせる。

## 4. 残す（push）

`RESULT.json`（リポジトリ直下）を次の形で書き、結果だけをコミットして push する:

```json
{"wp": "rollout-teach", "shard": "K/30", "rows": <行数>, "fail": <失敗の数>, "sec": <PLAY_DONE の秒>,
 "files": ["n1_results/rollout_teach/rows_sKK.jsonl", "n1_results/rollout_teach/logs/rt_sKK.log"]}
```

```bash
mkdir -p n1_results/rollout_teach/logs && cp ~/rt_sKK.log n1_results/rollout_teach/logs/
git add RESULT.json n1_results/rollout_teach/rows_sKK.jsonl n1_results/rollout_teach/logs/rt_sKK.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "rollout-teach シャード KK/30 の打ち続け"
git push -u origin claude/rollout-teach-sKK            # 失敗したら 2・4・8・16 秒あけて最大 4 回
```

結果ブランチは **`claude/rollout-teach-sKK`**（零詰め 2 桁）。PR は作らない・マージしない。コミットにモデル名を書かない。
