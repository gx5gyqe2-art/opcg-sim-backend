# pair-diff-loss のシャードの回し方（16 本・1 本 ≒ 2.5〜3 時間）

WP `pair-diff-loss`（`docs/search_efficiency_plan.md` §7.88）の段 1 の教材（ドン ±1 の対の打ち続け）。シャード番号 `K`（1〜16）、2 桁の零詰めを `KK`（例 `07`）とする。
**コードは変えない**。道具は `tests/scripts/pair_diff.py`、盤面の一覧は `n1_results/pair_diff_loss/positions.json.gz`（このブランチに同梱・作り直さない）。
1 本のシャードは約 1,000 局・約 4,100 対。各対で両側から配り直し 16 本を読み 16 回・r28x4 で最後まで打つ（1 対 約 9 コア秒）。

## 1. 用意（約 15 分）

```bash
cd /home/user/opcg-sim-backend            # リポジトリの場所（違えば読み替える）
git fetch origin claude/pair-diff-loss
git checkout -B claude/pair-diff-loss-sKK FETCH_HEAD
pip install -q maturin && pip install -q numpy
make rust-develop                         # エンジンの wheel（数分）
python -c "import opcg_engine"            # 通ること
```

このシャードが使う記録（8 ブランチ・約 130MB）だけを取る（約 3 分）:

```bash
for br in $(python tests/scripts/pair_diff.py need --shard K/16); do
  wv=$(echo $br | sed 's#claude/n\([0-9a-z]*\)-w.*#\1#')     # 例 70f
  mkdir -p ~/n$wv
  for t in 1 2 3 4; do git fetch -q --depth 1 origin $br && break; sleep $((2**t)); done
  git archive FETCH_HEAD n_records | tar -x -C ~/n$wv
done
ls -d ~/n7*f/n_records/*_p[0-3] | wc -l    # 32 であること
```

## 2. 回す（約 2.5〜3 時間・4 並列）

```bash
nohup env OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py play --shard K/16 --workers 4 > ~/pd_sKK.log 2>&1 &
```

- 背景で回し、**9 分以内の待ちを繰り返して** `~/pd_sKK.log` を見張る（`PLAY_DONE` が出たら終わり）。ターンを閉じない。同じコンテナで他の重い処理を並べない。
- 進み具合は 10 局ごとに `[n/… 局] 盤面 …・失敗 …・残り見込み … 分` と出る。
- 落ちたら同じコマンドを打ち直せば、`n1_results/pair_diff_loss/pairs_sKK.jsonl` に書けた盤面を飛ばして続きから回る。

## 3. 確かめる

```bash
wc -l n1_results/pair_diff_loss/pairs_sKK.jsonl                          # 約 3,800〜4,500（`PLAY` の行の「盤面」の数と同じ）
grep -c '"ok": false' n1_results/pair_diff_loss/pairs_sKK.jsonl          # 失敗の数（数十までは普通）
```

失敗が 1 割を超えたら push の前にコーディネータへ知らせる。

## 4. 残す（push）

`RESULT.json`（リポジトリ直下）を次の形で書き、結果だけをコミットして push する（行は大きいので gzip する）:

```json
{"wp": "pair-diff-loss", "shard": "K/16", "rows": <行数>, "fail": <失敗の数>, "sec": <PLAY_DONE の秒>,
 "files": ["n1_results/pair_diff_loss/pairs_sKK.jsonl.gz", "n1_results/pair_diff_loss/logs/pd_sKK.log"]}
```

```bash
gzip -k n1_results/pair_diff_loss/pairs_sKK.jsonl
mkdir -p n1_results/pair_diff_loss/logs && cp ~/pd_sKK.log n1_results/pair_diff_loss/logs/
git add RESULT.json n1_results/pair_diff_loss/pairs_sKK.jsonl.gz n1_results/pair_diff_loss/logs/pd_sKK.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "pair-diff-loss シャード KK/16 の対の打ち続け"
git push -u origin claude/pair-diff-loss-sKK            # 失敗したら 2・4・8・16 秒あけて最大 4 回
```

結果ブランチは **`claude/pair-diff-loss-sKK`**（零詰め 2 桁）。PR は作らない・マージしない。コミットにモデル名を書かない。
