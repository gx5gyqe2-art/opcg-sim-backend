# 色クラスタ分散学習 ワーカー共通プロンプト

新しいセッションにこの内容を貼る（またはこのファイルを指して「これを読んで従って」と指示する）。
各セッションが**空いている1色を自動で選んで担当**し、連続学習を回す。セッション間はコンテナ隔離のため、
**協調は git origin 経由のみ**（＝各色の checkpoint 枝の進行状況で判断する）。

> **運用の要**: セッションは**1つずつ開き、各セッションが「担当色＝X、学習開始」と報告してから次を開く**。
> これで色の二重取りを確実に防げる（コンテナ隔離のためロックは git の進行状況しか無い）。

---

## プロンプト本文（ここから貼る）

あなたは分散CPU学習の1ワーカーです。赤/緑/青/紫/黒/黄の6色クラスタごとに別セッションが1つずつ担当し、
弱いネットから自己対戦で連続学習します。あなたの仕事は「**まだ誰も学習していない空き色を1つ選び、担当になって
連続学習を起動し、走り続ける**」こと。以下を順に実行:

### 1. リポジトリ準備
```bash
cd /home/user/opcg-sim-backend && git fetch origin -q && git checkout claude/opcg-cluster-learning && git pull -q
```

### 2. 各色の稼働状況を調べる
cum_games が進んでいる色＝誰かが担当中。全色を読み、**90秒待ってもう一度読み、増えた色＝稼働中（取られている）**。
```bash
COLORS="red green blue purple black yellow"
snap() { for c in $COLORS; do git fetch origin claude/p3-cluster-$c-checkpoints -q 2>/dev/null; \
  v=$(git show origin/claude/p3-cluster-$c-checkpoints:p3ckpt/manifest.json 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("cum_games","?"))'); \
  echo "$c cum=$v"; done; }
echo "=== 1回目 ==="; A=$(snap); echo "$A"; echo "=== 90秒待つ ==="; sleep 90; echo "=== 2回目 ==="; B=$(snap); echo "$B"
```
`A` と `B` を比べ、**cum_games が増えていない色＝空き**、増えている色＝稼働中（担当済み）。

### 3. 空き色を1つ選ぶ
空き色のうち **cum_games が最小**（最も手付かず）を優先。同点は red>green>blue>purple>black>yellow の順。
**空きが無ければ「6色すべて担当済み」と報告して停止**。

### 4. 連続学習を起動（バックグラウンド）
色→日本語の対応: **red=赤 green=緑 blue=青 purple=紫 black=黒 yellow=黄**。
選んだ色を `<c>`（英語）・`<jp>`（日本語）として:
```bash
OPCG_LEADER_COLORS=<jp> \
OPCG_P3_WT=/tmp/cluster-<c>-wt \
OPCG_P3_BRANCH=claude/p3-cluster-<c>-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --workers 4 \
  --target 100000000 --max-shards 100000000
```
（例: 緑を担当するなら `OPCG_LEADER_COLORS=緑 OPCG_P3_WT=/tmp/cluster-green-wt OPCG_P3_BRANCH=claude/p3-cluster-green-checkpoints ...`）

### 5. 起動確認と報告
5分以内に manifest.shards が増える（=走り出した）ことを確認:
```bash
cat /tmp/cluster-<c>-wt/p3ckpt/manifest.json
```
確認できたら「**担当色=<c>、学習開始（shards進行中）**」と報告する。

### 6. 運用（走り続ける）
- 連続モード（target/max-shards 実質無限）なので**世代境界で止まらない**。各shardで value.npz を checkpoint 枝へ force-push＝回収耐性あり。
- **コンテナ再起動でプロセスが死んだら、同じ起動コマンドで resume**（checkpointから続きを自動再開）。プロセス停止を検知するウォッチャを仕掛けておくとよい。
- 強度確認は後日: `value.npz` を凍結コピー→ その色のデッキで vs-L1（`p3_vs_l1.py --value-path <凍結> --rotate-leaders --sims 160`）。目標＝出荷v1のその色での性能超え。

---

## 設計メモ（担当者向け）
- **弱い出発点**: 各色の checkpoint 枝には v1由来のSLネットをv2温スタートした「弱いGen0」が置いてある（`gen0_value.npz`＝`value.npz`）。`load_nets` がこれを拾うので出荷強ネットからは始まらない（＝v1と同じ伸びしろ）。
- **狭い分布**: `OPCG_LEADER_COLORS` でその色を含むリーダーだけに絞る（赤31/緑23/青27/紫24/黒22/黄24種）。97全部より狭い＝v1的な速い climb を狙う。
- **隔離**: `OPCG_P3_WT`/`OPCG_P3_BRANCH` で checkpoint を色ごとに分離＝互いに衝突しない。
- **並列の本体**: 別セッション＝別コンテナ＝別CPU。だから6色まで真に並列に回せる（同一セッション内では4コア競合で不可）。
