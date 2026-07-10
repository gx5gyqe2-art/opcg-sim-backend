# バッチ式アクター/ラーナー分離ハーネス 設計・運用（自己対戦の並列化）— 2026-07-10

追い学習(b)などの**オンライン自己対戦**は「シャードN更新→N+1生成」の直列フィードバックループで、1本の run
としては並列化できない（コンテナ内4コア並列が上限）。本ハーネスは、これを**ラウンド制のアクター/ラーナー分離**に
組み替え、**生成を独立作業にして複数セッション（別コンテナ）で完全並列化**する。凍結教師の並列生成は蒸留v2で
実証済み（`distill2_gen.py`）で、その形をオンライン学習へ一般化したもの。

## 1. 仕組み（git 協調・単独writer原則）

```
  [generator w1]─┐   各generatorは「現netで凍結生成→自分のdata枝へpush」を繰り返す（別コンテナ・完全並列）
  [generator w2]─┼─→ data枝群 ──→ [learner]（1本）: 新鮮バッチを集めて1ラウンド学習→net枝へpush
  [generator w3]─┘        ↑__________________________|  generatorは次周でnet枝を再ロード＝更新に追従
```

- **net枝**（`claude/p3-pd-net`・learner 単独writer）: `p3ckpt/{value.npz, policy.npz, manifest.json}`。
  manifest = `{round, cum_games, consumed{wid:batch_id}}`。
- **data枝**（`claude/p3-pd-data-<wid>`・そのgenerator 単独writer）: `p3data/{batch.npz, meta.json}`。
  meta = `{worker, batch_id（generator内で単調増加）, against_round（生成に使ったnetのround）, games, states}`。
- **単独writer原則**: 各枝の書き手は1人だけ＝`commit --amend`+`push --force` が安全（LC事故＝多重writerの
  force-push衝突を構造的に回避）。generatorを増やすほどdata枝が増えるだけで衝突しない。

## 2. off-policy 制御（鮮度フィルタ）

learner はバッチ採用を2条件で絞る（`pd_batch_common.is_fresh`・単体テスト済み）:
1. **未消費**: `batch_id > consumed[wid]`（同じバッチを二度学習しない＝冪等）。
2. **十分新鮮**: `against_round >= round - max_staleness`（既定3）。古すぎる off-policy データは捨てる（log）。

これで「生成が数ラウンド遅れたnetで打たれても、遅れが小さければ使う／大きければ捨てる」を保証。AlphaZero系の
アクター/ラーナー分離と同じ off-policy 許容で、数ラウンドの遅れは実害が無いことが知られている。

## 3. ファイル

| ファイル | 役割 |
|---|---|
| `tests/scripts/pd_batch_common.py` | 純粋協調ロジック（is_fresh/plan_consumption/update_consumed/ring_append）＝git非依存・単体テスト対象 |
| `tests/scripts/pd_setup.py` | net枝＋data枝N本を種付け（司令塔が1回） |
| `tests/scripts/pd_gen.py` | アクター: 現netで並列自己対戦（p3_run の _gen_task/selfplay_shard 再利用）→data枝push。**複数セッションで並列** |
| `tests/scripts/pd_learn.py` | ラーナー（1本）: 新鮮バッチ集約→低LR学習→net枝push・consumed更新 |
| `tests/test_pd_batch_common.py` | 純粋ロジックの回帰（accept/seen/stale・消費単調性・リングバッファcap） |

ガードは既存の `OPCG_P3_LEAD_SLOTS` / `OPCG_P3_EFF_DIM`（p3_run.load_nets）を流用＝種取り違えは起動前に停止。

## 4. 起動手順（本番）

**① 種付け（司令塔・1回）** — postdistill生徒(or 任意のv3種)を net枝の起点に:
```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_setup.py \
  --net-branch claude/p3-pd-net \
  --seed-ref origin/claude/p3-postdistill97-checkpoints:p3ckpt \
  --data-branches claude/p3-pd-data-w1,claude/p3-pd-data-w2,claude/p3-pd-data-w3
```

**② generator（別セッションを N 本・並列）** — 各自 DATA_BRANCH を変える:
```bash
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
OPCG_PD_NET_BRANCH=claude/p3-pd-net OPCG_PD_DATA_BRANCH=claude/p3-pd-data-w1 \
OPCG_PD_WT=/tmp/pd-w1 OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_gen.py \
  --enc-version 3 --sims 160 --games 128 --workers 4 --dirichlet-eps 0.15
```
（w2 は `-w2`・`/tmp/pd-w2`、w3 は `-w3`… と変えるだけ。プロンプト: `docs/pd_gen_worker_prompt.md`）

**③ learner（司令塔セッションで1本）**:
```bash
OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
OPCG_PD_NET_BRANCH=claude/p3-pd-net \
OPCG_PD_DATA_BRANCHES=claude/p3-pd-data-w1,claude/p3-pd-data-w2,claude/p3-pd-data-w3 \
OPCG_PD_WT=/tmp/pd-learn OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_learn.py \
  --enc-version 3 --lr 2e-4 --epochs 2 --buffer 60000 --min-new 300 --max-staleness 3
```

**測定**: net枝 `p3ckpt/value.npz`+`policy.npz` を凍結し `p3_vs_l1`（従来どおり）。cum_games は manifest 参照。

## 4b. 薄まり防止（1局あたりの勾配露出を K に依らず一定に保つ）

学習を決めるのは**局数でなく更新回数(勾配ステップ)**。素朴に「1波=1ラウンド」だと、並列で K 本ぶんの
データが一度に来ても更新1回＝**1局あたりの勾配露出が K 分の1に薄まる**（さらに buffer cap を超えた
ぶんは学習前に溢れる）。これを防ぐため learner は:
- **`--games-per-update`（既定128＝1バッチの games）ごとに学習1ラウンド**を回す＝新規games に比例して
  学習回数(epoch)をスケール（`pd_batch_common.updates_for`・単体テスト済）。K=1 なら 1ラウンド＝従来と同一、
  K=6 なら 6ラウンド＝直列の games:updates 比を維持。`--max-updates-per-round`(既定16)で暴発を抑止。
- **`round` は net バージョン数（1push=1版・staleness基準）**、学習量の真の指標は manifest の
  **`updates`（累積勾配パス）**。監視は cum_games と updates の両方を見る。
- buffer 既定を 120,000 に拡大（K≈8バッチを収容）。1波が buffer を超えたら warn（generator数/バッチを下げる合図）。

**結論**: この scaling を入れると、並列は「速いが薄い」ではなく**「直列と同じ per-game 学習量のまま K 倍速い」**に
なる。逆に scaling 無しだと到達点が変わらない（or 悪化）＝並列の意味が消える。

## 5. スループット

generator 1本 = コンテナ内4コア = 現行 p3_run 1本と同等（sims160 で ~1 g/s級）。generator を K 本並列に
すれば**生成が実質 K 倍**。learner の学習は軽い（数百局面/ラウンド・2 epoch）ので律速にならない。
＝「本判定10kまで18時間」が generator 3本で ~6時間級に短縮できる。

## 6. スモーク（検証済み・2026-07-10）

ローカル bare リポジトリ（file://・ネットワーク不要）で end-to-end 疎通:
- pd_setup: net枝＋data枝2本 種付けOK
- pd_gen w1: batch0（460局面・against_round=0）を data枝へ push
- pd_learn: w1採用→1ラウンド学習→net枝 round 0→1・consumed{w1:0}・cum_games更新 push
- 冪等性: 新データ無しで再実行→round=1のまま（消費済み再学習せず）
- 鮮度追従: learner が round1 に進んだ後、generator w2 は against_round=1 で生成（更新に追従）

## 6b. 運用前レビューで塞いだ穴（2026-07-10・全てe2e検証済み）

1. **generator再起動で batch_id が0に戻る** → consumed 未満のIDが全部「seen」で黙って捨てられ生成が無駄に。
   → 起動時に自分の data枝 meta.json から **batch_id を復元**（`再開batch_id=N` ログ）。
2. **learner停止中の上書き全損**: data枝は最新1バッチのみ保持＝learnerが止まると amend+force で前バッチが消える。
   → **バックプレッシャ**（`--pipeline-depth` 既定2）: 未消費バッチが depth 本を超えたら生成を待機
   （`should_generate`・待機ログあり）。learner未稼働でも depth 本までは先行生成できる。
3. **policyの凍結**（旧実装は pol を捨てていた＝直列と挙動乖離）→ バッチに同梱し learner が毎ラウンド学習
   （pack/unpack_policy・往復単体テスト＋net枝の policy.npz ハッシュ変化で確認）。
4. 軽微: net ロード時の db/vocab 再構築を排除（vocab をループ外でキャッシュ）。

**受容したリスク（把握済み・実害小）**:
- learner 再起動でメモリ上のリプレイバッファは失われる（consumed は manifest で永続＝二重学習はしない。
  バッファは新規バッチで再構築される＝数ラウンドだけ学習の質が薄い）。
- generator の push 失敗時は batch_id を進めて続行（IDに欠番＝learnerは単調性で問題なく処理・そのバッチは損失）。
- data枝サフィックス（w1/w2…）の一意性は運用規約（重複させると衝突）＝ワーカープロンプトで番号を司令塔が配布。

## 7. スコープ外・注意

- learner は1本厳守（net枝の単独writer）。2本立てると force-push で衝突する。
- data枝は generator と1:1。使い回すと衝突。
- policy は generator がバッチに同梱（pack_policy）し learner が毎ラウンド学習＝**直列(p3_run)とパリティ**。
- 極端な staleness（generatorが大量遅延）時は stale 破棄ログが増える＝learner を一時停止して generator を
  追いつかせるか、max-staleness を上げる。
