# Cloud Run ジョブで再分析とアリーナを回す（手順書）

出発点: ユーザ決定 2026-10-17「(c) 計算の短縮を先に」（台帳 `docs/cpu_theory_gap.md` 2+242）。再分析（`opcg_sim/loop/reanalyze.py`）とアリーナ（`opcg_sim/loop/arena_shard.py`）を、Google Cloud の **Cloud Run ジョブ**で多数のタスクに並べて回す。

- **入口**: `opcg_sim/loop/cloud_job.py`。
- **イメージ**: `Dockerfile.job`。
- **シャードの割り当てのテスト**: `tests/test_cloud_job.py`。

**状態（2026-10-17）**: 認証がまだ無いので、実際のデプロイはしていない。コード・この手順書・このコンテナでの模擬実行（§9）まで。コマンドはそのまま貼れる形にしてあり、プロジェクト名などは §1 の変数で置き換える。

---

## 0. 仕組み

- Cloud Run ジョブは同じイメージを `--tasks N` 個並べて走らせ、各タスクに `CLOUD_RUN_TASK_INDEX`（0..N−1）と `CLOUD_RUN_TASK_COUNT`（N）を渡す。
- 入口はそれを読んで自分の分を決め、既存の器をそのまま子プロセスで呼ぶ:
  - **再分析**: `reanalyze.py --shard i/N`（局を N 分割した i 番目・器の既定の分け方）。
  - **アリーナ**: 帯の計画（`arena.plan_bands`）を通しで N 等分し、自分の分を「連続した seed の区間」ごとに `arena_shard --bands 1 --seed-base <区間の先頭> --pairs <長さ>` で呼ぶ。全タスクを合わせると、1 本で回したときと同じ seed を同じ設定で打つ。帯からはみ出さず、タスクの間で重ならない（`tests/test_cloud_job.py`）。
- **入力**（記録の dir・ネット）はバケットからタスクの作業場所へ落とす。
- **出力**はバケットの `<out>/task-<i>-of-<N>/` に置く（タスクごとに衝突しない・N を名前に入れる＝タスクの数を変えて回し直しても混ざらない）。
- **冪等**: 始める前に自分の出力を落とし直す。器はどちらも「終えた局・ペアを飛ばす」再開の作りなので、再試行（`--max-retries`）で同じタスクがもう一度走っても、終えた分はやり直さない。走っている間も `--sync-every` 秒（既定 300）ごとに出力を上げる。
- **合算**は手元で既存の器を使う（`arena_merge`・`reanalyze --merge`／`--summarize`）。§6。

## 1. 前提と変数

```bash
export PROJECT=my-gcp-project          # プロジェクト ID
export REGION=asia-northeast1          # 地域（東京）。費用は地域で変わる（§7）
export REPO=opcg                       # Artifact Registry のリポジトリ名
export BUCKET=my-opcg-bucket           # Cloud Storage のバケット名（gs:// は付けない）
export JOB_SA=opcg-job                 # ジョブが動くサービスアカウント
export TAG=$(git rev-parse --short HEAD)
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/opcg-job
gcloud config set project $PROJECT
```

### 1.1 有効にする API（1 回だけ）

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com storage.googleapis.com
```

### 1.2 権限の最小集合

| 誰 | 役割 | 何のため |
|---|---|---|
| 手で操作する人（ユーザ・鍵を入れるアカウント） | `roles/run.developer` | Cloud Run ジョブの作成・更新・実行・削除 |
| 同上 | `roles/artifactregistry.writer`（リポジトリに付ける） | イメージの push |
| 同上 | `roles/iam.serviceAccountUser`（`$JOB_SA` に付ける） | ジョブを `$JOB_SA` で動かす |
| 同上 | `roles/cloudbuild.builds.editor` | `gcloud builds submit` でビルドするときだけ（手元の docker で push するなら不要） |
| ジョブのサービスアカウント `$JOB_SA` | `roles/storage.objectAdmin`（**バケットに付ける**・プロジェクト全体には付けない） | 入力の読み・出力の書き（同じ名前の上書き＝冪等の再アップロードに要る） |
| 同上 | `roles/artifactregistry.reader`（リポジトリに付ける） | イメージの pull（同じプロジェクトなら既定で足りることが多い） |

```bash
# リポジトリ・バケット・サービスアカウント（1 回だけ）
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION
gcloud storage buckets create gs://$BUCKET --location=$REGION --uniform-bucket-level-access
gcloud iam service-accounts create $JOB_SA --display-name="opcg cloud run jobs"
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
    --member=serviceAccount:$JOB_SA@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
gcloud artifacts repositories add-iam-policy-binding $REPO --location=$REGION \
    --member=serviceAccount:$JOB_SA@$PROJECT.iam.gserviceaccount.com --role=roles/artifactregistry.reader
```

### 1.3 認証（鍵を環境変数で入れる場合）

```bash
# サービスアカウントの鍵（JSON）の置き場を環境変数で渡す
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud auth configure-docker $REGION-docker.pkg.dev    # 手元の docker で push するとき
```

（ジョブの中は `$JOB_SA` の権限で動くので、ジョブに鍵は渡さない。）

## 2. 入力をバケットに置く

```bash
# 記録（record_gen の出力 dir）とネット
gcloud storage cp -r n_records/n29_w01 gs://$BUCKET/opcg/records/
gcloud storage cp opcg_sim/data/learned/nrel_r11a.npz gs://$BUCKET/opcg/nets/
```

## 3. イメージのビルドと push

### 3.1 アリーナ用（今のコミット）

```bash
gcloud builds submit --tag $IMAGE:arena-$TAG --file Dockerfile.job .     # Cloud Build
# または手元の docker:
# docker build -f Dockerfile.job -t $IMAGE:arena-$TAG . && docker push $IMAGE:arena-$TAG
```

### 3.2 再分析用（記録を作ったコミット）

再分析は記録を作ったコミットの wheel と効果 JSON で打ち直す（新しいエンジンでは手が分岐する）。記録の `meta_n_record.json` に記されたコミット（例: `a28ededa`）の worktree に、今の木から 3 つのファイルを写してビルドする。

```bash
export REC_COMMIT=a28ededa                      # 記録を作ったコミット
git worktree add /tmp/wt_$REC_COMMIT $REC_COMMIT
cp opcg_sim/loop/cloud_job.py opcg_sim/loop/reanalyze.py /tmp/wt_$REC_COMMIT/opcg_sim/loop/
cp Dockerfile.job /tmp/wt_$REC_COMMIT/
(cd /tmp/wt_$REC_COMMIT && gcloud builds submit --tag $IMAGE:rean-$REC_COMMIT --file Dockerfile.job .)
git worktree remove /tmp/wt_$REC_COMMIT
```

- `cloud_job.py` は標準ライブラリと `google-cloud-storage` 以外を import しないので、古い木でも動く。
- `reanalyze.py` は古いコミットにもある口だけを使う作り（器の docstring）。

## 4. ジョブの作成

```bash
# 再分析: 200 局を 50 タスク（1 タスク 4 局・1 局 1 コア）
gcloud run jobs create rean-n29w01 --image $IMAGE:rean-$REC_COMMIT --region $REGION \
    --service-account $JOB_SA@$PROJECT.iam.gserviceaccount.com \
    --tasks 50 --parallelism 50 --cpu 4 --memory 8Gi --task-timeout 3h --max-retries 3 \
    --args="^@^reanalyze@--storage@gs://$BUCKET/opcg@--records@records/n29_w01@--out@rean/n29_w01@--inputs@nets/nrel_r11a.npz@--@--mult@4@--games@200@--quota@4,4,2,2@--net@{inputs}/nets/nrel_r11a.npz@--engine-commit@$REC_COMMIT"

# アリーナ: 2,800 ペアを 100 タスク（1 タスク 28 ペア・4 並列）
gcloud run jobs create arena-r12-vs-r11a --image $IMAGE:arena-$TAG --region $REGION \
    --service-account $JOB_SA@$PROJECT.iam.gserviceaccount.com \
    --tasks 100 --parallelism 100 --cpu 4 --memory 8Gi --task-timeout 3h --max-retries 3 \
    --args="^@^arena@--storage@gs://$BUCKET/opcg@--out@arena/r12_vs_r11a_main@--pairs@2800@--bands@4@--seed-base@71000@--inputs@nets/nrel_r12.npz@--@--candidate@{inputs}/nets/nrel_r12.npz@--leaders@random@--decks@synth_roles@--workers@4"
```

- `--args` は既定ではカンマ区切りで、カンマを含む引数（`--quota 4,4,2,2`）が割れる。そのため、上の例は区切りを `@` に変えてある（先頭の `^@^`・`gcloud topic escaping`）。
- `--` の後ろは器（`reanalyze.py`／`arena_shard`）の引数をそのまま渡す。`{inputs}` は入口が作業場所の入力の置き場に置き換える。
- `--parallelism` は地域の割り当て（同時に動く vCPU の上限）を超えると、超えた分は待つ。割り当ては「IAM と管理 → 割り当て」で `Cloud Run Jobs` の CPU を見る。
- 1 タスクのコア数は 4 にして、器の並列（`--workers 4`）と合わせる。再分析は 1 局 1 コアで逐次（器の作り）なので、`--cpu 1 --memory 2Gi` にしてタスクを増やしてもよい。
- `--task-timeout` は 1 タスクの上限（最大 168h）。冪等なので、短めにして再試行に任せてもよい。

## 5. 実行と見張り

```bash
gcloud run jobs execute rean-n29w01 --region $REGION
gcloud run jobs executions list --job rean-n29w01 --region $REGION
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="rean-n29w01" AND textPayload:"CLOUD_JOB_DONE"' --limit 100
```

- 各タスクは `CLOUD_JOB_START`（自分の分・落とし直した出力の数）と `CLOUD_JOB_DONE`（終了の値）をログに出す。
- 落ちたタスクは `--max-retries` まで同じ番号で再試行され、終えた局・ペアは飛ばす。
- 実行全体を回し直すときも同じ `execute` でよい（終えた分は飛ばす）。

## 6. 結果の回収と合算

```bash
python -m opcg_sim.loop.cloud_job collect --storage gs://$BUCKET/opcg --out arena/r12_vs_r11a_main --to /tmp/r12_main
python -m opcg_sim.loop.arena_merge --in "/tmp/r12_main/*/*.jsonl" --label main      # 判定は ARENA_MERGE_FINAL

python -m opcg_sim.loop.cloud_job collect --storage gs://$BUCKET/opcg --out rean/n29_w01 --to /tmp/rean_n29w01
python opcg_sim/loop/reanalyze.py --summarize /tmp/rean_n29w01/task-*                 # 再生率・深い最善 ≠ 記録
python opcg_sim/loop/reanalyze.py --merge /tmp/rean_n29w01/task-* --in n_records/n29_w01 --out /tmp/rean_merged   # r11b 型の教材
```

`arena_merge` は、同じ seed が 2 つのタスクに現れたら判定を出さずに落とす（帯設計のミスの見張り）。この入口の割り当てでは起きない（テストで確かめている）。

## 7. 費用の目安

**定価からの概算で、地域・時期・割引で変わる**（Cloud Run の料金ページで、使う地域の「CPU（vCPU 秒）」「メモリ（GiB 秒）」の単価を確かめてから使うこと）。ここでは CPU を **1 コア時間 約 0.065 ドル**として数える。メモリの課金は別にかかる。無料枠は考えない。

**r11 の 1 サイクル（再分析 ＋ 高分解能アリーナ）の換算**:

| 段 | 量 | 1 単位のコア秒 | コア時間 | CPU の費用 |
|---|---|---:|---:|---:|
| 再分析 | 14 dir × 200 局（＝33,600 点） | 約 95（1 局・12 点・実側のパイロットの実測） | 約 74 | 約 4.8 ドル |
| 高分解能アリーナ | 2,800 ペア（1 条件） | 約 40（1 ペア・このコンテナの模擬実行で 24〜51 秒） | 約 31 | 約 2.0 ドル |
| 合計 | — | — | 約 105 | **約 7 ドル**（＋メモリ） |

- 訓練（`n_rel_train.py`）は Cloud Run ジョブに載せていない（GPU を使わない CPU の訓練で、1 本のプロセス）。
- 壁時計: 100 コアを同時に使えれば、再分析は 1 時間弱、アリーナは 20 分前後（割り当て次第）。
- 1 単位のコア秒は、このコンテナと Cloud Run の CPU の速さの違いで変わる。初回は小さく回して（§8）、ログの秒から測り直す。

## 8. 初回の確かめ（小さく回す）

```bash
gcloud run jobs create arena-smoke --image $IMAGE:arena-$TAG --region $REGION \
    --service-account $JOB_SA@$PROJECT.iam.gserviceaccount.com \
    --tasks 4 --parallelism 4 --cpu 2 --memory 4Gi --task-timeout 30m --max-retries 1 \
    --args="^@^arena@--storage@gs://$BUCKET/opcg@--out@arena/smoke@--pairs@4@--bands@4@--inputs@nets/nrel_r11a.npz@--@--candidate@{inputs}/nets/nrel_r11a.npz@--leaders@random@--decks@synth_roles@--workers@2"
gcloud run jobs execute arena-smoke --region $REGION --wait
gcloud storage ls -r gs://$BUCKET/opcg/arena/smoke/
```

確かめること:

- 4 タスクとも `CLOUD_JOB_DONE` の `rc` が 0。
- 台帳が `task-000i-of-0004/` に 1 本ずつ。
- もう一度 `execute` すると、どのタスクも `消化済み 1/1 ペア・残り 0` で数秒で終わる（冪等）。

## 9. 片付け

```bash
gcloud run jobs delete arena-smoke --region $REGION --quiet
gcloud run jobs list --region $REGION                                  # 残りを確認
gcloud artifacts docker images list $IMAGE                             # 古いタグを消すなら:
# gcloud artifacts docker images delete $IMAGE:arena-<古いタグ> --quiet
gcloud storage rm -r gs://$BUCKET/opcg/arena/smoke/                    # 出力は回収してから消す
```

ジョブそのものは、実行していなければ費用はかからない。費用がかかるのは、イメージの保管（Artifact Registry）とバケットの保管（Cloud Storage）。

---

## 10. 模擬実行の結果（このコンテナ・2026-10-17）

バケットの代わりにローカルの dir（`--storage file:///…/cj/bucket`）を使い、`CLOUD_RUN_TASK_INDEX=0..3`・`CLOUD_RUN_TASK_COUNT=4` を順に与えて回した。

### 10.1 アリーナ（2 ペア・今のエンジン）

候補 `nrel_r3` 対 出荷既定 `nrel_r11a`・`--leaders random --decks synth_roles --workers 2`・`--pairs 2 --bands 4`。

| タスク | 割り当て（seed の区間） | 結果 |
|---|---|---|
| 0 | なし | `CLOUD_JOB_DONE rc=0`（何も打たない） |
| 1 | `[71000, 1]` | 1 ペア 51 秒・台帳 `task-0001-of-0004/pairs_71000_1.jsonl` |
| 2 | なし | `rc=0` |
| 3 | `[171000, 1]` | 1 ペア 24 秒・台帳 `task-0003-of-0004/pairs_171000_1.jsonl` |

- **冪等**: タスク 1 を作業場所を消して再実行すると、`resumed_files 1`・`消化済み 1/1 ペア・残り 0` で 0.2 秒で終わる（打ち直さない）。
- **回収と合算**: `collect` で 2 タスクの台帳を落とし、`arena_merge` で `ARENA_MERGE_FINAL {"pairs": 2, "games": 4, "void": 0, …, "shards": 2, "dup_seeds": 0}`（2 ペアなので勝率に意味は無い）。

### 10.2 再分析（2 局・記録を作った古いエンジン）

記録 w39（a28ededa で生成・300 局）をバケットに置く。古いコミットの worktree に `cloud_job.py`・`reanalyze.py` を写し、その木の wheel（`make rust-develop`）と効果 JSON で、その木を cwd にして回した（§3.2 のイメージと同じ組み立て）。器の引数は `--mult 4 --games 2 --quota 4,4,2,2 --net {inputs}/nets/nrel_r3.npz`。

| タスク | 割り当て（`--shard i/4` × 先頭 2 局） | 結果 |
|---|---|---|
| 0 | seed 9700 | 12 点・失った点 0・再生 11.7 秒・深い探索 9.0 秒 |
| 1 | seed 9701 | 12 点・失った点 0・7 秒 |
| 2・3 | なし | `rc=0`（空の meta と `points.jsonl` だけ） |

- **冪等**: タスク 0 を作業場所を消して再実行すると、`resumed_files 4`・`REANALYZE_START {"done": 1, "todo": 0}` で打ち直さない。
- **回収と集計**: `collect` → `reanalyze.py --summarize` で再生率 1.00・分かれた局 0・24 点・対応しない点 0・深い最善 ≠ 記録 25%（実側のパイロットの 25〜27% と同じ水準）。
- 模擬実行の後、今のエンジンの wheel に戻した。

### 10.3 まだ確かめていないこと（認証が要る）

- `gs://` の読み書き（`google-cloud-storage` の経路）。コードは `file://` と同じ手順で、`gcloud storage` への逃げ道もある。
- Cloud Run の上での CPU の速さ（§7 の 1 単位のコア秒）。
- `--args` の区切りの書き換え（`^@^`）。

いずれも §8 の小さな実行で確かめる。

### 10.4 ユーザに用意してもらうもの

1. プロジェクト ID・地域・バケット名（§1 の変数）。課金を有効にしたプロジェクト。
2. §1.1 の API の有効化。
3. 手で操作するアカウント（または鍵を入れるサービスアカウント）の権限: §1.2 の上 4 行。
4. ジョブ用のサービスアカウント（`$JOB_SA`）と、バケットへの `roles/storage.objectAdmin`。
5. 鍵（JSON）を環境変数 `GOOGLE_APPLICATION_CREDENTIALS` で渡す（このコンテナから操作するなら）。
6. Cloud Run ジョブの CPU の割り当て（同時に使うコア数・既定は地域で数百 vCPU 程度・要るなら引き上げの申請）。
