# LC青パイロット ワーカープロンプト（単発・専用）

新しいセッションにこの内容を貼る（またはこのファイルを指して「これを読んで従って」と指示する）。
このセッションは **LC-ValueNet（リーダー条件付け）の青パイロット訓練を1本回すだけ**の専用ワーカー。
司令塔（別セッション）が git 経由で軌跡を測定・判定するので、あなたは**起動して走り続ける**ことに専念する。

> 背景（なぜやるか）: 現行の汎用CPUは、value ネットがカード埋め込みを平均プールで潰しマッチアップを条件付け
> できないのが天井の正体（`docs/orchestrator_handoff.md` §2）。その最小修正＝リーダー埋め込みを専用枠で直結した
> LC-ValueNet を、最も劣化した青（legacy: gen0 0.417→538局で 0.208）で検証する。設計は
> `docs/reports/lc_value_net_plan_20260708.md`。

---

## プロンプト本文（ここから貼る）

あなたは LC青パイロットの訓練ワーカーです。以下を**順に・省略せず**実行してください。
**重要（2026-07-08 の事故対策）**: 前回、LCコード枝に居ないワーカーが次元不一致エラーで checkpoint を掃除し、
legacy net を silently 訓練して実験が無効になりました。今回は下記のガードを必ず守ること＝
**checkpoint を勝手に掃除しない・`OPCG_P3_LEAD_SLOTS=2` を必ず付ける・LCコード枝を検証してから起動する**。

### 1. リポジトリ準備（LCコード枝が必須）
種netは lead_slots=2 なので、**LC実装が入った枝で回さないと読めません**。必ずこの枝に切り替えること。
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
rm -rf /tmp/lc-blue-wt          # 前回の残骸worktreeを掃除（古い状態を掴まないため）
```

### 2. LCコード枝であることを自己検証（ここで失敗したら起動しない）
`lead_slots` 実装が入っていること・種が LC（lead_slots=2）であることを機械的に確認する。
```bash
PYTHONPATH=tests python3 - <<'PY'
import sys; sys.path.insert(0,"tests")
import _bootstrap  # noqa
import rl_net as RN
assert hasattr(RN.ValueNet(10,4,8,96), "lead_slots"), "❌ LCコード枝に居ない（value_net に lead_slots が無い）"
import subprocess, io
subprocess.run(["git","fetch","origin","claude/p3-lc-blue-checkpoints","-q"])
raw = subprocess.check_output(["git","show","origin/claude/p3-lc-blue-checkpoints:p3ckpt/value.npz"])
v = RN.ValueNet.load(io.BytesIO(raw))
assert v.lead_slots == 2, f"❌ 種が LC でない (lead_slots={v.lead_slots})＝司令塔に報告"
print("✅ LCコード枝＋LC種を確認（value.npz lead_slots=2）")
PY
git show origin/claude/p3-lc-blue-checkpoints:p3ckpt/manifest.json   # {"gen":0,"cum_games":0,...} のはず
```
- `❌` が出たら**起動せず司令塔に報告**する。`cum_games` が 0 でなければ二重起動＝止めて確認。
- **どんなエラーが出ても checkpoint 枝を掃除・リセットしない**（種を消すと事故が再発する）。

### 3. 連続学習を起動（LC ガード付き）
`OPCG_P3_LEAD_SLOTS=2` を**必ず**付ける（legacy を読んだら黙って走らずエラー停止する安全弁）。
```bash
OPCG_LEADER_COLORS=青 \
OPCG_P3_LEAD_SLOTS=2 \
OPCG_P3_WT=/tmp/lc-blue-wt \
OPCG_P3_BRANCH=claude/p3-lc-blue-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --workers 4 \
  --target 100000000 --max-shards 100000000
```
（案Bの青と**完全同条件**。変えたのは net アーキ（lead_slots=2）だけ＝legacy青と直接比較できる。）

### 4. 起動確認と報告
5分以内に manifest.shards が増える（=走り出した）ことを確認:
```bash
cat /tmp/lc-blue-wt/p3ckpt/manifest.json
python3 -c "import sys;sys.path.insert(0,'tests');import _bootstrap,rl_net;print('走行中netの lead_slots=', rl_net.ValueNet.load('/tmp/lc-blue-wt/p3ckpt/value.npz').lead_slots)"
```
- 起動ログに `再開: gen=0 cum_games=0 ... enc=v2` が出て、**エラーで止まっていないこと**を確認。
- `走行中netの lead_slots= 2` を確認（**0 なら legacy＝即停止して司令塔に報告**）。
- 確認できたら「**LC青、学習開始（shards進行中・cum=◯◯・lead_slots=2）**」と報告する。

### 5. 運用（走り続ける）
- 連続モード（target/max-shards 実質無限）＝世代境界で止まらない。各shardで value.npz を checkpoint 枝へ
  約2.5分ごとに force-push＝回収耐性あり。**強度測定は司令塔がやる**ので、あなたは測定しなくてよい。
- **プロセスが死んだら同じ起動コマンドで resume**（checkpoint から自動継続・`OPCG_P3_LEAD_SLOTS=2` を忘れず）。
  停止検知ウォッチャを仕掛けておくとよい。**checkpoint 枝の掃除・force リセットは絶対にしない**。
- 司令塔から「cum=14,520 まで到達したら/判定が出たら止めて」と言われたら停止する。**それまでは回し続ける**。

---

## 設計メモ（担当者向け）
- **種netの正体**: 共通弱Gen0（SHA1 92ae0c1f）を `ValueNet.to_leader_conditioned()` で lead_slots=2 化したもの
  （W1末尾に自/相手リーダー埋め込み専用枠48行をゼロ追加＝**拡張直後は弱Gen0と完全恒等**）。policyなし=uniform開始。
- **legacy青との比較点**: legacy青は cum=14,520（538局/リーダー）で対L1 **0.208**（gen0は 0.417）。
  LCが同じ点で **0.42以上を保てば「劣化を止めた」・0.55以上なら「アーキが効いた」**（`lc_value_net_plan_20260708.md` §4-5）。
- **隔離**: `OPCG_P3_WT=/tmp/lc-blue-wt`・`OPCG_P3_BRANCH=claude/p3-lc-blue-checkpoints` で他runと衝突しない。
- **競合ゼロ**: このセッション＝別コンテナ＝別CPU。司令塔の測定（別コンテナ）とCPUを食い合わない。
