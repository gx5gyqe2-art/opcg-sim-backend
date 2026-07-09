# 蒸留v2 ワーカープロンプト（単発・専用）

新しいセッションにこの内容を貼る。このセッションは **蒸留v2（出荷v1教師→v3生徒・97多様分布）のデータ生成と
生徒訓練を1本回すだけ**の専用ワーカー。測定は司令塔（別セッション）が成果物枝から回収して行う。
背景: 蒸留パイロット（青2千ゲーム）で生徒0.812を実証済み（`docs/orchestrator_handoff.md` §2 最下部）。
v2は 97リーダー多様×10,000ゲームで「出荷v1級の汎用v3生徒」を作る。

---

## プロンプト本文（ここから貼る）

あなたは蒸留v2のワーカーです。以下を順に実行してください。**途中でエラーが出ても checkpoint/コード枝を
掃除・リセットしないこと**。

### 1. リポジトリ準備
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
```

### 2. 自己検証（❌なら起動せず司令塔に報告）
```bash
PYTHONPATH=tests python3 -c "
import sys; sys.path.insert(0,'tests')
import _bootstrap, rl_net
assert hasattr(rl_net.ValueNet(10,4,8,96), 'eff_dim'), '❌ v3コード枝に居ない'
import os
assert os.path.exists('tests/scripts/distill2_gen.py'), '❌ 蒸留スクリプトが無い'
print('✅ 蒸留v2コード確認')"
```

### 3. データ生成（~3時間・250ゲーム毎にシャード保存＝再開可能）
```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/distill2_gen.py \
  --games 10000 --shard 250 --outdir /home/user/distill2_data 2>&1 | tee /home/user/distill2_gen.log
```
- 進捗は `shard N/40` で出る。**プロセスが死んだら同じコマンドで再実行**（既存シャードはスキップ＝続きから）。
- 40シャード完走で `GENERATION_COMPLETE` が出る。

### 4. 生徒訓練＋成果物push（~1〜2時間）
```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/distill2_train.py \
  --outdir /home/user/distill2_data --epochs 4 \
  --push-branch claude/distill2-artifacts 2>&1 | tee /home/user/distill2_train.log
```
- `TRAIN_COMPLETE` と `成果物push: claude/distill2-artifacts` が出れば完了。
- 成果物枝には生徒net（value/policy）とmanifestだけが載る（データ本体は載せない）。

### 5. 報告
「**蒸留v2完了（states=◯◯・val_mse=◯◯・成果物push済み）**」と報告して終了。
測定（多様97＋青の対L1）は司令塔が成果物枝を検知して自動で行う。

---

## 設計メモ
- 生成＝出荷v1(value+policy・sims40・Dirichlet/温度)の自己対戦を**全97リーダー**で回し、各局面に
  (v3符号化, 出荷v1のvalue生予測) を記録。教師ラベルの分散が±1勝敗より遥かに小さい＝教師あり回帰が効く。
- 生徒種は git 上のv3種（`claude/p3-v3-blue-checkpoints:p3ckpt/gen0_value.npz`）から自動取得＝恒久。
- 比較バー: 出荷v1 多様=0.833／青=0.938。パイロット生徒（青のみ2千ゲーム）=青0.812。
