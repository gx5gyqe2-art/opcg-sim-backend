# 追い学習（postdistill）ワーカープロンプト — 残タスク(b)・単発専用

新しいセッションにこの内容を貼る。**蒸留v2生徒（多様97=0.771/青=0.896）を出発点に、自身+MCTS sims160 の
自己対戦で追い学習し「出荷v1(0.833)超え」を狙う**訓練を1本回す専用ワーカー。測定は司令塔が行う。

> 機序: 蒸留は写経＝教師(v1)を超えない。しかし教師はv1エンコード＝**v3の追加特徴（ターン1・山札・EffFeat細部）に
> 信号を載せられない**（使いこなしプローブで「ターン1フラグは休眠」を確認済み）。追い学習は sims160 の探索増幅値
>（生netより強い教師信号）と実戦勝敗で、教師に見えなかった特徴を活性化させる＝**教師超えの唯一の経路**。

---

## プロンプト本文（ここから貼る）

あなたは追い学習（postdistill）の訓練ワーカーです。以下を順に・省略せず実行してください。
重要: **checkpoint を勝手に掃除しない・ガード環境変数2つと --enc-version 3 を必ず付ける・
OPCG_LEADER_COLORS は設定しない（97多様が評価土俵）・sims は160（追い学習の本体）**。

### 1. リポジトリ準備
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
rm -rf /tmp/pd97-wt
```

### 2. コード枝＋種の自己検証（❌が出たら起動せず司令塔に報告）
```bash
PYTHONPATH=tests python3 - <<'PY'
import sys; sys.path.insert(0,"tests")
import _bootstrap  # noqa
import rl_net as RN
assert hasattr(RN.ValueNet(10,4,8,96), "eff_dim"), "❌ v3コード枝に居ない"
import subprocess, io
subprocess.run(["git","fetch","origin","claude/p3-postdistill97-checkpoints","-q"])
raw = subprocess.check_output(["git","show","origin/claude/p3-postdistill97-checkpoints:p3ckpt/value.npz"])
v = RN.ValueNet.load(io.BytesIO(raw))
assert v.lead_slots == 2 and v.eff_dim == 116, f"❌ 種が蒸留生徒でない (lead={v.lead_slots} eff={v.eff_dim})"
print("✅ v3コード枝＋蒸留生徒種を確認")
PY
git show origin/claude/p3-postdistill97-checkpoints:p3ckpt/manifest.json   # cum_games=0 のはず
```
**どんなエラーが出ても checkpoint 枝を掃除・リセットしない。**

### 3. 追い学習を起動（sims160・低LR・低ノイズ＝フィネチューン設定）
```bash
OPCG_P3_LEAD_SLOTS=2 \
OPCG_P3_EFF_DIM=116 \
OPCG_P3_WT=/tmp/pd97-wt \
OPCG_P3_BRANCH=claude/p3-postdistill97-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 3 --rotate-leaders --shard-games 32 --sims 160 --workers 4 \
  --lr 2e-4 --dirichlet-eps 0.15 --target 100000000 --max-shards 100000000
```
（sims160=25s/局/コア実測 → 32局シャード≈3.5分＝push間隔は従来並み。lr2e-4/eps0.15 は
「良い出発点を壊さず・探索の質を教師にする」ためのフィネチューン設定。）

### 4. 起動確認と報告
```bash
cat /tmp/pd97-wt/p3ckpt/manifest.json
python3 -c "import sys;sys.path.insert(0,'tests');import _bootstrap,rl_net;v=rl_net.ValueNet.load('/tmp/pd97-wt/p3ckpt/value.npz');print('lead=',v.lead_slots,' eff=',v.eff_dim)"
```
- 起動ログ「リーダーローテーション ON: **97 種**」「再開: gen=0 cum_games=0 ... enc=v3」を確認。
- 確認できたら「**追い学習、開始（cum=◯◯・97種・sims160）**」と報告する。

### 5. 運用
- 連続モード＝止まらない。push間隔~4分。測定は司令塔（**cum≈2,000 で早期判定**＝劣化していたら司令塔が停止を指示する）。
- プロセス死は同コマンドで resume（ガード変数を忘れず）。checkpoint枝の掃除・forceリセットは絶対にしない。

---

## 設計メモ（司令塔用）
- **バー**: 生徒(出発点)=多様97 **0.771**／出荷v1=**0.833**／写経上限を超えたら(b)成立。
- **測定計画**: cum≈2,000（早期・劣化ガード）→ 5,000 → 10,000 で多様97 vs L1（pairs12）。
  併せて**マーク再評価**（@3 が ATTACH_DON へ転じるか＝休眠特徴活性化の決定的証拠）と
  **ターン1フラグ感度**（使いこなしプローブ再実行・休眠→使用に転じるか）。
- **中止基準**: cum≈2,000 時点で 0.70 未満（出発点0.771から有意劣化）→ 停止し、ラベルを勝敗から
  **MCTS改善値（root値ブレンド）**に替えるフェーズ2（要ハーネス小改修）へ。
- 種: `claude/p3-postdistill97-checkpoints`（`0980583`・distill2-artifacts の生徒そのまま・恒等）。
