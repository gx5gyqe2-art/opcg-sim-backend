# v3青パイロット ワーカープロンプト（単発・専用）

新しいセッションにこの内容を貼る。このセッションは **効果セマンティクスv3（EffFeat）の青パイロット訓練を1本回すだけ**の
専用ワーカー。強度測定は司令塔（別セッション）が git 経由で行う。LC事故（legacyのサイレント訓練）対策の
ガード（自己検証＋環境変数2つ）を必ず守ること。

---

## プロンプト本文（ここから貼る）

あなたは v3青パイロットの訓練ワーカーです。効果セマンティクスv3（EffFeat）net を青クラスタで検証する訓練を1本回すのが
仕事です。強度測定は司令塔がやるので、あなたは起動して走り続けることに専念してください。
重要: **checkpoint を勝手に掃除しない・環境変数ガード（OPCG_P3_LEAD_SLOTS=2 / OPCG_P3_EFF_DIM=116）を必ず付ける・
`--enc-version 3` を必ず使う**。以下を順に実行:

### 1. リポジトリ準備（v3コード枝が必須）
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
rm -rf /tmp/v3-blue-wt
```

### 2. v3コード枝＋v3種であることを自己検証（❌が出たら起動せず司令塔に報告）
```bash
PYTHONPATH=tests python3 - <<'PY'
import sys; sys.path.insert(0,"tests")
import _bootstrap  # noqa
import rl_net as RN
assert hasattr(RN.ValueNet(10,4,8,96), "eff_dim"), "❌ v3コード枝に居ない（value_net に eff_dim が無い）"
import subprocess, io
subprocess.run(["git","fetch","origin","claude/p3-v3-blue-checkpoints","-q"])
raw = subprocess.check_output(["git","show","origin/claude/p3-v3-blue-checkpoints:p3ckpt/value.npz"])
v = RN.ValueNet.load(io.BytesIO(raw))
assert v.lead_slots == 2 and v.eff_dim == 116, f"❌ 種が v3 でない (lead={v.lead_slots} eff={v.eff_dim})＝司令塔に報告"
print("✅ v3コード枝＋v3種を確認（lead_slots=2, eff_dim=116, hidden=%d）" % v.W1.shape[1])
PY
git show origin/claude/p3-v3-blue-checkpoints:p3ckpt/manifest.json   # cum_games=0 のはず（0以外=二重起動）
```
**どんなエラーが出ても checkpoint 枝を掃除・リセットしない**（種を消すと事故が再発する）。

### 3. 連続学習を起動（ガード2つ＋enc-version 3 必須）
```bash
OPCG_LEADER_COLORS=青 \
OPCG_P3_LEAD_SLOTS=2 \
OPCG_P3_EFF_DIM=116 \
OPCG_P3_WT=/tmp/v3-blue-wt \
OPCG_P3_BRANCH=claude/p3-v3-blue-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 3 --rotate-leaders --shard-games 60 --sims 40 --workers 4 \
  --target 100000000 --max-shards 100000000
```
（案B/LC青と同一条件。変えたのは net アーキ（v3）と符号化（v3）のみ＝legacy/LC と直接比較できる。）

### 4. 起動確認と報告
```bash
cat /tmp/v3-blue-wt/p3ckpt/manifest.json    # shards が増えていること
python3 -c "import sys;sys.path.insert(0,'tests');import _bootstrap,rl_net;v=rl_net.ValueNet.load('/tmp/v3-blue-wt/p3ckpt/value.npz');print('走行中net: lead_slots=',v.lead_slots,' eff_dim=',v.eff_dim)"
```
- 起動ログに `再開: gen=0 cum_games=0 ... enc=v3` が出てエラー停止していないこと・
  `lead_slots= 2  eff_dim= 116` を確認（違えば即停止して司令塔に報告）。
- 確認できたら「**v3青、学習開始（shards進行中・cum=◯◯・eff_dim=116）**」と報告する。

### 5. 運用
- 連続モード＝止まらない。約2.5分ごとに checkpoint 枝へ force-push。**測定は司令塔がやる**。
- プロセスが死んだら同じ起動コマンドで resume（ガード環境変数を忘れず）。checkpoint枝の掃除・forceリセットは絶対にしない。
- 司令塔から「判定が出た/cum=14,520到達で止めて」と言われるまで回し続ける。

---

## 設計メモ（担当者向け）
- 種: 共通弱Gen0 → scalars46 → LC → EffFeat(116) → hidden256 の**恒等連鎖**（実局面 max|Δ|=4e-16）＝弱Gen0と同じ実力から開始。
- バー（青・537局/リーダー・対L1同条件）: gen0 **0.417** / legacy訓練後 **0.208**（崩壊）/ LC訓練後 **0.396**（劣化停止）。
  **v3 が 0.42 を超えれば「効果セマンティクスで登り始めた」**＝97汎用パイロットへ進む判定材料。
- 設計: `docs/reports/effect_semantics_v3_plan_20260708.md`／特徴の一次資料: `effect_feature_inventory_20260708.md`。
