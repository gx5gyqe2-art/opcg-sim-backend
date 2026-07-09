# v3×97汎用パイロット ワーカープロンプト（単発・専用）

新しいセッションにこの内容を貼る。このセッションは **効果セマンティクスv3 の97リーダー汎用訓練を1本回すだけ**の
専用ワーカー。青パイロットで climb 実証済み（537局/リーダーで対L1 0.604）の v3 を、本命の「汎用CPU」設定で検証する。
強度測定は司令塔（別セッション）が行う。

---

## プロンプト本文（ここから貼る）

あなたは v3×97汎用パイロットの訓練ワーカーです。v3 net を**全97リーダー**で訓練するのが仕事です。
強度測定は司令塔がやるので、あなたは起動して走り続けることに専念してください。
重要: **checkpoint を勝手に掃除しない・環境変数ガード（OPCG_P3_LEAD_SLOTS=2 / OPCG_P3_EFF_DIM=116）を必ず付ける・
`--enc-version 3` を必ず使う・OPCG_LEADER_COLORS は付けない（全97リーダーが対象）**。以下を順に実行:

### 1. リポジトリ準備（v3コード枝が必須）
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
rm -rf /tmp/v3-gen97-wt
```

### 2. v3コード枝＋v3種であることを自己検証（❌が出たら起動せず司令塔に報告）
```bash
PYTHONPATH=tests python3 - <<'PY'
import sys; sys.path.insert(0,"tests")
import _bootstrap  # noqa
import rl_net as RN
assert hasattr(RN.ValueNet(10,4,8,96), "eff_dim"), "❌ v3コード枝に居ない（value_net に eff_dim が無い）"
import subprocess, io
subprocess.run(["git","fetch","origin","claude/p3-v3-gen97-checkpoints","-q"])
raw = subprocess.check_output(["git","show","origin/claude/p3-v3-gen97-checkpoints:p3ckpt/value.npz"])
v = RN.ValueNet.load(io.BytesIO(raw))
assert v.lead_slots == 2 and v.eff_dim == 116, f"❌ 種が v3 でない (lead={v.lead_slots} eff={v.eff_dim})＝司令塔に報告"
print("✅ v3コード枝＋v3種を確認（lead_slots=2, eff_dim=116, hidden=%d）" % v.W1.shape[1])
PY
git show origin/claude/p3-v3-gen97-checkpoints:p3ckpt/manifest.json   # cum_games=0 のはず（0以外=二重起動）
```
**どんなエラーが出ても checkpoint 枝を掃除・リセットしない。**

### 3. 連続学習を起動（色フィルタ無し＝全97リーダー）
```bash
OPCG_P3_LEAD_SLOTS=2 \
OPCG_P3_EFF_DIM=116 \
OPCG_P3_WT=/tmp/v3-gen97-wt \
OPCG_P3_BRANCH=claude/p3-v3-gen97-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 3 --rotate-leaders --shard-games 60 --sims 40 --workers 4 \
  --target 100000000 --max-shards 100000000
```
（案A（legacy 97汎化）と同一条件。変えたのは net アーキ（v3）と符号化（v3）のみ＝直接比較できる。）

### 4. 起動確認と報告
```bash
cat /tmp/v3-gen97-wt/p3ckpt/manifest.json
python3 -c "import sys;sys.path.insert(0,'tests');import _bootstrap,rl_net;v=rl_net.ValueNet.load('/tmp/v3-gen97-wt/p3ckpt/value.npz');print('走行中net: lead_slots=',v.lead_slots,' eff_dim=',v.eff_dim)"
```
- 起動ログに `リーダーローテーション ON: 97 種` と `再開: gen=0 cum_games=0 ... enc=v3` が出ること・
  `lead_slots= 2  eff_dim= 116` を確認（**97種でなければ色フィルタが残っている＝環境変数を確認**）。
- 確認できたら「**v3汎用97、学習開始（shards進行中・cum=◯◯・eff_dim=116・リーダー97種）**」と報告する。

### 5. 運用
- 連続モード＝止まらない。約2.5分ごとに checkpoint 枝へ force-push。**測定は司令塔がやる**。
- プロセスが死んだら同じ起動コマンドで resume（ガード環境変数を忘れず）。checkpoint枝の掃除・forceリセットは絶対にしない。
- 司令塔から停止指示があるまで回し続ける。

---

## 設計メモ（担当者向け）
- 種: 青パイロットと同一の v3 弱Gen0（恒等連鎖・EffFeatは全2652カード対応＝97リーダーでそのまま有効）。
- 比較バー（多様デッキ・対L1同条件）: **案A(legacy 97汎化) Gen1 = cum10,000 で 0.433**／出荷v1 多様 = **0.833**。
  v3青は537局/リーダーで0.604まで登った（`docs/orchestrator_handoff.md` §2 最下部）。97でも登れるかが本検証の問い。
- 設計: `docs/reports/effect_semantics_v3_plan_20260708.md`。
