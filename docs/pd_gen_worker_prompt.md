# バッチ生成ワーカー（pd_gen）プロンプト — 並列アクター・複数セッション

新しいセッションにこの内容を貼る。**バッチ式自己対戦の generator を1本**回す。複数セッションで同時に走らせられる
（各自 DATA_BRANCH を変える＝単独writer＝衝突なし）。learner は別に1本だけ走っている前提。
設計: `docs/reports/batched_selfplay_design_20260710.md`。

> **重要**: あなたに割り当てられた **ワーカー番号 <N>**（w1/w2/w3…）を司令塔から受け取ること。
> 番号がなければ司令塔に聞く（他ワーカーと同じ番号にすると data枝が衝突する）。

---

## プロンプト本文（<N> を自分の番号に置換してから貼る）

あなたはバッチ式自己対戦の生成ワーカー w<N> です。以下を順に実行:

### 1. 準備
```bash
cd /home/user/opcg-sim-backend
git fetch origin claude/opcg-cluster-learning -q && git checkout claude/opcg-cluster-learning && git pull -q
python -m pip install -q numpy
rm -rf /tmp/pd-w<N>
```

### 2. net枝の種を自己検証（❌なら起動せず司令塔に報告）
```bash
PYTHONPATH=tests python3 - <<'PY'
import sys; sys.path.insert(0,"tests")
import _bootstrap  # noqa
import rl_net as RN
import subprocess, io
subprocess.run(["git","fetch","origin","claude/p3-pd-net","-q"])
raw = subprocess.check_output(["git","show","origin/claude/p3-pd-net:p3ckpt/value.npz"])
v = RN.ValueNet.load(io.BytesIO(raw))
assert v.lead_slots==2 and v.eff_dim==116, f"❌ net枝の種がv3でない (lead={v.lead_slots} eff={v.eff_dim})"
print("✅ net枝の種を確認（v3・lead=2 eff=116）")
PY
```
checkpoint 枝は絶対に掃除しない。

### 3. 生成ループを起動（自分の番号で DATA_BRANCH と WT を変える）
```bash
OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
OPCG_PD_NET_BRANCH=claude/p3-pd-net \
OPCG_PD_DATA_BRANCH=claude/p3-pd-data-w<N> \
OPCG_PD_WT=/tmp/pd-w<N> \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_gen.py \
  --enc-version 3 --sims 160 --games 128 --workers 4 --dirichlet-eps 0.15
```

### 4. 確認と報告
- ログに `generator w<N>: ... リーダー=97 sims=160` が出て、数分内に `batch0 r0 ...局面 push=OK` が出ることを確認。
- `push=FAIL` が続く場合は司令塔に報告（他ワーカーと番号が被っている可能性）。
- 確認できたら「**生成ワーカー w<N> 開始（batch0 push OK）**」と報告する。

### 5. 運用
- 連続して batch を生成し続ける（net枝が更新されれば次周から自動追従）。
- プロセス死は同コマンドで resume。data枝の掃除・force リセットはしない。
- 司令塔から停止指示があるまで回し続ける。

---

## メモ
- 1バッチ = 128局・sims160 ＝ 4コアで数分。生成した (v3符号化, 最終勝敗) が learner の教師データになる。
- あなたは**測定しない**（司令塔が net枝を凍結測定）。あなたの仕事は「新鮮なデータを供給し続ける」こと。
