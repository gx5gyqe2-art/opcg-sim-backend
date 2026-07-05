"""学習型CPUの探索オプション既定（本番・自己対戦の単一定義。skew防止＋意図的差の明示）。

本番 `cpu_learned.decide`／自己対戦 `p3_run._gen_task` 等、呼び出し側が個別にオプションを
指定する箇所を1定義に集約する。値そのものは変更しない（挙動不変・マジックナンバーの名前化のみ）。

- **共有既定**（本番も自己対戦も同一であるべき値。ここを唯一の正にする）: `C_PUCT` / `DIRICHLET_ALPHA` /
  `VALUE_SCALE`。
- **意図的な差**（文脈で値が異なることを明示する。揃えない）: `SERVE_*`（本番・強さ優先）と
  `SELFPLAY_*`（自己対戦・速度/探索多様化優先）。
- **触らない**（このモジュールの対象外）: `enc_version`（ロードした net の重み次元から自動判別）・
  ネット形状 `d_emb`/`hidden`/`feat_dim`（ロードした `.npz` が正）。
"""

# --- 共有（本番も自己対戦も同一であるべき。ここを唯一の正にする）---
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
VALUE_SCALE = 10000.0            # GATE-B の L1 tanh 圧縮用（learned 経路では未使用）

# --- 文脈で意図的に異なる値（揃えない・差を明示的にレビュー可能にする）---
SERVE_SIMS = 160                 # 本番 decide_learned（強さ優先）
SERVE_DIRICHLET_EPS = 0.0        # 本番は決定的
SELFPLAY_SIMS = 40               # 自己対戦（速度優先）
SELFPLAY_DIRICHLET_EPS = 0.25    # 自己対戦は探索ノイズ
SELFPLAY_TEMP_MOVES = 8          # 序盤サンプリング手数
