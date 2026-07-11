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
# 終局値の深さ減衰: terminal を ±max(TERM_FLOOR, 1 − TERM_DECAY·depth) で返す（TreeMCTS）。
# L1 の ±(W_WIN − ply) と同じ原理＝「速い勝ち＞遅い勝ち／遅い負け＞速い負け」。減衰が無いと
# 敗勢で全候補 q=-1 に飽和し、防御（カウンター）と無抵抗（PASS）が無差別になる
# （docs/reports/cpu_learned_mark_review_20260711.md §F2）。
TERM_DECAY = 0.02
TERM_FLOOR = 0.5

# --- 文脈で意図的に異なる値（揃えない・差を明示的にレビュー可能にする）---
SERVE_SIMS = 160                 # 本番 decide_learned（強さ優先）
SERVE_DIRICHLET_EPS = 0.0        # 本番は決定的
SELFPLAY_SIMS = 40               # 自己対戦（速度優先）
SELFPLAY_DIRICHLET_EPS = 0.25    # 自己対戦は探索ノイズ
SELFPLAY_TEMP_MOVES = 8          # 序盤サンプリング手数

# --- serve 専用: root 読み出し・PIMC 世界線（探索・自己対戦データ生成は不変）---
# root 読み出しの LCB 乗り換え（cpu_learned._select_root_group）: 最多訪問グループを基準に、
# 訪問数が SERVE_ROOT_LCB_MIN_FRAC·n_top 以上の代替の LCB(q − z/√n) が上回れば乗り換える。
# z=0 で従来の argmax(N) に一致（マーク回帰: @12/@24 の Q 劣後手への貼り付きを解消）。
SERVE_ROOT_LCB_Z = 1.0
SERVE_ROOT_LCB_MIN_FRAC = 0.2
# ターン内 sticky 世界線: 同一 (game, turn, player) の連続 decide で PIMC 決定化 seed を固定し、
# 「ドン付与→（別世界を引いて）攻撃取り止め」型の計画非一貫（無駄ドン）を抑える。
SERVE_STICKY_WORLD = True
