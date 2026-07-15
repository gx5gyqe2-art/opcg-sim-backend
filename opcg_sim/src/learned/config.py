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
# root 読み出しの Q 乗り換え（cpu_learned._select_root_group）: 最多訪問グループを基準に、
# 「十分競った訪問（n ≥ MIN_FRAC·n_top）」かつ「明確な Q 差（q ≥ q_top + MIN_GAP）」の代替
# だけへ乗り換える（二重ゲート）。MIN_GAP=inf で従来の argmax(N) に一致（ロールバック）。
# 較正: 実対局2局×16人間マークに対する回帰（cpu_learned_mark_review2_20260711.md §S1）。
# 初版 LCB(z=1, frac=0.2) は低訪問 Q の楽観バイアス（実測 +0.14〜+0.54 ≫ 1/√n）に対して
# 甘く、ドン付与へ誤って乗り換える退行を起こしたため、この二重ゲートへ置換した。
SERVE_ROOT_SWITCH_MIN_FRAC = 0.4
SERVE_ROOT_SWITCH_MIN_GAP = 0.05
# ターン内 sticky 世界線: 同一 (game, turn, player) の連続 decide で PIMC 決定化 seed を固定し、
# 「ドン付与→（別世界を引いて）攻撃取り止め」型の計画非一貫（無駄ドン）を抑える。
SERVE_STICKY_WORLD = True

# learned MCTS の候補生成で無駄攻撃（倒せない/届かない）・無意味なドン付与を除外する（L1/α-β と同じ
# 枝刈りを learned 候補にも適用）。False で従来（v4 まで）＝枝刈り無し。docs/cpu_v5_plan.md §4-1補。
# serve に効く（OPCGGame.legal_actions 経由・インスタンス未指定時の既定）。
SERVE_PRUNE_FUTILE = True

# v6 柱⑤（生成/serve の探索設定分離・docs/reports/v5_adoption_20260715.md §4-5）: 自己対戦**生成**の
# 枝刈り既定。生成側は枝刈りを外す＝探索が訪れない枝は学習できないため、serve 用ヒューリスティクスを
# 生成に入れると「刈った枝の反例をネットが二度と見ない」自己強化盲点になる（v5 は serve と生成の両方に
# 掛けていた）。生成ハーネス（p3_run）が OPCGGame(prune_futile=GEN_PRUNE_FUTILE) で適用する。
GEN_PRUNE_FUTILE = False

# aux 粘り項（v5 §4-1・C4 負けq飽和の緩和）: 葉評価が飽和域（|v| ≥ SAT_START）のとき、残りターン
# 補助ヘッドの予測 t̂ で振幅を減衰 v' = v·max(TERM_FLOOR, 1 − AUX_TIE_DECAY·t̂·sat) する。
# 終局の深さ減衰（TERM_DECAY）の「終局に届かない葉」への拡張＝敗勢では『本当に延命する手』を、
# 優勢では『速い勝ち』を選好。sat = clip((|v|−SAT_START)/(1−SAT_START), 0, 1)＝非飽和域は不変。
# 再学習不要（v4 学習済みの aux ヘッドを手選択に初活用）。False で従来（v4）＝減衰なし。
SERVE_AUX_TIEBREAK = True
AUX_TIE_DECAY = 0.02      # 1予測ターンあたりの減衰（TERM_DECAY と同スケール）
AUX_SAT_START = 0.8       # この |v| から減衰を線形に効かせ始める（中間域の較正は不変）

# --- v4 学習（docs/cpu_v4_plan.md §4）---
# value 混合ラベル: y = α·z(勝敗±1) + (1−α)·q_root(探索後 root Q・終局減衰込み)。
# 勝敗単独（α=1）は v3 で忘却を実証済み・q_root が「何手で負けるか」の距離を持ち込む。
V4_LABEL_ALPHA = 0.5
# 残りターン数の補助損失（ValueNet の aux ヘッド・「2つの時計」をラベルで明示的に教える）。
V4_AUX_TURNS_WEIGHT = 0.25
# turns_left の正規化: min(turns_left, V4_TURNS_SCALE) / V4_TURNS_SCALE ∈ [0,1]。
V4_TURNS_SCALE = 15.0
