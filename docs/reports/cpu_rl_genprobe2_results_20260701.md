# pre-flight ②: ドメインランダム化の効き＆教師の信頼性（＋①結果の訂正）

日付: 2026-07-01 / 計画: `cpu_rl_generalization_plan_20260701.md`
先行: `cpu_rl_genprobe_results_20260701.md`（pre-flight ①）
コード: `tests/probe_generalization.py`（`--holdout {color,leader}` 追加）

> スナップショット（改変しない）。**本書は先行①結果の一部主張を訂正する**（後述）。

## 目的
pre-flight ① で outcome 教師の held-out が崩壊した。仮説「未見領域を訓練が覆えば（＝ドメイン
ランダム化②）崩壊は消える」を安く検証する。held-out の切り方を2種で比較:
- **color**: 黄を丸ごと訓練から除外（被覆ゼロ）
- **leader**: 黄リーダーの8個だけ held-out・残りの黄は訓練に残す（被覆あり＝②の代理）

## 結果（160games×2seed）

### 教師=L1 evaluate（信頼できる密教師）: fingerprint の held-out 上乗せ ΔR²
| holdout | seed0 | seed1 |
|---|---|---|
| color（被覆ゼロ） | +0.121 | +0.193 |
| leader（被覆あり） | **+0.264** | +0.200 |

→ どちらでも **fingerprint は転移（正）**、identity は ≈0。被覆ありで**やや改善**（seed0 明確・seed1 横ばい）。

### 教師=game outcome（ランダム続行・ノイズ大）: fingerprint の held-out 上乗せ ΔR²
| holdout | seed0 | seed1 |
|---|---|---|
| color（被覆ゼロ） | −0.596 | −0.223 |
| leader（被覆あり） | **−0.612** | **−1.285** |

→ **被覆しても崩壊は消えない**（むしろ悪化）。in-dist は +0.25〜+0.30 と最大＝**ノイズをin-distで
過学習し外挿で崩れている**。崩壊は「被覆の有無」ではなく「教師のノイズ」に起因する。

## 解釈と訂正

1. **outcome-under-ランダム続行 は有効なオフライン汎化ゲートではない**。被覆の有無に関わらず
   held-out が崩壊する＝この教師の数値は表現の汎化ではなく**線形ヘッドによるノイズ過学習**を測っている。
2. **【訂正】** 先行 `cpu_rl_genprobe_results_20260701.md` の「**②ドメインランダム化は必須と実測された**」
   （§解釈4・計画反映）は **over-claim**。outcome 崩壊は**教師ノイズの副産物**であり、被覆の必要性の
   証拠ではない（本実験で被覆しても崩壊が残ることを確認）。
3. **確定している事実（信頼できる密教師・全regime・全seedで一貫）**:
   - **① 効果フィンガープリントは汎化価値を足し held-out へ転移する（+0.12〜+0.26）** → 表現GO（不変）。
   - **識別子(card_id)は汎化価値ゼロ（≈0）** → 残差降格（不変）。
   - 被覆(②)は密教師で**弱い正の効き**（seed依存）＝**方向性は支持されるが offline では必要性を証明できない**。

## 計画への反映（更新）
- **① fingerprint: GO**（変更なし。本番 encoder 移植へ）。
- **② domain randomization: “必須と実証”ではなく“妥当な事前策”へ格下げ**。sim-to-real の定石として採用するが、
  **効き/必要性の決定的判定は本走の held-out 勝率ゲート（③・強い方策下の対局）で行う**。offline の
  安価な代替（ランダム続行 outcome）は不可と確定。
- **③ held-out 勝率ゲート: 汎化の唯一の決定的判定点**（本実験で offline 代替の限界が確定したため、重みが増した）。

## 学び（方法論）
pre-flight の価値は「安く GO/NO-GO」だけでなく「**当てにならない指標を早期に棄却する**」ことにもある。
今回、cheap な outcome ゲートを campaign で信じていたら誤判定していた。①（表現）は密教師で確定・
②③（被覆と最終判定）は本走 held-out 勝率へ、と役割分担が明確になった。

## 再現
```bash
OPCG_LOG_SILENT=1 python tests/probe_generalization.py --games 160 --ply-cap 400 --sample-every 5 --holdout color  --seed 0
OPCG_LOG_SILENT=1 python tests/probe_generalization.py --games 160 --ply-cap 400 --sample-every 5 --holdout leader --seed 0
```
