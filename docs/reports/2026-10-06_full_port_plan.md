# 理論の器の Rust 全移植——棚卸しと計画（2026-10-06・読み取りだけの準備）

起点: `claude/cpu-spec-improvements-yw91jd` @ `19e8d85f7`（波C の台帳まで）。リポジトリは 1 文字も変えていない。
作業物は計画を作った作業者の一時領域（`scratchpad/port_plan/`）にあった。リポジトリには本書と `2026-10-06_full_port_plan/tools.csv`・`tests.csv` だけを保存した（`prof/`・`work/`・試験用の写し `tree/` は保存していない）。

読み方: **【事実】**＝ファイル・行・実測で確かめたこと。**【推測】**＝それを元にした見立て・提案。

---

## 0. 要点（先に）

1. **理論の器（`tests/scripts`＋`tests/harness` のうち理論フェーズのもの）は 86 本・32,200 行**。分類（`tools.csv`）:
   **移植する 26 本（19,193 行）**／**退役 50 本（14,468 行）**＝閉じた課題 34・CPU の診断 15・答え合わせの原文 1／**判断待ち 10 本（3,137 行）**。
   範囲外（エンジン・学習・golden・カード監査・アリーナ）は 36 本（6,560 行）。
2. **移植すべき計算の実体は約 1 万行**【事実＋推測】: 現行の 8 器を既定で走らせたとき実際に実行された関数は、移植対象 26 本の中で **11,013 行**
   （うち CLI・集計 ≈1,000 行、`rd_kernel` の継ぎ目 85 行）。残りの約 8,000 行は既定では一度も呼ばれない（閉じた測定の CLI・旧い切替の枝・試験だけが呼ぶ関数）。
3. **値付けの核は 7 ファイルにまたがる 1 つの相互再帰**【事実】: 既定経路の関数呼び出しを強連結成分に分けると、
   `theory_order`（`nu_of`・`attack_value(_don)`・`attack_stream`・`option_value`・`block_cost`）・`effect_value`（`card_value` ほか 12）・
   `hand_plan`・`hand_spend`・`search_price`・`cut_price`・`hand_joint`・`theory_bridge.joint_valuer` の **33 関数が 1 つの輪**になる。
   **下から 1 層ずつは移せない**＝この輪はひとかたまりで移す（最大の山・§4）。
4. **速さ**【事実】: 5 局で、Rust（`rd_solve`）が占める壁時計は 交点の橋 25〜34%・線形の橋 49〜62%・相対の帳簿 48〜53%・較正 24〜34%。
   残りの Python の理論計算を全部 Rust にしても、**道具まるごとの速さは 1.5〜2 倍程度**【推測】（守る側の計算は既に Rust で、それが一番重い）。
   **全移植の価値は速さではなく「式の写しを 1 つにする」こと**（M-2 を Rust で 1 回だけ書ける）。
5. **試験**（`tests.csv`・理論の器に触れる 86 ファイル・1,378 関数）: **移す 645**／器と一緒に退役 437／古い算術・番人で退役 82／判断待ち 203／範囲外 11。
   **試験の自動の固定 4 つ（`conftest.py`）を外して落ちるのは 43 本だけ**（実測・§2.3）。
6. **見積もり（悲観・やり直しと確認込み）: 作業担当の時間で 30〜50 時間**（5〜9 回の作業セッション）。最大の不確かさは値付けの核（効果の木を
   Rust で歩く部分）と、numpy の足し算・`math.erf` のビット一致（§5）。
7. **ユーザの判断は 11 項目**（§8）。前提を決めるのは 1（退役の基準）・3（答え合わせを残すか）・4/5（旧い値・保留の値を持っていくか）・7（出力をどこで書くか）。

---

## 1. 器の棚卸し（`tools.csv`）

### 1.1 列の意味
`lines`＝ファイルの行数。`default_path_funcs_used`＝既定の 8 器（＋4 器の追加計測）を 5 局走らせたときに実行されたトップレベル関数の数／全数
（`prof/used_lines.json`・cProfile から）。`newest_report_any`＝`docs/reports/` で名前が出る最新の日付（10-05 の棚卸し・設計の報告を含む）、
`newest_report_substantive`＝それらの棚卸し系 6 本を除いた最新の報告。`imported_by_*`＝AST で拾った import（関数の中の遅延 import も含む）。

### 1.2 移植する（KEEP-PORT・26 本）

| 区分 | 器 | 行 | 既定で実行 | 根拠 |
|---|---|---|---|---|
| 現行の計測（ユーザ指定 8） | `crossing_bridge` 3,787・`theory_bridge` 1,589・`relative_ledger` 591・`transition_ledger` 468・`win_calib` 246・`pre_settle_asymmetry` 485・`kappa_vector` 699・`price_realised` 530 | 8,395 | 計 5,859 行 | 10-05 の表の作り直し（`2026-10-05_table_rebuild_rust.md`）と D-6 採用で回したのがこの集合 |
| 計算の依存（全体） | `theory_order` 2,403・`effect_value` 2,743・`condition_value` 611・`search_price` 372・`hand_plan` 754・`hand_joint` 223・`hand_guard` 162・`cut_price` 668・`deck_refill` 363・`lethal_rule` 457・`kappa_needed` 170 | 8,926 | 計 4,694 行 | 呼び出しの実測（`prof/callgraph.txt`） |
| 計算の依存（一部の関数だけ） | `hand_spend`（`use_value`・`free_value`・`hand_ids`・`spent_cards`）・`guard_afford`（`knapsack`・`hand_counters`・`_vocab`）・`attack_response`（`parts`・`parts_mirror`）・`don_ledger`（`zones_of`）・`order_acc`（`band_of`）・`clock_calib`（定数 `D_BINS`・`d_bin`） | 1,523 | 計 149 行 | 器そのもの（各自の `collect`/`main`）は閉じた測定＝**その部分は移さず消す** |
| 継ぎ目 | `rd_kernel` | 349 | 85 行 | 最後に PyO3 の入口へ吸収して消える |

【事実】`plan_value_map`・`race_state`・`plan_drift` は `guard_afford` の先頭で import されるが、既定経路で関数は 1 度も実行されない
（`Leaders` はクラス本体の評価だけ）。`onplay_parts` も `hand_plan` から遅延 import されるが 0 回。

### 1.3 退役（RETIRE・50 本）
* **閉じた課題の器 34 本（9,086 行）**: `attack_axis_by_result`(C-1)・`attack_resid_defender_split`(P8-7c)・`attack_theta_parts`(C-5)・`band_screen`・
  `budget_audit`・`don_active_price`(T57)・`entry_gain`・`exit_ledger`(T27)・`h4_tables`(H-4g)・`ko_by_power`(T22)・`mu_branch`(T26)・`nu_calib`・
  `nu_ledger`(T44)・`nu_measure`・`nu_stock`(T50)・`onplay_parts`(T59)・`opp_board_dist`(T46)・`play_body_check`(T157)・`pre_settle_decomp`(T145)・
  `price_scaling`(T24)・`rate_tracking`(T128)・`removal_demand`(T27-b)・`search_gain`(T65)・`seat_pair_symmetry`(T151-1)・`settlement_split`(T139)・
  `shield_rate`(T21)・`t139_defender_check`(P8-7b)・`target_choice`・`theory_residual`(1c)・`theory_vs_search`(T1)・`theta_at_settle`(T143)・
  `theta_price`・`two_curves_metrics`(T137b)・`two_curves_settle`(T137d)。最新の実質的な報告はどれも 2026-09-25 以前（`h4_tables` だけ 10-04）。
  **残す器が呼ぶ関数は 1 つも無い**（呼び出しの実測と import の両方で確認）。
  注意: `opp_board_dist` は `tests/fixtures/opp_boards.json`（既定の `option_value` が読む）を作った器。**fixture は残し、作り直しは凍結ブランチで**。
  `budget_audit` は `guard_afford` を import しているので、`guard_afford` を縮める前に消す。
* **CPU の診断 15 本（4,880 行）**: `zero_sum_check`・`pimc_diag`・`policy_entropy`・`search_vs_prior`・`action_coverage`・`order_option`・`plan_drift`・
  `plan_value_map`・`time_plan_map`・`race_state`・`attack_budget`・`deck_profile`・`hazard_curve`・`hand_value_slope`・`life_price`。
  理論の式を持たない（`theory_order` を import しない）ネット・探索の測定で、すべて 09-12〜09-16。**移植の対象ではない**——消すか今のまま置くかは
  理論の Python を消すこととは独立（§8 判断 11）。
* **答え合わせの原文 `rule_don_ref`（502 行）**: ユーザ決定（10-06）どおり削除。ただし**最後に**（§4 段 7）。

### 1.4 判断待ち（ASK・10 本）
* **T18 の出口実験 5 本**（`t18_arena` 550・`shadow_forbid` 447・`live_theory` 400・`theory_trace` 372・`theory_trace_build` 120）と
  その依存 2 本（`two_curves` 270 の `_price_row`・`two_curves_state` 176 の `state_by_turn`）。【事実】台帳 §0.6 の「T18 を据え置く」
  （`docs/cpu_theory_gap.md:440-441`）は「理論を直してから**同じ器（`t18_arena.py`）で測り直す**」「目で確かめる道具（`theory_trace`・`theory_trace_build`）は残す」と書いている。
  【事実】`t18_arena`・`shadow_forbid`・`live_theory` は対局を回す（`opcg_sim.loop` の `driver`・`engine`・`record_gen`）＝器の大半は対局の駆動で、理論の計算は
  `kappa_vector.state_of_row`・`two_curves._price_row`・`theory_order` の採点を呼ぶだけ。
* **`theory_gate`**（323 行）: §0.1 の「完成の 4 条件」の器として名前が残るが、最後の報告は 09-17。
* **`plan_store`**（計画のディスク保存）と **`rd_kernel_golden`**（golden を原文で作り直す道具）: Python が消えると鍵（Python の原文のハッシュ）と
  作り直しの正本（原文）が意味を失う。

### 1.5 範囲外（36 本）
`rs_*`（エンジン・探索の計測 13）・`az_*`/`tictactoe`/`gate_a_tictactoe`（AZ 5）・`n_rel_torch*`・`train_profile*`・`dump_f16_check`・`label_worker`・
`pd_batch_common`・`peak_alert`・`phase1_sweep`・`q_pi_sweep`（学習）・`effect_oracle`・`expected_effects`・`mistarget_diagnostics`・`compare_parsers`・
`effect_diagnostics`（カード監査・パーサ）・`structural_invariants`・`play_coverage`（エンジン監査）・`rs_golden`・`rs_replay`（harness）。

---

## 2. 試験の棚卸し（`tests.csv`）

### 2.1 数（関数単位・parametrize の展開前。pytest の収集は 1,400 件）

| 分類 | 関数 | 行 | 意味 |
|---|---|---|---|
| **PORT** | 645 | 6,862 | 移植する器の**既定の式**を確かめる。うち 563 は `cpu_infra` |
| **RETIRE-WITH-TOOL** | 437 | 4,263 | 退役する器（と一部だけ残す器の測定 CLI 部分）を確かめる |
| **RETIRE-OBSOLETE** | 82 | 1,308 | 試験の自動の固定に依存（43）・足場の値を明示（9）・旧い既定（比較の基準）を明示（22）・Python 同士／原文との一致の番人（下の 2.4）|
| **ASK** | 203 | 2,401 | T18 系 146・中間ゲート 8・保留の値を入れる 43（攻撃時能力・常在の体・ドンの配分ずれ・減衰・対称の耐久・`game` 決着 ほか）・計画のディスク保存 6 |
| OUT-OF-SCOPE | 11 | 240 | 同じファイルにあるだけで理論の器に触れない（`test_plan_heads` の 9 など） |

ファイルごとの内訳は `work/tests_per_file.json` と `work/tests_summary.txt`。PORT の多いファイル: `test_crossing_bridge` 95・`test_effect_value` 81・
`test_theory_order` 59・`test_kappa_vector` 45・`test_pre_settle_asymmetry` 35・`test_lethal_rule` 34・`test_theory_bridge` 32・`test_relative_ledger` 30。

### 2.2 分類の規則（機械的・`work/classify_tests.py`）
1. ファイル名の主題の器（`test_X.py` の `X`）が退役 → RETIRE-WITH-TOOL、判断待ち → ASK。
2. 主題が「一部の関数だけ残す」器なら、試験が触る属性に残す関数があれば PORT、無ければ RETIRE-WITH-TOOL。
3. それ以外は PORT。ただし本文が足場の値（`surv=once`・`w=flat/clock`・`cbar=loose`・`option=off`・`nu=base`・`SPEED_MEMO=False`）を入れる、
   または**自動の固定を外すと落ちる** → RETIRE-OBSOLETE。旧い既定（`cuttable_forced`・`flat`/`mu`・`curve`・`token`・`none`・`printed`・鏡 off）→ RETIRE-OBSOLETE。
   保留の値 → ASK。
4. 番人の個別の上書き（079e73b8・`OPCG_RD_KERNEL`・Python の解き方の不在）は表の `reason` に書いた。

【限界】静的に拾えない設定（辞書で渡す、ヘルパの奥で切り替える）は取りこぼしうる。PORT の 645 は上限寄り。

### 2.3 試験の自動の固定（`tests/conftest.py:43-78`）を外す実験【事実】
写し（`git archive HEAD` を `tree/` に展開・`conftest.py` だけ環境変数で固定を個別に外せるようにした）で、理論の 86 ファイルを回した:

| 外した固定 | 落ちた試験 |
|---|---|
| なし（基準） | 0（1,400 passed） |
| 4 つ全部 | **43** |
| 潜在価値を切る（`option=off`） | 15 |
| 平均の傾き（`w=flat`） | 0 |
| 生存の重み（`surv=once`） | 13 |
| 費用曲線（`cbar=loose`） | 21 |

落ちる 43 本の内訳: `test_theory_order` 19・`test_attack_passive_wiring` 8・`test_nu_calib`/`test_nu_measure` 各 3 ほか（`tests.csv` の `pins_needed`）。
**`w=flat` の固定はもう何も守っていない**。【推測】固定は「43 本の算術のため」だけに残っている＝Rust に古い算術（`loose`・`once`・`off`）を持ち込む理由にならない。

### 2.4 ビット一致の番人と、Python が消えた後の置き換え

| 番人 | 今 | 置き換え（提案） |
|---|---|---|
| `test_attack_passive_wiring.py:780` `test_switches_off_match_recorded_base_outputs_of_079e73b8` | 旧い既定 6 つ＋印字読みで 3 器を回し、079e73b8 の記録と一致 | **退役**（旧い既定を Rust に持ち込まない場合・凍結ブランチ `claude/theory-switches-final` で再現） |
| `test_rd_speed` の 4・`test_rd_kernel` の 11・`test_rd_kernel_plans` の 5（原文・速くした Python との一致） | Rust ＝原文の `repr` | cargo の**記録ビットの再生**（既に在る `rd_dp_golden.jsonl`・`rd_solve_golden.jsonl.gz`）に一本化。同点・退化の入力の作り方は cargo の手計算試験へ写す |
| 残る Rust 側の検査（`test_rd_kernel` の記録再生・状態の数・冷/温/並べ替えで同じ判断 ほか 11） | PyO3 経由 | PORT（そのまま pytest か cargo へ） |
| `test_speed_memo` の 4 | 覚え書きの有無で 1 ビットも変わらない | 退役（覚え書きは Python のもの）。Rust 側は覚え書きを持つなら同じ形の試験を cargo に |
| 計画のディスク保存の 6 | 鍵・版の上げ忘れ | 判断 8 しだい |
| **新しく要るもの** | — | 各層の**呼び出しの記録 golden**（§5.2）を cargo で再生＝Python が消えた後の答え合わせ。道具まるごとの JSON は `tests/fixtures/theory_out/`（実・合成 各 2〜5 局）を pytest で 1 本 |

---

## 3. 測った速さ（`prof/`）

設定: `OPCG_LOG_SILENT=1`・Rust 核は既定・計画のディスク保存なし。記録 `slice/w41`（実 10 局）・`slice/w39`（合成）・**5 局**。1 本ずつ。
cProfile ありの全 8 本＋追加 4 本で約 5 分、cProfile なしの再計測 8 本で約 2 分。cProfile の有無で出力は同一（交点の橋・実で確認）。

### 3.1 壁時計（cProfile なし）と Rust の割合

| 器 | 実 w41（秒） | うち Rust | 合成 w39（秒） | うち Rust |
|---|---|---|---|---|
| crossing_bridge | 7.2 | 2.48（34%） | 9.6 | 2.38（25%） |
| theory_bridge | 20.7 | 10.22（49%） | 20.4 | 12.74（62%） |
| relative_ledger | 15.3 | 8.12（53%） | 16.2 | 7.72（48%） |
| win_calib（`--pre-settle on`） | 7.0 | 2.38（34%） | 9.4 | 2.25（24%） |

Rust の秒数は cProfile の `rd_solve` の自己時間（Rust の中は cProfile の負担を受けない）。追加 4 本（実・cProfile あり）: transition_ledger 10.7 s（Rust 0）・
pre_settle_asymmetry 14.8 s（Rust 2.46）・kappa_vector 5.8 s（Rust 0）・price_realised 7.3 s（Rust 0）。

### 3.2 cProfile の内訳（秒・5 局・実／合成）

| 器 | 合計 | Rust | カード DB の読み込み（パーサ） | 合成デッキの作り直し | 理論の Python（自己時間） | 組み込み関数（dict.get・round・isinstance・sum ほか） |
|---|---|---|---|---|---|---|
| crossing_bridge | 15.7／27.0 | 2.5／2.4 | 3.3／3.1 | 0／14.1 | 5.1／4.5 | 3.5／4.8 |
| theory_bridge | 40.2／35.1 | 10.2／12.7 | 3.1／2.9 | 0／0.3 | 20.1／14.5 | 6.6／4.6 |
| relative_ledger | 28.3／36.1 | 8.1／7.7 | 3.0／3.0 | 0／14.2 | 12.8／8.4 | 4.3／5.5 |
| win_calib | 14.6／27.2 | 2.4／2.3 | 3.1／3.2 | 0／14.3 | 6.7／5.4 | 2.7／5.0 |

理論の Python の内訳（自己時間の多い順）: `theory_order`（2〜7 s）＞ `crossing_bridge`（1〜5）＞ `effect_value`（1〜3.5）＞ `hand_plan` ＞ `deck_refill` ＞ `cut_price` ＞ `hand_guard`。

### 3.3 重い関数（累積・theory_bridge 実）
`_kappa_of_row` 26.9 s → `crossing_bridge.curve_d_of_row` 26.9 → `threshold`/`threshold_parts_side` 16.6 → `_rule_don_term` → `rule_don_solve` 16.5
（うち `rd_solve` 10.2 s／499 回）・`hand_spend.use_value`/`free_value` 12.9（17,731 回）・`hand_plan.hand_items` 10.6・`theory_order.nu_of` 9.5（55,657 回）・
`attack_stream` 9.4（227,750 回）・`option_value` 9.2・`attacker_ctx` 7.0。自己時間の上位: `rd_solve` 10.2・`crossing_bridge._attach_gain` 2.0（**342 万回**）・
`attack_value_don` 1.9（71 万回）・`attack_stream` 1.6・`rules_steps` 1.5・`dict.get` 1.4（721 万回）・`round` 1.2（366 万回）。全表は `prof/summary.txt`。

### 3.4 読み取り
* **【事実】合成の記録では、`--limit-games 5` でも合成デッキ 300 組を全部作り直している**（`deck_refill.shares_by_seed` → `opcg_sim.loop.deck_synth.synth_deck` 600 回・
  cProfile で 14 s・cProfile なしでは数秒）。【推測】理論とは無関係の固定費で、使う種だけ作れば消える（Python だけで直せる・値は不変）。
* **【事実】カード DB の読み込み（パーサの走行）が毎回 3 s**。パーサは Python に残る（CLAUDE.md）。
* **【推測】全移植後の速さ**: 1 局あたりの Python の理論計算（cProfile なしで交点の橋 実 ≈0.45 s/局、線形の橋 ≈1.6 s/局）が 10〜30 倍速くなっても、
  `rd_solve`（≈0.5〜2.5 s/局）は変わらない＝**道具まるごとで 1.5〜2 倍**。300 局の本番では固定費が薄まり、`rd_solve` の比率がさらに上がる。

---

## 4. 移植の計画（下の層から）

### 4.1 境界の提案
| 置き場所 | 中身 |
|---|---|
| **Rust（`rust/opcg_engine/src/theory/`）** | すべての式: 値付けの核・耐久 Θ・速さ・守る側の計算（既存）・付与の計画（段 2b 相当）・行ごとの読み（`seat_row`・`state_of_row`・`ledger_value`・`lethal_of_row` …）・局ごとの `collect` の中身 |
| **Python に残す（薄い包み）** | CLI（argparse）・記録の読み込み（`plan_labels.iter_games`・`n_rel_feat` の復号・`opcg_sim.loop.decks`/`deck_synth` の合成デッキ）・カード DB（パーサ）から**カード表と効果の木を Rust へ渡す**こと・**集計と JSON の書き出し**（ブートストラップ・AUC・較正・`np.random.default_rng` を使う部分） |

理由【事実】: 集計は `theory_bridge.py:1358`・`pre_settle_asymmetry.py:228`・`price_realised.py:358` で numpy の乱数（PCG64）を、`summarise` 系で
`np.mean`/`np.percentile`/`np.corrcoef`/`np.var` を使う。これを Rust でビット一致させるのは割に合わない【推測】。JSON の浮動小数の書式も Python の `repr` のまま保てる。
**集計は「理論の計算」ではなく測定の後処理**なので、ユーザの「Python 版を全部消す」の趣旨とは矛盾しないと考える（判断 7）。

API【推測】: 局ごとに 1 回の呼び出し（「局の枠」＝行ごとの `scalars`/`tokens`（float32 のまま）・候補・uuid→カード・デッキ）で、器ごとの行の結果を返す。
細かい呼び出し（`attack_value_don` を 71 万回）を PyO3 越しにすると境界の費用で遅くなる。移植の途中だけは、層ごとに PyO3 の関数を出して Python から差し替えて突き合わせる（段 1〜3 と同じやり方）。

### 4.2 段（下から・依存の実測 `prof/layers.json` に基づく）

| 段 | 中身（移す関数） | Python 行（既定で実行される分） | 一致の確かめ方 |
|---|---|---|---|
| **0 片づけ** | 退役 50 本と試験 437＋82 を削除・試験の自動の固定を外す・一部だけ残す器から残す関数を移す（または移植まで温存）・合成デッキの作り直しを使う種だけに（任意） | −14,468（器）・−5,571（試験） | 8 器×実・合成 5 局の JSON が 1 バイト一致（秒数欄を除く・波 B/C と同じ網） |
| **1 入力と土台** | 局の枠の受け渡し（float32 のまま）・カード表（`Cards.info` の 9 項目）と効果の木（`opcg_effects.json` と同じ形）・fixture（`harm_profile.json`・`opp_boards.json`）・数値の道具の追加（§5.1）・**呼び出し記録の器**（`OPCG_THEORY_CAPTURE`） | ≈400 | 枠の往復が恒等・カード表と効果の木が Python の物と項目ごとに一致 |
| **2 葉** | `deck_refill`（`a_of`・`e_of`・`removal_harm`・`cut_share`）・`lethal_rule` の葉（`attach_don`・`avg_counter`・`stop_min_counter`）・`guard_afford.knapsack`・`hand_guard`・`condition_value`（17 関数）・`price_realised.nu_meas_of`・`theory_order` の葉（`_attack_bound`・c̄・生存・時計 w・`theta_of` ほか）・`cut_price` の葉 | ≈2,300 | 関数ごとの呼び出し記録（実・合成 5 局）をビットで再生 |
| **3 値付けの核（33 関数の輪）** | `theory_order`（`nu_of`・`attack_value`・`attack_value_don`・`attack_stream`・`option_value`・`block_cost`）・`effect_value`（`card_value`・`ability_value`・`action_value`・`selection_dist` ほか 85 関数中の残り）・`hand_plan`（`_base_value`・`inflow_item`・`apply_inflow` …）・`hand_spend`・`search_price`・`cut_price`（`CutView`・`CutCurve`・`defending`）・`hand_joint.JointValuer` | ≈4,600 | 入口 5 つ（`nu_of`・`attack_value_don`・`card_value`・`search_value`・`JointValuer`）の呼び出し記録＋`both` 運転（Rust と Python を両方解き違えば例外） |
| **4 守る側の外側（旧 段 2b）と耐久** | `attacker_ctx`・`rules_steps`・`purse_plan_witness`・`_attach_gain`・`_rule_don_masks`・`model_horizon`/`tau_grow`・`rule_don_solve` の外側（覚え書き・計画の辞書の組み立て）・`threshold_parts_side`・`hand_items`・`search_context` | ≈1,500 | `rule_don_solve` の結果（辞書）の記録ビット＋既存の `rd_*_golden` |
| **5 行の読み** | `crossing_bridge`（`_seat_row`・`curve_d_of_row`・`threshold`）・`theory_bridge`（`ledger_value`・`_kappa_of_row`・`_state_of`・`guard_step`）・`kappa_vector`（`state_of_row`・`g_of_row`・`rate_of_row`）・`lethal_rule`（`lethal_of_row`・`settled_map`）・`price_realised`（`row_ctx`・`state_meas`）・`relative_ledger`・`transition_ledger`（`shapley`）・`win_calib.probs_of`・`pre_settle_asymmetry` の行 | ≈1,600 | 行の結果の記録ビット |
| **6 局の駆動** | 各器の `collect` の中身（局の枠 → 行の表）を Rust に、Python は記録を回して集計・JSON | ≈1,900（`collect` 群）| **道具まるごとの JSON**: 8 器×実・合成 5 局＋`f_identity/rec` 2 局で 1 バイト一致・`both` で 1 通し |
| **7 削除** | 理論の Python（計算）・`rule_don_ref`・`rd_kernel`・`OPCG_RD_KERNEL`・（判断 8 なら）`plan_store`・RETIRE-OBSOLETE の試験・PORT 試験の付け替え・TEST_SPEC §2/§3・台帳 §9.2 | −≈12,000 | `make test`（cargo＋pytest） |

【事実】段 3 の輪は呼び出しの強連結成分（`prof/layers.py` の出力）。他に小さな輪が 2 つ: `crossing_bridge.curve_d_of_row`⇄`theory_bridge._kappa_of_row`/`collect`/`ledger_value`、
`kappa_vector.state_of_row`⇄`relative_ledger.collect`（段 5 で一緒に移す）。

### 4.3 大きさ
* **Python の計算の核 ≈9,900 行**（既定で実行された 11,013 − 集計・CLI ≈1,000 − 継ぎ目 85）＋既定では通らないが生きている枝 ≈10〜15%。
* 第 2a 段の比（Python 1.2k → Rust 2.9k・ビット一致込み）を当てると **Rust ≈24,000〜27,000 行**（試験込み）。今の `src/theory/` は 2,854 行、クレート全体 42,880 行。
* 残る Python: CLI と記録の読み込み・集計・JSON（8 器の `main`/`summarise`/集計 ≈1,000 行＋読み込み ≈300 行）。

### 4.4 計画の覚え書き（`plan_store`）への影響
【事実】鍵は `SOLVER_VERSION`（`rd-speed-3`）＋`tests/scripts` の解き方の関数の原文のハッシュ＋Rust の `src/theory` のハッシュ＋入力（`plan_store.py`）。
【推測】段 4 で `rule_don_solve` の外側が Rust に移ると、鍵の「Python の原文」は空になる。選択肢は (a) Rust の原文のハッシュと入力だけで鍵を作り直す
（版を上げて一度冷える）、(b) 捨てる（Rust の解き直しは十分速いか？——`rd_solve` は今でも壁時計の 25〜60% を占めるので、表の作り直しのように同じ記録を何度も回す作業では効く）。判断 8。

### 4.5 M-2 との順序
ユーザの順序（全移植 → Python 全削除 → M-2 を Rust で）のとおりなら、M-2 が触る `rules_steps`・速さの歩き・`model_horizon` は段 4 で**一度 Rust に写してから**
変える。段 4 を「写すだけ・値は不変」に保つと、M-2 の差分がきれいに分かれる【推測】。

---

## 5. 一致の守り方（E の一覧の延長）

### 5.1 新しく見つけた順序・数値の落とし穴（設計 §4.2 の E1〜E24 の続き）

| # | 場所（行） | 何を守るか |
|---|---|---|
| E25 | `hand_plan.py:466`（`inflow_item`）・`lethal_rule.avg_counter`・`crossing_bridge.collect`/`harm_profile`・`relative_ledger.collect`・`kappa_vector._score`（`np.corrcoef`） | **`np.mean` は 8 要素以上で numpy の対ごとの足し算**（8 本の部分和を展開・128 超で二分）＝Python の `sum` とも左からの足し算とも違う。Rust に numpy と同じ足し方を書く。`np.median`/`np.quantile` の補間も同様 |
| E26 | `theory_order.py:754/805/807` | `np.round(float(float32) * 1e4 / PWR_EPS) * PWR_EPS`＝記録は float32。Rust は f32 で読み f64 に広げてから同じ順で計算（f32 のまま計算しない） |
| E27 | `math.erf`×5・`math.erfc`×2・`exp`・`log`・`log1p`（既定経路の器に計 22 か所） | **Rust の std に `erf` が無い**（クレートの追加は禁止）。glibc の `erf`/`erfc`/`exp`/`log` を `extern "C"` で呼べば同じ機械でビット一致【推測・未検証】。コンパイル時の定数畳み込みを避ける |
| E28 | 辞書の反復（集計・候補の並び） | Python の辞書は挿入順。Rust は `HashMap` を使わず挿入順の並び（Vec）で持つ |
| E29 | `effect_value`（`set(` 16 か所）・`hand_joint` 5・`cut_price` 7 | 集合を反復して浮動小数を足す所があれば順序が要る。読んだ範囲では所属の判定だけ（`effect_value.py:457-463`）——1 つずつ確かめる |
| E30 | `sorted(..., key=...)` | Python の並べ替えは安定。Rust も安定な `sort_by` だけを使う |
| E31 | fixture の JSON（`harm_profile.json`・`opp_boards.json`） | `serde_json` は `float_roundtrip` 機能なしでは最後の桁を誤りうる（設計 §3.4）。機能を有効にするか Python から f64 で渡す |
| E32 | 効果の木 | `export_effects_json` は集合を `repr` で並べ替えて書く・列挙は名前。`effect_value` が `raw_text` に正規表現を当てる所が 4 か所（`effect_value.py:1043/1068/1232/1529`）——正規表現のクレートは無いので、手書きの照合か Python 側で旗を前計算 |
| E33 | 丸めた鍵の覚え書き（`_attach_gain` の `round(x,3)`・`_TW_MEMO`・`_C_OF_MEMO`・`_OPTION_CACHE`・`_uv_memo`） | 鍵が衝突すると「先に書いた方が勝つ」＝呼び出し順に依存しうる。鍵が正確か証明するか、同じ規則を写す |
| E34 | 負の数の `//`・`%` | Python は床・Rust は切り捨て。`py_floordiv`/`py_mod` を用意 |
| E35 | `max(seq, key=...)`／同点 | 最初のものが勝つ（設計 §4.1 の `py_max` の列版） |
| E36 | numpy のスカラー（`np.float64`）と Python の float の混在 | 0 除算で例外にならず inf になる等、端の挙動が分かれる。記録ビットの再生に端の例を入れる |
| E37 | 器を `__main__` で走らせたとき、同じファイルが別名で 2 度読み込まれる | 切替の値が 2 つの複製で食い違いうる（段 3 の報告で `OPCG_RD_KERNEL` について直した）。Rust では設定を明示の引数にして消す |
| E38 | Python 3.11 の `sum()` | 設計 E 一覧と同じ（3.12 以降は補償加算）。移植中は 3.11 で固定 |

### 5.2 層ごとの確かめ方（第 1〜3 段と同じ型）
1. **呼び出しの記録**: Python の各層の入口に記録の器を掛け、実・合成の各 5 局で入力と出力（浮動小数は 16 桁の 16 進ビット）を `jsonl.gz` に書く。
2. **cargo で再生**: Rust だけで記録を読みビットで比べる（Python 不要＝Python を消した後も残る答え合わせ）。
3. **PyO3 で突き合わせ**: `both` 運転（両方解いて違えば例外）で実・合成の 5 局を 1 通し。
4. **道具まるごと**: 段 6 で 8 器の JSON が 1 バイト一致（秒数欄を除く）。
5. **Python を消す前に**、2 の記録を最終のものに取り直してコミット＝これが新しい「正本の写し」。

---

## 6. 見積もり（悲観・作業担当の時間）

物差し【事実・依頼文】: 第 2a 段（Python 1.2k → Rust 2.9k・ビット一致）45 分・第 3 段 45 分・波 C 50 分・波 B 4.5 時間（大半は一致の通し）。
第 2a 段の前には設計の報告（E1〜E24 の洗い出し）が丸 1 本あった——これから移す部分にはまだ無い。

| 段 | 楽観 | 悲観 | 主な不確かさ |
|---|---|---|---|
| 0 片づけ（削除・固定外し・一致の通し） | 1.5 h | 3 h | 遅延 import・試験の付け替え漏れ |
| 1 入力と土台・記録の器 | 2 h | 5 h | 効果の木を Rust で読む形、カード表の項目 |
| 2 葉 | 2 h | 4 h | numpy の足し算・`erf` |
| 3 値付けの核 | 6 h | 14 h | 効果の木を歩く 1,800 行・辞書の多い Python・1 回はやり直す前提 |
| 4 守る側の外側＋耐久 | 2 h | 4 h | `_attach_gain` の覚え書き（E33） |
| 5 行の読み | 3 h | 6 h | 器ごとの差異 |
| 6 局の駆動＋道具まるごとの一致 | 2 h | 5 h | 2 回目以降の通し（1 回 ≈10 分 × 器 8 × 2 記録） |
| 7 削除・試験の付け替え（PORT 645）・文書 | 3 h | 6 h | PyO3 で同じ名前の関数を出せば試験の大半は差し替えだけ |
| ゲート（`make rust-develop`＋`make test` を段ごと） | 1 h | 2 h | |
| **計** | **≈23 h** | **≈49 h** | |

**提案する言い方: 30〜50 時間（5〜9 回の作業セッション）**。【推測】最初の 2 段が見込みどおり進むかで後半の幅が決まるので、段 2 の終わりで見積もりを言い直す。

---

## 7. 試験はどう置き換わるか
* **PORT 645 本**: 段 7 で Rust を呼ぶ同名の Python の薄い包み（例 `theory_rs.attack_value(...)`）に向け直せば、手計算の期待値はそのまま使える【推測】。
  Rust だけで閉じる性質（状態の数・予算の判断・記録ビット）は cargo へ。全部を cargo に書き直すのは 6,862 行の書き直しで、得るものが少ない。
* **番人 22 本**（§2.4）は記録ビットの再生（cargo）と道具まるごとの JSON 1 本（pytest）に置き換え。
* **RETIRE-OBSOLETE 82 本・RETIRE-WITH-TOOL 437 本**は削除（凍結ブランチと git の歴史に残る）。
* 試験の自動の固定（`conftest.py`）は段 0 で外す（43 本を消すか書き直す）。

---

## 8. ユーザに決めてもらうこと（CLAUDE.md の書式・言葉で）

### 判断 1: どの器を移し、どれを消すか
1. **何を決めるか**: 今の計測に使っている器と、その計算が頼っている部分だけを Rust に移し、閉じた課題のためだけに作った器を消してよいか。
2. **選択肢**: (a) 移すのは 26 本（実際に動く計算は約 1 万行）、閉じた課題の器 34 本（約 9,100 行）と試験 437 本を消す。(b) 全部移す（約 3 万行・見積もり 2〜3 倍）。(c) 閉じた器を Python のまま残す（理論の Python が消えると動かなくなるので、実際は (a) と同じ結果になる）。
3. **なぜ出てきたか**: 10-06 に「全部を Rust に移してから Python を全部消し、その後で速さの 2 段階を Rust で直す」と決まった。棚卸しすると器は 86 本あり、今の計測で動くのはその一部だった。
4. **推薦**: (a)。消す器はどれも 9 月 25 日以前の課題で、残す器から 1 度も呼ばれない（実測）。必要になれば凍結ブランチで動かせる。
5. **決めないと**: 移植の範囲が決まらず、見積もりが 2〜3 倍の幅のまま。
6. **留保**: 相手の場の分布の表（既定の計算が読む）を作り直す器も消える。作り直すときは凍結ブランチで。

### 判断 2: 据え置き中の「理論で手を止める実験」の道具
1. **何を決めるか**: 出口の実験（理論が損と言う手を止めて勝率が動くか）と、棋譜に理論値を注釈して目で見る道具（計 5 本＋その下請け 2 本・試験 146 本）をどうするか。
2. **選択肢**: (a) 対局を回す部分は Python のまま残し、理論の計算だけ Rust 版を呼ぶよう付け替える（約 1 時間の追加）。(b) 道具ごと Rust に移す（対局の駆動まで・数時間〜1 日の追加）。(c) 消して、再開するときに作り直す。
3. **なぜ出てきたか**: 9 月 24 日に「理論を整えてから同じ道具で測り直す」と据え置きにし、注釈の道具は「残す」と書いた。一方で道具は理論の Python を直接呼んでいる。
4. **推薦**: (a)。道具の大半は対局を回す仕事で、Rust にしても得が無い。理論の部分だけ付け替えれば据え置きの約束を守れる。
5. **決めないと**: Python の理論を消す段で、この 5 本が動かなくなる。
6. **留保**: 付け替えた後の数字は、理論の式が同じでも乱数の引きが違えば 9 月の数字と比べられない（実験の性質上もとからそう）。

### 判断 3: 答え合わせ用の Python を一切残さないか
1. **何を決めるか**: 消した後の「正しさの拠り所」を何にするか。
2. **選択肢**: (a) Python は全部消し、消す直前に取った「呼び出しごとの入力と出力のビットの記録」を正本の写しにする。(b) 守る側の計算の原文（約 500 行）だけ残す。(c) 速さの 2 段階が終わるまで Python 全体を残す。
3. **なぜ出てきたか**: 10-05 には原文を残すと決め、10-06 にそれを覆して全部消すと決めた。消すと、後で式を変えたときに比べる相手は記録だけになる。
4. **推薦**: (a)。10-06 の決定どおり。記録は Rust だけで再生できるので、Python が無い機械でも確かめられる。
5. **決めないと**: 段 7 で何を残すかが決まらない。
6. **留保**: 記録は「その時点の Rust の出力」で、正しさの独立した証拠ではない（golden と同じ注意）。式を意図して変えたら記録を取り直し、差分を読む運用が要る。

### 判断 4: 旧い既定（比較の基準）を Rust へ持っていくか
1. **何を決めるか**: 9 月末〜10 月初めに既定から外れた旧い形（切らせた札を一律の値段で読む・手札の項の旧い形・守る費用の旧い曲線・旧い守り手のパワー・値付けの直し無し・印字だけの数え方・自分の耐久を別の作り方で読む）を Rust にも書くか。
2. **選択肢**: (a) 書かない。それらで 9 月の数字を再現する試験（1 本）と旧い形を名指す試験 22 本を消し、再現は凍結ブランチで。(b) 全部書く（Rust が約 1 割増え、一致の確かめも倍）。(c) 印字だけの数え方（10-04 に「比較用に残す」と決めたもの）だけ書く。
3. **なぜ出てきたか**: 10-05 の切替の削除でも「比較の基準の値」と古い記録との同一性の試験は残した。Rust に移すとき、それも写すかが問われる。
4. **推薦**: (a)。ただし印字だけの数え方は 10-04 の明示の決定があるので、(c) との差を確かめてほしい。
5. **決めないと**: 移植する式の数が決まらない。
6. **留保**: (a) では、旧い形との差を今のコードで測ることはできなくなる（凍結ブランチを使う）。

### 判断 5: 保留中の候補の式を Rust へ持っていくか
1. **何を決めるか**: 既定ではないが「保留」として残っている形——ドンの配分ずれの値段（10-05 に残すと決めた）・攻撃時の能力と常在の体の値付け・耐久の対称な読み方・場の体の減衰・探す効果の 1 枚 1 役の読み・決着の時刻を局ごとに読む形・出す価値の長い勘定と短い勘定の切替 など——を写すか。
2. **選択肢**: (a) ユーザが明示で残すと決めたもの（ドンの配分ずれ）だけ写し、他は凍結ブランチへ。(b) 全部写す。(c) どれも写さない。
3. **なぜ出てきたか**: これらの試験 43 本が「保留の値」を使っている。写さなければ試験ごと消える。
4. **推薦**: (a) に加え、探す効果の 1 枚 1 役（N-5 で既定にする予定の候補）は写す。
5. **決めないと**: 段 3 の範囲が決まらない（多くは値付けの核の中の枝）。
6. **留保**: 写さなかった候補を後で試すときは、Rust に書き直すことになる。

### 判断 6: 試験の自動の固定（4 つ）
1. **何を決めるか**: 試験だけが使う古い算術（潜在価値を切る・平均の傾きを一定・生存の重みを一回・費用曲線の旧い引き方）をどうするか。
2. **選択肢**: (a) 固定を外し、それに頼る 43 本の試験を消すか今の式で期待値を書き直す。(b) Rust にも古い算術を写して固定を続ける。
3. **なぜ出てきたか**: 固定を 1 つずつ外して回した（実測）。落ちるのは 43 本だけで、平均の傾きの固定はもう何も守っていなかった。
4. **推薦**: (a)。
5. **決めないと**: 古い算術を Rust に書くかどうかが決まらない。
6. **留保**: 43 本のうち手計算の値を今の式で出し直すものは、1 本ずつ手で確かめる手間がかかる（約 1 時間）。

### 判断 7: 集計と JSON の書き出しをどこに置くか
1. **何を決めるか**: 行ごとの計算は Rust にするとして、ブートストラップ・AUC・較正などの集計と JSON の書き出しも Rust にするか。
2. **選択肢**: (a) Python に残す（約 1,000 行・薄い包み）。(b) Rust に移す（numpy の乱数と足し算をビットで再現する必要がある）。
3. **なぜ出てきたか**: 集計の一部は numpy の乱数を使っていて、Rust で同じ数を出すには乱数の生成器ごと写す必要がある。
4. **推薦**: (a)。集計は理論の式ではなく測定の後処理。
5. **決めないと**: 「Python を全部消す」の範囲の解釈が定まらない。
6. **留保**: 記録の読み込みとカード DB（パーサ）も Python に残る。

### 判断 8: 計画のディスク保存
1. **何を決めるか**: 守る側の計算の結果をディスクに保存する仕組みを、Rust 用に作り直すか、やめるか。
2. **選択肢**: (a) Rust の原文と入力だけで鍵を作り直す（一度冷える・約 1〜2 時間）。(b) やめる。
3. **なぜ出てきたか**: 鍵に Python の原文のハッシュが入っていて、Python が消えると意味を失う。
4. **推薦**: (a)。守る側の計算は今でも時間の 25〜60% を占め、表の作り直しのように同じ記録を何度も回す作業では効く。
5. **決めないと**: 段 4 の出口の形が決まらない。
6. **留保**: 鍵の付け方を誤ると古い計画を使ってしまう（今と同じ注意）。

### 判断 9: 中間ゲートの道具
1. **何を決めるか**: 「完成の 4 条件」を測る道具（9 月 17 日以降は使っていない）を移すか消すか。
2. **選択肢**: (a) 消す（凍結ブランチで動く）。(b) 移す（約 1 時間）。
3. **なぜ出てきたか**: 台帳の「完成の定義」の章でこの道具が名指しされている。
4. **推薦**: (a)。完成の判定は今は 2 つの橋で読んでいる。
5. **決めないと**: 試験 8 本の扱いが決まらない。
6. **留保**: 台帳の文言の直しが要る。

### 判断 10: 記録の読み込みと合成デッキの作り直しの場所
1. **何を決めるか**: 記録ファイルの読み込みと、合成の記録のデッキを種から作り直す処理を Python に残すか。
2. **選択肢**: (a) Python に残し、局ごとにまとめて Rust に渡す。(b) Rust で記録を直接読む（読み込みの部品は Rust に既にある）。
3. **なぜ出てきたか**: 合成の記録では、5 局しか読まなくても 300 組のデッキを作り直していた（測って判明・数秒〜14 秒）。
4. **推薦**: (a)。ついでに「使う種だけ作る」直しを入れる（値は変わらない）。
5. **決めないと**: 段 1 の入口の形が決まらない。
6. **留保**: 局ごとの受け渡しの費用は、守る側の計算に比べて小さいと見ているが未測定。

### 判断 11: 理論の式を持たない CPU の診断 15 本
1. **何を決めるか**: 9 月中旬にネットと探索を調べた道具（零和の検査・方針の地図など）も一緒に消すか。
2. **選択肢**: (a) 消す（試験 約 100 本も）。(b) 今のまま置く（理論の Python を消しても、1 本〔ライフ予算の監査〕を除いて動く）。
3. **なぜ出てきたか**: 理論の式は持たないので移植の対象ではないが、守りの支払いの道具を縮めるとライフ予算の監査は壊れる。
4. **推薦**: (a)。どれも 9 月 16 日以前で、訓練を再開するときは別の道具で測り直す見込み。
5. **決めないと**: 段 0 の削除の範囲が決まらない。
6. **留保**: 訓練の課題（台帳 §8.5）を調べるとき、凍結ブランチから持ち出す手間がある。

**独立性**: 1・2・9・11 は互いに独立で先に決められる。4・5・6 は「Rust に何を書くか」で互いに独立だが、3 と合わせて段 3 の範囲を決める。7・8・10 は境界の形で互いに独立。
**段 0 を始めるのに要るのは 1・6・11（と 4 の「079e73b8 の試験を消すか」）だけ**。

---

## 付録 A: 作ったもの（`scratchpad/port_plan/`）
* `tools.csv`（122 行）・`tests.csv`（1,378 行）
* `prof/*.prof`（cProfile 12 本）・`prof/summary.txt`・`prof/breakdown.json`・`prof/used_lines.json`・`prof/callgraph.txt`・`prof/layers.json`・`prof/noprof_times.txt`
* `work/`（`deps.py`・`closure.py`・`meta.py`・`testfuncs.py`・`classify_tools.py`・`classify_tests.py`・`layers.py`・試験の JUnit XML 6 本・`pinrun.sh`）
* `tree/`（`git archive HEAD` の写し・`conftest.py` に `PIN_SKIP` を足したもの。リポジトリとは無関係）

## 付録 B: 気づいた不具合・注意（範囲外・未修正）
* 合成の記録では `--limit-games` に関係なく全種のデッキを作り直している（§3.4）。
* `guard_afford` の先頭の import が `plan_value_map`・`race_state`（→ `plan_drift`）を道連れにする（既定では使わない）。
* `test_rd_kernel_plans` の 3 本は `rust/opcg_engine/tests/fixtures/` が無いと落ちる（写しで再現・リポジトリでは問題なし）。
