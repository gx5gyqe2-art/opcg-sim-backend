# 表（`tests/fixtures/harm_profile.json`）の作り直し——4ec745f の既定・Rust 版で（2026-10-05）

作業担当セッションの報告。出どころ: 引き継ぎ書 `2026-10-05_theory_orchestrator_handoff.md` §3・§4・§5-3
（ユーザ決定 2026-10-05「Rust 化の後に、輪郭 → ばらつき → 勝率の幅 → 平均の傾き の順で表を一度だけ作り直す」）。
手順の前例は `2026-09-24_sigma_rebuild.md`（T154）§0・§8。**式・切替・器のコードは触っていない（新定数ゼロ）**。

* 起点: `claude/cpu-spec-improvements-yw91jd`@`4ec745f`／記録: `claude/sigma-wip`@`18131d5`
  （実 = `w41` 300 局・合成 = `w39 w42` 900 局・全局）
* 設定: `OPCG_LOG_SILENT=1 OPCG_RD_KERNEL=rs`・`OPCG_PLAN_STORE`（リポジトリの外）・既定の切替のまま
  （`win_calib` だけ前例どおり `--pre-settle on`）
* 出力: ブランチ `claude/table-rebuild`・結果 json は `table_rebuild/results/`（`before_*`／`s2_*`／`s3_*`／`after_*`）

---

## 0. 結論（先に書く）

1. **表は全部書き換えた**（§1）。いちばん大きく動いたのは**合成の終局時刻のばらつき（曲線）1.154 → 2.133** と
   **平均の傾き 実 0.1894 → 0.0948／合成 0.2238 → 0.1589**（ほぼ半分）。輪郭も実・合成とも動いた。
2. **輪郭は 1 回で収まった**——新しい輪郭で `crossing_bridge` を回し直した出力は、事前の出力と
   **秒数欄を除き 647 項目すべて 1 ビット一致**（実・合成とも）。**理由（事実）: `crossing_bridge` の既定の出力は
   表を 1 つも読まない**（曲線の読みは記録自身から作った輪郭を使う）。なので手順 1〜2 の繰り返しは不要だった。
3. **予告 5 本のうち当たり 4・外れ 1**（§3）。外れは P3（相対の帳簿の偏りが −0.09 の側へ戻る）——
   **相対の帳簿は表の平均の傾きを読んでいない**（事実・§4）ので、2026-10-05 の仮説はこの道具では検証できない形だった。
4. 較正（決着前・`rel`）: 実は対数損失 0.5273 → 0.5266・ECE 0.0187 → 0.0168 と**わずかに良く**、
   合成は 0.6252 → 0.6314・ECE 0.0486 → 0.0584 と**わずかに悪く**なった。どちらもコインは下回ったまま。順位づけ（AUC）は不変。

## 1. 表（旧 → 新・`blockers`・cross 規約＝キーは出所の記録の名前）

| キー | 実（`real`） | 合成（`syn`） | 出所 |
|---|---|---|---|
| `sigma_t.blockers`（curve の `sigma_T`） | 0.994 → **1.242** | 1.154 → **2.133** | `s2_cb_*` `by_slope.curve.sigma_T` |
| `sigma_rel.blockers.theory` | 0.3078 → **0.2383** | 0.3301 → **0.2737** | `by_slope.theory.sigma_rel` |
| `sigma_rel.blockers.curve` | 0.2456 → **0.2262** | 0.2693 → **0.2567** | `by_slope.curve.sigma_rel` |
| `sigma_rel_whole.blockers.theory` | 0.4146 → **0.387** | 0.4068 → **0.3875** | `by_slope.theory.sigma_rel_whole` |
| `sigma_rel_whole.blockers.curve` | 0.4282 → **0.3874** | 0.4481 → **0.4427** | `by_slope.curve.sigma_rel_whole` |
| `w_bar.blockers` | 0.1894 → **0.0948** | 0.2238 → **0.1589** | `s3_tb_*` `stats.w_mean` |
| `n_scale0`（記録のみ） | theory 0・curve 2 | theory 18・curve 20 | `by_slope.*.n_scale0` |

輪郭（`real`／`syn`＝`harm_by_turn`・`theory_slope`＝`theory_slope_by_turn`・13 個・後ろは道具自身が最後の値で埋めた
＝`harm_profile(j_max=12)` の出力をそのまま写した）:

| | 旧 | 新 |
|---|---|---|
| 輪郭・実 | 0.0007, 0.0783, 0.1059, 0.1799, 0.2554, 0.2666, 0.2164, 0.1955… | 0.0007, 0.0775, 0.1073, 0.186, 0.2597, 0.2809, 0.2277, 0.226… |
| 輪郭・合成 | 0.0004, 0.0682, 0.1121, 0.2, 0.2468, 0.2155, 0.1884, 0.1531, 0.1426, 0.1001… | 0.0004, 0.0601, 0.1035, 0.1965, 0.248, 0.2283, 0.2022, 0.1708, 0.1558, 0.0971… |
| 理論の速さ・実 | 0.0, 0.0773, 0.1196, 0.1949, 0.2444, 0.3, 0.3114, 0.336… | 0.0, 0.0953, 0.1036, 0.1875, 0.2678, 0.3401, 0.3108, 0.3728… |
| 理論の速さ・合成 | 0.0, 0.0815, 0.1231, 0.186, 0.2277, 0.2453, 0.2526, 0.2438, 0.2225, 0.2162… | 0.0, 0.0947, 0.1227, 0.1913, 0.274, 0.298, 0.2964, 0.2951, 0.3223, 0.2315… |

触っていないキー: `attackable`／`all`（死んだ切替の入力）・`sigma_rel_floor`。`_note*` には旧 → 新を 1 文ずつ足した（既存の文は残した）。

## 2. 事前／事後の評価（同じ 4 つの道具・同じ記録）

| 量 | 実 before | 実 after | 合成 before | 合成 after |
|---|---|---|---|---|
| 較正 `win_calib` rel: 使った幅（`sigma_rel_for`） | 0.4068 | 0.3875 | 0.4146 | 0.387 |
| 　対数損失（コイン 0.692／0.693） | 0.5273 | **0.5266** | 0.6252 | **0.6314** |
| 　AUC | 0.8076 | 0.8076 | 0.7241 | 0.7241 |
| 　ECE | 0.0187 | 0.0168 | 0.0486 | 0.0584 |
| 　Brier | 0.1785 | 0.1784 | 0.2148 | 0.2161 |
| 有利側・不利側 `pre_settle_asymmetry`: 有利側の gap | −0.0152 | −0.0085 | +0.0568 | +0.0662 |
| 　不利側の gap | +0.0122 | +0.0050 | −0.0416 | −0.0518 |
| 相対の帳簿 `relative_ledger`（60 局）`abs_kappa` の偏り（0 が理想） | −0.180 | **−0.1835** | −0.1272 | −0.1167 |
| 　勝ち − 負けの平均（`sep`・1 が理想） | 0.561 | 0.5905 | 0.360 | 0.4138 |
| 　AUC | 0.8316 | 0.8283 | 0.6537 | 0.6549 |
| 　使った幅（curve・`sigma_rel_for`） | 0.4481 | 0.4427 | 0.4282 | 0.3874 |
| 交点の橋 `crossing_bridge` theory の勝者の的中 | 0.7048 | 0.7048 | 0.6602 | 0.6602 |
| 　theory の `sigma_rel`（測った値） | 0.2383 | 0.2383 | 0.2737 | 0.2737 |

注: 既定は `SETTLE_COND_MODE=whole` なので、較正・有利不利・帳簿が引く幅は `sigma_rel_whole` の表（`sigma_rel` ではない）。

## 3. 予告（司令塔が測る前に固定）と当否

| # | 予告 | 結果 | 当否 |
|---|---|---|---|
| P1 | 較正の AUC は ±0.002 以内 | 実 0.8076 → 0.8076・合成 0.7241 → 0.7241（変化 0） | **当たり** |
| P2 | after の対数損失は実・合成ともコインを下回る | 0.5266 < 0.692・0.6314 < 0.693 | **当たり**（ただし合成は before より 0.006 悪化） |
| P3 | 帳簿の `abs_kappa` の偏りは実で −0.19 付近から −0.09 の側へ、`sep` は 1.0 の側へ。合成も同じ向き | 実: 偏り −0.180 → −0.1835（**逆向きにわずか**）・`sep` 0.561 → 0.5905（1 の側へわずか）。合成: 偏り −0.127 → −0.117・`sep` 0.360 → 0.414（どちらも予告の向き） | **外れ**（主張の中心＝実の偏りが −0.09 へ戻る、が起きない） |
| P4 | 交点の橋の theory の的中は ±0.02 以内 | 0.7048 → 0.7048・0.6602 → 0.6602 | **当たり**（自明——§0-2 のとおり道具が表を読まない） |
| P5 | theory 読みの `sigma_rel` は表の値（0.3078／0.3301）から動く | 0.2383／0.2737（どちらも小さくなる向き） | **当たり** |

## 4. 事実と推測

**事実**:

* `crossing_bridge` の既定の出力は表を読まない（before と s2 の 2 回で 647 項目が一致）。表の輪郭は
  `theory_bridge`・`relative_ledger`・`kappa_vector`・`rate_tracking`・`transition_ledger` が `profile_for` で読む。
* `relative_ledger` は表の `w_bar` を読まない——κ の分母は `theory_order.W_BAR = 0.5 / R_TURNS`（コードの既定値）のまま。
  表の `w_bar` を `set_w_bar` するのは `theory_bridge` だけ（`theory_bridge.py` 871〜873 行）。帳簿が表から読むのは
  `sigma_rel_for(slope="curve")`（既定では `sigma_rel_whole.curve`）だけ。
* したがって P3 の仮説（「平均の傾きの分母が旧い物差しの幅のままだったのが悪化の原因」）は、表の書き換えでは
  帳簿に届かない。帳簿の after の変化は、幅（curve の `sigma_rel_whole`）が 0.4481 → 0.4427（実の行）・0.4282 → 0.3874（合成の行）
  に動いた分だけ。
* 平均の傾き `w̄` はほぼ半分になった（κ の平均＝`w_mean` ÷ 相手セットの `w̄`: 実 0.4236・合成 0.8392）。
  合成の `sigma_T`（curve）が 1.154 → 2.133 と倍近くになったのと同じ方向（幅が広がれば密度の平均は下がる）。
* `make test` の結果は §5。

**推測**（検証していない）:

* P3 を本当に試すには、帳簿の κ の分母を表の `w_bar`（の match 版）にする配線が要る——これはコードの変更（器）で、
  この作業の範囲外。
* 合成の較正がわずかに悪くなったのは、`sigma_rel_whole.theory` の合成の行が実のセットから取った値（0.4068 → 0.3875）に縮み、
  合成の記録の上では幅がやや狭すぎる側に寄ったため（有利側 gap +0.057 → +0.066・不利側 −0.042 → −0.052＝自信過剰の向き）。
* 合成の `sigma_T` の倍増（handoff §8 の下書き 2.131 と同じ）は、輪郭ではなく既定の構成（Rust 化までに入った標準の変更）
  そのものの性質。`crossing_bridge` は表を読まないので「古い輪郭の影響」という見立ては当たらない。

## 5. テスト

`make rust-develop` → `make test` を 1 回（結果は下に追記）。表の値を固定している期待値は見つからなかった
（`test_settle_cond_floor.py` の 0.3078 は式の検算用の任意の数で、表を読まない＝変えていない）。
表を読む不等式（`sigma_rel_floor` < `sigma_rel.theory` < `sigma_rel_whole.theory`）は新しい値でも成り立つ
（0.2084 < 0.2383 < 0.387・0.1946 < 0.2737 < 0.3875）。

## 6. 再現

```bash
git fetch origin claude/sigma-wip && git archive origin/claude/sigma-wip sigma_wip/data | tar -x -C /home/user/tr
make rust-develop
bash table_rebuild/run.sh before                       # 手順 0（OPCG_RD_KERNEL=rs・OPCG_PLAN_STORE を設定する）
python table_rebuild/write_table.py contour before     # 手順 1: harm_by_turn → real/syn・theory_slope_by_turn → theory_slope
bash table_rebuild/run.sh s2                           # 手順 2: crossing_bridge を回し直し
python table_rebuild/write_table.py sigma s2           #   sigma_t / sigma_rel / sigma_rel_whole（blockers）
bash table_rebuild/run.sh s3                           # 手順 3: theory_bridge
python table_rebuild/write_table.py wbar               #   stats.w_mean → w_bar.blockers
bash table_rebuild/run.sh after                        # 手順 5
```

所要（Rust 版・計画の保存あり）: `crossing_bridge` 実 6 分・合成 約 18 分、`theory_bridge` 実 20 分・合成 65 分、
評価 3 つは各 1〜7 分。
