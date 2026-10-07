# 理論の覚え書きを正確にする（M-2 の前の独立した 1 変更）——全部の覚え書きの一覧・直したもの・正確さの証拠・効果（2026-10-07）

起点: `claude/cpu-spec-improvements-yw91jd` @ `a4a5755ab`（段 7 の後）。ユーザ決定（10-07）: 移植で**ビットごと写した覚え書きの癖**
（E39・E52・`_OPTION_CACHE`・`_GAIN`・E60・E61・段 5／6 の E74〜E78）を今直す——**覚え書きが返してよい値は、解き直した値と同じ値だけ**
（鍵は値が読む入力の全部を持つ・計数は 1 つの出来事を 1 度だけ数える）。式と既定は変えない（覚え書きの正しさの直しで、模型の変更ではない）。

読み方: **【実行】**＝この作業で回して確かめたこと。**【推測】**＝それを元にした見立て。

---

## 0. 要点

1. **理論の Rust の覚え書きを全部洗い出した**（§1・25 項目＋計数 4 つ）。癖があったもの（丸めた鍵・文脈の抜け・先勝ちの共有）は
   **12 個**——`_OPTION_CACHE`・`_GAIN`・`_FLOW`（E39）・選択の分布・`JointValuer` の `_plan`／`_val`（E52）・曲線の `L`／`ḡ`・値段の枠の
   曲線と補正・手札の値の `_uv_memo`（丸めた `ḡ` と状態の一部だけ）・`_attach_gain`（E60）・`_RULE_DON_CACHE`（E61）・計画のディスクの鍵・
   `theory_bridge` の `g_cache`（E77）・`_AVD_MEMO`（丸めた `ḡ`・ただし `CUT_TAKE_CARD` のビットが鍵に入っていて局の駆動では実害なし）。
   全部「値が読む入力の全部（値段の文脈は `ḡ` のビット・効果の値を通るものは核の文脈と切替の全部）」を鍵にした。計数は
   `cut_causal_skip` の 2 度数え（E76）・`g_for` の 2 度呼び・鏡の行の計数の行き先（E75）・鏡の行の `t_left`（E74・出力に出ない欄）を直した。
2. **正確さの証拠**【実行】: 当たるたびに解き直してビットで比べる検算（`OPCG_MEMO_CHECK`）を 8 器 × 3 入力（`f_identity/rec`・実 w41・
   合成 w39 の各 5 局）で回した——**旧い鍵では 9,631 件が違い**（`option` 8,483／145,851・`flow` 666／79,218・`tb_g_cache` 475／1,092・
   `gain` 6／188・`sel` 1／10,099）、**直した後は 2,007 万件の検算で 0 件**。検算つきの通しの出力は検算なしと 24/24 一致。
   覚え書きを引かない通し（`OPCG_MEMO_OFF`）の出力は、解いた回数の計数を除き覚え書きつきと一致（§3.3）。新しい `cargo test`
   （`tests_memo` 8 本）は旧い鍵の写しで 8 本とも落ち、直した後は全部通る。
3. **効果（5 局の切れ端）**【実行】: 値の欄は小さく動く（`relative_ledger`・`transition_ledger` は 4 桁目以下・最大 6.5e-4）。
   大きく動くのは計数（鏡の行の計数を捨てたので `g_n`・`rule_n` などが約半分）と `theory_bridge` の行ごとの `D` の帯（`g_cache` を行ごとに
   した）。`crossing_bridge`・`win_calib`・`pre_settle_asymmetry`・`price_realised`・`kappa_vector` の値の欄は 1 つも動かない（§4.1）。
4. **全記録**（実 w41 300 局・合成 w39＋w42 900 局）【実行】: 交点の橋と勝率の較正は実質不変（実の勝者の的中 0.7048→0.7051・
   偏り 0.026→0.027・log-loss 0.5266→0.5267・合成は全部同じ・σ_T／sigma_rel／sigma_rel_whole は実・合成とも同じ）。`theory_bridge` は
   `w_mean` 0.0948→0.0961（実）・0.1589→0.1603（合成）・`dS_mean` 0.00123→0.00401・−0.0298→−0.0359・帯ごとの較正が動く（判定は不変）。
   `relative_ledger` の `abs_kappa` は偏り・AUC とも 4 桁目以下（§4.2）。**損害の輪郭の表**を作り直すと変わるのは **`w̄` の 2 つだけ**
   （0.0948→0.0961・0.1589→0.1603・輪郭／理論の傾き／σ_T／sigma_rel_whole は同じ・表は書き換えていない・§5）。
   時間は `theory_bridge` が遅くなった（5 局の w39 21→32 秒・行ごとの手札の値段）・他はほぼ同じ（§7）。`make test` は green（§8）。
5. **golden**【実行】: `theory_core` 413 行・`theory_outer` 355 行・`theory_rows` 9 行の戻りを Rust の出力で作り直した（入力は不変・
   `pre` は全部の行から外した）。作り直しは Rust だけの器（`tests_regen`・§6）。守る側の計算の指紋のファイルは触っていない＝
   `SOLVER_VERSION` は上げていない。
6. **別に見つけて直した既存の落ち**【実行】: 合成の全記録（w39＋w42）では `a4a5755ab` が `py_str: 未対応の型 List([...])` で落ちる
   （条件の値が素性の並びのとき）。Python の `str(list)` を写して直した（`9bda9dcc0`）。比べる基準（`a4a5755ab` の写し）にも同じ 1 か所を
   当てて回した（§4.2）。

---

## 1. 覚え書きの一覧と分類

`grep`（`HashMap`／`KeyTable`／`Vec` の引き表／`Option` の 1 度だけの値・`memo`／`cache`）で `rust/opcg_engine/src/theory/**` を全部見た。
「解き直し」の検算の名前（§3）を括弧に書く。

### 1.1 癖があった（直した）

| # | 覚え書き（Python の名前） | 場所 | 旧い鍵の癖 | 新しい鍵 | 検算 旧→新（違い／検算） |
|---|---|---|---|---|---|
| 1 | `_OPTION_CACHE`（`option_value`） | `core/to.rs` | パワー・リーダーのパワーを 100 単位・`R` を帯 1〜5・θ 4 桁・μ 5 桁・ko_p 4 桁・`ḡ` 12 桁・`CUT_TAKE_CARD` は有るかだけ・先勝ち | 全部の引数のビット＋`ḡ` と `CUT_TAKE_CARD` のビット（`Ctx::price_key`） | **8,483**／145,851 → 0／150,041 |
| 2 | `search_price._GAIN`（`card_gain`） | `core/sp.rs` | `_ctx_key`（手札の `(cid, round(v,6))`・`caps`・`round(x,1)`・`round(take,6)`・`round(olp,1)`・`round(r,3)`）＝デッキ・場・`st_base`・札のコスト／カウンター／イベントを持たない・`ḡ` 12 桁 | 手札の読み `ctx` の中身の全部（`key_deep`）＋札＋核の文脈と切替の全部（`Core::ctx_key`） | **6**／188 → 0／8 |
| 3 | `deck_refill._FLOW`（`a_of`・E39） | `core/entry.rs` | `round(olp,1)`・θ／μ なし・`ḡ` 12 桁・`CUT_TAKE_CARD` は有るかだけ | デッキ・`olp`・ドンの上限・速攻だけか・付与つきか・θ・μ・`price_key` | **666**／79,218 → 0／78,572 |
| 4 | 選択の分布 `_CACHE["sel"]`・`_sel_premium` | `core/ev.rs` | 最初に解いた文脈の分布をプロセスの間ずっと使う | 核の文脈と切替ごと | **1**／10,099 → 0／10,016 |
| 5 | `theory_bridge` の `g_cache`（E77） | `core/drv_tb.rs` | `(席, ターン)`＝同じターンの後の行が最初の行の手札の値を受け取る・相手の読みの落ち先 `(seat, opp["t"])` と同じ鍵の空間を先勝ちで共有 | 行（局の中の位置）＋核の文脈 | **475**／1,092 → 0／647 |
| 6 | `JointValuer._plan`／`_val`（E52） | `core/hj.rs` | 残った札だけ（手札を読み直す物は値が値段の文脈に依るのに文脈なし） | 読み直す物は（残った札, 核の文脈と切替）・読み直さない物は残った札だけのまま（値が文脈に依らない） | 0／194,534（`jv_plan`）・0／1,374（`jv_val`）→ 0／0。この通しでは窓をまたいで同じ物を引く場面が無かった【実行】。`tests_memo` の作った場面では旧い鍵で違う |
| 7 | 曲線の `L`・`ḡ`（`_L`・`_gbar_memo`） | `core/curve.rs` | 曲線ごとに 1 度（最初に解いた文脈の値） | `JointValuer` の文脈の部分ごと | 0 → 0（この通しでは曲線を作った文脈でしか引いていない） |
| 8 | 値段の枠の曲線と補正（`CutFrames` の `curves`・`corr`） | `core/cutframes.rs` | 枠・`(席, ターン)` だけ（曲線の手札の値は作ったときの文脈で決まる） | （枠, 核の文脈と切替） | 0 → 0（同上） |
| 9 | 手札の値の `_uv_memo` | `core/hp.rs` | 札・相方・時計を進めた欄 `_PROJ_KEYS`・丸めた `ḡ`（`olp`／`r`／場・状態の残りの欄は 1 つの読みの中で同じ） | 札・相方・`olp`・`r`・場・状態の中身の全部・核の文脈と切替 | 0／135,772 → 0／135,772 |
| 10 | `_attach_gain`（攻め手の財布の `_gain`・E60） | `core/outer.rs` | `(round(x,3), k)`・値段の文脈なし | `x`・`k` のビット＋`price_key` | 0／12.5M → 0／14.3M（`tests_memo` の作った場面では旧い鍵で違う） |
| 11 | `_RULE_DON_CACHE`（E61・＋地平つきの鍵） | `core/outer.rs` | 守る側の入力を並べ替え・ライフを丸め・財布は `actx["key"]`（値の表を 12 桁に丸めた要約）・値段の文脈は丸めた `ḡ` | 守る側の入力をそのまま（並べ替えない・ライフも丸めない）・財布の dict の中身の全部・`price_key`・`CUT_OTHER_SIDE > 0` | 0／884 → 0／614（`tests_memo`: 財布の表を 12 桁より下だけ動かすと旧い鍵で違う） |
| 12 | 計画のディスクの覚え書き（`OPCG_PLAN_STORE`） | `core/store.rs` | `cut_context_key`＝丸めた `CUT_PRICER_KEY` | `cut_context_key` を `ḡ` のビットに（財布の dict の `cp` も同じ） | 検算の通しではディスクを使っていない。全記録の通しは版ごとに別の場所（§4.2） |
| 13 | `_AVD_MEMO`（`attack_value_don`） | `core/to.rs` | `ḡ` は 12 桁に丸めた `CUT_PRICER_KEY`・`CUT_TAKE_CARD` はビット＝局の駆動（窓が両方に `ḡ` を入れる）では実害なし | `price_key` | 0／4.06M → 0／4.25M（`tests_memo`: `CUT_TAKE_CARD` を共通にして `ḡ` だけ違えると旧い鍵で違う） |

付随: 記録の再生の `pre`（Python が前から在った値を読んだ丸めた鍵の覚え書きを前もって入れる）は読まなくなった（鍵が正確なので要らない）。
記録の財布の dict の `_gain` も読まない（`Actx::from_v`）。記録の財布の `cp` が旧い形（`("avg", round(ḡ,12))`）で今の窓の `ḡ` を丸めた値と
同じなら今の形に直す（`entry_outer::legacy_cp`・golden の入力のためだけ）。

### 1.2 鍵が正確だった（そのまま・検算は付けたものだけ）

| # | 覚え書き | 場所 | 理由 | 検算 |
|---|---|---|---|---|
| 14 | `_BODIES_MEMO`（`attack_stream` の相手の体） | `core/to.rs` | 鍵＝相手の体・`mlp`・`r_turns`・θ・μ・ko_p・`_OPTION_DEPTH > 0`。値は `_nu_of_other_side`（値段の文脈を外して解く）だけ | 0／0.86M → 0／0.89M |
| 15 | `_hand_stats`（読み直しの間の手札の要約） | `core/hp.rs` | 鍵＝札・カウンター・値・イベント・`xs`・`take`・探すか。デッキは 1 つの読みの中で同じ | 0／5,703 |
| 16 | `_RULE_PLAN_CACHE`（`rule_guard_plan3`） | `core/outer.rs` | 並べ替えた鍵だが、値（付与 0 の守りの計画）は並びに依らない（中で並べ替える）・ライフは中で丸める | 0／284 |
| 17 | `GuardTable`／`GuardPlan`（動的計画の表）・守る側の動的計画（`defender`・`SetCache`・`layers`・共有の `glob`） | `hj.rs`・`outer.rs`・`defender.rs` ほか | 1 つの問題の中の状態の表・文脈の鍵（`ctx_ids`）は問題の入力の全部（第 1〜3 段で確かめた） | — |
| 18 | `info`・`ident`・`enabler`・`has_cond` | `core/state.rs`・`sp.rs`・`hp.rs` | 札の番号だけの関数（カード表と効果の木は不変） | — |
| 19 | `pyval_of`（`V` → `PyVal` の写し） | `core/state.rs` | 番地で引くが、写しの元を生かしておく（番地が再利用されない）・`V` は不変 | — |
| 20 | `lethal_rule` の `avg_cache` | `core/drv_lr.rs` | 1 局ごとに空で始める（`carry: {}`）・鍵＝席＝その局のデッキ | — |
| 21 | 局の駆動の `rate_at`・`g_at`・`g_self`・`hb_self` ほか | `drv_*.rs` | 覚え書きではなく**定義**（「そのターンの最初の行の値」）——鍵が違う行の値を返すことは無い | — |
| 22 | Python の器の表（`pre_settle_asymmetry` の山のばらつき・`deck_refill._DB`／`_SEAT_DECKS`） | `tests/scripts` | `(seed, 席)` のデッキの関数・DB の読み込み | — |

### 1.3 計数（1 つの出来事を 1 度だけ）

| # | 何か | 旧 | 新 |
|---|---|---|---|
| C1 | `CutFrames.view` の `frame_key`（E76） | 2 度呼んで `cut_causal_skip` を 2 度数える | 枠を 1 度だけ引く |
| C2 | 交点の橋の `seat_row` の `g_for` | `r_opp` のためにもう 1 度呼び `g_*` を 2 度数える | 行ごとに 1 度（`r_opp` は同じ値を使う） |
| C3 | 鏡の行の計数（E75） | 行の計数は捨てるが `g_*`・枠の引きの `cut_view_flat`・`RULE_STATS` の行の読み（`rule_*`・`plan_*`）は本物に | **行ごとの計数は全部捨てる**。**解いた回数**（枠を作った `cut_frames` ほか・守る側の計算の `ex:*`）は、どの行が最初に引いたかに依らず本物に 1 度 |
| C4 | 鏡の行の `t_left`（E74） | 外のループの席の `ts` で数える | 鏡の席の残りの自席ターン（Python が渡していた引数）。この欄は出力に出ない（値は不変） |

---

## 2. 直し方

* **値段の文脈の鍵** `Ctx::price_key`: 窓の `ḡ` のビットと、窓があるときだけ `CUT_TAKE_CARD` のビット（窓が無いと `CUT_TAKE_CARD` を読む式は無い）。
* **核の文脈と切替の鍵** `Core::ctx_key`: `price_key`＋`CUT_OTHER_SIDE > 0`＋`_OPTION_DEPTH > 0`＋`FLOW_PRICING`＋`ATTACK_DON_COST_MODE`＋
  `SEARCH_VALUE_MODE`＋`UNKNOWN_FACTOR`。効果の値（`card_value`）を通る覚え書きに足した。
* **中身の鍵** `obj::key_deep`: dict（挿入順の対の並び）・numpy の配列（型・形・バイト）も中身ごと。数は従来の鍵と同じ規則（型をまたいで値で・`-0.0 == 0.0`）。
* 鍵の丸めを外したので**当たりが減ったもの・増えたもの**がある（§7 の時間）。値は変えない（式は 1 文字も変えていない）。

---

## 3. 正確さの証拠

### 3.1 当たるたびの検算（`OPCG_MEMO_CHECK`）【実行】

`core/memock.rs`: 環境変数 `OPCG_MEMO_CHECK` があると、§1 の覚え書き（1〜16 と枠の曲線・補正）に当たるたびに本体を解き直し、覚えた値と
**ビットで**比べる（返すのは覚えた値・解き直しの副作用〔条件の計数・`RULE_STATS`・局の `stats`〕は戻す）。解き直しの中の当たりは検算しない
（入れ子にすると覚え書きの無い木を丸ごと解き指数的に遅い）＝「覚えた値＝今の下の覚え書きの答えから組んだ本体の値」を当たりのたびに確かめる。
プロセスの終わりに `MEMOCK <名前> checked=<n> bad=<m>` を書く。

8 器（`crossing_bridge`・`theory_bridge`・`relative_ledger`・`transition_ledger`・`price_realised`・`win_calib --pre-settle on`・
`pre_settle_asymmetry`・`kappa_vector`）× 3 入力（`f_identity/rec`・実 w41・合成 w39 の各 5 局＝`--limit-games 5`／`--games 5`）:

| 覚え書き | 旧い鍵（`9f1178450`＝検算の器だけ足した写し） 違い／検算 | 直した後 違い／検算 |
|---|---|---|
| `option` | **8,483** / 145,851 | 0 / 150,041 |
| `flow` | **666** / 79,218 | 0 / 78,572 |
| `tb_g_cache` | **475** / 1,092 | 0 / 647 |
| `gain` | **6** / 188 | 0 / 8 |
| `sel` | **1** / 10,099 | 0 / 10,016 |
| `sel_prem` | 0 / 10,066 | 0 / 9,975 |
| `avd` | 0 / 4,061,406 | 0 / 4,252,498 |
| `attach_gain` | 0 / 12,531,307 | 0 / 14,328,320 |
| `bodies` | 0 / 859,700 | 0 / 889,838 |
| `uv` | 0 / 135,772 | 0 / 135,772 |
| `jv_plan`・`jv_val` | 0 / 194,534・0 / 1,374 | 0 / 194,486・0 / 1,374 |
| `hand_stats` | 0 / 5,703 | 0 / 5,703 |
| `rdc` | 0 / 884 | 0 / 614 |
| `rpc` | 0 / 284 | 0 / 284 |
| `curve_L`・`curve_gbar` | 0 / 1,236・0 / 5,791 | 0 / 1,236・0 / 5,791 |
| `cut_curves`・`cut_corr` | 0 / 5,791・0 / 1,797 | 0 / 5,791・0 / 1,797 |
| **合計** | **9,631** / 18,052,093 | **0** / 20,072,763 |

旧い鍵の違いの例: `option` 1.5524e-2 対 解き直し 1.5735e-2（パワーを 100 単位に丸めた鍵の先勝ち）・`flow` 1.4493950673771510e-2 対
1.4493950673771506e-2（`ḡ` の 12 桁の丸め）・`gain` 0.1087 対 0.0810・`tb_g_cache` は同じターンの別の行の手札の読み。
**検算つきの通しの出力は検算なしの通しと 24/24 一致**（計数も・`seconds` だけ除く）【実行】。全部の器が rc=0。

### 3.2 `cargo test` の見張り（`core/tests_memo.rs`・8 本）【実行】

直した覚え書きごとに、旧い鍵では同じ鍵になる 2 つの入力を温めた核で続けて解き、後の値が新しい核で解いた値とビットで同じこと・入力の違いが
値を変えること（空振りしない）を見る（`option_value`・`attack_value_don`・`a_of`・`card_gain`〔golden の 40 行〕・選択の分布・
`_attach_gain`・`JointValuer`〔golden の `jv.value` の全部＋窓の外と中〕・`_RULE_DON_CACHE`〔golden の守る側の全部＋財布の表を 12 桁より下だけ動かす〕）。
**同じ 8 本を旧い鍵の写し（`9f1178450`）に入れると 8 本とも落ちる**（当たり ≠ 解き直し）・直した後は全部通る。所要 約 30 秒（debug・並列）。

### 3.3 覚え書きを引かない通し（`OPCG_MEMO_OFF`）【実行】

§1.1／1.2 の覚え書き（選択の分布を除く——引かないと 1 呼び出しごとに 2,803 枚を解き直し通しが終わらない・§3.1／3.2 が見る）を
**全部引かない**通しの出力を、覚え書きつきと比べた。
8 器 × `rec`（2 局）で回した（覚え書きを引かないと遅い: `crossing_bridge` 979 秒・`theory_bridge` 1,931 秒・`relative_ledger` 2,152 秒・
`transition_ledger` 1,952 秒・`win_calib` 1,039 秒・`pre_settle_asymmetry` 1,015 秒・`price_realised` 6 秒・`kappa_vector` 3 秒＝実・合成の 5 局は回していない）。
**8 器とも、違うのは値段の枠を作った回数の計数だけ**（`cut_frames`・`cut_mlp_next`／`_rule`・`cut_cand_sum`・`cut_L1_n`／`_sum`・
`cut_corr_*`＝覚え書きを引かないと枠の曲線と補正を引くたびに作り直して数える）。**値の欄（`summary`・`arms`・行・較正・`RULE_STATS`）は
1 つも違わない**＝覚え書きは値を変えていない（通しで）。

---

## 4. 効果

### 4.1 5 局の切れ端（8 器 × 3 入力・基準 `a4a5755ab`〔自前の wheel〕→ 直した後）【実行】

「変わった値の数／値の総数・最大の絶対差・最大の相対差」。`seconds` は除く。表の「計数」は数え方を直した欄（§1.3）。

| 器 | rec | w41 | w39 | 変わった欄 |
|---|---|---|---|---|
| `crossing_bridge` | 15／596 | 17／610 | 19／612 | **計数だけ**: `stats` の `g_*`（`g_n` は 1/4＝鏡の行を捨て 2 度呼びをやめた〔w41 240→60〕・平均 `g_mean` は本物の行だけの平均になった〔rec 0.0392→0.0386・w41 0.0359→0.0362・w39 0.0367→0.0377〕）・`rule_n`／`plan_n`／`rule_*_sum`／`rule_don_*`（約 −50%・鏡の行の読み）・`sched_j5_sum`（w41 23.3487→23.3501・6e-5＝`option` の正確な値）・`cut_corr_sum`（1e-16）。`summary` の値（交点・σ・表）は 1 つも動かない |
| `theory_bridge` | 21／279 | 27／309 | 53／310 | `stats.w_mean`（rec 0.0986→0.0980・w41 0.0922→0.0944・w39 0.1507→0.1522）・`kappa_sum`／`kappa_mean`・`D` の帯の行の数 `d_bins`／`d_win`（`g_cache` を行ごとに）・`kappa_needed`・`kappa_split`（4 桁目）・`ledger_rescored`・`rule_stats`（`rule_cut_sum` ほか） |
| `relative_ledger` | 0 | 9／154 | 16／154 | `arms.*.sep`／`bias`／`w0_gap`・`K_mean`（いずれも ±1〜2e-4・4 桁に丸めた値の最後の桁）・`cut_price.cut_L1_sum`（3〜13e-5 相対） |
| `transition_ledger` | 0 | 10／76 | 10／76 | `by_axis_mean`・`by_cause_mean`・`by_family`・`resid_abs_per_gap`（±1〜6e-6）・`priced_share`（w41 ±1e-4） |
| `win_calib --pre-settle on` | 6／138 | 6／168 | 6／168 | **`rule_stats` だけ**（鏡の行の読みの計数）。勝率の較正（log-loss・AUC・ECE）は不変 |
| `pre_settle_asymmetry` | 0 | 0 | 0 | — |
| `price_realised` | 0 | 0 | 0 | — |
| `kappa_vector` | 0 | 0 | 0 | — |

詳しい表（欄ごと）は作業の記録 `effect_slices.txt`（下の §9 の場所）。

### 4.2 全記録（実 w41 300 局・合成 w39＋w42 900 局）【実行】

器の既定（`crossing_bridge`・`win_calib --pre-settle on`・`relative_ledger --games 60`・`theory_bridge`）。基準は `a4a5755ab` の写し（自前の wheel・
合成は §9-1 の 1 か所を当てた）・直した後は `9bda9dcc0` の木の wheel（後のコミットは golden・試験・診断の切替〔`OPCG_MEMO_OFF`〕だけ＝既定の通しの値は同じ）。計画のディスクは版ごとに別の場所。

| 見出し | 実 w41 旧 → 新 | 合成 w39＋w42 旧 → 新 |
|---|---|---|
| 交点の橋（`theory`）の勝者の的中 `sign_accuracy` | 0.7048 → **0.7051** | 0.6602 → 0.6602 |
| 同 偏り `bias`（ターン） | 0.026 → **0.027** | 0.386 → 0.386 |
| 同 `σ_T`・`sigma_rel`・`sigma_rel_whole` | 1.412・0.2383・0.387 → 同じ | 2.004・0.2737・0.3875 → 同じ |
| 同 `within1` | 0.6732 → **0.6735** | 0.6688 → 0.6688 |
| 交点の橋（`curve`）`sign_accuracy`・`bias`・`σ_T`・`sigma_rel`・`sigma_rel_whole` | 0.6081・0.582・1.242・0.2262・0.3874 → 同じ（`within1` 0.5577 → **0.5574**） | 0.6097・1.105・2.133・0.2567・0.4427 → 同じ |
| 勝率の較正 `win_calib`（`rel`）log-loss・AUC・ECE | 0.5266・0.8076・0.0168 → **0.5267**・0.8076・0.0168 | 0.6314・0.7241・0.0584 → 同じ |
| 同（`abs`）log-loss・AUC・ECE・Brier | 0.6090・0.7991・0.0666・0.1876 → **0.6091**・0.7991・**0.0667**・**0.1877** | 0.8639・0.7182・0.1245・0.2331 → 同じ |
| `theory_bridge` の平均の傾き `w_mean` | 0.0948 → **0.0961** | 0.1589 → **0.1603** |
| 同 `κ` の平均 `kappa_mean` | 0.5965 → **0.6046** | 1.6766 → **1.6909** |
| 同 `dS_mean`（`dS_sd`）・`auc_all` | 0.00123（0.5717）・0.5519 → **0.00401**（**0.5653**）・**0.5518** | −0.02982（1.9103）・0.4461 → **−0.03587**（**1.9098**）・**0.4408** |
| 同 判定 `verdict`・`gain_verdict` | `no_link`・`bridge_holds` → 同じ | 同じ |
| `relative_ledger`（60 局）`abs_kappa` の偏り・AUC | −0.1835・0.8283 → 同じ（`corr` 0.5698 → **0.5697**） | −0.1167・0.6549 → **−0.1166**・0.6549 |
| 同 `rel_K` の AUC・偏り・`K_mean` | 0.8967・−0.079・0.4213 → 同じ（`K_mean` **0.4212**） | 0.7349・−0.2209・0.4905 → 同じ |

変わった値の数（器ごと・`seconds` を除く・作業の記録 `effect_full.txt`）: `crossing_bridge` 実 66／646・合成 52／646（`theta_check` の
`tau_matched` の行の数 ±1・`by_slope` の 4 桁目・ほかは計数）・`win_calib` 実 22／170・合成 9／170（較正の箱の 4 桁目と `rule_stats`）・
`relative_ledger` 実 31／154・合成 30／154（腕の `slope`／`corr`／`auc` の 4 桁目・最大の相対差 1.6%＝合成の `rel_flat.bias` −0.0063→−0.0062・実の `rel_exact.auc` 0.8507→0.8496）・
`theory_bridge` 実 371／866・合成 409／883（**帯ごとの較正と `dS`／`dG`**——`g_cache` を行ごとにしたので行ごとの `D` の帯が動く・
帯の勝率の箱で最大 0.069・`kappa_needed` の行の数 ±1〜2）。

**読み**【推測】: 交点の橋と勝率の較正は実質不変（4 桁目が 1 つ動く程度）——覚え書きの癖は小さな値の差（`option` の丸め・`ḡ` の 12 桁）で、
これらの器の出力の丸めに届かない。大きく動いたのは `theory_bridge` の行ごとの読み（`g_cache` の E77）で、帯ごとの集計の中身が入れ替わる。
判定（`verdict`）は変わらない。

---

## 5. 損害の輪郭の表（`tests/fixtures/harm_profile.json`）——作り直すと何が変わるか（**表は書き換えていない**）

`table_rebuild/write_table.py` の規則（輪郭・平均の傾き＝`crossing_bridge` の `harm_profile`・`σ_T`／`sigma_rel_whole`＝`by_slope`・
`w̄`＝`theory_bridge` の `stats.w_mean`）で全記録の通しから値を作った。**基準（`a4a5755ab`）の通しは表と全部同じ値**（表は 10-05 の作り直しの
値・そのまま再現する）【実行】。

| 表の欄 | 表（＝基準の通し） | 直した後の通し |
|---|---|---|
| 輪郭 `real`（13 欄）・`syn`（13 欄） | [0.0007, 0.0775, 0.1073, 0.186, 0.2597, 0.2809, 0.2277, 0.226…]・[0.0004, 0.0601, 0.1035, 0.1965, 0.248, 0.2283, 0.2022, 0.1708, 0.1558, 0.0971…] | **同じ** |
| 平均の理論の傾き `theory_slope`（実・合成） | [0, 0.0953, 0.1036, 0.1875, 0.2678, 0.3401, 0.3108, 0.3728…]・[0, 0.0947, 0.1227, 0.1913, 0.274, 0.298, 0.2964, 0.2951, 0.3223, 0.2315…] | **同じ** |
| `sigma_t.blockers`（実・合成） | 1.242・2.133 | **同じ** |
| `sigma_rel_whole.blockers`（`theory` 実・合成／`curve` 実・合成） | 0.387・0.3875／0.3874・0.4427 | **同じ** |
| **`w_bar.blockers`**（実・合成） | **0.0948・0.1589** | **0.0961・0.1603** |

**作り直すなら変わるのは `w̄` の 2 つだけ**（+1.4%・+0.9%）。表は書き換えていない（ユーザの判断が要る）。`w̄` を読むのは `theory_bridge`
の `κ = w/w̄` の分母（`relative_ledger` は表の `w_bar` を読まない＝10-05 の報告）【推測・読み手は確かめていない】。

---

## 6. golden の作り直し【実行】

器: `rust/opcg_engine/src/theory/core/tests_regen.rs`（`OPCG_THEORY_GOLDEN_WRITE=1 cargo test --no-default-features tests_regen -- --include-ignored`・
既定は数えるだけ）。各行を試験と同じ道（`tests_core`／`tests_outer`／`tests_rows` の `run_one`・同じ順・同じ核）で解き、戻りの欄
（`r`・`cs`・`ev`／`result`）が違う行だけ書き換える。入力の欄は変えない（`golddiff.py` で全行を確かめた）・`pre` は全部の行から外した。
手順は `rust/opcg_engine/tests/fixtures/README.md` に書いた。

| golden | 変わった行 | 内訳（関数・行数・最大の絶対差／相対差） | なぜ |
|---|---|---|---|
| `theory_core_golden`（9,984 行） | **413**（戻り `r` が変わったのは 389・残りは計数 `cs` だけ） | 戻り: `to._nu_of_other_side` 323／840（≤1.0e-3・≤0.38%）・`to.option_value` 21／840（≤8.0e-4・≤8.7%）・`sp.search_value` 17／140（≤9.1e-3・≤11%）・`sp.card_gain` 12／420（≤2.8e-2・≤26%）・`hs.free_value` 6（≤1.6e-4）・`dr.a_of` 3（≤6.9e-18）・`hp.apply_inflow` 2・`hs.use_value` 2（≤8.0e-4）・`to.nu_of` 2（≤2.8e-17）・`ev.card_value` 1（4.0e-4）。計数だけ: `sp.search_value` 13・`sp.card_gain` 5・`to._don_cost_total` 4・`ev.card_value` 2 | 記録の `r` は Python が丸めた鍵の覚え書き（`pre`）から読んだ値。前もって入れなくなり正確な値になった（`option` の 100 単位・`_GAIN`・`_FLOW`）。計数 `cs` は解き直した分だけ増える |
| `theory_outer_golden`（4,950 行） | **355** | `cb.attacker_ctx` 276／280（`cp` と `key` の形 276・`a_tab` 113・`flow` 63・`blk_a` 44・≤4.2e-4）・`cb.opp_blockers_of` 60／507（≤4.2e-4・≤0.13%）・`hp.hand_items` 7・`hp.search_context` 7・`tb.guard_hand_reading` 5（≤8.0e-4・≤2.3%） | 財布の `cp`（値段の文脈）を `ḡ` のビットにした（ほぼ全部の行）・`option` の正確な値が相手の体の `ν` と表へ |
| `theory_rows_golden`（113 行） | **9** | `main/rec/crossing_bridge` 1（鏡の行の計数・`ev` の長さ・`turn_harm[].d_blk_nu`）・`main/rec/theory_bridge` 1（`per` の帯ごとの `g`・`s`・計数）・`main/w39/kappa_vector` 2・`main/w39/transition_ledger` 2・`main/w41/transition_ledger` 3（≤6.5e-4） | `g_cache` を行ごとに・鏡の行の計数・`option` の正確な値 |
| `theory_leaves_golden`・`rd_*` | 0 | — | 覚え書きを通らない |

`cargo test theory`（44 本・作り直した後）は全部通る【実行】。

---

## 7. 時間【実行・壁時計・秒・8 器の合計】

5 局の切れ端は、全記録の通しが終わった後の静かな機械で基準と直した後を続けて回した（計画のディスクなし・どちらの出力も前の通しと 24/24 一致）。

| 器 | rec 前 | rec 後 | w41 前 | w41 後 | w39 前 | w39 後 |
|---|---|---|---|---|---|---|
| `crossing_bridge` | 4.0 | 4.4 | 6.8 | 6.6 | 6.3 | 6.2 |
| `theory_bridge` | 7.2 | 9.2 | 20.4 | 23.8 | 21.1 | **31.8** |
| `relative_ledger` | 5.6 | 5.7 | 16.0 | 15.1 | 14.5 | 14.6 |
| `transition_ledger` | 2.6 | 3.0 | 3.1 | 3.8 | 3.3 | 3.4 |
| `price_realised` | 2.4 | 2.5 | 3.0 | 2.9 | 2.8 | 2.6 |
| `win_calib` | 4.0 | 4.2 | 6.2 | 7.4 | 6.0 | 6.2 |
| `pre_settle_asymmetry` | 3.7 | 4.5 | 7.0 | 8.3 | 6.3 | 7.0 |
| `kappa_vector` | 2.3 | 2.1 | 2.5 | 2.5 | 2.5 | 2.7 |
| **合計** | **31.8** | **35.6** | **65.0** | **70.4** | **62.8** | **74.5** |

**遅くなったのは主に `theory_bridge`**（w39 +51%）——`g_cache` を行ごとにしたので手札の値段（`hand_price_mean`）を行ごとに解く【推測・
行の数だけ解く回数が増える】。他は ±1 秒前後（器 1 本の床 約 2.5 秒の中の揺れ）。全記録（計画のディスクつき・並行して別の通しが走って
いた時間があるので目安）: `crossing_bridge` 実 375→401 秒・合成 1,279→1,405・`win_calib` 67→95・266→258・`relative_ledger` 211→212・
175→182・`theory_bridge` 実 1,217→1,478・合成 4,159→5,116。

---

## 8. `make test`

【実行】`37d4176ba` の木で `make rust-develop`（114 秒）→ `make test` を 1 回（221 秒）: **`cargo test` 473 passed・1 ignored（`tests_regen` の作り直し）・
pytest 621 passed・12 skipped・失敗 0**。この報告（`docs/` だけ）はゲートの後に足した。

---

## 9. 別に見つけたこと・残り

1. **合成の全記録で `a4a5755ab` が落ちる**（`py_str: 未対応の型 List([Str("魚人族"), Str("人魚族")])`・`cond.rs` の `holds_deck`＝リーダーの素性の
   条件の値が並びのとき）。Python は `str(want) in traits`（並びの `str` は `"['魚人族', '人魚族']"` で常に偽）。`py_str` に list／tuple の
   Python の `str` を足した（`9bda9dcc0`・`cargo test` 1 本）。5 局の切れ端と実 w41 の全記録はこの道を通らない（基準の出力は変わらない）。
   基準の合成の全記録はこの 1 か所を当てた写しで回した。
2. `theory_bridge` の守りの行（`drv_tb.rs` 681 行）は相手の窓の `ḡ` を `cut_me_fr`（自席の枠）から引いている（攻撃の行〔499 行〕は
   `cut_opp_fr`）——式の写しなので直していない【推測・Python の原文と同じかは確かめていない】。
3. `OPCG_MEMO_CHECK`・`OPCG_MEMO_OFF` は診断の切替（既定は無効・値を変えない）。
4. `cargo clippy -D warnings` は `src/ops.rs` の既存の 1 件（`collapsible_match`・新しい clippy の lint）で落ちる——この作業の外。

## 10. コミット

`git log --oneline a4a5755ab..memofix`:

```
ad802a0d9 報告（このファイル）
37d4176ba OPCG_MEMO_OFF は選択の分布を対象外に
b13e85f5d tests_regen の clippy・TEST_SPEC に tests_memo／tests_regen の行
67a611cf7 golden の戻りを作り直す（core 413・outer 355・rows 9 行・pre を外す）
9bda9dcc0 py_str の list／tuple（合成の全記録の既存の落ち）
28ef935ee tests_memo（当たり＝解き直しの見張り 8 本）・fixtures README
358e8e28e tests_regen（golden を Rust だけで作り直す器）・記録の財布の旧い cp
8145e5728 覚え書きの鍵を全部正確に・計数を 1 度ずつ・OPCG_MEMO_OFF
9f1178450 覚え書きの検算の器（OPCG_MEMO_CHECK）——鍵はまだ変えない
```

作業の記録（一時の場所・`/tmp/claude-0/-home-user/4b89ca80-da31-5772-bfe4-312f948e10f9/scratchpad/mf/`）: `MEMOCK_old.txt`・`out/ck_new/MEMOCK.txt`・
`effect_slices.txt`・`effect_full.txt`・`old_tests.log`・`regen.log`・`full/{base,new}/`。
