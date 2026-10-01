# WP `rs-select-default`: 対象なし解決の原因分析（2026-09-08）

`docs/rust_engine_plan.md` §8.27／分析 #3 §5 結論 5 の追跡。判定はコーディネータ、分析は本 WP。

## 先に結論

- **(a) 万雷カウンターの BUFF 空選択**と**(b) 神の裁きの KO 空選択**は、**別の原因**で起きている。
  §8.27 は「既定解決 `choose_selection` が自分側を常に min 件（コスト系）にする」1 本の欠陥に
  まとめていたが、実際に手を返した経路をたどると (a) と (b) は違う場所で壊れている。
- **(a) は既定解決の欠陥＋タイブレークの合わせ技**: 自分のリーダーとキャラが混在する対象（`zone`
  が `"leader"`／`"field"` に分かれる）は `choose_selection` の `zones_in(&["hand","field"])` に
  一致せず `None` になり、フォールバック `uuids[:min_n]`（`min_n=0`＝「1 枚まで」）で**空**になる。
  この空選択が `get_legal_actions` の唯一の既定手として**候補の先頭**に来る（`merged_search_actions`
  の重複排除の順序）。さらに、このバフは `THIS_BATTLE` 持続＝**評価する頃には戦闘が終わって効果が
  切れている**ため、空／サトリ選択／リーダー選択の 3 枝が**価値関数上で完全に同値**になる。
  `best_branch` は同値なら**先頭を残す**ので、たまたま先頭にある空選択が勝つ。
- **(b) は既定解決の欠陥ではない**。相手側（対象系）の候補は `choose_selection` の `all_opp` 分岐で
  正しく「候補を選ぶ」を返す（空にならない）。実際に手を返した経路を追うと、**空選択と非空選択が
  同値ではなく、価値関数が「KO しない」方を実際に高く評価していた**（`resolved_branch_values` の
  値を直接ログした：`-0.5376` (KO) vs `-0.5006` (何もしない) 等）。これは分析 #3 §1.2/§1.3 で既に
  観測されている**除去イベントの過小評価**が、単発の対話選択（対象を選ぶかどうか）というもっと
  細かい粒度でも起きているということ。**§8.27 の「既定解決のゾーン意味論バグ」という説明は (b) には
  当てはまらない**。
- 修正は **(a) のみ**を対象にした。(b) は「モデルの価値評価が除去を過小評価する」という既存の
  既知課題（分析 #3 §5-2）の一形態であり、対話の既定解決やタイブレークを直しても再発しうる
  （ネットの訓練データ／符号化側の問題）。今回はコードを変えていない（**fix_included: false**）。
  変更の要否・優先度はコーディネータ判断。

以下、Q1〜Q6 に沿って証拠を示す。

## 計測方法

`tests/scripts/rs_scenario_play.py` の `RsGame._decide` を丸ごと差し替え、Rust `Game.decide` の
生の戻り値 `out`（`kind`／`sig`／`commit`／`stats.legal`／`groups`）を `decisions[i]["_raw_out"]` に
残すようにした（差し替えスクリプトは `/tmp/.../scratchpad/instrumented_play.py`・本コミットには
含めない使い捨てツール）。加えて `rust/opcg_engine/src/search/quiesce.rs::resolved_branch_values`
に `OPCG_DEBUG_SELECT=1` 環境変数で有効化する一時的な `eprintln!`（`legal` と `vals` をダンプ）を
入れて 1 回だけビルドし、該当シナリオを再走した後、**この計測用コードは元に戻した**（このブランチの
diff には残っていない）。再現に使ったコマンドは WP 指示書のとおり:

```bash
OPCG_LOG_SILENT=1 python tests/scripts/rs_scenario_play.py play \
  --scenario enel_human_20260810_t3-4 --net opcg_sim/data/learned/nrel_r3.npz --seeds 2 --sims 160 --out /tmp/sel
OPCG_LOG_SILENT=1 python tests/scripts/rs_scenario_play.py play \
  --scenario human_enel_vs_luffy_20260904_t4-5 --net opcg_sim/data/learned/nrel_r3.npz --seeds 2 --sims 160 --out /tmp/sel
```

(a) は `enel_human_20260810_t3-4` の seed 0（`r3_s0.md` 手 #11〜#12）、(b) は
`human_enel_vs_luffy_20260904_t4-5` の seed 1（`r3_s1.md` 手 #10〜#13）で再現した。

## Q1: 手を返した経路（kind: main／window／commit）

### (a) 万雷カウンターの BUFF 空選択

`r3_s0.md` 手 #11（turn=4 SELECT_COUNTER）〜手 #12（SEARCH_AND_SELECT）の `_raw_out`:

```
決定11: kind=window, sig=SELECT_COUNTER(万雷),
        commit=[{"sig":["RESOLVE_EFFECT_SELECTION",null,[],[],true]},
                {"sig":["PASS",null,[],[],null]}]
        move={"kind":"battle","action_type":"SELECT_COUNTER","card_uuid":"8f8198e8...(万雷)"}
決定12: kind=commit, commit=[{"sig":["PASS",...]}]
        move={"kind":"game","action_type":"RESOLVE_EFFECT_SELECTION",
              "payload":{"selected_uuids":[],...}}     ← ここで BUFF targets:[] が確定
決定13: kind=commit
        move={"kind":"battle","action_type":"PASS",...}
```

**手を決めたのは決定 #11（`kind=window`）**。5 択（SELECT_COUNTER×4 + PASS）の窓評価
（`search/decide.rs::window_choice`）が「万雷でカウンターする」を選んだ**その場で**、
`commit_window_continuation`（同ファイル）が続きの対話（BUFF 対象選択→PASS）を
`resolve_battle_inplace` で**先読みして焼き込んで**いる。決定 #12・#13 はその焼き込み済みの
`commit` を機械的に消化しているだけ（`kind=commit`＝`commit_step` が `find_move` で一致する手を
適用するだけで、選び直しはしない）。つまり**BUFF の空選択は「対話の窓」ではなく「その手前の
戦闘カウンター選択窓が内部で行ったクイエス評価」の中で決まっていた**。これが `(候補なし)` と
`.md` に出る理由（決定 #12・#13 は `stats.legal` が空＝`commit` 消化は候補を持たない）。

### (b) 神の裁きの KO 空選択

`r3_s1.md` 手 #10（PLAY 神の裁き）〜手 #13（KO の SEARCH_AND_SELECT）の `_raw_out`:

```
決定10: kind=main（MCTS 木で PLAY 神の裁きを選択）
        commit=[{"sig":["RESOLVE_EFFECT_SELECTION",null,[],["5ee3...(エネル)"],true]},
                {"sig":["RESOLVE_EFFECT_SELECTION",null,[],[],true]}]   ← 2 手目が KO 空
        move=PLAY 神の裁き
決定11: kind=main（SELECT_RESOURCE＝ドン!!-1 の返却先選択。commit=[] で独立に決定）
決定12: kind=window（BUFF 対象選択。commit を消化せず窓で決め直し、
        同じ「エネルを選ぶ」に到達し、続きの KO 空選択を自分の commit として焼き直す）
        commit=[{"sig":["RESOLVE_EFFECT_SELECTION",null,[],[],true]}]
決定13: kind=commit（KO の空選択を機械的に適用。ここで targets:[] が確定）
```

**手を決めたのは決定 #10（`kind=main`＝MCTS 木の直後に走る `commit_play_dialog`）**。
「PLAY 神の裁き」を選んだ直後、`commit_play_dialog`（`search/decide.rs`）が
`resolve_battle_inplace(Window::Dialog, box_depth=1)` で BUFF→KO の対話を**先読みして**
`commit` に焼き込む。決定 #12 は（`SELECT_RESOURCE` が `commit` に挟まったせいで一度 `carry` の
`(turn, seat)` 鍵がずれ）窓を取り直しているが、**同じ盤面・同じ乱数系列なので同じ答え（BUFF は
エネル・KO は空）に収束**し、決定 #13 がそれを消化する。(a) と違い、決定 #10 の時点で**すでに
KO は空で焼き込まれている**＝(b) の分岐は決定 #12 の窓ではなく決定 #10 のツリー探索直後の
先読みで起きている。

## Q2: (a) 万雷カウンターの選択窓 — 合法手の数と勝因

`OPCG_DEBUG_SELECT=1` でダンプした `resolved_branch_values` の入力／出力（対象選択が絡む
呼び出しだけを抽出。これは決定 #11 の内部で `commit_window_continuation` → 入れ子の
`resolve_battle_inplace(Window::Battle, box_depth=1)` が 2 段目の分岐を評価している箇所）:

```
box_depth=0 window=Battle
legal=[RESOLVE_EFFECT_SELECTION(selected=[]),                 ← 空（既定解決からの重複排除で先頭）
       RESOLVE_EFFECT_SELECTION(selected=[サトリ f45ae0b5]),
       RESOLVE_EFFECT_SELECTION(selected=[リーダー 41a8a61d])]
vals=[Some(-0.3961171805858612), Some(-0.3961171805858612), Some(-0.3961171805858612)]
```

**合法手は 3 つ**（空／サトリ／リーダー）＝`adapter::selection_moves` は正しく分岐している
（`min_n=0` なので「選ばない」も候補になるのは仕様どおり）。`request_actor`／`action`／`uuids`／
`min`／`max` のどれかで枝生成が落ちたわけではない。**3 枝の価値が小数点以下 15 桁まで完全一致**
している。原因は「このバトル中」持続の BUFF が、評価する時点（`resolve_battle_inplace` が
`Window::Battle` を抜けた後＝戦闘解決後）にはすでに失効しており、盤面のエンコードに差が
出ないため。`best_branch`（`quiesce.rs`）は `>` 比較で同値なら**先頭を残す**契約なので、
**先頭に来た「空」が機械的に勝つ**。

先頭に来る理由: `rules::legal.rs::get_legal_actions` は対話状態を「妥当な既定解決」1 手として
返す（`default_interaction_payload`）。この対象は自分のリーダー(`zone="leader"`)とサトリ
(`zone="field"`)の**混在**なので `choose_selection`（`effects/interact.rs:1008-1019`）の
`zones_in(&["hand","field"])`（リーダーは含まない）に一致せず `None`。フォールバック
（`uuids[:take]`・`take = min_n.max(0).min(max_n).min(n) = min(0,1,2) = 0`）で**空**になる。
`adapter::merged_search_actions` は `base_moves`（＝この空の既定解決 1 手）を先頭に置き、
`selection_moves` が生成した alt（[サトリ, リーダー, 空]）のうち「空」は同じ鍵として重複排除
されるので、**最終的な並びは [空, サトリ, リーダー]** になる。これが tie 時に「空」を選ばせる
直接の原因。

## Q3: (b) 神の裁きの KO 空選択 — コミット／窓／枝評価のどれか

**枝の評価**（値の比較そのもの）。`OPCG_DEBUG_SELECT` のログから、KO 選択の 2 択
（`selected=[ルフィ]` と `selected=[]`）は**先頭がルフィ**（`choose_selection` の `all_opp` 分岐が
機能して既定解決は正しく非空になるため、Q2 の (a) とは並びが逆になる）。にもかかわらず複数回の
評価で「選ばない」が上回った:

```
legal=[RESOLVE_EFFECT_SELECTION(selected=[ルフィ 51f4f70a]),
       RESOLVE_EFFECT_SELECTION(selected=[])]
vals=[Some(0.049433...),  Some(-0.015628...)]   ← ルフィが勝った回（3 回中 1 回）
vals=[Some(-0.537643...), Some(-0.500643...)]   ← 「選ばない」が勝った回
vals=[Some(-0.524546...), Some(-0.491620...)]   ← 「選ばない」が勝った回
```

3 サンプル中 2 回で「選ばない」の評価値が高い。これは (a) のような**完全な同値（タイブレーク）
ではない**——ネット自身が「パワー 0 のルフィを KO するより放置する方が良い」と評価している。
`BOX_RESOLVE_DEPTH=1` により、この KO 選択の**先の展開**（KO 後にどちらが得かの数手先）は
`quiesce_choice`（方策優先→PASS 優先→先頭）で機械的に打ち切られる**浅い**先読みしか見ていない
ため、除去の長期的な価値（分析 #3 §1.2-3 で報告済みの「除去イベントの事前分布／価値が低い」現象）
が短い箱の中で正しく反映されない。**`choose_selection`／`default_interaction_payload` のゾーン
意味論はここでは無罪**（相手側候補は正しく最大件・価値降順を返す）。

## Q4: 「自分側＝コスト系」の既定解決が実際は利益系である割合

`opcg_sim/data/opcg_effects.json` の全カードの効果木を走査し、「対象が自分（`player=SELF`）・
ゾーンが手札／場（`zone in {HAND, FIELD}`）・`is_up_to=true`（＝§8.27 のバグが刺さる形＝
`min_n=0` の「N 枚まで」）」の `GameAction` を数えた（1 アビリティに複数アクションがあれば
それぞれ 1 件）。

| 区分 | 件数 | 判定基準 |
|---|---:|---|
| 該当アクション総数 | 660 | 自分・up_to・hand/field 系 |
| 利益系（`BUFF`／`GRANT_KEYWORD`／`RAMP`／`DRAW`／`ACTIVE`／`ADD_COUNTER` 等） | 339（51%） | 選ぶほど得 |
| コスト系（`REST`／`KO`／`TRASH`／`RETURN_DON`／`DISCARD`／`DEBUFF` 等・自分に対して） | 13（2%） | 選ぶほど損＝min が正しい |
| 分類不能（`SEARCH`／`ARRANGE`／`MODIFY_COST` 等・文脈依存） | 308（47%） | 個別判定が必要 |

`choose_selection` が「自分側＝コスト系」と仮定している場面のうち、**少なくとも半分は実際には
利益系**で、min 件（＝しばしば 0 件）を選ぶのは誤り。代表 10 件:

| カード | トリガー | 効果 |
|---|---|---|
| 万雷 (OP15-078) | COUNTER | 自分のリーダーかキャラ1枚までを、このバトル中、パワー+1000 |
| 神の裁き (OP15-075) | ACTIVATE_MAIN | 自分のリーダーかキャラ1枚までを、このターン中、パワー+1000 |
| 神避 (OP10-019) | COUNTER | 自分のリーダー1枚までを、このバトル中、パワー+3000 |
| 神避 (OP13-076) | COUNTER | 自分のリーダーかキャラ1枚までを、このバトル中、パワー+3000 |
| 放電 (OP15-074) | COUNTER | 自分の「エネル」1枚までを、このバトル中、パワー+2000 |
| 雷獣 (OP15-076) | COUNTER | 自分の「エネル」1枚までを、このバトル中、パワー+2000 |
| EB01-019 | COUNTER | 自分のリーダーかキャラ1枚までを、このバトル中、パワー+4000 |
| EB02-007 | ACTIVATE_MAIN | 自分のリーダーとキャラ合計3枚までを、このターン中、パワー+1000 |
| EB02-018 | ON_PLAY | 自分のリーダー1枚までは、このターン中、【ダブルアタック】を得る |
| EB03-001 | ACTIVATE_MAIN | 自分の【アタック時】効果を持たないキャラ1枚までは、このターン中、【速攻】を得る |

## Q5: Python 版（`py-engine-final`）でも同じ盤面で空選択になるか

**未確認**。`legacy/python_engine/core/engine/interaction.py::choose_selection` は Rust の
docstring どおり同じゾーン意味論（自分の手札／場＝min 件）を持つため、(a) と**同じ構造の欠陥**
（自分のリーダー・キャラ混在で `None`→`uuids[:0]`）は理論上再現するはずだが、Python 側の
CPU（`cpu_ai.py`／`mcts.py`）は探索の実装（箱化・タイブレーク規則）が Rust 版と完全に同一とは
限らず、(a) を再現させるには **同じ tie が起きる保証がない**。(b) はそもそも「ネットの価値評価」
起因なので、Python 側で別のネット（c10 等）を使えば違う結果になりうる。時間の制約でこの WP では
実行して確かめていない（tag `py-engine-final` の checkout・別環境構築が必要）。

## Q6: 修正案の比較

| # | 案 | 変更箇所 | (a) を直すか | (b) を直すか | 影響範囲 | Python との一致 |
|---|---|---|---|---|---|---|
| ① | 既定解決に効果種別（利益／コスト）を渡し、利益系の自分側 up_to は max 件・価値降順にする | `effects/interact.rs::choose_selection`／`default_interaction_payload` のシグネチャ変更＋呼び出し元（`suspend_for_target_selection` は `GameAction.type` を知っているのでそこから配線）／`rules/legal.rs`／`search/adapter.rs` | ○ 直る（既定解決そのものが正しい対象を返すようになるので、Q2 のタイブレークで先頭に来ても実害が無くなる） | × 直らない（(b) はゾーン意味論の問題ではない） | **大**: 監査 golden 3,386 件・再生 golden 200 局の両方が変わりうる（自分側 up_to の対話がある能力全部＝Q4 の 660 件が対象）。差分レビューが必須 | 崩れる（Python は既存のまま）。ユーザ決定 2026-08-25 により許容 |
| ② | 探索の枝で対話を既定解決で畳まず、`BOX_RESOLVE_DEPTH` を対話の連鎖ぶん深くする（浅い先読みを止める） | `search/quiesce.rs`（`BOX_RESOLVE_DEPTH` 定数／`resolve_battle_inplace` の再帰条件） | △ 部分的（タイブレークの原因＝評価時点でバフが失効している問題は深さを増やしても直らない。同値のまま） | △ 部分的（(b) の「浅い先読みで除去の価値が見えない」は緩和されうるが、除去の真の価値は数手先の展開に依存するため根本解決ではない上、分岐が指数的に増える） | **大**: 探索コスト（レイテンシ）が対話の連鎖数に応じて跳ね上がる。sims を維持すると生成／アリーナが遅くなる | 変わらない（探索の内部実装のみ） |
| ③ | 適用側で `can_skip=False` かつ候補ありの空選択を拒む | `effects/interact.rs`（`SelectTarget` の解決）または `default_interaction_payload` | × **効かない**（下記） | × **効かない** | — | — |

③ は**前提が誤り**だったことが今回わかった: `can_skip` は `InteractionKind::SelectTarget` では
**常に `false`**（`suspend_for_target_selection`・`effects/interact.rs:140`）で、真の必須／任意は
`constraints.min` が持つ。(a)(b) はどちらも `is_up_to=true`＝`min=0`＝**正当に「選ばない」が
許される**選択なので、「`can_skip=False` なら空を拒む」を実装すると**この 2 件は直らず**、かつ
本来 0 件で正しいはずの選択（Q4 の「コスト系」13 件や、本当に対象が無い場面）まで壊れる。
③ は不採用（この分析で判明した新事実）。

**推奨: ①**。(a) はこれで根本から直る（Q2 の「タイブレークで先頭が勝つ」という運任せの挙動が、
そもそも「正しい既定値」に置き換わることで問題化しなくなる）。ただし変更が Q4 の 660 件（カード
効果の対象選択全般）に及ぶため golden 差分は大きくなる見込みで、**必ず差分をレビューする**こと。
(b) は①②のどちらでも直らない、別カテゴリの課題（ネットの除去過小評価・分析 #3 §5-2）として
切り離し、評価帯／訓練データ側の是正（分析 #3 の「次にやること 2」）で扱うべき。

## 本 WP での対応

原因は Q1〜Q3 で確定したが、①の実装は「対象種別を対話システム全体に配線する」設計変更で、
影響が Q4 の 660 件・golden 全体に及ぶため**局所修正とは言えない**と判断し、**このセッションでは
コードの修正は行わなかった**（`fix_included: false`）。修正の要否・実施タイミングはコーディネータ
判断に委ねる。

## 参考: 決定的な証拠ログの要約

- (a): `決定#11` (`kind=window`) の内部評価で `legal=[空,サトリ,リーダー]`・
  `vals=[-0.3961171805858612]×3`（完全一致）→ `best_branch` が先頭（空）を返す。
- (b): `決定#10` (`kind=main`→`commit_play_dialog`) の内部評価で `legal=[ルフィ,空]`・
  `vals` は `[0.049,-0.016]`（ルフィ優位）と `[-0.538,-0.501]`／`[-0.525,-0.492]`（空優位）が混在
  し、空優位が多数派。
