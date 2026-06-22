# CPU 強化（強さ=Elo 優先・フェア制約）ロードマップ メモ（2026-06-22）

> 設計メモ（点）。CPU の主目的を **強さ（Elo）** に置き直し、かつ **フェア制約**（相手手札・裏ライフを
> 読まない）の下で追う版。コード接地レビュー（αβ探索 `cpu_ai`／マクロMCTS `cpu_mcts`／アリーナ
> `tests/cpu_arena.py`／学習スキャフォールド）を経て確定。進捗の正本は WBS（`gx5gyqe2-art/WBS` の
> `projects/opcg-sim-backend.md`）。実装が固まった内容は `docs/SPEC.md §2.5` に吸収する（本メモは時点
> スナップショットとして改変しない）。

## 0. 確定スコープ・前提（ユーザ判断）

- **作るもの = フェアな最強 CPU を 1 体**（難易度ラダー easy/normal/hard は今回作らない）。
- **チートは即フェアに切替**（出荷デフォルトを fair へ。一時的な弱化を許容し、Phase 2 の PIMC で回復・超過）。
- **強さは「まず無制限レイテンシで測り、後で 1 秒に最適化」**（探索診断と本番最適化を分離）。

これにより dual-shipping・フラグ分岐・出荷ゲートの複雑さは消え、設計が一本化される。

## 1. コード接地で判明した重要事実（暗黙仮定の補正）

1. **出荷 hard はチート**: `cpu_ai.decide` が difficulty に関係なく `see_opp_hand=True, opp_public_only=False`
   をハードコード（相手のカウンター総量・隠れ登場手まで読んで min を取る強い形のチート）。easy/normal は
   decide から実質消えている。
2. **アリーナは hard vs hard 限定**: `cpu_arena.py` の challenger/baseline choices が `["hard"]` のみ。
   fair vs 凍結 fair が測れない。
3. **フェア化は配線済み（殺されているだけ）**: `decide` の 2 行（`see_opp_hand`/`opp_public_only`）を
   `False, True` にすれば `evaluate`/`_search`/`_consumes_hand_card`/`_estimate_counter_buffer` が
   全部フェア経路へ切り替わる。
4. **フェア化は"別種の歪み"を生む**: チートを外すと相手 min ノードが「相手はカウンターも手札からの
   登場もできない」超楽観モデルになり、止まる攻撃を"通る"と誤読して突っ込む。→ **決定化はこの穴を
   埋める必須要素**（"あれば良い"から格上げ）。
5. **決定化の土台は既存**: `cpu_mcts._determinize_opponent`（相手手札の再サンプリング）と worlds>1 の
   root 手投票（`MCTS_WORLDS`/`MCTS_DETERMINIZE`・既定 OFF）が実装済み。ただし本番 `decide` 経路は
   αβ のみで MCTS に繋がっていない。
6. **決定化は現状 uniform**: 既知デッキリスト（`cpu_opponent_model` のテンプレート）を参照しない。
   informed 化（推定デッキ母集団から引く）は **改造規模"小"** で質が大きく上がる。
7. **線形 value model は未配線（αβ 葉に blend なし）**: blend は MCTS 葉のみ。`cpu_features` は
   `see_opp_hand=False` 固定でラベル分散に相手手札起因の交絡が残る（線形不発 val acc 0.645/0.67 の一因）。

## 2. 設計の核（強さ × フェアの両立）

- **強さ = 探索の質 × 評価の質**。片方だけ伸ばすと壁。バイアス評価のまま探索だけ深くすると地平線で
  バイアスを増幅する（horizon amplification）。
- **フェア制約下では決定化(PIMC)が中心**。相手の隠れ情報を K 通りサンプルして各々を完全情報 αβ で
  解き、**毎手再計算＋root 手投票**で集約する。これがフェア化の損失（相手過小評価の楽観突撃）を埋める。
  - PIMC 病理: **non-locality は OPCG では無視可**（カウンター在庫を隠す戦略的価値が薄い）。
    **strategy fusion は root 投票＋毎手再計算で緩和**（`decide` がステートレス毎手再探索なので相性良。
    ターンプラン replay より毎手 PIMC 再計算が安全）。
- **決定化は相手起因の評価誤差を正則化する**が、自分側・option value・時間地平線の誤差は直交。
  → 評価改善(Phase 3)の対象は「決定化で消えない残差」だけに縮む。

## 3. 測定基準（二層）

- **主指標 = 凍結 fair-hard 比 Elo**: チートを外した時点の公平 αβ を凍結スナップショットとして保存し、
  以降の全フェア CPU をこれに対する Elo で測る（フェア制約下で単調に強くなったか）。
- **参考 = cheat-hard 比勝率**: 完全情報の上限性能の代理。情報欠損ギャップをどれだけ埋めたかの進捗
  メータ（例 30%→42%）。**主指標にしない**（永遠に 50% に届かない設計）。
- **崩落検出**: `realize_trace`（value-realization gap）/`regret_trace` をフェア経路でも回し、楽観突撃
  （big_gaps）の増減を定量化＝決定化導入前後の改善証拠。

## 4. フェーズ・合格ゲート・依存順

### Phase -1 — 前提整備（規模:小・最優先）
- `decide` の情報方針を引数化し、**出荷デフォルトを fair に即切替**（固定値撤廃）。
- `cpu_arena` を多ポリシー対応（`fair_hard`/`cheat_hard`/`fair_pimc`…）。
- **デッキ配り seed と方策 rng を分離**（CRN の土台）＋ **PIMC world 用 rng を親 rng から独立種付け**
  （テスト/自己対戦の決定論契約「同入力→同手」を維持）。
- **合格ゲート**: 既存全テスト pass（CLAUDE.md 品質ゲート）・`fair_hard` が CLI 指名可・決定論維持。

### Phase 0 — 測定（規模:中）
- 凍結 `fair_hard` スナップショット作成。CRN＋antithetic seeds でアリーナの ±35 Elo ノイズ帯を破る。
  `realize_trace`/`regret_trace` をフェア経路で取得可能にする。
- **統計目標**: +30 Elo（≒ペア勝率 54.3%）を 95%信頼・80%検出力で検出（CRN＋antithetic で独立 ~1000 局
  相当を **300〜500 ペア局**へ削減）。**合格ゲート＝同一構成 2 回で Elo 信頼区間の半幅 < 15**。

### Phase 1 — 切り分け（規模:小・チート除去後）
- `fair_hard` 上で horizon(2→4→6)・beam を**無制限レイテンシ**で振り、「深く読めば cheat-hard 比勝率が
  単調増/飽和するか」を測る（CRN で少局数）。
- **合格ゲート**: 限界が探索か（横の情報欠損か）を確定。情報欠損支配 → Phase 2(PIMC)。探索支配 →
  Phase 4(TT) を前倒し。

### Phase 2 — PIMC（本命・規模:小〜中・前倒し）
- `decide` を K 世界の決定化 → αβ(`_search`) → root 手投票（毎手再計算）に。`decide_cached` のプラン
  replay 路線は本番最適化として後段に温存。
- **2a（小）**: informed determinization（`OpponentProfile` の推定デッキリスト母集団から引く・消費済み
  除外）＋ **K=2・予算 1/K 按分**（1 手 ~385ms 維持＝レイテンシほぼタダ）。
- **2b（中）**: consistency フィルタ（場/トラッシュ/ライフの公開カードをサンプルから除外）＋ K=4 を
  multiprocessing 並列（実験/arena 用・決定論は捨てる。出荷は決定論不要だがテストは固定反復＋種固定）。
- **合格ゲート**: ①凍結 fair_hard 比 Elo +Δ（主指標）②cheat-hard 比勝率の回復 ③realize big_gaps 非増加
  ④決定化クローン上で make/unmake 等価ゲート pass（`tests/test_cpu_make_unmake.py` 系を決定化 clone でも）。

### Phase 3 — 評価改善（決定化で消えない直交誤差のみ・段階）
- **3a（小）**: 線形モデルのまま **ラベルを"決定化平均勝率"に貼り替えて再学習**。val acc が跳ねれば
  「線形不発の主因＝相手起因ノイズの交絡」が確定し、これで足りる可能性。
- **3b（大）**: 3a でも頭打ちなら非線形（GBDT/小 MLP・pure-Python/stdlib 縛り）へ。αβ 葉にも winprob
  blend を配線（現状 MCTS 葉のみ）。対象は option value・時間地平線の残差。
- **合格ゲート**: 凍結 fair_hard(+PIMC) 比で Elo 非悪化かつ +Δ。

### Phase 4 — レイテンシ最適化（強さ確定後・規模:大）
- TT（局面ハッシュ＝探索内の着手順転置のみ健全。decide 跨ぎは uuid 再生成で不健全）＋反復深化＋壁時計
  デッドライン。K 世界をデッドライン共有で動的配分。
- **合格ゲート**: 1 秒/手で Phase 2/3 の Elo を維持（±10）。

### Phase 5 — 本格 ISMCTS（条件付き・規模:大）
- PIMC が strategy fusion で頭打ち（root 投票でも改善飽和）した場合のみ、`cpu_mcts` のマクロ木を
  information-set 対応へ。それ以外は着手しない。

## 5. 依存関係・マイルストーン

```
Phase-1 前提整備 ─→ Phase0 測定 ─→ Phase1 切り分け ─┬→ Phase2 PIMC(本命) ─→ Phase3 評価残差 ─→ Phase4 1秒化 ─→ Phase5 ISMCTS(条件付)
                                                    └→ (探索支配なら) Phase4 TT を前倒し
```

- **-1 → 0 → 1 は直列必須**（測定基盤が無いと以降全部が盲目）。
- **2(PIMC) は 3(評価) より前**（決定化が相手起因の評価誤差を正則化し、3 の対象が縮む）。
- **4 は強さ確定後**（無制限で強さを固めてから 1 秒へ）。
- **M-1** 前提整備＝フェア配線＆多ポリシー arena ／ **M0** ノイズ帯を破る計測器 ／ **M1** 限界要因確定 ／
  **M2** PIMC で fair_hard 比 Elo 有意向上 ／ **M3** 評価残差回収 ／ **M4** 1 秒化 ／ **M5** 条件付き ISMCTS。

## 6. 主なリスク

- 即フェア切替の一時的弱化 → Phase 2(PIMC) で回復・超過する前提。回復まではユーザ承認済みの許容範囲。
- アリーナの分散（±35 Elo）→ Phase 0 で先に潰す（順序厳守）。CRN は「デッキ配り seed と方策 rng の分離」
  が前提（現状 global random 一本で混線）。
- horizon amplification → Phase 1 で探索/評価を切り分けてから投資。
- 決定化と journal の整合 → 決定化クローン上で make/unmake 等価ゲートを Phase 2 合格条件に含める。
- PIMC × 無制限レイテンシの計算量（K 世界 × αβ × 数百局）→ K=2 予算按分で 1 手 ~385ms 維持。重い実験は
  multiprocessing 並列（決定論は捨て、テストは固定反復＋種固定で担保）。
- 学習の交絡（`cpu_features` が相手手札を見ない）→ Phase 3a で決定化平均勝率ラベルに貼り替えて切り分け。
