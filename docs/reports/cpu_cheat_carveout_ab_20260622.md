# 計測報告: hard カンニング切り分け A/B（Phase 1）

- 日付: 2026-06-22
- 種別: 報告（点・スナップショット／改変しない）
- 対象: `cpu_ai.decide`/`decide_guarded` の情報方針（`see_opp_hand`／`opp_public_only`）
- 関連: 計画 `reports/cpu_weird_move_remediation_plan_20260622.md` §5（Phase 1）、Phase 0 物差し
  `tests/cpu_weird_move_audit.py`、仕様 `SPEC.md §2.5.2`（難易度＝情報方針）

## 目的

Phase 0 で「CPU の変な手は 100% 評価関数起因（eval_gap）・探索バグはゼロ（search_dispreferred=0）」と
確定した。Phase 1 は残る前提＝**hard の相手手札カンニング（`see_opp_hand=True`）をフェア化すると、変な手／
強さがどう変わるか**だけを切り分ける測定（探索バグ探しは Phase 0 で否定済みのため対象外）。出す結論は
2 つ: (a) 配信 hard をフェア化すべきか、(b) Phase 2 の学習データをフェア対戦で生成すべきか。

## 「フェア hard」の定義と実装方法

**定義**: hard の探索深さ（α-β＋ビーム＋ horizon=4）は維持したまま、**情報方針だけを公開情報のみへ
落とす**。具体的には既存の normal 情報方針をそのまま再利用する:

- `see_opp_hand=False` … 葉の評価で相手手札の中身（カウンター値）を読まない（枚数のみ）。
- `opp_public_only=True` … 相手 min ノードで相手の隠れ手札に依存する手（手札からの登場／カウンター）を
  使わない保守モデルで応答する。

**なぜこの定義か（既存資産の何を使ったか）**: `_search`/`evaluate` には既に `see_opp_hand`/
`opp_public_only` の2フラグが全段に配線され、normal がこの公開情報方針で動いている。隠れ手札の
determinize（複数サンプルの平均化）は新規インフラが要りコスト大なので採らず、**最も安価で忠実な
フェア化＝既存の公開情報モデルの再利用**を選んだ。これは「相手手札を一切覗かず、相手は隠れ札に依存
する手を打たない前提で保守的に読む」という、配信フェア AI に最も近い情報状態を表す。

**実装（観測専用・既定 OFF＝現状不変）**: `decide`/`decide_guarded` に `info_policy: str = "hard"` 引数を
追加。既定 `"hard"` は従来どおり `see_opp_hand=True, opp_public_only=False`（**完全同値**）。`"fair"` の
ときだけ `see_opp_hand=False, opp_public_only=True` へ切替える。評価重み・補償パッチ・探索の既定挙動は
一切変えていない。実験ハーネス（`cpu_weird_move_audit.py --fair`／`cpu_arena.py --challenger-info/
--baseline-info`）からのみ `"fair"` を渡す。監査の `classify_decision` 内 `see_opp_hand` も測定方策に一致
させる（cheat 測定時=True／fair 測定時=False）。

> 既定 OFF の根拠: 本タスクは測定のみ。配信 hard の挙動を変えると決定論ベースライン・既存テストに波及し、
> 「観測専用」の原則を破る。フェア化の是非は本報告の結論を見てマネージャーが判断する。

## A/B 数値表

### 変な手（`cpu_weird_move_audit.py`・同 seed=0・各 30 局・hard 自己対戦）

| 指標（件数/100局） | cheat（info=hard, see_opp_hand=True） | fair（info=fair, see_opp_hand=False） | 差分 |
|---|---|---|---|
| ①差≤0で行動 | 1433.3 | 1230.0 | **−14.2%** |
| ②自殺攻撃 | 50.0 | 60.0 | **+20.0%**（悪化） |
| ③無駄ドン | 1080.0 | 940.0 | **−13.0%** |
| ④届かないカウンター | 0.0 | 0.0 | ±0 |
| 第2軸 search_dispreferred（探索バグ） | 0.0 | 0.0 | ±0 |
| 第2軸 eval_gap（評価ギャップ） | 1433.3 | 1230.0 | −14.2% |
| decisions_scored（母数） | 1625 | 1590 | — |
| 局完走 / 失敗 | 30/30・失敗0 | 30/30・失敗0 | — |

> cheat 列は Phase 0 ベースライン（①1433 ②50 ③1080 ④0・search_dispreferred=0）と**完全一致**＝既定 OFF が
> 現状同値であることの追加確認になっている。

### 強さ（`cpu_arena.py` 直接対決・席交互・seed0=0）

| 対戦 | 局数 | hard-cheat 勝率（cheat の勝ち数/局数） | Elo（cheat − fair） |
|---|---|---|---|
| hard-cheat（挑戦者） vs hard-fair（ベースライン） | 30（seed0=0） | 0.567（17.0/30） | **+47** |

（席交互＝偶数 seed で cheat=p1／奇数 seed で cheat=p2・先手有利相殺。引き分けは 0.5 勝計上・本走は
引き分け0。）cheat は生の強さで **+47 Elo** 優位だが、アリーナの既知ノイズ帯（±35 Elo・
`cpu_plan_ideal_line_ab_20260616.md`）と同オーダー＝**統計的に小さい優位**。カンニングを外しても強さの
低下は限定的。

## 防御の歪みへの効果の所見

計画 §1 の「防御の歪み」に最も近い監査カテゴリは **②自殺攻撃** と **④届かないカウンター**。

- **④届かないカウンター = 両モードとも 0 件**。カンニング有無に関係なく、CPU は「積んだカウンターが
  最終的に届かず無駄になる」歪みを（この 30 局・seed0 帯では）起こしていない。フェア化の効果を測る
  対象がそもそも存在しない。
- **②自殺攻撃 = フェア化で減らない（むしろ +20%）**。代表局面は cheat/fair とも `OP01-064`/`OP01-063`
  等の小型でレスト中の相手キャラ（`EB01-021`）へ突っ込み、settle regret が負（−500〜−1100）になる
  キャラ間トレードの誤り。これは「相手手札を覗けたから打った不可解な正解」ではなく、**戦闘トレードの
  価値を評価関数が過大評価している eval_gap**（search_dispreferred=0＝探索は TURN_END 超と判断して打って
  いる）。fair でわずかに増えるのは、相手手札の中身が見えない分カウンター緩衝の評価が変わり、トレード
  期待値の符号が一部局面で反転するため（評価の根の問題はそのまま）。
- ①③が −13〜14% 減るのは、相手手札のカウンター値（`W_COUNTER`）が評価から消えて盤面差の絶対値が
  縮み、`_ACT_MARGIN`(300) の畳み判定がわずかに発火しやすくなる**副次効果**であって、歪みの根（eval_gap）
  が直ったわけではない。第2軸は依然 100% eval_gap・search_dispreferred=0 のまま。

**結論（切り分け）**: カンニングは「変な手」の主因ではない。フェア化は①③をテンポマージン経由で軽く
押し下げるが、**防御の歪みの核（②自殺攻撃・eval_gap）は是正しない／一部悪化**する。Phase 0 の「100%
評価起因」所見はフェア化後も成立する＝**残課題は評価のキャリブレーション（Phase 2）に一本化される**。

## 意思決定

### (a) 配信 hard をフェア化すべきか

**推奨: フェア化してよい（むしろ推奨寄り）。ただし「変な手対策」としてではなく「公平性／納得感」目的で。**

- 根拠1（変な手）: フェア化で変な手は減りこそすれ増えない（①−14% ③−13%、②は微増だが核は eval_gap で
  カンニングと無関係）。少なくとも**変な手を悪化させずに**透視を外せる。
- 根拠2（強さ）: 直接対決は cheat +47 Elo（17/30）＝ノイズ帯（±35 Elo）と同オーダーの**小さい優位**。
  カンニングを外して失う強さは限定的で、納得感の改善に見合う。
- 根拠3（納得感）: 相手手札を覗く AI は人間に「読まれている」不快感・不可解さを与える。Phase 0 で
  「変な手は評価起因」と確定した今、透視を残す積極的理由は薄い。

判断の境界: 配信挙動の変更（既定 ON 化）は本タスクのスコープ外（観測専用に留める指示）。フラグは入れた
ので、マネージャーが上の Elo トレードオフを見て切替えるだけで済む。

### (b) Phase 2 の学習データはフェア対戦で生成すべきか

**推奨: フェアで生成すべき（強く推奨）。**

- 根拠: 計画 §6 も「特徴は公開情報ベースで算出・hard の相手手札透視を学習評価に混ぜない」と既定して
  いる。カンニング方策が打つ手は**フェア AI には観測不能な情報に依存し得る**ため、その軌跡で学習した
  価値関数は配信フェア AI の評価を歪める（分布シフト＋観測不能特徴への依存）。
- 本実験で、フェア方策は決定論・完走性・インバリアントを cheat と同等に保ち（30/30・失敗0）、変な手も
  悪化しないことを確認した＝**フェアでのデータ生成は安全に運用できる**。`info_policy="fair"` をデータ生成
  ランナーに渡せばよい（本 PR で配線済みの観測専用フラグを流用可能）。

## 限界

- サンプルは seed0 帯の 30 局（変な手）／30 局（強さ・直接対決）。アリーナは ±35 Elo 程度のノイズ帯が既知
  （`cpu_plan_ideal_line_ab_20260616.md`）＝強さの差分は方向性の目安。確定チューニングには数百局規模が要る。
- 自己対戦経路のため相手プロファイル（テンプレ）は None＝マッチアップ補正は不活性。配信の対人/対テンプレ
  分布とは異なる。
- 「フェア hard」は公開情報モデルの再利用であって determinize ではない＝隠れ札の確率的読みは行わない
  （保守モデル）。より忠実なフェア強度を測るなら determinize 版が将来課題（本タスクでは scope 外）。
- ④届かないカウンターが両モード 0 のため、防御の歪みのうちカウンター系への効果は本データでは判定不能。

## 付録: 実験条件・再現コマンド・品質ゲート

```
# 変な手 A/B（各 30 局・seed0）
OPCG_LOG_SILENT=1 python tests/cpu_weird_move_audit.py --games 30 --seed 0 --json /tmp/audit_cheat.json
OPCG_LOG_SILENT=1 python tests/cpu_weird_move_audit.py --games 30 --seed 0 --fair --json /tmp/audit_fair.json
# 強さ直接対決（30 局・席交互）
OPCG_LOG_SILENT=1 python tests/cpu_arena.py arena --challenger hard --baseline hard \
    --challenger-info hard --baseline-info fair --games 30 --seed 0
```

品質ゲート（フェア化フラグ既定 OFF で全テスト不変＋回帰テスト追加）:

```
OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_puzzles.py tests/test_cpu_replay.py tests/test_cpu_ai.py -q -s -p no:cacheprovider
# -> 63 passed（従来 59 + 新規 4）。
```

追加した回帰テスト（`tests/test_cpu_ai.py`・観測専用フラグの非侵襲性を機械照合）:
- `test_info_policy_default_is_hard_cheat`: 既定は see_opp_hand=True / opp_public_only=False。
- `test_info_policy_fair_switches_to_public_only`: fair で see_opp_hand=False / opp_public_only=True。
- `test_info_policy_default_decision_unchanged`: 既定省略と明示 "hard" が同一手（現状不変）。
- `test_info_policy_guarded_threads_flag`: `decide_guarded` も info_policy を素通しする。
