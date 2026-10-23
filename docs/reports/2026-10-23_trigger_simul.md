# 同時に誘発した自動効果は手番側から・同じプレイヤーの効果は本人が順を選ぶ（WP `rs-trigger-simul`・2026-10-23）

指示書: `docs/search_efficiency_plan.md` §7.27（本線 `claude/cpu-spec-improvements-yw91jd` @ 4f846bb・台帳 2+350）。
前段: §7.20・§7.23、`docs/reports/2026-10-22_trigger_order.md`・`2026-10-22_trigger_order2.md`（両報告の留保「待ち行列の中の順は積んだ順」）。
ブランチ `claude/rs-trigger-simul`。

## 0. 結論（先に）

- **直した**。同時に誘発した自動効果の束を **ライフの【トリガー】 → 手番のプレイヤーの効果 → 非手番のプレイヤーの効果** の順に並べ、
  同じプレイヤーの効果が（別の札で）2 つ以上ある組では、その組の番が来たときに本人へ**順を選ぶ対話**（ORDER_TRIGGERS）を立てる。
  既定解決（監査・判定器・ドレイン）は積んだ順＝先頭。CPU の合法手には候補ごとに 1 手として出る（MCTS が順を選ぶ）。
  待ち行列（`pending_triggers`）に加えて、**アタック時の待ち行列（`battle_triggers`）にも同じ規則**を入れた（§2.3）。
- Rust テスト 2 本を足した（`effects/tests_trigger_order.rs`・どちらも**修正前のエンジンで落ちる**ことを確かめた）。`cargo test` 606 本 green。
- `cargo clippy -D warnings` の本線由来の指摘を挙動を変えずに直した＝**0 件**（指示の 4 件＋本線で増えていた 1 件・§4）。
- **golden**: 監査 3,387 件は **0 件**、再生 200 局のうち **4 局**が変わった（予告は監査 0〜5・再生 5〜20）。4 局とも**アタックされた側が
  【相手のアタック時】を 2 つ持つ局面で順の選択が立った**＝正しくなった 4・疑わしい 0（§5）。
- `make card-audit`: IMPL_CHANGED **0**（既定・`--all` とも）・指摘の集計は前回と同一。`make test` green（Rust 606・pytest 855 passed）。
- `make audit-cross` **void 0**（§7）。手元アリーナ 40 ペア（seed 1685000〜1685039・出荷の設定）: 修正前・修正後とも **void 0**・
  **40 ペア中 36 ペアが勝敗もターン数も同一**・候補席の勝ちは 41 で同数（§8）。

## 1. 欠陥

同時に誘発した自動効果（同じ出来事で条件を満たした複数の効果）は、待ち行列に**積んだ順**に解決していた。総合ルールでは、
**手番のプレイヤーの効果を先に、非手番のプレイヤーの効果を後に**解決し、同じプレイヤーの効果が複数あればその**プレイヤーが順を選ぶ**。

積む順は誘発を拾う関数の都合で決まっていた:
- 効果の中の KO（`triggers::resolve_on_ko`）は **KO された札自身の【KO時】を先に**、他の札の「KOされた時」リスナーを後に積む。
  手番の p1 が相手（p2）のキャラを効果で KO すると、p2 の【KO時】が p1 の「相手のキャラがKOされた時」より先に解決されていた。
- アタック時（`triggers::enqueue_battle_triggers`）はアタックした札の【アタック時】 → 第三者の「アタックした時／された時」（**p1 → p2 の席順**）
  → アタックされた側の【相手のアタック時】の順。手番が p2 のとき、p1 の第三者の効果が手番の p2 の効果より先に入りうる。
- 同じプレイヤーの 2 つの効果（例: リーダーとキャラの【相手のアタック時】）は、選ばせずに積んだ順で解決していた。

## 2. 直したもの（`rust/opcg_engine`）

| 変更 | 場所 |
|---|---|
| 待ち行列の 1 件に「束の中の状態」（`Fresh`＝未整理／`Choose`＝本人が順を選ぶ組／`Fixed`＝順が決まった）を持たせる | `model.rs::PendingTrigger.bundle`・`TriggerBundle` |
| 消化の先頭で束を並べ替え、同じプレイヤーの組に別の札が 2 枚以上あれば順を選ぶ対話を立てる | `effects/triggers.rs::order_front_bundle` |
| 順を選ぶ対話の型（フロントへは `SEARCH_AND_SELECT`＝既存の 1 枚選択の UI・`constraints {min:1,max:1}`・`can_skip=false`・`intent "ORDER"`）| `model.rs::InteractionKind::OrderTriggers`・`SelectionIntent::Order`・`Continuation.order_queue` |
| 対話の解決: 選んだ札の（組の中で最初の）効果を `Fixed` で先頭へ移し、消化を再開（CONFIRM_TRIGGER と同じ後処理を共有） | `effects/triggers.rs::resolve_order_triggers`・`after_trigger_queue_interaction`・`effects/interact.rs::resolve_interaction` |
| 既定解決は先頭（積んだ順） | `effects/interact.rs::choose_selection`（`Order` は序列を使わない）|
| 消化点: 待ち行列は `advance_pending_triggers_inner` の先頭、アタック時の待ち行列は `advance_battle_triggers` の先頭で `order_front_bundle` を呼ぶ | `effects/triggers.rs`・`rules/battle.rs` |

状態の追加は待ち行列の各件の 1 欄だけ（待ち行列は journal 済み＝undo も従来どおり）。API 契約（`shared_constants.json`・`contract/`）は変わらない
（新しい対話はフロントへ既存の `SEARCH_AND_SELECT` で出る）。

### 2.1 束の切れ目（「同じ出来事」の定義）

**束＝前の消化が始まってから今の消化が始まるまでに積まれた誘発**（先頭から続く `Fresh` の並び）。待ち行列は効果の解決中・中断した効果の残りの
退避中は消化されない（§7.20）ので、実際には「一番外の効果・行動・戦闘の段の 1 つの区切りの中で条件を満たした誘発」が 1 つの束になる。
- 束の中は **ライフの【トリガー】 → 手番の組 → 非手番の組**（各組の中は積んだ順＝既定の答え）。ライフの【トリガー】は自動効果ではなく
  ダメージの処理の一部なので先頭に置き、順の選択に入れない。
- 束を解決している間に新しく誘発した効果は束の**後ろ**に積まれ、次の束になる（今の束の残りより先には解決しない＝従来の FIFO を保つ）。
- アタック時の待ち行列は `declare_attack` が 1 度に積む＝その全体が 1 つの束。

### 2.2 順の選択の形

組の番が来たとき（先頭が `Choose`）に、先頭から続く同じプレイヤーの `Choose` の並びに**別の札が 2 枚以上**あれば、その札を候補に
「先に解決する効果のカード」を 1 枚選ばせる。選んだ効果を解決した後、残りが 2 つ以上ならまた選ぶ（1 件ずつ選べば全ての順を選べる・
全順列を一度に出すより候補の数が小さい＝探索の枝が増えにくい）。非手番の組の選択は、手番の組を解決し終えてから立つ。
- **同じ札の 2 つの能力**（1 枚が 2 つの誘発を同時に満たす）は札で選ぶ形なので区別できず、積んだ順（留保）。
- 任意の効果（CONFIRM_TRIGGER）の確認は、順が決まってから先頭に来たときに立つ（従来どおり）。

### 2.3 アタック時の待ち行列にも入れた理由

指示は `pending_triggers` だが、アタック時の誘発（【アタック時】・【相手のアタック時】・「アタックした時／された時」）も同じ「同時に誘発した
自動効果」で、席順（p1 → p2）で積む欠陥も同じ形だった。同じ関数（`order_front_bundle`）を通すだけで済むので入れた（notes に記録）。
再生 golden の変化 4 局はすべてこちらの経路（§5）。

### 2.4 ルールの根拠

総合ルールの自動効果の解決（「複数の自動効果が同時に発動した場合、ターンプレイヤーの効果から解決し、同じプレイヤーの効果が複数あるときは
そのプレイヤーが順番を決める」の趣旨）。**リポジトリに総合ルールの本文は無く、条文番号と文言は記憶による**（留保）。

## 3. テスト

`rust/opcg_engine/src/effects/tests_trigger_order.rs`（実カード・`docs/TEST_SPEC.md` §2 の行に追記）:
- `the_turn_player_trigger_resolves_first`: p1（手番）の場にドーマとコビー（EB04-044・【自分のターン中】【ターン1回】相手のキャラがKOされた時、
  カード1枚を引く）、手札に衝撃貝（OP15-115）、p2 の場にマルコ（OP10-055・【KO時】相手のコスト4以下のキャラを手札に戻す）。衝撃貝でマルコを KO
  すると、マルコの【KO時】（非手番）とコビー（手番）が同じ KO で誘発する。**マルコの対象選択（p2 の対話）が立つ時点で p1 の山札が 5**
  （コビーが先に引いた）。各プレイヤー 1 つずつなので順の選択は立たない。**修正前は山札 6 で落ちる**（積んだ順＝自身の【KO時】が先）。
- `the_same_player_chooses_the_order`: 同じ盤面でコビーを 2 枚にする。衝撃貝の KO の後、**手番の p1 に ORDER_TRIGGERS が立つ**（p2 の対話より先）。
  候補は積んだ順の 2 枚・既定解決は先頭・`search::adapter::selection_moves` は 2 手。2 枚目を選ぶと 2 枚目のドローが先に記録され、
  その後に 1 枚目、それから p2 のマルコの対象選択（山札 4）。**修正前は ORDER_TRIGGERS が立たず（SELECT_TARGET）落ちる**。

修正前のエンジンでの確認は、本線 4f846bb の worktree に新しいテストファイルを置き（存在しない `OrderTriggers` を一度も立たない別の種別に
置き換えてコンパイル）回した: 既存 4 本 ok・新 2 本 FAILED（上の理由）。

`cargo test --no-default-features`: **606 passed**（既存 604 は無修正で green・新 2）。

## 4. clippy（挙動を変えない修正）

このコンテナの clippy 1.97 で `cargo clippy --no-default-features --all-targets -- -D warnings` が出していた指摘を全て直した（修正後 0 件）。

| 指摘 | 場所 | 直し方 |
|---|---|---|
| `collapsible_match` | `ops.rs::move_card`（ステージ枠から外す） | 内側の `if` を match のガードへ（ガードが偽のときは何もしない＝同じ） |
| `type_complexity` | `search/mcts.rs::BLIND_TRACE`（テスト用） | 型の別名 `BlindTrace` を置く |
| `type_complexity` | `search/decide.rs::blind_run`（テスト用） | 戻り値の型を `mcts::BlindTrace` に（**関数の署名 1 行だけ・中身は不変**＝並行の WP `rs-commit-sig` との衝突を小さく保つ） |
| `incompatible_msrv`（本線で増えていた 5 件目） | `search/lethal.rs`（`Option::is_none_or` は 1.82〜・crate の MSRV は 1.75） | 同じ意味の `match` に |

指示の「4 件」は前段の報告の時点の数で、その後の本線（WP `rs-lethal-root` 系）で `is_none_or` の 1 件が増えていた。

## 5. golden の差分のレビュー

作り直し: `make golden-audit`・`make golden-replay`。比較は作り直し前の golden（本線 4f846bb）。

### 5.1 監査 golden（3,387 件中 **0 件**）

`audit.json` はバイト単位で同一。監査の汎用盤面（`audit.rs::build_test_state`）は相手側がバニラで、検査対象の札の能力だけが誘発する＝
同じ束に 2 人の効果、または同じプレイヤーの別の札の効果が並ぶことが無い（予告 0〜5 の下限）。

### 5.2 再生 golden（200 局中 **4 局**・すべて random 帯・a1 帯 50 局は 0）

最初に食い違う段を探し、旧記録をその段まで旧新の両エンジンで再生して要求と盤面を比べた。4 局とも**最初に違う段の盤面は同じ**で、
違いは「アタックされた側に順の選択（ORDER_TRIGGERS）が立つ」こと。random 方策は合法手の数が変わると以降の引きが変わる＝局の残りは
別の局になる（勝者が変わった 1 局〔random_5000108〕もこのため）。

| 局 | 段 | 手番（アタック） | 順を選んだ側の 2 つの効果 | 判定 |
|---|---|---|---|---|
| random_5000039 | 16 | p2 のエース（【アタック時】を先に解決） | p1: OP11-062 カタクリ（リーダー・【相手のアタック時】ドン!!-1…）／EB04-034 プリン（【相手のアタック時】手札1枚を捨てることができる…） | **正しくなった**（手番の【アタック時】が先・同じプレイヤーの 2 つは本人が選ぶ） |
| random_5000085 | 105 | p1 のキッド | p2: OP17-058 カイドウ（リーダー・【相手のアタック時】ドン!!-1…）／EB01-037 Mr.9（【相手のアタック時】ドン!!-1…） | **正しくなった** |
| random_5000088 | 91 | p2 のロシナンテ | p1: OP17-001 ニューゲート（リーダー・【相手のアタック時】手札1枚を捨てることができる…）／EB01-002 イゾウ（【相手のアタック時】…） | **正しくなった** |
| random_5000108 | 25 | p2 のロジャー | p1: OP17-058 カイドウ（リーダー）／EB01-034 ミス・ウェンズデー（【相手のアタック時】ドン!!-1…） | **正しくなった** |

**疑わしいもの: 0**。待ち行列（`pending_triggers`）側で手番側が先になった局（効果の KO で両者の誘発が同じ束に入る形）は 200 局に現れなかった
（§3 のテストが見る）。予告（再生 5〜20）より少ないのは、同じ束に 2 人の効果、または同じプレイヤーの別の札の効果が並ぶ局面がまれなため。

## 6. カード効果監査（`make card-audit`）

- 既定（台帳）: 対象 285 枚（UNREVIEWED のバニラのみ）・IMPL_CHANGED 0・TEXT_CHANGED 0。
- 全走査（`--all`）: QUALIFIER_GAP 133・ACTION_NOT_FIRED 5・UNCOVERED_TEXT 2・TAG_TRIGGER_MISMATCH 1・DEFERRED 91・FIRED_EMPTY 28
  ＝前回（§7.23 の報告 §5）と同一。新たな指摘 0・指紋の変化 0（監査の盤面では本 WP の経路を通らない・§5.1）。台帳は変更なし。

## 7. 交差監査

`make audit-cross`（120 ペア・240 局・32 回・synth・修正後の wheel）: **void 0**（AUDIT_CROSS_RESULT の行は RESULT.json）。

## 8. 手元アリーナ（40 ペア・seed 1685000〜1685039）

§7.20・§7.23 と同じ読み方: 1 局の中で旧新 2 つのルールを混ぜることはできないので、**同じ 40 ペア**（両席 r13d・出荷の設定＝1,280 回・枝予算なし・
世界 2 本・詰みの証明 root・公平・synth_roles・leaders random・seed base 1685000）を修正前の wheel（本線 4f846bb）と修正後の wheel でそれぞれ打ち、
ペアごとに突き合わせた（候補席＝`--candidate nrel_r13d.npz`・基準席＝出荷既定〔同じ r13d〕・両席とも `--worlds 2 --budget 0 --lethal-proof root`）。

| wheel | ペア | 局 | void | 候補席の勝ち | 勝率（95% CI） | 平均ターン |
|---|---|---|---|---|---|---|
| 修正後 | 40 | 80 | **0** | 41 | 0.513（0.389–0.637） | 12.66 |
| 修正前 | 40 | 80 | **0** | 41 | 0.513（0.394–0.631） | 12.54 |

- **void 0**（修正後・修正前とも）。
- **40 ペア中 36 ペアは勝敗もターン数も同一**（予告「35 以上」と整合）。違った 4 ペア: 1685021（OP03-001 vs OP17-001・1 局目のターン 14 → 16）、
  1685025（OP17-001 vs OP12-041・1 局目のターン 11 → 13）、1685030（OP08-002 vs OP10-022・1 局目の勝敗が反転）、1685038（OP02-093 vs EB01-021・
  1 局目の勝敗が反転）。違った 4 ペアのうち 2 ペアは OP17-001 ニューゲート（リーダーの【相手のアタック時】＝§5.2 と同じ形）。
  候補席の勝ちは 41 で同数＝**強さは動かない**（ルールの正しさの修正）。
- 勝率は同じネット同士の席の差と 40 ペアの引きの偏り（旧新でほぼ同じ局＝独立な 2 標本ではない）。強さの判定には使わない。
- 台帳: `n1_results/trigger_simul/arena_after_b1685000.jsonl`（修正後）・`arena_before_b1685000.jsonl`（修正前）。

## 9. 留保

- 総合ルールの条文番号と文言は記憶による（リポジトリに本文が無い）。
- **戦闘の結果の KO の自身の【KO時】**はバトルの文脈を見るためその場で解決する（§7.23 の報告 §2.2）＝待ち行列のリスナー（他の札の
  「KOされた時」）とは束にならない。手番の p1 のアタックで p2 のキャラが KO されると、p2 の【KO時】が p1 の「相手のキャラがKOされた時」より
  先に解決される（効果の KO では §3 のとおり手番側が先）。直すなら、自身の【KO時】をバトルの文脈ごと束に入れる形が要る（別 WP）。
- 束を解決している間に誘発した効果は今の束の後ろ（次の束）に積む＝今の束の残りより後に解決する。ルールの読み方によっては「今解決した
  効果の直後」もありうるが、従来の FIFO を保った。
- 同じ札の 2 つの能力は札で選ぶ形なので区別できず、積んだ順で解決する。
- 順の選択の候補の上限は探索の 1 枚選択の上限（`HARD_SELECT_CAP`）に従う（同じプレイヤーの効果がそれを超えて並ぶ局面は実際には無い）。

## 10. 再現

```bash
make rust-develop
cd rust/opcg_engine && cargo test --no-default-features tests_trigger_order \
  && cargo clippy --no-default-features --all-targets -- -D warnings && cd -
make golden-audit && make golden-replay
make card-audit && make card-audit CARD_AUDIT_ARGS=--all
make test && make audit-cross
# 手元アリーナ（修正後の wheel。修正前は PYTHONPATH に本線 4f846bb の wheel を入れた環境で同じコマンド）
OPCG_LOG_SILENT=1 python -m opcg_sim.loop.arena_shard --candidate opcg_sim/data/learned/nrel_r13d.npz \
  --pairs 40 --bands 1 --max-pairs 40 --workers 2 --sims 1280 --budget 0 --cand-budget 0 \
  --worlds 2 --cand-worlds 2 --lethal-proof root --cand-lethal-proof root \
  --leaders random --decks synth_roles --seed-base 1685000 --out arena_after_b1685000.jsonl
```
