# 詰みの証明の守りの強い配り方：カウンターイベントは相手が払える枚数までだけ数える（WP `lethal-deal-cost`・2026-10-23）

> 仕様: `docs/search_efficiency_plan.md` §7.33（本線 `claude/cpu-spec-improvements-yw91jd` @ 81d02bb・台帳 2+357）。
> ユーザ決定（2026-10-23「a でお願いします」）。根拠は `docs/reports/2026-10-23_lethal_own_hidden.md`。
> 正本は `docs/SPEC.md` §2.5.10・コード `rust/opcg_engine/src/search/lethal.rs::defensive_deal`。

## 0. 結論

- **既定を変えた**: 詰みの証明の「守りの強い配り方」2 通りで、カウンターイベントは**相手のアクティブのドン!!で払える枚数まで**だけ
  守りに数える（払えない分は守りの値 0＝手札の枠は他の札で埋まる）。形は WP `lethal-own-hidden` の計測の切替 `cost`
  （コミット `afd94ef`・環境変数 `OPCG_LETHAL_OWN=cost`）と同じ。環境変数の切替は残していない。
- **測定用の旗** `DecideOptions.lethal_deal_legacy`（opts の鍵 `lethal_deal_legacy`・既定 false・true で旧い形）を足し、
  `RsGame.decide(lethal_deal_legacy=…)`（`opcg_sim/api/engine_rs.py`）・`arena --cand-lethal-deal-legacy`（候補席）・
  `arena_shard --lethal-deal-legacy`（基準席）・`Game.lethal_json` の `lethal_deal_legacy`・
  `lethal_speed_check.py judge --lethal-deal-legacy` に通した（§7.21・§7.26 と同じ配線）。`root_reply` の木の中の証明
  （`ReplyProof`）にも同じ旗が効く。
- **不変量は全部成立**（§2）: 12,710 点で証明 1,064 → **860**（切替 `cost` の数字と全点一致）・旗 true で **1,064**（本線の判定と全点一致）。
  `lethal_proof=off` は配り方を作らない＝1 bit も変わらない（golden 2 本を含む `make test` green）。
- 本番アリーナ（候補＝既定・基準＝`--lethal-deal-legacy`）は §7.33 の 5 のとおりコーディネータが起こす（本 WP では回していない）。

## 1. 変更

`defensive_deal(state, masters, me, mode, legacy, rng)`: 山を混ぜて守りの値で安定に並べる（旧い形と同じ）。`legacy` が false なら続けて、
並びの順にカウンターイベントのコストを相手のアクティブのドン!!の枚数（予算）と比べ、収まるものは値をそのまま残して予算を減らし、
収まらないものは値 0・イベントでない扱いの鍵に置き換え、もう一度安定に並べ直す。その上から手札の枠へ入れる（残りは今までどおり
混ぜて山札・ライフの枠へ）。乱数の引き方は旧い形と同じ（並べ直しは乱数を引かない）＝配り方の公平な 6 通りは不変。

`make_deals`・`prove`・`root_proof` に `legacy` を通し、`decide` の ⓪（根）と `ReplyProof`（`root_reply`）が `opts.lethal_deal_legacy` を渡す。

テスト (g) `defensive_deal_counts_counter_events_only_within_payable_don`（`tests_lethal.rs`・`df5d251` を既定の形へ直したもの）:
相手の知らない山＝手札の枠 2＋山札 6（コスト 1・+3000 のカウンターイベント 2・カウンター 2000 のキャラ 2・ふつうのキャラ 2）。
アクティブのドン!! 0／1／2 枚で、手札の枠のイベントが 0／1／2 枚（残りはカウンター 2000）。旗 true では 0 枚でも 2 枚（旧い形）。2 通りの配り方とも。

## 2. 不変量（`lethal_speed_check.py judge`・記録 `n1_results/lethal_speed/record_3012000_300.jsonl.gz`・300 局）

| 判定 | 点 | 証明 | 突き合わせ | status の差 | 手の差 | 節点の差 |
|---|---|---|---|---|---|---|
| 旗 true（`judge_legacy`） | 12,710 | **1,064** | `lethal_own_hidden/judge_base`（本線） | 0 | 0 | 0 |
| 既定（`judge_cost`） | 12,710 | **860** | `lethal_own_hidden/judge_cost`（切替 `cost`） | 0 | 0 | 0 |

（`n1_results/lethal_deal_cost/identity.json`。点は記録 300 局のうち今の wheel で当て直せる 12,710 点＝報告 `lethal_own_hidden` §4 と同じ。
`judge` の出力は 12,735 行・うち 25 行は当て直しのずれの行。）`line_len`（stats の欄だけ）は付与で始まる証明 449 点で基準と違う＝
基準の `judge_base` は `line_len` の修正〔付与で始まる手順を 1 と数えていた〕の前の wheel で作られたため（449 点とも基準が 1・手が `ATTACH_DON`）。
打つ手・判定には関わらない。

旧い形 → 既定の遷移（`compare_legacy_vs_cost.json`）: win→win 859・win→no 121・win→unknown 84・unknown→win 1・
証明した手が変わった 39・節点 ×1.021。報告 `lethal_own_hidden` §4 の `cost` の行（失う 205＝本当の配り方で勝てない 2＋勝てる 203・
増える 1〔3012249:p2:12:120・空いた予算で打ち切り→証明〕）とそのまま同じ。

## 3. ゲート

- `make test` green（cargo 607 passed・1 ignored〔テスト (g) で +1〕／pytest 857 passed・13 skipped・`n1_results/lethal_deal_cost/make_test.log`）。
  golden 2 本（`lethal_proof=off`）は不変。
- `make audit-cross` void 0（120 ペア・`n1_results/lethal_deal_cost/audit_cross.log`）。ただし監査は `lethal_proof=off` で回る＝
  本変更の経路は通らない。配線の確かめとして `arena_shard --lethal-proof root --lethal-deal-legacy --cand-lethal-proof root`
  （8 ペア・32 回・random × real・seed 3191000〜）を回し、void 0・台帳の `best_opts` に `lethal_deal_legacy: true` が載ることを見た。

## 4. 予告（本番アリーナ・§7.33 の 5）

候補席＝既定・基準席＝`--lethal-deal-legacy`・両席 r13d・出荷の設定（1,280 回・世界 2 本・枝予算なし・`--lethal-proof root` を両席）。
予告は勝率 0.495〜0.505（失う 203 点の多くは探索でもほぼ同じ手を打つ・打つ手が変わる局は全体の 1〜2%）。採用の規約は 95% 下限 ≥ 0.49。
割れたら旗の既定を true に戻す。

## 5. 出力

`n1_results/lethal_deal_cost/`: `judge_{cost,legacy}.jsonl.gz`・`judge_{cost,legacy}.log`・`identity.json`・`compare_legacy_vs_cost.json`・
`make_test.log`・`audit_cross.log`。
