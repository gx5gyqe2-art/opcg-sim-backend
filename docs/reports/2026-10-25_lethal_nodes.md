# 根の詰みの証明の節点の上限（WP `rs-lethal-nodes`・2026-10-25）

仕様: `docs/search_efficiency_plan.md` §7.49（本線 `claude/cpu-spec-improvements-yw91jd` @ 1899954・台帳 2+405）。
根拠: `docs/reports/2026-10-24_weak_points.md` §4.4（上限 20,000 → 80,000 で証明 +13%・偽の証明は増えない）。
ブランチ `claude/rs-lethal-nodes`・成果物 `n1_results/lethal_nodes/`。

## 0. まとめ

- **直したもの**: 根の詰みの証明（出荷の CPU だけが使う `PLAY_LETHAL_PROOF = "root"`）の節点の上限を、席ごとに替えられるようにした。
  エンジンには前から opts の鍵 **`lethal_nodes`**（`DecideOptions::lethal_nodes`・既定 `LETHAL_ROOT_NODES` = 20,000）が在り、
  根の判定（`decide.rs` の ⓪ → `lethal::root_proof`）はこれを使っていた。足したのは Python 側の配線だけ:
  `RsGame.decide(lethal_nodes=…)`・`arena --cand-lethal-root-nodes`（候補席）・`arena_shard --lethal-root-nodes`（基準席）・
  `rs_decide_latency.py --lethal-nodes`。**既定は 20,000 のまま**（エンジン・生成・アリーナ・serve の挙動は 1 bit も変わらない）。
- **選んだ上限: 80,000**（予告どおり）。300 局 12,716 点で証明 853 → **966（+113・+13%）**、増えた 113 点は**全部、本当の配り方でも勝てる**。
  偽の証明は 1 点（3012157:p2:12:124・既知）のまま増えない・証明した手は 1 点も変わらない・判定の節点 ×3.06。
- **費用**: 出荷の設定（世界 2 本・1,280 回・枝予算なし・証明 root・r23・20 局・1 コアずつ同時）で **1 手の平均 +19.6%（0.387 → 0.463 秒）・99% 点 +10.3%**。
  予告（+10〜30%）の中。160,000 にすると証明はさらに +42（1,008）だが、平均 +43%・99% 点 +60% と費用が跳ねる。
- **本番アリーナはコーディネータが起こす**: ジョブ定義の案 `opcg_sim/loop/jobs/arena_lnodes.json`（候補＝上限 80,000・基準＝今の形・両席 r23・
  出荷の設定・random × real・32 シャード × 84 ペア・seed 3681000 + (k−1)×1000）。予告 +1〜+3 Elo。

## 1. 直したもの

| 場所 | 中身 |
|---|---|
| `opcg_sim/api/engine_rs.py` | `RsGame.decide`／`_decide` に `lethal_nodes`（None＝渡さない＝Rust の既定）。 |
| `opcg_sim/loop/arena.py` | `--cand-lethal-root-nodes N` → 候補席の opts の `lethal_nodes`（`CAND_VALUE_FLAGS`）。 |
| `opcg_sim/loop/arena_shard.py` | `--lethal-root-nodes N` → 基準席の opts の `lethal_nodes`（`_best_kw`）。 |
| `tests/scripts/rs_decide_latency.py` | `--lethal-nodes N`（両席・根の秒の測り直し〔`Game.lethal_json` の `max_nodes`〕も同じ上限）。 |
| `rust/opcg_engine/src/search/tests_lethal.rs` | (h) `lethal_nodes` の鍵で上限が替わる・鍵が無ければ 20,000。 |
| `tests/test_arena_search_opts.py` | 鍵は名指しした席にだけ載る（基盤健全性）。 |
| `opcg_sim/loop/jobs/arena_lnodes.json` | 本番アリーナのジョブ定義の案（`worker_job --dry-run` で k=1／32 の seed が 3681000／3712000 になるのを確かめた）。 |

- 鍵の名前は仕様の例の `lethal_root_nodes` ではなく、**既に在った `lethal_nodes`** を使った（`decide` の opts・戻り値の `stats.lethal_nodes` と同じ名前・
  新しい鍵を足すと同じものに 2 つの名前ができる）。アリーナの旗だけは仕様どおり `--cand-lethal-root-nodes`（`root_reply` の応手の上限 `lethal_reply_nodes` と紛れないように）。
- `lethal.rs` の手順の書き出し（`Solver::walk`・計測用の `trace` のときだけ動く）は応手ごとの読み直しに定数 `LETHAL_ROOT_NODES` を使ったまま。
  打つ手には関わらない（証明と最初の 1 手は `solve_root` が `max_nodes` で決める）ので触っていない＝同時に走る `rs-defense-proof`（§7.51）と触る場所が重ならない。

## 2. 証明の数と偽の証明（`lethal_speed_check.py judge`・§7.25 の記録 300 局）

器は §7.25 の記録（`n1_results/lethal_speed/record_3012000_300.jsonl.gz`）をそのまま当て直し、上限だけを替えて判定した（`run_judge.sh`）。
偽の証明は、本当の配り方 1 通りだけで上限 160,000 の判定（`--true-deal`）が「勝てない（no）」と言う点。

| 上限 | 証明 | 20k から増えた証明 | うち本当の配り方で勝てる | 偽の証明 | 判定の節点 | 判定の秒（4 並列の合計） |
|---|---|---|---|---|---|---|
| 20,000（今） | 853 | — | — | 1 | ×1.00 | ×1.00 |
| 40,000 | 913 | +60 | 60 | 1 | ×1.76 | ×1.74 |
| **80,000** | **966** | **+113** | **113** | **1** | **×3.06** | ×3.31 |
| 160,000 | 1,008 | +155 | 155 | 1 | ×5.30 | ×5.37 |

- どの上限でも**減った証明 0・証明した手が変わった点 0**（`compare` の `invariant_ok`）。上限を上げると判定不能（unknown）が証明（win）か「無い」（no）に分かれるだけ。
  20k → 80k: 判定不能 3,500 点のうち 113 点が証明・668 点が「無い」に変わる。
- 偽の証明は 4 つの上限とも同じ **3012157:p2:12:124 の 1 点**（相手の手札 0・本当のライフの OP05-096 の【トリガー】で止まる・`2026-10-24_weak_points.md` §4.2）。
  本当の配り方で判定不能の 2 点（3012043:p1:11:78・3012044:p2:14:144）も同じ（器の癖・実局は証明した側の勝ち・同 §4.1）。
- 証明 853 は weak-points の 860 と 7 点違う。違いは 2 局（3012167・3012213）だけで、weak-points の判定は `rs-unpayable-activate`（払えない【起動メイン】を合法手から外す・
  3145b25）の前の本線で回っていた（weak-points のブランチに 3145b25 が無い）。記録した手を今の規則で当て直すと 3012213 の 7 点の証明が消え（6 点は no・1 点は判定不能）、
  点の数も 12,710 → 12,716 に変わる。**今の本線の数として 853 を基準にした**（`compare_now_vs_n20k.json`）。上限 4 倍で +113 は weak-points の +112 と同じ。

## 3. 費用（`rs_decide_latency.py`・出荷の設定）

両席 r23・世界 2 本・1,280 回・枝予算なし・証明 root・random × synth_roles・20 局（seed 2999900〜）。4 つの上限を 1 コアずつ同時に回した（`run_latency.sh`）。

| 上限 | 手数 | 証明で打った手 | 1 手の平均 | 中央 | 90% 点 | 99% 点 | 最大 | 根の判定 1 回の平均秒 | 同 99% 点 |
|---|---|---|---|---|---|---|---|---|---|
| 20,000 | 2,786 | 68 | 0.387 | 0.228 | 0.997 | 2.515 | 4.33 | 0.085 | 0.46 |
| 40,000 | 2,787 | 73 | 0.414（+7.0%） | 0.271 | 1.082 | 2.549（+1.4%） | 4.71 | 0.149 | 0.88 |
| **80,000** | 2,783 | 81 | **0.463（+19.6%）** | 0.277 | 1.280 | **2.775（+10.3%）** | 4.57 | 0.267 | 1.73 |
| 160,000 | 2,795 | 95 | 0.555（+43.4%） | 0.276 | 1.720 | 4.015（+59.6%） | 5.87 | 0.470 | 3.25 |

- 根の判定は自分のメインの判断点の 88% で走り、判定不能の点は上限まで読むので、増分はほぼ「判定不能の点の数 × 増えた節点」。
- 局の進み方は上限で変わる（証明で打つ手が増える）ので、手数は同じではない。差は 1 局あたり 1 手未満で、時間の比較には効かない。
- 1 手の平均の基準 0.387 秒は `2026-10-21` の 0.24 秒より大きい: 4 本を同時に回した（コアの取り合い）こと、その後に窓の読み直し（§7.42）が入ったこと。比較は同じ条件の 4 本の間でだけ読む。

## 4. 上限の選び方

- 20k → 40k → 80k → 160k で、増える証明は +60 → +53 → +42 と緩やかに減る一方、費用は 1 手の平均 +7% → +12% → +20%（いずれも前の上限から）と増える。99% 点は 80k までほぼ平ら（+10%）で、160k で +45%（80k から）と跳ねる。
- 予告（80k・1 手 +10〜30%）の範囲に収まり、99% 点が跳ねる手前の **80,000 を候補にする**。160k は証明 +4% のために 99% 点 4 秒（対戦 API の 1 手の体感）を払う。
- Elo の見積もりは予告のまま（+1〜+3）: 証明全体の価値が +5.2 Elo（`2026-10-22_lethal_prod_arena.md`）で、増える 113 点の多くは探索でも同じ手を打つ。アリーナで測る。

## 5. 確かめ

- Rust 単体テスト (h)（`tests_lethal.rs::root_node_cap_comes_from_the_lethal_nodes_key`）: 鍵が無ければ 20,000・鍵で 80,000／1 に替わる・上限 1 では判定不能で探索へ戻り既定 off と同じ手・N、既定の上限では `kind="lethal"`。
- `tests/test_arena_search_opts.py`: `--cand-lethal-root-nodes` は候補席の、`--lethal-root-nodes` は基準席の `lethal_nodes` にだけ載る。
- golden: エンジンの非テストのコードは触っていない（wheel の挙動は同じ）・証明は既定 off。golden の 2 本（`test_rs_golden_audit.py`・`test_rs_golden_replay.py`）は `make test` で記録と一致＝差分 0。作り直しはしていない（作り直しても同じ記録になる）。
- `make test`・`make audit-cross`: §6。

## 6. ゲート

- `make rust-develop` → `make test` **green**: cargo 633 passed・1 ignored／pytest 879 passed・13 skipped（`n1_results/lethal_nodes/make_test.log`）。
- `make audit-cross` **void 0**（120 ペア・240 局・`n1_results/lethal_nodes/audit_cross.log`）。

## 7. 次

- 本番アリーナ（`arena_lnodes.json`・コーディネータが起こす）。採るなら `opcg_sim/loop/engine.py` に `PLAY_LETHAL_NODES = 80000` を足して serve（`routers.py`）・監査の器へ渡す 1 行の変更（このブランチではしていない）。
