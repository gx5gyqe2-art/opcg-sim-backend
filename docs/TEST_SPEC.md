# テスト仕様書 — opcg-sim-backend

本書は `opcg-sim-backend` の**テスト仕様書**である。対になる **システム仕様書** は
[`docs/SPEC.md`](SPEC.md)。リーダー個別のテスト方針は [`docs/leader_specs/_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md)、
既知の挙動差異は [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md)。

---

## 1. テスト戦略・原則

- **効果の意味的正しさ（テキスト準拠で正しく発動するか）は、自動テストではなく
  「デッキ単位の手動検証」で担保する**（→ §8）。本書のテスト群は
  「**壊れていないこと**」——クラッシュ／カード消失／場超過を起こさない、
  既存挙動が退行しない——の保証に役割を絞る。
- **挙動を変えたら全カード挙動ベースライン（`full_card_baseline.json`）を再生成**し、
  差分をレビューして品質ゲートを通す。

### ⚠️ 注意：「成功するが何もしない」効果の死角
`RULE_PROCESSING`（「ルール上、〜になる」等の常在ルール注記）は**実行時 no-op** で、
resolver は `success = True` を返す。エラー・フォールバック・OTHER のいずれにもならず、
**構造監査も挙動ベースラインも素通りする**。「パースできた＝動く」ではない。

- 実例：リーダー OP15-058 エネル「ルール上、自分のドン!!デッキは6枚になる」が
  長期間 **未適用（10枚のまま）** だった。`RULE_PROCESSING` が no-op で、ドン!!デッキ
  枚数は別経路（`GameManager` 構築時）で初期化し直さないと既定の10枚のままになるため。
- 教訓：`RULE_PROCESSING` に落ちる能力は、**別経路でルールが強制されているかを必ず
  実機で確認**する。セットアップ／経済ルール（ドン!!デッキ枚数等）は per-ability の
  盤面差分の外側にあるので、**ゲーム不変条件として個別テストを足す**こと。

### 重要度分類（3階層）とテスト追加ルール

テストは「無ければ実プレイのゲームプレイ退行を見逃すか」で3階層に分類する。
**時間（重い/軽い）ではなく重要度が分類基準**——探索/自己対戦を回す内部機構の健全性テストは
性質上重くなりがちだが、それは結果であって基準ではない。

| 階層 | 判定基準 | マーカー |
|---|---|---|
| **必須** | 壊れたら実プレイが直接崩壊する（構造不変条件・コアルール・ラチェット・API契約） | 不要（常時実行） |
| **標準** | 機能単位の回帰保証（リーダー効果・パーサ・CPU判断の質等） | 不要（常時実行） |
| **基盤健全性** | ゲームプレイの正しさとは別軸。探索/自己対戦/学習パイプラインの内部機構（決定論・キャッシュ一致・make/unmake整合性等）のみを見る | `@pytest.mark.cpu_infra` |

**新しいテストを追加するとき**:
1. 上記基準で重要度を判定する。迷ったら必須/標準側に倒す。
2. 基盤健全性の場合のみ `@pytest.mark.cpu_infra`（module-level `pytestmark` 可）を付与する。
3. §2 のスイート表に1行追記する（既存ルール）。基盤健全性の場合はその旨を明記する。

現在 `cpu_infra` に分類済み: `test_game_driver.py` / `test_cpu_arena.py` /
`test_replay_roundtrip.py` / `test_cpu_pv_order.py` / `test_plan_cache.py` /
`test_cpu_make_unmake.py` / `test_card_cache.py` / `test_cpu_search_override.py` /
`test_cpu_replay.py` / `test_perf_gate.py` / `test_p2_harness.py` / `test_p3_components.py` /
`test_rl_datagen.py` / `test_turn_solver.py` / `test_learned_root_readout.py` /
`test_mcts_terminal_decay.py` / `test_selfplay_v4_datagen.py` / `test_value_net_aux_turns.py` /
`test_pd_mixed_label.py` / `test_learned_candidate_prune.py` / `test_learned_aux_tiebreak.py` /
`test_rl_encoder_v4.py` / `test_mark_seeds.py` / `test_value_net_distill.py` / `test_peak_alert.py` /
`test_journal.py`（`test_real_playout_make_unmake_roundtrip`のみ）。

### 実行方法（重要）
logger が `sys.stdout` を直接掴むため、pytest はキャプチャ無効で実行する。

```bash
make test        # フルスコープ（push前ゲート）。-n auto = pytest-xdist 並列。-m "not slow" = 通常ゲートの既定条件
make test-fast    # 開発中のイテレーション用（cpu_infra 除外。push前ゲートの代替ではない）
```

コマンドの正本は `Makefile`。`-s/-p no:capture` を付けないと I/O error になる。CI は無く、
`make test` がマージ前の唯一の確認手段（2026-07-11 廃止・詳細は `CLAUDE.md`）。

`slow` マーカー（`pytest_configure` で登録）は **`make test` から除外**する重テスト（手動実行前提）。現状の対象は
`test_journal.py::test_parked_resume_make_unmake_roundtrip`（8 seed × 全手の make/unmake 照合 ~245s＝
スイート単独最重量・並列でも壁時計上限を作る）。**make/unmake（journal）周辺を変更したら手動実行**する:

```bash
make test-slow   # 重テストだけ
```

合格条件: 出力が `passed` / `xfailed` / `skipped` のみ。`failed` / `xpassed` を残さない。

---

## 2. テストスイート一覧

### コアルール（ターン/戦闘/召喚酔い/場上限）
| ファイル | 役割 |
|---|---|
| `tests/test_rules_summoning_field_limit.py` | **召喚酔い/速攻**（登場ターン攻撃不可・速攻例外・リーダー非対象）と**場5体上限**（6体目で `FIELD_OVERFLOW_TRASH` 強制トラッシュ／効果登場でも発火／境界／**押し出し確定が【登場時】解決より先**）の検証 |
| `tests/test_turn_start_trigger.py` | **ターン開始時トリガー**（TURN_START。「自分のターン開始時、発動できる」OP11-040＝確認→受諾/拒否、ドン8枚条件は**ドン!!展開前**判定の裁定込み） |
| `tests/test_event_main_playability.py` | **イベントのメイン発動可否**（【メイン】効果を持つイベントのみ手札からメインで発動可。【カウンター】/【トリガー】専用イベントは合法手に出ず play_card_action も拒否。OP09-078/OP06-059/OP11-080） |
| `tests/test_event_listener_triggers.py` | **イベントリスナー誘発**（他カードの「…が登場した時」/「…キャラがKOされた時」を登場/KO地点から走査して発火。側・特徴・元々のパワー・出所ゾーン・タイミングのフィルタとドン条件/ターン1回。OP14-041/OP01-061） |
| `tests/test_power_filter_don.py` | **パワー参照対象と付与ドン**（+1000/枚は持ち主のターン中のみ＝相手ターン残置ドンは「パワーN以下」判定に乗らない。matcher 単体＋神の裁き OP15-075 の KO e2e） |
| `tests/test_trigger_cost_confirm.py` | **自動誘発のコスト使用確認**（コスト句の支払いは常に任意＝`CONFIRM_OPTIONAL`。拒否で未払い・受諾で支払い解決／同時複数の誘発が中断で消えない／起動メインは確認なし。OP16-073/065） |
| `tests/test_effects_engine.py` | エンジン実行系の盤面変化（プレイ/アタック/ブロック/カウンター/効果解決） |
| `tests/test_realdeck_play.py` | 実カードでの盤面変化・除去保護・対話 |
| `tests/test_self_cannot.py` | 自己制限（CANNOT_*）の enforce |
| `tests/test_arrange_deck.py` | デッキ配置/並び替え対話 |

### オンライン対戦（ルーム/WS）
| ファイル | 役割 |
|---|---|
| `tests/test_rule_online.py` | ルール対戦のルーム生成→デッキ選択→開始→アクションの WS 同期、開始の ready ガード（`load_deck_mixed` をモックし Firestore 非依存） |

### API 層（FastAPI・HTTP/WS スモーク）
| ファイル | 役割 |
|---|---|
| `tests/test_api.py` | `opcg_sim/api/app.py` の **API 契約**を `fastapi.testclient.TestClient` で検証（エンジン挙動は他スイートが担保するためスモーク粒度）。対象: health／cards／log／対局生成→state→マリガン→TURN_END／CPU step の契約（`cpu_acted`・`waiting_for`）／sandbox 生成・list・WS ブロードキャスト（STATE_UPDATE）／rule ルーム生成→SET_DECK→START／未知 ID・DB 未初期化（デッキ CRUD）の整形済みエラー応答・`X-Session-ID` 往復。`load_deck_mixed` をローカルカード DB の stub に差し替え Firestore 非依存 |
| `tests/test_flagship_api.py` | フラッグシップ結果集計 API（`opcg_sim/api/flagship/`、設計は flagship リポジトリ docs/design.md §12）。リーダー辞書（カードDB `種類=リーダー` 137件）配信／結果の登録（開催単位の全置換・冪等 PUT）→サマリ→詳細→削除の一連／ポストURL重複 409／placement・リーダーのバリデーション／SQLite 遅延作成（`OPCG_FLAGSHIP_DB` を tmp に向ける） |
| `tests/test_flagship_extract.py` | フラッグシップ結果抽出（`opcg_sim/api/flagship/extract.py`、LLM不使用の辞書マッチング、設計 docs/design.md §13）。137リーダーのエイリアス生成（正規名・短縮名・色略称）／順位パターン写像（優勝/準優勝/N位/ベストN）／色略称の card_number 一意化／同名（クロコダイル等）の曖昧化／confidence／NFKC正規化／`/extract`・`/oembed` の API 契約 |
| `tests/test_flagship_xfetch.py` | X ポスト本文取得（`opcg_sim/api/flagship/xfetch.py`、syndication API 主軸・oEmbed フォールバック、設計 docs/design.md §15）。URL→tweet id 抽出／決定的トークン算出／syndication JSON の本文組み立て（note_tweet 優先＝長文対応）／oEmbed フォールバック／取得不可時 None／`/ingest`（取得+抽出の一気通貫）・`/oembed` の API 契約。ネットワークは monkeypatch で遮断（ヘルメティック） |
| `tests/test_flagship_xsearch.py` | X recent search による結果ポスト発見（`opcg_sim/api/flagship/xsearch.py`、有料 X API v2、設計 docs/design.md §16）。クエリ構築（ハッシュタグ×アカウントの OR＋`-is:retweet`/`lang`）／@handle・URL からの username 抽出／v2 レスポンス整形（author 突き合わせ・note_tweet 優先・url 生成・空本文除外）／`X_BEARER_TOKEN` 無効時の graceful degrade／`/discover`・`/discover/status` の API 契約（無効=503・上流エラー=502・空指定=400）。ネットワークは monkeypatch で遮断 |
| `tests/test_flagship_store.py` | flagship 結果永続化ストア（`opcg_sim/api/flagship/store.py`、設計 docs/design.md §17）。`get_store()` の選択（Firestore 有→FirestoreStore／無→SqliteStore の graceful degrade）／FirestoreStore の全置換・取得・削除（スナップショット保持）・シリーズサマリ・URL 重複判定を**インメモリ Fake Firestore** で検証／`resources.db` を差し替えて API 全経路（PUT→サマリ→詳細→409→DELETE）が Firestore バックエンドでも SQLite と同挙動になることを確認 |
| `tests/test_flagship_trend.py` | 全国の優勝リーダー傾向集計（`opcg_sim/api/flagship/trend.py`、設計 docs/design.md §16.6）。(投稿者×日) 重複除去／集計アカウント除外／キャラ単位正規化（card 解決・未解決別名の合流）／`/trend` の API 契約（既定トレンドクエリ・503）。実リーダー辞書使用・ネットワークは monkeypatch 遮断 |
| `tests/test_flagship_match.py` | 収集ポスト × TCG+開催 の照合（`opcg_sim/api/flagship/match.py`、設計 docs/design.md §16.7）。handle 一致（自動確定候補）／表示名ファジー一致（要承認・閾値0.6）／同チェーン別店の誤爆除外／日付近接での絞り込み／個人ポスト=候補ゼロを実データ実例で検証（純粋関数） |
| `tests/test_flagship_storesns.py` | 店舗X の手動ディレクトリ（`opcg_sim/api/flagship/storesns.py`、設計 docs/design.md §16.9）。店名→店舗X の登録/更新/解除／開催マスターへの**上書き優先オーバーレイ**（TCG+ 値より手動優先）／`POST /stores/sns`（@handle→URL 正規化・空で解除）／`/events` が手動店舗X を TCG+ より優先して返すことを SQLite（tmp）と Fake Firestore の両実装で検証 |
| `tests/test_flagship_winnerstore.py` | 収集優勝ポストの一時保管（`opcg_sim/api/flagship/winnerstore.py`、設計 docs/design.md §16.7）。tweet_id 重複除去／再収集で event_id 保持／未紐付け抽出／開催割り当て／**削除（承認時の掃除）**を SQLite（tmp）と Fake Firestore の両実装で検証・`get_winner_store()` 選択 |
| `tests/test_flagship_link.py` | 収集の蓄積と開催紐付け（`/collect`・`/link/review`・`/link/approve`、設計 docs/design.md §16.7）。収集→DB蓄積／未紐付けポストの開催マスターへの照合レビュー（handle自動候補・個人ポストは候補ゼロ）／**承認で収集ポストを削除**（ポスト内容は恒久保持しない・結果は別途保存）・`event_id=null` は解除で行を残す／TCG+不達でもマスターにフォールバック・未設定503。検索と TCG+（`tcgplus.py`）は monkeypatch 遮断・SQLite(tmp) 永続 |
| `tests/test_flagship_eventmaster.py` | 開催マスターの永続化（`opcg_sim/api/flagship/eventmaster.py`・`GET /events`、設計 docs/design.md §16.8）。`get_event_master()` 選択／シリーズ別 upsert・list を SQLite(tmp)・Fake Firestore の両実装で検証／**TCG+ が過去開催を消しても `/events` が過去+現行を返す**（スナップショット保持）／TCG+不達でもマスターを返す |

### カード効果（パーサ/ゴールデン/全カード・回帰/安定性）
| ファイル | 役割 |
|---|---|
| `tests/test_parser.py` | レガシーパーサ単体 |
| `tests/test_golden.py` / `tests/golden/*` | ゴールデンコーパス（AST 指紋の部分一致） |
| `tests/test_full_card_audit.py` | 全カード構造不変条件ゲート（EXCEPTION/CARD_LOSS/TEMP_LEAK=0） |
| `tests/test_full_card_baseline.py` | 全カード挙動ベースライン回帰（`full_card_baseline.json` と一致） |
| `tests/test_verified_decks.py` | **手動検証済みデッキの効果回帰**（§8）。ベースラインが捕捉できない常在ルール（RULE_PROCESSING）・ON_LEAVE 誘発・勝利条件・ドンデッキ枚数・カード名別名・持続時間等を意味的に固定 |
| `tests/test_cpu_selfplay.py` | CPU 対 CPU 自己対戦の完走・決定論・clone 非破壊・合法手適用・インバリアント検出 |

### CPU 対戦・AI（評価/探索/相手モデル・SPEC §2.5）<!-- 自デッキ勝ち筋プランは 2026-06-27 全廃 -->
| ファイル | 役割 |
|---|---|
| `tests/test_cpu_ai.py` | 評価関数・α-βビーム探索・難易度情報方針（easy/normal/hard）・リーサル認識・有効パワー閾値・単一対象選択探索・horizon（B1/B2-lite）の保証テスト＋**B-2 ドン付与の手生成プルーニング**（意味ある配分のみ＝閾値跨ぎ／付与ドン条件残し・overcap/レスト除外・非ドン素通し） |
| ~~`tests/test_cpu_self_plan.py`~~ | **【削除 2026-06-27】** 自デッキ勝ち筋プラン／アーキタイプ・プリセット系の全廃（control 倍率が vs-midrange −5.7pp の A/B を受けたフラット評価ベースライン化）に伴い、テスト対象（`cpu_self_plan.py`・plan-gated 評価項）ごと削除。旧内容＝aggro/midrange/control 自動分類・plan 限定の置物/カウンター/ライフ/攻め圧重み・逆算リーサル/マイルストーン・脅威キーワード資産・C-4 settle 不確実性ディスカウント・時間割引・探索地平線越え価値（いずれも plan=None 完全同値の回帰ガード）。**注**: plan 非依存で存続した concave ライフ（`test_life_value_is_concave_*`）は本ファイル削除に伴い回帰ガードを失う＝再カバーは未整備 |
| `tests/test_cpu_puzzles.py` | **CPU 検証基盤（フェーズ0・全変更のゲート）**: 正解手種が既知の局面（致死を取る）＋アクティブドンの線形評価ピン。**2026-06 レビュー収束項（存続）**: A-3・E-1 min ビーム剪定の sort 方向。**【撤去 2026-06-27】** plan-gated 機能のテスト（B-1(a) アイドルドン末端減価／A-1 アンブロッカブル評価／A-2 アーキタイプ依存スケール）は自デッキ勝ち筋プラン全廃に伴い、**B-1(b) カウンター強要（推定カウンター応答モデル）／公開情報ベリーフ更新（手札枚数・トラッシュ）は CPU 評価の L1 単一系統化（profile ベース eval 補正の撤去）に伴い**削除 |
| `tests/test_cpu_arena.py` | **基盤健全性**（`cpu_infra`）。**検証基盤の絶対強度メトリクスの機械健全性**（`tests/harness/cpu_arena.py`）: 凍結ベースライン Elo 変換（勝率→Elo の 0.5→0／単調／対称）・非対称対局＋席交互アリーナ・regret ログ（`cpu_ai.decide_with_regret`＝非負・有限・easy/単一手で 0）。実ゲームは低速なので機械健全性のみ高速・有界に固定 |
| `tests/test_cpu_replay.py` | **基盤健全性**（`cpu_infra`）。**CPU 思考トレースの健全性**（`tests/harness/cpu_replay.py`）: trace は観測専用で手を変えない・RNG 中立（trace 有無で進行が分岐しない）・同一 seed の決定論再現・トレース 4 項目（候補スコア/regret/J値成分/読み筋）の存在と読み筋 PV の有界性 |
| `tests/test_game_driver.py` | **基盤健全性**（`cpu_infra`）。**共通対局ドライバ**（`tests/harness/game_driver.py`・設計⑥)の機械健全性: 同一 seed の決定論・observer 不干渉（観測専用の契約）・席の写像等価（run_one_game/play_game と一致）・`stop_after_decisions` 有界化・**learned(既定Gen＝現v4/gen4) 自己対戦の seed 再現** |
| `tests/test_replay_roundtrip.py` | **基盤健全性**（`cpu_infra`）。**実対局リプレイのラウンドトリップ**（`tests/harness/replay_runner.py`）: 録画（人間=private rng・card_id 基準記録）→記述子から再生（人間手注入＋CPU 再 decide）→勝敗・手数・ターン一致＋逆写像 miss=0。hard／**learned(既定Gen＝現v4/gen4)**／**coin toss（first_player=random）** の3系統＋リゾルバ単体 |
| `tests/test_replay_frames.py` | **リプレイ盤面フレーム**（`services/replay.py::_replay_record_frame`＋`GET /replay/frames`・リプレイビューアのデータ供給契約）: frames↔actions↔decisions の action_index 整合（フレーム0＝初期盤面のみ None）・フレームカードは動的状態のみ（マスター情報を持たない＝サイズ抑制）・`_FRAME_CAP` 超過で記録停止＋`frames_truncated`・非 traced 対局は記録なし＋整形エラー |
| `tests/test_perf_gate.py` | **基盤健全性**（`cpu_infra`）。**CPU 性能ゲートの判定ロジック**（`tests/scripts/perf_gate.py`・§5.1）: `evaluate_gate` 純関数（強度不足/レイテンシ超過/失敗局/データ不足→FAIL・理由の蓄積）＋ gen2/gen3/gen4_*.npz ハッシュの安定性（gen4(v4)＝本番既定・2026-07-12採用）。実対局は回さず高速固定 |
| `tests/test_cpu_learned.py` | **学習型CPU本番配線**（既定＝v4(gen4)・温スタート検証は v1(gen2) を明示ロード。`opcg_sim/src/core/cpu_learned.py`／`opcg_sim/src/learned/`）: 合法手・decide_client ルーティング・seed 決定論・席別エンジン（net-vs-net 等価）・**符号化/行動特徴の訓練時ドリフト検知（v1/v2）**（`tests/harness/{rl_encoder,opcg_action,rl_net,az_policy,az_mcts_tree}.py` は本番 `opcg_sim/src/learned/{encoder,action,value_net,policy,mcts}.py` への委譲shim＝TEST_E/TEST_A は本番と同一オブジェクトでドリフトは構造的に不可能・退行検知として存続。`tests/harness/opcg_game.py` は本番 `adapter.OPCGGame` の薄い継承＋研究専用 `new_game` のみ追加）・選択対話の併合（CONFIRM_OPTIONAL accept/decline・up-to ライフ追加・**ARRANGE_DECK の並び替え/上下選択**・position キー）・**ルート等価手マージ**（同名複製の訪問数分裂で PASS に負ける実害の反転ケース＋複製なし恒等）・トレース記述（decline の accepted 明示・dialog 種別）・**符号化世代 v2**（リーダー付与ドン特徴＝v1 では不可視・v1 出力不変・npz 入力次元からの自動判別）・**温スタート拡張**（v1→v2 の重み拡張が恒等＝拡張ネット×v2符号化 == 出荷×v1符号化・policy も恒等・縮小拒否・版差は scalars_dim のみが seam＝将来版に同一コード対応） |
| `tests/test_learned_root_readout.py` | **基盤健全性**（`cpu_infra`）。**learned root 読み出しの二重ゲート乗り換え規則**（`cpu_learned._select_root_group`・`docs/reports/cpu_learned_mark_review2_20260711.md` §S1）: 実対局2局×16人間マークの記録統計（visit%/Q）を固定入力とした全数回帰＝g1@12/@24 は乗り換え（人間指摘と一致）・g2@20 の低訪問楽観 Q（次decideで−0.54崩落）は訪問比ゲートで棄却・g2@22/@23 の微小 Q 差（同格ノイズ）は Q差ゲートで棄却・Q同値（−1飽和）は訪問トップ維持・min_gap=inf は argmax(N) と一致（ロールバック経路） |
| `tests/test_mcts_terminal_decay.py` | **基盤健全性**（`cpu_infra`）。**TreeMCTS 終局値の深さ減衰**（`±max(TERM_FLOOR, 1−TERM_DECAY·depth)`・同レポート §F2）: 決定的グラフゲーム（汎用 make/unmake IF）で「負け確定なら終局が遠い方（粘る手）」「勝ち確定なら近い方（最短リーサル）」を選ぶ・終局直行 edge の backup 値が正確に ±scale・減衰0=従来の −1 飽和・床で下げ止まり |
| `tests/test_value_net_leader_slots.py` | **ValueNet のリーダー条件付け専用枠**（`lead_slots`・`docs/reports/lc_value_net_plan_20260708.md`）: `to_leader_conditioned()` の恒等性（追加ゼロ行＝拡張直後は旧net予測と一致）・二重適用拒否・save/load 往復（旧形式npz=lead_slots無しの後方互換込み）・`expanded()`（enc版温スタート）との直交併用・解析勾配=数値微分一致・**リーダーIDのみで決まる合成ターゲットを lead_slots=2 だけが fit できる**回帰 |
| `tests/test_effect_features.py` | **EffFeat＝効果セマンティクス特徴テーブル**（`opcg_sim/src/learned/effect_features.py`・`docs/reports/effect_semantics_v3_plan_20260708.md` §1）: 決定性（2回構築一致）・PAD行ゼロ・次元・効果持ち全カードの能力ブロック非ゼロ・実カードのスポットチェック（OP03ナミ=VICTORY独立枠＋資源条件／OP11ナミ=ON_OPP_ATTACK+2kバフ+HAS_DON+手札コスト／コスト操作とパワーバフの status×値スケール分離／ATTACH_DON全体センチネル／印刷キーワード・カウンター値・種別の静的ブロック） |
| `tests/scripts/replay_reeval.py` | **マーク付きリプレイ再評価CLI**（`opcg-replay/v1`のframes+marksから各マーク直前フレームの盤面を復元し候補ネットにdecideさせ「人間の指摘どおり手が変わるか」を検証＝ネット改善の人間フィードバック回帰。全編再生は山札覗き効果＋ドン経済で漂流するため局所復元方式を採用。カウンター系マークは直前の PASS（ブロッカー段の見送り等）・RESOLVE_EFFECT_SELECTION（【アタック時】効果の選択）を遡って攻撃宣言に着地し、宣言〜マーク間の記録応答を再生して復元する。`.json.gz` 直読み可） |
| `tests/scripts/defense_rate_probe.py` | **防御応答の守り採択率 計器**（v5 R1 調査・`docs/cpu_v5_plan.md` §3-R1）: 既定 net で自己対戦し、防御応答（SELECT_COUNTER/BLOCKER）局面の「守る(非PASS)採択率」を net argmax／温度1期待（データ挙動）／L1-hard（良質目安）の3系統で集計。温度延長が守りを過剰注入したか（R1）を切り分ける読み取り専用計器。**実測結論: R1 否定**（net argmax はむしろ L1 より守らず・温度延長は L1 水準への補正＝過剰注入でない） |
| `tests/scripts/defense_rate_probe.py` | **防御応答の守り採択率 計測CLI**（v5計画 §3-R1 の調査計器・読み取り専用）: 既定 net（gen4）で自己対戦し、SELECT_COUNTER/BLOCKER 局面の守り率を net argmax／温度1期待（データ挙動）／L1-hard（良質目安）の3系統で比較。「守りすぎ」の原因が防御温度延長の過剰注入か net 体質かを切り分ける（24局630局面で否定＝netはむしろ守らなさすぎ・温度延長は補正的） |
| `tests/scripts/clock_error_by_leader.py` | **時計誤差の対面別分解CLI**（v4監視 diagnostics・`docs/reports/v4_adoption_20260712.md` §3/§6）: batch.npz（スキーマv2）の局面を自リーダー/対面ペアでグループ化し、残りターン補助ヘッドの MAE・bias（ターン換算）を分解。平均誤差が隠す対面別の系統偏りを可視化＝§5.5-2（自デッキ残特徴）の切り分け材料。読み取り専用 |
| `tests/scripts/mark_gate.py` | **v4 マーク回帰ゲートCLI**（`docs/reports/v4_adoption_20260712.md` §5＝v4採用ゲート）: `tests/fixtures/replays/` の2局×16人間マークを復元し、challenger / baseline（既定=v3）ネットで各Kシード decide→「人間指摘方向率」を比較。判定＝F4代表6件の過半で改善 かつ 既存正着ガード3件（g1@12/@24・g2@20）非退行で PASS（exit 0）。v3 vs v3 で「改善0/6・非退行OK・FAIL」を確認済み（＝ゲート感度の基準線） |
| `tests/test_learned_candidate_prune.py` | **基盤健全性**（`cpu_infra`）。**learned 候補の無駄手枝刈り**（`adapter.OPCGGame.legal_actions`・v5 §4補）: L1/α-β と同じ `_prune_futile_attacks`/`_prune_don_moves` を learned MCTS 候補にも適用（`SERVE_PRUNE_FUTILE`）。枝刈りON が _prune_* 適用後と一致・OFF で merged 素集合へ復帰（ゲート）・候補を空にしない（TURN_END 常在） |
| `tests/test_learned_aux_tiebreak.py` | **基盤健全性**（`cpu_infra`）。**aux 粘り項**（`cpu_learned._aux_tie_scale`・`SERVE_AUX_TIEBREAK`・v5 §4-1）: 飽和域の葉価値を残りターン予測 t̂ で減衰 `v·max(TERM_FLOOR, 1−AUX_TIE_DECAY·t̂·sat)`。非飽和域は恒等・敗勢は t̂ 大（延命）を選好・優勢は t̂ 小（速い勝ち）を選好・床で下げ止まり・t̂<0 クランプ（増幅しない）・`predict_with_aux`＝分離呼び出しと一致・ゲート OFF/終局 ±1 は従来どおり |
| `tests/test_selfplay_v4_datagen.py` | **基盤健全性**（`cpu_infra`）。**v4 自己対戦データ生成**（`p3_loop.selfplay_game` 拡張・`docs/reports/v4_adoption_20260712.md` §1）: q_root∈[-1,1]/turns_left（非負・終局で0）の記録・batch スキーマ v2（pack_vdata のキー/形状）・同一seed決定論・**sticky世界線**（同一(turn,手番)で決定化seed固定＝戦闘応答の交互手番でも dict で保持・ターンが変われば引き直す）・**防御応答の温度延長**（temp_moves=0 でも SELECT_BLOCKER/COUNTER は温度1でサンプリング）・**L1混合席**（policy教師はnet席のみ・L1席のvalueはq_root=NaN→mergeで勝敗へ退化・決定論維持） |
| `tests/test_value_net_aux_turns.py` | **基盤健全性**（`cpu_infra`）。**ValueNet 残りターン補助ヘッド**（W2t/b2t・v4 §4-2）: 補助ヘッドの value 出力からの独立性（＝恒等温スタートの根拠）・旧npz（v3=gen3）ロードで aux ゼロ＋save/load往復・解析勾配=数値微分一致（W2t/b2t＋共有層への寄与・NaNラベルのマスク）・構造拡張4種（expanded/LC/to_v3/widened）の aux 引き継ぎ・合成ターゲットの学習可能性（NaN混在可） |
| `tests/test_pd_mixed_label.py` | **基盤健全性**（`cpu_infra`）。**v4 混合ラベルとスキーマv2後方互換**（`pd_batch_common`）: normalize_batch_v2（v1バッチ→q_root=value/turns_left=NaN の退化規則・v2素通し）・mixed_value_label（α=1で勝敗単独と一致・線形補間）・ring_append の v2キー連結と v1/v2 混在 |
| `tests/test_pd_batch_common.py` | **バッチ式アクター/ラーナー分離の純粋協調ロジック**（`tests/scripts/pd_batch_common.py`・`docs/reports/batched_selfplay_design_20260710.md`）: 鮮度フィルタ is_fresh（accept/seen/stale の境界＝未消費かつ against_round>=round-staleness）・plan_consumption の採用/スキップ内訳・update_consumed の単調性と非破壊・ring_append のcap切りとキー整合。git入出力を含むe2eはpd_*スモークで別途疎通確認済み |
| `tests/test_mark_seeds.py` | **基盤健全性**（`cpu_infra`）。**マーク局面シード**（`mark_seeds.load_mark_boards`・`p3_loop.selfplay_game` の `seed_boards`/`seed_frac`・`docs/cpu_v5_plan.md` §4-2）: 実対局の失敗局面（MAIN手番マーク）を静的フレームから復元し自己対戦の開始局面プールにする。復元プールが非空・非終局・合法手あり・中盤（turn≥2）・決定論（同一プール）・シード開始で完走しラベル採取・**seed_frac=0 は seed_boards を渡しても turn1開始と完全一致**（rng消費順不変＝シードOFFの本走が v4生成とbit同一）のゲート |
| `tests/test_rl_encoder_v4.py` | **基盤健全性**（`cpu_infra`）。**符号化世代 v4（自デッキ残の集約特徴）**（`rl_encoder` version=4・`docs/cpu_v5_plan.md` §4-3）: v3(scalars46)末尾に自ライブラリの守り/資源集約5値（残カウンター総量/密度・ブロッカー残・イベント残・高コストキャラ残）を append-only 追加（51）。版マップ単調増加・先頭46がv3と一致（並び不変）・集約値の定義一致・**自デッキのみ**（相手デッキ改変で不変＝公平性契約）・空デッキ安全・**恒等温スタート必達**（v3→v4 拡張で value/aux 出力が数値的完全一致＝新5行ゼロ） |
| `tests/test_value_net_distill.py` | **基盤健全性**（`cpu_infra`）。**value 蒸留（忘却抑制・教師アンカー）**（`ValueNet.backward`/`train` の `distill_weight`・`docs/cpu_v5_plan.md` §4-4b）: 凍結v4教師の value 予測へ引く MSE アンカーを value ヘッドに加算。distill_weight=0 は素の MSE 勾配と完全一致（挙動不変ゲート）・合成損失の解析勾配=数値微分一致（W2/b2/W1/Emb）・強めの蒸留で予測が教師値へ寄る（ラベルとのバランスで暴走しない） |
| `tests/test_peak_alert.py` | **基盤健全性**（`cpu_infra`）。**ピーク自動アラート**（`peak_alert.detect_peak`・`docs/cpu_v5_plan.md` §4-4a）: 本走の checkpoint 評価系列（mark_improved・arena_wr）から忘却開始を検知。改善中はアラートせず凍結候補=best round・2指標の同時後退が patience 回連続でアラート・単一指標後退や許容内ノイズでは誤報しない・空/単一入力の安全性 |
| `tests/test_value_net_v3.py` | **ValueNet v3（EffFeat組み込み）**（設計 §2/§5）: 恒等温スタート連鎖（scalars拡張→LC→to_v3→widened で出力完全一致・22/24幅idx双方）・順序ガード（LC前to_v3拒否/二重適用拒否/eff後LC拒否/widened縮小拒否）・W_eff含む勾配=数値微分一致・save/load往復＋旧形式後方互換（eff_dim=0）・**ゼロショット回帰**（効果特徴で決まるターゲットを未見リーダーへ汎化できるのはv3のみ＝LC埋め込みは不可）・encoder v3（scalars46/card_idx24/ステージ末尾・v2不変）とのe2e結線 |
| `tests/test_p3_components.py` | **基盤健全性**（`cpu_infra`）。**P3学習ループ部品の高速単体**（重い loop は `p3_loop.py --smoke --enc-version 1`）: action 特徴 one-hot・action_key の区別・policy 学習・**自己対戦のリーダーローテーション**（`OPCGGame.new_game(leaders=…)` が全リーダープールから両席を抽選＋リアルデッキ化＝【ドン‼×1】系リーダー効果を学習データに載せる「穴B」対策・seed 決定論／leaders 未指定は build_deck 固定＝後方互換） |
| `tests/test_p3_loop.py` | **P3学習ループの疎通**（slow・`make test`除外）: 自己対戦→value/policy 学習→クロス評価が例外なく完走（勝率シグナルは見ない）。`p3_loop.py`／`p3_run.py` は `--enc-version 2`（必須・符号化v2）・`--rotate-leaders`（穴B）を配管。p3_run の v2 Gen0 は出荷 v1 Gen2 から**温スタート**（乱数初期化しない） |
| `tests/test_p2_harness.py` | **基盤健全性**（`cpu_infra`）。**P2 harness（`tests/harness/p2_gen0.py`）の高速単体**: SL価値の配線（encode→net→[-1,1]）・SL-MCTSエージェントの合法手・save/loadラウンドトリップ・**`match()` のリーダーローテーション配管**（`leaders=…` が `new_game` へ伝播＝`p3_vs_l1.py --rotate-leaders` の土台。未指定は従来の固定リーダーで後方互換）。世代ゲート本体（`p3_gate.py`＝Gen_k vs Gen_{k-1} 損切り判定）と直接対戦参考測定（`p3_vs_l1.py`＝vs 製品L1）は、符号化世代をロード重みの入力次元から自動判別（`cpu_learned._net_enc_version`）してエージェントを構築＝チェックポイントの実際の版とズレない |

### 効果メカニクス・対話モデル
| ファイル | 役割 |
|---|---|
| `tests/test_effect_oracle_gate.py` | 静的 text↔AST 整合性 HAS_OTHER/PER_TURN_LIMIT_GAP/UP_TO_GAP = 0 のラチェット（§5） |
| `tests/test_effect_event_dest.py` | **EFFECT イベントの行き先（dest）記録**: 移動系（MOVE_CARD 等）の eventLog に dest（"LIFE" 等）が additive に載る／非移動系（LOOK）には載せない。実カード OP16-119 のライフ追加で固定（フロントの効果表示の根拠） |
| `tests/test_structural_gate.py` | 構造不変条件4スキャン＋条件偽パスのラチェット（カテゴリH 再発防止。§5/§8.5） |
| `tests/test_interaction_stack.py` | 中断スタック（`active_interaction` 互換プロパティ／`push_interaction`）のセマンティクス |
| `tests/test_replacement_interactive.py` | 置換 sub_effect のネスト中断（終端=UI提示+resume／非終端=自動解決）。SPEC §6.1 |
| `tests/test_both_sides_interactive.py` | 「お互いの〜」両側効果の各プレイヤー個別選択（相手→自分の逐次中断）。SPEC §6.1 |
| `tests/test_freeze_don.py` | FREEZE_DON（OP07-026 ドン側）＝レストのドン!!を1回リフレッシュ据え置き |
| `tests/test_on_rest_trigger.py` / `tests/test_on_rest_subject.py` | ON_REST 誘発（このキャラ／任意主語＋自分の/相手の効果で）。アタック宣言・効果レスト両経路 |
| `tests/test_execute_trash_event_main.py` | EB03-031 トラッシュのイベント【メイン】効果の発動（EXECUTE_MAIN_EFFECT + 対象選択） |
| `tests/test_char_or_don_mixed.py` | 「キャラかドン合計N枚」の混在選択（CHAR_OR_DON 候補プール） |
| `tests/test_counter_affordability.py` | **カウンター合法手の支払い可能性**: SELECT_COUNTER がコストを払えないイベントカウンターを提示しない（実デッキ×ランダム自己対戦の property・「ドン!!不足」クラッシュの回帰） |

### リーダー効果（全137枚）
| ファイル | 役割 |
|---|---|
| `tests/test_leader_*.py`（13本） | 全リーダーの挙動テスト（既存の回帰アンカー）。方針は [`_TEST_GUIDE.md`](leader_specs/_TEST_GUIDE.md) |
| `tests/harness/leader_test_helpers.py` | リーダー挙動テスト用ヘルパ（盤面構築・対話駆動・観測） |
| `tests/harness/engine_helpers.py` | 最小 GameManager 構築ヘルパ（`make_game`/`make_instance`/`make_master`/`action`） |

---

## 3. 診断・監査ツール（pytest 外）

| ツール | 役割 |
|---|---|
| `tests/scripts/compare_parsers.py` | レガシー vs V2 の全カード差分（退行検知） |
| `tests/harness/full_card_audit.py` | 全カード構造不変条件検証＋挙動ベースライン生成（`--regen` で更新） |
| `tests/harness/game_driver.py` | **共通対局ドライバ**（設計⑥ `docs/refactoring_harness_driver.md`）: 統一対局ループ `run_game`（決定論契約＝global random の消費順保存・`first_player` 再現）＋席生成 `make_seat`（random/ai/arena/**learned**・engine 注入で net-vs-net）＋観測専用 observer。全 CPU 検証ハーネスの土台（新計器の追加＝observer 1 個） |
| `tests/harness/replay_runner.py` | **実対局リプレイヤ**（`docs/replay_verification_plan.md` R1-R3）: 記録記述子（seed＋leaders＋decks＋人間アクション列）から対局を再構築・再生。人間手＝決定論タイブレーク逆引き（`resolve_recorded_action`）・CPU＝再 decide・分岐は `reproduced`/`misses` に記録（サイレント誤再生なし）。API `/replay` 記述子を直接食える |
| `tests/harness/cpu_selfplay.py` | 決定論的 CPU 対 CPU 自己対戦（効果検証ハーネス）。詳細は §3.1 |
| `tests/harness/cpu_arena.py` | **CPU 検証基盤の絶対強度メトリクス**（SPEC §2.5.3／強さ=Elo 優先は §2.5.8）: `arena`＝固定参照相手への挑戦者勝率→**凍結ベースライン Elo**（席交互）／`regret`＝greedy regret 集計／**`arena-paired`＝分散低減（antithetic 席ペアリング＋Wilson 区間）で per-decider に情報方針(`--challenger-policy fair/cheat`)・PIMC(`--challenger-pimc K`)・学習ブレンド(`--challenger-blend α`)・予算按分(`--challenger-budget`) を A/B**。実ゲームは低速なので本走は手動/定期実行 |
| `tests/harness/phase1_sweep.py` | **Phase 1 切り分け実験**（SPEC §2.5.8）: 探索ノブ env（`OPCG_HARD_HORIZON` 等）を設定ごとに別プロセスで `arena-paired`（fair vs cheat）起動し horizon 掃引＋**同一 seed ペア差の符号検定**で「深さが効くか（探索 vs 情報の限界）」を判定。純関数テスト＝`tests/test_phase1_sweep.py` |
| `tests/harness/cpu_replay.py` | **CPU 思考トレース＋決定論リプレイ**（CPU 挙動改善用）。1 局を seed で再生し、各意思決定の「選んだ手・上位候補スコア（1-ply prelim／深掘り deep）・regret・J値成分内訳・読み筋（貪欲 PV）」をローカル JSONL へ出力する（GCS 不要）。詳細は §3.2 |
| `tests/harness/expected_effects.py` | 各カード×能力の「期待する動き」を AST から機械生成（`--regen`→`expected_effects.json`、`--card ID`）。効果オラクルの期待マニフェスト |
| `tests/harness/effect_oracle.py` | 期待 vs テキスト/AST の静的整合性コンパレータ（既存ゲートが拾わない高シグナル候補のみ抽出。`--category`/`--json`） |
| `tests/harness/structural_invariants.py` | 構造不変条件4スキャン（H先頭ゲート漏れ／Duration write-off／chooser欠落／「すべて」count退化）の一括検出（`--show`）。カテゴリH 横展開の回帰ツール化 |
| `tests/harness/false_path_coverage.py` | 条件を偽にして発動し、ゲートされた効果が走らない（盤面変化ゼロ）かを動的検証（`--show`/`--card`） |
| `tests/scripts/arena_parallel.py` | **並列アリーナ**（対照ペア×コア並列・旧 depth/thinktime_arena を統合）: 挑戦者の探索深さ/予算/PIMC/L1係数/**難易度（--challenger-difficulty learned＝既定Gen＝現Gen3）/sims** を席別に振って Elo A/B。SPSA の f(θ) 評価にも使う |
| `tests/scripts/perf_gate.py` | **CPU 性能ゲート**（§5.1）: learned(既定Gen＝現Gen3) vs 凍結 hard(L1) の Elo＋ペア単位 CI・1手レイテンシ・失敗局0・npz ハッシュを1コマンドで PASS/FAIL（`--quick`/`--full`） |
| `tests/scripts/replay_ambiguity_probe.py` | **実対局リプレイの曖昧性計測**（R0）: 記録アクション（card_id 基準）の一意復元可否を実デッキで実測（`--real-decks`・アクション種別ごとの曖昧率）。報告は `docs/reports/cpu_replay_ambiguity_r0_20260704.md` |
| `tests/scripts/sample_audit.py` | 各弾から決定的ランダム抽出＋自動スクリーニング＋精査素材出力（§8.4 ✓信頼度の実測。`--per-set`/`--seed`/`--dump`）。報告は `docs/reports/sample_audit_*.md` |
| `tests/scripts/leader_spec_probe.py` | リーダー1枚のテキスト/AST要約/実行観測の出力（`<ID>`/`--set`/`--all`/`--json`）。手動検証（§8）の補助に使う |
| `tests/scripts/card_spec_probe.py` | 上記を非リーダー含む全カードに拡張し**弾×色**で絞る（`--set OP16 --color 赤`/`--buckets`/`--type`/`--json`）。デッキを跨いで弾×色バケット単位に検証する起点（§8） |
| `tests/scripts/rl_purepy_probe.py` | **PyPy自己対戦ワーカー投資可否の判定プローブ**: MCTSホットループ（value/policy forward・PUCT選択）を numpy版と純Python版で同型実装し正しさ照合＋CPython計時。1手あたり合成コストの py/np 比から「numpy剥がし＋PyPy で現行を逆転できるか」を数字で判定（`--sims`/`--legal`/`--depth`）。結論=NN行列積は numpy/BLAS が純Pythonを71〜592×圧倒しPyPyでは届かない＝④見送りの根拠 |

### 3.1 効果検証ハーネス（CPU 対 CPU 自己対戦）

`tests/harness/cpu_selfplay.py` は「遊ぶ機能」と同じ AI（`core/cpu_ai.py`）を流用した**決定論的・自動異常検出
付きの効果検証ツール**。弱い AI でも長時間の自己対戦で効果を踏めるため、検証品質と AI の強さは分離
できる。長時間対戦で効果を踏ませ、実行時の破綻を fail-fast で炙り出す。

- **決定論・再現性**: 全乱数を seed 付き RNG に集約（`--seed N` で完全再現）。適用した
  `(player, action_type, payload)` を順序記録し、同 seed＋同手順で 1 ステップ単位に再現する。
  バグ報告 ＝「seed＋手順＋停止ステップ」で完結する。
- **方策・実行**: `--policy random|ai` / `--difficulty easy|normal|hard` / `--games K` /
  `--p1-leader`/`--p2-leader`（リーダー指定）。特定カードを強制投入して効果を踏ませる用途にも使う。
- **トレース**: `--out trace.jsonl`（1 行＝1 ステップ：step/turn/phase/player/action/events/
  snapshot_diff/flags）。`grep`/`diff` で異常箇所へ直行できる。`--verbose` で 1 手ずつ表示。
- **実行時インバリアント**（`core/invariants.py` の `check_invariants`/`check_turn_boundary`）: 各
  ステップ後に検査し、破れたら**即停止＋リプロ出力**（fail-fast）。
  - `SUSPEND_LEAK`（手番をまたいで未解決の `active_interaction` / `pending_request` / temp_zone が残る）
  - `HIDDEN_LEAK`（隠しゾーンの中身露出）
  - `FIELD_LIMIT`（場のキャラ ≤ 5）・DON 総数保存・パワー非負
  - UUID ユニーク・ゾーン間の重複 / 消失なし・ライフ枚数とゾーンの整合
  - `STUCK`（合法手が空＝詰み / スタック）・無限ループ（同状態反復・ステップ上限）

これにより「効果が静かに失敗する（`OTHER` 化・no-op）」「中断が解決されない」を**進行中から**自動
検出する（AI の自動解決が本番のバグを覆い隠さないよう、本番の中断は握り潰さず必ずここで表面化する）。
`tests/test_cpu_selfplay.py` がスモーク（完走・決定論・`clone` 非破壊・合法手の `_validate_action` 適合・
インバリアント検出）を回帰として固定する。

### 3.2 CPU 思考トレース＋決定論リプレイ（挙動改善用・Phase 1）

`tests/harness/cpu_replay.py` は §3.1 と同じ決定論エンジン（全乱数を global random に集約・`action_api` で本番
同一コアパス）の上に、**CPU の意思決定の中身**を 1 局ぶん 1 ファイルへローカル出力する。GCS（本番
テレメトリ）に撮りに行かずに、手元で `grep`/`diff` して「なぜその手か」を読める。

- **思考トレース（4 項目）**: 各意思決定（`type:"decision"` 行）に以下を記録する。
  - `chosen`／`folded`: 選んだ手（card_id 基準）とターンを畳んだか。
  - `candidates`: 上位候補（`prelim`＝1-ply 事前スコア／`deep`＝深掘りスコア。easy は prelim のみ）。
  - `regret`: deep 最善 − 1-ply 貪欲手の deep 値（`decide_with_regret` と同義の崖エラー代理）。
  - `j_components`: 選んだ手の結果盤面の **L1 評価成分内訳**（`evaluate(out=…)` が L1 評価の内訳を `out["v2"]`
    キーに格納したもの＝カード通貨ベースの内訳＋`total`）。<!-- 旧 `_side_score` 由来の me/opp 別ライフ/手札/場…成分は 2026-06-27 の CPU 評価 L1 単一系統化で撤去。`plan_progress`/`telegraph` 成分は同日 plan 全廃で削除 -->
  - `read_ahead`: 読み筋（各手番で 1-ply 最善を辿った貪欲 PV。`max_steps` で有界。`REPEAT_CAP` は 2026-06-27 撤去）。
- **手記述は card_id 基準**（uuid は実行ごとに変わるため）＝同一 seed で安定再現・比較できる。
- **トレースは観測専用**: `decide`/`decide_guarded` の `trace` 引数（既定 None＝**完全に無
  オーバーヘッド・挙動不変**）で採取する。トレース構築の追加クローンは getstate/setstate で
  RNG 中立化し、**トレース有無でゲーム進行が分岐しない**（評価関数の `evaluate(out=...)`／L1 評価
  `cpu_eval_v2.evaluate_v2(out=...)` も `out=None` 時は採点を一切変えない＝ベースライン不変。`_side_score(out=...)`
  は手書き J値評価の撤去〔2026-06-27〕で消滅）。
- **リプレイ種**: `--record seed.json` で `{seed, リーダー, 難易度}` の極小記述子を残し、
  `--descriptor seed.json` で完全再現する。
- **learned（Gen2・本番既定 CPU）のトレース**: `--difficulty learned`（席別 `--p1-/--p2-difficulty learned`）で
  Gen2 学習型（`game_driver.make_seat(kind="learned")`）を再生する。思考トレースは L1 の 4 項目に代わり
  **MCTS root 統計**を記録する（`candidates`＝訪問%＋行動価値Q／`value`＝採用手のQ／`l1_move`・`l1_disagrees`
  ＝独立評価器 L1 の第二意見）。learned の numpy rng は global random 由来（PR-D2）なので **seed から
  決定論再生できる**（本番既定 CPU の「なぜその手か」を手元で読める）。`tests/test_game_driver.py` が
  learned 自己対戦の決定論を、`tests/test_cpu_learned.py` が単発意思決定の seed 再現を固定する。

#### 実アプリ対局の取得（Phase 2・`opcg_sim/api/app.py`）

実アプリの CPU 対戦も、**GCS（本番テレメトリ）に撮りに行かずに**思考トレース＋リプレイ種を残せる。

- **opt-in**: `POST /api/game/create` に `cpu_trace=true`（任意で `seed`）を渡した対局のみ記録する。
  未指定の本番対局は seed も触らず追加処理ゼロ＝**従来挙動を完全維持**（トレースは観測専用）。
- **記録内容**: create 時に seed を固定（コイントス＋シャッフルを再現可能化）し、
  人間/CPU 双方のアクションを card_id 基準で、CPU の各意思決定の思考トレース（4 項目）をメモリに蓄積する。
- **取得**: `GET /api/game/{game_id}/replay` が `{replay: 種(schema/seed/leaders/decks/difficulty/actions),
  decisions: 思考トレース列}` を返す。対局はメモリ常駐（Cloud Run は揮発）なので、対局中〜終了直後に
  取得して保存/共有する想定。崩れた局面はそのまま `test_cpu_puzzles.py` の決定論ケースへ落とせる。
- **盤面フレーム**（リプレイビューア用）: traced 対局は各アクション適用後の盤面スナップショット
  （コンパクト形＝動的状態のみ）も蓄積し、`GET /api/game/{game_id}/replay/frames` が
  `/replay` の内容＋`frames`（`action_index` で actions/decisions と整合）を一括で返す。
  詳細は [`LOGGING.md`](LOGGING.md)「盤面フレーム」、回帰は `tests/test_replay_frames.py`。
- **ライブは軽量トレース**（`trace_read_ahead=False`）: 最も重い `read_ahead`（読み筋＝各手番で全合法手を
  クローンする貪欲 PV）を**省く**＝CPU 思考のレイテンシをトレース無しとほぼ同等に保つ（実測: light≒none、
  full は約 +50%）。候補スコア・regret・J値成分は探索の回収＋クローン1回で安価なので残す。**読み筋は
  オフライン（`cpu_replay.py`／リプレイ種の再生）でのみ**採る。
- **実行例**:
  ```bash
  OPCG_LOG_SILENT=1 python tests/harness/cpu_replay.py --seed 7 --difficulty hard --out /tmp/replay.jsonl
  OPCG_LOG_SILENT=1 python tests/harness/cpu_replay.py --seed 7 --difficulty hard --record /tmp/seed.json
  OPCG_LOG_SILENT=1 python tests/harness/cpu_replay.py --descriptor /tmp/seed.json --decisions-only --out -
  ```

`tests/test_cpu_replay.py` が回帰（trace の挙動不変・RNG 中立・決定論再現・トレース 4 項目の存在）を固定する。

> 注: 汎用ログ（`log_event`／`logger_config.py`／`/api/log`／GCS/Slack 転送）は撤去済み。ログの扱いの
> 正本は [`LOGGING.md`](LOGGING.md)。本番は Cloud Run の素の stdout 以外に明示的なアプリログを出さない。

---

## 4. 変更・回帰検証フロー

```bash
# 1) ルール追加（opcg_sim/src/core/effects/rules/atoms.py に @rule）
#    エンジン実行が要るなら gamestate/resolver も実装し test_effects_engine に検証追加
#    コアルール（ターン/戦闘等）の変更は gamestate.py を直接修正し test_rules_* に検証追加

# 2) 回帰・退行（構造不変条件チェック込み。コマンドの正本は Makefile）
make test
OPCG_LOG_SILENT=1 python tests/scripts/compare_parsers.py        # レガシー比の新規OTHER（退行）

# 3) 挙動を意図的に変えた場合のみベースライン更新
make regen-baseline
```

`@rule(name, priority)` で関数登録（priority 大ほど先に試行、不一致は `None`、一致は `EffectNode`）。

---

## 5. 品質ゲート

| ツール | 合格条件 |
|---|---|
| `tests/harness/full_card_audit.py` | EXCEPTION / CARD_LOSS / TEMP_LEAK = 0 |
| `tests/test_full_card_baseline.py` | `full_card_baseline.json` と一致 |
| `tests/scripts/compare_parsers.py` | 新規 OTHER（退行）= 0 |
| `tests/test_effect_oracle_gate.py` | 静的 text↔AST 整合性 HAS_OTHER / PER_TURN_LIMIT_GAP / UP_TO_GAP = 0（**ラチェット**） |
| `tests/test_verified_decks.py` | 検証済みデッキの効果回帰 = 全合格（**ラチェット**: 検証済みの挙動は減らさない） |
| `tests/test_structural_gate.py` | 構造不変条件4スキャン（H先頭ゲート漏れ／Duration write-off／chooser欠落／「すべて」count退化）= 0 ＋ 条件偽パスで盤面変化ゼロ（**ラチェット**。カテゴリH 再発防止） |
| `tests/test_verified_buckets.py` | §8.2 台帳「✓」弾×色がベースライン全数登録・H違反0（ドキュメント主張の機械保証） |

挙動を変更したら差分をレビューのうえ `full_card_audit.py --regen` でベースライン更新し、上記ゲートを通す。
**検証済みデッキ（§8.2 台帳）の挙動を直したら `tests/test_verified_decks.py` にアサートを追記**し、
以後それを割らないことをマージ条件とする（カバレッジは単調増加）。

### 5.1 CPU 性能ゲート（Gen2 非退行・手動/定期）

本番既定 CPU＝**Gen2(learned)** の強度・非退行を測る運用ツール（実対局は重いので `make test` 外・手動/定期）:

```bash
OPCG_LOG_SILENT=1 python tests/scripts/perf_gate.py --quick     # 疎通/軽い確認（pairs6・sims40）
OPCG_LOG_SILENT=1 python tests/scripts/perf_gate.py --full      # 本走（pairs40・sims160）
```

- 測るもの: learned(Gen2) vs 凍結ベースライン **hard(L1)**（決定論・不変の物差し）を対照ペア並列で戦わせ
  勝率→Elo＋ペア単位 CI／learned の 1 手レイテンシ（1手1秒予算）／失敗局=0／gen2_*.npz ハッシュ記録。
- 合格: `evaluate_gate`（純関数・`tests/test_perf_gate.py` が固定）が Elo 下限>閾値 ∧ median<予算 ∧ 失敗局0 で PASS。
- net 更新（新 Gen）の**昇格判定**は net-vs-net（`cpu_learned.LearnedEngine` で2ネット同居・`play_game(pX_engine=…)`）。
  詳細と運用ルール（凍結＝出荷 Gen2 の npz ハッシュ・昇格条件 elo_lo>0／非退行 elo_hi>−15）は
  [`cpu_perf_testing_plan.md`](cpu_perf_testing_plan.md)。強度 A/B は `tests/harness/cpu_arena.py arena-paired --challenger learned` /
  並列 `tests/scripts/arena_parallel.py --challenger-difficulty learned`。

---

## 6. 直近の変更で追加されたテスト（参考）

- **API 層スモーク**: `tests/test_api.py`（18件）。FastAPI の HTTP/WS 契約（対局生成→state→マリガン→TURN_END／CPU step／sandbox WS ブロードキャスト／rule ルーム→START／未知 ID・DB 未初期化のエラー応答・`X-Session-ID` 往復）を `TestClient` で検証。`fastapi`/`httpx` 導入により collection 可能になった層。
- **オンライン対戦**: `tests/test_rule_online.py`（2件）。ルーム生成→WS購読→SET_DECK→START→`/api/game/action` のブロードキャスト同期、開始の両者 ready ガードを検証。
- **コアルール修正**: `tests/test_rules_summoning_field_limit.py`（9件）。召喚酔い/速攻、場5体上限（強制トラッシュ＝`FIELD_OVERFLOW_TRASH`）を検証。
- これらの修正に伴い `full_card_baseline.json` を更新（`OP06-086`: ON_PLAY で場が6体になる挙動が5体上限により `INTERACTIVE`＝選択待ちへ変化）。

---

## 7. 既知の挙動差異
記載先は対象の種別で分ける:

- **リーダー効果**のテキスト準拠期待と現挙動の差異 → [`docs/leader_specs/ISSUES.md`](leader_specs/ISSUES.md) に集約
  （各項目は対応する `tests/test_leader_*.py` の xfail で固定）。差異が解消されればマーカーを外して通常テスト化する。
- **エンジンのモデル化制約**（「お互い」の同時両側処理・置換のネスト中断 等） → [`docs/SPEC.md`](SPEC.md) §6.1。
- **パーサの構造分解・未対応表現** → [`docs/parser_v2.md`](parser_v2.md)「既知のパース制約」。未対応原子句は
  `test_parser_fallback_ratchet`（上限0）で監視する。
- **非リーダーカードの個別挙動**でバグ確定・修正したもの → §8 の手動検証フローに従い `tests/test_verified_decks.py` に回帰アサートを追加。

---

## 8. 効果の正しさ検証（デッキ単位の手動方式）

効果の意味的正しさ（テキスト準拠で正しく発動するか）は、自動オラクル／監査では検出
しきれない細部が多い。そこで**実際に組んだデッキを起点に、カードを1枚ずつ実装と
突合する手動方式**を採用する。

手順:

1. フロントの**デッキビルダーからデッキを「検証向け Markdown」でエクスポート**する
   （リーダー＋各カードを「枚数 番号 名前 / 効果テキスト / トリガー」で列挙）。
2. 各カードについて、効果テキストを実装（`parser.py` / `resolver.py` /
   `rules/atoms.py` / `matcher.py` / `gamestate.py`）の挙動と突合する。
   AST のダンプだけで判断せず、**実機（実効パワー・条件評価・対象選択・盤面差分）
   まで確認**する（§1 の `RULE_PROCESSING` 死角に注意）。
3. バグ確定なら修正し、可能なら同型テンプレートのカードへ横展開する。挙動を変えた
   場合は §4・§5 の回帰フロー（ベースライン再生成・退行ゼロ・構造ゲート）を通す。
4. リーダーの常在「ルール」効果（ドン!!デッキ枚数等）は per-ability 差分に現れない
   ため、**ゲーム不変条件として個別テストを足す**。

検証で固定した挙動は `tests/test_verified_decks.py` に1ケースずつ集約する（ベースライン
が見られない常在ルール・ON_LEAVE・勝利条件・別名・持続時間等の意味的回帰ガード）。
新しいデッキを検証して挙動を直したら、同ファイルに対応するアサートを追記すること。

### 8.1 品質管理の考え方
全 2652 枚ではなく**実際にプレイされるカード（組まれたデッキ）**を対象に、検証カバレッジで
品質を管理する。検証済みは回帰テストで固定し、**二度と落とさない**（ラチェット）。新しい
デッキを検証するたびに台帳（§8.2）へ1行、回帰テストへ数ケースを足し、保証を積み上げる。

### 8.2 検証進捗台帳
手動検証したデッキを記録する。新規デッキを見たら1行追記（単調増加）。

| デッキ | リーダー | 検証日 | 発見/修正バグ | 回帰テスト |
|---|---|---|---|---|
| 新エネル（除去コン） | OP15-058 | 2026-06-13 | 6（場を離れず+2000欠落 / サンジ‖イベント / 雷龍レスト / 神避付与ドン / エンドフェイズ持続 / ドンデッキ=6 未適用） | ✓ |
| ロシナンテ | OP12-061 | 2026-06-13 | 2（カード名別名未適用 / お互いライフ合計） | ✓ |
| バギー（インペルダウン） | OP16-041 | 2026-06-13 | 1（ON_LEAVE トリガー未実装） | ✓ |
| 赤紫ルフィ | ST10-002 | 2026-06-13 | 4（得て+パワーの付与欠落 / パワー厳密一致 / 複数リーダー名OR / ロジャー誤自動勝利） | ✓ |
| 青緑ルフィ（インペルダウン） | OP16-022 | 2026-06-13 | 2（レストにできない対象誤フィルタ / distinct-name スケール） | ✓ |
| ミホーク（緑レスト） | OP14-020 | 2026-06-13 | 3（属性‖種類の跨ぎOR=ペローナ / on-restトリガー未実装 / キャラ‖ドン合計枚数） | ✓ |
| OP11 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 0（新規の系統的バグなし。OP16〜OP12 の横断修正＝leader特徴OR・_ko_trigger_matches・TRAIT_OR_NAME・AND分割・FIELD_COUNTのcostフィルタ・LIFE_COUNT_COMPARE 等で全てカバー）。残: OP11-001「速攻:キャラ」が「速攻」緩和／OP11-050 戻し先「手札かデッキ下」のゾーン解釈／OP11-110「ステージかリーダー」レスト混在 は各1枚・未対応 | ✓ |
| OP10 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP10-119 ロー / ST13-005 イワンコフ「手札から…公開し、ライフの上に裏向きで加える」が hand_to_life の正規表現で「表向きで」しか許容せず、reveal_hand に落ちて REVEAL のみ＝手札→ライフ移動が脱落。`[表裏]向きで` に拡張）。赤/緑/青/紫/黒は新規バグなし（REST費用・DON_COUNT_COMPARE相互比較・FIELD_COST_SUM・LIFE_COUNT_COMPARE・dual-tier除去 等は横断修正でカバー済） | ✓ |
| OP09 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 5（OP09-017 リーダー「パワーN以上でかつ特徴《X》」AND の片側脱落＝でかつ無読点 / OP09-036 「キャラ1枚かドン‼1枚をレスト」の択一でキャラ側脱落＝rest_char_or_don が枚数を挟む形に不一致 / OP09-097 カウンター「効果を無効にし、パワー-4000」で negate 脱落＝negate_then_buff 追加 / OP09-084 「【A】か【B】か【C】を得る」キーワード3択の2番目以降脱落＝grant_keyword_choice 追加 / OP09-101・EB01-053・OP06-103 「場のキャラを…ライフの上か下に表向きで置く」が field_char_to_life の「加える」限定で FACE_UP_LIFE に誤落＝「置く」も移動として許容）。残（各1枚・未対応）: OP09-005/024/092 等の「…場合、引き／捨てる」条件分岐の後続アクション脱落（=OP15-104 と同型の節分割問題）／OP09-092 「手札が相手より3枚以上少ない」相対比較／OP09-098 「そのキャラのコスト4以下ならKO」の参照・対象退化 | ✓ |
| OP05 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（「このキャラ以外の自分の…（パワーN以上／名前）キャラがいる場合」が「このキャラ」を含むため SOURCE_STATE（自身の状態）に誤分類＝他キャラ存在条件(FIELD_COUNT, 自身除外)であるべきが自身条件に化けていた。FIELD_COUNT 分岐で「このキャラ以外」を許容＋SOURCE_STATE 分岐から除外。OP05-003 イナズマ／OP04-005 クンフージュゴン に波及）。残（各1枚・未対応）: OP05-007「パワー合計が4000以下になるようにKO」選択制約／OP05-040「すべて」側未指定スコープ／OP05-058「手札が5枚になるように捨てる」可変枚数／OP05-002「特徴か【トリガー】」対象OR／OP05-100 自己効果無効。赤/緑/青/紫/黒 の DON条件・search・dual_tier・cost範囲・field_char_to_life・REPLACE_EFFECT 等は健全 | ✓ |
| **個別: 残カテゴリG（取りこぼし整理）** | — | 2026-06-14 | 修正2（OP06-117/OP05-089「このカード（キャラ）と〈X〉をレストにできる」の自身レスト脱落＝複合レストを Sequence[REST(自身), REST(X)] に／OP09-118「自分か相手のライフが0枚」を OR(自分0,相手0) に）。確認: OP05-002「特徴か【トリガー】」(TRAIT_OR_TRIGGER 既対応)・OP05-058「手札が5枚になるように捨てる」(DOWN_TO_N 既対応) は健全。**未対応（継続効果/タイミング/特殊条件のアーキ拡張が必要・各1枚）**: OP08-043 アタック税（アタック時に手札2枚を捨てねば不可）／OP08-114 属性《斬》限定のバトルKO耐性／OP08-101 「このターン終了時」遅延ライフ追加／OP08-006 トラッシュに特定名2種がある条件／OP05-100 自己効果無効の置換。回帰 `test_g_compound_self_rest_and_life_or` | ✓ |
| **横断: 側未指定の「すべて」/KO スコープ** | — | 2026-06-14 | 1（**残カテゴリF を解消**。側の明示が無い「コストN以下のキャラ(すべて)をKO」が SELF 既定で自分のキャラだけ、「お互いの…アクティブにならない」FREEZE が OPPONENT 固定で相手だけ、になっていた。KO ルールは側未指定かつ対象キャラ絞りありなら ALL、FREEZE ルールは「お互い」/BOTH_SIDES なら ALL（それ以外は従来どおり OPPONENT 既定）。OP05-040（鳥カゴ）・OP06-081・ST08-005・ST27-005 を是正。素の「KOする」(そのキャラ/選んだキャラ参照系) は対象外。回帰 `test_side_unspecified_removal_is_all`） | ✓ |
| **横断: 「パワー合計N以下になるようにKO」** | — | 2026-06-14 | 1（**残カテゴリE を解消**。「相手のキャラ2枚までを、パワーの合計が4000以下になるようにKO」で合計上限の選択制約が脱落し合計超過でもKOできていた。`TargetQuery.power_sum_max` を新設・matcher で解析、resolver が合計≤N の有効な選択に限定（低パワー順に貪欲＝ルール違反を起こさず最大枚数を確保）。OP05-007・OP09-018 を是正。回帰 `test_power_sum_max_ko_constraint`） | ✓ |
| **横断: 「リーダーとキャラを選ぶ」SELECT** | — | 2026-06-14 | 1（**残カテゴリD を解消**。「（相手/自分の）リーダーとキャラN枚(まで)を選ぶ」で SELECT がリーダーを含まず1枚しか選べず、後続の「選んだカード」効果が片側/不発になっていた。SELECT を CHARACTER 選択＋`INCLUDE_LEADER` フラグとし、resolver `_with_leader` が対象側リーダーを選択群へ常に含める。OP07-059（リーダー＋キャラを凍結）・OP14-009（リーダー↔キャラのパワー入替）を是正。回帰 `test_select_leader_and_char_includes_leader`） | ✓ |
| **横断: オフセット相対比較** | — | 2026-06-14 | 1（**残カテゴリC を解消**。「自分の〈手札/場のドン/キャラ〉が相手より N枚以上少ない/多い場合」が、オフセット「N枚以上」の『以上』を方向と誤認して GE に化けたり、手札比較が型不在で HAND_COUNT(相手) に退化していた。比較演算子＋オフセット抽出 `_compare_op_offset` を追加し、resolver は相手枚数±N をしきい値に評価（`_offset_threshold`）。HAND_COUNT_COMPARE を新設。OP09-092（手札-3）・OP07-064/OP06-072（ドン-2）・OP10-098（キャラ-2）を是正。回帰 `test_offset_relative_count_compare`） | ✓ |
| **横断: 公開/トラッシュ済みカードの条件** | — | 2026-06-14 | 1（**残カテゴリB を解消**。「公開したカードが〈特徴/コスト/パワー/種別〉の場合」「置いたカードが〈コスト〉の場合」が GENERIC（常時真）に退化し、公開/トラッシュしたカードの内容を問わず発動していた。REVEALED_CARD_TRAIT 検出を「公開したカード」「置いたカード」へ拡張＋パワー/種別(カード語尾なし)条件を追加、resolver にパワー判定と TRASH_FROM_DECK 後の last_revealed_card 記録を追加。OP08-049/096・EB01-029・OP01-063・OP04-011・OP15-065 を是正。回帰 `test_revealed_placed_card_condition_not_generic`） | ✓ |
| **横断: 節分割（条件ゲートのスコープ）** | — | 2026-06-14 | 1（**残カテゴリA を解消**。「〈条件〉場合、AしてB」が文内連用接続（引き、捨て、…し 等）で区切られる際、後続アクション B が条件分岐の外へ出て**条件不成立でも実行**されていた＝`_parse_to_node` の分割で条件ゲートを含む文を一体化（「。」「その後、」等の手順境界は従来どおり分割）。ゲートは能力条件へ lift／Branch 化され本体全体を覆う。OP09-005/024・OP08-082/086・OP15-104・OP01-002・OP03-069・OP16-087/103/106/109・EB01-020/EB04-031 ほか計36能力の挙動を是正（ベースライン regen 済）。回帰 `test_conditional_clause_gates_all_trailing_actions`） | ✓ |
| OP06 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（対象指定「パワーNからM」(N以上M以下)が単一しきい値判定に落ち「パワーN」だけ拾って power_min=power_max=N に縮退＝上限Mが脱落。matcher にパワー範囲判定を追加＝OP06-015 リリーカーネーション／EB02-039／PRB02-010 の「パワー2000から5000」等に波及）。残（各1枚・未対応）: OP06-044「相手がイベント発動時」のイベント種別が手札捨て対象に漏れ／OP06-081 側未指定KOが SELF 既定／OP06-117「このカードとエネルをレスト」コストの自身レスト脱落／OP06-082 等の節分割。赤/緑/青/黒/黄の REST費用・DON返却・search・dual_tier・rest_char_or_don・field_char_to_life 等は健全 | ✓ |
| OP07 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（「（相手/自分の）リーダーとキャラN枚(ずつ)(まで)を、…パワー±N／効果を無効」が「と」(両方)を「か」(択一)と同一視して単一 count=1 対象に潰し、リーダー＋キャラの双方へ掛かるべき効果が片方しか掛からなかった＝leader_and_char_dual 追加で BUFF/NEGATE を Sequence 分割。OP07-075／OP10-098 に波及。ドン付与(OP13-042)・選ぶ(OP07-059/OP14-009)は別構造で対象外）。残（各1枚・未対応）: OP07-064「ドンが相手より2枚以上少ない」オフセット相対比較／OP07-059・OP14-009 「リーダーとキャラを選ぶ」SELECT 構造。赤/緑/青/紫/黄の REST費用・DON返却・search・dual_tier除去・REVEALED_CARD_TRAIT 等は健全 | ✓ |
| OP08 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 0（新規の独立バグなし。発見した不備は既知の残カテゴリに該当＝(a) 節分割（「…場合、AしてB」の後続Bが分岐外）: OP08-079/086/097 ほか、OP15-104/OP09-005 と同型／(b) 公開・トラッシュしたカードの条件が GENERIC に退化: OP08-049「公開カードが白ひげ」・OP08-096「置いたカードがコスト6以上」／(c) アタック税・属性限定KO耐性など複合継続効果: OP08-043「アタックする際手札2枚を捨てねば不可」・OP08-114「属性《斬》とのバトルでKOされず」／(d) OP08-006 トラッシュに特定名2種がある条件。**(b)(c)(d) は是正済**（OP08-043 ATTACK_TAX_DISCARD・OP08-114 属性限定保護・OP08-006 HAS_CHARACTER(zone=TRASH)・OP08-101 遅延ライフ・OP05-100 自己無効置換。回帰 test_op08_*/test_op05_100_*）。(a) 節分割は別カテゴリで対応）。赤/緑/青/紫/黒/黄の REST費用・DON返却・FREEZE・COST増減・search・dynamic cost 等は健全 | ✓ |
| OP12 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP12-006/014「「モンキー・Ｄ・ルフィ」か赤のイベント」が 名前∧色∧種類 AND に縮退＝NAME_OR_COLORTYPE 追加で 名前OR(色∧種類) に。3枚）。残: OP12-073「名前と特徴を持つキャラすべて」の和集合／OP12-096 条件付き対象コスト上限アップグレード／OP12-081 leader の登場時トリガー条件 は未対応 | ✓ |
| OP13 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（dual-tier 除去「<f1>のキャラ1枚と<f2>のキャラ/ステージ1枚を、KO/手札に戻す/デッキの下」が単一化し第2対象脱落＝OP13-077/OP07-017/OP07-118/OP03-018/OP04-044/OP06-056/OP05-093/OP10-098/EB03-021 等11枚）。残: OP13-025「特徴か属性」/OP13-051「名前か多色」のリーダー条件OR各1枚、OP13-064 全体効果無効の対象範囲 は未対応 | ✓ |
| OP14 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 4（リーダー特徴の複数OR《X》か《Y》＝12枚 / 「リーダーが「X」で、…」AND分割の名前条件脱落＝6枚 / 「相手の場のドンがN枚以上」が相互比較に誤吸収＝5枚＋複合「多色で」分割 / 「コスト0か8以上のキャラ」condが cost0 のみに縮退＝B・W 5枚）。残: OP14-084「コスト4以下と1の1枚ずつ」dual-tier 登場が片方のみ／OP14-041 ハンコック leader の自軍キャラKO監視は別アーキ未対応 | ✓ |
| OP15 全色（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 6（OP15-073/101 name-or-trait語順AND化 / OP15-018/015 付与ドンfilter脱落 / OP15-005 付与ドン存在条件の常時真化 / P-107 「自分か相手」OR / OP15-024 レスト耐性複合の脱落 / LIFE_COUNT_COMPARE 未対応＝「自分のライフが相手より少ない」12枚が相手ライフ0判定に退化）。残: OP15-104 の DISCARD が条件分岐外（条件不成立でも手札2枚捨て）／OP15-080 power10000フィルタ／OP15-092 20枚分岐の相手ターン文脈 は未対応 | ✓ |
| マルコ（白ひげ） | OP08-002 | 2026-06-14 | 3（手札のこのカード=コスト軽減が一切不発：条件HAND_COUNT誤判定＋手札PASSIVE未評価／元々のパワー指定が現在パワーで誤絞り＝ナミュール／「リーダーとキャラ…ずつ」付与が片側1体に縮退） | ✓ |
| ナミ（スリラーバーク） | OP11-041 | 2026-06-14 | 1（「（トラッシュから）…レストで登場させる」の「レスト」が対象 is_rest フィルタに誤漏れ＝蘇生候補を全除外し完全不発：OP14-102/110/111 ほか「レストで登場/加える/追加」109枚に波及） | ✓ |
| **OP16 × 赤**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-015 ルフィ「リーダーが『エース』を含むカード名で、ドン!!6枚以上」の AND がパーサで分割されず**リーダー名条件が脱落**＝ドン!!枚数だけで誤発動。さらに『』内をカード名でなく特徴扱いしていた二重退化） | ✓ |
| **OP16 × 緑**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-024 イナズマ「**相手の効果で**KOされた時」が要因を問わず全KOで発火＝戦闘KO・自分の効果KOでも誤誘発。書き下し形KO誘発の要因/ターン修飾を `_ko_trigger_matches` で尊重。OP09-052/OP11-024/OP11-035/OP11-051/EB01-057/ST15-003/OP03-015/OP02-085 へ横展開） | ✓ |
| **OP16 × 青**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-047 ドフラミンゴ「**相手は自身の**手札2枚を…デッキの下に置く」の対象選択者が既定(自分)のまま＝自分が相手の手札を選べる退行。`相手は自身の` を chooser=OPPONENT に。OP16-094/OP12-087/OP09-087/EB04-022/EB03-026/OP06-047/OP06-051/OP11-072/OP15-048 ほか「相手は自身の…捨てる/置く/戻す」系へ横展開） | ✓ |
| **OP16 × 紫**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-074 マゼラン「相手は自身の場のドン!!を戻す」が、RETURN_DON の対象選択 resume を応答者(相手)視点で再実行＝`_don_pool_player` が自分プールを指し空振り。`SELECT_RESOURCE` resume を発生源の持ち主＝効果責任者視点に修正。「相手は…ドン!!を戻す」系全般に波及） | ✓ |
| **OP16 × 黒**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-100 氷諸斬り「このターン中、相手のキャラがKOされている場合」が、「KOされて<いる>」の "いる" で FIELD_COUNT（相手の場キャラ存在）に誤吸収され逆の意味に化けていた。ターン内KOイベント記録＋専用条件 `CHAR_KOED_THIS_TURN` を追加して是正） | ✓ |
| **OP16 × 黄**（弾×色, §8 デッキ非依存） | — | 2026-06-14 | 1（OP16-102 アバロ・ピサロ「自分の**手札かトラッシュ**から登場」が、play_card_from_zone ルールの `has_trash` 上書きで zone=TRASH 単一に退化し手札からの登場が不可だった。「手札かトラッシュ／トラッシュか手札」の隣接並列を両ゾーンに。OP06-060/064/066/068・EB01-033・EB03-042・EB04-047・OP14-091・PRB02-018 ほか13枚に波及。併せて parse_target の「手札から…場」誤マルチゾーン検出を本ルールが上書きで吸収） | ✓ |
| **EB01〜EB04 全色**（EX ブースター, §8 デッキ非依存） | — | 2026-06-15 | 0（独立新規バグなし・走査245枚。dual-tier 除去 EB03-021／leader+char の BUFF・ATTACH_DON EB02-007・EB03-026・EB03-037／name-or-type EB04-029 はいずれも既存横断修正でカバー、WARN 群は self-target ACTIVE/GRANT/RAMP/PLAY に対する分類器の方向ヒューリスティック誤検知で健全。**全弾横断の系統的バグ「カテゴリH」**（先頭条件が「。その後、」をまたぐ漏れ）の検出箇所: EB02-028/032・EB03-003/013/017/039/051/052・EB04-030/036 ほか22能力（いずれも `_lift_h_gate` で是正済）） | ✓（H是正済） |
| **PRB01・PRB02 全色**（プレミアムブースター, §8 デッキ非依存） | — | 2026-06-15 | 0（独立新規バグなし・走査19枚。PRB02-010/013 がカテゴリH に該当・他は健全） | ✓（H是正済） |
| **P（プロモ）全色**（§8 デッキ非依存） | — | 2026-06-15 | 0（独立新規バグなし・走査120枚。WARN 3枚 P-029/091/099 は self-target ACTIVE/GRANT の誤検知で健全。カテゴリH 該当: P-059/112） | ✓（H是正済） |
| **ST01〜ST09 全色**（スターターデッキ, §8 デッキ非依存） | — | 2026-06-15 | 0（系統的な新規バグなし・走査149枚。カテゴリH 該当0。WARN 群は self-target ACTIVE/KO/GRANT/BUFF の分類器誤検知で健全。ST02-014 X・ドレーク「特徴《超新星》か《海軍》を持つリーダーとキャラ**の**（無印）パワー+1000」の数量詞なし count=1/CHOOSE 退化は **是正済**（`leader_and_char_dual` に所有格形を追加し ALL 適用・特徴フィルタ保持。回帰 `test_st02_014_*`）。「すべて」明記の OP02-120 等は正しく ALL。§8.3「対象フィルタ/count退化」型・低優先） | ✓ |
| **OP01〜OP04 全色**（OP ブースター, §8 デッキ非依存） | — | 2026-06-15 | 0（独立新規バグなし・走査約480枚。dual-tier/leader+char合計/お互いライフ合計(LIFE_COUNT_BOTH)/FREEZE合計まで＝OP02-120・OP04-031・OP04-112・OP04-116 等はいずれも既存横断修正でカバー、WARN は self-target 誤検知。**全弾横断の系統的バグ「カテゴリH」の検出箇所**＝15能力: OP01:1/OP02:1/OP03:4/OP04:9（いずれも `_lift_h_gate` で是正済）） | ✓（H是正済） |
| **ST10〜ST30 全色**（スターターデッキ, §8 デッキ非依存） | — | 2026-06-15 | 0（走査203枚・系統的な新規バグなし。ST はほぼ既存 OP カードのリプリントで、横断修正カテゴリA〜G が網羅。`card_spec_probe` の classify＋§8.3 危険パターンで17枚を抽出し全数精査＝(a)「条件＋後続」10枚は条件が後続アクション全体を覆い健全（ST13-015/ST14-008/ST17-001/ST18-002/ST22-006/ST22-012/ST24-001/ST25-001/ST29-001）(b) WARN 6枚は self-target ACTIVE/GRANT_KEYWORD/RAMP_DON/PLAY_CARD に対する分類器の方向ヒューリスティック誤検知で全て正常。回帰 `test_st29001_*`/`test_st24001_*`/`test_st30001_*`）。ST11-004「新時代」の「リーダーがウタの場合…その後、…ドン!!1枚アクティブ」で ACTIVE_DON が条件外に出ていた件は **カテゴリH 是正で解消**（先頭ゲートを能力全体へ引き上げ。回帰 `test_st11_004_*`） | ✓ |

> on-rest 誘発の残課題のうち、相手効果の**発生源がキャラかリーダーかの区別**（「相手のキャラの
> 効果で」OP14-070）は**実装済**: 発生源カードを resolver→apply_action_to_engine→
> _fire_on_rest_triggers へ伝播し、発生源が判明していればキャラ限定を厳密化する（不明時は後方
> 互換で発火許容。回帰 `test_op14_070_*`）。**ブロック宣言によるレスト**（ブロッカー自身のレスト）
> は未対応のまま残すが、現行 on-rest カードは全て「自分のターン中」または「効果で」限定で、
> ブロック宣言（相手ターン・非効果）では発火条件を満たさないため実害ゼロ。
>
> **解消済み**: 「キャラかドン!!**合計N枚**」（N≥2。OP06-035／OP12-037）の**混在選択**（1キャラ+1ドン
> 等）は実装済み。パーサが単一 REST に `CHAR_OR_DON` フラグの混在候補（相手のキャラ＋ドン!!）を
> 持たせ、`matcher` が候補プールを構築、`resolver` の SELECT_TARGET で最大N枚を自由選択、REST
> ハンドラがキャラ/ドンを各々レストにする（回帰 `tests/test_char_or_don_mixed.py`）。「N枚まで」
> （合計でない・total=1）は混在の余地が無いため従来の Choice のまま（OP06-020 等）。
>
> **解消済み**: FREEZE 版「レストのキャラかドン」（OP07-026）のドン側は実装済み。パーサが
> Choice[FREEZE(キャラ), FREEZE_DON] を生成し、`FREEZE_DON` がレストのドン!!を `is_frozen` 化、
> `refresh_all` が次のリフレッシュで1回だけアクティブ化を据え置く（回帰 `tests/test_freeze_don.py`）。
>
> **解消済み**: 「キャラがレストになった時」は専用トリガー **`TriggerType.ON_REST`** として実装。
> パーサが「レストになった時」を ON_REST へ写像し（ターン文脈は CONTEXT 条件・ターン1回は
> TURN_LIMIT として保全）、エンジンは**アタック宣言**（`declare_attack`）と**効果による
> レスト**（`apply_action_to_engine` の REST 経路）の双方で `_fire_on_rest_triggers` を呼ぶ。
> 主語・要因は `_rest_subject_matches` が raw_text から解釈する（「このキャラ／キャラ」＝主語、
> 「自分の効果で／相手の効果で／アタック」＝要因）。対象: OP14-021/027/028/032/035/119（このキャラ）、
> OP07-031/OP10-036（任意主語・自分の効果で）、PRB02-009/OP14-070（このキャラ・相手の効果で）。
> 回帰 `tests/test_on_rest_trigger.py`・`tests/test_on_rest_subject.py`。

### 8.3 バグ類型カタログ（次に何を疑い、どう探すか）
発見した不具合は少数の再発パターンに収まる。検証時はまずこれらを疑う。

| 類型 | 具体例 | 検出手段 |
|---|---|---|
| **parse されるが実行系が無い**（最頻・最危険） | `RULE_PROCESSING`（エネルのドンデッキ=6、カード名別名）、`ON_LEAVE` がエンジン未発火 | 当該 ActionType/TriggerType を `grep` し、**gamestate/resolver に発火・適用箇所があるか**を確認。無ければ死んでいる（または no-op）。「パースできた＝動く」ではない |
| 複合句の取りこぼし | 「（相手の効果で）場を離れず、パワー+N」「【X】を得て、パワー+N」でキーワード/バフ片方が脱落 | AST に両アクション（PREVENT_LEAVE/GRANT_KEYWORD＋BUFF）が並ぶか |
| 条件の退化 | 「お互いのライフ合計」→自分のみ、「付与されているドン」→場のドン、複数リーダー名→先頭のみ、存在条件「ある/ない」の反転 | 条件の type / player / operator / value を実機（`_check_condition`）で真偽確認 |
| 対象フィルタの誤り | 「パワー8000の」を≤8000扱い、「レスト/アクティブにできない」等のアクション語を状態フィルタと誤認、「名前か種類」を AND 化 | `TargetQuery` の power_min/max・is_rest・flags・names/exclude_names を確認 |
| 持続時間の写像漏れ | 「次の相手のエンドフェイズ終了時まで」が `INSTANT` に退化し即失効 | 対象アクションの `duration` を確認（UNTIL_NEXT_TURN_END 等） |
| スケール値の脱落 | 「カード名の異なるキャラ1枚につき+N」がフラット値に退化 | `ValueSource.dynamic_source`（COUNT_QUERY 等）と count_query を確認 |
| 危険な常在 | `PASSIVE`+`VICTORY` 等が再計算ループで誤発火（相手ライフ0で自動勝利） | 不変条件テスト（誤って勝利/除去しないこと）を追加 |
| **先頭条件が「。その後、」をまたいで漏れる**（カテゴリH・是正済） | 「〈条件〉の場合、A**。その後、**B」でBが条件の外に出て無条件実行（EB02-032 ドン<3でもガレーラ登場／EB03-017 超新星でなくても相手レスト不可／OP04-033・ST11-004 等・全弾~119能力） | 能力 effect の先頭要素が `branch`(if_false=None) かつ後続に実効果アクション（PLAY_CARD/KO/REST/BUFF/付与/ACTIVE_DON 等）が並ぶか。先頭条件は能力全体（その後 B 含む）をゲートすべき。TEMP/REMAINING のデッキ整理だけの後続は no-op で無害。**`EffectParser._lift_h_gate` で先頭ゲートを能力全体へ引き上げ済み**。再混入は `tests/test_structural_gate.py`（構造不変条件＝上限0）で検出する |

> **カテゴリH の修正（是正済み）**: パーサ `EffectParser._lift_h_gate` が「能力 effect の先頭要素が分岐
> （if_false=None＝先頭条件）」のとき、その条件でシーケンス全体を包む（後続を if_true に取り込み
> `ability.condition` へ引き上げる）。条件成立時は従来と同一、不成立時のデッキ整理系は元々 no-op なので
> 観測挙動は不変、実効果のみ正しくゲートされる。
> 「公開→無条件でデッキ下」型（OP04-011 ナミ／EB01-029）は**公開したカードを必ず戻す＝無条件で正しい**ため対象外。
> 先頭が無条件 LOOK/宣言→分岐→「その後、報酬」の形（**OP11-066 シャーロット・オーブン**＝当たりならKO、
> その後ドン追加が漏れる）も非 index-0 だが同根の実害として同じ検出器・修正でカバーした。
> ~119能力・全弾（OP05〜OP16 の既検証弾も含む）に波及するためベースラインを再生成し（漏れ抑止の差分）、
> golden／検証デッキを更新、構造ゲートで違反0に固定した。再混入は `tests/test_structural_gate.py` で機械検出する。
> 見逃し原因の分析と横展開調査は [`reports/quality_postmortem_categoryH.md`](reports/quality_postmortem_categoryH.md)。

### 8.4 1枚あたりの検証チェックリスト
カードの各能力について、AST だけでなく**実機**で次を確認する。

1. **発火**: トリガー種別がエンジンで実際に発火するか（§8.3「実行系が無い」を疑う）。
2. **条件**: player / 比較 / 値 / 複数条件の AND-OR が正しいか（境界で真偽を実測）。
3. **条件“偽”パス**: **条件を偽にして発動し、ゲートされた効果が一切走らない（状態変化ゼロ）**か。
   先頭ゲート条件は「。その後、」をまたいで能力全体を支配する（カテゴリH）。真パスだけ見ると
   ベースラインが latent bug を凍結する死角がある（→ `tests/harness/false_path_coverage.py`）。
4. **対象**: ゾーン・側・種類・特徴・名前（別名含む）・パワー/コスト範囲・レスト状態・除外が正しいか。
5. **値**: 固定値か動的スケールか（「N枚につき」「同じパワー」等）。
6. **持続時間**: INSTANT / THIS_TURN / THIS_BATTLE / UNTIL_NEXT_TURN_END の写像。
7. **複合句**: 「〜得て」「〜ず、」で2アクションに割れているか（片方脱落していないか）。
8. **コスト**: 任意（できる）か必須か、支払い不能時にスキップされるか。
9. **副作用の安全性**: 誤って勝利/除去/無限ループ等を起こさないか。

### 8.5 二層回帰モデル（責務分担）
- **`full_card_baseline.json`**（構造・盤面差分）: 能力1つを単発の汎用盤面で動かした
  指紋。クラッシュ/カード消失/対象方向/単発の盤面変化の退行を広く検出。**意味的な
  細部・常在ルール・複数ターン・トリガー発火の有無は対象外**。挙動変更時は `--regen`。
- **`tests/test_verified_decks.py`**（意味的）: 手動検証で確定した「あるべき挙動」を
  ゲーム不変条件として固定。ベースラインの死角（§8.3）を埋める。意味挙動を直したら
  **必ずここへ追記**し、以後割らない（§5 ラチェット）。
- **`tests/test_structural_gate.py`**（構造不変条件・ランタイム偽パス）: ベースライン／オラクルが
  測れない *条件スコープ／期間／選択者／全体性* の死角を埋めるラチェット（上限0）。
  `tests/harness/structural_invariants.py` の4スキャン（先頭ゲート漏れH／Duration write-off／chooser欠落／
  「すべて」count退化）＋ `tests/harness/false_path_coverage.py`（条件偽で盤面変化ゼロ）。
  カテゴリH ポストモーテム（`docs/reports/quality_postmortem_categoryH.md` §6）の再発防止策の実装。
- **`tests/test_verified_buckets.py`**（台帳の機械保証）: §8.2 台帳「✓」の弾×色バケットが
  ベースライン指紋に全数登録され、カテゴリH 構造違反0であることを固定（ドキュメント主張→機械保証）。

### 8.6 未検証弾の弾×色検証計画
OP05〜OP16 は弾×色の横断検証（§8.2 台帳「弾×色, §8 デッキ非依存」行）で一巡済み。
**残る全弾（OP05〜OP16 以外）も、同じ弾×色バケット単位で効果の正しさを検証する**。
起点は `tests/scripts/card_spec_probe.py`（`--set <弾> --color <色>`）で、§8.4 のチェックリストに
沿って1枚ずつ実装と突合し、確定した挙動は `tests/test_verified_decks.py` に集約する。

対象弾と色（カードが存在する弾×色のみをタスク化。計 94 バケット）:

| 弾 | 色 |
|---|---|
| OP01 | 赤・緑・青・紫 |
| OP02 | 赤・緑・青・紫・黒 |
| OP03 / OP04 | 赤・緑・青・紫・黒・黄（各6色） |
| EB01 / EB02 / EB03 / EB04 | 赤・緑・青・紫・黒・黄（各6色） |
| PRB01 | 赤 |
| PRB02 | 赤・緑・青・紫・黒・黄 |
| P（プロモ） | 赤・緑・青・紫・黒・黄 |
| ST01 赤 / ST02 緑 / ST03 青 / ST04 紫 / ST05 紫 / ST06 黒 / ST07 黄 / ST08 黒 / ST09 黄 | 単色 |
| ST10 赤・紫 / ST12 緑・青 / ST13 赤・青・黒・黄 / ST30 赤・緑 | 複色 |
| ST11 緑 / ST14 黒 / ST15 赤 / ST16 緑 / ST17 青 / ST18 紫 / ST19 黒 / ST20 黄 | 単色 |
| ST21 赤 / ST22 青 / ST23 赤 / ST24 緑 / ST25 青 / ST26 紫 / ST27 黒 / ST28 黄 / ST29 黄 | 単色 |

進捗は WBS（`gx5gyqe2-art/WBS` の `projects/opcg-sim-backend.md`）の
「未検証弾の効果検証（弾×色）」フェーズで1バケット=1タスクとして追跡する。
検証完了した弾×色は §8.2 台帳へ1行追記し、回帰アサートを足す（単調増加・ラチェット）。

**完了**: ST10〜ST30（27バケット・203枚）／EB01〜EB04（24バケット・245枚）／PRB01・PRB02
（7バケット・19枚）／P プロモ（6バケット・120枚）は §8.2 台帳の各行のとおり一巡済み（独立した
新規バグ0）。この過程で**全弾横断の系統的バグ「カテゴリH」**（先頭条件が「。その後、」をまたいで
漏れる・~119能力）を検出し、§8.3 カタログに記録のうえ `EffectParser._lift_h_gate` で是正済み
（ベースライン再生成・構造ゲート違反0で固定）。
ST01〜ST09（9バケット・149枚）と OP01〜OP04（21バケット・約480枚）も一巡し、独立した新規バグ0
（ST02-014 の単発 count 退化のみ低優先残）。**全 §8.6 バケット（94）の一巡が完了**した。
カテゴリH を含む系統的バグは是正済みで、残るは低優先の単発項目（ST02-014 等）のみ。
