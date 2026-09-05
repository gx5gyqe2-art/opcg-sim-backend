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
`test_rl_datagen.py` / `test_turn_solver.py` /
`test_selfplay_v4_datagen.py` / `test_value_net_aux_turns.py` /
`test_pd_mixed_label.py` / `test_learned_candidate_prune.py` /
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

> 2026-08-25 純正AZ化: 補償層（プラン読み出し/受け方針箱/戦闘読み出し・コミット/LCB/aux/深さ減衰/旧ドン箱）を削除（docs/reports/2026-08-25_pure_az_cleanup.md）。該当テスト/計器の行は本表から削除。

### コアルール（ターン/戦闘/召喚酔い/場上限）
| ファイル | 役割 |
|---|---|
| `tests/test_rules_summoning_field_limit.py` | **召喚酔い/速攻**（登場ターン攻撃不可・速攻例外・リーダー非対象）と**場5体上限**（6体目で `FIELD_OVERFLOW_TRASH` 強制トラッシュ／効果登場でも発火／境界／**押し出し確定が【登場時】解決より先**）の検証 |
| `tests/test_leader_move_guard.py` | **リーダーのゾーン移動ガード**（bb0 発見の欠陥A・2026-08-11）: `move_card` がリーダーへ届くと元リスト remove が素通りしたまま宛先へ append＝**カード複製**（合成世界の実測 seed880007・実カードでも INCLUDE_LEADER 選択→KO で到達可能）。中央 no-op ガードで全経路（KO/バウンス/トラッシュ/デッキ送り）の保存則を固定。通常キャラ移動の回帰対照込み |
| `tests/test_attack_legal_cannot_rest.py` | **CANNOT_REST と攻撃列挙の整合**（bb0 発見の欠陥B・2026-08-11）: 「レストにできない」継続効果持ちの攻撃が合法手に載るのに `declare_attack` は ValueError で拒否する不整合（実測 seed880014・dead_child 汚染源）。列挙が検証側と同じ timed_flags を見ることをリーダー/キャラ両方で固定＋「列挙された全 ATTACK は declare_attack を通る」の整合直接検査 |
| `tests/test_freeze_don_target.py` | **ドン!!を対象にしたフリーズ**（交差対面監査 `deck_synth_audit --cross` が検出・2026-08-16）: `FREEZE` は `card.flags` に書くカード専用アクションで、ドン!!は `is_frozen`（`FREEZE_DON`）。「キャラかドン‼」の択一形（OP07-026）は Choice に分解済みだったが、**ドン!!だけを対象にする形**（OP10-033 ナミ）が素の FREEZE に落ち、`'DonInstance' object has no attribute 'flags'` で**対局ごと落ちていた**（アリーナでは1ペア全損）。(1) パーサはコストエリア対象を FREEZE_DON（枚数処理・`status="OPPONENT"`）にする、(2) FREEZE ハンドラは flags を持たない対象を `is_frozen` で受ける、(3) フリーズは1回のリフレッシュのみ（次で解ける）、(4) キャラ側の経路は不変 |
| `tests/test_noop_activation_loop.py` | **「何も起きない手」の無限反復ガード**（生成デッキ監査 `deck_synth_audit` が 137リーダー中3リーダーで検出・2026-08-15。E は交差対面監査が検出・2026-08-16）: (A) **レストコストの候補はアクティブのみ**（`_can_satisfy_node` は除外するのに `_resolve_targets` は除外せず、レスト済みリーダーを選んで支払いが no-op 化＝OP10-083 光月モモの助が無限再起動）、(B) **継続効果の再計算は対象選択の対話を出さない**（静的なコスト軽減が該当カード全部でなく「どれに掛けるか」を訊き、再計算のたびに同じ対話が復活＝OP05-097 聖地マリージョア）、(C) **効果が完全な空振りと証明できる起動メインは合法手に出さない**（自己制限が既に有効／レスト中のドン!!が無い＝EB04-016 トリ・OP10-030 スモーカー。証明できない効果は従来どおり合法手に残す保守側）。(B)(C)(E)(A') は出荷カード DB の master をそのまま使って検証する。**(F) 「登場させる」の候補はキャラ／ステージのみ**（イベントは発動するもので登場しない）。ST31-002 ジンベエは「キャラカード」ではなく**「カード」**とだけ言うため対象種別が無制限に解析され、イベント ST01-016 が場に出ていた＝場のイベントの【メイン】がコストなしの起動メインとして列挙され426回連続で空振り（アリーナ seed 907006 の void）。候補側（resolver）と適用側（play_card）の両方で塞ぐ。**(A'') 自己参照コスト（「このXを〜できる：」）は発生源に固定**（`ref_id='self'`）。レスト以外（トラッシュ／手札に戻す／デッキの下）が汎用の対象解析へ落ち、**DB全体で96節が別のカードで払えるコスト**に化けていた＝P-081 ミホークは相手のキャラを手札に戻して払え、発生源が場に残るので3手周期の無限ループ（アリーナ seed 904002 の void）。OP02-035/P-074 は挙動ベースライン上でも「相手のキャラを手札に戻す」だった。**該当0件をラチェット**で固定する。**(A') 状態を変えるコストは「既にその状態」の対象では払えない**（OP15-099 ウルージ「自分のライフの上から1枚を裏向きにできる」＝ライフが全て裏向きでも払えることになり297回連続で撃てた。判定と支払いの規則を `_cost_state_noop` に一本化＝レスト以外にも同じ穴があった）。**(E) 継続効果の再計算はコスト確認の対話も出さない**（OP09-080 サウザンド・サニー号＝「【相手のターン中】このステージをレストにできる：…場を離れた時、…」の「離れた時」が反応型と判定されず毎回評価され、拒否しても何も変わらない確認が613回連続＝上限手数で終わらない対局。反応型の判定に「離れた」を追加し、再計算中はコストを持つ能力を発動しない） |
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
| `tests/test_flagship_tcgplus.py` | TCG+ 開催マスター取得クライアント（`opcg_sim/api/flagship/tcgplus.py`、設計 docs/design.md §5.1）。**User-Agent のラチェット**（TCG+ は `Mozilla/` 始まりでない UA を 403 で拒否する＝実測 2026-08-16。退行すると全シリーズの開催同期が黙って止まり新開催期が「0件」に見える）／UA が実リクエストに載ること／403→`TcgPlusError` 写像／**`pref_code` からの都道府県フォールバック**（店舗予選は `place` が null・§16.17）／`limit=100` の offset ページング（total 到達で停止）／シリーズ単位キャッシュで再取得しないこと・`is_cached`（`/events` の upsert 抑止が拠り所にする）。ネットワークは monkeypatch 遮断 |

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
| `tests/test_game_driver.py` | **基盤健全性**（`cpu_infra`）。**共通対局ドライバ**（`tests/harness/game_driver.py`・設計⑥)の機械健全性: 同一 seed の決定論・observer 不干渉（観測専用の契約）・席の写像等価（run_one_game/play_game と一致）・`stop_after_decisions` 有界化・**learned(既定Gen＝現v6/gen6) 自己対戦の seed 再現** |
| `tests/test_replay_roundtrip.py` | **基盤健全性**（`cpu_infra`）。**実対局リプレイのラウンドトリップ**（`tests/harness/replay_runner.py`）: 録画（人間=private rng・card_id 基準記録）→記述子から再生（人間手注入＋CPU 再 decide）→勝敗・手数・ターン一致＋逆写像 miss=0。hard／**learned(既定Gen＝現v6/gen6)**／**coin toss（first_player=random）** の3系統＋リゾルバ単体 |
| `tests/test_replay_true_state.py` | **基盤健全性**（`cpu_infra`）。**リプレイ真盤面再生**（`replay_runner.state_at_action`・反実仮想レフェリー `--true-board` の入力）: 実 API 録画（g3 fixture）を両席 scripted で action 64 まで再実行→公開情報が直前フレームと一致＋フレームに無い内部状態（OP15-119 のパワーデバフ 7000→1000）の再現・**フレーム差分による対象欠落 ATTACK_CONFIRM の特定**（ライフ減=リーダー/消えた札=その対象。リーダー優先推測が幻のトリガー対話で分岐した g3 step88 の回帰）込みで全159手を最後まで再生・効果対話リゾルバ（`_resolve_dialog_action`）の card_id→候補uuid 写像（列挙順・重複消費／uuid 記録は同質候補のみ先頭充当・異種混在は miss）・裸記録＝空選択＋index/accepted 上書き |
| `tests/test_action_feats_v2.py` | **行動特徴 v9 拡張＋幅互換層**（`action.py`/`policy.py`・PR#188 レビュー#7・必須/標準）: カウンター値（0/1000/2000→0/0.5/1.0）・対象=リーダー flag・**攻撃マージン（(攻撃側−対象)実効パワー/1e4・v9.2＝これが無いと候補が@64でリーダー攻撃 2/12 悪手を選び続けた実測）**の append-only 追加（@82 型「カウンター温存」を policy が吸収する素地・1.9k教師で支持一致 60→62% 頭打ちの実測が根拠）・**旧次元 net × 新次元行列＝末尾切詰で出力完全一致**（既定 gen5 の serve 挙動不変の防壁）・`extend_action_dim`（零行温スタート＝恒等→旧22次元記録のゼロ埋め混在で学習が回り新特徴に勾配が流れる） |
| `tests/test_label_worker.py` | **基盤健全性**（`cpu_infra`）。**ワーカーの seed 割当て**（`label_worker.next_seed0`・純関数）: 累計局数ベース＝`--games` を途中で変えても過去 seed 帯と重複しない（旧式 batch_id×games が 16→4 変更で過去帯を再割当てした w1 運用報告 2026-07-18 の回帰）・games 一定時は旧式と互換・メタ欠損は旧式フォールバック |
| `tests/test_referee_labeler.py` | **基盤健全性**（`cpu_infra`）。**レフェリー再ラベルの純ロジック**（v9 フェーズ1・`referee_labeler.py`・純関数のみ＝高速）: 採掘候補の選抜（sat/disagree/blind の3カテゴリ round-robin＝どのカテゴリも飢えさせない・disagree は損失降順/sat・blind は昇順・隣接間引き・上限・index昇順）・policy 教師の構築（同価値バンド上位プランの初手 multi-hot＝バンド外は 0・同一初手の合算・初手が合法手に無ければ None＝誤教師を作らない） |
| `tests/test_coach_gate.py` | **基盤健全性**（`cpu_infra`）。**コーチゲートの判定則**（`coach_gate.py`・純関数）: **点別差の信頼下限 `min_reliable_delta`（2σ=1.414/√n・5seed=0.63／16seed=0.35）の数値固定**（5seed の 0.60→0.20 を「退行」と読んだ v19 の誤りの再発防止）・hit の card/type-only 判定・judge の非退行（確実点の大幅落ちのみ退行・不確実点の揺れは不問）と改善（ヒット計比較）・VERIFIED 採録の整形（非空 accept・実在ゲーム） |
| `tests/test_defense_plan.py` | **基盤健全性**（`cpu_infra`）。**防御側プラン化**（v8・カウンター/ブロッカー窓）: 攻撃側と同じ終端規約（手番が自分から離れる手で終わる）が防御にも成立することを g3@82（SELECT_COUNTER 窓・人間マーク実測点）で固定＝素通し(PASS)/各カウンター単発/重ね切りの列挙・全プランの終端は戦闘解決(PASS)・実際の手（EB03切り→PASS）の記録からの2手復元。ロールアウトなし＝高速 |
| `tests/test_referee_comeback.py` | **基盤健全性**（`cpu_infra`）。**捲りモードの相手不完全性モデル**（v8・`counterfactual_referee._sample_by_visits`・純関数＝高速）: 相手手番の訪問数比例サンプル p∝N^(1/τ) の3性質＝低温 τ→0 は argmax（temp0 と連続）・τ=1 は訪問数比例＋固定 rng で決定論（CRN 再現性）・縮退（訪問ゼロ/1手/空） |
| `tests/test_referee_band.py` | **基盤健全性**（`cpu_infra`）。**同価値バンド v2 の対判定則**（v8 柱B・`counterfactual_referee.same_value`・純関数＝高速）: CRN の世界線共有を活かした符号検定風＝同一世界で勝敗が割れたペアの正味差 ≥3 のみ断定・未満はライフ差 < band で同価値（±1勝の揺れで断定が往復した @64 実測が較正根拠）・割れの相殺・飽和局面のライフ序列・不成立世界の共通部分対判定 |
| `tests/test_referee_plan_enum.py` | **基盤健全性**（`cpu_infra`）。**ターンプラン自動列挙＋実プラン復元**（v8 柱A/C・`counterfactual_referee.enumerate_turn_plans`／`coach_sweep.actual_plan_keys`）: g3@64 真盤面で比較の本命（素の攻撃・攻撃者自身への付与→攻撃・素の TURN_END）が縮約後も必ず残る（素朴な value 順が gen5 の付与バイアスで全滅させた回帰＝コミットメント別ラウンドロビン＋種内「短さ→自己強化→value」）・終端規約（手番が自分から離れる手で終わる）・縮約の必須ログ（無言の縮約禁止）・実際の手（素 ATTACK:PRB02-008）の記録からの復元一致。ロールアウトなし＝高速 |
| `tests/test_replay_frames.py` | **リプレイ盤面フレーム**（`services/replay.py::_replay_record_frame`＋`GET /replay/frames`・リプレイビューアのデータ供給契約）: frames↔actions↔decisions の action_index 整合（フレーム0＝初期盤面のみ None）・フレームカードは動的状態のみ（マスター情報を持たない＝サイズ抑制）・`_FRAME_CAP` 超過で記録停止＋`frames_truncated`・非 traced 対局は記録なし＋整形エラー |
| `tests/test_perf_gate.py` | **基盤健全性**（`cpu_infra`）。**CPU 性能ゲートの判定ロジック**（`tests/scripts/perf_gate.py`・§5.1）: `evaluate_gate` 純関数（強度不足/レイテンシ超過/失敗局/データ不足→FAIL・理由の蓄積）＋ gen2〜gen8_*.npz ハッシュの安定性（gen8＝本番既定・2026-07-29採用）。実対局は回さず高速固定 |
| `tests/test_promotion_gate.py` | **基盤健全性**（`cpu_infra`）。**昇格ゲートの判定ロジック**（`tests/scripts/promotion_gate.py`・v6 柱①）: 段階式判定の純関数＝stage1（24局で勝ち越しのみ継続・五分以下は即棄却）／final（累計勝率 ≥ 0.55 で昇格・境界は昇格側・frac 可変・浮動小数境界の安定）／anchor（v7・固定アンカー非退行 ≥0.5・五分は許容・r99 実測ケース 8/24 を棄却）。実対局 arena は回さず高速固定 |
| `tests/test_search_averse_probe.py` | **基盤健全性**（`cpu_infra`）。**SEARCH_AVERSE 追跡の判定則**（`tests/scripts/search_averse_probe.py`・純関数 `diagnose`/`world_sensitive`）: **アブレーションは base を上回って初めて原因**・base≥0.5 は NOT_FAILING@deep・SEARCH_Q_BOUND は全深さで accept の Q が下回ることを要求・世界依存フラグは 0<base<1 のみ。純正AZ化（2026-08-25）で腕は base/多世界のみに縮小 |
| `tests/test_prior_bound_probe.py` | **基盤健全性**（`cpu_infra`）。**prior/value 分解の分類則**（`tests/scripts/prior_bound_probe.py`・v20・純関数 `classify`）: 優先順（探索の浅さ→prior→value）と各機序の境界＝PRIOR_BOUND は「prior が薄い かつ 一様priorで立ち上がる」の**両方**が要る（片方だけで policy 起因と断定しない）・VALUE_BLIND は dv≤0（境界0含む）・**SEARCH_AVERSE**（prior 1位かつ dv>0 なのに深探索が選ばない第3の機序・2026-07-29 実測3件）は prior 1位でなければ断定しない。実 decide・実復元は回さず高速固定 |
| `tests/test_value_blind_probe.py` | **基盤健全性**（`cpu_infra`）。**VALUE_BLIND 原因分析プローブの純関数**（`tests/scripts/value_blind_probe.py`・v23）: 遮蔽帰属のグループ定義が符号化3キー（scalars55/field10行/card_idx24枠）の**完全分割**（漏れ・重複は帰属の見逃し/二重計上）・swap_group の非破壊/対象限定・**線形ネットでは帰属総和が gap に厳密一致**（fwd=rev＝分解の健全性）・scan_target の展開（自場ID新出）/付与（attached_don 増加）判別・contrast_stats の echo=dq−dz（NaN q 除外・対照空なら差を主張しない）。実ネット・実盤面は使わず高速固定 |
| `tests/test_rl_encoder_v7.py` | **基盤健全性**（`cpu_infra`）。**v7 符号化世代＝登場時オプションの実測3値**（`cpu_ai.onplay_option_scan`・v29・2026-08-01）: 手札の **ON_PLAY 持ち各札**を **make/unmake で適用→観測→巻き戻し**し（**ドン非依存**＝コスト分の一時ドンを txn 内で補う。2026-08-02 修正: 旧実装は「今払える PLAY」だけを見ており、ドン枯渇後の子で全札が非合法になり両方 (0,0,0) へ潰れて**オプションを温存した子と行使した子が判別できなかった**＝判別が要る唯一の場所で盲目だった）「バニラ設置以外の何か」（効果対話 or EFFECT イベント）が起きるかをエンジン自身に確かめさせる（判定子＝適用後 pending!=MAIN_ACTION or EFFECT。実測 0.6〜1.1ms/局面・decide 309→546ms）。「手札のパワー6000を2枚公開」等の**カード間関係の登場時条件**は埋め込みの線形和で表現できず（v24 representation-bound）、実測でしか全カードに一般化しない。固定する性質＝**子盤面での判別**（オプションを行使した子は live が減り温存した子は保たれる）・ドン非依存・**恒等温スタート**（v6→v7 で予測完全一致）・**副作用ゼロ**（global random 不消費・盤面不変＝探索/リプレイ/CRN の再現性を壊さない）・非メイン手番は (0,0,0)（今行使できるオプションの意味論） |
| `tests/test_rl_encoder_v8.py` | **基盤健全性**（`cpu_infra`）。**v8 符号化世代＝自場集約の純対称化**（v32・2026-08-02/03・ユーザ指摘「パワー2000以下のキャラの盤面価値は低い」）: v5 は相手場のみ集約を持ち自場はキャラ数の生カウントだけ＝gen10 実測（power_value_probe）でバニラ2000追加が 6000 体の 2/3 の加点（「体があれば加点」が支配・自側のパワー傾き 0.026/1000 < 相手側 0.036/1000）。v7 末尾に [自場総火力/高パワー数/ブロッカー数]＝相手 v5 と**同じ関数**（`_opp_field_aggregate`）の 3 を append。**しきい値つき弱ボディ特徴は設けない**（汎用性のユーザ方針 2026-08-03＝平均パワーは総火力÷キャラ数からネットが導出）。固定する性質＝版マップ（63+3=66）・**末尾3値の配線**（実盤面で直接計算と一致＝offset ズレ検出）・**純対称性**（自分視点の自場集約==相手視点の相手場集約）・接頭辞不変（v7 と完全一致）・**恒等温スタート**（v7→v8 で予測完全一致） |
| `tests/test_rl_encoder_v11.py` | **基盤健全性**（`cpu_infra`）。**v11 符号化世代＝リーダー物理要約24**（`opcg_sim/src/learned/leader_feat.py`・2026-08-14）: 接戦帯の帰趨を支配するリーダー再帰効果（ドンランプ・回復・ミル・常在修正）が v10 まで**0ビット**（消去はしご2.6σ・重み直し天井0.16 で確定）。能力木を ActionType で歩き毎ターン率12次元×自/相手を append（73+24=97）。ID非依存＝パースできる新リーダーへ即汎化。固定する性質＝版マップ・末尾24値の配線・接頭辞不変（v10）・恒等温スタート・**乱数無消費**（純粋な木walk＝符号化は観測）・**意味の錨**（ハンニャバル don_rate>0／ビビ atk_disable=1／ナミ rule_flag=1）。初回実測（bb6）: 同一対局A/Bで域内MSE−9%・実L中間帯 +0.117→+0.147（弱い正）→**配線修正（BUFF/ACTIVE_DON/防御系トリガー・ユーザ指摘起点）で +0.249・ナミ帯 +0.311**→被覆86%化（DISCARD=純手札経済・REST/ACTIVE）で同水準を確認し**v11 定義確定**（`backbone_bb7_v11final_20260814.md`） |
| `tests/test_rl_encoder_v12.py` | **基盤健全性**（`cpu_infra`）。**v12 符号化世代＝v9 + リーダー物理要約24（94列・リーサルΔ抜き）**: v11 から v10 のΔ3列だけを外した**安価版の分岐**（一本道の append-only 系譜ではない唯一の版）。固定＝版マップ登録と次元（94）・`encode(version=12)` が **v11 の列 [0:70]+[73:97] と bit 一致**（＝`corpus_v11_to_v12` の切り出しが正しい／コーパス再生成が不要であることの根拠）・前半70列が **v9 と一致**（G14 からの温スタートが末尾ゼロ追加の恒等拡張になる）・**符号化コストが v9 並み**（Δのエンジン台本再生を通らない＝`lethal_scan` を呼ばない）・`battle_resource_cols(12)` の列が範囲内で末尾24列がリーダー要約を指す。動機と実測は `docs/reports/gen15_adoption_20260815.md` §3 |
| `tests/test_rl_encoder_v12.py` | **基盤健全性**（`cpu_infra`）。**符号化 v12**（= v9 + リーダー物理要約24・**リーサル距離Δ抜き**・2026-08-15）: v10 のΔはエンジンで台本を再生する実測特徴で ~25ms/盤面あり、探索が1手で数百回符号化するため候補ネットの decide が**0.47s（v9）→13.5s（v11）**＝本番予算1秒を28倍超過した（アリーナ 10分/ペアで発覚・2026-08-15 実測）。リーダー要約はカードIDキャッシュで実質ゼロコストかつ gen15 系の改善の実体、Δは v53 で両系とも転移せず効果未実証——よって**安い側だけを v9 系譜に継ぐ**分岐版。固定＝次元94・レイアウト[v9 70｜リーダー 24]・**v11 行の列切り出し（[0:70]+[73:97]）と bit 一致**（既存コーパスを再生成せず教師にできる根拠＝`corpus_v11_to_v12.py`）・前半70列が v9 と bit 一致・**リーサルスキャンを呼ばない**（呼べば失敗する細工で証明）・warm_start_value(9→12) が恒等 |
| `tests/test_rl_encoder_v9.py` | **基盤健全性**（`cpu_infra`）。**v9 符号化世代＝ドンデッキ残＋自デッキ残キャラ頂点**（v49・2026-08-10）: リーダー固有のドン上限（紫エネル=6）と「don!!-X で山へ戻したドンがリーダー効果で再装填される」経済が v8 まで**原理的に不可視**＝h1@2（turn1 サトリで掘る/無行動）のターン末比較が value Δ=+0.011 の無差別になる根因（v48/v49 実測）。v8 末尾に [自ドンデッキ残/10, 相手ドンデッキ残/10, 自デッキ残キャラ最大パワー/10000, 同最大コスト/10] の 4 を append（66+4=70）。頂点＝連続量で「山に眠る勝ち筋」（OP15-118 cost6/8000＝v4 の cost≥7 カウントが落とす帯）を見せる——**しきい値特徴は新設しない**（ユーザ方針 2026-08-03）・既存 v4 特徴も append-only 契約で不変。固定する性質＝版マップ・末尾4値の配線・**ドンデッキ残の感度**（1枚差で該当特徴だけが動く）・頂点が cost6/8000 を見る・接頭辞不変・恒等温スタート（v8→v9 で予測完全一致） |
| `tests/test_rl_encoder_v6.py` | **基盤健全性**（`cpu_infra`）。**v6 符号化世代＝自手札の資源集約**（`encoder._hand_aggregate`・2026-07-30・ユーザ指摘「手札の価値をどう正確に判断するか」）: append-only（先頭55は v5 と完全一致）・集計の中身（カウンター総量/札枚数/最大カウンター/手札ブロッカー/イベント・正規化込み）・**公平性契約**（相手手札の中身は漏れない）・空手札安全・「カウンターを失うと資源集約が下がる」＝v23 の相貌学習（手札減＝良い）への逆向きの取っ手が特徴として成立すること・**v5→v6 温スタートの恒等性**（新5行ゼロで予測完全一致） |
| `tests/test_defense_cf_gen.py` | **基盤健全性**（`cpu_infra`）。**防御窓CFコーパス生成の純関数**（`tests/scripts/defense_cf_gen.py`・フェーズ2）: causal_z の値域/正規化（v24 と同一規約）・spread=0 が「どの防御でも結果が同じ＝無情報窓」を表すこと（有情報率モニタの土台）・ターン分散サンプリング（防御窓は攻撃連打の1ターンに固まるため一様抽出は偏る）・選択肢の同一視は probe と同一定義を共有（定義の二重化を防ぐ）。実対局・実ロールアウトは回さず高速固定 |
| `tests/test_defense_cf_probe.py` | **基盤健全性**（`cpu_infra`）。**防御窓CF×人間整合プローブの純関数**（`tests/scripts/defense_cf_probe.py`・①防御応答矯正フェーズ1）: 選択肢の同一視（行動種×card_id・同名複製は等価）・整合判定（agree_top は同数タイを人間側有利・agree_band は勝ち数差<band・**人間選択が列挙に無ければ None＝列挙漏れとして表面化**）。実ロールアウトは回さず高速固定 |
| `tests/test_default_interaction.py` | **効果対話の既定解決**（`engine/interaction.py` `choose_selection`＝ゾーン意味論・2026-07-30 実測欠陥の回帰ガード）: **統一「残す価値」`card_keep_value`**（コスト・現在パワー・カウンター値・効果保有・カウンタートリガー・【トリガー】の合成＝ドレイン既定と探索分岐順序 `_rank_select_candidates` の共通序列）の序数性質＝同コストなら効果+カウンター持ち＞バニラ・カウンターイベント＞バニラ低コスト・大型の順位維持（コストだけで捨て札を決めない＝ユーザ指摘の修正）。コスト系（自手札/場）は min 件を価値昇順・獲得系（自山札/トラッシュ・TEMP は min=0 のみ）は max 件を価値降順・対象系（相手側）は降順・判別不能（ドン/混在/強制 TEMP）は旧既定へ退避（RETURN_DON の候補順細工を維持）。統合＝実測欠陥2点の真盤面で固定: m4@2 イワンコフの引き3捨て2が公開 6000 2枚を温存し低価値2枚を捨てる／m1@3 ウタの「1枚まで」が見送らず最高コストを加える。旧既定は探索・自己対戦・レフェリー裁定を汚染していた（先頭min件＝最良札から捨てる・min=0＝常に見送る） |
| `tests/test_counterfactual_pair_gen.py` | **基盤健全性**（`cpu_infra`）。**反実仮想ペア教師生成の純関数**（`tests/scripts/counterfactual_pair_gen.py`・v24）: 採掘条件（化粧系 PLAY/ATTACH_DON/ACTIVATE_MAIN と進行系 ATTACK/TURN_END が同時に合法＝防御窓は自然に除外）・行動種別代表の選抜（種ごとに1手先 value 最良・種1つでは対照を組めない契約）・causal_z の値域・ターン分散サンプリング（メイン決定は同一ターンに連続するため一様抽出は固まる＝ラウンドロビンでカバレッジ優先・決定的）。実対局・実ロールアウトは回さず高速固定 |
| `tests/test_arena_gate.py` | **基盤健全性**（`cpu_infra`）。**固定N・帯層別アリーナ判定器**（`tests/scripts/arena_gate.py`・v16・`docs/reports/cpu_v15_ensemble_power_20260726.md` §2＝24〜120局の判定は検定力不足だった反省）: 帯分割の純関数（全ペアを過不足なく分配・帯間 seed 基点が stride ぶん離れ全 seed 一意）／一次スクリーン（floor 未満で早期棄却・境界は継続）／本判定（**勝率 ≥ frac かつペア水準95%CI下限 > 0.50 の2条件**＝点推定だけの偽陽性を塞ぐ・五分のヌル対照は PASS しない）。実対局 arena は回さず高速固定 |
| `tests/test_dense_finetune.py` | **基盤健全性**（`cpu_infra`）。**密ラベル追い学習のラベル生成**（`tests/scripts/dense_finetune.py`・v16・純関数 `build_labels`）: 混合ラベル y=α·勝敗+(1−α)·q_root（α=1 で勝敗単独＝レフェリー教師と地続き）・**q_root が非有限な行（L1 席等）は勝敗単独へ退化**＝NaN がラベルへ伝播して学習を壊さない・aux は [0,1] 正規化＋飽和で NaN は欠損のまま通す（`ValueNet.backward` が補助損失から除外する契約） |
| `tests/test_option_pair_gen.py` | **基盤健全性**（`cpu_infra`）。**オプションペア教師生成の純関数**（`tests/scripts/option_pair_gen.py`・v31・`docs/reports/cpu_v30_option_feature_20260802.md` §4-B）: v7特徴で m4@2 の value は方向づいたが回帰では符号反転に届かない（拮抗2子の勾配が弱い・v30 §3-2）ため、v12.1 の順位ヒンジへ渡す**カード単位ペア**を生成する計器。固定＝**カード単位の枝**（v24 の行動種分岐では PLAY vs PLAY を対照できなかった修正の核心＝同card_idは代表1つ・別card_idは別枝・TURN_END を温存枝に足す）・qualifies（ON_PLAY 持ちPLAYが2枚以上＝m4@2 パターン）・causal_z/spread の値域・生成物（group+value）が `build_rank_pairs` にそのまま繋がること。実対局・実ロールアウトは回さない |
| `tests/test_rank_anchor.py` | **基盤健全性**（`cpu_infra`）。**蒸留アンカー付き順位微調整**（`ref_finetune_smoke.rank_finetune_anchored`／`dead_weighted_pairs`・v33・2026-08-03）: v32 の負の結果（アンカー無し順位ヒンジは順位が上がるほど防御較正 m2@12/58 が先に壊れる・3回再現）への機構対応＝順位バッチと「アンカー盤面で base 予測へ引き戻す蒸留バッチ」を交互に流す。固定＝**錘の効果そのもの**（同一ペア学習でアンカー有りのドリフト < 無し）・順位はそれでも学習される・anchor_scale=0 は素の rank_finetune と厳密一致（rng 消費経路も同一）・dead_weighted_pairs は負け側が不発PLAYのペアだけ k 倍（k=1 恒等）。合成小ネットのみ＝実盤面不使用 |
| `tests/test_effect_selection_wiring.py` | **基盤健全性**（`cpu_infra`）。**自分の効果対話が探索の決定点として現れる**（`adapter.OPCGGame.apply`／`mcts` の各 apply に `stop_at_select=True`・v39・2026-08-06）: 学習型CPUの遷移は手の適用後に **actor 自身の効果対話を既定解決でドレイン**していたため、CPU は自分の手で生じた対象選択を一度も決定できなかった（人間には選択リクエストが飛ぶのに CPU だけ既定ヒューリスティクス＝非対称）。実害（m2@44）: 相手リーダー OP09-001 の「アタック時 −1000」が攻撃者でなくパワー最大のキャラに当たり、**アタックを止める唯一の手段を毎回捨てていた**。`merged_search_actions` は当初からこの分岐を候補化する設計だったが、apply 側のドレインで保留が立たず「相手の手で生じた対話」にしか効いていなかった。固定＝確認と対象選択が2つの決定点として現れる・選んだ対象が実際に適用される（既定解決に上書きされない）・出口 value の箱は選べさえすれば攻撃者への −1000（＝アタック不成立）を最良に並べる（判断力でなく提示の欠陥だったことの切り分け） |
| `tests/test_exit_heads.py` | **基盤健全性**（`cpu_infra`）。**出口専用 value ヘッド**（`ValueNet.EXIT_HEADS`／`enable_exit_head`／`predict_exit`／`backward_exit`・v39 ターン末 We1..be2 / v41 戦闘出口 Wb1..bb2）: 箱の階層ごとに較正を分ける出力の append-only 拡張。全性質を階層で parametrize＝**新しい箱を足すときの雛形**。動機は2つの負の結果——v38（ターン出口教師を既存ヘッドへ同居させると m5@7/m2@66 は満点になるが m1@15 が 1.00→0.00・8点合計 3.06 < 本番 3.44・α補間でも救えない＝gen12 の m1@15 のマージンが +0.062 しかない）と v40（防御CFで**本体 value を直接**順位学習するとコーチ 8.00/8.00 満点なのにアリーナ 0.447 CI[0.409,0.485]・284ペア＝有意な退行。全面学習は盤面評価そのものを全域で動かす）。固定＝有効化は残差ゼロで**恒等**・未有効/旧 npz は `predict_exit` が既存ヘッドへフォールバック（同梱ネットは無改修）・ヘッド学習は胴体/既存ヘッド/補助ヘッド/**他階層のヘッド**を **bit 不変**に保つ・順位ヒンジがその出口の順位を実際に改善・save/load と複製（expanded/widened）でヘッドが残る・消費側の結線（ヘッド無しネットでは v39/v41 以前と同計算／`evaluate_plan` は出口盤面だけをターン末ヘッドで測り戦闘窓は戦闘ヘッドへ渡す／`_window_choice`（窓の根畳み）が箱へ渡す物差しが戦闘出口ヘッドである＝**この引数が本体 value に戻ると v41 は無効化される**）・**リソースヘッド**（`in_cols`・2026-08-14 ユーザ提案「手札/盤面/ライフの束で交換レートを学ぶ」）＝胴体 A1 でなく生 scalars の指定列を読む変種でも同性質が成立（恒等・胴体凍結・save/load 往復・`expanded` 挿入時の列番号追随＝恒等保存）＋`E.battle_resource_cols` の列健全性（範囲内・重複なし・世代間 append-only） |
| `tests/test_mcts_box_battle.py` | **基盤健全性**（`cpu_infra`）。**木の中の箱化**（`TreeMCTS._expand` の戦闘窓畳み込み・v35・2026-08-05・ユーザ指摘）: 実対局の窓を出口 value で選ぶようにしても（decide の「窓の根畳み」）**木の中の戦闘窓は通常ノードのまま**訪問を配っており、同じ場面を木と実対局で違う規約で扱う唯一のずれが残っていた。二人零和では相手は最善応手を返すのが正しく、PUCT の訪問混合は**収束前の副産物**であって設計された保険ではない＝畳む方がミニマックスに近い（幅も失われない。木には別の攻撃順・別盤面の戦闘が無数にあり各々が独立した箱を持つ）。副次効果＝カウンターの組合せに配っていた訪問がメイン判断へ回る／**攻撃の帰結が具体的な出口として立ち上がる**（「相手手札−1＝カウンターを絞り出した」か「相手ライフ−1・手札+1＝通した」の二択。攻撃は必ず相手に損失を強いるので『止められる＝無駄』ではない）。固定＝戦闘窓ノードが**単一辺**へ畳まれその辺が出口最良（m1@15 で止まる 2000 カウンター）・葉見積もりも同じ出口の値（木と読み出しの規約一致）・OFF は全合法手を子に持つ（ロールバック可能）・メインフェーズは畳まない・探索後も盤面と global random が復元。**gen12 で既定 ON**（m2@44 0.00→0.62・m5@7 0.00→0.56 を初めて動かした） |
| `tests/test_anchor_strata.py` | **基盤健全性**（`cpu_infra`）。**層別アンカー**（`option_pair_finetune.load_anchor(own_turn_only=True)`・v35・2026-08-05）: 蒸留アンカー（v33）は dense 一般盤面（約4割が相手ターン＝防御判断側）を base 予測へ MSE で釘付けにするため、**ライフ↔手札の交換レートという評価尺度そのものを動かす**防御較正では順位教師の押しが木の深部で打ち消される（v35 実測: 1手先は正解カウンターが +0.117 上なのに探索後 root Q は PASS が上へ逆転）。相手ターン盤面をアンカーから除外し自ターン挙動だけを固定する。固定＝`IDX_IS_MY_TURN`（scalars 列11・append-only 契約で全版不変）が encode() の実出力と一致（両手番で反転確認）・own_turn_only=True は自ターン行のみ残す・y はフィルタ後盤面への base 予測（行対応がズレない）・未指定は従来挙動（後方互換） |
| `tests/test_mcts_quiesce.py` | **基盤健全性**（`cpu_infra`）。**静止探索**（`TreeMCTS._leaf_value`・v35・2026-08-04・ユーザ提案「防御に入ったら防御処理が終わるまで探索を続ける」）: カウンター選択の最中は**戦闘が未解決**で「1000 を切った子」と「2000 を切った子」が符号化上ほぼ同一（手札-1・ライフ不変）＝どちらが命を救うかは解決後にしか盤面へ現れない。gen11 実測で正解の 2000 が3択の最下位（-0.4502）・止まらない 1000 が最高（-0.4281）＝手札の最大カウンターを温存する汎用癖が逆向きに働いていた。戦闘中の葉は**解決まで進めてから**評価する。**葉の意味論＝解決した時点の手札と盤面**（ユーザ指摘 2026-08-04）。延長は policy 最良手→PASS→先頭手の順（priors 不在時に先頭手を機械的に採ると m1@15 では「もう1枚足す」が先頭で**両枝とも助かって判別不能**になる実測落とし穴があるため PASS へフォールバック）。実測の解決時盤面: 素通し=手札5/ライフ4、正解2000=手札3/ライフ5、不発1000=手札4/ライフ4（＝不発は素通しにカード1枚分だけ純粋に劣る・正解は2枚と1ライフの交換）。固定する性質＝不発カウンターの評価が下がること（存在理由）・**副作用ゼロ**（盤面と global random が完全復元＝CRN 一貫性/リプレイ再現性）・quiesce=False で従来と同一・戦闘中でない葉は no-op。**gen12 で既定 ON**（単体では決定が変わらないが、教師の解決後符号化・実対局の箱読み出し・木の箱化と3機構そろって機能する） |
| `tests/test_forced_defense_gen.py` | **基盤健全性**（`cpu_infra`）。**ε強制防御**（`tests/harness/p3_loop._forced_defense_index`・v26・`docs/reports/cpu_v25_dense_regen_20260731.md` §5）: v4(c) の防御応答温度は**訪問分布依存**のため、守りに価値を認めないネットでは守る対局が生成されない循環（v25 実測＝温度延長込み2048局でも守り採択率 0.281＝gen8 と同値）を切るための強制抽選。固定する性質＝**eps=0 で None かつ乱数を引かない**（既存コーパスとの rng 消費順の互換／`_NoDrawRng` で機械的に検証）・守る手が無い窓も乱数を引かず None・eps=1 は必ず非PASS・**守る手の一様抽選**（訪問分布に依存しない＝分布の新規性の源）・ε が強制率そのもの・生成コアの引数既定が 0（配線の存在確認） |
| `tests/test_move_regret.py` | **基盤健全性**（`cpu_infra`）。**手の監査 段2の判定規約**（`tests/scripts/move_regret.py`・実対局は回さない）: regret＝最良の勝率 − 打った手の勝率（打った手が最良なら0）／**全選択肢が同率は `saturated`＝判別不能**として集計から外す（v49「両腕とも0/32勝でラベル飽和」の教訓）／容疑者は優先度降順で読む／同じ seed は**1回の再生**にまとめる／飽和・未測定は母数から外して件数を必ず出す |
| `tests/test_plan_struct.py` | **基盤健全性**（`cpu_infra`）。**構造化プラン提案器**（2026-08-20 ユーザ設計「プレイするカードの組×浮ドンの使い途」）: プレイ組はアクティブドン予算内・同名の組は畳む・空集合と最大コスト組を必ず含む／intent は正準順序（登場→付与→攻撃）で付与/攻撃の対象は今アクティブな既存ユニットのみ（P1/P2 を生成側で守る）／実現器は実現不能な指示を縮退し対話は policy 最良手で埋めて記録／`select_plan` の diag.kinds に提案の由来ラベルが出る |
| `tests/test_plandef_gen.py` | **基盤健全性**（`cpu_infra`）。**D族生成器の判定規約**（`tests/scripts/plandef_gen.py`）: battle_need の算術（atk≥defで通る・buff考慮）／D1=総量不足で素通しが支配／D2=最小札組が+1枚を支配／必要0・札なし・余りなしでは対を立てない |
| `tests/test_plan_dom_gen.py` | **基盤健全性**（`cpu_infra`）。**P1/P2支配ペア生成器の判定規約**（`tests/scripts/plan_dom_gen.py`）: 死に付与先は V1（レスト済み・ドン条件なし・レスト/相手ターン常在の文言なし）と V1s（【ドン!!×N】持ちでも attached≥N＝閾値達成済みなら追加付与は死に・段3裁定 #1/#2 のドレーク型）／「相手のターン」常在（チョッパー型）と閾値未達の1枚目は死ににしない／V1系の対はドンの置き場所のみ・V4の対は付与と攻撃の順序のみが違う |
| `tests/test_macro_moves.py` | **基盤健全性**（`cpu_infra`）。**マクロ手化 P1＝配分箱の契約**: `don_alloc_candidates` は枝刈り済み原始付与から「対象へk枚」（k=1/閾値開放/全振り・DON_BOX の target_ids=[] 形）を合成／適用展開は付与のみで攻撃しない／行動特徴は ATTACH_DON として符号化（素付与の prior 継承）／adapter の macro_moves seam は ON で原始 ATTACH_DON を撤廃・OFF（既定）は挙動不変 |
| `tests/test_dialog_box.py` | **基盤健全性**（`cpu_infra`）。**マクロ手化 P3/P5＝対話箱の契約**: `in_dialog` の語彙（SEARCH_AND_SELECT/CONFIRM_OPTIONAL/CONFIRM_TRIGGER/CHOICE/DECLARE_COST は対象・外周 MULLIGAN/ARRANGE_DECK/SELECT_RESOURCE と MAIN_ACTION/戦闘窓は対象外）／実盤面 m2@22 の対話窓検出／対話窓読み出し（`_dialog_window_choice`）が readout=dialog_resolved で**OFF の合法手集合に含まれる手**を返す（新手型を作らない）／`resolved_branch_values(window_pred=in_dialog)` の枝が窓の外の出口へ到達する／`TREE_BOX_DIALOG` 既定 OFF |
| `tests/test_box_commit.py` | **基盤健全性**（`cpu_infra`）。**箱コミット実行の契約**（ユーザ決定 2026-08-26「箱は選ぶ時だけ判断し、中身は機械実行」＝箱の原子性の完成・`SERVE_BOX_COMMIT` 既定 ON・`LearnedEngine._commits`）: アタック箱コミットは ATTACH_DON×k の後に必ず ATTACK が出る（プラン半消化バグ 2026-08-25 の再発ガード）／配分箱 k=2 は付与2回で手順が空になる（攻撃しない）／契約違反（存在しない uuid の sig）はコミットを**全破棄**して通常判断の手が返る（縮退して続きだけ拾わない＝箱単位で再入札・クラッシュしない）／seam: box_commit=False は従来（毎 decide 判断・コミットキャッシュを読まない/書かない）／PLAY 対話コミットの手順は評価（`resolve_battle_inplace(window_pred=in_dialog, box_depth=…)` を直接呼んだ結果）と一致＝評価が正当化した継続と実行される継続が同一／**適用検証**（2026-08-26 void 修正）: コミットの返す手（カウントダウンの合成 ATTACK/ATTACH_DON 含む）は実盤面クローンで適用可能か検証し、不可なら箱ごと全破棄して通常判断へ（新しい箱の再コミットは正当＝箱単位で再入札） |
| `tests/test_gen_explore.py` | **基盤健全性**（`cpu_infra`）。**生成の探索多様性の契約**（純正AZ 2026-08-27）: `LearnedEngine(dirichlet_eps=…, temp_turns=…)` seam——serve 既定（両方 None）は同一 seed 同一手＝従来どおり決定的／temp_turns 有効時は序盤メイン窓が訪問分布 τ=1 サンプリングになり seed で分散・選択は常に候補（record.groups の (sig,k)）の一員／dirichlet_eps 有効でも合法手が返る |
| `tests/test_n_record.py` | **基盤健全性**（`cpu_infra`）。**棋譜ダンプの record 観測の契約**（純正Nループ① 2026-08-26・`n_record_gen.py` の基盤）: `LearnedEngine.decide(record=dict)` は観測専用＝record の有無で選択が変わらない／main 窓は kind="main"・groups（`_merge_root_stats` と同一集計の全候補 {sig,k,n,q}・n 降順・候補の同一性は (sig,k)＝配分箱の k 違いは同 sig）・選択 (sig,k) は候補の一員（DON_BOX は箱レベル記録・返り値は原始手化）／commit 消化は kind="commit"・sig=返った原始手・groups 無し／戦闘・対話窓は kind="window" |
| `tests/test_defense_box.py` | **基盤健全性**（`cpu_infra`）。**マクロ手化 P4-c＝防御箱 v1 の契約**: `defense_battle_need` の算術（atk≥def で通る・counter_buff 考慮・戦闘外は None）／`defense_box_prune` は D1'（印字総量<need→SELECT_COUNTER 全落とし＝素通しのみ）・D2'（need==0 の窓で払わない）を SELECT_COUNTER+PASS のみの窓に限って適用（混在窓・印字0混入は不変・PASS は常に残る）／adapter の defense_box seam は実盤面 m2@58（総量不足のユーザ裁定点）で ON=素通しのみ・OFF（既定）=従来のまま |
| `tests/test_plan_lethal_gen.py` | **基盤健全性**（`cpu_infra`）。**V7 リーサル族生成器の契約**（`tests/scripts/plan_lethal_gen.py`）: `lethal_exit` は「今ターンに name が勝ち切る」時だけ勝利済み manager を返す（相手勝ち・手詰まり・台本が TURN_END を返すターン跨ぎは None）／`cut_at_group` はシャード境界で組（win+負例の可変行数）を割らない |
| `tests/test_move_audit.py` | **基盤健全性**（`cpu_infra`）。**手の監査 段1の判定規約**（`tests/scripts/move_audit.py`）: カテゴリ分類は `dialog` が MAIN_ACTION にも付くため**効果の対話種別だけ**を効果選択に落とす／迷いは**1位と2位の Q 差**（`q_margin`）で見る（`q_gap`＝打った手 vs 最良手は CPU がほぼ常に最良 Q を選ぶため中央値0で指標にならない・実測）／L1 単独の不一致は容疑者にしない（4割に出て絞り込めない）／欠測は疑いに数えない／優先度は three_way > policy_low > off_top_q > toss_up。加えて**トレースはグローバル乱数を消費しない**（`_fill_trace`）＝観測が対局を変えると段2が `(seed, 決定番号)` で局面を復元できなくなる |
| `tests/test_arena_breakdown.py` | **基盤健全性**（`cpu_infra`）。**対面別内訳の割当規約**（`arena_breakdown`）: `games=[wa,wb]` があれば候補が実際に握ったリーダーへ局単位で割り当て、無い古い台帳は score を半分ずつ割る／対面は席順を無視した組で数える／シャード間の seed 重複は落とす。実対局は回さない純集計の固定 |
| `tests/test_arena_resume.py` | **基盤健全性**（`cpu_infra`）。**再開可能アリーナ**（`tests/scripts/arena_resume.py`・v25）: 台帳 jsonl の読み戻し＋残り seed 抽出（計画順・重複なし＝10分制限下の分割実行の土台）・最終判定は全ペア消化後にのみ出る（部分結果で promoted を出さない）・判定規約（勝ち数0..2→ペア水準0/0.5/1 正規化・wr≥frac かつ CI下限>0.50）が arena_gate と同値。実対局は回さず高速固定。候補席にだけ機構を与える A/B の seam（`--cand-macro` 等・席別 seam の検査は `test_promotion_gate.py`。補償層系の cand フラグは純正AZ化 2026-08-25 で削除） |
| `tests/test_arena_merge.py` | **基盤健全性**（`cpu_infra`）。**分散アリーナ台帳のマージ**（`tests/scripts/arena_merge.py`・2026-09-01）: シャードに割った台帳の合算が単一台帳の判定と一致する（勝率も CI も）・void は母数から外して件数を残す・**seed 衝突を黙って畳まない**（同じ対局の二重計上は CI を不当に狭める＝帯設計のミスを検出して落とす）・promoted は wr≥frac かつ CI下限>0.50 で `arena_resume` と同規約。実対局は回さず合成台帳で高速固定 |
| `tests/test_residual_dig.py` | **基盤健全性**（`cpu_infra`）。**残ドン掘り seam**（`LearnedEngine(residual_dig=True)`・対照生成の腕A・2026-09-02）: serve 既定（None）は挙動不変（同一 seed で同一手・events 空）／`_is_dig_card` は**構造判定**（CHARACTER・cost1・ON_PLAY に DRAW・コストに RETURN_DON）でサトリ/シュラを真・ドローしないコスト1やコスト≠1 を偽（全カードで真⇒cost1 キャラを機械確認＝カードID非依存）／seam 有効でも合法手が返り、差し替え時は PLAY かつ `residual_dig_events` に発火が記録される |
| `tests/test_deck_dig.py` | **基盤健全性**（`cpu_infra`）。**掘りカード差し込み合成デッキ**（`tests/harness/deck_dig.py`・2026-09-02）: 紫を含むリーダー（テーマ外で合成では0枚のクロコダイル等）に 8 枚差し込まれ 50枚・同名4枚以下を維持／テーマで既に12枚のエネルでも上限を壊さない／紫を含まないリーダーには差し込まず不変／builder は決定論で run_game 契約（leader, cards, leader, cards）を返す |
| `tests/test_residual_activate.py` | **基盤健全性**（`cpu_infra`）。**残り起動 seam**（`LearnedEngine(residual_activate="low"\|"high")`・対照生成の腕A2・2026-09-02）: serve 既定（None）は挙動不変／`_pick_attach_target` の方針（low＝攻撃できる最低パワー・無ければ全体最低／high＝最高／同点は uuid 順／候補なし None）／`_leader_has_don_ramp` はテキスト構造語（エネル真・シャンクス偽）／h1@2 掘り分岐 turn2 で起動手→適用→付与対話で low 方針がサトリを選ぶ→適用→付与ドンが乗りドンデッキ0（起動→付与の1周がエンジン上で通る） |
| `tests/test_net_vocab_pinning.py` | **ネット付属 vocab**（`value_net.vocab_ids`・2026-07-15 索引ズレ事故の恒久対策・`docs/reports/net_vocab_pinning_20260715.md`）: カードDB増加で `build_vocab`（card_id ソート）が途中挿入され学習済み Emb/EffF 行との対応が破壊された事故（既存371枚+2ズレ＋新カード範囲外クラッシュ）の回帰を直接見張る**必須テスト**＝同梱 gen2〜5 の vocab_ids 保持／既定エンジンの訓練時 idx 復元（PRB01-001=2282）・新カード UNK／学習側拡張（`extend_to_vocab`）の append-only・既存盤面の出力恒等・EffF 行補充／vocab_ids 無し＋行数不一致の明示エラー／save-load 往復 |
| `tests/test_neff_default.py` | **出荷既定 CPU＝N系 c10 の serve 配線契約**（2026-09-03 採用・`docs/reports/c10_adoption_20260903.md`）。**必須**（壊れると実プレイの CPU がクラッシュ／黙って別ネットで打つ）。固定＝`LearnedEngine()` が同梱 `neff_c10.npz` を N系として読む（`vnet`=`NEffValueAdapter`・`pnet` 無し・`priors_override` あり・符号化 v12・出口ヘッド無し）／同梱 npz の **vocab_ids が gen15 系譜と同一**（N系の訓練 vocab）／vocab_ids 無しの旧 N系 npz は同梱既定の vocab_ids へフォールバック（現行 DB ソートには落とさない）／アダプタの `predict`・`predict_exit` が `NEffNet.value` と一致／priors が合法手上の確率／`decide` が合法手で同一 seed 決定論（既定と明示パスで同一手）／表・重みはプロセス内共有だが `vnet` はエンジンごと別インスタンス／G15 ペアの明示ロードは G系配線のまま（ロールバック先）／`is_neff_npz` の G/N 判別 |
| `tests/test_n_rel_feat.py` | **基盤健全性**（`cpu_infra`）。**NRel P0 符号化**（`opcg_sim/src/learned/n_rel_feat.py`・`docs/n_attention_plan.md` §2）: 形状（tokens 22×S・rel_om 16×6×R・rel_oo 16×16×R・extra）／**組**＝h2 turn 6 で 神の裁き（KO≤3000）単独は囚人 6000 に届かず（gap +0.30）、ガンマナイフ（−5000）との自×自は届く（gap −0.20・feasible・リーダーは対象外）／しきい値＝ゴムゴムの雷（KO≤6000）はバギー 6000 に届き、キャラ限定の除去はリーダーに届かない／**条件の充足はエンジンの真偽**（バレット「ドン 8 枚以上」が turn 4 で偽・ウタ「10000 以上がいる」が turn 12 で真）／1c 登場ドローの戻すドン=1・エネルの起動が合法なら leader_act_avail=1／**v13 = v12 + EXTRA_DIM の append-only**（先頭 94 列が bit 一致）／encode_rel < 10ms |
| `tests/test_n_record_v2.py` | **基盤健全性**（`cpu_infra`）。**dump v2**（`n_record_gen --dump-v2`・NRel P1）: 行が符号化 v13（94+29）・tokens float32 [n,22,S]（float16 は境界で反転するため不可）・候補ごとの主体/対象の 22 枠 index（pol_si/pol_ti・無ければ −1）を持つ／main 窓の候補に主体が枠にあるものが存在／dump 1 行から `relations_from_dump` で R を再計算できる／v1（既定）は tokens 無し・scalars 94 のまま。生成器を in-process で 1 局（sims 4） |
| `tests/test_n_rel_grad.py` | **基盤健全性**（`cpu_infra`）。**NRel 本体（Stage A・`opcg_sim/src/learned/n_rel.py`）と訓練器（`tests/scripts/n_rel_train.py`）**: 手書き backward（value・policy）が中心差分と一致（|grad|>2e-3 のエントリで相対誤差 <5%・見本 h2/h5/h6 の実盤面 6 点）／forward の形状・決定論・空枠（PAD）不変性／save→load で value/policy が bit 一致・`is_nrel_npz` が N系 c10 の npz と衝突しない／`relations_batch`（一括）が参照実装と 22 盤面で bit 一致／**切り分け（ablation・2026-09-05）**: `ablate={"rel"}`（関係 R を 0）・`{"opp_pool"}`（相手デッキ知識の列を 0）が forward の入口で遮断＝遮断した入力の変化に不変・他は効く・save→load で復元・遮断ありでも数値勾配一致 |
| `tests/test_n_rel_serve.py` | **基盤健全性**（`cpu_infra`）。**NRel の serve 配線**（P3・`cpu_learned` × `n_rel.NRelValueAdapter`）: `LearnedEngine(value_path=<NRel npz>)` が NRel を判別（`vnet`=アダプタ・`pnet` 無し・`priors_override` あり・v13・出口ヘッド無し）し既定 c10 は不変／葉価値は `predict_state`（盤面から直接）／priors は合法手上の確率／`decide` は合法手で同一 seed 決定論・表と重みは席間で共有／レイテンシは情報出力（sims 32・c10 比 約 2.7 倍・2026-09-04 実測）／**R 遮断ネット（切り分け a1・2026-09-05）**は `encode_state` が関係の計算を省いて零の R を返し、葉価値は「R を計算して渡した value」と一致 |
| `tests/test_cpu_learned.py` | **学習型CPU本番配線**（既定＝gen11(符号化v8・2026-08-03採用＝gen10＋自場集約の純対称化・蒸留アンカー付き順位学習×α0.3補間)・温スタート検証は v1(gen2) を明示ロード。`opcg_sim/src/core/cpu_learned.py`／`opcg_sim/src/learned/`）: 合法手・decide_client ルーティング・seed 決定論・席別エンジン（net-vs-net 等価）・**符号化/行動特徴の訓練時ドリフト検知（v1/v2）**（`tests/harness/{rl_encoder,opcg_action,rl_net,az_policy,az_mcts_tree}.py` は本番 `opcg_sim/src/learned/{encoder,action,value_net,policy,mcts}.py` への委譲shim＝TEST_E/TEST_A は本番と同一オブジェクトでドリフトは構造的に不可能・退行検知として存続。`tests/harness/opcg_game.py` は本番 `adapter.OPCGGame` の薄い継承＋研究専用 `new_game` のみ追加）・選択対話の併合（CONFIRM_OPTIONAL accept/decline・up-to ライフ追加・**ARRANGE_DECK の並び替え/上下選択**・position キー）・**ルート等価手マージ**（同名複製の訪問数分裂で PASS に負ける実害の反転ケース＋複製なし恒等）・トレース記述（decline の accepted 明示・dialog 種別）・**符号化世代 v2**（リーダー付与ドン特徴＝v1 では不可視・v1 出力不変・npz 入力次元からの自動判別）・**温スタート拡張**（v1→v2 の重み拡張が恒等＝拡張ネット×v2符号化 == 出荷×v1符号化・policy も恒等・縮小拒否・版差は scalars_dim のみが seam＝将来版に同一コード対応） |
| `tests/test_value_net_leader_slots.py` | **ValueNet のリーダー条件付け専用枠**（`lead_slots`・`docs/reports/lc_value_net_plan_20260708.md`）: `to_leader_conditioned()` の恒等性（追加ゼロ行＝拡張直後は旧net予測と一致）・二重適用拒否・save/load 往復（旧形式npz=lead_slots無しの後方互換込み）・`expanded()`（enc版温スタート）との直交併用・解析勾配=数値微分一致・**リーダーIDのみで決まる合成ターゲットを lead_slots=2 だけが fit できる**回帰 |
| `tests/test_effect_features.py` | **EffFeat＝効果セマンティクス特徴テーブル**（`opcg_sim/src/learned/effect_features.py`・`docs/reports/effect_semantics_v3_plan_20260708.md` §1）: 決定性（2回構築一致）・PAD行ゼロ・次元・効果持ち全カードの能力ブロック非ゼロ・実カードのスポットチェック（OP03ナミ=VICTORY独立枠＋資源条件／OP11ナミ=ON_OPP_ATTACK+2kバフ+HAS_DON+手札コスト／コスト操作とパワーバフの status×値スケール分離／ATTACH_DON全体センチネル／印刷キーワード・カウンター値・種別の静的ブロック） |
| `tests/scripts/replay_reeval.py` | **マーク付きリプレイ再評価CLI**（`opcg-replay/v1`のframes+marksから各マーク直前フレームの盤面を復元し候補ネットにdecideさせ「人間の指摘どおり手が変わるか」を検証＝ネット改善の人間フィードバック回帰。全編再生は山札覗き効果＋ドン経済で漂流するため局所復元方式を採用。カウンター系マークは直前の PASS（ブロッカー段の見送り等）・RESOLVE_EFFECT_SELECTION（【アタック時】効果の選択）を遡って攻撃宣言に着地し、宣言〜マーク間の記録応答を再生して復元する。`.json.gz` 直読み可） |
| `tests/scripts/defense_rate_probe.py` | **防御応答の守り採択率 計器**（v5 R1 調査・`docs/cpu_v5_plan.md` §3-R1）: 既定 net で自己対戦し、防御応答（SELECT_COUNTER/BLOCKER）局面の「守る(非PASS)採択率」を net argmax／温度1期待（データ挙動）／L1-hard（良質目安）の3系統で集計。温度延長が守りを過剰注入したか（R1）を切り分ける読み取り専用計器。**実測結論: R1 否定**（net argmax はむしろ L1 より守らず・温度延長は L1 水準への補正＝過剰注入でない） |
| `tests/scripts/defense_rate_probe.py` | **防御応答の守り採択率 計測CLI**（v5計画 §3-R1 の調査計器・読み取り専用）: 既定 net（gen4）で自己対戦し、SELECT_COUNTER/BLOCKER 局面の守り率を net argmax／温度1期待（データ挙動）／L1-hard（良質目安）の3系統で比較。「守りすぎ」の原因が防御温度延長の過剰注入か net 体質かを切り分ける（24局630局面で否定＝netはむしろ守らなさすぎ・温度延長は補正的） |
| `tests/scripts/clock_error_by_leader.py` | **時計誤差の対面別分解CLI**（v4監視 diagnostics・`docs/reports/v4_adoption_20260712.md` §3/§6）: batch.npz（スキーマv2）の局面を自リーダー/対面ペアでグループ化し、残りターン補助ヘッドの MAE・bias（ターン換算）を分解。平均誤差が隠す対面別の系統偏りを可視化＝§5.5-2（自デッキ残特徴）の切り分け材料。読み取り専用 |
| `tests/scripts/mark_gate.py` | **v4 マーク回帰ゲートCLI**（`docs/reports/v4_adoption_20260712.md` §5＝v4採用ゲート）: `tests/fixtures/replays/` の2局×16人間マークを復元し、challenger / baseline（既定=v3）ネットで各Kシード decide→「人間指摘方向率」を比較。判定＝F4代表6件の過半で改善 かつ 既存正着ガード3件（g1@12/@24・g2@20）非退行で PASS（exit 0）。v3 vs v3 で「改善0/6・非退行OK・FAIL」を確認済み（＝ゲート感度の基準線） |
| `tests/scripts/promotion_gate.py` | **昇格ゲートCLI**（v6 柱①・`docs/reports/v5_adoption_20260715.md` §4-1）: candidate ネットが現行 best（未指定は出荷既定）に段階式 arena（stage1=12ペア24局で勝ち越しのみ継続 → stage2=累計50ペア100局・勝率≥0.55）で勝った場合のみ昇格 PASS（exit 0）。**`--anchor`（v7）**: 対best 通過後に固定アンカー（空文字=出荷既定）へ非退行（勝率≥0.5・`anchor_decision`）を追加要求＝**血統過適合**（対best 連鎖では昇格するが祖先に直接負ける偽の前進・v6 実測 r99 対gen5 0.33）を弾く。席入替CRNペア・multiprocessing 並列。learner（`pd_learn.py --promote-every`）が candidate(p3ckpt)→best(p3best) の昇格判定に呼ぶ＝「run をいつ止めてもベストが残る」ピーク一過性対策。生成側（`pd_gen.py --gen-from best`＝既定）は p3best があればベストから生成 |
| `tests/scripts/counterfactual_referee.py` | **反実仮想レフェリー**（教師CPU構想の核・読み取り専用）: 1つの決定点で root の合法手を**枝刈りなしで全列挙**し、同一世界線（隠れ情報を決定化で再サンプル・全 root 手で共有＝CRN）×終局までのロールアウト（固定教師ネット・既定 gen5＝学習でドリフトしない錨）で「その1手の因果効果」を数世界で測る。何万局の独立平均でなく対照実験＝少数試行で判定。出力=手ランキング・勝ち数・**勝ち方の質**（残ライフ差・決着ターン）・人間指摘との一致（`REFEREE_RESULT`）。**--plans（プランモード）**: 手順列を固定適用してから比較＝root prefix 比較がロールアウト役の盲点に汚染される問題（g3@64 で実証）を回避。`--plans auto`＝ターンプラン自動列挙（v8 柱A・手番遷移を終端規約に DFS＋並べ替え等価除去＋ビーム＋コミットメント別ラウンドロビン縮約・切り捨ては必ずログ）。`--true-board`＝盤面をフレーム復元でなく記録全手順の再実行（`state_at_action`）で用意。`--band`＝同価値バンド v2（v8 柱B・対判定: 同一世界での勝敗割れの正味差 ≥3 のみ断定・未満はライフ差 < band で ≈ 表示）。初回検証は `docs/reports/counterfactual_referee_20260716.md`／真盤面判定は `referee_true_board_20260716.md`。--worlds/--sims/--net 可変  **`--plans` の落とし穴2件（v46・2026-08-08 実測）**: (1) 攻撃を含む複数手順は表現できない——1手適用するたびに手番が相手の防御窓へ移り次の手順が `legal_actions` に現れず「不成立」になる。(2) **自分の効果解決ダイアログが保留された位置でプランを終えると結果が壊れる**（単手順 `ACTIVATE_MAIN:OP16-056` は 0/32・`>RESOLVE_EFFECT_SELECTION>ATTACK:…` まで含めると 18〜19/32）。手順は対話が保留されない位置で終えること |
| `tests/scripts/coach_gate.py` | **コーチゲート（診断計器）**（v9 §4 で mark_gate 後継として導入・v16 総括 §4 で合否ゲートから診断へ降格）: レフェリー検証済み決定点の**同価値バンド初手集合**への所属で出荷既定と候補を同条件 decide 比較（`COACH_GATE_RESULT`）。**既定 seeds=16**（v22 で 5seed の点別差の信頼下限が 0.63＝ほぼ何も言えないと実測・`min_reliable_delta` を出力に併記して bar 未満の増減を「治った/壊れた」と読ませない・`docs/reports/coach_gate_variance_20260729.md`）。**既定プロファイル v2**（2026-07-28・`docs/reports/coach_gate_v2_20260728.md`）＝gen7 実対局マーク由来の**14点**（v3・2026-07-30 修正後エンジン worlds16 で全点再裁定＝`docs/reports/verified_v3_20260730.md`。m1@3 は 2026-07-30 に判別不能で取り下げ→**2026-08-03 再採録**＝修正済み評価（def_temp0.7＋マージン）でウタが両指標最上位・gen10 の行動欠陥（非発動イワンコフ 5/8）を実測・accept はウタのみ。m4@2 は同日ユーザ最終裁定で accept＝イワンコフ（効果発動＝手札 5→5 の実質ノーコスト入替）＝非退行ガード化）・4対局・両対面方向（ガード4点＋gen7 が 0.00 の改善ターゲット7点＝無駄低コスト展開/リーダー付与過小評価/無駄カウンター）。旧 g3（単一対局・gen4期・7点）は --profile g3 で診断用に存続 |
| `tests/scripts/coach_sweep.py` | **コーチングスイープ**（v8 柱C・`docs/cpu_v8_plan.md` §3・読み取り専用）: 録画1局の指定範囲の決定点（MAIN_ACTION＋**防御窓 SELECT_COUNTER/SELECT_BLOCKER**）ごとに真盤面（`state_at_action`）→ターンプラン自動列挙（柱A・`enumerate_turn_plans`・防御は素通し/カウンター単発/重ね切り）→実際に打たれたプラン（記録から `actual_plan_keys` で復元・列挙漏れなら必ず追加・強制手=選択肢1はスキップ）を同じ CRN 世界線で判定し、最良プランとの差が同価値バンド v2（柱B・`--band`・対判定）を超えた決定だけを「損失」として報告（`COACH_RESULT` 行）。**`--comeback τ`（既定0.7）**: 飽和負け（最善でも勝ち≤1）の決定は上位＋実プランを世界数×4＋相手温度で再判定し捲り率で採点（柱B+）。教師の答え合わせは少数局面を深く＝`--range`/--worlds/--sims でコスト制御 |
| `tests/scripts/referee_labeler.py` | **レフェリー再ラベル・パイプライン**（v9 フェーズ1・`docs/cpu_v9_plan.md` §2）: ①両席 learned(gen5) の記録つき自己対戦（`record_selfplay_descriptor`＝decide の rng 隔離で scripted 再生可能）→②採掘（同一パスの読み取り専用 observer・**miner v3**: 効率盲点＝**上位M(4)手**の1-ply後 value spread<ε（全手 spread は明確な悪手混入で@64型を取り逃す）／飽和負け＝value<しきい／**反例（disagree）＝policy top1 が 1-ply value 最善と食い違い policy が明確に劣る手を推す点**（@82/@68/@93 型の policy 矯正点を能動採掘・採掘はノイジーでよく最終ラベルはレフェリーが付ける）。**sat/disagree/blind の3カテゴリ round-robin**＝敗者側終盤の sat 洪水でどれかが飢える構造の防止・1局上限K点）→③真盤面再生（**採掘時の actor/pend 種と照合**＝index ズレで別局面を黙ってラベルする事故の防止）＋プラン自動列挙＋CRN 対照評価（飽和は捲りエスカレーション）→ policy 教師＝バンド上位プラン初手 multi-hot・value 教師＝z=2·wr−1（捲り率）。出力=バッチスキーマ v2＋meta(source="referee_label"・miner版数。worker はワーカー運用時に w1 等へ上書き)＝ pd_learn 直結（staleness は source で免除）。教師ネット gen5 固定＝外部の錨 |
| `tests/scripts/divergence_probe.py` | **乖離診断プローブ**（v12・読み取り専用）: 候補 vs 既定の記録対戦（席別 engine・rng 隔離）→候補敗北局の候補手番を真盤面再生し両者 decide の equiv キー比較で乖離点を採掘→レフェリー（gen5 錨・プラン列挙＋バンド判定・捲りエスカレーション）で cand_bad/best_bad/両可/両外 に裁定→行動種遷移×局面相の悪癖カタログ（`DIVERGE_RESULT`）。コーチ PASS↔アリーナ敗退（v11/v12 実測 19/48）の構造乖離の特定計器。wall-clock 予算の安全弁つき |
| `tests/scripts/ref_finetune_smoke.py` | **v9 フェーズ2 スモーク**（`docs/cpu_v9_plan.md` §3 の当たり付け・読み取り専用）: v9-label 全枝の教師バッチを収集→決定単位 train/val 分割（固定seed）→同梱ベース（`--base`・既定 gen6）温スタートで value/policy 微調整（LR 候補比較・policy-smooth 床＝未評価ハードゼロの緩和）→前後評価（val の policy 支持一致率・KL・value MAE/corr）。同梱ネットは書き換えない（--out 指定時のみ候補 npz 保存＝後続ゲート運転用） |
| `tests/scripts/ensemble_probe.py` | **value アンサンブル A/B プローブ**（v15・読み取り専用・`docs/reports/cpu_v15_ensemble_power_20260726.md`）: 2つの ValueNet の予測を重み平均する薄いシム（`AvgValueNet`＝predict/predict_with_aux のみ・符号化版の不一致は起動時に検査）を `LearnedEngine.vnet` に差し、葉評価だけを差し替えて既定と CRN 対戦（policy prior・探索・root 読み出しは不変）。**実測: gen6＋scratch512 は通算300局 0.517 で棄却**（帯ごとに 0.442-0.583 と振れる＝60-120局では 5pt 差を判別できないことの実例） |
| `tests/scripts/search_config_probe.py` | **探索設定 A/B プローブ**（v14・読み取り専用・`docs/reports/cpu_v13_v14_plateau_20260726.md`）: ネットを固定したまま serve の探索設定（`LearnedEngine` のエンジン別上書き sims/c_puct/root_frac/root_gap）だけを変えた席と既定の席をCRN ペア（同 seed・席入替）で対戦させ勝率と 95%CI を出す（`SEARCH_PROBE_RESULT`）。null 条件（同一設定同士＝0.5）で計器サニティも取れる。**実測は全条件で有意差なし**＝探索設定は gen6 踊り場の原因ではない（c_puct の山型は小標本ノイズ・確証120局で 0.458） |
| `tests/scripts/value_scratch_train.py` | **大容量 value フルスクラッチ訓練**（v13・`docs/reports/cpu_v13_v14_plateau_20260726.md`）: 蓄積コーパス（root／`--use-child` で子盤面併合）でValueNet を初期値から訓練（hidden/d_emb 可変・policy は gen6 凍結・vocab と効果表は gen6 引き継ぎ＝index ズレ防止）。**実測: hidden 512 で held-out MAE 0.338→0.242（−28%）でも対gen6 アリーナ 0.425**＝ラベルへの当てはまりと実戦力の非相関を示す計器 |
| `tests/scripts/dense_selfplay_gen.py` | **密ラベル自己対戦生成**（v16・v19 で `--matchup a:b` 追加＝ユーザ実デッキの固定対面生成。`_W["game"]` の new_game 差し替えのみ・seed偶奇で席入替）: 既存の p3 生成コア（`p3_run.selfplay_shard`→`p3_loop.selfplay_game`）を無改修で駆動し、gen6 の自己対戦から**全決定点**の (z, q_root, turns_left) を集める（実測 **112〜122 行/局**・q_root 有限率 1.00・符号化 v5）。教師密度が p3期 102.9 点/局 → v9 レフェリー期 2.52 点/局 へ 40.8 倍崩壊しており、gen2→gen5 を支えた密度レジームを gen6 以降で一度も再現していない、という仮説の検証用。**閉ループにしない**（生成ネットは gen6 固定・判定は外部の `arena_gate.py`）＝v7 の閉ループ（15,776局・246更新で昇格ゼロ）の機序を再演しないため。シャード単位で npz 保存＝途中終了しても既存分をそのまま学習に使える。**v25**: `--base` にパス接頭辞を許可（温スタート拡張ネットでの生成）。**v26**: `--def-force-eps`（ε強制防御＝分布の新規性を作る主レバー・`p3_loop._forced_defense_index`）と、meta の `hand_counter_mean`（手札カウンター保有の平均＝ε が実際に効いているかの走行中モニタ。実測 ε=0.6 で 0.115→0.080） |
| `tests/scripts/dense_finetune.py` | **密ラベル追い学習**（v16・`dense_selfplay_gen.py` の対）: 密コーパスで gen6 value を追い学習する。`ref_finetune_smoke.py` と分けるのは、あちらの `collect_ref_batches` が q_root/turns_left を読まずラベルが勝敗単独へ退化する＝v16 が検証したい学習仕様（混合ラベル＋残りターン補助＋任意の distill アンカー）そのものを再現できないため。**policy は学習しない**（v12 確定＝policy 微調整は1エポックでも対gen6 を 0.33 に落とす）＝出力は value.npz ＋ base の policy.npz コピー。保存後に **vocab_ids が base と一致すること**を assert（2026-07-22 の index ズレ事故の再発ガード） |
| `tests/scripts/arena_gate.py` | **固定N・帯層別アリーナ判定器**（v16・判定専用の正本・`docs/reports/cpu_v15_ensemble_power_20260726.md` §2）: 一次スクリーン（既定48ペア・floor 未満で早期棄却）→本判定（既定 **400ペア=800局**・4帯に分けて seed 空間を stride 10万ずつ離す）で、**勝率 ≥ 0.55 かつペア水準95%CI下限 > 0.50** の2条件で PASS（`ARENA_GATE_RESULT`・帯間ばらつき `band_spread` 併記）。`promotion_gate.py` は昇格運用の逐次ゲート（stage1 24局）で固定N・帯層別・ペア水準CIを出さないため分離。対局実行は promotion_gate の席入替CRN（`_init_pool`/`_play_pair`）を import して再利用 |
| `tests/scripts/move_audit_shard.sh` | **手の監査のシャード実行**（セッション分割・2026-08-17）: `move_audit_shard.sh <シャード番号> [局数] [カテゴリ毎の点数]` で **seed 帯ごとに段1→段2を完結**させる。帯は `500000 + シャード番号×1000`＝決定論なので帯が違えば判断点も重複しない＝**ワーカー間で入力ファイルの受け渡しが要らない**（作業台帳は実行環境の巻き戻しで消えるため、結果はチャットへ貼り戻す運用）。段2 は層化抽出（`--per-category`）でカテゴリ別平均を作れる形に配る |
| `tests/scripts/move_regret.py` | **手の監査 段2＝容疑者だけ反実仮想で regret 実測**（2026-08-17）: 段1 の `(seed, 決定番号)` から**対局を再生して判断点の直前の manager を複製**し（同じ seed に複数の容疑者があっても再生は1回）、選択肢を**同一世界・共通乱数**で終局まで打ち分けて regret＝最良の勝率−打った手の勝率 を出す。世界の作り方は `serve_referee` と同規約（手札・ライフ・場は真値のまま**山札の並びだけ**を世界 seed で振る）。**再生は監査時の条件**（`audit_sims`/対面/デッキ/ネットを行に埋めて持ち回る）でないと対局が分岐して別の判断点に着地する（実測: sims を変えたら席までずれた）＝ロールアウトの探索数だけ別ノブ。全選択肢同率は `saturated`（判別不能）として集計から外し件数を出す |
| `tests/scripts/plan_dom_gen.py` | **P1/P2 支配ペアのターン出口教師（系統3・π非依存・A段 2026-08-20）**: 自己対戦の自席メイン判断から、V1=浮ドンを「死に先」（レスト済み・ドン条件なし・相手ターン常在なし）へ付与 vs アクティブな攻撃可能ユニットへ付与、V4=同一攻撃者で「付与→攻撃」（正順・P1）vs「攻撃→付与」の対を `plan.scripted_plan` で実現・`plan.execute_plan` で出口化（serve と同一規約）。ラベルは順位のみ（good=+0.5/bad=−0.5）＝審判ロールアウト無し・V^π 汚染なし。出力 `plandom_*.npz`（plancf 互換）→ `exit_head_finetune.py --head turn --globs "plandom_*.npz"` |
| `tests/scripts/plan_exit_anchor_gate.py` | **段3裁定アンカーのターン出口検査（系統1・A段ゲート①）**: 裁定 fixture（`move_audit_stage3.json`）の best_correct/cpu_correct から「上位に来るべき手 vs 下位の手」の対を作り、監査時条件の決定的リプレイで局面復元→各手を打って閉じた出口盤面を候補ネットの `predict_exit("turn")` が正しく並べる数を出す。**訓練には使わない完全ホールドアウト**（系統3のみで学習・リークゼロ） |
| `tests/scripts/plan_cf2_gen.py` | **ターン出口教師・系統2＝プラン構造化審判の反実仮想**（B段・2026-08-20）: 各判断点の候補プラン（policy argmax/温度＋構造化 `struct_intents`・≤6本）を `execute_plan` で出口化し、**両席とも候補ヘッド搭載エンジンの審判**（旧 π_plan＝plan_readout serve 配線は純正AZ化 2026-08-25 で削除・立案APIは plan.py 計器を直接使用）で決定化K世界（プラン間共有=CRN・相手手札もサンプル＝手札真値固定の穴を教師では踏まない）を終局まで打って z=2·wr−1 をラベル。全プラン同率の決定点は捨てる。`--drift N` で旧審判（plan OFF）とのラベル乖離 \|Δz\| を測る＝V^π汚染の定量化。出力 `plancf2_*.npz`（plancf互換） |
| `tests/scripts/plan_lethal_gen.py` | **ターン出口教師・系統3＝V7 リーサル族**（π非依存・2026-08-21）: 自己対戦の自席メイン判断のうち**防御込み台本リーサル**（`lethal.lethal_distance(..., defend=True)==0`＝相手が実カウンターで最大防御しても今ターン勝ち切る）が立つ点で、リーサル台本を勝利まで実行した出口（+0.5）vs 取らない出口（pass=即閉幕／dump=リーダー全付与閉幕・段3裁定 #15 の実挙動、各−0.5）の論理支配ペアを作る。防御込み検証は生成側が相手手札を読むがラベルの保守化であり符号化特徴には入れない。B2a/B2b/B2c 切り分け（2026-08-21・系統2の自己対戦結果ラベルはアンカーと系統的に衝突＝どの混合比でも 6/9→3/9）を受けた系統3拡張の第1弾。出力 `planlet_*.npz`（plandom 互換） |
| `tests/scripts/macro_p0_probe.py` | **マクロ手化 P0＝読みの浪費の実測**（2026-08-24・読み取り専用）: gen15 既定の自己対戦から全判断点を観測し、窓型別分布（細部窓の割合）・serve 実分岐数・ドン付与の順序重複率（同一配分への原始経路の倍率= targets^k / C(targets+k-1,k)）を出す。初回実測(8局1240点): 細部窓42.3%・メイン窓の最頻手はドン1枚付与(46%)・順序重複 中央値5.3x/最大9756x＝木の候補マクロ化（P1）の効果測定の基準線 |
| `tests/scripts/macro_equiv_probe.py` | **マクロ手化の等価性監査**（2026-08-24・読み取り専用）: 自己対戦の全メイン窓で箱（DON_BOX 配分形/アタック形）の3契約を全数検査——A 適用等価（箱1手の終状態＝等価な原始手列の逐次適用。乱数は両経路同一 seed）／B 被覆（ONの候補から到達できる先頭原始手 ⊇ OFF の原始手）／C 出力合法（don_box_first_primitive の返す手が OFF 合法手に含まれる）。指紋は盤面そのもの（勝敗・ライフ/手札/場/ドン・場の(ID,レスト,付与)集合・保留）＝符号化のバグと独立 |
| `tests/scripts/defense_audit_probe.py` | **防御監査 P4-a＝防御窓の結合判断の質の実測**（2026-08-24・読み取り専用）: 自己対戦の被攻撃窓（SELECT_BLOCKER/SELECT_COUNTER 入口）で結合防御の変種（素通し／守り切る最小の札組／+1000 余裕・lethal 防御台本と同じ印字値算術）を台本適用→戦闘出口 value で順位づけし、エンジン実選択（窓の根畳み）との乖離率と型（過少/過剰防御）を出す。マクロ手化 P4（防御箱＝要点総量の候補化）の設計根拠となる入口計測。教師には使わない |
| `tests/scripts/plandef_gen.py` | **D族＝防御の支配ペア教師**（π非依存・P4-b・2026-08-24）: 防御窓で D1（印字カウンター総量<必要値→素通しが任意の支払いを支配＝m2@58型の一般化）と D2（最小で守り切れる札組が+1枚を支配＝過剰防御の矯正）の対を作り、戦闘解決後の出口盤面を ±0.5 で出力（`plandef_*.npz`・defcf互換）。学習は battle 出口ヘッドの再訓練（--replace-head・既存 defcf を混ぜて較正保持）。設計根拠は defense_audit_probe の実測（乖離21%・過剰防御が主・現行 value は防御差をほぼ感じない） |
| `tests/scripts/d1_branch_probe.py` | **ヘッド差し替え退行の枝別診断**（P4-b・2026-08-24・読み取り専用）: coach_gate で退行した防御裁定点（既定 m1@14 / m2@58）の合法手を `resolved_branch_values`（serve と同一の戦闘箱の物差し）で既定ネットと候補ネットの双方で採点し、枝ごとの出口値と順位を並べて表示する。ゲート FAIL の「どの枝の順位がどう崩れたか」を1分で特定する計器。初回実測（2026-08-24）: cand_D0/cand_D1 とも素通し1位が反転し枝間差が ~0.005 に平坦化＝出口ヘッド再学習ルート打ち止めの根拠（`docs/reports/2026-08-24_p4_defense_verdict.md`） |
| `tests/scripts/move_audit.py` | **手の監査 段1＝安い一次フィルタ**（ロールアウト無し・2026-08-17）: 「CPU が打った手は正しかったか」を全判断点で測るのは高すぎるので、3段ファネルの1段目として**信号だけで容疑者を絞る**。各判断点で CPU の手／**L1 の手**（別評価軸の第二意見）／**policy 事前分布の順位**／**Q の1位2位差**を突き合わせ、**勝敗がほぼ決している点（|Q|≥0.9）は容疑者にしない**（段2 の実測で飽和した判断点は全選択肢 wr=1.000＝何を選んでも勝つ局面だった＝18本のロールアウトが無駄になる）。`three_way`（三者食い違い）・`policy_low`（policy 3番手以下を選んだ）・`off_top_q`（Q 最良でない手を読み出しが選んだ）・`toss_up`（実質同着）を付ける。出力は `(seed, 決定番号)` 付きのjsonl で、段2 は `run_game(stop_after_decisions=決定番号)` で局面を厳密復元してから反実仮想で regret を測る。`--top` で優先度上位だけに絞れる。**値打ちは集計**（カテゴリ別の平均マージン・容疑者率・L1不一致率）にあり、次にどこへ物量を投じるかがデータで決まる。初回実測（2局269判断点）: ドン付与は L1 と**90%**食い違い、防御はマージン0.047／容疑者20% |
| `tests/scripts/plan_dom_gen.py` | **ターン出口教師・系統3＝P1/P2支配ペアの生成**（π非依存・A段・2026-08-20）: 自己対戦の自席メイン判断から V1/V1s（死に先への付与 vs アクティブ付与）と V4（付与→攻撃 vs 攻撃→付与）の対を作り、`plan.scripted_plan` で実現・`plan.execute_plan`（serve と同一規約）で出口化して plancf 互換の npz（`plandom_*.npz`）に落とす。ラベルは順位のみ（±0.5）＝審判ロールアウト無し・V^π 汚染なし。学習は `exit_head_finetune.py --head turn --globs plandom_*.npz` |
| `tests/scripts/plan_exit_anchor_gate.py` | **ターン出口ヘッドの裁定アンカーゲート**（系統1・完全ホールドアウト・2026-08-20）: `tests/fixtures/move_audit_stage3.json` の裁定（best_correct/cpu_correct）から「正しい手 vs 誤った手」の出口順位対を作り、候補 value の `predict_exit('turn')` の正答数を出す。候補は serve と同じ符号化世代へ温スタートして測る。**訓練には使わない**（系統3のみで学習・リークゼロ）。初回実測: gen14 素 value 3/9 → A段候補 5/9 |
| `tests/scripts/arena_merge.py` | **分散アリーナの台帳マージ**（2026-09-01 ユーザ決定「アリーナの分散化」・読み取り専用）: 1セッションで回せるのは24ペア×2条件＝192局程度で、この母数では**世代交代の実力差（実測 +0.04＝約+29 Elo）が CI に埋もれて判定できない**（c9 vs c8 は4本とも 0.521〜0.563 に収まったが、どの1本も「wr≥0.55 かつ CI下限>0.50」を満たせなかった）。生成波と同じくオーケストレータでセッションを分散し、**シャードごとに別 seed 帯**で回した `arena_resume` 台帳をここで合算する。判定規約は `arena_resume.final_result` と同一（`arena_parallel._pair_level_ci` を import＝二重化しない）。seed 衝突は既定で判定を出さずに落とす（`--allow-dup-seeds` で強制可） |
| `tests/scripts/arena_breakdown.py` | **アリーナ台帳の対面別内訳**（2026-08-16 ユーザ決定「どの対面が強いかは記録する」・読み取り専用）: ランダムリーダー帯は総合勝率しか残さず、**特定の系統だけ打てていない**種類の汎化の穴が平均の陰に隠れる。台帳 jsonl（複数シャード可）を読み直し、リーダー別（候補席がそのリーダーを握った局の勝率）と対面別（順序を無視した組）に割り直す。`arena_resume` の台帳は 2026-08-16 から `leaders`/`games` を記録するが、それ以前の台帳でも**seed からリーダー対を再計算**して集計できる（対面は seed の決定論関数）。ただし `games`（2局の内訳）が無い行は score を両リーダーへ半分ずつ割る＝不偏だが分解能は落ちる。seed がシャード間で重複したら二重計上せず落とす |
| `tests/scripts/arena_resume.py` | **再開可能アリーナ**（v25・`arena_gate.py` の chunk 実行版）: 実行環境がフォアグラウンド約10分制限＋ターン終了でバックグラウンド回収のため、800局判定を一度に走り切れない。ペアスコアを jsonl 台帳へ追記し、再実行のたびに未消化 seed から `--max-pairs` ぶん進める＝10分×N回で同一判定を積み上げる。帯設計は `arena_gate.plan_bands`・対局は `promotion_gate._play_pair`・集計は `arena_parallel._pair_level_ci` を import（判定規約の二重化なし）。全ペア消化後の実行が `ARENA_RESUME_FINAL` を出す。**void（2026-08-16）**: 対局がエンジン欠陥で成立しなかったペア（上限手数 MAX_STEPS 等）は score=null で台帳に残し、消化済みとして数えつつ勝率の母数からは外す＝1ペアの失敗でシャードを落とさない。結果には `void` 件数を必ず載せ、全ペア void なら判定を出さない |
| `tests/harness/deck_dig.py` | **掘りカードを保証した合成デッキ**（残ドン掘りの方針対照実験用・2026-09-02）: `deck_synth` はテーマ整合で採るため掘りカード（登場時ドン-Xドローのコスト1・全5種・全て紫）はエネルにしか入らず（12枚）他の紫リーダーは0枚＝ランダム対面で腕Aが空振りする。合成後にリーダーで構築可能な掘りカードを末尾の非掘り札と差し替えて差し込む（同名4枚まで・50枚維持・決定論・両席同規則＝腕A/腕Bは同一デッキ）。母集団は紫を含む34リーダー（`promotion_gate --leaders purple`・`arena_resume --decks synth_dig`） |
| `tests/scripts/dig_cf_breakdown.py` | **残ドン掘りの方針対照の層化集計**（2026-09-02・読み取り専用）: 掘り1回の効果（数%）は1局の勝敗に埋もれるため、`arena_resume --cand-residual-dig`（候補席＝腕A「木が TURN_END を選んだ時だけ、捨てるはずのアクティブドンで登場時ドン-Xドローのコスト1キャラを出す」／基準席＝素の同一ネット）の台帳を **(1) リーダー層**（ドンデッキからドンを追加する効果の有無＝raw_text の構造語「ドン!!デッキから」・12リーダー）と **(2) 掘り発火の区分**（ターン帯・場のドン・ドンデッキ残・カード）で割り直す。全体はペア水準 CI、層内は局単位の Wilson 区間（席入替の対が崩れる旨を明示）。ラベルは勝敗のみ＝人間裁定を教えない。台帳行の `dig`（`promotion_gate._play_pair_detail` が候補席の events を game a/b 別に記録）を読む |
| `tests/scripts/don_refund_audit.py` | **エネル h1@2 のドン経済とリーダー起動の監査**（2026-09-02・読み取り専用）: 残ドン掘りの方針対照で「掘りは報われない」と出た原因を、h1@2（エネル turn1・手札サトリ）から掘る／掘らないの2分岐をエンジンで実際に進めて切り分ける。(1) don‼-1 返還→ドロー→翌ターンのドンフェイズ＋リーダー起動後に**場のドン合計が両分岐で一致**するか（実質無料の力学）(2) 付与対話（自キャラへ〜まで）を学習エンジンの adapter が選択肢として列挙するか（gamestate 既定は0件）(3) `--net` の N 系ネットの探索根統計で ACTIVATE_MAIN の訪問/Q。実測: (1)(2) 成立・(3) **c10 は起動を負に評価**（Q −0.03〜−0.09・512 sims でも同じ＝価値ネットの盲点）。`docs/reports/2026-09-02_enel_leader_activation_audit.md` |
| `tests/scripts/activate_cf_breakdown.py` | **残り起動（腕A2）の方針対照の層化集計**（2026-09-02・読み取り専用）: `arena_resume --cand-residual-activate low\|high`（候補席＝「木が TURN_END を選んだ時に未使用のドン追加起動効果を起動し付与対話を方針で解く」／基準席＝素の同一ネット）の台帳を `dig_cf_breakdown` と同じ規約で割る。全体＝ペア水準 CI・発火/無発火ペア・リーダー層（ドン追加あり/なし）・起動回数／ターン帯／ドンデッキ残／付与先パワー帯／付与先の攻撃可否。台帳行 `act`（kind=activate/attach）を読む。監査 `don_refund_audit` で c10 が起動を負に評価する盲点が出たための勝敗検証 |
| `tests/scripts/deck_synth_audit.py` | **生成デッキの実プレイ監査**（2026-08-15。`--cross` は 2026-08-16）: `deck_synth` のデッキで自己対戦し、**終局しない対局と例外を洗い出す**。既定は全リーダーのミラー1局ずつ（ok/hang=上限手数/timeout=実時間上限/error=例外を集計）。固定ハンニャバル（ステージ0・イベント0）のミラーだけを回していた歴代の測定では通らない経路が実プレイに乗るため、ここでしか出ない実バグを掘れる（初回で3ハング＋2タイムアウト＝`test_noop_activation_loop.py` の4欠陥）。時間上限は `_GameTimeout`＝**BaseException 派生**（エンジンの広い `except Exception` に食われないため）。**`--cross N`**＝ミラーではなく交差対面を N 件（対面の引き方はアリーナ `promotion_gate._leader_pair` と同一＝void の再現条件と揃う）。hang 時は**繰り返した対話の発生元**に加えて**繰り返した手**も出す（対話を伴わないループは発生元が付かない）。`InvariantError` は上限手数のみ hang、適用中の例外は error に分類する |
| `tests/scripts/option_pair_gen.py` | **オプションペア教師生成**（v31）: 固定対面の自己対戦（序盤で打切り `--max-mine-steps`）＋マーク局面シードで「ON_PLAY 持ちPLAYが2枚以上」の点を採掘し、各カードを別枝＋TURN_END 温存枝として CRN ロールアウト（**def_temp 既定0.7**・v32: argmax 防御は「手札を回して即出しする枝」に偽の優位を与える＝温存カウンターが使われる世界が生成されない。m4@2 実測で def_temp0→0.7 によりイワンコフ 18/32→11/32 と偽優位が消失）→**マージン混合ラベル** `margin_blend`（勝敗z＋0.25·clip(平均残ライフ差/4)＝拮抗群でも勝ち方の質で順位が立つタイブレーク・z の符号は単独で覆せない）を付け、**同一 group** の子盤面を吐く。`--enc-version` で子盤面ラベルのみ新版符号化（採掘・ロールアウトはエンジン版のまま）。各行に **dead_play フラグ**（v33＝ON_PLAY 持ちPLAYが発動しない不発設置・判定子は onplay_option_scan と同一の単手版 `_play_fires`・旧シャードは 0 既定）。v24 が行動種で分岐して m4@2（PLAY vs PLAY）を対照できなかった穴をカード単位で埋める。符号化は出荷ネットの版（gen10=v7）。出力は `build_rank_pairs`/`rank_finetune`（v12.1）が読む形式 |
| `tests/scripts/option_pair_finetune.py` | **オプションペア順位微調整**（v31・`option_pair_gen` の対）: optpair コーパスを child dict へ連結→`build_rank_pairs`（group 内 z差>δ）→`rank_finetune`／**`rank_finetune_anchored`**（v33・`--anchor-dirs`＝dense 一般盤面で base 予測へ引き戻す蒸留錘・`--anchor-scale`）で gen10 を微調整。v32 実測: アンカー無しヒンジは順位が上がるほど防御較正（m2@12/58）が先に壊れる（3回再現）。`--dead-weight`＝負け側が不発PLAYのペアを複製で重み増し（m1@3 型へ信号集中）。policy は base のまま（v12）だが**符号化版は value に揃える**（版違いコピーは行動特徴列ズレで黙って壊れる・2026-07-31 実害）。順位正答率（学習前後・base 対比）を出力・判定は外部（coach/arena） |
| `tests/scripts/plan_cf_gen.py` | **ターン出口CFコーパス生成**（v38・2026-08-06）: 自己対戦の**ターン開始点**を一様抽出（防御窓CFの round-robin は「毎ターン1つ」の性質で最序盤へ偏るため専用の `pick_turn_starts`）し、候補プラン（argmax＋**先頭手が異なるプラン各1本**＝温度サンプルだけでは argmax へ潰れる実測に対処）を `plan.execute_plan`（serve と共有する実行規約の単一の正）で実盤面に打ち切り、到達した**ターン末盤面**を符号化。ラベルは K 決定化世界を def_temp=0.7 で終局まで回した因果 z を `margin_blend` したもの。group は seed_base 込み＝並列ワーカーのシャードを衝突なく併合できる。出力は optpair/defcf と同スキーマ（順位学習にそのまま流せる） |
| `tests/scripts/exit_head_finetune.py` | **出口ヘッドの順位学習**（v39 ターン末＝`plan_cf_gen` の対 / v41 戦闘出口＝`defense_cf_gen` の対）: `--head turn|battle` が「どの箱の出口を較正するか」を選び、それが (a) 読むシャード種（`HEAD_GLOBS`＝plancf / defcf） (b) 有効化するヘッド (c) serve で誰がその値を見るか、を同時に決める。順位ペアを**そのヘッドの4パラメタだけ**へ流す（`ValueNet.backward_exit`＝胴体も既存ヘッドも他階層のヘッドも凍結）。共有重みを動かさないので**蒸留アンカーが要らない**（v33 以降の錘は「共有重みを動かすと既存挙動が壊れる」ことへの対処であり、錘と教師の綱引き自体が消える）。学習後に凍結対象の bit 一致を自己検査し、保存 npz の再読込でヘッドが復元されることも検査する。順位正答率（学習前後）を出力・判定は外部（coach_gate/arena） |
| `tests/scripts/corpus_v11_to_v12.py` | **コーパスの符号化 v11→v12 変換**（列切り出し・2026-08-15）: v12 = v9(70列) + リーダー物理要約(24列) ＝ v11 から**リーサル距離Δ3列（70..72）を抜いた**もの。v11 行は `[v9 70 | Δ 3 | リーダー 24]` の並びなので `scalars[:, [0:70]+[73:97]]` の切り出しだけで v12 教師になる（**対局の再生成が不要**）。列定義はエンコーダの版レイアウト（`scalars_dim`）から導出＝二重定義しない。冪等（既に94列なら素通し）。動機は serve コスト: v10 のΔは台本再生の実測特徴で ~25ms/盤面あり、探索が1手で数百回符号化するため decide が 0.47s(v9)→13.5s(v11) と本番予算1秒を28倍超過していた（`docs/reports/gen15_adoption_20260815.md`） |
| `tests/scripts/value_invariant_audit.py` | **価値ネットの不変量監査**（オラクル不要の破れ探し・2026-08-15 ユーザ提案「過去バージョンを使っても今確認できていない問題は検出できないのでは」への回答）: 対戦相手も正解ラベルも使わず、価値関数が満たすべき性質の破れを数える。**(1) 零和対称性**＝同一盤面を自分視点/相手視点で評価した和は0のはず（符号化は非対称〔自手札は中身・相手は枚数＝公平性契約〕なので厳密反転は期待せず、**平均バイアス**＝席/手番バイアスと順位相関を見る）。**(2) 支配単調性**＝他が全く同じでライフ/手札/ドン/パワーが増えた側の評価は悪化してはいけない。改変は**無から足す**（複製/新規生成）＝山札・ドンデッキを減らさない（初版は山から引いていたため「デッキ残減という正当なコスト」と欠陥が区別できなかった・設計修正 2026-08-15）。**初回実測（85盤面）**: gen15 は power_opp 39%・don_opp 41%・零和 +0.113、gen14 も同水準（power_opp 39%・don_opp 29%）＝**世代を跨いで存在する既存欠陥**で、破れの大きさ 0.02〜0.05 は**枝間マージン 0.02〜0.03 と同規模**＝接戦の選択を裏返す力がある。ライフ/手札はほぼ無傷（勝敗ラベルから直接学べる量）で、**パワー/ドンだけが秩序づけられていない** |
| `tests/scripts/monotonicity_pair_gen.py` | **支配単調性の教師生成**（不変量監査で見つけた既存欠陥への処方・2026-08-15）: 任意の盤面に資源を1単位だけ足したペアを順位教師にする（value=+1 が良い側）。**ロールアウト不要・対局不要・実デッキ非依存**でラベルが論理的に正しい（勝敗の推定でも人間の裁定でもない＝ブートストラップ問題が無い）＝**物量が素直に効く唯一の教師**。スキーマは defcf/vinj/optpair と同一で `option_pair_finetune`（蒸留アンカー付き順位学習）にそのまま流せる。**初回実測**: 139盤面→1,042ペアで gen15 の破れが don_opp 41%→7%・power_opp 39%→21%・零和バイアス +0.113→+0.009 に低下し、**同時に ns2 接戦帯が +0.709→+0.759 と改善**（歪みは純粋なノイズで、除くだけ性能が上がる＝トレードオフ無し）。本体 value を動かす教師なのでアンカー必須＋戦闘出口ヘッドの載せ直しが要る |
| `tests/scripts/candidate_screen.py` | **候補ネットの一次スクリーニング**（重い検証の前段の足切り・2026-08-15）: **(1) decide レイテンシ**（本番 sims160・予算1秒）→ **(2) ns2 相関**（接戦帯/全帯）→ **(3) 裁定3点**（m1@3 展開／m1@14 入口素通し／m1@15 払い切る）の順に測る。**順序に意味がある**——v11 候補は decide 13.5s＝予算の28倍だったのに ns2（事前計算行列のバッチ予測）でもコーチゲート（1点ずつ）でも表面化せず、**アリーナが10分/ペアになって初めて発覚**した（符号化コストは v9 1.3ms / v10 25.1ms / v12 1.3ms で、探索は1手で数百回符号化する＝1盤面のコストが桁で効く）。判定は出さず数字を並べるだけ（正式判定は coach_gate と arena_resume） |
| `tests/scripts/exit_head_probe.py` | **出口ヘッド候補の安価な足切り**（v41・読み取り専用）: MCTS を回さず2列だけ測る——(1) 各検証点の合法手を `resolved_branch_values`（＝実対局の防御窓読み出しそのもの）で並べ、argmax が裁定済み accept に入るか（gap 列＝枝間マージン）、(2) 教師コーパス盤面での `predict_exit(kind) − predict()` の平均/標準偏差（平均が支配的＝一律バイアスを学んだ／標準偏差が支配的＝盤面ごとの差を学んだ）。**実測 2026-08-07**: gen12 の枝間マージンは 0.02〜0.03 なのに defcf（584群775ペア）で学習したヘッドの摂動は標準偏差 0.23〜0.27＝約10倍で、m1@14 を直す腕は必ず m2@44 を壊した（5腕すべて 4/8）。腕を増やす前にこの2列でコーパスの信号がマージンを超えているかを判定する。注意＝戦闘窓でない点（m1@3/m4@2）の行は「root を戦闘箱の規約で並べたら」の近似で実対局の decide とは経路が違う（一次情報は m1@14/m1@15/m2@58）。`turn_all` 形式の点（m2@66）は初手1手の枝順位では原理的に判定できないので**分母から除外**して `--` 表示（正しい計器はコーチゲートの `turn_all_rate`）——v44 まで `CG.hit` に dict を渡して黙って常時不一致に数えており gen13 を 7/8 と過小に見せていた（修正後 7/7・腕どうしの比較は同じ偏りを共有していたため過去の判定は無傷） |
| `tests/scripts/value_calib_fit.py` | **単調再較正のフィットと検証**（v47 手順2）: 純粋ラベルコーパスから等調回帰（PAVA）で `ValueNet.set_calib` 用のノットを当て、**対局単位のホールドアウト**で汎化を見る（盤面単位で割ると同一対局の相関で楽観的に出る）。**単調変換で消えない誤りも必ず出す**——ライフ差など特徴依存のバイアスは出力の変換では消せないので、変換前後の層別バイアスを併記して「残った分＝本体の再学習でしか直らない分」を明示する。単調変換から試す理由は、v40（本体全面学習でアリーナ 0.447）と違い**あらゆる直接比較を bit 保存**するため壊しうる範囲が構造的に限定されること |
| `tests/scripts/value_label_gen.py` | **一般盤面の純粋勝率ラベル生成**（v47 手順1）: 自己対戦の**任意の決定点**（自ターン/相手ターン・戦闘中も含む＝木の葉が実際に見る分布）をターン帯クォータで層化抽出し、K世界 CRN のロールアウトで `win_w`/`life_w` を生のまま保存する。**`value` は純粋 z**（=2*勝率-1）で `margin_blend` を掛けない——本器の目的が blend 汚染の除去だから。`plan_cf_gen`/`defense_cf_gen` との棲み分け: あちらは**同一決定点の兄弟枝**を対照する順位ペア教師で盤面が出口（ターン末/戦闘後）に限定される。本器は**1盤面=1行**で水準の較正用＝group は盤面ごとに一意なので順位学習には使えない。**なぜ要るか**: 手順0 の監査が blend ラベルで劣勢側バイアスを過大に見せていた（純粋 worlds=32 では消えた）＝**較正の議論には純粋ラベルが要る** |
| `tests/scripts/value_calibration_audit.py` | **本体 value の較正監査**（v47・読み取り専用）: ラベル済みコーパス全体で predict() とレフェリー実測を突き合わせ、較正曲線（予測ビン→実測平均）と層別バイアス（ライフ差/手札差/自ライフ/ターン/**相手ドン総数**/**ドン差**）＋**手番リーダー別バイアス**を出す。SE はクラスタ単位（`--cluster group|game`）。`win_w` があれば純粋勝率 z を再計算し、無ければ `value`（margin_blend 込み）。**手番リーダー別が要る理由**: 対面をプールした全体バイアスは席の打ち消し合いを隠す。片方の席で +x・他方で −x なら、その誤りは「対面依存」ではなく**片方のリーダーに紐づく**（`card_idx[:,0]`＝to-move リーダーの語彙 index・0=パディングの 1-origin）。ドン層は OP15-058 紫エネルの「ドン‼デッキ6枚」ルール変更のように、**ドン数そのものが対面の識別子になる**場合の切り分け用。**実測 2026-08-09（gen13・110対局）**: 共通の系統誤差は無い（プール +0.031 ±0.044）／訓練対面は水準一致（±0.03）／エネル席のみ +0.42（相手席 −0.23〜−0.27）だが、同席の実測勝率が 3〜34% とエンジン側の疑いが濃く**value の較正問題として扱ってはいけない**（手順0 の「ライフ差過大・中間域圧縮」は blend ラベルと出口分布の産物で、純粋ラベルでは再現しない＝`cpu_v47_step0_correction_20260809.md` で訂正済み） |
| `tests/scripts/label_reliability.py` | **教師ラベルの信頼度を予算の関数として測る**（v44・読み取り専用・**本生成前の足切り**）: `plan_cf_gen` が保存する**世界ごとの生の結果**（`win_w`/`life_w`）から任意の予算 K≤worlds の z を再計算し、世界を互いに素な2組へ分けて (1) 半々の順位一致率（0.5=コインフリップ＝ノイズのみ）、(2) **δ選抜後の一致率**＝学習が実際に教わる順位の正答率（真の差が小さい点では |Δz|>δ が立つのは引きが偏ったときだけなので、平均への回帰でここが 0.5 を割りうる）、(3) 推定ラベル1σ と枝間マージン（gen12 実測 0.02〜0.03）の比、を出す。**コーパスを作り直さずに「worlds をいくつにすべきか」に答えるための計器**——生成コストのほぼ全部は worlds×プラン数×終局までのロールアウト（実測 245秒/窓@worlds8）なので、世界を使い回す以外に予算を変える安価な方法が無い。**K の行は「K世界のラベル」の信頼度**（互いに素な K世界の組を2つ取って比較）なので `2K ≤ worlds` が要る＝worlds=32 のコーパスからは K=16 までしか測れない。**実測（v44・worlds=32・33群174ペア）**: 半々一致率は K=2→16 で 0.629→0.722、δ選抜後は 0.656→0.818、ラベル1σ は 0.716→0.242。世界を倍にしても δ選抜後は +0.05 程度で、**予算では 0.9 に届かない**（16世界ラベルの真正答率は 0.899 と逆算される） |
| `tests/scripts/verified_inject_gen.py` | **裁定済み点の注入コーパス生成**（v42・ユーザ判断 (A)・2026-08-07）: `coach_gate.VERIFIED_V2` の各点で root 合法手を箱の規約で出口盤面まで解決し、accept を勝ち（value=+1）・それ以外を負け（value=−1）とする1点=1群の順位ペアを吐く（スキーマは defcf/plancf と同一＝`exit_head_finetune.py` にそのまま流せる）。**なぜ**＝v41 でロールアウト由来のラベル（worlds=4 で z 粒度 0.5）は枝間マージン 0.02〜0.03 に届かないことが確定したため、裁定そのものを最も鋭い信号として使う。**代償（必読）**＝コーチゲート8点はこの生成器の入力そのものなので、注入後のゲートは**独立した検査ではなくなる**（自分の訓練データを測る）＝採否の一次証拠はアリーナへ移る。`--battle-only`（既定 ON）＝戦闘出口ヘッドが serve で評価するのは戦闘箱の出口盤面だけなので、戦闘窓の点（実測 m1@14/m1@15/m2@58 の3点のみ）に限定する（メインフェーズの子盤面はヘッドが一度も見ない＝管轄外の教師）。`turn_all` 基準の点（m2@66）は単一 root 手で採点できないため常に除外 |
| `tests/scripts/turn_exhaust_probe.py` | **ターン消化の打ち切り診断**（v45・読み取り専用）: `turn_all` 形式の点（m2@66）で自ターンを終端まで指させ、**TURN_END を選んだ瞬間**の (1) 必須アクションの消化数と残り、(2) TURN_END と残存必須の visit%/Q（`decide(trace=…)`）、(3) 同じ2者の policy prior、(4) **その時点の合法手集合**を並べる。コーチゲートの `turn_all_rate` は消化率（gen13 で 0.62〜0.69）しか返さず、打ち切り位置も原因も分からないため。**読み方**: TURN_END の Q が最高＝value の問題／Q は低いのに visit% が最高＝prior の問題／残存必須が**合法手に無い**＝判断以前の問題（枝刈りや盤面変化）。**実測 2026-08-08（m2@66・16seed・160sims・gen13）**: 消化率 0.69。未消化5件は**すべて枝刈りで除外**（選べたのに選ばなかった 0件／ルール上不可 0件）＝16seed 全てで TURN_END が唯一の合法手で、CPU は一度も早期に畳んでいない。原因は相手の2つの**このターン中**デバフ（OP09-001 シャンクス −1000／OP10-018 カマクラ十草紙 −2000）が自リーダーに集中し 7000→4000 となり、以後リーダー攻撃がどの標的にも届かず `_prune_futile_attacks` が正しく落とすこと。**最初の攻撃がリーダーなら 11/11 達成・ナミなら 0/5**＝例外ゼロの完全分離（詳細は `docs/reports/cpu_v45_m2at66_root_cause_20260808.md`）。**設計上の要点**: prior/Q を見る前に合法手集合を残すこと——無ければ「TURN_END の Q が最高（−0.41）」だけを見て value の問題と誤診する（他に候補が無いのだから当然である） |
| `tests/scripts/game_replay_log.py` | **1局まるごとの人間可読プレイログ**（v48・読み取り専用）: 自己対戦を1局通しで回し、各決定点の「自分と相手の盤面（ライフ/ドン活・レ/手札/場のキャラ＝パワー・付与ドン・レスト）→ 合法手数 → 選んだ手」を平たく出し、末尾に行動種別・プレイしたカードの集計を付ける。**用途**: あるデッキをCPU が**まともに回せているか**を統計でなく打ち回しで確かめる（v47b で `nami:p_enel` のエネル席が実測 5.3%＝ユーザの実プレイ知見「エネルはナミに不利」の幅を超えたため、value の較正を論じる前にエンジン側を目視する必要が出た）。**棲み分け**: `turn_exhaust_probe` は1ターンの探索内部（visit%/Q/prior）、`divergence_probe` は乖離の集計、本器は1局の通しで探索内部を出さない代わりにデッキの機能を人間が読める形にする。`--focus <リーダー card_id>` で片席のみ詳細化、手番側の手札は `card_id(cコスト,Cカウンター)` で表示（相手手札は encoder と同じく非表示＝公平性契約）。**`--viewer-json` でリプレイビューアが読む形式**（`opcg-replay/v1` 封筒＋frames）も出せる——API と同じ `opcg_sim/api/services/replay.py` の記録関数を呼ぶので、フレームの形も action_index の対応付け規約（action は適用前・frame は適用後）も本番の traced 対局と一致する。既定 `--eps 0`（serve と同じ決定的プレイ）で、生成時の play を再現するときだけ 0.15。**射程の限定**: ラベルは探索でなく `CR.rollout`（より安い方策）で付くので、本器で見えるのは**自己対戦の質であってラベルの質そのものではない** |
| `tests/scripts/power_threshold_probe.py` | **打点しきい値の未達スキャン**（v48・読み取り専用）: リプレイのフレームだけで「その**自ターン**、しきい値以上の打点を作れたのに作らなかったか」を数える。到達可能打点 = max(ターン開始時に場にいてレストでないキャラ／レストでないリーダーのパワー + アクティブドン×1000)、実際 = そのターンの攻撃の攻撃側パワー最大値。**用途**: 勝率が使えない対面の裁定裏取り——飽和負け局面ではレフェリー12世界でも ±1勝に揺れて識別できないが、打点は盤面とドン枚数から決定論的に決まるのでノイズがゼロ。ユーザ裁定（2026-08-10）「エネルがナミに勝つには 7000 以上で殴る必要がある／キャラクターのエネル OP15-118 を出すのもパワー10000 が重要だから」を指標化した。**実測 2026-08-10（しきい値7000）**: **人間（h1・エネルを握って CPU ナミに勝った実対局）は 6自ターン中 0件未達**、**CPU 自己対戦（e1/e2・gen13）は 12自ターン中 3件未達**（e1@5 到達11000/実際5000・e2@2 到達7000/攻撃なし・e2@6 到達11000/実際6000）。人間は turn5 で 11000 作れる場面を**あえて 7000 ちょうど**で殴っており、最大化ではなく**しきい値を満たして残りを他に回す**打ち方＝指標の設計と一致する。**2つの実装上の落とし穴**（どちらも実測で発覚し修正済み）: (1) 攻撃の action_type は経路で異なる——CPU は `ATTACK`、アプリの人間操作は `ATTACK_CONFIRM`（宣言→確定の2段UI）。片方だけ数えると人間側が全ターン攻撃0＝全未達に化ける。(2) 自ターンの判定は**フレームの `active`** で行う——`player == 自分` のアクションには相手ターン中の防御応答（SELECT_COUNTER/PASS）が含まれ、それを自ターンとして数えると攻撃0＝未達に化ける（初報で e2@7/@9 を未達と誤報告した原因）。**射程の限定**: リーダー能力による一時的な打点増（エネルの「レストのドン4枚まで付与」等）は数えないので到達可能打点は**過小評価**＝偽陽性を出さない側に倒してあり、未達を全部拾えてはいない |
| `tests/scripts/human_replay_divergence.py` | **人間リプレイとの選択乖離スキャン**（v48・読み取り専用）: 人間が打った traced 対局を決定点ごとに復元し（`mark_gate._restore`）、同じ盤面で `LearnedEngine.decide` を seeds 回まわして**人間の手と一致するか**を出す。一致率0＝CPU が一度も人間の手を選ばない点＝最も濃い裁定候補。**狙い**: 裁定を1点ずつ言葉で取るのは高コスト（v18 は34マークに人手を要した）だが、人間が1局打てば**その手がそのまま正解ラベル**になり、食い違いを機械的に抽出できる。**棲み分け**: `divergence_probe`（v12）は候補ネット vs 既定ネットの生成対戦、`mark_referee_verify` は既知マークの裁定、本器は**人間の実対局 vs CPU**。**落とし穴**: 攻撃の action_type は経路で異なる（CPU=`ATTACK` / アプリの人間操作=`ATTACK_CONFIRM`＝宣言→確定の2段UI）ため正規化しないと全攻撃が不一致に化ける。**読み方**: 同一ターン内の乖離は最初の1点の下流に連鎖するので、**ターンごとの最初の乖離**を見る。`--net neff:<npz>`／`n1:<npz>` で N 系ネットも照合可（2026-09-03）。**2026-09-04 の訂正 2 点**: (1) 人間の ATTACK_CONFIRM 記録は `targets` を持たないため、参照側に対象が無い点は対象を比較から外す（外さないと全攻撃が不一致に化ける＝h2〜h4 で 61 点が 0.00 になった実害）／(2) RESOLVE_EFFECT_SELECTION は復元器が効果対話の途中状態を再現できない（復元盤面は常にメイン窓）ため既定で除外。見本棋譜 h2/h3/h4（`coach_gate.REPLAYS_HUMAN`）も対象 |
| `tests/scripts/serve_referee.py` | **本番仕様レフェリー**（v48・読み取り専用）: 決定点の各候補手を適用し、以降を**両席とも出荷の `LearnedEngine.decide`（同梱既定ネット＝gen13・sims=SERVE_SIMS・測定用温度なし）**で終局まで打って勝率を比べる。ユーザ指示（2026-08-10）「選択手を測る時は本番仕様で測るべき」に基づく。世界＝フレーム復元が持つ**両者の手札・ライフの真値**を保ち、山札の並びだけを共有 seed でシャッフル（CRN・神視点の対照実験）。**棲み分け**: `counterfactual_referee` は教師を gen5 に固定する設計（学習で漂流しない錨＝教師ラベル生成の正本）で、本器は**製品の挙動そのもの**での最終確認。両方で同方向に出た点だけを頑健な裁定として扱う |
| `tests/scripts/dig_inject_gen.py` | **掘り裁定の注入コーパス生成**（v49・2026-08-10・`verified_inject_gen` の兄弟＝ターン末盤面版）: 早期ターン（turn≤turn-max）の復元盤面から「低コスト登場時発火キャラを出して効果を受けてEND」(+1) と「無行動END」(−1) のターン末ペアを1点=1群で吐く（符号化 v9・`option_pair_finetune` がそのまま読む）。**なぜ勝率CFでなく注入か（v49 実測）**: h1@2 の掘り/無行動を教師正本設定×32世界CRNで測ると両腕 0/32 勝＝ラベル完全飽和、ライフ差タイブレークは逆向き（ロールアウト方策自身が掘った札を活かせない＝ブートストラップ問題）。既定 `--leader OP15-058`＝**裁定の射程（エネル席）だけを採る**。掘り手判定はカードIDハードコード無し（適用で効果対話が立つ低コストキャラPLAY）。注入点は以後独立した検査にならない（v42 と同じ取引）＝効果確認は h1@2 3層分解・非注入ゲート点の不変・アリーナ中立で行う |
| `tests/scripts/satori_transplant_probe.py` | **サトリ移植プローブ**（v49・2026-08-11・読み取り専用・ユーザ発案）: ナミ（ドン再装填なし）のデッキにサトリ OP15-066 を移植した turn1 合成盤面で「掘ってEND vs 無行動END」のマージンと decide を測り、学習した掘りが**リーダーの経済に紐づいたか／カード特徴で他デッキへ漏れたか**を判定する。ユーザ裁定（2026-08-11）＝再装填のないリーダーでは**出さない（TURN_END）が正解**（負マージン・掘り率0が OK）。v49 実測: gen13 は順序が逆（ナミ+0.12 で掘る/エネル+0.011 で掘らない）・B腕は順序修正するが絶対水準はカード特徴で漏れる・E腕（逆ラベル対照）のみ通るが防御較正を壊す。候補ネットの**常設検査**（盤面 seed は学習に使った 10..20 を避け 22,24 が既定） |
| `tests/scripts/bb_card_factory.py` | **骨組み線 bb0: 合成カードファクトリ**（2026-08-11・分離規約 `bb_*`・`docs/cpu_backbone_plan.md`）: 実カードのパース済み Ability を収穫（2963件）し、元ホストコスト c±1（低確率±2＝包絡拡張）の予算で BB- カードへ再結合する。バニラリーダー（能力なし）合成込み。数値変異は既定 OFF。ドメインランダム化訓練（同じIDが対局ごとに違う効果＝埋め込み依存を構造的に不可能にする）の材料源 |
| `tests/scripts/bb_selfplay_audit.py` | **骨組み線 bb0: 合成デッキ自己対戦の実現性監査**（2026-08-11・`bb_card_factory` の対）: 対局ごとに新合成デッキを生成し出荷既定エンジンで打ち、構造不変条件（EXCEPTION/CARD_LOSS/TEMP_LEAK/APPLY_NONE）と内在品質基準（完走・退化率・意味行動密度＝実対局分布との類似は使わない）で Go/No-Go を出す。**初回300局の実測（backbone_bb0 レポート）**: 完走295/300・退化0.7%・現行エンジンの実在欠陥2件を発見（キャラ対象へのリーダー混入→カード増殖 seed880007／合法手列挙と declare_attack の不整合 seed880014） |
| `tests/scripts/lethal_teacher_gen.py` | **リーサル帯・乖離盤面の教師採掘**（G系 v51・2026-08-12・`lethal_calibration_probe` の対）: v50 の欠陥族「見かけと実質の乖離盤面」の較正教師を**盤面ごとの証明**で選別して量産する。採掘規則（ユーザ合意）＝①非エネル席（既定 nami:shanks）②実現による自己証明（|EV|≥2/3＝勝った側が5/6以上実演）③確信して外した盤面のみ（|予測|≥0.5 かつ |予測−EV|≥0.5）④ラベル器は教師正本（CR.rollout sims48 def_temp0.7）×真値世界。出力は G系通常符号化＋value=EV（dense 系 MSE 微調整が読む形）。スモーク実測: 3局→5ラベル→教師1（+0.585 予測 vs 0/6 実測） |
| `tests/scripts/bb_train.py` | **B系 bb1: 骨組みネット訓練**（2026-08-12・`bb_gen` の対）: 合成世界コーパスで value を MSE 訓練。ID排除は card_idx 全 PAD(0) 固定＝ValueNet 実装は G系と同一（分離規約・無変更）。初回実測: 8652行・val MSE 0.79 |
| `tests/scripts/bb_eval.py` | **B系 bb1: 実盤面ホールドアウト評価**（Phase 1 の Go/No-Go）: レフェリー勝率ラベル（教師正本 sims48×6世界）の実盤面に対する骨組みの MAE/RMSE/r/符号一致を、G14 参考線・常に0 基準と並記（gen14 一致は判定に使わない＝固有性監査 #2）。第1ラウンド実測（60点）: 骨組み r=0.20 vs G14 r=0.42＝学習は成立・水準は未達（主因＝バニラリーダー世界と実盤面のリーダー効果ギャップ・Phase 3 第2段の前倒しが本命） |
| `tests/scripts/bb_gen.py` | **B系 bb1: 骨組みコーパス生成**（2026-08-12・`bb_train` の対）: バニラリーダー×合成カード（`bb_card_factory`）の自己対戦から1ターン1行（勝者 z ラベル・card_idx 省略＝ID情報なし）を採る。退化局は落とす。初回実測: 400局→8,652行。`--engine l1`（2026-08-14 bb5）＝駆動を古典CPUへ交代する A/B つまみ——実測は**負**（接戦読み改善なし・G14 は UNK でも物理＋探索で十分な先生だった・`backbone_bb5_l1_20260814.md`）・既定は learned のまま |
| `tests/scripts/bb_relabel.py` | **B系 bb4: 接戦盤面のレフェリー再ラベル**（2026-08-13・分散可＝seed帯分割）: bb3 コーパスと同一シードの対局を決定論再生（bb_gen と同一乱数規約）し、接戦フィルタ（turn≥4・ライフ差昇順・5点/局）で選んだ盤面へ教師正本 EV（CR canon sims48×6世界 CRN・ラベリングは対局完走後＝v51 の global random 教訓）を貼る。出力は bb_train 互換シャード。実測: 466局→2327行（非飽和ラベル率19〜28%）・**結果は負＝ラベル品質仮説を対照実験で棄却**（±1 訓練の bb3 が既に域内・未見の接戦盤面 r=+0.436＝律速は合成↔実世界の接戦力学ギャップ・`backbone_bb4_20260813.md`） |
| `tests/scripts/bb_contested_isolation.py` | **B系 bb4b: 接戦帯の分離プローブ**（2026-08-13・`backbone_bb4_20260813.md` §4 の切り分け）: 実デッキ×**バニラリーダー**の対局を bb4 と同一の接戦フィルタ（turn≥4・ライフ差昇順・5点/局）＋同一教師正本（CR canon sims48×6世界 CRN・完走後ラベル）で採り、ID無しネットの r を**非飽和帯に層別**して測る。読み方: bb の非飽和 r が 0.4 級へ回復＝接戦盲目の主因はリーダー効果／0 のまま＝特徴不足（実カード能力意味論が物理特徴に映らない）。`--strip-card-abilities`＝キャラ・イベントの能力も消した純ステータス世界（消去はしごの最下段・カード能力の寄与を単離） |
| `tests/scripts/g15_gen.py` | **G系 g15: v11 スパイクのコーパス生成**（2026-08-14・`g15_train` の対）: 実デッキ4種の全6対面ローテ自己対戦（出荷 G14 駆動・seed 決定論）から (盤面, 勝敗±1) 行を v11 符号化＋実 card_idx で採る。分散は対面×シード帯で子セッションへ分割（12子で720局実測）。ハンニャバルは訓練に入れない＝未見リーダー検査帯 |
| `tests/scripts/n_record_gen.py` | **純正Nループ①: 棋譜ダンプ生成器**（2026-08-26 ユーザ決定「純正に準じてやりましょう」）: ランダムリーダー×生成デッキ（`_leader_pair`/`deck_synth` 同規約）の自己対戦から**全判断点**（main/window/commit）の v12 符号化＋勝敗 z（純正 AZ の素の ±1・勝敗の付かない対局は棄却）と、main 窓の**候補と訪問分布**（`decide(record=…)`＝選択と同一の等価マージ集計・(sig,k) で一意・主体/対象の**カードID**をダンプ時に解決＝uuid は事後解決不能）を ragged npz で採る。教師の採掘（z 密・方策ターゲット）は別計器＝ダンプは生のまま＝教師の取り方を後から変えられる。候補生成は GEN_PRUNE_FUTILE・探索は serve 既定（箱化＋箱コミット ON）＝実対局と同じ行動列・seed 決定論。**`--dump-v2`（2026-09-04・NRel P1）**: 符号化 v13＋トークン状態 S（float32）＋候補の枠 index を追加、関係 R は保存せず訓練時に再計算（meta の dump_version=2） |
| `tests/scripts/n_mine_z.py` | **純正Nループ②-a: 素の z 教師採掘器**（`n_record_gen` の対）: 棋譜ダンプの全判断点（既定=main+window+commit・`--kinds` で絞り込み可）を訓練互換形式（scalars/field/card_idx/value）へ落とす。value は素の z=±1 のみ（TD・blend・margin 合成はしない＝純正 AZ） |
| `tests/scripts/n1_train.py` | **純正Nループ③④: N1ネット訓練器**（value+方策チャネル・胴体共有）: 棋譜ダンプを直接読み（seed で**対局単位**の train/val 分割＝行リーク防止）、単一 value（素の z・ctx 出口ヘッド無し）と方策チャネル（状態埋め込み＋候補素性49次元→点内 softmax）を、訪問分布 π への交差エントロピーと z の MSE の多課題で同時学習。胴体は N0 の芯（`n0_spike.build_card_table`/`card_channel`）を再利用。val は v_mse/v_sign・π top1・実選択 top1・CE を印字 |
| `tests/scripts/n1_gate.py` | **純正Nループ④: N1 の serve 接続とゲート**: N1 を LearnedEngine へ両輪注入——value は `N1ValueAdapter`（出口ヘッド無し has_exit_head=False＝戦闘/対話箱の物差しは本体 value へ自動フォールバック＝単一価値関数）、policy は `priors_override` seam（訓練と同一の候補素性49次元→点内 softmax。失敗時 None=一様）。`gate`=coach 13点（既定 vs N1・n0_spike.gate と同じ判定）／`smoke`=N1 同士の実対局1局完走（配線の煙試験） |
| `tests/scripts/n_eff_feat.py` | **効果構造符号化（N系カード表現v2・2026-08-27）**: パーサ正本の能力列→能力ベクトル167次元（トリガーonehot23＋op62×自/相手量＋対象フィルタ要約〔コスト/パワー閾値・特徴参照・相手対象〕＋付与キーワード8＋構造フラグ7〔条件/任意/Choice/up-to/持続/回数制限〕＋コスト量）×最大4本＋基礎統計16（stats8+印字キーワード8）。トリガー×op×量×閾値の結合を保存（12次元合算の`leader_feat`と違い「ON_PLAYで2枚掘る」が固有の型になる）・重み付けはネットに学ばせる。**2026-09-03 c10 採用で forward・表・アダプタ・priors の正本は `opcg_sim/src/learned/n_eff.py` へ昇格**（本器は継承／再輸出の互換窓口） |
| `tests/scripts/n_eff_train.py` | **効果構造版の訓練器**（`n1_train`とのA/B＝カード表現だけが変数）: 効果埋め込み48は学習対象（全語彙の能力集合→共有MLP(167→24)+mean/maxプール→カード表を毎stepWaから計算・backwardは語彙indexで勾配合算＝端から端）。候補素性139次元（主体/対象のカード表現64×2＋printedパワーマージン＋対象=リーダー）。value+policy多課題・対局単位分割・ベストチェックポイント。**2026-09-03 c10 採用で forward・表・アダプタ・priors の正本は `opcg_sim/src/learned/n_eff.py` へ昇格**（本器は継承／再輸出の互換窓口）。dump v2（v13・123 列）を読んだときは先頭 94 列（v12）へ切り詰める（`_to_v12`・2026-09-05・c12 対照実験） |
| `tests/scripts/n_eff_gate.py` | **効果構造版の serve 接続とゲート**: カード表はアダプタ初期化時に1回前計算（serve凍結）・候補素性は訓練と同一139次元（train/serve一致）。`gate`=coach 13点（--base-net で前世代比）／`smoke`=実対局1局完走。**2026-09-03 c10 採用で forward・表・アダプタ・priors の正本は `opcg_sim/src/learned/n_eff.py` へ昇格**（本器は継承／再輸出の互換窓口） |
| `tests/scripts/n_rel_train.py` | **NRel（Stage A）の訓練器**（2026-09-04・`n_rel` の対）: forward は `opcg_sim/src/learned/n_rel.py` を継承し backward（対の MLP・max/mean プール・候補の枠 index への散布）と Adam を足す。dump v2 を読み、関係 R は `n_rel_feat.relations_batch` でバッチごとに再計算。候補の予算 3（戻すドン・次ターンの最大の札が出せるか・ドンコスト）を `budget_feats` で作る。`--in`（π）/`--z-in`（z 専用）/`--warm-start`/`--holdout-mod 7`・保存時に vocab_ids と meta.kind=nrel-a を焼き込む。**メモリ（2026-09-05）**: 行数を先に数えて V を一度だけ確保し、方策点は盤面を複製せず V の行 index（`P["row"]`・`prow`）で参照＝1 シャード（≈13.6 万行）あたり 363MB・16 シャードで約 5.8GB（cgroup 14GB 内）。epoch ごとに暫定最良を `--out` へ書き出す（16 シャード×2 epoch ≒ 2 時間 51 分・r1 実測）。`--ablate rel,opp_pool`＝切り分け訓練（遮断は `n_rel.NRelNet.ablate`・npz の meta に焼き込まれ serve でも遮断される） |
| `tests/scripts/n_rel_band.py` | **評価帯（dump v2 の holdout 行・seed%7==0）で N系 c ネットと NRel r ネットの value を同じ行で比べる**（2026-09-05・r1 の判定用）。dump v2 の scalars は v13＝v12 の末尾に 29 列を足した append-only なので c ネットには先頭 94 列と card_idx を渡す。`--neff`/`--nrel` に複数 npz 可・`N_REL_BAND` 行に v_mse/v_sign（全体・ターン帯別）を出す。`--zero-rel`/`--zero-opp-pool`＝serve 時の遮断（訓練なしで r ネットの依存を見る・r1 実測: R 遮断 0.531→0.574／opp_pool 遮断 →0.637／両方 →0.751） |
| `tests/scripts/n_mine_pi.py` | **純正Nループ②-b: 方策ターゲット採掘器**（`n_record_gen` の対）: main 窓の実質選択（候補2つ以上・chosen 解決済み）だけを採り、訪問分布 π=n/Σn（**選んだ手のクローンではない**）と候補素性（action_type・主体/第1対象カードID・don_k）を ragged（cand_ptr）で保存。カードIDは文字列のまま＝索引化は訓練側の語彙（採掘器は語彙非依存） |
| `tests/scripts/g15_train.py` | **G系 g15: 実ID訓練（A/B両腕）**（2026-08-14・`g15_gen` の対）: card_idx を実IDのまま MSE 訓練。`--scalar-cols` の接頭辞切り出しで**同一コーパスから v10腕/v11腕**（対局・行・分割が完全同一の A/B）。実測（720局・ns2判定）: v11 は域内−6%・ナミ帯+0.10・エネル帯+0.13 だが**未見リーダーで−0.32（4リーダー過適合）**＝処方は B/G 混合訓練（`g15_v11_spike_20260814.md`） |
| `tests/scripts/ns2_rebuild.py` | **ns2 評価行列の再構築器**（2026-08-14・ロールバック対策の計器化）: `holdout_ns2` fixture（24世界ラベル）から復元→任意世代で符号化→npz。ラベル再ロールアウト不要・数分。/tmp の評価行列がコンテナロールバックで消えるたびに本器で再構築する（本セッションで5回実害） |
| `tests/scripts/bb_isolation_probe.py` | **B系 bb1: ドメインギャップ分離テスト**（2026-08-12・ユーザ確認質問が設計動機）: 実デッキ×**バニラリーダー**の盤面（レフェリー勝率ラベル）で骨組みを測り、実デッキ×実リーダー60点との差分から「カード分布ずれ vs リーダー効果」を分離する。実測: バニラ側 r=0.51/符号73% vs 実リーダー側 r=0.20＝**落差の主因はリーダー効果**（`backbone_bb1_20260812.md`） |
| `tests/scripts/v51_finetune.py` | **G系 v51: 乖離教師の較正学習**（2026-08-12・負の結果の記録器）: `lethal_teacher_gen` の教師50点を MSE＋蒸留アンカー（一般633盤面・v33規約）で G14 に注入し、教師/学習外乖離7点/リーサル45点/一般60点を前後比較。実測: 暗記成立（MAE 1.68→0.37）・**転移ゼロ**（学習外 1.32→1.33）＝representation-bound 3例目（`cpu_v51_negative_20260812.md`） |
| `tests/scripts/lethal_distance_probe.py` | **リーサル距離Δ特徴スパイク**（v52/v52b・2026-08-12・G/B 合流点）: 台本レース（MCTS なし・決定論・数十〜数百ms/盤面）で「双方あと何ターンで詰むか」を測り、説明不能58点（v51教師50＋v50乖離8）への説明力を判定。v1=無抵抗/v2=カウンター防御込み/v3=忠実度改善（ブロッカー・イベント実測・ドン付与算術。don_mode/parts で切り分け）。実測: v2 が頂点（26/58・r+0.351）・**v3 は退行**（引分爆発 16→34＝対称強化の相殺・`cpu_v52b_lethal_v3_fidelity_20260812.md`） |
| `tests/scripts/lethal_distance_holdout.py` | **リーサル距離の一般60点検査**（v52b・2026-08-12・`lethal_distance_probe` の対）: bb1 ホールドアウトと同一の一般60盤面（`holdout60_boards.json`・レフェリー勝率ラベル）でΔ特徴の副作用を測る。実測: v2 37/55・r+0.517（ライフ差 15/55・r−0.10）＝Δ族は一般盤面でも最有力の静的信号 |
| `tests/scripts/v51_retest_v10.py` | **G系 v51 再試験 @ v10**（2026-08-12・`v51_finetune` の対）: Δ特徴（符号化 v10）ありで乖離族教師50点が汎化するかの Go/No-Go。データは全て盤面から v10 再符号化（教師=決定論再生・L45=スキャン再走の v9 内容一致照合・一般60=保全盤面表・アンカー=リプレイ新規採取）。実測: 転移ほぼゼロ（乖離7点 1.281→1.25）＋残差回帰 R²0.11-0.19＝**特徴パッチも埋め込み本体には効かない**（`cpu_v53_v10_delta_retest_20260812.md`） |
| `tests/scripts/don_attach_audit.py` | **ドン付与の行動空間監査**（2026-08-12・読み取り専用・ユーザ指摘「CPUはドン付与が苦手」の診断）: 全リプレイの実 ATTACH_DON を復元盤面上で現行枝刈り `_prune_don_moves` に照らし、kept/DROPPED（overcap=過剰盛り・unreachable・rested）に分類する。初回実測: **人間の付与58手中31手（53%）が「過剰」として候補外**＝7000作り（カウンター強要）の手を CPU は考えることすらできなかった → (C) マージン付与規則（リーダー+2000未満・有界）を追加、監査再走で人間手の可視率 36%→74% |
| `tests/scripts/don_reserve_probe.py` | **温存診断プローブ**（ドン箱化 P2-a・2026-08-12・読み取り専用）: コスト付きカウンターイベント保持側（既定 bg_luffy:nami）の「残ドンがイベントコスト境界帯」のメイン判断で、G14 の本番選択（sims160）が「しきい値を割る消費」だった点を、消費線 vs 最も怠惰な温存線（即 TURN_END）の教師正本 EV（CR sims48×6世界 CRN）で二重ラベル。gap≥1/3＝欠陥確定（怠惰な温存にすら負ける消費・過小検出側に倒す保守判定）。ユーザの同通貨理論（温存ドンの価値＝来ターンのカウンター値）の検証器 |
| `tests/scripts/don_margin_enel_ab.py` | **(C) マージン付与のエネル対面 A/B**（2026-08-12・`don_attach_audit` の対）: 主アリーナ（既定リーダー EB01-021 固定ミラー）ではエネル対面の効果が測れないため、ユーザデッキ fixture の固定対面（既定 p_enel:nami）で同一 gen14・候補席だけ (C) 有効の席入替ペアを回す（OPCG_DON_MARGIN=0 で走らせ既定側=旧規則）。jsonl 追記台帳・再開可。エネル局は長い（1ペア10分超あり）＝ペア数控えめ |
| `tests/scripts/lethal_calibration_probe.py` | **リーサル近接の較正プローブ**（v50・2026-08-11・読み取り専用＋ラベル出力）: 全リプレイから min(自ライフ, 相手ライフ)≤1 の復元盤面を採り、本体 value の予測（手番視点・aux無し）と本番仕様ロールアウト実勝率（真値世界・山札シャッフルCRN・既定6世界）の較正誤差をライフ帯（敵0/敵1/自0/自1）で層別する。背景＝v49 A2 実測: h1 turn9 の実ターン末（相手ライフ0）を gen13 が −0.181 と悲観・実測 5/6 勝＝誤差 ≈0.85。**A1 と違い勝率ラベルが飽和しない**（終盤盤面はエンジンが打ち切れる）ため、`--out` で教師ラベル行（npz・value=実測EV）を同時に吐く採掘器を兼ねる。**接戦帯 v2（2026-08-13 監査後）**: `--max-life-diff/--min-both-life`＝帯の**状態による事前定義**（測定EVでの事後定義は雑音盤面を混入させる）・`--board-offset/--board-stride`＝分散ストライプ・`--shard-rows`＝逐次シャード＋**provenance.json（git rev・config 記録＝ラベル来歴）** |
| `tests/scripts/search_averse_probe.py` | **SEARCH_AVERSE 追跡プローブ**（v21・読み取り専用）: root の Q/N を sims 昇順にトレースし、多世界(8)平均アブレーションで「単一世界 PIMC が犯人か」を切り分ける。旧アブレーション腕（argmax(N) 読み出し／終局減衰 off／aux off）は純正AZ化（2026-08-25）で対象機構ごと削除＝base 自体がそれら無しの計算 |
| `tests/scripts/prior_bound_probe.py` | **prior/value 分解プローブ**（v20・読み取り専用・`docs/reports/cpu_v20_prior_value_20260729.md`）: VERIFIED v2 の各点を真盤面復元し、**prior 直読み**（accept 集合の確率質量と順位＝探索を介さない一次証拠）・**着手後 value 差 dv**・深探索率・prior一様化率の4列で測って OK/EXPLORABLE/PRIOR_BOUND/VALUE_BLIND/**SEARCH_AVERSE**/UNRESOLVED に分類する。`mark_deep_probe.py`（フレーム復元＋人間述語）とは復元方式も基準も別。**実測(gen8): PRIOR_BOUND 0件＝policy prior は犯人ではない**・VALUE_BLIND 4（低コスト展開の過大評価で一貫）・SEARCH_AVERSE 3（prior 1位かつ dv>0 なのに深探索が離れる第3の機序） |
| `tests/scripts/value_blind_probe.py` | **VALUE_BLIND 原因分析プローブ**（v23・読み取り専用・`docs/reports/cpu_v23_value_blind_cause_20260729.md`）: v20 の VALUE_BLIND 4点を真盤面復元し、①**遮蔽帰属**＝正着/誤着の子盤面符号化を特徴グループ単位で入れ替え「誤着の value 優位がどの特徴群に乗るか」を両向きで測る、②**コーパス対照走査**＝密シャードの「同リーダー・同ターン帯・対象カード在/無」の z/q_root 群間差（echo=dq−dz）。**実測(gen8): 過大評価は「勝者の相貌」特徴（手札減・体が並ぶ・使用済/召喚酔いフラグ）が運び、攻撃の子盤面は打点未実現でコストのみ可視＝構造的に不利。密ラベルが直せないのは q_root エコー（m4@2: dz+0.02/dq+0.15）と z の状態交絡（m2@64: dz+0.31）のため** |
| `tests/scripts/defense_cf_gen.py` | **防御窓CFコーパス生成**（①防御応答矯正フェーズ2・2026-07-30）: 固定対面の自己対戦から防御窓（SELECT_COUNTER/SELECT_BLOCKER）をターン分散で採掘し、各選択肢（素通し/各カウンター/各ブロッカー）を**同一決定化世界＋枝間共有のロールアウト乱数**（フェーズ1で発見した CRN 破れの修正後）×**def_temp ロールアウト**で終局測定 → 子盤面に因果 z=2·wr−1 を付与。行の符号化は `--enc-version`（既定 6＝手札資源集約つき。ロールアウト自体は出荷ネット＝v5 でも、教師行には新特徴を載せる）。出力は `dense_finetune` 互換（q_root=NaN＝勝敗単独＝エコー遮断）。生成中に **z幅平均・有情報率**（全枝同値でない窓の割合）を出す＝教師の情報量を走りながら監視する  **v35: 子盤面は戦闘を解決してから符号化する**（`mcts.resolve_battle_inplace` を探索と共有＝1定義）。戦闘途中の子は「1000 を切った子」と「2000 を切った子」が手札-1・ライフ不変で**ほぼ同一入力**になり、そこへ異なるラベル（z=-0.875 と -0.562）を付けていた＝**学習不可能な教師**だった（v34 で教えても動かず強く押すと他点が壊れた原因の疑い）。探索側の静止探索と対で有効化する必要がある（片方だけでは train/serve skew）。 |
| `tests/scripts/defense_cf_probe.py` | **防御窓の反実仮想×人間整合**（①防御応答矯正フェーズ1・2026-07-30・読み取り専用）: ユーザ実対局5リプレイの**人間側**防御窓（素通し/カウンター/ブロック）を真盤面復元し、各選択肢を同一CRN世界×**防御温度つきロールアウト**（`counterfactual_referee.rollout` の `def_temp`＝防御窓のみ訪問数比例サンプリング。argmax ロールアウトでは「温存カウンターが将来使われる世界」が生成されず手札温存の価値が測定から消える循環を切る・L1 不使用）で終局測定し、人間の実選択との整合（top一致/band内）を検証する。フェーズ2（防御窓コーパス量産→学習）に進む前の測定系健全性の関門。人間選択はラベルでなく監査に使う |
| `tests/scripts/counterfactual_pair_gen.py` | **反実仮想ペア教師の生成**（v24・v23 §4 の本命・`docs/reports/cpu_v23_value_blind_cause_20260729.md`）: 固定対面の自己対戦（copy-apply で親盤面保持・Dirichlet ノイズで局面多様化）から「化粧系と進行系が同時に合法」な決定点をターン分散で採掘し、行動種別代表（現ネットの1手先 value 最良）を**同一決定化世界（CRN・`counterfactual_referee.rollout` 再利用）**から終局ロールアウト→**子盤面に因果 z=2·wr−1 をラベル付け**。出力は `dense_finetune` 互換シャード（**q_root=NaN**＝勝敗単独ラベルへ自動退化＝エコー遮断が構造的に入る）。状態相関でなく行動の因果差を value に教える＝VALUE_BLIND（勝者の相貌学習）への処方。閉ループにしない（生成ネット固定・判定は外部 coach/arena） |
| `tests/scripts/mark_referee_verify.py` | **人間マークのレフェリー裏取り**（v18・`docs/reports/coach_gate_v2_20260728.md`）: `gen7_marks_20260728` fixture の各マークを真盤面復元（`state_at_action`・復元盤面ターン照合つき＝別局面を黙って裁かない）し、レフェリー（gen5錨・プラン列挙＋同価値バンド worlds8・捲りエスカレーション）で cpu_out_band（マーク支持）/cpu_in_band（同価値）/no_plans に3値裁定。JSONL 追記（再開可能）・4並列。**実測: 34マーク→裁定14点（支持9）→ VERIFIED v2 採録13点**。裏取り中に再生系の早停止バグと ATTACH_DON 照合不能（=行動空間の穴の発見）を検出 |
| `tests/scripts/matchup_balance_probe.py` | **固定デッキ対面バランス計測**（v18・読み取り専用・`docs/reports/matchup_balance_20260728.md`）: ユーザ提供の固定リスト（`tests/fixtures/decks/user_decks_20260728.json`）の全ペアを gen7 同士の CRN ペア（同 seed・席入替・`run_game(deck_builder=...)` で正確な50枚を毎局再構築）で対戦させ、**CPU 運用時のデッキ相性**を測る。マーク採取の対面選定用（v16/v17 教訓＝一方的対面のマークは飽和局面に落ち教材にならない）。実測: ナミvsシャンクスのみ互角 0.583・紫エネルは対シャンクス 0.021＝gen7 のドン経済運用不全の候補 |
| `tests/scripts/label_worker.py` | **ラベル量産ワーカー**（v9 フェーズ1 外注用）: `referee_labeler.py` をバッチループで回し教師バッチを自分専用 data枝（`claude/v9-label-<worker>`）へ**蓄積 push**（pd_gen と同じ worktree・単独writer だが、消費者不在のため amend+force でなく通常コミット追記＝全バッチ残置）。seed 空間はワーカー別に分離（10M×worker番号＋連番）・停止/再開は git の既存バッチ数から自動（ローカル状態なし）・push 失敗はバックオフ再試行 |
| `tests/scripts/teacher_echo_probe.py` | **教師エコー計器**（v7 §4-1・読み取り専用・`docs/reports/seesaw_probe_20260716.md` 追試1の計器化）: data枝バッチの policy 教師（訪問分布）と生成ネット prior の相関を全決定点で実測（全体＋付与決定点サブセット・corr 分位・top1一致）。v6 実測=中央値 0.93〜0.95（エコー）。v7 スモーク合否＝付与決定点のcorr 中央値 < 0.8（`ECHO_RESULT` 行・exit 0/1）。--data-branch/--net-branch で枝から直接測定 |
| `tests/scripts/mark_deep_probe.py` | **マーク深探索プローブ**（v6 柱③の前提チェック計器・読み取り専用・`docs/reports/v5_adoption_20260715.md` §4-3）: mark_gate と同じ復元・述語でマーク盤面を **sims だけ深く**（既定 160/800/3200）decide し、人間指摘方向率の立ち上がりで分類＝ **EXPLORABLE**（深探索で解ける→柱②深探索再ラベルの守備範囲）／**PRIOR_BOUND**（一様 prior の深探索なら解ける＝policy が正着を読ませていない→柱②の射程・prior平坦化つき再ラベル）／**VALUE_BOUND**（それでも解けない→柱④特徴設計行き）／OK@base（現既定で正着）。誤るマークには確定用の独立証拠2列を既定で追加測定: **flat**（prior 平坦化＝`priors_fn=None` の一様・製品コード無改変の診断パッチ）と **L1**（製品 L1 α-β の第二意見オラクル＝手作り評価が差を感じるかの独立証拠）。既定は基準 sims で誤るマークのみ深掘り（コスト抑制・`--all` で全件） |
| `tests/test_learned_candidate_prune.py` | **基盤健全性**（`cpu_infra`）。**learned 候補の無駄手枝刈り**（`adapter.OPCGGame.legal_actions`・v5 §4補）: L1/α-β と同じ `_prune_futile_attacks`/`_prune_don_moves` を learned MCTS 候補にも適用（`SERVE_PRUNE_FUTILE`）。枝刈りON が _prune_* 適用後と一致・OFF で merged 素集合へ復帰（ゲート）・候補を空にしない（TURN_END 常在）・**インスタンス指定 `OPCGGame(prune_futile=...)` が config より優先**（v6 柱⑤＝生成は `GEN_PRUNE_FUTILE=False` で枝刈り無し・serve は従来どおり） |
| `tests/test_policy_teacher_shaping.py` | **基盤健全性**（`cpu_infra`）。**v7 教師整形（案D/E/F）**（`docs/cpu_v7_plan.md`・確定原因=教師が prior のエコー corr0.93 で prior が独立酔歩）: D=`p3_loop.priors_fn_of(flatten)`（生成 prior 部分平坦化 p′=(1−λ)p+λ/K・床・正規化保存）／E=`policy.smooth_target`（教師ラベル平滑化・床・順位保存）／F=`p3_loop.q_reweight(N,Q,β)`（Q補正教師 t∝N·exp(βQ)・高Q昇格・N=0不動）。**全て 0 で恒等**（後方互換）を固定 |
| `tests/test_selfplay_deep_relabel.py` | **基盤健全性**（`cpu_infra`）。**v6 深探索再ラベル**（`p3_loop.selfplay_game` の relabel_frac/relabel_sims・v5採用報告 §4-2＝PRIOR_BOUND 対策）: 各決定点を確率 frac で「深 sims × **prior 平坦化** × ノイズ無し」の教師探索にかけ policy 教師（訪問分布）だけを差し替える機構の配管＝ON で完走し教師分布が正規（Σ=1・legal と同数）・同一 seed 決定論・**frac=0 は乱数を追加消費しない**（従来生成列と完全一致＝後方互換） |
| `tests/test_selfplay_v4_datagen.py` | **基盤健全性**（`cpu_infra`）。**v4 自己対戦データ生成**（`p3_loop.selfplay_game` 拡張・`docs/reports/v4_adoption_20260712.md` §1）: q_root∈[-1,1]/turns_left（非負・終局で0）の記録・batch スキーマ v2（pack_vdata のキー/形状）・同一seed決定論・**sticky世界線**（同一(turn,手番)で決定化seed固定＝戦闘応答の交互手番でも dict で保持・ターンが変われば引き直す）・**防御応答の温度延長**（temp_moves=0 でも SELECT_BLOCKER/COUNTER は温度1でサンプリング）・**L1混合席**（policy教師はnet席のみ・L1席のvalueはq_root=NaN→mergeで勝敗へ退化・決定論維持） |
| `tests/test_value_net_aux_turns.py` | **基盤健全性**（`cpu_infra`）。**ValueNet 残りターン補助ヘッド**（W2t/b2t・v4 §4-2）: 補助ヘッドの value 出力からの独立性（＝恒等温スタートの根拠）・旧npz（v3=gen3）ロードで aux ゼロ＋save/load往復・解析勾配=数値微分一致（W2t/b2t＋共有層への寄与・NaNラベルのマスク）・構造拡張4種（expanded/LC/to_v3/widened）の aux 引き継ぎ・合成ターゲットの学習可能性（NaN混在可） |
| `tests/test_pd_mixed_label.py` | **基盤健全性**（`cpu_infra`）。**v4 混合ラベルとスキーマv2後方互換**（`pd_batch_common`）: normalize_batch_v2（v1バッチ→q_root=value/turns_left=NaN の退化規則・v2素通し）・mixed_value_label（α=1で勝敗単独と一致・線形補間）・ring_append の v2キー連結と v1/v2 混在 |
| `tests/test_pd_batch_common.py` | **バッチ式アクター/ラーナー分離の純粋協調ロジック**（`tests/scripts/pd_batch_common.py`・`docs/reports/batched_selfplay_design_20260710.md`）: 鮮度フィルタ is_fresh（accept/seen/stale の境界＝未消費かつ against_round>=round-staleness・**source="referee_label" は staleness 免除**＝gen5固定アンカー由来で腐らない/v9・seen 判定は免除しない）・plan_consumption の採用/スキップ内訳・update_consumed の単調性と非破壊・ring_append のcap切りとキー整合。git入出力を含むe2eはpd_*スモークで別途疎通確認済み |
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
| `tests/scripts/arena_parallel.py` | **並列アリーナ**（対照ペア×コア並列・旧 depth/thinktime_arena を統合）: 挑戦者の探索深さ/予算/PIMC/L1係数/**難易度（--challenger-difficulty learned＝既定Gen＝現gen8）/sims** を席別に振って Elo A/B。SPSA の f(θ) 評価にも使う |
| `tests/scripts/perf_gate.py` | **CPU 性能ゲート**（§5.1）: learned(既定Gen＝現gen8) vs 凍結 hard(L1) の Elo＋ペア単位 CI・1手レイテンシ・失敗局0・npz ハッシュを1コマンドで PASS/FAIL（`--quick`/`--full`） |
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
- **learned（本番既定 CPU＝現gen8）のトレース**: `--difficulty learned`（席別 `--p1-/--p2-difficulty learned`）で
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

### 5.0 交差対面の実プレイ監査（エンジン/パーサを変更したときの追加ゲート・2026-08-16）

```bash
make audit-cross                      # 既定 120 件・約10分（CROSS/CROSS_SEED で件数と対面集合を変更）
make audit-cross CROSS=240            # 変更が広いときは件数を増やす
```

**合格条件: hang / timeout / error = 0**（`ok` 以外が1件でも出たら push しない）。

なぜ `make test` に入れないか: 1件あたり実プレイ1局で 120件≈10分かかり、`make test`（約7分）を倍にする。
一方で**掛ける価値があるのはエンジン/パーサを触ったときだけ**なので、その作業単位でのみ追加する。

なぜミラー監査では足りないか: 137リーダーの**ミラー**（同一リーダー同士）監査は ok=137 / hang=0 なのに、
**交差対面**（左右で別リーダー）に広げると 240件中3件が落ちた（`void_root_causes_20260816.md`）。
3件はいずれも「ミラーでは一度も通らない経路」で、ドン!!へのフリーズ（対局が例外で落ちる）・
継続効果の再計算がコスト確認を無限に出す・自己参照コストが別カードで払える（無限ループ）だった。
**対面の組み合わせは測定の広さそのもの**で、狭い条件の green は品質の証明にならない。

`CROSS_SEED` は既定 0 固定（回帰確認は同じ対面集合で比べる）。**探索を広げたいときだけ変え、
変えたなら使った値をコミットメッセージかレポートに残す**（後から同じ集合を引けるように）。

### 5.05 アリーナ昇格判定の運用（2026-08-16 改定）

gen15 は「アリーナ 225ペア 0.5756 CI[0.533,0.619]＝歴代初の昇格」で出荷既定に採用したが、
**同じ条件で測り直すと再現しなかった**（3条件で 0.5028〜0.5222・いずれも昇格基準未達。
`gen15_recheck_20260816.md`）。1本の測定を昇格の証拠にしていたことが原因なので、以下を要求する。

1. **条件を2本以上**。主判定＝`--leaders random --decks synth`（ランダム対面×生成デッキ＝新カード・
   新リーダーへの汎化を直接測る条件）、副判定＝`--leaders fixed`（固定ミラー＝歴代と地続きの比較）。
   **主判定で昇格基準（wr≥0.55 かつペア水準CI下限>0.50）を満たし、副判定で退行していない**
   （CI 下限が 0.45 を下回らない）ことを昇格の条件とする。
2. **判定は測定時のコミットに紐づく**。エンジン（探索・解決・パーサ）を変更したら、
   過去の昇格証拠は**失効しうる**ものとして扱う。レポートには測定時の HEAD を必ず書く。
3. **void は件数を必ず出す**（`arena_resume` が `void` を結果に載せる）。void は勝率の母数から
   外れるだけで系統的な偏りを持ちうる（負けそうな側が延命する形のループなら勝率が歪む）ため、
   **void 率が 2% を超えたら判定を出さずに原因を潰す**（2026-08-16 の 5.4% は実バグ3件だった）。
4. **判定に使ったネットを消さない**。不採用候補でもリポジトリ（または退避先）に残す。
   gen16 は不採用後にネットが失われ、**当時の void 17件を同一条件で再現できなくなった**。
   判定が文書に生きている間は、その判定を再現できる成果物を保持する。
5. **台帳は消える前提**で運用する（作業台帳は gitignore・実行環境は巻き戻る）。
   チャンク境界ごとに中間集計（ペア数・wr・CI）を記録し、**結果はレポートへ落とす**。

### 5.1 CPU 性能ゲート（Gen2 非退行・手動/定期）

本番既定 CPU＝**learned（現gen8）** の強度・非退行を測る運用ツール（実対局は重いので `make test` 外・手動/定期）:

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
