# プロジェクト運用ルール（エージェント向け）

このファイルは Claude（エージェント）が本リポジトリで作業する際の取り決め。毎セッション従う。

## 開発・PR・マージのリズム

CI は無い（2026-07-11 廃止・下記参照）。**品質ゲート（次節）をローカルで通すことが唯一の確認手段**。

1. **ブランチで開発**しコミットする（指定があればそのブランチ。無ければ作業用ブランチを作る）。
2. **push 前に品質ゲート（`make test`）をローカルで実行し、全て green にする**。
   `make test` は約 2 分（2026-09-07 実測 418 passed・122 秒。Rust 化前は約 10 分だった）。
   長く待つコマンド（アリーナ・生成）は**バックグラウンドで実行**し、完了を待つ間もチャット
   対応・他の作業を止めない（フォアグラウンドでブロッキング実行しない）。
3. **push → PR 作成**する。PR 作成の時点で、チャットへ「PR出しました（ローカルテスト green）」と一報する
   （ユーザは iPhone アプリ利用。アプリを離れている間はこの一報が端末プッシュ通知になる）。
4. **マージはユーザの明示の指示があるまで実行しない**（`enable_pr_auto_merge` 等の自動マージ機能も使わない）。
5. マージすると PR 購読は自動解除される。マージ済み PR は再オープンしない。

## マージ前に緑であるべき品質ゲート

コマンドの正本は `Makefile`（CLAUDE.md / README 系はここを参照する。生コマンドを
個別に書き換えない）。`OPCG_LOG_SILENT=1` と `-s`（キャプチャ無効）は本スイートの必須フラグ。

```bash
make test        # push前ゲート（**これ 1 本**）。cargo test（Rust）＋ pytest -m "not slow and not legacy"
make test-slow   # 重テスト（slow マーカー・現状は空。追加したときだけ手動）
make test-fast   # 開発中のイテレーション用（slow・cpu_infra除外）
```

> **エンジンは Rust**（`rust/opcg_engine`・PyO3 wheel）。ルール・効果・探索は全て Rust にあり、
> Python が持つのは**カード本文の解釈（パーサ）・学習・API**だけ（`docs/rust_engine_plan.md` §17）。
> `make test` の前に **`make rust-develop` で wheel を入れておく**こと（無いと golden ゲートは
> skip ではなく fail する＝ゲートが黙って通らない）。
>
> **ゲートは `make test` 1 本**（2026-09-07・第 2 段 `rs-archive-cutover`）。Python エンジンは
> `legacy/python_engine/` へ退避してテスト対象外になった＝旧 `make test-legacy` は無い。
> 凍結した 1,786 本を回したいときは **tag（またはブランチ）を checkout して回す**:
>
> ```bash
> git checkout py-engine-final              # tag（ローカル）
> git checkout claude/py-engine-final       # 同じコミット c22f0a62 のブランチ（origin にある）
> OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -n auto -m "not slow" -p no:cacheprovider
> ```
>
> **ブランチを併記する理由**: この環境のプロキシは tag ref の push を 403 で弾く（実測・
> ユーザ環境でも同じ）。**リモートに置けるのはブランチだけ**なので、同じコミットを
> `claude/py-engine-final` としても push してある。中身は完全に同一。

> **ゲームプレイ退行の一次防衛線は golden 2 本**（いずれも Rust だけで回り
> `tests/fixtures/rs_goldens/` の記録と照合する）:
> `tests/test_rs_golden_audit.py`（カード×トリガー 3,386 件を汎用盤面で発動→既定解決）と
> `tests/test_rs_golden_replay.py`（実対局 200 局＝random 150・a1 50 の盤面／合法手／イベント）。
> **golden の正本は Rust**（2026-09-07・§16.3-10）＝作り直しも Rust だけで回る
> （`make golden-audit`／`make golden-replay`＝`tests/scripts/rs_golden_make.py`）。
> **挙動を意図的に変えたときだけ**作り直し、**差分は必ずレビューする**——golden は
> 「その時点の Rust の出力」であって、正しさの独立した証拠ではない。

> テストは**重要度**で3階層に分ける（時間ではない。詳細は `docs/TEST_SPEC.md` §重要度分類）。**基盤健全性**
> （探索/自己対戦/学習パイプラインの内部機構の健全性のみを見る。ゲームプレイの正しさ自体は
> 必須/標準テストが別途担保）は `cpu_infra` マーカーを付け、`make test-fast` で除外する。
> **push前は必ず `make test`（cpu_infra を含むフルスコープ）を green にする**——
> `make test-fast` はイテレーション用の速い一次チェックであり、push前ゲートを代替しない。
>
> **新しいテストを追加するとき**: 「無ければ実プレイのゲームプレイ退行（誤った効果解決／クラッシュ／
> カード消失／API契約破壊）を見逃すか」で判定する。Yes＝必須/標準（マーカー不要・常時実行）。No＝
> 探索/自己対戦/学習パイプラインの内部機構の健全性のみを見る基盤健全性＝`@pytest.mark.cpu_infra`
> （module-level `pytestmark` 可）を明示する（迷ったら必須/標準側に倒す）。
> `docs/TEST_SPEC.md` §2 への追記時、基盤健全性はその旨を明記する。

> **エンジン/パーサを変更したときは `make audit-cross`（交差対面の実プレイ監査）も通す**
> （合格＝void〔決着せず〕= 0）。ミラー監査では一度も通らない経路があり、そこにしか出ない
> 欠陥が実在する（2026-08-16 に3件検出）。詳細は `docs/TEST_SPEC.md` §5.0。

> **アリーナで新世代を昇格させるときは条件を2本以上測る**（主=ランダム対面×生成デッキ／
> 副=固定ミラー）。1本の測定は昇格の証拠にならない——gen15 の「歴代初の昇格 0.5756」は
> 同条件で再現しなかった。判定は測定時のコミットに紐づき、エンジンを触れば失効しうる。
> void 率が2%を超えたら判定を出さず原因を潰す。**判定に使ったネットは不採用でも消さない**
> ——同梱の npz は消さず、消したくなったら**そのコミットの履歴と tag で辿れること**を確かめてから
> にする（gen16 は失われ、当時の void を再現できなくなった。tag 例: `py-engine-final`）。
> 詳細は `docs/TEST_SPEC.md` §5.05。

- 全テスト pass（`test_effect_oracle_gate.py`＝HAS_OTHER/PER_TURN_LIMIT_GAP/UP_TO_GAP = 0 の
  ラチェットを含む）。挙動ベースライン（旧 `test_full_card_baseline.py`）は監査 golden が置き換えた。

> **CI 廃止の経緯（2026-07-11）**: GitHub Actions CI（lint+pytest）はローカル品質ゲートと全く同じ
> コマンドを二重実行していただけだった（`.github/workflows/ci.yml` は削除済み）。ローカルで
> `make test` が green であることをそのままマージ可否の判断材料にする。CI という独立した
> クリーン環境での再検証が無くなる分、**push 前に品質ゲートを飛ばさないこと**が今まで以上に重要。

詳細は `docs/TEST_SPEC.md`（§4 検証フロー／§5 品質ゲート）、`docs/README.md`（文書索引）を参照。

> **配置規約**（2026-09-07・計画 §17）:
> `opcg_sim/api`＝FastAPI／`opcg_sim/loop`＝生成・アリーナ・ゲート判定（Rust で回す）／
> `opcg_sim/learned`＝ネット定義と符号化の仕様、`learned/train`＝訓練 CLI／
> `opcg_sim/src/effects`＝パーサ（裁定を書く場所その1）／`opcg_sim/src/models`＝型／
> `opcg_sim/tools`＝効果 JSON・API 契約の生成／`rust/opcg_engine`＝エンジン（裁定を書く場所その2）。
> `tests/test_*.py`＝pytest テスト、`tests/harness/`＝テストが import する基盤ライブラリ、
> `tests/scripts/`＝単体実行の実験/計測/監査 CLI、`tests/fixtures/`＝データ資産。
> 共通ブートは `tests/_bootstrap.py`（sys.path＋google スタブ）。
> `legacy/python_engine/`＝凍結した Python エンジンとその周辺（テスト対象外・
> **`opcg_sim/` からここを import しない**）。
> **1トピック=1ファイルを維持し、ファイル数削減のための統合はしない**（ユーザ決定 2026-07-04・
> ファイル名を索引として使う方針）。見通しは `docs/TEST_SPEC.md` §2/§3 のスイート一覧（正本索引）で
> 確保する＝**テスト/計器を追加したら同表へ1行追記**する。

> API 契約: `shared_constants.json` または `opcg_sim/api/schemas.py` を変更したら
> `python -m opcg_sim.tools.export_contract` を実行し `contract/`（api_schema.json / manifest.json）を
> **同じ作業単位でコミット**する。`test_contract_export.py` が再生成差分ゼロをラチェットする
> （生成物が古いと `make test` が落ちる）。フロント側の型生成・定数同期は別リポジトリの PR で追従する。

## 開発判断の前提（ユーザ決定 2026-08-25）

- 本シミュレータは**私用**であり、CPU はまだ実用水準に達していない＝ほぼ使われていない。
  **「出荷・本番運用・既存利用者への配慮」を判断材料にしない**。互換性・実デッキ適合より、
  汎化と強さの根本改善を常に優先する（例: G17 は実デッキを訓練に入れず全ランダムで学習し、
  実デッキは完全ホールドアウトの検証帯とする）。

## 学習データ窓の運用（ユーザ決定 2026-08-28・N系）

- **π（訪問分布）は現 era（生成器の世代）のみ**・**z（勝敗）は現 era＋直前2 era**を訓練に使う
  （z/π 分離の実証: `docs/reports/2026-08-28_zpi_split.md`。旧 era の π は方策を引き戻す毒・
  z は価値教師として古びにくいが、打ち方が変わるほど「その後の進行」が別物になるため区切る）。
- **z は新しい波から順に、実行環境に載る最大数を使う**（ユーザ決定 2026-09-01・上の
  「現+直前2 era」を置き換える）。波が積み上がるとメモリ上限（コンテナ cgroup 14GB）に
  当たるため、**古い波から順に落として上限内に収める**——落とす基準は era 境界ではなく
  「新しい順に入るだけ」。訓練起動時に RSS を実測し、OOM したら1波減らして再開する
  （2026-09-01 に波1〜11＋π波15 の 694万行で OOM＝13.9GB を実測）。
- **例外: 世代交代時のアリーナ勝率が極端に高い場合は z 窓を狭める**——勝率 ≥0.65（+100 Elo級）
  なら現+直前1 era・≥0.75 なら現 era のみ（強くなるほど過去の z が早く古びるため）。
- 引退したデータもブランチ保全は継続（評価帯・再採掘・監査用）。
- **評価帯（固定の定点）は世代交代のたびに更新する**——古い波の holdout 行は「古いスタイルの
  進行」を予測する課題なので、打ち方が変わった新世代に構造的に不利に働く（2026-09-01 実測:
  旧帯=波7+8 では c8pre が最良で c8 は退行に見えたが、全ネットが未見の新帯=波16〜18 では
  世代順に単調改善し c8 が最良。順位が逆転した）。**新帯は「既存の全ネットが訓練後に生成された
  波」から採る**＝比較性が仮説ではなく事実として成立する。判定はあくまでアリーナを主とし、
  評価帯は補助指標として読む。

## N系ループの分散運用（ユーザ決定 2026-09-01）

- 教材収集・学習・検証は**全て独立した作業セッション**で実行し、コーディネータ（本セッション）は
  指示書を書いて成果物を回収・判定するだけ。受け渡しは **origin のブランチ経由のみ**・成果物には
  `RESULT.json` を添える・seed 帯はコーディネータが台帳で払い出す。正本は `docs/n_loop_ops.md`。

## CPU 系統の呼称（ユーザ決定 2026-08-11・指示・会話・文書でこの略称を使う）

> **エンジンは Rust**（`rust/opcg_engine`）。ルール・効果・探索（MCTS＋PIMC＋箱化）と
> ネットの forward は全て Rust にあり、Python が持つのは**パーサ・学習・API** だけ。
> 生成・アリーナ・serve は同じ `Game.decide` を呼ぶ＝**物差しは 1 本**
> （2026-09-07・第 2 段 `rs-archive-cutover`・`docs/rust_engine_plan.md` §17）。
>
> **L1（手作り評価 L1・α-β＋ビーム＋PIMC）は廃止**（ユーザ決定 2026-09-07・§16.3-8）。
> `cpu_ai.py`／`cpu_eval_v2.py` は `legacy/python_engine/core/` にあり、tag `py-engine-final` を
> checkout すれば動く。`--policy l1` の記録・golden の L1 帯は **N系（a1）で打ち直した**。
> 対戦 API の `cpu_difficulty` は "hard"／"learned" のどちらも同じ N系エンジンが打つ
> （フロント互換のために欄だけ残っている）。

- **N系** = 効果構造符号化の学習CPU（カードID埋め込みなし）。2 世代ある:
  - **NRel（N系 v3・rN/aN）** = 22 枠の「効果構造＋今の状態＋ゾーン」トークン＋対の小ネット
    （`opcg_sim/learned/n_rel.py`・`n_rel_feat.py`・符号化 v13・`docs/n_attention_plan.md`）。
    **現在の出荷既定は NRel a1**〔2026-09-05 採用・`nrel_a1.npz`・`cpu_learned._DEFAULT_VALUE` が正本・
    `docs/reports/a1_adoption_20260905.md`〕。構成は**関係 R なし・相手デッキ知識あり**（ユーザ決定
    2026-09-05・訓練は `n_rel_train.py --ablate rel` が既定・系譜は r2, r3, … を続ける）。
  - **c 系（cN）** = 旧 N系（`n_eff.py`・符号化 v12・最終 c10=2026-09-03 採用）。ロールバック先
    （`_C10_VALUE`）。その先は G15 ペア（`_G15_VALUE`/`_G15_POLICY`）。
- **G系** = 旧本流の学習CPU（genN・埋め込みあり・最終世代 **G15=gen15**
  〔2026-08-15 採用・符号化v12＋戦闘出口ヘッド。2026-09-03 に既定を c10 へ譲った〕。
  **Rust エンジンは G系の value/policy ペアを載せない**（物差しを 1 本に保つ・§8.17 の設計 1）＝
  同梱は続けるが、生成・アリーナ・serve では使えない。動かすなら tag `py-engine-final`。
  **候補ヘッドの土台や比較基準は必ず現既定に合わせる**
  ——2026-08-22 に gen14 土台で作った候補が coach_gate で「偽の退行」を出した実害あり）
- **B系** = 骨組み線（リーダー非依存・合成カードのドメインランダム化・`bb_*`/bbN・
  `docs/cpu_backbone_plan.md`。G系とはバージョン空間・コード・成果物を分離、依存は B→G の一方向のみ）

## ドキュメントの更新

- 仕様（正本: `docs/SPEC.md` / `TEST_SPEC.md` / `parser_v2.md` / `leader_specs/`）は実装変更に追従して最新に保つ。
- 報告（`docs/reports/`）は特定時点のスナップショットで追記・改変しない。
