# v9 学習ループ運用手順書（教師CPU→外部アンカー学習）

v9 開発ループ（ラベル量産→学習→ゲート診断→特徴/レシピ修正→自動反映）の運用の正本。
設計の背景は `docs/cpu_v9_plan.md`、判定器の仕様は `TEST_SPEC.md` の各行（coach_gate /
referee_labeler / label_worker）を参照。**進捗の真実源は常に git**（ログではない）。

## 1. ループの全体像

```
[外部セッション×N] label_worker → claude/v9-label-wN 枝へ教師バッチ蓄積 push
        │（while ループ内蔵の git pull で本体修正を自動取込み）
        ▼
[本セッション] 蓄積監視 → しきい値到達で学習（ref_finetune_smoke）
        ▼
コーチゲート（進歩検出）─ FAIL → 点別診断 → 特徴追加 or レシピ修正 → push → 自動反映
        ▼ PASS
stage1 アリーナ（退行防壁）→ アンカーゲート（血統防壁）
        ▼ 全通過
gen6 採用提案（同梱 npz・既定切替・採用レポート）→ ユーザ承認 → マージ
```

## 2. ラベルワーカーの運用

### 起動プロンプト（正本・wN だけ変える）

```
opcg-sim-backend の v9 教師ラベル量産ワーカー（wN）を再開してください。
1. 既存のワーカープロセスがあれば停止
2. リポジトリのルートで次を実行（フォアグラウンドで回し続ける・クラッシュ時は自動再開）:
   while true; do
     git pull --ff-only
     OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/label_worker.py --worker wN --games 4
     echo "worker exited; restart in 60s"; sleep 60
   done
3. 進捗の確認は git 優先（claude/v9-label-wN 枝のコミット）。ログへの sed -i 等の in-place 編集は禁止
4. コンテナ再起動でループごと消えた場合は、このプロンプトの再実行だけで安全に再開できます
```

### 設計上の保証（w1 運用報告 2026-07-18 で確立）

- **停止・再開は常に安全**: 連番と seed は git（メタの累計局数）から自動再開。push 済みバッチは
  セッション状態に依存しない。失われるのは実行途中の1バッチ分の計算時間のみ（`--games 4` で
  10〜25分に抑制）。
- **コード更新の自動追従**: `label_worker` は `--batches`（既定20）を回しきると**正常終了して
  while ループの `git pull` に戻る**＝本体の新特徴/修正を自動で取り込む。無限ループにすると
  追従できない（2026-07-18 の25次元切替遅延の教訓）。**稼働中のワーカーが古いコードのままなら
  一度手動再起動が必要**（既に無限ループに入っているため）。以降は自動。
- **seed 無重複**: 割当ては累計局数ベース（`--games` を途中で変えても過去帯と重複しない）。
- **worktree 巻き戻り**: バッチごとに fetch→reset --hard origin で自己修復。手動 reset は
  fetch 失敗が続くときのみ。
- **禁止事項**: 稼働中プロセスのログへの in-place 編集（sed -i / truncate / mv 上書き）＝
  fd が切り離され進捗が見えなくなる。加工は別ファイルへコピーしてから。

### 健全性の目安

- push 間隔: `--games 4` で 10〜30分。**1時間以上無音なら停止とみなし再起動**（外部セッションは
  数時間でアイドル回収されるのが常態＝異常ではない）。
- ラベル形式は pol_am 幅（22/24/25…）と meta の `miner` 版数で判別（混在OK・学習側が自動吸収）。

## 3. 監視（本セッション・オーケストレータ）

集計はデータ枝の直接読みで行う（例）:

```bash
git fetch origin 'refs/heads/claude/v9-label-*:refs/remotes/origin/claude/v9-label-*'
# 各枝の p9label/batch_*.npz を読み、教師数・pol_am 幅（形式版）・最終 push 時刻を集計
```

- 報告粒度: ワーカー別（バッチ数・教師数・形式別内訳・最終push・生死判定）。
- 重複除外: `--games` 変更期の旧コード生成分は meta の seed0 帯の重なりで機械的に識別して
  学習から除外できる（通常は発生しない）。

## 4. 学習イテレーション

**発火条件**: 直近の特徴/採掘版の新形式ラベルが目安 500件（当たり付け）〜2,500件（本判定）。

```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/ref_finetune_smoke.py \
  --lrs 1e-4,5e-5 --epochs 8 --distill-weight 0.5 --policy-selfdistill 1.0 --out <出力dir>
```

- 既定レシピ: gen5 温スタート＋自己蒸留（value=distill 0.5・policy=gen5 prior 混合 1:1）＋
  policy-smooth 0.05。**正則化なしの素の微調整は mark ガード退行の実測があるため使わない**。
- 行動特徴が増えている場合は自動で温スタート拡張（零行＝恒等）される。旧形式ラベルはゼロ埋めで
  併用される（新特徴の学習だけが新形式依存）。

## 5. ゲート運転と判定

```bash
# ①コーチゲート（進歩検出・数分）
PYTHONPATH=tests python tests/scripts/coach_gate.py --challenger v.npz,p.npz --seeds 5
#   PASS 条件: 非退行（base≥0.8 の点で −0.4 超の下落なし）＋改善（ヒット計 ≥ gen5 の 3.0/7）
# ②stage1 アリーナ（退行防壁・約3分）
PYTHONPATH=tests python tests/scripts/promotion_gate.py --candidate v.npz,p.npz --best "" --pairs1 12
#   v9 の見方: 非退行（勝率 ≥ 0.5 目安）。勝ち越しは要求しない（効率改善は勝率に映らない）
# ③アンカー（血統防壁）: 世代が進む gen6 以降で --anchor を固定 gen5 に
```

**FAIL 時の診断手順**（コーチゲートの点別出力から）:
1. 落ちた点で候補が実際に何を選んだかを decide で確認（真盤面・数シード）
2. 分類: (a) 特徴飢餓＝選択の判断材料が行動特徴に無い → append-only で特徴追加
   （前例: カウンター値→+10pt・攻撃マージン）。(b) 反例不足＝文脈違いへの過汎化 →
   蓄積継続（採掘が反例を自然に拾う）。(c) レシピ過強＝広範な忘却 → 正則化強化。
3. 特徴追加時の規約: **append-only**（幅互換層が serve 恒等と旧記録ゼロ埋めを保証）・
   serve 挙動不変をテストで固定・TEST_SPEC 追記。push すればワーカーが自動で新形式へ切替。
4. ゲート合格手集合（coach_gate.VERIFIED）の変更は必ずレフェリー実測（真盤面・世界数明記）を
   出典にする。ゲートと候補が割れたら先にレフェリーで裏取り（@115 でゲート側が誤りだった前例）。

## 6. gen6 採用（全ゲート通過後）

1. 候補 npz を `opcg_sim/data/learned/gen6_*.npz` として同梱・既定切替（vocab_ids 継承を確認）
2. 採用レポート（`docs/reports/`・ゲート3種の数値・学習データの規模と形式内訳）
3. SPEC/TEST_SPEC 追従 → `make test` green → push → PR → **ユーザ承認でマージ**
4. 以降のレフェリー/ラベラーの教師ネットは gen5 のまま（錨は動かさない。錨の世代更新は
   別途ユーザ判断）

## 7. トラブルシューティング早見

| 症状 | 原因 | 対処 |
|---|---|---|
| ワーカー1時間無音 | セッション回収 | 起動プロンプト再投入（データ損失なし） |
| push=FAIL 連発 | ネットワーク/枝競合 | 放置で自動リトライ。続くなら fetch 失敗を疑い手動 reset |
| ラベラー失敗ループ | 本体コード不整合 | ワーカーに git pull 再実行を指示（コード修正はさせない） |
| 学習後にガード退行 | §5 の診断手順へ | 特徴飢餓/反例不足/レシピ過強を切り分け |
| ゲートと候補の見解相違 | ゲート定義が古い可能性 | レフェリーで裏取りしてから VERIFIED を更新 |
