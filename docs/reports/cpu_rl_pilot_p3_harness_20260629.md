# 学習RL パイロット P3 ハーネス構築 スナップショット（2026-06-29）

> **種別: 報告（点）**。P3（AZ自己対戦RLループ）ハーネスの構築完了時点の記録。上書きしない。
> 前提＝`cpu_rl_pilot_plan_20260629.md`（計画）＋`cpu_rl_pilot_results_20260629.md`（GATE A/B/P2 結果・
> P3損切りライン）。本書は dev環境で構築可能な P3ハーネス（＋疎通試走）の到達点を記録する。

## 0. 結論

P3 の**本走ハーネスを構築し、OPCG 上で end-to-end に動くことを疎通確認**した
（自己対戦→value/policy学習→世代間クロス評価が例外なく完走）。
最難関だった**アクションの正準符号化**を解決。**勝率シグナルは判定に使わない**
（レビュー確定＝数局モデルの勝率は乱数）。

**本走（1万局/世代・N=400 CRN・3世代上限の損切り）は常設CPU VM前提**。
dev環境（揮発性）でできるのはここまで＝ハーネス＋疎通。

## 1. 解決した最難関＝アクションの正準符号化

OPCG の合法手は heterogeneous（self-play 実測で11種：MULLIGAN/KEEP_HAND/TURN_END/PASS/
PLAY/ATTACK/ATTACH_DON/SELECT_COUNTER/SELECT_BLOCKER/ACTIVATE_MAIN/RESOLVE_EFFECT_SELECTION）。
これを policy head が出力するための表現として、**ポインタ/marginal方式**を採用:

- 巨大疎な直積空間（カード×型×対象）は作らない（レビューで「numpyでは地獄」と判定された方）。
- 各合法手を**固定長 ACTION_DIM 次元**へ符号化＝`[action_type one-hot] ++ [関与カード特徴(rl_encoder._char_feats同型)] ++ [所有者flag, has_card, has_target]`。
- policy は `[状態文脈(94) ++ action特徴]` から同一MLPでスカラ logit を出し、**合法手上で softmax**＝可変個の手に自然対応。
- `action_key`＝探索木の手の同一性キー（dict非hashable対策）。

（`opcg_action.py`）

## 2. value/policy 分離型 Dual と RLループ

- **value**: `rl_net.ValueNet`（半生表現＋カードID Embedding・outcome 教師）。MCTS の葉価値。
- **policy**: `az_policy.PolicyScorer`（ポインタ型・MCTS訪問分布が教師）。MCTS の事前確率。
  - value と policy は**別ネット**（共有trunkより numpy 実装が単純で正しさを担保しやすい・AZ的には等価）。
- **RLループ** `p3_loop.py`: NN誘導MCTSで自己対戦→`(局面, 訪問分布, 最終勝敗)`採取→value/policy学習→
  世代間クロス評価（CRN・先後交互）。
  - **policy は uniform から RL で育てる**（P2でL1模倣policyを足さない＝模倣の天井回避・レビュー確定）。
  - `az_mcts_tree.run` を `(move, N, legal)` 返しに拡張（policy教師の整列用）。

## 3. 疎通検証（インフラのみ・勝率は無視）

```
自己対戦6局 → value局面1339 + policy1339 採取
 → value net 学習(mse 0.07) + policy 学習(CE 2.17)
 → Gen1 vs Gen0 クロス評価   ← 例外なく完走
```

テンソル不整合・パイプライン詰まり無し＝**RLループ機械が OPCG 上で end-to-end に動く**ことを確認。
※ Gen0 は乱数 value net の疎通試走。**勝率(0.333)は乱数＝判定に一切使わない**（レビュー確定）。

### テスト
- CI内: `test_p3_components.py`（6件: 符号化の形/one-hot/同一性キー/policy正規化/policy過学習/状態文脈次元）。
- slow(CI除外): `test_p3_loop.py`（end-to-end 疎通＝例外なく完走）。

## 4. 残作業（dev環境でできる仕上げ／VM前提の本走）

dev環境で可能:
- **損切り判定ロジックの p3_loop 組込み**（N=400 CRN・「Gen_k vs Gen0 ≥0.55」・Wilson CI 等）＝
  VM投入前に判定基準をコード化（再現性確保）。

VM前提（揮発性devでは不可）:
- **P3本走**: Gen0=P2のSL net 起点・**1万局/世代**の自己対戦・**N=400 CRN** の損切り判定・**3世代上限**。
  - 続行: Gen1 vs Gen0 ≥0.55 → Gen2（以降の対前世代は 0.51〜0.52＝後退でないこと）。
  - NO-GO: Gen3 までに「Gen_k vs Gen0」が0.55未達。断定前に容量ラダー(P4)＋c_puct再較正。
  - 時間見積り（補正）: 1万局MCTS自己対戦＋N=400は重く、第1トランシェは「半日」でなく**1〜3日**。
- 環境: 安価な常設CPU VM（GPU不要＝探索律速はCPU/Rust層・NN推論はL1と同オーダー・実測済み）。

## 5. パイロット到達サマリ

| 段階 | 状態 |
|---|---|
| GATE A（ループ機械の実装正しさ） | PASS（三目並べ最適収束） |
| GATE B（探索の健全性＋PIMC） | PASS（more search=stronger） |
| P2（value→プレイ強度の転換） | 陽性（本番CPUと互角・N=20） |
| **P3 ハーネス＋疎通** | **完了（dev環境の上限）** |
| P3 本走（GO/NO-GO） | 未（常設CPU VM 待ち） |

## 6. P3 成果物一覧（本走で再利用）

- `opcg_action.py`（アクション正準符号化） / `az_policy.py`（ポインタ型 policy）
- `p3_loop.py`（自己対戦RLループ＋クロス評価） / `az_mcts_tree.py`（(move,N,legal)拡張）
- value=`rl_net.py`（save/load） / 状態符号化=`rl_encoder.py`
- テスト: `test_p3_components.py`(CI) / `test_p3_loop.py`(slow)
