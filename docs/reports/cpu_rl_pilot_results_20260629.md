# 学習RL Pythonパイロット 結果スナップショット（2026-06-29・GATE A/B/P2）

> **種別: 報告（点）**。実行結果のスナップショット。上書きしない。
> 計画＝`cpu_rl_pilot_plan_20260629.md`。本書はその GATE A→GATE B→P2 の実行結果と、
> 外部AIレビュー（計9往復）で確定した P3 への移行方針・損切りラインを記録する。

## 0. 結論

パイロット3関門すべて**陽性で通過**。実装バグ・探索ハイパラ・value→プレイ強度転換の
3交絡を順に潰し、いずれも前向き。レビュアー懸念の「偽陰性の未練」は発生せず。
→ **計画変更（P2固め）はレビューで却下＝先延ばし**。直ちに **P3（RL本走・アジャイル fail-fast）** へ。

| 関門 | 目的 | 結果 |
|---|---|---|
| GATE A | RLループ機械の実装正しさ | **PASS** |
| GATE B | OPCG MCTS探索の健全性＋PIMC | **PASS** |
| P2 | value予測の優位→プレイ強度の転換 | **陽性（本番CPUと互角・N=20）** |

## 1. GATE A（三目並べ＝RLループ機械の実装正しさ）

自己対戦→学習→重み更新＋NN誘導MCTS が既知最適へ収束するか（seed=0・8世代）。

- gen0（未学習）vs 完全プレイ(ミニマックス) = **14敗/40**（suboptimal）。
- **final（8世代）vs 完全プレイ = 0敗/40（全引分＝最適）**、vs ランダム = 0敗/100。
- 学習進行: value MSE 0.33→0.09、policy CE 1.51→0.82（単調改善）。
- 改善は「対完全プレイ敗北数」で測る（引分ゲームゆえ最適同士の直接対戦＝全引分で無情報）。

→ ループ機械は健全。OPCGの陰性を「実装バグ」から切り離せる。
（`gate_a_tictactoe.py` / `test_gate_a.py`(slow) / `test_az_components.py`(CI)）

## 2. GATE B（OPCG＝MCTS探索の健全性＋PIMC）

評価器を固定し sims だけ動かして「more search = stronger」か。

- **sims=270 vs 30 = 勝率0.81**（+0.50・厳密単調）。
- PIMC determinize: 相手伏せ手札を再サンプル（枚数保存・中身変化・自分の手札不変＝チート除去）。
- MCTS機械的健全性: MCTS(60/200) vs ランダム = 6勝0敗・訪問分布は尖る。

### 途中で踏んだバグ（記録）
当初は単調性NG。診断で **L1 evaluate の生スコアが桁違いに大きく**（中央 ≈ -5800・範囲[-11920,7091]）、
`tanh(score/6)` が**100%飽和**＝葉価値が符号だけの粗信号に潰れていた。`value_scale=10000`（飽和率0%・
std0.25）に較正してPASS。＝instrument②の設計意図通り「探索が効かない」をまず探索/評価の不具合として切り分けた。
（`opcg_game.py` / `az_mcts_tree.py` / `gate_b_opcg.py` / `gate_b_diag.py` / `test_opcg_adapter.py` / `test_az_mcts_tree.py`(CI) / `test_gate_b.py`(slow)）

## 3. P2/Gen0（必要条件チェック＝value優位のプレイ強度転換）

GATE B のMCTS葉価値を **学習SL価値net** に差し替え、相手を **本番 L1+α-β+PIMC(4)** にした対戦。

- **SL-net+MCTS(sims=160) vs L1+α-β(pimc=4) = 勝率0.450（9勝11敗・N=20・CRN）**。
- 構成は意図的に「下限」: 価値netは**7,505局面のみ**で訓練（本来10⁵〜10⁶）、MCTSは**policy head無し・
  一様prior**、単一合成デッキ。それでも本番CPUとほぼ互角。
- 訓練: self-play 140局→7505局面（397s）、train_mse=0.372 val_mse=0.493。

### 正直な留保
- **N=20は方向性のみ**（95%CI ±0.22）。統計的本ゲートは400戦CRN（外部計算資源）。
- 下限構成（小データ・uniform prior・単一デッキ）＝いずれもSLに不利な方向。
- レビュアー評: 「下限構成として異常な好成績＝メカニズムの勝利。極小Valueネットが盤面価値を高精度に
  捉え、重い探索木を正解へ導いている」。
（`p2_gen0.py` / `test_p2_harness.py`(CI)・`rl_net.ValueNet.save/load`）

## 4. レビュー確定事項（計画変更の却下と P3 方針）

外部AIレビューで「P3前のP2固め」案は**却下**。理由と確定方針:
- **P2固め（policy追加/データ拡大/N増/多様デッキ）は先延ばし**。静的SLを磨くだけでP3の目的
  （世代交代で強くなるか）のリスクヘッジにならない。P3のクロス評価がP2固めを包含。
- **P2でpolicy head（L1模倣）を足すのは悪手**＝模倣の天井を持ち込み、RLの「L1超え」を阻害。
  **uniform prior のまま P3 に入り、policy は RL で育てる**。
- **多様デッキ追加もNG**＝下限netをパンクさせ不要な偽陰性を生む。
- N=20は「致命的バグ無し・探索とvalue連動」の証明として十分。N増は無駄。

## 5. P3 移行設計（アジャイル fail-fast・損切りライン）

外部資源の浪費を最小化（空振りダメージを数時間〜半日に限定）:

```
[第1トランシェ・3世代上限]
 ① Gen0(現SL-net)で自己対戦 数千局 → Gen1 学習
 ② Gen1 vs Gen0 を N=100〜200 CRN でクロス評価
 ③ 損切り:
    - 続行 : Gen1 勝率 ≥0.55（95%CI下限>0.50）→ Gen2
    - 停滞 : <0.55 → Gen2 を1世代だけ猶予。Gen2 も対前世代<0.55 なら停滞確定
    - NO-GO: 2世代連続で対前世代 0.55 未達＝改善曲線がノイズ床で平坦
 ④ NO-GO 断定の前に必ず: 容量ラダー(P4) で表現力除外 ＋ c_puct 再較正。
    両方効かず GATE A 通過済み（機械は正しい）→ 初めて手法限界=NO-GO
```

- **数値損切り**: Gen3 までに「Gen_k vs Gen0」が0.55をCI付きで超えなければ停止（数日コミットしない）。
- 環境: 本走は安価な常設CPU VM（GPU不要・探索律速はCPU/Rust層／NN推論はL1と同オーダー＝実測済み）。

## 6. 成果物一覧（共通AZ部品＝P3/本走で再利用）

- `az_net.py`(Dual-Net value+policy) / `az_mcts.py`(状態キーMCTS・三目並べ) / `az_mcts_tree.py`(ノード型MCTS・OPCG)
- `az_loop.py`(自己対戦→学習→世代反復) / `tictactoe.py`(オラクル)
- `opcg_game.py`(エンジンアダプタ・PIMC・L1葉価値較正) / `p2_gen0.py`(SL net訓練＋対L1対戦)
- `rl_encoder.py`/`rl_net.py`(半生表現＋カードEmbedding value net・save/load) / `rl_datagen.py`(self-playデータ)
- CI内テスト: test_az_components / test_az_mcts_tree / test_opcg_adapter / test_p2_harness
- slow(CI除外): test_gate_a / test_gate_b
