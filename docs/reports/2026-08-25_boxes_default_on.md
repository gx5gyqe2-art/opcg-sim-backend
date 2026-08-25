# 箱化インフラ一括採用: 全箱 ON をアリーナで確認し既定化（2026-08-25）

スナップショット。マクロ手化（`docs/cpu_macro_plan.md`）の箱インフラ完成と、
**測定済みフラグの既定 ON 切替（ユーザ決定 2026-08-25）**の記録。

## 完成したインフラ（本日時点）

| 層 | 機構 | 状態 |
|---|---|---|
| 箱（自分ターン） | 配分箱 P1・アタック箱 P2（`SERVE_MACRO_MOVES`）／カード使用箱・効果起動箱=対話箱 P3/P5（`TREE_BOX_DIALOG`） | **既定 ON** |
| 箱（相手ターン） | 防御箱 P4-c（`SERVE_DEFENSE_BOX`）／応答箱・トリガー=対話箱 | **既定 ON** |
| 既存の戦闘箱 | 静止探索・窓読み出し・入口コミット・木の箱化 | 既定 ON（gen12〜） |
| プラン層 | ターン箱 P6-a（プラン機構の箱語彙対応・`--cand-plan-box`） | 実装済・**既定 OFF**（未測定） |
| 方針層 | 受け方針箱 P6-c（`guard.py`・local/pass/minimal/hold・`--cand-guard-policy`） | 実装済・**既定 OFF**（未測定） |
| 生成側 | p3_run 等は箱 kwargs 未指定＝**serve 既定に自動追従**（生成/serve 一貫） | 配線済 |

## 判定材料（全箱 ON vs 既定 OFF・両席 gen15 同ネット＝機構だけの A/B）

分散測定: ローカル96ペア×2条件＋ワーカー6セッション（各48ペア・台帳ブランチ
`claude/arena-boxes-{r,m,w1r,w3r,w5r,w2m,w4m,w6m}`）をペア水準で合算。

| 条件 | ペア | void | 勝率 | 95%CI | Elo |
|---|---|---|---|---|---|
| 主: ランダム対面×生成デッキ | 224 | 0 | 0.5223 | [0.4720, 0.5726] | +16 |
| 副: 固定ミラー | 240 | 0 | **0.5625** | [**0.5224**, 0.6026] | **+44** |

- **ミラーで有意プラス**（下限>0.50・n=240）。ランダム対面は中立プラス（有意ではない）。
- P1（0.505/0.510）・P2（0.474/0.458）・P4-c（0.442/0.526）と**単独では3回連続 PAR** だった
  箱が、対話箱を加えて全部同時に ON にすると初めて有意差が出た＝「読みの浪費を全窓型で
  消すと同じ探索量の価値が上がる」（複利仮説）と整合。
- coach_gate（裁定点・16seed・`--chall-boxes`）: **PASS**（9.0 vs 9.0・ノイズ幅超の変化ゼロ）
  ＝積み上げた裁定挙動（ウタ/素通し/付与/掘り/エネル）を箱化は壊さない。
- 対話窓の decide 実測 ~1.2s → ~0.01s（約100倍）。
- 昇格ルール（2条件優位）には主条件が届かないが、本件はネット昇格ではなく**探索機構の
  既定**の判断。前例のドン箱（A/B 中立でもユーザ判断で ON）より強い証拠
  （片条件有意＋片条件中立プラス＋高速化＋一貫性）で**ユーザ決定により一括 ON**。

## 切替内容（コミット参照）

- `SERVE_MACRO_MOVES = True`／`SERVE_DEFENSE_BOX = True`／`TREE_BOX_DIALOG = True`
- `SERVE_GUARD_POLICY = False`・プラン読み出し OFF のまま（P6 は seam 完備・測定待ち）
- ロールバック手順: 3フラグを False に戻せば 32a31c1 以前の serve 挙動（ネット不変）。

## 残課題（次の測定対象）

1. ターン箱（`--cand-plan-box`）と受け方針箱（`--cand-guard-policy`）のアリーナ2条件
2. turn 出口ヘッドの学習接続（枝順位ゲートを検証に組み込む条件つき・P4 の教訓）
3. ランダム対面での有意化（箱の上の評価改善＝採点係の較正が本丸）

## 再現手順

```bash
# 全箱 ON vs OFF（両席 gen15）
python tests/scripts/arena_resume.py --candidate <gen15paths> --cand-boxes-all \
  --leaders random --decks synth --pairs 96 --seed-base 77000 \
  --workers 2 --pair-timeout 1800 --out <ledger>
python tests/scripts/coach_gate.py --challenger <gen15paths> --chall-boxes --seeds 16
```
