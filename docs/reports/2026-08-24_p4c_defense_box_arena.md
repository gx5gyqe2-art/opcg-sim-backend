# P4-c 防御箱 v1 アリーナ判定: PAR・現形は既定 OFF 維持（2026-08-24）

スナップショット。マクロ手化 P4-c（`docs/cpu_macro_plan.md` §5・防御窓の D1'/D2' 支配則に
よる候補整形）のアリーナ A/B 測定と判定の記録。実装と前段の判定は
`docs/reports/2026-08-24_p4_defense_verdict.md`（D族ヘッド不採用・出口採点ルート打ち止め）。

## 測定

- 対象: 防御箱 ON（候補席）vs OFF（基準席）。**両席とも gen15 既定ネット**＝機構だけの A/B。
- 実装コミット: 33988c29（make test green 1729・契約テスト6件・監査 ON/OFF 比較済み）。
- 方式: `arena_resume.py --cand-defense-box`・96ペア（CRN 席交換対）・ペア水準CI・
  チャンク8ペアごとに台帳を専用ブランチへ逐次退避（`claude/arena-p4c-r` / `claude/arena-p4c-m`）。
- 監査の事前確認: 防御窓の乖離 5/22 → 2/13（消えた窓は支配則が算術で確定させたもの・
  残る乖離2は「払えるが割に合わない」経済判断＝支配則の守備範囲外）。

## 結果

| 条件 | ペア | void | 勝率 | 95%CI | Elo |
|---|---|---|---|---|---|
| 主: ランダム対面×生成デッキ | 95/96 | 1（MAX_STEPS 4000・1.04%≦2%） | 0.4421 | [0.3705, 0.5137] | −40 |
| 副: 固定ミラー | 96/96 | 0 | 0.5260 | [0.4638, 0.5882] | +18 |

- 両条件とも CI が 0.5 を跨ぐ＝**有意差なし（PAR）**。昇格条件（2条件で優位）は不成立。
- 主条件はやや下振れ・副条件はやや上振れで、方向も揃わない。
- void 1件はステップ上限（`InvariantError MAX_STEPS`・seed 376010 ST02-001 vs OP07-019）で
  防御箱起因のクラッシュ/hangではない。

## 判定

**現形は不採用＝`SERVE_MACRO_MOVES` 同様、`SERVE_DEFENSE_BOX` は既定 OFF を維持**。
seam・純関数・契約テストは残す（防御箱 v2 以降の土台）。

## 読み取り（P1/P2 と同じ教訓の再確認）

1. **支配則は局所的に正しくても勝率を動かさなかった**。防御監査で消えた乖離（過剰防御の
   一部）は実対局の勝敗を左右する頻度が低い、または既存の出口評価が同じ結論に達していた。
2. P1（配分箱 PAR）・P2（アタック箱 PAR下限）に続き、**探索の単位や候補整形を変えても
   採点係（value）が同じなら勝率は動かない**という系全体の教訓がまた再現した。
3. 残る防御の乖離2件は「払えるが割に合わない」**経済判断**（m1@14 型）で、これは支配則では
   閉じない。防御で勝率を動かす余地があるとすれば、(a) 経済判断を扱う評価の較正
   （ただし出口ヘッド再学習は打ち止め済み＝裁定点の枝順位を検証に組み込まない限り再開しない）、
   (b) ターン全体の受け方針（P6 受け方針箱）の側にある。

## 再現手順

```bash
export PYTHONPATH=tests:tests/harness:tests/scripts OPCG_LOG_SILENT=1
python tests/scripts/arena_resume.py \
  --candidate opcg_sim/data/learned/gen15_value.npz,opcg_sim/data/learned/gen15_policy.npz \
  --cand-defense-box --leaders random --decks synth --pairs 96 --seed-base 76000 \
  --workers 2 --pair-timeout 1800 --out <ledger_r.jsonl>   # 主（副は fixed/singleton・76500）
```

台帳の正本: `claude/arena-p4c-r` / `claude/arena-p4c-m` の `arena_p4c/ledger_*.jsonl`。
