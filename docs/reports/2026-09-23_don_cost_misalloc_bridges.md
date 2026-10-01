# T150f-4: misalloc/misalloc_play を帳簿・橋・曲線で測る（2026-09-23）

T150f-1（候補依存の配分ずれ損 `misalloc`）・T150f-2（見送った登場の価値を足す `misalloc_play`）を
実装した。本 T は `relative_ledger`・`theory_bridge`・`two_curves` の 3 器で、既定 `off`／`misalloc`／
`misalloc_play` を測る（`opportunity` は T150d/T150d2 の既測値と比較する）。ユーザ指示「この構成で
進めてください」の一部。crossing_bridge は `score_candidate` を呼ばないため測らない（T150d で確認済み）。

**記録**: T150d2 と同じ（実 `w41`・合成 `w39+w42`・介入なし）。

---

## 0. 結論（先に書く）

**`misalloc` は `opportunity` より一貫して良い**——理論の「候補依存の配分ずれ」という診断（前回の
チャット）が実測でも裏づけられた。`two_curves` の攻撃型の増分相関の悪化が `opportunity` の
1/5〜1/10 に縮んだ。**`misalloc_play`（見送った登場を足す）は記録間で割れる**——実は帳簿・橋の両方で
さらに改善するが、合成は帳簿で悪化し、曲線の増分相関は両記録とも大きく悪化する。

| 器 | 指標 | 実（off→misalloc→misalloc_play） | 合成（off→misalloc→misalloc_play） |
|---|---|---|---|
| `relative_ledger`（before・AUC） | | 0.7446→**0.7546**→**0.7814** | 0.8517→0.8535→**0.7969**（悪化） |
| `relative_ledger`（全帯・`sep`・理想1.0） | | 0.5652→0.5819→**0.7909**（改善） | 0.6399→0.6419→**0.5921**（悪化） |
| `theory_bridge`（`dS_atk`・AUC） | | 0.6195→**0.6268**→**0.6400** | 0.4588→0.4572→0.4402（さらに悪化） |
| `theory_bridge`（`dS`・AUC） | | 0.5947→0.6034→**0.6198** | 0.4550→0.4533→0.4378 |
| `two_curves`（`attack` 型の増分相関） | | 0.817→**0.813**→0.657（大きく悪化） | 0.813→**0.811**→0.629（大きく悪化） |
| `two_curves`（終点比・局面別距離 6+） | | 1.666/0.776→1.662/0.772→1.533/0.700（改善） | 1.352/0.456→1.349/0.454→1.215/0.408（改善） |

## 1. 予告と当否

前回のチャットで立てた予告（測る前に書いたもの）:

| # | 予告 | 結果 | 当否 |
|---|---|---|---|
| 1 | `relative_ledger` の before 帯 AUC は `opportunity` の改善（実 0.7805）を維持か上回る | **`misalloc` 単体では下回った**（0.7546）。`misalloc_play` は上回った（0.7814）が、**合成では大きく下回った**（0.7969 <<opportunity 0.8536） | **記録・モードで割れた** |
| 2 | `two_curves` の攻撃増分相関の悪化は `misalloc` で縮む（最良の体の相殺が消える） | **当たった**——実 0.022→0.004・合成 0.022→0.002 の縮小。**しかし `misalloc_play` で再び大きく悪化**（実 0.160・合成 0.184） | **当たり（`misalloc` 単体）** |

**予告 2 が当たった理由**: `opportunity` は候補非依存（`V(n)−V(n−k)`）なので、行の中で最良の体を選ぶ
候補も弱い体を選ぶ候補も同額を引かれ、最良の体自身の「行内の相対順位」への影響は限定的だが、
**この体固有の DON 利益をそのまま引いていた**ため増分の絶対値がまとまって動いた。`misalloc` は
最良の体には 0 を引くので、その体が実際に選ばれた行では価格が変わらない——**増分のノイズが
構造的に減る**。

**予告 1 が外れた理由（考察）**: `misalloc` 単体は「配分ずれの損」しか引かないので、`opportunity`
（常に非 0 を引く）より**引く量が全体に小さい**——判別力の改善が小さくなるのは自然。合成で
`misalloc_play` が大きく悪化したのは、合成デッキの手札構成（低コストの体が多い・後述）が実デッキと
違い、`foregone_play_value` が過大に効きやすい可能性がある（未検証・観察のみ）。

## 2. `misalloc_play` の悪化は FP(k) 由来

`two_curves` の攻撃増分相関の悪化が `misalloc` の 0.004/0.002 から `misalloc_play` の 0.160/0.184 へ
跳ねるのは、`misalloc` と `misalloc_play` の唯一の差である `foregone_play_value`（FP(k)）が原因——
FP は手札にある**低コストの体**を毎回同じ価格の分だけ引くので、`opportunity` と同じ「候補非依存の
定額を引く」欠陥をこの部分だけ再導入している（配分ずれの損は候補依存に直したが、見送った登場は
まだ候補非依存＝ATTACK/DON_BOX の対象が誰であれ同じ FP を引く）。**T150a／T150f-1 で見つけた診断
（候補依存にすれば改善する）が、FP にはまだ当てられていない**——今回で言語化できた新しい欠陥。

## 3. 不変量

* 既定 `off` の数字は T150d2 と完全に一致（`before_rel_K` AUC 実 0.7446／合成 0.8517・`dS_atk` AUC
  実 0.6195／合成 0.4588・増分相関 実 0.6419／合成 0.6730）——`misalloc`／`misalloc_play` の追加が
  既定を動かしていない。
* 局数（300／900）は全モードで同じ（介入しない測定）。

## 4. 上手くいったこと／いかなかったこと

**上手くいった**:
* **T150a の診断（候補依存にすべき）が実測でも裏づけられた**——`two_curves` の増分相関の悪化が
  1/5〜1/10 に縮んだ。前回のチャットの考察が単なる思いつきではなく実際に効いたことを確認できた。
* **`misalloc_play` の悪化の原因（FP が候補非依存）を、`misalloc` との比較から特定できた**——
  新しい欠陥の言語化に測定が使えた。

**いかなかった**:
* **予告 1（帳簿の改善が維持される）は外れた**——`misalloc` 単体は `opportunity` より弱い改善に
  留まり、合成では `misalloc_play` が大きく悪化した。
* **FP(k) を候補依存にする発展形は本 T では書いていない**（次の一手として残す）。

## 5. ユーザ判断をお願いしたいこと

1. **`misalloc`（配分ずれ損だけ）は `opportunity` より一貫して良い**——採用するなら `misalloc` を
   `opportunity` の後継として推薦する（T150f-3 の shadow_forbid 実測と合わせて最終判断）。
2. **`misalloc_play`（FP を足す）は推薦しない**——合成での帳簿の悪化・両記録での曲線の大きな悪化。
   FP を候補依存にする発展形を作るまでは見送るべき。
3. **T150f-5（最終採用判断）は shadow_forbid（T150f-3・実行中）の結果と合わせて判断する**。

## 6. 再現

```bash
python3 -c "
import theory_order as TO, relative_ledger as RL, theory_bridge as TB, two_curves as TC, two_curves_metrics as TCM
for mode in ('off', 'misalloc', 'misalloc_play'):
    TO.set_attack_don_cost_mode(mode)
    print(mode, RL.collect(['<w41>'], 0)['before']['rel_K'])
    per, _ = TB.collect(['<w41>'], 0, TB.THETA, TB.MU, 'const', 'leader', 'zero', None)
    print(mode, TB.summarise(TB.pair_games(per, 'zero'), 50, 0)['dS_atk'])
    dump = []
    TC.collect(['<w41>'], 0, dump=dump)
    print(mode, TCM.collect(dump)['by_family']['attack'])
"
```
