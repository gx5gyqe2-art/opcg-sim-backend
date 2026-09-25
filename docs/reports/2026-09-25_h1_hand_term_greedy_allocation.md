# H-1: 耐久の手札の項に「攻撃ごとに安い順へ割り当てる」式を実装(2026-09-25)

`crossing_bridge.py`の`THETA_HAND_MODE`に新しい値`cuttable_seq`(T158)を追加し、T99が指した
「攻撃ごとに割り当てる」形を実装した。既存の`hand_absorb`(T99)・`hand_absorb_forced`(T100)は
どちらも「同じ大きさの組」で手札を割る粗さが残っていた——本作業はそれを「攻撃ごとに実際の
費用をそのまま使う」真の貪欲法に置き換える新モードを、既定を変えずに切替として追加した。
新定数ゼロ・実装/テスト/ゲート/健全性チェックのみ(実・合成の本格測定はH-2)。

## 背景——既存2式との違い

* `hand_absorb(n_cut, x_max, mu)` = `μ × c(x_max) × floor(n_cut / c(x_max))` ——**1本の最重量攻撃**
  だけで手札全体を割る(端数を捨てすぎる・T99の欠陥)。
* `hand_absorb_forced(n_cut, xs, life_opp, n_blockers_opp, mu)` = `μ × c_eff × floor(n_cut / c_eff)`・
  `c_eff` = 強制される守り回数`G`本の**平均費用**(安い順に`G`本取って平均)——単一の攻撃ではなく
  複数を考慮するが、まだ「同じ大きさの組」で割っている。
* 新形`hand_absorb_seq(n_cut, xs, mu)`(T158) = 攻撃を`c(x)`の安い順に並べ、手札が尽きるまで
  **攻撃ごとに実際の`c(x_i)`枚をそのまま割り当てる**(真の貪欲法・T64「一番安い札から切る」と同じ規則)。

```python
cs = sorted(c_of(x) for x in xs if c_of(x) > 0)
remaining = n_cut
absorbed = 0.0
for c in cs:
    if remaining >= c:
        absorbed += c
        remaining -= c
    else:
        break   # 端数は一生 F に入らない(次以降の攻撃も止められない)
return mu * absorbed
```

## 実装

`tests/scripts/crossing_bridge.py`:
- `THETA_HAND_MODES`に`"cuttable_seq"`を追加(既定は`cuttable_forced`のまま・据え置き)。
- `THETA_HAND_PART["cuttable_seq"] = "cuttable"`(1枚あたりの価格の出どころは`cuttable_cx`/`cuttable_forced`と同じ)。
- `hand_absorb_seq(n_cut, xs, mu=MU)`を`hand_absorb_forced`の直後に実装(同じ関数シグネチャ・
  docstringの文体・T番号付きコメントに合わせた)。
- `threshold_parts_side`の既存分岐(`THETA_HAND_MODE in ("cuttable_cx", "cuttable_forced")`)を
  `"cuttable_seq"`も含む3値に広げ、`elif THETA_HAND_MODE == "cuttable_seq": hand = hand_absorb_seq(n_cut, xs, mu)`
  を追加。
- CLI `--theta-hand`のヘルプ文言に新モードを追記。

## テスト(`tests/test_crossing_bridge.py`・4本追加)

1. **`test_the_hand_absorbs_by_price_order_per_attack`**: 手で構成した3攻撃シナリオ
   (超過0/1000/3000・`CBAR_MODE=strict`)で、貪欲割り当てが最初の2組だけを丸ごと止め3組目は
   端数不足で打ち切ること、`hand_absorb`とも`hand_absorb_forced`とも異なる値になることを確認。
   通らない攻撃のみ・1回分に届かない薄い手札は0になることも押さえる。
2. **`test_the_hand_absorbed_by_price_order_does_not_carry_the_remainder_forward`**: 手札を
   ちょうど2組の費用の和に合わせ、割り切った後の余り(正確に0)が3組目に一部でも回らないことを
   直接確認(手札を少し増やしても3組目は依然0本のまま)。
3. **`test_selecting_cuttable_seq_leaves_the_older_hand_modes_byte_for_byte`**: `count`／
   `cuttable_cx`／`cuttable_forced`のそれぞれについて、`threshold_parts`が返す手札の項が
   既存の素の式(`μ×枚数`／`hand_absorb`／`hand_absorb_forced`)と完全に一致することを確認——
   「新モードを選ばない限り旧モードは変わらない」不変量。
4. **`test_cuttable_seq_is_wired_into_threshold_parts_and_rejects_unknown_modes`**: 新モードを
   選ぶと`threshold_parts`の手札の項が`hand_absorb_seq`そのものになること、知らないモード名で
   `set_theta_hand_mode`が`ValueError`を出し既定を汚さないことを確認。

`tests/test_crossing_bridge.py`は89本全て pass(追加前84本)。コーディネータ側でも独立に
再実行し(`89 passed in 1.56s`)、実装エージェントの自己報告と一致することを確認した。

## ゲート

`make test`(cargo test 429 passed ＋ pytest)は green。実装エージェント自身のworktree内では
**1824 passed, 8 warnings in 106.37s**、コーディネータが同じworktreeで独立に再実行しても
**429 passed**(cargo)＋**1824 passed**(pytest, 95.69s)、さらにこのコミットをメイン
ブランチへcherry-pick(`71d1b9fe`)した上で再実行しても**1824 passed**(99.06s)で、3回とも
一致してgreen。

途中、フレッシュな worktree での初回実行で`tests/test_effect_value.py`の5本が
`opcg_sim/data/opcg_effects.json`の`FileNotFoundError`で落ちた(実装エージェントの報告)。これは
本diffと無関係——このファイルは gitignore 対象の生成物で、`RsGame.__init__`
(`export_effects_json.ensure`)がオンデマンドで作る設計になっており、`effect_value.load_cards()`は
`ensure()`を経由せず直接開くため、真にフレッシュな worktree で最初にこのファイルへ触れるテストが
どの worker で先に走るかという pytest-xdist のスケジューリング次第で稀に競合する。ファイルが
一度生成された後に`make test`を再実行すると完全に green になることを確認済み(コーディネータの
3回の再実行でもこの競合は再現しなかった)。

## 健全性チェック(壊れていないことの確認のみ・本格測定はH-2)

```bash
python tests/scripts/crossing_bridge.py --in <w41> --theta-hand cuttable_seq
python tests/scripts/crossing_bridge.py --in <w39> <w42> --theta-hand cuttable_seq
```

出力全体(JSON全ノード)を走査してNaN/Infが0件であることを確認。`Θ`/要(`theta_over_need`)を
残りターン別に4モードで並べると、新モードは既存3モードのどれとも重ならない値を出す:

| 残りターン | `cuttable` | `cuttable_cx` | `cuttable_forced`(既定) | `cuttable_seq`(新・T158) |
|---|---|---|---|---|
| 1 | 1.292 | 1.142 | 1.231 | 1.261 |
| 2 | 1.066 | 0.967 | 1.056 | 1.017 |
| 3 | 0.947 | 0.893 | 0.946 | 0.876 |
| 6+ | 0.765 | 0.764 | 0.765 | 0.627 |

(実・w41・300局。合成・w39+w42・900局でも同様に1(1.442)/2(1.101)/3(0.929)/6+(0.655)でNaN無く
既存モードと明確に異なる。)

残り1ターン帯では新形は既定(`cuttable_forced`)とほぼ同水準(実1.261対1.231・合成側はやや高い
1.442)——事前予告していた「終盤の過大がさらに縮む」を必ずしも裏付けない。一方、残り3以上の
帯では既定より明確に小さい値(実0.876対0.946・6+で0.627対0.765)を示す。これは「壊れていない」
ことの確認に過ぎず本格測定ではないので、この傾向(終盤は横ばい・中盤以降はさらに縮む)が本当に
予告と整合するかは、`ΔG`のAUCや必要な`κ`も含めてH-2で確認する。

## 台帳への追記

`docs/cpu_theory_gap.md`に本作業のエントリを1段追記する(項目番号2・アポストロフィ123個)。

## 次(H-2/H-3への引き渡し)

- 4形(`cuttable`／`cuttable_cx`／`cuttable_forced`／`cuttable_seq`)を実(w41)・合成(w39+w42)で
  並べ、`Θ`/要・`ΔG`のAUC・必要な`κ`・交点の橋・帳簿を測り直す(H-2)。
- 既定の採否・報告・台帳・§0.6・切替一覧・ゲート・pushはH-3。
