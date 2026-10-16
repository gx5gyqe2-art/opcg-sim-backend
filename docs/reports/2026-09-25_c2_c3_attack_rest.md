# C-2/C-3: 攻撃した体のレスト費用をΘ_meに足す——残差は縮むが「決着前」の帳簿は悪化(2026-09-25)

C-1(`2026-09-25_c1_attack_axis_by_result.md`)は、攻撃の型の残差がほぼ`th_me`(自分の耐久)と
`th_opp`(相手の耐久)の2軸だけに乗り、`th_me`は"nothing"(何も変わらない結果)でも負のままであることから、
**候補(a)(攻撃した体がレストになり、規則上`Θ_me`の体の項〔`THETA_BODY_MODE=blockers`〕から次の自席
ターンまで抜ける)** を最有力とした。C-2でこれを`ATTACK_REST_MODE`(既定`off`・`body`)として実装し、
C-3でその効果を実測した。

## 式(C-2・新定数ゼロ)

`kappa_vector.axis_of_move`の`attack`枝に追記(既定`off`では何もしない):

```python
if fam == "attack":
    out["th_opp"] = -v
    if ATTACK_REST_MODE == "body":
        info = cards.info(cid) if (cards is not None and cid) else None
        if info and info.get("blocker") and not info.get("event"):
            nu = float(CB.nu_meas_of(float(info.get("power") or 0.0), olp))
            if nu > 0.0:
                out["th_me"] = -nu
```

値は`_body_term`(`crossing_bridge.py`)が体の項を数えるのと**同じ単位**(`nu_meas_of`)。攻撃した体が
ブロッカーでない・イベントなら`th_me`は付かない(元々`Θ_me`の体の項に入っていないので削るものが無い)。
`--attack-rest {off,body}`を`kappa_vector.py`・`transition_ledger.py`・`relative_ledger.py`・
`attack_axis_by_result.py`のCLIに配線した。テストは`tests/test_kappa_vector.py`(6本追加・計69本)・
`tests/test_attack_axis_by_result.py`(2本追加・計6本)。

## 計測(C-3・実局300局／合成局900局)

### 1. 攻撃行残差の割合(`attack_axis_by_result.py`・T123の残差のうち攻撃行が占める分)

| 読み | 記録 | off | body | 差 |
|---|---|---|---|---|
| `curve`(帳簿の正本) | 実 | 0.3787 | 0.3661 | -0.0126(-3.3%) |
| `curve` | 合成 | 0.3838 | 0.3721 | -0.0117(-3.0%) |
| `theory`(速さの軸あり) | 実 | 0.1832 | 0.1769 | -0.0063(-3.4%) |
| `theory` | 合成 | 0.1623 | 0.1575 | -0.0030(-3.0% 相対) |

**4条件全てで縮む**(相対3%程度・小さいが一貫)。

### 2. 結果別の`th_me`/`th_opp`(`curve`読み・符号つき平均)

| 結果 | 記録 | th_me off | th_me body | th_opp off | th_opp body |
|---|---|---|---|---|---|
| countered | 実 | -0.0108 | -0.0075 | +0.0136 | +0.0077 |
| took | 実 | -0.0247 | -0.0222 | +0.0214 | +0.0176 |
| nothing | 実 | -0.0080 | -0.0070 | +0.0139 | +0.0124 |
| took_and_countered | 実 | -0.0798 | -0.0801 | +0.0838 | +0.0820 |
| **blocker_died** | 実 | **+0.0421** | **+0.0457** | **-0.0346** | **-0.0396** |
| countered | 合成 | -0.0035 | -0.0014 | +0.0070 | +0.0032 |
| took | 合成 | -0.0244 | -0.0216 | +0.0258 | +0.0220 |
| nothing | 合成 | -0.0124 | -0.0110 | +0.0127 | +0.0104 |
| took_and_countered | 合成 | -0.0773 | -0.0740 | +0.0794 | +0.0765 |
| **blocker_died** | 合成 | **+0.0362** | **+0.0394** | **-0.0392** | **-0.0430** |

`countered`・`took`・`nothing`・`took_and_countered`は両記録で`th_me`・`th_opp`とも**0へ縮む**
(候補(a)の想定どおり)。**`blocker_died`だけ逆に膨らむ**(両記録で`th_me`・`th_opp`とも絶対値が増える)
——攻撃した体がブロッカーを倒す行では、レスト費用を足すとかえって残差が増える。機構は未確認
(相手のブロッカーを倒せる体はパワーが高くブロッカー率も高い可能性があるが未検算)。`theory`読みでも
同じ向き(実+0.0253→+0.0271・合成+0.0252→+0.0273)。

### 3. 帳簿(`relative_ledger.py`・線形の橋)のAUC

| 帯 | 腕 | 実 off | 実 body | 実差 | 合成 off | 合成 body | 合成差 |
|---|---|---|---|---|---|---|---|
| 全帯(`arms`) | `rel_K` | 0.6420 | 0.6694 | +0.0274 | 0.7023 | 0.7259 | +0.0236 |
| 全帯 | `rel_flat` | 0.8680 | 0.8732 | +0.0052 | 0.8963 | 0.8811 | -0.0152 |
| **主に読む帯(`before`)** | `rel_K` | 0.7757 | 0.7702 | **-0.0055** | 0.8358 | 0.8306 | **-0.0052** |
| 主に読む帯 | `rel_flat` | 0.9393 | 0.9333 | **-0.0060** | 0.9476 | 0.9319 | **-0.0157** |
| とどめの帯(`last_turn`) | `rel_K` | 0.3435 | 0.4010 | +0.0575 | 0.4365 | 0.4943 | +0.0578 |
| とどめの帯 | `rel_flat` | 0.6255 | 0.6446 | +0.0191 | 0.6465 | 0.6596 | +0.0131 |
| (対照)全帯 | `abs_kappa` | 0.5908 | 0.5908 | 0.0000 | 0.7342 | 0.7342 | 0.0000 |

`abs_kappa`(`axis_of_move`のdx配分を使わない旧来のスカラー腕)は**厳密に不変**——切替が意図した箇所
だけを触っていることの検算になる。

**「とどめの帯」は両記録・両腕で明確に改善**(`rel_K` +0.057〜+0.058)——決着の瞬間に近い行では
レスト費用が効いている。**「全帯」も`rel_K`は両記録で改善**するが`rel_flat`は合成で悪化(-0.0152)。
**台帳が「主に読む帯」と呼ぶ`before`(最後の自席ターンを外した帯)は両記録・両腕で悪化**
(-0.0052〜-0.0157)——決着直前でない行では、レスト費用を足すとかえって順序づけが崩れる。

### 4. 両橋のもう一方(交点の橋)と曲線

`crossing_bridge.py`(交点の橋)・`theory_bridge.py`・`two_curves.py`はいずれも`kappa_vector`を
importしておらず、`axis_of_move`も呼んでいない(`grep`で確認済み)——**`ATTACK_REST_MODE`はこれらの
出力にコード上不可避的に影響しない**(実測ではなく静的な保証)。実測が要るのは`axis_of_move`を呼ぶ
`relative_ledger.py`(上記)と`transition_ledger.py`(上記1・2)の2本のみ。

## 読み

**攻撃行残差の割合・とどめの帯の帳簿は一貫して改善**する一方、**台帳が主に読むと定めた`before`帯の
帳簿は一貫して悪化**する。`blocker_died`の結果だけ残差が逆に膨らむことも未解明。**「新定数ゼロ」の
式自体は規則どおり**(攻撃した体はレストになり、次の自席ターンまでブロッカーとして数えられない)だが、
**測った効果は場所によって符号が割れている**——単純な「採用/不採用」では済まない。

## 再現

```bash
OPCG_LOG_SILENT=1 python tests/scripts/attack_axis_by_result.py --in <n_records>... [--d-mode curve|theory] --attack-rest {off,body}
OPCG_LOG_SILENT=1 python tests/scripts/relative_ledger.py --in <n_records>... --attack-rest {off,body}
```
