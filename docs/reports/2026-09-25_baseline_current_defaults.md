# 基準線の取り直し(今の既定・P8記録・2026-09-25)

B(止められる本数)・C(攻撃の値段)・D(型ごとの値段・旧#43)・E(1枚が2つの時計)に着手する前に、
**今の既定**(`OPP_CLOCK_MODE=mirror`・`W_MOVER_MODE=half`・`SCHED_T1_MODE=game`・`THETA_BODY_MODE=blockers`・
`PLAY_BOOK_MODE=next`・`ATTACH_LEDGER_MODE=in_attack`・`D_MODE=curve`・`LETHAL_STOP_MODE=max`・
`LETHAL_HAND_MODE=actual`・`LETHAL_LIFE_MODE=draw`)と**P8の記録**(実300局・合成900局・dump v5)で、
型ごとの「値段 対 実現」の表(`price_realised.py`)と、値段の付いていない遷移の残差の内訳
(`transition_ledger.py`)を測り直した。読み取りだけ・新定数ゼロ・式は変えていない。

**訂正**: 前回のまとめで「`THETA_BODY_MODE`は`attackable`が既定」と書いたのは誤り。T83で`attackable`を
採用した後、同日T97で**`blockers`に戻している**(避けて通れるレストの体は耐久ではない、という規則の
読み直し)。現在の既定は`blockers`。

## 型ごとの価格対実現(`price_realised.py`・T41)

| 型 | 実 price | 実 real | 実 real/price | 合成 price | 合成 real | 合成 real/price |
|---|---|---|---|---|---|---|
| attack | 0.0646 | 0.0655 | **1.014** | 0.0718 | 0.0796 | **1.108** |
| attach | 0.0196 | 0.0014 | 0.072 | 0.0193 | 0.0001 | 0.006 |
| play | 0.0828 | 0.0710 | 0.858 | 0.0689 | 0.0753 | 1.092 |
| effect | 0.0546 | 0.0285 | 0.522 | 0.0003 | 0.0028 | **10.949** |

`dPrice`/`dReal`(局全体の勝敗との相関): 実 AUC 0.7714/0.8572・傾き 0.3787/0.4428。
合成 AUC 0.8164/0.8110・傾き 0.4164/0.5497。

**読み**: `attack`と`play`は実・合成とも0.8〜1.25倍の枠に入っており、今の既定では大きく外れていない。
`attach`のreal/priceが両記録で0に近いのは**既知の型**(T144/T155/T156・純付与は価格を「その付与が
準備する攻撃」から借りて読み替えないと単体では無意味に見える・`shadow_forbid`のseqモードの対象)。
`effect`は実0.522・合成10.949で**両記録の食い違いが最大**——ただし合成のprice_meanが0.00026とほぼ0なので、
比自体がゼロ割りに近い不安定な量である可能性がある(price側のp10/p50/p90が-0.08/0/0.07と符号が割れている)。
D-1で対象を絞るときは、この「比の不安定さ」を先に切り分ける必要がある。

## 値段の付いていない遷移の残差(`transition_ledger.py`・T123)

| | 実(300局) | 合成(900局) |
|---|---|---|
| `priced_share` | 0.3637 | 0.4254 |
| `resid_share` | 0.8891 | 0.8332 |
| 攻撃の行(`attack_rows`) | **0.3787** | **0.3838** |
| 攻撃でない行(`nonattack_rows`) | 0.300 | 0.2576 |
| ターンの境目(`turn_boundary`) | 0.3213 | 0.3586 |
| 軸: `th_me`/`th_opp`/`a_me`/`a_opp`/`j` | 0.485/0.495/0.0/0.0/0.020 | 0.470/0.498/0.0/0.0/0.032 |
| attack行のresid/priced比 | 0.926 | 0.758 |

**読み**: 攻撃の行の残差割合(37.9%/38.4%)は、T123の初回計測(37.8%/38.9%)・P8-7(c)の再計測時と
**ほぼ変わっていない**——直近の既定変更(mirror/half/game等)はこの3分割そのものには影響していない。
これがC-1診断の直前の基準値になる。

## 再現

```bash
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <実 n_records dump v5> --out real.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <合成 n_records ×2> --out syn.json
OPCG_LOG_SILENT=1 python tests/scripts/transition_ledger.py --in <実 n_records dump v5>
OPCG_LOG_SILENT=1 python tests/scripts/transition_ledger.py --in <合成 n_records ×2>
```

記録は`docs/reports/2026-09-25_p8_7a_body_removal.md`と同じ実300局(`--decks user`・seed_base 8800)・
合成900局(`--decks synth`・seed_base 9700/7000)を再利用した。
