# D-2: playの"?"バケットの符号反転——ドン!!コストの支払い可否を値付けに足す(2026-09-25)

D-1が挙げた「playの"?"バケットの符号反転」の原因（`effect_value.py::_cost_unpayable`が
`RETURN_DON`型のコストを一度も判定していない）を直した。新定数ゼロ・切替は既定offで追加のみ・
`price_realised.py`でこの型のバケットの符号が実測どおり直ることを確認した。

## 原因（D-1からの申し送りの確定）

D-1の「未確定点」——OP15-074〜078（【メイン】ドン‼️-N の効果を持つコスト0イベント）が
記録上どこかで実際に値を実現している形跡があるか——は**確認不要だった**。コードを直接読んで
確定した:

* `_cost_unpayable(cost_acts, card, st)`（`effect_value.py`）は`e.get("target")`が
  在るコスト（戻す・捨てる・KO＝`FIELD`/`HAND`ゾーン）しか見ていない。
* `RETURN_DON`（ドン‼️を払うコスト）は**`target=None`**（`walk_actions`で確認）なので、
  `if not t or _side(t) != "SELF": continue`で**毎回スキップ**され、関数はループを素通りして
  `False`（払える）を返す。
* 一方`action_value()`はコストの**金額**自体は正しく`don_loss`として値付けしている
  （`RETURN_DON`→`kind="don_loss"`→`amt = cnt * delta`）。**欠けていたのは金額ではなく
  「払えるか」の判定だけ**。
* 必要なデータ（アクティブなドンの実数）はすでに存在していた——`condition_value.state_from_scalars`
  （`theory_bridge._state_of`が呼ぶ・T72）が`st["my_don_active"]`を積んでおり、`_don_attach_cost`
  （T74・【ドン!!×N】の判定）がすでに同じ場所を読んでいる。**新しい状態の配線は要らなかった**。

## 式（新定数ゼロ・既存パターンの流用）

`effect_value.py`に`DON_COST_GATE_MODE`（`off`/`check`・既定`off`）を追加。`check`のとき
`_cost_unpayable`は`DON_LOSS`型（`RETURN_DON`）のコストを`st["my_don_active"]`と比べ、
足りなければ`True`（払えない）を返す——場・手札の判定（既存のFIELD/HANDゾーンの分岐）とは
独立に効く別ブロックとして足した。`status`が無い行・`my_don_active`が無い行は従来どおり
「払える」に落ちる（上限）。`price_realised.py`のCLIに`--don-cost-gate`を配線した。

```python
if DON_COST_GATE_MODE == "check" and st:
    have = st.get("my_don_active")
    if have is not None:
        for e in cost_acts:
            if str(e.get("type") or "") in DON_LOSS:
                need = _magnitude(e)
                if need > float(have) + 1e-9:
                    return True
```

`ability_value`の既存のゲート（`if cost_acts and not offered and _cost_unpayable(...): return 0.0, []`）
がそのまま使う——起動メインの`offered=True`候補は従来どおりゲートを通らない（エンジンが
`has_activatable_main`で既に確かめてから候補に出しているため）。

## 実装時に踏んだ配線漏れ

最初の測定（`--don-cost-gate check`）は`off`と1バイトも変わらない出力を返した。
`price_realised.py`に`EV.add_don_cost_gate_arg(ap)`は足したが`EV.apply_don_cost_gate(a)`の
呼び出しを足し忘れていた（CLI引数が受理されるだけでモードに反映されない）ため。直接
`effect_value._cost_unpayable`をレコードの`st`（`theory_bridge._state_of`経由）と実際の
OP15-074〜078の選ばれた手で呼ぶ検算（886回中614回=69%がアクティブなドン不足で「払えない」
と正しく判定）で仕組み自体は健全と確認してから配線漏れを特定・修正した。

## 計測（`price_realised.py`・実300局・合成900局・w41／w39+w42）

| | 実 off | 実 check | 合成 off | 合成 check |
|---|---|---|---|---|
| play "?" `price_mean` | +0.01076 | **-0.00387** | +0.02698 | +0.02688 |
| play "?" `real_mean` | -0.01005 | -0.01005（不変） | +0.03848 | +0.03848（不変） |
| play "?" `ratio_real_over_price` | **-0.934** | **+2.597** | +1.426 | +1.431 |
| play family `n` | 4013 | 4013（不変） | 8623 | 8623（不変） |
| play family `corr` | 0.8112 | 0.8274 | 0.6983 | 0.6987 |
| `by_family_games` play `auc_price` | 0.6635 | 0.6873 | 0.582 | 0.5818 |

**符号は直った**——実測の"?"バケットの`price_mean`が正(+0.01076)から負(-0.00387)へ動き、
`real_mean`(-0.01005)と同じ符号になった（`ratio_real_over_price`が-0.934→+2.597で符号が揃う・
合成の+1.426/+1.431と符号が一致）。`corr`・`auc_price`も実測でわずかに改善。

**effect/attack/attach族は実・合成とも1バイトも変わらない**——ゲートは`play`族の登場時価格
（`offered=False`の経路）だけを通り、`ACTIVATE_MAIN`候補（`offered=True`）はゲートを通らない
設計どおりに、他の族には触れていないことを確認した（`by_family`の該当3族の値を突き合わせ・完全一致）。

**未解決の残差**: `ratio_real_over_price`が+2.597で1から離れている——ゲートは「払えなければ0」
にするだけで、払えるときの価格・実現の水準自体は直していない（"?"バケットには他の無効果カードも
混ざる）。この残差はD-3（線形の橋の傾き・符号）で改めて読む。

## 再現

```bash
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <実 n_records dump v5> --don-cost-gate off   --out real_off.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <実 n_records dump v5> --don-cost-gate check --out real_check.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <合成 n_records ×2> --don-cost-gate off   --out syn_off.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <合成 n_records ×2> --don-cost-gate check --out syn_check.json
```

記録は`docs/reports/2026-09-25_d1_price_mismatch_diagnosis.md`と同じ実300局(`--decks user`・
seed_base 8800)・合成900局(`--decks synth`・seed_base 9700/7000)を再利用した。

## D-3への申し送り

* `DON_COST_GATE_MODE`はまだ既定`off`（採否はD-4）。
* D-3は`--don-cost-gate check`で線形の橋（`theory_bridge.py`のΔG恒等式）の傾き・符号が
  実・合成の両方で一致するかを測る（#43「Rewrite the family prices...re-run dG」に対応する
  playファミリー分の再測定）。
