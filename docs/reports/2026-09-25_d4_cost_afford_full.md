# D-4: コスト支払い判定を全種に広げる（2026-09-25）

D-2/D-3は`RETURN_DON`型（ドン‼️を払うコスト）だけを払える判定に入れた。「ドン以外のコストは
正しくなってるの？」という問いを受けてカードDBの全コスト型を棚卸しし、`RETURN_DON`以外に
5種の取りこぼしがあることを確認——それらを含めて`_cost_unpayable`を直し、実300局・合成900局
（D-2/D-3と同じ記録）で測り直した。新定数ゼロ・`off`は1バイトも変えない（フューズ8万回超＋
手読みで検算済み）。

## 直した取りこぼし（`RETURN_DON`以外）

1. **`REST_DON`**（ドン‼️をレストにするコスト・`target=None`）——D-2は`RETURN_DON`だけ見ており
   同じ穴が残っていた。
2. **自分自身が対象**（`ref_id=="self"`・「このカードをレストにする」等）——場の一覧の中から
   自分を探していたが、判定文脈（`hand_plan.py`の計画価格評価など）では対象の札がまだ実際の
   場に無いことがあり、常に「払えない」と誤判定していた。
3. **リーダーが対象**（`card_type`に`"LEADER"`）——場の絞り込み関数がリーダー型を最初から除く
   ため常に「払えない」と誤判定していた。
4. **要る枚数**（`target.count`）——見ておらず、1枚合えば何枚要る効果でも「払える」扱いだった。
5. **ライフ／トラッシュゾーンのコスト**——ゾーンとして判定対象にすら入っていなかった。

`COST_AFFORD_MODE`（旧`DON_COST_GATE_MODE`を改称・`off`/`check`・既定`off`）にまとめ、
CLIは`--cost-afford`（旧`--don-cost-gate`）に統一した。

### 実装レビューで見つかった2件の実害（その場で直した）

* **リーダー対象の絞り込み無視**: 3.の直しだけだと`card_type`に`"LEADER"`が在れば特徴・名前の
  絞り込みを無視して常に払える扱いになる誤りが残っていた（「特徴《ドレスローザ》のリーダー」等
  15枚）——`st["my_leader"]`と`_matches_identity`で実際に絞り込みに合うか見るよう直した。
* **`ATTACH_DON`を自分のアクティブなドン起点にするコスト**（5枚・raw_textに「アクティブ」と
  明記）——ドンの在庫チェック自体が`RETURN_DON`／`REST_DON`しか見ておらず、この型は対象を持つ
  ぶん3.までのチェックには掛かるが在庫チェックからは漏れていた。同じ在庫チェックに含めた。

いずれも独立ワークフロー（3観点レビュー×検証4体）で見つかり、実カード（OP10-043・EB04-009）で
再現・直したことを確認した。テストは`tests/test_search_price.py`に追加（計22本）。

## 計測（`price_realised.py`・`theory_bridge.py`・実300局・合成900局・w41／w39+w42）

### 型ごとの値段（実現との比・`price_realised.py`）

| | 実 off | 実 check | 合成 off | 合成 check |
|---|---|---|---|---|
| play `price_mean`／`real_mean`／比 | 0.08887／0.07776／0.875 | **0.08301**／0.07776／**0.937** | 0.07034／0.07641／1.086 | 0.07036／0.0764／1.086 |
| effect `price_mean`／`real_mean`／比 | 0.05348／0.02481／0.464 | 0.05348／0.02481／0.464（不変） | 0.00044／0.00273／6.248 | 0.00054／0.00277／5.081 |
| attack `price_mean`／`real_mean`／比 | 0.06684／0.06783／1.015 | 0.06684／0.06783／1.015（不変） | 0.07245／0.08036／1.109 | 0.07245／0.08037／1.109 |
| attach `price_mean`／`real_mean`／比 | 0.01866／0.0013／0.07 | 0.01866／0.0013／0.07（不変） | 0.01903／0.00013／0.007 | 0.01903／0.00013／0.007 |

### 線形の橋（`theory_bridge.py`）

| | 実 off | 実 check | 合成 off | 合成 check |
|---|---|---|---|---|
| `ΔG` AUC／傾き | 0.6528／0.3631 | 0.6528／0.3631（不変） | 0.7004／0.3651 | 0.7004／0.3651（不変） |
| `ΔS` AUC／傾き | 0.6014／0.1229 | **0.6147**／**0.1387** | 0.4678／-0.0137 | 0.4687／-0.0131 |
| `ΔS`（攻め手の行） AUC／傾き | 0.6253／0.1516 | **0.6368**／**0.1646** | 0.4715／-0.0122 | 0.4722／-0.0115 |

## 読み

1. **`ΔG`（線形の橋の本体）は今回も両記録ともビット単位で不変**——D-3と同じ理由
   （`LEDGER_HARM_MODE=realised`・T87で登場の値段が`ΔG`に入らない）。型の広げ方に関わらず
   D系の直しは`ΔG`に届かない（#43は今回も閉じない）。
2. **`ΔS`（実）はD-2/D-3の狭い直し（0.622／0.1495）よりわずかに低い（0.6147／0.1387）**——
   ただし`off`（0.6014／0.1229）よりは改善している。原因を追った:
   D-2/D-3期のコードで既に「自分自身が対象」の場チェック（2.の穴）は`DON_COST_GATE_MODE`に
   関わらず**無条件で動いていた**（`git show c32c90be`で確認）——`hand_plan.py`の計画価格評価
   （手札に入れた札の価値を評価する経路・対象の札がまだ実際の場に無い）では、この穴により
   自分自身コスト付きの能力が**`off`でもD-2/D-3の`check`でも同じく誤って0円**になっていた。
   D-4はこの穴を`check`だけで直した（2.）ので、この種の能力は`check`で正しく非0に戻る——
   これは3.〜6.（リーダー・枚数・ライフ／トラッシュ・ATTACH_DONドン起点）が新たに0円へ落とす
   効果と**逆向き**に働き、一部相殺する。D-2/D-3の`ΔS`改善は、本来無関係な2.の穴が
   たまたま`ΔS`にプラスに効いていた分を含んでいたことになる。両方向とも個々には正しい直しで、
   `off`比では依然改善（実+0.0133／+0.0158の傾き）。
3. **合成の`by_family`は`play`・`effect`でわずかに動く（14/1800行・0.78%）——実は0行**。
   機構を直接確認した: `hand_plan.py`/`hand_spend.py`の計画価格（T68の探索効果の期待値・T69の
   手札品質補正で使う「その札を今持っていたらいくらか」の評価）は常に`offered=False`で
   `effect_value.card_value`を呼ぶ——これは「起動メインが候補に出た時点で条件確認済み」と
   いう`offered=True`のバイパス（`ACTIVATE_MAIN`本体の値付け）とは**別の経路**で、意図どおり
   D-4の新しい判定が正しく届く。具体例で検算: `EB04-009`（アクティブなドンを払うイベント）を
   `my_don_active=0`・場にRayleigh型の対象なしの状態で`hand_spend.free_value`に通すと
   `off`＝0.0487・`check`＝0.0（払えないので0）——同じ関数が`_search_plan`（effect族の
   探索効果価格）と`hand_quality_delta`（`real`側の手札品質補正・どの族の窓でも掛かりうる）
   の両方から呼ばれるため、`attack`・`effect`族の`price`・`real`どちらにも波及しうる。
   実測の300局にこの経路で影響を受ける具体的な組み合わせ（該当コスト型を持つ札＋その札が
   計画評価の対象になる窓）がたまたま無かっただけで、合成900局（デッキ分布が広い）では
   14行で起きた——**バグではなく直しが意図どおり届いた結果**。絶対値はごく小さい
   （`effect`族`price_mean`は+0.0001前後・`play`族は+0.00002）。

## D系のまとめ（採否はユーザ判断）

* 既定はまだ`off`（`COST_AFFORD_MODE`）。採否の材料:
  - 規則どおりの形になった（払えないコストは払えないとして値付けに反映）。
  - `ΔG`・攻撃／付与族・実測の効果族は不変。合成のplay/effect族の変化は上記の機構どおりで
    小さい（サブパーセントの行数・4〜5桁目の平均シフト）。
  - `ΔS`（実）は`off`比で改善するが、D-2/D-3の狭い直しより改善幅が小さい（2.の穴の相殺分）。
  - `ΔS`（合成）はほぼ不動（4桁目まで）——D-2/D-3と同じ理由（該当デッキが合成にほぼ無い）。
* #43（型の値段を直して`ΔG`の傾き→1）はD-3同様、今の帳簿（`realised`）の下では
  playファミリー分の直しが`ΔG`に届かないため、この道では閉じない。

## 再現

```bash
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <実 n_records dump v5> --cost-afford off   --out real_off.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <実 n_records dump v5> --cost-afford check --out real_check.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <合成 n_records ×2> --cost-afford off   --out syn_off.json
OPCG_LOG_SILENT=1 python tests/scripts/price_realised.py --in <合成 n_records ×2> --cost-afford check --out syn_check.json
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py  --in <実 n_records dump v5> --cost-afford off   --out tb_real_off.json
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py  --in <実 n_records dump v5> --cost-afford check --out tb_real_check.json
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py  --in <合成 n_records ×2> --cost-afford off   --out tb_syn_off.json
OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py  --in <合成 n_records ×2> --cost-afford check --out tb_syn_check.json
```

記録はD-1〜D-3と同じ実300局（`--decks user`・seed_base 8800）・合成900局（`--decks synth`・
seed_base 9700/7000）。`off`側の数値はD-2/D-3の測定を再利用（`off`モードは今回の変更で
1バイトも変わらない不変量として、フューズ80,000回超＋手読みで検算済み）。
