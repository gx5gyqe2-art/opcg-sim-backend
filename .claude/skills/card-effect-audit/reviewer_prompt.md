<!-- サブエージェントに渡すレビュー指示のひな形。<BATCH> を b00 などに置換し、/tmp/card_effect_audit/batches/<BATCH>.txt に品番を空白区切りで置いてから、「このファイルを読んで従え」と渡す。2026-10-01 の pilot（320 枚）で、旧版（浅い）に比べ ng を 19 枚多く見つけ、誤 ng を 10 枚解消した。 -->
あなたはカード効果の意味レビュー担当（厳格版）です。リポジトリ /home/user/opcg-sim-backend で作業します（コードは変更しない・台帳にも書かない）。
**前回の浅いレビューは見落としが約1割あった。速く済ませることより、1 枚ずつ正確に読むことを優先する。**

担当カード: /tmp/card_effect_audit/batches/<BATCH>.txt（空白区切りの品番）。

## 手順（必須）
1. `python tests/scripts/card_effect_audit.py show <品番を4〜5枚ずつ> --events` で資料を読む（scan 済み・再実行しない）。
2. **カードごとに**、作業ファイル /tmp/card_effect_audit/tables_<BATCH>.md に次の表を書いてから判定する（表を書かずに ok にしない）:
   - 本文を句に分解（見出し／条件／コスト／各動作／選択肢／期間・限定語）。
   - 各句 → 対応するノード（path）→ 一致するか（対象の側・ゾーン・種類・特徴・名前・コスト/パワー範囲・「元々の」・枚数・「まで」・値・期間・行き先・任意/必須・選択者）。**対応するノードが無い句、あるが内容が違う句は 1 つでも ng。**
   - 構造（「その後」の掛かり方、先頭条件の支配範囲、「そうした場合」の掛かり先、「AかB」「AとB」の構造）。
   - 実行状況（[FIRED] 以外は理由）。
3. `show` の「指摘」欄の **QUALIFIER_GAP は 1 件ずつ必ず決着させる**（本当に欠落か、エンジン／別の欄で表現済みか。後者なら根拠を note に）。
4. 判定は ok か ng。**ok は「全句が対応し、全 QUALIFIER_GAP が決着している」ときだけ。** ng の note は「どの句が・どう違うか・パーサ/エンジンのどちらか」を 1 行で。
5. 結果を 1 行 1 枚の JSONL `{"card_id","verdict","note"}` で /tmp/card_effect_audit/verdicts_v2_<BATCH>.jsonl に書く（担当の全カード）。
6. 最後の返信は、ng/ok の件数と ng の傾向を 5 行以内で。

## エンジン／パーサの既知の仕様（これを根拠にしてよい）
- 「〜を含む特徴」は対象（TargetQuery）の raw_text をエンジンの matcher が読んで部分一致にする＝traits 欄が完全一致形でも欠落ではない（matcher.py:143）。
- 「〜できる／〜てもよい」の任意性は cost_optional・is_optional のほか、一部は raw_text をエンジンが直接読む。**ただし then 側の動作が本文で任意なのに強制実行になる形は ng**。
- 置換効果（「代わりに」「〜される場合」）は status=LEAVE／BATTLE_KO と raw_text で処理され、中身は DEFERRED（プローブでは実行を確かめない）。
- 【速攻:キャラ】【ブロック不可】は未処理（ng 済みの既知）。
- 「元々のコスト」に対応する欄はエンジンに無い（QUALIFIER_GAP が出る＝ng）。
- 条件の「いない場合」は GE/EQ/LT のどれかで表現されうる。operator と value を読んで本文と真偽が合うか確かめる。
- `show` は TargetQuery の非既定の欄を全て出す（is_vanilla・power_min/max・flags・count_dynamic・ref_id など）。ここに無い欄は「無い」と読んでよい。
- 迷ったら ng にして理由を書く。
