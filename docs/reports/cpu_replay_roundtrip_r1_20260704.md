# 実対局リプレイ R1/R2: リプレイヤ＋ラウンドトリップ 実装結果

日付: 2026-07-04 / 計画: `docs/replay_verification_plan.md` R1/R2。**報告（点）＝以後改変しない**。
コード: `tests/harness/replay_runner.py`・`tests/test_replay_roundtrip.py`。

## 1. やったこと

記録記述子（seed＋leaders＋decks＋人間アクション列）から実対局を**再構築・再生**する再生側を実装:
- **デッキ復元** `build_deck_from_ids`: 記録の card_id 列から復元。`Player` は deck を JournaledList へ**コピー**して
  `setup_game` で shuffle するため、記録される `decks` は **pre-shuffle 順**＝`random.seed(seed)`＋`start_game` で
  同一シャッフルを再現できる（この順序性を実測で確認）。
- **人間手注入** `resolve_recorded_action`: R0 確定の (A) 決定論タイブレーク逆引き（記述子に一致する合法手の
  **列挙順先頭**）。CPU 手は `run_game` の席で**再 decide**（同一 seed から再計算＝一致が決定論の証明）。
- 逆写像不能（分岐）は crash させず `reproduced=False`/`misses` に記録＝**分岐検出**。

## 2. 結果（held-out 実デッキ・4-of 複製あり）

| | EXACT（勝敗+手数+ターン一致・miss=0） | 分岐 |
|---|---|---|
| `_find_card` 修正前・10 seed | 8/10 | 2/10 |
| `_find_card` 修正後・10 seed | **10/10** | 0/10 |

各局で人間手 30〜50 手を card_id 記述子から注入し、tie-break で復元して**完全一致**。

## 3. 判明した欠落と修正（R1 の副産物）

- 分岐 2 件はいずれも **`ACTIVATE_MAIN` の手記述が card_id でなく uuid** になっていた（`_describe_move`→`_card_label`→
  `_find_card` が uuid を解決できず raw uuid を返す）。原因: **`_find_card` が `stage` / `temp_zone` を探索していない**
  ＝それらのゾーンのカードは card_id に解決されず、card_id 基準の記録が**再現不能**になる。
- 修正: `cpu_ai._find_card` の探索ゾーンに `stage` と `temp_zone` を追加 → 8/10→**10/10**。
  trace テスト（`test_cpu_replay`/`test_cpu_learned` 16 passed）無退行・ruff clean。
- 含意: これは**記録側（`_describe_move`）の欠落**なので、API の実対局記録にも同じ穴があった＝本修正で実対局の
  card_id 記録の再現性も改善する。

## 4. R0 予測との対比（重要）

R0（§5）は「場複製（ATTACK/ATTACH_DON/SELECT）でタイブレークが誤個体を選び分岐しうる」を残差リスクとした。
**R1 実測では場複製由来の分岐は 0**（10局・人間手 300〜500 手）。曖昧だった場複製手も、複製個体が実質同一
（同状態）か、tie-break の先頭選択が録画と一致したため分岐しなかった。→ **(A) タイブレークは R0 見積りより頑健**。
(B)-lite（記録の個体弁別子追加）は現時点で**不要**（将来 net 更新等で分岐が出れば round-trip が検出する）。

## 5. スコープと残り（R3）

- 本 R1/R2 は **hard**＋**合成録画**（`record_descriptor`＝人間を private rng で代替・global random 非消費）で検証。
- 残り（R3）: **learned** のラウンドトリップ／**API 記述子の実結線**（`REPLAY_SCHEMA` 直食い・first_player の
  coin toss 再現）／API 記録テスト `test_replay_capture_and_fetch` の learned 追加。
- 調査用途（計画 §2.0・read_ahead 込み再取得／崩れ局面のパズル化）は再生が一致する土台の上に R3 以降で載せる。

## 6. 付随（別途フォロー・R0 から継続）

- random 方策×実デッキで稀に `apply_counter` の DON 不足 `ACTION_EXCEPTION`（合法手生成が支払い可能性を絞れて
  いるか）。R1 の hard 録画では未発生。エンジン側の要確認事項として継続。
