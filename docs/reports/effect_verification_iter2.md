# 効果検証 修正報告 — イテレーション2

本書は「カード効果の正しさ検証」第2イテレーション（2026-06）の**修正報告**である（特定時点の
スナップショット）。イテレーション1（[`effect_verification_iter1.md`](effect_verification_iter1.md)）の
トリアージを受け、方針判断のうえ修正を実施した。検証ハーネスの仕様は
[`docs/TEST_SPEC.md`](../TEST_SPEC.md) §3.1。

実施対象（A〜E はドキュメント由来の課題分類）:

| 区分 | 対応 | 状態 |
|---|---|---|
| A. PER_TURN_LIMIT_GAP | iter1 で修正済み（置換/保護系【ターン1回】enforce） | 回帰確認のみ（残1=OP10-118 は表現負債） |
| B. EB01-001 | **手札ベースで実装** | 完了 |
| C-1/D-1. 「お互いの〜」同時両側 | **両側同時適用を実装** | 完了（選択を伴う両側は既定選択で非中断） |
| C-2. 置換 sub_effect ネスト対話 | 据え置き（SPEC §6.1 の既知制約として継続） | 対象外 |
| D-2. フォールバック負債 | ratchet 上限0で違反なし | 対象なし |
| E. UP_TO_GAP / 検出器誤検知 | **検出器の誤検知除去＋精査** | 完了（真の取りこぼしは無し＝下記） |

---

## 1. B. EB01-001（手札ベースのカウンター+1000）

「ルール上、自分の特徴《ワノ国》を持ちカウンターを持たないキャラカードすべてはカウンター+1000を
持つ」を、カウンターは手札から使うという実ルールに合わせ**手札ベース**で実装した。

- パーサ `counter_buff`（`atoms.py`）: 「カウンター+Nになる」に加え「カウンター+N**を持つ**」を一致。
  「カウンターを持たない」を `NO_COUNTER` フラグ化。→ PASSIVE `BUFF(COUNTER, zone=HAND, select_mode=ALL,
  traits=[ワノ国], NO_COUNTER, value=1000)`。
- `matcher`: `NO_COUNTER` フラグで基礎カウンター値を持つカードを候補から除外（二重加算しない）。
- `gamestate` BATTLE_COUNTER 候補列挙: `master.counter > 0` → **`current_counter > 0`** に変更。
  +1000 化した非所持《ワノ国》キャラがカウンター候補に出る。既存「手札の…はカウンター+Nになる」系も
  同時に正しく候補化される（副次的改善）。
- テスト: `test_eb01_001_passive_grants_counter_to_wano`（xfail 解除・通常テスト化。非所持→+1000、
  既所持→据置、非《ワノ国》→不変、候補列挙を検証）。

## 2. C-1/D-1.「お互いの〜」同時両側処理

`OP11-102`（お互いのライフ上1枚トラッシュ）/`OP05-058`（お互い手札5枚に調整）が `Player.ALL` の
単一対象選択で片側のみ解決されていた問題を解消した。

- `matcher`: 「お互い」を `Player.ALL` ＋ **`BOTH_SIDES`** フラグとして解析（側無指定 ALL と区別）。
- `resolver._resolve_targets`: `BOTH_SIDES` のとき各プレイヤーで候補・枚数を**個別に解決して結合**
  （隠しゾーン自動取得・`DOWN_TO_N`・`select_mode=ALL` を各サイド独立に評価）。
- `discard` ルール: 「手札がN枚になるように…捨てる」を `DOWN_TO_N` として解釈（従来は literal の
  N 枚と誤読。単独カード `OP14-054` も同時に修正）。
- **残る制約**: 選択を伴う両側効果は各サイド既定選択（候補先頭）で非中断確定する。人間が両側を個別に
  選ぶ完全な同時対話化は未実装（`active_interaction` 単一スロット）。→ `SPEC.md §6.1`。
- テスト: `test_op11_102_mutual_life_to_trash_both_sides` / `test_op05_058_mutual_discard_down_to_5_both_sides`。
- 挙動ベースライン差分（意図どおり・`full_card_baseline.json` 再生成済み）:
  - `OP11-102|YOUR_TURN`: 相手のみ → **両者**のライフ -1。
  - `OP05-058|ACTIVATE_MAIN`: 誤った全捨て → 手札 5 枚以下なら捨てない（DOWN_TO_N）。
  - `OP14-054|TURN_END`: 5枚を全捨て → **据置**（手札5枚なら捨てない）。

## 3. E. UP_TO_GAP 精査と検出器の誤検知除去

### 3.1 検出器の誤検知除去（`tests/effect_oracle.py` / `tests/text_execution_audit.py`）
- **UP_TO_GAP**: 「まで」判定の前処理で非カード選択の「まで」を除外（`ドン!!N枚まで`／`N枚まで…で追加`
  ／`…開始時/終了時まで`）。さらに任意性が **is_up_to（`sub_effect` 内も走査）／opt-out Choice
  （空 Sequence）／自動カウント（`HEAL`/`DRAW`）** で表現済みのものを除外。検出 **203 → 1**。
- **MISSING_ACTION**: 「公開」の許容アクションに `LOOK_LIFE` を追加（`text_execution_audit`）。
  OP10-022/ST13-007/ST13-010/ST13-014 の誤検知 **4 → 0**。effect_oracle 側の旧 `KNOWN_FALSE_POSITIVES`
  表示（根本解消により陳腐化）も削除。

### 3.2 精査結果（真の取りこぼし）
DON 等の誤検知を除いた残 ~53→（自動カウント/opt-out/入れ子 is_up_to を除外して）**1 枚**まで絞り込んだ。
精査の結論: **カード選択の `is_up_to` 取りこぼし（選択可能対象で 0 枚可が欠落）に該当する確定バグは無い**。
内訳は次の構造的事由で、いずれも取りこぼしではない:

- `HEAL`（デッキ上→ライフ「N枚まで加える」）・`DRAW`（「カードN枚まで引く」）: 対象選択を伴わない
  自動カウントで、engine は固定 N としてモデル化（隠しゾーン上から自動取得）。
- `life_scry_top`（「ライフ上1枚までを見て…」）: 任意性を「見ない」**opt-out Choice** で表現済み。
- `OP05-032`: `is_up_to=True` は付いているが置換の `sub_effect` 内にあり、旧検出器の走査対象外だった。

残 **1 枚＝EB03-031**（「トラッシュのコスト7以下のイベント1枚までの【メイン】効果を発動」）は、
`is_up_to` 単独の問題ではなく **`EXECUTE_MAIN_EFFECT` がトラッシュのイベントを選んでその効果を実行する**
機構が未実装（現状は発生源自身の【メイン】を再展開する）という別個・複雑なモデル化課題であり、
今回の `is_up_to` 修正の対象外として**要レビュー**に残す。

---

## 4. 検証（このイテレーションのゲート結果）

```bash
cd opcg-sim-backend
OPCG_LOG_SILENT=1 python -m pytest tests/ -q -s -p no:cacheprovider   # 769 passed
OPCG_LOG_SILENT=1 python tests/full_card_audit.py                     # EXCEPTION/CARD_LOSS/TEMP_LEAK = 0
OPCG_LOG_SILENT=1 python tests/text_execution_audit.py                # FLAG_* = 0（MISSING_ACTION 0）
OPCG_LOG_SILENT=1 python tests/effect_oracle.py                       # HAS_OTHER 0 / UP_TO_GAP 1 / PER_TURN_LIMIT_GAP 1
```

- 全テスト 769 passed（API テストは `fastapi` 導入後に collection 可）。
- 構造不変条件ゲート 0、パーサ・フォールバック 0、品質ゲート緑。
- 挙動ベースラインは §2 の 3 件のみ意図的に更新（`full_card_audit.py --regen`）。
