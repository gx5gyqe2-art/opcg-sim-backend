# リーダー効果 pytest 化ガイド

仕様書(`docs/leader_specs/<SET>.md`)のテストケースを pytest に落として
**挙動を固定**するためのルール。

## 基本方針
- **常にテキスト準拠の期待挙動をアサートする**（現実装に合わせない）。
- 現挙動との関係でマーカーを決める:
  - 期待挙動と現挙動が一致 → 通常テスト。
  - 現挙動が期待と異なる → `@pytest.mark.xfail(strict=True, reason="差異の内容")`。
    差異が解消されると xpass→strict で赤になるため、マーカーを外して通常テスト化できる。
  - 汎用盤面で安定検証できない／対話が複雑な場合 → まず通常テストで書き、不安定なら
    `@pytest.mark.xfail(strict=False, reason="要確認: ...")`（strict=False なので xpass でも緑）。

## 実行コマンド（**`-s` 必須**。付けないとログ干渉で I/O エラー）
```
OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_<set>.py -q -s -p no:cacheprovider
```
**合格条件**: 出力が `passed` / `xfailed` のみ。`failed` や `xpassed` を残さない
（xpassed が出たら、マーカーを外して通常テスト化するか、strict=False に変える）。

## ヘルパ API（`tests/leader_test_helpers.py`）
```python
from leader_test_helpers import (
    build, get_ability, abilities_of, auto_resolve,
    select_uuids, confirm, choose,
    add_char, make_char, clear_field, set_life,
    leader_power, don_total, zone_counts, leader_master,
)

gm, p1, p2, L = build("ST10-003")   # p1=ターンP, L=リーダーinstance
ab = get_ability(L.master, "ON_ATTACK", n=0)   # trigger種別のn番目
gm.resolve_ability(p1, ab, L)        # 能力発動（source=Lはリーダー）
auto_resolve(gm, p1)                  # 対話を賢い既定で駆動
assert leader_power(p1) == 7000
```
- 盤面: p1=ドン10/手札5/トラッシュ10/デッキ20/ライフ5/フィールド3(フィラー),
  p2=フィールド3/手札3/ドン5/デッキ20/ライフ5。必要に応じ各ゾーンを上書き。
- `add_char(p1, name=, cost=, power=, traits=[...], colors=[...], rest=False)` で
  特徴・コスト・パワー付きキャラを場に追加（戻り値は instance）。
- `clear_field(p1)` で場を空に、`set_life(p1, n)` でライフ枚数調整。
- レストドンを作る: `for _ in range(2): d=p1.don_active.pop(); d.is_rest=True; p1.don_rested.append(d)`

### 対話駆動
- `auto_resolve(gm, player, plan=None)`: plan省略時は
  CONFIRM系=受諾 / SELECT系=min枚(最低1)を先頭選択 / CHOICE=index0。
- 精密制御は plan に payload を順に渡す:
  `auto_resolve(gm, p1, plan=[confirm(True), select_uuids([victim.uuid]), choose(1)])`
- `gm.active_interaction` で現在の `action_type` / `candidates` / `constraints` を確認可。

### 観測
- `leader_power(p1)` / `don_total(p1)` / `zone_counts(p1)` → dict(hand/field/trash/deck/life/don_active/don_rested)
- キャラの `inst.get_power(True)` / `inst.current_cost` / `inst.is_rest` / `inst.attached_don` /
  `'速攻' in (inst.current_keywords|inst.timed_keywords)`

## 命名・構成
- ファイル: `tests/test_leader_<set>.py`（例 `test_leader_op01.py`, `test_leader_st01_10.py`）
- 関数: `test_<id小文字>_<能力概要>()`。1能力につき代表1〜数ケース。
  条件分岐がある効果は「条件成立／不成立」を別ケースにすると差異が明確になる。
- 各テストに docstring で「カードID / 能力 / 期待挙動」を1行。
- `@pytest.mark.xfail` の reason には期待と現挙動の差異を具体的に書く。

## 例（期待と現挙動が異なる場合の xfail）
```python
@pytest.mark.xfail(strict=True, reason="OP10-001: パワー条件『7000以上』が現挙動では power_max(以下) として扱われる")
def test_op10_001_active_don_requires_power_ge_7000():
    """OP10-001 起動メイン: 自パワー7000以上のキャラがいる場合のみドン2枚アクティブ。"""
    gm, p1, p2, L = build("OP10-001")
    clear_field(p1)
    add_char(p1, power=8000)
    for _ in range(2):
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == 10 and len(p1.don_rested) == 0   # 条件成立で2枚復帰
```
