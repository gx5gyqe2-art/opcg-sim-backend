# リファクタリング詳細設計: apply_action_to_engine のディスパッチ化と GameManager の責務分割

- 対象: `opcg_sim/src/core/gamestate.py`（2,893行 / GameManager 2,715行・約90メソッド）
- 目的: 挙動を **一切変えずに** 構造だけを改善する（pure refactoring）。
- ステータス: **Phase A（ディスパッチ化）実装済み**（A-1/A-2/A-3・PR #157）。Phase B（GameManager 分割）は未着手。
- 関連: `docs/SPEC.md` §2.5（クローン/シミュレーション）、`docs/TEST_SPEC.md` §4–5（品質ゲート）

> 実装時の設計差分（Phase A）: `FIELD_LIMIT` は `rules_constants.py` へ移さず `gamestate.py` に残置した。
> actions パッケージからは参照されず（循環回避の必要がない）、gamestate 内部で多用され外部
> （`invariants.py`／テスト）からも `gamestate` 経由で import されるため。`SELF_RESTRICTION_KEYS`
> のみ `rules_constants.py` へ集約した（actions が参照＝循環回避が必要なため）。

---

## 0. 現状の問題（要約）

1. `apply_action_to_engine`（L2281–2840, **561行**）が、`act_name = action.type.name` で
   enum を文字列に戻し、**45箇所の文字列比較**で全アクション型を単一関数内で分岐している。
   - 型安全性の放棄（タイポが静的検査で捕まらない）。
   - アクション追加のたびに巨大関数が肥大化。
   - CPU 探索（make/unmake シミュレーション）のホットパスで最悪45回の文字列比較。
2. GameManager がターン進行・戦闘・トリガー・対話（中断）・保護/置換・受動効果・
   カード移動・動的値計算の全責務を1クラスに抱えている。変更影響範囲が読めず、
   並行開発とレビューのボトルネック。

## 1. 設計原則（全フェーズ共通の不変条件）

本エンジンには2つの強い技術的制約があり、設計全体を規定する。

### 1-1. journal（make/unmake）互換

- `GameManager.__setattr__` / `Player.__setattr__` は `journal.record_attr` で属性変更を記録する
  （gamestate.py L180–185）。コンテナは `JournaledList/Dict/Set`。
- **したがって「可変状態は今後も GameManager / Player / CardInstance 上にのみ置く」。**
  分割先のモジュール・ハンドラは一切の可変状態を持たない（モジュールレベル可変変数の禁止、
  インスタンス属性への状態保持の禁止）。ハンドラ表（レジストリ）は import 時に確定する
  **不変** の dict のみ許可。
- これを守る限り、ロジックをどこへ移しても journal の記録経路は変わらず、make/unmake の
  正しさに影響しない。

### 1-2. clone（deepcopy）互換

- `GameManager.clone()` は `copy.deepcopy(self)`（L308–318）。
- 分割方式が「gm への後方参照を持つサービスオブジェクト」だと deepcopy 対象が増える。
  → **分割はステートレスなモジュール関数（第1引数 gm）で行い、GameManager 側に
  1行デリゲートを残す**方式を採る。インスタンスを増やさないため clone への影響ゼロ。

### 1-3. 挙動不変の証明方法

- `full_card_baseline.json` の**再生成（--regen）を禁止**する。ベースライン一致＝挙動不変の証明。
- 各PRで CLAUDE.md の品質ゲート（全テスト / 構造監査 = 0）を green にする。
- 実行経路（apply_action_to_engine / move_card 等）は journal transaction 内で走るため、
  **各PRで `-m slow`（journal roundtrip ~245s）も手動実行**する。
- 性能ゲート: `tests/bench_decide.py` / `tests/_profile_decide.py` で decide 時間が
  リファクタ前比 ±5% 以内であること（ホットパスのため）。

---

## 2. Phase A: apply_action_to_engine のディスパッチテーブル化

### 2-1. 現構造の分析（保存すべき事実）

現関数は構造的に **2セクション** から成る:

**(a) プレイヤーレベル・アクション**（L2287–2574）: 対象ループ前に `if act_name == X: ...; return`
で完結する22分岐。対象を取らない/枚数ベースの処理。

| act_name | 備考（保存すべき挙動） |
|---|---|
| RULE_PROCESSING | **status ∈ SELF_RESTRICTION_KEYS の場合のみ**ここで処理（制限を player.restrictions に記録）。それ以外は対象ループ側の no-op へ落ちる |
| DRAW | CANNOT_DRAW_BY_EFFECT 制限で抑止（True を返す）。target.player=OPPONENT で相手ドロー |
| DEAL_DAMAGE / DAMAGE | enum エイリアス（`ActionType.DAMAGE is DEAL_DAMAGE`、`.name` は "DEAL_DAMAGE"）。ライフ→手札、【トリガー】enqueue、_enqueue_life_decrease、_advance_pending_triggers、デッキ切れで winner 設定 |
| SHUFFLE / LOOK / LOOK_LIFE | LOOK(status=OPPONENT) は盤面不変で True。LOOK_LIFE は `card._temp_origin="LIFE"` を付けて temp へ |
| MOVE_ATTACHED_DON | **唯一 False を返しうる**（`moved >= n` をコスト成否として返す） |
| REDIRECT_ATTACK | active_battle の target/target_owner を差し替え |
| DISABLE_ABILITY | **status=="OPP_ONPLAY" の場合のみ**ここ（negate_onplay_until 設定）。他は対象ループへ |
| EXTRA_TURN / VICTORY / ORDER_LIFE / EXECUTE_EVENT / SELECT | VICTORY は status=="REPLACE_DECKOUT_LOSS" なら無視して True |
| HEAL / LIFE_RECOVER | 2つの enum が同一処理（デッキ上→ライフ） |
| TRASH_FROM_DECK / SWAP_POWER / RAMP_DON | |
| RETURN_DON | `gm._return_don_selection` を**消費して None に戻す**。返却数を `record_turn_event("DON_RETURNED")` |
| REST_DON / FREEZE_DON / ACTIVE_DON | 実処理枚数を **`gm._last_resource_count` に記録**（§7-5「1枚につき」の resolver 連携）。ACTIVE_DON は **`not action.target` の場合のみ**ここ（target ありは対象ループの ACTIVE/ACTIVE_DON へ） |

**(b) 対象ループ**（L2576–2840）: `success = True` で開始し、`for target in targets` で
21分岐の `elif` チェーン。横断的関心事として:

- 除去系（`_LEAVE_ACTIONS` = KO/DISCARD/TRASH/BOUNCE/MOVE_TO_HAND/MOVE/DECK_BOTTOM/DECK_TOP/MOVE_CARD）
  に対する**保護ゲート**: 相手効果 × 場のカードのとき `_active_protection`（KO は LEAVE+EFFECT_KO、
  非KO は LEAVE のみ）→ 該当したら `continue`。
- **置換ゲート（B2）**: `_active_replacement` が内側中断を提示したら、残対象を
  `_defer_removal_targets` へ退避して **`return success` で即座に抜ける**。
- `owner` 不明の対象は skip（`if not owner: continue`）。
- **success は一度も False にならない**（初期値 True、各分岐が True を設定）。つまり
  対象ループ側の戻り値は常に True。未知の act_name も no-op で True（parser の
  ActionType.OTHER がここに落ちる）。この寛容さは**仕様**であり保存する。
- BUFF は status（POWER_OVERRIDE/COST_OVERRIDE/COST_REDUCTION/COUNTER/BLOCKER_DISABLE/既定）
  でさらに内部分岐。`gm._in_passive_recalc` フラグで書き込み先レイヤが変わる。
- REST は ON_REST 誘発（`_fire_on_rest_triggers`）、KO は `_resolve_on_ko`、PLAY_CARD は
  ON_PLAY 解決＋`_apply_passive_effects`＋`_enforce_field_limit` を呼ぶ（gm メソッドへの再入あり）。

### 2-2. 新パッケージ構成

```
opcg_sim/src/core/actions/
├── __init__.py        # apply_action(gm, player, action, targets, value, source_card) を公開
├── registry.py        # レジストリ本体・登録デコレータ・ActionType 正規化
├── player_level.py    # (a) プレイヤーレベル・ハンドラ 22種（~300行）
├── per_target.py      # (b) 対象ループハンドラ 21種（~260行）
└── target_loop.py     # 対象ループランナー（保護/置換ゲート・B2退避・success 規約を一元化）
```

- `gamestate.py` の `apply_action_to_engine` は次の1行デリゲートになる（**公開シグネチャ不変**。
  テスト74箇所・resolver L420 の呼び出しはそのまま動く）:

```python
def apply_action_to_engine(self, player, action, targets, value, source_card=None) -> bool:
    return actions.apply_action(self, player, action, targets, value, source_card)
```

- 依存の向き: `actions` → `models`（enums/models/effect_types）のみ。
  **actions から gamestate を import しない**（gm はダックタイピングで受ける）。
  `gamestate` → `actions` の一方向。`SELF_RESTRICTION_KEYS` / `FIELD_LIMIT` 等の定数は
  循環回避のため `opcg_sim/src/core/rules_constants.py`（新設・定数のみ）へ移し、
  gamestate からは再エクスポートで互換維持する。

### 2-3. レジストリ設計

```python
# registry.py
from typing import Callable, Optional
from ...models.enums import ActionType

# ホットパス配慮: コンテキストオブジェクトは作らない。位置引数で渡す。
GameHandler   = Callable[..., bool]   # (gm, player, action, targets, value, source_card) -> bool
TargetHandler = Callable[..., None]   # (gm, player, action, target, owner, source_list, value, source_card) -> None

_GAME_HANDLERS:   dict[ActionType, tuple[GameHandler, Optional[Callable]]] = {}
_TARGET_HANDLERS: dict[ActionType, TargetHandler] = {}

def game_handler(*types: ActionType, when: Optional[Callable] = None):
    """プレイヤーレベル・ハンドラの登録。when はガード述語（action を受け bool を返す）。
    when が False のアクションは対象ループへフォールスルーする（現行の条件付き分岐を保存）。"""
    def deco(fn):
        for t in types:
            _GAME_HANDLERS[t] = (fn, when)
        return fn
    return deco

def target_handler(*types: ActionType):
    def deco(fn):
        for t in types:
            _TARGET_HANDLERS[t] = fn
        return fn
    return deco

def normalize(action_type) -> Optional[ActionType]:
    """現行の `action.type.name if hasattr(...) else str(...)` の防御を enum 側で吸収する。
    ActionType ならそのまま。文字列なら ActionType[name] を試み、未知なら None（=no-op 経路）。"""
    if isinstance(action_type, ActionType):
        return action_type
    try:
        return ActionType[str(action_type)]
    except KeyError:
        return None
```

エントリポイント:

```python
# __init__.py
def apply_action(gm, player, action, targets, value, source_card=None) -> bool:
    if not action:
        return False
    atype = normalize(action.type)
    entry = _GAME_HANDLERS.get(atype)
    if entry is not None:
        fn, when = entry
        if when is None or when(action):
            return fn(gm, player, action, targets, value, source_card)
    return run_target_loop(gm, player, action, atype, targets, value, source_card)
```

ガード付き登録の例（現行の条件付き分岐を宣言的に保存）:

```python
@game_handler(ActionType.RULE_PROCESSING,
              when=lambda a: getattr(a, "status", None) in SELF_RESTRICTION_KEYS)
def rule_processing_self_restriction(gm, player, action, targets, value, source_card):
    rec = {"expire": gm.turn_count}
    if action.value and getattr(action.value, "base", None):
        rec["min_cost"] = action.value.base
    player.restrictions[action.status] = rec
    return True

@game_handler(ActionType.DISABLE_ABILITY,
              when=lambda a: getattr(a, "status", None) == "OPP_ONPLAY")
def disable_opp_onplay(gm, player, action, targets, value, source_card): ...

@game_handler(ActionType.ACTIVE_DON,
              when=lambda a: not getattr(a, "target", None))
def active_don_by_count(gm, player, action, targets, value, source_card): ...

@game_handler(ActionType.HEAL, ActionType.LIFE_RECOVER)
def heal(gm, player, action, targets, value, source_card): ...
```

enum エイリアス（`DAMAGE is DEAL_DAMAGE`、`DEBUFF is BUFF`）は同一メンバーなので
登録は1回でよい（現行の `act_name in ("DEAL_DAMAGE", "DAMAGE")` はエイリアスの
name が常に "DEAL_DAMAGE" になるための防御であり、enum キー化で不要になる）。

### 2-4. 対象ループランナー

横断的関心事（保護・置換・B2退避・success 規約）は**ランナーに一元化**し、
各ハンドラは「1対象への適用」だけを書く:

```python
# target_loop.py
_LEAVE_ACTIONS = frozenset({ActionType.KO, ActionType.DISCARD, ActionType.TRASH,
                            ActionType.BOUNCE, ActionType.MOVE_TO_HAND, ActionType.MOVE,
                            ActionType.DECK_BOTTOM, ActionType.DECK_TOP, ActionType.MOVE_CARD})

def run_target_loop(gm, player, action, atype, targets, value, source_card) -> bool:
    handler = _TARGET_HANDLERS.get(atype)   # None でもループは回す（現行の no-op 挙動）
    success = True                          # 現行規約: 対象0でも「何もしないことに成功」
    for target in targets:
        owner, source_list = gm._find_card_location(target)
        if not owner:
            continue
        if (atype in _LEAVE_ACTIONS and player.name != owner.name
                and source_list is owner.field):
            guard = ("LEAVE", "EFFECT_KO") if atype is ActionType.KO else ("LEAVE",)
            if gm._active_protection(target, guard, actor=player):
                continue
            if gm._active_replacement(target, guard, can_suspend=True):
                if gm.active_interaction is not None:          # B2: 内側中断が立った
                    remaining = targets[targets.index(target) + 1:]
                    if remaining:
                        gm._defer_removal_targets(player, action, remaining, value)
                    return success                              # 早期 return（現行同一）
                continue
        if handler is not None:
            handler(gm, player, action, target, owner, source_list, value, source_card)
    return success
```

ハンドラ側は現行の各 `elif` 本体を**逐語コピー**する（`self` → `gm`、`continue` →
`return` に置換するだけ）。例:

```python
# per_target.py
@target_handler(ActionType.KO)
def ko(gm, player, action, target, owner, source_list, value, source_card):
    gm.move_card(target, Zone.TRASH, owner)
    gm._resolve_on_ko(target, owner, cause="EFFECT", effect_controller=player)

@target_handler(ActionType.REST)
def rest(gm, player, action, target, owner, source_list, value, source_card):
    was_rested = target.is_rest
    target.is_rest = True
    ...  # DonInstance の don_rested への移送（現行 L2726–2733 逐語）
    if not was_rested and not isinstance(target, DonInstance):
        gm._fire_on_rest_triggers(target, by_attack=False, effect_controller=player,
                                  cause_source=source_card)
```

BUFF の status 内部分岐は1ハンドラ内のまま維持する（status→関数の副表に割るのは
過剰分割。`_in_passive_recalc` の読み取りが3系統に跨るため1関数の方が読みやすい）。

### 2-5. 重複ヘルパの集約

現関数内に**6回**出現する「OPPONENT なら相手プレイヤー」解決を1ヘルパへ:

```python
def resolve_side(gm, player, action, *, by_status: bool = False):
    """action.target.player もしくは action.status の OPPONENT 指定から対象側プレイヤーを返す。"""
```

（DRAW/SHUFFLE は `action.target.player.name == 'OPPONENT'`、LOOK/LOOK_LIFE/TRASH_FROM_DECK/
ORDER_LIFE は `action.status == "OPPONENT"` と**参照箇所が異なる**ため、フラグで区別し
現行挙動を厳密に保存する。）

### 2-6. Phase A で保存すべき挙動チェックリスト（実装PRのレビュー観点）

- [ ] `action` が falsy → False（先頭ガード）
- [ ] 未知/未登録の ActionType（OTHER 含む）→ 対象ループ no-op で True
- [ ] `action.type` が enum でない場合（文字列）も動く（normalize の fallback）
- [ ] MOVE_ATTACHED_DON のみ False を返しうる（moved >= n）
- [ ] `gm._last_resource_count`（REST_DON/FREEZE_DON/ACTIVE_DON）と
      `gm._return_don_selection` の**消費（読んだら None に戻す）**
- [ ] B2 退避の早期 return（残対象の deferred 化）と `_replacement_suspended` の
      resolver 連携（resolver.py L419–426）
- [ ] `_in_passive_recalc` による BUFF の書き込みレイヤ分岐
- [ ] REST_DON は `don_active.pop(0)`、ACTIVE_DON(枚数) は `don_rested.pop()`、
      ATTACH_DON は `pool.pop(0)` — **pop 位置の向きまで逐語保存**（journal の
      diff 順序・ベースライン一致に影響）
- [ ] `random.shuffle` の呼び出し回数・順序が不変（SHUFFLE ハンドラ）— 乱数消費順が
      変わるとベースラインが崩れる
- [ ] ハンドラモジュールに可変なモジュールレベル状態を置かない（§1-1）
- [ ] レジストリ dict は import 完了後に変更されない（登録はデコレータのみ）

### 2-7. Phase A の PR 分割

| PR | 内容 | ゲート |
|---|---|---|
| A-1 | actions パッケージ新設＋プレイヤーレベル22ハンドラ移設（対象ループは gamestate に残置） | 全テスト・監査・ベースライン |
| A-2 | 対象ループランナー＋21ハンドラ移設。gamestate 側は1行デリゲート化 | 上記＋ `-m slow`（journal）＋ bench |
| A-3 | 文字列比較の残骸除去・`rules_constants.py` 集約・SPEC.md 該当節更新 | 全テスト・監査 |

### 2-8. Phase A セルフレビュー由来の followup（挙動不変・別PR）

PR #157 の high-effort セルフレビューで挙がった、挙動に影響しない cleanup（正しさのバグは検出ゼロ）。
コメント/文書修正は #157 で対応済み。以下は別 followup（各々テスト再実行を伴うため分離）:

- **OPPONENT 解決の重複（player_level の6ハンドラ）**: `draw`/`shuffle` は `action.target.player`、
  `look`/`look_life`/`order_life`/`trash_from_deck` は `action.status` を見る**2方式が意図的に別条件**。
  `_don_pool_player` は両方を見るため単純統合は挙動を変える。統合するなら parser のエンコーディングを
  棚卸しした上で `resolve_side(gm, player, action, *, by_status)`（設計 §2-5）を導入する。
- **duration→`continuous.apply` の定型 ×7（per_target）**: 期間文字列は `continuous.py` の
  名前付き定数（`THIS_TURN` 等）を import し、`_apply_timed` ヘルパへ集約（typo の早期検出・失効仕様の一元化）。
- **保護語彙の層分裂（target_loop の `_LEAVE_ACTIONS`＋guard タプル）**: `BATTLE_KO` 等の兄弟バケツと
  解釈器は gamestate 側。除去/バトルの保護語彙を `rules_constants.py` へ集約し両者で共有する。
- **no-op ハンドラ（`reveal`/`rule_processing`）とランナーのフォールスルー（未登録=no-op）が二重機構**。
  どちらかに統一（「仕様として明示 no-op」なら登録を残しコメント化、そうでなければ削除）。
- **二重 dict lookup**（対象ループ系は `_GAME_HANDLERS` 空振り→`_TARGET_HANDLERS`）: 統合レジストリ
  `ActionType→(game_fn, guard, target_fn)` で1回に。ホットパスの微小最適化（bench で要測定）。
- **`normalize` の文字列エイリアス解決**（`ActionType["DEBUFF"]`）: 現状 `GameAction.type` は常に enum で
  到達不能だが、旧・生文字列比較とは非等価。文字列は防御専用である旨を維持（将来 string 経路を作らない）。

---

## 3. Phase B: GameManager の責務分割

### 3-1. 方式: ステートレス・モジュール関数 + 1行デリゲート

- 新設 `opcg_sim/src/core/engine/` パッケージに **状態を持たない関数群** を移す。
  各関数は第1引数に gm を取る。
- GameManager には**同名の1行デリゲートを残す**（公開API・テスト74ファイル・resolver/api の
  呼び出し互換を完全維持）。gamestate.py は「状態定義＋デリゲート＋少数の中核メソッド」
  （目標 **600行以下**）になる。
- クラス化（サービスオブジェクト）を採らない理由: journal（§1-1）と clone（§1-2）に
  対して証明が自明（インスタンスも可変状態も増えない）。将来 DI が必要になった時点で
  関数→クラスへの昇格は機械的にできる。

### 3-2. メソッド移管表

| 新モジュール | 移すメソッド（現 gamestate.py の行） | 概算行数 |
|---|---|---|
| `engine/turn_flow.py` | start_game(925) / do_mulligan(950) / keep_hand(967) / _check_mulligan_complete(976) / finish_setup(982) / end_turn(988) / _fire_turn_end_triggers(997) / _flush_pending_end_of_turn(1012) / switch_turn(1050) / refresh_phase(1063) / _reset_player_status(1066) / refresh_all(1075) / draw_phase(1104) / don_phase(1108) / main_phase(1115) | ~230 |
| `engine/battle.py` | declare_attack(1364) / _advance_battle_triggers(1429) / handle_block(1507) / apply_counter(1523) / resolve_attack(1540) / _finish_attack(1587) / _suspend_for_battle_ko_replacement(1599) / has_blocker(1356) / check_victory(1618) / _has_deckout_win_replace(1627) | ~270 |
| `engine/triggers.py` | _enqueue_trigger(1448) / _advance_pending_triggers(1458) / _relocate_activated_trigger_card(1474) / _suspend_for_trigger_confirm(1492) / _ko_trigger_matches(2080) / _resolve_on_ko(2114) / _rest_subject_matches(2126) / _fire_on_rest_triggers(2158) / _leave_subject_matches(2184) / _enqueue_on_leave(2212) / _enqueue_life_decrease(2228) / _fire_on_life_decrease(2245) | ~230 |
| `engine/interaction.py` | resolve_interaction(636, **232行**) / get_pending_request(522) / default_interaction_payload(470) / pending_actor_action(605) / _defer_resolver_stack(2034) / _defer_removal_targets(2044) / _resume_deferred_continuations(2055) | ~450 |
| `engine/guards.py` | _active_protection(1775) / _find_replacement(1859) / _register_granted_replacements(1933) / _active_replacement(1962) / _auto_resolve_replacement(1996) / _active_restriction(1703) / _blocks_effect_play(1715) / _has_rested_play(1689) | ~300 |
| `engine/passives.py` | refresh_passive_state(868) / _apply_passive_effects(1154) / _is_reactive_passive(1121) / _find_first_action(1135) / _apply_hand_self_cost(1248) | ~180 |
| `engine/card_moves.py` | move_card(1287) / _find_card_location(1275) / _find_card_by_uuid(505) / draw_card(1269) / pay_cost(1346) / _enforce_field_limit(877) / _suspend_for_field_overflow(886) / _return_one_don(2251) / _don_pool_player(2270) / _apply_leader_don_deck_rule(261) | ~180 |
| `engine/values.py` | get_dynamic_value(2842) / _resolve_power_reference(2878) | ~55 |
| `actions/`（Phase A） | apply_action_to_engine 本体 | ~560 |
| **gamestate.py に残す** | `__init__` / `__setattr__` / clone / 対話スタック property 群 / get_legal_actions(320) / _has_activatable_main(444) / play_card_action(1643) / resolve_ability(1727) / _find_action(1753) / record_turn_event / _record_event_played / _validate_action / get_debug_snapshot / Player クラス / モジュール定数 | ~600 |

> 残置の理由: `get_legal_actions` と `resolve_ability` / `play_card_action` は
> 「エンジンの正面玄関」であり、分割各層を編成する薄いオーケストレーションとして
> GameManager 本体が持つのが自然。行数的にも許容範囲。

### 3-3. 移管の実装規約

1. **逐語移動**: 関数本体は `self` → `gm` の置換以外、変更しない（挙動改善・改名・
   型付け強化は本リファクタでは行わない。やる場合は挙動不変PRの後に別PR）。
2. **デリゲート形式**（例）:
   ```python
   # gamestate.py
   from .engine import battle as _battle
   class GameManager:
       def declare_attack(self, attacker, target):
           return _battle.declare_attack(self, attacker, target)
   ```
3. **プライベートメソッドもデリゲートを残す**（`_active_protection` 等は actions/ の
   ランナーや engine 内他モジュールから gm 経由で呼ばれるため。gm がハブである限り
   engine モジュール間の相互 import は発生しない）。
4. 依存の向き: `engine/*` → `models` / `effects.continuous` のみ。**engine 同士の直接
   import 禁止**（必要な相互呼び出しはすべて gm のデリゲート経由。循環を構造的に排除）。
5. docstring・コメント（日本語の仕様注記）は**本体側へ移動**し、デリゲートには付けない。

### 3-4. Phase B の PR 分割と順序（結合度の低い順）

| PR | モジュール | リスク | 追加ゲート |
|---|---|---|---|
| B-1 | values.py + guards.py | 低（読み取り中心） | 通常ゲート |
| B-2 | card_moves.py + passives.py | 中（move_card は全域から呼ばれる） | `-m slow` + bench |
| B-3 | triggers.py | 中 | `-m slow` |
| B-4 | battle.py + turn_flow.py | 中 | 通常ゲート + bench |
| B-5 | interaction.py | 高（resolve_interaction 232行・対話スタック） | `-m slow` + `test_interaction_stack.py` / `test_journal_concurrency.py` 個別確認 |
| B-6 | 仕上げ: gamestate.py の整理・SPEC.md / TEST_SPEC.md の該当節更新 | 低 | 全ゲート |

各PRは**1モジュール群のみ**を動かす。レビューは「git diff が純粋な移動＋デリゲート追加に
見えること」を第一観点とする（`git diff --color-moved=dimmed-zebra` で移動検証）。

---

## 4. リスクと対策

| リスク | 対策 |
|---|---|
| if-chain の隠れた順序依存 | 分岐キー（act_name）は相互排他であることを確認済み。条件付き3分岐（RULE_PROCESSING / DISABLE_ABILITY / ACTIVE_DON）は when ガードで順序ごと保存（§2-3） |
| 乱数消費順の変化でベースライン崩壊 | ハンドラ逐語コピー＋ pop 方向/shuffle 呼び出し位置の保存（§2-6）。ベースライン再生成禁止 |
| journal 記録漏れ（新しい可変状態の混入） | §1-1 の禁止事項をレビュー観点化。`-m slow` の roundtrip テストを A-2 以降の全PRで実行 |
| deepcopy（clone）対象の増加 | ステートレス関数方式で構造的に回避（§3-1） |
| ホットパス性能劣化 | コンテキストオブジェクト非採用（位置引数）。dict ディスパッチは現行の最大45回文字列比較より速い見込みだが、bench_decide で ±5% を確認 |
| tests/ の直接呼び出し（74ファイル）破壊 | 公開シグネチャ完全維持（1行デリゲート）。テスト側の変更はゼロが正 |
| PR の巨大化・レビュー不能 | 9PR に分割（A-1〜A-3, B-1〜B-6）。各PRの diff は「移動」が主で新規ロジックなし |

## 5. 非目標（今回やらないこと）

- 挙動の変更・バグ修正（発見したら本リファクタに混ぜず、別Issue/PRで）
- `Dict[str, Any]` で流れる interaction/pending の dataclass 化（別リファクタ項目）
- api/app.py の分割（洗い出し項目2として別設計）
- parser/resolver/atoms の分割（別項目）
- 型ヒントの全面強化・改名（挙動不変PRの純度を守るため）

## 6. 完了条件（達成状況）

- gamestate.py を **796行**へ縮小（Phase B 着手前 ~2355行 → -66%）。8 責務群を `engine/`
  （values/guards/card_moves/passives/triggers/battle/turn_flow/interaction・計 ~1880行）へ分割。
  当初目標「600行以下・最長関数100行以下」は未達だが、これは設計どおり **本体に残す方針の
  オーケストレーションメソッド**（`get_legal_actions`＝123行が最長・`play_card_action`/`resolve_ability`）
  と `Player` クラス・全デリゲートの合計であり、意図的な残置（§3-2）。さらなる分割は非目標。
- ✅ `apply_action_to_engine` の文字列比較 0 箇所（enum キーのレジストリのみ・Phase A）。
- ✅ 全品質ゲート green（ベースライン**無変更**・構造監査 0・slow journal pass・bench ±5%内）。
  各 B-PR で個別確認済み。engine 各モジュールの未解決グローバル参照ゼロを co_names 検査で担保。
- ✅ `docs/SPEC.md` のモジュール構成節が新レイアウト（`engine/` パッケージ）を反映。

> B-5 で `resolve_interaction` の場札超過継続経路が参照する `FIELD_LIMIT` の import 漏れを
> 挙動ベースライン（OP06-086）が検出。値スキャンでは拾えない稀経路の依存は、関数の
> `co_names` を gamestate モジュールグローバルと突合する検査で網羅確認する運用とした。
