"""実行カバレッジスクリプト。

全カードの全トリガータイプ能力を GameManager 上で発動し、
手動テストの優先順位付けに使う分類レポートを出力する。

分類:
  ERROR        : 例外発生 → エンジン修正が必要（最優先）
  INTERACTIVE  : 発動中にプレイヤー選択が発生 → 手動テスト必須リスト
  EXECUTED     : 盤面変化ありまたは action_events 記録あり・自動確認済み
  EXECUTED+WARN: EXECUTED かつ効果タイプと盤面変化の方向が不一致（要確認）
  NO_CHANGE    : 発動完了したが変化なし（条件未達 or OTHER の疑い）

has_other フラグ: AST 中に ActionType.OTHER が含まれる（effect_diagnostics の OTHER と同義）

実行:
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show ERROR
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show NO_CHANGE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --trigger ON_PLAY
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001
"""
import os
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance, CardMaster
from opcg_sim.src.models.enums import ActionType, CardType, TriggerType, Phase
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.utils.loader import CardLoader
from engine_helpers import make_master, make_player

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "opcg_sim", "data",
)


# ---------------------------------------------------------------------------
# フィラーカード
# ---------------------------------------------------------------------------

_filler_master: Optional[CardMaster] = None


def _get_filler_master() -> CardMaster:
    global _filler_master
    if _filler_master is None:
        _filler_master = make_master(card_id="FILLER", name="フィラー", power=1000)
    return _filler_master


def _instances(n: int, owner: str) -> List[CardInstance]:
    m = _get_filler_master()
    return [CardInstance(m, owner) for _ in range(n)]


# ---------------------------------------------------------------------------
# テスト用ゲームステート
# ---------------------------------------------------------------------------

def _build_test_state(
    test_card: CardMaster,
    source_in_hand: bool = False,
) -> Tuple[GameManager, Player, Player, CardInstance]:
    """メインフェイズ・リソース豊富な状態のテストゲームを構築する。

    source_in_hand=True のときはテストカードを手札に置く（ON_PLAY 用）。
    """
    p1 = make_player("P1")
    p2 = make_player("P2")

    # P1: ドン 10 枚・手札 5 枚・トラッシュ 10 枚・デッキ 20 枚・ライフ 5 枚
    # フィールド 3 体（フィールド条件・コスト条件を満たす）
    p1.don_deck   = _instances(5,  "P1")
    p1.don_active = _instances(10, "P1")
    p1.hand       = _instances(5,  "P1")
    p1.trash      = _instances(10, "P1")
    p1.deck       = _instances(20, "P1")
    p1.life       = _instances(5,  "P1")
    p1.field      = _instances(3,  "P1")

    # P2: フィールド 3 体・手札 3 枚・ドン 5 枚・デッキ 20 枚・ライフ 5 枚
    p2.don_active = _instances(5,  "P2")
    p2.field      = _instances(3,  "P2")
    p2.hand       = _instances(3,  "P2")
    p2.deck       = _instances(20, "P2")
    p2.life       = _instances(5,  "P2")

    gm             = GameManager(p1, p2)
    gm.turn_player = p1
    gm.opponent    = p2
    gm.turn_count  = 2
    gm.phase       = Phase.MAIN

    source = CardInstance(test_card, "P1")
    if source_in_hand:
        p1.hand.append(source)
    elif test_card.type == CardType.LEADER:
        p1.leader = source
    elif test_card.type == CardType.STAGE:
        p1.stage = source
    else:
        p1.field.append(source)

    return gm, p1, p2, source


def _snap(p1: Player, p2: Player) -> tuple:
    return (
        len(p1.hand), len(p1.field), len(p1.trash), len(p1.deck), len(p1.life),
        len(p1.don_active) + len(p1.don_rested),
        len(p2.hand), len(p2.field), len(p2.trash), len(p2.deck), len(p2.life),
    )


_SNAP_KEYS = (
    "p1_hand", "p1_field", "p1_trash", "p1_deck", "p1_life", "p1_don",
    "p2_hand", "p2_field", "p2_trash", "p2_deck", "p2_life",
)


def _snap_diff(before: tuple, after: tuple) -> str:
    return ", ".join(
        f"{_SNAP_KEYS[i]}:{before[i]}→{after[i]}"
        for i in range(len(_SNAP_KEYS))
        if before[i] != after[i]
    )


def _smart_drain(gm: GameManager, limit: int = 30) -> Tuple[bool, int]:
    """インタラクションを賢㍖に解決する。

    - SELECT_TARGET: selectable_uuids から constraints.min 枚を選ぶ（空選択でなく有効な候補を渡す）
    - CHOICE: 先頭の選戠肢（index=0）を選ぶ
    - (stuck, 処理数) を返す
    """
    count = 0
    while gm.active_interaction and count < limit:
        ia          = gm.active_interaction
        player      = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        action_type = ia.get("action_type", "")

        if action_type == "SELECT_TARGET":
            candidates = ia.get("selectable_uuids") or [
                c.uuid for c in ia.get("candidates", [])
            ]
            cons     = ia.get("constraints") or {}
            min_req  = cons.get("min", 0)
            max_req  = cons.get("max", 1)
            # max が -1（=ALL/REMAINING、例「残りをデッキの下に置く」）の場合は全候補を選ぶ。
            # 従来は max(min,1)=1枚しか選ばず、残りの temp カードが取り残されて
            # TEMP リーク（デッキから消失）に見える測定アーティファクトを生んでいた。
            if max_req is not None and max_req < 0:
                n_select = len(candidates)
            else:
                n_select = max(min_req, 1) if candidates else 0
                if max_req:
                    n_select = min(n_select, max_req)
            selected = candidates[:n_select]
            payload  = {"selected_uuids": selected, "index": 0}
        else:  # CHOICE 等は先頭の選戠肢
            payload = {"selected_uuids": [], "index": 0}

        try:
            gm.resolve_interaction(player, payload)
        except Exception:
            break
        count += 1
    return count >= limit, count


# ---------------------------------------------------------------------------
# AST ウォーク
# ---------------------------------------------------------------------------

def _walk(node):
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk(a)
    elif isinstance(node, Branch):
        yield from _walk(node.if_true)
        if node.if_false:
            yield from _walk(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk(o)


def _has_other_in(abilities) -> bool:
    for ab in abilities:
        for act in list(_walk(ab.effect)) + list(_walk(ab.cost)):
            if act and act.type == ActionType.OTHER:
                return True
    return False


# ---------------------------------------------------------------------------
# 方向性アサーション（改善第二値なし）
# ---------------------------------------------------------------------------

# ActionType 名 → (変化前 tuple, 変化後 tuple) から「期待方向」を返す関数
# snap インデックス: 0=p1_hand 1=p1_field 2=p1_trash 3=p1_deck 4=p1_life 5=p1_don
#                       6=p2_hand 7=p2_field 8=p2_trash 9=p2_deck 10=p2_life
# 方向チェックは「対象がどちら側か」に依存しない side-agnostic で書く。対象側を固定して
# 書くと、相手キャラを KO/デッキ送り/ハンデスする効果や自己バウンス等が軒並み誤検知になる
# （effect_coverage の汎用足場では対象側が一意に定まらないため）。
_DIRECTION: Dict[str, any] = {
    "DRAW":             lambda b, a: a[0] > b[0],                       # 自分の手札増
    "DISCARD":          lambda b, a: a[0] < b[0] or a[6] < b[6],        # どちらかの手札減
    "KO":               lambda b, a: a[1] < b[1] or a[7] < b[7],        # どちらかのフィールド減
    "BOUNCE":           lambda b, a: a[1] < b[1] or a[7] < b[7],        # どちらかのフィールド減
    "RAMP_DON":         lambda b, a: a[5] > b[5],                       # ドン増
    "RETURN_DON":       lambda b, a: a[5] < b[5],                       # ドン減
    "HEAL":             lambda b, a: a[4] > b[4] or a[10] > b[10],      # どちらかのライフ増
    "TRASH_FROM_DECK":  lambda b, a: a[3] < b[3] or a[9] < b[9],        # どちらかのデッキ減
    # PLAY_CARD: フィールド増、または公開/サーチでデッキが減る派生も許容（look-and-play 系）
    "PLAY_CARD":        lambda b, a: a[1] > b[1] or a[7] > b[7] or a[3] < b[3] or a[9] < b[9],
    # DECK_BOTTOM: どちらかのフィールド減＋どちらかのデッキ増（場→デッキ）
    "DECK_BOTTOM":      lambda b, a: (a[1] < b[1] or a[7] < b[7]) and (a[3] > b[3] or a[9] > b[9]),
    "TRASH_FROM_HAND":  lambda b, a: a[0] < b[0] or a[6] < b[6],        # どちらかの手札減
    "MOVE":             lambda b, a: b != a,                            # 何か変わればOK
}


def _soft_assert(abilities, before: tuple, after: tuple) -> Optional[str]:
    """単一効果タイプの能力に限り、盤面変化の方向が期待値と逆の場合に警告を返す。

    複合効果（DRAW+KO 等）は誤検知が多いためチェック対象外。
    変化なし（条件未達の可能性）は警告しない。
    """
    if before == after:
        return None  # 変化なし = 条件未達の可能性が高いためスキップ

    type_names = [
        act.type.name
        for ab in abilities
        for act in _walk(ab.effect)
        if act and act.type != ActionType.OTHER
    ]
    if not type_names:
        return None

    # 複数のタイプが混在する場合は誤検知が多いためチェックしない
    unique_types = {n for n in type_names if n in _DIRECTION}
    if len(unique_types) != 1:
        return None

    (dominant,) = unique_types
    checker = _DIRECTION[dominant]
    if not checker(before, after):
        return f"WARN: {dominant} 期待だが方向不一致"
    return None


# ---------------------------------------------------------------------------
# 分類
# ---------------------------------------------------------------------------

_PRIORITY = {"ERROR": 4, "INTERACTIVE": 3, "EXECUTED": 2, "NO_CHANGE": 1}


@dataclass
class AbilityResult:
    card_id:   str
    name:      str
    trigger:   str    # TriggerType.name
    status:    str    # ERROR / INTERACTIVE / EXECUTED / NO_CHANGE
    has_other: bool = False
    detail:    str  = ""


def _outcome(
    gm: GameManager,
    p1: Player,
    before: tuple,
    card_id: str,
    name: str,
    trig: str,
    h_other: bool,
    abilities=None,
) -> AbilityResult:
    """発動後のゲーム状態から結果を返す。インタラクションは賢㍖選戠で消化。"""
    stuck, n_ia = _smart_drain(gm)
    after   = _snap(p1, gm.p2)
    changed = before != after
    # resolve_ability が登録するアクション履歴（REVEAL/SHUFFLE を EXECUTED に分類するため）
    has_events = bool(getattr(gm, "action_events", []))

    if stuck or gm.active_interaction:
        msg = gm.active_interaction.get("message", "") if gm.active_interaction else "stuck in drain loop"
        return AbilityResult(card_id, name, trig, "INTERACTIVE", h_other, msg)
    if changed:
        diff = _snap_diff(before, after)
        warn = _soft_assert(abilities or [], before, after) if abilities else None
        detail = f"{diff}  |  {warn}" if warn else diff
        return AbilityResult(card_id, name, trig, "EXECUTED", h_other, detail)
    if n_ia > 0:
        return AbilityResult(card_id, name, trig, "INTERACTIVE", h_other,
                             f"{n_ia} interaction(s) smart-resolved (no state change)")
    if has_events:
        return AbilityResult(card_id, name, trig, "EXECUTED", h_other,
                             "ability fired (no visible state change)")
    return AbilityResult(card_id, name, trig, "NO_CHANGE", h_other,
                         "no state change, no interaction")


def _test_on_play(master: CardMaster, h_other: bool) -> AbilityResult:
    """ON_PLAY: play_card_action を経由（全 ON_PLAY 能力を一括発動）。"""
    on_play_abs = [ab for ab in master.abilities
                   if (ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)) == "ON_PLAY"]
    try:
        gm, p1, p2, source = _build_test_state(master, source_in_hand=True)
    except Exception as e:
        return AbilityResult(master.card_id, master.name, "ON_PLAY", "ERROR", h_other, f"setup: {e}")

    before = _snap(p1, p2)
    try:
        gm.play_card_action(p1, source)
    except Exception:
        return AbilityResult(master.card_id, master.name, "ON_PLAY", "ERROR", h_other,
                             traceback.format_exc(limit=2))

    return _outcome(gm, p1, before, master.card_id, master.name, "ON_PLAY", h_other, on_play_abs)


def _test_ability(master: CardMaster, ability, trig: str, h_other: bool) -> AbilityResult:
    """resolve_ability を直接呼んでテスト（ON_PLAY 以外の全トリガー）。"""
    try:
        gm, p1, p2, source = _build_test_state(master)
    except Exception as e:
        return AbilityResult(master.card_id, master.name, trig, "ERROR", h_other, f"setup: {e}")

    before = _snap(p1, p2)
    try:
        gm.resolve_ability(p1, ability, source)
    except Exception:
        return AbilityResult(master.card_id, master.name, trig, "ERROR", h_other,
                             traceback.format_exc(limit=2))

    return _outcome(gm, p1, before, master.card_id, master.name, trig, h_other, [ability])


def classify(master: CardMaster) -> List[AbilityResult]:
    """カードの全能力をトリガータイプ別にテストし、各タイプの最悪結果を返す。"""
    if not master.abilities:
        return []

    h_other_all = _has_other_in(master.abilities)

    by_trigger: Dict[str, list] = defaultdict(list)
    for ab in master.abilities:
        k = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
        by_trigger[k].append(ab)

    results: List[AbilityResult] = []
    for trig, abilities in by_trigger.items():
        h_other = _has_other_in(abilities) or h_other_all

        if trig == "ON_PLAY":
            # play_card_action が全 ON_PLAY を発動するためまとめてテスト
            r = _test_on_play(master, h_other)
        else:
            r = None
            for ability in abilities:
                ri = _test_ability(master, ability, trig, h_other)
                if r is None or _PRIORITY.get(ri.status, 0) > _PRIORITY.get(r.status, 0):
                    r = ri

        if r:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

_LABELS = {
    "ERROR":       "--- ERROR ({n} 件) --- ← 要修正",
    "INTERACTIVE": "--- INTERACTIVE ({n} 件) --- ← 手動テスト優先リスト",
    "EXECUTED":    "--- EXECUTED ({n} 件) ---",
    "NO_CHANGE":   "--- NO_CHANGE ({n} 件) --- ← 条件未達 or OTHER の疑い",
}


def _print_section(items: List[AbilityResult], status: str) -> None:
    label = _LABELS.get(status, f"--- {status} ({{n}} 件) ---").format(n=len(items))
    print(label)
    if not items:
        print("  (なし)")
    else:
        for r in sorted(items, key=lambda x: (x.card_id, x.trigger)):
            other_mark = " [OTHER]" if r.has_other else ""
            warn_mark  = " [WARN]"  if r.detail and "WARN" in r.detail else ""
            print(f"  [{r.trigger:<20}] {r.card_id:<12}  {r.name}{other_mark}{warn_mark}")
            if r.detail:
                print(f"    └ {r.detail[:120]}")
    print()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run(
    show:        Optional[str] = None,
    card_filter: Optional[str] = None,
    trig_filter: Optional[str] = None,
) -> None:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()

    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    all_results: List[AbilityResult] = []
    skipped = 0
    total   = len(card_ids)

    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None:
            continue
        card_results = classify(master)
        if not card_results:
            skipped += 1
            continue
        for r in card_results:
            if trig_filter is None or r.trigger == trig_filter:
                all_results.append(r)

    sys.stderr.write(f"\r完了: {total} カード処理済み\n")

    # --- サマリ ---
    counts = Counter(r.status for r in all_results)
    warn_count = sum(1 for r in all_results if r.status == "EXECUTED" and "WARN" in r.detail)
    print("=== 実行カバレッジ (能力単位) ===")
    print(f"  能力なし(SKIP)  : {skipped:4d}")
    for s in ("ERROR", "INTERACTIVE", "EXECUTED", "NO_CHANGE"):
        o    = sum(1 for r in all_results if r.status == s and r.has_other)
        note = f"  (うち OTHER フラグ {o} 件)" if o else ""
        print(f"  {s:<14}: {counts[s]:4d}{note}")
    if warn_count:
        print(f"  WARN 付き EXECUTED: {warn_count:4d}  ← 方向不一致の疑い（要確認）")
    print()

    # --- トリガー別内訳 ---
    print("--- トリガー別内訳 ---")
    trig_status: Dict[str, Counter] = defaultdict(Counter)
    for r in all_results:
        trig_status[r.trigger][r.status] += 1
    for trig in sorted(trig_status):
        cs    = trig_status[trig]
        parts = [f"{s}={cs[s]}" for s in ("ERROR", "INTERACTIVE", "EXECUTED", "NO_CHANGE") if cs[s]]
        print(f"  {trig:<22}  {', '.join(parts)}")
    print()

    # --- 詳細リスト ---
    # --show 指定時はその分類のみ、デフォルトは ERROR と INTERACTIVE
    targets = [show] if show else ["ERROR", "INTERACTIVE"]
    for target in targets:
        items = [r for r in all_results if r.status == target]
        _print_section(items, target)

    # WARN 付き EXECUTED も常に表示（--show 指定なし or --show EXECUTED 時）
    if not show or show == "EXECUTED":
        warn_items = [r for r in all_results if r.status == "EXECUTED" and "WARN" in r.detail]
        if warn_items:
            print(f"--- WARN 付き EXECUTED ({len(warn_items)} 件) --- ← 方向不一致の疑い（要確認）")
            for r in sorted(warn_items, key=lambda x: (x.card_id, x.trigger)):
                print(f"  [{r.trigger:<20}] {r.card_id:<12}  {r.name}")
                print(f"    └ {r.detail[:120]}")
            print()


if __name__ == "__main__":
    show_opt = None
    card_opt = None
    trig_opt = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--show" and i + 1 < len(args):
            show_opt = args[i + 1]; i += 2
        elif args[i] == "--card" and i + 1 < len(args):
            card_opt = args[i + 1]; i += 2
        elif args[i] == "--trigger" and i + 1 < len(args):
            trig_opt = args[i + 1]; i += 2
        else:
            i += 1

    run(show=show_opt, card_filter=card_opt, trig_filter=trig_opt)
