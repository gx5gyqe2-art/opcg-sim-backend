"""実行カバレッジスクリプト。

全カードの ACTIVATE_MAIN 能力を GameManager 上で直接発動し、
「実行時にどう動いたか」を分類して手動テストの優先順位付けに使う。

分類:
  ERROR        : 例外発生 → エンジン修正が必要（最優先）
  INTERACTIVE  : 発動中にプレイヤー選択が発生 → 手動テスト必須リスト
  EXECUTED     : 盤面変化あり・自動確認済み
  NO_CHANGE    : 発動完了したが盤面変化なし（条件未達 or OTHER の疑い）
  SKIP         : ACTIVATE_MAIN 能力なし（対象外）

has_other フラグ: AST 中に ActionType.OTHER が含まれる（effect_diagnostics の OTHER と同義）

実行:
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show ERROR
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show NO_CHANGE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001
"""
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
# フィラーカード（テスト用ダミー）
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
# テスト用ゲームステート構築
# ---------------------------------------------------------------------------

def _build_test_state(
    test_card: CardMaster,
) -> Tuple["GameManager", Player, Player, CardInstance]:
    """メインフェイズ・リソース豊富な状態のゲームを組み立てる。"""
    p1 = make_player("P1")
    p2 = make_player("P2")

    # P1: ドン 10 枚・手札 5 枚・トラッシュ 10 枚・デッキ 20 枚・ライフ 5 枚
    # フィールドに 3 体（フィールド数条件・コスト払い条件を満たすため）
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

    # テストカードを適切なゾーンに配置
    source = CardInstance(test_card, "P1")
    if test_card.type == CardType.LEADER:
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


def _drain(gm: "GameManager", limit: int = 30) -> Tuple[bool, int]:
    """空選択でインタラクションを消化。(スタック検知, 処理数) を返す。"""
    count = 0
    while gm.active_interaction and count < limit:
        ia     = gm.active_interaction
        player = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        try:
            gm.resolve_interaction(player, {"selected_uuids": [], "index": 0})
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


def _has_other(abilities) -> bool:
    for ab in abilities:
        for act in list(_walk(ab.effect)) + list(_walk(ab.cost)):
            if act and act.type == ActionType.OTHER:
                return True
    return False


# ---------------------------------------------------------------------------
# 分類
# ---------------------------------------------------------------------------

_PRIORITY = {"ERROR": 4, "INTERACTIVE": 3, "EXECUTED": 2, "NO_CHANGE": 1, "SKIP": 0}


@dataclass
class CardResult:
    card_id:   str
    name:      str
    status:    str    # ERROR / INTERACTIVE / EXECUTED / NO_CHANGE / SKIP
    has_other: bool = False
    detail:    str  = ""


def classify(master: CardMaster) -> CardResult:
    activate_abs = [ab for ab in master.abilities if ab.trigger == TriggerType.ACTIVATE_MAIN]
    h_other      = _has_other(master.abilities)

    if not activate_abs:
        return CardResult(master.card_id, master.name, "SKIP", h_other)

    best = CardResult(master.card_id, master.name, "NO_CHANGE", h_other, "no ability fired")

    for ability in activate_abs:
        try:
            gm, p1, p2, source = _build_test_state(master)
        except Exception as e:
            return CardResult(master.card_id, master.name, "ERROR", h_other, f"setup: {e}")

        before = _snap(p1, p2)

        try:
            gm.resolve_ability(p1, ability, source)
        except Exception:
            return CardResult(master.card_id, master.name, "ERROR", h_other,
                              traceback.format_exc(limit=2))

        stuck, n_ia = _drain(gm)
        after   = _snap(p1, p2)
        changed = before != after

        if stuck or gm.active_interaction:
            msg = gm.active_interaction.get("message", "") if gm.active_interaction else "stuck in drain loop"
            r = CardResult(master.card_id, master.name, "INTERACTIVE", h_other, msg)
        elif changed:
            r = CardResult(master.card_id, master.name, "EXECUTED", h_other, _snap_diff(before, after))
        elif n_ia > 0:
            # インタラクションがあったが空選択で解決 → 実際のプレイヤー選択が必要
            r = CardResult(master.card_id, master.name, "INTERACTIVE", h_other,
                           f"{n_ia} interaction(s) auto-resolved (no state change)")
        else:
            r = CardResult(master.card_id, master.name, "NO_CHANGE", h_other,
                           "no state change, no interaction (condition failed or OTHER)")

        if _PRIORITY.get(r.status, 0) > _PRIORITY.get(best.status, 0):
            best = r

    return best


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run(show: Optional[str] = None, card_filter: Optional[str] = None) -> None:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()

    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    results: List[CardResult] = []
    total = len(card_ids)

    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None:
            continue
        results.append(classify(master))

    sys.stderr.write(f"\r完了: {total} カード処理済み\n")

    # --- サマリ ---
    counts     = Counter(r.status for r in results)
    other_flag = Counter(r.status for r in results if r.has_other)
    print("=== 実行カバレッジ ===")
    for s in ("ERROR", "INTERACTIVE", "EXECUTED", "NO_CHANGE", "SKIP"):
        n    = counts[s]
        o    = other_flag.get(s, 0)
        note = f"  (うち OTHER フラグ {o} 件)" if o else ""
        print(f"  {s:<14}: {n:4d}{note}")
    print()

    # --- 詳細出力 ---
    # --show 指定時はその分類のみ、デフォルトは ERROR と INTERACTIVE
    targets = [show] if show else ["ERROR", "INTERACTIVE"]
    _LABELS = {
        "ERROR":       "--- ERROR ({n} 件) --- ← 要修正",
        "INTERACTIVE": "--- INTERACTIVE ({n} 件) --- ← 手動テスト優先リスト",
        "EXECUTED":    "--- EXECUTED ({n} 件) ---",
        "NO_CHANGE":   "--- NO_CHANGE ({n} 件) --- ← 条件未達 or OTHER の疑い",
        "SKIP":        "--- SKIP ({n} 件) ---",
    }
    for target in targets:
        items = [r for r in results if r.status == target]
        label = _LABELS.get(target, "--- {target} ({n} 件) ---").format(n=len(items), target=target)
        print(label)
        if not items:
            print("  (なし)")
            print()
            continue
        for r in sorted(items, key=lambda x: x.card_id):
            other_mark = " [OTHER]" if r.has_other else ""
            print(f"  {r.card_id:<12}  {r.name}{other_mark}")
            if r.detail:
                print(f"    └ {r.detail[:120]}")
        print()


if __name__ == "__main__":
    show_opt = None
    card_opt = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--show" and i + 1 < len(args):
            show_opt = args[i + 1]
            i += 2
        elif args[i] == "--card" and i + 1 < len(args):
            card_opt = args[i + 1]
            i += 2
        else:
            i += 1

    run(show=show_opt, card_filter=card_opt)
