"""対局中インバリアント（不変条件）チェック。

自己対戦（CPU 対 CPU）・テストの各ステップ後に呼び、ルール上ありえない盤面を
**即座に検出** するための実行時アサート群（docs/TEST_SPEC.md §3.1）。
`text_execution_audit` の実行時フラグ思想（SUSPEND_LEAK 等）を「ゲーム進行中」に
常時作動させ、効果のサイレント失敗・中断リークを進行から自動炙り出しする。

`check_invariants(manager)` は違反の (code, message) リストを返す。空ならクリーン。
"""
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from .gamestate import FIELD_LIMIT

Violation = Tuple[str, str]

_DON_RULE_RE = re.compile(r"ドン!!デッキは(\d+)枚")


def _expected_don_total(player) -> int:
    """そのプレイヤーのドン!!総数の期待値（保存量）。

    既定 10 枚だが、リーダーの常在ルール「ルール上、自分のドン!!デッキはN枚になる」（エネル OP15-058=6 等）が
    あればその枚数。`gamestate._apply_leader_don_deck_rule` と同じテキストを読む（NFKC で ‼/全角数字も吸収）。
    """
    leader = getattr(player, "leader", None)
    master = getattr(leader, "master", None) if leader else None
    text = getattr(master, "effect_text", "") if master else ""
    m = _DON_RULE_RE.search(unicodedata.normalize("NFKC", text or ""))
    return int(m.group(1)) if m else 10


def _all_card_zones(p) -> Dict[str, List]:
    """カードインスタンスを持つ全ゾーン（ドン!! は別系統なので除く）。"""
    zones = {
        "hand": list(p.hand),
        "field": list(p.field),
        "life": list(p.life),
        "trash": list(p.trash),
        "deck": list(p.deck),
        "temp_zone": list(getattr(p, "temp_zone", []) or []),
    }
    if p.leader:
        zones["leader"] = [p.leader]
    if p.stage:
        zones["stage"] = [p.stage]
    return zones


def check_invariants(manager) -> List[Violation]:
    """現在の盤面に対する不変条件違反を列挙する（空＝健全）。"""
    violations: List[Violation] = []

    # 対話が進行中は場の超過が一時的に許容される: _enforce_field_limit は
    # active_interaction がある間は起動を見送り（中断のネスト回避）、対話完了時に
    # resolve_interaction 末尾で強制トラッシュを立てるため。よって「対話が一切無いのに
    # 超過している」場合のみ真の違反とする。
    interaction_pending = manager.active_interaction is not None

    seen_uuids: Dict[str, str] = {}
    for p in (manager.p1, manager.p2):
        # --- 場のキャラ上限（対話進行中は超過を許容） ---
        if len(p.field) > FIELD_LIMIT and not interaction_pending:
            violations.append((
                "FIELD_LIMIT",
                f"{p.name}: field has {len(p.field)} characters (> {FIELD_LIMIT}) without overflow interaction",
            ))

        # --- ドン!! 総数保存（既定 10 枚・リーダールールで調整: エネル OP15-058=6 等） ---
        expected_don = _expected_don_total(p)
        don_total = (
            len(p.don_deck) + len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)
        )
        if don_total != expected_don:
            violations.append((
                "DON_CONSERVATION",
                f"{p.name}: total DON = {don_total} (expected {expected_don})",
            ))

        # --- 付与ドン!! 数が非負・整合 ---
        for c in p.field + ([p.leader] if p.leader else []):
            if getattr(c, "attached_don", 0) < 0:
                violations.append((
                    "NEGATIVE_DON",
                    f"{p.name}: {c.master.name} has negative attached_don ({c.attached_don})",
                ))

        # --- UUID ユニーク（同一カードが複数ゾーンに重複しない） ---
        for zone_name, cards in _all_card_zones(p).items():
            for c in cards:
                uid = getattr(c, "uuid", None)
                if uid is None:
                    continue
                if uid in seen_uuids:
                    violations.append((
                        "UUID_DUPLICATE",
                        f"card uuid {uid[:8]} ({c.master.name}) appears in both "
                        f"{seen_uuids[uid]} and {p.name}.{zone_name}",
                    ))
                else:
                    seen_uuids[uid] = f"{p.name}.{zone_name}"

    # --- 勝者整合: winner が立っているなら p1/p2 いずれかの名前 ---
    if manager.winner is not None and manager.winner not in (manager.p1.name, manager.p2.name):
        violations.append((
            "WINNER_INVALID",
            f"winner='{manager.winner}' is not a known player name",
        ))

    return violations


def check_turn_boundary(manager) -> List[Violation]:
    """ターン境界（end_turn 直後）で追加チェックする一時ゾーンのリーク検出。

    注意: ターン終了時トリガー（任意効果など）が `active_interaction` を立てたまま境界を
    跨ぐのは **正常**（次ステップで解決される）。よって対話/誘発が残ること自体は違反としない。
    一方、対話も誘発も無い「落ち着いた」状態なのに `temp_zone` にカードが残っているのは、
    効果解決が一時ゾーンを片付け損ねた真のリーク（`text_execution_audit` の
    FLAG_SUSPEND_LEAK 相当）として検出する。
    """
    violations: List[Violation] = []
    settled = manager.active_interaction is None and not getattr(manager, "_pending_triggers", None)
    if settled:
        for p in (manager.p1, manager.p2):
            if getattr(p, "temp_zone", None):
                violations.append((
                    "TEMP_ZONE_LEAK",
                    f"{p.name}: temp_zone not empty in settled state at turn boundary "
                    f"({len(p.temp_zone)} card(s))",
                ))
    return violations
