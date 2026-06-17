"""継続効果（期間付き効果）の管理（改善策④の continuous 版）。

「このバトル中」「このターン中」「次の相手のターン終了時まで」のように、
適用してから特定のタイミングで失効する効果を一元管理する。

設計（既存エンジンと衝突しない方針）:
  - 効果は CardInstance の *専用フィールド* `timed_power` / `timed_flags` に反映する。
    これらは `reset_turn_status()` でクリアされない（=ターン境界を跨いで存続できる）。
    既存の `power_buff` / `flags`（ターン境界でリセットされる）とは独立。
  - 失効は本マネージャの `expire(event)` を、バトル終了・ターン終了のフックで
    呼ぶことで行う。リセット後の再適用(reapply)が不要になり、二重適用を避けられる。

対応 kind:
  - "POWER": timed_power に加算（パワー増減）
  - "FLAG" : timed_flags に追加（例: ATTACK_DISABLE などの制限）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Duration 定数
THIS_TURN = "THIS_TURN"               # 現在のターン終了時に失効
THIS_BATTLE = "THIS_BATTLE"           # 現在のバトル解決時に失効
UNTIL_NEXT_TURN_END = "UNTIL_NEXT_TURN_END"  # 次のターン終了時に失効（複数ターン跨ぎ）
PERMANENT = "PERMANENT"               # ターン/バトルでは失効しない（場を離れるまで持続）

# expire() に渡すイベント
EV_TURN_END = "TURN_END"
EV_BATTLE_END = "BATTLE_END"


@dataclass
class ContinuousEffect:
    target_uuid: str
    kind: str          # "POWER" | "COST" | "FLAG" | "KEYWORD"
    amount: int = 0
    flag: str = ""
    keyword: str = ""
    duration: str = THIS_TURN
    expire_turn: int = 0  # UNTIL_NEXT_TURN_END 用: この turn_count の TURN_END で失効


class ContinuousEffectManager:
    def __init__(self, game_manager):
        self.gm = game_manager
        self.effects: List[ContinuousEffect] = []

    def apply(self, card, kind, duration, amount=0, flag="", keyword="", expire_turn=0) -> ContinuousEffect:
        # KEYWORD/FLAG は集合セマンティクス（重複付与は無意味）のため、同一内容の効果が
        # 既に生きていれば再登録しない。PASSIVE 再計算が同じ付与を繰り返すと effects
        # リストが際限なく成長するのを防ぐ（POWER/COST は正当な重ね掛けがあるため対象外）。
        if kind in ("KEYWORD", "FLAG"):
            for e in self.effects:
                if (e.target_uuid == card.uuid and e.kind == kind and e.flag == flag
                        and e.keyword == keyword and e.duration == duration
                        and e.expire_turn == expire_turn):
                    return e
        eff = ContinuousEffect(
            target_uuid=card.uuid,
            kind=kind,
            amount=amount,
            flag=flag,
            keyword=keyword,
            duration=duration,
            expire_turn=expire_turn,
        )
        self._apply_to_card(card, eff)
        self.effects.append(eff)
        return eff

    def _apply_to_card(self, card, eff: ContinuousEffect) -> None:
        if eff.kind == "POWER":
            card.timed_power += eff.amount
        elif eff.kind == "COST":
            card.timed_cost += eff.amount
        elif eff.kind == "FLAG":
            card.timed_flags.add(eff.flag)
        elif eff.kind == "KEYWORD":
            card.timed_keywords.add(eff.keyword)

    def _remove_from_card(self, card, eff: ContinuousEffect) -> None:
        if eff.kind == "POWER":
            card.timed_power -= eff.amount
        elif eff.kind == "COST":
            card.timed_cost -= eff.amount
        elif eff.kind == "FLAG":
            card.timed_flags.discard(eff.flag)
        elif eff.kind == "KEYWORD":
            card.timed_keywords.discard(eff.keyword)

    def _is_expired(self, eff: ContinuousEffect, event: str, turn_count: int) -> bool:
        if event == EV_BATTLE_END:
            return eff.duration == THIS_BATTLE
        if event == EV_TURN_END:
            if eff.duration == THIS_TURN:
                return True
            if eff.duration == UNTIL_NEXT_TURN_END:
                return turn_count >= eff.expire_turn
        return False

    def expire(self, event: str, turn_count: int) -> None:
        """指定イベント時点で失効する効果をカードから取り除く。"""
        remaining: List[ContinuousEffect] = []
        removed = 0
        for eff in self.effects:
            if self._is_expired(eff, event, turn_count):
                card = self.gm._find_card_by_uuid(eff.target_uuid)
                if card:
                    self._remove_from_card(card, eff)
                removed += 1
            else:
                remaining.append(eff)
        self.effects = remaining
        if removed:
            pass

    def drop_for(self, uuid: str) -> None:
        """カードが場を離れた等で、その uuid 宛ての継続効果を破棄する。"""
        kept = []
        for eff in self.effects:
            if eff.target_uuid == uuid:
                card = self.gm._find_card_by_uuid(uuid)
                if card:
                    self._remove_from_card(card, eff)
            else:
                kept.append(eff)
        self.effects = kept
