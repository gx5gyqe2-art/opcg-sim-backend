"""エンジン層の葉ヘルパ（NFC 正規化・ターン制限判定・能力インデックス）。

gamestate.py と engine/* の双方から使う純粋関数。依存は stdlib + models.enums のみ
（循環回避のための葉モジュール）。gamestate.py はこれらを後方互換で再エクスポートする。
"""
import re
import unicodedata
from typing import Optional, Any

from ...models.enums import ConditionType


def _nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


# 【ターン1回】系の表記（置換/保護能力は parser が TURN_LIMIT 条件を落とすため raw_text からも拾う）。
_TURN1_RE = re.compile(r'ターン1回|ターンに1回|1ターンに1回')


def _condition_turn_limit(cond) -> Optional[int]:
    """条件ツリー中の TURN_LIMIT 上限値を返す（無ければ None）。AND/OR の入れ子も探索。"""
    if cond is None:
        return None
    if cond.type == ConditionType.TURN_LIMIT:
        v = cond.value
        return v if isinstance(v, int) and v > 0 else 1
    if cond.type in (ConditionType.AND, ConditionType.OR):
        for a in (cond.args or []):
            r = _condition_turn_limit(a)
            if r is not None:
                return r
    return None


def _ability_turn_limit(ab) -> Optional[int]:
    """能力の per-turn 使用上限。条件の TURN_LIMIT を優先し、無ければ raw_text の【ターン1回】表記から 1。

    置換/保護能力（_active_replacement / _active_protection 経由）は parser が TURN_LIMIT を
    ab.condition へ載せない（自己置換は final_condition=None）ため、raw_text を併用する。
    """
    lim = _condition_turn_limit(getattr(ab, "condition", None))
    if lim is not None:
        return lim
    if _TURN1_RE.search(_nfc(getattr(ab, "raw_text", "") or "")):
        return 1
    return None


def _ability_index(card, ab) -> Any:
    """ability_used_this_turn のキー（master.abilities 内の位置。resolver._ability_key と整合）。"""
    abilities = getattr(getattr(card, "master", None), "abilities", ()) or ()
    for i, a in enumerate(abilities):
        if a is ab:
            return i
    return id(ab)
