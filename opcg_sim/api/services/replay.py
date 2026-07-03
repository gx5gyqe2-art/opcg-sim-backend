"""リプレイ種＋CPU思考トレース（実アプリ対局・opt-in）。

すべて opt-in（create リクエストの cpu_trace=true）でのみ作動し、未指定の本番対局には
一切の追加処理・レイテンシ・挙動変化を与えない（トレースは観測専用＝進行不変）。
スキーマ識別子 `REPLAY_SCHEMA` は config を参照。
"""
from typing import Any, Dict

from opcg_sim.src.core import cpu_ai


def _replay_enabled(meta) -> bool:
    return bool(meta and meta.get("cpu_trace"))


def _replay_record_action(meta, manager, src: str, player_id: str, movelike: Dict[str, Any]):
    """traced CPU 対局のアクションを card_id 基準で記録する（再現用・例外安全・適用前に呼ぶ）。"""
    if not _replay_enabled(meta):
        return
    try:
        desc = cpu_ai._describe_move(manager, movelike) or {"action_type": movelike.get("action_type")}
        meta.setdefault("actions", []).append(
            {"src": src, "turn": manager.turn_count, "player": player_id, **desc})
    except Exception:
        pass


def _capture_final_winner(meta, manager):
    """traced 対局の終局勝者を meta に保持する（WS 切断後の cleanup が manager を退避しても replay で参照可能）。

    opt-in（cpu_trace）時のみ作動＝本番対局にはオーバーヘッド・挙動変化なし。**アクション適用後**に呼ぶ。
    （旧 _capture_value_samples の価値学習データ採取は学習価値サブシステムごと撤去・2026-06-28。）
    """
    if not _replay_enabled(meta):
        return
    try:
        if manager.winner is not None:
            meta["_winner"] = manager.winner
    except Exception:
        pass
