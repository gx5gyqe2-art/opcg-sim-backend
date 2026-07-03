"""対局レジストリ（プロセス内メモリ状態）。

ルールモード対局・フリーモード・CPU 対戦メタ・ルールルームの4レジストリを1箇所へ集約する。
現状は Cloud Run 単一インスタンス前提のプロセス内 dict（SPEC 準拠）。外部ストア化・ロック導入は
本リファクタの非目標＝将来の差し替え点として本モジュールに集約するに留める。
"""
from typing import Any, Dict

from opcg_sim.src.core.gamestate import GameManager

# ルールモード対局本体（ソロ／オンライン／CPU 共通で GameManager を格納）。
GAMES: Dict[str, GameManager] = {}
# フリーモード（サンドボックス）の SandboxManager。
SANDBOX_GAMES: Dict[str, Any] = {}
# CPU 対戦のメタ情報: {game_id: {"cpu_player_id": "p2", "difficulty": "hard", ...}}。
# GAMES[game_id] に GameManager 本体を、ここに CPU 側の識別子と難易度を保持する。
CPU_GAMES: Dict[str, Dict[str, Any]] = {}
# ルールモード・オンライン対戦のルーム（ロビー）レジストリ。
# 各値: {game_id, room_name, created_at, status(WAITING/PLAYING/FINISHED),
#        ready{p1,p2}, decks{p1,p2:deck_id}, deck_preview{p1,p2:{leader_id,leader_name}}}
RULE_ROOMS: Dict[str, Dict[str, Any]] = {}


def clear_all() -> None:
    """全レジストリをクリアする（テストのセットアップ/ティアダウン用）。"""
    for reg in (GAMES, SANDBOX_GAMES, CPU_GAMES, RULE_ROOMS):
        reg.clear()
