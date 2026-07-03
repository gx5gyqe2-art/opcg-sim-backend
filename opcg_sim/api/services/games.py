"""対局セットアップのサービス（先行プレイヤー解決ほか）。

対局生成本体（/api/game/create・/api/rule/action START）の共通化は C-5（routers 分割）で
route を薄くする際に扱う。ここでは純粋なヘルパーのみを持つ。
"""
import random
from typing import Any, Optional

from opcg_sim.src.core.gamestate import Player


def _resolve_first_player(value: Any, player1: Player, player2: Player) -> Optional[Player]:
    """リクエストの first_player 指定を先行 Player に解決する。
      "p1"/"p2" : 明示指定（ソロでプレイヤーが選択）
      "random"  : ランダム（CPU/対戦のコイントス用。結果は turn_info に反映される）
      その他/None: 従来通り既定（start_game 側で p1 先行）
    """
    if value == "random":
        return random.choice([player1, player2])
    if value == "p1":
        return player1
    if value == "p2":
        return player2
    return None
