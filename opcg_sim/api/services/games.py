"""対局セットアップのサービス（先行プレイヤー解決ほか）。

対局生成本体（/api/game/create・/api/rule/action START）の共通化は C-5（routers 分割）で
route を薄くする際に扱う。ここでは純粋なヘルパーのみを持つ。
"""
import random
from typing import Any, Optional


def _resolve_first_player(value: Any, player1: Any, player2: Any) -> Optional[Any]:
    """リクエストの first_player 指定を先行プレイヤーに解決する。
      "p1"/"p2" : 明示指定（ソロでプレイヤーが選択）
      "random"  : ランダム（CPU/対戦のコイントス用。結果は turn_info に反映される）
      その他/None: 従来通り既定（start_game 側で p1 先行）

    対戦 API は席名版（[`_resolve_first_player_seat`]）を使う。こちらは Python の `Player`
    を渡す旧経路の互換（型注釈は `Any`＝エンジンを import しない・計画 §16.3-6）。
    """
    if value == "random":
        return random.choice([player1, player2])
    if value == "p1":
        return player1
    if value == "p2":
        return player2
    return None


def _resolve_first_player_seat(value: Any) -> Optional[str]:
    """`_resolve_first_player` の席名版（Rust エンジン用・計画 §15.2）。

    戻り値は "p1"／"p2"／None（既定）。"random" は `random.choice` を **1 回** 消費する＝
    `_resolve_first_player` と乱数の消費が同じなので、種からの再現（`replay_runner`）が保たれる。
    """
    if value == "random":
        return random.choice(["p1", "p2"])
    if value in ("p1", "p2"):
        return value
    return None
