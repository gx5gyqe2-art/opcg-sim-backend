"""価値学習データの『ターン境界サンプリング』共通ロジック（自己対戦・実対局で同一）。

`tests/collect_value_data.py`（オフライン自己対戦）と `api/app.py`（実対局のライブ採取）が、
**同じ観測点（ターン境界）・同じラベル規約（終局の勝者視点=1）・同じ特徴**を使うための単一情報源。
特徴抽出は `cpu_features.extract_features`（既定 `see_opp_hand=False`＝公平＝相手手札の中身を読まない）。
すべて読み取り専用（manager を変更しない）・stdlib-only。
"""
from typing import Any, Dict, List

from . import cpu_features


def turn_boundary_samples(manager) -> List[Dict[str, Any]]:
    """現盤面（ターン境界）の両プレイヤー視点の特徴を `{"f":[...], "p":<name>}` で返す（manager 非破壊）。

    プレイヤー識別はプレイヤー**名**（`manager.p1.name`/`manager.p2.name`）で行う＝実対局のカスタム名にも
    追従する（`manager.winner` も名前なのでラベル付けと整合）。
    """
    return [{"f": cpu_features.extract_features(manager, p.name), "p": p.name}
            for p in (manager.p1, manager.p2)]


def label_samples(samples: List[Dict[str, Any]], winner) -> List[Dict[str, Any]]:
    """`{"f","p"}` サンプル列を終局 `winner`（プレイヤー名）で `{"f","y"}` に確定する（勝者視点なら y=1）。"""
    return [{"f": s["f"], "y": 1 if winner == s["p"] else 0} for s in samples]
