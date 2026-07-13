"""マーク局面シード（v5 §4-2・cpu_v5_plan.md）: 人間が悪手をマークした実対局の**失敗局面**を、
自己対戦の開始局面プールとして復元する。

狙い: 観測された失敗モードそのものを in-distribution 化する＝net が「その盤面から先」を実際に
プレイして学ぶ機会を作る（C2/D1 付与の程度・C5 資源・守りすぎ等は、自己対戦の turn1 開始分布には
稀にしか現れない）。開始局面を差し替えるだけで、以降の軌跡・ラベル（勝敗/q_root/turns_left）は
通常の自己対戦と同じ経路で採れる。

対象は **MAIN 手番マークのみ**（フレーム i-1 から盤面を静的復元＝乱数不要・決定論）。カウンター/
戦闘マーク（SELECT_COUNTER/BLOCKER・PASS）は戦闘途中の再開が近似的で扱いにくいため除外する。
復元は replay_reeval の静的フレーム復元を流用（同一ソース＝盤面近似の限界も replay_reeval と同じ）。
"""
import glob
import os

from replay_reeval import load_replay_json, _board_from_frame
from cpu_selfplay import _load_db

_FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "replays")
_SKIP_AT = {"SELECT_COUNTER", "SELECT_BLOCKER", "PASS"}


def _fixture_paths():
    """マーク付きリプレイ fixtures（決定論順＝sorted）。"""
    return sorted(glob.glob(os.path.join(_FIX, "*.json.gz")))


def load_mark_boards(db=None, fixtures=None):
    """マーク fixtures の MAIN 手番マーク盤面を復元し、プレイ可能な GameManager のリストで返す。

    決定論（fixtures は sorted・各 fixture 内はマーク順）＝ワーカー間・再実行間で同一プール。
    復元失敗・終局・合法手ゼロの盤面は落とす（自己対戦の開始点として不適格なもの）。
    """
    db = db or _load_db()
    paths = fixtures if fixtures is not None else _fixture_paths()
    boards = []
    for path in paths:
        raw = load_replay_json(path)
        rec = raw.get("replay", raw)
        frames = raw.get("frames") or []
        marks = raw.get("marks") or []
        fbi = {f.get("action_index"): f for f in frames}
        actions = rec.get("actions") or []
        for mk in marks:
            i = mk.get("action_index")
            if not isinstance(i, int) or not (0 <= i < len(actions)):
                continue
            if actions[i].get("action_type") in _SKIP_AT:
                continue
            pre = fbi.get(i - 1)
            if pre is None:
                continue
            try:
                m = _board_from_frame(db, rec, pre, actions[i]["player"])
                if m.winner is None and m.get_legal_actions(m.turn_player):
                    boards.append(m)
            except Exception:
                continue      # 近似復元の限界＝落とすだけ（生成を止めない）
    return boards
