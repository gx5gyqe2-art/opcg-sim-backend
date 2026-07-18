"""ラベル量産ワーカーの seed 割当て（`label_worker.next_seed0`・純関数）。

w1 運用報告（2026-07-18）の実バグの回帰: 旧式 `base + batch_id × games` は `--games` を
途中で変えると過去 seed 帯を再割当てし、重複対局（＝重複教師・計算の無駄）を生んだ。
累計局数ベースなら games 設定の変更を跨いでも連続・無重複。基盤健全性＝cpu_infra。
"""
import os
import sys

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
from label_worker import next_seed0  # noqa: E402

pytestmark = pytest.mark.cpu_infra

BASE = 20_000_000


def test_games_change_does_not_reuse_seed_band():
    """16局×7バッチ後に --games 4 へ変更しても、過去帯（+0〜111）と重複しない。"""
    metas = [{"games": 16} for _ in range(7)]
    s = next_seed0(BASE, metas, batch_id=7, games=4)
    assert s == BASE + 112               # 旧式なら BASE+28＝batch1 帯の再割当てだった
    metas.append({"games": 4})
    assert next_seed0(BASE, metas, batch_id=8, games=4) == BASE + 116


def test_constant_games_matches_legacy():
    """--games 一定なら旧式と同じ値（既存ワーカーの連番と互換）。"""
    metas = [{"games": 16} for _ in range(5)]
    assert next_seed0(BASE, metas, batch_id=5, games=16) == BASE + 5 * 16


def test_missing_games_falls_back_to_legacy():
    """メタ欠損（games 無し）は旧式へフォールバック（黙って0扱いにしない）。"""
    metas = [{"games": 16}, {}]
    assert next_seed0(BASE, metas, batch_id=2, games=16) == BASE + 32
