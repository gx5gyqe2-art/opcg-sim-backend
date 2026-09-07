"""`opcg_sim.loop.deck_synth` の別名（正本はそちら・2026-09-07 計画 §17 の移動）。

デッキ生成は「学習ループのオーケストレーション」なので `opcg_sim/loop/` が置き場になった。
旧ラインの実験/計測 CLI（`tests/scripts/` の多数）は `import deck_synth` で引いているので、
**モジュールそのもの**を差し替えて名前を保つ（`from X import *` では私有名が落ちる）。
"""
import sys

from opcg_sim.loop import deck_synth as _mod

sys.modules[__name__] = _mod
