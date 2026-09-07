"""GameManager から分割したステートレス・エンジン関数群（各関数の第1引数は gm）。

gamestate.py が同名の1行デリゲートを保持し公開APIを維持する（engine 同士の直接 import は
行わず、相互呼び出しは gm 経由）。詳細は docs/refactoring_gamestate.md §3。
"""
