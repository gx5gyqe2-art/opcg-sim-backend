"""フラッグシップバトル結果集計（flagship）ドメイン。

ゲーム本体の API とは独立した小さなドメイン（設計は flagship リポジトリの
docs/design.md §12）。ルーターは `router.py`、永続化は `db.py`（SQLite・遅延初期化）、
入出力モデルは `schemas.py`（contract/ のラチェット対象外）に分離する。
"""
