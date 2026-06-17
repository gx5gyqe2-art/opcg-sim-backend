"""カードを事前パースして pickle キャッシュを生成する（起動高速化用）。

docker build 内で実行し、image に `opcg_cards.cache.pkl` を焼き込む。起動時は
`CardLoader.load_cache()` がこれを採用し、コールドスタート時の全件パース(~1.8s)を
回避する。キャッシュには生成元 json のハッシュが同梱され、起動時に現行 json と
照合される（不一致なら採用せずフルパースに安全劣化する）。

使い方:
    python -m opcg_sim.tools.build_card_cache
"""
import os

from opcg_sim.src.utils.loader import CardLoader

# このファイル: opcg_sim/tools/build_card_cache.py -> data は opcg_sim/data
_DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
CARD_DB_PATH = os.path.join(_DATA_DIR, "opcg_cards.json")


def main() -> str:
    db = CardLoader(CARD_DB_PATH)
    db.load()
    n = db.parse_all()
    path = db.save_cache()
    print(f"[build_card_cache] parsed {n} cards -> {path} (hash={db.db_hash()[:8]})")
    return path


if __name__ == "__main__":
    main()
