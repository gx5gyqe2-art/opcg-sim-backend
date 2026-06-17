"""パース済みカードキャッシュ（ビルド時 pickle）の健全性テスト。

- pickle 往復でカード内容（to_dict / abilities）が完全一致する＝挙動が変わらない
- 生成元 json と不整合なキャッシュは採用されない（古いデータを出さない＝安全劣化）

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_card_cache.py -q -s -p no:cacheprovider
"""
import json
import conftest  # noqa: F401  (sys.path 設定)

from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.tools.build_card_cache import CARD_DB_PATH


def _fresh_parsed():
    db = CardLoader(CARD_DB_PATH)
    db.load()
    db.parse_all()
    return db


def test_cache_roundtrip_is_identical(tmp_path):
    """キャッシュ採用後のカードは、フルパース結果と完全に一致する。"""
    fresh = _fresh_parsed()
    cache_path = str(tmp_path / "cards.cache.pkl")
    fresh.save_cache(cache_path)

    cached = CardLoader(CARD_DB_PATH)
    cached.load()
    assert cached.load_cache(cache_path) is True

    assert set(cached.cards.keys()) == set(fresh.cards.keys())
    assert len(cached.cards) == len(fresh.cards) > 0

    for cid, fm in fresh.cards.items():
        cm = cached.cards[cid]
        # データフィールドの一致
        assert cm.to_dict() == fm.to_dict()
        # パース済み能力の一致（数・内容）
        assert len(cm.abilities) == len(fm.abilities)
        assert repr(cm.abilities) == repr(fm.abilities)


def test_cache_rejected_on_hash_mismatch(tmp_path):
    """生成元と異なる json に対しては、整合しないキャッシュを採用しない。"""
    fresh = _fresh_parsed()
    cache_path = str(tmp_path / "cards.cache.pkl")
    fresh.save_cache(cache_path)

    # 別内容の json を指す CardLoader（db_hash が変わる）→ 採用されないこと
    other_json = tmp_path / "other_cards.json"
    other_json.write_text(json.dumps([{"number": "ZZZ-999", "name": "dummy"}]), encoding="utf-8")
    other = CardLoader(str(other_json))
    other.load()
    assert other.load_cache(cache_path) is False


def test_cache_absent_is_safe():
    """キャッシュ不在でも例外にならず False（=遅延パースに劣化）。"""
    db = CardLoader(CARD_DB_PATH)
    db.load()
    assert db.load_cache("/nonexistent/path/cards.cache.pkl") is False
