"""パース済みカードキャッシュ（ビルド時 pickle）の健全性テスト。

- pickle 往復でカード内容（to_dict / abilities）が完全一致する＝挙動が変わらない
- 生成元 json と不整合なキャッシュは採用されない（古いデータを出さない＝安全劣化）

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_card_cache.py -q -s -p no:cacheprovider
"""
import json
import conftest  # noqa: F401  (sys.path 設定)

from opcg_sim.src.utils.loader import CardLoader
from opcg_sim.tools.build_card_cache import CARD_DB_PATH


def _canon(x, _seen=None):
    """set/dict の順序に依存しない正準形（pickle 往復で set の repr 順が変わるフレーク対策）。

    IR の back-reference 等による循環を、パスごとの訪問集合 `_seen` で打ち切る（journal.deep_diff 同様）。
    """
    if _seen is None:
        _seen = frozenset()
    if isinstance(x, (set, frozenset)):
        return ("set", sorted(repr(e) for e in x))  # 要素は基本 str/enum＝repr で順序非依存
    if isinstance(x, dict):
        return ("dict", sorted((repr(k), _canon(v, _seen)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_canon(e, _seen) for e in x])
    d = getattr(x, "__dict__", None)
    if d is not None:
        if id(x) in _seen:
            return ("cycle", type(x).__name__)
        _seen = _seen | {id(x)}
        return (type(x).__name__, sorted((k, _canon(v, _seen)) for k, v in d.items()))
    return repr(x)


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
        # パース済み能力の一致（数・内容）。set フィールド（TargetQuery.flags 等）の repr 順は
        # pickle 往復で変わりフレークになるため、set/dict の順序に依存しない正準形で比較する。
        assert len(cm.abilities) == len(fm.abilities)
        assert _canon(list(cm.abilities)) == _canon(list(fm.abilities))


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
