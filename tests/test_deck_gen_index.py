"""デッキ生成用オフラインインデックス（deck_gen_index）の検証。

v4b 生成器の部品が壊れていないことを CI で保証: Trait両側・機序ペア・役割クラスタ・
コアパッケージ機械導出（held-out 実デッキとの照合は「構造導出が実デッキを覆うか」の
一方向確認＝許可用途(b)。実リストを生成に使うわけではない）。
"""
import conftest  # noqa: F401
import deck_gen_index as DGI
import heldout_decks as HD
from cpu_selfplay import _load_db

_CACHE = {}


def _built():
    if not _CACHE:
        db = _load_db()
        tm, tr, mech, pool = DGI.build_indexes(db)
        _CACHE.update(db=db, tm=tm, tr=tr, mech=mech, pool=pool)
    return _CACHE


def test_indexes_populated():
    c = _built()
    assert len(c["pool"]) > 2000, "デッキ投入可プールが少なすぎ"
    assert len(c["tm"]) > 100, "trait members が少なすぎ"
    assert len(c["tr"]) > 30, "trait referencers が少なすぎ"
    for k, v in c["mech"].items():
        assert len(v) > 0, f"機序ペア {k} が空（パース形の変化を疑う）"


def test_trait_bilateral_blackbeard():
    c = _built()
    t = "黒ひげ海賊団"
    assert len(c["tm"].get(t, [])) >= 5, "黒ひげ海賊団の保持者が引けない"
    assert len(c["tr"].get(t, [])) >= 1, "黒ひげ海賊団の参照者が引けない"


def test_leader_core_is_color_legal_and_recovers_real_deck():
    c = _built()
    db = c["db"]
    core = DGI.leader_core_candidates(db, "OP16-080", c["tm"], c["tr"], c["mech"])
    assert len(core) >= 10
    lcol = {getattr(x, "value", x) for x in db.get_card("OP16-080").colors}
    for cid in core:
        ccol = {getattr(x, "value", x) for x in db.get_card(cid).colors}
        assert lcol & ccol, f"コア候補 {cid} が色不一致"
    # 一方向確認: DB構造だけの導出が held-out 黒ひげ実デッキの過半近くを再発見できる
    bb = set(HD.all_lists()["blackbeard_black_yellow"].keys())
    overlap = len(bb & set(core))
    assert overlap >= 7, f"実デッキ再発見が弱すぎ: {overlap}/{len(bb)}"


def test_role_clusters_partition_pool():
    c = _built()
    clusters = DGI.build_role_clusters(c["db"], c["pool"], k=8, iters=8, seed=0)
    total = sum(len(v) for v in clusters.values())
    assert total == len(c["pool"]), "クラスタがプールを分割していない"
    assert len(clusters) >= 4, "クラスタが縮退"
