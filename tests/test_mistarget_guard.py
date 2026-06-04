"""隠れミスターゲット／lift 不具合の回帰ガード。

`mistarget_diagnostics.scan()` の detector A/B は、PHoSv ブランチの修正で 0 になった。
これらは「実行はされるが盤面操作が誤る」OTHER 指標に出ない不具合であり、
compare_parsers（新規OTHER検知）では捕捉できないため、専用の回帰ガードで 0 を固定する。

- A. PLAY_CARD zone=FIELD            … 場からは登場できない＝ミスターゲット → 0 を維持
- B. REVEALED_CARD_TRAIT lift(公開消失) … 公開(LOOK)消失＋順序矛盾 → 0 を維持

C/D は段階的に burn down する想定のため、ここでは「ベースラインから増えていないこと」を
緩く確認する（上限はベースライン実測値）。値を下げたら上限も更新すること。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from mistarget_diagnostics import KEY_A, KEY_B, KEY_C, KEY_D, scan


def _unique_cards(hits, key):
    return {h[0] for h in hits.get(key, [])}


def test_no_play_card_field_mistarget():
    """A: PLAY_CARD が zone=FIELD に誤ターゲットするカードが無いこと。"""
    _, hits = scan()
    offenders = _unique_cards(hits, KEY_A)
    assert offenders == set(), f"PLAY_CARD zone=FIELD ミスターゲットが復活: {sorted(offenders)}"


def test_no_revealed_card_trait_lift():
    """B: REVEALED_CARD_TRAIT がアビリティ条件へ lift（公開消失）していないこと。"""
    _, hits = scan()
    offenders = _unique_cards(hits, KEY_B)
    assert offenders == set(), f"REVEALED_CARD_TRAIT lift が復活: {sorted(offenders)}"


def test_cd_detectors_do_not_grow():
    """C/D: 段階的に減らす想定。ベースライン実測値を超えて増えていないこと。"""
    _, hits = scan()
    # ベースライン（PHoSv ラウンド3 時点の実測。減ったら下げること）
    assert len(_unique_cards(hits, KEY_C)) <= 8
    assert len(_unique_cards(hits, KEY_D)) <= 56
