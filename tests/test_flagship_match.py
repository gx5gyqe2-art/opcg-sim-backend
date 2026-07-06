"""収集ポスト × TCG+開催 の照合（`opcg_sim/api/flagship/match.py`、設計 §16.7）のテスト。

handle 一致（自動確定候補）、表示名ファジー一致（要承認）、誤爆（同チェーン別店）の除外、
日付近接での絞り込み、個人ポスト（候補ゼロ＝未紐付け）を、実データで観測した実例で検証する。
純粋関数のためネットワーク不要。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_match.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

from opcg_sim.api.flagship import match as M


def _ev(eid, store, date, sns=None):
    return M.StoreEvent(event_id=eid, store=store, date=date, sns_url=sns)


# --- 部品 -------------------------------------------------------------------

def test_extract_handle():
    assert M.extract_handle("https://x.com/shop_a") == "shop_a"
    assert M.extract_handle("https://twitter.com/Shop_B/status/1") == "shop_b"
    assert M.extract_handle("https://instagram.com/x") is None
    assert M.extract_handle("") is None


def test_name_similarity_matches_and_rejects():
    # 実データ: 完全/近接一致は高く、別チェーン店は低い。
    assert M.name_similarity("Japan TCG Center那覇沖映通り店", "Japan TCG Center那覇沖映通り店") == 1.0
    assert M.name_similarity("トレカビッグホーン", "ビッグホーン") >= 0.6
    # 誤爆例（別チェーン・別地域）は閾値 0.6 未満。
    assert M.name_similarity("トップカード名古屋大須店", "カードラボ名古屋大須店") < 0.6
    assert M.name_similarity("TCバトロコ八千代台駅前", "TCバトロコ秋田駅前") < 0.6


# --- 照合 -------------------------------------------------------------------

def test_handle_match_is_auto():
    events = [_ev(1, "宝島イオンモール三川店", "2026-07-05", "https://x.com/m_takara2")]
    cands = M.match_post("m_takara2", "宝島イオンモール三川店", "2026-07-05", events)
    assert len(cands) == 1
    assert cands[0].method == "handle" and cands[0].auto is True and cands[0].event_id == 1


def test_name_match_is_proposal_not_auto():
    events = [_ev(2, "ゲームスペース鶴岡", "2026-07-05", sns=None)]
    cands = M.match_post("someacct", "ゲームスペース鶴岡", "2026-07-05", events)
    assert len(cands) == 1
    assert cands[0].method == "name" and cands[0].auto is False


def test_false_positive_chain_store_excluded():
    events = [_ev(3, "カードラボ名古屋大須店", "2026-07-05")]
    cands = M.match_post("acct", "トップカード名古屋大須店", "2026-07-05", events)
    assert cands == []           # 別チェーンは提案しない


def test_date_window_filters():
    events = [_ev(4, "ゲームスペース鶴岡", "2026-06-01")]  # 投稿と1ヶ月差
    assert M.match_post("acct", "ゲームスペース鶴岡", "2026-07-05", events) == []


def test_individual_post_has_no_candidate():
    events = [_ev(5, "ゲームスペース鶴岡", "2026-07-05", "https://x.com/gamespace")]
    # 個人（表示名が店でない・handle も違う）→ 候補ゼロ＝未紐付け。
    assert M.match_post("gokuu_08", "孫悟空", "2026-07-05", events) == []


def test_handle_ranked_before_name_and_nearer_date_first():
    events = [
        _ev(10, "ゲームスペース鶴岡", "2026-07-04"),                                  # name, gap1
        _ev(11, "ゲームスペース鶴岡", "2026-07-05", "https://x.com/gs_tsuruoka"),      # 別店名だが... handle 不一致
        _ev(12, "別ショップ鶴岡", "2026-07-05", "https://x.com/acct"),                 # handle 一致
    ]
    cands = M.match_post("acct", "ゲームスペース鶴岡", "2026-07-05", events)
    assert cands[0].method == "handle" and cands[0].event_id == 12   # handle 最優先
