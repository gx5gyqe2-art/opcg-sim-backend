"""フラッグシップ結果抽出（LLM不使用・辞書マッチング、設計 §13）のテスト。

エイリアス生成（137リーダーで正規名・短縮名が引ける）、順位パターン写像、色略称の
card_number 一意化、同名（クロコダイル等）の曖昧化、confidence の付与、正規化、
`/api/flagship/extract`・`/oembed` の API 契約を検証する。全処理はカードDBのみ依存で無料。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_extract.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import unicodedata

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api.flagship import extract as E
from opcg_sim.api.resources import card_db


def _leader_rows():
    rows = []
    for number, item in card_db.raw_db.items():
        norm = {unicodedata.normalize("NFC", str(k)): v for k, v in item.items()}
        if unicodedata.normalize("NFC", str(norm.get("種類", ""))) == unicodedata.normalize("NFC", "リーダー"):
            rows.append((number, str(norm.get("name", "")), str(norm.get("色", ""))))
    return rows


# --- エイリアス生成 ---------------------------------------------------------

def test_alias_index_covers_all_137_leaders():
    idx = E._index()
    rows = _leader_rows()
    assert len(rows) == 137
    # 全リーダーの正規名・短縮名が辞書から引け、その card_number を含む。
    for number, name, _color in rows:
        full = E._norm(name)
        short = E._norm(E._short_name(name))
        assert full in idx.aliases, f"正規名が引けない: {name}"
        assert number in idx.aliases[full]["numbers"]
        assert short in idx.aliases, f"短縮名が引けない: {name}"
        assert number in idx.aliases[short]["numbers"]


def test_color_prefix_two_colors_both_orders():
    # 2色は順序両方（黒黄 / 黄黒）を登録する。
    assert E._color_prefixes(["黒", "黄"]) == ["黒黄", "黄黒"]
    assert E._color_prefixes(["赤"]) == ["赤"]


# --- 正規化 -----------------------------------------------------------------

def test_normalization_strips_spaces_and_middle_dots_and_fullwidth():
    # NFKC で全角→半角、空白・中黒除去。
    assert E._norm("ロロノア・ゾロ") == E._norm("ロロノアゾロ")
    assert E._norm("赤 ゾロ") == E._norm("赤ゾロ")
    assert E._norm("ＢＥＳＴ４") == "BEST4"


# --- 抽出（順位×リーダー） --------------------------------------------------

def test_winner_with_color_shorthand_resolves_unique():
    entries, _ = E.extract_results("本日のフラッグシップ 優勝：赤ゾロ")
    assert len(entries) == 1
    e = entries[0]
    assert e.placement == 1
    assert e.card_number == "OP01-001"  # 赤 ロロノア・ゾロ
    assert e.confidence >= 0.9


def test_runner_up_placement_mapping():
    # 準優勝 は 優勝 を含むが placement=2 に正しく写像される。
    entries, _ = E.extract_results("優勝：赤ゾロ\n準優勝：黒黄ルフィ")
    by_p = {e.placement: e for e in entries}
    assert set(by_p) == {1, 2}
    assert by_p[1].card_number == "OP01-001"
    assert by_p[2].leader_raw  # 黒黄ルフィ が入る


def test_same_name_ambiguous_without_color():
    # 同名複数（クロコダイルは4枚）→ 色が無ければ card_number は一意化できず None。
    entries, _ = E.extract_results("優勝 クロコダイル")
    assert len(entries) == 1
    assert entries[0].card_number is None
    assert entries[0].leader_raw
    assert entries[0].confidence == pytest.approx(E._CONF_AMBIGUOUS)


def test_same_name_disambiguated_by_color():
    # 色プレフィックスで一意化。青クロコダイル → 単一 card_number。
    entries, _ = E.extract_results("優勝 青クロコダイル")
    assert len(entries) == 1
    assert entries[0].card_number is not None
    assert entries[0].confidence >= 0.9


def test_numeric_placement_pattern():
    entries, _ = E.extract_results("第3位 赤紫ロー")
    assert len(entries) == 1
    assert entries[0].placement == 3


def test_no_leader_text_returns_empty():
    # 画像のみ相当（テキストにリーダー名なし）→ 候補ゼロ。
    entries, unmatched = E.extract_results("本日開催しました！ご参加ありがとうございました")
    assert entries == []


def test_empty_text():
    assert E.extract_results("") == ([], [])
    assert E.extract_results("   ") == ([], [])


def test_no_marker_falls_back_to_winner():
    # 順位語が無くてもリーダー名があれば優勝(1)候補を1件出す。
    entries, _ = E.extract_results("赤ゾロ でした")
    assert len(entries) == 1
    assert entries[0].placement == 1


# --- 精度: 賞品告知はマーカーにしない（設計 §16.5） --------------------------

def test_prize_mention_does_not_override_winner():
    # 「優勝景品はルフィ」は賞品告知でマーカーにしない → 優勝は赤ゾロのまま。
    entries, _ = E.extract_results("優勝：赤ゾロ 準優勝：青クロコダイル 優勝景品はモンキー・D・ルフィ")
    by = {e.placement: e for e in entries}
    assert by[1].card_number == "OP01-001"          # 赤ゾロ（景品ルフィに奪われない）
    assert by[2].card_number == "ST03-001"          # 青クロコダイル


def test_prize_word_is_not_a_win_marker():
    # 「優勝景品」の優勝はマーカー非成立（準優勝は別扱い）。
    hits = [m.group(0) for m in E._MARKER_RE.finditer("優勝景品プレゼント")]
    assert all(not h.startswith("優勝") for h in hits)


# --- API 契約 ---------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(A.app) as c:
        yield c


def test_extract_endpoint(client):
    res = client.post("/api/flagship/extract", json={"text": "優勝：赤ゾロ 準優勝：青クロコダイル"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 2
    winner = body["results"][0]
    assert winner["placement"] == 1
    assert winner["leader_card_number"] == "OP01-001"
    # 一意化できたリーダーは辞書情報が付く。
    assert winner["leader"]["name"] == "ロロノア・ゾロ"


def test_extract_endpoint_empty(client):
    res = client.post("/api/flagship/extract", json={"text": "テキストなし"})
    assert res.status_code == 200
    assert res.json()["results"] == []


def test_oembed_bad_url(client):
    assert client.get("/api/flagship/oembed", params={"url": "not-a-url"}).status_code == 400
