"""`theory_trace_build.py`（棋譜 JSON → カード画像を同梱した単体ビューアー）の算術と配線を固める。

押さえるのは 3 つ（画像の取得はネットワークが要るので、取得済みの縮小画像を置いた `--cache` で代える）:

1. **`card_ids`**: 棋譜に現れる全カード（リーダー・場・手札・ステージ・トラッシュの一番上）を重複なく拾う。
2. **`build_html`**: 雛形の 2 つの差し込み口に棋譜と画像表を入れる（`</` を逃がす・差し込み口が無ければ落ちる）。
   **本物の雛形（`docs/tools/theory_trace_viewer.html`）に差し込み口が在る**ことも確かめる。
3. **CLI**: `--cache` に在る画像だけで組み立て、取れなかった札を `missing` に出す。`--no-images` は画像表を `null` にする。

**基盤健全性ではない**（札の取りこぼしは人が目で確かめる盤面そのものを欠く）。必須側。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import theory_trace_build as TB  # noqa: E402


def _trace():
    side_a = {"leader": {"card_id": "L-1"}, "stage": {"card_id": "S-1"}, "field": [{"card_id": "F-1"}],
              "hand": [{"card_id": "H-1"}, {"card_id": "F-1"}], "trash_top": "T-1"}
    side_b = {"leader": {"card_id": "L-2"}, "stage": None, "field": [], "hand": [], "trash_top": None}
    return {"meta": {"seed_base": 1}, "games": [{"decisions": [{"board": {"p1": side_a, "p2": side_b}},
                                                               {"board": {"p1": side_b, "p2": side_a}}]}]}


def test_card_ids_collects_every_zone_without_duplicates():
    assert TB.card_ids(_trace()) == ["F-1", "H-1", "L-1", "L-2", "S-1", "T-1"]
    assert TB.card_ids({}) == []


def test_build_html_fills_both_slots_and_escapes_closing_tags():
    tpl = f"<p>{TB.TRACE_SLOT}</p><p>{TB.IMAGE_SLOT}</p>"
    out = TB.build_html(tpl, {"x": "</script>"}, {"A": "data:image/webp;base64,AA"})
    assert "<\\/script>" in out and "</script>" not in out
    assert '"A":"data:image/webp;base64,AA"' in out
    assert TB.build_html(tpl, {"x": 1}, None).endswith("<p>null</p>")
    with pytest.raises(ValueError):
        TB.build_html("<p>no slots</p>", {}, None)


def test_the_real_template_has_both_slots():
    with open(TB.TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    assert TB.TRACE_SLOT in tpl and TB.IMAGE_SLOT in tpl


def test_cli_uses_the_cache_and_reports_missing_cards(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("OPCG_back", "DON", "L-1", "F-1"):
        (cache / f"{name}.webp").write_bytes(b"RIFF")
    monkeypatch.setattr(TB.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(_trace()), encoding="utf-8")
    tpl = tmp_path / "tpl.html"
    tpl.write_text(f"{TB.TRACE_SLOT}|{TB.IMAGE_SLOT}", encoding="utf-8")
    out = tmp_path / "v.html"
    assert TB.main(["--trace", str(trace), "--out", str(out), "--template", str(tpl), "--cache", str(cache)]) == 0
    trace_part, image_part = out.read_text(encoding="utf-8").split("|")
    images = json.loads(image_part)
    assert set(images) == {"__back", "__don", "L-1", "F-1"}
    assert images["L-1"].startswith("data:image/webp;base64,")
    assert json.loads(trace_part)["meta"]["seed_base"] == 1
    assert TB.main(["--trace", str(trace), "--out", str(out), "--template", str(tpl), "--no-images"]) == 0
    assert out.read_text(encoding="utf-8").endswith("|null")
