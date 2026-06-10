"""全カード横断検証の回帰ガード（トラックB）。

全2652枚の全能力を汎用盤面で発動し、カード保全に関する構造不変条件を固定する:
  - 例外（クラッシュ）= 0
  - カード消失（CARD_LOSS）= 0
  - TEMP リーク（解決完了後の temp 残留）= 0

意味的正しさではなく「カードが消えない/落ちない」を全カードで保証する回帰テスト。
ここが赤になったら、ある効果がカードを temp に取り残す/消失させる退行が入った合図。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_full_card_audit.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

from full_card_audit import audit


def test_all_cards_no_structural_anomaly():
    anomalies = audit()
    by_kind = {}
    for a in anomalies:
        by_kind.setdefault(a.kind, []).append(a)

    # 退行時に原因が分かるよう、種別ごとに最初の数件を添えて落とす。
    def _fmt(kind):
        items = by_kind.get(kind, [])
        sample = "; ".join(f"{a.card_id}/{a.trigger}({a.detail})" for a in items[:8])
        return f"{kind}={len(items)} [{sample}]"

    assert not anomalies, (
        "全カード横断検証で構造異常を検出:\n  "
        + "\n  ".join(_fmt(k) for k in ("EXCEPTION", "CARD_LOSS", "TEMP_LEAK") if by_kind.get(k))
    )
