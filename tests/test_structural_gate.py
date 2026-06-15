"""構造不変条件ゲート — 横展開4スキャンのラチェット（カテゴリH 再発防止）。

`tests/structural_invariants.py` の4検出器を pytest から実行し、いずれも **0 件** で固定する。
カテゴリH（`docs/reports/quality_postmortem_categoryH.md`）の見逃しは、ベースラインが latent bug を
凍結し、オラクルが別軸を測っていたため検出できなかった。本ゲートは *条件スコープ／期間／選択者／
全体性* という従来死角だった軸を構造不変条件として常設し、同種バグの再混入を機械的に検出する。

新たに 0 を超えたら、真のバグかを精査し、カードを直してから 0 に戻すこと（誤検知なら検出器側の
除外条件を明示して調整する）。
"""
import os

import conftest  # noqa: F401

import structural_invariants as si
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data", "opcg_cards.json")

# 各カテゴリのラチェット上限（現状すべて 0。引き上げは退行）。
MAX = {
    "H_LEADING_GATE_LEAK": 0,
    "DURATION_WRITEOFF": 0,
    "CHOOSER_MISSING": 0,
    "SUBETE_COUNT_DEGRADE": 0,
}


def _findings():
    db = CardLoader(DATA)
    db.load()
    return si.scan(db)


def test_structural_invariants_are_zero():
    findings = _findings()
    for cat, limit in MAX.items():
        items = findings.get(cat, [])
        assert len(items) <= limit, (
            f"{cat} {len(items)} > {limit}: "
            + ", ".join(f"{cid}/{trig}" for cid, trig, _ in items[:10]))


def test_false_path_no_leak():
    """条件“偽”パスの被覆（§6.2）: ability レベルのゲート条件を偽にして発動しても
    実盤面が一切動かないことを動的に確認する。漏れ（H 類型の退行）があれば顕在化する。

    ベースラインが“真パス”しか踏まない死角の逆側を埋める標準観点（docs/TEST_SPEC.md §8.4）。"""
    import false_path_coverage as fp
    buckets = fp.collect()
    leaks = buckets.get("FALSE_PATH_LEAK", [])
    errs = buckets.get("ERROR", [])
    assert not leaks, (
        f"FALSE_PATH_LEAK {len(leaks)}: "
        + ", ".join(f"{cid}/{trig}" for cid, _, trig, _ in leaks[:10]))
    assert not errs, (
        f"偽パス実行で例外 {len(errs)}: "
        + ", ".join(f"{cid}/{trig}" for cid, _, trig, _ in errs[:10]))
    # 取りこぼし検出（測定が機能している担保）: 相当数が実際にゲート検証されている。
    assert len(buckets.get("GATED_OK", [])) >= 500
