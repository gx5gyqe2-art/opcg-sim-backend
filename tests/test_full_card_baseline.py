"""全カード挙動ベースライン回帰（トラックB）。

全2652枚の各能力の実行シグネチャ（ゾーン差分 + power/keyword/rest/cost 差分、または
INTERACTIVE/ERROR）を `full_card_baseline.json` に凍結し、現在の挙動と比較する。
どのカードの挙動が変わってもここで検出される＝全カードの回帰網。

意図的に挙動を変えた（バグ修正・新ルール）場合は、差分を確認した上でベースラインを再生成:
    OPCG_LOG_SILENT=1 python tests/full_card_audit.py --regen

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_full_card_baseline.py -q -s -p no:cacheprovider
"""
import json
import os

import conftest  # noqa: F401

from full_card_audit import BASELINE, signatures


def test_behavior_matches_baseline():
    assert os.path.exists(BASELINE), (
        "ベースライン未生成。`python tests/full_card_audit.py --regen` で生成してください。"
    )
    with open(BASELINE, encoding="utf-8") as f:
        base = json.load(f)
    now = signatures()

    changed = []
    for key, sig in now.items():
        if key in base and base[key] != sig:
            changed.append(f"{key}: '{base[key]}' → '{sig}'")
    missing = sorted(set(base) - set(now))
    added = sorted(set(now) - set(base))

    problems = []
    if changed:
        problems.append(f"挙動変化 {len(changed)} 件:\n  " + "\n  ".join(changed[:25]))
    if missing:
        problems.append(f"消失キー {len(missing)} 件: {missing[:15]}")
    if added:
        problems.append(f"新規キー {len(added)} 件: {added[:15]}（新カード追加時は --regen）")

    assert not problems, (
        "全カード挙動ベースラインと差分あり。意図的なら "
        "`python tests/full_card_audit.py --regen` で更新。\n" + "\n".join(problems)
    )
