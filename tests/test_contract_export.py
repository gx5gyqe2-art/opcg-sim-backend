"""契約生成物のラチェット。

`opcg_sim/tools/export_contract.py` を再生成した結果が、コミット済みの
`contract/api_schema.json` / `contract/manifest.json` と一致することを検証する。
スキーマ（api/schemas.py）や共有定数を変えて export を忘れた PR を落とす。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_contract_export.py -q -s -p no:cacheprovider
"""
import os
import json

import conftest  # noqa: F401  (sys.path 設定)

from opcg_sim.tools import export_contract as EC

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTRACT = os.path.join(_ROOT, "contract")


def test_api_schema_up_to_date():
    """再生成した api_schema がコミット済みと一致する（export 忘れ検出）。"""
    committed = open(os.path.join(_CONTRACT, "api_schema.json"), encoding="utf-8").read()
    regenerated = EC._canon(EC.build_schema()) + "\n"
    assert regenerated == committed, (
        "contract/api_schema.json が最新でない。`python -m opcg_sim.tools.export_contract` を実行して"
        "生成物をコミットしてください。"
    )


def test_manifest_hashes_match():
    """manifest の schema_sha256 が現行スキーマのハッシュと一致する（tool と同一算出）。"""
    import hashlib
    schema_text = EC._canon(EC.build_schema())
    expect = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:12]
    manifest = json.load(open(os.path.join(_CONTRACT, "manifest.json"), encoding="utf-8"))
    assert manifest["schema_sha256"] == expect, "manifest の schema_sha256 が api_schema と不一致（再 export が必要）"
