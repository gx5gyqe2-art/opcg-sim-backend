"""API スキーマ契約の生成（`contract/api_schema.json` + `contract/manifest.json`）。

pydantic モデル（`api/schemas.py`）から JSON Schema を生成し、フロントの型生成の入力にする。
生成物はコミットし、CI で「再生成して差分ゼロ」を検証する（スキーマを変えて export を忘れた
PR を落とすラチェット）。google.cloud 不要（schemas は config/enums のみに依存）。

使い方:
    python -m opcg_sim.tools.export_contract
"""
import os
import json
import hashlib

from opcg_sim.api import schemas as S
from opcg_sim.src.utils.shared_constants import constants_hash, load_shared_constants

# 契約に含める公開モデル（フロントの受信型生成の対象）。
_MODELS = [
    "CardSchema", "ZoneSchema", "PlayerSchema", "BattleStateSchema",
    "GameStateSchema", "PendingRequestSchema", "GameActionResultSchema",
    "BattleActionRequest",
]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # リポジトリルート
_OUT_DIR = os.path.join(_ROOT, "contract")


def build_schema() -> dict:
    """公開モデルの JSON Schema を1つの契約ドキュメントへ集約する。"""
    models = {name: getattr(S, name).model_json_schema(by_alias=True) for name in _MODELS}
    return {"schema_version": "opcg-api/v1", "models": models}


def _canon(obj) -> str:
    # 決定的出力（CI ラチェットのため sort_keys 固定）。
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def main() -> None:
    # 共有定数が読めない環境では、schemas の別名がフォールバック既定に化け、constants_sha256 が
    # 空 dict のハッシュになる＝「もっともらしく間違った契約」を成功終了で書いてしまう。明示的に落とす。
    if not load_shared_constants():
        raise SystemExit(
            "shared_constants.json を読めませんでした（契約が誤生成されるため中断）。"
            "作業ディレクトリ／パッケージ配置を確認してください。"
        )
    os.makedirs(_OUT_DIR, exist_ok=True)
    schema_text = _canon(build_schema())
    schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:12]
    manifest = {"constants_sha256": constants_hash(), "schema_sha256": schema_hash}

    with open(os.path.join(_OUT_DIR, "api_schema.json"), "w", encoding="utf-8") as f:
        f.write(schema_text + "\n")
    with open(os.path.join(_OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(_canon(manifest) + "\n")
    print(f"wrote contract/ (schema_sha256={schema_hash}, constants_sha256={manifest['constants_sha256']})")


if __name__ == "__main__":
    main()
