"""フロント采取ログ（人間 vs CPU）→ 価値学習データ JSONL 変換（オフライン・dev専用・stdlib-only）。

フロントの「采取」ボタンが出す JSON（`replay` descriptor を内包）から、バックエンドが**ライブ採取**して
勝者でラベル確定した `value_samples`（`{"f":[...],"y":0/1}`）を取り出し、`train_value`/`eval_value_on_set` が
読む JSONL（1行1サンプル）へ書き出す。リプレイ再現は不要（特徴は実対局の生盤面でサーバ側計算済み）。

入力は单一 JSON でもディレクトリ（*.json を走査）でも可。`value_samples` は采取エンベロープのどの階層に
あっても拾える（envelope→replay→descriptor のいずれか）。終局していない采取は value_samples が空＝0行。

実行例:
    python tests/human_log_ingest.py --in capture1.json capture2.json --out /tmp/human_value.jsonl
    python tests/human_log_ingest.py --in ./captures/ --out /tmp/human_value.jsonl
"""
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features


def _find_value_samples(obj: Any, depth: int = 0) -> List[Dict[str, Any]]:
    """采取エンベロープから最初に見つかる value_samples（list）を返す（浅い再帰・envelope/replay/descriptor 対応）。"""
    if depth > 4 or not isinstance(obj, dict):
        return []
    vs = obj.get("value_samples")
    if isinstance(vs, list):
        return vs
    for key in ("replay", "data", "result"):
        found = _find_value_samples(obj.get(key), depth + 1)
        if found:
            return found
    return []


def _valid_row(r: Any) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("f"), list)
            and len(r["f"]) == cpu_features.N_FEATURES and r.get("y") in (0, 1))


def rows_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        dump = json.load(f)
    return [{"f": [float(v) for v in r["f"]], "y": int(r["y"])}
            for r in _find_value_samples(dump) if _valid_row(r)]


def _expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="フロント采取ログ→価値学習 JSONL")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="采取 JSON ファイル または ディレクトリ（複数可）")
    ap.add_argument("--out", default="/tmp/human_value.jsonl")
    ap.add_argument("--append", action="store_true", help="既存 out に追記")
    args = ap.parse_args(argv)

    paths = _expand_inputs(args.inputs)
    if not paths:
        print("入力なし（ファイル/ディレクトリを確認）"); return 1

    n_files = n_rows = n_empty = 0
    with open(args.out, "a" if args.append else "w", encoding="utf-8") as out:
        for p in paths:
            try:
                rows = rows_from_file(p)
            except (OSError, ValueError) as e:
                print(f"  skip {p}: {e}"); continue
            n_files += 1
            if not rows:
                n_empty += 1
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"done: {n_files} files ({n_empty} に有効サンプルなし) → {n_rows} rows → {args.out} "
          f"(features={cpu_features.N_FEATURES})")
    if n_rows == 0:
        print("注意: 0 行。终局済みの cpu_trace 対局を采取したか確認（未決着は value_samples が空）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
