"""人間ログ（フロント采取）→ 価値モデル学習のワンショット・パイプライン（dev専用・stdlib-only）。

`tests/human_captures/` に貯めた采取 JSON を一括で:
  ① ingest  : `human_log_ingest.py` でラベル付き特徴 JSONL に変換
  ② train   : `train_value.py` で**候補モデル**を学習（既定では同梱 value_model.json は上書きしない）
  ③ eval    : `eval_value_on_set.py` で候補モデル vs 同梱モデルの汎化を同一データで比較

安全方針: 本スクリプトは**学習データを集約して候補モデルを作り、数値を見せるだけ**。本番同梱の
`opcg_sim/src/core/value_model.json` の差し替え（昇格）と `OPCG_VALUE_BLEND_*` の有効化は、Elo 検証を
経た上での**別の明示操作**に委ねる（詳細は docs/human_log_collection.md）。

実行例:
    OPCG_LOG_SILENT=1 python tests/human_value_pipeline.py
    OPCG_LOG_SILENT=1 python tests/human_value_pipeline.py --captures tests/human_captures --epochs 400
"""
import argparse
import os
import subprocess
import sys

import conftest  # noqa: F401  (tests/ を import path に載せる)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.join(_REPO, "tests")
_DEFAULT_CAPTURES = os.path.join(_TESTS, "human_captures")
_DATA_OUT = os.path.join(_TESTS, "human_value.jsonl")            # 集約データ（dev中間物）
_CANDIDATE = os.path.join(_TESTS, "human_value_model.candidate.json")  # 候補モデル（非同梱）


def _run(argv):
    """同じ Python で tests 配下のツールを呼ぶ。失敗時はそのまま終了コードを返す。"""
    print(f"$ {' '.join(argv)}")
    return subprocess.run([sys.executable, *argv], cwd=_REPO).returncode


def _count_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description="人間ログ→価値モデル ワンショット")
    ap.add_argument("--captures", default=_DEFAULT_CAPTURES,
                    help="采取 JSON のディレクトリ（既定 tests/human_captures）")
    ap.add_argument("--data", default=_DATA_OUT, help="集約 JSONL の出力先")
    ap.add_argument("--candidate", default=_CANDIDATE, help="候補モデルの出力先（非同梱）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--min-rows", type=int, default=50, help="学習に必要な最小行数")
    args = ap.parse_args(argv)

    # ① ingest
    rc = _run(["tests/human_log_ingest.py", "--in", args.captures, "--out", args.data])
    if rc != 0:
        return rc
    rows = _count_rows(args.data)
    print(f"\n[ingest] 集約 {rows} 行 → {os.path.relpath(args.data, _REPO)}")

    if rows < args.min_rows:
        print(f"\n[stop] 学習には >= {args.min_rows} 行が必要（現在 {rows}）。"
              f"采取 JSON を {os.path.relpath(args.captures, _REPO)}/ に追加して再実行してください。")
        return 0

    # ② train（候補モデルへ。同梱 value_model.json は触らない）
    print()
    rc = _run(["tests/train_value.py", "--data", args.data,
               "--epochs", str(args.epochs), "--out", args.candidate])
    if rc != 0:
        return rc

    # ③ eval（候補 vs 同梱・同一データ）
    print("\n[eval] 候補モデル:")
    _run(["tests/eval_value_on_set.py", "--data", args.data, "--model", args.candidate])
    print("\n[eval] 同梱 value_model.json:")
    _run(["tests/eval_value_on_set.py", "--data", args.data])

    print(f"\n[done] 候補モデル → {os.path.relpath(args.candidate, _REPO)}（非同梱）。"
          "\n       昇格（本番同梱の差し替え・OPCG_VALUE_BLEND_* 有効化）は Elo 検証後の別操作。"
          "\n       手順は docs/human_log_collection.md を参照。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
