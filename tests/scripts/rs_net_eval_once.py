"""rs_net_eval_once: npz 1 本を Rust に読ませて `net_eval` を 1 回だけ回す（照合用の下請け）。

**別プロセスで回すための器**。`opcg_engine` の既定ネット（`net_eval` が使うもの）は
**プロセスで最初に `load_net` したもの**に固定されるので、「同じプロセスで 2 本のネットを
比べる」ことができない。v13／v14 のように 2 本を比べるときは、この器を 2 回起動する。

入力はすべてファイル（JSON）で受け、結果を 1 行の JSON で標準出力に出す:

  {"summary": {…load_net の要約…}, "value": …, "priors": [...]}

実行例:
  python tests/scripts/rs_net_eval_once.py --net net.npz --tables tables.npz \\
      --enc enc.json --legal legal.json
"""
import argparse
import json
import sys

import os as _os, sys as _sys                                            # noqa: E401
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                        # noqa: E402,F401


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--enc", required=True, help="`encode::Encoding` と同じ鍵の JSON ファイル")
    ap.add_argument("--legal", default=None, help="候補の JSON ファイル（省略＝候補なし）")
    args = ap.parse_args(argv)

    import opcg_engine

    summary = json.loads(opcg_engine.load_net(args.net, args.tables))
    with open(args.enc) as f:
        enc = f.read()
    if args.legal:
        with open(args.legal) as f:
            legal = f.read()
    else:
        legal = json.dumps({"moves": []})
    out = json.loads(opcg_engine.net_eval(enc, legal))
    out["summary"] = summary
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
