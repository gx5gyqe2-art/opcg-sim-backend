"""v5 本走の温スタート種ネット生成（v5 §4-5）: 出荷 gen4（v3符号化）を v4 符号化へ温スタート拡張し、
value/policy の種 .npz を出力する。本走はこの種を checkpoint 枝（`p3ckpt/{value,policy}.npz` と
`gen0_value.npz`）に置いて起動する＝load_nets が cold-fallback（gen2 から v1→v4）でなく **gen4 の実力を
引き継ぐ**（§4-3補で恒等温スタートを数値確認済み）。

出力は「拡張直後＝gen4 と挙動恒等」（新スカラーが W1 のゼロ行に当たる）＝学習前は v4 と同じ手を指す。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/v5_seed_net.py --enc-version 4 --out /tmp/v5seed
    → /tmp/v5seed/value.npz, /tmp/v5seed/policy.npz を出力
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import warm_start_value, warm_start_policy, _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_seed(enc_version, gen_value=None, gen_policy=None):
    """gen4（既定）→ enc_version へ温スタートした (vnet, pnet, from_version) を返す。"""
    gv = gen_value or os.path.join(REPO, "opcg_sim", "data", "learned", "gen4_value.npz")
    gp = gen_policy or os.path.join(REPO, "opcg_sim", "data", "learned", "gen4_policy.npz")
    v = RN.ValueNet.load(gv)
    base_v = _net_enc_version(v)
    vnet = warm_start_value(v, base_v, enc_version)
    pnet = warm_start_policy(PolicyScorer.load(gp), base_v, enc_version) if os.path.exists(gp) else None
    return vnet, pnet, base_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc-version", type=int, required=True, choices=(2, 3, 4))
    ap.add_argument("--out", required=True, help="出力ディレクトリ（value.npz/policy.npz を書く）")
    ap.add_argument("--gen-value", default=None, help="種の value（既定 gen4_value.npz）")
    ap.add_argument("--gen-policy", default=None, help="種の policy（既定 gen4_policy.npz）")
    args = ap.parse_args()

    vnet, pnet, base_v = build_seed(args.enc_version, args.gen_value, args.gen_policy)
    os.makedirs(args.out, exist_ok=True)
    vnet.save(os.path.join(args.out, "value.npz"))
    if pnet is not None:
        pnet.save(os.path.join(args.out, "policy.npz"))

    # 恒等性の自己検証（1局面）: 種は gen4 と value/aux が一致すべき（学習前は挙動恒等）。
    import numpy as np
    from opcg_sim.src.core.gamestate import GameManager, Player
    from cpu_selfplay import build_deck
    import random
    random.seed(0)
    db = _load_db(); vocab = E.build_vocab(db)
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    gv = args.gen_value or os.path.join(REPO, "opcg_sim", "data", "learned", "gen4_value.npz")
    v0 = RN.ValueNet.load(gv)
    def b(enc): return {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
    e_old = E.encode(m, "p1", vocab, version=base_v)
    e_new = E.encode(m, "p1", vocab, version=args.enc_version)
    d = abs(float(v0.predict(b(e_old))[0]) - float(vnet.predict(b(e_new))[0]))
    print(f"v5 種ネット出力: {args.out}（gen4 v{base_v} → v{args.enc_version}・value 恒等誤差 Δ={d:.2e}）", flush=True)
    assert d < 1e-6, f"恒等温スタートが破れている（Δ={d}）"
    print("種を checkpoint 枝の p3ckpt/{value,policy}.npz と gen0_value.npz に配置して本走を起動してください。",
          flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
