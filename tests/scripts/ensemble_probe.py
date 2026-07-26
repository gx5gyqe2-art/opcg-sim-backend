"""value アンサンブルの A/B プローブ（v15）: 2つの value を平均して葉評価に使い、既定と対戦させる。

背景（v9〜v14 の確定・`docs/reports/cpu_v13_v14_plateau_20260726.md`）: gen6 への微調整・容量拡大・
探索設定はすべて無効（対gen6 0.41-0.49）。診断で判明した弱点は **子盤面（兄弟）の順位判断が5割前後**
＝評価のばらつき。ばらつきを直接減らす手段としてアンサンブルは機構的に的を射ており、かつ**学習不要**
（既存ネットの組み合わせのみ）。gen6（実戦で較正・MAE 0.338）と scratch512（教師に精密・MAE 0.242）は
初期値も容量も違う＝誤差が独立に近く、平均で打ち消しが効くことを期待する。

葉評価のみ差し替え（policy prior・探索・root 読み出しは不変）。`predict`/`predict_with_aux` を
持つ薄いシムを `LearnedEngine.vnet` に差すだけで serve 経路に載る。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/ensemble_probe.py \
    --second /tmp/sc512/value.npz --weights 0.5,0.3 --pairs 30
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

_G = {}


class AvgValueNet:
    """2つの ValueNet の予測を重み平均する薄いシム（`_value_fn` が使う API のみ実装）。

    `w` は primary（既定=gen6）の重み。aux（残りターン補助）は primary のものを使う
    ——aux は粘り項のスケール決めにしか使われず、平均の意味が薄いため（挙動の説明性を優先）。
    符号化版・vocab は primary と同一である前提（呼び出し側で検査）。
    """

    def __init__(self, primary, second, w=0.5):
        self.a, self.b, self.w = primary, second, float(w)
        self.feat_dim = primary.feat_dim
        self.vocab_ids = primary.vocab_ids
        self.d_emb = primary.d_emb

    def predict(self, batch):
        return self.w * self.a.predict(batch) + (1.0 - self.w) * self.b.predict(batch)

    def predict_with_aux(self, batch):
        pa, aux = self.a.predict_with_aux(batch)
        pb = self.b.predict(batch)
        return self.w * pa + (1.0 - self.w) * pb, aux


def _init_pool(second_path, w):
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned.value_net import ValueNet
    _G["db"] = _load_db()
    ens = LearnedEngine()                      # 既定（gen6）ベース＝vocab/enc_version はこのまま
    second = ValueNet.load(second_path)
    if second.feat_dim != ens.vnet.feat_dim:
        raise SystemExit(f"符号化不一致: primary={ens.vnet.feat_dim} second={second.feat_dim}")
    ens.vnet = AvgValueNet(ens.vnet, second, w)
    _G["var"] = ens
    _G["base"] = LearnedEngine()


def _play_pair(seed):
    from cpu_arena import play_game
    a = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["var"], p2_engine=_G["base"])
    b = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["base"], p2_engine=_G["var"])
    return (1.0 if a["winner"] == "p1" else 0.0) + (1.0 if b["winner"] == "p2" else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--second", required=True, help="平均に加える value.npz（符号化版は既定と同一）")
    ap.add_argument("--weights", default="0.5", help="primary(gen6) 側の重み（カンマ区切りで複数）")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=64000)
    ap.add_argument("--workers", type=int, default=3)
    ARGS = ap.parse_args()

    seeds = list(range(ARGS.seed0, ARGS.seed0 + ARGS.pairs))
    rows = []
    for w in [float(x) for x in ARGS.weights.split(",")]:
        t0 = time.time()
        with mp.Pool(ARGS.workers, initializer=_init_pool, initargs=(ARGS.second, w)) as pool:
            wins = sum(pool.imap_unordered(_play_pair, seeds))
        games = 2 * len(seeds)
        wr = wins / games
        se = math.sqrt(max(wr * (1 - wr), 1e-9) / games)
        rows.append({"w_gen6": w, "wins": wins, "games": games, "wr": round(wr, 4),
                     "ci95": [round(wr - 1.96 * se, 3), round(wr + 1.96 * se, 3)],
                     "sec": int(time.time() - t0)})
        print(f"w(gen6)={w:<5} {wins:>5.1f}/{games}  wr={wr:.3f} "
              f"CI[{wr - 1.96 * se:.3f},{wr + 1.96 * se:.3f}]  ({time.time() - t0:.0f}s)", flush=True)
    print(f"ENSEMBLE_PROBE_RESULT {json.dumps(rows, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
