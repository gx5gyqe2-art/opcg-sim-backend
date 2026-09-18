"""**規則から出る補充 `r`**——耐久 `Θ` は毎ターン「引いた 1 枚」ぶん補充される。T91・2026-09-18・読み取り専用。

**問い**（ユーザ指摘 2026-09-18「穴の大きさを測るのは CPU の打ち方によるんじゃない？」）:
T90 は動く的の下がる速さ `r` に**帳簿の `g`**（＝相手が**実際にどう打ったか**の記録から出た 1 枚あたりの価格）を
使っていた。これは**打ち筋を式に入れている**＝記録を取り直せば別の値になる。**式に入れてよいのは規則とデッキの中身だけ**。

**分け方**:

| | 誰が決めるか | `r` に入れてよいか |
|---|---|---|
| 毎ターン 1 枚引く・ドンが 1 枚増える | **規則** | ○ |
| 引いた 1 枚が**切れる札**である確率 | **デッキの中身**（デッキ表から数えられる） | ○ |
| その札を手札に残すか出すか | **打ち方** | **×** |

3 番目は `Θ` の**中の引っ越し**（手札の項 `μ` ↔ 体の項 `ν`）であって `Θ` の増減ではない＝`r` には入らない。
`Θ` の手札項は既定 `cuttable`（T77・**切れる札だけが `μ` を持つ**）なので:

```
r = μ × （そのデッキの切れる札の割合）            # 新定数ゼロ・記録も打ち方も見ない
```

**デッキは seed から決定論で作れる**（`decks.build_pair`・`deck_profile` と同じ経路）ので、記録の
`meta_games.json`（`decks` のモードと各局の `leaders`・`seed`）だけで席ごとの `r` が引ける。

**切れる札の定義は符号化に合わせる**——`Θ` の手札項が読んでいるのは `hand_guard.counter_of`
＝**印字カウンター ＋ カウンターイベントの上げ幅**なので、デッキ側も
「印字カウンター > 0 **または** 【カウンター】能力を持つイベント」を切れる札として数える。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/deck_refill.py --in ~/w41 --out ~/deck_refill.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned import n_rel_feat as NF  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402
from theory_order import MU  # noqa: E402

_DB = {}
_SHARE = {}          # (leader_id, tuple(deck_ids)) は重いので id 列の署名でキャッシュ
_PAIR = {}           # (decks_mode, seed, la, lb) -> (share_p1, share_p2)


def db():
    if "db" not in _DB:
        _DB["db"] = D.load_db()
    return _DB["db"]


def is_cuttable(m):
    """その札が**切られうる**か。**符号化と同じ式**を使う（`n_rel_feat` の `counter_value` 欄・
    印字カウンター > 0 **または**【カウンター】でパワーを上げるイベント）＝`Θ` の手札項が読んでいる物と一致する。
    `m` はカード原本（`CardLoader.get_card`）。"""
    if float(getattr(m, "counter", 0) or 0) > 0:
        return True
    return float(NF.profile(m).get("counter_event") or 0.0) > 0.0


def cut_share(deck_ids):
    """デッキ（card_id の列・50 枚）の**切れる札の割合**。引けなかった札は分母から外す。"""
    key = tuple(deck_ids)
    if key in _SHARE:
        return _SHARE[key]
    d = db()
    n = c = 0
    for cid in deck_ids:
        m = d.get_card(cid)
        if m is None:
            continue
        n += 1
        c += 1 if is_cuttable(m) else 0
    out = (float(c) / n) if n else 0.0
    _SHARE[key] = out
    return out


def pair_shares(seed, decks_mode, leaders=(None, None)):
    """1 局の**両席の切れる札の割合** `(p1, p2)`（デッキは seed から決定論で作り直す）。"""
    la, lb = (leaders or (None, None))[:2]
    key = (decks_mode, int(seed), la, lb)
    if key in _PAIR:
        return _PAIR[key]
    (_l1, d1), (_l2, d2) = D.build_pair(db(), la, lb, int(seed), decks_mode)
    out = (cut_share(d1), cut_share(d2))
    _PAIR[key] = out
    return out


def r_of(share, mu=MU):
    """**補充**＝`μ ×（切れる札の割合）`（1 守備ターンあたり・引き 1 枚ぶん）。"""
    return float(mu) * float(share)


def shares_by_seed(dirs):
    """記録のディレクトリ群 → `{seed: (share_p1, share_p2)}`（`meta_games.json` を読んで作り直す）。

    `who=0`＝p1・`who=1`＝p2（記録の規約）。同じ seed が複数のディレクトリに在れば先勝ち。"""
    out = {}
    for d in dirs:
        p = os.path.join(d, "meta_games.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            meta = json.load(fh)
        mode = str(meta.get("decks") or "singleton")
        for g in meta.get("games", ()):
            s = int(g.get("seed"))
            if s in out:
                continue
            try:
                out[s] = pair_shares(s, mode, tuple(g.get("leaders") or (None, None)))
            except Exception:                                # noqa: BLE001
                continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", nargs="+", required=True, help="記録のディレクトリ（複数可）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    sh = shares_by_seed(a.inp)
    vals = np.array([v for pair in sh.values() for v in pair], float)
    out = {"dirs": list(a.inp), "games": len(sh), "seats": int(vals.size),
           "cut_share": {"mean": round(float(vals.mean()), 4) if vals.size else None,
                         "p10": round(float(np.percentile(vals, 10)), 4) if vals.size else None,
                         "p50": round(float(np.percentile(vals, 50)), 4) if vals.size else None,
                         "p90": round(float(np.percentile(vals, 90)), 4) if vals.size else None},
           "r": {"mu": MU,
                 "mean": round(float(r_of(vals.mean())), 5) if vals.size else None,
                 "p10": round(float(r_of(np.percentile(vals, 10))), 5) if vals.size else None,
                 "p90": round(float(r_of(np.percentile(vals, 90))), 5) if vals.size else None},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
