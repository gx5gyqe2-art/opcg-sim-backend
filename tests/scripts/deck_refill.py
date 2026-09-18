"""**毎ターン引く 1 枚がもたらすもの**——デッキの中身と規則だけから出す「流入」。T91／T93・2026-09-18・読み取り専用。

2 つ在る（どちらも**記録も打ち方も読まない**）:

* **`r`（T91・守る側）**＝耐久 `Θ` の補充＝`μ ×（切れる札の割合）`
* **`a`（T93・攻める側）**＝速さ `A` の流入＝`E_デッキ[出せる体 1 枚の攻撃の価格]`

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
import theory_order as TO  # noqa: E402
from price_realised import nu_meas_of  # noqa: E402
from theory_order import MU, THETA, attack_value_don  # noqa: E402

_DB = {}
_SHARE = {}          # (leader_id, tuple(deck_ids)) は重いので id 列の署名でキャッシュ
_PAIR = {}           # (decks_mode, seed, la, lb) -> (share_p1, share_p2)
_DECKS = {}          # (decks_mode, seed, la, lb) -> (deck_ids_p1, deck_ids_p2)
_FLOW = {}           # (deck_ids, 相手リーダー, ドンの枠) -> 流入する速さ `a`
_EFF = {}            # (deck_ids, R, 自リーダー, ドンの枠) -> 効果が出す損害 `e`（T105）


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


def pair_decks(seed, decks_mode, leaders=(None, None)):
    """1 局の**両席のデッキ**（card_id の列・seed から決定論で作り直す）。"""
    la, lb = (leaders or (None, None))[:2]
    key = (decks_mode, int(seed), la, lb)
    if key in _DECKS:
        return _DECKS[key]
    (_l1, d1), (_l2, d2) = D.build_pair(db(), la, lb, int(seed), decks_mode)
    out = (tuple(d1), tuple(d2))
    _DECKS[key] = out
    return out


def pair_shares(seed, decks_mode, leaders=(None, None)):
    """1 局の**両席の切れる札の割合** `(p1, p2)`（デッキは seed から決定論で作り直す）。"""
    key = (decks_mode, int(seed), (leaders or (None, None))[0], (leaders or (None, None))[1])
    if key in _PAIR:
        return _PAIR[key]
    d1, d2 = pair_decks(seed, decks_mode, leaders)
    out = (cut_share(d1), cut_share(d2))
    _PAIR[key] = out
    return out


def r_of(share, mu=MU):
    """**補充**＝`μ ×（切れる札の割合）`（1 守備ターンあたり・引き 1 枚ぶん）。"""
    return float(mu) * float(share)


def body_of(m):
    """その札は**場に出て殴れる体**か（キャラでパワー > 0）。イベント・ステージ・リーダーは違う。"""
    if getattr(getattr(m, "type", None), "name", "") != "CHARACTER":
        return False
    return float(getattr(m, "power", 0) or 0) > 0.0


def a_of(deck_ids, opp_leader_power, don=None, theta=THETA, mu=MU, rush_only=False):
    """**流入する速さ `a`**（T93）＝**引いた 1 枚がもたらす攻撃の価格の期待値**（デッキ平均）。

    ```
    a = (1/N) Σ_{札 ∈ デッキ}  attack_value_don(パワー, 相手リーダー, リーダー狙い)
    ```

    体でない札（イベント・ステージ）は 0。`don` を渡すと**そのドンで出せない札**（コスト > ドン）も 0
    ＝規則の枠（ドンは毎ターン +1・上限 10）で絞る。**新定数ゼロ**（攻撃の価格は `attack_value_don`・
    残りはデッキの中身）。**打ち方は入らない**——どの札を選ぶかではなく**山の平均**を取る。

    **T103**: `rush_only=True` なら**速攻の札だけ**を数える。速攻は**引いたターンからもう殴れる**ので、
    歩き（`rate_at`）では 1 ターン早く積む（規則・`play_starts_next_turn` が帳簿側で既に使っている例外）。
    """
    olp = float(opp_leader_power)
    cap = None if don is None else int(round(float(don)))
    key = (tuple(deck_ids), round(olp, 1), cap, bool(rush_only))
    if key in _FLOW:
        return _FLOW[key]
    d = db()
    n = 0
    tot = 0.0
    for cid in deck_ids:
        m = d.get_card(cid)
        if m is None:
            continue
        n += 1
        if not body_of(m):
            continue
        if cap is not None and int(getattr(m, "cost", 0) or 0) > cap:
            continue
        # **T103**: `rush_only` なら**速攻の札だけ**（引いたターンからもう殴れる＝1 ターン早い）。
        if rush_only and "速攻" not in (getattr(m, "keywords", ()) or ()):
            continue
        tot += float(attack_value_don(float(getattr(m, "power", 0) or 0), olp, True, theta, mu))
    out = (tot / n) if n else 0.0
    _FLOW[key] = out
    return out


def e_of(deck_ids, my_leader_power=5000.0, r_turns=3, don=None, boards=None):
    """**引いた 1 枚が出す「効果の損害」の期待値**（T105・デッキ平均）。

    **問い**（T103 の速さの検算）: **実際に打った攻撃は 1 ターンの損害の 79〜92% しか説明しない**。
    残り 8〜21% は**効果が出した損害**（KO・除去）で、速さ `A` には 1 項も入っていなかった。

    ```
    e = (1/N) Σ_{札 ∈ デッキ}  max_{その札の除去能力}  E_盤面[ max ν(倒せる体) ]
    ```

    * **除去能力としきい値は原本から読む**（`n_rel_feat.profile` の `thr`＝相手を対象にした
      KO／バウンス／レスト系と `power_max`）。**パーサの出力であって打ち方ではない**。
    * **倒せる体の損害は `ν_meas`**（`price_realised`・`Θ` の体の項と同じ式）。
    * **盤面は測った分布**（`theory_order.load_opp_boards`・T46 の `OPTION_MODE=dist` と同じ資産）。
      **ここだけが記録由来**で、他は全部デッキ表と規則。
    * **1 枚は 1 回しか使えない**ので、これは**流量**（毎ターン 1 枚引く ＝ 毎ターン `e` ずつ）であって
      積み上がらない——体の攻撃（`a_of`）が**毎ターン殴り続ける**のと役割が違う。

    `don` を渡すとそのドンで出せない札は 0（`a_of` と同じ規則の枠）。
    """
    rb = int(max(1, min(5, round(float(r_turns)))))
    bs = (TO.load_opp_boards() if boards is None else boards).get(rb) or []
    if not bs:
        return 0.0
    mlp = float(my_leader_power)
    cap = None if don is None else int(round(float(don)))
    key = (tuple(deck_ids), rb, round(mlp, 1), cap)
    if boards is None and key in _EFF:
        return _EFF[key]
    d = db()
    n = 0
    tot = 0.0
    for cid in deck_ids:
        m = d.get_card(cid)
        if m is None:
            continue
        n += 1
        if cap is not None and int(getattr(m, "cost", 0) or 0) > cap:
            continue
        tot += removal_harm(m, mlp, bs)
    out = (tot / n) if n else 0.0
    if boards is None:
        _EFF[key] = out
    return out


def removal_harm(m, my_leader_power, boards):
    """その札 1 枚が**相手から奪える体の損害**（`ν_meas`）の期待値。除去能力が無ければ 0。

    しきい値（`power_max`）に合う体だけが対象。**1 枚で 1 体**（複数体を取る能力も 1 体ぶんで数える＝
    過小側に倒す）。**コストのしきい値（`cost_max`）は盤面の分布がコストを持たないので見ない**（限界）。"""
    thr = [t for t in (NF.profile(m).get("thr") or ()) if len(t) >= 4 and t[3] == "removal"]
    if not thr:
        return 0.0
    mlp = float(my_leader_power)
    best = 0.0
    for t in thr:
        pmax = t[0]
        tot = 0.0
        for _rec_mlp, bodies in boards:
            v = 0.0
            for tp, _blk in bodies:
                if pmax is not None and float(tp) > float(pmax) + 1e-6:
                    continue
                v = max(v, float(nu_meas_of(float(tp), mlp)))
            tot += v
        best = max(best, tot / len(boards))
    return float(best)


def _by_seed(dirs, fn):
    """記録のディレクトリ群 → `{seed: fn(seed, mode, leaders)}`（`meta_games.json` を読んで作り直す）。

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
                out[s] = fn(s, mode, tuple(g.get("leaders") or (None, None)))
            except Exception:                                # noqa: BLE001
                continue
    return out


def shares_by_seed(dirs):
    """記録のディレクトリ群 → `{seed: (切れる札の割合 p1, p2)}`（T91 の `r` の材料）。"""
    return _by_seed(dirs, pair_shares)


def decks_by_seed(dirs):
    """記録のディレクトリ群 → `{seed: (デッキ p1, デッキ p2)}`（T93 の流入 `a` の材料）。"""
    return _by_seed(dirs, pair_decks)


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
