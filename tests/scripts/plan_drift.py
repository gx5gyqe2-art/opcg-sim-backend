"""自己対戦の「方針」は続いているか——ターンごとの方針ラベルと、その切り替えと勝敗の関係。

ユーザの仮説（2026-09-12）: π が平らな局面は「どちらの方針でも進行が別物になる分岐」であり、
ネットは状態を持たないので、片方を選んだ後に他方の方針で打つ（ブレる）と弱くなる。
既存の dump（v14）の `sig`（打った手）・`pol_*`（候補と訪問）・`who`／`turn`／`seed`／`z` だけで
測る。ネットは読まない（読み取り専用の計器）。

**方針ラベル（自席ターンごと・粗い 5 値）**——その手番の main 決定の中身から:
  face    … リーダーへの攻撃あり・キャラ攻撃も除去効果も無し
  board   … キャラへの攻撃 or 除去効果（`deck_roles.classify` の型を持つ PLAY／ACTIVATE_MAIN）
            あり・リーダー攻撃なし
  mixed   … 両方あり
  develop … 攻撃も除去も無く PLAY あり
  pass    … 何もせず TURN_END
攻撃の対象は候補列（`pol_sig`×`pol_cid`/`pol_tcid`）から uuid→カード ID を引いて
リーダー／キャラを判別する（対局内で一度も候補に出ない uuid は unknown＝ラベルに使わない）。

出すもの:
  1. ラベルの分布・遷移行列（自席ターン t→t+1）・ターン帯別
  2. 席ごとの切り替え率（`switch`＝連続する自席ターンでラベルが変わった割合・`fb_switch`＝
     face↔board の反転）を**勝者と敗者で比較**。同じ対局の両席を対にして「敗者の方が多く
     切り替えた割合」（対局の長さは両席で同じ＝長さの交絡を消す）。序盤（t≤6）だけの版も出す
     （負けが決まってから打ち方が変わる＝逆因果を弱める）。
  3. **分岐点の粘り**: main 行で face 系と board 系の候補が両方あり、訪問が拮抗
     （pi_2/pi_1 ≥ --fork-ratio）な局面を「分岐」とし、打った手の方針が**直前の自席ターンの
     方針と同じ**だった割合（sticky）。状態を持たないなら基準率（その方針の出現率）と変わらない
     はず。勝者／敗者・教師（argmax pi）でも出す。
  4. don_waste … 自席ターンに DON を付けたのに攻撃しなかったカードの数（ブレの実体の一例）
  5. lean／drift … ラベルの 5 値だと mixed（リーダー攻撃とキャラ攻撃の同居）が中盤の半分を
     占めて face/board の対比が薄れるので、ターンの**傾き** lean＝リーダー攻撃 /（リーダー攻撃＋
     キャラ攻撃＋除去）（攻撃も除去も無いターンは欠損）を連続値で持ち、drift＝連続する自席
     ターンの |lean の差| の平均を勝者／敗者で比べる。分岐点の「前ターンの方針」も lean で判定
     （>0.5＝face・<0.5＝board・0.5 ちょうどと欠損は判定なし）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/plan_drift.py \\
    --in ~/n32_wave/w01/n_records/* ~/n32_wave/w02/n_records/* --out ~/pd_w32.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.learned.train import dump_io as DIO  # noqa: E402
from opcg_sim.loop import deck_roles as DR_ROLES  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402

LABELS = ("face", "board", "mixed", "develop", "pass")
_ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "forced", "pol_len", "pol_chosen")
_POL_COLS = ("pol_sig", "pol_cid", "pol_tcid", "pol_n")


def _turn_band(t):
    return "t1-4" if t <= 4 else "t5-8" if t <= 8 else "t9+"


def _mean_ci(xs):
    xs = np.asarray(xs, np.float64)
    if len(xs) == 0:
        return {"n": 0}
    m = float(xs.mean())
    se = float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return {"n": int(len(xs)), "mean": m, "ci95": [m - 1.96 * se, m + 1.96 * se]}


def _prop_ci(k, n):
    if n == 0:
        return {"n": 0}
    p = k / n
    se = float(np.sqrt(p * (1 - p) / n))
    return {"n": int(n), "p": float(p), "ci95": [p - 1.96 * se, p + 1.96 * se]}


class _Cards:
    """カード ID → リーダーか／除去の型を持つか（マスター単位でキャッシュ）。"""

    def __init__(self, db):
        self.db = db
        self._t = {}

    def info(self, cid):
        if cid not in self._t:
            m = self.db.get_card(cid) if cid else None
            if m is None:
                self._t[cid] = None
            else:
                forms = {k.split(":", 1)[0] for k in DR_ROLES.classify(m)}
                self._t[cid] = {"leader": getattr(getattr(m, "type", None), "name", "") == "LEADER",
                                "removal": bool(forms & set(DR_ROLES.FORMS))}
        return self._t[cid]


def _cls(sj, u2c, cards):
    """1 手（move_sig の JSON）→ 方針クラス（face/board/develop/end/don/None・unknown）。"""
    at = sj[0]
    if at in ("ATTACK", "DON_BOX"):
        tg = sj[2][0] if sj[2] else None
        if tg is None:
            return "don" if at == "DON_BOX" else None
        inf = cards.info(u2c.get(tg))
        if inf is None:
            return "unknown"
        return "face" if inf["leader"] else "board"
    if at in ("PLAY", "ACTIVATE_MAIN"):
        inf = cards.info(u2c.get(sj[1]))
        if inf is not None and inf["removal"]:
            return "board"
        return "develop"
    if at == "TURN_END":
        return "end"
    return None


def _label(c):
    """自席ターンのクラス集計 → 方針ラベル。"""
    board = c["board"]
    if c["face"] and board:
        return "mixed"
    if c["face"]:
        return "face"
    if board:
        return "board"
    if c["develop"]:
        return "develop"
    return "pass"


def _lean(c):
    """ターンの傾き＝リーダー攻撃 /（リーダー攻撃＋キャラ攻撃＋除去）。攻撃も除去も無ければ None。"""
    tot = c["face"] + c["board"]
    return (c["face"] / tot) if tot else None


def _iter_shards(dirs, row_cols, pol_cols, extra_fn=None):
    """シャード npz を 1 本ずつ開いて (rows, pol, extra) を返す（**1 波を丸ごと載せない**）。

    波 28／30／31 を一括で読むと 14GB を超えて OOM した（2026-09-12 実測）。シャードは対局単位で
    切られている（`record_gen` は `shard_games` 局ごとに丸ごと書く）ので、1 本ずつで集計できる。
    """
    files = [f for d in dirs for f in DIO.shard_files(d)]
    if not files:
        raise SystemExit("シャードが無い")
    for f in files:
        with np.load(f, allow_pickle=True) as dd:
            n = int(dd["z"].shape[0])
            rows = {k: (np.asarray(dd[k])[:n] if k in dd.files else np.zeros(n, np.int64)) for k in row_cols}
            pol = {k: np.asarray(dd[k]) for k in pol_cols}
            extra = extra_fn(dd, n) if extra_fn else None
        yield rows, pol, extra


def _iter_games(dirs, row_cols, pol_cols, extra_fn=None):
    """対局ごとに (rows, pol, extra, L, ptr, idx) を返す。idx は手順どおりの行 index。"""
    for rows, pol, extra in _iter_shards(dirs, row_cols, pol_cols, extra_fn):
        L = rows["pol_len"].astype(np.int64)
        ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
        order = np.lexsort((rows["step"], rows["seed"]))
        seeds = rows["seed"][order]
        bounds = np.flatnonzero(np.diff(seeds)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(seeds)]])
        for s_, e_ in zip(starts, ends):
            yield rows, pol, extra, L, ptr, order[s_:e_]


def analyze(dirs, fork_ratio=0.5, early_turn=6, limit_games=0):
    t0 = time.time()
    cards = _Cards(D.load_db())
    n_rows = 0

    per_seat = []                       # 1 席 1 局 = 1 レコード
    trans = collections.Counter()       # (prev, cur) → n
    trans_band = collections.Counter()  # (band, prev, cur)
    label_band = collections.Counter()  # (band, label)
    forks = []                          # 分岐点レコード
    unknown_targets = 0
    games = 0
    for rows, pol, _x, L, ptr, idx in _iter_games(dirs, _ROW_COLS, _POL_COLS):
        games += 1
        if limit_games and games > limit_games:
            break
        n_rows += len(idx)
        seed = int(rows["seed"][idx[0]])
        # uuid → カード ID（対局内の全候補から）
        u2c = {}
        for i in idx:
            k = int(L[i])
            if k <= 0:
                continue
            for j in range(ptr[i], ptr[i] + k):
                sj = json.loads(pol["pol_sig"][j])
                if sj[1] and pol["pol_cid"][j]:
                    u2c[sj[1]] = str(pol["pol_cid"][j])
                if sj[2] and pol["pol_tcid"][j]:
                    u2c[sj[2][0]] = str(pol["pol_tcid"][j])
                if sj[3] and pol["pol_tcid"][j]:
                    u2c[sj[3][0]] = str(pol["pol_tcid"][j])
        # 自席ターンごとのクラス集計（攻撃の単位＝対象付き DON_BOX の main 行・
        # 箱の外に出た ATTACK の main 行。箱の中の ATTACK（kind 2）は数えない＝二重計上を避ける）
        turns = {}                      # (who, turn) → Counter
        don_att = collections.defaultdict(set)   # (who, turn) → DON を付けた uuid
        attackers = collections.defaultdict(set)
        zs = {}
        forced_n = collections.Counter()
        main_rows = []                  # (who, turn, i)
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            zs[w] = float(rows["z"][i])
            if t < 1 or (t % 2 == 1) != (w == 0):
                continue                # 相手番の行（カウンター・ブロッカー）は方針に数えない
            key = (w, t)
            c = turns.setdefault(key, collections.Counter())
            sj = json.loads(rows["sig"][i])
            kind = int(rows["kind"][i])
            if int(rows["forced"][i]):
                forced_n[w] += 1
            if kind == 0:
                main_rows.append((w, t, i))
                cl = _cls(sj, u2c, cards)
                if cl == "unknown":
                    unknown_targets += 1
                elif cl == "don":
                    don_att[key].add(sj[1])
                elif cl is not None:
                    c[cl] += 1
                    if sj[0] in ("ATTACK", "DON_BOX"):
                        attackers[key].add(sj[1])
                if sj[0] == "DON_BOX" and sj[2]:
                    don_att[key].add(sj[1])
            elif sj[0] == "ATTACK":
                attackers[key].add(sj[1])
        # 席ごとの方針列
        seat_labels = {}
        seat_lean = {}
        for w in (0, 1):
            ts = sorted(t for (ww, t) in turns if ww == w)
            labs = [_label(turns[(w, t)]) for t in ts]
            seat_labels[w] = dict(zip(ts, labs))
            leans = [_lean(turns[(w, t)]) for t in ts]
            seat_lean[w] = dict(zip(ts, leans))
            dr = [abs(a - b) for a, b in zip(leans, leans[1:]) if a is not None and b is not None]
            for t, lab in zip(ts, labs):
                label_band[(_turn_band(t), lab)] += 1
            sw = []; fb = []; sw_early = []
            for a, b, t in zip(labs, labs[1:], ts[1:]):
                trans[(a, b)] += 1
                trans_band[(_turn_band(t), a, b)] += 1
                sw.append(a != b)
                if t <= early_turn:
                    sw_early.append(a != b)
                if a in ("face", "board") and b in ("face", "board"):
                    fb.append(a != b)
            waste = sum(len(don_att[(w, t)] - attackers[(w, t)]) for t in ts)
            per_seat.append({
                "seed": seed, "who": w, "z": zs.get(w, 0.0), "own_turns": len(ts),
                "labels": labs, "switch": (float(np.mean(sw)) if sw else None),
                "switch_early": (float(np.mean(sw_early)) if sw_early else None),
                "fb_switch": (float(np.mean(fb)) if fb else None), "fb_pairs": len(fb),
                "don_waste": waste / max(len(ts), 1), "forced": int(forced_n[w]),
                "drift": (float(np.mean(dr)) if dr else None), "drift_pairs": len(dr),
            })
        # 分岐点: face 系と board 系の候補が拮抗する main 行
        for w, t, i in main_rows:
            k = int(L[i])
            if k < 2:
                continue
            sl = slice(ptr[i], ptr[i] + k)
            pi = pol["pol_n"][sl].astype(np.float64)
            if pi.sum() <= 0:
                continue
            pi = pi / pi.sum()
            cls = [_cls(json.loads(pol["pol_sig"][j], ), u2c, cards) for j in range(ptr[i], ptr[i] + k)]
            best = {}
            for c_, p_ in zip(cls, pi):
                if c_ in ("face", "board") and p_ > best.get(c_, -1.0):
                    best[c_] = p_
            if len(best) < 2:
                continue
            hi, lo = max(best.values()), min(best.values())
            if lo < fork_ratio * hi:
                continue
            ch = int(rows["pol_chosen"][i])
            ch_cls = cls[ch] if 0 <= ch < k else None
            top_cls = cls[int(np.argmax(pi))]
            pl = seat_lean[w].get(t - 2)
            prev = None if pl is None or pl == 0.5 else ("face" if pl > 0.5 else "board")
            forks.append({"who": w, "z": zs.get(w, 0.0), "turn": t, "k": k,
                          "h_pi": float(-(np.clip(pi, 1e-12, None) * np.log(np.clip(pi, 1e-12, None))).sum()),
                          "chosen": ch_cls, "top": top_cls, "prev": prev,
                          "face_share": float(best["face"] / max(best["face"] + best["board"], 1e-12))})
    print(f"行 {n_rows}・対局 {games}（{time.time()-t0:.0f}s）", flush=True)
    return _aggregate(per_seat, trans, trans_band, label_band, forks, unknown_targets, games,
                      fork_ratio, early_turn, t0)


def _aggregate(per_seat, trans, trans_band, label_band, forks, unknown_targets, games,
               fork_ratio, early_turn, t0):
    out = {"games": games, "seats": len(per_seat), "unknown_targets": unknown_targets,
           "fork_ratio": fork_ratio, "early_turn": early_turn}
    # 1. 分布・遷移
    out["label_dist"] = dict(collections.Counter(l for r in per_seat for l in r["labels"]))
    out["label_by_turn"] = {b: {l: label_band[(b, l)] for l in LABELS if label_band[(b, l)]}
                            for b in ("t1-4", "t5-8", "t9+")}
    out["transitions"] = {a: {b: trans[(a, b)] for b in LABELS if trans[(a, b)]} for a in LABELS}
    out["transitions_by_turn"] = {
        band: {a: {b: trans_band[(band, a, b)] for b in LABELS if trans_band[(band, a, b)]}
               for a in LABELS} for band in ("t1-4", "t5-8", "t9+")}
    # 2. 切り替え率 × 勝敗
    def split(key):
        w = [r[key] for r in per_seat if r["z"] > 0 and r[key] is not None]
        l = [r[key] for r in per_seat if r["z"] < 0 and r[key] is not None]
        return {"win": _mean_ci(w), "lose": _mean_ci(l),
                "diff_lose_minus_win": ((float(np.mean(l)) - float(np.mean(w))) if w and l else None)}
    out["switch"] = {k: split(k) for k in ("switch", "switch_early", "fb_switch", "drift", "don_waste")}
    # 対にした比較（同じ対局の両席）
    by_seed = collections.defaultdict(dict)
    for r in per_seat:
        by_seed[r["seed"]][r["who"]] = r
    paired = {}
    for key in ("switch", "switch_early", "fb_switch", "drift", "don_waste"):
        more = 0; less = 0; tie = 0
        for g in by_seed.values():
            if len(g) < 2 or g[0][key] is None or g[1][key] is None:
                continue
            lo = g[0] if g[0]["z"] < 0 else g[1]; wi = g[1] if lo is g[0] else g[0]
            if lo[key] > wi[key]:
                more += 1
            elif lo[key] < wi[key]:
                less += 1
            else:
                tie += 1
        paired[key] = {"loser_more": more, "loser_less": less, "tie": tie,
                       "p_loser_more_given_diff": _prop_ci(more, more + less)}
    out["paired"] = paired
    # 対局の長さ帯で層別（自席ターン数）
    def lb(n):
        return "T<=5" if n <= 5 else "T6-8" if n <= 8 else "T9+"
    out["switch_by_length"] = {}
    for band in ("T<=5", "T6-8", "T9+"):
        sub = [r for r in per_seat if lb(r["own_turns"]) == band]
        w = [r["switch"] for r in sub if r["z"] > 0 and r["switch"] is not None]
        l = [r["switch"] for r in sub if r["z"] < 0 and r["switch"] is not None]
        out["switch_by_length"][band] = {"win": _mean_ci(w), "lose": _mean_ci(l)}
    # 3. 分岐点の粘り
    fk = {"points": len(forks)}
    if forks:
        fk["h_pi"] = float(np.mean([f["h_pi"] for f in forks]))
        fk["face_share"] = float(np.mean([f["face_share"] for f in forks]))
        fk["chosen_dist"] = dict(collections.Counter(f["chosen"] for f in forks))
        withprev = [f for f in forks if f["prev"] is not None and f["chosen"] in ("face", "board")]
        fk["with_prev"] = len(withprev)
        if withprev:
            base = collections.Counter(f["prev"] for f in withprev)
            fk["prev_dist"] = dict(base)
            # 基準率: 前ターンの方針を知らずに選ぶなら sticky の期待値は Σ P(prev=c)·P(chosen=c)
            chd = collections.Counter(f["chosen"] for f in withprev)
            nn = len(withprev)
            fk["sticky_expected_if_stateless"] = float(sum(
                (base[c] / nn) * (chd[c] / nn) for c in ("face", "board")))
            fk["sticky_chosen"] = _prop_ci(sum(f["chosen"] == f["prev"] for f in withprev), nn)
            fk["sticky_teacher"] = _prop_ci(sum(f["top"] == f["prev"] for f in withprev), nn)
            for name, cond in (("win", lambda f: f["z"] > 0), ("lose", lambda f: f["z"] < 0)):
                sub = [f for f in withprev if cond(f)]
                fk[f"sticky_chosen_{name}"] = _prop_ci(sum(f["chosen"] == f["prev"] for f in sub), len(sub))
            fk["sticky_by_turn"] = {}
            for band in ("t1-4", "t5-8", "t9+"):
                sub = [f for f in withprev if _turn_band(f["turn"]) == band]
                fk["sticky_by_turn"][band] = _prop_ci(sum(f["chosen"] == f["prev"] for f in sub), len(sub))
        # 分岐で「勝者は face を選ぶか」: 選んだ方針 × 勝敗
        fk["chosen_x_z"] = {c: {"win": sum(1 for f in forks if f["chosen"] == c and f["z"] > 0),
                                "lose": sum(1 for f in forks if f["chosen"] == c and f["z"] < 0)}
                            for c in ("face", "board")}
    out["fork"] = fk
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _print(out):
    print(f"対局 {out['games']}・席 {out['seats']}・unknown 対象 {out['unknown_targets']}")
    print("ラベル分布:", out["label_dist"])
    for b, d in out["label_by_turn"].items():
        print(f"  {b}: {d}")
    print("遷移（prev → cur）:")
    for a, d in out["transitions"].items():
        if d:
            tot = sum(d.values())
            print(f"  {a:8} n {tot:6d}  " + "  ".join(f"{b} {v/tot:.2f}" for b, v in d.items()))
    print("切り替え率（勝者／敗者）:")
    for k, d in out["switch"].items():
        w, l = d["win"], d["lose"]
        if w.get("n") and l.get("n"):
            print(f"  {k:13} win {w['mean']:.3f} [{w['ci95'][0]:.3f},{w['ci95'][1]:.3f}] n {w['n']}"
                  f"   lose {l['mean']:.3f} [{l['ci95'][0]:.3f},{l['ci95'][1]:.3f}] n {l['n']}"
                  f"   diff {d['diff_lose_minus_win']:+.3f}")
    print("対にした比較（敗者の方が多い割合・差のある対局のみ）:")
    for k, d in out["paired"].items():
        p = d["p_loser_more_given_diff"]
        if p.get("n"):
            print(f"  {k:13} {p['p']:.3f} [{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] n {p['n']}"
                  f"  (more {d['loser_more']} / less {d['loser_less']} / tie {d['tie']})")
    print("長さ帯別 switch（win／lose）:")
    for b, d in out["switch_by_length"].items():
        if d["win"].get("n") and d["lose"].get("n"):
            print(f"  {b:6} win {d['win']['mean']:.3f} (n {d['win']['n']})  lose {d['lose']['mean']:.3f} (n {d['lose']['n']})")
    fk = out["fork"]
    print(f"分岐点（face 系 vs board 系が拮抗）: {fk['points']} 点")
    if fk.get("with_prev"):
        def f(p):
            return f"{p['p']:.3f} [{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}] n {p['n']}"
        print(f"  H_pi {fk['h_pi']:.3f}  face_share {fk['face_share']:.3f}  chosen {fk['chosen_dist']}")
        print(f"  前ターンの方針あり {fk['with_prev']}  prev {fk['prev_dist']}")
        print(f"  sticky 期待値（無状態）{fk['sticky_expected_if_stateless']:.3f}"
              f"  chosen {f(fk['sticky_chosen'])}  teacher {f(fk['sticky_teacher'])}")
        print(f"  sticky win {f(fk['sticky_chosen_win'])}  lose {f(fk['sticky_chosen_lose'])}")
        for b, p in fk["sticky_by_turn"].items():
            if p.get("n"):
                print(f"  sticky {b}: {f(p)}")
        print(f"  chosen×z {fk['chosen_x_z']}")
    print(f"  {out['seconds']}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ（v14 の波）")
    ap.add_argument("--fork-ratio", type=float, default=0.5,
                    help="分岐とみなす訪問の拮抗（劣る側 / 勝る側 ≥ この値）")
    ap.add_argument("--early-turn", type=int, default=6, help="switch_early の上限ターン")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = analyze(args.src, args.fork_ratio, args.early_turn, args.limit_games)
    out["src"] = [os.path.abspath(s) for s in args.src]
    _print(out)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  → {args.out}")


if __name__ == "__main__":
    main()
