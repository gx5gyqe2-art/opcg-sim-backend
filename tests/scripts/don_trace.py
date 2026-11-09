"""WP `don-trace`（`docs/search_efficiency_plan.md` §7.76・台帳 2+510）: ドン 1 枚の差がネットのどこで消えるかを段ごとに読む。
読み取りと小さな当て直しだけ＝訓練の作り・エンジン・出荷は変えない。1 回きりの計測＝テストは足さない。

材料は波 82j（`--don-jitter 1.0`・局ごとに 1 回、手番 3〜10 の頭に手番の席のドンを無作為な向きに ±1）。
「ドンを動かした行」＝動かした手番の、手番の席の最初の行（`don_jitter_rows.py` と同じ選び方）。符号は無作為なので、
+1 の群と −1 の群の平均の差＝ドン 2 枚ぶんの効き。主に手番 3〜6（ドンデッキが必ず残る＝向きが本当に半々）を読む。

  cache     … 82j の行を 1 つの npz に束ねる（ドンを動かした行・手番 3〜6 の全行・全行の印）。→ ~/don_trace_cache.npz
  invariant … 各ネットの価値の群の差（§7.75 の 0.30／0.31・r28x4 0.28 と一致するか）
  input     … 段 1: 同じ局面でドンを動かす前と後を符号化し、どの列がどれだけ動くかを数える（小さな探索で局を進める）
  linear    … 段 2: 入力の列だけの線形の当てはめ（リッジ）で勝敗を当て、ドンを動かした行での群の差を出す（ドンの列を抜いた版も）
  repr      … 段 3: ネットの中の 3 か所（全体の入力の束・1 段目の隠れ層・共有表現）で (i) 線形の当てはめ、
              (ii) 価値の出口（最後の層）だけを同じ損失で当て直す
  weights   … 段 3 (ii)＋段 4: 出口だけ／価値の 2 層＋出口の当て直し（numpy・手書きの勾配・Adam）で、訓練の作りの部品
              （補助の頭・周の数・学習の速さ・tanh・最後の層だけ・新しく始める）を 1 つずつ変える
  trajectory… 段 4 の続き: 価値の 2 層＋出口の当て直しを 1 周ごとに読む（ドンを動かした後の行を抜いた当てはめと比べる）

    python tests/scripts/don_trace.py cache
    python tests/scripts/don_trace.py invariant --nets opcg_sim/data/learned/nrel_r28x4.npz ~/djnets/nrel_dja82.ep1.npz ...
    python tests/scripts/don_trace.py input --games 200 --sims 16 --nets ...      # 段 1（Rust の wheel が要る）
    python tests/scripts/don_trace.py linear --alphas 1 100 1000                    # 段 2（段 1 の出力を読む）
    python tests/scripts/don_trace.py repr --alphas 1 100 --exit-epochs --nets ...  # 段 3 (i)
    python tests/scripts/don_trace.py weights --exit-epochs 5 20 --nets ...         # 段 3 (ii)・段 4
    python tests/scripts/don_trace.py trajectory --traj-epochs 12 --nets ...        # 段 4 の周ごと
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _bootstrap  # noqa: E402,F401

OUT = os.path.join("n1_results", "don_trace")
CACHE = os.path.expanduser("~/don_trace_cache.npz")
WAVE = "~/n82j/n_records/n82j_w*_p*/n_record_*.npz"
_G = {}


def _w(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(obj, open(os.path.join(OUT, name), "w"), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- 束ねる
def cmd_cache(a):
    """ドンを動かした行（局ごとに 1 行）と、手番 3〜6 の全行・全手番の間引き行を束ねる。"""
    J = {k: [] for k in ("sc", "ci", "tok", "z", "sign", "turn", "hold", "seed")}
    R = {k: [] for k in ("sc", "ci", "tok", "z", "turn", "hold", "seed", "jit", "shift", "aux", "aux_mask")}
    for f in sorted(glob.glob(os.path.expanduser(a.wave))):
        d = np.load(f, allow_pickle=True)
        dj, seed, turn, who, step = d["don_jitter"], d["seed"], d["turn"], d["who"], d["step"]
        sc, ci, tok, z = d["scalars"], d["card_idx"][:, :22], d["tokens"], d["z"]
        jit_row = np.zeros(len(seed), bool)
        shift = np.zeros(len(seed), np.int8)        # その行の席のドンのずれ（動かした席は符号・相手は −符号ではなく 0）
        for s in np.unique(seed):
            idx = np.flatnonzero(seed == s)
            t, seat, sg = dj[idx[0]]
            if sg == 0:
                continue
            after = idx[(who[idx] == seat) & ((turn[idx] > t) | ((turn[idx] == t)))]
            shift[after] = sg                      # 動かした後の、動かした席の行（ずれは以後も残る・10 枚の頭打ちまで）
            hit = idx[(turn[idx] == t) & (who[idx] == seat)]
            if not len(hit):
                continue
            i = int(hit[np.argmin(step[hit])])
            jit_row[i] = True
            for k, v in (("sc", sc[i]), ("ci", ci[i]), ("tok", tok[i]), ("z", z[i]), ("sign", sg),
                         ("turn", t), ("hold", int(s) % 7 == 0), ("seed", s)):
                J[k].append(v)
        m = (turn >= 3) & (turn <= 6)
        for k, v in (("sc", sc), ("ci", ci), ("tok", tok), ("z", z), ("turn", turn), ("seed", seed),
                     ("jit", jit_row), ("shift", shift), ("aux", d["aux"]), ("aux_mask", d["aux_mask"])):
            R[k].append(v[m])
        R["hold"].append(seed[m] % 7 == 0)
    out = {f"j_{k}": np.array(v) for k, v in J.items()}
    out.update({f"r_{k}": np.concatenate(v) for k, v in R.items()})
    np.savez(CACHE, **out)
    print("ドンを動かした行", len(J["z"]), "手番 3〜6 の全行", len(out["r_z"]), flush=True)


def load_cache():
    if "c" not in _G:
        d = np.load(CACHE)
        _G["c"] = {k: d[k] for k in d.files}
    return _G["c"]


def group_diff(v, sign, m):
    a, b = v[m & (sign == 1)], v[m & (sign == -1)]
    se = math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
    return {"diff": round(float(a.mean() - b.mean()), 4), "ci95": round(1.96 * se, 4), "n": [int(len(a)), int(len(b))]}


def jit_groups(c):
    early = (c["j_turn"] >= 3) & (c["j_turn"] <= 6)
    hold = c["j_hold"].astype(bool)
    return {"turn3_6": early, "turn3_6_trained": early & ~hold, "turn3_6_holdout": early & hold, "turn7_10": ~early}


def diffs(v, c):
    return {g: group_diff(v, c["j_sign"], m) for g, m in jit_groups(c).items()}


# ---------------------------------------------------------------- ネットの読み出し
def load_net(path):
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db
    if "tables" not in _G:
        stats, ab, abm, pwr, isl, vocab = build_eff_tables()
        _G["tables"] = (stats, ab, abm, pwr, isl)
        _G["rt"] = NR.RelTable(NR.profile_table(load_db(), vocab))
    return NL.NRelNet.load(path, tables=_G["tables"])


def net_layers(net, sc, ci, tok, bs=512):
    """価値の forward を手で回し、全体の入力の束（415）・1 段目の隠れ層（192）・共有表現（64）・価値を返す。"""
    from opcg_sim.learned.train import n_rel_train as NT
    tab = net.card_table()
    Z, R1, E, V = [], [], [], []
    for s in range(0, len(sc), bs):
        a_, c_, t_ = sc[s:s + bs].astype(np.float32), ci[s:s + bs].astype(np.int64), tok[s:s + bs].astype(np.float32)
        rom, roo = NT.relations_or_zeros(net, c_, t_, _G["rt"])
        h, present = net.tokens_forward(c_, t_, rom, roo, tab)
        keep = {}
        e = net.body(a_, h, present, keep=keep)
        Z.append(keep["z"].astype(np.float32)); R1.append(keep["r1"].astype(np.float32)); E.append(e.astype(np.float32))
        V.append(np.tanh((e @ net.Wv + net.bv)[:, 0]))
    return {"z": np.concatenate(Z), "r1": np.concatenate(R1), "e": np.concatenate(E), "v": np.concatenate(V)}


def cmd_invariant(a):
    c = load_cache()
    res = {"outcome_z": diffs(c["j_z"], c), "nets": {}}
    for n in a.nets:
        L = net_layers(load_net(n), c["j_sc"], c["j_ci"], c["j_tok"])
        res["nets"][os.path.basename(n)] = diffs(L["v"], c)
        print(os.path.basename(n), {g: v["diff"] for g, v in res["nets"][os.path.basename(n)].items()}, flush=True)
    print("勝敗", {g: v["diff"] for g, v in res["outcome_z"].items()})
    _w("invariant.json", res)


# ---------------------------------------------------------------- 段 1: 入力
class _Stop(Exception):
    pass


def _enc(game, name):
    enc = json.loads(game.encode(name))
    return (np.asarray(enc["scalars"], np.float32), np.asarray(enc["tokens"], np.float32).reshape(22, -1),
            np.asarray(enc["card_idx"], np.int64)[:22])


def cmd_input(a):
    """82j と同じ決め方（局の seed から手番・向き）で、手番の席の最初のメインの判断の頭に、動かす前と後を符号化する。
    局は小さな探索（`--sims`）で進める＝局面は 82j と同じではないが、問うのは「ドンを 1 枚動かすと符号化のどの列が動くか」。"""
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    from opcg_sim.loop import record_gen as RG
    E.engine()
    spec = E.SeatSpec(a.net, sims=a.sims)
    db = D.load_db()
    pairs = []
    for k in range(a.games):
        seed = a.seed0 + k
        jit = RG.DonJitter(seed, 1.0)
        cap = {}

        def pre(game, name, turn, step, jit=jit, cap=cap):
            if not jit.on or turn != jit.turn or name != game.turn_player() or game.has_interaction or game.phase != "MAIN":
                return
            before = _enc(game, name)
            jit(game, name, turn, step)
            if jit.done is None:
                raise _Stop()
            cap.update(before=before, after=_enc(game, name), turn=jit.done[0], sign=jit.done[2], src=jit.src)
            raise _Stop()
        try:
            la, lb = D.leader_pair(db, seed, "random")
            p1, p2 = D.build_pair(db, la, lb, seed, "real")
            DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=3000, pre=pre)
        except _Stop:
            pass
        except Exception as ex:  # noqa: BLE001
            print("skip", seed, repr(ex)[:120], flush=True)
            continue
        if cap:
            pairs.append(cap)
        if (k + 1) % 20 == 0:
            print(k + 1, "局", len(pairs), "対", flush=True)
    from opcg_sim.learned import n_rel_feat as NR
    sb = np.array([p["before"][0] for p in pairs]); sa = np.array([p["after"][0] for p in pairs])
    tb = np.array([p["before"][1] for p in pairs]); ta = np.array([p["after"][1] for p in pairs])
    cb = np.array([p["before"][2] for p in pairs]); ca = np.array([p["after"][2] for p in pairs])
    sign = np.array([p["sign"] for p in pairs]); turn = np.array([p["turn"] for p in pairs])
    ds = (sa - sb) * sign[:, None]                    # 向きを揃えた動き（+1 の向きに直す）
    dt = (ta - tb) * sign[:, None, None]
    res = {"pairs": len(pairs), "sign": {"+1": int((sign == 1).sum()), "-1": int((sign == -1).sum())},
           "card_idx_changed": int((ca != cb).any(1).sum()), "scalars": {}, "tokens": {}}
    names = scalar_names()
    for j in np.flatnonzero(np.abs(sa - sb).max(0) > 1e-6):
        res["scalars"][f"{j}:{names[j]}"] = {"moved_rows": int((np.abs(sa[:, j] - sb[:, j]) > 1e-6).sum()),
                                             "mean_signed": round(float(ds[:, j].mean()), 4),
                                             "mean_abs": round(float(np.abs(sa[:, j] - sb[:, j]).mean()), 4),
                                             "range_before": [round(float(sb[:, j].min()), 3), round(float(sb[:, j].max()), 3)]}
    for j in range(tb.shape[2]):
        mv = np.abs(ta[:, :, j] - tb[:, :, j]) > 1e-6
        if mv.any():
            res["tokens"][NR.S_COLS[j]] = {"moved_rows": int(mv.any(1).sum()), "moved_slots": int(mv.sum()),
                                           "mean_signed_per_row": round(float(dt[:, :, j].sum(1).mean()), 4)}
    np.savez(os.path.join(os.path.expanduser("~"), "don_trace_pairs.npz"), sb=sb, sa=sa, tb=tb, ta=ta, cb=cb, ca=ca, sign=sign, turn=turn)
    # ネットの価値が同じ対でどれだけ動くか（入力の段での差の大きさの参考）
    res["net_value_shift"] = {}
    for n in a.nets:
        net = load_net(n)
        vb = net_layers(net, sb, cb, tb)["v"]; va = net_layers(net, sa, ca, ta)["v"]
        dv = (va - vb) * sign
        e3 = (turn >= 3) & (turn <= 6)
        res["net_value_shift"][os.path.basename(n)] = {"turn3_6_per_don": round(float(dv[e3].mean()), 4),
                                                        "all_per_don": round(float(dv.mean()), 4), "n3_6": int(e3.sum())}
        print(os.path.basename(n), res["net_value_shift"][os.path.basename(n)], flush=True)
    res["normalization"] = NORMALIZATION
    _w("stage1_input.json", res)
    print(json.dumps({k: res[k] for k in ("pairs", "sign", "card_idx_changed", "scalars", "tokens")}, ensure_ascii=False, indent=1))


#: ドンが入る列の正規化（`opcg_sim/learned/encoder.py`・`n_rel_feat.py` から書き写し・Rust の `encode` が同じ列を作る）。
NORMALIZATION = {
    "scalars[2..5] 自分・相手のアクティブ／レストのドン": "枚数そのまま（割らない・上限で切らない）",
    "scalars[14..15] リーダーの付与ドン": "÷5",
    "v9 ドンデッキの残り（自分・相手）": "÷10",
    "v5 展開できる手札のキャラ数": "今のアクティブのドン以下のコストの手札キャラ数 ÷10",
    "EXTRA don_addable": "n_rel_feat の EXTRA 列（付けられるドン）",
    "EXTRA don_next_turn": "次の手番のドンの総数（10 で頭打ち）÷10",
    "EXTRA max_play_next_turn／leader_power_max": "次の手番のドンで出せる最大コスト・付けたときのリーダーの最大パワー",
    "tokens attached_don": "÷5",
    "tokens playable_now／can_attack_now 等": "今のドンで出せる・使える旗（0／1）",
}


def scalar_names():
    """127 列の名前（分かる所だけ・残りは番号）。"""
    from opcg_sim.learned import n_rel_feat as NR
    nm = [f"c{j}" for j in range(127)]
    base = ["life_me", "life_opp", "don_active_me", "don_rested_me", "don_active_opp", "don_rested_opp",
            "hand_me", "hand_opp", "field_me", "field_opp", "turn_count", "is_my_turn", "leader_power_me", "leader_power_opp",
            "leader_don_me", "leader_don_opp", "deck_me", "deck_opp", "trash_me", "trash_opp", "koed_me", "koed_opp"]
    nm[:len(base)] = base
    nm[54] = "playable_chars_me"; nm[60:63] = ["onplay_live", "onplay_keep", "onplay_dead"]
    nm[66] = "don_deck_me"; nm[67] = "don_deck_opp"
    for j, n in enumerate(NR.EXTRA_COLS):
        nm[94 + j] = n
    return nm


# ---------------------------------------------------------------- 段 2: 入力だけの当てはめ
def input_features(sc, ci, tok):
    """全スカラー（127）＋ゾーンごとのトークン状態の和（5 ゾーン × 22 列）と枚数（5）。"""
    from opcg_sim.learned import n_rel as NL
    present = (ci > 0).astype(np.float32)
    zones = []
    for z in range(5):
        cols = np.flatnonzero(NL.ZONE_ONEHOT[:, z] > 0)
        zones.append((tok[:, cols, :].astype(np.float32) * present[:, cols, None]).sum(1))
        zones.append(present[:, cols].sum(1, keepdims=True))
    return np.concatenate([sc.astype(np.float32)] + zones, 1)


def feature_names():
    from opcg_sim.learned import n_rel_feat as NR
    zn = ("own_leader", "opp_leader", "own_field", "opp_field", "hand")
    out = list(scalar_names())
    for z in zn:
        out += [f"{z}.{s}" for s in NR.S_COLS] + [f"{z}.count"]
    return out


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd
    A = Xs.T @ Xs + alpha * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xs.T @ (y - y.mean()))
    return lambda Q: ((Q - mu) / sd) @ w + y.mean()


def probe(Xr, y, fit_m, Xj, c, alpha, Xr_eval_m=None):
    """行の集まり（fit_m）で勝敗にリッジを当て、ドンを動かした行で群の差を出す。取り置きの局での当てはまりも。"""
    f = ridge_fit(Xr[fit_m], y[fit_m], alpha)
    out = diffs(f(Xj), c)
    if Xr_eval_m is not None:
        p = f(Xr[Xr_eval_m])
        out["holdout_mse"] = round(float(np.mean((p - y[Xr_eval_m]) ** 2)), 4)
        out["holdout_mse_const"] = round(float(np.mean((y[fit_m].mean() - y[Xr_eval_m]) ** 2)), 4)
    return out


def cmd_linear(a):
    c = load_cache()
    Xr = input_features(c["r_sc"], c["r_ci"], c["r_tok"]); Xj = input_features(c["j_sc"], c["j_ci"], c["j_tok"])
    y = c["r_z"].astype(np.float64); hold = c["r_hold"].astype(bool)
    names = feature_names()
    moved = json.load(open(os.path.join(OUT, "stage1_input.json")))
    # ドンの数そのものの列（アクティブ・レスト・ドンデッキの残り・次の手番のドン）
    don_direct = [j for j, n in enumerate(names) if n in ("don_active_me", "don_rested_me", "don_deck_me", "don_next_turn")]
    # 段 1 で動いた列（スカラーは番号・トークンは全ゾーンの同じ列）
    mv_sc = [int(k.split(":")[0]) for k in moved["scalars"]]
    mv_tok = set(moved["tokens"])
    don_all = sorted(set(mv_sc) | {j for j, n in enumerate(names) if "." in n and n.split(".", 1)[1] in mv_tok})
    res = {"features": len(names), "rows_fit": int((~hold).sum()), "rows_holdout": int(hold.sum()),
           "don_direct_cols": [names[j] for j in don_direct], "don_moved_cols": [names[j] for j in don_all], "fits": {}}
    keep_all = np.arange(Xr.shape[1])
    for label, cols in (("all", keep_all), ("drop_don_direct", np.setdiff1d(keep_all, don_direct)),
                        ("drop_don_moved", np.setdiff1d(keep_all, don_all)), ("only_don_moved_plus_turn", np.array(sorted(set(don_all) | {10, 11})))):
        for alpha in a.alphas:
            r = probe(Xr[:, cols], y, ~hold, Xj[:, cols], c, alpha, hold)
            res["fits"][f"{label}@{alpha:g}"] = r
            print(label, alpha, {g: r[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")},
                  "holdout_mse", r["holdout_mse"], "const", r["holdout_mse_const"], flush=True)
    # 当てはめの集まりからドンを動かした後の行（ずれが残る行）を抜くと、例の無い教材と同じになる＝比べる
    m_noshift = ~hold & (c["r_shift"] == 0)
    r = probe(Xr, y, m_noshift, Xj, c, a.alphas[0], hold)
    res["fits"]["all_fit_without_shifted_rows"] = r
    print("ずれた行を抜いて当てる", {g: r[g]["diff"] for g in ("turn3_6", "turn3_6_trained")}, flush=True)
    # 物差しの確かめ: 動かした向きそのもの（ずれの列）を足せば、当てはめは勝敗の差まで届くか
    Xs = np.concatenate([Xr, c["r_shift"][:, None].astype(np.float32)], 1)
    Xjs = np.concatenate([Xj, c["j_sign"][:, None].astype(np.float32)], 1)
    r = probe(Xs, y, ~hold, Xjs, c, a.alphas[0], hold)
    res["fits"]["all_plus_shift_sign"] = r
    print("向きの列を足す", {g: r[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")}, flush=True)
    res["outcome_z"] = diffs(c["j_z"], c)
    _w("stage2_linear.json", res)


# ---------------------------------------------------------------- 段 3: 中の表現
def layers_cached(path):
    """手番 3〜6 の全行とドンを動かした行で、ネットの 3 か所の表現（キャッシュ）。"""
    key = os.path.expanduser(f"~/don_trace_layers_{os.path.basename(path)}")
    if os.path.exists(key):
        d = np.load(key)
        return {k: d[k] for k in d.files}
    c = load_cache()
    net = load_net(path)
    t0 = time.time()
    Lr = net_layers(net, c["r_sc"], c["r_ci"], c["r_tok"]); Lj = net_layers(net, c["j_sc"], c["j_ci"], c["j_tok"])
    print("表現の読み出し", os.path.basename(path), f"{time.time() - t0:.0f}s", flush=True)
    out = {f"r_{k}": v for k, v in Lr.items()} | {f"j_{k}": v for k, v in Lj.items()}
    np.savez(key, **out)
    return out


class _Adam:
    def __init__(self, P, lr, wd=0.0):
        self.P, self.lr, self.wd, self.t = P, lr, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}; self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, G):
        self.t += 1
        for k, g in G.items():
            if self.wd:
                g = g + self.wd * self.P[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g; self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _huber_grad(d):
    return np.clip(d, -1.0, 1.0)


def refit_exit(E_fit, y, _unused, Wv, bv, epochs=20, lr=1e-3, bs=256, seed=0, linear_out=False, wd=0.0):
    """価値の出口（共有表現 → 1 → tanh）だけを 0.5·mean((v−z)²) で当て直す（Adam・ネットの値から始める）。"""
    P = {"Wv": Wv.astype(np.float64).copy(), "bv": bv.astype(np.float64).copy()}
    opt = _Adam(P, lr, wd)
    g = np.random.default_rng(seed)
    for _ep in range(epochs):
        perm = g.permutation(len(E_fit))
        for s in range(0, len(E_fit) - bs + 1, bs):
            i = perm[s:s + bs]
            x = E_fit[i].astype(np.float64)
            o = (x @ P["Wv"] + P["bv"])[:, 0]
            v = o if linear_out else np.tanh(o)
            do = (v - y[i]) / len(i) * (1.0 if linear_out else (1 - v * v))
            opt.step({"Wv": x.T @ do[:, None], "bv": np.array([do.sum()])})

    def f(Q):
        o = (Q @ P["Wv"] + P["bv"])[:, 0]
        return o if linear_out else np.tanh(o)
    return f


def cmd_repr(a):
    c = load_cache()
    y = c["r_z"].astype(np.float64); hold = c["r_hold"].astype(bool)
    res = {"nets": {}}
    for n in a.nets:
        L = layers_cached(n)
        net = load_net(n)
        r = {"net_value": diffs(L["j_v"], c), "probe": {}, "exit_refit": {}}
        for layer in ("z", "r1", "e"):
            for alpha in a.alphas:
                r["probe"][f"{layer}@{alpha:g}"] = probe(L[f"r_{layer}"], y, ~hold, L[f"j_{layer}"], c, alpha, hold)
                print(os.path.basename(n), layer, alpha, {g: r["probe"][f"{layer}@{alpha:g}"][g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")}, flush=True)
        # (ii) 出口だけの当て直し（同じ損失・ネットの値から）。手番 3〜6 の訓練側の行で。
        for ep in a.exit_epochs:
            f = refit_exit(L["r_e"][~hold], y[~hold], None, net.Wv, net.bv, epochs=ep, lr=a.exit_lr)
            rr = diffs(f(L["j_e"]), c)
            rr["holdout_mse"] = round(float(np.mean((f(L["r_e"][hold]) - y[hold]) ** 2)), 4)
            rr["fit_mse"] = round(float(np.mean((f(L["r_e"][~hold]) - y[~hold]) ** 2)), 4)
            r["exit_refit"][f"ep{ep}"] = rr
            print(os.path.basename(n), "出口の当て直し", ep, {g: rr[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")}, rr["holdout_mse"], flush=True)
        base_v = L["r_v"]
        r["net_mse"] = {"holdout": round(float(np.mean((base_v[hold] - y[hold]) ** 2)), 4), "fit": round(float(np.mean((base_v[~hold] - y[~hold]) ** 2)), 4)}
        res["nets"][os.path.basename(n)] = r
    res["outcome_z"] = diffs(c["j_z"], c)
    _w("stage3_repr.json", res)


# ---------------------------------------------------------------- 段 4: 訓練の重み
def refit_body(Z, y, aux, aux_mask, net, Zj, Zh, epochs, lr, aux_w=0.0, scratch=False, linear_out=False, bs=256, seed=0, layers="all",
               on_epoch=None):
    """全体の入力の束（415）を固定し、価値の 2 層と出口（＋補助の頭）を当て直す（numpy・手書きの勾配・Adam）。損失は訓練と同じ
    0.5·mean((v−z)²) ＋ 補助の重み × Huber（補助の 10 列の平均・相手の枠の補助は持たない）。方策の損失は入れない。"""
    if scratch:
        from opcg_sim.learned import n_rel as NL
        src = NL.NRelNet(_G["tables"], hidden=net.W1.shape[1], seed=13 + seed)
    else:
        src = net
    names = ["W1", "b1", "W2", "b2", "Wv", "bv"] + (["Wx1", "bx1", "Wx2", "bx2"] if aux_w > 0 else [])
    P = {k: getattr(src, k).astype(np.float64).copy() for k in names}
    train = [k for k in names if layers == "all" or k not in ("W1", "b1")]
    opt = _Adam({k: P[k] for k in train}, lr)
    M = aux_mask.astype(np.float64).reshape(len(aux_mask), -1)[:, 0]

    def fwd(x):
        h1 = x @ P["W1"] + P["b1"]; r1 = np.maximum(h1, 0)
        h2 = r1 @ P["W2"] + P["b2"]; e = np.maximum(h2, 0)
        o = (e @ P["Wv"] + P["bv"])[:, 0]
        return r1, h1, e, h2, o, (o if linear_out else np.tanh(o))
    g = np.random.default_rng(seed)
    for _ep in range(epochs):
        perm = g.permutation(len(Z))
        for s0 in range(0, len(Z) - bs + 1, bs):
            i = perm[s0:s0 + bs]
            x = Z[i].astype(np.float64)
            r1, h1, e, h2, o, v = fwd(x)
            B = len(i)
            do = (v - y[i]) / B * (1.0 if linear_out else (1 - v * v))
            G = {"Wv": e.T @ do[:, None], "bv": np.array([do.sum()])}
            de = do[:, None] @ P["Wv"].T
            if aux_w > 0:
                ha = e @ P["Wx1"] + P["bx1"]; ra = np.maximum(ha, 0); pa = ra @ P["Wx2"] + P["bx2"]
                m = M[i]; den = max(m.sum(), 1.0) * pa.shape[1]
                dpa = _huber_grad(pa - aux[i]) * m[:, None] * (aux_w / den)
                G["Wx2"] = ra.T @ dpa; G["bx2"] = dpa.sum(0)
                dha = (dpa @ P["Wx2"].T) * (ha > 0)
                G["Wx1"] = e.T @ dha; G["bx1"] = dha.sum(0)
                de = de + dha @ P["Wx1"].T
            dh2 = de * (h2 > 0)
            G["W2"] = r1.T @ dh2; G["b2"] = dh2.sum(0)
            if "W1" in train:
                dh1 = (dh2 @ P["W2"].T) * (h1 > 0)
                G["W1"] = x.T @ dh1; G["b1"] = dh1.sum(0)
            opt.step({k: G[k] for k in train})
        if on_epoch is not None:
            on_epoch(_ep + 1, [fwd(q.astype(np.float64))[5] for q in (Zj, Zh)])
    return [fwd(q.astype(np.float64))[5] for q in (Zj, Zh)]


def cmd_weights(a):
    """段 4: 訓練の作りの部品を 1 つずつ変えて、価値の 2 層＋出口を当て直す（手番 3〜6 の訓練側の行・全体の入力の束は固定）。"""
    c = load_cache()
    y = c["r_z"].astype(np.float64); hold = c["r_hold"].astype(bool)
    res = {"nets": {}, "rows_fit": int((~hold).sum())}
    variants = [("as_trained_lr2e-4_ep2", dict(epochs=2, lr=2e-4)),
                ("plus_aux0.1_lr2e-4_ep2", dict(epochs=2, lr=2e-4, aux_w=0.1)),
                ("lr2e-4_ep10", dict(epochs=10, lr=2e-4)),
                ("lr1e-3_ep10", dict(epochs=10, lr=1e-3)),
                ("last_layer_only_lr1e-3_ep10", dict(epochs=10, lr=1e-3, layers="last")),
                ("linear_out_lr1e-3_ep10", dict(epochs=10, lr=1e-3, linear_out=True)),
                ("scratch_lr1e-3_ep10", dict(epochs=10, lr=1e-3, scratch=True)),
                ("scratch_lr1e-3_ep30", dict(epochs=30, lr=1e-3, scratch=True))]
    if a.only:
        variants = [v for v in variants if v[0] in a.only]
    for n in a.nets:
        L = layers_cached(n)
        net = load_net(n)
        r = {}
        # 段 3 (ii): 出口だけ（共有表現は固定）
        for ep in a.exit_epochs:
            f = refit_exit(L["r_e"][~hold], y[~hold], None, net.Wv, net.bv, epochs=ep, lr=a.exit_lr)
            rr = diffs(f(L["j_e"]), c)
            rr["holdout_mse"] = round(float(np.mean((f(L["r_e"][hold]) - y[hold]) ** 2)), 4)
            r[f"exit_only_lr{a.exit_lr:g}_ep{ep}"] = rr
            print(os.path.basename(n), "出口だけ", ep, {g: rr[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")}, rr["holdout_mse"], flush=True)
        for label, kw in variants:
            t0 = time.time()
            vj, vh = refit_body(L["r_z"][~hold], y[~hold], c["r_aux"][~hold], c["r_aux_mask"][~hold], net,
                                L["j_z"], L["r_z"][hold], **kw)
            rr = diffs(vj, c)
            rr["holdout_mse"] = round(float(np.mean((vh - y[hold]) ** 2)), 4)
            r[label] = rr
            print(os.path.basename(n), label, {g: rr[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")},
                  rr["holdout_mse"], f"{time.time() - t0:.0f}s", flush=True)
            _w("stage4_weights.partial.json", res | {"nets": res["nets"] | {os.path.basename(n): r}})
        r["net_holdout_mse"] = round(float(np.mean((L["r_v"][hold] - y[hold]) ** 2)), 4)
        res["nets"][os.path.basename(n)] = r
    res["outcome_z"] = diffs(c["j_z"], c)
    _w("stage4_weights.json", res)


def cmd_trajectory(a):
    """段 4 の続き: 価値の 2 層＋出口の当て直しを 1 周ごとに読む（ドンを動かした行での群の差と、取り置きの局の誤差）。
    当てはめの行を「手番 3〜6 の訓練側の全行」と「そこからドンを動かした後の行を抜いたもの（例の無い教材）」の 2 通りで比べる。"""
    c = load_cache()
    y = c["r_z"].astype(np.float64); hold = c["r_hold"].astype(bool)
    res = {"nets": {}}
    for n in a.nets:
        L = layers_cached(n)
        net = load_net(n)
        r = {"net": {"turn3_6": diffs(L["j_v"], c)["turn3_6"]["diff"], "holdout_mse": round(float(np.mean((L["r_v"][hold] - y[hold]) ** 2)), 4)}}
        for label, fit_m, kw in (("from_net_lr2e-4", ~hold, dict(lr=2e-4)),
                                 ("from_net_lr2e-4_no_shifted_rows", ~hold & (c["r_shift"] == 0), dict(lr=2e-4)),
                                 ("scratch_lr1e-3", ~hold, dict(lr=1e-3, scratch=True))):
            rows = []

            def cb(ep, out, rows=rows):
                vj, vh = out
                d = diffs(vj, c)
                rows.append({"epoch": ep, **{g: d[g]["diff"] for g in ("turn3_6", "turn3_6_trained", "turn3_6_holdout")},
                             "holdout_mse": round(float(np.mean((vh - y[hold]) ** 2)), 4)})
                print(os.path.basename(n), label, rows[-1], flush=True)
            refit_body(L["r_z"][fit_m], y[fit_m], c["r_aux"][fit_m], c["r_aux_mask"][fit_m], net, L["j_z"], L["r_z"][hold],
                       epochs=a.traj_epochs, on_epoch=cb, **kw)
            r[label] = rows
            _w("stage4_trajectory.partial.json", res | {"nets": res["nets"] | {os.path.basename(n): r}})
        res["nets"][os.path.basename(n)] = r
    _w("stage4_trajectory.json", res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["cache", "invariant", "input", "linear", "repr", "weights", "trajectory"])
    ap.add_argument("--wave", default=WAVE)
    ap.add_argument("--nets", nargs="*", default=[])
    ap.add_argument("--net", default="opcg_sim/data/learned/nrel_r28x4.npz", help="段 1 で局を進めるネット")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=3481000)
    ap.add_argument("--alphas", type=float, nargs="*", default=[1.0, 100.0])
    ap.add_argument("--exit-epochs", type=int, nargs="*", default=[5, 20])
    ap.add_argument("--exit-lr", type=float, default=1e-3)
    ap.add_argument("--traj-epochs", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=[], help="段 4 の変え方のうち回すものだけ")
    a = ap.parse_args(argv)
    {"cache": cmd_cache, "invariant": cmd_invariant, "input": cmd_input, "linear": cmd_linear,
     "repr": cmd_repr, "weights": cmd_weights, "trajectory": cmd_trajectory}[a.cmd](a)


if __name__ == "__main__":
    main()
