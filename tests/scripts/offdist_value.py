"""窓の損の根本原因の検証: ネットは「判断点でない盤面」を見たことがないのか（WP `offdist-value`・
`docs/search_efficiency_plan.md` §7.69・台帳 2+481）。

読み取りと 1 周の試験訓練だけ（エンジン・出荷は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。

1. ``count``: 波の行を、自分の手番の main／自分の手番の窓／相手の手番の窓／commit に分け、窓は戦闘の窓（ブロック・カウンター）か効果の窓か、
   「直前の同じ席の行から自分の場のドンが減った（＝ドンを払った直後）」かで割って数える。``counts.json``。
2. ``nn``: §7.66 の試験台の「窓を閉じた直後の盤面」（自分の視点の符号化）と、比べる基準として同じ点の判断点そのものの盤面が、
   波の行（無作為の 20 万行）にどれだけ近いか（盤面の数の列を標準化した最近傍の距離）。``nn.json``。
3. ``timeval``: 時点ごとのドン 1 枚の値。§7.66 の紫カイドウ 24 点（打ち比べ済み）で、(a) 自分の手番の頭、(b) 窓を閉じた直後、
   (c) 相手の手番の終わり（相手のターン終了の直前）、(d) 次の自分の手番の頭、の盤面から自分の場のドンを 1 枚だけ減らした対を作り、
   ネットの値の差を出す。(b) の対はそのまま打ち続けて（今のターンの残りは両席出荷の設定・その先は審判）勝率の差も出す（``--shard``）。
4. ``gen``: 自己対戦（波 60 と同じ設定・r27）を打ち、相手の手番の窓（自分が応手を決める）で各候補を当てて窓を閉じた盤面を、
   自分の視点・その局の勝敗で「価値だけの行」として dump に書く（``--shard i/n``・seed は既存の波と重ならない帯）。
5. ``evalnet``: ネット（r27・対照・試験）ごとに、試験台の窓の決め方（今の窓の根畳み）が控えるを選ぶか・(3) の時点ごとのネットの差。
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import don_read as DRd  # noqa: E402
import matchup_5 as M5  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402
import window_read_all as W  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E, record_gen as RG  # noqa: E402

OUT_DIR = os.path.join("n1_results", "offdist_value")
NET = KR.NET
#: 配り方の番号のずらし（… main_read 12000／13000 と重ねない）。
OFFSET_TV = 14000
#: 生成の seed の帯（既存の波 2.6M〜2.8M・判定 4.1M〜4.3M と重ならない）。
GEN_SEED0 = 5_100_000
GEN_GAMES_PER_SHARD = 400
_G = {}


# ---------------------------------------------------------------- 1. count

IS_MY_TURN = 11
DON = slice(2, 6)  # 自アクティブ・自レスト・相手アクティブ・相手レスト（scalars の 2..6）


def _sig_class(sig):
    s = str(sig)
    if "COUNTER" in s or "BLOCK" in s or s.startswith('["PASS') or '"PASS"' in s:
        return "battle"
    return "effect"


def cmd_count(a):
    tot = {}
    nrow = 0
    for wd in a.waves:
        for d in sorted(glob.glob(os.path.join(os.path.expanduser(wd), "n_records", "*_p*"))):
            for f in sorted(glob.glob(os.path.join(d, "n_record_*.npz"))):
                z = np.load(f, allow_pickle=True)
                sc = z["scalars"].astype(np.float32)
                kind, who, seed, turn, step = z["kind"], z["who"], z["seed"], z["turn"], z["step"]
                sig = z["sig"]
                my = sc[:, IS_MY_TURN] > 0.5
                don_me = sc[:, 2] + sc[:, 3]
                order = np.lexsort((step, who, seed))
                prev_don = np.full(len(kind), np.nan)
                last = {}
                for i in order:
                    k = (int(seed[i]), int(who[i]), int(turn[i]))
                    if k in last:
                        prev_don[i] = don_me[last[k]]
                    last[k] = i
                paid = (~np.isnan(prev_don)) & (don_me < prev_don)
                for i in range(len(kind)):
                    if kind[i] == 0:
                        c = "main"
                    elif kind[i] == 2:
                        c = "commit"
                    else:
                        c = ("window_own_" if my[i] else "window_opp_") + _sig_class(sig[i])
                    key = c + ("|paid_don" if paid[i] else "")
                    tot[key] = tot.get(key, 0) + 1
                nrow += len(kind)
    out = {"rows": nrow, "counts": dict(sorted(tot.items())),
           "share": {k: round(v / nrow, 5) for k, v in sorted(tot.items())}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "counts.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 共通: 試験台の点

def kaido_points():
    """§7.66 の紫カイドウの打ち比べ済み 24 点（a＝CPU の手・b＝記事の手・diff は b − a）。"""
    pts = [p for p in BA.read_jsonl(os.path.join(DRd.OUT_DIR, "points.jsonl"))
           if p["judged"] and p["item"] != "mihawk_leader"]
    return pts


def _games():
    if "games" not in _G:
        _G["games"] = {"kr": KR.load_games(KR.OUT_DIR), "m5": DRd._m5_games()}
    return _G["games"]


def replay_point(p):
    g = _games()[p["src"]][p["idx"] if p["src"] == "kr" else (p["mu"], p["idx"])]
    return DRd.replay(p, g), g


def decline_move(p):
    """控える手（記事の手が控えるなら b・使うなら a）。"""
    return p["b"] if p["why"] in ("early", "big_next") else p["a"]


def accept_move(p):
    return p["a"] if p["why"] in ("early", "big_next") else p["b"]


def net_value_on(g, who):
    """プロセスの既定のネット（最初に読んだ npz）の生の値（`who` の視点）。"""
    enc = json.loads(g.encode(who))
    enc["tok"] = enc.pop("tokens", None) or []
    return float(json.loads(E.engine().net_eval(json.dumps(enc), "[]"))["value"])


def don_minus_one(g, seat, prefer):
    """自分の場のドンを 1 枚ドンデッキへ戻した複製（`prefer` の置き場を優先）。対話が立っていれば None。"""
    if g.has_interaction:
        return None, None
    h = json.loads(g.hidden_json())
    d = h["players"][seat]["don"]
    zone = prefer if d[prefer] else ("rested" if d["rested"] else ("active" if d["active"] else None))
    if zone is None:
        return None, None
    eng = E.engine()
    base = eng.Game.from_hidden(json.dumps(h))
    x = d[zone].pop()
    x["is_rest"] = False
    d["deck"].append(x)
    less = eng.Game.from_hidden(json.dumps(h))
    return base, less


# ---------------------------------------------------------------- 2. nn（分布の外か）

def _load_sc(paths, kinds=None, my=None, n=None, seed=0):
    """dump の scalars（float32）を集める。`kinds`／`my`（手番フラグ）で行を絞り、`n` 行を無作為に取る。"""
    xs = []
    for f in paths:
        z = np.load(f, allow_pickle=True)
        sc = z["scalars"].astype(np.float32)
        keep = np.ones(len(sc), bool)
        if kinds is not None:
            keep &= np.isin(z["kind"], kinds)
        if my is not None:
            keep &= (sc[:, IS_MY_TURN] > 0.5) == my
        xs.append(sc[keep])
    x = np.concatenate(xs) if xs else np.zeros((0, 127), np.float32)
    if n is not None and len(x) > n:
        x = x[np.random.default_rng(seed).choice(len(x), n, replace=False)]
    return x


def _nn_dist(q, ref, chunk=512):
    rr = (ref ** 2).sum(1)
    out = np.empty(len(q), np.float32)
    for s in range(0, len(q), chunk):
        qq = q[s:s + chunk]
        d2 = (qq ** 2).sum(1)[:, None] + rr[None, :] - 2.0 * qq @ ref.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def cmd_nn(a):
    """盤面の数の列（127）を標準化し、波の行の無作為 20 万行への最近傍の距離を、問いの群ごとに比べる。"""
    w60 = sorted(glob.glob(os.path.expanduser("~/w60/n_records/*_p*/n_record_*.npz")))
    w61 = sorted(glob.glob(os.path.expanduser("~/w61/n_records/*_p*/n_record_*.npz")))
    ref = _load_sc(w60, n=a.nn_ref, seed=1)
    mu, sd = ref.mean(0), ref.std(0) + 1e-3
    refz = (ref - mu) / sd
    folds = sorted(glob.glob(os.path.join(os.path.expanduser(a.gen_out), "fold_s*", "n_record_*.npz")))
    groups = {
        "fold_after_window（窓を閉じた直後）": _load_sc(folds, n=a.nn_q, seed=2),
        "main（自分の手番の判断点・別の波）": _load_sc(w61[:32], kinds=[0], n=a.nn_q, seed=3),
        "window_opp（相手の手番の窓の判断点・別の波）": _load_sc(w61[:32], kinds=[1], my=False, n=a.nn_q, seed=4),
        "window_own（自分の手番の窓の判断点・別の波）": _load_sc(w61[:32], kinds=[1], my=True, n=a.nn_q, seed=5),
    }
    out = {"ref_rows": len(ref), "groups": {}}
    for name, x in groups.items():
        if not len(x):
            continue
        d = _nn_dist((x - mu) / sd, refz)
        out["groups"][name] = {"n": len(x), "median": round(float(np.median(d)), 3), "mean": round(float(d.mean()), 3),
                               "p90": round(float(np.percentile(d, 90)), 3),
                               "my_turn_share": round(float((x[:, IS_MY_TURN] > 0.5).mean()), 3),
                               "my_don_active_mean": round(float(x[:, 2].mean()), 3),
                               "my_don_rested_mean": round(float(x[:, 3].mean()), 3)}
    # 波の中で「相手の手番」かつ「自分のドンが全部レスト」の行の割合（窓を閉じた直後の典型の形）
    allx = _load_sc(w60)
    opp = allx[:, IS_MY_TURN] < 0.5
    out["wave60_rows"] = int(len(allx))
    out["wave60_opp_turn_share"] = round(float(opp.mean()), 4)
    fx = groups["fold_after_window（窓を閉じた直後）"]
    out["fold_opp_turn_share"] = round(float((fx[:, IS_MY_TURN] < 0.5).mean()), 4) if len(fx) else None
    json.dump(out, open(os.path.join(a.out, "nn.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def _load_rows(paths, kinds=None, my=None, n=None, seed=0):
    """dump の (scalars, card_idx 22 枠, tokens) を集める（`_load_sc` と同じ絞り方）。"""
    S, C, T = [], [], []
    for f in paths:
        z = np.load(f, allow_pickle=True)
        sc = z["scalars"].astype(np.float32)
        keep = np.ones(len(sc), bool)
        if kinds is not None:
            keep &= np.isin(z["kind"], kinds)
        if my is not None:
            keep &= (sc[:, IS_MY_TURN] > 0.5) == my
        S.append(sc[keep])
        C.append(z["card_idx"][keep][:, :22].astype(np.int64))
        T.append(z["tokens"][keep].astype(np.float32))
    S, C, T = np.concatenate(S), np.concatenate(C), np.concatenate(T)
    if n is not None and len(S) > n:
        i = np.random.default_rng(seed).choice(len(S), n, replace=False)
        S, C, T = S[i], C[i], T[i]
    return S, C, T


def cmd_nn_embed(a):
    """ネット（r27）の共有表現（価値の頭の手前のベクトル）で同じ最近傍の距離を見る（`nn` の併記）。"""
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    net = NL.NRelNet.load(NET, tables=(stats, ab, abm, pwr, isl))
    rt = NR.RelTable(NR.profile_table(load_db(), vocab))

    def emb(S, C, T, bs=512):
        out = []
        tab = net.card_table()
        for s0 in range(0, len(S), bs):
            sc, ci, tok = S[s0:s0 + bs], C[s0:s0 + bs], T[s0:s0 + bs]
            rom, roo = NR.relations_batch(ci, tok, rt)
            h, present = net.tokens_forward(ci, tok, rom, roo, tab)
            out.append(net.body(sc, h, present))
        return np.concatenate(out)

    w60 = sorted(glob.glob(os.path.expanduser("~/w60/n_records/*_p*/n_record_*.npz")))
    w61 = sorted(glob.glob(os.path.expanduser("~/w61/n_records/*_p*/n_record_*.npz")))
    ref = emb(*_load_rows(w60, n=a.nn_ref, seed=1))
    mu, sd = ref.mean(0), ref.std(0) + 1e-3
    refz = (ref - mu) / sd
    folds = sorted(glob.glob(os.path.join(os.path.expanduser(a.gen_out), "fold_s*", "n_record_*.npz")))
    groups = {
        "fold_after_window（窓を閉じた直後）": _load_rows(folds, n=a.nn_q, seed=2),
        "main（自分の手番の判断点・別の波）": _load_rows(w61[:32], kinds=[0], n=a.nn_q, seed=3),
        "window_opp（相手の手番の窓の判断点・別の波）": _load_rows(w61[:32], kinds=[1], my=False, n=a.nn_q, seed=4),
        "window_own（自分の手番の窓の判断点・別の波）": _load_rows(w61[:32], kinds=[1], my=True, n=a.nn_q, seed=5),
    }
    out = {"net": NET, "ref_rows": len(ref), "dim": int(ref.shape[1]), "groups": {}}
    for name, (S, C, T) in groups.items():
        e = emb(S, C, T)
        d = _nn_dist((e - mu) / sd, refz)
        out["groups"][name] = {"n": len(e), "median": round(float(np.median(d)), 3), "mean": round(float(d.mean()), 3),
                               "p90": round(float(np.percentile(d, 90)), 3)}
    json.dump(out, open(os.path.join(a.out, "nn_embed.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 3. timeval

def _init_tv():
    E.engine()
    if _G.get("net"):
        E.load_net(_G["net"])  # 既定のネット（net_eval が使う）を先に読む
    W._boot()
    _games()


def own_turn_head(p, g):
    """記録の手の列で、点の直前の自分の手番の頭（自分の最初の main の判断）の盤面（無ければ None）。"""
    pt_game, _ = replay_point(p)
    t = int(pt_game.turn_count or 0)
    t_target = t if pt_game.turn_player() == "p1" else t - 1
    gm = KR.new_game(g["idx"]) if p["src"] == "kr" else M5.new_game(g["mu"], g["idx"])
    for s, mv in enumerate(g["moves"]):
        pend = json.loads(gm.pending_json())
        if int(gm.turn_count or 0) == t_target and gm.turn_player() == "p1" and pend.get("player_id") == "p1" \
                and not gm.has_interaction and pend.get("action") == "MAIN_ACTION":
            return gm
        if s >= p["step"]:
            break
        BA.apply(gm, pend["player_id"], mv)
    return None


def play_to(g, who, ws, stop):
    """両席とも出荷の設定で打ち進め、`stop(g, pend, mv)` が真になった盤面（手を当てる前）で止める。"""
    ship = W._G["ship"]
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps = 0
    while g.winner() is None and steps < 300:
        pend = json.loads(g.pending_json())
        if not pend:
            return g
        name = pend["player_id"]
        turn = int(g.turn_count or 0)
        if stop(g, pend, None):
            return g
        c = carries[name]
        out = json.loads(g.decide(name, ship.decide_opts(int(ws), turn, name, c.get(turn, name))))
        c.put(out)
        mv = out.get("move")
        if mv is None:
            return g
        if stop(g, pend, mv):
            return g
        BA.apply(g, name, mv)
        steps += 1
    return g


def tv_point(p):
    t0 = time.time()
    row = {"pid": p["pid"], "diff": p["diff"], "why": p["why"], "ok": False}
    who = "p1"
    try:
        game, g = replay_point(p)
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = DRd.game_seed(p)
    dec = decline_move(p)
    res = {"a_head": [], "b_close": [], "c_oppend": [], "d_next": []}
    roll = {"base": [], "less": []}
    # (a) 自分の手番の頭（記録の手の列から・世界は引かない）
    try:
        ga = own_turn_head(p, g)
        if ga is not None:
            b, l = don_minus_one(ga, who, "active")
            if b is not None:
                res["a_head"].append(net_value_on(b, who) - net_value_on(l, who))
    except Exception as exc:  # noqa: BLE001
        row["a_err"] = str(exc)[:100]
    t_now = int(game.turn_count or 0)
    target = t_now + (2 if game.turn_player() == who else 1)
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, p["step"], OFFSET_TV + i)
        try:
            g1 = game.fork_world(who, "fair", int(ws))
            g1.apply_search_move(who, json.dumps(dec, ensure_ascii=False))
            W.settle(g1, who, ws)
            # (b) 窓を閉じた直後
            b, l = don_minus_one(g1, who, "rested")
            if b is not None:
                res["b_close"].append(net_value_on(b, who) - net_value_on(l, who))
                if i < _G["k_roll"]:
                    roll["base"].append(_rollout_state(b, who, ws))
                    roll["less"].append(_rollout_state(l, who, ws))
            # (c) 相手の手番の終わり（相手がターン終了を打つ直前）
            gc = play_to(g1.fork(), who, ws,
                         lambda gg, pend, mv: mv is not None and pend["player_id"] != who
                         and mv.get("action_type") == "TURN_END")
            if gc.winner() is None:
                b, l = don_minus_one(gc, who, "rested")
                if b is not None:
                    res["c_oppend"].append(net_value_on(b, who) - net_value_on(l, who))
            # (d) 次の自分の手番の頭
            gd = play_to(g1.fork(), who, ws,
                         lambda gg, pend, mv: mv is None and int(gg.turn_count or 0) >= target
                         and pend["player_id"] == who and gg.turn_player() == who and not gg.has_interaction)
            if gd.winner() is None:
                b, l = don_minus_one(gd, who, "active")
                if b is not None:
                    res["d_next"].append(net_value_on(b, who) - net_value_on(l, who))
        except Exception as exc:  # noqa: BLE001
            row.setdefault("errs", []).append("%s: %s" % (type(exc).__name__, str(exc)[:100]))
    row["res"] = res
    row["roll"] = dict(MJ.paired(roll["less"], roll["base"]), base=roll["base"], less=roll["less"])
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def _rollout_state(g, who, ws):
    """盤面 `g`（手番の途中・対話なし）から、今のターンの残りは両席出荷の設定・その先は審判で最後まで → 1／0／None。"""
    gg = g.fork()
    ship, ref = W._G["ship"], W._G["ref"]
    t0 = int(gg.turn_count or 0)
    carries = {}
    steps = 0
    try:
        while gg.winner() is None and steps < MJ.MAX_STEPS:
            pend = json.loads(gg.pending_json())
            if not pend:
                break
            name = pend["player_id"]
            turn = int(gg.turn_count or 0)
            use_ship = turn == t0
            spec = ship if use_ship else ref
            c = carries.setdefault((name, use_ship), DR.Carry())
            out = json.loads(gg.decide(name, spec.decide_opts(int(ws), turn, name, c.get(turn, name))))
            c.put(out)
            mv = out.get("move")
            if mv is None:
                break
            BA.apply(gg, name, mv)
            steps += 1
    except Exception:  # noqa: BLE001
        return None
    w = gg.winner()
    return None if w is None else (1 if w == who else 0)


def _init_tv_k(k, k_roll, net=None):
    _G["net"] = net
    _init_tv()
    _G["k"], _G["k_roll"] = int(k), int(k_roll)


def cmd_timeval(a):
    pts = kaido_points()
    si, sn = KR.shard_of(a)
    jobs = [p for n, p in enumerate(pts) if n % sn == si]
    pre = "timeval" if not a.tv_net else "timeval_" + a.tag
    rpath = os.path.join(a.out, f"{pre}_shard{si}.jsonl" if a.shard else f"{pre}.jsonl")
    done = W._done(a.out, pre)
    jobs = [p for p in jobs if p["pid"] not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f"TIMEVAL shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} roll={a.k_roll} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_tv_k, initargs=(a.k, a.k_roll, a.tv_net)) as pool:
        for i, r in enumerate(pool.imap_unordered(tv_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} ok={r['ok']} roll={(r.get('roll') or {}).get('diff')} "
                  f"sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 4. gen

class _FoldRecorder(RG._Recorder):
    """生成の観測に、相手の手番の窓での「各候補を当てて窓を閉じた盤面」を足す（打ち方は変えない）。"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.folds = []
        _G["last_rec"] = self

    def __call__(self, game, name, turn, step, out, move):
        super().__call__(game, name, turn, step, out, move)
        try:
            if out.get("kind") == "window" and game.turn_player() != name:
                legal = (out.get("stats") or {}).get("legal") or []
                if len(legal) >= 2:
                    for j, mv in enumerate(legal):
                        g = game.fork()
                        g.apply_search_move(name, json.dumps(mv, ensure_ascii=False))
                        W.settle(g, name, int(self.seed) * 7919 + step)
                        if g.winner() is not None:
                            continue
                        enc = json.loads(g.encode(name))
                        self.folds.append((name, int(turn), int(step), j, enc))
        except Exception:  # noqa: BLE001  1 点の失敗で局を止めない
            pass


def _init_gen():
    RG._init_worker(1280, NET, 0.25, 4, decks="real", aux=False, eps_play=0.03, eps_hold=0.03,
                    search={"worlds": 1, "select_rule": "q_min_n", "q_min_frac": 0.125,
                            "root_prior_temp": 2.0, "budget": 0})
    W._boot()
    RG._Recorder = _FoldRecorder


def gen_one(seed):
    _G["last_rec"] = None
    r = RG.play_one(seed)
    rec = _G.get("last_rec")
    if r is None or rec is None or not rec.folds:
        return {"seed": seed, "rows": 0, "ok": r is not None}
    winner = r["_meta"]["winner"]
    rows = {"scalars": [], "field": [], "card_idx": [], "tokens": [], "z": [], "who": [], "turn": [], "step": [],
            "cand": []}
    for name, turn, step, j, enc in rec.folds:
        rows["scalars"].append(np.asarray(enc["scalars"], np.float32))
        rows["field"].append(np.asarray(enc["field"], np.float32).reshape(RG.FIELD_SHAPE))
        ci = np.zeros(RG.MAX_CI, np.int64)
        src = np.asarray(enc["card_idx"], np.int64)[:RG.MAX_CI]
        ci[:len(src)] = src
        rows["card_idx"].append(ci)
        rows["tokens"].append(np.asarray(enc["tokens"], np.float32).reshape(RG.TOKENS_SHAPE))
        rows["z"].append(1.0 if winner == name else -1.0)
        rows["who"].append(0 if name == "p1" else 1)
        rows["turn"].append(turn)
        rows["step"].append(step)
        rows["cand"].append(j)
    n = len(rows["z"])
    out = {"scalars": np.array(rows["scalars"]).astype(np.float16),
           "field": np.array(rows["field"], np.float32),
           "card_idx": np.array(rows["card_idx"]).astype(np.int16),
           "tokens": np.array(rows["tokens"]).astype(np.float16),
           "z": np.array(rows["z"], np.float32), "who": np.array(rows["who"], np.int8),
           "kind": np.full(n, 1, np.int8), "turn": np.array(rows["turn"], np.int16),
           "step": np.array(rows["step"], np.int32), "seed": np.full(n, seed, np.int64),
           "sig": np.array([""] * n), "pol_len": np.zeros(n, np.int32), "pol_chosen": np.full(n, -1, np.int16),
           "pol_n": np.zeros(0, np.float32), "pol_q": np.zeros(0, np.float32), "pol_k": np.zeros(0, np.int16),
           "pol_sig": np.array([], dtype="U1"), "pol_cid": np.array([], dtype="U1"), "pol_tcid": np.array([], dtype="U1"),
           "pol_p": np.zeros(0, np.float16), "pol_si": np.zeros(0, np.int16), "pol_ti": np.zeros(0, np.int16),
           "deck_kinds": np.array(["{}"] * n), "forced": np.zeros(n, np.int8), "forced_sig": np.array([""] * n),
           "pol_v0": np.zeros(n, np.float16), "fold_cand": np.array(rows["cand"], np.int16)}
    return {"seed": seed, "rows": n, "ok": True, "winner": winner, "base_rows": int(len(r["z"])), "data": out}


def cmd_gen(a):
    si, sn = KR.shard_of(a)
    seeds = [GEN_SEED0 + si * 10000 + i for i in range(a.games)]
    out_dir = os.path.join(os.path.expanduser(a.gen_out), f"fold_s{si}")
    os.makedirs(out_dir, exist_ok=True)
    prog = os.path.join(out_dir, "progress.jsonl")
    done = {x["seed"] for x in BA.read_jsonl(prog)} if os.path.exists(prog) else set()
    seeds = [s for s in seeds if s not in done]
    print(f"GEN shard={a.shard} todo {len(seeds)} (done {len(done)}) -> {out_dir}", flush=True)
    t0 = time.time()
    buf, nfile = [], len(glob.glob(os.path.join(out_dir, "n_record_*.npz")))
    with get_context("fork").Pool(a.workers, initializer=_init_gen) as pool:
        for i, r in enumerate(pool.imap_unordered(gen_one, seeds, chunksize=1)):
            if r.get("data") is not None:
                buf.append(r.pop("data"))
            if len(buf) >= 10 or (i + 1 == len(seeds) and buf):
                keys = buf[0].keys()
                path = os.path.join(out_dir, f"n_record_{nfile:05d}.npz")
                np.savez_compressed(path, **{k: np.concatenate([b[k] for b in buf]) for k in keys})
                nfile += 1
                buf = []
                # progress は書いたものだけ（落ちたら書いていない局はやり直す）
            with open(prog, "a") as f:
                f.write(json.dumps({k: v for k, v in r.items() if k != "data"}) + "\n")
            print(f"[{i + 1}/{len(seeds)}] seed={r['seed']} rows={r['rows']} ({time.time() - t0:.0f}s)", flush=True)
    meta = {"rows": sum(int(np.load(f)["z"].shape[0]) for f in sorted(glob.glob(os.path.join(out_dir, "n_record_*.npz")))),
            "kind": "fold_value_only", "enc_version": 15, "dump_version": 5, "net": NET, "seed0": GEN_SEED0 + si * 10000}
    json.dump(meta, open(os.path.join(out_dir, "meta_n_record.json"), "w"))
    print("GEN done", meta, flush=True)


# ---------------------------------------------------------------- 5. evalnet

def _init_ev(nets):
    W._boot()
    _games()
    for n in nets:
        E.load_net(n)
    _G["nets"] = nets


def ev_point(p):
    row = {"pid": p["pid"], "diff": p["diff"], "why": p["why"], "ok": False, "nets": {}}
    try:
        game, _g = replay_point(p)
    except Exception as exc:  # noqa: BLE001
        row["fail"] = str(exc)[:80]
        return row
    seed = DRd.game_seed(p)
    turn = int(game.turn_count or 0)
    dec = decline_move(p)
    for net in _G["nets"]:
        spec = BA.ship_spec(net)
        out = json.loads(game.fork().decide("p1", spec.decide_opts(seed, turn, "p1", DR.Carry().get(turn, "p1"))))
        rows = BA.ranked(out)
        q = {}
        for nm, mv in (("decline", dec), ("accept", accept_move(p))):
            hit = [r for r in rows if BA.key(r["mv"]) == BA.key(mv)]
            q[nm] = hit[0]["q"] if hit else None
        chose = "decline" if BA.key(out.get("move")) == BA.key(dec) else "accept"
        row["nets"][net] = {"q": q, "chose": chose}
    row["ok"] = True
    return row


def cmd_evalnet(a):
    pts = kaido_points()
    rpath = os.path.join(a.out, a.tag + "_evalnet.jsonl")
    with get_context("fork").Pool(a.workers, initializer=_init_ev, initargs=(a.nets,)) as pool:
        rows = list(pool.imap_unordered(ev_point, pts, chunksize=1))
    with open(rpath, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for net in a.nets:
        ok = [r for r in rows if r["ok"]]
        dec_side = [r for r in ok if r["why"] in ("early", "big_next")]
        use_side = [r for r in ok if r["why"] not in ("early", "big_next")]
        n_dec = sum(1 for r in ok if r["nets"][net]["chose"] == "decline")
        print(net, "points", len(ok), "chose_decline", n_dec,
              "| article_decline", len(dec_side), "chose_decline", sum(1 for r in dec_side if r["nets"][net]["chose"] == "decline"),
              "| article_use", len(use_side), "chose_decline", sum(1 for r in use_side if r["nets"][net]["chose"] == "decline"),
              "| early chose_decline", sum(1 for r in ok if r["why"] == "early" and r["nets"][net]["chose"] == "decline"),
              "of", sum(1 for r in ok if r["why"] == "early"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["count", "nn", "nn_embed", "timeval", "gen", "evalnet"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--waves", nargs="*", default=["~/w60", "~/w61"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--k-roll", dest="k_roll", type=int, default=16)
    ap.add_argument("--games", type=int, default=GEN_GAMES_PER_SHARD)
    ap.add_argument("--gen-out", default="~/fold_rows")
    ap.add_argument("--nets", nargs="*", default=[NET])
    ap.add_argument("--nn-ref", dest="nn_ref", type=int, default=200000)
    ap.add_argument("--nn-q", dest="nn_q", type=int, default=2000)
    ap.add_argument("--tag", default="r27")
    ap.add_argument("--tv-net", dest="tv_net", default=None, help="timeval のネットの値に使うネット（既定 r27・打ち続けは常に r27）")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"count": cmd_count, "nn": cmd_nn, "nn_embed": cmd_nn_embed, "timeval": cmd_timeval, "gen": cmd_gen, "evalnet": cmd_evalnet}[a.cmd](a)


if __name__ == "__main__":
    main()
