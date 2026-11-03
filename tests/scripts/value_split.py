"""WP `value-split`（`docs/search_efficiency_plan.md` §7.74・台帳 2+506）の計測 A: ドン 1 枚の差を
「教材の打ち方」で最後まで打ち続けたとき、勝敗にどれだけ残るか（物差し＝従来の打ち続けと対で比べる）。

読み取りだけ（エンジン・出荷・訓練器・既存の道具は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。

式（§7.74）: ネットの差 ＝（教材の打ち方での打ち続けた差）×（当てはめの縮み）。ここで測るのは右の 1 項目。

1. ``play``: §7.66 の紫カイドウ 24 点（`offdist_value.kaido_points`）の「窓を閉じた直後」のドン 1 枚違いの対
   （`offdist_value.tv_point` と同じ作り・`don_minus_one(…, "rested")`）を、配り直した世界 32 本（ずらし 17000）で作り、
   同じ対を次の 4 通りで最後まで打つ（世界ごとに対＝同じ盤面・同じ配り方・違うのは打ち続けの設定だけ）:
   - ``ref``        … (i) 従来の物差し: 今のターンの残りは両席出荷の設定・その先は審判 r27・160 回（`offdist_value._rollout_state`）。
   - ``teach_r29f4`` … (ii) r29f4 の教材の打ち方: 打ち手 r28x4・生成役の設定・後半の散らし 4 割（`gen_n70f.json`）。
   - ``teach_r28x4`` … (ii) r28x4 の教材の打ち方: 打ち手 r27・散らし 2 割（`gen_n62x.json`）。
   - ``teach_r27``   … (ii) r27 の教材の打ち方: 打ち手 r26・散らしなし（`gen_n56.json`）。
   (ii) は両席とも `record_gen._init_worker` が組む `SeatSpec`（1,280 回・世界 1 本・枝予算 0・q_min_n 1/8・根の事前分布の温度 2.0・
   根のノイズ 0.25・最初の 4 ターンの温度・後半の散らし）で最後まで打ち、ε 3% の差し替え（`_Recorder.swap`＝`eps_swap`）も
   そのまま使う（観測＝行の記録はしない）。あわせて同じ対の盤面での各ネットの値の差（numpy の forward）も残す。
   ``--shard k/16``（k＝1〜16・零詰めなし）・``--mode`` で 4 通りを分けられる。``--bench rocks`` で段 2（6キングだけ 53 点・
   a＝CPU の手・b＝戻す・`rocks_bounce.rollout` と同じ形で対象を 6キングに固定）。
2. ``summary``: 結果ブランチから集めた行（`n1_results/value_split/rows_*.jsonl`）を世代ごとにまとめる——ネットの差（保存済み盤面
   `n1_results/value_teach/tv_states/`・`value_teach tvval` の形）・(i) の差・(ii) の差・比 (i)・比 (ii)・点ごとの相関・(ii) の勝率そのもの。

    OPCG_LOG_SILENT=1 python tests/scripts/value_split.py play --k 32 --workers 4 --shard 1/16
    OPCG_LOG_SILENT=1 python tests/scripts/value_split.py play --bench rocks --k 16 --mode teach_r29f4 --shard 1/8
    OPCG_LOG_SILENT=1 python tests/scripts/value_split.py summary
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import don_read as DRd  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402
import offdist_value as OV  # noqa: E402
import value_teach as VT  # noqa: E402
import window_read_all as W  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E, record_gen as RG  # noqa: E402

OUT_DIR = os.path.join("n1_results", "value_split")
#: 配り方の番号のずらし（offdist_value 14000・rocks_bounce 15000・value_teach 16000 と重ねない）。
OFFSET = 17000
K_DEFAULT = 32
N_SHARDS = 16
LEARNED = os.path.join("opcg_sim", "data", "learned")
NETS = {n: os.path.join(LEARNED, f"nrel_{n}.npz") for n in ("r26", "r27", "r28x4", "r29f4")}
#: 生成役の探索の設定（`gen_n56.json`／`gen_n62x.json`／`gen_n70f.json` の `args` と同じ）。
GEN_SIMS, GEN_DIRICHLET, GEN_TEMP_TURNS, GEN_EPS = 1280, 0.25, 4, 0.03
GEN_SEARCH = {"worlds": 1, "select_rule": "q_min_n", "q_min_frac": 0.125, "root_prior_temp": 2.0, "budget": 0}
#: (ii) の 3 世代: 鍵＝そのネットの教材を打った設定（打ち手・後半の散らし）。
TEACH = {"teach_r29f4": {"player": "r28x4", "temp_late": 0.4, "waves": "70f〜73f"},
         "teach_r28x4": {"player": "r27", "temp_late": 0.2, "waves": "62x〜65x"},
         "teach_r27": {"player": "r26", "temp_late": 0.0, "waves": "56＋57"}}
MODES = ("ref",) + tuple(TEACH)
_G = {}


# ---------------------------------------------------------------- 生成役の席

def teach_spec(mode):
    """`record_gen._init_worker` が組むのと同じ `SeatSpec`（ジョブ定義の値・ε は `SeatSpec` の欄に載る）。"""
    t = TEACH[mode]
    explore = {"z": "game", "temp_late": t["temp_late"], "temp_late_pow": 1.0} if t["temp_late"] > 0 else None
    RG._init_worker(GEN_SIMS, NETS[t["player"]], GEN_DIRICHLET, GEN_TEMP_TURNS, decks="real", aux=False,
                    eps_play=GEN_EPS, eps_hold=GEN_EPS, search=dict(GEN_SEARCH), explore=explore)
    return RG._G["spec"]


def play_teach(g, who, ws, spec, k6=None, t0=None):
    """盤面 `g` から両席とも生成役の設定（`spec`）で最後まで。ε の差し替えは `_Recorder.swap` をそのまま使う（観測はしない）。
    `k6`（段 2）: 今のターンの中で手の直後にロックス海賊団の対象の問いが来たら 6キングを選ぶ（`rocks_bounce.rollout` と同じ）。
    → (1／0／None, 内訳)。"""
    gg = g.fork()
    rec = RG._Recorder(seed=int(ws), db=RG._G["db"], eps_play=spec.eps_play, eps_hold=spec.eps_hold,
                       eps_target=spec.eps_target)
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    info = {"steps": 0, "forced": 0, "late": 0, "k6": False, "err": None}
    try:
        while gg.winner() is None and info["steps"] < MJ.MAX_STEPS:
            pend = json.loads(gg.pending_json())
            if not pend:
                break
            name = pend["player_id"]
            turn = int(gg.turn_count or 0)
            if (k6 and not info["k6"] and name == who and turn == t0 and pend.get("action") == "SEARCH_AND_SELECT"
                    and "ロックス海賊団" in (pend.get("message") or "")):
                info["k6"] = True
                BA.apply(gg, name, {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                                    "payload": {"selected_uuids": [k6], "index": 0, "accepted": True,
                                                "position": "BOTTOM", "declared_value": 0}})
                info["steps"] += 1
                continue
            c = carries[name]
            out = json.loads(gg.decide(name, spec.decide_opts(int(ws), turn, name, c.get(turn, name))))
            mv = out.get("move")
            if mv is None:
                break
            if (out.get("temp_pick") or {}).get("explored"):
                info["late"] += 1
            # ε（§20.8.5）: `swap` は観測（`__call__`）が引いた uuid → card_id を使う＝ここで同じものを渡す。
            rec._cids = None
            if out.get("kind") == "main" and len(out.get("groups") or []) >= 2:
                rec._cids = json.loads(gg.dump_index_json(name))["cids"]
            new = rec.swap(gg, name, turn, info["steps"], out, mv)
            if new is not mv:
                info["forced"] += 1
            c.put(out)                          # `run_game` と同じ＝差し替えで捨てた箱の残り手順は持ち越さない
            BA.apply(gg, name, new)
            info["steps"] += 1
    except Exception as exc:  # noqa: BLE001
        info["err"] = "%s: %s" % (type(exc).__name__, str(exc)[:100])
        return None, info
    info["turns"] = int(gg.turn_count or 0)
    w = gg.winner()
    return (None if w is None else (1 if w == who else 0)), info


def play_ref(g, who, ws, k6=None, t0=None):
    """(i) 従来の物差し（`offdist_value._rollout_state`＝今のターンの残りは両席出荷の設定・その先は審判 r27・160 回）。"""
    if not k6:
        return OV._rollout_state(g, who, ws), {}
    # 段 2: `rocks_bounce.rollout` と同じ形（手は当ててある＝ここでは対象の固定だけ足す）。
    gg = g.fork()
    ship, ref = W._G["ship"], W._G["ref"]
    carries, steps, info = {}, 0, {"k6": False}
    try:
        while gg.winner() is None and steps < MJ.MAX_STEPS:
            pend = json.loads(gg.pending_json())
            if not pend:
                break
            name = pend["player_id"]
            turn = int(gg.turn_count or 0)
            if (not info["k6"] and name == who and turn == t0 and pend.get("action") == "SEARCH_AND_SELECT"
                    and "ロックス海賊団" in (pend.get("message") or "")):
                info["k6"] = True
                BA.apply(gg, name, {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                                    "payload": {"selected_uuids": [k6], "index": 0, "accepted": True,
                                                "position": "BOTTOM", "declared_value": 0}})
                steps += 1
                continue
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
        return None, info
    w = gg.winner()
    return (None if w is None else (1 if w == who else 0)), info


def state_key(g):
    return hashlib.sha1(g.hidden_json().encode("utf-8")).hexdigest()[:12]


def net_diff(gb, gl, who):
    """同じ対の盤面での各ネットの値の差（そのまま − 1 枚少ない・numpy の forward）。"""
    eb, el = VT.encode(gb, who), VT.encode(gl, who)
    out = {}
    for n, path in NETS.items():
        vb = VT.net_values(path, eb[0][None], eb[1][None], eb[2][None])
        vl = VT.net_values(path, el[0][None], el[1][None], el[2][None])
        out[n] = round(float(vb[0] - vl[0]), 4)
    return out


# ---------------------------------------------------------------- 1. play（紫カイドウ 24 点）

def _init_play(modes, bench):
    OV._init_tv_k(0, 0, None)                   # エンジン・r27（出荷の設定・審判）・記録の局
    _G["specs"] = {m: teach_spec(m) for m in modes if m != "ref"}
    _G["modes"] = list(modes)
    if bench == "rocks":
        _G["games"] = W.load_all()               # 段 2 の記録（kaido_rocks）


def kaido_pair(p, i):
    """点 `p` の i 本目の世界の、窓を閉じた直後のドン 1 枚違いの対（`offdist_value.tv_point` の (b) と同じ作り）。"""
    who = "p1"
    game, _g = OV.replay_point(p)
    seed = DRd.game_seed(p)
    ws = MJ.world_seed(seed, p["step"], OFFSET + i)
    g1 = game.fork_world(who, "fair", int(ws))
    g1.apply_search_move(who, json.dumps(OV.decline_move(p), ensure_ascii=False))
    W.settle(g1, who, ws)
    if g1.winner() is not None:
        return None, None, ws, who, None
    b, l = OV.don_minus_one(g1, who, "rested")
    return b, l, ws, who, None


def rocks_pair(p, i):
    """段 2: 6キングだけの点 `p` の i 本目の世界で、a（CPU の手）と b（戻す）を当てた盤面の対（対象の固定は打ち続けの中で）。"""
    who = "p2"
    game = W.replay({"g": list(VT.rocks_key(p)), "step": p["step"]}, _G["games"][VT.rocks_key(p)])
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != who or int(game.turn_count or 0) != p["turn"]:
        raise RuntimeError("seat/turn")
    ws = MJ.world_seed(p["seed"], p["step"], OFFSET + i)
    ga = game.fork_world(who, "fair", int(ws))
    ga.apply_search_move(who, json.dumps(p["a"], ensure_ascii=False))
    gb = game.fork_world(who, "fair", int(ws))
    gb.apply_search_move(who, json.dumps(p["b"], ensure_ascii=False))
    return ga, gb, ws, who, p["k6_uuid"]


def point_id(p, bench):
    """行に書く点の鍵（紫カイドウ＝pid・段 2＝src/idx/step）。"""
    if bench == "rocks":
        return {"src": p["src"], "idx": p["idx"], "step": p["step"], "ot": p["ot"]}
    return {"pid": p["pid"], "why": p["why"]}


def play_job(job):
    """1 つの (点, 世界) を、まだ済んでいない全部の mode で打つ → mode ごとの行。"""
    p, i, bench, todo = job
    t0 = time.time()
    rows = []
    pid = point_id(p, bench)
    try:
        if bench == "rocks":
            g_a, g_b, ws, who, k6 = rocks_pair(p, i)
        else:
            g_a, g_b, ws, who, k6 = kaido_pair(p, i)
    except Exception as exc:  # noqa: BLE001
        return [dict(pid, i=i, mode=m, ok=False, fail="pair: %s: %s" % (type(exc).__name__, str(exc)[:80])) for m in todo]
    if g_a is None:
        return [dict(pid, i=i, mode=m, ok=False, fail="ended") for m in todo]
    turn0 = int(g_a.turn_count or 0)
    key = [state_key(g_a), state_key(g_b)]
    nd = None
    try:
        nd = net_diff(g_a, g_b, who) if bench != "rocks" else None
    except Exception as exc:  # noqa: BLE001
        nd = {"err": str(exc)[:80]}
    for m in todo:
        t1 = time.time()
        # a＝そのまま（紫カイドウ）／CPU の手（段 2）、b＝1 枚少ない／戻す。差は常に b − a で持つ。
        if m == "ref":
            ra, ia = play_ref(g_a, who, ws, k6, turn0)
            rb, ib = play_ref(g_b, who, ws, k6, turn0)
        else:
            ra, ia = play_teach(g_a, who, ws, _G["specs"][m], k6, turn0)
            rb, ib = play_teach(g_b, who, ws, _G["specs"][m], k6, turn0)
        rows.append(dict(pid, i=i, ws=int(ws), mode=m, ok=ra is not None and rb is not None, a=ra, b=rb,
                         info_a=ia, info_b=ib, key=key, net=nd, turn0=turn0, sec=round(time.time() - t1, 1)))
    rows[-1]["sec_pair"] = round(time.time() - t0, 1)
    return rows


def _rows_path(out, bench, shard):
    tag = "rows" if bench != "rocks" else "rocks"
    return os.path.join(out, f"{tag}_s{int(shard.split('/')[0]):02d}.jsonl" if shard else f"{tag}_local.jsonl")


def _row_key(r):
    return (r.get("pid"), r.get("src"), r.get("idx"), r.get("step"), r["i"], r["mode"])


def load_rows(out, bench="kaido"):
    tag = "rows" if bench != "rocks" else "rocks"
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, f"{tag}_*.jsonl"))):
        for r in BA.read_jsonl(f):
            if r.get("ok"):
                rows[_row_key(r)] = r
    return rows


def cmd_play(a):
    modes = MODES if a.mode == "all" else tuple(a.mode.split(","))
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"--mode は {','.join(MODES)} か all: {m}")
    pts = OV.kaido_points() if a.bench != "rocks" else VT.rocks_points()
    if a.limit:
        pts = pts[:a.limit]
    jobs = [(p, i) for p in pts for i in range(a.k)]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        if not 1 <= si <= sn:
            raise SystemExit("--shard は k/n（k＝1〜n）")
        jobs = [j for n, j in enumerate(jobs) if n % sn == si - 1]
    os.makedirs(a.out, exist_ok=True)
    rpath = _rows_path(a.out, a.bench, a.shard)
    done = set()
    for f in glob.glob(os.path.join(a.out, ("rows" if a.bench != "rocks" else "rocks") + "_*.jsonl")):
        for r in BA.read_jsonl(f):
            if r.get("ok"):
                done.add(_row_key(r))
    todo = []
    for p, i in jobs:
        ms = [m for m in modes if _row_key(dict(point_id(p, a.bench), i=i, mode=m)) not in done]
        if ms:
            todo.append((p, i, a.bench, ms))
    print(f"PLAY bench={a.bench} shard={a.shard} modes={','.join(modes)} k={a.k} jobs {len(todo)} (of {len(jobs)}) "
          f"-> {rpath}", flush=True)
    t0 = time.time()
    n_rows = 0
    with get_context("fork").Pool(a.workers, initializer=_init_play, initargs=(modes, a.bench)) as pool:
        for n, rows in enumerate(pool.imap_unordered(play_job, todo, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += len(rows)
            r0 = rows[0]
            tag = r0.get("pid", f"{r0.get('src')}:{r0.get('idx')}:{r0.get('step')}")
            cells = " ".join("%s=%s(%ss)" % (r["mode"], ("%s/%s" % (r["a"], r["b"])) if r.get("ok") else "x", r.get("sec"))
                             for r in rows)
            print(f"[{n + 1}/{len(todo)}] pt={tag} i={r0['i']} {cells} pair={rows[-1].get('sec_pair')}s "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"PLAY_DONE rows {n_rows} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- 2. summary

def _tv_point_net_diff(tv_dir, t="b_close"):
    """保存済み盤面（`value_teach tvstates`・24 点 × 世界 8 本）から、点ごとの各ネットのドン 1 枚の値の差（価値の目盛り）。"""
    out = {n: {} for n in NETS}
    for f in sorted(glob.glob(os.path.join(tv_dir, "pid*.npz"))):
        pid = int(os.path.basename(f)[3:6])
        d = dict(np.load(f, allow_pickle=True))
        m = d["t"] == t
        if not m.any():
            continue
        for n, path in NETS.items():
            dv = VT.net_values(path, d["sc_base"][m], d["ci_base"][m], d["tok_base"][m]) \
                - VT.net_values(path, d["sc_less"][m], d["ci_less"][m], d["tok_less"][m])
            out[n][pid] = float(dv.mean())
    return out


def _ci(xs):
    xs = np.asarray(xs, float)
    n = len(xs)
    if n < 2:
        return [None, None]
    h = 1.96 * xs.std(ddof=1) / np.sqrt(n)
    return [round(float(xs.mean() - h), 4), round(float(xs.mean() + h), 4)]


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return round(float(np.corrcoef(x, y)[0, 1]), 3) if len(x) >= 3 and x.std() > 0 and y.std() > 0 else None


def cmd_summary(a):
    rows = load_rows(a.out)
    res = {"offset": OFFSET, "rows": len(rows), "modes": {}, "points": {}}
    # 点ごと・mode ごとの対の差（a − b＝そのまま − 1 枚少ない・勝率）。世界ごとの対で集める。
    per = {}
    for r in rows.values():
        per.setdefault(r["mode"], {}).setdefault(r["pid"], []).append((r["i"], r["a"], r["b"], r))
    tv_net = _tv_point_net_diff(os.path.join("n1_results", "value_teach", "tv_states"))
    for m in MODES:
        pts = per.get(m, {})
        if not pts:
            continue
        d_pt, wa_pt, wb_pt, pids = [], [], [], []
        forced, late, steps, n_games = 0, 0, 0, 0
        world_net = {n: [] for n in NETS}
        world_d = []
        for pid in sorted(pts):
            xs = pts[pid]
            d_pt.append(float(np.mean([x[1] - x[2] for x in xs])))
            wa_pt.append(float(np.mean([x[1] for x in xs])))
            wb_pt.append(float(np.mean([x[2] for x in xs])))
            pids.append(pid)
            for _i, ra, rb, r in xs:
                for side in ("info_a", "info_b"):
                    inf = r.get(side) or {}
                    forced += int(inf.get("forced") or 0)
                    late += int(inf.get("late") or 0)
                    steps += int(inf.get("steps") or 0)
                n_games += 2
                if isinstance(r.get("net"), dict) and "err" not in r["net"]:
                    world_d.append(ra - rb)
                    for n in NETS:
                        world_net[n].append(r["net"][n])
        d_pt = np.asarray(d_pt)
        # 価値の目盛り（2w − 1 の差）＝勝率の差 × 2。
        res["modes"][m] = {
            "points": len(pids), "worlds_per_point": round(float(np.mean([len(pts[p]) for p in pids])), 1),
            "games": n_games, "win_diff": round(float(d_pt.mean()), 4), "value_diff": round(float(2 * d_pt.mean()), 4),
            "value_diff_ci": [None if x is None else round(2 * x, 4) for x in _ci(d_pt)],
            "win_a": round(float(np.mean(wa_pt)), 4), "win_b": round(float(np.mean(wb_pt)), 4),
            "per_point_win_diff": {str(p): round(float(x), 4) for p, x in zip(pids, d_pt)},
            "forced_per_game": round(forced / max(n_games, 1), 3), "late_per_game": round(late / max(n_games, 1), 3),
            "steps_per_game": round(steps / max(n_games, 1), 1),
            # 同じ世界の盤面でのネットの差（世界ごとの対・勝敗 1/0 の差との相関＝参考）
            "world_net_corr": {n: _corr(world_net[n], world_d) for n in NETS if world_net[n]},
            "world_net_mean": {n: round(float(np.mean(world_net[n])), 4) for n in NETS if world_net[n]},
        }
        res["points"][m] = {str(p): {"n": len(pts[p]), "wa": round(float(np.mean([x[1] for x in pts[p]])), 3),
                                     "wb": round(float(np.mean([x[2] for x in pts[p]])), 3)} for p in pids}
    # 世代ごとの表: ネットの差（保存済み盤面・窓を閉じた直後）÷ (i)／(ii)。
    gens = {"r29f4": "teach_r29f4", "r28x4": "teach_r28x4", "r27": "teach_r27"}
    ref = res["modes"].get("ref")
    res["generations"] = {}
    for net, m in gens.items():
        nd = tv_net[net]
        row = {"net_diff_saved": round(float(np.mean(list(nd.values()))), 4) if nd else None, "mode": m,
               "player": TEACH[m]["player"], "temp_late": TEACH[m]["temp_late"]}
        for lab, mm in (("i", "ref"), ("ii", m)):
            mr = res["modes"].get(mm)
            if not mr:
                continue
            row[f"play_diff_{lab}"] = mr["value_diff"]
            row[f"ratio_{lab}"] = round(row["net_diff_saved"] / mr["value_diff"], 3) if row["net_diff_saved"] and mr["value_diff"] else None
            common = [p for p in sorted(nd) if str(p) in mr["per_point_win_diff"]]
            row[f"corr_{lab}"] = _corr([nd[p] for p in common], [mr["per_point_win_diff"][str(p)] for p in common])
            row[f"n_points_{lab}"] = len(common)
        if ref and res["modes"].get(m):
            # 同じ世界の対で (ii) − (i)
            d = []
            for k, r in rows.items():
                if r["mode"] == m:
                    kr = k[:-1] + ("ref",)
                    if kr in rows:
                        d.append((r["a"] - r["b"]) - (rows[kr]["a"] - rows[kr]["b"]))
            row["ii_minus_i_value"] = round(2 * float(np.mean(d)), 4) if d else None
            row["ii_minus_i_ci"] = [None if x is None else round(2 * x, 4) for x in _ci(d)] if len(d) > 1 else None
            row["paired_n"] = len(d)
            # 比の区間（点を引き直す bootstrap・4,000 回）
            common = [p for p in sorted(nd) if str(p) in ref["per_point_win_diff"] and str(p) in res["modes"][m]["per_point_win_diff"]]
            if len(common) >= 3:
                N = np.array([nd[p] for p in common])
                R = 2 * np.array([ref["per_point_win_diff"][str(p)] for p in common])
                T = 2 * np.array([res["modes"][m]["per_point_win_diff"][str(p)] for p in common])
                rng = np.random.default_rng(0)
                bi, bii = [], []
                for _ in range(4000):
                    idx = rng.integers(0, len(common), len(common))
                    if R[idx].mean() > 0 and T[idx].mean() > 0:
                        bi.append(N[idx].mean() / R[idx].mean()); bii.append(N[idx].mean() / T[idx].mean())
                q = lambda x: [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]  # noqa: E731
                row["ratio_i_ci"], row["ratio_ii_ci"] = q(bi), q(bii)
        res["generations"][net] = row
    # 点の種類（記事が控える理由: early＝序盤・big_next＝次の大物・bk_on＝記事は使う）ごとの差（価値の目盛り）
    why = {str(p["pid"]): p["why"] for p in OV.kaido_points()}
    res["by_why"] = {}
    for w in sorted(set(why.values())):
        ps = [p for p in why if why[p] == w]
        res["by_why"][w] = {"points": len(ps)}
        for m, mr in res["modes"].items():
            xs = [mr["per_point_win_diff"][p] for p in ps if p in mr["per_point_win_diff"]]
            if xs:
                res["by_why"][w][m] = round(2 * float(np.mean(xs)), 4)
        for net in NETS:
            xs = [tv_net[net][int(p)] for p in ps if int(p) in tv_net[net]]
            if xs:
                res["by_why"][w]["net_" + net] = round(float(np.mean(xs)), 4)
    # 失敗の数（決着せず・対を作る段で決着）
    res["fails"] = {}
    for f in sorted(glob.glob(os.path.join(a.out, "rows_*.jsonl"))):
        for r in BA.read_jsonl(f):
            if not r.get("ok"):
                k = r["mode"] + "|" + (r.get("fail") or "void")
                res["fails"][k] = res["fails"].get(k, 0) + 1
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    for m, r in res["modes"].items():
        print(f"{m:13s} 点 {r['points']} 世界/点 {r['worlds_per_point']} 勝率 a {r['win_a']:.3f} b {r['win_b']:.3f} "
              f"差(目盛り) {r['value_diff']:+.3f} {r['value_diff_ci']} ε/局 {r['forced_per_game']} 散らし/局 {r['late_per_game']}")
    for net, r in res["generations"].items():
        print(f"{net:6s} ネットの差 {r['net_diff_saved']} (i) {r.get('play_diff_i')} 比 {r.get('ratio_i')} {r.get('ratio_i_ci')} 相関 {r.get('corr_i')} | "
              f"(ii) {r.get('play_diff_ii')} 比 {r.get('ratio_ii')} {r.get('ratio_ii_ci')} 相関 {r.get('corr_ii')} | "
              f"(ii)−(i) {r.get('ii_minus_i_value')} {r.get('ii_minus_i_ci')}")
    for w, r in res["by_why"].items():
        print("種類", w, json.dumps(r, ensure_ascii=False))
    print("失敗", json.dumps(res["fails"], ensure_ascii=False))
    # 段 2（あれば）
    rocks = load_rows(a.out, "rocks")
    if rocks:
        res2 = {}
        for m in MODES:
            xs = [r for r in rocks.values() if r["mode"] == m]
            if not xs:
                continue
            pts = {}
            for r in xs:
                pts.setdefault((r["src"], r["idx"], r["step"]), []).append(r["b"] - r["a"])
            d = np.asarray([np.mean(v) for v in pts.values()])
            res2[m] = {"points": len(pts), "games": 2 * len(xs), "win_diff_b_minus_a": round(float(d.mean()), 4),
                       "value_diff": round(float(2 * d.mean()), 4), "value_diff_ci": [None if x is None else round(2 * x, 4) for x in _ci(d)],
                       "win_a": round(float(np.mean([r["a"] for r in xs])), 4), "win_b": round(float(np.mean([r["b"] for r in xs])), 4),
                       "b_better_points": int((d > 0).sum())}
            print(f"rocks {m:13s} 点 {len(pts)} 戻す−CPU の手（目盛り）{res2[m]['value_diff']:+.3f} {res2[m]['value_diff_ci']} "
                  f"勝率 a {res2[m]['win_a']:.3f} b {res2[m]['win_b']:.3f}")
        res["rocks"] = res2
        json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["play", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--bench", default="kaido", choices=["kaido", "rocks"])
    ap.add_argument("--mode", default="all", help="ref,teach_r29f4,teach_r28x4,teach_r27（カンマ区切り）か all")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None, help="k/n（k＝1〜n）")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"play": cmd_play, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
