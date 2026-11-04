"""WP `light-rollout` 段 0（下見・`docs/search_efficiency_plan.md` §7.79・台帳 `cpu_theory_gap.md` 2+515）。

問い: 軽い打ち手で同じ盤面から（見えない札を配り直して）何度も打ち続けた勝敗の平均は、安く・ドン 1 枚の差を残した
価値の教師になるか。**測るだけ**（教材も訓練も作らない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。

打ち手の段（両席とも同じ・ネット r28x4・窓と対話の決め方は今の既定のまま）:
  L0   … 探索なし・ネットの打ち方の割合から引く（温度 1）＝ `sims=0`＋`prior_pick="sample"`（§7.79 で足した測定専用の口）
  L0f  … L0 に加えて戦闘の窓・効果の対話も割合から引く（配り直しの読み・木の箱畳み・静止探索を使わない＝`prior_pick="sample_all"`・
         コーディネータ追加 2026-11-04）
  L1   … 探索なし・割合が最大の手 ＝ `sims=0`＋`prior_pick="argmax"`
  L16  … 読み 16 回・世界 1 本
  L160 … 読み 160 回・世界 1 本（審判と同じ重さ）

サブコマンド:
  ``rows``   波 82j のドンを動かした行（手番 3〜6・+1／−1 を 200 ずつ）を選び、記録の手を当て直して盤面を復元できるかを確かめ、
             `rows.json`（局 seed・行の step・符号・勝敗 z・r28x4 の値）に書く。
  ``play``   ``--task speed``  … `rows.json` の先頭 100 盤面から各段 1 局（1 盤面 1 局・世界は配り直し 1 本）。
             ``--task kaido``  … 紫カイドウ 24 点の窓を閉じた直後のドン 1 枚違いの対（`value_split.kaido_pair`・世界 32 本）を各段で最後まで。
             ``--task jitter`` … `rows.json` の 400 行を各段で 16 局（配り直し 16 本）。
             配り方の番号のずらしは 18000。``--levels``・``--shard k/n``・``--workers`` で分ける。行は `<task>_<shard>.jsonl` に追記（再開できる）。
  ``summary`` 集めた行から 3 つ（速さ・差・ばらつき）をまとめて `summary.json`。

    OPCG_LOG_SILENT=1 python tests/scripts/light_rollout.py rows
    OPCG_LOG_SILENT=1 python tests/scripts/light_rollout.py play --task speed --levels L0,L1,L16,L160 --workers 1
    OPCG_LOG_SILENT=1 python tests/scripts/light_rollout.py play --task jitter --levels L160 --shard 3/16 --workers 4
    OPCG_LOG_SILENT=1 python tests/scripts/light_rollout.py summary
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from multiprocessing import get_context

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("OPCG_LOG_SILENT", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import move_judge as MJ  # noqa: E402
import omni_value_relabel as OR  # noqa: E402

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "light_rollout")
OFFSET = 18000
NET = os.path.join("opcg_sim", "data", "learned", "nrel_r28x4.npz")
WAVE = "~/n82j/n_records/n82j_w*_p*"
LEVELS = {
    "L0f": {"sims": 0, "prior_pick": "sample_all", "box_battle": False, "box_dialog": False, "quiesce": False},
    "L0": {"sims": 0, "prior_pick": "sample"},
    "L1": {"sims": 0, "prior_pick": "argmax"},
    "L16": {"sims": 16},
    "L160": {"sims": 160},
}
N_JIT_PER_SIGN = 200
K_JIT = 16
K_KAIDO = 32
N_SPEED = 100
_G = {}


# ---------------------------------------------------------------- 打ち続け

def spec_of(level):
    return E.SeatSpec(net=NET, worlds=1, **LEVELS[level])


def rollout(g, who, ws, level):
    """盤面 `g`（そのまま打ってよい複製）から、両席とも段 `level` で最後まで → (1／0／None, 手数, 秒)。

    探索の乱数の seed は `driver.run_game` と同じ（配り方の seed・ターン・席）。L0 だけは引きが 1 ターンの中で
    揃わないように手数も混ぜる（同じ seed だと同じターンの全部の判断が同じ一様乱数で引く）。"""
    spec = _G["specs"][level]
    sample = str(LEVELS[level].get("prior_pick", "")).startswith("sample")
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps, t0 = 0, time.time()
    try:
        while g.winner() is None and steps < MJ.MAX_STEPS:
            pend = json.loads(g.pending_json())
            if not pend:
                break
            name = pend["player_id"]
            turn = int(g.turn_count or 0)
            gs = (int(ws) + steps * 0x9E3779B1) & 0xFFFFFFFFFFFF if sample else int(ws)
            c = carries[name]
            dialog = bool(g.has_interaction)
            t1 = time.time()
            out = json.loads(g.decide(name, spec.decide_opts(gs, turn, name, c.get(turn, name))))
            t2 = time.time()
            c.put(out)
            mv = out.get("move")
            if mv is None:
                break
            kind = out.get("kind") or "?"
            # 内訳: 効果の対話（対話が立っている）・戦闘の窓（kind=window で対話なし）・メインの手（木）・箱の続き・詰み
            cat = "dialog" if dialog else ("battle" if kind == "window" else kind)
            if kind == "main" and not dialog and "BATTLE" in str(g.phase).upper():
                cat = "battle"
            BREAK[cat] = BREAK.get(cat, 0.0) + (t2 - t1)
            BREAK["n_" + cat] = BREAK.get("n_" + cat, 0) + 1
            BA.apply(g, name, mv)
            BREAK["apply"] = BREAK.get("apply", 0.0) + (time.time() - t2)
            steps += 1
    except Exception:  # noqa: BLE001
        return None, steps, round(time.time() - t0, 3)
    w = g.winner()
    return (None if w is None else (1 if w == who else 0)), steps, round(time.time() - t0, 3)


#: 打ち続けの時間の内訳（1 局ごとに play_job が読んで空にする）
BREAK = {}


# ---------------------------------------------------------------- 波 82j の行の復元

def _index(wave):
    """波の記録 → [(局 seed, ファイル, 動かした手番, 符号)]（列 seed と don_jitter だけ読む）。"""
    out = []
    for d in sorted(glob.glob(os.path.expanduser(wave))):
        for f in sorted(glob.glob(os.path.join(d, "n_record_*.npz"))):
            with np.load(f, allow_pickle=False) as z:
                seed, dj = z["seed"], z["don_jitter"]
            for s in np.unique(seed):
                i = int(np.flatnonzero(seed == s)[0])
                out.append((int(s), f, int(dj[i][0]), int(dj[i][2])))
    return out


def _file_games(f):
    """記録の 1 ファイル → {seed: 行の dict の列（step 順）}（プロセスの中で 1 ファイルだけ覚える）。"""
    if _G.get("file") == f:
        return _G["file_games"]
    out = {}
    with np.load(f, allow_pickle=False) as z:
        cols = {k: z[k] for k in ("seed", "step", "turn", "who", "z", "sig", "forced", "forced_sig",
                                  "scalars", "tokens", "card_idx", "don_jitter", "pol_len", "pol_chosen", "pol_k")}
    off = np.concatenate([[0], np.cumsum(cols["pol_len"])])
    for s in np.unique(cols["seed"]):
        rows = []
        for i in np.flatnonzero(cols["seed"] == s):
            r = {k: cols[k][i] for k in ("seed", "step", "turn", "who", "z", "sig", "forced", "forced_sig",
                                         "scalars", "tokens", "card_idx", "don_jitter")}
            pl, ch = int(cols["pol_len"][i]), int(cols["pol_chosen"][i])
            r["k"] = int(cols["pol_k"][off[i] + ch]) if pl and 0 <= ch < pl else -1
            rows.append(r)
        rows.sort(key=lambda r: int(r["step"]))
        out[int(s)] = rows
    _G["file"], _G["file_games"] = f, out
    return out


def _cands(game, who, sig):
    """`omni_value_relabel.candidates` に 6 つ目の区別の欄（position・selects・index・2026-10-23 から）を足したもの。"""
    if len(sig) <= 5:
        return OR.candidates(game, who, sig)
    from opcg_sim.loop import record_gen as RG
    out = []
    for mv in OR.candidates(game, who, sig[:5]) + [m for m in json.loads(game.legal_json(who)) if RG.move_sig(m) == sig]:
        new = dict(mv, payload=dict(mv.get("payload") or {}, **sig[5]))
        if RG.move_sig(new) == sig and new not in out:
            out.append(new)
    return out


def replay_to_jitter(seed, rows):
    """局 `seed` を記録の手で当て直し、ドンを動かした直後（その手番の手番の席の最初のメインの判断の頭）の盤面を返す。
    → (盤面, 行の添字, 席) か (None, 理由)。動かす口は生成と同じ（`Game.shift_don`・増やせなければ減らす側へ振り替え）。"""
    db = _G.setdefault("db", D.load_db())
    jt, jseat, jsign = (int(x) for x in rows[0]["don_jitter"])
    if jsign == 0:
        return None, "no_jitter"
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    seat = "p1" if jseat == 0 else "p2"
    beam = [(DR.new_game(seed, p1, p2), [])]
    for j, row in enumerate(rows):
        who = "p1" if row["who"] == 0 else "p2"
        if int(row["step"]) != j:
            return None, "step_gap"
        # 生成の `DonJitter`（driver の pre＝decide の前）と同じ条件で 1 回だけ動かす（記録の符号は振り替え後の値）
        g0 = beam[0][0]
        if (int(g0.turn_count or 0) == jt and who == seat and g0.turn_player() == seat
                and not g0.has_interaction and g0.phase == "MAIN"):
            ok = [g for g, _h in beam if g.shift_don(seat, jsign) is not None and OR.enc_matches(g, who, row)[0]]
            if not ok:
                return None, "enc_after_jitter"
            return ok[0], j
        live = [(g, h) for g, h in beam if OR.enc_matches(g, who, row)[0]]
        if not live:
            return None, f"enc_{j}"
        beam = live
        sig = OR.applied_sig(row)
        nxt = []
        for g, hist in beam:
            cs = _cands(g, who, sig)
            for mv in cs[1:]:
                h = DR.new_game(seed, p1, p2)
                try:
                    for w, m in hist:
                        OR.apply_move(h, w, m)
                    OR.apply_move(h, who, mv)
                except Exception:  # noqa: BLE001
                    continue
                nxt.append((h, hist + [(who, mv)]))
            if cs:
                try:
                    OR.apply_move(g, who, cs[0])
                    nxt.append((g, hist + [(who, cs[0])]))
                except Exception:  # noqa: BLE001
                    pass
            if len(nxt) >= OR.BEAM_MAX:
                break
        if not nxt:
            return None, f"apply_{j}"
        beam = nxt[:OR.BEAM_MAX]
    return None, "not_reached"


def _rows_job(item):
    seed, f = item
    try:
        rows = _file_games(f)[seed]
        g, j = replay_to_jitter(seed, rows)
    except Exception as exc:  # noqa: BLE001
        return {"seed": seed, "ok": False, "fail": "%s: %s" % (type(exc).__name__, str(exc)[:80])}
    if g is None:
        return {"seed": seed, "ok": False, "fail": j}
    r = rows[j]
    jt, jseat, jsign = (int(x) for x in r["don_jitter"])
    return {"seed": seed, "file": f.replace(os.path.expanduser("~"), "~", 1), "ok": True, "row": j, "turn": jt, "seat": "p1" if jseat == 0 else "p2", "sign": jsign,
            "who_row": int(r["who"]), "z": float(r["z"]), "hold": seed % 7 == 0,
            "sc": r["scalars"].astype(np.float32).tolist(), "ci": r["card_idx"][:22].astype(int).tolist(),
            "tok": r["tokens"].astype(np.float32).tolist()}


def _init_rows():
    E.engine()
    E.load_net(NET)


def cmd_rows(a):
    idx = _index(a.wave)
    rng = np.random.default_rng(0)
    cand = {1: [], -1: []}
    for s, f, t, sg in idx:
        if sg != 0 and 3 <= t <= 6:
            cand[sg].append((s, f))
    print("候補 +1 %d −1 %d" % (len(cand[1]), len(cand[-1])), flush=True)
    items = []
    for sg in (1, -1):
        xs = sorted(cand[sg])
        perm = rng.permutation(len(xs))
        items += [xs[i] for i in perm[:int(N_JIT_PER_SIGN * 1.15)]]   # 復元に失敗する局のぶん多めに当てる
    items.sort(key=lambda x: x[1])                                    # 同じファイルを続けて読む
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_rows) as pool:
        res = list(pool.imap(_rows_job, items, chunksize=4))
    ok = {1: [], -1: []}
    fails = {}
    for r in res:
        if r["ok"]:
            ok[r["sign"]].append(r)
        else:
            fails[r["fail"]] = fails.get(r["fail"], 0) + 1
    sel = sorted(ok[1], key=lambda r: r["seed"])[:N_JIT_PER_SIGN] + sorted(ok[-1], key=lambda r: r["seed"])[:N_JIT_PER_SIGN]
    from value_teach import net_values
    sc = np.array([r["sc"] for r in sel], np.float32)
    ci = np.array([r["ci"] for r in sel], np.int16)
    tok = np.array([r["tok"] for r in sel], np.float32)
    v = net_values(NET, sc, ci, tok)
    out = []
    for r, x in zip(sel, v):
        out.append({k: r[k] for k in ("seed", "file", "row", "turn", "seat", "sign", "z", "hold")} | {"net": round(float(x), 4)})
    os.makedirs(a.out, exist_ok=True)
    json.dump({"rows": out, "fails": fails, "tried": len(items), "sec": round(time.time() - t0, 1)},
              open(os.path.join(a.out, "rows.json"), "w"), ensure_ascii=False, indent=0)
    zp = [r["z"] for r in out if r["sign"] == 1]
    zm = [r["z"] for r in out if r["sign"] == -1]
    print(f"rows {len(out)} (+1 {len(zp)} −1 {len(zm)}) fails {fails} 勝敗の差 {np.mean(zp) - np.mean(zm):+.3f} "
          f"r28x4 の差 {np.mean([r['net'] for r in out if r['sign'] == 1]) - np.mean([r['net'] for r in out if r['sign'] == -1]):+.3f} "
          f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- play

def _init_play(levels, task):
    E.engine()
    E.load_net(NET)
    _G["specs"] = {lv: spec_of(lv) for lv in levels}
    if task == "kaido":
        import offdist_value as OV
        import value_split as VS
        OV._init_tv_k(0, 0, None)          # エンジン・r27（窓を片付ける出荷の設定）・記録の局（value_split と同じ）
        E.load_net(NET)
        VS.OFFSET = OFFSET
        _G["VS"] = VS
    else:
        rows = json.load(open(os.path.join(_G["out"], "rows.json")))["rows"]
        _G["rows"] = {(r["seed"]): r for r in rows}


def _jit_game(seed):
    """`rows.json` の局 seed の、ドンを動かした直後の盤面（プロセスの中で覚える）。"""
    cache = _G.setdefault("gcache", {})
    if seed not in cache:
        r = _G["rows"][seed]
        g, j = replay_to_jitter(seed, _file_games(os.path.expanduser(r["file"]))[seed])
        if g is None or j != r["row"]:
            raise RuntimeError("replay: %s" % j)
        if len(cache) > 32:
            cache.clear()
        cache[seed] = g
    return cache[seed]


def play_job(job):
    task, key, i, level = job
    t0 = time.time()
    base = {"task": task, "key": key, "i": i, "level": level}
    try:
        if task == "kaido":
            VS = _G["VS"]
            pts = _G.setdefault("kpts", {p["pid"]: p for p in VS.OV.kaido_points()})
            ga, gb, ws, who, _ = VS.kaido_pair(pts[key], i)
            if ga is None:
                return dict(base, ok=False, fail="ended")
            ra, sa, ta = rollout(ga.fork(), who, ws, level)
            rb, sb, tb = rollout(gb.fork(), who, ws, level)
            return dict(base, ok=ra is not None and rb is not None, a=ra, b=rb, steps=[sa, sb], sec=[ta, tb],
                        ws=int(ws), why=pts[key]["why"])
        r = _G["rows"][key]
        g = _jit_game(key)
        ws = MJ.world_seed(key, r["row"], OFFSET + i)
        gw = g.fork_world(r["seat"], "fair", int(ws))
        BREAK.clear()
        res, st, sec = rollout(gw, r["seat"], ws, level)
        brk = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in BREAK.items()}
        return dict(base, ok=res is not None, w=res, steps=st, sec=sec, ws=int(ws), brk=brk)
    except Exception as exc:  # noqa: BLE001
        return dict(base, ok=False, fail="%s: %s" % (type(exc).__name__, str(exc)[:100]), sec=round(time.time() - t0, 2))


def _jobs(a, levels):
    if a.task == "kaido":
        import offdist_value as OV
        pids = [p["pid"] for p in OV.kaido_points()]
        jobs = [("kaido", pid, i, lv) for lv in levels for pid in pids for i in range(a.k or K_KAIDO)]
    else:
        rows = json.load(open(os.path.join(a.out, "rows.json")))["rows"]
        if a.task == "speed":
            rows = rows[::max(1, len(rows) // N_SPEED)][:N_SPEED]
            jobs = [("speed", r["seed"], 0, lv) for lv in levels for r in rows]
        else:
            jobs = [("jitter", r["seed"], i, lv) for lv in levels for r in rows for i in range(a.k or K_JIT)]
    if a.limit:
        jobs = jobs[:a.limit]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        jobs = [j for n, j in enumerate(jobs) if n % sn == si - 1]
    return jobs


def cmd_play(a):
    levels = a.levels.split(",")
    for lv in levels:
        if lv not in LEVELS:
            raise SystemExit("--levels は %s" % ",".join(LEVELS))
    _G["out"], _G["wave_glob"] = a.out, a.wave
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or (f"s{int(a.shard.split('/')[0]):02d}" if a.shard else "local")
    path = os.path.join(a.out, f"{a.task}_{tag}.jsonl")
    done = set()
    for f in glob.glob(os.path.join(a.out, f"{a.task}_*.jsonl")):
        for r in BA.read_jsonl(f):
            if r.get("ok") or r.get("fail") == "ended":
                done.add((r["key"], r["i"], r["level"], r.get("wk", 0)))
    jobs = [j for j in _jobs(a, levels) if (j[1], j[2], j[3], a.workers if a.task == "speed" else 0) not in done]
    # 同じ盤面の行をまとめて 1 プロセスへ（当て直しの覚えを効かせる）
    print(f"PLAY task={a.task} levels={a.levels} shard={a.shard} workers={a.workers} jobs {len(jobs)} -> {path}", flush=True)
    t0 = time.time()
    n = 0
    with get_context("fork").Pool(a.workers, initializer=_init_play, initargs=(levels, a.task)) as pool:
        for r in pool.imap(play_job, jobs, chunksize=1 if a.task != "jitter" else 4):
            if a.task == "speed":
                r["wk"] = a.workers
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {r['level']} key={r['key']} ok={r.get('ok')} sec={r.get('sec')} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"PLAY_DONE {n} {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- summary

def _ci_mean(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return None
    return round(float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs))), 4)


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return round(float(np.corrcoef(x, y)[0, 1]), 3) if len(x) >= 3 and x.std() > 0 and y.std() > 0 else None


def _load(out, task):
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, f"{task}_*.jsonl"))):
        for r in BA.read_jsonl(f):
            rows[(r["key"], r["i"], r["level"], r.get("wk", 0))] = r
    return list(rows.values())


def cmd_summary(a):
    res = {"offset": OFFSET, "net": NET, "levels": LEVELS}
    # 1. 速さ
    sp = _load(a.out, "speed")
    res["speed"] = {}
    for lv in LEVELS:
        for wk in sorted({r.get("wk", 1) for r in sp}):
            xs = [r for r in sp if r["level"] == lv and r.get("wk", 1) == wk and r.get("ok")]
            if not xs:
                continue
            sec = np.array([r["sec"] for r in xs])
            st = np.array([r["steps"] for r in xs])
            res["speed"][f"{lv}_w{wk}"] = {"games": len(xs), "sec_mean": round(float(sec.mean()), 3),
                                           "sec_median": round(float(np.median(sec)), 3), "sec_max": round(float(sec.max()), 2),
                                           "steps_mean": round(float(st.mean()), 1),
                                           "ms_per_step": round(1000 * float(sec.sum() / st.sum()), 2),
                                           "fail": sum(1 for r in sp if r["level"] == lv and r.get("wk", 1) == wk and not r.get("ok"))}
            bk = [r["brk"] for r in xs if r.get("brk")]
            if bk:
                keys = sorted({k for b in bk for k in b})
                res["speed"][f"{lv}_w{wk}"]["breakdown_sec"] = {k: round(float(np.mean([b.get(k, 0) for b in bk])), 4)
                                                                for k in keys}
    # 2-(i). 紫カイドウ 24 点
    kd = [r for r in _load(a.out, "kaido") if r.get("ok")]
    res["kaido"] = {}
    for lv in LEVELS:
        xs = [r for r in kd if r["level"] == lv]
        if not xs:
            continue
        per = {}
        for r in xs:
            per.setdefault(r["key"], []).append(r["a"] - r["b"])
        d = np.array([np.mean(v) for v in per.values()])
        why = {}
        for r in xs:
            why.setdefault(r["why"], {}).setdefault(r["key"], []).append(r["a"] - r["b"])
        res["kaido"][lv] = {"points": len(per), "games": 2 * len(xs), "value_diff": round(2 * float(d.mean()), 4),
                            "value_diff_ci": round(2 * (_ci_mean(d) or 0), 4),
                            "win_a": round(float(np.mean([r["a"] for r in xs])), 4),
                            "win_b": round(float(np.mean([r["b"] for r in xs])), 4),
                            "by_why": {w: round(2 * float(np.mean([np.mean(v) for v in ps.values()])), 4) for w, ps in why.items()},
                            "per_point": {str(k): round(2 * float(np.mean(v)), 3) for k, v in per.items()},
                            "sec_game": round(float(np.mean([s for r in xs for s in r["sec"]])), 3)}
    # 同じ (点, 世界) で段どうしの対の差（L160 を基準）
    if "L160" in res["kaido"]:
        by = {(r["key"], r["i"], r["level"]): r["a"] - r["b"] for r in kd}
        for lv in res["kaido"]:
            pts = {}
            for (k, i, l), v in by.items():
                if l == lv and (k, i, "L160") in by:
                    pts.setdefault(k, []).append(v - by[(k, i, "L160")])
            if pts and lv != "L160":
                d = np.array([np.mean(v) for v in pts.values()])
                res["kaido"][lv]["minus_L160"] = round(2 * float(d.mean()), 4)
                res["kaido"][lv]["minus_L160_ci"] = round(2 * (_ci_mean(d) or 0), 4)
                res["kaido"][lv]["corr_points_L160"] = _corr([np.mean([by[(k, i, lv)] for i in range(64) if (k, i, lv) in by]) for k in pts],
                                                            [np.mean([by[(k, i, "L160")] for i in range(64) if (k, i, "L160") in by]) for k in pts])
    # 2-(ii)・3. 波 82j の動かした行
    meta = json.load(open(os.path.join(a.out, "rows.json")))
    R = {r["seed"]: r for r in meta["rows"]}
    sign = np.array([R[s]["sign"] for s in R])
    z = np.array([R[s]["z"] for s in R])
    net = np.array([R[s]["net"] for s in R])
    seeds = list(R)
    res["jitter"] = {"rows": len(R), "fails_replay": meta["fails"],
                     "z_diff": round(float(z[sign == 1].mean() - z[sign == -1].mean()), 4),
                     "z_diff_ci": round(1.96 * math.hypot(z[sign == 1].std() / math.sqrt((sign == 1).sum()),
                                                          z[sign == -1].std() / math.sqrt((sign == -1).sum())), 4),
                     "net_diff": round(float(net[sign == 1].mean() - net[sign == -1].mean()), 4), "levels": {}}
    jt = [r for r in _load(a.out, "jitter") if r.get("ok")]
    for lv in LEVELS:
        xs = [r for r in jt if r["level"] == lv]
        if not xs:
            continue
        per = {}
        for r in xs:
            per.setdefault(r["key"], []).append(2 * r["w"] - 1)
        keep = [s for s in seeds if len(per.get(s, [])) >= 2]
        m = np.array([np.mean(per[s]) for s in keep])
        nk = np.array([len(per[s]) for s in keep])
        sg = np.array([R[s]["sign"] for s in keep])
        zz = np.array([R[s]["z"] for s in keep])
        nn = np.array([R[s]["net"] for s in keep])
        # 平均の標準誤差（盤面ごと・その盤面の 16 局の標本分散から）と、勝敗 1 局（±1）のばらつき
        se = np.array([np.std(per[s], ddof=1) / math.sqrt(len(per[s])) for s in keep])
        sd1 = np.array([np.std(per[s], ddof=1) for s in keep])
        dm = float(m[sg == 1].mean() - m[sg == -1].mean())
        dci = 1.96 * math.hypot(m[sg == 1].std() / math.sqrt((sg == 1).sum()), m[sg == -1].std() / math.sqrt((sg == -1).sum()))
        # 群の差に効く雑音の比べ: 平均の盤面の間のばらつき vs 勝敗 1 局の盤面の間のばらつき
        res["jitter"]["levels"][lv] = {
            "rows": len(keep), "games": int(nk.sum()), "games_per_row": round(float(nk.mean()), 2),
            "mean_value": round(float(m.mean()), 4),
            "group_diff": round(dm, 4), "group_diff_ci": round(dci, 4),
            "se_mean_rms": round(float(np.sqrt((se ** 2).mean())), 4),
            "sd_one_game_within": round(float(np.sqrt((sd1 ** 2).mean())), 4),
            "sd_outcome_z": round(float(zz.std()), 4),
            "ratio_se_to_z": round(float(np.sqrt((se ** 2).mean()) / zz.std()), 4),
            "sd_between_rows_mean": round(float(m.std()), 4),
            "corr_net": _corr(m, nn), "corr_z": _corr(m, zz),
            "corr_net_z": _corr(nn, zz),
            "sec_game": round(float(np.mean([r["sec"] for r in xs])), 3),
        }
        # 16 局の平均の、局数ごとの見え方（先頭 1・2・4・8・16 局）——差とばらつき
        curve = {}
        for kk in (1, 2, 4, 8, 16, 32):
            if nk.min() < kk:
                continue
            mk = np.array([np.mean(sorted_by_i(xs, s)[:kk]) for s in keep])
            curve[kk] = {"group_diff": round(float(mk[sg == 1].mean() - mk[sg == -1].mean()), 4),
                         "sd_between_rows": round(float(mk.std()), 4), "corr_z": _corr(mk, zz), "corr_net": _corr(mk, nn)}
        res["jitter"]["levels"][lv]["by_k"] = curve
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ("speed", "kaido", "jitter")}, ensure_ascii=False, indent=1)[:6000])


def sorted_by_i(xs, seed):
    _G.setdefault("_byi", {})
    key = id(xs)
    if _G.get("_byi_key") != key:
        d = {}
        for r in xs:
            d.setdefault(r["key"], []).append((r["i"], 2 * r["w"] - 1))
        _G["_byi"] = {k: [v for _, v in sorted(vs)] for k, vs in d.items()}
        _G["_byi_key"] = key
    return _G["_byi"][seed]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["rows", "play", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--wave", default=WAVE)
    ap.add_argument("--task", default="speed", choices=["speed", "kaido", "jitter"])
    ap.add_argument("--levels", default="L0f,L0,L1,L16,L160")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None, help="k/n（k＝1〜n）")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"rows": cmd_rows, "play": cmd_play, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
