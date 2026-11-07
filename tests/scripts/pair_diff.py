"""WP `pair-diff-loss`（`docs/search_efficiency_plan.md` §7.88・台帳 2+534）: 同じ局面のドン ±1 の対で、価値に差そのものを教える。

問い: 勝敗の二乗誤差は各盤面の値を当てるだけで、同じ盤面のドン 1 枚違いの値の差を直接は求めない。差を損失で求め、対を
いろいろな形の盤面（自分の手番の頭・窓を閉じた直後・ドンを控えた後・相手の手番の終わり）で作れば、差は形を越えて移るか。

サブコマンド:
  ``teach``  段 0 (2): 対の教師の候補を比べる。紫カイドウ 24 点の窓を閉じた直後の対（`value_split.kaido_pair`・世界 32 本・
             ずらし 18000＝§7.79 と同じ世界）と、波 82j のドンを動かした 400 行（§7.79 の `rows.json`）で、
             探索の根の値（読み 160 回／1,280 回・世界 1 本）が「打ち続けの差」をどれだけ持つかを測る。
             打ち続けの側（読み 16 回・160 回）は §7.79 の 1 局ごとの記録（`n1_results/light_rollout/*_local.jsonl`）から読む。
  ``plan``   波 70f〜73f の局から 4 つの形（＋評価だけの 5 つ目）の盤面を選び `positions.json.gz` に書く（決定的）。
  ``play``   ``--shard i/n``: 自分の局だけ記録の手で当て直し、各盤面でドンを ±1 動かした対を作り、両方から同じ配り直し 16 本
             （公平・ずらし 20000）を読み 16 回・世界 1 本・r28x4 で最後まで打つ。対の符号化と勝敗を `pairs_sKK.npz`／`.jsonl` に書く（再開できる）。
  ``merge``  シャードの対を 1 つの `pairs.npz`（訓練器の `--pair-diff` が読む）にまとめる。
  ``beds``   対の当てはまり（形ごと・取り置きの局／訓練の局・ネットの差と教師の差の比〔傾き〕・相関）。

    OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py teach --workers 4
    OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py plan
    OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py play --shard 3/16 --workers 4
    OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py merge
    OPCG_LOG_SILENT=1 python tests/scripts/pair_diff.py beds --nets a.npz b.npz

1 回きりの測定と教材の作り＝道具のテストは足さない（CLAUDE.md の理論の道具の規約）。訓練器の受け口のテストは `tests/test_pair_diff.py`。
"""
import argparse
import glob
import gzip
import hashlib
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
import light_rollout as LR  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "pair_diff_loss")
LR_DIR = os.path.join("n1_results", "light_rollout")
NET = LR.NET                                   # r28x4（波 70f〜73f の生成役）
#: 根の値の候補（世界 1 本・窓と対話は今の既定）
ROOT_LEVELS = {"R160": {"sims": 160}, "R1280": {"sims": 1280}}
_G = {}


def _ci(xs):
    xs = np.asarray(xs, np.float64)
    return float(xs.mean()), (1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


# ---------------------------------------------------------------- 根の値

def root_value(g, who, ws, spec):
    """盤面 `g` の手番の席が探索した根の値（訪問で重み付けた Q の平均＝記録の `pol_v0` と同じ式）を `who` の視点で。
    根に候補が 1 つしか無い（訪問 0）・手番が無いときは None。"""
    pend = json.loads(g.pending_json())
    if not pend:
        return None, "no_pending"
    name = pend["player_id"]
    turn = int(g.turn_count or 0)
    out = json.loads(g.fork().decide(name, spec.decide_opts(int(ws), turn, name, None)))
    groups = out.get("groups") or []
    ntot = sum(float(x["n"]) for x in groups)
    if ntot <= 0:
        return None, "kind=%s" % out.get("kind")
    v = sum(float(x["n"]) * float(x["q"]) for x in groups) / ntot
    return (v if name == who else -v), out.get("kind")


def _init_teach():
    import offdist_value as OV
    import value_split as VS
    OV._init_tv_k(0, 0, None)          # エンジン・r27（窓を片付ける出荷の設定）・記録の局（value_split と同じ）
    E.load_net(NET)
    VS.OFFSET = LR.OFFSET              # 18000＝§7.79 の kaido と同じ世界
    _G["VS"] = VS
    _G["specs"] = {k: E.SeatSpec(net=NET, worlds=1, **v) for k, v in ROOT_LEVELS.items()}
    LR._G["out"] = LR_DIR
    rows = json.load(open(os.path.join(LR_DIR, "rows.json")))["rows"]
    LR._G["rows"] = {r["seed"]: r for r in rows}


def teach_job(job):
    task, key, i = job
    base = {"task": task, "key": key, "i": i}
    try:
        if task == "kaido":
            VS = _G["VS"]
            pts = _G.setdefault("kpts", {p["pid"]: p for p in VS.OV.kaido_points()})
            gb, gl, ws, who, _ = VS.kaido_pair(pts[key], i)
            if gb is None:
                return dict(base, ok=False, fail="ended")
            out = dict(base, ok=True, why=pts[key]["why"])
            for lv, spec in _G["specs"].items():
                vb, kb = root_value(gb, who, ws, spec)
                vl, kl = root_value(gl, who, ws, spec)
                out[lv] = [None if vb is None else round(vb, 5), None if vl is None else round(vl, 5)]
                out[lv + "_kind"] = [kb, kl]
            return out
        r = LR._G["rows"][key]
        g = LR._jit_game(key)
        ws = MJ.world_seed(key, r["row"], LR.OFFSET)
        out = dict(base, ok=True, sign=r["sign"], hold=r["hold"], z=r["z"], net=r["net"])
        for lv, spec in _G["specs"].items():
            v, k = root_value(g, r["seat"], ws, spec)
            out[lv] = None if v is None else round(v, 5)
            out[lv + "_kind"] = k
        return out
    except Exception as exc:  # noqa: BLE001
        return dict(base, ok=False, fail="%s: %s" % (type(exc).__name__, str(exc)[:100]))


def cmd_teach(a):
    import offdist_value as OV
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "teach_root.jsonl")
    done = ({(r["task"], r["key"], r["i"]) for r in BA.read_jsonl(path) if r.get("ok") or r.get("fail") == "ended"}
            if os.path.exists(path) else set())
    jobs = []
    if a.task in ("all", "kaido"):
        jobs += [("kaido", p["pid"], i) for p in OV.kaido_points() for i in range(LR.K_KAIDO)]
    if a.task in ("all", "jitter"):
        rows = json.load(open(os.path.join(LR_DIR, "rows.json")))["rows"]
        jobs += [("jitter", r["seed"], 0) for r in rows]
    jobs = [j for j in jobs if j not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"TEACH jobs {len(jobs)} -> {path}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_teach) as pool:
        for n, r in enumerate(pool.imap(teach_job, jobs, chunksize=1), 1):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 50 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {r['task']} ok={r.get('ok')} ({time.time() - t0:.0f}s)", flush=True)
    print(f"TEACH_DONE {time.time() - t0:.0f}s", flush=True)
    teach_summary(a)


def teach_summary(a):
    """候補ごとに: 紫カイドウの対の差（24 点の平均・点ごとの平均の平均）と §7.79 の物差しとの比、82j の群の差。"""
    rt = [r for r in BA.read_jsonl(os.path.join(a.out, "teach_root.jsonl")) if r.get("ok") or r.get("fail") == "ended"]
    out = {"kaido": {}, "jitter": {}}
    # 紫カイドウ: 打ち続け（§7.79・1 局ずつ・価値の目盛り＝2×(勝 a − 勝 b)）と根の値
    kl = [r for r in BA.read_jsonl(os.path.join(LR_DIR, "kaido_local.jsonl")) if r.get("ok")]
    per = {}
    for r in kl:
        if r["level"] in ("L16", "L160"):
            per.setdefault(r["level"], {}).setdefault(r["key"], []).append(2.0 * (r["a"] - r["b"]))
    for r in rt:
        if r["task"] != "kaido" or not r.get("ok"):
            continue
        for lv in ROOT_LEVELS:
            vb, vl = r.get(lv) or [None, None]
            if vb is not None and vl is not None:
                per.setdefault(lv, {}).setdefault(r["key"], []).append(vb - vl)
    from value_teach import TV_TRUTH
    for lv, d in per.items():
        pm = [float(np.mean(v)) for v in d.values()]
        m, h = _ci(pm)
        out["kaido"][lv] = {"diff": round(m, 4), "ci": round(h, 4), "ratio_to_truth": round(m / TV_TRUTH, 3),
                            "points": len(pm), "pairs": int(sum(len(v) for v in d.values()))}
    # 点ごとの相関（読み 160 回の打ち続けと）
    ref = {k: float(np.mean(v)) for k, v in per.get("L160", {}).items()}
    for lv, d in per.items():
        ks = sorted(set(ref) & set(d))
        if len(ks) > 2 and lv != "L160":
            out["kaido"][lv]["corr_L160"] = round(float(np.corrcoef([ref[k] for k in ks], [np.mean(d[k]) for k in ks])[0, 1]), 3)
    # 82j の 400 行: 群の差（+1 − −1）
    jl = [r for r in BA.read_jsonl(os.path.join(LR_DIR, "jitter_local.jsonl")) if r.get("ok")]
    mean_by = {}
    for r in jl:
        if r["level"] in ("L16", "L160"):
            mean_by.setdefault(r["level"], {}).setdefault(r["key"], []).append(2 * r["w"] - 1)
    rows = {r["seed"]: r for r in json.load(open(os.path.join(LR_DIR, "rows.json")))["rows"]}
    vals = {lv: {k: float(np.mean(v)) for k, v in d.items()} for lv, d in mean_by.items()}
    vals["z"] = {k: r["z"] for k, r in rows.items()}
    vals["net_r28x4"] = {k: r["net"] for k, r in rows.items()}
    for r in rt:
        if r["task"] == "jitter" and r.get("ok"):
            for lv in ROOT_LEVELS:
                if r.get(lv) is not None:
                    vals.setdefault(lv, {})[r["key"]] = r[lv]
    for lv, d in vals.items():
        p = [v for k, v in d.items() if rows[k]["sign"] == 1]
        m = [v for k, v in d.items() if rows[k]["sign"] == -1]
        g = float(np.mean(p) - np.mean(m))
        h = 1.96 * math.hypot(np.std(p) / math.sqrt(len(p)), np.std(m) / math.sqrt(len(m)))
        out["jitter"][lv] = {"gap": round(g, 4), "ci": round(h, 4), "n": [len(p), len(m)]}
    for lv in out["jitter"]:
        out["jitter"][lv]["ratio_to_L160"] = round(out["jitter"][lv]["gap"] / out["jitter"]["L160"]["gap"], 3)
    fails = {}
    for r in rt:
        if not r.get("ok"):
            fails[r.get("fail")] = fails.get(r.get("fail"), 0) + 1
    out["fails"] = fails
    json.dump(out, open(os.path.join(a.out, "teach_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


# ---------------------------------------------------------------- 対の盤面の選び（段 0 (1)）

#: 形（4 つは訓練に使う・"mid" は評価だけ＝対を作っていない形）
SHAPES = ("head", "close", "hold", "oppend")
EVAL_SHAPES = ("mid",)
WAVES = ("n70f", "n71f", "n72f", "n73f")
WAVE_GLOB = "~/{w}/n_records/{w}_w*_p[0-3]"
POSITIONS = os.path.join(OUT_DIR, "positions.json.gz")
#: 配り方の番号のずらし（`move_judge.world_seed` の i に足す・`docs/n_loop_ops.md` の 14000〜19000 と重ねない）
OFFSET = 20000
LEVEL = "L16"                                  # 読み 16 回・世界 1 本（§7.79 の推薦・§7.80 と同じ）
K = 16                                         # 1 つの盤面の配り直しの本数（対の両側で同じ 16 本）
MIN_TURN = 3
N_TOK = 22


def _h(*xs):
    return int(hashlib.sha1(("pair-diff:" + ":".join(map(str, xs))).encode()).hexdigest()[:12], 16)


def shape_rows(turn, who, kind, step):
    """1 局の行 → {形: [(行の添字, step, 視点の席 0/1, ターン)]}（候補の全部）。

    - head  : そのターンの手番の席の最初の main の行（自分の手番の頭）・視点は手番の席。
    - close : 守る側の窓（kind=1）の行の直後の行が、同じターンの手番の席の main（窓を閉じて攻める側の手に戻った）・視点は守る側。
    - hold  : そのターンの手番の席の最初の main の行・視点は**手番でない席**（前のターンを終えた席＝ドンを残して終えた直後）。
    - oppend: そのターンの手番の席の最後の main の行（手番を終える判断）・視点は手番でない席（相手の手番の終わり）。
    - mid   : そのターンの手番の席の main の行で、最初でも最後でもないもの（メインの途中）・視点は手番の席（評価だけ）。"""
    o = np.argsort(step, kind="stable")
    tv, wv, kv, sv = turn[o], who[o], kind[o], step[o]
    out = {s: [] for s in SHAPES + EVAL_SHAPES}
    mains = {}
    for j in range(len(o)):
        t, w = int(tv[j]), int(wv[j])
        tp = 0 if t % 2 == 1 else 1
        if t >= MIN_TURN and int(kv[j]) == 0 and w == tp:
            mains.setdefault(t, []).append(j)
        if t >= MIN_TURN and int(kv[j]) == 1 and w != tp and j + 1 < len(o):
            if int(tv[j + 1]) == t and int(kv[j + 1]) == 0 and int(wv[j + 1]) == tp:
                out["close"].append((int(o[j + 1]), int(sv[j + 1]), w, t))
    for t, js in mains.items():
        tp = 0 if t % 2 == 1 else 1
        f, l = js[0], js[-1]
        out["head"].append((int(o[f]), int(sv[f]), tp, t))
        out["hold"].append((int(o[f]), int(sv[f]), 1 - tp, t))
        out["oppend"].append((int(o[l]), int(sv[l]), 1 - tp, t))
        for j in js[1:-1]:
            out["mid"].append((int(o[j]), int(sv[j]), tp, t))
    return out


def cmd_plan(a):
    t0 = time.time()
    games = []
    for w in WAVES:
        dirs = sorted(glob.glob(os.path.expanduser(WAVE_GLOB.format(w=w))))
        if not dirs:
            raise SystemExit(f"波 {w} の記録が無い（{WAVE_GLOB.format(w=w)}）")
        for d in dirs:
            for f in sorted(glob.glob(os.path.join(d, "n_record_*.npz"))):
                with np.load(f, allow_pickle=False) as z:
                    seed, step, turn, who, kind = (z[k] for k in ("seed", "step", "turn", "who", "kind"))
                rel = f.replace(os.path.expanduser("~"), "~", 1)
                for s in np.unique(seed):
                    ix = np.flatnonzero(seed == s)
                    cand = shape_rows(turn[ix], who[ix], kind[ix], step[ix])
                    pos = []
                    for sh in SHAPES + EVAL_SHAPES:
                        cs = cand[sh]
                        if not cs:
                            continue
                        r, st, wh, t = cs[_h(s, sh) % len(cs)]
                        pos.append([sh, int(ix[r]), st, wh, t, 1 if _h(s, sh, "sign") % 2 else -1])
                    if all(any(p[0] == sh for p in pos) for sh in SHAPES):
                        games.append([w, rel, int(s), pos])
    print(f"局 {len(games)}（{time.time() - t0:.0f}s）", flush=True)
    games.sort(key=lambda g: _h(g[2]))
    sel = games[:a.games]
    # 評価だけの形（mid）は先頭 `eval_games` 局だけに残す（費用を抑える）
    for n, g in enumerate(sel):
        if n >= a.eval_games:
            g[3] = [p for p in g[3] if p[0] not in EVAL_SHAPES]
    npos = {sh: sum(1 for g in sel for p in g[3] if p[0] == sh) for sh in SHAPES + EVAL_SHAPES}
    os.makedirs(os.path.dirname(a.positions), exist_ok=True)
    with gzip.open(a.positions, "wt", encoding="utf-8") as fo:
        json.dump({"offset": OFFSET, "net": NET, "level": LEVEL, "k": K, "games_total": len(games), "games": sel},
                  fo, separators=(",", ":"))
    print(f"選んだ局 {len(sel)}（取り置き seed%7==0 {sum(1 for g in sel if g[2] % 7 == 0)}）・形ごとの盤面 {npos} → {a.positions}",
          flush=True)


def load_positions(path):
    with gzip.open(path, "rt", encoding="utf-8") as fi:
        return json.load(fi)


# ---------------------------------------------------------------- 対を作って打ち続ける（段 1 の教材）

def encode(g, who):
    enc = json.loads(g.encode(who))
    ci = np.zeros(N_TOK, np.int16)
    src = np.asarray(enc["card_idx"], np.int64)[:N_TOK]
    ci[:len(src)] = src
    return (np.asarray(enc["scalars"], np.float16), ci, np.asarray(enc["tokens"], np.float16).reshape(N_TOK, -1))


def make_pair(g, seat, shape, sign):
    """盤面 `g` から (少ない側, 多い側, 動かした向き) を作る（`Game.shift_don`・`g` は書き換えない）。
    sign=−1 → (g から 1 枚戻した盤面, g)・sign=+1 → (g, g に 1 枚足した盤面)。できない向きならもう一方へ振り替え。
    hold は「控えたドンが 1 枚多い／少ない」にしたいので、アクティブが 0 枚なら必ず +1（アクティブを 1 枚足す）。"""
    h = json.loads(g.hidden_json())["players"][seat]["don"]
    if shape == "hold" and not h["active"]:
        sign = 1
    order = (sign, -sign)
    for sg in order:
        x = g.fork()
        src = x.shift_don(seat, sg)
        if src is None:
            continue
        return (x, g.fork(), sg, src) if sg == -1 else (g.fork(), x, sg, src)
    return None, None, 0, None


def _init_play():
    E.engine()
    E.load_net(NET)
    LR._G["specs"] = {LEVEL: LR.spec_of(LEVEL)}


def play_game(job):
    import rollout_teach as RT
    wave, rel, seed, pos = job
    t0 = time.time()
    rows = []
    try:
        rows = RT._file_games(os.path.expanduser(rel))[seed]
        got, why = RT.replay_to_steps(seed, rows, {p[2] for p in pos})
    except Exception as exc:  # noqa: BLE001
        got, why = {}, "%s: %s" % (type(exc).__name__, str(exc)[:80])
    out = []
    zby = {r["_i"]: float(r["z"]) for r in rows}
    for sh, i, st, wh, t, sign in pos:
        base = {"wave": wave, "file": rel, "seed": seed, "shape": sh, "row": i, "step": st, "who": wh, "turn": t}
        if st not in got:
            out.append(dict(base, ok=False, fail=why or "missing"))
            continue
        seat = "p1" if wh == 0 else "p2"
        try:
            g = got[st]
            if g.winner() is not None:
                out.append(dict(base, ok=False, fail="ended"))
                continue
            gl, gm, sg, src = make_pair(g, seat, sh, sign)
            if gl is None:
                out.append(dict(base, ok=False, fail="no_shift"))
                continue
            wl, wm, secs = [], [], 0.0
            for k in range(K):
                ws = MJ.world_seed(seed, st, OFFSET + k)
                for gg, acc in ((gl, wl), (gm, wm)):
                    r, _n, sec = LR.rollout(gg.fork_world(seat, "fair", int(ws)), seat, ws, LEVEL)
                    acc.append(r)
                    secs += sec
            el, em = encode(gl, seat), encode(gm, seat)
            both = [(x, y) for x, y in zip(wl, wm) if x is not None and y is not None]
            out.append(dict(base, ok=bool(both), sign=sg, src=src, z=zby.get(i),
                            wl="".join("x" if x is None else str(x) for x in wl),
                            wm="".join("x" if x is None else str(x) for x in wm),
                            n=len(both), d=round(2.0 * float(np.mean([y - x for x, y in both])), 5) if both else None,
                            ml=round(float(np.mean([2 * x - 1 for x, _ in both])), 5) if both else None,
                            mm=round(float(np.mean([2 * y - 1 for _, y in both])), 5) if both else None,
                            sec=round(secs, 2),
                            enc=[[e[0].tolist(), e[1].tolist(), e[2].tolist()] for e in (el, em)]))
        except Exception as exc:  # noqa: BLE001
            out.append(dict(base, ok=False, fail="%s: %s" % (type(exc).__name__, str(exc)[:80])))
    return out, round(time.time() - t0, 2)


def branch_of(rel):
    """記録のパス → 生成の結果ブランチ（例 `~/n70f/n_records/n70f_w07_p2/…` → `claude/n70f-w07`）。"""
    d = rel.split("/n_records/")[1].split("/")[0]          # n70f_w07_p2
    w, k = d.split("_")[0], d.split("_")[1]
    return f"claude/{w}-{k}"


def shard_of_branch(br, sn):
    """ブランチ → シャードの番号（1 始まり）。波 4 つ × 32 本＝128 本を順に並べて sn で割った余り＝1 本のシャードは 128/sn 本だけ取ればよい。"""
    w, k = br.split("/")[1].split("-")
    return (WAVES.index(w) * 32 + int(k[1:]) - 1) % sn + 1


def shard_games(P, shard):
    si, sn = (int(x) for x in shard.split("/"))
    gs = [g for g in P["games"] if shard_of_branch(branch_of(g[1]), sn) == si]
    return sorted(gs, key=lambda g: (g[1], g[2]))      # 同じファイルを続けて読む


def cmd_need(a):
    """シャードが取ってくる生成の結果ブランチの一覧（1 行 1 本）。"""
    P = load_positions(a.positions)
    print("\n".join(sorted({branch_of(g[1]) for g in shard_games(P, a.shard)})))


def read_rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def cmd_play(a):
    P = load_positions(a.positions)
    games = shard_games(P, a.shard) if a.shard else list(P["games"])
    if a.limit:
        games = games[:a.limit]
    tag = a.tag or (f"s{int(a.shard.split('/')[0]):02d}" if a.shard else "local")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"pairs_{tag}.jsonl")
    done = {(r["seed"], r["shape"]) for r in BA.read_jsonl(path)} if os.path.exists(path) else set()
    todo = [[g[0], g[1], g[2], [p for p in g[3] if (g[2], p[0]) not in done]] for g in games]
    todo = [g for g in todo if g[3]]
    npos = sum(len(g[3]) for g in todo)
    print(f"PLAY shard={a.shard} 局 {len(todo)}/{len(games)}・盤面 {npos} workers={a.workers} -> {path}", flush=True)
    t0 = time.time()
    n = nf = 0
    with get_context("fork").Pool(a.workers, initializer=_init_play) as pool:
        for k, (rs, sec) in enumerate(pool.imap_unordered(play_game, todo, chunksize=1), 1):
            with open(path, "a", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rs)
            nf += sum(1 for r in rs if not r.get("ok"))
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k}/{len(todo)} 局] 盤面 {n}/{npos}・失敗 {nf}・{el:.0f}s・残り見込み {el / n * (npos - n) / 60:.0f} 分",
                      flush=True)
    print(f"PLAY_DONE 盤面 {n}・失敗 {nf}・{time.time() - t0:.0f}s", flush=True)


def cmd_merge(a):
    """シャードの行 → `pairs.npz`（訓練器の `--pair-diff` が読む・`opcg_sim.learned.train.pair_diff.PAIR_KEYS`）。"""
    from opcg_sim.learned.train import pair_diff as PD
    rows = {}
    for f in sorted({f for pat in a.rows for f in glob.glob(pat)}):
        for r in read_rows(f):
            if r.get("ok") and r.get("n"):
                rows[(r["seed"], r["shape"])] = r
    rs = [rows[k] for k in sorted(rows)]
    if not rs:
        raise SystemExit("対が 1 つも無い")
    out = {"sc_l": np.array([r["enc"][0][0] for r in rs], np.float16), "ci_l": np.array([r["enc"][0][1] for r in rs], np.int16),
           "tok_l": np.array([r["enc"][0][2] for r in rs], np.float16),
           "sc_m": np.array([r["enc"][1][0] for r in rs], np.float16), "ci_m": np.array([r["enc"][1][1] for r in rs], np.int16),
           "tok_m": np.array([r["enc"][1][2] for r in rs], np.float16),
           "d": np.array([r["d"] for r in rs], np.float32), "n": np.array([r["n"] for r in rs], np.int16),
           "ml": np.array([r["ml"] for r in rs], np.float32), "mm": np.array([r["mm"] for r in rs], np.float32),
           "seed": np.array([r["seed"] for r in rs], np.int64), "turn": np.array([r["turn"] for r in rs], np.int16),
           "shape": np.array([r["shape"] for r in rs]), "sign": np.array([r["sign"] for r in rs], np.int8)}
    assert set(PD.PAIR_KEYS) <= set(out), set(PD.PAIR_KEYS) - set(out)
    np.savez_compressed(a.pairs, **out)
    by = {sh: int((out["shape"] == sh).sum()) for sh in SHAPES + EVAL_SHAPES}
    hold = out["seed"] % 7 == 0
    print(f"MERGE 対 {len(rs)}・形ごと {by}・取り置き {int(hold.sum())} → {a.pairs}", flush=True)
    for sh in SHAPES + EVAL_SHAPES:
        m = out["shape"] == sh
        if m.any():
            print(f"  {sh}: 教師の差の平均 {out['d'][m].mean():+.4f}（符号 +1 の対 {int((out['sign'][m] == 1).sum())}）"
                  f" 平均 {out['ml'][m].mean():+.3f}/{out['mm'][m].mean():+.3f}", flush=True)


# ---------------------------------------------------------------- 試験台: 対の当てはまり（形ごと）

def cmd_beds(a):
    """各ネットの予測の差（多い − 少ない）を対ごとに出し、形 × {取り置きの局, 訓練の局} で教師の差と比べる。
    比＝予測の差の平均 ÷ 教師の差の平均（教師の雑音で縮まない読み）。傾き・相関は対ごと（教師の雑音で縮む）。
    `--ref` のネットとの対にした差（同じ対で 予測の差 − ref の予測の差 の平均と 95% 区間）も出す。"""
    from value_teach import net_values
    from opcg_sim.learned.train import pair_diff as PD
    P = PD.load_pairs([a.pairs])
    hold = P["seed"] % 7 == 0
    allix = np.arange(len(P["d"]))
    dv = {}
    for net in a.nets:
        dv[net] = PD.predict_diff(lambda sc, ci, tok, n=net: net_values(n, sc, ci, tok), P, allix)
    res = {"pairs": int(len(allix)), "groups": {}, "nets": {}}
    groups = {}
    for sh in sorted(set(P["shape"].tolist())):
        for nm, m in (("hold", hold), ("train", ~hold)):
            mm = (P["shape"] == sh) & m
            if mm.any():
                groups[f"{sh}_{nm}"] = mm
    for g, mm in groups.items():
        dt = P["d"][mm].astype(np.float64)
        res["groups"][g] = {"n": int(mm.sum()), "teacher_mean": round(float(dt.mean()), 4),
                            "teacher_ci": round(1.96 * float(dt.std(ddof=1)) / math.sqrt(len(dt)), 4)}
    for net in a.nets:
        row = {}
        for g, mm in groups.items():
            st = PD.fit_stats(dv[net][mm], P["d"][mm])
            st["ratio"] = round(st["dv_mean"] / st["dt_mean"], 3) if st["dt_mean"] else None
            if a.ref and a.ref in dv and net != a.ref:
                x = dv[net][mm] - dv[a.ref][mm]
                m_, h_ = _ci(x)
                st["vs_ref"] = [round(m_, 4), round(h_, 4)]
            row[g] = st
        res["nets"][os.path.basename(net)] = row
        print(os.path.basename(net), " ".join(f"{g} {x['dv_mean']:+.3f}({x['ratio']})" for g, x in row.items()), flush=True)
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    json.dump(res, open(a.out_json, "w"), ensure_ascii=False, indent=1)
    print("teacher", " ".join(f"{g} {x['teacher_mean']:+.3f}±{x['teacher_ci']:.3f}" for g, x in res["groups"].items()), flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--out", default=OUT_DIR)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--summary-only", action="store_true")
    t.add_argument("--task", choices=("all", "kaido", "jitter"), default="all")
    p = sub.add_parser("plan")
    p.add_argument("--positions", default=POSITIONS)
    p.add_argument("--games", type=int, default=12000)
    p.add_argument("--eval-games", type=int, default=2000)
    q = sub.add_parser("play")
    q.add_argument("--positions", default=POSITIONS)
    q.add_argument("--out", default=OUT_DIR)
    q.add_argument("--shard", default="")
    q.add_argument("--tag", default="")
    q.add_argument("--workers", type=int, default=4)
    q.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--rows", nargs="+", default=[os.path.join(OUT_DIR, "pairs_*.jsonl*")])
    m.add_argument("--pairs", default=os.path.join(OUT_DIR, "pairs.npz"))
    b = sub.add_parser("beds")
    b.add_argument("--pairs", default=os.path.join(OUT_DIR, "pairs.npz"))
    b.add_argument("--nets", nargs="+", required=True)
    b.add_argument("--ref", default="")
    b.add_argument("--out-json", default=os.path.join(OUT_DIR, "beds_pairs.json"))
    nd = sub.add_parser("need")
    nd.add_argument("--positions", default=POSITIONS)
    nd.add_argument("--shard", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "need":
        return cmd_need(a)
    if a.cmd == "teach":
        return teach_summary(a) if a.summary_only else cmd_teach(a)
    return {"plan": cmd_plan, "play": cmd_play, "merge": cmd_merge, "beds": cmd_beds}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
