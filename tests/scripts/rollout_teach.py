"""WP `rollout-teach` 段 1（`docs/search_efficiency_plan.md` §7.80・台帳 `cpu_theory_gap.md` 2+518）: 打ち続けの平均の教師を作る。

問い: 価値の教師の雑音を減らすと、小さな効き（ドン 1 枚の差）を縮めずに覚えるか。§7.79 の推薦どおり、
既存の波（70f〜73f＝r29f4 と同じ教材）の局から 1 局 4 盤面（自分の手番の頭・手番 3 以降から等間隔・両席を交互に）を選び、
各盤面で見えない札を配り直して（公平・配り方の番号のずらし 19000）、両席とも r28x4・読み 16 回・世界 1 本で最後まで 16 局打ち、
勝敗（その行の手番の席から見て ±1）の平均を教師にする。盤面は記録の手の当て直しで復元する（符号化がビット一致した盤面だけ使う）。

サブコマンド:
  ``plan``     4 波の記録から盤面を選び `positions.json.gz`（局ごとの [波, シャードのパス, 局 seed, [(行, step, 席, ターン)…]]）を書く。
               局の選びは局 seed のハッシュの順に `N_GAMES` 局（決定的）。シャードの割り当ては選んだ局の順位 % 本数。
  ``play``     ``--shard i/30``: 自分の局だけ当て直して打ち続け、`rows_sKK.jsonl` に 1 盤面 1 行で追記（再開できる）。
  ``sidecar``  集めた行（`--rows` の glob）から記録の隣に `n_record_XXXXX.roll.npz`（`opcg_sim.learned.train.rollout_labels`）を書く。
  ``check``    不変量: §7.79 の 400 行（`n1_results/light_rollout/rows.json`）の一部を、同じ世界（ずらし 18000）で
               この道具の打ち続けで打ち直し、`jitter_local.jsonl` の読み 16 回の勝敗と 1 局ずつ一致するかを見る。

    OPCG_LOG_SILENT=1 python tests/scripts/rollout_teach.py plan
    OPCG_LOG_SILENT=1 python tests/scripts/rollout_teach.py play --shard 3/30 --workers 4
    OPCG_LOG_SILENT=1 python tests/scripts/rollout_teach.py sidecar --rows 'n1_results/rollout_teach/rows_s*.jsonl'
    OPCG_LOG_SILENT=1 python tests/scripts/rollout_teach.py check --n 12

理論の道具ではなく教師の作りだが、1 回きりの生成＝道具のテストは足さない（訓練器の受け口は `tests/test_rollout_teach.py`）。
"""
import argparse
import glob
import gzip
import hashlib
import json
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
import omni_value_relabel as OR  # noqa: E402

from opcg_sim.learned.train import dump_io as DIO  # noqa: E402
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "rollout_teach")
POSITIONS = os.path.join(OUT_DIR, "positions.json.gz")
#: 配り方の番号のずらし（§7.79 の 18000 とも、それより前の 14000〜17000 とも重ねない）
OFFSET = 19000
NET = LR.NET                                   # r28x4（波 70f〜73f の生成役）
LEVEL = "L16"                                  # 読み 16 回・世界 1 本（§7.79 の推薦）
K = 16                                         # 1 盤面の局数
N_GAMES = 25000                                # 1 局 4 盤面 × 2.5 万局 ≒ 10 万盤面
PER_GAME = 4
MIN_TURN = 3
N_SHARDS = 30
WAVES = ("n70f", "n71f", "n72f", "n73f")
WAVE_GLOB = "~/{w}/n_records/{w}_w*_p[0-3]"
COLS = ("seed", "step", "turn", "who", "z", "sig", "forced", "forced_sig", "scalars", "tokens", "card_idx",
        "pol_len", "pol_chosen", "pol_k")
_G = {}


# ---------------------------------------------------------------- 盤面の選び

def _h(seed):
    return hashlib.sha1(f"rollout-teach:{int(seed)}".encode()).hexdigest()


def heads_of(turn, who, kind, step):
    """1 局の行（step 順）→ 自分の手番の頭 [(行の添字, step, 席, ターン)]（そのターンの最初の main 行・ターン ≥ MIN_TURN）。"""
    out, seen = [], set()
    for i in np.argsort(step, kind="stable"):
        t, w = int(turn[i]), int(who[i])
        if t >= MIN_TURN and int(kind[i]) == 0 and ((t % 2 == 1) == (w == 0)) and (w, t) not in seen:
            seen.add((w, t))
            out.append((int(i), int(step[i]), w, t))
    return out


def pick(heads, seed):
    """手番の頭から 4 つ: 席を交互に（始めの席は局 seed のハッシュ）、各席の頭から等間隔に。足りなければある分だけ。"""
    by = {0: [h for h in heads if h[2] == 0], 1: [h for h in heads if h[2] == 1]}
    s0 = int(_h(seed)[-1], 16) % 2
    slots = {0: [], 1: []}
    for i in range(PER_GAME):
        slots[(s0 + i) % 2].append(i)
    out = []
    for w in (0, 1):
        hs, m = by[w], len(slots[w])
        if not hs:
            continue
        ix = sorted({min(len(hs) - 1, int((j + 0.5) * len(hs) / m)) for j in range(m)})
        out += [hs[j] for j in ix]
    return sorted(out, key=lambda h: h[1])


def cmd_plan(a):
    t0 = time.time()
    games = []
    for w in WAVES:
        dirs = sorted(glob.glob(os.path.expanduser(WAVE_GLOB.format(w=w))))
        if not dirs:
            raise SystemExit(f"波 {w} の記録が無い（{WAVE_GLOB.format(w=w)}）")
        for d in dirs:
            for f in DIO.shard_files(d):
                with np.load(f, allow_pickle=False) as z:
                    seed, step, turn, who, kind = (z[k] for k in ("seed", "step", "turn", "who", "kind"))
                rel = f.replace(os.path.expanduser("~"), "~", 1)
                for s in np.unique(seed):
                    ix = np.flatnonzero(seed == s)
                    hs = heads_of(turn[ix], who[ix], kind[ix], step[ix])
                    ps = pick(hs, s)
                    if ps:
                        games.append([w, rel, int(s), [[int(ix[i]), st, wh, t] for i, st, wh, t in ps]])
    print(f"局 {len(games)}（{time.time() - t0:.0f}s）", flush=True)
    games.sort(key=lambda g: _h(g[2]))
    sel = games[:a.games]
    npos = sum(len(g[3]) for g in sel)
    os.makedirs(os.path.dirname(a.positions), exist_ok=True)
    with gzip.open(a.positions, "wt", encoding="utf-8") as fo:
        json.dump({"offset": OFFSET, "net": NET, "level": LEVEL, "k": K, "games_total": len(games),
                   "games": sel}, fo, separators=(",", ":"))
    by_w = {w: sum(len(g[3]) for g in sel if g[0] == w) for w in WAVES}
    print(f"選んだ局 {len(sel)}・盤面 {npos}・波ごと {by_w}・取り置き（seed%7==0）の盤面 "
          f"{sum(len(g[3]) for g in sel if g[2] % 7 == 0)} → {a.positions}", flush=True)


def load_positions(path):
    with gzip.open(path, "rt", encoding="utf-8") as fi:
        return json.load(fi)


# ---------------------------------------------------------------- 当て直し

def _file_games(f):
    """記録の 1 ファイル → {seed: 行の dict の列（step 順・各 dict に "_i"＝シャードの行）}（1 ファイルだけ覚える）。"""
    if _G.get("file") == f:
        return _G["file_games"]
    with np.load(f, allow_pickle=False) as z:
        cols = {k: z[k] for k in COLS}
    off = np.concatenate([[0], np.cumsum(cols["pol_len"])])
    out = {}
    for s in np.unique(cols["seed"]):
        rows = []
        for i in np.flatnonzero(cols["seed"] == s):
            r = {k: cols[k][i] for k in COLS if not k.startswith("pol_")}
            pl, ch = int(cols["pol_len"][i]), int(cols["pol_chosen"][i])
            r["k"] = int(cols["pol_k"][off[i] + ch]) if pl and 0 <= ch < pl else -1
            r["_i"] = int(i)
            rows.append(r)
        rows.sort(key=lambda r: int(r["step"]))
        out[int(s)] = rows
    _G["file"], _G["file_games"] = f, out
    return out


def replay_to_steps(seed, rows, targets):
    """局 `seed` を記録の手で当て直し、`targets`（step の集合）の各行の直前の盤面（符号化が記録と一致したもの）の複製を返す
    → ({step: 盤面}, 失敗の理由 or None)。手の当て方は `light_rollout.replay_to_jitter` と同じ（候補の幅の探し）。"""
    db = _G.setdefault("db", D.load_db())
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "real")
    beam = [(DR.new_game(seed, p1, p2), [])]
    got, last = {}, max(targets)
    for j, row in enumerate(rows):
        who = "p1" if row["who"] == 0 else "p2"
        if int(row["step"]) != j:
            return got, "step_gap"
        live = [(g, h) for g, h in beam if OR.enc_matches(g, who, row)[0]]
        if not live:
            return got, f"enc_{j}"
        beam = live
        if j in targets:
            got[j] = beam[0][0].fork()
            if j == last:
                return got, None
        sig = OR.applied_sig(row)
        nxt = []
        for g, hist in beam:
            cs = LR._cands(g, who, sig)
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
            return got, f"apply_{j}"
        beam = nxt[:OR.BEAM_MAX]
    return got, "not_reached"


# ---------------------------------------------------------------- 打ち続け

def _init_play():
    E.engine()
    E.load_net(NET)
    LR._G["specs"] = {LEVEL: LR.spec_of(LEVEL)}


def rollouts(g, seat, seed, step, k, offset):
    """盤面 `g` から配り直し `k` 本（公平・`MJ.world_seed(seed, step, offset+i)`）を読み 16 回で最後まで → 勝敗の列（1／0／None）と秒。"""
    ws_, secs = [], []
    for i in range(k):
        ws = MJ.world_seed(seed, step, offset + i)
        gw = g.fork_world(seat, "fair", int(ws))
        r, _st, sec = LR.rollout(gw, seat, ws, LEVEL)
        ws_.append(r)
        secs.append(sec)
    return ws_, secs


def play_game(job):
    wave, rel, seed, pos = job
    t0 = time.time()
    try:
        rows = _file_games(os.path.expanduser(rel))[seed]
        got, why = replay_to_steps(seed, rows, {p[1] for p in pos})
    except Exception as exc:  # noqa: BLE001
        got, why = {}, "%s: %s" % (type(exc).__name__, str(exc)[:80])
    out = []
    for i, st, wh, t in pos:
        base = {"wave": wave, "file": rel, "seed": seed, "row": i, "step": st, "who": wh, "turn": t}
        if st not in got:
            out.append(dict(base, ok=False, fail=why or "missing"))
            continue
        seat = "p1" if wh == 0 else "p2"
        try:
            ws, secs = rollouts(got[st], seat, seed, st, K, OFFSET)
        except Exception as exc:  # noqa: BLE001
            out.append(dict(base, ok=False, fail="%s: %s" % (type(exc).__name__, str(exc)[:80])))
            continue
        done = [2 * x - 1 for x in ws if x is not None]
        z = float(rows[[r["_i"] for r in rows].index(i)]["z"])
        out.append(dict(base, ok=bool(done), w="".join("x" if x is None else str(x) for x in ws),
                        n=len(done), mean=round(float(np.mean(done)), 5) if done else None, z=z,
                        sec=round(float(sum(secs)), 2)))
    return out, round(time.time() - t0, 2)


def shard_games(P, shard):
    si, sn = (int(x) for x in shard.split("/"))
    gs = [g for r, g in enumerate(P["games"]) if r % sn == si - 1]
    return sorted(gs, key=lambda g: (g[1], g[2]))      # 同じファイルを続けて読む


def cmd_play(a):
    P = load_positions(a.positions)
    games = shard_games(P, a.shard) if a.shard else list(P["games"])
    if a.limit:
        games = games[:a.limit]
    tag = a.tag or (f"s{int(a.shard.split('/')[0]):02d}" if a.shard else "local")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"rows_{tag}.jsonl")
    done = {(r["seed"], r["step"]) for r in BA.read_jsonl(path)}
    todo = [g for g in games if any((g[2], p[1]) not in done for p in g[3])]
    todo = [[g[0], g[1], g[2], [p for p in g[3] if (g[2], p[1]) not in done]] for g in todo]
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


# ---------------------------------------------------------------- sidecar

def cmd_sidecar(a):
    from opcg_sim.learned.train import rollout_labels as RL
    files = sorted({f for pat in a.rows for f in glob.glob(pat)})
    by = {}
    nrow = nbad = 0
    for f in files:
        for r in BA.read_jsonl(f):
            nrow += 1
            if not r.get("ok") or not r.get("n"):
                nbad += 1
                continue
            by.setdefault(r["file"], {})[r["row"]] = r
    nw = npos = 0
    for rel, rs in sorted(by.items()):
        sh = os.path.expanduser(rel)
        if not os.path.exists(sh):
            raise SystemExit(f"記録が無い: {sh}")
        rr = sorted(rs)
        RL.write_sidecar(sh, rr, [rs[i]["mean"] for i in rr], [rs[i]["n"] for i in rr],
                         [rs[i]["seed"] for i in rr], [rs[i]["step"] for i in rr])
        nw += 1
        npos += len(rr)
    print(f"sidecar {nw} 本・盤面 {npos}（行 {nrow}・使えない {nbad}）", flush=True)


# ---------------------------------------------------------------- 不変量（§7.79 と同じ世界で一致するか）

def _check_job(item):
    r, k = item
    g = LR._jit_game(r["seed"])
    out = []
    for i in range(k):
        ws = MJ.world_seed(r["seed"], r["row"], LR.OFFSET + i)
        gw = g.fork_world(r["seat"], "fair", int(ws))
        res, _st, _sec = LR.rollout(gw, r["seat"], ws, LEVEL)
        out.append(res)
    return r["seed"], out


def cmd_check(a):
    src = os.path.join("n1_results", "light_rollout")
    rows = json.load(open(os.path.join(src, "rows.json")))["rows"]
    ref = {}
    for x in BA.read_jsonl(os.path.join(src, "jitter_local.jsonl")):
        if x.get("level") == LEVEL and x.get("ok"):
            ref[(x["key"], x["i"])] = x["w"]
    pick_ = rows[::max(1, len(rows) // a.n)][:a.n]
    LR._G["out"] = src

    def init():
        _init_play()
        LR._G["rows"] = {r["seed"]: r for r in rows}

    same = diff = 0
    with get_context("fork").Pool(a.workers, initializer=init) as pool:
        for seed, out in pool.imap_unordered(_check_job, [(r, a.k) for r in pick_]):
            for i, w in enumerate(out):
                if (seed, i) in ref:
                    same += int(ref[(seed, i)] == w)
                    diff += int(ref[(seed, i)] != w)
    res = {"rows": len(pick_), "games": same + diff, "same": same, "diff": diff}
    print("CHECK", json.dumps(res), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "check_light_rollout.json"), "w"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["plan", "play", "sidecar", "check"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--positions", default=POSITIONS)
    ap.add_argument("--games", type=int, default=N_GAMES)
    ap.add_argument("--shard", default=None, help="i/n（i＝1〜n）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0, help="play: 先頭の局だけ（煙試験）")
    ap.add_argument("--rows", nargs="*", default=[os.path.join(OUT_DIR, "rows_s*.jsonl")])
    ap.add_argument("--n", type=int, default=12, help="check: 打ち直す行の数")
    ap.add_argument("--k", type=int, default=K, help="check: 1 行の局数")
    a = ap.parse_args(argv)
    {"plan": cmd_plan, "play": cmd_play, "sidecar": cmd_sidecar, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
