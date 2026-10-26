"""n_rel_after_desk: 方策に「候補を打った後の盤面」を見せると壁が動くかの机上検証（WP `policy-after-desk`・2026-10-25）。

計画 `docs/search_efficiency_plan.md` §7.53・報告 `docs/reports/2026-10-25_policy_after_desk.md`。
**1 回きりの測定の道具**（テストは足さない・CLAUDE.md の「理論の道具」と同じ扱い）。既存のネット定義
（`n_rel.py`・`n_rel_torch.py`）も npz も、エンジン・符号化も変えない。比べ方・指標は §7.52 の
`n_rel_attn_desk.py`（ブランチ `claude/policy-attn-desk`）と同じ定義（必要な部分をここへ写した）。

2 つの段:

  build — 記録（`record_gen` の dump）の局を**記録の手の列から**再生し（探索は回さない）、方策点
          （main・候補 2 本以上・選択が判っている行＝`dump_io._read_policy` と同じ行）の各候補について、
          `Game.fork_world(席, "fair", seed)` → `apply_search_move`（探索が木の中で使う適用・DON_BOX は
          付与 k 回＋攻撃・分岐する自分の選択は残す）→ `encode(席)` で**打った直後の盤面**を符号化 v15 のまま
          書く（`after_<shard>.npz`・候補の並びは `_read_policy` の C と同じ）。
          - 隠れた札: 公平な世界（自分の山札と裏向きのライフも混ぜ・相手の見えない札は引き直し・公開・見た札は固定）で
            打つ＝打った後に引く札・覗く札は**本物を見せない**（その世界の札になる）。
          - 戦闘・対話の途中で止まる候補は、その時点の盤面（窓は閉じない）。
          - 再生は各行で `encode` が記録の `scalars`／`tokens`／`card_idx` とビット一致することを確かめる。
            合わなくなった局はそこで打ち切り、以降の点は `ok=0`。
  train — §7.52 と同じ訓練ループで `--head base`（今の頭）と `--head after`（打った後の盤面を見る頭）を比べる。
          after の頭: 候補ごとに打った後の盤面を**今の胴体（トークン → 要約 e）と共有の重み**で通し、
          e_after・e_after − e_now・v_after（今の価値の頭を打った後の盤面に当てたもの）・手番が自分のままか・
          決着（自分の勝ち/負け）を、今の頭の第 1 層の入力に足す:
            logit = relu(u·Wp1 + x·Wp1a + bp1)·Wp2 + bp2,  x = [e_a, e_a − e, v_a, pend, win]
          r24 から始める組は Wp1a を 0 で始める（最初の出力は r24 と同じ）。

実行例:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_after_desk build \\
    --in "data/n_records/n54_w0[1-8]_p*" --out data/after --workers 4
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_after_desk train \\
    --in "data/n_records/n54_w0[1-8]_p*" --after data/after --head after \\
    --warm-start opcg_sim/data/learned/nrel_r24.npz --epochs 3 --lr 2e-4 --out-json runs/after_r24.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import hashlib
import json
import sys
import time

import numpy as np

N_TOK_ = 22
D_SC_ = 127
D_X_EXTRA = 3                 # v_a・手番が自分のままか・決着
Q_GAP = 0.1                   # 本当の読み違いの閾値（`2026-10-20_worlds4_and_policy_errors.md` と同じ）
_AT_PLAY = 0
_AT_ATK = (2, 3, 4)           # 攻撃・戦闘箱（DON_BOX）・ドンの付与（`dump_io._atype_idx` の並び）


# ---------------------------------------------------------------------------
# build: 記録の再生と、打った直後の盤面の符号化
# ---------------------------------------------------------------------------
def _move_sig(mv):
    from opcg_sim.loop import record_gen as RG
    return RG.move_sig(mv)


def cand_move(sig, k, legal):
    """記録の候補の鍵（`pol_sig`・`pol_k`）→ 適用できる手（JSON）。作れなければ None。"""
    at = sig[0]
    if at == "DON_BOX":
        return {"kind": "game", "action_type": "DON_BOX",
                "payload": {"uuid": sig[1], "target_ids": list(sig[2]), "don_k": int(k)}}
    m = [mv for mv in legal if _move_sig(mv) == sig]
    if m:
        return m[0]
    if at == "RESOLVE_EFFECT_SELECTION":
        # `legal_json` は対象選択を既定の 1 本でしか返さない＝その形に記録の選択を写す
        base = [mv for mv in legal if mv.get("action_type") == at]
        if base:
            mv = json.loads(json.dumps(base[0]))
            p = mv.setdefault("payload", {})
            p["selected_uuids"] = list(sig[3])
            p["accepted"] = sig[4]
            if len(sig) > 5:
                p.update(sig[5])
            return mv
    return None


def played_move(z, i, ptr, legal):
    """記録の行 i で実対局に出た原始手（箱は先頭原始手・ε で差し替えた行は `forced_sig`）。"""
    if int(z["forced"][i]):
        sig = json.loads(str(z["forced_sig"][i]))
    else:
        sig = json.loads(str(z["sig"][i]))
        if sig[0] == "DON_BOX":
            k = int(z["pol_k"][ptr[i] + int(z["pol_chosen"][i])])
            sig = (["ATTACH_DON", sig[1], [], [], None] if k > 0 else ["ATTACK", sig[1], list(sig[2]), [], None])
    return cand_move(sig, 0, legal)


def _apply(game, name, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(name, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(name, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def _world_seed(seed, step):
    return (int(seed) * 1000003 + int(step) * 7919 + 0x5EED) & 0xFFFFFFFFFFFF


def build_file(args):
    """1 シャード（`n_record_*.npz`）→ `after_*.npz`。戻り値は数え上げ。

    `persp`＝"actor"（打った席から符号化・既定）／"pending"（打った後に決める席から符号化。相手へ移った候補は
    相手の席から見た盤面になり、`a_pend=0` が「符号化の席＝相手」の印。価値は訓練側で符号を返す）。
    """
    path, out_path = args[:2]
    persp = args[2] if len(args) > 2 else "actor"
    resolve = bool(args[3]) if len(args) > 3 else False
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    E.load_net(None)                         # 符号化の語彙（ネット付属の vocab_ids）
    # `resolve`＝候補を当てた後の戦闘・対話の窓を、探索が葉を評価する前に閉じる規則で閉じる（`Game.resolve_windows`）
    spec = _G_DB.setdefault("spec", E.SeatSpec(None, sims=160)) if resolve else None
    db = _G_DB.setdefault("db", D.load_db())
    d = os.path.dirname(path)
    gm = {int(g["seed"]): g for g in json.load(open(os.path.join(d, "meta_games.json")))["games"]}
    z = dict(np.load(path))
    t0 = time.time()
    pl, pc, kind = z["pol_len"], z["pol_chosen"], z["kind"]
    ptr = np.concatenate([[0], np.cumsum(pl)]).astype(np.int64)
    take = np.where((kind == 0) & (pl >= 2) & (pc >= 0))[0]
    # 候補行 → 出力の位置（`_read_policy` の並び＝行の昇順・行の中は候補の順）
    out_off = {}
    o = 0
    for i in take:
        out_off[int(i)] = o
        o += int(pl[i])
    n_out = o
    a_sc = np.zeros((n_out, D_SC_), np.float16)
    a_ci = np.zeros((n_out, N_TOK_), np.int16)
    a_tok = np.zeros((n_out, N_TOK_, N_TOK_), np.float16)
    a_ok = np.zeros(n_out, np.uint8)
    a_pend = np.zeros(n_out, np.uint8)
    a_win = np.zeros(n_out, np.int8)
    st = {"games": 0, "diverged": 0, "pts": len(take), "pts_ok": 0, "cands": n_out, "cand_fail": 0}
    for sd in np.unique(z["seed"]):
        g = gm.get(int(sd))
        if g is None:
            continue
        st["games"] += 1
        idx = np.nonzero(z["seed"] == sd)[0]
        idx = idx[np.argsort(z["step"][idx], kind="stable")]
        p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], int(sd), "real")
        game = DR.new_game(int(sd), p1, p2)
        for i in idx:
            try:
                pend = json.loads(game.pending_json()) or {}
                name = pend.get("player_id")
                if name != ("p1" if int(z["who"][i]) == 0 else "p2"):
                    raise ValueError("手番が合わない")
                enc = json.loads(game.encode(name))
                if not (np.array_equal(np.asarray(enc["scalars"], np.float32).astype(np.float16), z["scalars"][i])
                        and np.array_equal(np.asarray(enc["tokens"], np.float32).reshape(N_TOK_, N_TOK_)
                                           .astype(np.float16), z["tokens"][i])):
                    raise ValueError("符号化が記録と合わない")
                legal = json.loads(game.legal_json(name))
            except Exception:                                  # noqa: BLE001  局の再生が外れたら打ち切る
                st["diverged"] += 1
                break
            if int(i) in out_off:
                o0 = out_off[int(i)]
                wseed = _world_seed(sd, z["step"][i])
                good = True
                for j in range(int(ptr[i]), int(ptr[i + 1])):
                    q = o0 + j - int(ptr[i])
                    mv = cand_move(json.loads(str(z["pol_sig"][j])), z["pol_k"][j], legal)
                    if mv is None:
                        good = False
                        continue
                    try:
                        w = game.fork_world(name, "fair", wseed)
                        w.apply_search_move(name, json.dumps(mv, ensure_ascii=False))
                        if resolve and w.winner() is None:
                            w.resolve_windows(spec.decide_opts(int(sd), int(z["turn"][i]), name, None))
                        pn = (json.loads(w.pending_json()) or {}).get("player_id")
                        ea = json.loads(w.encode(pn if (persp == "pending" and pn) else name))
                        win = w.winner()
                    except Exception:                          # noqa: BLE001
                        good = False
                        continue
                    a_sc[q] = np.asarray(ea["scalars"], np.float32)
                    ci = np.asarray(ea["card_idx"], np.int64)[:N_TOK_]
                    a_ci[q, :len(ci)] = ci
                    a_tok[q] = np.asarray(ea["tokens"], np.float32).reshape(N_TOK_, N_TOK_)
                    a_ok[q] = 1
                    a_pend[q] = 1 if pn == name else 0
                    a_win[q] = 0 if win is None else (1 if win == name else -1)
                if good:
                    st["pts_ok"] += 1
                else:
                    st["cand_fail"] += 1
            mv = played_move(z, i, ptr, legal)
            try:
                if mv is None:
                    raise ValueError("手が作れない")
                _apply(game, name, mv)
            except Exception:                                  # noqa: BLE001
                st["diverged"] += 1
                break
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, a_sc=a_sc, a_ci=a_ci, a_tok=a_tok, a_ok=a_ok, a_pend=a_pend, a_win=a_win)
    os.replace(tmp, out_path)
    st["sec"] = round(time.time() - t0, 1)
    st["file"] = path
    return st


_G_DB = {}


def after_path(after_root, shard):
    part = os.path.basename(os.path.dirname(shard))
    return os.path.join(after_root, part, os.path.basename(shard).replace("n_record_", "after_"))


def cmd_build(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--persp", choices=("actor", "pending"), default="actor")
    ap.add_argument("--resolve", action="store_true", help="候補を当てた後の窓を葉の規則で閉じてから符号化する")
    args = ap.parse_args(argv)
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_train as NT
    jobs = []
    for d in NT._expand(args.src):
        for f in DIO.shard_files(d):
            op = after_path(args.out, f)
            if os.path.exists(op):
                continue
            os.makedirs(os.path.dirname(op), exist_ok=True)
            jobs.append((f, op, args.persp, args.resolve))
    print(f"build {len(jobs)} shards", flush=True)
    log = open(os.path.join(args.out, "build_log.jsonl"), "a")
    t0 = time.time()
    if args.workers <= 1:
        it = map(build_file, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(args.workers)
        it = pool.imap_unordered(build_file, jobs)
    for k, st in enumerate(it):
        log.write(json.dumps(st) + "\n"); log.flush()
        print(f"[{k+1}/{len(jobs)}] {st['file']} pts {st['pts_ok']}/{st['pts']} div {st['diverged']} "
              f"{st['sec']}s total {time.time()-t0:.0f}s", flush=True)
    print("AFTER_BUILD_DONE", flush=True)
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def load_after(dirs, after_root, cache_dir):
    """`after_*.npz` を `load_dump` の C と同じ並びで 1 本につなぎ、cache に .npy で置いて memmap で返す。"""
    from opcg_sim.learned.train import dump_io as DIO
    files = [f for d in dirs for f in DIO.shard_files(d)]
    key = hashlib.sha1("\n".join(os.path.abspath(after_path(after_root, f)) for f in files).encode()).hexdigest()[:12]
    cd = os.path.join(cache_dir, "after_" + key)
    cols = ("a_sc", "a_ci", "a_tok", "a_ok", "a_pend", "a_win")
    if not os.path.exists(os.path.join(cd, "done")):
        os.makedirs(cd, exist_ok=True)
        lens = []
        for f in files:
            with np.load(after_path(after_root, f)) as a:
                lens.append(len(a["a_ok"]))
        n = int(sum(lens))
        mm = {}
        for c in cols:
            with np.load(after_path(after_root, files[0])) as a:
                shp, dt = a[c].shape[1:], a[c].dtype
            mm[c] = np.lib.format.open_memmap(os.path.join(cd, c + ".npy"), mode="w+", dtype=dt, shape=(n,) + shp)
        o = 0
        for f, ln in zip(files, lens):
            with np.load(after_path(after_root, f)) as a:
                for c in cols:
                    mm[c][o:o + ln] = a[c]
            o += ln
        for c in cols:
            mm[c].flush()
        del mm
        open(os.path.join(cd, "done"), "w").write(str(n))
    return {c: np.load(os.path.join(cd, c + ".npy"), mmap_mode="r") for c in cols}


def _make_classes():
    import torch
    from opcg_sim.learned.n_rel import D_E
    from opcg_sim.learned.train import n_rel_torch as TT

    class TorchNRelAfter(TT.TorchNRel):
        """`TorchNRel` の方策だけを差し替えたもの（`after=False` なら今の頭そのもの）。"""

        def __init__(self, net, after=False, x_scale=0.0, seed=13, detach=False, persp_pending=False):
            super().__init__(net)
            self.persp_pending = bool(persp_pending)
            self.after = bool(after)
            self.detach = bool(detach)
            self.factor_pairs = True        # R を遮断した式（値は丸めの範囲で同じ・全組で同じ）
            self.fast_max = True
            if self.after:
                torch.manual_seed(seed)
                d_in = 2 * D_E + D_X_EXTRA
                hp = self.Wp1.shape[1]
                self.Wp1a = torch.nn.Parameter(torch.randn(d_in, hp) * (x_scale / np.sqrt(d_in)))

        def rel_zeros(self, B):
            return None, None               # factor_pairs＋R 遮断では読まない（幅ごとの使い回しも要らない）

        def policy_logits_after(self, sc, ci, tok, b, A=None):
            tab = self.card_table()
            feats = self.cand_feats(tab, b.at, b.cid, b.tcid, b.tail)
            h, present = self.tokens_forward(ci, tok, None, None, tab)
            e = self.body(sc, h, present)
            u = self.cand_input(e, h, b.seg, b.si, b.ti, b.ok_s, b.ok_t, b.rr, feats, b.budget)
            pre = u @ self.Wp1 + self.bp1
            if self.after:
                a_sc, a_ci, a_tok, a_pend, a_win = A
                if self.detach:
                    with torch.no_grad():
                        ha, pa = self.tokens_forward(a_ci, a_tok, None, None, tab)
                        ea = self.body(a_sc, ha, pa)
                else:
                    ha, pa = self.tokens_forward(a_ci, a_tok, None, None, tab)
                    ea = self.body(a_sc, ha, pa)
                va = torch.tanh((ea @ self.Wv + self.bv)[:, 0])
                if self.persp_pending:      # 相手の席から符号化した候補は、価値の符号を返して打った席から見る
                    va = torch.where(a_pend > 0, va, -va)
                x = torch.cat([ea, ea - e[b.seg], va[:, None], a_pend[:, None], a_win[:, None]], 1)
                pre = pre + x @ self.Wp1a
            return (torch.relu(pre) @ self.Wp2 + self.bp2)[:, 0]

    class AfterBatches(TT.EpochBatches):
        """`EpochBatches` に、候補行ごとの打った後の盤面の切り出しを足したもの。"""

        def __init__(self, *a, after=None, **kw):
            super().__init__(*a, **kw)
            self.A = after

        def begin(self, order_v, bs_v, order_p, bs_p):
            super().begin(order_v, bs_v, order_p, bs_p)
            pts = np.ascontiguousarray(order_p, np.int64)
            lens = self.plen[pts].astype(np.int64)
            self.idx_p = TT.ragged_index(self.ptr[pts], lens)

        def after_rows(self, idx):
            A = self.A
            idx = np.asarray(idx, np.int64)
            return (torch.from_numpy(np.asarray(A["a_sc"][idx], np.float32)),
                    torch.from_numpy(np.asarray(A["a_ci"][idx], np.int64)),
                    torch.from_numpy(np.asarray(A["a_tok"][idx], np.float32)),
                    torch.from_numpy(np.asarray(A["a_pend"][idx], np.float32)),
                    torch.from_numpy(np.asarray(A["a_win"][idx], np.float32)))

        def policy_after(self, i):
            sc, ci, tok, rom, roo, b = self.policy(i)
            p0 = i * self.bs_p; p1 = p0 + self.bs_p
            c0 = int(self.c_off[p0]); c1 = int(self.c_off[p1])
            return sc, ci, tok, b, self.after_rows(self.idx_p[c0:c1])

        def eval_logits_after(self, tn, pts):
            pts = np.ascontiguousarray(pts, np.int64)
            lens = self.plen[pts].astype(np.int64)
            idx = TT.ragged_index(self.ptr[pts], lens)
            seg_np = np.repeat(np.arange(len(pts), dtype=np.int64), lens)
            bn = np.ascontiguousarray(self.prow[pts]); bi = torch.from_numpy(bn)
            si = self.C["si"][idx]; ti = self.C["ti"][idx]
            b = TT.PolicyBatch(len(pts), torch.from_numpy(seg_np), TT._i64(si), TT._i64(ti),
                               TT._i64(self.C["at"][idx]), TT._i64(self.C["cid"][idx]),
                               TT._i64(self.C["tcid"][idx]), TT._f32(self.tail[idx]),
                               self._rr(len(idx), None, seg_np, si, ti), TT._f32(self.budget[idx]),
                               TT._f32(self.C["pi"][idx]))
            with torch.no_grad():
                return tn.policy_logits_after(self.v_sc.take(bi, bn), self.v_ci.take(bi, bn),
                                              self.v_tok.take(bi, bn), b,
                                              self.after_rows(idx) if tn.after else None).numpy()

    return TorchNRelAfter, AfterBatches


def _seg_argmax(x, seg, n):
    """区間ごとの argmax（全体の行 index・同点は先頭）。"""
    m = np.full(n, -np.inf)
    np.maximum.at(m, seg, x)
    hit = np.where(x >= m[seg])[0]
    out = np.full(n, -1, np.int64)
    out[seg[hit][::-1]] = hit[::-1]
    return out


def eval_points(tn, src, P, C, ptr, pts, bs=256):
    """方策点で: 方策誤差・1 位の一致・本当の読み違いの率と内訳（§7.52 の `n_rel_attn_desk.eval_points` と同じ定義）。"""
    from opcg_sim.learned.train import n_rel_torch as TT
    ce = 0.0; hit = 0; n = 0
    mis = {"all": 0, "atk_atk": 0, "order": 0, "play_play": 0, "other": 0, "unvisited": 0}
    for s in range(0, len(pts), bs):
        bi = pts[s:s + bs]
        lo = src.eval_logits_after(tn, bi).astype(np.float64)
        lens = P["len"][bi].astype(np.int64)
        idx = TT.ragged_index(ptr[bi], lens)
        seg = np.repeat(np.arange(len(bi)), lens)
        mx = np.full(len(bi), -np.inf); np.maximum.at(mx, seg, lo)
        ex = np.exp(lo - mx[seg]); zz = np.zeros(len(bi)); np.add.at(zz, seg, ex)
        logp = lo - mx[seg] - np.log(zz[seg])
        pi = C["pi"][idx].astype(np.float64)
        ce += float(-(pi * logp).sum())
        a_net = _seg_argmax(lo, seg, len(bi))
        a_t = _seg_argmax(pi, seg, len(bi))
        hit += int((a_net == a_t).sum())
        q = C["q"][idx]; nv = C["n"][idx]; at = C["at"][idx]
        for j in np.where(a_net != a_t)[0]:
            i_n, i_t = a_net[j], a_t[j]
            if nv[i_n] <= 0:
                mis["unvisited"] += 1
                continue
            if q[i_t] - q[i_n] < Q_GAP:
                continue
            mis["all"] += 1
            tn_, tt_ = int(at[i_n]), int(at[i_t])
            if tn_ in _AT_ATK and tt_ in _AT_ATK:
                mis["atk_atk"] += 1
            elif {tn_, tt_} <= set(_AT_ATK) | {_AT_PLAY} and (tn_ == _AT_PLAY) != (tt_ == _AT_PLAY):
                mis["order"] += 1
            elif tn_ == _AT_PLAY and tt_ == _AT_PLAY:
                mis["play_play"] += 1
            else:
                mis["other"] += 1
        n += len(bi)
    out = {"ce": ce / max(n, 1), "top1": hit / max(n, 1), "n": n}
    out.update({f"mis_{k}": v / max(n, 1) for k, v in mis.items()})
    return out


def cmd_train(argv):
    import torch
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.n_rel import N_TOK
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--after", required=True, help="build の出力の根")
    ap.add_argument("--head", choices=("base", "after"), required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--val-pts", type=int, default=20000)
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--train-probe", type=int, default=8000)
    ap.add_argument("--x-scale", type=float, default=None, help="Wp1a の初期値の大きさ（既定: r24 から＝0・一から＝1）")
    ap.add_argument("--detach", action="store_true", help="打った後の盤面の胴体へ勾配を流さない")
    ap.add_argument("--persp-pending", action="store_true", help="`build --persp pending` の材料（価値の符号を返す）")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(args.threads or (os.cpu_count() or 1))
    TorchNRelAfter, AfterBatches = _make_classes()

    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    rt = NR.RelTable(NR.profile_table(db, vocab))
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in NR.profile_table(db, vocab)], np.float32)
    dirs = NT._expand(args.src)
    cache = args.cache_dir or DIO.default_cache_dir()
    V, P, C = DIO.load_dump(dirs, vocab, cache_dir=cache, n_tok=N_TOK)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    A = load_after(dirs, args.after, cache)
    assert len(A["a_ok"]) == int(ptr[-1]), (len(A["a_ok"]), int(ptr[-1]))
    # 打った後の盤面が全候補で作れた点だけを使う（base も同じ点＝比べる組は同じ材料）
    okc = np.asarray(A["a_ok"]).astype(np.int64)
    ok_pt = np.add.reduceat(okc, ptr[:-1]) == P["len"]
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    print(f"dirs {len(dirs)} value {len(V['z'])}行（val {int(va_v.sum())}） policy {len(P['len'])}点"
          f"（全候補 ok {int(ok_pt.sum())}・val {int((va_p & ok_pt).sum())}） 候補 {int(ptr[-1])} {time.time()-t0:.0f}s",
          flush=True)
    if args.warm_start:
        net = NT.NRelNet.load(args.warm_start, tables=tables)
    else:
        net = NT.NRelNet(tables, hidden=args.hidden, seed=args.seed)
    net.ablate = {"rel"}                     # 出荷の構成（関係 R なし）
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    tail = NT.cand_tail_all(net, C)
    src = AfterBatches(V, P, C, ptr, C["budget"], tail, rt, net.ablate, after=A)
    xs = args.x_scale if args.x_scale is not None else (0.0 if args.warm_start else 1.0)
    tn = TorchNRelAfter(net, after=(args.head == "after"), x_scale=xs, seed=args.seed, detach=args.detach,
                        persp_pending=args.persp_pending)
    n_par = sum(p.numel() for p in tn.parameters())
    opt = torch.optim.Adam(tn.parameters(), lr=args.lr)
    print(f"head {args.head} params {n_par}（足した {tn.Wp1a.numel() if tn.after else 0}） warm {args.warm_start}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v)[0]; tr_p = np.where(~va_p & ok_pt)[0]
    vi = np.where(va_v)[0][:args.val_rows]
    vp = np.where(va_p & ok_pt)[0][:args.val_pts]
    probe = np.sort(np.random.default_rng(7).choice(tr_p, min(args.train_probe, len(tr_p)), replace=False))
    fo = open(args.out_json, "a")

    def evaluate(ep, train_ce, train_mse, sec):
        tn.eval()
        with torch.no_grad():
            ev = eval_points(tn, src, P, C, ptr, vp)
            pr = eval_points(tn, src, P, C, ptr, probe)
            vm = float(np.mean((src.eval_value(tn, vi) - np.asarray(V["z"][vi], np.float32)) ** 2))
        tn.train()
        row = {"ep": ep, "head": args.head, "warm": bool(args.warm_start), "lr": args.lr, "detach": args.detach,
               "persp_pending": args.persp_pending, "after_root": args.after,
               "x_scale": xs if tn.after else None,
               "wp1a_norm": float(tn.Wp1a.detach().norm()) if tn.after else None,
               "train_ce_run": train_ce, "train_mse_run": train_mse, "train_probe": pr, "val": ev,
               "val_vmse": vm, "n_val_pts": len(vp), "n_train_pts": len(tr_p),
               "sec": round(sec, 1), "wall": round(time.time() - t0, 1)}
        print("AFTER_DESK_EPOCH " + json.dumps(row), flush=True)
        fo.write(json.dumps(row) + "\n"); fo.flush()

    evaluate(-1, None, None, 0.0)
    for ep in range(args.epochs):
        t_ep = time.time()
        rng.shuffle(tr_v); rng.shuffle(tr_p)
        nv = len(tr_v) // args.bs_v
        npi, ord_p = NT._policy_order(tr_p, args.bs_p, 1.0, rng)
        src.begin(tr_v[:nv * args.bs_v], args.bs_v, ord_p, args.bs_p)
        sched = [0] * nv + [1] * npi
        rng.shuffle(sched)
        iv = ip = 0; mse = ce = 0.0
        for st, what in enumerate(sched):
            if st and st % 2000 == 0:
                print(f"  ep{ep} step {st}/{len(sched)} mse {mse/max(iv,1):.4f} ce {ce/max(ip,1):.4f}"
                      f" {time.time()-t0:.0f}s", flush=True)
            if what == 0:
                sc, ci, tok, rom, roo, zt = src.value(iv)
                v = tn.value(sc, ci, tok, None, None)
                l = ((v - zt) ** 2).mean()
                mse += float(l.detach()); loss = 0.5 * l; iv += 1
            else:
                sc, ci, tok, b, Ab = src.policy_after(ip)
                lo = tn.policy_logits_after(sc, ci, tok, b, Ab if tn.after else None)
                logp = TT.seg_log_softmax(lo, b.seg, b.n_pts)
                loss = -(b.pi * logp).sum() / b.n_pts
                ce += float(loss.detach()); ip += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        evaluate(ep, ce / max(npi, 1), mse / max(nv, 1), time.time() - t_ep)
    print("AFTER_DESK_DONE", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("build", "train"):
        print("usage: n_rel_after_desk {build|train} ...", file=sys.stderr)
        return 2
    return (cmd_build if argv[0] == "build" else cmd_train)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
