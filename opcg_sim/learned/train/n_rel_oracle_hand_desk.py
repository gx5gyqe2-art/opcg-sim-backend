"""n_rel_oracle_hand_desk: 相手の今の手札の本当の中身を方策（と価値）に見せると壁が動くかの上限の机上検証
（WP `policy-oracle-hand-desk`・2026-10-25）。

計画 `docs/search_efficiency_plan.md` §7.54・報告 `docs/reports/2026-10-25_policy_oracle_hand_desk.md`。
**1 回きりの測定の道具**（テストは足さない・CLAUDE.md の「理論の道具」と同じ扱い）。既存のネット定義
（`n_rel.py`・`n_rel_torch.py`）も npz も、エンジン・符号化も変えない。比べ方・指標は §7.52 の
`n_rel_attn_desk.py` と同じ定義（`eval_points` をそのまま使う）。局の再生は §7.53 の
`n_rel_after_desk.py`（ブランチ `claude/policy-after-desk`）の作法を写した。

2 つの段:

  build — 記録（`record_gen` の dump）の局を**記録の手の列から**再生し（探索は回さない）、**全ての行**
          （価値の行＝`load_dump` の V と同じ並び）で、その行で決める席から見た**相手の今の手札の本当の中身**の
          要約 `HAND_COLS`（18 列）を書く（`hand_<shard>.npz`）。役割の判定は `n_rel_feat.profile`／`roles_of`。
          各行で `encode` が記録の `scalars`／`tokens` とビット一致することを確かめ、合わなくなった局はそこで打ち切る
          （以降の行は `ok=0`）。あわせて「決める席が手番の席か」（`tp`）を書く＝攻め／守りの分け。
  train — §7.52 と同じ訓練ループ。`--hand none|true|shuffle`:
          - none＝今の入力（足した重みは在るが列は 0＝今の頭と同じ）
          - true＝本当の相手の手札の列
          - shuffle＝対照。列を**別の行の値**に置き換える（全行の並べ替え＝同じ分布・局面と無関係）
          列は訓練行の平均・標準偏差で標準化し、(1) 胴体の第 1 層（価値と方策の両方が通る）に x·W1h、
          (2) 方策の頭の第 1 層に x·Wp1h を足す。足した重みは 0 で始める（r24 から始める組は最初の出力が r24 と同じ）。

実行例:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_oracle_hand_desk build \\
    --in "data/n_records/n54_w0[1-8]_p*" --out data/hand --workers 4
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_oracle_hand_desk train \\
    --in "data/n_records/n54_w0[1-8]_p*" --hand-root data/hand --hand true \\
    --warm-start opcg_sim/data/learned/nrel_r24.npz --epochs 4 --lr 2e-4 --out-json runs/true_r24.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

N_TOK_ = 22
#: 相手の今の手札の要約（決める席から見た相手・本物の中身）。値の単位は列名のとおり（パワー・カウンターは 1000 単位）。
HAND_COLS = (
    "n_hand",              # 枚数（今の符号化にも在る・基準）
    "ctr_sum_k",           # カウンター値の合計（キャラの印字カウンター＋【カウンター】イベントの +パワー）
    "ctr_max_k",           # その最大
    "n_ctr_char",          # カウンター値を持つキャラ
    "n_ctr_event",         # 【カウンター】イベント
    "n_ctr_event_now",     # そのうち相手の今のアクティブなドンで撃てる枚数
    "ctr_event_now_k",     # その +パワーの合計
    "def_now_k",           # 今すぐ足せる守り＝キャラのカウンター合計＋撃てるイベントの合計
    "n_blocker",           # ブロッカー
    "n_removal",           # 除去（役割 removal または v14 の直した除去）
    "n_reduction",         # パワー減算
    "n_lock",              # 封じ（レスト・効果無効など）
    "n_draw",              # ドロー
    "n_big",               # 大型キャラ（役割 big）
    "n_char",              # キャラ
    "n_event",             # イベント
    "max_pow_now_k",       # 相手の今のドン総数で出せるキャラの最大パワー
    "max_pow_next_k",      # 相手の次の番のドン（総数＋1・上限 10）で出せるキャラの最大パワー
)
D_HAND = len(HAND_COLS)


# ---------------------------------------------------------------------------
# build: 記録の再生（`n_rel_after_desk` の作法）と、相手の手札の要約
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


_G = {}


def _card_stats(cid):
    """card_id → 手札の要約に要る量（マスター単位でキャッシュ）。"""
    hit = _G.setdefault("stats", {}).get(cid)
    if hit is not None:
        return hit
    from opcg_sim.learned import n_rel_feat as NR
    m = _G["db"].get_card(cid)
    if m is None:
        out = None
    else:
        p = NR.profile(m)
        r = NR.roles_of(m)
        r2 = NR.roles2_of(m)
        t = getattr(getattr(m, "type", None), "name", "")
        ctr = float(getattr(m, "counter", 0) or 0) if t == "CHARACTER" else 0.0
        cev = float(p["counter_event"]) if t == "EVENT" else 0.0
        out = {"t": t, "cost": int(getattr(m, "cost", 0) or 0), "power": float(getattr(m, "power", 0) or 0),
               "ctr": ctr, "cev": cev, "blocker": bool(p["blocker"]),
               "removal": bool(r[0] > 0 or r2[0] > 0), "red": bool(r[1] > 0), "lock": bool(r[2] > 0),
               "draw": bool(r[3] > 0), "big": bool(r[6] > 0)}
    _G["stats"][cid] = out
    return out


def hand_summary(hand_ids, don_active, don_total):
    """相手の手札（card_id の列）と相手のドン → `HAND_COLS` の 1 行。"""
    x = np.zeros(D_HAND, np.float32)
    x[0] = len(hand_ids)
    don_next = min(10, don_total + 1)
    pw_now = pw_next = 0.0
    for cid in hand_ids:
        s = _card_stats(cid)
        if s is None:
            continue
        cv = s["ctr"] + s["cev"]
        x[1] += cv / 1000.0
        x[2] = max(x[2], cv / 1000.0)
        if s["ctr"] > 0:
            x[3] += 1
            x[7] += s["ctr"] / 1000.0
        if s["cev"] > 0:
            x[4] += 1
            if s["cost"] <= don_active:
                x[5] += 1
                x[6] += s["cev"] / 1000.0
                x[7] += s["cev"] / 1000.0
        x[8] += s["blocker"]
        x[9] += s["removal"]
        x[10] += s["red"]
        x[11] += s["lock"]
        x[12] += s["draw"]
        x[13] += s["big"]
        if s["t"] == "CHARACTER":
            x[14] += 1
            if s["cost"] <= don_total:
                pw_now = max(pw_now, s["power"])
            if s["cost"] <= don_next:
                pw_next = max(pw_next, s["power"])
        elif s["t"] == "EVENT":
            x[15] += 1
    x[16] = pw_now / 1000.0
    x[17] = pw_next / 1000.0
    return x


def _opp_hand(game, name):
    """盤面（`board_json`）→ 決める席 `name` の相手の (手札の card_id 列, アクティブなドン, ドン総数)。"""
    b = json.loads(game.board_json())
    opp = "p2" if name == "p1" else "p1"
    p = b["players"][opp]
    hand = [c.get("card_id") for c in ((p.get("zones") or {}).get("hand") or [])]
    da = len(p.get("don_active") or [])
    dr = len(p.get("don_rested") or [])
    att = 0
    for c in ((p.get("zones") or {}).get("field") or []) + [p.get("leader") or {}]:
        att += int(c.get("attached_don") or 0)
    return hand, da, da + dr + att, (b.get("turn_info") or {}).get("active_player_id")


def build_file(args):
    """1 シャード（`n_record_*.npz`）→ `hand_*.npz`（行ごと: x [n, D_HAND]・ok・tp）。戻り値は数え上げ。"""
    path, out_path = args
    from opcg_sim.loop import decks as D
    from opcg_sim.loop import driver as DR
    from opcg_sim.loop import engine as E
    E.load_net(None)                         # 符号化の語彙（ネット付属の vocab_ids）
    db = _G.setdefault("db", D.load_db())
    d = os.path.dirname(path)
    gm = {int(g["seed"]): g for g in json.load(open(os.path.join(d, "meta_games.json")))["games"]}
    z = dict(np.load(path))
    t0 = time.time()
    n = len(z["seed"])
    ptr = np.concatenate([[0], np.cumsum(z["pol_len"])]).astype(np.int64)
    X = np.zeros((n, D_HAND), np.float32)
    ok = np.zeros(n, np.uint8)
    tp = np.zeros(n, np.uint8)
    st = {"games": 0, "diverged": 0, "rows": n, "rows_ok": 0, "missing_ids": 0}
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
                hand, da, dt, tpn = _opp_hand(game, name)
            except Exception:                                  # noqa: BLE001  局の再生が外れたら打ち切る
                st["diverged"] += 1
                break
            if any(c is None for c in hand):
                st["missing_ids"] += 1
            X[i] = hand_summary([c for c in hand if c is not None], da, dt)
            ok[i] = 1
            tp[i] = 1 if tpn == name else 0
            st["rows_ok"] += 1
            mv = played_move(z, i, ptr, legal)
            try:
                if mv is None:
                    raise ValueError("手が作れない")
                _apply(game, name, mv)
            except Exception:                                  # noqa: BLE001
                st["diverged"] += 1
                break
    tmp = out_path + ".tmp.npz"
    np.savez(tmp, x=X, ok=ok, tp=tp)
    os.replace(tmp, out_path)
    st["sec"] = round(time.time() - t0, 1)
    st["file"] = path
    return st


def hand_path(root, shard):
    part = os.path.basename(os.path.dirname(shard))
    return os.path.join(root, part, os.path.basename(shard).replace("n_record_", "hand_"))


def cmd_build(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_train as NT
    jobs = []
    for d in NT._expand(args.src):
        for f in DIO.shard_files(d):
            op = hand_path(args.out, f)
            if os.path.exists(op):
                continue
            os.makedirs(os.path.dirname(op), exist_ok=True)
            jobs.append((f, op))
    print(f"build {len(jobs)} shards", flush=True)
    os.makedirs(args.out, exist_ok=True)
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
        print(f"[{k+1}/{len(jobs)}] {st['file']} rows {st['rows_ok']}/{st['rows']} div {st['diverged']} "
              f"{st['sec']}s total {time.time()-t0:.0f}s", flush=True)
    print("HAND_BUILD_DONE", flush=True)
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def load_hand(dirs, root):
    """`hand_*.npz` を `load_dump` の V と同じ並び（波 → シャード → 行）で 1 本につなぐ。"""
    from opcg_sim.learned.train import dump_io as DIO
    xs, oks, tps = [], [], []
    for d in dirs:
        for f in DIO.shard_files(d):
            with np.load(hand_path(root, f)) as a:
                xs.append(a["x"]); oks.append(a["ok"]); tps.append(a["tp"])
    return np.concatenate(xs), np.concatenate(oks).astype(bool), np.concatenate(tps).astype(bool)


def _make_class():
    import torch
    from opcg_sim.learned.train import n_rel_torch as TT

    class TorchNRelHand(TT.TorchNRel):
        """`TorchNRel` の胴体の第 1 層と方策の頭の第 1 層に、行ごとの列 x を足したもの（重みは 0 で始める）。

        x は forward の前に `self.x_cur`（[B, D_HAND]）へ置く（値の行・方策点の行のどちらでも同じ）。"""

        def __init__(self, net, d_x, x_scale=0.0, seed=13):
            super().__init__(net)
            torch.manual_seed(seed)
            h1 = self.W1.shape[1]; hp = self.Wp1.shape[1]
            self.W1h = torch.nn.Parameter(torch.randn(d_x, h1) * (x_scale / np.sqrt(d_x)))
            self.Wp1h = torch.nn.Parameter(torch.randn(d_x, hp) * (x_scale / np.sqrt(d_x)))
            self.x_cur = None

        def body(self, sc, h, present):
            if self._sc_masked:
                sc = sc * self.sc_keep
            n = torch.clamp(present.sum(1, keepdim=True), min=1.0)
            mean = h.sum(1) / n
            if self.fast_max:
                mx = h.max(1).values
            else:
                h_masked = torch.where(present[:, :, None] > 0, h, TT.NEG)
                mx = h_masked.max(1).values
                mx = torch.where(mx < -1e8, 0.0, mx)
            zz = torch.cat([sc, mean, mx], 1)
            r1 = torch.relu(zz @ self.W1 + self.b1 + self.x_cur @ self.W1h)
            return torch.relu(r1 @ self.W2 + self.b2)

        def policy_logits(self, sc, ci, tok, rel_om, rel_oo, b):
            tab = self.card_table()
            feats = self.cand_feats(tab, b.at, b.cid, b.tcid, b.tail)
            h, present = self.tokens_forward(ci, tok, rel_om, rel_oo, tab)
            e = self.body(sc, h, present)
            u = self.cand_input(e, h, b.seg, b.si, b.ti, b.ok_s, b.ok_t, b.rr, feats, b.budget)
            pre = u @ self.Wp1 + self.bp1 + (self.x_cur @ self.Wp1h)[b.seg]
            return (torch.relu(pre) @ self.Wp2 + self.bp2)[:, 0]

    return TorchNRelHand


def cmd_train(argv):
    import torch
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.n_rel import N_TOK
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_torch as TT
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train import n_rel_attn_desk as AD
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned.vocab import load_db

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--hand-root", required=True, help="build の出力の根")
    ap.add_argument("--hand", choices=("none", "true", "shuffle"), required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--bs-v", type=int, default=256)
    ap.add_argument("--bs-p", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--val-pts", type=int, default=20000)
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--train-probe", type=int, default=8000)
    ap.add_argument("--x-scale", type=float, default=None, help="足した重みの初期値の大きさ（既定: r24 から＝0・一から＝1）")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(args.threads or (os.cpu_count() or 1))
    TorchNRelHand = _make_class()

    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    rt = NR.RelTable(NR.profile_table(db, vocab))
    ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in NR.profile_table(db, vocab)], np.float32)
    dirs = NT._expand(args.src)
    cache = args.cache_dir or DIO.default_cache_dir()
    V, P, C = DIO.load_dump(dirs, vocab, cache_dir=cache, n_tok=N_TOK)
    ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
    X, ok, tp = load_hand(dirs, args.hand_root)
    assert len(X) == len(V["z"]), (len(X), len(V["z"]))
    va_v = V["seed"] % args.holdout_mod == 0
    va_p = P["seed"] % args.holdout_mod == 0
    ok_pt = ok[P["row"]]
    # 標準化（訓練行の平均・標準偏差）→ 組ごとの列
    mu = X[ok & ~va_v].mean(0); sd = X[ok & ~va_v].std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    Xs[~ok] = 0.0
    if args.hand == "none":
        Xs = np.zeros_like(Xs)
    elif args.hand == "shuffle":
        okr = np.where(ok)[0]
        perm = np.random.default_rng(args.seed + 101).permutation(okr)
        Xs2 = np.zeros_like(Xs); Xs2[okr] = Xs[perm]; Xs = Xs2
    Xt = torch.from_numpy(Xs)
    print(f"dirs {len(dirs)} value {len(V['z'])}行（ok {int(ok.sum())}・val {int(va_v.sum())}） policy {len(P['len'])}点"
          f"（ok {int(ok_pt.sum())}・val {int((va_p & ok_pt).sum())}・守り {int((~tp[P['row']]).sum())}） hand {args.hand}"
          f" {time.time()-t0:.0f}s", flush=True)
    if args.warm_start:
        net = NT.NRelNet.load(args.warm_start, tables=tables)
    else:
        net = NT.NRelNet(tables, hidden=args.hidden, seed=args.seed)
    net.ablate = {"rel"}                     # 出荷の構成（関係 R なし）
    C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
    tail = NT.cand_tail_all(net, C)
    src = TT.EpochBatches(V, P, C, ptr, C["budget"], tail, rt, net.ablate)
    xs = args.x_scale if args.x_scale is not None else (0.0 if args.warm_start else 1.0)
    tn = TorchNRelHand(net, D_HAND, x_scale=xs, seed=args.seed)
    n_par = sum(p.numel() for p in tn.parameters())
    opt = torch.optim.Adam(tn.parameters(), lr=args.lr)
    print(f"hand {args.hand} params {n_par}（足した {tn.W1h.numel() + tn.Wp1h.numel()}） warm {args.warm_start}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    tr_v = np.where(~va_v & ok)[0]; tr_p = np.where(~va_p & ok_pt)[0]
    vi = np.where(va_v & ok)[0][:args.val_rows]
    vp = np.where(va_p & ok_pt)[0][:args.val_pts]
    vp_atk = vp[tp[P["row"][vp]]]; vp_def = vp[~tp[P["row"][vp]]]
    vi_atk = vi[tp[vi]]; vi_def = vi[~tp[vi]]
    probe = np.sort(np.random.default_rng(7).choice(tr_p, min(args.train_probe, len(tr_p)), replace=False))
    fo = open(args.out_json, "a")

    class _Src:
        """`eval_points`／`eval_value` に渡す口（forward の前に x を置く）。"""

        def eval_logits(self, tn_, bi):
            tn_.x_cur = Xt[torch.from_numpy(np.ascontiguousarray(P["row"][bi], np.int64))]
            return src.eval_logits(tn_, bi)

    def eval_value(rows, bs=512):
        out = []
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            tn.x_cur = Xt[torch.from_numpy(np.ascontiguousarray(r, np.int64))]
            out.append(src.eval_value(tn, r, bs=bs))
        vv = np.concatenate(out) if out else np.zeros(0, np.float32)
        return float(np.mean((vv - np.asarray(V["z"][rows], np.float32)) ** 2)) if len(rows) else None

    def evaluate(ep, train_ce, train_mse, sec):
        tn.eval()
        with torch.no_grad():
            ev = AD.eval_points(tn, _Src(), P, C, ptr, vp)
            ev_atk = AD.eval_points(tn, _Src(), P, C, ptr, vp_atk)
            ev_def = AD.eval_points(tn, _Src(), P, C, ptr, vp_def) if len(vp_def) else None
            pr = AD.eval_points(tn, _Src(), P, C, ptr, probe)
            vm = eval_value(vi); vm_atk = eval_value(vi_atk); vm_def = eval_value(vi_def)
        tn.train()
        row = {"ep": ep, "hand": args.hand, "warm": bool(args.warm_start), "lr": args.lr, "x_scale": xs,
               "w1h_norm": float(tn.W1h.detach().norm()), "wp1h_norm": float(tn.Wp1h.detach().norm()),
               "train_ce_run": train_ce, "train_mse_run": train_mse, "train_probe": pr, "val": ev,
               "val_atk": ev_atk, "val_def": ev_def, "val_vmse": vm, "val_vmse_atk": vm_atk, "val_vmse_def": vm_def,
               "n_val_pts": len(vp), "n_val_def": len(vp_def), "n_val_rows": len(vi), "n_val_rows_def": len(vi_def),
               "n_train_pts": len(tr_p), "sec": round(sec, 1), "wall": round(time.time() - t0, 1)}
        print("HAND_DESK_EPOCH " + json.dumps(row), flush=True)
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
                tn.x_cur = Xt[src.ord_v_t[iv * src.bs_v:(iv + 1) * src.bs_v]]
                sc, ci, tok, rom, roo, zt = src.value(iv)[:6]
                v = tn.value(sc, ci, tok, rom, roo)
                l = ((v - zt) ** 2).mean()
                mse += float(l.detach()); loss = 0.5 * l; iv += 1
            else:
                tn.x_cur = Xt[src.rows_p_t[ip * src.bs_p:(ip + 1) * src.bs_p]]
                sc, ci, tok, rom, roo, b = src.policy(ip)
                lo = tn.policy_logits(sc, ci, tok, rom, roo, b)
                logp = TT.seg_log_softmax(lo, b.seg, b.n_pts)
                loss = -(b.pi * logp).sum() / b.n_pts
                ce += float(loss.detach()); ip += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        evaluate(ep, ce / max(npi, 1), mse / max(nv, 1), time.time() - t_ep)
    print("HAND_DESK_DONE", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("build", "train"):
        print("usage: n_rel_oracle_hand_desk {build|train} ...", file=sys.stderr)
        return 2
    return (cmd_build if argv[0] == "build" else cmd_train)(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
