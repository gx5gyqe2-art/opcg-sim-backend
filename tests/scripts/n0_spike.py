"""N0 スパイク＝N系（0から設計・ユーザ発案 2026-08-25）の最小実証。

G17 の実証（`docs/reports/2026-08-25_g17_baseline.md`）を受けた設計の芯の検証:
  柱1 カード物理要約: leader_feat の効果木要約を**全カード**に適用（基礎統計＋12次元物理）
  柱2 集合符号化: 盤面のカード群をゾーンタグ付き集合として共有MLP＋mean/maxプールで読む
  柱3 文脈条件付き出口採点: leaf / battle出口 / turn出口 を one-hot 文脈で単一胴体が採点
  柱4 教師の本体統合: 勝敗 z（G17コーパス）＋支配対（defcf/D族=battle・plandom=turn）を
       後付けヘッドでなく**最初から多課題損失**で同時学習

データは既存 npz（scalars94/card_idx24）から導出する＝盤面再符号化不要。card_idx の
ゾーンは位置固定（[自L,相手L,自場5,相手場5,自手札10,pad2]・encoder.py §card_idx）。

サブコマンド:
  train  … 学習して /home/user/n0_net.npz に保存・val 指標を印字
  probe  … m1@14 の枝別出口を N0 で採点（枝順位ゲートの N0 版）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

MAX_CI = 24
CARD_STAT = 8       # cost, power, counter, is_leader, is_char, is_event, is_stage, blocker
PHYS = 12           # leader_feat.DIMS
ZONE = 5            # own_leader, opp_leader, own_field, opp_field, own_hand
D_IN = CARD_STAT + PHYS + ZONE + 1   # +1 present flag
D_CARD = 24
CTX = 3             # 0=leaf, 1=battle出口, 2=turn出口
_SLOT_ZONE = [0, 1] + [2] * 5 + [3] * 5 + [4] * 10 + [2] * 2   # pad2はダミー(present=0)


def build_card_table():
    """vocab index → 物理特徴ベクトル（D_IN-ZONE-1 次元・PAD/UNK=0行）。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned.leader_feat import leader_static_vector
    eng = LearnedEngine()
    vocab = eng.vocab                     # card_id -> idx（生成時と同一＝gen15 vocab）
    from cpu_selfplay import _load_db
    db = _load_db()
    n = max(vocab.values()) + 1
    tab = np.zeros((n, CARD_STAT + PHYS), np.float32)
    for cid, idx in vocab.items():
        c = db.get_card(cid)
        if c is None:
            continue
        t = getattr(getattr(c, "type", None), "name", "")
        text = getattr(c, "effect_text", "") or ""
        stats = [float(getattr(c, "cost", 0) or 0) / 5.0,
                 float(getattr(c, "power", 0) or 0) / 5000.0,
                 float(getattr(c, "counter", 0) or 0) / 2000.0,
                 1.0 if t == "LEADER" else 0.0,
                 1.0 if t == "CHARACTER" else 0.0,
                 1.0 if t == "EVENT" else 0.0,
                 1.0 if t == "STAGE" else 0.0,
                 1.0 if "ブロッカー" in text else 0.0]
        tab[idx, :CARD_STAT] = stats
        tab[idx, CARD_STAT:] = leader_static_vector(c)
    return tab, vocab


def card_channel(card_idx, tab):
    """card_idx[B,24] → 集合入力 [B,24,D_IN]（物理特徴＋ゾーンonehot＋present）。"""
    B = card_idx.shape[0]
    x = np.zeros((B, MAX_CI, D_IN), np.float32)
    feats = tab[np.clip(card_idx, 0, len(tab) - 1)]
    x[:, :, :CARD_STAT + PHYS] = feats
    for s in range(MAX_CI):
        x[:, s, CARD_STAT + PHYS + _SLOT_ZONE[s]] = 1.0
    x[:, :, -1] = (card_idx > 0).astype(np.float32)
    x[card_idx <= 0] = 0.0                # PAD/UNK 行は完全ゼロ（present=0）
    return x


class N0Net:
    def __init__(self, d_scalar=94, hidden=96, seed=7):
        r = np.random.default_rng(seed)
        def W(a, b):
            return (r.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
        self.Wc = W(D_IN, D_CARD); self.bc = np.zeros(D_CARD, np.float32)
        d_tr = d_scalar + 2 * D_CARD + CTX
        self.W1 = W(d_tr, hidden); self.b1 = np.zeros(hidden, np.float32)
        self.W2 = W(hidden, 64);   self.b2 = np.zeros(64, np.float32)
        self.W3 = W(64, 1);        self.b3 = np.zeros(1, np.float32)
        self.params = ["Wc", "bc", "W1", "b1", "W2", "b2", "W3", "b3"]
        self._adam = {p: [np.zeros_like(getattr(self, p)), np.zeros_like(getattr(self, p))]
                      for p in self.params}
        self._t = 0

    def forward(self, sc, cards, ctx, keep=None):
        """sc[B,94] cards[B,24,D_IN] ctx[B,3] → v[B]・keep に中間を格納（backward 用）。"""
        h_c = cards @ self.Wc + self.bc                 # [B,24,D_CARD]
        r_c = np.maximum(h_c, 0.0)
        present = cards[:, :, -1:]                       # [B,24,1]
        n = np.maximum(present.sum(1), 1.0)              # [B,1]
        mean = (r_c * present).sum(1) / n                # [B,D_CARD]
        masked = np.where(present > 0, r_c, -1e9)
        mx = masked.max(1)                               # [B,D_CARD]
        mx = np.where(mx < -1e8, 0.0, mx)
        z = np.concatenate([sc, mean, mx, ctx], 1)       # [B,d_tr]
        h1 = z @ self.W1 + self.b1; r1 = np.maximum(h1, 0.0)
        h2 = r1 @ self.W2 + self.b2; r2 = np.maximum(h2, 0.0)
        o = (r2 @ self.W3 + self.b3)[:, 0]
        v = np.tanh(o)
        if keep is not None:
            keep.update(cards=cards, r_c=r_c, present=present, n=n, masked=masked,
                        mx=mx, z=z, r1=r1, r2=r2, o=o, v=v)
        return v

    def backward(self, k, dv):
        """dL/dv[B] → 勾配 dict（forward の keep を使用）。"""
        g = {}
        do = dv * (1.0 - k["v"] ** 2)                    # tanh'
        g["W3"] = k["r2"].T @ do[:, None]; g["b3"] = np.array([do.sum()], np.float32)
        dr2 = do[:, None] @ self.W3.T
        dh2 = dr2 * (k["r2"] > 0)
        g["W2"] = k["r1"].T @ dh2; g["b2"] = dh2.sum(0)
        dr1 = dh2 @ self.W2.T
        dh1 = dr1 * (k["r1"] > 0)
        g["W1"] = k["z"].T @ dh1; g["b1"] = dh1.sum(0)
        dz = dh1 @ self.W1.T
        d_sc = 94
        dmean = dz[:, d_sc:d_sc + D_CARD]
        dmx = dz[:, d_sc + D_CARD:d_sc + 2 * D_CARD]
        # mean 経路
        dr_c = k["present"] / k["n"][:, None]             # [B,24,1]（n は [B,1]）
        dr_c = dr_c * dmean[:, None, :]                   # [B,24,D_CARD]
        # max 経路（argmax の位置にだけ流す）
        am = k["masked"].argmax(1)                        # [B,D_CARD]
        B = dmx.shape[0]
        dmax = np.zeros_like(k["r_c"])
        bi = np.repeat(np.arange(B), D_CARD)
        ci = np.tile(np.arange(D_CARD), B)
        dmax[bi, am.reshape(-1), ci] = dmx.reshape(-1)
        dmax *= (k["present"] > 0)
        dr_c = dr_c + dmax
        dh_c = dr_c * (k["r_c"] > 0)
        g["Wc"] = np.einsum("bsi,bsj->ij", k["cards"], dh_c).astype(np.float32)
        g["bc"] = dh_c.sum((0, 1))
        return g

    def step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for p in self.params:
            gp = grads.get(p)
            if gp is None:
                continue
            m, v = self._adam[p]
            m[:] = b1 * m + (1 - b1) * gp
            v[:] = b2 * v + (1 - b2) * gp * gp
            mh = m / (1 - b1 ** self._t)
            vh = v / (1 - b2 ** self._t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + eps))

    def save(self, path, meta=None):
        np.savez_compressed(path, **{p: getattr(self, p) for p in self.params},
                            meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        net = cls()
        for p in net.params:
            setattr(net, p, d[p])
        return net


def _load_rows(dirs, tab):
    sc, ci, y = [], [], []
    for dd in dirs:
        for f in sorted(glob.glob(os.path.join(dd, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            s = d["scalars"].astype(np.float32)
            if s.shape[1] != 94:
                continue
            n = len(s)
            sc.append(s)
            c = d["card_idx"] if "card_idx" in d else np.zeros((n, MAX_CI), np.int64)
            if c.shape[1] < MAX_CI:
                c = np.concatenate([c, np.zeros((n, MAX_CI - c.shape[1]), np.int64)], 1)
            ci.append(c[:, :MAX_CI])
            y.append(d["value"].astype(np.float32))
    return np.concatenate(sc), np.concatenate(ci), np.concatenate(y)


def _load_pairs(dirs, tab, holdout_mod=7):
    """群内 (良, 悪) 対の (scalars, card_idx) を train/val に分ける。"""
    tr, va = [], []
    for dd in dirs:
        for f in sorted(glob.glob(os.path.join(dd, "*.npz"))):
            d = np.load(f, allow_pickle=True)
            s = d["scalars"].astype(np.float32)
            if s.shape[1] != 94 or "group" not in d:
                continue
            c = d["card_idx"][:, :MAX_CI]
            g = d["group"]
            y = d["value"]
            for gid in np.unique(g):
                m = g == gid
                good = np.where(m & (y > 0.1))[0]
                bad = np.where(m & (y < -0.1))[0]
                dst = va if (int(gid) % holdout_mod == 0) else tr
                for a in good:
                    for b in bad:
                        dst.append((s[a], c[a], s[b], c[b]))
    return tr, va


def _pair_batch(pairs, idxs, tab, ctx_id):
    sa = np.stack([pairs[i][0] for i in idxs]); ca = np.stack([pairs[i][1] for i in idxs])
    sb = np.stack([pairs[i][2] for i in idxs]); cb = np.stack([pairs[i][3] for i in idxs])
    ctx = np.zeros((len(idxs), CTX), np.float32); ctx[:, ctx_id] = 1.0
    return (sa, card_channel(ca, tab), ctx), (sb, card_channel(cb, tab), ctx)


def pair_acc(net, pairs, tab, ctx_id, cap=2000):
    if not pairs:
        return float("nan")
    idxs = list(range(min(len(pairs), cap)))
    (sa, ca, ctx), (sb, cb, _) = _pair_batch(pairs, idxs, tab, ctx_id)
    va = net.forward(sa, ca, ctx); vb = net.forward(sb, cb, ctx)
    return float((va > vb).mean())


def train(args):
    tab, vocab = build_card_table()
    z_dirs = [d for d in args.z_dirs.split(",") if d]
    sc, ci, y = _load_rows(z_dirs, tab)
    rng = np.random.default_rng(11)
    val_mask = (np.arange(len(y)) % 10) == 0
    tr_i = np.where(~val_mask)[0]; va_i = np.where(val_mask)[0]
    bt_tr, bt_va = _load_pairs([d for d in args.battle_pair_dirs.split(",") if d], tab)
    tn_tr, tn_va = _load_pairs([d for d in args.turn_pair_dirs.split(",") if d], tab)
    print(f"z行 {len(y)}（train {len(tr_i)}/val {len(va_i)}）・battle対 {len(bt_tr)}+{len(bt_va)}"
          f"・turn対 {len(tn_tr)}+{len(tn_va)}・カード表 {tab.shape}", flush=True)
    net = N0Net()
    B = 256
    margin = 0.2
    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr_i)
        losses = []
        for s0 in range(0, len(tr_i), B):
            bi = tr_i[s0:s0 + B]
            ctx = np.zeros((len(bi), CTX), np.float32); ctx[:, 0] = 1.0
            k = {}
            v = net.forward(sc[bi], card_channel(ci[bi], tab), ctx, keep=k)
            dv = 2.0 * (v - y[bi]) / len(bi)
            g = net.backward(k, dv)
            # 対の勾配を同ステップに合流（多課題）: battle と turn を交互に1ミニバッチ
            for pairs, cid in ((bt_tr, 1), (tn_tr, 2)):
                if not pairs:
                    continue
                pidx = rng.integers(0, len(pairs), size=min(64, len(pairs)))
                (sa, ca, ctxp), (sb, cb, _) = _pair_batch(pairs, pidx, tab, cid)
                ka, kb = {}, {}
                va = net.forward(sa, ca, ctxp, keep=ka)
                vb = net.forward(sb, cb, ctxp, keep=kb)
                viol = (margin - (va - vb)) > 0
                w = args.pair_w / max(viol.sum(), 1)
                ga = net.backward(ka, np.where(viol, -w, 0.0).astype(np.float32))
                gb = net.backward(kb, np.where(viol, +w, 0.0).astype(np.float32))
                for p in net.params:
                    g[p] = g[p] + ga[p] + gb[p]
            net.step(g, lr=args.lr)
            losses.append(float(((v - y[bi]) ** 2).mean()))
        ctxv = np.zeros((len(va_i), CTX), np.float32); ctxv[:, 0] = 1.0
        vv = net.forward(sc[va_i], card_channel(ci[va_i], tab), ctxv)
        vmse = float(((vv - y[va_i]) ** 2).mean())
        print(f"  ep{ep}: train {np.mean(losses):.4f} / val {vmse:.4f}"
              f" / battle対val {pair_acc(net, bt_va, tab, 1):.3f}"
              f" / turn対val {pair_acc(net, tn_va, tab, 2):.3f}", flush=True)
    net.save(args.out, meta={"val_mse": vmse,
                             "battle_pair_val": pair_acc(net, bt_va, tab, 1),
                             "battle_pair_train": pair_acc(net, bt_tr, tab, 1),
                             "turn_pair_val": pair_acc(net, tn_va, tab, 2)})
    print("N0_TRAIN_DONE", json.dumps({
        "val_mse": round(vmse, 4),
        "battle_pair_val": round(pair_acc(net, bt_va, tab, 1), 4),
        "turn_pair_val": round(pair_acc(net, tn_va, tab, 2), 4),
        "out": args.out}))


def probe(args):
    """m1@14 の枝別出口を N0 で採点（枝順位の N0 版・素通しが1位であるべき）。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from opcg_sim.src.core import cpu_ai
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned import encoder as ENC
    from opcg_sim.src.learned.mcts import resolve_battle_inplace
    tab, vocab = build_card_table()
    net = N0Net.load(args.net)
    CR.ARGS = argparse.Namespace(true_board=True); CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m1"]); rec = raw.get("replay", raw)
    CR.GAMES["m1"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m1", 14)
    name = who if isinstance(who, str) else who.name
    eng = LearnedEngine(macro_moves=False, defense_box=False)   # 原始手空間で枝を出す
    legal = eng.game.legal_actions(m)
    print("=== m1@14 N0(battle出口文脈) ===")
    scored = []
    for mv in legal:
        c = m.clone(); c.action_events = []
        try:
            cpu_ai._apply_move_inplace(c, name, mv, stop_at_select=True)
            resolve_battle_inplace(eng.game, c)
            enc = ENC.encode(c, name, eng.vocab, version=12)
            sc = np.asarray(enc["scalars"], np.float32)[None]
            cix = np.zeros((1, MAX_CI), np.int64)
            src = np.asarray(enc["card_idx"])[:MAX_CI]
            cix[0, :len(src)] = src
            ctx = np.zeros((1, CTX), np.float32); ctx[0, 1] = 1.0
            v = float(net.forward(sc, card_channel(cix, tab), ctx)[0])
        except Exception as e:
            v = None
        d = cpu_ai._describe_move(m, mv) or {}
        scored.append((v, f"{d.get('action_type')}:{d.get('card')}"))
    for v, s in sorted(scored, key=lambda t: -(t[0] if t[0] is not None else -9)):
        print(f"  {v:+.4f}  {s}" if v is not None else f"  None  {s}")


class N0ValueAdapter:
    """N0Net を ValueNet のダックタイプ（predict/predict_exit/has_exit_head）に適合させ、
    LearnedEngine の葉評価・戦闘出口・ターン出口の物差しを**単一胴体の文脈切替**で供給する
    （N1＝エンジン統合・2026-08-25）。aux は持たない（本体の aux 粘り項も純正AZ化で削除済み）。"""

    def __init__(self, net, tab):
        self.net = net
        self.tab = tab

    def _fwd(self, batch, ctx_id):
        sc = np.asarray(batch["scalars"], np.float32)
        ci = np.asarray(batch["card_idx"])
        if ci.shape[1] < MAX_CI:
            ci = np.concatenate([ci, np.zeros((len(ci), MAX_CI - ci.shape[1]), ci.dtype)], 1)
        ctx = np.zeros((len(sc), CTX), np.float32)
        ctx[:, ctx_id] = 1.0
        return self.net.forward(sc, card_channel(ci[:, :MAX_CI], self.tab), ctx)

    def predict(self, batch):
        return self._fwd(batch, 0)

    def predict_with_aux(self, batch):
        v = self._fwd(batch, 0)
        return v, np.zeros(len(v), np.float32)

    def has_exit_head(self, kind):
        return kind in ("battle", "turn")

    def predict_exit(self, batch, kind):
        return self._fwd(batch, 1 if kind == "battle" else 2)


def n0_engine(net_path):
    """N0 を積んだ LearnedEngine（符号化 v12 は共通・policy は現行 gen15 のまま）。"""
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    tab, _vocab = build_card_table()
    eng = LearnedEngine()
    eng.vnet = N0ValueAdapter(N0Net.load(net_path), tab)
    return eng


def gate(args):
    """coach 13点（既定 vs N0エンジン）＝N1 の行動ゲート。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import mark_gate as MG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    base = LearnedEngine()
    chall = n0_engine(args.net)
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    rows = []
    for tag, i, accept in CG.VERIFIED_V2:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            rec, fbi, actions = CR.GAMES[tag]
            built = MG._restore(db, rec, fbi, actions, i)
            if isinstance(built, str) or built is None:
                print(f"{tag}@{i}: 復元不可"); continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        b = CG.decide_rate(base, m0, actor, accept, args.seeds, 160)
        c = CG.decide_rate(chall, m0, actor, accept, args.seeds, 160)
        rows.append((tag, i, b, c))
        print(f"  {tag}@{i:<4} base={b:.2f} n0={c:.2f}", flush=True)
    ok_nr, ok_imp, regs = CG.judge(rows)
    print(f"改善: {'OK' if ok_imp else 'NG'}（n0計 {sum(c for *_, c in rows):.1f}"
          f" vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {regs}")
    print("N0_GATE_RESULT", json.dumps({"verdict": "PASS" if (ok_nr and ok_imp) else "FAIL"}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--z-dirs", required=True)
    t.add_argument("--battle-pair-dirs", default="")
    t.add_argument("--turn-pair-dirs", default="")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--pair-w", type=float, default=1.0)
    t.add_argument("--out", default="/home/user/n0_net.npz")
    p = sub.add_parser("probe")
    p.add_argument("--net", default="/home/user/n0_net.npz")
    g = sub.add_parser("gate")
    g.add_argument("--net", default="/home/user/n0_net.npz")
    g.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "gate":
        gate(args)
    else:
        probe(args)


if __name__ == "__main__":
    main()
