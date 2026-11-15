#!/usr/bin/env python3
"""value_calib: ネットの価値の系統的な見誤りの地図（WP `ceiling-map` 段 C・`docs/search_efficiency_plan.md` §7.92）。

**1 回きりの計測**（テストは足さない）。エンジン・ネット・既定の挙動には触らない。

2 段に分かれる:

    # 段 1: 波の記録（結果ブランチの `n_records/`）1 本ぶんを読み、ネットを前向きに通して「行ごとの表」を npz に書く
    #       （作業用・リポジトリの外に置く。npz は結果ブランチに置かない）。`--root` は波を打ったコミットの checkout
    python tests/scripts/value_calib.py extract --root ~/wt_wave --in ~/waves/n83f_w01/n_records --tag n83f_w01 --out ~/vc_rows

    # 段 2: 表を全部読み、切り口ごとに集計して要約と jsonl を書く
    python tests/scripts/value_calib.py report --root ~/wt_wave --rows ~/vc_rows --out n1_results/value_calib

**測るもの**

- 価値: 行ごとに「ネットの価値の予測（その行の席の視点・−1〜+1）− 局の結末（勝ち +1／負け −1）」。
  切り口ごとに平均（偏り）・二乗誤差・「その切り口の結末の平均を言い当てるだけの定数予測」に対する二乗誤差の比。
  あわせて main の行では探索の根の評価（訪問で重み付けた行動価値）の偏りも同じ切り口で出す。
- 打ち方: main の行（候補 2 つ以上）で、訪問の最多手とネットの方策の最上位が食い違う割合。

**区間は局を単位に出す**（局の中の行は独立でない）。切り口の中の行を局ごとにまとめ、
「局ごとの誤差の和 ÷ 局ごとの行数の和」の比の推定として、局をまたぐ散らばりから 95% 区間を作る。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# トークンの並び（`n_rel_feat`）: 自L, 相L, 自場5, 相場5, 手札10
OWN_F = slice(2, 7)
OPP_F = slice(7, 12)
HAND = slice(12, 22)


# ---------------------------------------------------------------------------
# 段 1: 行ごとの表
# ---------------------------------------------------------------------------
def _use_root(root):
    """`opcg_sim` を読む木を選ぶ（**波を打ったコミットの木**で読むこと——カード DB が替わると語彙の番号がずれ、
    記録の `card_idx` を今の木の表で引くと別のカードになる。2026-11-09 の DB 差し替えで実際にずれた）。"""
    root = os.path.abspath(os.path.expanduser(root))
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return root


def extract(args):
    root = _use_root(args.root)
    if not args.net:
        args.net = os.path.join(root, "opcg_sim", "data", "learned", "nrel_r29f4.npz")
    from opcg_sim.learned.train import dump_io as DIO
    from opcg_sim.learned.train import n_rel_train as NT
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    from opcg_sim.learned import n_rel as NL
    from opcg_sim.learned import n_rel_feat as NR
    from opcg_sim.learned.vocab import load_db

    t0 = time.time()
    out_f = os.path.join(os.path.expanduser(args.out), f"{args.tag}.npz")
    if os.path.exists(out_f):
        print("skip (exists)", out_f)
        return 0
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(os.path.expanduser(args.src), "*")) if os.path.isdir(d))
    db = load_db()
    stats, ab, abm, pwr, isl, vocab = build_eff_tables()
    tables = (stats, ab, abm, pwr, isl)
    inv = {i: c for c, i in vocab.items()}
    V, P, C = DIO.load_dump(dirs, vocab, with_policy=True, cache_dir=os.path.expanduser(args.cache),
                            n_tok=NL.N_TOK)
    n = len(V["z"])
    ptab = NR.profile_table(db, vocab)
    rt = NR.RelTable(ptab)
    net = NT.NRelNet.load(args.net, tables=tables)

    # ---- 価値の前向き（全行）
    v = np.empty(n, np.float32)
    for s in range(0, n, args.bs):
        bi = np.arange(s, min(n, s + args.bs))
        sc, ci, tok = DIO.rows_f32(V, bi)
        rel_om, rel_oo = NT.relations_or_zeros(net, ci, tok, rt)
        v[s:s + len(bi)] = net.value(sc[:, :NL.D_SC], ci, tok, rel_om, rel_oo)
    print(f"value {n} rows {time.time()-t0:.0f}s", flush=True)

    # ---- 方策の前向き（main・候補 2 つ以上）
    pol_dis = np.full(n, -1, np.int8)          # 訪問の最多手 ≠ 方策の最上位
    pol_pvis = np.full(n, np.nan, np.float32)  # 訪問の最多手にネットの方策が置く確率
    pol_ncand = np.zeros(n, np.int16)
    pol_vtop = np.full(n, np.nan, np.float32)  # 訪問の最多手の訪問の割合
    pol_qgap = np.full(n, np.nan, np.float32)  # 訪問の最多手の行動価値 − 方策の最上位の行動価値
    if len(P["len"]):
        ptr = np.concatenate([[0], np.cumsum(P["len"])]).astype(np.int64)
        ptab_ret = np.array([(p["ret_don"] if p else 0.0) for p in ptab], np.float32)
        C["budget"] = NT.budget_feats_all(V, P, C, ptr, ptab_ret)
        tab = net.card_table()
        npt = len(P["len"])
        for s in range(0, npt, args.bs):
            bi = np.arange(s, min(npt, s + args.bs))
            lens = P["len"][bi]
            idx = np.arange(ptr[bi[0]], ptr[bi[-1] + 1])
            seg = np.repeat(np.arange(len(bi)), lens)
            sc, ci, tok = NT.prow(V, P, bi)
            rel_om, rel_oo = NT.relations_or_zeros(net, ci, tok, rt)
            lo = net.policy_logits(sc[:, :NL.D_SC], ci, tok, rel_om, rel_oo, seg,
                                   C["si"][idx].astype(np.int64), C["ti"][idx].astype(np.int64),
                                   net.cand_feats(C, idx, tab), C["budget"][idx], tab=tab)
            p = net.seg_softmax(lo, seg, len(bi))
            pos = 0
            for j, L in enumerate(lens):
                sl = slice(pos, pos + L)
                nn = C["n"][idx][sl]
                a_v = int(np.argmax(nn)); a_p = int(np.argmax(p[sl]))
                r = int(P["row"][bi[j]])
                pol_dis[r] = int(a_v != a_p)
                pol_pvis[r] = p[sl][a_v]
                pol_ncand[r] = L
                pol_vtop[r] = nn[a_v] / max(1e-9, float(nn.sum()))
                q = C["q"][idx][sl]
                pol_qgap[r] = q[a_v] - q[a_p]
                pos += L
    print(f"policy {len(P['len'])} points {time.time()-t0:.0f}s", flush=True)

    # ---- 行ごとの列（npz を pack と同じ順で読む）
    cols = {k: DIO.load_row_col(dirs, k) for k in ("who", "kind", "step", "explore", "forced", "pol_v0", "pol_len")}
    sc_all = np.asarray(V["sc"][np.arange(n)], np.float16)
    tok = np.asarray(V["tok"][np.arange(n)], np.float32)
    ci = np.asarray(V["ci"][np.arange(n)], np.int64)
    unit_own = ci[:, OWN_F] > 0
    unit_opp = ci[:, OPP_F] > 0

    def fsum(col, sl, m):
        return (tok[:, sl, col] * m).sum(1).astype(np.float32)

    def fmax(col, sl, m):
        return np.where(m, tok[:, sl, col], 0).max(1).astype(np.float32)
    S = {c: i for i, c in enumerate(NR.S_COLS)}
    feats = {
        "own_fmax": fmax(S["power_now"], OWN_F, unit_own), "opp_fmax": fmax(S["power_now"], OPP_F, unit_opp),
        "own_fsum": fsum(S["power_now"], OWN_F, unit_own), "opp_fsum": fsum(S["power_now"], OPP_F, unit_opp),
        "own_blk": fsum(S["is_blocker_active"], OWN_F, unit_own), "opp_blk": fsum(S["is_blocker_active"], OPP_F, unit_opp),
        "own_rest": fsum(S["is_rest"], OWN_F, unit_own), "opp_rest": fsum(S["is_rest"], OPP_F, unit_opp),
        "own_fdon": fsum(S["attached_don"], OWN_F, unit_own), "opp_fdon": fsum(S["attached_don"], OPP_F, unit_opp),
        "own_canatk": (tok[:, 0, S["can_attack_now"]] + fsum(S["can_attack_now"], OWN_F, unit_own)).astype(np.float32),
        "hand_playable": fsum(S["playable_now"], HAND, ci[:, HAND] > 0),
        "hand_counter": fsum(S["counter_value"], HAND, ci[:, HAND] > 0),
        "own_trig_ko": fsum(S["trig_ko"], OWN_F, unit_own), "opp_trig_ko": fsum(S["trig_ko"], OPP_F, unit_opp),
        "own_trig_oppatk": fsum(S["trig_opp_attack"], OWN_F, unit_own) + tok[:, 0, S["trig_opp_attack"]],
        "opp_trig_oppatk": fsum(S["trig_opp_attack"], OPP_F, unit_opp) + tok[:, 1, S["trig_opp_attack"]],
        "own_act": fsum(S["act_avail"], OWN_F, unit_own) + tok[:, 0, S["act_avail"]],
        "opp_threat": fsum(S["threat_next"], OWN_F, unit_own),
    }
    lead = np.array([[inv.get(int(a), "?"), inv.get(int(b), "?")] for a, b in ci[:, :2]])
    np.savez_compressed(
        out_f, v=v, z=np.asarray(V["z"][np.arange(n)], np.float32), seed=V["seed"], turn=V["turn"],
        sc=sc_all, lead=lead, pol_dis=pol_dis, pol_pvis=pol_pvis, pol_ncand=pol_ncand, pol_vtop=pol_vtop,
        pol_qgap=pol_qgap, **{k: a for k, a in cols.items() if a is not None}, **feats)
    print(f"wrote {out_f} rows {n} {time.time()-t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 段 2: 切り口ごとの集計
# ---------------------------------------------------------------------------
COLOR_JA = {"RED": "赤", "GREEN": "緑", "BLUE": "青", "PURPLE": "紫", "BLACK": "黒", "YELLOW": "黄"}
#: 掛け合わせの切り口に入れる最低の局数・行数（これ未満のマスは集計しない）
MIN_GAMES = 200
MIN_ROWS = 2000


def _band(x, edges, labels):
    """`edges` で切った帯の番号とラベル（`x < edges[0]` が 0 番）。"""
    return np.digitize(x, edges).astype(np.int32), list(labels)


def load_rows(rows_dir):
    fs = sorted(glob.glob(os.path.join(os.path.expanduser(rows_dir), "*.npz")))
    parts = [np.load(f) for f in fs]
    keys = parts[0].files
    D = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    D["wave"] = np.concatenate([np.full(len(p["v"]), 0 if "n83f" in os.path.basename(f) else 1, np.int8)
                                for f, p in zip(fs, parts)])
    return D, len(fs)


def build_cuts(D, db, decks_mod):
    """行ごとの切り口（名前 → (番号の列, ラベル, 説明)）。番号 −1 はその切り口に入らない行。"""
    sc = D["sc"]
    n = len(D["v"])
    f32 = lambda j: sc[:, j].astype(np.float32)                                   # noqa: E731
    life_me, life_opp = np.rint(f32(0)).astype(int), np.rint(f32(1)).astype(int)
    hand_me, hand_opp = np.rint(f32(6)).astype(int), np.rint(f32(7)).astype(int)
    f_me, f_opp = np.rint(f32(8)).astype(int), np.rint(f32(9)).astype(int)
    turn = D["turn"].astype(int)
    my_turn = f32(11) > 0.5
    deck_me, deck_opp = np.rint(f32(16) * 50).astype(int), np.rint(f32(17) * 50).astype(int)
    don_me = np.rint(f32(2) + f32(3) + f32(14) * 5 + D["own_fdon"] * 5).astype(int)
    don_opp = np.rint(f32(4) + f32(5) + f32(15) * 5 + D["opp_fdon"] * 5).astype(int)
    kind = D["kind"].astype(int)
    who = D["who"].astype(int)
    seed = D["seed"]
    C = {}

    rt = np.where(kind == 0, 0, np.where(kind == 2, 3, np.where(my_turn, 1, 2)))
    C["行の種類"] = (rt, ["メインの判断", "自分の手番の効果の対話", "相手の手番の応手・対話", "箱の残りの確定"])
    C["手番の帯"] = _band(turn, [1, 3, 5, 7, 9, 11, 13, 17],
                        ["0（マリガン）", "1〜2", "3〜4", "5〜6", "7〜8", "9〜10", "11〜12", "13〜16", "17〜"])
    lab_l = ["0", "1", "2", "3", "4", "5〜"]
    C["自分のライフ"] = (np.minimum(life_me, 5), lab_l)
    C["相手のライフ"] = (np.minimum(life_opp, 5), lab_l)
    C["ライフの差"] = _band(life_me - life_opp, [-2, -1, 0, 1, 2, 3], ["−3 以下", "−2", "−1", "0", "+1", "+2", "+3 以上"])
    C["手札の差"] = _band(hand_me - hand_opp, [-3, -1, 0, 1, 2, 4],
                        ["−4 以下", "−3〜−2", "−1", "0", "+1", "+2〜+3", "+4 以上"])
    C["自分の手札"] = _band(hand_me, [2, 4, 6, 8], ["0〜1", "2〜3", "4〜5", "6〜7", "8〜"])
    C["相手の手札"] = _band(hand_opp, [2, 4, 6, 8], ["0〜1", "2〜3", "4〜5", "6〜7", "8〜"])
    C["場のドンの差"] = _band(don_me - don_opp, [-2, -1, 0, 1, 2, 3], ["−3 以下", "−2", "−1", "0", "+1", "+2", "+3 以上"])
    C["自分の場のドン"] = _band(don_me, [3, 5, 7, 9, 10], ["0〜2", "3〜4", "5〜6", "7〜8", "9", "10〜"])
    C["自分のキャラの数"] = (np.minimum(f_me, 5), ["0", "1", "2", "3", "4", "5"])
    C["相手のキャラの数"] = (np.minimum(f_opp, 5), ["0", "1", "2", "3", "4", "5"])
    C["キャラの数の差"] = _band(f_me - f_opp, [-2, -1, 0, 1, 2, 3], ["−3 以下", "−2", "−1", "0", "+1", "+2", "+3 以上"])
    pd_ = (D["own_fsum"] - D["opp_fsum"]) * 10000.0
    C["場のパワー合計の差"] = _band(pd_, [-12000, -6000, -2000, 2001, 6001, 12001],
                              ["−12000 以下", "−12000〜−6000", "−6000〜−2000", "±2000 以内", "+2000〜+6000", "+6000〜+12000", "+12000 超"])
    pw_lab = ["キャラなし", "〜4000", "5000〜6000", "7000〜8000", "9000〜"]
    C["自分のキャラの最大パワー"] = _band(D["own_fmax"] * 10000.0, [1, 4500, 6500, 8500], pw_lab)
    C["相手のキャラの最大パワー"] = _band(D["opp_fmax"] * 10000.0, [1, 4500, 6500, 8500], pw_lab)
    dk_lab = ["〜5", "6〜10", "11〜15", "16〜20", "21〜30", "31〜"]
    C["自分の山札の残り"] = _band(deck_me, [6, 11, 16, 21, 31], dk_lab)
    C["相手の山札の残り"] = _band(deck_opp, [6, 11, 16, 21, 31], dk_lab)
    C["自分の立っているブロッカー"] = (np.minimum(np.rint(D["own_blk"]).astype(int), 2), ["0", "1", "2〜"])
    C["相手の立っているブロッカー"] = (np.minimum(np.rint(D["opp_blk"]).astype(int), 2), ["0", "1", "2〜"])
    C["自分の攻撃できる駒"] = (np.minimum(np.rint(D["own_canatk"]).astype(int), 4), ["0", "1", "2", "3", "4〜"])
    C["相手の休んでいるキャラ"] = (np.minimum(np.rint(D["opp_rest"]).astype(int), 2), ["0", "1", "2〜"])
    C["手札のカウンターの合計"] = _band(D["hand_counter"] * 2000.0, [1, 2001, 4001, 6001],
                                 ["0", "〜2000", "〜4000", "〜6000", "6000 超"])
    C["未使用の起動メイン"] = ((D["own_act"] > 0.5).astype(int), ["なし", "あり"])
    C["相手の攻撃に反応する効果（自分の場）"] = ((D["own_trig_oppatk"] > 0.5).astype(int), ["なし", "あり"])
    C["KO 時の効果を持つ相手のキャラ"] = ((D["opp_trig_ko"] > 0.5).astype(int), ["なし", "あり"])
    C["波"] = (D["wave"].astype(int), ["83f", "84f"])

    # ---- 局ごとの量（デッキ・リーダー・先攻）
    useeds, ginv = np.unique(seed, return_inverse=True)
    G = len(useeds)
    # 先攻の席: 手番 1 の行を打った席
    first = np.full(G, -1, int)
    m1 = (turn == 1) & (kind == 0)
    first[ginv[m1]] = who[m1]
    C["先攻・後攻"] = (np.where(first[ginv] < 0, -1, (who != first[ginv]).astype(int)), ["先攻", "後攻"])

    def col_of(cid):
        m = db.get_card(cid)
        names = sorted(getattr(c, "name", str(c)) for c in (getattr(m, "colors", ()) or ()))
        order = ["RED", "GREEN", "BLUE", "PURPLE", "BLACK", "YELLOW"]
        return "".join(COLOR_JA[c] for c in order if c in names) or "?"

    def trig_n(cards):
        k = 0
        for c in cards:
            m = db.get_card(c)
            if any(getattr(getattr(ab, "trigger", None), "name", "") == "TRIGGER" for ab in (m.abilities or [])):
                k += 1
        return k
    lead_g = np.empty((G, 2), object); trig_g = np.zeros((G, 2), int); life0_g = np.zeros((G, 2), int)
    for i, s_ in enumerate(useeds.tolist()):
        pr = decks_mod.real_deck_pair(int(s_), "real")
        for k in (0, 1):
            lead_g[i, k] = pr[k][0]; trig_g[i, k] = trig_n(pr[k][1])
            life0_g[i, k] = int(getattr(db.get_card(pr[k][0]), "life", 0) or 0)
    # 記録の先頭 2 枠（自分・相手のリーダー）と突き合わせる（席の対応の検算）
    lm = lead_g[ginv, who]; lo = lead_g[ginv, 1 - who]
    chk = float(np.mean((lm == D["lead"][:, 0]) & (lo == D["lead"][:, 1])))

    def cat(vals, top=None):
        u, inv = np.unique(vals.astype(str), return_inverse=True)
        return inv.astype(np.int32), list(u)
    cm = np.array([col_of(c) for c in lead_g.ravel()]).reshape(G, 2)
    C["自分のリーダーの色"] = cat(cm[ginv, who])
    C["相手のリーダーの色"] = cat(cm[ginv, 1 - who])
    for cj in COLOR_JA.values():
        has = np.char.find(cm.astype(str), cj) >= 0
        C[f"自分のリーダーが{cj}を含む"] = (np.where(has[ginv, who], 0, -1), ["含む"])
        C[f"相手のリーダーが{cj}を含む"] = (np.where(has[ginv, 1 - who], 0, -1), ["含む"])
    nm = {c: f"{c} {getattr(db.get_card(c), 'name', '')}（{col_of(c)}）" for c in set(lead_g.ravel())}
    C["自分のリーダー（個別）"] = cat(np.array([nm[c] for c in lm]))
    C["相手のリーダー（個別）"] = cat(np.array([nm[c] for c in lo]))
    C["同じリーダー同士か"] = ((lm == lo).astype(int), ["違う", "同じ"])
    C["自分のリーダーの初期ライフ"] = cat(life0_g[ginv, who])
    tg_lab = ["0〜3 枚", "4〜7 枚", "8〜11 枚", "12 枚〜"]
    C["自分のデッキのトリガー札"] = _band(trig_g[ginv, who], [4, 8, 12], tg_lab)
    C["相手のデッキのトリガー札"] = _band(trig_g[ginv, 1 - who], [4, 8, 12], tg_lab)

    # ---- ネット・探索・散らしに関わる切り口
    v = D["v"]
    C["ネットの値の帯"] = _band(v, np.linspace(-0.8, 0.8, 9), [f"{a:+.1f}〜{a+0.2:+.1f}" for a in np.linspace(-1, 0.8, 10)])
    has_v0 = (kind == 0) & (D["pol_len"] > 0)
    dv = np.where(has_v0, D["pol_v0"].astype(np.float32) - v, 0)
    b, lab = _band(dv, [-0.3, -0.1, 0.1, 0.3], ["探索が 0.3 以上 低い", "0.1〜0.3 低い", "±0.1 以内", "0.1〜0.3 高い", "探索が 0.3 以上 高い"])
    C["探索の根の評価 − ネットの値（メイン）"] = (np.where(has_v0, b, -1), lab)
    ex = D["explore"].astype(int) if "explore" in D else np.zeros(n, int)
    C["この行の手を散らしたか"] = (ex, ["散らしていない", "序盤の散らし", "後半の散らし"])
    # この行より前に、この局で後半の散らしがあったか（局ごとに手順の順で累積）
    order = np.lexsort((D["step"], ginv))
    e2 = (ex == 2).astype(np.int64)[order]
    cs = np.cumsum(e2) - e2
    g_o = ginv[order]
    start = np.r_[0, np.flatnonzero(np.diff(g_o)) + 1]
    base = np.repeat(cs[start], np.diff(np.r_[start, n]))
    prior = np.empty(n, np.int64); prior[order] = cs - base
    C["この行より前の後半の散らしの回数"] = _band(prior, [1, 3, 6, 11], ["0", "1〜2", "3〜5", "6〜10", "11〜"])
    C["局面の時期"] = _band(turn, [1, 7, 13], ["マリガン", "手番 1〜6", "手番 7〜12", "手番 13〜"])
    return C, ginv, G, {"seat_check": chk, "games": G}


class Agg:
    """局を単位にした比の推定（切り口のマスごとの平均と 95% 区間）。"""

    def __init__(self, D, ginv, G):
        self.g = ginv.astype(np.int64); self.G = G
        z = D["z"].astype(np.float64); v = D["v"].astype(np.float64)
        self.e = v - z; self.e2 = self.e ** 2; self.z = z; self.v = v
        kind = D["kind"]
        self.m_v0 = (kind == 0) & (D["pol_len"] > 0)
        self.e0 = np.where(self.m_v0, D["pol_v0"].astype(np.float64) - z, 0.0)
        self.m_p = D["pol_dis"] >= 0
        self.dis = np.where(self.m_p, D["pol_dis"], 0).astype(np.float64)
        self.qg = np.where(self.m_p, np.nan_to_num(D["pol_qgap"].astype(np.float64)), 0.0)
        self.n = len(z)

    def _pg(self, key, K, w):
        return np.bincount(key, weights=w, minlength=K * self.G).reshape(K, self.G)

    @staticmethod
    def _ratio(num, den):
        """マスごとの「局ごとの和の比」と局を単位にした標準誤差。num, den: [K, G]"""
        N = den.sum(1); Gc = (den > 0).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = num.sum(1) / N
            res = num - r[:, None] * den
            var = (res ** 2).sum(1) / N ** 2 * Gc / np.maximum(Gc - 1, 1)
        return r, np.sqrt(var), N, Gc

    def cells(self, code, K):
        ok = code >= 0
        key = (code[ok].astype(np.int64) * self.G + self.g[ok])
        one = np.ones(int(ok.sum()))
        n_pg = self._pg(key, K, one)
        b, se, N, Gc = self._ratio(self._pg(key, K, self.e[ok]), n_pg)
        mse = self._pg(key, K, self.e2[ok]).sum(1) / np.maximum(N, 1)
        zb = self._pg(key, K, self.z[ok]).sum(1) / np.maximum(N, 1)
        vb = self._pg(key, K, self.v[ok]).sum(1) / np.maximum(N, 1)
        n0 = self._pg(key, K, self.m_v0[ok].astype(float))
        b0, se0, N0, _ = self._ratio(self._pg(key, K, self.e0[ok]), n0)
        # メインの行（探索の根の評価がある行）での二乗誤差: ネットの値と探索の根の評価を同じ行で比べる
        mse_s = self._pg(key, K, self.e0[ok] ** 2).sum(1) / np.maximum(N0, 1)
        mse_n = self._pg(key, K, np.where(self.m_v0[ok], self.e2[ok], 0.0)).sum(1) / np.maximum(N0, 1)
        npp = self._pg(key, K, self.m_p[ok].astype(float))
        dr, dse, Np, Gp = self._ratio(self._pg(key, K, self.dis[ok]), npp)
        qg = self._pg(key, K, self.qg[ok]).sum(1) / np.maximum(Np, 1)
        out = []
        for k in range(K):
            if N[k] == 0:
                out.append(None); continue
            var_z = max(1e-9, 1.0 - zb[k] ** 2)
            out.append({
                "rows": int(N[k]), "games": int(Gc[k]), "share": float(N[k] / self.n),
                "v_mean": float(vb[k]), "z_mean": float(zb[k]), "bias": float(b[k]),
                "lo": float(b[k] - 1.96 * se[k]), "hi": float(b[k] + 1.96 * se[k]),
                "zscore": float(b[k] / se[k]) if se[k] > 0 else 0.0,
                "mse": float(mse[k]), "mse_const": float(var_z), "skill": float(1.0 - mse[k] / var_z),
                "bias2_cost": float(N[k] / self.n * b[k] ** 2),
                "search_rows": int(N0[k]),
                "net_mse_main": float(mse_n[k]) if N0[k] else None,
                "search_mse_main": float(mse_s[k]) if N0[k] else None,
                "search_bias": float(b0[k]) if N0[k] else None,
                "search_lo": float(b0[k] - 1.96 * se0[k]) if N0[k] else None,
                "search_hi": float(b0[k] + 1.96 * se0[k]) if N0[k] else None,
                "pol_rows": int(Np[k]), "pol_games": int(Gp[k]),
                "pol_disagree": float(dr[k]) if Np[k] else None,
                "pol_lo": float(dr[k] - 1.96 * dse[k]) if Np[k] else None,
                "pol_hi": float(dr[k] + 1.96 * dse[k]) if Np[k] else None,
                "pol_qgap": float(qg[k]) if Np[k] else None})
        return out


def linear_map(D, ginv, G, first_code, matchup):
    """誤差（ネットの値 − 結末）を盤面の量に同時に回帰する（どの量を高く／低く買っているか・他の量をそろえたうえでの 1 単位あたり）。
    区間は局を単位にしたサンドイッチ推定。層（全行・手番の帯 3 つ・行の種類 2 つ）ごとに出す。"""
    sc = D["sc"]
    f = lambda j: sc[:, j].astype(np.float64)                                     # noqa: E731
    X = {
        "自分のライフ（1 枚）": f(0), "相手のライフ（1 枚）": f(1),
        "自分の手札（1 枚）": f(6), "相手の手札（1 枚）": f(7),
        "自分の場のドン（1 枚）": f(2) + f(3) + f(14) * 5 + D["own_fdon"] * 5.0,
        "相手の場のドン（1 枚）": f(4) + f(5) + f(15) * 5 + D["opp_fdon"] * 5.0,
        "自分のキャラ（1 体）": f(8), "相手のキャラ（1 体）": f(9),
        "自分の場のパワー合計（1000）": D["own_fsum"] * 10.0, "相手の場のパワー合計（1000）": D["opp_fsum"] * 10.0,
        "自分の立っているブロッカー（1 体）": D["own_blk"].astype(np.float64),
        "相手の立っているブロッカー（1 体）": D["opp_blk"].astype(np.float64),
        "手札のカウンターの合計（1000）": D["hand_counter"] * 2.0,
        "自分の山札（1 枚）": f(16) * 50.0, "相手の山札（1 枚）": f(17) * 50.0,
        "自分の手番か": f(11), "後攻か": (first_code == 1).astype(np.float64),
    }
    names = list(X)
    M = np.column_stack([np.ones(len(D["v"]))] + [X[k] for k in names])
    e = D["v"].astype(np.float64) - D["z"].astype(np.float64)
    turn = D["turn"].astype(int); kind = D["kind"].astype(int); myt = f(11) > 0.5
    strata = {"全行": np.ones(len(e), bool), "手番 1〜6": (turn >= 1) & (turn <= 6),
              "手番 7〜12": (turn >= 7) & (turn <= 12), "手番 13〜": turn >= 13,
              "メインの判断": kind == 0, "相手の手番の応手・対話": (kind == 1) & ~myt}
    out = {}
    # 対面（自分のリーダー × 相手のリーダー）ごとの平均を全部の列と誤差から引いた版＝リーダーの見誤りを取り除いたうえでの傾き
    _u, mi = np.unique(matchup, return_inverse=True)
    cnt = np.bincount(mi).astype(np.float64)
    Mw = M.copy()
    for j in range(1, M.shape[1]):
        Mw[:, j] -= (np.bincount(mi, weights=M[:, j]) / cnt)[mi]
    ew = e - (np.bincount(mi, weights=e) / cnt)[mi]
    strata["全行（対面ごとの平均を引いた版）"] = None
    for sn, m in strata.items():
        if m is None:
            Xm, em, gm, m = Mw, ew, ginv, np.ones(len(e), bool)
        else:
            Xm, em, gm = M[m], e[m], ginv[m]
        XtX = Xm.T @ Xm
        inv = np.linalg.pinv(XtX)                 # 層の中で動かない列（例: 応手の行の「自分の手番か」）は係数 0 になる
        beta = inv @ (Xm.T @ em)
        r = em - Xm @ beta
        S = np.column_stack([np.bincount(gm, weights=Xm[:, j] * r, minlength=G) for j in range(Xm.shape[1])])
        se = np.sqrt(np.diag(inv @ (S.T @ S) @ inv))
        sd = Xm.std(0)
        out[sn] = {"rows": int(m.sum()),
                   "coef": [{"name": n_, "per_unit": float(beta[j + 1]), "lo": float(beta[j + 1] - 1.96 * se[j + 1]),
                             "hi": float(beta[j + 1] + 1.96 * se[j + 1]), "sd": float(sd[j + 1]),
                             "per_sd": float(beta[j + 1] * sd[j + 1])} for j, n_ in enumerate(names)]}
    return out


#: 掛け合わせに使う切り口（行数が足りるマスだけ集計する）
#: 既定の総当たりとは別に足す掛け合わせ
EXTRA_PAIRS = (("自分のリーダー（個別）", "局面の時期"), ("相手のリーダー（個別）", "局面の時期"),
               ("行の種類", "局面の時期"))
#: 上位の表から外す診断の切り口（条件が誤差そのものを選ぶ・生成の仕掛けで偏りが機械的に出る）
DIAG_ONLY = ("この行の手を散らしたか", "探索の根の評価 − ネットの値（メイン）", "波")
PAIR_CUTS = ("行の種類", "手番の帯", "自分のライフ", "相手のライフ", "ライフの差", "手札の差", "場のドンの差",
             "キャラの数の差", "場のパワー合計の差", "自分のリーダーの色", "相手のリーダーの色", "先攻・後攻",
             "自分の山札の残り", "自分の立っているブロッカー", "相手の立っているブロッカー",
             "自分のデッキのトリガー札", "相手のデッキのトリガー札", "手札のカウンターの合計", "ネットの値の帯")


def report(args):
    root = _use_root(args.root)
    from opcg_sim.loop import decks as DK
    from opcg_sim.learned.vocab import load_db
    t0 = time.time()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    D, n_files = load_rows(args.rows)
    db = load_db()
    C, ginv, G, info = build_cuts(D, db, DK)
    A = Agg(D, ginv, G)
    print(f"rows {A.n} games {G} files {n_files} seat_check {info['seat_check']:.4f} {time.time()-t0:.0f}s", flush=True)

    # **散らした手の行も本体の集計に入れる**: ネットの教師は「散らしも含む生成の打ち方のもとでの結末」なので、
    # 散らした行だけ外すと残りの行が逆向きに寄る（外した版は −0.015・全行は −0.008＝試して戻した）。
    # 散らした行そのものの偏りは診断の切り口「この行の手を散らしたか」で別に出す。
    overall = A.cells(np.zeros(A.n, np.int32), 1)[0]
    single, pair = [], []
    for name, (code, labels) in C.items():
        for k, c in enumerate(A.cells(np.asarray(code), len(labels))):
            if c is not None:
                single.append({"cut": name, "cell": str(labels[k]), **c})
    names = [x for x in PAIR_CUTS if x in C]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, la = C[a]; cb, lb = C[b]
            ca = np.asarray(ca); cb = np.asarray(cb)
            code = np.where((ca >= 0) & (cb >= 0), ca * len(lb) + cb, -1)
            for k, c in enumerate(A.cells(code, len(la) * len(lb))):
                if c is not None and c["games"] >= MIN_GAMES and c["rows"] >= MIN_ROWS:
                    pair.append({"cut": f"{a} × {b}", "cell": f"{la[k // len(lb)]} × {lb[k % len(lb)]}", **c})
        print(f"pairs {a} {time.time()-t0:.0f}s", flush=True)
    for a, b in EXTRA_PAIRS:
        ca, la = C[a]; cb, lb = C[b]
        code = np.where((np.asarray(ca) >= 0) & (np.asarray(cb) >= 0), np.asarray(ca) * len(lb) + np.asarray(cb), -1)
        for k, c in enumerate(A.cells(code, len(la) * len(lb))):
            if c is not None and c["games"] >= MIN_GAMES and c["rows"] >= MIN_ROWS:
                pair.append({"cut": f"{a} × {b}", "cell": f"{la[k // len(lb)]} × {lb[k % len(lb)]}", **c})
    lin = linear_map(D, ginv, G, np.asarray(C["先攻・後攻"][0]),
                     np.asarray(C["自分のリーダー（個別）"][0]).astype(np.int64) * 1000 + np.asarray(C["相手のリーダー（個別）"][0]))
    with open(os.path.join(out_dir, "linear_map.json"), "w", encoding="utf-8") as f:
        json.dump(lin, f, ensure_ascii=False, indent=1)
    for fn, rows in (("cells_single.jsonl", single), ("cells_pair.jsonl", pair)):
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from statistics import NormalDist

    def sig(rows, min_games=MIN_GAMES, zc=0.0):
        return [r for r in rows if r["games"] >= min_games and r["cut"] not in DIAG_ONLY
                and (r["lo"] > 0 or r["hi"] < 0) and abs(r["zscore"]) >= zc]
    tested_s = [r for r in single if r["games"] >= MIN_GAMES and r["cut"] not in DIAG_ONLY]
    n_tests = len(tested_s) + len(pair)
    z_bonf = NormalDist().inv_cdf(1.0 - 0.025 / max(1, n_tests))
    summ = {
        "net": "nrel_r29f4.npz", "waves": ["83f", "84f"], "files": n_files, "rows": A.n, "games": G,
        "seat_check": info["seat_check"], "overall": overall,
        "bonferroni": {"tests": n_tests, "z": round(z_bonf, 3),
                       "single": len(sig(single, zc=z_bonf)), "pair": len(sig(pair, zc=z_bonf))},
        "tested": {"single": len(tested_s), "pair": len(pair)},
        "flagged": {"single": len(sig(single)), "pair": len(sig(pair))},
        "expected_false_flags": {"single": round(0.05 * len(tested_s), 1), "pair": round(0.05 * len(pair), 1)},
        "top_single_by_bias": sorted(sig(single), key=lambda r: -abs(r["bias"]))[:40],
        "top_single_by_cost": sorted(sig(single), key=lambda r: -r["bias2_cost"])[:40],
        "top_pair_by_bias": sorted(sig(pair), key=lambda r: -abs(r["bias"]))[:60],
        "top_pair_by_cost": sorted(sig(pair), key=lambda r: -r["bias2_cost"])[:60],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: summ[k] for k in ("rows", "games", "overall", "tested", "flagged")}, ensure_ascii=False, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="ネットの価値の系統的な見誤りの地図（§7.92 段 C）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--in", dest="src", required=True, help="波 1 本の `n_records/`（下に part のディレクトリ）")
    e.add_argument("--tag", required=True)
    e.add_argument("--out", default="~/vc_rows")
    e.add_argument("--cache", default="~/vc_pack")
    e.add_argument("--net", default="")
    e.add_argument("--root", default=_ROOT, help="`opcg_sim` を読む木（波を打ったコミットの checkout）")
    e.add_argument("--bs", type=int, default=512)
    e.set_defaults(fn=extract)
    r = sub.add_parser("report")
    r.add_argument("--rows", default="~/vc_rows")
    r.add_argument("--out", default="n1_results/value_calib")
    r.add_argument("--root", default=_ROOT, help="`opcg_sim` を読む木（波を打ったコミットの checkout）")
    r.set_defaults(fn=report)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
