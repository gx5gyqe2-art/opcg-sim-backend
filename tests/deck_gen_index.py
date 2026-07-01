"""デッキ生成用オフラインインデックス（v4b 実装ステップ2）。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md §生成。エンジン駆動型パラメトリック生成の部品:
  1. **Trait両側インデックス**: trait→保持者(members) / trait→効果参照者(referencers)。
     二部サンプリング（λ=集中度・α=members:referencers 比）の母集団（レビュー第6巡）。
  2. **機序ペア（Action-Condition Pair）**: Traitを共有しないメカニクス由来コンボの抽出
     （コスト操作↔コスト閾値参照・パワー低下↔パワー閾値参照・レスト付与↔レスト参照）（第6巡）。
  3. **役割クラスタ**: fingerprint（色除去）の k-means 事前分類。パラメトリック充填は毎ゲームの
     ナップサックを解かずクラスタから一様サンプル（第5巡）。
全てオフライン1回の計算（決定的・seed固定）。実デッキリストは一切参照しない（リーク禁止）。
"""
from collections import defaultdict

import numpy as np

import rl_fingerprint as FP

COLOR_SLICE = (7, 13)   # fingerprint の raw 色次元（クラスタリングから除外）
DECK_TYPES = ("CHARACTER", "EVENT", "STAGE")   # デッキに入る種別（LEADER以外）


def _iter_nodes(node):
    """effect/cost 木から GameAction を列挙（Sequence/Branch/sub_effect を再帰）。"""
    yield from FP._iter_actions(node)


def _iter_all_actions(master):
    for ab in (getattr(master, "abilities", None) or []):
        for node in (getattr(ab, "effect", None), getattr(ab, "cost", None)):
            if node is not None:
                yield from _iter_nodes(node)


def _enum(x):
    return getattr(x, "name", str(x))


def build_indexes(db):
    """DB全カードから (trait_members, trait_referencers, mech_pairs, deck_pool) を構築。"""
    trait_members = defaultdict(list)      # trait -> [card_id] （そのTraitを持つデッキ投入可カード）
    trait_referencers = defaultdict(list)  # trait -> [card_id] （効果対象条件にそのTraitを指定）
    mech = {k: [] for k in (
        "cost_setter", "cost_referencer",      # コスト操作 ↔ コスト閾値参照（黒コンボ等）
        "power_debuffer", "power_referencer",  # パワー低下 ↔ パワー閾値参照
        "rest_setter", "rest_referencer",      # レスト付与 ↔ レスト状態参照
    )}
    deck_pool = []                          # デッキ投入可の全 card_id
    for cid in db.raw_db.keys():
        m = db.get_card(cid)
        if m is None:
            continue
        tname = _enum(getattr(m, "type", None))
        if tname not in DECK_TYPES:
            continue
        deck_pool.append(cid)
        for t in (getattr(m, "traits", None) or []):
            trait_members[t].append(cid)
        ref_traits = set()
        is_cost_setter = is_cost_ref = False
        is_pow_debuff = is_pow_ref = False
        is_rest_setter = is_rest_ref = False
        for a in _iter_all_actions(m):
            at = _enum(getattr(a, "type", None))
            tq = getattr(a, "target", None)
            status = getattr(a, "status", None)
            base = (getattr(getattr(a, "value", None), "base", 0) or 0)
            # コスト操作: 専用型に加え、総称 BUFF は status で判別（COST_REDUCTION/COST_OVERRIDE）。
            if at in ("COST_BUFF", "SET_COST", "COST_CHANGE") or \
               (at == "BUFF" and status in ("COST_REDUCTION", "COST_OVERRIDE")):
                is_cost_setter = True
            if at in ("REST", "FREEZE"):
                is_rest_setter = True
            # パワー低下: BP_BUFF/総称 BUFF（status がコスト/カウンター以外）の負値、または上書きの負値。
            if base < 0 and (at == "BP_BUFF" or
                             (at == "BUFF" and status not in ("COST_REDUCTION", "COST_OVERRIDE", "COUNTER"))):
                is_pow_debuff = True
            if tq is not None:
                for t in (getattr(tq, "traits", None) or []):
                    ref_traits.add(t)
                if getattr(tq, "cost_max", None) is not None or getattr(tq, "cost_min", None) is not None:
                    is_cost_ref = True
                if getattr(tq, "power_max", None) is not None or getattr(tq, "power_min", None) is not None:
                    is_pow_ref = True
                if getattr(tq, "is_rest", None) is True:
                    is_rest_ref = True
        for t in ref_traits:
            trait_referencers[t].append(cid)
        if is_cost_setter: mech["cost_setter"].append(cid)
        if is_cost_ref: mech["cost_referencer"].append(cid)
        if is_pow_debuff: mech["power_debuffer"].append(cid)
        if is_pow_ref: mech["power_referencer"].append(cid)
        if is_rest_setter: mech["rest_setter"].append(cid)
        if is_rest_ref: mech["rest_referencer"].append(cid)
    return dict(trait_members), dict(trait_referencers), mech, deck_pool


def leader_core_candidates(db, leader_id, trait_members, trait_referencers, mech, cap=40):
    """リーダーのコアパッケージ候補をDB構造から機械導出（実リスト不参照）。

    (a) リーダーTraitの保持者/参照者（同色のみ）、(b) リーダー効果が参照するTraitの保持者、
    (c) 機序ペア（リーダーが setter なら対応 referencer 側、逆も）。同色フィルタはデッキ構築ルール。
    """
    lm = db.get_card(leader_id)
    lcol = {getattr(x, "value", x) for x in (getattr(lm, "colors", []) or [])}

    def legal(cid):
        m = db.get_card(cid)
        ccol = {getattr(x, "value", x) for x in (getattr(m, "colors", []) or [])}
        return bool(lcol & ccol)

    cands = []
    seen = set()

    def add(cids):
        for c in cids:
            if c not in seen and legal(c):
                seen.add(c); cands.append(c)

    for t in (getattr(lm, "traits", None) or []):
        add(trait_members.get(t, []))
        add(trait_referencers.get(t, []))
    ref_traits = set()
    l_cost_setter = l_rest_setter = False
    for a in _iter_all_actions(lm):
        at = _enum(getattr(a, "type", None))
        tq = getattr(a, "target", None)
        if at in ("COST_BUFF", "SET_COST", "COST_CHANGE"):
            l_cost_setter = True
        if at in ("REST", "FREEZE"):
            l_rest_setter = True
        if tq is not None:
            for t in (getattr(tq, "traits", None) or []):
                ref_traits.add(t)
    for t in ref_traits:
        add(trait_members.get(t, []))
    if l_cost_setter:
        add(mech["cost_referencer"])
    if l_rest_setter:
        add(mech["rest_referencer"])
    return cands[:cap]


def build_role_clusters(db, deck_pool, k=12, iters=25, seed=0):
    """fingerprint（色次元除去）の k-means 役割クラスタ。{cluster_id: [card_id]} を返す（決定的）。"""
    rng = np.random.default_rng(seed)
    fps = FP.build_fingerprints(db)
    a, b = COLOR_SLICE
    X = []
    ids = []
    for cid in deck_pool:
        v = fps[cid].copy()
        v[a:b] = 0.0
        X.append(v); ids.append(cid)
    X = np.stack(X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    C = Xn[rng.choice(len(Xn), size=k, replace=False)]
    for _ in range(iters):
        d = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            pts = Xn[lab == j]
            if len(pts):
                C[j] = pts.mean(0)
    out = defaultdict(list)
    for cid, l in zip(ids, lab):
        out[int(l)].append(cid)
    return dict(out)
