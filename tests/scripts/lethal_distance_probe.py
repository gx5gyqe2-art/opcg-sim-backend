"""リーサル距離のΔ特徴スパイク（G/B共用・2026-08-12・v52/bb2 の起案測定）。

**問い**: 「双方あと何ターンで詰むか」をエンジン仮実行で測った**動力学の要約2値**は、
現行特徴で説明不能と実証済みの乖離盤面族（v51 教師50点・v50 乖離8点）を説明できるか。

**距離の測り方（v0・台本レース）**: 盤面クローン上で、両者が「開発を捨てて全力で顔面を
殴り続ける」台本方策を打つ:
  - 自ターン: リーダー起動能力があれば起動（エネル再装填等を発火させる）→ 全アタッカーで
    相手リーダーへ攻撃（ドンは能力/攻撃の合法手が出るまま）→ TURN_END
  - 防御側: 常に素通し（PASS/ブロックなし）＝**無抵抗リーサル距離**（カウンター資源は
    既存特徴 v6 が別途持っているので、距離は純粋なレース速度を測る）
  - 効果対話は既定解決（受ける側）。K=12 ターンで打ち切り。
  - MCTS 不使用＝決定論・軽量（1盤面あたり数十〜数百ms）。
**限界（明示）**: カード起因の持続回復・追加展開の伸びしろは台本が打たないため映らない。
これで58点が説明できるか自体がこのスパイクの判定対象。

**v3（忠実度改善・2026-08-12 承認）**: v2 の残欠 a〜c を台本に追加。効果の**静的解釈はゼロ**
（帰結は全てエンジン実行/実測で得る）:
  (a) ブロッカー: リーダーへの攻撃をカウンターで止め切れない時だけブロック
      （生き残れる最弱ブロッカー優先・いなければ最弱でチャンプ。キャラへの攻撃は守らない）。
  (b) イベントカウンター: 値が静的に読めないため**クローン適用で counter_buff の実変化を実測**
      （v7 登場時実測と同型）。手札カウンターと合算した支払い計画（ドン予算考慮・大きい値から）で
      「止め切れる時だけ」発動する。
  (c) 攻め側のドン付与算術: ①不足アタッカーの底上げ（安い順＝攻撃本数の最大化）→
      ②余りは最強アタッカーへ集中（相手の手札カウンター印字総量を超える need を作れる時だけ）。
      「7000ちょうど×複数本」（ユーザ理論）の一般形。v3 は枝刈り無し gs（prune_futile=False）で
      測る＝過剰付与も台本の意思で打てる。
  残欠（v3 でも打たない）: トリガー(d)・攻め側のイベント/キャラ追加展開。相手カウンター総量の
  見積りは印字値のみ（イベント分は過小評価＝保守側）。

出力: 各盤面の (自距離, 相手距離, 距離差) と、
  (a) 乖離族での符号説明力（距離差の符号 vs 実測EVの符号）
  (b) 既存単純特徴（ライフ差・盤面火力差）への上乗せ（最小二乗の r 比較）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/lethal_distance_probe.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai  # noqa: E402

MAX_TURNS = 12
MAX_STEPS = 240


def _desc(m, mv):
    try:
        return cpu_ai._describe_move(m, mv) or {}
    except Exception:
        return {}


def _counter_value(owner, card_id):
    for c in owner.hand:
        if getattr(c.master, "card_id", None) == card_id:
            v = int(getattr(c, "current_counter", 0) or 0)
            if v > 0:
                return v
    return 0


def _find_card(cards, uuid):
    for c in cards or ():
        if getattr(c, "uuid", None) == uuid:
            return c
    return None


def _battle_need(ab):
    """今の攻防で防御側が止めるのに必要なカウンター量（0=素で不成立）。"""
    try:
        atk = int(ab["attacker"].get_power(True))
        tgt = int(ab["target"].get_power(False)) + int(ab.get("counter_buff", 0) or 0)
    except Exception:
        return 0
    return atk - tgt + 1000 if atk >= tgt else 0   # 攻撃は atk>=def で通る


def _counter_plan(gs, m, cur_name, owner, need, ctr_moves):
    """カウンター窓の支払い計画（v3）: 手札カウンターは印字値、イベントは**クローン適用で
    counter_buff の実変化を実測**。ドン予算を考慮した貪欲（値の大きい順）で need を満たせる
    計画の先頭手を返す（満たせなければ None＝温存）。"""
    ab = getattr(m, "active_battle", None)
    if ab is None:
        return None
    buf0 = int(ab.get("counter_buff", 0) or 0)
    items = []                                     # (実効値, ドンコスト, mv)
    for mv in ctr_moves:
        c = _find_card(owner.hand, mv.get("card_uuid"))
        if c is None:
            continue
        v = int(getattr(c, "current_counter", 0) or 0)
        if v > 0:
            items.append((v, 0, mv))
            continue
        m2 = gs.apply(m, mv, cur_name)             # イベント: 実測（静的解釈しない）
        ab2 = getattr(m2, "active_battle", None) if m2 is not None else None
        if ab2 is None:
            continue
        dv = int(ab2.get("counter_buff", 0) or 0) - buf0
        if dv > 0:
            items.append((dv, int(getattr(c.master, "cost", 0) or 0), mv))
    items.sort(key=lambda x: -x[0])
    don = len(owner.don_active)
    got, first = 0, None
    for v, cost, mv in items:
        if got >= need:
            break
        if cost > don:
            continue
        don -= cost
        got += v
        if first is None:
            first = mv
    return first if got >= need else None


def _counter_total(gs, m, cur_name, owner, ctr_moves):
    """支払い計画と同じ実測規約での「今出せるカウンター総量」（ブロック判断用）。"""
    ab = getattr(m, "active_battle", None)
    if ab is None:
        return 0
    buf0 = int(ab.get("counter_buff", 0) or 0)
    total, don = 0, len(owner.don_active)
    ev = []
    for mv in ctr_moves:
        c = _find_card(owner.hand, mv.get("card_uuid"))
        if c is None:
            continue
        v = int(getattr(c, "current_counter", 0) or 0)
        if v > 0:
            total += v
            continue
        m2 = gs.apply(m, mv, cur_name)
        ab2 = getattr(m2, "active_battle", None) if m2 is not None else None
        if ab2 is None:
            continue
        dv = int(ab2.get("counter_buff", 0) or 0) - buf0
        if dv > 0:
            ev.append((dv, int(getattr(c.master, "cost", 0) or 0)))
    for dv, cost in sorted(ev, key=lambda x: -x[0]):
        if cost <= don:
            don -= cost
            total += dv
    return total


def _v3_attacker_move(gs, m, name, descs):
    """v3 攻め側のドン付与算術（c）: ①不足アタッカーの底上げ（安い順・攻撃本数最大化）→
    ②余りは最強へ集中（相手手札の印字カウンター総量を超える need を作れる時だけ）→
    ③最強から顔面攻撃。付与・攻撃とも 1 手ずつ返す（次の手番で再計画）。"""
    me = m.p1 if m.p1.name == name else m.p2
    opp = m.p2 if m.p1.name == name else m.p1
    if opp.leader is None:
        return None
    lid_uuid = opp.leader.uuid
    try:
        def_p = int(opp.leader.get_power(False))
    except Exception:
        def_p = int(getattr(opp.leader.master, "power", 0) or 0)
    budget = len(me.don_active)
    opp_ctr = sum(int(getattr(c, "current_counter", 0) or 0) for c in opp.hand)
    mine = ([me.leader] if me.leader is not None else []) + list(me.field)

    face, attach = [], {}
    for _d, mv in descs:
        at = mv.get("action_type")
        pl = mv.get("payload") or {}
        if at == "ATTACK" and (pl.get("target_ids") or [None])[0] == lid_uuid:
            c = _find_card(mine, pl.get("uuid"))
            if c is not None:
                try:
                    p = int(c.get_power(True))
                except Exception:
                    p = int(getattr(c.master, "power", 0) or 0)
                face.append((p, pl.get("uuid"), mv))
        elif at == "ATTACH_DON":
            attach[pl.get("uuid")] = mv
    if not face:
        return None
    face.sort(key=lambda x: -x[0])

    if budget > 0:
        # ① 不足アタッカーの底上げ（必要ドンが最少の順＝本数最大化）
        deficits = sorted(((def_p - p + 999) // 1000, u) for p, u, _mv in face if p < def_p)
        fixable = [(k, u) for k, u in deficits if k <= budget]
        if fixable and fixable[0][1] in attach:
            return attach[fixable[0][1]]
        # ② 余りは最強へ集中（止め切れない need を作れる見込みがある時だけ）
        p0, u0, _ = face[0]
        if p0 >= def_p and u0 in attach:
            need = p0 - def_p + 1000
            if opp_ctr >= need and need + budget * 1000 > opp_ctr:
                return attach[u0]
    for p, _u, mv in face:                          # ③ 最強から顔面攻撃（素で通る体のみ）
        if p >= def_p:
            return mv
    return None


def _script_move(gs, m, name, defend=False, v3=False):
    """台本方策の1手: 手番側=能力起動→顔面攻撃→END。非手番側= defend=False なら素通し、
    True なら**カウンター防御台本**（v2）: 攻撃が通る場合のみ、手持ちカウンター合計で
    止め切れるなら最大値から切る（止め切れないなら温存＝PASS）。v2 ではイベントカウンターは
    使わない（値が静的に読めない）。v3=True で忠実度改善 a〜c（モジュール docstring）が入る。"""
    legal = gs.legal_actions(m)
    if not legal:
        return None
    cur = gs.current_player(m)
    descs = [(_desc(m, mv), mv) for mv in legal]
    if cur != name:
        if defend:
            ab = getattr(m, "active_battle", None)
            # (a) ブロック窓（v3）: リーダーが対象で、カウンターで止め切れない時だけブロック。
            blk = [mv for _d, mv in descs if mv.get("action_type") == "SELECT_BLOCKER"]
            if v3 and blk and ab:
                owner = ab["target_owner"]
                if ab["target"] is owner.leader:
                    need = _battle_need(ab)
                    if need > 0:
                        # ブロック判断用の総量: カウンター窓と同じ実測規約（窓前なのでイベントは
                        # 印字0のまま＝クローン適用が効かないため印字値のみ＝保守側）。
                        total = sum(int(getattr(c, "current_counter", 0) or 0)
                                    for c in owner.hand)
                        if total < need:
                            cand = []
                            try:
                                atk = int(ab["attacker"].get_power(True))
                            except Exception:
                                atk = 0
                            for mv in blk:
                                b = _find_card(owner.field, mv.get("card_uuid"))
                                if b is None:
                                    continue
                                try:
                                    bp = int(b.get_power(False))
                                except Exception:
                                    bp = 0
                                cand.append((0 if bp > atk else 1, bp, mv))
                            if cand:               # 生き残れる最弱 → いなければ最弱でチャンプ
                                cand.sort(key=lambda x: (x[0], x[1]))
                                return cand[0][2]
            ctr = [(d, mv) for d, mv in descs if d.get("action_type") == "SELECT_COUNTER"]
            if ctr and ab:
                owner = ab["target_owner"]
                if v3 and ab["target"] is not owner.leader:
                    pass                            # キャラ/ブロッカーは守らない（ライフ優先）
                else:
                    need = _battle_need(ab)
                    if need > 0:
                        if v3:
                            # (b) イベント込みの支払い計画（実測・止め切れる時だけ）
                            mv3 = _counter_plan(gs, m, cur, owner, need,
                                                [mv for _d, mv in ctr])
                            if mv3 is not None:
                                return mv3
                        else:
                            vals = sorted(((_counter_value(owner, d.get("card")), d, mv)
                                           for d, mv in ctr), key=lambda x: -x[0])
                            total = sum(v for v, _d, _m in vals if v > 0)
                            if total >= need and vals and vals[0][0] > 0:
                                return vals[0][2]  # 止め切れる時だけ最大値から切る
        # 相手の自ターン: **何もせず END**（無抵抗レース＝相手の時間だけが流れる）。
        # v0 は誤って legal[0]（任意のプレイ）を打っており距離が 0/13 に崩壊していた
        # （2026-08-12 実測 19/58・引分21）。応答窓: PASS/「しない」で素通し。
        for d, mv in descs:
            if d.get("action_type") == "TURN_END":
                return mv
        for d, mv in descs:
            if d.get("action_type") in ("PASS",):
                return mv
        for d, mv in descs:
            if d.get("action_type") == "RESOLVE_EFFECT_SELECTION" and d.get("accepted") is False:
                return mv
        return legal[0]
    # 効果対話: 既定解決（受ける側・列挙順先頭）
    resolves = [(d, mv) for d, mv in descs if d.get("action_type") == "RESOLVE_EFFECT_SELECTION"]
    if resolves:
        for d, mv in resolves:
            if d.get("accepted") is not False:
                return mv
        return resolves[0][1]
    # リーダー起動能力（再装填等の経済を発火させる）
    for d, mv in descs:
        if d.get("action_type") == "ACTIVATE_MAIN":
            me = m.p1 if m.p1.name == name else m.p2
            if me.leader is not None and d.get("card") == me.leader.master.card_id:
                return mv
    # (c) v3: ドン付与算術＋最強から顔面攻撃
    if v3:
        mv3 = _v3_attacker_move(gs, m, name, descs)
        if mv3 is not None:
            return mv3
        for d, mv in descs:
            if d.get("action_type") == "TURN_END":
                return mv
        return legal[0]
    # 相手リーダーへの攻撃を優先（パワー最大の攻撃者から）
    opp = m.p2 if m.p1.name == name else m.p1
    lid = opp.leader.master.card_id if opp.leader else None
    atks = [(d, mv) for d, mv in descs
            if d.get("action_type") == "ATTACK" and (d.get("targets") or [None])[0] == lid]
    if atks:
        return atks[0][1]
    for d, mv in descs:                               # 顔面攻撃が無ければ TURN_END
        if d.get("action_type") == "TURN_END":
            return mv
    return legal[0]


def lethal_distance(gs, m0, name, max_turns=MAX_TURNS, defend=False, v3=False):
    """name 視点: 台本レースで相手を削り切るまでの自ターン数（詰まねば max+1）。
    defend=True＝相手がカウンター防御台本で抵抗する（v2・防御込みリーサル距離）。
    v3=True＝忠実度改善 a〜c（ブロッカー/イベント実測/ドン付与算術）。v3 は枝刈り無し gs
    （OPCGGame(prune_futile=False)）で呼ぶこと（過剰付与の手が枝刈りで消えるため）。"""
    m = m0.clone()
    my_turns = 0
    steps = 0
    max_steps = MAX_STEPS * (2 if v3 else 1)          # v3 は付与手の分ステップが増える
    while steps < max_steps:
        if m.winner is not None:
            return my_turns if m.winner == name else max_turns + 1
        cur = gs.current_player(m)
        if cur is None:
            return max_turns + 1
        mv = _script_move(gs, m, name, defend=defend, v3=v3)
        if mv is None:
            return max_turns + 1
        d = _desc(m, mv)
        is_end = (cur == name and d.get("action_type") == "TURN_END")
        m2 = gs.apply(m, mv, cur)
        if m2 is None:
            return max_turns + 1
        m = m2
        steps += 1
        if is_end:
            my_turns += 1
            if my_turns >= max_turns:
                return max_turns + 1
    return max_turns + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default=None, help="既定=fixtures の v51_teacher")
    args = ap.parse_args()
    import glob
    import time
    import coach_gate as CG  # noqa: F401  （表は使わないが依存を明示）
    from cpu_selfplay import _load_db
    from opcg_game import OPCGGame

    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fixt = args.teachers or os.path.join(REPO, "tests", "fixtures", "candidates", "v51_teacher")

    # 58点の盤面は npz（符号化済み）にしか無い→盤面そのものが要る。教師50点は meta の
    # (seed, turn, who) から**決定論再生**で復元できる（lethal_teacher_gen と同一 seed 規約）。
    # v50 乖離8点はリプレイ復元（tag,i が meta にある）。
    import mark_gate as MG
    import replay_reeval as RE
    import coach_gate as CG2
    from dense_selfplay_gen import _make_fixed_matchup_game
    from opcg_sim.src.core.cpu_learned import LearnedEngine

    db = _load_db()
    gs = OPCGGame()
    gs_raw = OPCGGame(prune_futile=False)   # v3 用（過剰ドン付与の手が枝刈りで消えないように）
    eng = LearnedEngine()
    game = _make_fixed_matchup_game(
        os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json"),
        "nami", "shanks")

    rows = []
    t0 = time.time()

    # --- 教師50点（meta から決定論再生: 同じ seed で自己対戦を再走し (turn,who) の盤面を取る）
    for meta_f in ("meta_r1.json", "meta_r2.json"):
        meta = json.load(open(os.path.join(fixt, meta_f)))
        wanted = {}
        for d in meta["diag"]:
            if d.get("teach"):
                wanted.setdefault(d["seed"], []).append((d["turn"], d["who"], d["ev"]))
        for seed, pts in sorted(wanted.items()):
            m = game.new_game(db, seed)
            drng = np.random.default_rng(seed * 17 + 3)
            left = {(t, w): ev for t, w, ev in pts}
            steps = 0
            while left and m.winner is None and not gs.is_terminal(m) and steps < 400:
                name = gs.current_player(m)
                if name is None:
                    break
                t = int(getattr(m, "turn_count", 0) or 0)
                if (t, name) in left:
                    ev = left.pop((t, name))
                    dme = lethal_distance(gs, m, name)
                    opp = m.p2.name if m.p1.name == name else m.p1.name
                    dop = lethal_distance(gs, m, opp)
                    dme_d = lethal_distance(gs, m, name, defend=True)
                    dop_d = lethal_distance(gs, m, opp, defend=True)
                    dme_3 = lethal_distance(gs_raw, m, name, defend=True, v3=True)
                    dop_3 = lethal_distance(gs_raw, m, opp, defend=True, v3=True)
                    s = None  # ライフ差/火力差は再計算
                    me = m.p1 if m.p1.name == name else m.p2
                    op_ = m.p2 if m.p1.name == name else m.p1
                    rows.append({"src": meta_f[:-5], "seed": seed, "turn": t, "who": name,
                                 "ev": ev, "d_me": dme, "d_opp": dop,
                                 "d_me_def": dme_d, "d_opp_def": dop_d,
                                 "d_me_v3": dme_3, "d_opp_v3": dop_3,
                                 "life_diff": len(me.life or []) - len(op_.life or [])})
                actor = m.p1 if m.p1.name == name else m.p2
                eng._world_seeds = {}
                mv = eng.decide(m, actor, sims=32, rng=drng)
                if mv is None:
                    break
                m2 = gs.apply(m, mv, name)
                if m2 is None:
                    break
                m = m2
                steps += 1
            print(f"  {meta_f} seed{seed}: 残り{len(left)} 累計{len(rows)}行"
                  f" {time.time()-t0:.0f}s", flush=True)

    # --- v50 乖離8点（リプレイ復元・|err|>=0.85）
    V50_PTS = [("g2", 80), ("g2", 82), ("m2", 59), ("m2", 60), ("m2", 76),
               ("m5", 62), ("e2", 121), ("h1", 91)]
    table = {**MG.REPLAYS, **CG2.REPLAYS_V2, **CG2.REPLAYS_V48, **CG2.REPLAYS_HUMAN}
    # v50 の EV（v50 スキャンの実測・e2/h1 はエネル対面＝ラベルにブートストラップ留保あり）
    V50_EV = {("g2", 80): 0.667, ("g2", 82): 0.667, ("m2", 59): -0.667, ("m2", 60): 0.667,
              ("m2", 76): 1.0, ("m5", 62): 1.0, ("e2", 121): 0.0, ("h1", 91): -0.333}
    for tag, i in V50_PTS:
        raw = RE.load_replay_json(table[tag])
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        built = MG._restore(db, rec, fbi, acts, i)
        if isinstance(built, str) or built is None:
            continue
        m0, actor = built
        name = actor.name if hasattr(actor, "name") else actor
        dme = lethal_distance(gs, m0, name)
        opp = m0.p2.name if m0.p1.name == name else m0.p1.name
        dop = lethal_distance(gs, m0, opp)
        dme_d = lethal_distance(gs, m0, name, defend=True)
        dop_d = lethal_distance(gs, m0, opp, defend=True)
        dme_3 = lethal_distance(gs_raw, m0, name, defend=True, v3=True)
        dop_3 = lethal_distance(gs_raw, m0, opp, defend=True, v3=True)
        me = m0.p1 if m0.p1.name == name else m0.p2
        op_ = m0.p2 if m0.p1.name == name else m0.p1
        rows.append({"src": "v50", "seed": f"{tag}@{i}", "turn": int(acts[i].get("turn", 0) or 0),
                     "who": name, "ev": V50_EV[(tag, i)], "d_me": dme, "d_opp": dop,
                     "d_me_def": dme_d, "d_opp_def": dop_d,
                     "d_me_v3": dme_3, "d_opp_v3": dop_3,
                     "life_diff": len(me.life or []) - len(op_.life or [])})
        print(f"  v50 {tag}@{i}: d_me={dme} d_opp={dop} ev={V50_EV[(tag, i)]:+.2f}", flush=True)

    # --- 説明力の集計
    ev = np.array([r["ev"] for r in rows], float)
    dd = np.array([r["d_opp"] - r["d_me"] for r in rows], float)   # 正＝自分が先に詰ませる
    ddf = np.array([r.get("d_opp_def", 0) - r.get("d_me_def", 0) for r in rows], float)
    dd3 = np.array([r.get("d_opp_v3", 0) - r.get("d_me_v3", 0) for r in rows], float)
    ld = np.array([r["life_diff"] for r in rows], float)
    ok = np.sign(dd) == np.sign(ev)
    okf = np.sign(ddf) == np.sign(ev)
    ok3 = np.sign(dd3) == np.sign(ev)
    tiedf = ddf == 0
    tied = dd == 0
    tied3 = dd3 == 0
    print(f"\n=== リーサル距離の説明力（{len(rows)}点＝全て現行特徴で説明不能の乖離盤面）")
    print(f"  無抵抗:      符号一致 {int(ok.sum())}/{len(rows)}（引分 {int(tied.sum())}）")
    print(f"  防御込みv2:  符号一致 {int(okf.sum())}/{len(rows)}（引分 {int(tiedf.sum())}）")
    print(f"  防御込みv3:  符号一致 {int(ok3.sum())}/{len(rows)}（引分 {int(tied3.sum())}）")
    print(f"  参照: ライフ差の符号一致 {int((np.sign(ld) == np.sign(ev)).sum())}/{len(rows)}")
    if len(rows) > 3:
        print(f"  相関: 無抵抗 r={np.corrcoef(dd, ev)[0,1]:+.3f} / 防御込みv2 r={np.corrcoef(ddf, ev)[0,1]:+.3f}"
              f" / 防御込みv3 r={np.corrcoef(dd3, ev)[0,1]:+.3f} / ライフ差 r={np.corrcoef(ld, ev)[0,1]:+.3f}")
    print("LETHAL_DISTANCE_SPIKE " + json.dumps(
        {"n": len(rows), "sign_ok": int(ok.sum()), "tied": int(tied.sum()),
         "sign_ok_v3": int(ok3.sum()), "tied_v3": int(tied3.sum()),
         "rows": rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
