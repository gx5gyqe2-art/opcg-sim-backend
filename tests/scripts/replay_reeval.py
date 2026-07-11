"""マーク付きリプレイの再評価ツール（フレーム盤面復元・司令塔 2026-07-09）。

実アプリ対局のリプレイ記録（opcg-replay/v1: decks+actions+frames+marks）の**各マーク地点の直前フレーム
から盤面を直接復元**し、候補ネットに同じ決断をさせて「人間の指摘どおりに手が変わるか」を検証する。
ネット改善（LC/v3/蒸留生徒）の before/after を人間フィードバックで回帰テストできる。

なぜ全編再生でなく盤面復元か（実測に基づく設計）:
  リプレイ種(seed)の乱数同期再現はサーバ版の乱数消費順に脆く、フレーム誘導の全編再生も「山札を覗いて
  戻す効果（覗いた中身がフレームに残らない）」＋ドン経済の1枚差で漂流する。一方フレームは各手番で
  全ゾーンのcard_id・レスト・付与ドン・ドン数・山札数を持つ＝**マーク地点だけを局所復元すれば、そこに
  至る経路の乱数を一切必要としない**。これがネット評価には十分（ネットは盤面スナップのみ参照）。

対象:
  - MAIN_ACTION のマーク＝直前フレーム(i-1)から盤面を復元して decide。
  - SELECT_COUNTER/SELECT_BLOCKER 等の**戦闘中**マーク＝攻撃前フレーム(i-2)から復元し、記録された攻撃
    (action i-1)を declare_attack で再宣言してカウンター待ちを engine に正しく作らせてから decide。
  限界: ごく早い巡目は passive 状態等の微差で復元が近似的（recorded_is_legal で自己診断）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/replay_reeval.py \
    --replay /path/replay.json \
    --net ship=opcg_sim/data/learned/gen2_value.npz,opcg_sim/data/learned/gen2_policy.npz \
    --net v3=/tmp/v3_value.npz,/tmp/v3_policy.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player, Phase
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.models.models import CardInstance, DonInstance
from cpu_selfplay import _load_db

MATCH_KEYS_IGNORE = {"src", "turn", "player"}


def _ci(db, cid, owner, spec=None):
    c = CardInstance(db.get_card(cid), owner)
    if spec:
        if spec.get("is_rest"):
            c.is_rest = True
        ad = spec.get("attached_don") or 0
        if ad:
            c.attached_don = ad
        if spec.get("is_frozen"):
            try:
                c.is_frozen = True
            except Exception:
                pass
    return c


def _dons(owner, n, rested=False):
    out = []
    for _ in range(n):
        d = DonInstance(owner_id=owner)
        out.append(d)
    return out


def _board_from_frame(db, rec, fr, actor_pid):
    """フレーム fr（あるマーク直前の盤面）から MAIN 手番の GameManager を復元する。"""
    from collections import Counter
    players = {}
    for pid in ("p1", "p2"):
        s = fr["players"][pid]
        leader = _ci(db, rec["leaders"][pid], pid, s.get("leader"))
        # デッキ = 全デッキリスト − 可視ゾーンのcard_id（多重集合差・順は任意＝ネットは枚数のみ参照）
        seen = Counter()
        for zone in ("hand", "field", "life", "trash"):
            for c in s.get(zone) or []:
                seen[c["card_id"]] += 1
        if s.get("stage"):
            seen[s["stage"]["card_id"]] += 1
        deck_ids = []
        for cid in rec["decks"][pid]:
            if seen[cid] > 0:
                seen[cid] -= 1
            else:
                deck_ids.append(cid)
        pl = Player(pid, [CardInstance(db.get_card(cid), pid) for cid in deck_ids], leader)
        pl.hand = [_ci(db, c["card_id"], pid, c) for c in s.get("hand") or []]
        pl.field = [_ci(db, c["card_id"], pid, c) for c in s.get("field") or []]
        pl.life = [_ci(db, c["card_id"], pid, c) for c in s.get("life") or []]
        pl.trash = [_ci(db, c["card_id"], pid, c) for c in s.get("trash") or []]
        pl.stage = _ci(db, s["stage"]["card_id"], pid, s["stage"]) if s.get("stage") else None
        leader.attached_don = (s.get("leader") or {}).get("attached_don", 0) or 0
        # ドン: 非付与ぶんを don_active/don_rested に、付与ぶんは attached_to 付きで積む。
        pl.don_active = _dons(pid, s.get("don_active", 0))
        pl.don_rested = _dons(pid, s.get("don_rested", 0))
        attached = leader.attached_don + sum(c.attached_don for c in pl.field)
        pl.don_attached_cards = _dons(pid, attached)
        players[pid] = pl
    m = GameManager(players["p1"], players["p2"])
    m.turn_count = fr.get("turn", 1)
    m.phase = Phase.MAIN
    m.turn_player = players[actor_pid]
    m.opponent = players["p2" if actor_pid == "p1" else "p1"]
    try:
        m.refresh_passive_state()
    except Exception:
        pass
    return m


def _describe(m, mv):
    try:
        return cpu_ai._describe_move(m, mv) or {"action_type": mv.get("action_type")}
    except Exception:
        return {"action_type": mv.get("action_type")}


def _uuid_to_cardid(fr, uuid):
    """フレーム内の全ゾーンから uuid→card_id を引く（元対局の uuid をcard_idへ翻訳）。"""
    for pid in ("p1", "p2"):
        s = fr["players"][pid]
        for zone in ("hand", "field", "life", "trash"):
            for c in s.get(zone) or []:
                if c.get("uuid") == uuid:
                    return pid, c.get("card_id")
        for key in ("leader", "stage"):
            c = s.get(key)
            if c and c.get("uuid") == uuid:
                return pid, c.get("card_id")
    return None, None


def _find_unit(pl, card_id, active_only=False):
    units = ([pl.leader] if pl.leader else []) + list(pl.field) + ([pl.stage] if pl.stage else [])
    cands = [u for u in units if u.master.card_id == card_id and (not active_only or not u.is_rest)]
    return cands[0] if cands else None


def _board_for_counter_mark(db, rec, frames_by_idx, actions, i):
    """カウンター系マーク(action i)の盤面を復元する。

    直近の攻撃宣言(ATTACK/ATTACK_CONFIRM)を i-1 から後方に探す（間の PASS＝ブロッカー段の
    見送り等・同一戦闘内の応答は飛ばす）。攻撃前フレームから盤面を復元して記録された攻撃を
    再宣言し、SELECT_COUNTER 待ちを engine に正しく作らせる（ブロッカー段があれば PASS で進める）。
    返り値 (manager, defender) または理由文字列。
    """
    from opcg_sim.src.core import action_api
    j = i - 1
    while j >= 0 and actions[j].get("action_type") in ("PASS", "SELECT_BLOCKER"):
        j -= 1   # 同一戦闘の応答（ブロッカー見送り等）を遡って攻撃宣言に着地する
    atk_act = actions[j] if j >= 0 else {}
    if atk_act.get("action_type") not in ("ATTACK", "ATTACK_CONFIRM"):
        return f"直近の攻撃宣言が見つからない（直前手={actions[i-1].get('action_type')}）"
    pre = frames_by_idx.get(j - 1)
    postatk = frames_by_idx.get(j)
    if pre is None or postatk is None:
        return "攻撃前/後フレームが欠落"
    atk_pid = atk_act["player"]
    m = _board_from_frame(db, rec, pre, atk_pid)
    atk_pl = m.turn_player
    defender = m.opponent
    attacker = _find_unit(atk_pl, atk_act.get("card"), active_only=True)
    if attacker is None:
        return f"攻撃者 {atk_act.get('card')} が復元盤面に見つからない"
    # 対象: 記録の targets（card_id）優先、無ければ frame の battle.target_uuid を翻訳
    tgt_cid = (atk_act.get("targets") or [None])[0]
    if tgt_cid is None:
        b = postatk.get("battle") or {}
        _, tgt_cid = _uuid_to_cardid(postatk, b.get("target_uuid"))
    target = _find_unit(defender, tgt_cid)
    if target is None:
        return f"攻撃対象 {tgt_cid} が復元盤面に見つからない"
    try:
        m.declare_attack(attacker, target)
    except Exception as e:
        return f"declare_attack 失敗: {e}"
    # ブロッカー段が立ったら PASS（記録ストリームに block が無い＝ブロックしない選択）で進める
    pend = m.get_pending_request() or {}
    if pend.get("action") == "SELECT_BLOCKER":
        try:
            action_api.apply_battle_action(m, defender, "PASS", None)
            pend = m.get_pending_request() or {}
        except Exception as e:
            return f"ブロッカー段の PASS 失敗: {e}"
    if pend.get("action") != "SELECT_COUNTER" or pend.get("player_id") != defender.name:
        return f"カウンター待ちに到達しない（pending={pend.get('action')}/{pend.get('player_id')}）"
    return m, defender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--marks", default=None)
    ap.add_argument("--net", action="append", default=[], help="label=value.npz[,policy.npz]")
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    raw = json.load(open(args.replay))
    rec = raw.get("replay", raw)
    frames = raw.get("frames") or []
    marks = (json.load(open(args.marks)) if args.marks else raw).get("marks") or []
    frames_by_idx = {f.get("action_index"): f for f in frames}
    actions = rec["actions"]

    db = _load_db()
    engines = []
    for spec in args.net:
        label, paths = spec.split("=", 1)
        parts = paths.split(",")
        engines.append((label, LearnedEngine(value_path=parts[0],
                                             policy_path=parts[1] if len(parts) > 1 else None)))
    if not engines:
        print("警告: --net 未指定＝復元と合法手照合のみ（ネット評価なし）", flush=True)

    print(f"リーダー: p1={rec['leaders']['p1']} p2={rec['leaders']['p2']} / marks={[mk['action_index'] for mk in marks]}",
          flush=True)
    results, supported = [], 0
    for mk in marks:
        i = mk["action_index"]
        act = actions[i]
        pid = act["player"]
        rec_move = {k: v for k, v in act.items() if k not in MATCH_KEYS_IGNORE}
        at = rec_move.get("action_type")
        print(f"\n=== mark@{i} T{mk.get('turn')} [{pid}] 記録手={rec_move}", flush=True)
        print(f"    指摘: {mk.get('note')}", flush=True)
        if at in ("SELECT_COUNTER", "SELECT_BLOCKER", "PASS"):
            built = _board_for_counter_mark(db, rec, frames_by_idx, actions, i)
            if isinstance(built, str):
                print(f"    → 戦闘状態の復元に失敗（{built}）＝スキップ", flush=True)
                continue
            m, actor = built
            print(f"    （攻撃再現: {actions[i-1].get('card')} → カウンター待ちを復元）", flush=True)
        else:
            pre = frames_by_idx.get(i - 1)
            if pre is None:
                print("    → 直前フレームが無い＝スキップ", flush=True)
                continue
            m = _board_from_frame(db, rec, pre, pid)
            actor = m.turn_player
        legal = m.get_legal_actions(actor)
        rec_legal = any(all(_describe(m, x).get(k) == v for k, v in rec_move.items()) for x in legal)
        supported += 1
        row = {"action_index": i, "turn": mk.get("turn"), "player": pid,
               "note": mk.get("note"), "recorded": rec_move, "recorded_is_legal": rec_legal, "nets": {}}
        if not rec_legal:
            print(f"    ⚠️ 復元盤面で記録手が合法手に無い（復元近似の限界）。合法手例: "
                  f"{[_describe(m, x) for x in legal][:6]}", flush=True)
        for label, eng in engines:
            trace = {}
            try:
                mv = eng.decide(m, actor, sims=args.sims, rng=np.random.default_rng(777 + i), trace=trace)
                d = _describe(m, mv)
                same = all(d.get(k) == v for k, v in rec_move.items())
            except Exception as e:
                d, same = {"error": str(e)}, False
            row["nets"][label] = {"chosen": d, "value": trace.get("value"),
                                  "candidates": (trace.get("candidates") or [])[:4], "same_as_recorded": same}
            tag = "＝記録と同じ" if same else "→ 変化!"
            cand = "; ".join(f"{(_c.get('move') or {}).get('action_type','?')}"
                             f"{('/'+str((_c.get('move') or {}).get('card'))) if (_c.get('move') or {}).get('card') else ''}"
                             f":{_c.get('visit_pct')}%" for _c in (trace.get("candidates") or [])[:3])
            print(f"    [{label:12s}] {d} v={trace.get('value')} {tag}", flush=True)
            if cand:
                print(f"                  上位候補: {cand}", flush=True)
        results.append(row)

    print(f"\n完了: マーク{len(marks)}件中 復元評価{supported}件（MAIN手番＋カウンター戦闘マーク対応）", flush=True)
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), ensure_ascii=False, indent=1)
        print(f"JSON出力: {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
