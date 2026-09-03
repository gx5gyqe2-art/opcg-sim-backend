"""エネル h1@2 のドン経済とリーダー起動の監査（2026-09-02・読み取り専用の計器）。

なぜ要るか: 残ドン掘りの方針対照（`docs/reports/2026-09-02_residual_dig_cf.md`）で「掘りは
報われない」と出た。原因が (H1) エンジンの力学が実戦と違う／(H2) CPU が引いた札・戻した
ドンを活かせない／(H3) 実験規則が裁定より広い、のどれかを切り分けるため、h1@2（エネル・
turn1・手札にサトリ）から**掘る／掘らない**の2分岐をエンジンで実際に進めて数える:

  1. サトリ登場 → ドン‼-1（SELECT_RESOURCE）→ ドロー が成立するか、翌ターン（自分の第2
     ターン）のドンフェイズ＋リーダー起動後に**場のドン合計が両分岐で一致**するか（＝「実質
     無料」がエンジン上で成り立つか・H1）
  2. リーダー起動の付与対話（自分のキャラへ「〜まで」）を学習エンジンの adapter が
     **選択肢として列挙**しているか（gamestate の既定解決は「付与しない」なので、列挙が無いと
     付与は CPU に到達不能＝H1 の別形）
  3. その局面で c10（`--net`）が実際に何を選ぶか、探索の根統計（訪問数・Q）で
     ACTIVATE_MAIN がどう評価されているか（H2）

実測（2026-09-02・c10）: 1 は成立（両分岐とも turn2 で場のドン 6・掘り側は手札+1 相当の
体が場に残る）。2 は列挙あり（付与先にサトリを選ぶ手が合法手に出る）。3 は **この局面では
c10 がリーダー効果を起動しない**——ACTIVATE_MAIN は探索されている（128 sims で 11〜22 訪問）が
Q≈−0.03〜−0.09 で、リーダーに2枚付けて殴る DON_BOX（Q +0.10〜+0.15）に負ける。512 sims でも
変わらない。**ただし局面依存**（2026-09-03 追記）: 一般の対局（合成デッキ・対クロコダイル
3 seed）では c10 は第2ターン（t3）で毎回起動し、飛ばすターンが 1〜2 回ある程度＝系統的な
盲点ではない。`docs/reports/2026-09-03_residual_activate_cf.md`。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/scripts python tests/scripts/don_refund_audit.py \\
    --net /home/user/neff_net_c10.npz --sims 128
"""
import argparse
import random as _r

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core import action_api  # noqa: E402

DIG_CARD = "OP15-066"     # サトリ（h1@2 の裁定で出すカード）


def _load_h1():
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_HUMAN["h1"])
    rec = raw.get("replay", raw)
    CR.GAMES["h1"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])


def restore(db):
    m, who = CR._restore_board(db, "h1", 2)
    m.action_events = []
    return m, (who if isinstance(who, str) else who.name)


def P(m, name):
    return m.p1 if m.p1.name == name else m.p2


def _ad(c):
    v = getattr(c, "attached_don", 0) or 0
    return v if isinstance(v, int) else len(v)


def don_state(pl):
    att = sum(_ad(c) for c in list(pl.field) + [pl.leader])
    return {"active": len(pl.don_active), "rested": len(pl.don_rested), "attached": att,
            "field_total": len(pl.don_active) + len(pl.don_rested) + att,
            "deck": len(pl.don_deck), "hand": len(pl.hand)}


def apply(m, name, at, payload):
    m.action_events = []
    action_api.apply_game_action(m, P(m, name), at, payload or {})


def drain(m, log=None):
    """効果対話を gamestate の既定解決で消化（合法手が RESOLVE_EFFECT_SELECTION だけの間）。"""
    for _ in range(30):
        pa = m.pending_actor_action()
        if pa is None:
            return
        legal = m.get_legal_actions()
        if legal and all(x.get("action_type") == "RESOLVE_EFFECT_SELECTION" for x in legal):
            if log:
                req = m.get_pending_request() or {}
                log(f"     [対話] {pa[1]} 候補{len(req.get('selectable_uuids') or [])}件 "
                    f"制約{req.get('constraints')} → 既定 sel="
                    f"{len(legal[0]['payload'].get('selected_uuids') or [])}件")
            apply(m, pa[0], "RESOLVE_EFFECT_SELECTION", legal[0]["payload"])
        else:
            return


def end_turn(m, name):
    apply(m, name, "TURN_END", {})
    drain(m)


def opp_pass(m, me):
    """相手の手番を TURN_END だけで流す（対話は既定解決）。"""
    for _ in range(10):
        pa = m.pending_actor_action()
        if pa is None or pa[0] == me:
            return
        drain(m)
        pa = m.pending_actor_action()
        if pa and pa[0] != me:
            end_turn(m, pa[0])


def to_turn2(db, dig, log=print):
    m, me = restore(db)
    pl = P(m, me)
    if dig:
        legal = m.get_legal_actions(pl)
        mv = [x for x in legal if x.get("action_type") == "PLAY"
              and any(c.uuid == x["payload"]["uuid"] and c.master.card_id == DIG_CARD
                      for c in pl.hand)]
        if not mv:
            raise SystemExit(f"{DIG_CARD} の PLAY が合法手に無い")
        apply(m, me, "PLAY", mv[0]["payload"])
        drain(m, log)
        log(f"  掘り直後: {don_state(pl)}")
    end_turn(m, me)
    opp_pass(m, me)
    return m, me


def audit_refund(db):
    print("=== 1) ドン返還の力学（turn2 リーダー起動後の場のドン合計）")
    out = {}
    for dig in (True, False):
        m, me = to_turn2(db, dig)
        pl = P(m, me)
        before = don_state(pl)
        am = [x for x in m.get_legal_actions(pl) if x.get("action_type") == "ACTIVATE_MAIN"
              and x["payload"].get("uuid") == pl.leader.uuid]
        if am:
            apply(m, me, "ACTIVATE_MAIN", am[0]["payload"])
            drain(m)
        out[dig] = don_state(pl)
        print(f"  {'掘る  ' if dig else '掘らない'}: ドンフェイズ後 {before} → 起動後 {out[dig]}")
    same = out[True]["field_total"] == out[False]["field_total"]
    print(f"  場のドン合計 一致={same}（掘る {out[True]['field_total']} / 掘らない "
          f"{out[False]['field_total']}）・掘り側 active 差 {out[True]['active'] - out[False]['active']:+d}")
    return same


def audit_attach_enumeration(db):
    print("=== 2) 付与対話の列挙（学習エンジン adapter）")
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    m, me = to_turn2(db, True)
    pl = P(m, me)
    am = [x for x in m.get_legal_actions(pl) if x.get("action_type") == "ACTIVATE_MAIN"
          and x["payload"].get("uuid") == pl.leader.uuid][0]
    apply(m, me, "ACTIVATE_MAIN", am["payload"])
    g = LearnedEngine().game
    la = g.legal_actions(m)
    pick = [x for x in la if (x.get("payload") or {}).get("selected_uuids")]
    print(f"  pending={m.pending_actor_action()} adapter 合法手 {len(la)} 本・付与先を選ぶ手 {len(pick)} 本")
    if pick:
        apply(m, me, "RESOLVE_EFFECT_SELECTION", pick[0]["payload"])
        drain(m)
        sat = [c for c in pl.field if c.master.card_id == DIG_CARD]
        print(f"  付与後: {don_state(pl)} サトリ付与ドン={_ad(sat[0]) if sat else None}")
    return bool(pick)


def audit_root_stats(db, net_path, sims):
    print(f"=== 3) c10 の探索根統計（sims={sims}）")
    import n_eff_gate
    for dig in (True, False):
        m, me = to_turn2(db, dig)
        pl = P(m, me)
        eng = n_eff_gate.neff_engine(net_path)
        _r.seed(1)
        eng._world_seeds = {}
        eng._commits.clear()
        rec = {}
        eng.decide(m, pl, sims=sims, rng=np.random.default_rng(0), record=rec)
        gs = sorted(rec.get("groups", []), key=lambda g: -g["n"])
        tot = sum(g["n"] for g in gs) or 1
        print(f"  [{'掘り' if dig else '掘らない'}分岐 turn2] 選択={rec.get('sig')[0] if rec.get('sig') else None}")
        for g in gs[:6]:
            sig = g["sig"]
            c = m._find_card_by_uuid(sig[1]) if sig[1] else None
            cid = getattr(getattr(c, "master", None), "card_id", "") if c is not None else ""
            print(f"     {sig[0]:14s} {cid:10s} k={g.get('k')} n={g['n']:6.1f} ({g['n']/tot:5.1%}) q={g['q']:+.3f}")
        am = [g for g in gs if g["sig"][0] == "ACTIVATE_MAIN"]
        print("     ACTIVATE_MAIN:", [(g["n"], round(g["q"], 3)) for g in am] or "候補に無い/訪問0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="N系ネット npz（neff）。未指定なら 3) を省く")
    ap.add_argument("--sims", type=int, default=128)
    args = ap.parse_args()
    db = _load_db()
    _load_h1()
    audit_refund(db)
    audit_attach_enumeration(db)
    if args.net:
        audit_root_stats(db, args.net, args.sims)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
