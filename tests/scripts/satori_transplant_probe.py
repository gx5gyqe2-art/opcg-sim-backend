"""サトリ移植プローブ（v49・2026-08-11・読み取り専用・ユーザ発案）。

**問い**: 学習した「掘り」は**リーダーの経済**（ドン再装填）に紐づいているか、
**カード/浅いパターン**に紐づいて他デッキへ漏れているか。

ナミ（OP11-041・ドン再装填なし）のデッキにサトリ OP15-066 を移植した turn 1 を合成し、
「サトリで掘ってEND vs 無行動END」のターン末マージンと decide を測る。
ユーザ裁定（2026-08-11）: **ドン再装填のないリーダーでは出さない（TURN_END）が正解**＝
don!!-1 持ちの低コストは再装填が無ければ出す価値自体が低い。よって正しいネットは
ここで負のマージン・TURN_END を出すべき（エネル h1@2 では正のマージン・PLAY が正解）。

v49 実測の記録: gen13 は**逆順**（ナミ +0.12 で掘る / エネル +0.011 で掘らない）。
B腕注入は順序を直した（エネル +0.41 > ナミ +0.30）が絶対水準はカード特徴で漏れる。
E腕（同カード逆ラベルの対照ペア）は本プローブを通した（ナミ −0.47/−0.28・TURN_END 3/3）が
防御較正（m1@15）を壊しゲート FAIL＝50ペア規模では「エネル獲得・漏れ抑制・逆裁定」は
同時に買えない（v49 レポート§トリレンマ）。豊かな教師が入るまで本プローブは
**候補ネットの常設検査**（経済紐づけの有無の判定器）として使う。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/satori_transplant_probe.py \\
    --net /tmp/cand/value.npz,/tmp/cand/policy.npz --seeds 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import rl_encoder as E  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from dense_selfplay_gen import _make_fixed_matchup_game  # noqa: E402
from opcg_game import OPCGGame  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine  # noqa: E402
from dig_inject_gen import _desc, _apply_dialogs, _end_turn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECKS_JSON = os.path.join(REPO, "tests", "fixtures", "decks", "user_decks_20260728.json")
SWAP_OUT = "OP16-119"          # 4枚をサトリ×4へ置換（枚数維持・プローブ規約）
SATORI = "OP15-066"


def make_game(tmpdir="/tmp"):
    specs = json.load(open(DECKS_JSON))
    cards = dict(specs["nami"]["cards"])
    cards.pop(SWAP_OUT)
    cards[SATORI] = 4
    mod = {**specs, "nami_satori": {**specs["nami"], "label": "ナミ+サトリ(probe)",
                                    "cards": cards}}
    path = os.path.join(tmpdir, "user_decks_satori_probe.json")
    json.dump(mod, open(path, "w"))
    return _make_fixed_matchup_game(path, "nami_satori", "shanks")


def build_turn1(game, gs, db, seed):
    """seed 偶数＝nami_satori が p1。KEEP×2 で開始を通過し、サトリを手札に保証する。"""
    m = game.new_game(db, seed)
    for _ in range(6):
        cur = gs.current_player(m)
        if cur is None:
            break
        keep = next((v for v in gs.legal_actions(m)
                     if (cpu_ai._describe_move(m, v) or {}).get("action_type") == "KEEP_HAND"),
                    None)
        if keep is None:
            break
        m = gs.apply(m, keep, cur)
    name = gs.current_player(m)
    pl = m.p1 if m.p1.name == name else m.p2
    if pl.leader.master.card_id != "OP11-041":
        return None, None
    if not any(c.master.card_id == SATORI for c in pl.hand):
        k = next((j for j, c in enumerate(pl.deck) if c.master.card_id == SATORI), None)
        if k is None:
            return None, None
        pl.hand[0], pl.deck[k] = pl.deck[k], pl.hand[0]
    return m, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="", help="value.npz[,policy.npz]（空＝出荷既定）")
    ap.add_argument("--seeds", type=int, default=4, help="decide の試行数")
    ap.add_argument("--board-seeds", default="22,24", help="盤面生成 seed（学習に使った 10..20 は避ける）")
    args = ap.parse_args()

    db = _load_db()
    gs = OPCGGame()
    gr = OPCGGame(prune_futile=False)
    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0],
                            policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()
    game = make_game()

    out = []
    for seed in [int(s) for s in args.board_seeds.split(",") if s.strip()]:
        m0, name = build_turn1(game, gs, db, seed)
        if m0 is None:
            print(f"seed{seed}: ナミ手番でない（skip）")
            continue
        m2, _ = build_turn1(game, gs, db, seed)
        mpass = _end_turn(gs, m2, name)
        mt, _ = build_turn1(game, gs, db, seed)
        mv = next((v for v in gr.legal_actions(mt)
                   if _desc(mt, v).get("action_type") == "PLAY"
                   and _desc(mt, v).get("card") == SATORI), None)
        if mv is None or mpass is None:
            continue
        mp = gs.apply(mt, mv, name)
        md = _apply_dialogs(gs, mp, name) if mp is not None else None
        mend = _end_turn(gs, md, name) if md is not None else None
        if mend is None:
            continue

        def vv(m):
            enc = E.encode(m, name, eng.vocab, version=eng.enc_version)
            return float(eng.vnet.predict(
                {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")})[0])
        delta = vv(mend) - vv(mpass)
        got = collections.Counter()
        for s_ in range(args.seeds):
            m, nm2 = build_turn1(game, gs, db, seed)
            a = m.p1 if m.p1.name == nm2 else m.p2
            eng._world_seeds = {}
            mv2 = eng.decide(m, a, rng=np.random.default_rng(7100 + s_))
            dd = cpu_ai._describe_move(m, mv2) or {}
            got[f"{dd.get('action_type')} {dd.get('card') or ''}".strip()] += 1
        dig_rate = sum(v for k, v in got.items() if SATORI in k) / max(args.seeds, 1)
        ok = delta < 0 and dig_rate == 0
        out.append({"seed": seed, "delta": round(delta, 3), "dig_rate": dig_rate, "ok": ok})
        print(f"seed{seed}: Δ(掘り−無行動)={delta:+.3f}  decide={dict(got)}"
              f"  → {'OK（出さない）' if ok else 'NG（経済でなくカードに紐づいている）'}")
    n_ok = sum(1 for r in out if r["ok"])
    print(f"\n=== 裁定「ナミでは出さない」に整合: {n_ok}/{len(out)} 盤面")
    print("SATORI_TRANSPLANT " + json.dumps({"ok": n_ok, "n": len(out), "rows": out},
                                            ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
