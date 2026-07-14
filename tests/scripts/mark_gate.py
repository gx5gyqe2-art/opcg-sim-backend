"""v4 マーク回帰ゲート（docs/cpu_v4_plan.md §4-3-1）: 人間マークに対する before/after 判定 CLI。

`tests/fixtures/replays/` の frames 同梱リプレイ2局（16人間マーク）の各マーク盤面を復元し、
challenger / baseline（既定=現本番 v3）ネットで各 K シード decide して「人間指摘方向の手を
選ぶ率」を比較する。判定規則:
  - **改善**: F4代表マーク（net起因＝g1@16/@43/@63・g2@21/@32/@38）のうち過半で
    challenger の指摘方向率 > baseline。
  - **非退行**: 既存の正着ガード（g1@12/@24・g2@20）で challenger の率が baseline−0.2 を下回らない。
両方満たせば PASS（exit 0）。復元は replay_reeval の機構を流用（counter系マークも対応）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_gate.py \
    --challenger /tmp/v4_value.npz,/tmp/v4_policy.npz --seeds 5
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine

_FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "fixtures", "replays")
REPLAYS = {
    "g1": os.path.join(_FIX, "g1_replay_8410492561010605030.json.gz"),
    "g2": os.path.join(_FIX, "g2_replay_2635670571334674537.json.gz"),
}

# 人間指摘方向の述語（described move -> bool）。docs/cpu_v4_plan.md §4-3-1 の代表/ガード。
def _is(at, card=None):
    def pred(d):
        if d.get("action_type") != at:
            return False
        return card is None or d.get("card") == card
    return pred


def _not(at):
    return lambda d: d.get("action_type") != at


def _involves(card):
    return lambda d: d.get("card") == card


# (game, action_index) -> (説明, 述語)。target=F4代表（改善を要求）・guard=既存正着（非退行を要求）。
TARGETS = {
    ("g1", 16): ("無駄ドン付与をしない", _not("ATTACH_DON")),
    ("g1", 43): ("リーダーにドン付与", _is("ATTACH_DON", "OP11-041")),
    ("g1", 63): ("カウンターで守る", _is("SELECT_COUNTER")),
    ("g2", 21): ("ドン付与連打をしない", _not("ATTACH_DON")),
    ("g2", 32): ("カウンターで守る", _is("SELECT_COUNTER")),
    ("g2", 38): ("防御ドンをリーダーへ", _is("ATTACH_DON", "OP11-041")),
}
GUARDS = {
    ("g1", 12): ("ドン付与を優先（素の攻撃でない）", _is("ATTACH_DON")),
    ("g1", 24): ("波(OP15-105)を絡める", _involves("OP15-105")),
    ("g2", 20): ("ボンクレー登場", _is("PLAY", "OP16-056")),
}
GUARD_TOLERANCE = 0.2


def _restore(db, rec, frames_by_idx, actions, i):
    """マーク i の盤面を復元して (manager, actor) を返す（失敗は理由文字列）。"""
    at = actions[i].get("action_type")
    if at in ("SELECT_COUNTER", "SELECT_BLOCKER", "PASS"):
        built = RE._board_for_counter_mark(db, rec, frames_by_idx, actions, i)
        return built
    pre = frames_by_idx.get(i - 1)
    if pre is None:
        return "直前フレーム欠落"
    m = RE._board_from_frame(db, rec, pre, actions[i]["player"])
    return m, m.turn_player


def _rate(db, rec, frames_by_idx, actions, i, eng, pred, seeds, sims):
    """マーク i を seeds 回 decide し、述語を満たした率を返す（復元失敗は None）。"""
    hit = 0
    for w in range(seeds):
        built = _restore(db, rec, frames_by_idx, actions, i)
        if isinstance(built, str):
            return None
        m, actor = built
        eng._world_seeds = {}
        mv = eng.decide(m, actor, sims=sims, rng=np.random.default_rng(9000 + 97 * i + w))
        try:
            d = cpu_ai._describe_move(m, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        hit += 1 if pred(d) else 0
    return hit / seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--baseline",
                    default="opcg_sim/data/learned/gen3_value.npz,opcg_sim/data/learned/gen3_policy.npz")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sims", type=int, default=160)
    args = ap.parse_args()

    def engine(spec):
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)

    db = _load_db()
    chall, base = engine(args.challenger), engine(args.baseline)
    games = {}
    for tag, path in REPLAYS.items():
        raw = RE.load_replay_json(path)
        rec = raw["replay"]
        games[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])

    def run(table, label):
        rows = []
        for (tag, i), (desc, pred) in sorted(table.items()):
            rec, fbi, actions = games[tag]
            rb = _rate(db, rec, fbi, actions, i, base, pred, args.seeds, args.sims)
            rc = _rate(db, rec, fbi, actions, i, chall, pred, args.seeds, args.sims)
            rows.append(((tag, i), desc, rb, rc))
            fmt = (lambda r: "復元不可" if r is None else f"{r:.2f}")
            print(f"  [{label}] {tag}@{i:<3} {desc:<22} base={fmt(rb)} chall={fmt(rc)}", flush=True)
        return rows

    print(f"=== マーク回帰ゲート（seeds={args.seeds} sims={args.sims}）===", flush=True)
    t_rows = run(TARGETS, "改善対象")
    g_rows = run(GUARDS, "ガード")

    evals = [(rb, rc) for _, _, rb, rc in t_rows if rb is not None and rc is not None]
    improved = sum(1 for rb, rc in evals if rc > rb)
    need = math.floor(len(evals) / 2) + 1
    ok_improve = improved >= need
    regressed = [(k, rb, rc) for k, _, rb, rc in g_rows
                 if rb is not None and rc is not None and rc < rb - GUARD_TOLERANCE]
    ok_guard = not regressed

    print(f"\n改善: {improved}/{len(evals)}（必要 {need}）→ {'OK' if ok_improve else 'NG'}")
    print(f"非退行: {'OK' if ok_guard else 'NG ' + str(regressed)}")
    verdict = ok_improve and ok_guard
    print(f"判定: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
