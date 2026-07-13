"""マーク回帰ゲート: 人間マークに対する before/after 判定 CLI（プロファイル制）。

`tests/fixtures/replays/` の frames 同梱リプレイの各マーク盤面を復元し、challenger / baseline
ネットで各 K シード decide して「人間指摘方向の手を選ぶ率」を比較する。判定規則:
  - **改善**: target マークのうち過半で challenger の指摘方向率 > baseline。
  - **非退行**: guard マークで challenger の率が baseline−0.2 を下回らない。
両方満たせば PASS（exit 0）。復元は replay_reeval の機構を流用（counter系マークも対応）。

**プロファイル**（`--profile`）:
  - `v4`（docs/cpu_v4_plan.md §4-3-1）: g1/g2 の16マーク・baseline=v3(gen3)。v4採用ゲート。
  - `v5`（docs/cpu_v5_plan.md §4-6・既定）: g3_v4 追加・baseline=v4(gen4)。
    **注意（実測に基づく設計）**: g3_v4 の14マークは、gen4 を復元盤面で decide すると 9/10 で
    既に人間希望方向を選ぶ（誤るのは @82 のみ）。実対局の誤りは「その対局の文脈（sticky世界線・
    実際に辿った手順）」で起きたもので、盤面を単独復元すると再現されない（盤面復元の構造的限界）。
    ⇒ v5 ゲートは **@82 の1点改善 ＋ 残り全マークの非退行**に徹する（改善余地ゼロのマークを
    改善ターゲットにしても無意味）。**v5 の主判定は arena（対v4/対L1）＋本走後の再プレイ再マーク**で、
    本ゲートは「gen4 の正着を壊さない」退行ガードが主目的。gen4 vs gen4 は @82 で改善0＝FAIL（感度基準線）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_gate.py \
    --challenger /tmp/v5_value.npz,/tmp/v5_policy.npz --seeds 5           # 既定 profile=v5
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
    "g3": os.path.join(_FIX, "g3_v4_replay_7943918224969915818.json.gz"),
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


def _not_attack_with(card):
    """『そのカードでの攻撃をしない』（無駄攻撃マーク用・攻撃自体は禁じない）。"""
    return lambda d: not (d.get("action_type") == "ATTACK" and d.get("card") == card)


# --- v4 プロファイル（v4採用ゲート・baseline=v3 gen3・docs/cpu_v4_plan.md §4-3-1）---
# target=F4代表（改善を要求）・guard=既存正着（非退行を要求）。
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

# --- v5 プロファイル（v5評価ゲート・baseline=v4 gen4・docs/cpu_v5_plan.md §4-6）---
# **実測に基づく分類**（gen4 を復元盤面で各6シード decide した率・2026-07-13）:
# g3_v4 の14マークのうち、gen4 が復元盤面で人間希望方向を**既に選ぶ**（率≈1.0）のが 9/10、
# 誤るのは @82（無駄カウンター）のみ。実対局の誤りは「その対局の文脈＝sticky世界線/実際に辿った
# 手順」で起きたもので、盤面を単独復元すると再現されない（盤面復元の構造的限界＝replay_reeval と同じ）。
# ⇒ mark_gate は g3_v4 に対し **@82 の1点改善 ＋ 残りの非退行ガード**に徹する（改善余地ゼロの
# マークを改善ターゲットにしても無意味なため）。v5 の主判定は arena（対v4/対L1）＋本走後の再プレイ再マーク。
V5_TARGETS = {
    ("g3", 82): ("無駄なカウンターを切らない", _not("SELECT_COUNTER")),   # gen4 実測率 0.0＝改善余地
}
V5_GUARDS = {
    # g3_v4: gen4 が復元盤面で正着（実測率≈1.0）＝v5 は非退行を要求。
    ("g3", 33): ("6000→6000攻撃を維持", _is("ATTACK")),
    ("g3", 43): ("ルフィで攻撃する", _is("ATTACK")),
    ("g3", 64): ("付与でブースト（7000→7000）", _is("ATTACH_DON")),
    ("g3", 68): ("付与でブースト", _is("ATTACH_DON")),
    ("g3", 93): ("無駄なドン付与をしない", _not("ATTACH_DON")),
    ("g3", 115): ("無意味な守りをしない", _not("SELECT_COUNTER")),
    ("g3", 19): ("OP11-106で無駄攻撃しない", _not_attack_with("OP11-106")),
    ("g3", 72): ("OP15-119で無駄攻撃しない", _not_attack_with("OP15-119")),
    ("g3", 102): ("OP11-106で無駄攻撃しない", _not_attack_with("OP11-106")),
    # v4 期に獲得した守り（gen4 が持つ）を維持＝v5 で忘却しないことのガード。
    ("g1", 63): ("カウンターで守る（v4獲得）", _is("SELECT_COUNTER")),
    ("g2", 32): ("カウンターで守る（v4獲得）", _is("SELECT_COUNTER")),
    ("g1", 12): ("ドン付与を優先", _is("ATTACH_DON")),
    ("g1", 24): ("波(OP15-105)を絡める", _involves("OP15-105")),
    ("g2", 20): ("ボンクレー登場", _is("PLAY", "OP16-056")),
}

PROFILES = {
    "v4": (TARGETS, GUARDS, "opcg_sim/data/learned/gen3_value.npz,opcg_sim/data/learned/gen3_policy.npz"),
    "v5": (V5_TARGETS, V5_GUARDS, "opcg_sim/data/learned/gen4_value.npz,opcg_sim/data/learned/gen4_policy.npz"),
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
    ap.add_argument("--profile", choices=("v4", "v5"), default="v5",
                    help="v4=旧ゲート(g1/g2・baseline gen3)／v5=現ゲート(g3_v4追加・baseline gen4・§4-6)")
    ap.add_argument("--baseline", default=None,
                    help="value.npz[,policy.npz]（未指定はプロファイル既定＝v5:gen4 / v4:gen3）")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sims", type=int, default=160)
    args = ap.parse_args()

    targets, guards, default_base = PROFILES[args.profile]
    baseline = args.baseline or default_base

    def engine(spec):
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)

    db = _load_db()
    chall, base = engine(args.challenger), engine(baseline)
    used_tags = {tag for tag, _ in list(targets) + list(guards)}
    games = {}
    for tag, path in REPLAYS.items():
        if tag not in used_tags:
            continue
        raw = RE.load_replay_json(path)
        rec = raw.get("replay", raw)
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

    print(f"=== マーク回帰ゲート profile={args.profile}（baseline={baseline.split('/')[-1]} "
          f"seeds={args.seeds} sims={args.sims}）===", flush=True)
    t_rows = run(targets, "改善対象")
    g_rows = run(guards, "ガード")

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
