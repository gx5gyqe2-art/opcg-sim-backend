"""出口専用ヘッドを持つ候補の**安価な足切り**（v41・2026-08-07）。

コーチゲートは1腕あたり10分以上かかるので腕のスイープに使えない。本プローブは MCTS を回さず、
2つの読み取り専用の測定だけを出す:

  (1) **出口順位**: 各検証点の合法手を `resolved_branch_values`（＝実対局の防御窓読み出しその
      もの）で1回だけ並べ、argmax が裁定済み accept 集合に入るか。物差しは候補の
      `LearnedEngine._battle_value_fn()`＝戦闘出口ヘッド。
  (2) **ヘッドのオフセット分布**: 教師コーパスの盤面で `predict_exit(kind) − predict()` の
      平均と標準偏差。平均が支配的＝ヘッドは**バイアス**を学んだ（全ての出口を一律に上下）、
      標準偏差が支配的＝**盤面ごとの差**を学んだ（＝狙いどおり、ただし正しいとは限らない）。

**なぜこの2つか**（v41 実測 2026-08-07）: gen12 の出口 value の**枝間マージンは 0.02〜0.03**
しかない（(1) の gap 列）。一方 defcf コーパス（584群775ペア）で学習したヘッドの摂動は
**標準偏差 0.23〜0.27**＝マージンの約10倍。順位を動かせてしまうが、それは較正ではなく
ノイズの当たり外れで、実際に m1@14 を直す腕は必ず m2@44 を壊した（5腕すべて 4/8）。
腕を増やす前にこの2列を見れば、コーパスの信号がマージンを超えているかが分かる。

**(1) の読み方の注意**: 検証点には戦闘窓でないもの（メインフェーズの判断＝m1@3/m4@2 など）も
含まれる。それらの行は「root の合法手を戦闘箱と同じ規約で並べたら」という**近似**で、
実対局の decide（木／プラン読み出し）とは経路が違う＝順位が一致しなくても即 NG ではない。
戦闘窓の点（m1@14/m1@15/m2@58）が一次情報。
**2026-08-08 現在 `turn_all` 形式の点は存在しない**（唯一の m2@66 は v46 で取り下げ）。以下は
機構の説明として残す。
`turn_all` 形式の点（ターン内で全ての攻撃と起動を消化することが条件）は**初手1手の
枝順位では原理的に判定できない**ので分母から外して `--` と表示する（コーチゲートの
`turn_all_rate` が正しい計器）。v44 まで `CG.hit` に dict を渡して**黙って常時不一致**に
数えており、gen13 の出口順位を 7/8 と過小に見せていた（腕どうしの比較は同じ偏りなので
無傷だが、絶対値の読み違いを招いた）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/exit_head_probe.py \\
    --nets "base;/tmp/cand_v41_a/value.npz,/tmp/cand_v41_a/policy.npz" \\
    --labels "gen12,v41_a" --offset-dirs /tmp/plancf_bal --offset-glob "defcf_*.npz"
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine, _priors_fn  # noqa: E402
from opcg_sim.src.learned.mcts import resolved_branch_values  # noqa: E402


def load_boards():
    """VERIFIED_V2 の各点を真盤面復元して (名前, manager, 手番, accept) の列にする。"""
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    out = []
    for tag, i, accept in CG.VERIFIED_V2:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            print(f"{tag}@{i}: 復元不可（スキップ）: {built}", flush=True)
            continue
        m0, who = built
        out.append((f"{tag}@{i}", m0, who if isinstance(who, str) else who.name, accept))
    return out


def load_corpus(dirs, pattern, limit_files=6):
    d = {k: [] for k in ("scalars", "field", "card_idx")}
    n = 0
    for dd in dirs:
        for f in sorted(glob.glob(os.path.join(dd, pattern)))[:limit_files]:
            z = np.load(f)
            for k in d:
                d[k].append(z[k])
            n += 1
    if not n:
        return None
    return {k: np.concatenate(v) for k, v in d.items()}


def _engine(spec):
    parts = spec.split(",")
    if parts[0] in ("", "base"):
        return LearnedEngine()
    return LearnedEngine(value_path=parts[0],
                         policy_path=parts[1] if len(parts) > 1 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", required=True,
                    help="'value.npz,policy.npz' を ';' 区切りで複数（'base'=出荷既定）")
    ap.add_argument("--labels", default="", help="表示名（カンマ区切り・--nets と同数）")
    ap.add_argument("--head", default="battle", help="オフセットを見る出口ヘッドの種別")
    ap.add_argument("--offset-dirs", default="", help="オフセット分布を測るコーパス（カンマ区切り）")
    ap.add_argument("--offset-glob", default="defcf_*.npz")
    args = ap.parse_args()

    boards = load_boards()
    corpus = load_corpus([d for d in args.offset_dirs.split(",") if d],
                         args.offset_glob) if args.offset_dirs else None

    specs = [s for s in args.nets.split(";") if s]
    labels = (args.labels.split(",") + [""] * len(specs))[:len(specs)]
    for spec, lab in zip(specs, labels):
        eng = _engine(spec)
        has = eng.vnet.has_exit_head(args.head)
        print(f"\n=== {lab or spec}（{args.head} ヘッド: {'あり' if has else 'なし'}）===",
              flush=True)
        bf = eng._battle_value_fn()
        pf = _priors_fn(eng.pnet, eng.vocab, eng.enc_version)
        hits = scored = 0
        for name, m0, actor, accept in boards:
            # turn_all 形式（{"turn_all": {...}}＝ターン内で全て実行する必要がある点・m2@66）は
            # **初手1手の枝順位では判定できない**ので分母から外す。`CG.hit` に dict を渡すと
            # キー文字列との照合になり黙って常時不一致になる（v44 で計器側の欠陥として発見）。
            if CG.turn_all_required(accept) is not None:
                print(f"  {name:<8} --  判定不能（turn_all 形式＝ターン全消化・"
                      f"分母から除外。コーチゲートの turn_all_rate で見る）", flush=True)
                continue
            legal = eng.game.legal_actions(m0)
            vals = resolved_branch_values(eng.game, m0, actor, legal, bf, pf)
            ok = [i for i, v in enumerate(vals) if v is not None]
            if not ok:
                print(f"  {name:<8} 評価不能", flush=True)
                continue
            order = sorted(ok, key=lambda i: -vals[i])
            best = order[0]
            d = cpu_ai._describe_move(m0, legal[best]) or {}
            good = CG.hit(d, accept)
            hits += bool(good)
            scored += 1
            gap = vals[best] - (vals[order[1]] if len(order) > 1 else vals[best])
            print(f"  {name:<8} {'OK ' if good else '   '} "
                  f"best={(d.get('action_type'), d.get('card'))} "
                  f"v={vals[best]:+.4f} gap={gap:+.4f}", flush=True)
        print(f"  → 出口順位一致 {hits}/{scored}"
              f"（判定可能な点のみ・全{len(boards)}点）", flush=True)
        if corpus is not None and has:
            off = eng.vnet.predict_exit(corpus, args.head) - eng.vnet.predict(corpus)
            m, s = float(off.mean()), float(off.std())
            kind = "バイアス寄り" if abs(m) > s else "盤面ごとの差寄り"
            print(f"  → ヘッドのオフセット（{len(off)}盤面）平均{m:+.4f} 標準偏差{s:.4f}"
                  f"（{kind}）", flush=True)
    print("EXIT_HEAD_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
