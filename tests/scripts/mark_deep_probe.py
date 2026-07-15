"""マーク深探索プローブ: 各マークが「深い探索で解けるか」を分類する計器 CLI（v6 柱③・読み取り専用）。

`docs/reports/v5_adoption_20260715.md` §4-3 の前提チェック。v6 柱②（深い探索による部分再ラベル＝
expert iteration）は「生成時 sims=160 では見えない差が、深い探索の policy ターゲットには現れる」ことを
前提にする。本計器はマーク盤面（mark_gate と同じ復元・述語）を **現既定ネットのまま sims だけ深く**
decide し、人間指摘方向率が sims と共に立ち上がるかを見る:

  - **EXPLORABLE**: 浅い sims で誤り、深い sims で指摘方向を選ぶ＝探索の浅さが原因。
    柱②（深探索再ラベル）の守備範囲＝再ラベルで policy に写せば直る見込み。
  - **VALUE_BOUND**: 深くしても選ばない＝value/表現の問題（いくら読んでも評価が差を感じない）。
    柱④（失敗モード逆算の特徴設計）行き。再ラベルでは直らない。
  - **OK@160**: 現既定が浅い sims で既に指摘方向＝プローブ対象外（参考表示のみ）。

判定は既定で「160 で率<0.5 のマーク」だけ深掘りし（コスト抑制）、
EXPLORABLE = 最深 sims の率 ≥ 0.67 かつ 160 比 +0.3 以上。分類は次サイクルの実装先の
振り分けが目的なので、しきい値は感度より説明可能性を優先した固定値。

誤るマークには**確定用の独立証拠2列**を追加測定する（いずれも読み取り専用・既定ON）:
  - **flat**（prior平坦化・`--no-flat-prior` で省略）: policy prior を一様にして最深 sims を再測定。
    PUCT の深部は prior に誘導されるため、「value が差を感じない」と「policy が正着を読ませない」を
    分離する。flat で立ち上がれば **PRIOR_BOUND**（policy 起因＝深探索再ラベルの射程に戻る）、
    立ち上がらなければ VALUE_BOUND が確定に近づく。
  - **L1**（第二意見オラクル・`--no-l1` で省略）: 同じ復元盤面を製品 L1（α-β・手作り評価）で decide。
    L1 が人間指摘方向を選ぶ＝「手作り評価は差を感じる」＝学習 value の表現/品質問題という
    独立証拠（選ばなければ、そもそも評価困難な局面の可能性）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_deep_probe.py            # 既定 gen5
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_deep_probe.py \
    --net /tmp/v.npz,/tmp/p.npz --sims-levels 160,800,3200 --seeds 6,4,3 --all
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import random
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import mark_gate as MG
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.core.cpu_learned import LearnedEngine

FAIL_AT_BASE = 0.5     # 基準 sims でこの率未満＝「誤るマーク」として深掘り対象
DEEP_PASS = 2 / 3      # 最深 sims でこの率以上（かつ +0.3 改善）＝ EXPLORABLE。seeds=3 の 2/3 を通す有理数
DEEP_GAIN = 0.3


_EPS = 1e-9   # 率は seeds 分の整数比（2/3=0.666…）＝しきい値比較は許容誤差付き（浮動小数の際で落とさない）


def classify(rates, flat=None):
    """sims昇順の率リスト（＋prior平坦化の最深率）→ 分類名。復元不可(None)は 'UNRESTORABLE'。"""
    if any(r is None for r in rates):
        return "UNRESTORABLE"
    if rates[0] >= FAIL_AT_BASE:
        return "OK@base"
    if rates[-1] >= DEEP_PASS - _EPS and rates[-1] - rates[0] >= DEEP_GAIN:
        return "EXPLORABLE"
    if flat is not None and flat >= DEEP_PASS - _EPS:
        return "PRIOR_BOUND"   # 一様priorの深探索なら解ける＝policyが正着を読ませていない
    return "VALUE_BOUND"


@contextlib.contextmanager
def _flat_prior():
    """policy prior を一様化（TreeMCTS は priors_fn=None で一様＝製品コード無改変の診断パッチ）。"""
    orig = CL._priors_fn
    CL._priors_fn = lambda pnet, vocab, enc_version=1: None
    try:
        yield
    finally:
        CL._priors_fn = orig


def _l1_rate(db, rec, fbi, actions, i, pred, seeds):
    """同じ復元盤面を製品 L1（hard・α-β）で decide した指摘方向率（第二意見オラクル）。"""
    hit = 0
    for w in range(seeds):
        built = MG._restore(db, rec, fbi, actions, i)
        if isinstance(built, str):
            return None
        m, actor = built
        mv = cpu_ai.decide(m, actor, "hard", random.Random(9000 + 97 * i + w))
        try:
            d = cpu_ai._describe_move(m, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        hit += 1 if pred(d) else 0
    return hit / seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="value.npz[,policy.npz]（未指定＝出荷既定 gen5）")
    ap.add_argument("--profile", choices=("v4", "v5"), default="v5",
                    help="プローブ対象マーク集合（mark_gate のプロファイルを流用・target+guard 全件）")
    ap.add_argument("--sims-levels", default="160,800,3200")
    ap.add_argument("--seeds", default="6,4,3", help="sims 水準ごとの decide 回数（levels と同数）")
    ap.add_argument("--all", action="store_true",
                    help="基準 sims で正着のマークも深掘りする（既定は誤るマークのみ＝コスト抑制）")
    ap.add_argument("--no-flat-prior", action="store_true",
                    help="prior平坦化の再測定（policy起因の分離）を省略する")
    ap.add_argument("--no-l1", action="store_true",
                    help="L1 第二意見オラクル（独立評価の証拠列）を省略する")
    args = ap.parse_args()

    levels = [int(x) for x in args.sims_levels.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    assert len(levels) == len(seeds) >= 2 and levels == sorted(levels), "levels/seeds 不整合"

    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)
        net_label = parts[0].split("/")[-1]
    else:
        eng = LearnedEngine()
        net_label = "gen5(既定)"

    targets, guards, _ = MG.PROFILES[args.profile]
    marks = {**targets, **guards}
    db = _load_db()
    used_tags = {tag for tag, _ in marks}
    games = {}
    for tag, path in MG.REPLAYS.items():
        if tag not in used_tags:
            continue
        raw = RE.load_replay_json(path)
        rec = raw.get("replay", raw)
        games[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])

    print(f"=== マーク深探索プローブ net={net_label} profile={args.profile} "
          f"sims={levels} seeds={seeds} ===", flush=True)
    results = []
    t0 = time.time()
    fmt = lambda r: "  ---" if r is None else f"{r:5.2f}"
    for (tag, i), (desc, pred) in sorted(marks.items()):
        rec, fbi, actions = games[tag]
        rates = [MG._rate(db, rec, fbi, actions, i, eng, pred, seeds[0], levels[0])]
        flat = l1 = None
        failing = rates[0] is not None and rates[0] < FAIL_AT_BASE
        if rates[0] is not None and (args.all or failing):
            for lv, sd in zip(levels[1:], seeds[1:]):
                rates.append(MG._rate(db, rec, fbi, actions, i, eng, pred, sd, lv))
        else:
            rates += [None] * (len(levels) - 1)   # 深掘りスキップ（OK@base / 復元不可）
        if failing:
            if not args.no_flat_prior:
                with _flat_prior():
                    flat = MG._rate(db, rec, fbi, actions, i, eng, pred, seeds[-1], levels[-1])
            if not args.no_l1:
                l1 = _l1_rate(db, rec, fbi, actions, i, pred, seeds[0])
        cls = ("OK@base" if rates[0] is not None and rates[0] >= FAIL_AT_BASE
               else classify(rates, flat))
        results.append(((tag, i), desc, rates, cls))
        extra = ""
        if flat is not None:
            extra += f" flat{levels[-1]}={fmt(flat)}"
        if l1 is not None:
            extra += f" L1={fmt(l1)}"
        print(f"  {tag}@{i:<3} {desc:<22} " +
              " ".join(f"s{lv}={fmt(r)}" for lv, r in zip(levels, rates)) +
              extra + f"  → {cls} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== 分類サマリ（v6 実装先の振り分け）===")
    for cls, dest in (("EXPLORABLE", "柱②深探索再ラベルの守備範囲"),
                      ("PRIOR_BOUND", "policy起因（一様priorの深探索なら解ける）＝柱②の射程・再ラベルはprior平坦化つきで"),
                      ("VALUE_BOUND", "柱④特徴設計行き（再ラベルでは直らない）"),
                      ("OK@base", "現既定で正着（対象外）"),
                      ("UNRESTORABLE", "盤面復元不可")):
        ks = [f"{t}@{i}" for (t, i), _, _, c in results if c == cls]
        if ks:
            print(f"  {cls:<13} {len(ks)}件: {', '.join(ks)}  ＝ {dest}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
