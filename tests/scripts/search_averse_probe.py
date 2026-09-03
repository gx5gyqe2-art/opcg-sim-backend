"""SEARCH_AVERSE 点の追跡プローブ（v21・読み取り専用）: 探索のどこで正着から離れるかを特定する。

背景（`docs/reports/cpu_v20_prior_value_20260729.md`）: VERIFIED v2 の3点（m2@44/m4@12/m5@7）は
**prior が正着を1位に置き、1手先 value も正の差を付けている**のに、sims を10倍にしても選ばない。
prior でも 1手先 value でもない第3の機序＝探索/読み出し側の問題として分離された。

**純正AZ化（2026-08-25）後の縮小版**: 旧アブレーション A（root LCB 乗り換え）・B（終局深さ
減衰）・C（aux 粘り項）の対象機構は補償層として削除済み＝base 自体がそれら無しの計算に
なった。残る容疑者は:

  D. **単一世界の PIMC**（`TreeMCTS.run` は決定化を**1回**だけ行う＝1決定=1世界）— 隠れ情報の
     引き方1つで root Q が偏っていないか。レフェリーは 8〜16 世界で裁いており、探索だけが
     1世界で決めている構造的な非対称がある

測り方（すべて製品コード無改変・診断側で TreeMCTS を直接駆動して root 統計を読む）:
  1. **Q/N トレース**: sims 昇順に、accept グループと訪問最多グループの (N 割合, Q) を並べる。
  2. **アブレーション**: D（多世界平均）で accept 率を再測定する（同一 seed）。
     **base を上回って 0.5 以上**になった要素が犯人（≥0.5 だけでは base が元々高い点を誤診する）。
  3. **世界依存フラグ**: 0 < base < 1 ＝ seed（隠れ情報の引き）で結論が割れる点。こういう点の
     単発 0.00/1.00 を「その点の実力」と読まないための注記。

`prior_bound_probe.py`（v20）が「prior/value/探索」の三択を決める計器なのに対し、本プローブは
**探索と決まった後の内訳**を見る（1トピック=1ファイル）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/search_averse_probe.py --out /tmp/sa.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random as _random
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import coach_gate as CG
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.mcts import TreeMCTS

# v20 で SEARCH_AVERSE と分類された点（`docs/reports/cpu_v20_prior_value_20260729.md` §2-3）。
DEFAULT_POINTS = ("m2@44", "m4@12", "m5@7")


def world_sensitive(base_rate):
    """同一局面・同一 sims でも**隠れ情報の引き（決定化世界）で結論が割れる**か（pure）。

    0 < base < 1 ＝ seed（世界）によって accept を選んだり選ばなかったりする。この状態の点は
    5 seed 程度の命中率（コーチゲートの読み方）では分散が大きく、単発の 0.00/1.00 を
    「その点の実力」と読んではいけない。"""
    return base_rate is not None and 0.0 < base_rate < 1.0


def diagnose(trace, abl):
    """Q/Nトレースとアブレーション結果 → 犯人の見立て（pure）。

    `trace`: [{"sims":s, "acc_n":割合, "acc_q":Q, "top_n":割合, "top_q":Q, "readout_ok":bool}, ...]
    `abl`  : {"base": 率, "multiworld": 率}。
    **アブレーションは base を上回って初めて『それが原因』**（≥0.5 だけでは base が元々高い点を
    誤診する）。どれも効かず探索 Q が全深さで accept を下に見るなら、探索の評価そのものが根本。"""
    base = abl.get("base")
    if base is None:
        return "UNRESOLVED"
    if base >= 0.5:
        return "NOT_FAILING@deep"          # この sims/世界では失敗していない（world_sensitive を併記）
    def fixed(k):
        v = abl.get(k)
        return v is not None and v >= 0.5 and v > base
    for key, name in (("multiworld", "PIMC_WORLD_BOUND"),):
        if fixed(key):
            return name
    if trace and all(t["acc_q"] is not None and t["acc_q"] <= t["top_q"] for t in trace):
        return "SEARCH_Q_BOUND"            # どの深さでも探索 Q が accept を下に見る＝評価の積み上げ
    return "UNRESOLVED"


def _root_stats(eng, m0, name, sims, seed):
    """TreeMCTS を1回回して root 統計（legal/N/Q）を返す（製品と同じ構成）。"""
    rng = np.random.default_rng(seed)
    mcts = TreeMCTS(eng.game,
                    value_fn=CL._value_fn(eng.vnet, eng.vocab, eng.enc_version),
                    priors_fn=CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version),
                    c_puct=CL.C_PUCT, n_sims=sims, dirichlet_eps=0.0,
                    determinize_fn=lambda s, r: eng.game.determinize(s, name, r), rng=rng)
    mcts.run(m0)
    return getattr(mcts, "last_stats", None) or {}


def _groups_with_accept(m0, stats, accept):
    """root 統計 → (グループ列, accept を含むグループの index 集合)。`_merge_root_stats` と同じ併合。"""
    groups = CL._merge_root_stats(m0, stats["legal"], stats["N"], stats["Q"])
    acc = set()
    for gi, g in enumerate(groups):
        for j in g["idxs"]:
            try:
                d = cpu_ai._describe_move(m0, stats["legal"][j]) or {}
            except Exception:
                d = {}
            if CG.hit(d, accept):
                acc.add(gi)
                break
    return groups, acc


def _rate_with(eng, m0, actor, accept, seeds, sims):
    """指定条件で decide 相当（探索＋製品読み出し＝合算 argmax(N)）を seeds 回まわし accept 率を返す。"""
    name = actor if isinstance(actor, str) else actor.name
    hit = 0
    for s in range(seeds):
        stats = _root_stats(eng, m0, name, sims, 9100 + 97 * s)
        if not stats.get("legal"):
            continue
        groups, acc = _groups_with_accept(m0, stats, accept)
        hit += 1 if 0 in acc else 0     # 製品読み出し＝合算後の先頭グループ（argmax N）
    return hit / max(seeds, 1)


def multiworld_rate(eng, m0, actor, accept, worlds, sims, seed0=9100):
    """W 世界の root 統計を**グループ単位で N 加重平均**してから読み出す（レフェリーと同じ扱い）。

    `TreeMCTS.run` は決定化を1回だけ行う＝1決定=1世界。隠れ情報の引き1つで root Q が偏りうるので、
    世界を跨いで平均した場合に正着へ戻るかを見る。返り値は accept を選べば 1.0・でなければ 0.0。"""
    name = actor if isinstance(actor, str) else actor.name
    agg, acc_keys = {}, set()
    for w in range(worlds):
        stats = _root_stats(eng, m0, name, sims, seed0 + 97 * w)
        if not stats.get("legal"):
            continue
        groups, acc = _groups_with_accept(m0, stats, accept)
        for gi, g in enumerate(groups):
            try:
                key = repr(cpu_ai._move_equiv_key(m0, stats["legal"][g["rep"]]))
            except Exception:
                key = f"g{gi}"
            a = agg.setdefault(key, {"n": 0.0, "wq": 0.0})
            a["n"] += g["n"]; a["wq"] += g["n"] * g["q"]
            if gi in acc:
                acc_keys.add(key)
    if not agg:
        return 0.0
    merged = [{"rep": k, "idxs": [], "n": v["n"], "q": (v["wq"] / v["n"] if v["n"] else 0.0)}
              for k, v in agg.items()]
    merged.sort(key=lambda g: -g["n"])
    return 1.0 if merged[0]["rep"] in acc_keys else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=None, help="value.npz[,policy.npz]（未指定＝出荷既定＝現 gen8）")
    ap.add_argument("--points", default=",".join(DEFAULT_POINTS),
                    help="tag@index のカンマ区切り（既定＝v20 の SEARCH_AVERSE 3点）")
    ap.add_argument("--sims-levels", default="40,160,640,1600")
    ap.add_argument("--seeds", type=int, default=3, help="アブレーションの decide 回数")
    ap.add_argument("--worlds", type=int, default=8, help="多世界平均アブレーションの世界数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.sims_levels.split(",")]
    want = {p.strip() for p in args.points.split(",") if p.strip()}
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    if args.net:
        parts = args.net.split(",")
        eng = LearnedEngine(value_path=parts[0], policy_path=parts[1] if len(parts) > 1 else None)
        label = parts[0].split("/")[-1]
    else:
        eng = LearnedEngine()
        label = "既定(gen8)"

    replays = {**__import__("mark_gate").REPLAYS, **CG.REPLAYS_V2}
    CR.GAMES = {}
    out, t0 = [], time.time()
    print(f"=== SEARCH_AVERSE 追跡 net={label} sims={levels} ===", flush=True)
    for tag, i, accept in CG.VERIFIED_V2:
        if f"{tag}@{i}" not in want:
            continue
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            print(f"  {tag}@{i}: 復元不可 {built}", flush=True)
            continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        print(f"\n--- {tag}@{i}  accept={sorted(accept)}", flush=True)

        trace = []
        for sims in levels:
            stats = _root_stats(eng, m0, name, sims, 9100)
            if not stats.get("legal"):
                continue
            groups, acc = _groups_with_accept(m0, stats, accept)
            tot = sum(g["n"] for g in groups) or 1
            top = groups[0]
            best_acc = max((groups[gi] for gi in acc), key=lambda g: g["q"], default=None)
            row = {"sims": sims,
                   "acc_n": (best_acc["n"] / tot) if best_acc else None,
                   "acc_q": best_acc["q"] if best_acc else None,
                   "top_n": top["n"] / tot, "top_q": top["q"],
                   "readout_ok": 0 in acc,
                   "argmaxN_ok": 0 in acc}
            trace.append(row)
            f = lambda x: " ----" if x is None else f"{x:+5.2f}"
            print(f"  s{sims:<5} accept: N={0.0 if row['acc_n'] is None else row['acc_n']:.2f} "
                  f"Q={f(row['acc_q'])}   最多: N={row['top_n']:.2f} Q={f(row['top_q'])}   "
                  f"読み出し={'accept' if row['readout_ok'] else '×'} "
                  f"argmaxN={'accept' if row['argmaxN_ok'] else '×'}", flush=True)

        abl = {}
        deep = levels[-1]
        abl["base"] = _rate_with(eng, m0, actor, accept, args.seeds, deep)
        abl["multiworld"] = multiworld_rate(eng, m0, actor, accept, args.worlds, deep)
        dg = diagnose(trace, abl)
        ws = world_sensitive(abl.get("base"))
        print(f"  アブレーション(s{deep}): base={abl['base']:.2f} "
              f"多世界({args.worlds})={abl['multiworld']:.2f} → {dg}"
              f"{'＋世界依存' if ws else ''} "
              f"({time.time() - t0:.0f}s)", flush=True)
        out.append({"tag": tag, "i": i, "trace": trace, "abl": abl,
                    "diagnosis": dg, "world_sensitive": ws})

    print("\n=== 診断サマリ ===")
    for r in out:
        print(f"  {r['tag']}@{r['i']:<4} → {r['diagnosis']}"
              f"{'（世界依存＝seed で結論が割れる）' if r['world_sensitive'] else ''}")
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"SEARCH_AVERSE_RESULT {json.dumps({r['tag'] + '@' + str(r['i']): r['diagnosis'] for r in out}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
