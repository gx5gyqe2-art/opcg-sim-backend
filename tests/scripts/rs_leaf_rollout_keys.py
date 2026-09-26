"""WP `rs-leaf-rollout`（§20.7.9）の計測を `RESULT.json` の形へ集計する（判定は書かない）。

使い方: `OPCG_LOG_SILENT=1 python tests/scripts/rs_leaf_rollout_keys.py scenario_out_f`

keys は分析 #6 の表と同じ 4 つ:
  kami_before_attack … enel_human_20260810_t9-9 で「オームの最初の ATTACK より前に神の裁きの PLAY」
  lethal             … 同シナリオで seat（p1）が決着させた run の本数
  gammaknife         … human_enel_vs_roger_20260904_t7-8 で seat がガンマナイフを PLAY
  kami_ko            … human_enel_vs_luffy_20260904_t4-5 で神の裁きの KO 対象の選択が非空
rollout は全 run の全決定の trace["rollout"] を束ねた実績。
"""
import glob
import json
import os
import sys

sys.path.insert(0, "tests/scripts")
sys.path.insert(0, "tests")
import rs_search_a_keys as K  # noqa: E402

SCN = {
    "kami_before_attack": (K.SCN_ENEL_T9, K._kami_before_oom_attack),
    "gammaknife": (K.SCN_ENEL_ROGER, lambda r: K._played(r, K.GAMMA)),
    "kami_ko": (K.SCN_ENEL_LUFFY, K._kami_ko_target),
}


def _runs(cond_dir, scn):
    for path in sorted(glob.glob(os.path.join(cond_dir, scn, "r3_s*.frames.json"))):
        seed = os.path.basename(path).split(".")[0].split("_s")[-1]
        with open(path, encoding="utf-8") as f:
            yield f"s{seed}", json.load(f)


def _lethal(res):
    """seat（分岐点の手番）が決着させたか（最後のフレームの winner が seat）。"""
    ds = res.get("decisions") or []
    seat = ds[0]["player"] if ds else None
    frames = res.get("frames") or []
    return (frames[-1] if frames else {}).get("winner") == seat


def cond_row(cond_dir):
    detail, counts = {}, {}
    for key, (scn, fn) in SCN.items():
        per = {s: fn(r) for s, r in _runs(cond_dir, scn)}
        detail[key] = {"scenario": scn, "per_seed": per}
        counts[key] = sum(1 for v in per.values() if v is True)
    per = {s: _lethal(r) for s, r in _runs(cond_dir, K.SCN_ENEL_T9)}
    detail["lethal"] = {"scenario": K.SCN_ENEL_T9, "per_seed": per}
    counts["lethal"] = sum(1 for v in per.values() if v)
    # 神の裁きを打った run の本数（KO の分母・分析 #6 の「打った」）。
    played = {s: (v != "no_kami") for s, v in detail["kami_ko"]["per_seed"].items()}
    detail["kami_ko"]["played"] = sum(1 for v in played.values() if v)
    counts["runs"] = len(per)
    return counts, detail


def rollout_row(cond_dir):
    leaves = rolled = plies = capped = 0
    decisions = with_rollout = 0
    for scn in {v[0] for v in SCN.values()}:
        for _, res in _runs(cond_dir, scn):
            for d in res.get("decisions") or []:
                decisions += 1
                r = d.get("rollout")
                if not r:
                    continue
                with_rollout += 1
                leaves += r["leaves"]
                rolled += r["rolled"]
                plies += r["plies"]
                capped += r["capped"]
    rnd = lambda x: round(x, 4)  # noqa: E731
    return {
        "decisions": decisions, "decisions_with_rollout": with_rollout,
        "leaves": leaves, "rolled": rolled, "plies": plies, "capped": capped,
        "mean_plies": rnd(plies / leaves) if leaves else None,
        "mean_plies_when_rolled": rnd(plies / rolled) if rolled else None,
        "rolled_frac": rnd(rolled / leaves) if leaves else None,
        "capped_frac": rnd(capped / leaves) if leaves else None,
    }


def main(root):
    out = {"keys": {}, "keys_detail": {}, "rollout": {}}
    for cond_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(cond_dir):
            continue
        cond = os.path.basename(cond_dir)
        counts, detail = cond_row(cond_dir)
        out["keys"][cond] = counts
        out["keys_detail"][cond] = detail
        out["rollout"][cond] = rollout_row(cond_dir)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scenario_out_f")
