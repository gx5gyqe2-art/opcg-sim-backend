"""Copy measured values into tests/fixtures/harm_profile.json, preserving key order and formatting (indent=1)."""
import json, sys
FX = "tests/fixtures/harm_profile.json"; R = "table_rebuild/results/"
def load(p): return json.load(open(R + p))["summary"]
def contour(prefix):
    d = json.load(open(FX)); old = {}
    for t in ("real", "syn"):
        hp = load(f"{prefix}_cb_{t}.json")["harm_profile"]
        assert len(hp["harm_by_turn"]) == 13 and len(hp["theory_slope_by_turn"]) == 13
        old[t] = (d[t], d["theory_slope"][t])
        d[t] = hp["harm_by_turn"]; d["theory_slope"][t] = hp["theory_slope_by_turn"]
    json.dump(d, open(FX, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(old, ensure_ascii=False))
def sigma(prefix):
    d = json.load(open(FX)); b = {t: load(f"{prefix}_cb_{t}.json")["by_slope"] for t in ("real", "syn")}
    for t in ("real", "syn"):
        d["sigma_t"]["blockers"][t] = b[t]["curve"]["sigma_T"]
        for sv in ("theory", "curve"):
            d["sigma_rel"]["blockers"][sv][t] = b[t][sv]["sigma_rel"]
            d["sigma_rel_whole"]["blockers"][sv][t] = b[t][sv]["sigma_rel_whole"]
    json.dump(d, open(FX, "w"), ensure_ascii=False, indent=1)
def wbar():
    d = json.load(open(FX))
    for t in ("real", "syn"):
        d["w_bar"]["blockers"][t] = json.load(open(R + f"s3_tb_{t}.json"))["stats"]["w_mean"]
    json.dump(d, open(FX, "w"), ensure_ascii=False, indent=1)
{"contour": contour, "sigma": sigma, "wbar": lambda p=None: wbar()}[sys.argv[1]](*sys.argv[2:])
