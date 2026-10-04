import json, sys, os
D = os.path.dirname(os.path.abspath(__file__))
for s in ("real", "syn"):
    for m in ("cuttable_forced", "rule_don", "rule", "rule_don_purse", "noplan", "spdavg"):
        p = "%s/cb_%s_%s.json" % (D, s, m)
        if not os.path.exists(p) or not os.path.getsize(p):
            continue
        j = json.load(open(p))
        th = j["summary"]["by_slope"]["theory"]; st = j["stats"]
        print(s, m, "%.4f %+.3f %.3f %.4f %.3f" % (th["sign_accuracy"], th["bias"], th["sigma_T"], th["within1"], th["mae"]),
              "capped", st["tau_capped"], j.get("rule_stats"))
