"""型 × 色のカバレッジ計測（WP `rs-removal-decks`・§20.8.1-5・単体実行の計測 CLI）。

143 リーダー × seed 0〜1 で `deck_synth` → `deck_roles.inject_roles` を回し、
  - **型 × 色のカバレッジ**（各型が入ったデッキ数・差し込み前／後）
  - **組めない型**（その色にカードが `MIN_KIND_CARDS` 未満＝評価の層から外す根拠）
  - 死に札監査の落ち数・差し込みが割れて減らした回数
を JSON で出す（`docs/reports/2026-09-10_removal_decks.RESULT.json` の材料）。

  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_removal_decks_report.py \\
    --seeds 0 1 --out /tmp/removal_decks.json
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

from opcg_sim.loop import deck_roles as R                                # noqa: E402
from opcg_sim.loop import deck_synth as DS                               # noqa: E402
from opcg_sim.loop import decks as D                                     # noqa: E402

COLORS = ("BLACK", "BLUE", "GREEN", "PURPLE", "RED", "YELLOW")


def color_report(db):
    """色ごとの「組める型／組めない型」（テンプレート単位・型の鍵単位の両方）。"""
    tmpl, kinds, unbuildable = {}, {}, {}
    for col in COLORS:
        pool = R.color_pool(db, {col})
        names = [t["name"] for t in R.buildable(pool)]
        tmpl[col] = names
        kinds[col] = R.kind_counts(pool)
        thin = sorted(k for k, n in kinds[col].items() if n < R.MIN_KIND_CARDS)
        missing = [t["name"] for t in R.TEMPLATES if t["name"] not in names]
        unbuildable[col] = {"templates": missing, "thin_kinds": thin}
    return tmpl, kinds, unbuildable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    ap.add_argument("--leaders", type=int, default=0, help="先頭 N リーダーだけ（0=全部）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    t0 = time.time()
    db = D.load_db()
    leaders = D.leader_pool(db)
    if args.leaders:
        leaders = leaders[:args.leaders]
    tmpl, kind_pool, unbuildable = color_report(db)

    # 色 → 型 → その型を持つデッキ数（差し込み前 native／差し込み後 after）
    cov_native = collections.defaultdict(collections.Counter)
    cov_after = collections.defaultdict(collections.Counter)
    # form 単位（KO/bounce/…）で「その form を 1 つでも持つデッキ数」＝読みやすい要約
    form_native = collections.defaultdict(collections.Counter)
    form_after = collections.defaultdict(collections.Counter)
    rates = collections.Counter()
    n_decks = audit_fail = shrunk = empty = 0
    dead_worse = 0
    hard = ("KO", "bounce", "deck", "trash")      # 硬い除去（盤面から駒が減る）
    no_rm = {"native": 0, "after": 0}             # §20.7.11 の「除去 0 枚が 27%」と較べる
    per_leader = []
    for lid in leaders:
        for seed in args.seeds:
            leader, cards = DS.synth_deck(db, lid, seed=seed, owner="p1")
            lm = DS._master(leader)
            cols = sorted(DS.colors_of(lm))
            base_dead = DS.audit_deck(leader, cards)["dead_rate"]
            out, kinds = R.inject_roles(db, leader, cards, "p1", seed=seed)
            after = set(R.deck_kinds(out))
            n_decks += 1
            rates[kinds["rate"]] += 1
            if kinds["reduced"]:
                shrunk += 1
            if kinds["rate"] > 0 and kinds["n"] == 0:
                empty += 1
            au = DS.audit_deck(leader, out)
            if au["dead_rate"] > base_dead:
                dead_worse += 1
            if au["dead_rate"] > 0:
                audit_fail += 1
            no_rm["native"] += 0 if any(k.split(":")[0] in hard for k in kinds["native"]) else 1
            no_rm["after"] += 0 if any(k.split(":")[0] in hard for k in after) else 1
            for col in cols:
                for k in kinds["native"]:
                    cov_native[col][k] += 1
                for k in after:
                    cov_after[col][k] += 1
                for f in {k.split(":")[0] for k in kinds["native"]}:
                    form_native[col][f] += 1
                for f in {k.split(":")[0] for k in after}:
                    form_after[col][f] += 1
            per_leader.append({"leader": lid, "seed": seed, "colors": cols,
                               "rate": kinds["rate"], "n": kinds["n"],
                               "templates": kinds["templates"], "reduced": kinds["reduced"],
                               "dead_rate": au["dead_rate"],
                               "counter_cards": R._counter_cards([DS._master(x) for x in out])})
    out = {
        "decks": n_decks, "leaders": len(leaders), "seeds": list(args.seeds),
        "templates_by_color": tmpl,
        "unbuildable": unbuildable,
        "pool_kind_counts": {c: dict(sorted(v.items())) for c, v in kind_pool.items()},
        "coverage_native": {c: dict(sorted(v.items())) for c, v in cov_native.items()},
        "coverage_after": {c: dict(sorted(v.items())) for c, v in cov_after.items()},
        "coverage_form_native": {c: dict(sorted(v.items())) for c, v in form_native.items()},
        "coverage_form_after": {c: dict(sorted(v.items())) for c, v in form_after.items()},
        "decks_by_color": dict(collections.Counter(
            c for r in per_leader for c in r["colors"])),
        "rates": {str(k): v for k, v in sorted(rates.items())},
        "no_hard_removal": dict(no_rm),
        "audit_dead_decks": audit_fail, "audit_worse_than_base": dead_worse,
        "shrunk": shrunk, "empty_after_shrink": empty,
        "elapsed_s": round(time.time() - t0, 1),
        "per_leader": per_leader,
    }
    js = json.dumps(out, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(js)
    print(f"decks={n_decks} shrunk={shrunk} empty={empty} dead_worse={dead_worse} "
          f"{out['elapsed_s']}s")
    print("REMOVAL_DECKS_DONE " + json.dumps(
        {"decks": n_decks, "shrunk": shrunk, "empty": empty}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
