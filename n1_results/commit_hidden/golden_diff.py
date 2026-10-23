"""a1 帯の再生 golden の差分を 1 局ずつ分類する（WP rs-commit-hidden・1 回きり）。

各 seed で (1) HEAD の golden（git show）の手の列、(2) 既定（修正後）、(3) commit_hidden_legacy で打ち直し、
最初に分かれた手・その手番の決め方（kind）・旧い旗で HEAD を再現するかを書く。
"""
import json
import os
import subprocess
import sys

REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
os.chdir(REPO)
import _bootstrap  # noqa: F401,E402
from harness.rs_golden import load_engine  # noqa: E402
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

MAX_STEPS = 400


def play(seed, spec):
    db = D.load_db()
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "singleton")
    game = DR.new_game(seed, p1, p2)
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    moves, kinds = [], []
    while game.winner() is None and len(moves) < MAX_STEPS:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        turn = int(game.turn_count or 0)
        out = json.loads(game.decide(name, spec.decide_opts(seed, turn, name, carry[name].get(turn, name))))
        carry[name].put(out)
        move = out.get("move")
        if move is None:
            break
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
        moves.append(move)
        kinds.append((out.get("kind"), name, turn, len(out.get("commit") or [])))
    return moves, kinds, game.winner(), int(game.turn_count or 0)


def head_moves(seed, ref):
    txt = subprocess.check_output(["git", "show", f"{ref}:tests/fixtures/rs_goldens/replay/a1_{seed}.json"])
    g = json.loads(txt)
    return [s["move"] for s in g["input"]["steps"]], g["summary"]["winner"]


def norm(m):
    return json.dumps(m, sort_keys=True, ensure_ascii=False)


def main():
    ref = sys.argv[1]
    out_path = sys.argv[2]
    seeds = range(5000150, 5000200)
    load_engine()
    E.engine()
    new = E.SeatSpec(None, sims=32)
    old = E.SeatSpec(None, sims=32, commit_hidden_legacy=True)
    rows = []
    for seed in seeds:
        hm, hw = head_moves(seed, ref)
        nm, nk, nw, _ = play(seed, new)
        lm, lk, lw, _ = play(seed, old)
        hs, ns, ls = [norm(m) for m in hm], [norm(m) for m in nm], [norm(m) for m in lm]
        legacy_ok = hs == ls
        if hs == ns:
            rows.append({"seed": seed, "changed": False, "legacy_reproduces_head": legacy_ok})
            continue
        i = next((k for k in range(min(len(hs), len(ns))) if hs[k] != ns[k]), min(len(hs), len(ns)))
        rows.append({
            "seed": seed, "changed": True, "first_div": i,
            "old": hm[i] if i < len(hm) else None, "new": nm[i] if i < len(nm) else None,
            "kind_old": lk[i][0] if i < len(lk) else None, "kind_new": nk[i][0] if i < len(nk) else None,
            "seat_turn": list(nk[i][1:3]) if i < len(nk) else None,
            "prev_kinds_legacy": [k[0] for k in lk[max(0, i - 3):i + 1]],
            "prev_moves": [m.get("action_type") for m in hm[max(0, i - 3):i]],
            "legacy_reproduces_head": legacy_ok, "len": [len(hm), len(nm)], "winner": [hw, nw],
        })
        print(json.dumps(rows[-1], ensure_ascii=False)[:400], flush=True)
    json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=1)
    ch = [r for r in rows if r["changed"]]
    print(f"changed {len(ch)}/50 legacy_reproduces_head {sum(r['legacy_reproduces_head'] for r in rows)}/50 "
          f"winner_changed {sum(1 for r in ch if r['winner'][0] != r['winner'][1])}")


if __name__ == "__main__":
    main()
