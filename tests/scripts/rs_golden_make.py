"""golden の作り直し（**Rust だけで回る**・2026-09-07 第 2 段 `rs-archive-cutover`）。

第 1 段（§16.1）の golden は「Python エンジンで打った局／発動した能力」を記録し、その場で
Rust と突き合わせて焼き付けたものだった。第 2 段で **golden の正本は Rust** になった
（計画 §16.3-10）ので、作り直しも Rust だけで回る形にする。Python エンジンは要らない。

    python -m ... rs_golden_make.py audit                    # 監査 golden（3,386 能力）
    python -m ... rs_golden_make.py replay --policy a1 \\
        --games 50 --seed-base 5000150                       # 再生 golden（a1 の 50 局）

**いつ作り直すか**: 挙動を意図的に変えたときだけ（裁定の変更・記録契約 `RECORD_VERSION` の
更新・正規化規約 `GOLDEN_VERSION` の更新）。差分は必ずレビューする——golden は「その時点の
Rust の出力」であって、正しさの独立した証拠ではない。

**方策**（`--policy`）:
  `a1`     … 出荷既定のネットで打つ（`Game.decide`）。旧 `l1` 帯の置き換え（L1 は廃止・§16.3-8）
  `random` … 合法手から決定的に引く（seed → 同じ局）
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import random                                                          # noqa: E402
import sys                                                             # noqa: E402

import os as _os, sys as _sys                                          # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap                                                      # noqa: E402,F401

from harness.rs_golden import (AUDIT_GOLDEN, GOLDEN_VERSION, REPLAY_GOLDEN_DIR,  # noqa: E402
                               load_engine, replay_hashes, strip_request_id)
from opcg_sim.loop import decks as D                                   # noqa: E402
from opcg_sim.loop import driver as DR                                 # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402

MAX_STEPS = 400
SOURCE = "tests/scripts/rs_golden_make.py"


# --- 監査 golden -------------------------------------------------------------

def _abilities(effects_path):
    """効果構造 JSON から (card_id, trigger, ability_index) を列挙する（ID 順・能力順）。

    Python エンジンは通らない（JSON はパーサの生成物）。件数は監査 golden の収録件数と一致する。
    """
    with open(effects_path, encoding="utf-8") as f:
        cards = json.load(f)["cards"]
    out = []
    for cid in sorted(cards):
        for i, ab in enumerate(cards[cid].get("abilities") or []):
            out.append((cid, ab.get("trigger"), i))
    return out


def cmd_audit(args) -> int:
    engine = load_engine()
    entries = []
    for cid, trigger, idx in _abilities(args.effects):
        got = json.loads(engine.golden_audit(cid, trigger, idx))
        entries.append({"card_id": cid, "trigger": trigger, "ability_index": idx,
                        "hashes": got["hashes"], "summary": got["summary"]})
    body = {"version": GOLDEN_VERSION, "record_version": engine.record_version(),
            "source": f"{SOURCE} audit", "entries": entries}
    os.makedirs(os.path.dirname(AUDIT_GOLDEN), exist_ok=True)
    with open(AUDIT_GOLDEN, "w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print(f"[golden] {AUDIT_GOLDEN} ({len(entries)} abilities, "
          f"{os.path.getsize(AUDIT_GOLDEN)/1e6:.2f} MB)")
    return 0


# --- 再生 golden -------------------------------------------------------------

def _record_game(seed, policy, spec, decks_mode):
    """1 局を Rust で打ち、記録（`opcg_engine.replay` が食える形）と期待 sha1 を作る。"""
    db = D.load_db()
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, decks_mode)
    game = DR.new_game(seed, p1, p2)
    setup_hidden = json.loads(game.hidden_json())
    rng = random.Random(seed)          # policy=random 用（seed → 決定的）
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    steps, states, legals, events, shuffled = [], [], [], [], []
    while game.winner() is None and len(steps) < MAX_STEPS:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        legal = json.loads(game.legal_json(name))
        if not legal:
            break
        if policy == "random":
            move = legal[rng.randrange(len(legal))]
        else:
            turn = int(game.turn_count or 0)
            out = json.loads(game.decide(name, spec.decide_opts(seed, turn, name,
                                                                carry[name].get(turn, name))))
            carry[name].put(out)
            move = out.get("move")
            if move is None:
                break
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"],
                                   json.dumps(move.get("payload") or {}, ensure_ascii=False))
        sh = json.loads(game.shuffled_json())
        hidden = json.loads(game.hidden_json())
        step = {"index": len(steps), "move": move, "shuffled": sh}
        if sh:
            # 再同期に要るのは `deck`／`hand` の uuid の並びだけ（`state::resync_shuffled`）。
            step["hidden"] = {"players": {
                s: {"deck": [{"uuid": c["uuid"]} for c in hidden["players"][s]["deck"]],
                    "hand": [{"uuid": c["uuid"]} for c in hidden["players"][s]["hand"]]}
                for s in sh}}
        steps.append(step)
        # 再生（`opcg_engine.replay`）が出す `states` は **盤面 dict ＋ `pending_request`**
        # （`state.rs` の組み立て順）。同じ形で記録しないと sha1 が揃わない。
        board = json.loads(game.board_json())
        board["pending_request"] = json.loads(game.pending_json())
        states.append(board)
        legals.append(legal)
        events.append(json.loads(game.events_json()))
        shuffled.append(sh)
    return {
        "version": GOLDEN_VERSION, "record_version": 5, "source": f"{SOURCE} replay",
        "seed": seed, "policy": policy, "vanilla": False,
        "input": {"version": 5, "seed": seed, "policy": policy, "vanilla": False,
                  "setup": {"first_player": None, "hidden": setup_hidden}, "steps": steps},
        "hashes": replay_hashes(states, legals, events, shuffled),
        "summary": {"actions": len(steps), "steps": len(steps),
                    "turns": int(game.turn_count or 0), "winner": game.winner(),
                    "p1_leader": p1[0], "p2_leader": p2[0]},
    }


def cmd_replay(args) -> int:
    engine = load_engine()
    E.engine()
    spec = E.SeatSpec(None, sims=args.sims) if args.policy != "random" else None
    os.makedirs(REPLAY_GOLDEN_DIR, exist_ok=True)
    written = mismatched = 0
    for i in range(args.games):
        seed = args.seed_base + i
        g = _record_game(seed, args.policy, spec, args.decks)
        # 焼き付ける前に**その場で再生して確かめる**（記録が replay の契約に合っているか）。
        out = json.loads(engine.replay(json.dumps(g["input"], ensure_ascii=False, default=str)))
        got = replay_hashes(
            [strip_request_id(s) for s in out["states"]],
            out.get("legal") or [None] * len(out["states"]),
            out.get("events") or [None] * len(out["states"]),
            [s["shuffled"] for s in g["input"]["steps"]])
        if got["states"] != g["hashes"]["states"]:
            mismatched += 1
            print(f"  seed={seed}: 再生が記録と一致しない（この局は書かない）", flush=True)
            continue
        path = os.path.join(REPLAY_GOLDEN_DIR, f"{args.policy}_{seed}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(g, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            f.write("\n")
        written += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.games} 局", flush=True)
    print("RS_GOLDEN_REPLAY " + json.dumps(
        {"policy": args.policy, "written": written, "mismatched": mismatched,
         "seed_base": args.seed_base}, ensure_ascii=False))
    return 1 if mismatched else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="監査 golden を作り直す")
    p.add_argument("--effects", default=None)
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("replay", help="再生 golden を作り直す")
    p.add_argument("--policy", default="a1", choices=("a1", "random"))
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--sims", type=int, default=32,
                   help="a1 の探索数（golden は再生の照合だけなので serve 既定より軽くてよい）")
    p.add_argument("--decks", default="singleton", choices=("singleton", "synth"))
    p.set_defaults(func=cmd_replay)
    args = ap.parse_args(argv)
    if getattr(args, "effects", None) is None:
        from harness.rs_golden import DEFAULT_EFFECTS_PATH
        args.effects = DEFAULT_EFFECTS_PATH
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
