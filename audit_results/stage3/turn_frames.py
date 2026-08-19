"""段3用: regret 上位20件の「ターンのコマ送りデータ」を作る（2026-08-19）。

各件について
  start / steps[] … 実対局の監査対象ターン（各手の直後の盤面スナップショット付き）
  cf_start / cf_steps[] … 監査対象の判断点で◎最良手に差し替えた続き（同一世界・1例）
  cards … 登場する全カードの名前/コスト/パワー/カウンター/種類/効果テキスト
を JSON に落とす。アーティファクトのコマ送りビューアがこれを描く。

出力: /home/user/turn_frames20.json ＋人間可読ログ
"""
import os, sys, json, collections, subprocess
os.environ.setdefault("OPCG_LOG_SILENT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, f"{REPO}/tests")
sys.path.insert(0, f"{REPO}/tests/scripts")
sys.path.insert(0, f"{REPO}/tests/harness")
os.chdir(REPO)
import _bootstrap  # noqa

import numpy as np

TOP_N = 20
OUT = "/home/user/turn_frames20.json"
USED_IDS = set()


def _sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True).stdout


def load_regrets():
    rows, seen = [], set()
    for br in sorted(b for b in _sh(
            "git for-each-ref refs/remotes/origin --format='%(refname:short)'").split()
            if "moveaudit-shard" in b):
        for l in _sh(f"git show {br}:audit_results 2>/dev/null").splitlines():
            if not l.startswith("shard"):
                continue
            d = f"audit_results/{l.strip('/')}"
            for line in _sh(f"git show {br}:{d}/regret.jsonl 2>/dev/null").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = (r["seed"], r["decision"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
    return rows


def card_ref(c):
    m = c.master
    USED_IDS.add(m.card_id)
    ad = getattr(c, "attached_don", None)
    don = ad if isinstance(ad, int) else len(ad or [])
    pw = getattr(c, "current_power", None) or m.power
    return {"id": m.card_id, "pw": pw, "r": 1 if getattr(c, "is_rest", False) else 0,
            "d": don}


def snap_player(p, with_hand):
    d = {"lead": card_ref(p.leader) if p.leader else None,
         "life": len(p.life or []), "hand": len(p.hand or []),
         "da": len(p.don_active or []), "dr": len(p.don_rested or []),
         "dt": len(p.don_attached_cards or []),
         "field": [card_ref(c) for c in (p.field or [])],
         "stage": card_ref(p.stage) if getattr(p, "stage", None) else None}
    if with_hand:
        d["handc"] = []
        for c in (p.hand or []):
            USED_IDS.add(c.master.card_id)
            d["handc"].append(c.master.card_id)
    return d


def snap(m, _me_name=None):
    # 両席とも手札込みで保存する（リプレイは全情報を持つ）。表示側が監査席の手札だけ見せる。
    return {"p1": snap_player(m.p1, True), "p2": snap_player(m.p2, True)}


class _Done(BaseException):
    pass


class _Cap:
    """局面複製＋全判断の記録（手の説明＋直前スナップショット）。
    最後の監査対象決定のターンが終わるまで記録してから _Done で打ち切る。"""

    def __init__(self, wanted, describe, me_name_of):
        self.wanted = set(wanted); self.n = 0; self.frames = {}
        self.describe = describe; self.me_name_of = me_name_of
        self.log = []; self.last_wanted = max(wanted); self.last_turn = None

    def on_decision_point(self, ctx):
        if (self.n + 1) in self.wanted:
            self.frames[self.n + 1] = (ctx.manager.clone(), ctx.actor.name)

    def on_decision(self, ctx, move):
        self.n += 1
        try:
            desc = self.describe(ctx.manager, move) or {}
        except Exception:
            desc = {"action_type": (move or {}).get("action_type", "?")}
        turn = getattr(ctx.manager, "turn_count", None)
        self.log.append({"d": self.n, "turn": turn,
                         "seat": getattr(ctx.actor, "name", None), "desc": desc,
                         "snap": snap(ctx.manager, self.me_name_of)})
        if self.n == self.last_wanted:
            self.last_turn = turn
        if self.last_turn is not None and turn is not None and turn > self.last_turn:
            raise _Done()


def main():
    from cpu_arena import _load_db
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.core import cpu_ai
    from opcg_game import OPCGGame

    db = _load_db()

    def nm(cid):
        if not cid:
            return cid
        try:
            c = db.get_card(cid)
            USED_IDS.add(cid)
            return getattr(c, "name", None) or cid
        except Exception:
            return cid

    def label(desc):
        if not desc:
            return "-"
        s = desc.get("action_type", "?")
        if desc.get("card"):
            s += f" {nm(desc['card'])}"
        if desc.get("targets"):
            s += " → " + ", ".join(nm(t) for t in desc["targets"])
        return s

    regrets = load_regrets()
    top = sorted((r for r in regrets if r.get("regret") and not r.get("saturated")),
                 key=lambda r: -r["regret"])[:TOP_N]
    by_seed = collections.defaultdict(list)
    for r in top:
        by_seed[r["seed"]].append(r)

    eng = LearnedEngine(sims=160)
    gr = OPCGGame(prune_futile=False)
    gs = OPCGGame()
    items = {}

    for seed, rows in sorted(by_seed.items()):
        la, lb = _leader_pair(db, seed, rows[0].get("audit_leaders") or "random")
        cap = _Cap([r["decision"] for r in rows], cpu_ai._describe_move, "p1")
        seat = make_seat(kind="learned", want_trace=False,
                         sims=rows[0].get("audit_sims") or 160, engine=eng)
        try:
            run_game(seed, db, seats={"p1": seat, "p2": seat},
                     deck_builder=synth_deck_builder(la, lb, seed=seed),
                     observers=(cap,), max_steps=1500, legal_moves="skip",
                     invariants="raise",
                     stop_after_decisions=max(r["decision"] for r in rows) + 60)
        except _Done:
            pass
        except BaseException as e:
            print(f"seed {seed}: 再生失敗 {type(e).__name__}: {str(e)[:80]}", flush=True)

        for r in rows:
            key = f"{r['seed']}@{r['decision']}"
            hit_turn = next((e["turn"] for e in cap.log if e["d"] == r["decision"]),
                            r.get("turn"))
            turn_entries = [e for e in cap.log if e["turn"] == hit_turn]
            sentinel = next((e for e in cap.log if (e["turn"] or 0) > (hit_turn or 0)), None)
            start = turn_entries[0]["snap"] if turn_entries else None
            steps = []
            for i, e in enumerate(turn_entries):
                after = (turn_entries[i + 1]["snap"] if i + 1 < len(turn_entries)
                         else (sentinel["snap"] if sentinel else None))
                steps.append({"d": e["d"], "seat": e["seat"], "mv": label(e["desc"]),
                              "cid": e["desc"].get("card"), "hit": e["d"] == r["decision"],
                              "after": after})
            # --- 反実仮想: ◎最良手に差し替えて同一世界を1例だけ進める ---
            cf_steps, cf_start, err = [], None, None
            got = cap.frames.get(r["decision"])
            if got is None:
                err = "局面を復元できなかった"
            else:
                frame, actor_name = got
                cf_start = snap(frame, "p1")
                best = r.get("best")
                mv0 = None
                for mv in gr.legal_actions(frame):
                    if (cpu_ai._describe_move(frame, mv) or {}) == best:
                        mv0 = mv
                        break
                if mv0 is None:
                    err = "最良手が候補に見つからない"
                else:
                    m = gs.apply(frame.clone(), mv0, actor_name)
                    if m is None:
                        err = "最良手の適用に失敗"
                    else:
                        cf_steps.append({"seat": actor_name, "mv": label(best),
                                         "cid": (best or {}).get("card"), "first": True,
                                         "after": snap(m, "p1")})
                        eng._world_seeds = {}
                        rng = np.random.default_rng(424200 + r["decision"])
                        for _ in range(25):
                            if m.winner is not None or gs.is_terminal(m):
                                break
                            name = gs.current_player(m)
                            if name is None:
                                break
                            actor = m.p1 if m.p1.name == name else m.p2
                            mv = eng.decide(m, actor, rng=rng)
                            if mv is None:
                                break
                            desc = cpu_ai._describe_move(m, mv) or {}
                            m2 = gs.apply(m, mv, name)
                            if m2 is None:
                                break
                            m = m2
                            cf_steps.append({"seat": name, "mv": label(desc),
                                             "cid": desc.get("card"),
                                             "after": snap(m, "p1")})
                            if name == actor_name and desc.get("action_type") == "TURN_END":
                                break
                        if m.winner is not None:
                            cf_steps.append({"seat": "-", "mv": f"終局（勝者 {m.winner}）",
                                             "cid": None, "after": snap(m, "p1")})
            items[key] = {"turn": hit_turn, "seat": r.get("seat"),
                          "start": start, "steps": steps,
                          "cf_start": cf_start, "cf_steps": cf_steps, "cf_error": err}
            print(f"■ {key} T{hit_turn} 実{len(steps)}手 / 仮{len(cf_steps)}手"
                  f"{'  !! ' + err if err else ''}", flush=True)

    cards = {}
    for cid in sorted(USED_IDS):
        try:
            c = db.get_card(cid)
        except Exception:
            continue
        t = str(getattr(c, "type", "") or "")
        tl = ("L" if "LEADER" in t.upper() else
              "S" if "STAGE" in t.upper() else
              "E" if "EVENT" in t.upper() else "C")
        cards[cid] = {"n": getattr(c, "name", cid), "c": getattr(c, "cost", None),
                      "p": getattr(c, "power", None), "ct": getattr(c, "counter", None),
                      "t": tl, "x": getattr(c, "effect_text", "") or "",
                      "tr": "/".join(getattr(c, "traits", None) or []),
                      "lf": getattr(c, "life", None)}

    with open(OUT, "w") as f:
        json.dump({"cards": cards, "items": items}, f, ensure_ascii=False)
    print(f"TURN_FRAMES_DONE items={len(items)} cards={len(cards)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
