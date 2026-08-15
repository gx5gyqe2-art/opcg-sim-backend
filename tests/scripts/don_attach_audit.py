"""ドン付与の行動空間監査（2026-08-12・ユーザ指摘「CPUはドン付与が苦手」の診断）。

**問い**: 実対局リプレイで観測される ATTACH_DON（特に人間の手）のうち、現行の
候補枝刈り `cpu_ai._prune_don_moves`（B-2・`_attach_don_meaningful`）が**候補から
落としてしまう手**はどれだけあるか。落ちている手は CPU が「評価して捨てた」のではなく
**考えることすらできない**＝行動空間の穴。

分類:
  kept      = 現行枝刈りが候補に残す（しきい値越え or 【ドン!!×N】開放）
  DROPPED   = 枝刈りが落とす。内訳:
    overcap    : 付与先が既に全防御パワー以上＝「過剰盛り」（7000作り・カウンター強要の族。
                 ユーザのエネル理論の手はここに落ちる）
    unreachable: 全ドンを載せても届かない付与先
    rested/sick: 付与先がこのターン攻撃できない（防御ドン・次ターン仕込み等）
    other      : 上記以外（列挙不能・復元不能含む）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/don_attach_audit.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
from collections import Counter

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai  # noqa: E402


def _label(card):
    return getattr(getattr(card, "master", None), "card_id", None)


def classify(m, actor_name, tgt_label):
    """記録された付与（actor が tgt_label へ）を現行枝刈りに照らして分類する。"""
    actor = m.p1 if m.p1.name == actor_name else m.p2
    raw = m.get_legal_actions(actor)
    att = [mv for mv in raw if mv.get("action_type") == "ATTACH_DON"]
    by_uuid = {}
    for u in ([actor.leader] if actor.leader is not None else []) + list(actor.field):
        by_uuid[getattr(u, "uuid", None)] = u
    cands = []
    for mv in att:
        c = by_uuid.get((mv.get("payload") or {}).get("uuid"))
        if c is not None and _label(c) == tgt_label:
            cands.append((mv, c))
    if not cands:
        return "other", {}
    kept = cpu_ai._prune_don_moves(m, actor_name, [mv for mv, _c in cands])
    if kept:
        return "kept", {}
    # DROPPED の内訳（先頭候補で判定＝同ラベル複数体は同条件とみなす）
    _mv, c = cands[0]
    detail = {}
    if getattr(c, "is_rest", False) or (getattr(c, "is_newly_played", False)
                                        and not c.has_keyword("速攻")):
        return "dropped:rested/sick", detail
    try:
        p = float(c.get_power(True))
    except Exception:
        p = float(getattr(getattr(c, "master", None), "power", 0) or 0)
    opp = m.p2 if actor is m.p1 else m.p1
    defs = []
    for u in ([opp.leader] if opp.leader is not None else []) + list(opp.field):
        try:
            defs.append(float(u.get_power(False)))
        except Exception:
            defs.append(float(getattr(getattr(u, "master", None), "power", 0) or 0))
    budget = len(actor.don_active)
    if defs and p >= max(defs):
        detail = {"p": int(p), "max_def": int(max(defs))}
        return "dropped:overcap", detail
    if defs and all(p + budget * 1000 < d for d in defs if d > p):
        return "dropped:unreachable", detail
    return "dropped:other", detail


def main():
    import mark_gate as MG
    import replay_reeval as RE
    import coach_gate as CG
    from cpu_selfplay import _load_db

    db = _load_db()
    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    stats = {"human": Counter(), "cpu": Counter()}
    examples = []
    for tag in sorted(table):
        raw = RE.load_replay_json(table[tag])
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        for i, a in enumerate(acts):
            if a.get("action_type") != "ATTACH_DON":
                continue
            src = a.get("src") or "?"
            built = MG._restore(db, rec, fbi, acts, i)
            if isinstance(built, str) or built is None:
                stats.setdefault(src, Counter())["other"] += 1
                continue
            m0, _actor = built
            cls, detail = classify(m0, a.get("player"), a.get("card"))
            stats.setdefault(src, Counter())[cls] += 1
            if cls.startswith("dropped") and cls != "dropped:rested/sick":
                examples.append({"tag": tag, "i": i, "turn": a.get("turn"),
                                 "src": src, "card": a.get("card"), "cls": cls, **detail})
    print("\n=== ドン付与の行動空間監査（全リプレイ・実際に打たれた ATTACH_DON） ===")
    for src in sorted(stats):
        c = stats[src]
        total = sum(c.values())
        if not total:
            continue
        drop = sum(v for k, v in c.items() if k.startswith("dropped"))
        print(f"  {src:>6}: 計{total:3d}  kept {c.get('kept',0):3d}  DROPPED {drop:3d} "
              f"（overcap {c.get('dropped:overcap',0)} / rested {c.get('dropped:rested/sick',0)}"
              f" / unreachable {c.get('dropped:unreachable',0)} / other {c.get('dropped:other',0)}"
              f" / 復元不能等 {c.get('other',0)}）")
    print("\n  DROPPED の実例（rested を除く）:")
    for e in examples[:20]:
        print(f"    {e['tag']}@{e['i']} T{e['turn']} {e['src']} {e['card']} {e['cls']}"
              + (f" p={e.get('p')} vs max_def={e.get('max_def')}" if 'p' in e else ""))
    print("\nDON_AUDIT " + json.dumps({s: dict(c) for s, c in stats.items()},
                                      ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
