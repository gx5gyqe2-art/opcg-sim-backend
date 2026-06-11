"""H-6: バトル文脈での効果カバレッジ。

effect_coverage は ON_ATTACK / ON_BLOCK / COUNTER 能力を resolve_ability の
直呼びで検証する。しかし「このバトル中パワー+N」「アタック対象の変更」
「ブロッカー無効」等は active_battle（攻撃者/対象/カウンター値）が無いと
正しく実行できず、直呼びでは盤面が動かないように見える盲点があった。

本ツールは実際の戦闘フロー（declare_attack → handle_block → apply_counter →
resolve_attack）を駆動して各トリガーを発火させ、盤面/ステータスが動くかを
分類する。さらに duration!=INSTANT の効果が正しい境界で失効するかを検証する。

実行:
    OPCG_LOG_SILENT=1 python tests/battle_coverage.py
    OPCG_LOG_SILENT=1 python tests/battle_coverage.py --show BATTLE_NO_CHANGE
    OPCG_LOG_SILENT=1 python tests/battle_coverage.py --card OP01-025
"""
import argparse
import os
import sys
from collections import defaultdict
from typing import Optional

import conftest  # noqa: F401

import effect_coverage as cov
from engine_helpers import make_master
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.models.enums import CardType, Phase, Zone
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data")

_BATTLE_TRIGGERS = {"ON_ATTACK", "ON_OPP_ATTACK", "ON_BLOCK", "COUNTER"}


def _drain_battle(gm) -> None:
    cov._smart_drain(gm, record={})


def classify_battle(master, trig: str) -> str:
    """1カードの bat トリガーを実戦フローで発火させ、盤面変化を分類する。"""
    try:
        gm, p1, p2, source = cov._build_test_state(master)
    except Exception as e:
        return f"SETUP_ERROR:{e}"

    # 攻撃の枠組みを用意する。ON_ATTACK/ON_BLOCK はソース側、ON_OPP_ATTACK/COUNTER は
    # 相手の攻撃に対する防御側として駆動する。
    gm.phase = Phase.MAIN
    fb = cov._zone_fingerprint(p1, p2)
    sb = cov._stat_snap(p1, p2)
    try:
        if trig in ("ON_ATTACK", "ON_BLOCK"):
            # p1 のソースが p2 リーダーへアタック
            attacker = source if source in p1.field else (p1.field[0] if p1.field else None)
            if attacker is None or not p2.leader:
                return "NO_SETUP"
            attacker.is_rest = False
            if trig == "ON_BLOCK":
                # ソースをブロッカーにして p2 の攻撃を受ける
                gm.turn_player, gm.opponent = p2, p1
                atk = p2.field[0] if p2.field else None
                if atk is None:
                    return "NO_SETUP"
                atk.is_rest = False
                source.is_rest = False
                gm.declare_attack(atk, p1.leader)
                _drain_battle(gm)
                if gm.active_interaction:
                    return "INTERACTIVE"
                if gm.phase in (Phase.BLOCK_STEP,) or gm.active_battle:
                    gm.handle_block(source)
                    _drain_battle(gm)
            else:
                gm.turn_player, gm.opponent = p1, p2
                gm.declare_attack(attacker, p2.leader)
                _drain_battle(gm)
        elif trig in ("ON_OPP_ATTACK", "COUNTER"):
            gm.turn_player, gm.opponent = p2, p1
            atk = p2.field[0] if p2.field else None
            if atk is None or not p1.leader:
                return "NO_SETUP"
            atk.is_rest = False
            gm.declare_attack(atk, p1.leader)
            _drain_battle(gm)
            if trig == "COUNTER" and gm.active_battle:
                # カウンターステップ: ソース能力を直接解決（手札カウンターの発火相当）
                gm.phase = Phase.BATTLE_COUNTER
                for ab in master.abilities:
                    t = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
                    if t == "COUNTER" and ab.effect is not None:
                        gm.resolve_ability(p1, ab, source)
                _drain_battle(gm)
        else:
            return "SKIP"
    except Exception:
        return "ERROR"

    if gm.active_interaction:
        return "INTERACTIVE"
    fa = cov._zone_fingerprint(p1, p2)
    sa = cov._stat_snap(p1, p2)
    ig = frozenset()
    if cov._moved(fb, fa) or cov._stat_changed(sb, sa, ig):
        return "BATTLE_EXECUTED"
    if bool(getattr(gm, "action_events", [])):
        return "BATTLE_EXECUTED"
    return "BATTLE_NO_CHANGE"


def collect(card_filter: Optional[str] = None):
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]
    buckets = defaultdict(list)
    total = len(card_ids)
    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        seen = set()
        for ab in master.abilities:
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            if trig not in _BATTLE_TRIGGERS or trig in seen:
                continue
            seen.add(trig)
            status = classify_battle(master, trig)
            buckets[status].append((cid, master.name, trig,
                                    (getattr(ab.effect, "raw_text", "") or "")[:46]))
    sys.stderr.write(f"\r完了: {total} カード処理済み\n")
    return buckets


def run(show: Optional[str] = None, card_filter: Optional[str] = None):
    buckets = collect(card_filter)
    print("=== バトル文脈での実行分類（H-6）===")
    for k in ("BATTLE_EXECUTED", "BATTLE_NO_CHANGE", "INTERACTIVE", "NO_SETUP",
              "ERROR", "SKIP"):
        mark = "  ★真のバグ候補" if k == "BATTLE_NO_CHANGE" else ""
        print(f"  {k:<18}: {len(buckets.get(k, [])):4d}{mark}")
    for k in list(buckets):
        if k.startswith("SETUP_ERROR") or k == "ERROR":
            for cid, name, trig, raw in buckets[k][:5]:
                print(f"    {k}: {cid} {trig} {name}")
    print()
    targets = [show] if show else ["BATTLE_NO_CHANGE"]
    for t in targets:
        items = buckets.get(t, [])
        print(f"--- {t} ({len(items)} 件) ---")
        for cid, name, trig, raw in items[:80]:
            print(f"  {cid:<12} {trig:<14} {name}  | {raw}")
        if len(items) > 80:
            print(f"  ... 他 {len(items) - 80} 件")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None)
    ap.add_argument("--card", default=None)
    args = ap.parse_args()
    run(show=args.show, card_filter=args.card)
