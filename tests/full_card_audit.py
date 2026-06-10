"""全カード横断検証ハーネス（トラックB）。

2デッキ限定だった深い検証を全2652枚へ広げる。各カードの全能力を汎用盤面で発動し、
**カードの保全に関する構造不変条件**を全カードで自動検証する。意味的な正しさ（効果が
テキスト通りか）は手動/golden で詰めるが、以下は意味を知らずとも自動で守れる:

  - EXCEPTION : 発動中に例外（クラッシュ）
  - CARD_LOSS : カード総数（全ゾーン＋temp、don除く）が減少＝カードが消失
  - TEMP_LEAK : 解決完了（中断なし）後も temp_zone にカードが残る＝デッキ等から消失予備軍

`effect_coverage` の足場（_build_test_state/_smart_drain）を再利用する。
セット単位（OP01.., EB.., ST..）の内訳も出す。回帰ガードは test_full_card_audit.py。

使い方:
    OPCG_LOG_SILENT=1 python tests/full_card_audit.py            # 全体＋セット内訳
    OPCG_LOG_SILENT=1 python tests/full_card_audit.py --show     # 異常カード一覧
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import conftest  # noqa: F401

import effect_coverage as cov
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data")


def _total_cards(p) -> int:
    """don を除く全ゾーンのカード総数（保全チェック用）。"""
    n = len(p.hand) + len(p.field) + len(p.trash) + len(p.deck) + len(p.life) + len(p.temp_zone)
    if p.leader:
        n += 1
    if getattr(p, "stage", None):
        n += 1
    return n


def _set_of(card_id: str) -> str:
    m = re.match(r"([A-Z]+\d+|[A-Z]+)", card_id)
    return m.group(1) if m else card_id


@dataclass
class Anomaly:
    card_id: str
    trigger: str
    kind: str   # EXCEPTION / CARD_LOSS / TEMP_LEAK
    detail: str = ""


def audit() -> List[Anomaly]:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    anomalies: List[Anomaly] = []
    total = len(card_ids)

    for i, cid in enumerate(card_ids, 1):
        if i % 300 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        for ab in master.abilities:
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            try:
                if trig == "ON_PLAY":
                    gm, p1, p2, src = cov._build_test_state(master, source_in_hand=True)
                    t0 = _total_cards(p1) + _total_cards(p2)
                    gm.play_card_action(p1, src)
                else:
                    gm, p1, p2, src = cov._build_test_state(master)
                    t0 = _total_cards(p1) + _total_cards(p2)
                    gm.resolve_ability(p1, ab, src)
                cov._smart_drain(gm)
            except Exception as e:
                anomalies.append(Anomaly(cid, trig, "EXCEPTION", str(e)[:60]))
                continue
            if gm.active_interaction:
                continue  # 手動操作待ち（INTERACTIVE）は検証対象外
            leak = len(p1.temp_zone) + len(p2.temp_zone)
            if leak > 0:
                anomalies.append(Anomaly(cid, trig, "TEMP_LEAK", f"{leak} 枚 temp 残留"))
            t1 = _total_cards(p1) + _total_cards(p2)
            if t1 < t0:
                anomalies.append(Anomaly(cid, trig, "CARD_LOSS", f"{t0 - t1} 枚消失"))
    sys.stderr.write(f"\r完了: {total} カード処理済み\n")
    return anomalies


def run(show: bool = False) -> None:
    anomalies = audit()
    by_kind = defaultdict(int)
    by_set: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for a in anomalies:
        by_kind[a.kind] += 1
        by_set[_set_of(a.card_id)][a.kind] += 1

    print("=== 全カード横断検証（構造不変条件）===")
    for k in ("EXCEPTION", "CARD_LOSS", "TEMP_LEAK"):
        print(f"  {k:<10}: {by_kind[k]:4d}")
    print()
    if by_set:
        print("--- セット別 異常 ---")
        for s in sorted(by_set):
            parts = [f"{k}={v}" for k, v in sorted(by_set[s].items())]
            print(f"  {s:<8}  {', '.join(parts)}")
        print()
    if show:
        for a in anomalies:
            print(f"  [{a.kind}] {a.card_id} {a.trigger}  {a.detail}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    run(show=args.show)
