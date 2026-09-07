"""held-out 実デッキ集合のローダ（汎化ゲート専用・凍結 v4b）。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md。`heldout_decks.json` はユーザの実対局リプレイ
から抽出した実構築（スナップショット・改変禁止）。**訓練（自己対戦・デッキ生成）に使うことは
リークであり禁止**。許可される用途は (a) vs L1 勝率ゲート (b) Covering Radius の一方向確認 のみ。
"""
import json
import os

from opcg_sim.src.models.models import CardInstance

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(_HERE, "fixtures", "heldout_decks.json")


def load_spec():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def deck_ids():
    return [d["id"] for d in load_spec()["decks"]]


def build(db, deck_id, owner_id):
    """held-out デッキを (leader CardInstance, [CardInstance]*50) で構築する。"""
    spec = next(d for d in load_spec()["decks"] if d["id"] == deck_id)
    lm = db.get_card(spec["leader"])
    if lm is None:
        raise ValueError(f"leader {spec['leader']} が DB に無い")
    leader = CardInstance(lm, owner_id)
    cards = []
    for cid, n in spec["cards"].items():
        m = db.get_card(cid)
        if m is None:
            raise ValueError(f"{cid} が DB に無い")
        for _ in range(int(n)):
            cards.append(CardInstance(m, owner_id))
    if len(cards) != 50:
        raise ValueError(f"{deck_id}: {len(cards)}枚（50枚必須）")
    return leader, cards


def all_lists():
    """{deck_id: {card_id: count}}（Covering Radius の一方向確認用の読み出し）。"""
    return {d["id"]: dict(d["cards"]) for d in load_spec()["decks"]}
