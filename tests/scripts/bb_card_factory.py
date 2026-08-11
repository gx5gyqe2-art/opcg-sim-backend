"""bb0: 合成カードファクトリ（骨組み線 Phase 0・2026-08-11・分離規約 `bb_*`）。

**目的**（docs/cpu_backbone_plan.md Phase 0）: 「実在しないカード」で構成されたデッキを
量産し、骨組み価値ネットのドメインランダム化訓練の材料にする。本ファイルは**カード合成のみ**
（自己対戦と監査は `bb_selfplay_audit.py`）。

**合成方式＝実カードからの能力収穫＋再結合**:
  - 実カード（CHARACTER/EVENT）のパース済み `Ability`（trigger/condition/cost/effect の効果木）
    をそのまま収穫し、合成カードへ載せ替える。**動くことが保証された断片だけを使う**ので
    パーサ/エンジンの未踏経路を踏まず、日本語テキスト生成も不要。
  - 効果予算の代理: 能力には「元ホストのコスト」を記録し、合成時はホストコスト c±1 の能力
    だけを許す（±2 は低確率＝**包絡を広げる**・固有性監査 #4）。ステータス（パワー/カウンター）
    も実カードのコスト条件付き分布からサンプルする。
  - 数値変異（--mutate）は既定 OFF: Phase 0 の問いは実行可能性であり、変異は Phase 3 で
    被覆を広げたくなった時のつまみ（効果木の int を ±1 する）。

**バニラリーダー**: 能力なし・パワー5000・ライフ5 の LEADER を合成する（リーダー効果の
影響を骨組みから排除する＝計画 §0 原理4）。

実行例（スモーク・合成デッキ1つの目視）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_card_factory.py --seed 7
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import dataclasses

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from engine_helpers import make_master  # noqa: E402
from opcg_sim.src.models.models import CardType  # noqa: E402

# 収穫対象のトリガー。UNKNOWN（パース不能の残骸）は除外。リーダー由来能力も除外
# （バニラリーダー原則＝リーダー的な文脈を前提にした能力を骨組み世界へ持ち込まない）。
_DECK_TYPES = (CardType.CHARACTER, CardType.EVENT)


def harvest(db):
    """実カードから (ability, 元ホストcost, 元ホストtype) を収穫する。

    戻り値: {"CHARACTER": [(cost, ability), ...], "EVENT": [...]}・
            {"power": {cost: [実測パワー...]}, "counter": {cost: [...]}}
    """
    pool = {"CHARACTER": [], "EVENT": []}
    stats = {"power": {}, "counter": {}, "cost_hist": {}}
    db.parse_all()                                 # 全カードを cards へロード（キャッシュ済なら軽い）
    for m in db.cards.values():
        t = getattr(getattr(m, "type", None), "name", "")
        if getattr(m, "type", None) not in _DECK_TYPES:
            continue
        c = int(getattr(m, "cost", 0) or 0)
        stats["cost_hist"][c] = stats["cost_hist"].get(c, 0) + 1
        if t == "CHARACTER":
            stats["power"].setdefault(c, []).append(int(getattr(m, "power", 0) or 0))
            stats["counter"].setdefault(c, []).append(int(getattr(m, "counter", 0) or 0))
        for ab in getattr(m, "abilities", ()) or ():
            trig = getattr(getattr(ab, "trigger", None), "name", "UNKNOWN")
            if trig == "UNKNOWN":
                continue
            pool[t].append((c, ab))
    return pool, stats


def _mutate_ints(node, rng, prob=0.3):
    """効果木の int フィールド（count/value 等）を ±1 変異する（--mutate 用・破壊的）。"""
    if node is None or not dataclasses.is_dataclass(node):
        return
    for f in dataclasses.fields(node):
        v = getattr(node, f.name, None)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and 1 <= v <= 5 and rng.random() < prob:
            setattr(node, f.name, max(1, v + int(rng.choice([-1, 1]))))
        elif dataclasses.is_dataclass(v):
            _mutate_ints(v, rng, prob)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _mutate_ints(x, rng, prob)


def _pick_abilities(pool_t, cost, rng, mutate=False):
    """ホストコスト cost の合成カードに載せる能力を 0〜2 個選ぶ（予算＝元ホスト c±1・低確率で±2）。"""
    n = int(rng.choice([0, 1, 1, 1, 2]))          # 実カードの能力数分布の粗い近似（能力1が最頻）
    if n == 0 or not pool_t:
        return ()
    span = 2 if rng.random() < 0.15 else 1        # 包絡を広げる（固有性監査 #4）
    cands = [ab for c, ab in pool_t if abs(c - cost) <= span]
    if not cands:
        return ()
    out = []
    for _ in range(n):
        ab = copy.deepcopy(cands[int(rng.integers(len(cands)))])
        if mutate:
            _mutate_ints(ab.cost, rng)
            _mutate_ints(ab.effect, rng)
        out.append(ab)
    return tuple(out)


def synth_card(pool, stats, rng, seq, mutate=False):
    """合成カード1枚（CHARACTER 80% / EVENT 20%＝実デッキの粗い比率）。card_id は BB- 接頭辞。"""
    is_char = rng.random() < 0.8
    costs = sorted(stats["cost_hist"])
    weights = np.array([stats["cost_hist"][c] for c in costs], dtype=float)
    cost = int(rng.choice(costs, p=weights / weights.sum()))
    cid = f"BB-{seq:04d}"
    if is_char:
        pw_pool = stats["power"].get(cost) or [max(1000, cost * 1000)]
        ct_pool = stats["counter"].get(cost) or [1000]
        power = int(rng.choice(pw_pool))
        counter = int(rng.choice(ct_pool))
        abilities = _pick_abilities(pool["CHARACTER"], cost, rng, mutate)
        return make_master(card_id=cid, name=f"合成{seq}", type=CardType.CHARACTER,
                           cost=cost, power=power, counter=counter, abilities=abilities)
    abilities = _pick_abilities(pool["EVENT"], cost, rng, mutate)
    return make_master(card_id=cid, name=f"合成E{seq}", type=CardType.EVENT,
                       cost=cost, power=0, counter=0, abilities=abilities)


def vanilla_leader(card_id="BB-L000", name="バニラ", power=5000, life=5):
    return make_master(card_id=card_id, name=name, type=CardType.LEADER,
                       cost=0, power=power, counter=0, life=life, abilities=())


def synth_deck(pool, stats, rng, seq_base, n_distinct=15, deck_size=50, mutate=False):
    """合成デッキ1つ: n_distinct 種を 2〜4 枚ずつ（計 deck_size 枚）。戻り値 (masters, counts)。"""
    masters = [synth_card(pool, stats, rng, seq_base + k, mutate) for k in range(n_distinct)]
    counts = [4] * n_distinct
    over = 4 * n_distinct - deck_size
    k = 0
    while over > 0:                                # 4枚基調から均等に削って 50 枚に合わせる
        if counts[k % n_distinct] > 2:
            counts[k % n_distinct] -= 1
            over -= 1
        k += 1
    return masters, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mutate", action="store_true")
    args = ap.parse_args()
    from cpu_selfplay import _load_db
    db = _load_db()
    pool, stats = harvest(db)
    print(f"収穫: CHARACTER能力 {len(pool['CHARACTER'])}・EVENT能力 {len(pool['EVENT'])}"
          f"・コスト帯 {min(stats['cost_hist'])}〜{max(stats['cost_hist'])}")
    rng = np.random.default_rng(args.seed)
    masters, counts = synth_deck(pool, stats, rng, seq_base=0, mutate=args.mutate)
    print(f"\n合成デッキ（seed={args.seed}・{sum(counts)}枚）:")
    for m, n in zip(masters, counts):
        abs_ = [getattr(ab.trigger, "name", "?") for ab in m.abilities]
        print(f"  {m.card_id} ×{n} {getattr(m.type, 'name', '?'):>9} c{m.cost}"
              f" P{m.power} 能力{abs_ or 'なし'}"
              + (f"  «{(m.abilities[0].raw_text or '')[:42]}»" if m.abilities else ""))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
