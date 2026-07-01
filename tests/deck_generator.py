"""エンジン駆動型パラメトリック デッキ生成器（v4b 実装ステップ3・訓練データ分布の本体）。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md §生成。**実デッキリストは一切参照しない**。
毎ゲーム新規サンプル＝デッキ識別の学習を構造的に不可能にする（＝データレベルの暗記禁止）。

生成手順:
  1. リーダーを全リーダーから一様サンプル（色多様性はデッキ構築ルール経由で自動・色数パラメータ無し）
  2. コアパッケージ: leader_core_candidates（DB構造由来）から core_n 種を 3〜4 枚積む
     ＝「リーダー中心の非線形コンボ」を分布に保つ（レビュー第5巡）
  3. Trait二部サンプリング: リーダーTrait T の members(T)/referencers(T) から
     λ（集中度）×α（構成比）で充填（レビュー第6巡・発動側/受け側のバランス保証）
  4. 残りを役割クラスタから充填: クラスタ重み〜Dirichlet ＋ コストカーブ目標の soft 受理
  5. 数%は「極端だが合法」ノイズ構成（高コスト偏重/低コストバニラ偏重）＝OOD耐性（第5巡）
制約（ルール・常に強制）: 50枚・同名≤4・リーダーと色一致。
"""
import random
from collections import Counter

import deck_gen_index as DGI


def _colors(m):
    return {getattr(x, "value", x) for x in (getattr(m, "colors", []) or [])}


class DeckGenerator:
    def __init__(self, db, k_clusters=12, seed=0):
        self.db = db
        tm, tr, mech, pool = DGI.build_indexes(db)
        self.tm, self.tr, self.mech, self.pool = tm, tr, mech, pool
        self.clusters = DGI.build_role_clusters(db, pool, k=k_clusters, seed=seed)
        self.leaders = [cid for cid in db.raw_db.keys()
                        if (m := db.get_card(cid)) is not None and m.type.name == "LEADER"]
        # 色→デッキ投入可カード（合法フィルタの前計算）
        self._by_color = {}
        for cid in pool:
            for c in _colors(db.get_card(cid)):
                self._by_color.setdefault(c, []).append(cid)

    # --- 内部ヘルパ ---
    def _legal(self, cid, lcol):
        return bool(_colors(self.db.get_card(cid)) & lcol)

    def _cost(self, cid):
        return int(getattr(self.db.get_card(cid), "cost", 0) or 0)

    def _add(self, deck, cid, n, lcol):
        """同名≤4・色一致・50枚を守って n 枚まで積む。実際に積んだ枚数を返す。"""
        if not self._legal(cid, lcol):
            return 0
        cur = deck.get(cid, 0)
        take = min(n, 4 - cur, 50 - sum(deck.values()))
        if take > 0:
            deck[cid] = cur + take
        return max(take, 0)

    def _fill_from(self, deck, cands, budget, lcol, rng, copies=(2, 4), accept=None):
        """候補群から budget 枚を目標に充填（soft 受理関数 accept 対応）。"""
        cands = [c for c in cands if self._legal(c, lcol)]
        rng.shuffle(cands)
        filled = 0
        for cid in cands:
            if filled >= budget or sum(deck.values()) >= 50:
                break
            if accept is not None and not accept(cid):
                continue
            filled += self._add(deck, cid, rng.randint(*copies), lcol)
        return filled

    # --- 生成本体 ---
    def generate(self, rng: random.Random, leader_id=None, noise_prob=0.03):
        leader_id = leader_id or self.leaders[rng.randrange(len(self.leaders))]
        lm = self.db.get_card(leader_id)
        lcol = _colors(lm)
        deck = {}

        if rng.random() < noise_prob:
            self._generate_noise(deck, lcol, rng)
        else:
            self._generate_normal(deck, leader_id, lm, lcol, rng)

        # 不足分は色合法プールから無条件で埋める（最終保証）
        if sum(deck.values()) < 50:
            all_legal = [c for col in lcol for c in self._by_color.get(col, [])]
            self._fill_from(deck, all_legal, 50 - sum(deck.values()), lcol, rng, copies=(1, 4))
        assert sum(deck.values()) == 50, f"生成失敗: {sum(deck.values())}枚 leader={leader_id}"
        return leader_id, deck

    def _generate_normal(self, deck, leader_id, lm, lcol, rng):
        # 構造パラメータのサンプル（振る舞い軸のみ・色数パラメータは無し）
        lam = rng.uniform(0.1, 0.8)          # Trait集中度
        alpha = rng.uniform(0.5, 0.9)        # members:referencers 構成比
        curve = rng.choice(["aggro", "mid", "control"])
        core_n = rng.randint(3, 8)           # コアパッケージ種数

        # 1) コアパッケージ
        core = DGI.leader_core_candidates(self.db, leader_id, self.tm, self.tr, self.mech)
        rng.shuffle(core)
        for cid in core[:core_n]:
            self._add(deck, cid, rng.randint(3, 4), lcol)

        # 2) Trait 二部サンプリング
        traits = list(getattr(lm, "traits", None) or [])
        remain = 50 - sum(deck.values())
        if traits and remain > 0:
            t = traits[rng.randrange(len(traits))]
            quota = int(lam * remain)
            n_mem = int(alpha * quota)
            self._fill_from(deck, list(self.tm.get(t, [])), n_mem, lcol, rng)
            self._fill_from(deck, list(self.tr.get(t, [])), quota - n_mem, lcol, rng)

        # 3) 役割クラスタ充填（コストカーブの soft 受理）
        lo, hi = {"aggro": (0, 4), "mid": (2, 6), "control": (4, 99)}[curve]
        def accept(cid):
            c = self._cost(cid)
            return (lo <= c <= hi) or rng.random() < 0.25   # 25%はカーブ外も許す（多様性）
        weights = [rng.random() for _ in self.clusters]
        order = sorted(self.clusters.keys(), key=lambda j: -weights[j % len(weights)])
        for j in order:
            remain = 50 - sum(deck.values())
            if remain <= 0:
                break
            self._fill_from(deck, list(self.clusters[j]), remain, lcol, rng, accept=accept)

    def _generate_noise(self, deck, lcol, rng):
        """極端だが合法な構成（OOD耐性の常設ノイズ）。"""
        mode = rng.choice(["top_heavy", "cheap_flood"])
        all_legal = [c for col in lcol for c in self._by_color.get(col, [])]
        if mode == "top_heavy":
            cands = [c for c in all_legal if self._cost(c) >= 6]
        else:
            cands = [c for c in all_legal if self._cost(c) <= 2]
        self._fill_from(deck, cands, 50, lcol, rng, copies=(3, 4))


def build_instances(db, leader_id, deck_counts, owner_id):
    """生成結果 {card_id: n} を (leader CardInstance, [CardInstance]) に実体化。"""
    from opcg_sim.src.models.models import CardInstance
    leader = CardInstance(db.get_card(leader_id), owner_id)
    cards = []
    for cid, n in deck_counts.items():
        m = db.get_card(cid)
        for _ in range(n):
            cards.append(CardInstance(m, owner_id))
    return leader, cards
