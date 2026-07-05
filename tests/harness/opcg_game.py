"""委譲shim（薄い継承）: 本番 `opcg_sim.src.learned.adapter.OPCGGame` を正とし、
研究専用の `new_game`（対局生成ヘルパ・本番は既存 manager を駆動するため不要）だけをここで足す。
"""
from opcg_sim.src.learned.adapter import OPCGGame as _BaseGame
from cpu_selfplay import build_deck, _load_db  # noqa: F401  (loader は互換のため再エクスポート)


class OPCGGame(_BaseGame):
    # --- 局面生成（gate ランナー用ヘルパ・研究専用） ---
    def new_game(self, db, seed, leaders=None):
        """seed から決定論的に対局を生成する（同一 seed→同一局面＝CRN 可能）。

        `leaders`（リーダー card_id のプール）指定時は、両席のリーダーを seed から独立に
        抽選し `deckgen.build_realistic_deck`（イベント込み・4枚積み・カーブあり）で組む
        ＝自己対戦の**盤面分布を広げる**（固定1リーダーのミラー戦だと【ドン‼×1】系の
        リーダー効果等が学習データに一度も現れない・cpu_rl_pilot の穴B）。未指定は従来の
        `build_deck`（先頭リーダー・単色自動充填）＝後方互換（既存テスト・スモークが依存）。
        """
        import random
        random.seed(seed)
        if leaders:
            import numpy as _np
            from deckgen import build_realistic_deck
            r = _np.random.default_rng(seed)
            lp1 = leaders[int(r.integers(len(leaders)))]
            lp2 = leaders[int(r.integers(len(leaders)))]
            # デッキ構築は席ごとの private rng（seed 由来）＝global random は start_game の
            # シャッフル再現用に温存する（消費順を変えず CRN を保つ）。
            l1, c1 = build_realistic_deck(db, "p1", lp1, rng=random.Random(seed * 2 + 1))
            l2, c2 = build_realistic_deck(db, "p2", lp2, rng=random.Random(seed * 2 + 2))
        else:
            l1, c1 = build_deck(db, "p1")
            l2, c2 = build_deck(db, "p2")
        from opcg_sim.src.core.gamestate import GameManager, Player
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        m.start_game()
        return m
