"""OPCG エンジンを AZ/MCTS の Game プロトコルに適合させるアダプタ（GATE B〜パイロット）。

docs/.../cpu_rl_pilot_plan_20260629.md GATE B。状態＝GameManager（可変・非hashable）ゆえ
ノード型MCTS（az_mcts_tree）と組む。手番は pending_actor_action（同一プレイヤーが連続する＝
ドン→攻撃→…→ターン終了）。遷移は実エンジン（_apply_clone＝適用＋対話ドレイン）で行い本番挙動と乖離しない。
determinize は cpu_ai._determinize_opponent（相手の伏せ手札を相手ライブラリから再サンプル＝チート除去）。

GATE B の葉価値は **固定評価器**（既定 L1 cpu_ai.evaluate を tanh で[-1,1]へ）。探索の健全性
（playout単調性）を「評価器を固定して sims だけ動かす」純粋比較で測るのが目的。policy/encode は本段では不要。
"""
import math

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from .config import VALUE_SCALE


class OPCGGame:
    # L1 生スコアは card-currency で桁が大きい（実測 中央 ~-5800・範囲[-11920,7091]）。
    # scale=10000 で tanh 飽和率0%・std0.25＝探索が勾配を使える値域（GATE B 診断で較正）。
    def __init__(self, value_scale=VALUE_SCALE, see_opp_hand=False, prune_futile=None,
                 don_margin=None, macro_moves=None, defense_box=None):
        self.value_scale = value_scale
        self.see_opp_hand = see_opp_hand
        # (C) マージン付与の席別上書き（None=cpu_ai.DON_MARGIN_ATTACH に従う・アリーナ A/B 用）
        self.don_margin = don_margin
        # マクロ手化 P1（ユーザ設計 2026-08-24）: True で原始 ATTACH_DON を配分箱
        # 「対象へk枚」に置換（None=config.SERVE_MACRO_MOVES に従う・席別 seam）
        self.macro_moves = macro_moves
        # マクロ手化 P4-c（防御箱 v1・2026-08-24）: True で防御窓の候補を D1'/D2' 支配則で
        # 整形（None=config.SERVE_DEFENSE_BOX に従う・席別 seam）
        self.defense_box = defense_box
        # v6 柱⑤（生成/serve の探索設定分離・docs/reports/v5_adoption_20260715.md §4-5）:
        # None=config の SERVE_PRUNE_FUTILE に従う（serve 既定）。自己対戦生成は False を渡して
        # 枝刈りを外す＝「刈った枝は学習データに現れない→枝刈りの誤りが学習で固定化される」
        # 自己強化盲点を断つ（探索が訪れない枝は学習できない、というループの性質への対策）。
        self.prune_futile = prune_futile

    # 注: 研究用の new_game(deck構築) は製品版では除外（本番は既存 manager を駆動するため不要）。

    # --- Game プロトコル ---
    def current_player(self, state):
        pa = state.pending_actor_action()
        return pa[0] if pa else None

    def _actor(self, state, name):
        return state.p1 if state.p1.name == name else state.p2

    def is_terminal(self, state):
        return state.winner is not None or state.pending_actor_action() is None

    def winner(self, state):
        return state.winner

    def legal_actions(self, state):
        name = self.current_player(state)
        if name is None:
            return []
        base = state.get_legal_actions(self._actor(state, name))
        # 効果選択対話では get_legal_actions は既定解決1手のみ。L1 と同じ候補ごと／
        # accept・decline の代替手を併合し、MCTS が選択肢を評価できるようにする
        # （併合しないと任意効果を常に発動・up-to効果を常に見送る配線バグになる）。
        moves = cpu_ai.merged_search_actions(state, name, base)
        # v5: 無駄攻撃（倒せない/届かない）・無意味なドン付与を候補から除外する（L1/α-β と同じ枝刈り）。
        # 学習型 MCTS の候補生成は従来これを掛けておらず、net が無駄手に visit を割いて選ぶ実害が
        # あった（v4 実測マーク @19/@102/@38＝枝刈りで直る・docs/cpu_v5_plan.md §4-1補）。
        # CPU の探索/方策のみで作用しエンジンの合法手列挙は変えない。TURN_END 等は常に残る。
        pf = self.prune_futile
        if pf is None:   # インスタンス未指定＝config（serve 既定）に従う
            from opcg_sim.src.learned.config import SERVE_PRUNE_FUTILE
            pf = SERVE_PRUNE_FUTILE
        if pf:
            moves = cpu_ai._prune_don_moves(state, name, moves, margin=self.don_margin)
            moves = cpu_ai._prune_futile_attacks(state, name, moves)
        # 旧ドン箱（SERVE_DON_BOX・don_box_candidates のリーダー限定攻撃形の合成）は
        # 純正AZ化（2026-08-25）で削除＝P2 アタック箱（下の macro_moves）が上位互換。
        mm = self.macro_moves
        if mm is None:
            from opcg_sim.src.learned.config import SERVE_MACRO_MOVES
            mm = SERVE_MACRO_MOVES
        if mm:
            # マクロ手化 P1: 原始 ATTACH_DON（1枚単位）を配分箱「対象へk枚」に置換。
            # 順序重複（同一配分への原始経路 中央値5.3x/最大9756x・macro_p0_probe）を潰す。
            # 配分箱の対象は枝刈り済み原始手から導出＝意味フィルタを継承（死に先は来ない）。
            attaches = [m for m in moves if m.get("action_type") == "ATTACH_DON"]
            if attaches:
                allocs = cpu_ai.don_alloc_candidates(state, name, attaches)
                if allocs:
                    moves = [m for m in moves
                             if m.get("action_type") != "ATTACH_DON"] + allocs
            # マクロ手化 P2: 原始 ATTACK を「（付与k→）対象Yへ攻撃」のアタック箱に置換
            # （素の攻撃は k=0 の箱として吸収）。DON_BOX の target_ids 付き＝攻撃形の
            # 重複はここで除く。
            attacks = [m for m in moves if m.get("action_type") == "ATTACK"]
            if attacks:
                atk_boxes = cpu_ai.attack_box_candidates(state, name, attacks)
                if atk_boxes:
                    moves = [m for m in moves
                             if m.get("action_type") != "ATTACK"
                             and not (m.get("action_type") == "DON_BOX"
                                      and (m.get("payload") or {}).get("target_ids"))]
                    moves = moves + atk_boxes
        dbx = self.defense_box
        if dbx is None:   # インスタンス未指定＝config（serve 既定）に従う
            from opcg_sim.src.learned.config import SERVE_DEFENSE_BOX
            dbx = SERVE_DEFENSE_BOX
        if dbx:
            # マクロ手化 P4-c: 防御窓（SELECT_COUNTER+PASS のみの窓）の候補を D1'（総量不足
            # →素通し以外を落とす）/ D2'（止まった戦闘に払わない）の支配則で整形。
            # 同一インスタンス経由の全経路（serve 窓読み出し・resolved_branch_values の
            # 内部窓・木の展開）に一様に効く。
            moves = cpu_ai.defense_box_prune(state, name, moves)
        return moves

    def apply(self, state, move, actor_name):
        """move を新クローンへ適用（対話ドレイン込み）。例外手は None（呼び出し側で除外）。"""
        return cpu_ai._apply_clone(state, actor_name, move, stop_at_select=True)

    def determinize(self, state, me_name, rng):
        """探索の世界線を固定＝相手の伏せ手札を再サンプリングしたクローンを返す（PIMC）。"""
        return cpu_ai._determinize_opponent(state, me_name, rng)

    def value(self, state, to_move):
        """葉価値∈[-1,1]（to_move 視点）。終局は ±1。途中は L1 を tanh で圧縮。"""
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        score = cpu_ai.evaluate(state, to_move, see_opp_hand=self.see_opp_hand)
        return math.tanh(score / self.value_scale)
