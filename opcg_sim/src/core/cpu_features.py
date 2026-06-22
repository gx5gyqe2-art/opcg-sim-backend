"""学習価値関数（§2.5.7 残5・GBDT/線形 価値葉）の **特徴抽出**（stdlib-only・PyPy/CPython 両対応）。

目的: 静的葉MCTSの構造的な穴＝**状況札の温存価値（option value）**を一般理論で埋めるため、盤面＋手札を
固定長の特徴ベクトルへ落とす。学習した価値関数（`cpu_value_model`）がこの特徴から勝率を推定し、葉評価に
ブレンドされる。**「手札の答え在庫（除去/カウンター/ブロッカー）× 相手盤面の脅威」の交互作用項**を明示的に
入れることで、線形モデルでも option value（＝相手が脅威を並べた時に答えを握っている価値）を捉えられる。

設計上の制約（重要）:
  - **stdlib-only**（lightgbm/numpy を持ち込まない）＝探索の PyPy ワーカーでそのまま動く。
  - **葉で数千回呼ばれる**＝安価な状態読みのみ（パワー/枚数/ライフ/ドン/カウンター値/キーワード）。
    カードテキストの重い走査は**マスタ単位で1回だけ**判定してキャッシュ（`_REMOVAL_CACHE`）。
  - **決定論・manager 非破壊**（読み取りのみ）。`see_opp_hand=False`（既定）は相手手札の中身を読まない
    （枚数のみ＝公開情報＝公平モードの葉と整合）。

`evaluate` と同じ視点規約: `get_power(is_turn)` の `is_turn` は「そのカードの持ち主の手番か」。
"""
from typing import Any, Dict, List

from ..models.enums import ActionType

# 相手キャラを盤面から除去する効果（ハード除去）。手札の「答え在庫」判定に使う。
_REMOVAL_TYPES = frozenset({
    ActionType.KO, ActionType.BOUNCE, ActionType.MOVE_TO_HAND, ActionType.DECK_BOTTOM,
})
_BLOCKER_KW = "ブロッカー"
# 攻撃的キーワード脅威（手作り評価 `_threat_value` の `_KEYWORD_ASSETS`＋アンブロッカブルに対応）。
# 場のキャラが持つ「攻め脅威性」を数で捉える＝学習が脅威キーワードの価値を表現できるようにする。
_THREAT_KW = ("ダブルアタック", "速攻", "アンブロッカブル", "ブロック不可", "バニッシュ")
_RUSH_KW = "速攻"

# マスタ（card_id）単位の「除去効果を持つか」キャッシュ（テキスト走査は1回だけ）。
_REMOVAL_CACHE: Dict[str, bool] = {}


def _walk_actions(node, out: List[Any]) -> None:
    """効果ツリー（GameAction/Sequence/Branch/Choice）を再帰走査し ActionType を集める（getattr で疎結合）。"""
    if node is None:
        return
    t = getattr(node, "type", None)
    if t is not None:
        out.append(t)
    for attr in ("actions", "options"):
        seq = getattr(node, attr, None)
        if seq:
            for a in seq:
                _walk_actions(a, out)
    for attr in ("if_true", "if_false", "effect", "cost", "sub_effect"):
        sub = getattr(node, attr, None)
        if sub is not None:
            _walk_actions(sub, out)


def _is_removal_master(master) -> bool:
    """このカードマスタが（相手）除去効果を持つか。card_id 単位でキャッシュ。"""
    cid = getattr(master, "card_id", None)
    if cid is not None and cid in _REMOVAL_CACHE:
        return _REMOVAL_CACHE[cid]
    found = False
    for ab in (getattr(master, "abilities", None) or []):
        acts: List[Any] = []
        _walk_actions(getattr(ab, "effect", None), acts)
        _walk_actions(getattr(ab, "cost", None), acts)
        if any(a in _REMOVAL_TYPES for a in acts):
            found = True
            break
    if cid is not None:
        _REMOVAL_CACHE[cid] = found
    return found


def _other(manager, me_name: str):
    return manager.p2 if manager.p1.name == me_name else manager.p1


# 特徴名（順序＝特徴ベクトルのインデックス）。モデルの重みとこの順序が対応する。
FEATURE_NAMES: List[str] = [
    # --- ライフ（最重要資源） ---
    "life_me", "life_opp", "life_diff",
    # 非線形ライフ（薄域の高限界価値）。手作り評価の膝カーブ（W_LIFE_LOW・膝=2）に対応＝線形 life_* だけ
    # では表現できない「薄いほど 1 枚が高い」を min(life,2) のバケットで近似（concave の片側区分）。
    "life_thin_me", "life_thin_opp",
    # --- デッキ危険域（デッキアウト近接・非線形）。手作り評価 W_DECK_DANGER/DECK_DANGER に対応。 ---
    "deck_danger_me", "deck_danger_opp",      # max(0, 4 - deck_n)
    # --- ドン経済 ---
    "don_active_me", "don_rested_me", "don_deck_me", "turn_count", "is_my_turn",
    # --- 盤面 ---
    "field_n_me", "field_n_opp",
    "field_pow_me_k", "field_pow_opp_k",      # 合計パワー /1000
    "leader_pow_me_k", "leader_pow_opp_k",    # 有効パワー /1000
    "blocker_n_me", "blocker_n_opp",
    "rested_n_me", "rested_n_opp",
    # 実攻撃可能体数（召喚酔いを除く＝手作り評価 W_ATTACKER と同じ「このターン/次に攻撃できる体」）。
    # 付与ドン/展開の「将来の攻め圧へ繋がる」準備手価値を表す土台。
    "attacker_n_me", "attacker_n_opp",
    # 脅威キーワード体数（攻め脅威）。手作り評価 _threat_value（ダブルアタック/速攻/アンブロッカブル/
    # バニッシュ）に対応＝学習が脅威の価値を捉えられるようにする。
    "threat_n_me", "threat_n_opp",
    # ステージ（永続リソース）。手作り評価 W_STAGE_COUNT に対応。
    "stage_me", "stage_opp",
    # --- 手札（答え在庫） ---
    "hand_n_me", "hand_n_opp",
    "hand_counter_total_me_k", "hand_counter_cards_me",
    "hand_removal_me", "hand_blocker_me", "hand_event_me", "hand_char_me",
    # --- option value 交互作用（答え在庫 × 相手脅威） ---
    "removal_x_oppfield",     # 除去在庫 × 相手場の体数
    "blocker_x_oppfield",     # ブロッカー在庫 × 相手場の体数
    "counter_x_opppow",       # カウンター総量 × 相手場の合計パワー
    "bias",                   # 定数項（線形モデル用）
]
N_FEATURES = len(FEATURE_NAMES)


def extract_features(manager, me_name: str, see_opp_hand: bool = False) -> List[float]:
    """`me_name` 視点の固定長特徴ベクトル（`FEATURE_NAMES` 順）。読み取りのみ・決定論。

    `see_opp_hand=False`（既定）は相手手札の**中身を読まない**（枚数のみ＝公平モードの葉と整合）。
    """
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = _other(manager, me_name)
    is_my_turn = (getattr(manager, "turn_player", None) is not None
                  and manager.turn_player.name == me_name)

    def field_pow(player, own_turn):
        return sum(c.get_power(own_turn) for c in player.field)

    def leader_pow(player, own_turn):
        return player.leader.get_power(own_turn) if player.leader else 0

    def blockers(player):
        return sum(1 for c in player.field if (not c.is_rest) and c.has_keyword(_BLOCKER_KW))

    def rested(player):
        return sum(1 for c in player.field if c.is_rest)

    def attackers(player, own_turn):
        """このターン（own_turn=True）/次の攻撃で実際に攻撃できるアクティブ体数。

        手作り評価 `W_ATTACKER` と同じく、自分の手番に出したばかりの体（召喚酔い・速攻なし）は
        今攻撃できないので数えない。相手の手番（own_turn=False）から見た自分の体は酔いが解けている。
        """
        n = 0
        for c in player.field:
            if c.is_rest:
                continue
            sick = own_turn and getattr(c, "is_newly_played", False) and not c.has_keyword(_RUSH_KW)
            if not sick:
                n += 1
        return n

    def threats(player):
        n = 0
        for c in player.field:
            try:
                if any(c.has_keyword(kw) for kw in _THREAT_KW):
                    n += 1
            except Exception:
                pass
        return n

    life_me, life_opp = len(me.life), len(opp.life)
    field_n_me, field_n_opp = len(me.field), len(opp.field)
    field_pow_opp_k = field_pow(opp, not is_my_turn) / 1000.0

    # 手札（答え在庫）。自分は実物・相手は枚数のみ（see_opp_hand=False）。
    hand_counter_total = sum((c.current_counter or 0) for c in me.hand)
    hand_counter_cards = sum(1 for c in me.hand if (c.current_counter or 0) > 0)
    hand_removal = sum(1 for c in me.hand if _is_removal_master(c.master))
    hand_blocker = sum(1 for c in me.hand
                       if _BLOCKER_KW in (getattr(c.master, "keywords", None) or []))
    hand_char = sum(1 for c in me.hand if c.master.type.name == "CHARACTER")
    hand_event = sum(1 for c in me.hand if c.master.type.name == "EVENT")

    feats = {
        "life_me": float(life_me),
        "life_opp": float(life_opp),
        "life_diff": float(life_me - life_opp),
        "life_thin_me": float(min(life_me, 2)),
        "life_thin_opp": float(min(life_opp, 2)),
        "deck_danger_me": float(max(0, 4 - len(me.deck))),
        "deck_danger_opp": float(max(0, 4 - len(opp.deck))),
        "don_active_me": float(len(me.don_active)),
        "don_rested_me": float(len(me.don_rested)),
        "don_deck_me": float(len(me.don_deck)),
        "turn_count": float(getattr(manager, "turn_count", 0)),
        "is_my_turn": 1.0 if is_my_turn else 0.0,
        "field_n_me": float(field_n_me),
        "field_n_opp": float(field_n_opp),
        "field_pow_me_k": field_pow(me, is_my_turn) / 1000.0,
        "field_pow_opp_k": field_pow_opp_k,
        "leader_pow_me_k": leader_pow(me, is_my_turn) / 1000.0,
        "leader_pow_opp_k": leader_pow(opp, not is_my_turn) / 1000.0,
        "blocker_n_me": float(blockers(me)),
        "blocker_n_opp": float(blockers(opp)),
        "rested_n_me": float(rested(me)),
        "rested_n_opp": float(rested(opp)),
        "attacker_n_me": float(attackers(me, is_my_turn)),
        "attacker_n_opp": float(attackers(opp, not is_my_turn)),
        "threat_n_me": float(threats(me)),
        "threat_n_opp": float(threats(opp)),
        "stage_me": 1.0 if getattr(me, "stage", None) is not None else 0.0,
        "stage_opp": 1.0 if getattr(opp, "stage", None) is not None else 0.0,
        "hand_n_me": float(len(me.hand)),
        "hand_n_opp": float(len(opp.hand)),
        "hand_counter_total_me_k": hand_counter_total / 1000.0,
        "hand_counter_cards_me": float(hand_counter_cards),
        "hand_removal_me": float(hand_removal),
        "hand_blocker_me": float(hand_blocker),
        "hand_event_me": float(hand_event),
        "hand_char_me": float(hand_char),
        "removal_x_oppfield": float(hand_removal * field_n_opp),
        "blocker_x_oppfield": float(hand_blocker * field_n_opp),
        "counter_x_opppow": (hand_counter_total / 1000.0) * field_pow_opp_k,
        "bias": 1.0,
    }
    return [feats[name] for name in FEATURE_NAMES]
