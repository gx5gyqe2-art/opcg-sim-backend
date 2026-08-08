"""効果対話の既定解決（`engine/interaction.py` `choose_selection`/`default_interaction_payload`）。

2026-07-30 実測欠陥の回帰ガード（必須/標準＝実プレイのゲームプレイ退行を直接見張る）:
旧既定「候補先頭から min 件・min=0 は常に0件」が
  (a) イワンコフ ST30-004 の手札破棄コストで**公開した 6000 2枚（最良札）を捨て**、
  (b) ウタ OP09-002 の「1枚まで手札に加える」を**常に見送って**いた。
この既定は探索ドレイン・`get_legal_actions` の既定解決1手・自己対戦の全 CPU 経路で使われ、
レフェリー裁定（m4@2 の cpu_out_band）も汚染していた（v24 後続調査）。

固定する性質:
  - ゾーン意味論（pure `choose_selection`）: コスト系（自手札/場）= min 件を価値昇順、
    獲得系（自山札/トラッシュ・TEMP は min=0 のみ）= max 件を価値降順、対象系（相手側）= 降順
  - 判別不能（ゾーン不明=ドン等・混在・強制 TEMP）は None＝旧既定へ退避
    （RETURN_DON の候補順細工（レスト優先）を壊さない）
  - 実盤面統合: m4@2 / m1@3 の実測欠陥がどちらも直っていること
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import argparse

import pytest

from opcg_sim.src.core.engine.interaction import choose_selection


def _e(uuid, side, zone, cost, power):
    return (uuid, side, zone, cost * 100000 + power)


def test_hand_cost_picks_cheapest_min():
    """手札破棄コスト（m4@2 の形）: min=2 を価値昇順＝6000 2枚でなく低価値2枚を払う。"""
    entries = [_e("bb1", "own", "hand", 5, 6000), _e("bb2", "own", "hand", 5, 6000),
               _e("zoro", "own", "hand", 7, 9000), _e("asl", "own", "hand", 1, 1000),
               _e("uta", "own", "hand", 1, 2000)]
    assert choose_selection(entries, 2, 2) == ["asl", "uta"]


def test_temp_upto_gain_picks_best():
    """公開一時領域の「〜まで」（m1@3 ウタの形）: 0件見送りでなく最高価値を max 件取る。"""
    entries = [_e("yasopp", "own", "temp", 3, 5000), _e("shanks4", "own", "temp", 4, 6000),
               _e("shanks9", "own", "temp", 9, 10000), _e("bb", "own", "temp", 5, 6000)]
    assert choose_selection(entries, 0, 1) == ["shanks9"]


def test_deck_trash_gain_picks_best_max():
    entries = [_e("a", "own", "trash", 2, 3000), _e("b", "own", "trash", 6, 8000)]
    assert choose_selection(entries, 1, 2) == ["b", "a"]


def test_opponent_target_picks_strongest():
    entries = [_e("small", "opp", "field", 2, 3000), _e("big", "opp", "field", 8, 9000)]
    assert choose_selection(entries, 1, 1) == ["big"]


def test_unresolvable_and_forced_temp_fall_back_to_legacy():
    """ドン（ゾーン走査外＝side None）・混在・強制 TEMP は None＝旧既定（先頭 min 件）へ。
    RETURN_DON はこの退避で候補順細工（レスト優先）が維持される。"""
    assert choose_selection([("don1", None, None, 0)], 1, 1) is None
    mixed = [_e("h", "own", "hand", 1, 1000), _e("d", "own", "deck", 1, 1000)]
    assert choose_selection(mixed, 1, 1) is None
    forced_temp = [_e("t1", "own", "temp", 1, 1000), _e("t2", "own", "temp", 2, 2000)]
    assert choose_selection(forced_temp, 1, 1) is None       # ミル系コスト誤爆の防止
    assert choose_selection([], 0, 1) is None
    assert choose_selection([_e("x", "own", "hand", 1, 0)], 0, 0) is None   # max<1


def _stub(cost=0, power=0, counter=0, triggers=()):
    class _T:
        def __init__(self, name):
            self.name = name
    class _Ab:
        def __init__(self, name):
            self.trigger = _T(name)
    class _M:
        pass
    m = _M(); m.cost = cost; m.power = power; m.counter = counter
    m.abilities = [_Ab(t) for t in triggers]
    class _C:
        pass
    c = _C(); c.master = m
    return c


def test_card_keep_value_ordinal_properties():
    """統一「残す価値」の序数性質（重みの絶対値でなく順序を固定・ユーザ指摘 2026-07-30）:
    コストだけで決めない＝カウンター/効果/防御トリガーが価値に数えられること。"""
    from opcg_sim.src.core.engine.interaction import card_keep_value as v
    # 同コストなら 効果+カウンター持ち > バニラ
    assert v(_stub(1, 1000, 2000, ("ACTIVATE_MAIN",))) > v(_stub(1, 2000, 0))
    # カウンターイベント（コスト1・パワー0）はバニラ1コストキャラより残す価値が高い
    assert v(_stub(1, 0, 0, ("COUNTER",))) > v(_stub(1, 2000, 0))
    # 実測ケースの序列: エース&サボ&ルフィ(1/1000/カウンター2000/起動) > ウタ(1/2000/1000/登場時)
    assert v(_stub(1, 1000, 2000, ("ACTIVATE_MAIN",))) > v(_stub(1, 2000, 1000, ("ON_PLAY",)))
    # 高コスト大型は依然として上位（合成が逆転を起こさない）
    assert v(_stub(9, 10000, 0)) > v(_stub(1, 1000, 2000, ("ACTIVATE_MAIN", "TRIGGER")))


@pytest.fixture(scope="module")
def _marks():
    """実測欠陥点の真盤面（m4@2・m1@3）。tests/scripts の復元機構を再利用する。"""
    import counterfactual_referee as CR
    import coach_gate as CG
    import replay_reeval as RE
    from cpu_selfplay import _load_db
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    replays = {**__import__("mark_gate").REPLAYS, **CG.REPLAYS_V2}
    CR.GAMES = {}
    boards = {}
    for tag, i in (("m4", 2), ("m1", 3)):
        raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
        CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                         rec["actions"])
        boards[(tag, i)] = CR._restore_board(db, tag, i)
    return boards


def _apply_play(boards, tag, i, card):
    """card を出し、**残った効果対話を既定解決でドレイン**した後の自分側を返す。

    v39 以降、探索の apply（`OPCGGame.apply`）は分岐可能な対象選択で停止する（CPU が選べる
    ようにするため・`test_effect_selection_wiring.py`）。本ファイルの主題は「既定解決の意味論」
    なので、停止した対話をここで `_drain_own_interactions`（既定解決そのもの）へ明示的に流す。"""
    from opcg_game import OPCGGame
    from opcg_sim.src.core import cpu_ai
    m0, who = boards[(tag, i)]
    name = who if isinstance(who, str) else who.name
    g = OPCGGame(prune_futile=False)
    mv = next(x for x in g.legal_actions(m0)
              if (cpu_ai._describe_move(m0, x) or {}).get("card") == card)
    child = g.apply(m0, mv, name)
    cpu_ai._drain_own_interactions(child, name)      # 既定解決（分岐は探索でなくヒューリスティクス）
    return child.p1 if child.p1.name == name else child.p2


def test_m4_at_2_ivankov_discard_keeps_revealed_6000s(_marks):
    """イワンコフの引き3捨て2: 公開したベン・ベックマン(6000)2枚を温存し「残す価値」最低の
    2枚を捨てる。統一価値（card_keep_value）ではエース&サボ&ルフィ OP13-007（コスト1だが
    **カウンター2000＋起動効果**）はウタ OP09-002（カウンター1000）より上＝コストだけなら
    最下位の要札が温存される（2026-07-30 ユーザ指摘の修正）。"""
    me = _apply_play(_marks, "m4", 2, "ST30-004")
    hand = [c.master.card_id for c in me.hand]
    assert hand.count("OP16-012") == 2                        # ベックマン温存
    assert "OP13-007" in hand                                 # カウンター2000+効果の要札を温存
    assert sorted(c.master.card_id for c in me.trash) == ["OP09-002", "OP09-002"]


def test_m1_at_3_uta_takes_best_revealed_card(_marks):
    """ウタの「1枚まで手札に加える」: 見送らず、公開中の最高コスト（ST23-002 10000）を加える。"""
    me = _apply_play(_marks, "m1", 3, "OP09-002")
    hand = [c.master.card_id for c in me.hand]
    assert hand.count("ST23-002") == 2                        # 元々の1枚＋効果で追加
    assert not me.trash
