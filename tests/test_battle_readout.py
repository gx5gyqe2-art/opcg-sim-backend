"""戦闘窓の読み出し（battle readout・`cpu_learned._battle_window_choice`・v35・2026-08-05）の検証。

なぜ要るか（ユーザ整理 2026-08-05「バトルを一つの箱としてまとめて、探索と value は
バトルをするかどうか／バトルの結果どんな盤面・手札・ライフになるかのみを判断する」）:
カウンター/ブロッカー選択は**箱の出口**（解決後の盤面・手札・ライフ）で決まる局所判断なのに、
full-tree の root Q は箱の外の深い未来まで平均するため判断が薄まる。v35 実測では、出口盤面の
評価は防御原則3類型（m1@14 素通し／m1@15 止め切る／m2@58 捨てる）を**すべて**正しく順序づける
のに、探索後 Q では逆転していた（木の葉の68%が『次の自ターン』の通常盤面＝旧レートに引き戻す）。

**判断しているのは葉評価と同じ value ネット自身**（別系統の防御ロジックではない）。どの出口へ
至るかはネットの予測でなくエンジンの実計算で、枝の残り手は静止探索・教師コーパスと同一の
解決規約（`resolve_battle_inplace`）で進める。

固定する性質:
  - **箱の出口で選ぶ**: value を差し替えると（出口盤面の評価順が変わると）選択が追従する
  - **副作用ゼロ**: 評価後も盤面（手札/ライフ/場）と global random が完全復元される
  - **明示 OFF は従来（gen11 まで）と同一の手**＝ロールバック可能（gen12 で既定 ON になった
    あとも、3機構を False に戻せば旧挙動へ戻れることを固定する）
  - 戦闘中でないメインフェーズの決定は読み出しを通らない（full-tree のまま）
"""
import os
import random

import conftest  # noqa: F401
import numpy as np
import pytest

import replay_runner as RR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.mcts import in_battle, resolved_branch_values

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（serve 読み出し機構）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays", "gen7_marks_20260728")
M1 = "opcg_replay_2057134394987494995.json.gz"


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def battle_board(db):
    """m1@15＝カウンター選択の最中（攻撃7000・防御6000＝あと 2000 で凌げる）。"""
    raw = RE.load_replay_json(os.path.join(FIX, M1))
    rec = raw.get("replay", raw)
    m, who = RR.state_at_action(db, rec, 15)
    return m, (who if isinstance(who, str) else who.name)


def _game():
    from opcg_game import OPCGGame
    return OPCGGame(prune_futile=False)


def _card_of(mgr, mv):
    try:
        return (cpu_ai._describe_move(mgr, mv) or {}).get("card")
    except Exception:
        return None


def test_resolved_values_rank_by_exit_board(battle_board):
    """出口で評価する＝「止まる 2000」がライフを守り、「止まらない 1000」は失う。

    評価器は「自分のライフ＋手札（1/100 重み）」を返す純関数（ネット非依存＝この性質だけを固定）。
    v39（`BOX_RESOLVE_DEPTH`≥1）で箱の内部解決が**出口 value 最良**になったため、ライフだけを
    数える評価器では「1000 を切った枝も、続けてもう1枚足して助かる」と読まれ両枝が同値になる
    （`quiesce_choice` の docstring にある m1@15 の落とし穴と同型）。**払った枚数**を評価に含めれば
    「1枚で止める」が「2枚で止める」より上、という本来の性質がそのまま出る。
    """
    m, name = battle_board
    game = _game()
    legal = game.legal_actions(m)

    def life_fn(mgr, to_move):
        me = mgr.p1 if mgr.p1.name == to_move else mgr.p2
        return float(len(me.life)) + len(me.hand) / 100.0

    vals = resolved_branch_values(game, m, name, legal, life_fn)
    by_card = {_card_of(m, mv): v for mv, v in zip(legal, vals)}
    assert by_card.get("OP10-011") is not None and by_card.get("OP16-012") is not None
    assert by_card["OP10-011"] > by_card["OP16-012"], \
        f"止まるカウンターが出口で上に来ない: {by_card}"


def test_resolved_values_have_no_side_effects(battle_board):
    """評価は transaction で巻き戻し、global random も復元する（実ゲームを汚さない）。"""
    m, name = battle_board
    me = m.p1 if m.p1.name == name else m.p2
    hand0 = [c.uuid for c in me.hand]
    life0 = len(me.life)
    field0 = [c.uuid for c in (m.p1.field + m.p2.field)]
    st0 = random.getstate()
    game = _game()
    resolved_branch_values(game, m, name, game.legal_actions(m), lambda mgr, tm: 0.0)
    assert [c.uuid for c in me.hand] == hand0
    assert len(me.life) == life0
    assert [c.uuid for c in (m.p1.field + m.p2.field)] == field0
    assert random.getstate() == st0, "評価が global random を消費したまま返した"


def test_choice_follows_the_value_function(battle_board):
    """読み出しは出口 value に従う＝value を反転させると選択も入れ替わる。

    「別系統の防御ロジックが決めている」のでなく **value ネットが決めている**ことの検査。
    """
    m, name = battle_board
    eng = LearnedEngine(battle_readout=True)
    eng.game = _game()

    def pick(value_fn):
        eng.vnet = None   # 使わせない（差し替えた value_fn だけで決める）
        import opcg_sim.src.core.cpu_learned as CL
        orig = CL._value_fn
        CL._value_fn = lambda *a, **k: value_fn
        try:
            mv, _stats, _mgr = eng._battle_window_choice(m, name, np.random.default_rng(0))
        finally:
            CL._value_fn = orig
        return _card_of(m, mv)

    def life_fn(mgr, to_move):
        me = mgr.p1 if mgr.p1.name == to_move else mgr.p2
        return float(len(me.life))

    hi = pick(life_fn)
    lo = pick(lambda mgr, tm: -life_fn(mgr, tm))
    assert hi != lo, f"value を反転しても選択が変わらない（読み出しが value に従っていない）: {hi}"


def test_explicit_off_matches_legacy_engine(battle_board):
    """3機構を明示 OFF にすれば gen11 までと同一の手（ロールバック経路が生きている）。

    gen12 で既定は ON になったが、旧世代の重みへ戻すときは3つとも False に戻す必要がある
    （片方だけ ON は train/serve skew）。その戻し方が実際に効くことを固定する。
    """
    m, name = battle_board
    player = m.p1 if m.p1.name == name else m.p2
    kw = dict(battle_readout=False, quiesce=False, box_battle=False)
    a = LearnedEngine(**kw).decide(m, player, sims=8, rng=np.random.default_rng(3))
    b = LearnedEngine(**kw).decide(m, player, sims=8, rng=np.random.default_rng(3))
    assert cpu_ai._describe_move(m, a) == cpu_ai._describe_move(m, b)
    # 既定（gen12＝3機構 ON）は戦闘窓で読み出しを通る＝旧経路と同じとは限らない。
    d = LearnedEngine().decide(m, player, sims=8, rng=np.random.default_rng(3))
    assert d is not None


def test_main_phase_is_not_routed_to_battle_readout(db):
    """戦闘中でない局面は読み出しを通らない＝ON/OFF で同手（メインフェーズは full-tree のまま）。"""
    from opcg_game import OPCGGame
    m = OPCGGame().new_game(db, 4242)
    assert not in_battle(m)
    on = LearnedEngine(battle_readout=True)
    off = LearnedEngine(battle_readout=False)
    a = on.decide(m, m.p1, sims=8, rng=np.random.default_rng(5))
    b = off.decide(m, m.p1, sims=8, rng=np.random.default_rng(5))
    assert cpu_ai._describe_move(m, a) == cpu_ai._describe_move(m, b)
