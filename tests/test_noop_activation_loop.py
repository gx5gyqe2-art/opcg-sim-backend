"""「何も起きない手」を無限に打てる経路の回帰テスト（2026-08-15。A'/E は 2026-08-16 追加）。

生成デッキ監査（`tests/scripts/deck_synth_audit.py`）が 137 リーダー中 3 リーダーで
「上限手数まで終わらない」対局を検出した。原因はいずれも**盤面が 1 ミリも動かない手を
エンジンが合法手として出し続ける**ことだった。固定ハンニャバルデッキ（ステージ0・
イベント0・特徴の絡む効果ほぼ無し）では 1 度も通らない経路で、歴代のアリーナ／ゲート／
自己対戦では検出できなかった実バグ。**交差対面**（同監査の `--cross`）へ広げると、
ミラーでは出ない A'（コストの空振り・別種）と E（再計算がコスト確認を訊く）も出た。

  A. レストコストの候補にレスト済みカードが混ざる（OP10-083 光月モモの助）
     支払い可否（`_can_satisfy_node`）はレスト済みを候補から外すのに、実際の支払い
     （`_resolve_targets`）は外していなかった＝判定と支払いの不一致。レスト済みリーダーを
     選んで「支払い」が no-op になり、何も消費しないまま起動メインを無限再起動できた。

  B. 継続効果の再計算が対象選択の対話を出す（OP05-097 聖地マリージョア）
     「コスト2以上の特徴《天竜人》を持つキャラの支払うコストは1少なくなる」の対象が手札に
     2枚以上あると、静的効果なのに「どれに掛けるか」を訊いていた。再計算は盤面が動くたびに
     走るため、解決しても次の再計算がまた同じ対話を出す＝無限ループ。

  C. 効果が完全な空振りの起動メイン（EB04-016 トリ／OP10-030 スモーカー）
     「ドン!!1枚までをアクティブにする。その後、このターン中キャラの効果でドン!!を
     アクティブにできない」。2回目以降は自己制限で完全な no-op だが、コストもターン制限も
     無いため何も消費せず、合法手に出続けていた。

  E. 継続効果の再計算がコスト確認の対話を出す（OP09-080 サウザンド・サニー号・2026-08-16）
     「【相手のターン中】このステージをレストにできる：…場を離れた時、…」の「離れた時」が
     反応型と判定されず、常時効果として再計算のたびに評価されていた。任意コストなので
     「使用しますか？」を訊き、拒否しても使用回数も盤面も変わらないため次の再計算がまた
     同じ問いを立てる＝無限ループ（交差対面の対局で 613 回連続）。B と同じ「再計算は
     対話を出さない」原則で、こちらはコスト確認側。

  A'. 状態を変えるコストが「既にその状態」でも払える（OP15-099 ウルージ・2026-08-16）
     「自分のライフの上から1枚を裏向きにできる」は表向きのライフが要る。ライフが全て裏向き
     でも支払い可能と判定され、裏向きを裏向きにする空振りで何も消費せず起動メインを撃ち
     続けられた（交差対面の対局で 297 回連続）。A と同じ「判定と支払いの不一致」で、
     レスト以外にも同じ穴があったことを示す＝規則を `_cost_state_noop` に一本化した。

  D. 複合コストから「自分自身の分」が落ちる（パーサ・9枚）
     「この（カード/キャラ）と〈X〉を レスト／トラッシュ／デッキの下 にできる：」の自身の分が
     脱落し、コストが実質「X だけ」に化けていた。発生源が場に残るので起動メインを撃ち続け
     られる（EB01-030 ローグタウン＝手札1枚→2ドローの永久機関、OP13-099 虚の玉座、
     OP10-083/087/088/091、OP10-026/027、OP04-073）。A の「支払いが空振り」とは別で、
     こちらは**そもそもコストが安すぎる**という忠実度の欠陥。
"""
import functools

import conftest  # noqa: F401

from engine_helpers import make_game, make_master, make_instance
from opcg_sim.src.core.effects.parser import EffectParser
from opcg_sim.src.models.enums import CardType, TriggerType
from opcg_sim.src.models.models import CardInstance, DonInstance


def _ability(text, idx=0):
    return EffectParser().parse_card_text(text)[idx]


@functools.lru_cache(maxsize=1)
def _db():
    from game_driver import load_db
    return load_db()


def _real(card_id, owner="P1"):
    """出荷カード DB の master をそのまま使う（テキストを書き写さず実物で検証する）。"""
    return CardInstance(master=_db().get_card(card_id), owner_id=owner)


def _main_ability(card_inst):
    return next(ab for ab in card_inst.master.abilities
                if ab.trigger == TriggerType.ACTIVATE_MAIN)


# --- A. レストコストの候補はアクティブのみ ------------------------------------------
MOMO_TEXT = (
    "【起動メイン】このキャラと自分の特徴《ドレスローザ》を持つ、リーダーかステージ1枚を、"
    "レストにできる：相手のキャラ1枚までを、このターン中、コスト-2。"
)


def _momo_board():
    """アクティブなリーダーとレスト済みステージ（ともに特徴《ドレスローザ》）＋モモの助の場。

    コストの相方は「自分の特徴《ドレスローザ》を持つリーダーかステージ1枚」。レスト済みの
    ステージは支払いに使えないので、候補はアクティブなリーダー 1 枚に確定するのが正。
    """
    gm, p1, p2 = make_game()
    p1.leader = make_instance(
        make_master(card_id="OP12-081", name="コアラ", type=CardType.LEADER,
                    traits=["ドレスローザ", "革命軍"], life=5), owner="P1")
    p1.stage = make_instance(
        make_master(card_id="S-001", name="テストステージ", type=CardType.STAGE,
                    traits=["ドレスローザ"]), owner="P1")
    p1.stage.is_rest = True                       # レスト済み＝コストに使えない
    src = _real("OP10-083")
    assert MOMO_TEXT[:16] in src.master.effect_text            # 実物のテキストで検証している
    ab = _main_ability(src)
    p1.field = [src]
    gm.turn_player = p1
    return gm, p1, src, ab


def test_compound_cost_keeps_the_source_leg():
    """「このキャラと〜を、レストにできる」のコストは**自身のレストを落とさない**
    （落ちると発生源が場に残り続け、何度でも撃てる無限機関になる）。"""
    _gm, _p1, src, ab = _momo_board()
    legs = ab.cost.actions
    assert len(legs) == 2
    assert legs[0].target.ref_id == "self"                     # 自身のレスト
    assert legs[1].target.traits == ["ドレスローザ"]            # 相方のレスト


def test_rest_cost_selection_excludes_rested_cards():
    """レストコストの解決はレスト済みのカードを候補にしない＝支払いが空振りしない。"""
    gm, p1, src, ab = _momo_board()
    from opcg_sim.src.core.effects.resolver import EffectResolver
    res = EffectResolver(gm)
    other = ab.cost.actions[1]                     # コスト = Sequence[自身レスト, 相方レスト]
    chosen = res._resolve_targets(p1, other.target, src, action_node=other)
    assert gm.active_interaction is None           # 候補が1枚に確定＝対話は起きない
    assert [c.uuid for c in chosen] == [p1.leader.uuid]   # レスト済みステージは候補から外れる


def test_rest_cost_activation_consumes_the_source_and_cannot_repeat():
    """起動するとモモの助自身とリーダーがレストになり、同じ起動メインは 2 度は撃てない。"""
    gm, p1, src, ab = _momo_board()
    assert gm._has_activatable_main(src, p1) is True
    gm.resolve_ability(p1, ab, source_card=src)
    assert src.is_rest is True                      # コストが実際に支払われた
    assert p1.leader.is_rest is True
    assert gm._has_activatable_main(src, p1) is False


UROUGE_TEXT = "【起動メイン】自分のライフの上から1枚を裏向きにできる"


def _urouge_board(face_up_lives=0):
    """ウルージ（OP15-099）と、指定枚数だけ表向きのライフを持つ盤面。"""
    gm, p1, p2 = make_game()
    src = _real("OP15-099")
    assert UROUGE_TEXT[:18] in src.master.effect_text            # 実物のテキストで検証している
    p1.field = [src]
    p1.life = [make_instance(make_master(card_id=f"L-{i}", name=f"ライフ{i}"), owner="P1")
               for i in range(3)]
    for i in range(face_up_lives):
        p1.life[i].is_face_up = True
    p1.don_rested.append(DonInstance(owner_id="P1"))             # 効果自体は空振りではない
    gm.turn_player = p1
    return gm, p1, src, _main_ability(src)


def test_face_down_life_cost_is_unpayable_when_all_lives_are_face_down():
    """「ライフの上から1枚を裏向きにできる」は**表向きのライフ**が無いと払えない。

    払えることになっていると、裏向きのライフを何度も「裏向きに」して何も消費せず、
    起動メインを無限に撃てる（交差対面の対局で 297 回連続＝上限手数で終わらない）。
    """
    gm, p1, src, _ab = _urouge_board(face_up_lives=0)
    assert gm._has_activatable_main(src, p1) is False


def test_face_down_life_cost_is_paid_once_and_then_unpayable():
    """表向きのライフがあるときだけ撃て、支払うと同じ手はもう撃てない。"""
    gm, p1, src, ab = _urouge_board(face_up_lives=1)
    assert gm._has_activatable_main(src, p1) is True
    gm.resolve_ability(p1, ab, source_card=src)
    assert p1.life[0].is_face_up is False        # コストが実際に支払われた（裏向きになった）
    assert gm._has_activatable_main(src, p1) is False


ROGUETOWN_TEXT = "【起動メイン】このカードと自分の手札1枚を好きな順番でデッキの下に置くことができる"


def test_stage_deck_bottom_cost_consumes_the_stage_itself():
    """「このカードと手札1枚をデッキの下に置く：2枚引く」はステージ自身も下へ送る。

    自身の分が落ちるとステージが場に残り、手札1枚→2ドロー＝**毎回+1枚の永久機関**になる
    （EB01-030 ローグタウン。生成デッキ監査で1局が90秒を超えた原因）。
    """
    gm, p1, p2 = make_game()
    stage = _real("EB01-030")
    assert ROGUETOWN_TEXT[:20] in stage.master.effect_text
    p1.stage = stage
    p1.hand = [make_instance(make_master(card_id="H-0", name="手札0"), owner="P1")]
    p1.deck = [make_instance(make_master(card_id=f"D-{i}", name=f"デッキ{i}"), owner="P1")
               for i in range(5)]
    gm.turn_player = p1
    ab = _main_ability(stage)
    assert gm._has_activatable_main(stage, p1) is True
    gm.resolve_ability(p1, ab, source_card=stage)
    assert p1.stage is None                        # ステージ自身がデッキの下へ去った
    assert len(p1.hand) == 2                       # 手札1枚を下に置き、2枚引いた
    assert p1.deck[-1] is stage or stage in p1.deck


# --- B. 継続効果の再計算は対話を出さない ---------------------------------------------
MARIEJOA_TEXT = (
    "【自分のターン中】自分が手札から登場させるコスト2以上の特徴《天竜人》を持つ"
    "キャラカードの支払うコストは1少なくなる。"
)


def test_passive_recalc_applies_to_all_candidates_without_dialog():
    """静的なコスト軽減は手札の該当カード全部に掛かる（どれに掛けるかを訊かない）。"""
    gm, p1, p2 = make_game()
    p1.stage = _real("OP05-097")
    assert MARIEJOA_TEXT[:20] in p1.stage.master.effect_text   # 実物のテキストで検証している
    hand = [make_instance(make_master(card_id=f"H-{i}", name=f"天竜人{i}",
                                      type=CardType.CHARACTER, cost=3, traits=["天竜人"]),
                          owner="P1") for i in range(2)]
    p1.hand = list(hand)
    gm.turn_player = p1
    gm._apply_passive_effects(p1)
    assert gm.active_interaction is None            # 継続効果は対象選択を伴わない
    assert [c.cost_buff for c in hand] == [-1, -1]  # 該当カード全部が軽減される


# --- C. 空振りが確定している起動メインは合法手に出さない ------------------------------
TORI_TEXT = (
    "【起動メイン】自分のドン!!1枚までを、アクティブにする。"
    "その後、自分は、このターン中、キャラの効果でドン!!をアクティブにできない。"
)


def _tori_board():
    gm, p1, p2 = make_game()
    src = _real("EB04-016")
    assert TORI_TEXT[:24] in src.master.effect_text            # 実物のテキストで検証している
    ab = _main_ability(src)
    p1.field = [src]
    gm.turn_player = p1
    return gm, p1, src, ab


def test_active_don_main_stops_being_legal_after_one_empty_activation():
    """レスト中のドン!!が無くても 1 回目は合法（自己制限を課すのは盤面変化）。
    2 回目以降は制限も既に有効＝どの枝も完全な空振りなので合法手から外れる。"""
    gm, p1, src, ab = _tori_board()
    assert gm._has_activatable_main(src, p1) is True
    gm.resolve_ability(p1, ab, source_card=src)
    assert gm._has_activatable_main(src, p1) is False


SUNNY_TEXT = (
    "【相手のターン中】このステージをレストにできる:自分の特徴《麦わらの一味》を持つキャラが"
    "相手の効果で場を離れた時、"
)


def _sunny_board():
    """相手（＝非ターンプレイヤー）の場に OP09-080 サウザンド・サニー号が立っている盤面。"""
    gm, p1, p2 = make_game()
    stage = _real("OP09-080", owner="P2")
    assert SUNNY_TEXT[:20] in stage.master.effect_text          # 実物のテキストで検証している
    p2.stage = stage
    gm.turn_player = p1                                          # p2 から見て「相手のターン中」
    return gm, p1, p2, stage


def test_leave_field_trigger_is_not_a_continuous_effect():
    """「…場を離れた時、」はイベント誘発＝再計算ループで実行してはいけない反応型。"""
    _gm, _p1, _p2, stage = _sunny_board()
    gm = _gm
    ab = next(ab for ab in stage.master.abilities
              if ab.trigger == TriggerType.OPPONENT_TURN)
    assert gm._is_reactive_passive(ab) is True


def test_passive_recalc_never_asks_to_pay_a_cost():
    """継続効果の再計算は**コスト確認の対話を出さない**（出すと無限ループになる）。

    拒否しても使用回数は減らず盤面も変わらないので、次の再計算がまた同じ問いを立てる。
    OP09-080 は「離れた時」が反応型と判定されず継続効果として毎回評価されており、
    生成デッキの交差対面で 613 回連続の同一確認＝上限手数まで終わらない対局になっていた。
    """
    gm, p1, p2, stage = _sunny_board()
    gm._apply_passive_effects(p1)
    assert gm.active_interaction is None        # 問い合わせが立たない
    assert stage.is_rest is False               # 勝手にコストも払わない


def test_active_don_main_is_illegal_after_self_restriction():
    """1回撃つと自己制限が付き、2回目以降は完全な no-op なので合法手から外れる。"""
    gm, p1, src, ab = _tori_board()
    p1.don_rested.extend(DonInstance(owner_id="P1") for _ in range(2))
    gm.resolve_ability(p1, ab, source_card=src)
    assert len(p1.don_active) == 1                  # 1枚アクティブになった
    assert gm._active_restriction(p1, "CANNOT_ACTIVATE_DON") is not None
    assert len(p1.don_rested) == 1                  # まだレスト中のドン!!は残っている
    assert gm._has_activatable_main(src, p1) is False   # それでも撃つ意味は無い＝除外
