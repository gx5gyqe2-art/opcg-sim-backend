"""自分の効果対話が**探索の決定点**として現れることの検証（v39・2026-08-06）。

なぜ要るか（実測 m2@44・2026-08-06）: 学習型CPUの遷移は `adapter.OPCGGame.apply`
→ `cpu_ai._apply_clone` で、これが手を適用したあと **actor 自身の効果対話を既定解決で
ドレイン**していた。結果、「相手がアタックした時、リーダーかキャラ1枚を−1000」（OP09-001
シャンクス）の**対象**を CPU が一度も選べず、既定ヒューリスティクス（パワー最大）が
攻撃していないキャラを選ぶ＝**攻撃を止められる唯一の手段を使い損なう**という系統的悪手に
なっていた（実測: 攻撃側ナミ7000 に当てれば 6000 でアタック不成立なのに、無関係のロビン
8000 に当てて被弾）。`merged_search_actions` は当初からこの分岐を候補化する設計だったが、
**apply 側がドレインしてしまうので保留が立たず**、候補化が働くのは「相手の手で立った対話」
だけだった（配線の不整合）。

出口 value の箱（`resolved_branch_values`）は、選べさえすれば正しく並べる（実測: 攻撃者へ
−1000 が +0.307、他は −0.02）。つまり判断力ではなく**選択肢の提示**が欠けていた。

固定する性質:
  - 自分の手で立った効果選択が、次の `legal_actions` に**選択肢ごとの手**として現れる
  - 選んだ対象が実際に適用される（既定解決に上書きされない）
  - 対象選択の余地が無い対話は従来どおり自動解決（決定点を無駄に増やさない）
"""
import argparse
import os

import conftest  # noqa: F401
import pytest

import coach_gate as CG
import counterfactual_referee as CR
import mark_gate as MG
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn
from opcg_sim.src.learned.mcts import in_battle, resolved_branch_values

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（探索の配線）

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "replays")


@pytest.fixture(scope="module")
def m244():
    """m2@44（ナミ vs シャンクス・ターン8）の実盤面と、そこから攻撃を宣言した局面。"""
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2}
    raw = RE.load_replay_json(replays["m2"]); rec = raw.get("replay", raw)
    CR.GAMES = {"m2": (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                       rec["actions"])}
    built = CR._restore_board(db, "m2", 44)
    if isinstance(built, str):
        pytest.skip(f"盤面復元不可: {built}")
    m0, who = built
    name = who if isinstance(who, str) else who.name
    # 原始手（ATTACH_DON/ATTACK）で盤面を組み立てる配線テスト＝箱化は明示 OFF
    # （serve 既定の箱化 2026-08-25 と独立に効果選択の配線だけを見る）
    eng = LearnedEngine(macro_moves=False, defense_box=False)
    g = eng.game

    def find(mgr, atype, card=None):
        for mv in g.legal_actions(mgr):
            d = cpu_ai._describe_move(mgr, mv) or {}
            if d.get("action_type") == atype and (card is None or d.get("card") == card):
                return mv
        return None

    mgr = g.apply(m0.clone(), find(m0.clone(), "ATTACH_DON", "EB03-053"), name)
    assert mgr is not None
    atk = find(mgr, "ATTACK", "EB03-053")
    assert atk is not None, "ドン付与でナミ7000のアタックが成立していない（前提の変化）"
    mgr = g.apply(mgr, atk, name)
    return eng, g, mgr, name


def test_opponent_trigger_confirm_then_target_are_both_decisions(m244):
    """相手リーダーの −1000 は「発動するか」と「誰に当てるか」の2つの決定点になる。"""
    eng, g, mgr, _name = m244
    confirm = [cpu_ai._describe_move(mgr, x) or {} for x in g.legal_actions(mgr)]
    assert len(confirm) == 2, f"発動確認は accept/decline の2手のはず: {confirm}"
    nxt = g.apply(mgr, g.legal_actions(mgr)[0], "p1")       # 発動する
    assert nxt is not None
    sels = [(cpu_ai._describe_move(nxt, x) or {}).get("selected") for x in g.legal_actions(nxt)]
    assert any(s == ["EB03-053"] for s in sels), \
        f"対象選択が決定点として現れていない（攻撃者を選べない）: {sels}"
    assert len(sels) >= 4, f"対象候補が足りない: {sels}"


def test_selected_target_is_actually_applied(m244):
    """選んだ対象に −1000 が入る（既定解決に上書きされない）。"""
    eng, g, mgr, name = m244
    nxt = g.apply(mgr, g.legal_actions(mgr)[0], "p1")
    me = nxt.p1 if nxt.p1.name == name else nxt.p2
    before = {c.master.card_id: int(c.get_power(True)) for c in me.field}
    pick = [x for x in g.legal_actions(nxt)
            if (cpu_ai._describe_move(nxt, x) or {}).get("selected") == ["EB03-053"]][0]
    done = g.apply(nxt, pick, "p1")
    # 対象選択のあとに残る確認（「1枚まで」の実行確認）は accept で流す＝効果が解決するまで進める。
    for _ in range(4):
        acts = g.legal_actions(done)
        ds = [cpu_ai._describe_move(done, x) or {} for x in acts]
        if g.current_player(done) != "p1" or not acts or \
                any(d.get("action_type") != "RESOLVE_EFFECT_SELECTION" for d in ds):
            break
        done = g.apply(done, acts[0], "p1")
    me2 = done.p1 if done.p1.name == name else done.p2
    after = {c.master.card_id: int(c.get_power(True)) for c in me2.field}
    assert after["EB03-053"] == before["EB03-053"] - 1000, f"選んだ対象に入っていない: {before}→{after}"
    assert after["EB03-055"] == before["EB03-055"], "選んでいないキャラが下がっている"


def test_exit_value_box_prefers_blanking_the_attack(m244):
    """出口 value の箱は「攻撃者に当てる（＝アタック不成立）」を最良に並べる。

    選択肢が提示されさえすれば評価は正しい、という切り分け（判断力ではなく配線の欠陥だった）。
    箱の内部解決は PASS 寄りの規約（priors 無し）＝『相手がカウンターしない世界』での比較。"""
    eng, g, mgr, name = m244
    nxt = g.apply(mgr, g.legal_actions(mgr)[0], "p1")
    assert in_battle(nxt), "戦闘中のはず（active_battle）"
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version)
    legal = g.legal_actions(nxt)
    vals = resolved_branch_values(g, nxt, "p1", legal, vf, None)
    best = max(range(len(legal)), key=lambda i: (vals[i] if vals[i] is not None else -9))
    assert (cpu_ai._describe_move(nxt, legal[best]) or {}).get("selected") == ["EB03-053"], \
        f"箱が攻撃者への −1000 を最良にしていない: " \
        f"{[((cpu_ai._describe_move(nxt, m) or {}).get('selected'), v) for m, v in zip(legal, vals)]}"
