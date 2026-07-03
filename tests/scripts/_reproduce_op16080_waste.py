"""検証①-#1(ログ状況再現・決着用): OP16-080【相手のアタック時】任意リダイレクトの無駄撃ちを
ログと同じ状況で再現し、出荷Gen2 が「見送り(decline)」を選ぶか確認する。

再現する状況（ログ由来）:
  - 相手ターン中、相手キャラが p1(OP16-080)の**リーダーを攻撃**。
  - p1 手札に【トリガー】保有カードあり（捨てコストは払える＝accept 可能）。
  - p1 盤面に《黒ひげ海賊団》キャラ無し ⇒ リダイレクト先はリーダーのみ＝**発動しても対象は不変＝
    トリガー1枚を捨てるだけの純損**。⇒ 正解は decline(見送り)。
旧バグ: adapter が accept 既定1手のみ渡す→毎回 accept＝浪費。修正後は accept/decline を探索。
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import numpy as np
from leader_test_helpers import build, add_char, clear_field
from engine_helpers import make_master
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.core import cpu_learned


def setup():
    gm, p1, p2, L = build("OP16-080")
    gm.turn_player = p2
    gm.current_player = p2
    gm.turn_count = 4                      # 初手ターン制限を回避（相手の通常ターン）
    clear_field(p1)                       # 黒ひげキャラ無し＝リダイレクト先はリーダーのみ
    trig = CardInstance(make_master(card_id="TRIGX", name="トリガー餌", trigger_text="ドン!!-1"), p1.name)
    p1.hand = [trig]                       # 手札は【トリガー】1枚のみ（捨てコスト可・浪費が痛い）
    atk = add_char(p2, name="アタッカー", cost=3, power=5000)
    atk.is_rest = False
    gm.declare_attack(atk, p1.leader)      # 相手アタック宣言 → OP16-080 ON_OPP_ATTACK 発火
    return gm, p1


def main():
    gm, p1 = setup()
    ai = gm.active_interaction
    print(f"発火した interaction = {ai and ai.get('action_type')}  "
          f"source = {ai and ai.get('source_card_name')}")
    print(f"p1 手札(トリガー) = {[c.master.card_id for c in p1.hand]}  p1 盤面 = {[c.master.name for c in p1.field]}")
    moves = OPCGGame().legal_actions(gm)
    opts = sorted(bool((m.get('payload') or {}).get('accepted')) for m in moves)
    print(f"提示された選択肢(accepted) = {opts}   （[False, True]＝見送り/発動 両方提示＝配線修正済み）")

    # 出荷Gen2 に決めさせる（複数シードで安定性も見る）
    results = []
    for s in range(5):
        gm2, _ = setup()
        mv = cpu_learned.decide_learned(gm2, gm2.p1, sims=160, rng=np.random.default_rng(s))
        acc = bool((mv.get("payload") or {}).get("accepted")) if mv else None
        results.append(acc)
    n_decline = sum(1 for a in results if a is False)
    print(f"\n出荷Gen2 の選択(5seed) accepted = {results}")
    print(f"  → 見送り(decline)={n_decline}/5  発動(accept)={5 - n_decline}/5")
    print(f"  期待: この局面では見送りが正解（発動は対象不変でトリガー純損）")
    print("  " + ("✅ #1 ログ状況で見送りを選ぶ＝治っている" if n_decline >= 3
                  else "⚠ まだ発動寄り＝netがこの局面で浪費を選ぶ"))


if __name__ == "__main__":
    main()
