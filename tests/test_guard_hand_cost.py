"""**G-2（2026-09-25・ユーザ決定「これで行きましょう」）**: 守りの判断（`theory_bridge.guard_step`）の 2 つの直し。

1. **守れたかを規則どおりに判定する**——同じパワーは命中するので、超過 `x` を止めるには
   カウンター合計が `x + 1000` 以上要る。旧 `lenient`（合計 ≥ `x`・超過 0 ならカウンター 0 枚でも「守れた」）の切替は
   2026-10-05 に削除（下の `_old_guard_step` は比較の基準としてだけ残す）。
2. **守る費用をこの手札で実際に失う価値で測る切替**（`GUARD_S_COST_MODE`・`curve`＝従来の `c(x)·μ`・`hand` が G-2 の形
   〔2026-10-05 に削除・本ファイルの `hand` の代数のテストも同時に削除〕。
   **既定は 2026-09-26 から N-2 の `joint`**＝`test_hand_joint.py`。本ファイルの代数は `curve` を明示して固定する）——
   止める札の組 `S` のうち `V(手札) − V(手札 − S)` が一番小さい組の減り。`V` は出す計画（T66）＋ **これから来る**
   相手ターンの守る備え（T67 と同じ目的を**厳密な最大**で解いたもの・今の窓から **1 ラウンド割り引く**）を次の自席ターンの時点で読んだもの。

**G-2 の修正（2026-09-26・レビュー D1〜D5）**: 守る備えを貪欲な割り当てから厳密な最大へ（V が手札について単調になる）・
最初の相手ターンを `s = 1 − ko_p` で割り引く（「止める組がそれしか無い」手札が必ず受けると同点になっていた）・
V の差は T67 の札ごとの価値 `max(ΔH, ΔG)` とは違う量（`ΔH + ΔG`・割引前）であることを値で固定・実行全体の記録に実際に効いた閾値を刻む。

**符号と循環が 1 つ狂うと判断が反転する器**なので、向き・循環・規則の境目を値で押さえる。**基盤健全性**（`cpu_infra`）。
"""
import itertools
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import guard_afford as GA  # noqa: E402
import hand_guard as HG  # noqa: E402
import theory_bridge as B  # noqa: E402
import theory_order as T  # noqa: E402

MU = T.MU
TAKE = T.THETA * T.MU
S_DISC = 1.0 - T.KO_P
LIFE0_TAKE = T.theta_take(0.0) * T.MU                           # ライフ 0 で受ける損（致死）

_SHIPPED_S_COST = B.GUARD_S_COST_MODE          # 収集時（どのテストも切替を触る前）の出荷時の既定


@pytest.fixture(autouse=True)
def _pin_curve_guard_cost():
    """本ファイルの代数（旧との一致・`hand` との比べ）は、切替を省いた呼び出しを旧の `c(x)·μ` として書いてある——
    既定は N-2 の `joint`（手札の読みが要る）なので `curve` を明示して固定し、終わったら戻す。"""
    old = B.GUARD_S_COST_MODE
    B.set_guard_s_cost_mode("curve")
    yield
    B.set_guard_s_cost_mode(old)


def _tok(opp_lead=5000, my_lead=5000, blocker=False):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_lead / 1e4
    tok[1, T.S_POWER] = opp_lead / 1e4
    if blocker:
        tok[2, T.S_POWER] = 0.3
        tok[2, T.S_IS_CHAR] = 1.0
        tok[2, B.GA.S_BLOCKER] = 1.0
    return tok


def _sc(don=5):
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON] = don
    return sc


def _slot(counter, v, cost=0.0, event=False, paid=None):
    """手で作る手札の 1 枚（`guard_hand_reading` の枠と同じ形・無料の印字は event でない札だけ）。"""
    free = 0.0 if (event or counter <= 0.0) else float(counter)
    return {"cid": None, "cost": float(cost), "counter": float(counter), "event": event, "free": free,
            "paid": paid, "v": v, "item": None}


def _hand(slots, xs_future=(1000.0,), caps=(7, 9, 10, 10), take=TAKE):
    return {"slots": list(slots), "caps": list(caps), "xs_future": list(xs_future), "take": float(take),
            "mu": MU, "inflow": None}


def _old_guard_step(tok, sc, played, free, paid, theta=T.THETA, mu=T.MU, margin_comfort=None):
    """**G-2 より前の `guard_step` の写し**（HEAD f86e7d30 のまま）——規則どおりの判定で変わる行だけが変わることの基準。"""
    xs = [x for x in B.incoming_x(tok) if x >= -B.PWR_EPS]
    if not xs:
        return None
    x = max(xs)
    blocker = bool((np.asarray(tok)[B.GA.SLOT_OWN_FIELD][:, B.GA.S_BLOCKER] > 0.5).any())
    budget = int(round(float(sc[B.SC_MY_DON])))
    afford_pw = free + B.GA.knapsack(paid, budget)
    can_guard = bool(blocker or afford_pw >= x)
    cost_take = float(theta) * float(mu)
    cost_guard = float(B.c_of(x)) * float(mu)
    best = min(cost_take, cost_guard) if can_guard else cost_take
    actual = cost_guard if played == "guard" else cost_take
    g_paid = -float(actual)
    price = min(cost_take, cost_guard)
    g_delta = float(price) - float(actual)
    g = g_delta                                           # `GUARD_G_MODE=delta`（他の値は 2026-10-05 に削除）
    if played == "guard" and not can_guard:
        actual = best
    margin = (float("inf") if blocker else float(afford_pw) - float(x))
    return {"s": -max(0.0, actual - best), "g": g, "g_paid": g_paid, "g_delta": g_delta, "price": float(price),
            "x": x, "can_guard": can_guard, "margin": margin,
            "comfortable": bool(can_guard and margin >= (B.MARGIN_COMFORT
                                                        if margin_comfort is None
                                                        else float(margin_comfort))),
            "played": played, "theory_says": ("guard" if (can_guard and cost_guard < cost_take)
                                              else "take")}


def _grid():
    for opp in (4000, 5000, 5500, 6000, 7000, 8000, 10000, 14000):
        for free in (0.0, 500.0, 1000.0, 2000.0, 2500.0, 3000.0, 4000.0, 6000.0, 9000.0):
            for paid in ([], [(1, 2000.0)], [(3, 1000.0)]):
                for don in (0, 1, 3):
                    for blocker in (False, True):
                        for played in ("take", "guard"):
                            for theta in (T.THETA, 1.15, 8.18):
                                yield opp, free, paid, don, blocker, played, theta


# ---- 1. 既定と切替 ----

def test_the_defaults_are_the_rule_and_the_old_cost():
    assert _SHIPPED_S_COST == "joint"             # 判断の守る費用は N-2 の 1 枚 1 役（ユーザ決定 2026-09-26「判断1の続き→(a)」）
    assert B.GUARD_S_COST_MODES == ("curve", "joint")


def test_unknown_modes_are_refused_everywhere():
    for bad in ("", "strict", "なにか"):
        with pytest.raises(ValueError):
            B.set_guard_s_cost_mode(bad)
        with pytest.raises(ValueError):
            B.guard_step(_tok(6000), _sc(), "take", 9000.0, [], s_cost=bad)
    assert B.GUARD_S_COST_MODE == "curve"      # 弾かれた指定は今の値（ここでは固定した `curve`）を変えない


# ---- 2. 規則どおりの「守れたか」 ----

def test_an_equal_power_attack_is_not_stopped_by_no_counters():
    """超過 0（同じパワー）は命中する＝カウンター 0 枚では止まらない。旧は「守れた」にして受けた行を罰していた。"""
    tok, sc = _tok(opp_lead=5000), _sc(don=0)
    rule = B.guard_step(tok, sc, "take", 0.0, [])
    old = _old_guard_step(tok, sc, "take", 0.0, [])
    assert rule["x"] == pytest.approx(0.0)
    assert rule["can_guard"] is False and rule["s"] == 0.0            # 守れない＝受けても誤りでない
    assert old["can_guard"] is True and old["s"] < 0.0                # 旧は受けたことを罰していた


def test_a_counter_total_of_exactly_the_excess_still_loses_the_battle():
    tok, sc = _tok(opp_lead=7000), _sc(don=0)                   # 超過 2000
    assert B.guard_step(tok, sc, "take", 2000.0, [])["can_guard"] is False
    assert _old_guard_step(tok, sc, "take", 2000.0, [])["can_guard"] is True                 # 旧（lenient）は守れた扱い
    assert B.guard_step(tok, sc, "take", 2500.0, [])["can_guard"] is False    # 許容は PWR_EPS（丸め）だけ
    assert B.guard_step(tok, sc, "take", 3000.0, [])["can_guard"] is True     # x + 1000 ちょうどで勝つ
    assert B.afford_need(2000.0) == pytest.approx(3000.0 - T.PWR_EPS)


def test_paid_event_counters_still_need_the_don_under_the_rule():
    tok = _tok(opp_lead=6000)                                   # 超過 1000 → 2000 要る
    assert B.guard_step(tok, _sc(don=1), "take", 0.0, [(1, 2000.0)])["can_guard"] is True
    assert B.guard_step(tok, _sc(don=0), "take", 0.0, [(1, 2000.0)])["can_guard"] is False
    assert B.guard_step(tok, _sc(don=1), "take", 0.0, [(1, 1000.0)])["can_guard"] is False   # 払えても 1000 では足りない
    assert B.guard_step(_tok(opp_lead=9000, blocker=True), _sc(don=0), "take", 0.0, [])["can_guard"] is True   # ブロッカーは不変


def test_under_the_rule_only_the_rows_between_x_and_x_plus_1000_change():
    """`rule` ＋ `curve` で旧と違うのは**`x ≤ 守る力 < x + 1000` の行だけ**で、変わるのは「守れたか」とそれに従う欄だけ
    （帳簿の欄・余裕・価格は 1 ビットも動かない）。向きは必ず「守れた → 守れない」。"""
    derived = {"can_guard", "s", "theory_says", "comfortable"}
    changed = 0
    for opp, free, paid, don, blocker, played, theta in _grid():
        tok, sc = _tok(opp_lead=opp, blocker=blocker), _sc(don=don)
        old = _old_guard_step(tok, sc, played, free, paid, theta=theta)
        new = B.guard_step(tok, sc, played, free, paid, theta=theta)
        if old is None:
            continue
        diff = {k for k in old if new[k] != old[k]}
        assert diff <= derived, diff
        pw = free + GA.knapsack(paid, don)
        between = (not blocker) and (old["x"] <= pw < old["x"] + 1000.0 - T.PWR_EPS)
        if diff:
            changed += 1
            assert between and old["can_guard"] is True and new["can_guard"] is False
            assert new["s"] == 0.0                                  # 守れない行は受けても守っても誤りでない
        if between:
            assert new["can_guard"] is False
    assert changed > 0


def test_curve_mode_with_a_hand_reading_changes_no_old_field():
    """既定の `curve` で手札の読み（枚数の監査）を渡しても従来の欄は同じ——足すのは監査の欄だけ。"""
    rd = {"slots": [_slot(2000.0, None), _slot(0.0, None, cost=3), _slot(2000.0, None, cost=1, event=True, paid=(1, 2000.0))],
          "caps": None, "xs_future": None, "take": TAKE, "mu": MU, "inflow": None}
    for opp, free, paid, don, blocker, played, theta in itertools.islice(_grid(), 0, None, 7):
        tok, sc = _tok(opp_lead=opp, blocker=blocker), _sc(don=don)
        a = B.guard_step(tok, sc, played, free, paid, theta=theta)
        b = B.guard_step(tok, sc, played, free, paid, theta=theta, hand=rd)
        if a is None:
            assert b is None
            continue
        old_keys = set(_old_guard_step(tok, sc, played, free, paid, theta=theta))
        assert {k: a[k] for k in old_keys} == {k: b[k] for k in old_keys}
        assert b["cost_guard_hand"] is None and b["cost_guard_source"] == "curve" and b["s_cost_mode"] == "curve"
        assert b["cost_guard_curve"] == pytest.approx(B.c_of(b["x"]) * MU) and b["cost_guard_s"] == b["cost_guard_curve"]
        assert b["n_hand_cards"] == 3
        assert b["n_counter_cards"] == (2 if don >= 1 else 1)         # イベントはドンが払えるときだけ数える
        assert a["n_hand_cards"] is None and a["cost_guard_hand"] is None


# ---- 3. 手札で測る守る費用（`hand`）——2026-10-05 に切替ごと削除（`joint` は `test_hand_joint.py`） ----

# ---- 4. 記録の行からの読み（本物の札） ----

def _real_row():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    vocab = GA._vocab()
    idx2cid = {i: c for c, i in vocab.items()}
    # 2000 カウンターのキャラ 2・1000 カウンターのキャラ 2・【カウンター】イベント（上げ幅はトークンから・コスト 1）・カウンター無しのイベント
    hand = ["EB01-007", "EB01-014", "EB01-004", "EB01-005", "EB01-009", "EB01-010"]
    tok = np.zeros((22, 24), np.float32)
    ci = np.zeros(24, np.int64)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.6              # 超過 1000
    for j, cid in enumerate(hand):
        s = GA.SLOT_HAND.start + j
        inf = cards.info(cid)
        assert inf is not None, cid
        ci[s] = vocab[cid]
        cv = float(inf.get("counter") or 0.0)
        if inf.get("event"):
            cv = 2000.0 if cid == "EB01-009" else 0.0
            tok[s, GA.S_IS_EVENT] = 1.0
        tok[s, GA.S_COUNTER] = cv / GA.COUNTER_SCALE
        tok[s, T.S_POWER] = float(inf.get("power") or 0.0) / 1e4
        tok[s, 23] = 0.1
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON], sc[T.SC_OPP_LEADER_POWER], sc[T.SC_OPP_LIFE], sc[T.SC_MY_LIFE] = 1, 0.5, 3, 3
    return tok, sc, ci, idx2cid, cards


def test_the_reading_splits_counters_exactly_like_guard_afford():
    tok, sc, ci, idx2cid, cards = _real_row()
    free, paid, nslots = GA.hand_counters(tok, ci, idx2cid, cards)
    for values in (False, True):
        rd = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=values)
        assert sum(s["free"] for s in rd["slots"]) == pytest.approx(free)
        assert [s["paid"] for s in rd["slots"] if s["paid"] is not None] == paid
        assert len(rd["slots"]) == nslots == 6
    assert free == pytest.approx(6000.0) and paid == [(1.0, 2000.0)]


def test_the_reading_values_come_from_hand_plan_on_the_next_turn():
    import hand_plan as HP
    from price_realised import don_stock
    tok, sc, ci, idx2cid, cards = _real_row()
    rd = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=True)
    items = HP.hand_items(tok, ci, idx2cid, cards, 5000.0, 3.0)
    assert [s["item"] for s in rd["slots"] if s["item"] is not None] == items          # 札ごとの価値は hand_plan のもの
    # 出す計画は次の自席ターンから（**式を写さずに数で**）: この行のドンはアクティブ 1 枚だけ＝総在庫 1
    # → 次の自席ターンは 1 + 2 = 3 枚（全部アクティブ）・その後 5, 7・「それより後」は残り 3 ターンなので 10 × 1
    assert don_stock(sc, tok, "me") == pytest.approx(1.0)
    assert rd["caps"] == [3, 5, 7, 10]
    sc2 = sc.copy()
    sc2[T.SC_MY_DON] = 9                                                               # 総在庫 9 → 次は 10 で頭打ち
    rd2 = B.guard_hand_reading(tok, sc2, ci, idx2cid, cards, take=TAKE, values=True)
    assert rd2["caps"] == [10, 10, 10, 10]
    sc3 = sc.copy()
    sc3[T.SC_OPP_LIFE] = 5                                                             # 残り 5 ターン → 「それより後」は 10 × 2
    assert B.guard_hand_reading(tok, sc3, ci, idx2cid, cards, take=TAKE, values=True)["caps"] == [3, 5, 7, 20]
    assert rd["xs_future"] == HG.incoming(tok) == [pytest.approx(1000.0)]
    lite = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=False)
    assert lite["caps"] is None and all(s["v"] is None and s["item"] is None for s in lite["slots"])
    free, paid, _n = GA.hand_counters(tok, ci, idx2cid, cards)
    got = B.guard_step(tok, sc, "take", free, paid, s_cost="joint", hand=rd)
    assert got["cost_guard_source"] == "joint" and got["cost_guard_hand"] is not None
    assert got["cost_guard_hand"] >= 0.0 and got["hand_set_n"] >= 1
    assert got["n_hand_cards"] == 6 and got["n_counter_cards"] == 5                   # イベントはドン 1 で払える


# ---- 5. G-2 の修正（2026-09-26・レビュー D1〜D5） ----

def _brute_guard(items, xs, take, s=S_DISC, turns=HG.GUARD_TURNS, start=0):
    """**独立の総当たり**: 札ごとに「使わない／どの (相手ターン, 攻撃) に切るか」を全部試す（`guard_value_exact` の再帰とは別の数え方）。
    足りない組を割り当てた割り当ては捨てる。受ける方が安い組を割り当てた割り当ても数えるが、割り当てない方が必ず良いので最大は変わらない。"""
    pairs = [(t, x) for t in range(turns) for x in xs if x >= -T.PWR_EPS]
    n = len(items)
    best = 0.0
    for assign in itertools.product(range(len(pairs) + 1), repeat=n):
        tot, ok = 0.0, True
        for j, (t, x) in enumerate(pairs, start=1):
            S = [i for i in range(n) if assign[i] == j]
            if not S:
                continue
            if sum(items[i][0] for i in S) < x + 1000.0 - T.PWR_EPS:
                ok = False
                break
            tot += s ** (start + t) * (take - sum(0.0 if items[i][1] is None else items[i][1] for i in S))
        if ok and tot > best:
            best = tot
    return best


def _rand_items(rng, n):
    return [(float(rng.choice([0.0, 1000.0, 2000.0])), float(rng.choice([0.0, 0.01, 0.03, 0.05, 0.09]))) for _ in range(n)]


def test_the_exact_guard_readiness_is_the_true_maximum_and_never_below_the_greedy():
    """**D1**: 守る備えは全部の割り当ての最大——独立の総当たりと一致し、貪欲（`guard_value`・同じ目的の 1 つの割り当て）以上。
    `start=1` は全体をちょうど `s` 倍する。"""
    rng = np.random.default_rng(11)
    n_gap = 0
    for _ in range(150):
        items = _rand_items(rng, int(rng.integers(0, 5)))
        xs = [float(x) for x in rng.choice([0.0, 1000.0, 2000.0, 3000.0], size=int(rng.integers(0, 3)))]
        ex = HG.guard_value_exact(items, xs, TAKE)
        assert ex == pytest.approx(_brute_guard(items, xs, TAKE), abs=1e-12), (items, xs)
        greedy = HG.guard_value(items, xs, TAKE)
        assert ex >= greedy - 1e-12
        n_gap += int(ex > greedy + 1e-9)
        assert HG.guard_value_exact(items, xs, TAKE, start=1) == pytest.approx(S_DISC * ex, abs=1e-12)
    assert n_gap > 0                                             # 貪欲が取りこぼす手札は実在する
    # 負の v が在っても定義どおり（足りる組を全部調べる経路）
    items = [(1000.0, -0.01), (1000.0, 0.02), (2000.0, 0.0)]
    assert HG.guard_value_exact(items, [0.0, 1000.0], TAKE) == pytest.approx(_brute_guard(items, [0.0, 1000.0], TAKE))


def test_the_value_lost_is_not_t67s_per_card_value():
    """**D3**: 手札の差で読む失う価値は T67 の札ごとの価値 `max(ΔH, ΔG)` とは**違う量**。
    2000 カウンター（v = 0.03・コスト 2）1 枚・次の攻撃 1000: 割引前の差は `ΔH + ΔG`＝`max(v, Θμ)`＝0.0872、
    T67 は `max(ΔH, ΔG)`＝0.0572。差はちょうど `min(ΔH, ΔG)`。割り引くと `ΔH + s·ΔG`。"""
    import hand_plan as HP
    card = {"cid": None, "cost": 2.0, "v": 0.03, "counter": 2000.0, "event": False}
    d = HP.card_deltas([], card, [7, 9, 10, 10], [1000.0], TAKE)
    assert d["dh"] == pytest.approx(0.03) and d["dg"] == pytest.approx(TAKE - 0.03)
    assert d["dtotal"] == pytest.approx(0.0572, abs=1e-4)
    item = {"cost": 2.0, "v": 0.03, "counter": 2000.0}
    v0 = (B.hand_value_next([item], [7, 9, 10, 10], [1000.0], TAKE, guard_start=0)
          - B.hand_value_next([], [7, 9, 10, 10], [1000.0], TAKE, guard_start=0))
    assert v0 == pytest.approx(0.0872, abs=1e-4) and v0 == pytest.approx(TAKE)
    assert v0 == pytest.approx(d["dh"] + d["dg"])
    assert v0 == pytest.approx(d["dtotal"] + min(d["dh"], d["dg"]))


