"""**切らせた札の値段＝1 枚 1 役の手札の価値の減り**（N-3・2026-09-26・ユーザ決定 判断6(a)／判断7(a) の第 2 段）。

N-1（`hand_joint.py`）が手札の価値を「各札を 出す／カウンター／持つ のちょうど 1 つに割り当てた最大」と定め、
N-2 が守りの判断の守る費用をそれで測った（既定）。本器は同じ値段を **2 つの対の側に同時に**通す
（T77 の原則「損害の側と耐久の側は同じ札に同じ値段を付ける」・片側だけ変えない）:

* **損害の側**——攻め手が守り手に切らせた札（カウンター・効果の捨て札など手札から出ていった札）の値段。
  旧は 1 枚一律 `μ`（`attack_response.parts` の `μ × 手札の枚数の差`・`theory_order.attack_value` の `c(x)·μ`・
  `block_cost` の `c(x)·μ`）。
* **耐久の側**——守り手が生き延びるために使う手札の値段（`crossing_bridge.threshold_parts_side` の手札の項・
  旧は `μ × 切る枚数`〔`cuttable_forced` なら `μ × c_eff × floor(切れる枚数 / c_eff)`〕）。

## 式（新定数ゼロ）

守り手の手札の読み（**枠**＝その席の直近の自席ターンの行・N-2 と同じ時点の規約: 出す計画の枠は次の自席ターン、
守る備えはこれから来る相手ターンを 1 ラウンド割り引く・`theory_bridge.guard_hand_reading`／`joint_valuer` をそのまま使う）
の上で、

```
V(K)        = 1 枚 1 役の手札の価値（N-1）                     K は枠の札の部分集合
切った組 S の値段      = V(K) − V(K − S)                        （実現の損害の側・実際に切った組・順に読む）
L(k)        = min_{|T| = k, T ⊆ 切れる札} [ V(全部) − V(全部 − T) ]   （切る k 枚の最安の値段＝最安の組 T* を順に切った損害の和）
Lx(x)       = L を実数へ: 端数は隣の 2 点の直線・切れる札の数 n を越えた分は 1 枚 μ（手札に無い札は旧と同じ値段）
N_f         = 枠の時点で耐久の式が言う「切る枚数」（予約・`reserve_of_row`）
ḡ           = Lx(N_f) / N_f                                    （予約の 1 枚あたりの平均の値段）
耐久の手札の項 = ḡ × N*                                         N* は旧の式が言う「切る枚数」（数は変えない）
攻撃の守る値段 = ḡ × c(x)                                       c(x) は旧の費用曲線の枚数（数は変えない）
```

**理論の側は枠の中で線形のまま `μ` を `ḡ` に替える**——速さ（`A`）は攻撃を 1 本ずつ独立に値付けし、予約の何枚目を
どの攻撃が切らせるかを持たないので、予約の値段を攻撃の本数で配るには平均しか無い。これで**予約の `N_f` 枚ぶんの攻撃の値段の和
＝耐久の手札の項＝守り手が予約の組 `T*` を順に切ったときの実現の損害の和＝`L(N_f)`**（3 つが厳密に同じ額）。
安い順の切れ目で配る形（`joint_slice`）は診断として残す（`CutView` の注）。

**数は変えず、1 枚あたりの値段だけを変える**——`c(x)`・`N*`・実際に減った枚数はどれも旧の式のまま。
全部の札が一律 `μ` の札なら `L(k) = kμ`・`ḡ = μ` で旧の値に厳密に戻る（テスト）。数の側を規則から作り直す線（H-4 の
`rule`／`rule_don`）とは直交する: H-4 が「どの札を何枚」を出せば、本器の `set_loss`（組の値段）／`Lx`／`ḡ` がそれに値段を付ける。

**不変量**（`tests/test_cut_price.py`）:

1. **望遠鏡**: 枠の札を順に切る列 `S1, S2, …` の値段の和は `V(全部) − V(全部 − ∪S)`（途中の手札で読む限り厳密）。
   `L(k)` は最安の組 `T*` の値段なので、守り手が `T*` を順に切れば**予約された耐久と切らせた損害が厳密に一致**する。
   守り手が高い札を切れば損害は耐久の予約より大きい（損をした分がそのまま帳簿に出る）。
2. **一律の値段への還元**: 札が全部 `v = μ`・カウンターの役が効かない（来る攻撃が無い）なら `L(k) = kμ`・`ḡ = μ`。
3. **単調**: `L` は減らない（単調な `V` では証明つき・読み直しのある手札では累積の最大で床を打つ）。

**使い方**: 呼ぶ側が守り手の枠から `CutCurve` を作り（`curve_of_row`）、今の枚数で `view(今の枚数)` を取り、
`with defending(view):` の中で旧の式を呼ぶ。`theory_order.attack_value`／`block_cost`／`attach_value` と
`crossing_bridge` の手札の項・窓の上限は、この文脈の中でだけ新しい値段になる（外では 1 ビットも変わらない）。
**既定は `joint`＋`gbar`**（2026-10-01・ユーザ決定「規則どおりの N-3 の読みへ今切り替える（数字が落ちても）」）。
`CUT_PRICE_MODE=flat`（`--cut-price flat --cut-take mu`）では誰も文脈に入らない＝**旧と完全に同じ数字**（再現用）。
"""
import contextlib
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_order as TO  # noqa: E402
from theory_order import MU, SC_MY_HAND, SC_MY_LIFE  # noqa: E402

#: **切替**: `flat`＝旧（1 枚一律 `μ`・2026-10-01 までの既定）／`joint`＝N-3（**既定**・1 枚 1 役の価値の減り・損害と耐久の両側を同時に・
#: 予約の平均の値段 `ḡ = L(N_f)/N_f` で理論の側を線形に読む）／`joint_slice`＝同じ曲線を**安い順の切れ目**で読む診断の腕
#: （攻撃 1 本ごとに一番安い札から数える＝速さが攻撃を独立に数えるので安い札を何度も使い回す・下の注）。
#: 切り分けの腕（レビュー 2026-09-30）: `joint_theta`＝**耐久の側だけ**（`Θ` の手札の項と窓の上限は `ḡ`・攻撃の値段・
#: 速さ・実現の損害は旧の `μ`＝T77 を破る対照）／`joint_floor`＝`ḡ` を `μ` で床打ちした `joint`（安すぎる枠の影響の切り分け）。
CUT_PRICE_MODES = ("flat", "joint", "joint_slice", "joint_theta", "joint_floor")
CUT_PRICE_MODE = "joint"
#: **B4（T77）**: 受けたとき手札に入るライフの札の値段。`mu`＝旧（受ける費用 `λ − h·μ`・実現の `−μ`）／
#: `gbar`＝守る側と同じ `ḡ`（**既定**・2026-10-01・受ける費用 `λ − h·ḡ`・実現の損害でも攻め手のターンの間に入った札を `ḡ` で数える）。
CUT_TAKE_MODES = ("mu", "gbar")
CUT_TAKE_MODE = "gbar"


def set_cut_take_mode(mode):
    global CUT_TAKE_MODE
    if mode not in CUT_TAKE_MODES:
        raise ValueError("cut take mode は %s のどれか" % (CUT_TAKE_MODES,))
    CUT_TAKE_MODE = mode
    return CUT_TAKE_MODE

_EPS = 1e-12


def set_cut_price_mode(mode):
    global CUT_PRICE_MODE
    if mode not in CUT_PRICE_MODES:
        raise ValueError("cut price mode は %s のどれか" % (CUT_PRICE_MODES,))
    CUT_PRICE_MODE = mode
    return CUT_PRICE_MODE


def add_cut_price_arg(ap):
    ap.add_argument("--cut-price", default=None, choices=CUT_PRICE_MODES,
                    help="**N-3** 切らせた札の値段: `flat`（旧・1 枚一律 μ）／"
                         "`joint`（既定・1 枚 1 役の手札の価値の減り・損害の側と耐久の側を同時に）／"
                         "`joint_slice`（安い順の切れ目・診断）／`joint_theta`（耐久の側だけ・対照）／`joint_floor`（ḡ を μ で床打ち）")
    ap.add_argument("--cut-take", default=None, choices=CUT_TAKE_MODES,
                    help="**N-3 B4** 受けたとき手札に入るライフの札の値段: `gbar`（既定・守る側と同じ ḡ）／`mu`（旧）")


def apply_cut_price(a):
    if getattr(a, "cut_price", None) is not None:
        set_cut_price_mode(a.cut_price)
    if getattr(a, "cut_take", None) is not None:
        set_cut_take_mode(a.cut_take)
    return CUT_PRICE_MODE


def joint_on():
    return CUT_PRICE_MODE != "flat"


def harm_side_on():
    """損害の側（攻撃の値段・速さ・実現の損害）も新しい値段で読むか（`joint_theta` だけが耐久の側に限る）。"""
    return joint_on() and CUT_PRICE_MODE != "joint_theta"


# ---------------------------------------------------------------------------------------------------------------
# 切れる札（`crossing_bridge.cuttable_share` と同じ規則）
# ---------------------------------------------------------------------------------------------------------------

def cuttable_indices(items, don=None):
    """`items`（`hand_plan.hand_items` の並び）のうち**切れる札の添字**——`crossing_bridge.cuttable_share` と同じ規則:
    無料のカウンター（カウンター値 > 0 のイベントでない札）は全部・カウンター・イベントは `don` を渡せば
    **コストの安い順にドンが続く限り**（`don=None` なら全部）。`len(返り値) / len(items)` が `cuttable_share` と一致する（テスト）。"""
    free = [k for k, it in enumerate(items) if float(it.get("counter") or 0.0) > 0.0 and not it.get("event")]
    evs = sorted((k for k, it in enumerate(items) if float(it.get("counter") or 0.0) > 0.0 and it.get("event")),
                 key=lambda k: float(items[k].get("cost") or 0.0))
    if don is None:
        return sorted(free + evs)
    left = float(don)
    out = list(free)
    for k in evs:
        c = float(items[k].get("cost") or 0.0)
        if c > left + 1e-9:
            continue
        left -= c
        out.append(k)
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------
# 値段の曲線
# ---------------------------------------------------------------------------------------------------------------

class CutCurve:
    """**守り手の枠 1 つの上の値段**。`valuer` は `hand_joint.JointValuer`（枠の全部の枠〔札〕の上）、
    `cand` は切れる札の枠の添字、`h0` は枠の手札の枚数（今の枚数から `m` を出す基準）、`share` は切れる割合
    （`cuttable_share`・`m` の換算）、`cids` は枠ごとの札 id（実際に切った組を id から添字へ写す）。"""

    def __init__(self, valuer, cand, h0, share, mu=MU, cids=None, reserve=None):
        self.valuer = valuer
        self.cand = list(cand)
        self.n0 = len(self.cand)
        self.h0 = float(h0)
        self.share = float(share)
        self.mu = float(mu)
        self.cids = list(cids) if cids is not None else None
        self.reserve = None if reserve is None else float(reserve)
        self._L = None

    @property
    def gbar(self):
        """**予約の平均の値段** `ḡ = Lx(N_f) / N_f`——`N_f` は枠の時点で耐久の式が言う「切る枚数」（`reserve`）。
        予約が 0 枚（来る攻撃が無い等）なら切れる札全部 `n0` の平均・切れる札が無ければ `μ`。
        全部の札が一律 `μ` なら `μ`（テスト）。"""
        key = (self.reserve, CUT_PRICE_MODE)
        got = self.__dict__.get("_gbar_memo")
        if got is not None and got[0] == key:
            return got[1]
        n = self.reserve if (self.reserve is not None and self.reserve > 1e-9) else float(self.n0)
        if n <= 1e-9:
            g = self.mu
        else:
            g = float(self.Lx(n) / n)
            g = max(g, self.mu) if CUT_PRICE_MODE == "joint_floor" else g
        self._gbar_memo = (key, g)
        return g

    # --- 値段 ---
    def full(self):
        return frozenset(range(self.valuer.n))

    def set_loss(self, keep, S):
        """残っている枠 `keep` から組 `S` を切ったときの価値の減り `V(keep) − V(keep − S)`（0 で床）。"""
        keep = frozenset(keep)
        S = frozenset(S) & keep
        if not S:
            return 0.0
        return max(0.0, self.valuer.value(keep)[0] - self.valuer.value(keep - S)[0])

    def L(self):
        """`L(k)`（k = 0..n0）＝切れる札のうち k 枚を切る最安の価値の減り（全部の部分集合を調べる・厳密）。"""
        if self._L is not None:
            return self._L
        n = self.n0
        full = self.full()
        v_all = self.valuer.value(full)[0]
        best = [math.inf] * (n + 1)
        best[0] = 0.0
        for mask in range(1, 1 << n):
            T = frozenset(self.cand[b] for b in range(n) if mask >> b & 1)
            k = len(T)
            loss = max(0.0, v_all - self.valuer.value(full - T)[0])
            if loss < best[k]:
                best[k] = loss
        for k in range(1, n + 1):                               # 単調（読み直しのある手札でも床を打つ）
            best[k] = max(best[k], best[k - 1])
        self._L = [float(x) for x in best]
        return self._L

    def Lx(self, x):
        """`L` を実数の枚数へ（端数は直線・`n0` を越えた分は 1 枚 `μ`）。"""
        x = float(x)
        if x <= 0.0:
            return 0.0
        L = self.L()
        n = self.n0
        if x >= n:
            return L[n] + self.mu * (x - n)
        lo = int(math.floor(x))
        f = x - lo
        return L[lo] + f * (L[lo + 1] - L[lo]) if f > 0.0 else L[lo]

    def m_of(self, hand_now):
        """枠からこれまでに減った切れる枚数（切れる割合 × 減った枚数・0 で床）。"""
        return max(0.0, self.share * (self.h0 - float(hand_now)))

    def view(self, hand_now=None, kind=None):
        """今の枚数（省略＝枠のまま）での値段の窓。`kind` は `avg`（`joint`）／`slice`（`joint_slice`）・省略は切替から。"""
        if kind is None:
            kind = "slice" if CUT_PRICE_MODE == "joint_slice" else "avg"
        return CutView(self, 0.0 if hand_now is None else self.m_of(hand_now), kind)

    # --- 実際に切った組 ---
    def keep_of_ids(self, ids):
        """札 id の多重集合 `ids` のうち**枠の札**に当たる枠の添字（同じ id が複数なら枠の若い順に数だけ）。"""
        if self.cids is None:
            raise ValueError("cids の無い曲線では id から枠へ写せない")
        want = {}
        for c in ids:
            want[c] = want.get(c, 0) + 1
        out = []
        for i, c in enumerate(self.cids):
            if c is not None and want.get(c, 0) > 0:
                out.append(i)
                want[c] -= 1
        return frozenset(out)


class CutView:
    """`CutCurve` の値段の窓。

    * `avg`（`joint`・既定）: `price(k) = k · ḡ`——**枠の中では 1 枚あたり一定**（予約の平均）。理論の側（攻撃の守る値段・
      `Θ` の手札の項・窓の上限）は枚数に線形のまま `μ` を `ḡ` に替えるだけ＝予約の `N_f` 枚ぶんの攻撃の値段の和も、
      `Θ` の手札の項も、守り手が予約の組を順に切ったときの実現の損害の和も**同じ `L(N_f)`**。
    * `slice`（`joint_slice`・診断）: `price(k) = Lx(m + k) − Lx(m)`（`m` はこれまでに減った切れる枚数）——安い順の切れ目。
      速さ（`A`）は攻撃を 1 本ずつ独立に値付けする（`m` を積まない）ので、凸な曲線では**一番安い札を攻撃の本数だけ
      使い回す**＝攻撃の値段が系統的に安く出る（40 局の予備測定で理論の速さ −30%・終局の偏り +1.0 ターン）。"""

    def __init__(self, curve, m=0.0, kind="avg"):
        self.curve = curve
        self.m = float(m)
        if kind not in ("avg", "slice"):
            raise ValueError("kind は avg／slice")
        self.kind = kind

    def price(self, k, mu=None):
        k = float(k)
        if k <= 0.0:
            return 0.0
        if self.kind == "avg":
            return float(k * self.curve.gbar)
        return float(self.curve.Lx(self.m + k) - self.curve.Lx(self.m))


class FlatView:
    """旧の値段（1 枚 `μ`）の窓——テストで「文脈に入っても一律なら旧と同じ」を見るため。"""

    def __init__(self, mu=MU):
        self.mu = float(mu)
        self.m = 0.0

    def price(self, k, mu=None):
        return float(k) * (self.mu if mu is None else float(mu))


# ---------------------------------------------------------------------------------------------------------------
# 文脈（どの守り手の値段で読むか）
# ---------------------------------------------------------------------------------------------------------------

_STACK = []


def active():
    """今の文脈の値段の窓（無ければ `None`＝旧の値段）。**相手の体の値（逆の席）を読んでいる間は `None`**
    （`theory_order._nu_of_other_side`・B3: 攻撃の値段の差し替え口と手札の項の文脈を必ず一緒に外す）。"""
    if TO.CUT_OTHER_SIDE > 0:
        return None
    return _STACK[-1] if _STACK else None


@contextlib.contextmanager
def defending(view):
    """**この中で呼んだ旧の式は、守り手 `view` の値段で読む**。`view=None` なら旧の値段（何もしない）。
    `theory_order.CUT_PRICER`（攻撃の値段）・`CUT_PRICER_KEY`（覚えておく値の鍵）・`CUT_TAKE_CARD`（B4）を差し替え、
    出るときに必ず戻す（入れ子可）。`joint_theta` では攻撃の値段は差し替えない（耐久の側だけ）。"""
    prev = (TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD)
    _STACK.append(view)
    if view is None or CUT_PRICE_MODE == "joint_theta":
        TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD = None, None, None
    else:
        TO.CUT_PRICER = (lambda c, mu, _v=view: _v.price(c, mu))
        avg = getattr(view, "kind", None) == "avg"
        g = float(view.curve.gbar) if avg else None
        TO.CUT_PRICER_KEY = ("avg", round(g, 12)) if avg else None     # 覚えておく値の鍵（ḡ ごとに別）
        TO.CUT_TAKE_CARD = (g if CUT_TAKE_MODE == "gbar" else None) if avg else None
    try:
        yield view
    finally:
        _STACK.pop()
        TO.CUT_PRICER, TO.CUT_PRICER_KEY, TO.CUT_TAKE_CARD = prev


def price_active(k, mu=MU):
    """今の文脈で `k` 枚を切らせる値段（文脈が無ければ `k·μ`）。"""
    v = active()
    return float(k) * float(mu) if v is None else float(v.price(k, mu))


# ---------------------------------------------------------------------------------------------------------------
# 枠の行から曲線を作る
# ---------------------------------------------------------------------------------------------------------------

def curve_of_row(sc, tok, ci_row, idx2cid, cards, deck=None, don=None, mu=MU, mlp=None):
    """**守り手の枠の行から値段の曲線を作る**（N-2 の読み `guard_hand_reading(values=True)`・`joint_valuer` をそのまま）。

    受ける損は `theta_take(その席のライフ)·μ`（N-2 の窓の受ける費用と同じ規約・`const`）。`don` はカウンター・イベントを
    切れるかのドン（`cuttable_share` と同じ渡し方・`None` なら全部）。手札が空なら `None`。
    `mlp`＝守り手のリーダーが次の相手ターンに持つパワー（省略＝枠の行から規則で読む `frame_leader_power`）。"""
    import numpy as np
    import theory_bridge as TB
    sc = np.asarray(sc)
    take = float(TO.theta_take(float(sc[SC_MY_LIFE]))) * float(mu)
    hand = TB.guard_hand_reading(tok, sc, ci_row, idx2cid, cards, take=take, mu=mu, deck=deck, values=True)
    # **B1（レビュー 2026-09-30）**: 枠は守り手の**自席ターン**の行なので、トークンの自分のリーダーのパワーには自分が付けた
    # ドン（自分のターンだけ +1000）が乗っている＝相手のターンには規則上存在しないパワー。`HG.incoming` はそれを守る側に
    # 使うので来る攻撃が −2000 などに見え、カウンターの守りの役が全部 0 になっていた（`L(k)=0`・`ḡ≈0`）。
    # 相手のターンの規則どおりのパワー（`SC_MY_LEADER_POWER`＝付与ドンを載せない値）で読み、**予約 `N_f` と同じ攻撃の並び**を使う。
    # **残り 2a（2026-10-01）**: `SC_MY_LEADER_POWER` は付与ドンを外すが、自席のターンの増減（このターンだけの −1000 など）と
    # 手番つきの継続効果は残る＝実 w41 の枠の 18% で次の相手ターンのパワーと 1000 以上ずれた。`CutFrames` は
    # ターン末の枠では**次の相手ターンの最初の行**の値を渡す（`next_turn_leader_power`）。渡されなければ行から規則で読む。
    if mlp is None:
        mlp = frame_leader_power(sc, tok, ci_row, idx2cid)
    hand["xs_future"] = defender_incoming(sc, tok, mlp)
    slots = hand["slots"]
    if not slots:
        return None
    items_pos = [i for i, s_ in enumerate(slots) if s_.get("item") is not None]
    items = [slots[i]["item"] for i in items_pos]
    cand = [items_pos[k] for k in cuttable_indices(items, don)]
    share = (len(cand) / float(len(items))) if items else 0.0
    valuer = TB.joint_valuer(hand)
    cv = CutCurve(valuer, cand, float(sc[SC_MY_HAND]), share, mu,
                  cids=[s_.get("cid") for s_ in slots],
                  reserve=reserve_of_row(sc, tok, share, mu, mlp=mlp))
    cv.xs_future = list(hand["xs_future"])            # V が読んだ来る攻撃（テストで `N_f` の攻撃と一致を見る）
    cv.mlp = float(mlp)
    return cv


def frame_leader_power(sc, tok, ci_row=None, idx2cid=None):
    """**守り手のリーダーが次の相手ターンに持つパワーを、枠の行だけから規則で読む**（`theory_order.leader_power_opp_turn`：
    付与ドンを外し、リーダー自身の【自分のターン中】の上昇を外して【相手のターン中】の上昇を足す）。0 なら 5000（旧の床）。"""
    return float(TO.leader_power_opp_turn(tok, sc, ci_row, idx2cid)) or 5000.0


def next_turn_leader_power(order, rows, ex, d, t):
    """**ターン末の枠 `(d, t)` の守り手のリーダーが、次の相手ターンに実際に持つパワー**＝攻め手 `1 − d` のターン `t + 1` の
    最初の行の相手のリーダーのパワー（`SC_OPP_LEADER_POWER`＝付与ドンを載せない `get_power(False)`）。無ければ `None`。

    **なぜ次の行か**（規則から読めない分）: 記録のパワーは「印字＋恒久の増減＋このターンだけの増減＋今の手番の継続効果」の
    **和**しか持たない。このターンだけの増減（相手の【相手のアタック時】の −1000 など）はターン終了で消える＝規則では
    次の相手ターンに無いが、和から分けられない（実 w41 の 707 枠: 枠の行の値は 16% でずれ、印字＋本文の継続効果は
    ターン中に入った恒久の増減を落として 8% でずれる）。攻め手の最初の行はターン終了の失効とリフレッシュを経た直後の
    **同じ盤面**で、その間に守り手の選択は無い。**漏れない**: この値を使うのは攻め手のターン `t + 1` の行（最初の行は
    その行以前）だけ＝ターン末の枠でしか使わない（ターン頭の枠〔帳簿の `cut_me`〕は行から規則で読む）。"""
    import numpy as np
    for i in order:
        if int(rows["who"][i]) == 1 - int(d) and int(rows["turn"][i]) == int(t) + 1:
            v = float(np.asarray(ex["sc"][i])[TO.SC_OPP_LEADER_POWER]) * 1e4
            return v if v > 0.0 else None
    return None


def defender_incoming(sc, tok, mlp=None):
    """**守り手の枠から見た、次の相手ターンに来る攻撃の超過**（高い順・通らないものは落とす）。
    守る側は**相手のターンの**自分のリーダーのパワー（`mlp`・省略は `frame_leader_power`）、攻める側は相手のリーダーと
    場のキャラ全部（相手のターンにはリフレッシュで全部殴れる・`crossing_bridge.opp_attackers_of`）＝`reserve_of_row` と同じ並び。"""
    return sorted((float(x) for x in defender_attackers(sc, tok, mlp) if float(x) >= -TO.PWR_EPS), reverse=True)


def defender_attackers(sc, tok, mlp=None):
    """`reserve_of_row` と `defender_incoming` が共有する攻撃の並び（超過 `x`・通らないものも含む）。"""
    import numpy as np
    import crossing_bridge as CB
    sc = np.asarray(sc); tok = np.asarray(tok)
    if mlp is None:
        mlp = frame_leader_power(sc, tok)
    return CB.opp_attackers_of(tok, float(mlp))


def reserve_of_row(sc, tok, share, mu=MU, mlp=None):
    """**枠の時点の予約 `N_f`**＝その席の耐久の手札の項が言う「切る枚数」（`crossing_bridge.hand_cut_count`・
    `threshold_parts_side(…, "me")` と同じ入力: 相手の場の攻撃〔レフレッシュで全部殴れる〕・自分のライフ・自分のアクティブな
    ブロッカー・切れる割合 × 手札の枚数）。窓の上限（T116）は `τ` が要るので掛けない（限界）。"""
    import numpy as np
    import crossing_bridge as CB
    sc = np.asarray(sc); tok = np.asarray(tok)
    xs = defender_attackers(sc, tok, mlp)
    # **H-4g**: `rule`／`rule_don` の守る側の計算は札の枚数を自分で解く（予約はその結果に依る＝循環する）ので、`ḡ` を決める予約は
    # 既定の形（`cuttable_forced`）の枚数で作る＝**どの手札の形でも同じ `ḡ`**（N-3 の既定と同じ値段で比べられる）。
    mode = "cuttable_forced" if CB.THETA_HAND_MODE in ("rule",) + CB.RULE_DON_MODES else None
    return float(CB.hand_cut_count(float(mu) * float(share), float(sc[SC_MY_HAND]), xs, float(sc[SC_MY_LIFE]),
                                   CB._own_active_blockers(tok), mu, mode=mode))


# ---------------------------------------------------------------------------------------------------------------
# 実現の損害（攻め手のターンに守り手の手札から出ていった札）
# ---------------------------------------------------------------------------------------------------------------

def _multiset(ids):
    out = {}
    for c in ids:
        out[c] = out.get(c, 0) + 1
    return out


def _minus(a, b):
    """多重集合の差 `a − b`（id の並び）。"""
    mb = _multiset(b)
    out = []
    for c in a:
        if mb.get(c, 0) > 0:
            mb[c] -= 1
        else:
            out.append(c)
    return out


def realised_corrections(curve, frame_ids, snaps, final_ids=None, mu=MU, add_price=None):
    """**攻め手のターンの中で守り手の手札から出ていった札の値段の直し**（実現の損害 `F` の手札の部分）。

    `frame_ids`＝枠の手札（札 id）・`snaps`＝攻め手のターンの間の守り手の行の `[(位置, 手札の id), …]`（記録の順）・
    `final_ids`＝守り手の次の自席ターンの最初の行の手札（引いた 1 枚は足されるだけなので減った札の判定を変えない・
    無ければ最後の行の後の減りは読まない＝旧の値段のまま）。

    隣り合う読み（枠 → 1 つ目 → … → 最後 → 次の自席ターン）の間で**減った札**を、減らした応答の位置（前の読みの位置・
    枠 → 1 つ目は 1 つ目の位置）に置く。値段は**枠の札**なら `V(残っている枠の札) − V(残っている枠の札 − 減った枠の札)`
    （順に読む＝望遠鏡）・**枠に無い札**（ターンの間に入ったライフの札など）は旧の `μ`。返すのは
    `[(位置, 直し = 新しい値段 − μ × 減った枚数), …]`——旧の `μ × 枚数の差` に足せば新しい実現になる
    （全部の札が一律 `μ` なら直しは 0）。"""
    out = []
    keep = curve.keep_of_ids(frame_ids)          # 枠の札のうち、まだ残っている枠
    left_frame = list(frame_ids)                 # 残っている枠の札の id（多重集合）
    chain = [(None, list(frame_ids))] + [(p, list(ids)) for p, ids in snaps]
    if final_ids is not None and chain[-1][0] is not None:
        chain.append(("final", list(final_ids)))
    for k in range(len(chain) - 1):
        _p0, a = chain[k]
        p1, b = chain[k + 1]
        gone = _minus(a, b)
        if add_price is not None and p1 != "final":
            # **B4**: 攻め手のターンの間に手札へ入った札（受けたライフの札など）を `μ` ではなく `add_price` で数える
            # （旧の `μ × 枚数の差` は入った札を `−μ` で数えている＝直しは `+(μ − add_price)` × 入った枚数）。
            # 次の自席ターンの読みとの間の増えは引きを含むので数えない（旧の枚数の差にも入っていない）。
            added = _minus(b, a)
            if added:
                pos_a = p1 if p1 is not None else chain[k][0]
                out.append((pos_a, (float(mu) - float(add_price)) * len(added)))
        if not gone:
            continue
        pos = chain[k][0] if chain[k][0] is not None else chain[k + 1][0]
        if pos == "final":
            pos = chain[k][0]
        # 減った札のうち枠の札（残っている枠の札の多重集合から数だけ取る）
        g_frame = []
        rest = list(left_frame)
        for c in gone:
            if c in rest:
                rest.remove(c)
                g_frame.append(c)
        S = set()
        for c in g_frame:
            for i in sorted(keep - S):
                if curve.cids[i] == c:
                    S.add(i)
                    break
        price = curve.set_loss(keep, S) + float(mu) * (len(gone) - len(S))
        out.append((pos, float(price) - float(mu) * len(gone)))
        keep = keep - frozenset(S)
        left_frame = rest
    return out


def sum_in(corr, lo, hi):
    """位置が `(lo, hi)` に入る直しの和（括りの行 `lo` の後・閉じる行 `hi` の前の応答）。"""
    return float(sum(c for p, c in corr if p is not None and lo < p < hi))


# ---------------------------------------------------------------------------------------------------------------
# 1 局の枠の束（器が共有する）
# ---------------------------------------------------------------------------------------------------------------

class CutFrames:
    """**1 局ぶんの守り手の枠と値段の曲線**（器が共有する・`flat` では作らない）。

    `frame_rows`＝`{(席, 自席ターン): その枠の行}`（器が自分の規約で選ぶ——交点の橋は `turn_last`〔その席の直近の自席
    ターンの最後の行〕・帳簿は `rate_at`〔自席ターンの最初の行〕）。`order` は局の行の並び（位置 `n` は記録の順）。
    `decks`＝`(席 0 のデッキ, 席 1 のデッキ)`（相方待ちの読み直し・無ければ `None`）。`don_rule`＝カウンター・イベントを
    切れるかを枠の使い残しのドンで決めるか（交点の橋の `THETA_DON_MODE=rule` と同じ）・`False` なら全部切れる。"""

    def __init__(self, order, rows, ex, idx2cid, cards, frame_rows, mu=MU, decks=None, don_rule=True, stats=None,
                 end_of_turn=False):
        self.order = list(order)
        self.rows = rows
        self.ex = ex
        self.idx2cid = idx2cid
        self.cards = cards
        self.frame_rows = dict(frame_rows)
        self.mu = float(mu)
        self.decks = decks
        self.don_rule = bool(don_rule)
        self.stats = stats if stats is not None else {}
        self.end_of_turn = bool(end_of_turn)       # 枠がターン末の行か（次の相手ターンの最初の行のパワーを読む・残り 2a）
        self._curves = {}
        self._corr = {}
        self._by_seat = {}
        for (d, tt) in self.frame_rows:
            self._by_seat.setdefault(d, []).append(tt)
        for d in self._by_seat:
            self._by_seat[d].sort()

    def _st(self, k, v=1):
        self.stats[k] = self.stats.get(k, 0) + v

    def frame_key(self, d, t, at_n=None):
        """席 `d` の枠のうち自席ターンが `t` 以下で一番新しいもの（無ければ `None`）。
        `at_n`（呼ぶ行の `order` の中の位置）を渡せば、**その行より後ろの行を読む枠は飛ばす**（因果: 窓はその行までに
        在る行だけで作る・2026-10-01 の点検）。"""
        ts = [tt for tt in self._by_seat.get(int(d), ()) if tt <= int(t)]
        if at_n is None:
            return (int(d), ts[-1]) if ts else None
        for tt in reversed(ts):
            key = (int(d), tt)
            if self._key_last_pos(key) <= int(at_n):
                return key
            self._st("cut_causal_skip")
        return None

    def _key_last_pos(self, key):
        """枠 `key` が読む一番後ろの行の位置（枠の行と、ターン末の枠なら次の相手ターンの最初の行）。"""
        if not hasattr(self, "_posmap"):
            self._posmap = {int(i): n for n, i in enumerate(self.order)}
        last = self._posmap.get(int(self.frame_rows[key]), -1)
        if self.end_of_turn:
            for n, i in enumerate(self.order):
                if int(self.rows["who"][i]) == 1 - int(key[0]) and int(self.rows["turn"][i]) == int(key[1]) + 1:
                    last = max(last, n)
                    break
        return last

    def curve(self, d, t, at_n=None):
        key = self.frame_key(d, t, at_n)
        if key is None:
            return None
        if key not in self._curves:
            import numpy as np
            from theory_order import SC_MY_DON
            i = self.frame_rows[key]
            sc = np.asarray(self.ex["sc"][i])
            deck = None
            if self.decks is not None:
                dk = self.decks[int(d)]
                deck = None if dk is None else list(dk)
            don = float(sc[SC_MY_DON]) if self.don_rule else None
            mlp = next_turn_leader_power(self.order, self.rows, self.ex, key[0], key[1]) if self.end_of_turn else None
            self._st("cut_mlp_next" if mlp is not None else "cut_mlp_rule")
            if mlp is None:
                mlp = frame_leader_power(sc, self.ex["tok"][i], self.ex["ci"][i], self.idx2cid)
            cv = curve_of_row(sc, self.ex["tok"][i], self.ex["ci"][i], self.idx2cid, self.cards, deck=deck,
                              don=don, mu=self.mu, mlp=mlp)
            self._curves[key] = cv
            self._st("cut_frames")
            if cv is not None:
                self._st("cut_cand_sum", cv.n0)
                L = cv.L()
                if cv.n0 >= 1:
                    self._st("cut_L1_n")
                    self.stats["cut_L1_sum"] = self.stats.get("cut_L1_sum", 0.0) + float(L[1])
        return self._curves[key]

    def lookahead_pos(self, d, t):
        """席 `d`・ターン `t` の窓が読む**一番後ろの行の位置**（`order` の中の位置）——枠の行と、ターン末の枠なら
        次の相手ターンの最初の行（2a のパワー）。枠が無ければ `None`。"""
        key = self.frame_key(d, t)
        if key is None:
            return None
        return self._key_last_pos(key)

    def view(self, d, t, hand_now, at_n=None):
        """席 `d` が守り手の値段の窓（今の枚数 `hand_now`）。枠が無ければ `None`（旧の値段）。
        `at_n`（呼ぶ行の `order` の中の位置）を渡せば、窓がその行より後ろの行を読むかを数える（`cut_lookahead`・
        2026-10-01 の点検: 守りの窓で攻め手の進行中のターンの末の枠を読んでいた＝先読み。器は 0 を保つ）。"""
        if at_n is not None:
            self._st("cut_lookups")
            key = self.frame_key(d, t, at_n)
            if key is not None and self._key_last_pos(key) > int(at_n):
                self._st("cut_lookahead")                     # 因果の選び方なら常に 0（器の検算）
        cv = self.curve(d, t, at_n)
        if cv is None:
            self._st("cut_view_flat")
            return None
        return cv.view(hand_now)

    def corrections(self, w, t):
        """攻め手 `w` のターン `t` の間に守り手 `1 − w` の手札から出ていった札の値段の直し `[(位置, 直し), …]`。
        枠は守り手の `t` より前の直近の枠（`view(1 − w, t, …)` と同じ枠）。"""
        key = (int(w), int(t))
        if key in self._corr:
            return self._corr[key]
        import hand_spend as HS
        d = 1 - int(w)
        fk = self.frame_key(d, int(t))
        out = []
        cv = self.curve(d, int(t)) if fk is not None else None
        if cv is None:
            self._st("cut_corr_noframe")
        else:
            i_f = self.frame_rows[fk]
            frame_ids = HS.hand_ids(self.ex["ci"][i_f], self.idx2cid)
            snaps, final = [], None
            for n, i in enumerate(self.order):
                ww, tt = int(self.rows["who"][i]), int(self.rows["turn"][i])
                if ww == d and tt == int(t):
                    snaps.append((n, HS.hand_ids(self.ex["ci"][i], self.idx2cid)))
                elif ww == d and tt == int(t) + 1 and final is None:
                    final = HS.hand_ids(self.ex["ci"][i], self.idx2cid)
            if final is None:
                self._st("cut_corr_nofinal")
            out = realised_corrections(cv, frame_ids, snaps, final, self.mu,
                                       add_price=(cv.gbar if CUT_TAKE_MODE == "gbar" else None))
            self._st("cut_corr_turns")
            self._st("cut_corr_n", len(out))
            self.stats["cut_corr_sum"] = self.stats.get("cut_corr_sum", 0.0) + float(sum(c for _p, c in out))
        self._corr[key] = out
        return out

    def bracket_corr(self, w, t, n_lo, n_hi):
        """括り（攻め手の行の位置 `n_lo` から閉じる行の位置 `n_hi`）の中の応答の直しの和。"""
        if n_hi is None or not harm_side_on():
            return 0.0                                  # `joint_theta`（耐久の側だけ）は実現の損害を旧のまま
        return sum_in(self.corrections(w, t), n_lo, n_hi)
