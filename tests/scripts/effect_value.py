"""**効果をコストと実行内容で値付けする**（P2・`game_theory.md` §14.1.3）。

ユーザ決定 2026-09-14「イベント、起動メイン等の効果については**コストと実行内容で整理する**」。
続くユーザ決定「**一旦効果を全部理論に落とす**」「**数え方はエンジンベースで数える**
——エンジンが定義しているアクション（60 個くらいあったやつ）を変換したい」。

## 要点 1——**母数はエンジンの動作の全集合**（62 メンバ）

```
ENGINE_ACTIONS = opcg_sim.src.models.enums.ActionType の全メンバ（Rust の
                 rust/opcg_engine/src/effects/ast.rs::ActionType と 1:1）
```

**同梱カードに出てくる 45 種ではなく、エンジンが定義している 62 種を母数にする。**
初版はカードの出現から表を作ったので **(a) 実在しない名前を持ち**（`COST_REDUCTION`・
`ACTIVE_DON_OPP`・`LEAVE`＝どれも `ActionType` に無い）、**(b) 実在する 20 種近くが表に無かった**
（`MOVE_TO_HAND`・`BP_BUFF`・`COST_BUFF`・`LIFE_MANIPULATE`・`SELECT_OPTION` …）。
**どちらも「出現 0 のものは見えない」から起きた**——母数をエンジンに移せば構造的に起きない
（`test_effect_value.py` が**全メンバがちょうど 1 つの類に入ること**をラチェットする）。

## 要点 2——**新しい定数を足さない**（既存の実測値からの導出だけ）

```
効果の価値 = Σ（実行内容が動かした通貨 × その価格） − Σ（コストが動かした通貨 × その価格）
```

| 系統 | 導出 | 使う実測値 |
|---|---|---|
| 4 通貨 | そのまま | `μ`・`λ`・`δ`・`ν` |
| **パワー** | `+1000` ≈ ドン 1 個、**ただし 1 体から引き出せる上限で打ち切る** | `δ`・`Θ·μ`・`ν` |
| **テンポ**（レスト・凍結・攻撃不可） | **在庫 ÷ 残りターン**＝1 ターンぶんの流量 | `R = 4.128` |
| **選択**（サーチ・デッキ操作） | **k 枚見て 1 枚選ぶ利得** = `E[max_k W] − E[W]` | カード価値の分布（**つまみ無し**） |
| **生存**（KO されない・代わりに〜） | 体が消える確率 × 体 | `ko_p = 0.289` |
| **キーワード** | ブロッカー＝実測の上乗せ／速攻＝攻撃 1 回／バニッシュ＝トリガー 1 回 | `0.035`・`Θ·μ`・`τ_value` |
| **能力の参照**（起動メインを発動・能力を消す） | **同じカードなら再帰で解く**。判らなければ**能力 1 つ ≈ 札 1 枚** | `μ`（**暫定 P2-5**） |
| **恒等式** | 勝利 = 0.5・追加ターン = `0.5/R` | Σλ = 0.5 |
| **観測・印**（見る・選ぶ・ルール処理） | **0**（資源が動かない。選択の利得は能力ごとに 1 回だけ足す） | — |
| **盤面依存**（元々のパワーを X にする・入れ替える） | **`None`**（盤面を渡せば値が出る） | — |
| **`OTHER`** | **常に `None`**（パーサの「判らない」箱） | — |

> **`0` と `None` を混ぜない**——「効果が無い（資源が動かない）」と「値付けできていない」は
> 別のことである（`exit_ledger` の除去で踏んだ形）。**観測・印は 0 で正しい**（本当に何も動かない）。
> **盤面依存と `OTHER` だけが `None`。**

## 読み方（事前登録）

- **値が付いた能力の割合**（`coverage`）が上がるほど、橋（T28）の無言の行が減る。
- **`None` の内訳**を出す＝残りが「盤面が要る」のか「パーサが読めていない」のか分かれる。
- **「〜まで」（`is_up_to`）は満額**＝得なら全部取るので上限として読む（#39 と同じ形）。
- **個数は帯の容量で打ち切る**——場は 5・ライフは 5・ドンは 10・手札は実測 4.88。
  打ち切らないと `count = 50` が通って**勝率 −5.4 の能力**が出る（2026-09-14 に実際出た）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/effect_value.py --out ~/effect_value.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from theory_order import KO_P, LAM, MU, THETA  # noqa: E402

#: 実測の価格（`game_theory.md` §18）。**本器は新しい定数を足さない**。
DELTA = 0.0277
#: 場のキャラ 1 体の平均価格（§18 の実測 0.1087）。**パワーが判る場合は帯で置き換える**。
NU_AVG = 0.1087
#: `BUFF` の換算——**ドン 1 個 = +1000 パワー**なので `δ` で割り当てる（新しい定数ではない）。
POWER_PER_DON = 1000.0
#: 残りターン数（実測 4.128・§18）。**テンポ＝在庫 ÷ これ**。
R_TURNS = 4.128
#: 1 ターンの価値 = `0.5 / R`（§18 の `w` = 0.1211 と同じ導出）。`EXTRA_TURN` に使う。
W_TURN = 0.5 / R_TURNS
#: ブロッカーの上乗せ（実測 +0.035・`nu_measure.py`）。**キーワードの値付けに使う**。
BLOCK_PREMIUM = 0.035
#: トリガーの価値（枚・実測 0.148＝相手ターン・`2026-09-14_theta_price.md`）。
#: `Θ = λ/μ − 1 − τ_value` の `τ_value` そのもの＝**バニッシュが消す量**。
TAU_VALUE = 0.148
#: 個数が `-1`（すべて）のときに置く数——**場の上限 5 体**。ゾーンが判ればそちらを使う。
ALL_COUNT = 5
#: **帯の容量**（個数の打ち切りに使う）。場・ライフ・ドンは規則から、**手札は実測 4.88**
#: （w41 の 52,433 行の平均）。`count = 50` のような値を素通しすると桁が壊れる。
ZONE_CAPACITY = {"FIELD": 5.0, "LIFE": 5.0, "DON": 10.0, "COST_AREA": 10.0,
                 "HAND": 4.88, "TEMP": 5.0, "TRASH": 5.0, "DECK": 5.0}
#: **能力 1 つの値が判らないときに置く量**（**暫定 P2-5**・§0.4 の台帳）。
#: 値付けできた能力の**中央値が `μ` に一致した**（独立の一致）ので `μ` を置く。
#: 感度は `[0.5μ, 2μ]`（`--ability-unknown` で振る）。
ABILITY_UNKNOWN = MU
#: 能力の参照（`EXECUTE_MAIN_EFFECT` 等）を再帰で解く深さの上限。
MAX_DEPTH = 2

# ---------------------------------------------------------------------------
# 母数——**エンジンが定義している動作**
# ---------------------------------------------------------------------------


def engine_actions():
    """**エンジンの動作の全集合**（`ActionType` の 62 メンバ・エイリアス除く）。

    ここを表から作らずに enum から読むのが眼目——**表に書き忘れた動作が見えなくなる**のを
    構造的に防ぐ（初版はカードの出現から表を作ったので実在しない名前が混ざった）。
    """
    from opcg_sim.src.models.enums import ActionType
    return tuple(sorted({m.name for m in ActionType}))


#: 通貨に直接写る動作（**符号は `target.player` から決める**）
PRICED = {
    "KO": "nu_loss",
    "BOUNCE": "bounce",
    "DISCARD": "hand_loss",
    "DRAW": "hand_gain",
    "PLAY_CARD": "play",
    "BUFF": "power",
    "BP_BUFF": "power",
    "HEAL": "life_gain",
    "LIFE_RECOVER": "life_gain",
    "LIFE_MANIPULATE": "life_gain",
    "DEAL_DAMAGE": "life_loss",
    "RAMP_DON": "don_gain",
    "ACTIVE_DON": "don_gain",
    "ATTACH_DON": "don_gain",
    # **ドンがドンデッキへ戻る＝恒久の損**（`REST_DON` は戻ってくるのでテンポ）
    "RETURN_DON": "don_loss",
    # コストの増減はそのぶんのドン（`SET_COST` は絶対値なので盤面依存）
    "COST_BUFF": "cost",
    "COST_CHANGE": "cost",
}
#: **ドンが減る動作**（持ち主の損）。増える動作（`RAMP_DON` 等）と向きが逆。
DON_LOSS = ("RETURN_DON",)
#: **カードが場所を移る動作**——`zone`（どこから）と `destination`（どこへ）で意味が決まる。
MOVE_KINDS = ("MOVE_CARD", "MOVE", "MOVE_TO_HAND", "DECK_BOTTOM", "TRASH",
              "TRASH_FROM_DECK", "DECK_TOP")
#: 移った先が書かれていない動作の既定の行き先
MOVE_DEFAULT_DEST = {"DECK_BOTTOM": "DECK", "DECK_TOP": "DECK", "TRASH": "TRASH",
                     "TRASH_FROM_DECK": "TRASH", "MOVE_TO_HAND": "HAND"}
#: 移った先が書かれていない動作の既定の出どころ
MOVE_DEFAULT_ZONE = {"TRASH_FROM_DECK": "DECK", "DECK_TOP": "DECK"}
#: **テンポ**（一時的に使えなくする／使えるようにする）→ **在庫 ÷ 残りターン × ターン数**。
#: `stock` は動く在庫（`nu`＝体・`don`＝ドン・`attack`＝攻撃 1 回）、`turns` は既定のターン数、
#: `gain` は持ち主が**得する**向きか（`ACTIVE` は自分の体がもう 1 回動く）。
TEMPO = {
    "REST": ("nu", 1.0, False),
    "LOCK": ("nu", 1.0, False),
    "ATTACK_DISABLE": ("nu", 1.0, False),
    "PREVENT_REST": ("nu", 1.0, False),
    "RESTRICTION": ("nu", 1.0, False),
    # **凍結は次のリフレッシュでも起きない**＝2 ターンぶん
    "FREEZE": ("nu", 2.0, False),
    "ACTIVE": ("nu", 1.0, True),
    "REST_DON": ("don", 1.0, False),
    "FREEZE_DON": ("don", 2.0, False),
    "MODIFY_DON_PHASE": ("don", 1.0, False),
    # 攻撃の宛先を変える＝攻撃 1 回ぶんを付け替える
    "REDIRECT_ATTACK": ("attack", 1.0, True),
}
#: `duration` → ターン数（テンポと付与の長さ）。`PERMANENT` は残り全部。
DURATION_TURNS = {"THIS_BATTLE": 0.5, "THIS_TURN": 1.0, "UNTIL_NEXT_TURN_END": 2.0,
                  "PERMANENT": R_TURNS}
#: **キーワードの値**（`status` に日本語で入る）。**どれも既存の実測値からの導出**。
#:
#: | 様式 | 意味 | 期間の扱い |
#: |---|---|---|
#: | `stock` | 持っている間ずっと効く在庫（ブロッカー） | **掛けない**（実測が既に在庫の値） |
#: | `once`  | **1 回だけ得をする**（速攻＝出たターンに 1 回殴れる） | **掛けない** |
#: | `turn`  | **攻撃ごとに効く**（ダブルアタック・バニッシュ） | ターン数を掛ける |
#:
#: **`once` と `turn` を分けるのが要**——速攻に期間を掛けると「永続の速攻」が
#: 攻撃 R 回ぶんになり、5 体に付けば勝率 1.3 になる（2026-09-14 に実際に出た）。
KEYWORD_PER_TURN = {
    # ブロッカー: 実測の上乗せ（`nu_measure`）
    "ブロッカー": ("stock", BLOCK_PREMIUM),
    "ブロック不可": ("stock", BLOCK_PREMIUM),      # 相手のブロッカーを無効にするのと同じ量
    # 速攻: 出たターンに殴れる＝**攻撃 1 回きり**
    "速攻": ("once", THETA * MU),
    "速攻:キャラ": ("once", THETA * MU),
    "ATTACK_ACTIVE": ("once", THETA * MU),
    # ダブルアタック: 通ればライフ 1 枚ぶん余分に削る＝**攻撃ごと**
    "ダブルアタック": ("turn", THETA * MU),
    # バニッシュ: 削ったライフのトリガーを消す＝`τ_value` 枚ぶん（攻撃ごと）
    "バニッシュ": ("turn", TAU_VALUE * MU),
}
#: キーワードを付与する動作
KEYWORD_KINDS = ("GRANT_KEYWORD", "KEYWORD")
#: **生存**（場を離れない・代わりに〜）→ **体が消える確率 × 体**（`ko_p` は実測 0.289）
SURVIVE_KINDS = ("PREVENT_LEAVE", "REPLACE_EFFECT")
#: **能力そのものを足す／消す動作**——同じカードを指すなら再帰で解き、判らなければ `μ`。
#: `gain` は持ち主が得する向きか（`NEGATE_EFFECT` は相手の能力を消す＝対象の損）。
ABILITY_KINDS = {"EXECUTE_MAIN_EFFECT": True, "EXECUTE_EVENT": True,
                 "GRANT_EFFECT": True, "PASSIVE_EFFECT": True,
                 "NEGATE_EFFECT": False, "DISABLE_ABILITY": False}
#: **恒等式で決まる動作**（勝率そのもの）
IDENTITY = {"VICTORY": 0.5, "EXTRA_TURN": W_TURN}
#: **観測**——資源は動かないので **0**。ただし**見た枚数 `k` は能力ごとの選択の利得に効く**。
OBSERVE_KINDS = ("LOOK", "LOOK_LIFE", "REVEAL", "SHUFFLE", "ORDER_LIFE", "FACE_UP_LIFE",
                 "DECLARE_COST")
#: **印**（対象を保存する・ルール上の読み替え）——資源が動かないので **0**。
MARKER_KINDS = ("SELECT", "SELECT_OPTION", "RULE_PROCESSING", "MOVE_ATTACHED_DON")
#: **盤面が要る動作**（絶対値を設定する・入れ替える）——盤面を渡さない限り `None`。
BOARD_KINDS = ("SET_BASE_POWER", "SET_COST", "SWAP_POWER")
#: **パーサの「判らない」箱**——**常に `None`**（ここを 0 にすると穴が見えなくなる）。
UNKNOWN_KINDS = ("OTHER",)
#: **体を持つカードの種別**——`ν` を当てられるのはここだけ（ユーザ指摘 2026-09-14）。
#: **ステージは体を持たない**（T16・`2026-09-14_bodyless_and_power.md`）ので `ν` に入れない。
BODY_TYPES = ("CHARACTER", "LEADER")
#: **`PLAY_CARD` は「どこから出すか」で手札の損が付くかが変わる**（ユーザ指摘 2026-09-14）。
PLAY_FROM_HAND_ZONES = ("HAND",)

#: 動作 → 類（`family_of`）。**エンジンの全メンバがちょうど 1 つに入る**ことをテストが縛る。
_CLASSES = (
    ("currency", tuple(PRICED)),
    ("move", MOVE_KINDS),
    ("tempo", tuple(TEMPO)),
    ("keyword", KEYWORD_KINDS),
    ("survive", SURVIVE_KINDS),
    ("ability", tuple(ABILITY_KINDS)),
    ("identity", tuple(IDENTITY)),
    ("observe", OBSERVE_KINDS),
    ("marker", MARKER_KINDS),
    ("board", BOARD_KINDS),
    ("unknown", UNKNOWN_KINDS),
)


def family_of(action_type):
    """その動作がどの類か（`currency`／`move`／`tempo`／…／`unknown`）。

    **エンジンに無い名前は `absent`** を返す——表に架空の名前を書いた事故
    （`COST_REDUCTION` 等）をここで見えるようにする。
    """
    for fam, kinds in _CLASSES:
        if action_type in kinds:
            return fam
    return "absent"


# ---------------------------------------------------------------------------
# 値付けの部品
# ---------------------------------------------------------------------------


def _zone(target, key="zone"):
    """`zone` は list で来ることもある（選べるゾーン）。**選ぶ側が得な方を採る**ので
    ここでは最初に返し、呼び出し側が max を取る（`_zones` を使う）。"""
    z = (target or {}).get(key)
    if isinstance(z, list):
        return [str(x).upper() for x in z]
    return [str(z).upper()] if z else []


def _capacity(zones):
    """その帯の容量（個数の打ち切り）。ゾーンが判らなければ場の上限で代表する。"""
    caps = [ZONE_CAPACITY[z] for z in zones if z in ZONE_CAPACITY]
    return max(caps) if caps else float(ALL_COUNT)


def _count(target, zones=None):
    """対象の個数。**`-1`（すべて）も大きすぎる数も帯の容量で打ち切る**。

    打ち切らないと `count = 50`（「相手は手札をすべて捨てる」等の解析結果）が素通しになり、
    **勝率 −5.4 の能力**が出る（2026-09-14 に実際に出た）。
    """
    cap = _capacity(zones or _zone(target))
    if not target:
        return 1.0
    n = target.get("count")
    if n is None:
        return 1.0
    n = float(n)
    return cap if n < 0 else min(n, cap)


def _magnitude(effect):
    """`value` の大きさ（`base × multiplier / divisor`）。"""
    v = effect.get("value") or {}
    base = float(v.get("base") or 0.0)
    mul = float(v.get("multiplier") or 1.0)
    div = float(v.get("divisor") or 1.0) or 1.0
    return base * mul / div


def is_body(target):
    """**その対象は体を持つか**——`ν` を当てられるのは `CHARACTER`／`LEADER` だけ。

    `card_type` が空の動作も多い（全体の 4 割）ので、**書かれているときだけ除外に使う**。
    """
    types = (target or {}).get("card_type") or []
    if not types:
        return True                     # 書かれていない＝体かどうか判らない（通す）
    return any(str(t).upper() in BODY_TYPES for t in types)


def field_value(target, nu=NU_AVG):
    """**場に居る 1 枚の価値**。体（キャラ・リーダー）なら `ν`、
    **体を持たない常設（ステージ）なら「能力 1 つ」**（`ABILITY_UNKNOWN`）。

    T16 の「ステージは体を持たない」は**`ν` を当てるな**という意味で、
    **価値が 0 という意味ではない**——ステージの値打ちはその常在効果なので、
    判らない能力 1 つとして数える（`0` にすると除去が無料になり、`ν` にすると体を数える）。
    """
    return nu if is_body(target) else ABILITY_UNKNOWN


def _side(target):
    """**誰に効くか**——`OPPONENT` なら相手の資源が動く（自分にとっては逆符号）。"""
    if not target:
        return "SELF"
    return str(target.get("player") or "SELF").upper()


def power_value(power, delta=DELTA, theta=THETA, mu=MU, nu=NU_AVG):
    """**パワーの価値**＝ドン換算、**ただし 1 体から引き出せる上限で打ち切る**。

    `+1000` ≈ ドン 1 個は既定の換算だが、**そのまま線形に伸ばすと理論自身の飽和と矛盾する**
    ——リーダーを狙う攻撃は `Θ·μ` で頭打ちになり、キャラを倒しても `ν` しか取れない
    （§14.1 の `attack_value`）。**上限は `max(Θ·μ, ν)`**＝その 2 つの上限のどちらか。

    これを入れないと `パワー+792000`（パーサの生成物に実在）が**勝率 +21.7** になる。
    """
    amt = abs(power) / POWER_PER_DON * delta
    return min(amt, max(theta * mu, nu))


def move_value(zone, dest, mu=MU, lam=LAM, nu=NU_AVG):
    """**1 枚が `zone` から `dest` へ移ったときの持ち主の損得**（読めなければ `None`）。

    体（場）・札（手札）・ライフの 3 つだけが通貨で、**デッキとトラッシュは通貨を持たない**
    （順番と情報は別の系統＝選択の利得として能力ごとに 1 回だけ足す）。
    """
    src_v = {"FIELD": nu, "HAND": mu, "LIFE": lam,
             "DECK": 0.0, "TRASH": 0.0, "TEMP": 0.0, "COST_AREA": 0.0}
    dst_v = dict(src_v)
    if zone not in src_v or dest not in dst_v:
        return None
    return dst_v[dest] - src_v[zone]


def _sel_premium(k):
    """**k 枚見て 1 枚選べることの利得**＝`E[max_k W] − E[W]`（`W` はカードの中身の価値）。

    **つまみが無いのが眼目**——`W` の分布は**同梱のカードと既存の価格から計算する**ので
    選ぶ余地が無い。分布は遅延計算して使い回す（`selection_dist`）。
    """
    if k is None or k < 2:
        return 0.0
    d = selection_dist()
    if not d:
        return 0.0
    n = len(d)
    k = int(min(k, 10))
    mean = sum(d) / n
    emax = sum(d[i] * (((i + 1) ** k) - (i ** k)) / (n ** k) for i in range(n))
    return max(0.0, emax - mean)


def selection_dist(cards=None):
    """**カードの中身の価値の分布**（昇順・選択の利得を出すためだけに使う）。

    `W = 体（ν の帯） + 登場時の効果 − 費用·δ`。**選択の項を切って計算する**
    （切らないと選択の利得が自分自身に依る＝無限再帰）。
    """
    if "sel" in _CACHE:
        return _CACHE["sel"]
    cards = cards or _all_cards()
    try:
        sys.path.insert(0, os.path.join(_ROOT, "tests"))
        from opcg_sim.learned.train import plan_labels as PL
        info = PL.Cards()
    except Exception:
        _CACHE["sel"] = []
        return []
    out = []
    for cid in cards:
        i = info.info(cid) or {}
        if i.get("leader"):
            continue
        body = 0.0 if (i.get("event") or i.get("stage")) else _band_nu(i.get("power"))
        v, _unp = card_value(cid, ON_PLAY_TRIGGERS, cards=cards, selection=False)
        if v is None:
            continue
        out.append(body + v - float(i.get("cost") or 0) * DELTA)
    out.sort()
    _CACHE["sel"] = out
    return out


def _band_nu(power):
    """パワーの帯ごとの `ν`（実測・§18）。分布を作るときだけ使う。"""
    if power is None:
        return NU_AVG
    p = float(power)
    if p < 5000:
        return 0.0690
    if p <= 7000:
        return 0.1503
    return 0.2112


# ---------------------------------------------------------------------------
# 1 つの動作
# ---------------------------------------------------------------------------


def action_value(effect, mu=MU, lam=LAM, delta=DELTA, nu=NU_AVG, theta=THETA,
                 ko_p=KO_P, card=None, depth=0):
    """**1 つの動作の価値**（自分から見た勝率）。値付けできなければ `None`、
    **資源が動かない動作は 0**（観測・印）。

    `target.player == "OPPONENT"` なら**相手の資源が動く**＝零和なので符号が反転する。
    """
    at = str(effect.get("type") or "")
    target = effect.get("target") or {}
    side = _side(target)
    opp = (side == "OPPONENT")
    zones = _zone(target)
    n = _count(target, zones)
    fam = family_of(at)

    if fam in ("observe", "marker"):
        return 0.0
    if fam in ("board", "unknown", "absent"):
        return None
    if fam == "identity":
        v = IDENTITY[at]
        return v if not opp else -v
    if fam == "tempo":
        stock, turns, gain = TEMPO[at]
        per = {"nu": nu, "don": delta, "attack": theta * mu * R_TURNS}[stock]
        turns = DURATION_TURNS.get(str(effect.get("duration") or ""), turns)
        amt = n * per * turns / R_TURNS
        # 相手のものを止めれば自分の得・自分のものが止まれば損（`gain` は向きの既定）
        if gain:
            return amt if not opp else -amt
        return amt if opp else -amt
    if fam == "keyword":
        got = KEYWORD_PER_TURN.get(str(effect.get("status") or ""))
        if got is None:
            return None                       # 知らないキーワード（0 にしない）
        mode, per = got
        if mode == "turn":
            per *= DURATION_TURNS.get(str(effect.get("duration") or ""), 1.0)
        amt = n * per                         # `stock`／`once` は期間を掛けない
        return amt if not opp else -amt
    if fam == "survive":
        amt = n * ko_p * field_value(target, nu)
        return amt if not opp else -amt
    if fam == "ability":
        gain = ABILITY_KINDS[at]
        amt = n * _referenced_ability(at, effect, card, mu, lam, delta, nu, theta,
                                      ko_p, depth)
        if at == "EXECUTE_EVENT":
            amt -= mu                         # 手札のイベントを 1 枚払う
        if gain:
            return amt if not opp else -amt
        return amt if opp else -amt
    if fam == "move":
        dest = str(effect.get("destination") or MOVE_DEFAULT_DEST.get(at) or "").upper()
        zs = zones or [MOVE_DEFAULT_ZONE.get(at, "")]
        fv = field_value(target, nu)      # ステージは `ν` ではなく「能力 1 つ」
        vals = [move_value(z, dest, mu, lam, fv) for z in zs]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        # **ゾーンが選べるなら選ぶ側に得な方**（`is_up_to` と同じ扱い）
        per = max(vals) if not opp else min(vals)
        return (per * n) if not opp else -(per * n)

    # --- 通貨に直接写る動作 ---
    kind = PRICED.get(at)
    if kind is None:
        return None
    if kind == "nu_loss":
        amt = n * field_value(target, nu)
        return amt if opp else -amt
    if kind == "bounce":
        amt = n * (field_value(target, nu) - mu)   # 場から消えるが手札が 1 枚増える
        return amt if opp else -amt
    if kind == "play":
        zone = zones[0] if zones else ""
        fv = field_value(target, nu)
        per = fv - mu if zone in PLAY_FROM_HAND_ZONES else fv
        amt = n * per
        return amt if not opp else -amt
    if kind == "hand_gain":
        amt = n * mu
        return amt if not opp else -amt
    if kind == "hand_loss":
        amt = n * mu
        return amt if opp else -amt
    if kind == "life_gain":
        amt = n * lam
        return amt if not opp else -amt
    if kind == "life_loss":
        amt = n * lam
        return amt if opp else -amt
    if kind in ("don_gain", "don_loss", "cost"):
        # **ドンの枚数も打ち切る**——`base = 99`（パーサの「上限なし」）が素通しすると
        # 99·δ = 2.74 になる（2026-09-14 に実際に出た）。規則の上限は 10 枚。
        amt = min(max(1.0, abs(_magnitude(effect)) or n), ZONE_CAPACITY["DON"]) * delta
        if kind == "don_loss":
            return amt if opp else -amt
        if kind == "cost":
            # **コストが下がるのは持ち主の得**（払わずに済むドン）。上がるなら逆。
            down = _magnitude(effect) <= 0
            good = down if not opp else not down
            return amt if good else -amt
        return amt if not opp else -amt
    if kind == "power":
        amt = power_value(_magnitude(effect), delta, theta, mu, nu) * n
        up = _magnitude(effect) >= 0
        good = up if not opp else not up
        return amt if good else -amt
    return None


def _referenced_ability(at, effect, card, mu, lam, delta, nu, theta, ko_p, depth):
    """**参照された能力の値**。同じカードの起動メインなら**再帰で解く**、
    判らなければ **`μ`（暫定 P2-5）**。

    「能力 1 つ ≈ 札 1 枚」は**値付けできた能力の中央値が `μ` に一致した**という
    独立の実測から置いている（`2026-09-14_effect_value.md` §2）。
    """
    if at == "EXECUTE_MAIN_EFFECT" and card and depth < MAX_DEPTH:
        tot = 0.0
        hit = False
        for ab in (card.get("abilities") or []):
            if (ab.get("trigger") or ab.get("timing")) not in ACTIVATE_TRIGGERS:
                continue
            v, _u = ability_value(ab, mu, lam, delta, nu, theta, ko_p, card, depth + 1)
            if v is not None:
                tot += v
                hit = True
        if hit:
            return tot
    return ABILITY_UNKNOWN


# ---------------------------------------------------------------------------
# 能力・カード
# ---------------------------------------------------------------------------


def walk_actions(effect):
    """効果木から動作を平らに取り出す（`sub_effect` を辿る）。"""
    out = []
    if not isinstance(effect, dict):
        return out
    if effect.get("type"):
        out.append(effect)
    sub = effect.get("sub_effect")
    if isinstance(sub, dict):
        out.extend(walk_actions(sub))
    elif isinstance(sub, list):
        for e in sub:
            out.extend(walk_actions(e))
    for key in ("actions", "effects", "options"):
        v = effect.get(key)
        if isinstance(v, list):
            for e in v:
                out.extend(walk_actions(e))
    return out


def selection_k(actions):
    """**その能力が何枚から選べるか**（観測した枚数の最大）。

    **能力ごとに 1 回だけ足す**のが肝——`LOOK(5)` → `MOVE_CARD(TEMP→HAND)` →
    `DECK_BOTTOM(TEMP)` は**1 つの選択**なので、動作ごとに足すと 3 重に数える。
    """
    ks = []
    for e in actions:
        if family_of(str(e.get("type") or "")) != "observe":
            continue
        k = _magnitude(e) or _count(e.get("target") or {}, _zone(e.get("target") or {}))
        ks.append(float(k or 0))
    return max(ks) if ks else 0.0


def condition_factor(ab, st=None, offered=False):
    """**条件に掛かる係数**（`condition_value.py`・遅延 import）。

    `st` を渡さなければ 1.0＝**判らないものは割り引かない**（上限として読む）。
    `offered=True`（エンジンが候補に出した起動メイン）は**検査済みなので 1.0**。
    """
    if offered or not (ab or {}).get("condition"):
        return 1.0
    if st is None:
        return 1.0
    try:
        import condition_value as CV
    except Exception:
        return 1.0
    return CV.factor(ab, st)


def ability_value(ab, mu=MU, lam=LAM, delta=DELTA, nu=NU_AVG, theta=THETA, ko_p=KO_P,
                  card=None, depth=0, selection=True, st=None, offered=False):
    """**能力 1 つの価値**＝**条件** × （実行内容の和 ＋ 選択の利得 − コストの和）。

    **条件は足す項ではなく掛かる側**（ユーザ確認 2026-09-14）——成り立たなければ
    **実行内容もコストも起きない**ので、和ごと 0 になる。

    **値付けできない動作が 1 つでもあれば `None`**——**部分的に足して 0 扱いにしない**
    （「効果が小さい」と「読めていない」を混ぜない）。
    """
    unpriced = []
    total = 0.0
    acts = walk_actions(ab.get("effect"))
    for e in acts:
        v = action_value(e, mu, lam, delta, nu, theta, ko_p, card, depth)
        if v is None:
            unpriced.append((str(e.get("type") or "?"), family_of(str(e.get("type") or ""))))
        else:
            total += v
    if selection:
        total += _sel_premium(selection_k(acts))
    cost = ab.get("cost") or {}
    for e in walk_actions(cost):
        v = action_value(e, mu, lam, delta, nu, theta, ko_p, card, depth)
        if v is None:
            unpriced.append((str(e.get("type") or "?"), family_of(str(e.get("type") or ""))))
        else:
            total -= abs(v)          # **コストは必ず損**（向きは表ではなく役割で決まる）
    if unpriced:
        return None, unpriced
    return total * condition_factor(ab, st, offered), []


#: 登場時に解決する契機（イベントを `PLAY` したときに効くもの）。
#: **イベントの本文は `ACTIVATE_MAIN` として入っていることがある**（実測・EB04-009）ので含める。
ON_PLAY_TRIGGERS = ("ON_PLAY", "ACTIVATE_MAIN", "MAIN", "RULE", "PASSIVE", None)
#: 起動メインの契機
ACTIVATE_TRIGGERS = ("ACTIVATE_MAIN",)
_CACHE = {}


def _all_cards():
    if "cards" not in _CACHE:
        _CACHE["cards"] = load_cards()
    return _CACHE["cards"]


def card_value(cid, triggers, nu=NU_AVG, mu=MU, lam=LAM, delta=DELTA, cards=None,
               selection=True, st=None, offered=False):
    """**カード 1 枚の、その契機での価値**を `(値, 読めなかった動作)` で返す。

    同じ契機の能力が複数あれば**和**を取る（同時に解決するので）。
    **1 つでも読めない能力があれば値は `None`**——部分的に足して 0 扱いにしない
    （`ability_value` と同じ規約）。`nu` を渡せば盤面の帯で置き換えられる（配線側から渡す）。
    """
    c = (cards or _all_cards()).get(str(cid) or "")
    if not c:
        return None, [("<no_card>", "other")]
    hit = [ab for ab in (c.get("abilities") or [])
           if (ab.get("trigger") or ab.get("timing")) in triggers]
    if not hit:
        return None, [("<no_ability_for_trigger>", "other")]
    total = 0.0
    unp = []
    for ab in hit:
        v, u = ability_value(ab, mu, lam, delta, nu, card=c, selection=selection,
                             st=st, offered=offered)
        if v is None:
            unp.extend(u)
        else:
            total += v
    if unp:
        return None, unp
    return total, []


def load_cards(path=None):
    p = path or os.path.join(_ROOT, "opcg_sim", "data", "opcg_effects.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["cards"]


def summarise(cards, mu=MU, lam=LAM, delta=DELTA, nu=NU_AVG):
    """**どれだけ値付けできたか**と、できなかった動作の類。"""
    ok = 0
    ng = 0
    fams = Counter()
    kinds = Counter()
    vals = []
    for cid, c in cards.items():
        for ab in (c.get("abilities") or []):
            v, unp = ability_value(ab, mu, lam, delta, nu, card=c)
            if v is None:
                ng += 1
                for k, f in unp:
                    fams[f] += 1
                    kinds[k] += 1
            else:
                ok += 1
                vals.append(v)
    out = {"abilities": ok + ng, "priced": ok, "unpriced": ng,
           "coverage": round(ok / max(1, ok + ng), 4),
           "unpriced_by_family": dict(fams.most_common()),
           "unpriced_top_kinds": dict(kinds.most_common(15))}
    if vals:
        vals.sort()
        out["value"] = {"mean": round(sum(vals) / len(vals), 5),
                        "p10": round(vals[len(vals) // 10], 5),
                        "median": round(vals[len(vals) // 2], 5),
                        "p90": round(vals[9 * len(vals) // 10], 5),
                        "max_abs": round(max(abs(vals[0]), abs(vals[-1])), 5),
                        "negative_share": round(sum(1 for v in vals if v < 0) / len(vals), 4)}
    return out


def action_census(cards):
    """**エンジンの全メンバ**について、類・同梱での出現数・値付けできた数を出す。

    **母数はエンジン**（出現 0 の動作も行として出る）——ここが本器の眼目
    （ユーザ指摘 2026-09-14「数え方はエンジンベースで数えて」）。
    """
    seen = Counter()
    priced = Counter()
    for cid, c in cards.items():
        for ab in (c.get("abilities") or []):
            for e in walk_actions(ab.get("effect")) + walk_actions(ab.get("cost") or {}):
                at = str(e.get("type") or "?")
                seen[at] += 1
                if action_value(e, card=c) is not None:
                    priced[at] += 1
    rows = []
    for at in engine_actions():
        rows.append({"action": at, "family": family_of(at), "occurrences": seen[at],
                     "priced": priced[at]})
    extra = sorted(set(seen) - set(engine_actions()))
    return {"rows": rows, "in_engine": len(rows),
            "unclassified": [r["action"] for r in rows if r["family"] == "absent"],
            "outside_engine": extra}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--effects", default="", help="`opcg_effects.json`（既定は同梱のもの）")
    ap.add_argument("--census", action="store_true", help="エンジンの全動作の表も出す")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    cards = load_cards(a.effects or None)
    res = {"frozen": {"mu": MU, "lambda": LAM, "delta": DELTA, "nu_avg": NU_AVG,
                      "theta": THETA, "ko_p": KO_P, "r_turns": R_TURNS,
                      "w_turn": round(W_TURN, 4), "block_premium": BLOCK_PREMIUM,
                      "tau_value": TAU_VALUE,
                      "note": "§18 の実測値。**本器は新しい定数を足さない**"},
           "provisional": {"ability_unknown": ABILITY_UNKNOWN,
                           "note": "P2-5＝能力 1 つ ≈ 札 1 枚（感度 [0.5μ, 2μ]）"},
           "selection": {"k2": round(_sel_premium(2), 5), "k3": round(_sel_premium(3), 5),
                         "k5": round(_sel_premium(5), 5),
                         "pool": len(selection_dist(cards))},
           "summary": summarise(cards)}
    if a.census:
        res["census"] = action_census(cards)
    res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
