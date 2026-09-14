"""**効果をコストと実行内容で値付けする**（P2-1・`game_theory.md` §14.1.3）。

ユーザ決定 2026-09-14「イベント、起動メイン等の効果については**コストと実行内容で整理する**」。

## 要点——**新しい定数を 1 つも足さない**

```
効果の価値 = Σ（実行内容が動かした通貨 × その価格） − Σ（コストが動かした通貨 × その価格）
```

**実行内容が既にある通貨に写れば、係数を選ぶ余地が無い。**
T16 が「勘定は複数の互換でない仕方で閉じられる」と言って項を足す作業を止めたのは
**係数を選べたから**（§7.4）。**本器では選べない**——写像表だけで、自由なつまみが無い。

> **コストも `actions` で書かれている**（`ab["cost"]["actions"]`）ので、
> **同じ写像表を符号を反転して使う**。これがユーザの整理がそのまま実装になる理由。

## 写像表（`opcg_effects.json` の `effect.type` → 通貨）

| 動作 | 動く通貨 | 備考 |
|---|---|---|
| `KO` | **`ν`**（体が消える） | `target.player` で符号が決まる |
| `BOUNCE` | `ν` − `μ`（体が手札に戻る） | 相手に撃てば相手の手札が増える |
| `TRASH` | `zone` で分かれる（`FIELD`→`ν` / `HAND`→`μ` / **`LIFE`→`λ`**） | |
| `DISCARD` | **`μ`** | |
| `DRAW` | **`μ`** | |
| `PLAY_CARD` | **`ν`**（体が増える）。**手札から出すなら `ν − μ`** | **`zone` を読む** |
| `BUFF` | **`δ` 換算**（`+1000` パワー ≈ ドン 1 個） | **新しい定数を作らない鍵** |
| `HEAL` / `LIFE_RECOVER` | **`λ`** | |
| `RAMP_DON` / `ACTIVE_DON` / `ATTACH_DON` / `RETURN_DON` | **`δ`** | **`REST_DON` は除く**（戻ってくる＝テンポ） |
| `COST_REDUCTION` | **`δ`**（払わずに済んだドン） | |

> **対象の種別と場所を読む**（ユーザ指摘 2026-09-14）——**ステージは体を持たない**ので
> `ν` を当てない・**`PLAY_CARD` は手札から出すときだけ札 1 枚の損が付く**・
> **`TRASH` はライフを落とすときは `λ`**。`card_type` が書かれていない動作も多いので、
> **書かれているときだけ除外に使う**（書いていないものを弾くと範囲が不当に狭まる）。

**上に無い動作は `None` を返す**（値付けできない）——**0 にしない**。
0 にすると「効果が無い」と「値付けできていない」が区別できなくなる
（`exit_ledger` の除去で踏んだ形）。

## まだ価格が無い系統（§14.1.3 の訂正後の表）

| 系統 | 代表 | 出現の目安 |
|---|---|---|
| **情報・デッキ順** | `LOOK`・`ARRANGE`・`DECK_BOTTOM` | 約 860——**第 5 の通貨かもしれない** |
| **テンポ** | `REST`・`ACTIVE`・`ATTACK_DISABLE` | 約 560（backlog #34） |
| **能力付与** | `GRANT_KEYWORD`・`REPLACE_EFFECT` | 約 350（**再帰的**） |

## 読み方（事前登録）

- **値が付いた効果の割合**（`coverage`）が上がるほど、橋の無言の行が減る。
- **`None` を返した動作の内訳**を出す＝**次に価格を付けるべき系統**が判る。
- **「〜まで」（`is_up_to`）は選択肢の max** ＝**得なら全部取る**ので満額で数える
  （損な効果を強制されるわけではない・#39 と同じ形）。

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

from theory_order import LAM, MU  # noqa: E402

#: 実測の価格（`game_theory.md` §18）。**本器は新しい定数を足さない**。
DELTA = 0.0277
#: 場のキャラ 1 体の平均価格（§18 の実測 0.1087）。**パワーが判る場合は帯で置き換える**。
NU_AVG = 0.1087
#: `BUFF` の換算——**ドン 1 個 = +1000 パワー**なので `δ` で割り当てる（新しい定数ではない）。
POWER_PER_DON = 1000.0
#: 個数が `-1`（「すべて」）のときに置く数——**場の上限 5 体**を超えないので 5 で打ち切る。
ALL_COUNT = 5

#: 値付けできる動作 → どの通貨がどれだけ動くか。
#: 値は (通貨, 1 単位あたりの量) で、**符号は `target.player` から決める**。
PRICED = {
    "KO": ("nu", 1.0),
    "BOUNCE": ("nu_minus_hand", 1.0),
    "DISCARD": ("hand", 1.0),
    "DRAW": ("hand", 1.0),
    "PLAY_CARD": ("nu", 1.0),
    "BUFF": ("power", 1.0),
    "HEAL": ("life", 1.0),
    "LIFE_RECOVER": ("life", 1.0),
    "RAMP_DON": ("don", 1.0),
    "ACTIVE_DON": ("don", 1.0),
    "ATTACH_DON": ("don", 1.0),
    "COST_REDUCTION": ("don", 1.0),
    # **`RETURN_DON` はドンがドンデッキへ戻る＝恒久の損**なので `δ` で書ける。
    # **`REST_DON`／`FREEZE_DON` は戻ってくる**＝テンポなので写さない（下の系統へ）。
    "RETURN_DON": ("don", 1.0),
}
#: **ドンが減る動作**（持ち主の損）。増える動作（`RAMP_DON` 等）と向きが逆。
DON_LOSS = ("RETURN_DON",)
#: **体を持つカードの種別**——`ν` を当てられるのはここだけ（ユーザ指摘 2026-09-14）。
#: **ステージは体を持たない**（T16・`2026-09-14_bodyless_and_power.md`）ので `ν` に入れない。
BODY_TYPES = ("CHARACTER", "LEADER")
#: **`PLAY_CARD` は「どこから出すか」で手札の損が付くかが変わる**（ユーザ指摘 2026-09-14）。
#: 手札から出せば札が 1 枚減るが、トラッシュ・デッキ・見たカードからならその損は無い。
PLAY_FROM_HAND_ZONES = ("HAND",)
#: `TRASH` は `zone` で意味が変わる（場なら体・手札なら札）ので別扱い。
ZONE_SPLIT = {"TRASH": {"FIELD": ("nu", 1.0), "HAND": ("hand", 1.0),
                        # **ライフを落とすのはライフの損**（`ν` でも `μ` でもない）
                        "LIFE": ("life_loss", 1.0),
                        "DECK": ("deck", 1.0)}}
#: **価格がまだ無い系統**（`None` を返す理由を分類して出す）
UNPRICED_FAMILY = {
    "info": ("LOOK", "LOOK_LIFE", "REVEAL", "ARRANGE", "SHUFFLE", "DECK_BOTTOM",
             "DECK_TOP", "TRASH_FROM_DECK", "MOVE_CARD", "MOVE"),
    "tempo": ("REST", "RESTED", "ACTIVE", "FREEZE", "LOCK", "ATTACK_DISABLE",
              "PREVENT_REST", "REDIRECT_ATTACK",
              # **ドンをレスト／凍結させるのはテンポ**——戻ってくるので `δ` の損ではない
              "REST_DON", "FREEZE_DON", "ACTIVE_DON_OPP", "MODIFY_DON_PHASE"),
    "grant": ("GRANT_KEYWORD", "GRANT_EFFECT", "KEYWORD", "REPLACE_EFFECT",
              "EXECUTE_MAIN_EFFECT", "PASSIVE_EFFECT", "DISABLE_ABILITY",
              "NEGATE_EFFECT", "PREVENT_LEAVE", "LEAVE"),
}


def family_of(action_type):
    """値付けできない動作がどの系統か（次に価格を付けるべき系統を出すため）。"""
    for fam, kinds in UNPRICED_FAMILY.items():
        if action_type in kinds:
            return fam
    return "other"


def _count(target):
    """対象の個数。`-1`（すべて）は場の上限で打ち切る。"""
    if not target:
        return 1
    n = target.get("count")
    if n is None:
        return 1
    n = int(n)
    return ALL_COUNT if n < 0 else n


def _magnitude(effect):
    """`value` の大きさ（`base × multiplier / divisor`）。"""
    v = effect.get("value") or {}
    base = float(v.get("base") or 0.0)
    mul = float(v.get("multiplier") or 1.0)
    div = float(v.get("divisor") or 1.0) or 1.0
    return base * mul / div


def is_body(target):
    """**その対象は体を持つか**——`ν` を当てられるのは `CHARACTER`／`LEADER` だけ。

    `card_type` が空の動作も多い（全体の 4 割）ので、**書かれているときだけ除外に使う**
    ——書かれていないものを弾くと値付けできる範囲が不当に狭まる。
    """
    types = (target or {}).get("card_type") or []
    if not types:
        return True                     # 書かれていない＝体かどうか判らない（通す）
    return any(str(t).upper() in BODY_TYPES for t in types)


def _side(target):
    """**誰に効くか**——`OPPONENT` なら相手の資源が動く（自分にとっては逆符号）。"""
    if not target:
        return "SELF"
    return str(target.get("player") or "SELF").upper()


def action_value(effect, mu=MU, lam=LAM, delta=DELTA, nu=NU_AVG):
    """**1 つの動作の価値**（自分から見た勝率・値付けできなければ `None`）。

    `target.player == "OPPONENT"` なら**相手の資源が動く**＝零和なので符号が反転する。
    """
    at = str(effect.get("type") or "")
    target = effect.get("target") or {}
    side = _side(target)
    n = _count(target)
    opp = (side == "OPPONENT")
    if at in ZONE_SPLIT:
        zone = str(target.get("zone") or "").upper()
        got = ZONE_SPLIT[at].get(zone)
        if got is None:
            return None
        kind = got[0]
    elif at in PRICED:
        kind = PRICED[at][0]
    else:
        return None
    if kind == "nu":
        # **体を持たない対象には `ν` を当てない**（ステージ等・ユーザ指摘 2026-09-14）
        if not is_body(target):
            return None
        gain = n * nu
        if at == "PLAY_CARD":
            # **どこから出すか**で手札の損が付く（手札からなら札 1 枚を失う）
            zone = str(target.get("zone") or "").upper()
            per = nu - mu if zone in PLAY_FROM_HAND_ZONES else nu
            gain = n * per
            return gain if not opp else -gain
        # **体が消える動作**（KO・TRASH from FIELD）は相手に撃てば得・自分に撃てば損
        return gain if opp else -gain
    if kind == "nu_minus_hand":
        if not is_body(target):
            return None
        # BOUNCE: 体は消えるが持ち主の手札が 1 枚増える
        gain = n * (nu - mu)
        return gain if opp else -gain
    if kind == "hand":
        amt = n * mu if at == "DRAW" else n * mu
        if at == "DRAW":
            return amt if not opp else -amt          # 引くのは持ち主の得
        return amt if opp else -amt                  # 落とすのは持ち主の損
    if kind == "life":
        amt = n * lam
        return amt if not opp else -amt
    if kind == "life_loss":
        # ライフを**落とす**＝持ち主の損（`HEAL` と向きが逆）
        amt = n * lam
        return amt if opp else -amt
    if kind == "don":
        amt = max(1.0, abs(_magnitude(effect)) or float(n)) * delta
        # **向きは動作で決まる**——増える動作（RAMP など）は持ち主の得、
        # **減る動作（`RETURN_DON`）は持ち主の損**。`DRAW` 対 `DISCARD` と同じ形。
        if at in DON_LOSS:
            return amt if opp else -amt
        return amt if not opp else -amt
    if kind == "power":
        # **`+1000` パワー ≈ ドン 1 個**——新しい定数を作らずに `δ` で換算する
        amt = (_magnitude(effect) / POWER_PER_DON) * delta * n
        return amt if not opp else -amt
    if kind == "deck":
        return None                                  # デッキ操作は価格が無い（情報の系統）
    return None


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


def ability_value(ab, mu=MU, lam=LAM, delta=DELTA, nu=NU_AVG):
    """**能力 1 つの価値**＝実行内容の和 − コストの和。

    **値付けできない動作が 1 つでもあれば `None`**——**部分的に足して 0 扱いにしない**
    （「効果が小さい」と「読めていない」を混ぜない）。
    """
    unpriced = []
    total = 0.0
    for e in walk_actions(ab.get("effect")):
        v = action_value(e, mu, lam, delta, nu)
        if v is None:
            unpriced.append((str(e.get("type") or "?"), family_of(str(e.get("type") or ""))))
        else:
            total += v
    cost = ab.get("cost") or {}
    for e in walk_actions(cost):
        v = action_value(e, mu, lam, delta, nu)
        if v is None:
            unpriced.append((str(e.get("type") or "?"), family_of(str(e.get("type") or ""))))
        else:
            total -= abs(v)          # **コストは必ず損**（向きは表ではなく役割で決まる）
    if unpriced:
        return None, unpriced
    return total, []


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


def card_value(cid, triggers, nu=NU_AVG, mu=MU, lam=LAM, delta=DELTA, cards=None):
    """**カード 1 枚の、その契機での価値**を `(値, 読めなかった動作)` で返す。

    同じ契機の能力が複数あれば**和**を取る（同時に解決するので）。
    **1 つでも読めない能力があれば値は `None`**——部分的に足して 0 扱いにしない
    （`ability_value` と同じ規約）。`nu` を渡せば盤面の帯で置き換えられる（配線側から渡す）。
    """
    c = (cards or _all_cards()).get(str(cid) or "")
    if not c:
        return None, [("<no_card>", "other")]
    hit = []
    for ab in (c.get("abilities") or []):
        t = ab.get("trigger") or ab.get("timing")
        if t in triggers:
            hit.append(ab)
    if not hit:
        return None, [("<no_ability_for_trigger>", "other")]
    total = 0.0
    unp = []
    for ab in hit:
        v, u = ability_value(ab, mu, lam, delta, nu)
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
    """**どれだけ値付けできたか**と、できなかった動作の系統。"""
    ok = 0
    ng = 0
    fams = Counter()
    kinds = Counter()
    vals = []
    for cid, c in cards.items():
        for ab in (c.get("abilities") or []):
            v, unp = ability_value(ab, mu, lam, delta, nu)
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
                        "negative_share": round(sum(1 for v in vals if v < 0) / len(vals), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--effects", default="", help="`opcg_effects.json`（既定は同梱のもの）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    cards = load_cards(a.effects or None)
    res = {"frozen": {"mu": MU, "lambda": LAM, "delta": DELTA, "nu_avg": NU_AVG,
                      "note": "§18 の実測値。**本器は新しい定数を足さない**"},
           "summary": summarise(cards), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
