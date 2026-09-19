"""除去の「型」を色ごとに制御して差し込む合成デッキ（教材の対照・2026-09-10・計画 §20.8.1）。

**なぜ要るか**（教材の穴・§20.7.11 の実測）: `deck_synth` は**テーマ**でデッキを組む＝役割を見ない。
143 リーダー × 2 seed の実測で **除去 0 枚が 27%・減少 0 枚 41%・ロック 0 枚 45%** ＝自己対戦の棋譜に
「除去を打った／打たなかった」の対照がほとんど無い。z しか教師が無いので、**教材に無い対照は
学べない**（鶏と卵）。本器は `deck_dig.inject_dig` と同じ流儀で、**その色で組める型**の除去を
一定率で差し込み、**各局にその型を記録する**（`record_gen` の `deck_kinds` 列）。

**人は型の価値を書かない**。差し込みは対照を作るためだけで、良し悪しは z が決める（ユーザ決定
2026-09-09・§20.8 の冒頭）。記録した型は**層別の読み**（除去向きでないリーダーまで除去を高く
見ていないか）と**カバレッジの点検**（学べていないのか、教材に存在しないのか）に使う。

型の鍵 = `"<form>:<source>:<band>"`:
  form   … KO／bounce（手札へ）／deck（山札・ライフへのゾーン送り）／trash（直接トラッシュ）／
           lock（レスト・凍結・攻撃不可・レスト不可）／reduce（パワー・コスト減少）
  source … EVENT／ON_PLAY／ACTIVATE_MAIN／ON_ATTACK／TRIGGER／COUNTER／OTHER
  band   … c0-2／c3-5／c6+（カードの使用コスト）

**バウンスの分類（2026-09-09 の実測「相手対象のバウンス 0 種」の原因・§20.8.1-1）**:
パーサは「手札に戻す」を **`ActionType.BOUNCE`** で出す（`effects/parser.py` の
「手札に戻す / 手札に加える」節）。`MOVE_TO_HAND` は**同じ意味の別名**で、Rust 側も両者を同じ
ハンドラ（`per_target.bounce`）に載せている。ところが `n_rel_feat._REMOVAL_OPS`（と、それを写した
Rust `encode/tokens.rs::is_removal_op`）は **`MOVE_TO_HAND` だけ**を持ち `BOUNCE` を持たない＝
実カード 23 件のバウンスが「除去 0 種」に見えていた。ここでは両方を form=bounce として数える。
また `_REMOVAL_OPS` は**対象ゾーンを見ない**ので、手札のデッキ下送り（12）・ライフのトラッシュ
送り（13）・ライフ操作の MOVE_CARD（32）まで「盤面の除去」に混ざっていた。本器は
**対象ゾーン FIELD** に限る（＝盤面から相手の駒が減る効果だけを除去と呼ぶ）。
`n_rel_feat`／Rust の符号化そのものは**触っていない**（直せば符号化が変わり golden と既存ネットの
入力分布が動く＝別 WP）。判断は `docs/reports/2026-09-10_removal_decks.RESULT.json` の notes。
"""
import random
import zlib

from opcg_sim.loop import deck_synth as DS
from opcg_sim.learned import n_rel_feat as NF
from opcg_sim.src.models.effect_types import ActionType, TriggerType
from opcg_sim.src.models.models import CardInstance

# --- 型の語彙 ---------------------------------------------------------------
FORMS = ("KO", "bounce", "deck", "trash", "lock", "reduce")
SOURCES = ("EVENT", "ON_PLAY", "ACTIVATE_MAIN", "ON_ATTACK", "TRIGGER", "COUNTER", "OTHER")
BANDS = ("c0-2", "c3-5", "c6+")

#: 相手の**盤面**（zone=FIELD）に撃つ効果だけを除去と数える（ライフ操作・手札干渉は別の役割）。
_FIELD_ZONES = ("FIELD",)

_FORM_OF_OP = {
    ActionType.KO: "KO",
    ActionType.BOUNCE: "bounce",
    ActionType.MOVE_TO_HAND: "bounce",          # BOUNCE の別名（Rust も同じハンドラ）
    ActionType.DECK_BOTTOM: "deck",
    ActionType.DECK_TOP: "deck",
    ActionType.MOVE_CARD: "deck",               # 盤面 → ライフ/デッキ のゾーン送り
    ActionType.TRASH: "trash",
    ActionType.DISCARD: "trash",
    ActionType.REST: "lock",
    ActionType.FREEZE: "lock",
    ActionType.LOCK: "lock",
    ActionType.ATTACK_DISABLE: "lock",
    ActionType.PREVENT_REST: "lock",
}
#: 減少（パワー／コスト）。値が負のときだけ reduce（正の付与は自分向けのバフ）。
_RED_OPS = {ActionType.BP_BUFF, ActionType.BUFF, ActionType.COST_BUFF}

#: 差し込み率（seed から引く・0 も残す＝差し込み無しのデッキが 1/6）。
RATES = (0, 5, 10, 15, 20, 25)
#: その型を「組める」と言うための最低種類数（§20.8.1-2）。
MIN_KIND_CARDS = 3
#: 1 デッキに選ぶ型の数（組み合わせ型は 1 つで複数の要素を持つ）。
N_TEMPLATES = (1, 2)
#: 攻撃役（緑の {lock + 攻撃役} 用の疑似 form）。
ATTACKER_POWER = 5000

_CLASSIFY: dict = {}
_POOLS: dict = {}


def _zone_name(tgt):
    z = getattr(tgt, "zone", None)
    return getattr(z, "name", str(z) if z is not None else "")


def cost_band(master) -> str:
    c = getattr(master, "cost", 0) or 0
    return BANDS[0] if c <= 2 else (BANDS[1] if c <= 5 else BANDS[2])


def _source_of(master, ability) -> str:
    """発動源（EVENT／ON_PLAY／ACTIVATE_MAIN／ON_ATTACK／TRIGGER／COUNTER／OTHER）。

    トリガー／カウンターは**カード種別より先**（トリガー除去・カウンター除去は別の型）。
    次にイベント（【メイン】のイベントは「イベント」として数える＝§20.7.11 の表と同じ切り方）。
    """
    trig = getattr(ability, "trigger", None)
    if trig is TriggerType.TRIGGER:
        return "TRIGGER"
    if trig is TriggerType.COUNTER:
        return "COUNTER"
    if getattr(getattr(master, "type", None), "name", "") == "EVENT":
        return "EVENT"
    if trig is TriggerType.ON_PLAY:
        return "ON_PLAY"
    if trig is TriggerType.ACTIVATE_MAIN:
        return "ACTIVATE_MAIN"
    if trig is TriggerType.ON_ATTACK:
        return "ON_ATTACK"
    return "OTHER"


def _forms_of_ability(master, ability):
    """1 能力が相手の盤面へ撃つ form の集合（pure）。

    効果木だけでなく**能力コストも見る**（パーサが「相手のキャラ1枚を〜:…」をコスト側に置く
    カードがある＝OP09-101。ゾーン FIELD かつ相手対象という条件は同じなので誤検出は増えない）。
    """
    out = set()
    acts = []
    NF._walk(getattr(ability, "effect", None), acts)
    NF._walk(getattr(ability, "cost", None), acts)
    for a in acts:
        t = getattr(a, "type", None)
        tgt = getattr(a, "target", None)
        if not NF._is_opp(tgt) or _zone_name(tgt) not in _FIELD_ZONES:
            continue
        f = _FORM_OF_OP.get(t)
        if f is not None:
            out.add(f)
        elif t in _RED_OPS and NF._base(getattr(a, "value", None)) < 0:
            out.add("reduce")                   # パワー減少（-1000…）もコスト減少（-1…）も reduce
    return out


def classify(master) -> set:
    """マスター → 除去の「型」の集合（`"<form>:<source>:<band>"`・マスター単位でキャッシュ）。

    `n_rel_feat.roles_of` は符号化のための**役割**（removal/lock/reduce…）を返す器で、こちらは
    **教材の対照を作るための型**を返す（分類はここに 1 本化する・§20.8.1-1）。
    """
    key = getattr(master, "card_id", None) or id(master)
    hit = _CLASSIFY.get(key)
    if hit is not None:
        return hit
    band = cost_band(master)
    out = set()
    for ab in (getattr(master, "abilities", ()) or ()):
        src = _source_of(master, ab)
        for f in _forms_of_ability(master, ab):
            out.add(f"{f}:{src}:{band}")
    _CLASSIFY[key] = out
    return out


def is_attacker(master) -> bool:
    """攻撃役（緑の {lock + 攻撃役} の相方）＝パワー 5000 以上のキャラ（pure）。"""
    return (getattr(getattr(master, "type", None), "name", "") == "CHARACTER"
            and (getattr(master, "power", 0) or 0) >= ATTACKER_POWER)


# --- 型 × 色のテンプレート表 -------------------------------------------------
#: 型の**形**だけを書く（どの色で組めるかは DB から再計算する＝手書きで固定しない・§20.8.1-2）。
#: パターンは `"<form>:<source>:<band>"` で各要素に `*`（任意）を許す。
TEMPLATES = (
    # 「減少 → KO しきい値」の組み合わせ（黒・赤・紫でしか組めないはず＝§20.7.11 の読み）
    {"name": "reduce_then_ko", "parts": ("reduce:*:c0-2", "KO:*:c3-5")},
    {"name": "lock_and_attack", "parts": ("lock:*:*", "attacker:*:*")},      # 緑
    {"name": "deck_out", "parts": ("deck:*:*",)},                            # 青
    {"name": "trigger_removal", "parts": ("*:TRIGGER:*",)},                  # 黄
    {"name": "on_attack_removal", "parts": ("*:ON_ATTACK:*",)},              # 赤
    {"name": "activate_main_removal", "parts": ("*:ACTIVATE_MAIN:*",)},      # 黒
    {"name": "bounce", "parts": ("bounce:*:*",)},
    {"name": "trash", "parts": ("trash:*:*",)},
    {"name": "event_removal", "parts": ("*:EVENT:*",)},
)
TEMPLATE_NAMES = tuple(t["name"] for t in TEMPLATES)


def kind_matches(kind: str, pat: str) -> bool:
    """型の鍵がパターン（`*` を許す 3 要素）に一致するか（pure）。"""
    k, p = kind.split(":"), pat.split(":")
    return len(k) == len(p) == 3 and all(a == "*" or a == b for a, b in zip(p, k))


def part_cards(pool, pat: str):
    """プールのうちパターンに当たるマスター（card_id 順・pure）。

    form が `attacker` の擬似パターンだけは `is_attacker` で判定する（除去ではない相方）。
    """
    if pat.split(":")[0] == "attacker":
        return sorted((c for c in pool if is_attacker(c)), key=lambda c: c.card_id)
    return sorted((c for c in pool if any(kind_matches(k, pat) for k in classify(c))),
                  key=lambda c: c.card_id)


def color_pool(db, colors):
    """その色（集合）で構築できる効果持ちカード（硬い制約はリーダー依存なのでここでは見ない）。"""
    key = tuple(sorted(colors))
    hit = _POOLS.get(key)
    if hit is None:
        want = set(colors)
        hit = [c for c in DS.card_pool(db) if DS.colors_of(c) & want]
        _POOLS[key] = hit
    return hit


def leader_pool(db, leader):
    """そのリーダーのデッキに入れられるカード（色一致＋硬い制約＝`deck_synth._legal_for`）。"""
    lm = DS._master(leader)
    return [c for c in DS.card_pool(db) if DS._legal_for(lm, c)]


def buildable(pool):
    """プールで「組める」テンプレートの一覧（各要素に `MIN_KIND_CARDS` 種以上あること）。

    戻り値は `[{"name":…, "parts":(…), "cards":{part: [master,…]}}]`（parts の順は TEMPLATES 通り）。
    """
    out = []
    for t in TEMPLATES:
        cards = {p: part_cards(pool, p) for p in t["parts"]}
        if all(len(v) >= MIN_KIND_CARDS for v in cards.values()):
            out.append({"name": t["name"], "parts": t["parts"], "cards": cards})
    return out


def color_table(db, colors=("BLACK", "BLUE", "GREEN", "PURPLE", "RED", "YELLOW")):
    """色 → 組めるテンプレート名（§20.7.11 の実測表をコードで再計算した版）。"""
    return {col: [t["name"] for t in buildable(color_pool(db, {col}))] for col in colors}


def kind_counts(pool):
    """プールの型ごとの種類数（カバレッジ点検表・`{kind: n}`）。"""
    out = {}
    for c in pool:
        for k in classify(c):
            out[k] = out.get(k, 0) + 1
    return out


# --- 差し込み ---------------------------------------------------------------
def deck_kinds(cards) -> list:
    """デッキ（CardInstance か CardMaster の列）が持つ型の集合（ソート済み list）。"""
    out = set()
    for x in cards:
        out |= classify(DS._master(x))
    return sorted(out)


def _rng(leader_id, seed, salt=0):
    """(リーダー, seed) から決定論的な乱数（`zlib.crc32`＝実行ごとに変わらない・pure）。

    率をリーダーにも依らせるのは、同じ seed 帯でも**リーダーごとに違う率**が出るようにするため
    （seed だけだと 1 帯 1 率になり、型 × 色のカバレッジが率 1 種類に潰れる）。
    """
    return random.Random((seed * 7919 + 101 + salt) ^ zlib.crc32(str(leader_id).encode()))


def _counter_cards(masters):
    return sum(1 for m in masters if (getattr(m, "counter", 0) or 0) > 0)


def _plan(tmpls, have, n, rng):
    """差し込む札の並び（同名 `MAX_COPIES` まで・テンプレートの各要素を順に回す）。"""
    lanes = []
    for t in tmpls:
        for p in t["parts"]:
            pool = list(t["cards"][p])
            rng.shuffle(pool)                  # 種類の選び方だけ seed で振る（並びは決定論）
            lanes.append(pool)
    plan, taken = [], dict(have)
    i = 0
    while len(plan) < n and i < n * max(len(lanes), 1) * 4:
        lane = lanes[i % len(lanes)] if lanes else []
        i += 1
        for m in lane:
            if taken.get(m.name, 0) < DS.MAX_COPIES:
                plan.append(m)
                taken[m.name] = taken.get(m.name, 0) + 1
                break
        else:
            if all(taken.get(m.name, 0) >= DS.MAX_COPIES for ln in lanes for m in ln):
                break
    return plan


def _apply(cards, plan, owner, keep_kinds):
    """`plan` の札を末尾側の「型を持たない札」と差し替える（`deck_dig.inject_dig` と同じ流儀）。

    差し替え位置は**守り札（カウンター持ち）以外を先**に使う（合成デッキのカウンター枚数は
    18〜44 枚とばらつくので、素直に末尾から潰すと下限を割って差し込みが常に失敗する）。
    """
    out = list(cards)
    free = [k for k in range(len(out) - 1, -1, -1)
            if not (classify(DS._master(out[k])) & keep_kinds)]
    idx = ([k for k in free if not (getattr(DS._master(out[k]), "counter", 0) or 0)]
           + [k for k in free if (getattr(DS._master(out[k]), "counter", 0) or 0)])
    n = 0
    for k, m in zip(idx, plan):
        out[k] = CardInstance(m, owner)
        n += 1
    return out, n


def inject_roles(db, leader, cards, owner, seed=0):
    """合成デッキ `cards`（CardInstance×50）へ除去の型を差し込む（pure・決定論）。

    seed から差し込み率 r ∈ {0,5,10,15,20,25}% と、その色で組める型 1〜2 種を引き、`r×50` 枚を
    目安に差し込む。**死に札監査とカウンター比率を悪化させない**（割れたら 1 枚ずつ減らす）。
    戻り値 `(cards, kinds)`・`kinds` は `{"injected":[…],"native":[…],"rate":r,"colors":[…],
    "n":枚数,"templates":[…],"reduced":減らした枚数}`（`record_gen` の `deck_kinds` 列に載る JSON）。
    """
    lm = DS._master(leader)
    base = list(cards)
    base_masters = [DS._master(x) for x in base]
    kinds = {"injected": [], "native": deck_kinds(base_masters), "rate": 0,
             "colors": sorted(DS.colors_of(lm)), "n": 0, "templates": [], "reduced": 0}
    rng = _rng(getattr(lm, "card_id", ""), seed)
    rate = rng.choice(RATES)
    if rate <= 0:
        return base, kinds                                  # 差し込み無し（対照の基準・1/6）
    kinds["rate"] = rate                       # 引いた率は残す（n=0 なら「引いたが入らなかった」）
    tmpls = buildable(leader_pool(db, lm))
    if not tmpls:
        return base, kinds                                  # その色ではどの型も組めない
    k = min(rng.choice(N_TEMPLATES), len(tmpls))
    chosen = rng.sample(tmpls, k)
    keep = set()
    for t in chosen:
        for p in t["parts"]:
            keep |= {kk for c in t["cards"][p] for kk in classify(c)}
    have = {}
    for m in base_masters:
        have[m.name] = have.get(m.name, 0) + 1
    n_target = max(1, round(rate * DS.DECK_SIZE / 100))
    base_dead = DS.audit_deck(leader, base)["dead_rate"]
    base_ctr = _counter_cards(base_masters)
    for n in range(n_target, 0, -1):
        plan = _plan(chosen, have, n, _rng(getattr(lm, "card_id", ""), seed, salt=n))
        if not plan:
            continue
        out, got = _apply(base, plan, owner, keep)
        if got <= 0:
            continue
        masters = [DS._master(x) for x in out]
        if _counter_cards(masters) < min(DS.MIN_COUNTER_CARDS, base_ctr):
            continue                                        # 守り札を割った＝減らして再試行
        if DS.audit_deck(leader, out)["dead_rate"] > base_dead:
            continue                                        # 死に札を増やした＝減らして再試行
        random.Random(seed * 7919 + 31).shuffle(out)
        kinds.update({"injected": deck_kinds(plan[:got]), "n": got,
                      "templates": sorted(t["name"] for t in chosen),
                      "reduced": n_target - got})
        return out, kinds
    kinds["reduced"] = n_target                             # 1 枚も入れられなかった
    return base, kinds


def synth_roles_deck_builder(l1_id, l2_id=None, seed=0):
    """`run_game(deck_builder=…)` 互換（`deck_synth.synth_deck_builder` と同じ契約）。"""
    def _build(db, game_seed):
        l1, c1 = DS.synth_deck(db, l1_id, seed=seed, owner="p1")
        l2, c2 = DS.synth_deck(db, l2_id or l1_id, seed=seed + 1, owner="p2")
        c1, _ = inject_roles(db, l1, c1, "p1", seed=seed)
        c2, _ = inject_roles(db, l2, c2, "p2", seed=seed + 1)
        return l1, c1, l2, c2
    return _build
