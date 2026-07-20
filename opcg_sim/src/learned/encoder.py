"""学習evalスパイク D-1: 盤面エンコーダ（dev・docs/reports/cpu_learned_eval_spike_design_20260629.md §A/D）。

GameManager の状態を **to-move 視点**で固定長テンソルへ符号化する。Dual-Net の入力。
- **半生 numeric**：ライフ/ドン/手札数/場のキャラの cost/power/rest/付与don/キーワードflag 等
  （L1 が平坦化して捨てる相互作用を、ネットが学べる粒度で残す）。
- **カードID インデックス**：各カードの card_id を整数 idx へ（Embedding 用）。これが無いと「power8000のキャラ」
  止まりで「光月おでん」と認識できず＝L1超えの深層情報を拾えない（レビュー論点3）。

公平性：**相手手札の中身は符号化しない**（枚数のみ）。相手場/リーダーは公開情報なので符号化する。
決定的（盤面のみ参照・RNG不使用）＝同一局面→同一エンコード。numpy 実装（本走時 torch へ差し替え可）。

**version（符号化世代）**: v1=Gen2 出荷ネットの入力（scalars 14）。v2=**リーダー付与ドン**（自/相手）を
scalars に追加（16）。v1 はリーダーの付与ドンが完全に不可視で、「リーダーへのドン付与＝アクティブドンを
1枚失うだけの手」に見え、【ドン‼×1】条件のリーダー効果（OP11-041 ナミの防御+2000 等）を構造的に
学習できなかった。v3=山札/トラッシュ/KO数＋スロット別フラグ（ターン1使用済み/召喚酔い）で 46。
v4=**自デッキ残の集約**（残カウンター総量/密度・ブロッカー残・イベント残・高コストキャラ残の 5）で 51
＝「自分の山札にどれだけ守り札/カウンターが残るか」を可視化し、薄いライフの価値（C5）と残ターン読み
（D3）を底上げする（cpu_v5_plan.md §4-3）。v5=**相手場の脅威集約**（総火力/高パワー数/ブロッカー数）
＋**展開余力**（ドンで出せる手札キャラ数）で 55 ＝ policy が @33（相手無防備＝攻める）と @64（相手鉄壁
＝慎重）を区別でき、@93（ドン余剰＝展開すべき）も見える（cpu_v10・真盤面診断）。version は**ロード済み
ネットの入力次元から自動判別**する（cpu_learned 側）＝現行ネットは挙動不変・新版ネットへ差し替えた
時点で新特徴が有効になる。
"""
import numpy as np

MAX_FIELD = 5          # OPCG の場のキャラ上限
MAX_HAND = 10          # 自分の手札 ID を載せる上限（相手手札は枚数のみ＝公平）
KEYWORDS = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"]
PER_CHAR = 4 + len(KEYWORDS)   # [cost, power, is_rest, attached_don] + keyword flags
PAD = 0                # card_idx の PAD/UNK

# --- スカラー特徴の版マップ（拡張の唯一の seam） ------------------------------
# **不変条件（APPEND-ONLY）**: 新しい版は scalars を**末尾に追加**するだけ。既存の並びは
# 絶対に変更・並べ替えしない。これを守る限り、任意の版 old→new の温スタート（重み拡張）は
# 「old の重みをコピー＋末尾に増えたぶんゼロ行を挿入」で機械的に決まる（ValueNet/PolicyScorer
# .expanded()）。将来 v3 を足すときは (1) SCALARS_V3 を定義、(2) 下の dict に 1 行、(3) encode の
# version 分岐に append を足す——の3点だけで、拡張・温スタート・ドリフト検知が自動追従する。
SCALARS_V1 = 14        # v1 のグローバル数値特徴数（Gen2 出荷ネット）
SCALARS_V2 = 16        # v2 = v1 + [自リーダー付与ドン, 相手リーダー付与ドン]
SCALARS_V3 = 46        # v3 = v2 + [山札/トラッシュ/今ターンKO数 6] + [ターン1使用済み 12] + [召喚酔い 12]
SCALARS_V4 = 51        # v4 = v3 + 自デッキ残の集約5（残カウンター総量/密度・ブロッカー残・イベント残・高コストキャラ残）
SCALARS_V5 = 55        # v5 = v4 + 相手場の脅威集約3（総火力/高パワー数/ブロッカー数）＋展開余力1（ドンで出せる手札キャラ数）
_SCALARS_BY_VERSION = {1: SCALARS_V1, 2: SCALARS_V2, 3: SCALARS_V3, 4: SCALARS_V4, 5: SCALARS_V5}


def scalars_dim(version=1):
    """符号化世代 version のグローバル数値特徴数（append-only ＝ version が上がるほど単調増加）。"""
    if version not in _SCALARS_BY_VERSION:
        raise ValueError(f"未知の符号化世代 version={version}（_SCALARS_BY_VERSION に未登録）")
    return _SCALARS_BY_VERSION[version]


def known_versions():
    """登録済みの符号化世代（昇順）。次元→版の逆引き・拡張ループが版をハードコードしないため。"""
    return sorted(_SCALARS_BY_VERSION)


def build_vocab(db):
    """card_id → idx（1..N）。0=PAD/UNK。決定的（card_id ソート）。

    **注意（2026-07-15 実害）**: カードDBが増えるとソートの**途中挿入**で既存カードの idx がズレ、
    学習済みネットの Emb/EffF 行との対応が壊れる。学習済みネットと組む符号化は本関数でなく
    **ネット付属の vocab**（`ValueNet.vocab_ids` → `vocab_from_ids`）を使うこと。本関数は
    「新規ネットの初期 vocab を切る」用途のみ。"""
    ids = sorted(cid for cid in db.raw_db.keys() if db.get_card(cid) is not None)
    return {cid: i + 1 for i, cid in enumerate(ids)}


def vocab_from_ids(ids):
    """ネット付属の card_id 列（index 順）→ vocab dict（card_id → idx・0=PAD/UNK）。

    列に無いカード（ネットの訓練後に追加された新カード）は encode 側（`_vidx` の
    `vocab.get(..., PAD)`）で UNK=0 に落ちる＝範囲外参照もズレも起きない。"""
    return {cid: i + 1 for i, cid in enumerate(ids)}


def _power(c):
    try:
        return float(c.current_power)
    except Exception:
        return float(getattr(c.master, "power", 0) or 0)


def _char_feats(c):
    """場キャラ1体の numeric 特徴（power は 1e4 で正規化）。"""
    f = [float(getattr(c.master, "cost", 0) or 0) / 10.0,
         _power(c) / 10000.0,
         1.0 if getattr(c, "is_rest", False) else 0.0,
         float(getattr(c, "attached_don", 0) or 0) / 5.0]
    for kw in KEYWORDS:
        try:
            f.append(1.0 if c.has_keyword(kw) else 0.0)
        except Exception:
            f.append(0.0)
    return f


def _vidx(vocab, c):
    return vocab.get(getattr(getattr(c, "master", None), "card_id", None), PAD)


# v4: 自デッキ残（自分の残ライブラリ me.deck）の集約特徴。**カード個別に依存しない汎用量**
# （counter 値・keyword・type という全カード共通属性の集計のみ＝特定カードIDのハードコード無し）。
# 相手デッキは非公開なので対象外（公平性契約＝隠れ情報を符号化しない）。残ライフの precious 価値・
# 「あと何ターン守れるか」（時計）を、山札に残る守り資源から底上げする（D3/C5・cpu_v5_plan.md §4-3）。
_BLOCKER_KW = "ブロッカー"


def _deck_aggregate(deck):
    """me.deck（残ライブラリ）の守り/資源集約 5 値を返す。空デッキ・属性欠落に安全（探索クローン上で
    呼ばれるため決して例外を投げない）。正規化は有界（温スタートの恒等性は新W1行ゼロで保証されるので
    値域は学習安定性のためだけ・50=デッキ上限で割る/密度は分数）。"""
    n = 0
    counter_total = 0.0
    counter_cards = 0
    blockers = 0
    events = 0
    highcost_char = 0
    for c in deck:
        m = getattr(c, "master", None)
        if m is None:
            continue
        n += 1
        cv = getattr(m, "counter", 0) or 0
        counter_total += cv
        if cv > 0:
            counter_cards += 1
        try:
            if _BLOCKER_KW in (m.keywords or ()):
                blockers += 1
        except Exception:
            pass
        t = getattr(m, "type", None)
        tname = getattr(t, "name", None)
        if tname == "EVENT":
            events += 1
        elif tname == "CHARACTER" and (getattr(m, "cost", 0) or 0) >= 7:
            highcost_char += 1
    density = (counter_cards / n) if n else 0.0
    return [
        counter_total / (50.0 * 2000.0),   # 残カウンター総量（守りの総火力）
        density,                            # カウンター札密度（次に守り札を引く確率の代理）
        blockers / 50.0,                    # ブロッカー残（防御札残量）
        events / 50.0,                      # イベント残（カウンターイベント/トリック資源）
        highcost_char / 50.0,               # 高コストキャラ残（キーカード残の汎用代理）
    ]


# v5: 相手場（公開情報）の脅威集約＋自分の展開余力。個別キャラは field テンソルに入るが「集約」が
# scalars に無く、policy が @33（相手無防備＝攻める）と @64（相手鉄壁＝慎重）を区別できなかった。
# @93（ドン余剰＝展開すべき）も見えなかった（真盤面診断 cpu_v10）。
def _opp_field_aggregate(field):
    """相手場の脅威集約 3 値。空/属性欠落に安全（探索クローン上で呼ばれるため決して例外を投げない）。
    正規化は有界化のためだけ（恒等温スタートは新 W1 行ゼロで保証）。"""
    total_power = 0.0
    high = 0
    blockers = 0
    for c in field:
        p = _power(c)
        total_power += p
        if p >= 7000.0:
            high += 1
        try:
            if c.has_keyword(_BLOCKER_KW):
                blockers += 1
        except Exception:
            pass
    return [
        total_power / (5.0 * 10000.0),   # 相手場の総火力（守り/返しの厚さ）
        high / 5.0,                       # 高パワー(≥7000)脅威数（突破難度）
        blockers / 5.0,                   # ブロッカー数（実ブロック可能数）
    ]


def _playable_chars(me):
    """me.hand のうち今のアクティブドンで召喚できるキャラ数（@93「ドン余剰＝展開すべき」の素地）。
    ドン付与や効果コストの厳密計算はしない代理量（有界化のみ）。"""
    nd = len(getattr(me, "don_active", ()) or ())
    n = 0
    for c in getattr(me, "hand", ()) or ():
        m = getattr(c, "master", None)
        if m is None:
            continue
        tname = getattr(getattr(m, "type", None), "name", None)
        if tname == "CHARACTER" and (getattr(m, "cost", 0) or 0) <= nd:
            n += 1
    return n


def encode(manager, me_name, vocab, version=1):
    """to-move 視点 `me_name` で局面を符号化して dict（numpy 配列）を返す。

    returns:
      scalars  : float32[ S ]            グローバル数値特徴（S=SCALARS_V1/V2・version による）
      field    : float32[ 2*MAX_FIELD, PER_CHAR ]  自場(前半)→相手場(後半)・パディング
      card_idx : int32[ 2 + 2*MAX_FIELD + MAX_HAND ]  [自L, 相手L, 自場*5, 相手場*5, 自手札*10]
    """
    me = manager.p1 if manager.p1.name == me_name else manager.p2
    opp = manager.p2 if manager.p1.name == me_name else manager.p1
    is_my_turn = 1.0 if getattr(manager, "turn_player", me) is me else 0.0

    def lp(pl):
        try:
            return float(pl.leader.get_power(False)) / 10000.0 if pl.leader else 0.0
        except Exception:
            return 0.0

    vals = [
        len(me.life), len(opp.life),
        len(me.don_active), len(me.don_rested),
        len(opp.don_active), len(opp.don_rested),
        len(me.hand), len(opp.hand),                 # 相手手札は「枚数」だけ（中身は出さない）
        len(me.field), len(opp.field),
        float(getattr(manager, "turn_count", 0)),
        is_my_turn,
        lp(me), lp(opp),
    ]
    if version >= 2:
        # v2: リーダーの付与ドン（場キャラの attached_don 特徴と同じ /5 正規化）。
        def ldon(pl):
            if pl.leader is None:
                return 0.0
            return float(getattr(pl.leader, "attached_don", 0) or 0) / 5.0
        vals += [ldon(me), ldon(opp)]
    if version >= 3:
        # v3（docs/reports/effect_semantics_v3_plan_20260708.md §2）:
        # (a) 効果が参照するのに未符号化だった状態変数（棚卸し§3: TRASH_COUNT 29件・DECK_COUNT=OP03ナミの勝利条件変数）
        ev = getattr(manager, "_turn_events", None) or {}
        vals += [len(me.deck) / 50.0, len(opp.deck) / 50.0,
                 len(me.trash) / 20.0, len(opp.trash) / 20.0,
                 float(ev.get(f"CHAR_KOED_{me.name}", 0)) / 3.0,
                 float(ev.get(f"CHAR_KOED_{opp.name}", 0)) / 3.0]
        # (b) スロット別フラグ（[自L, 相L, 自場5, 相場5] の12枠×2種）。scalars に畳む＝新入力キーを
        #     増やさない（既存の append-only 温スタートと全配管がそのまま動く・MLPは位置に依存しない）。
        slots = [me.leader, opp.leader] + \
            (list(me.field)[:MAX_FIELD] + [None] * MAX_FIELD)[:MAX_FIELD] + \
            (list(opp.field)[:MAX_FIELD] + [None] * MAX_FIELD)[:MAX_FIELD]
        # ターン1使用済み（TURN_LIMIT=最頻の効果条件・出典 ability_used_this_turn=JournaledDict）
        vals += [1.0 if (c is not None and any(
            v > 0 for v in getattr(c, "ability_used_this_turn", {}).values())) else 0.0
            for c in slots]
        # 召喚酔い（battle.py の攻撃可否と同源・リーダーは is_newly_played=False）
        vals += [1.0 if (c is not None and getattr(c, "is_newly_played", False)) else 0.0
                 for c in slots]
    if version >= 4:
        # v4（cpu_v5_plan.md §4-3・D3/C5）: 自デッキ残の集約（守り/資源）を末尾追加。相手デッキは
        # 非公開ゆえ自分のみ（公平性契約）。カード個別でない汎用量＝counter/keyword/type の集計。
        vals += _deck_aggregate(getattr(me, "deck", ()) or ())
    if version >= 5:
        # v5（cpu_v10）: 相手場の脅威集約3＋自分の展開余力1。相手場は公開情報（公平性契約に適合）。
        vals += _opp_field_aggregate(getattr(opp, "field", ()) or ())
        vals += [_playable_chars(me) / float(MAX_HAND)]
    scalars = np.array(vals, dtype=np.float32)

    field = np.zeros((2 * MAX_FIELD, PER_CHAR), dtype=np.float32)
    for i, c in enumerate(list(me.field)[:MAX_FIELD]):
        field[i] = _char_feats(c)
    for i, c in enumerate(list(opp.field)[:MAX_FIELD]):
        field[MAX_FIELD + i] = _char_feats(c)

    n_idx = 2 + 2 * MAX_FIELD + MAX_HAND + (2 if version >= 3 else 0)
    idx = np.zeros(n_idx, dtype=np.int32)
    idx[0] = _vidx(vocab, me.leader) if me.leader else PAD
    idx[1] = _vidx(vocab, opp.leader) if opp.leader else PAD
    base = 2
    for i, c in enumerate(list(me.field)[:MAX_FIELD]):
        idx[base + i] = _vidx(vocab, c)
    base += MAX_FIELD
    for i, c in enumerate(list(opp.field)[:MAX_FIELD]):
        idx[base + i] = _vidx(vocab, c)
    base += MAX_FIELD
    for i, c in enumerate(list(me.hand)[:MAX_HAND]):   # 自分の手札のみ（公平）
        idx[base + i] = _vidx(vocab, c)
    if version >= 3:
        # v3: ステージ2枠を**末尾**に追加（ネット側はプール対象を先頭22枠に固定＝恒等温スタート維持。
        #     ステージは EffFeat 射影経路でのみ効く）。ステージ盲目の解消（設計書 改訂1）。
        base = 2 + 2 * MAX_FIELD + MAX_HAND
        idx[base] = _vidx(vocab, me.stage) if getattr(me, "stage", None) else PAD
        idx[base + 1] = _vidx(vocab, opp.stage) if getattr(opp, "stage", None) else PAD

    return {"scalars": scalars, "field": field, "card_idx": idx}


def field_dim():
    """場キャラ特徴 flatten の次元（自場+相手場・版に依らず一定）。温スタートの挿入位置計算に使う。"""
    return 2 * MAX_FIELD * PER_CHAR


def feature_dim(version=1):
    """flatten したときの次元（scalars + field）。card_idx は別経路（Embedding）。"""
    return scalars_dim(version) + field_dim()
