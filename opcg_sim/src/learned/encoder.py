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
学習できなかった。version は**ロード済みネットの入力次元から自動判別**する（cpu_learned 側）＝
現行 Gen2（v1）は挙動不変・v2 ネットへ差し替えた時点で新特徴が有効になる。
"""
import numpy as np

MAX_FIELD = 5          # OPCG の場のキャラ上限
MAX_HAND = 10          # 自分の手札 ID を載せる上限（相手手札は枚数のみ＝公平）
KEYWORDS = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"]
PER_CHAR = 4 + len(KEYWORDS)   # [cost, power, is_rest, attached_don] + keyword flags
PAD = 0                # card_idx の PAD/UNK
SCALARS_V1 = 14        # v1 のグローバル数値特徴数（Gen2 出荷ネット）
SCALARS_V2 = 16        # v2 = v1 + [自リーダー付与ドン, 相手リーダー付与ドン]


def build_vocab(db):
    """card_id → idx（1..N）。0=PAD/UNK。決定的（card_id ソート）。"""
    ids = sorted(cid for cid in db.raw_db.keys() if db.get_card(cid) is not None)
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
    scalars = np.array(vals, dtype=np.float32)

    field = np.zeros((2 * MAX_FIELD, PER_CHAR), dtype=np.float32)
    for i, c in enumerate(list(me.field)[:MAX_FIELD]):
        field[i] = _char_feats(c)
    for i, c in enumerate(list(opp.field)[:MAX_FIELD]):
        field[MAX_FIELD + i] = _char_feats(c)

    idx = np.zeros(2 + 2 * MAX_FIELD + MAX_HAND, dtype=np.int32)
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

    return {"scalars": scalars, "field": field, "card_idx": idx}


def feature_dim(version=1):
    """flatten したときの次元（scalars + field）。card_idx は別経路（Embedding）。"""
    scalars = SCALARS_V2 if version >= 2 else SCALARS_V1
    return scalars + 2 * MAX_FIELD * PER_CHAR
