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
＝慎重）を区別でき、@93（ドン余剰＝展開すべき）も見える（cpu_v10・真盤面診断）。v6=自手札の資源集約（60）。
v7=登場時オプション実測（63）。v8=自場集約の純対称化（66）。v9=**ドンデッキ残（自/相手）＋自デッキ残
キャラ頂点**（最大パワー/最大コスト）で 70 ＝リーダー固有のドン上限・don!!-X の再装填経済・「山に眠る
勝ち筋」を可視化する（v49・h1@2 の掘り無差別 Δ=+0.011 の根因）。version は**ロード済み
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
SCALARS_V6 = 60        # v6 = v5 + **自手札の資源集約5**（カウンター総量/カウンター札数/最大カウンター/ブロッカー数/イベント数）
SCALARS_V7 = 63        # v7 = v6 + **登場時オプション実測3**（発火するPLAY数/そのkeep値/ON_PLAY持ち不発数・v29）
SCALARS_V8 = 66        # v8 = v7 + **自場集約3**（総火力/高パワー数/ブロッカー数＝相手v5と純対称・v32）
SCALARS_V9 = 70        # v9 = v8 + **ドンデッキ残2**（自/相手）＋**自デッキ残キャラ頂点2**（最大パワー/最大コスト・v49）
SCALARS_V10 = 73       # v10 = v9 + **リーサル距離Δ3**（d_me/d_opp/d_opp_def・台本レース実測・v52b）
SCALARS_V11 = 97       # v11 = v10 + **リーダー物理要約24**（自12+相手12・能力木→毎ターン率・leader_feat・v54系）
# v12 = **v9 + リーダー物理要約24**（＝v11 から v10 のリーサル距離Δ3列を外した安価版・2026-08-15）。
# なぜ分岐させるか（実測）: v10 のΔはエンジンで台本を再生する実測特徴で **~25ms/盤面**。
# 探索は1手で数百回符号化するため decide が **0.47s（v9）→13.5s（v11）** と本番予算1秒を
# 28倍超過する。一方リーダー要約は能力木の走査結果をカードIDでキャッシュ＝**実質ゼロ**で、
# gen15 系の改善はこちら側の寄与だった。v10 のΔは v53 で両系とも転移せず効果未実証のまま
# なので、**出荷実績のある v9 系譜に無料の24列だけを継ぐ**のが v12。
# 列の並び: [v9 の 70 列 | リーダー要約 24 列]＝**v9 からは append-only**（G14 からの温スタートは
# 末尾ゼロ追加で恒等）。v11 行からは列 [0:70]+[73:97] の切り出しで教師を作れる（再生成不要）。
# **単調増加の例外**（v12 < v11）: 本 dict は「版→列数」の対応表であって順序の意味は持たない。
# 次元→版の逆引き（`{scalars_dim(v): v}`）は列数が一意なので成立する。温スタートは v9→v12 のみ
# 有効で、v11→v12 は縮小方向として `warm_start_value` が拒否する（設計どおり）。
SCALARS_V12 = 94
# v13（2026-09-04・NRel P0・`docs/n_attention_plan.md` §2.3）: v12 + グローバル追加列 29
#（起動の未使用・加速可能枚数・残りの攻撃可能数・速攻・ライフ圧・自デッキ残の役割別 7・
# 相手の未見プールの役割別 7＋脅威 4・リーダーパワー現在/見込み・次ターンのドン・次ターンに
# 出せる最大の札・守りの単価）。列の定義は `n_rel_feat.EXTRA_COLS` が正本。append-only。
from opcg_sim.src.learned.n_rel_feat import EXTRA_DIM as _NREL_EXTRA_DIM  # noqa: E402
SCALARS_V13 = SCALARS_V12 + _NREL_EXTRA_DIM
_SCALARS_BY_VERSION = {1: SCALARS_V1, 2: SCALARS_V2, 3: SCALARS_V3, 4: SCALARS_V4,
                       5: SCALARS_V5, 6: SCALARS_V6, 7: SCALARS_V7, 8: SCALARS_V8,
                       9: SCALARS_V9, 10: SCALARS_V10, 11: SCALARS_V11,
                       12: SCALARS_V12, 13: SCALARS_V13}

# 手番フラグ（is_my_turn）の scalars 列位置。append-only 契約により全版で不変＝
# コーパスの盤面を「自ターン/相手ターン」で層別するときの唯一の正（v35 層別アンカー）。
IDX_IS_MY_TURN = 11


def scalars_dim(version=1):
    """符号化世代 version のグローバル数値特徴数（append-only ＝ version が上がるほど単調増加）。"""
    if version not in _SCALARS_BY_VERSION:
        raise ValueError(f"未知の符号化世代 version={version}（_SCALARS_BY_VERSION に未登録）")
    return _SCALARS_BY_VERSION[version]


def known_versions():
    """登録済みの符号化世代（昇順）。次元→版の逆引き・拡張ループが版をハードコードしないため。"""
    return sorted(_SCALARS_BY_VERSION)


def battle_resource_cols(version):
    """戦闘リソースヘッドの入力列（scalars 列番号・ユーザ提案 2026-08-14）。

    カウンターを切る行為＝「盤面/ライフ ↔ 手札」のリソース交換（ワンピースカードの基本
    交換レート）を、勝敗相関で汚れた胴体表現でなく**物理量の束**から直接学ぶための列選択。
    束の中身: 中核リソース（ライフ/ドン/手札/盤面の量と質）＋交換レートを条件づける文脈
    （ターン・リーダー・デッキ残・脅威・リーサル距離・リーダー物理要約=デッキ進行の代理）。
    除外: スロット別フラグ（v3）・登場時実測（v7）・展開余力（v5）など攻め側のメイン判断
    専用の列。append-only 契約により列番号は世代を跨いで安定（挿入時は `expanded` が追随）。"""
    cols = [
        0, 1,                    # ライフ枚数（自/相手）
        2, 3, 4, 5,              # ドン（自 active/rested・相手 active/rested）
        6, 7,                    # 手札枚数（自/相手）
        8, 9,                    # 盤面キャラ数（自/相手）
        10, 11,                  # ターン数・手番フラグ
        12, 13,                  # リーダーパワー（自/相手）
    ]
    if version >= 2:
        cols += [14, 15]         # リーダー付与ドン
    if version >= 3:
        cols += [16, 17]         # デッキ残（自/相手・デッキアウト距離）
    if version >= 4:
        cols += [46, 47, 48, 49, 50]     # 自デッキ残の守り札集約（カウンター総量/密度…）
    if version >= 5:
        cols += [51, 52, 53]     # 相手場の脅威集約（総火力/高パワー/ブロッカー）
    if version >= 6:
        cols += [55, 56, 57, 58, 59]     # 自手札の資源集約（カウンター総量/札数/最大値…）
    if version >= 8:
        cols += [63, 64, 65]     # 自場の集約（総火力/高パワー/ブロッカー）
    if version >= 9:
        cols += [66, 67]         # ドンデッキ残（自/相手・ドン経済）
    if version == 10 or version == 11:
        cols += [70, 71, 72]     # リーサル距離Δ（v12 には無い＝コストで外した列）
    if version >= 11:
        # リーダー物理要約（自12+相手12＝デッキ依存の条件づけ）。v11 は 73 起点・
        # v12 は v10 の3列が無いぶん 70 起点（末尾24列であることは共通）。
        base = 73 if version == 11 else 70
        cols += list(range(base, base + 24))
    return cols


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


# v9: 自デッキ残キャラの頂点量。「デッキに何が眠っているか」を**連続量**で載せる（しきい値特徴は
# 設けない＝ユーザ方針 2026-08-03。v4 の cost≥7 カウントは OP15-118（cost6/8000＝紫エネルの勝ち筋）を
# 落とすが、既存特徴の意味は append-only 契約のため変更しない）。掘り（don!!-1 ドロー）の価値は
# 「山に眠る頂点の高さ」に比例する——これが無いと 1コスト掘りキャラの登場時効果が無差別になる
# （h1@2 実測 2026-08-10: 掘る線と無行動の線で value Δ=+0.011）。
def _deck_apex(deck):
    """me.deck（残ライブラリ）のキャラ頂点 2 値（最大パワー/10000・最大コスト/10）。
    空デッキ・属性欠落に安全（探索クローン上で呼ばれるため決して例外を投げない）。"""
    max_power = 0.0
    max_cost = 0.0
    for c in deck:
        m = getattr(c, "master", None)
        if m is None:
            continue
        if getattr(getattr(m, "type", None), "name", None) != "CHARACTER":
            continue
        max_power = max(max_power, float(getattr(m, "power", 0) or 0))
        max_cost = max(max_cost, float(getattr(m, "cost", 0) or 0))
    return [max_power / 10000.0, max_cost / 10.0]


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


# v6: **自手札の資源集約**（2026-07-30・ユーザ指摘「手札の価値をどう正しく判断するか」）。
# v5 までスカラーに載る手札情報は**枚数だけ**で、質（カウンター値・ブロッカー・イベント）は
# card_idx の埋め込み経由でしか見えなかった。その結果 value は「手札が減った＝勝者の相貌」という
# 逆向きの相関を学んでいた（v23 遮蔽帰属: 手札枚数 +0.084・手札ID +0.165 が誤着を押し上げ）。
# 山札残（v4）と同じ集計を**手札**にも与え、「手札は防御資源である」という線形の取っ手を作る。
# 相手手札は対象外（公平性契約＝中身を符号化しない）。カード個別知識は持たない汎用量。
def _hand_aggregate(hand):
    """自手札の資源集約 5 値。空・属性欠落に安全（探索クローン上で呼ばれるため例外を投げない）。

    正規化は有界化のためだけ（恒等温スタートは新 W1 行ゼロで保証）。"""
    counter_total = 0.0
    counter_cards = 0
    max_counter = 0.0
    blockers = 0
    events = 0
    for c in hand:
        m = getattr(c, "master", None)
        if m is None:
            continue
        try:
            cv = float(getattr(c, "current_counter", None) or 0) or float(getattr(m, "counter", 0) or 0)
        except Exception:
            cv = 0.0
        counter_total += cv
        if cv > 0:
            counter_cards += 1
        max_counter = max(max_counter, cv)
        try:
            if _BLOCKER_KW in (m.keywords or ()):
                blockers += 1
        except Exception:
            pass
        if getattr(getattr(m, "type", None), "name", None) == "EVENT":
            events += 1
    return [
        counter_total / (10.0 * 2000.0),   # 手札のカウンター総量（守りの総火力）
        counter_cards / float(MAX_HAND),   # カウンター札枚数（守れる回数）
        max_counter / 2000.0,              # 最大カウンター値（1回で止められる上限）
        blockers / float(MAX_HAND),        # 手札ブロッカー数（次ターンの防御設置）
        events / float(MAX_HAND),          # イベント数（カウンターイベント/トリック資源）
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
    if version >= 6:
        # v6（2026-07-30・防御応答矯正③）: 自手札の資源集約5。手札の「質」をスカラーに載せる
        # （v5 までは枚数のみ＝ネットが「手札減＝良い」を学ぶ素地になっていた）。自分のみ＝公平性契約。
        vals += _hand_aggregate(getattr(me, "hand", ()) or ())
    if version >= 7:
        # v7（2026-08-01・v29）: 登場時オプションの**実測**3値。手札の各 PLAY を make/unmake で
        # 試し「バニラ設置以外の何かが起きるか」をエンジン自身に確かめさせる（判定子は
        # 適用後 pending!=MAIN_ACTION or EFFECT イベント）。「手札のパワー6000を2枚公開」の
        # ような**カード間関係の条件**は埋め込みの線形和では表現できず（v24 representation-
        # bound）、静的特徴でなく実測でしか一般化しない。実測 0.08〜0.26ms/枚（clone の
        # 1/4〜1/14）＝探索の葉評価に載る。非メイン手番は (0,0,0)＝「今行使できる
        # オプション」の意味論。自分の手札のみ＝公平性契約。
        from opcg_sim.src.core.cpu_ai import onplay_option_scan
        n_live, n_dead, keep_live = onplay_option_scan(manager, me_name)
        vals += [n_live / 5.0, min(keep_live / 2000.0, 1.0), n_dead / 5.0]
    if version >= 8:
        # v8（2026-08-02/03・v32）: 自場集約＝相手（v5）と同じ [総火力, 高パワー数, ブロッカー数]
        # の**純対称化のみ**。v5 まで自場はキャラ数の生カウントだけ＝パワー2000も10000も同じ
        # 「1体」で、gen10 反実仮想実測（power_value_probe 2026-08-02）ではバニラ2000追加でも
        # 6000体の 2/3 の加点（「体があれば加点」が支配・自側のパワー傾きは相手側より緩い）。
        # 「低パワー体の盤面価値は低い」は総火力とキャラ数から平均としてネットが導出する
        # （汎用性のため新しいしきい値特徴は設けない＝ユーザ方針 2026-08-03）。
        vals += _opp_field_aggregate(getattr(me, "field", ()) or ())
    if version >= 9:
        # v9（2026-08-10・v49）: ドン経済とデッキ残の頂点量。
        # (a) ドンデッキ残（自/相手・公開情報＝公平性契約に適合）: リーダー固有のドン上限
        #     （紫エネル OP15-058 はドンデッキ6）と「don!!-X で山へ戻したドンがリーダー効果で
        #     再装填される」経済は、この量が無いと**原理的に**見えない。/10 正規化（通常上限）。
        # (b) 自デッキ残キャラの頂点2値（_deck_apex）: 掘りに行く先の価値。
        vals += [len(getattr(me, "don_deck", ()) or ()) / 10.0,
                 len(getattr(opp, "don_deck", ()) or ()) / 10.0]
        vals += _deck_apex(getattr(me, "deck", ()) or ())
    if 10 <= version <= 11:
        # v10（2026-08-12・v52b）: リーサル距離Δ3値＝台本レースのエンジン実測（lethal.py）。
        # 乖離族（見かけと実質が乖離・ライフ差が逆向きに壊れる盤面）で唯一正の説明力を持つ
        # 動力学の要約（v52: 乖離58点 r+0.35／一般60点 r+0.52）。v24/v41/v51 の
        # representation-bound（現行特徴で表現不能）への処方＝静的特徴でなく実測。
        # d_me_def（自攻撃 vs 相手の実防御）は相手手札を読む＝公平性契約違反のため入れない
        # （クリーン3成分の検証は v52b 追補）。/(MAX_TURNS+1) 正規化・実測 ~25ms/盤面。
        from opcg_sim.src.learned.lethal import lethal_scan, MAX_TURNS as _LMT
        d_me_l, d_opp_l, d_opp_def_l = lethal_scan(manager, me_name)
        _cap = float(_LMT + 1)
        vals += [d_me_l / _cap, d_opp_l / _cap, d_opp_def_l / _cap]
    if version >= 11:
        # v11（2026-08-14）/ v12（2026-08-15・リーサルΔ抜き）: リーダー物理要約（能力木→毎ターン率12次元×自/相手）。
        # 接戦帯の帰趨を支配するリーダー再帰効果（ドンランプ・回復・ミル・常在修正）が
        # 現行特徴に0ビットだった欠陥（消去はしご2.6σ）への処方。ID非依存＝新リーダーへ
        # パース即汎化。純粋な木walk＝乱数無消費・エンジン実行なし（符号化は観測）。
        from opcg_sim.src.learned.leader_feat import leader_pair_vectors
        lv_me, lv_opp = leader_pair_vectors(manager, me_name)
        vals += list(lv_me) + list(lv_opp)
    if version >= 13:
        # v13（2026-09-04・NRel P0）: グローバル追加列（n_rel_feat.EXTRA_COLS）。トークン状態 S と
        # 関係 R は scalars に畳まず `n_rel_feat.encode_rel` が別キーで返す（NRel 本体専用）。
        from opcg_sim.src.learned.n_rel_feat import extra_scalars
        vals += list(extra_scalars(manager, me_name))
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
