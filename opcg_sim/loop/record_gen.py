"""自己対戦の棋譜ダンプ（純正 N ループの生データ・旧 `tests/scripts/n_record_gen.py`）。

**エンジンは Rust**（`opcg_sim.loop.driver`）になったが、**記録形式（npz の列）は 1 バイトも
変えていない**＝`n_rel_train.py`／`n_eff_train.py` は無変更で読める（計画 §16.3-A1）。

**設計**（旧版から不変）: AlphaZero の自己対戦棋譜に相当する生データを 1 本で採る——
  ①全判断点（main=木探索／window=窓の根畳み／commit=箱コミット機械実行）の符号化＋勝敗 z
  ②main 窓は**候補と訪問分布**（等価マージ後＝decide の選択と同じ集計）
採掘器（z 密教師・方策ターゲット）は別計器＝ダンプは生のまま保存する。

**規約**: ランダムリーダー×生成デッキ（`decks.leader_pair` は旧 `promotion_gate._leader_pair` と
同じ `seed*7919+13`）。候補生成は `prune_futile=GEN_PRUNE_FUTILE`（生成は枝刈りを外す＝v6 柱⑤）。
探索プロファイルは serve 既定（箱化一式＋箱コミット ON）＝実対局と同じ行動列。

**乱数**（旧版との違い・計画 §8.17 の申告 (2)）: 対局と探索は Rust の決定的な生成器で回す
（旧版は CPython の `random` と `np.random.Generator`）。seed → 対局は決定的だが、**同じ seed で
旧版と同じ局にはならない**（乱数系統が違う）。ルールが一致していることは golden 2 本が担保する。

実行例（分散: 子が --seed-base を分担）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.loop.record_gen \\
    --games 60 --seed-base 200000 --workers 4 --sims 64 --out n_records/part1

シャード npz の列（D=判断点行・K=main 候補の flatten）— 旧版と同一:
  scalars(D,·) field(D,·) card_idx(D,24)   … 符号化（判断点の状態・手番視点）
  z(D)                                     … 勝敗 ±1（手番視点・純正 AZ の素の z）
  who(D) kind(D: 0=main/1=window/2=commit) turn(D) step(D) seed(D)
  sig(D str)                               … 選択手の move_sig（JSON・箱レベル）
  pol_len(D) pol_chosen(D)                 … main の候補数／選択候補の slice 内 index（無ければ 0/-1）
  pol_n(K) pol_q(K) pol_k(K) pol_sig(K str)… 訪問合算 n・行動価値 q・don_k（-1=無し）・候補 move_sig
  pol_cid(K str) pol_tcid(K str)           … 候補の主体/第1対象の**カードID**
  tokens(D,22,·) pol_si(K) pol_ti(K)      … NRel 用（dump v2 で追加・v3 で dtype だけ変えた）
  deck_kinds(D str)                        … **dump v4 で追加**（§20.8.1-4）。手番側デッキの除去の
                                             型（`deck_roles` の JSON・`--decks synth_roles` 以外は
                                             `{}`）。層別の読み（除去率別・型別）に使う
  forced(D int8)                           … **dump v4 で追加**（§20.8.5／§20.9-D）。ε で手を
                                             差し替えた行の印（0=木のまま／1=打つ側／2=保留側／
                                             3=対象を一様に引いた）。
                                             **教師ではない**（`n_rel_train` は読まない・層別用）
  pol_p(K float16) pol_v0(D float16)       … **dump v4 で追加**（§20.6.1・改良方策 π' の材料）。
                                             `pol_p`＝候補の根の事前分布（`stats.P` を group の
                                             `idxs` で合算）・`pol_v0`＝根の価値の推定（訪問で
                                             重み付けた Q の平均・main 行以外は 0）

**dump v3**（2026-09-07・計画 §18.5）: 常駐の 9 割を占める 3 本の保存 dtype を半分にする——
`tokens`／`scalars` を **float16**・`card_idx` を **int16**（`pol_si`/`pol_ti` は v2 から int16）。
値は float32／int64 を **cast しただけ**で、**符号化（`enc_version`=13）も列の形も変えていない**。
学習側は `learned/train/dump_io.py` が波ごとに memmap の pack を作って読む（v2 の npz＝
float32／int64 も同じ関数で読める＝過去の波はそのまま使える）。v1／v2 を書く経路は無い。

**dump v4**（既定・2026-09-10・計画 §20.8）: v3 の列は 1 バイトも変えず、**補助教師の 3 列**
（`aux`／`aux_tok`／`aux_mask`・WP `rs-aux-heads`）と **`deck_kinds(D str)`**（`--decks synth_roles` で
差し込んだ除去の型・型が無い経路では `{}`・WP `rs-removal-decks`）と **`forced(D int8)`**
（両方向 ε 探索で手を差し替えた行の印・`--eps-play`／`--eps-hold` が 0 の既定では全 0・
WP `rs-eps-explore`）を足すだけ。`dump_io.py` は**無い列を `None` で返す**＝v3 の波はそのまま
読める。補助教師の中身は下の `aux_from_ledger` が正本。
対局メタ（seed・両席のリーダー・両席の型）は part ごとの sidecar `meta_games.json` にも書く（層別の集計用）。

**符号化 v14**（2026-09-11・計画 §20.9・`meta_n_record.json` の `enc_version`＝14）: 列の**名前**は
1 つも変わらないが、`tokens` が `[22,20]→[22,22]`・`scalars` が `123→127` になる（append-only）。
`dump_io` が v13 の波を 0 埋めで v14 の形に揃えるので、**波を混ぜて訓練できる**。併せて
`pol_si`／`pol_ti`／`pol_cid`／`pol_tcid` は**対象選択の候補**に対して
`selected_uuids[0]`（対象）と効果の発生源（主体）を書く（§20.9 の A・正本は `n_rel.cand_ids`）。
`forced` に **3＝対象を一様に引いた行**が加わる（§20.9 の D・`--eps-target`）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import multiprocessing as mp                                           # noqa: E402
import random                                                          # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

import numpy as np                                                     # noqa: E402

from opcg_sim.learned import n_rel as NL                               # noqa: E402
from opcg_sim.learned import n_rel_feat as NF                         # noqa: E402
from opcg_sim.loop import deck_roles as DR_ROLES                       # noqa: E402
from opcg_sim.loop import decks as D                                   # noqa: E402
from opcg_sim.loop import driver as DR                                 # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402

MAX_STEPS = 400
MAX_CI = 24            # card_idx の PAD 長（G 系符号化の既定枠・列の形は据え置き）
ENC_VERSION = 12       # dump v1 の符号化世代（過去の波の meta にだけ残る）
ENC_VERSION_V2 = 13    # dump v2／v3／波 29 まで（v12 ＋ グローバル追加 29・append-only）
#: 符号化 v14（2026-09-11・§20.9）。**列の形が変わる唯一の欄**（tokens 22×22・scalars 127）。
#: `dump_io` は v13 の波を 0 埋めで v14 の形に揃えて読む＝波を混ぜて訓練できる。
ENC_VERSION_V14 = NL.NR_ENC_VERSION    # 14
DUMP_VERSION = 4       # v3 ＋ 補助教師の列（aux／aux_tok／aux_mask）＋ deck_kinds 列（§20.8）
#: `--decks` の既定（歴代の波は全て synth＝規約を変えない）。
DEFAULT_DECKS = "synth"

# v3 で dtype を落とす列（**float32／int64 を cast するだけ**＝符号化も列の形も変えない）。
# 常駐の 9 割はこの 3 本（tokens 1,760B／scalars 492B／card_idx 176B per 行）。float16 の丸めが
# forward に与える差は 1 バッチ最大 1.07e-4（`docs/reports/2026-09-07_train_profile.md` §3）。
DT_V3 = {"tokens": np.float16, "scalars": np.float16, "card_idx": np.int16}

# Rust の `encode` は平らな列で返す。**旧版と同じ形**へ戻してから積む（`n_rel_train` は
# `tokens.shape[1:]` をそのまま使う＝形が変われば読めない）。
FIELD_SHAPE = (10, 8)                   # `encoder.MAX_FIELD*2` × `PER_CHAR`
TOKENS_SHAPE = (NF.N_TOK, NF.S_DIM)     # `n_rel_feat` の 22 枠 × S_DIM（v14 は 22・v13 は 20）

_KIND = {"main": 0, "window": 1, "commit": 2}
_G = {}

# --- 補助教師（dump v4・計画 §20.8.2）--------------------------------------
#: `aux` の列数と名前（**「起きたこと」だけ**＝符号も重みも人が入れない）。
AUX_COLS = ("opp_turn_my_life_lost", "opp_turn_attacks", "opp_turn_effects",
            "my_turn_opp_life_lost", "my_turn_attacks", "my_turn_effects",
            "next_my_board_power", "next_opp_board_power",
            "next_my_board_n", "next_opp_board_n")
AUX_DIM = len(AUX_COLS)                    # 10
#: `aux_tok` の枠（相手 L ＋ 相手場 5＝`n_rel.OPP_SLOTS` の並び）と列。
AUX_TOK_SLOTS = 6
AUX_TOK_COLS = ("attacked", "life_pushed", "ability")
AUX_TOK_DIM = len(AUX_TOK_COLS)            # 3
#: パワーの目盛り（`aux` の 6/7 列はここで割る＝トークン状態 S と同じ流儀）。
AUX_POWER_SCALE = 10000.0
#: 攻撃宣言の action_type（`rules::actions` の `"ATTACK" | "ATTACK_CONFIRM"`）。
ATTACK_ATS = ("ATTACK", "ATTACK_CONFIRM")


def snapshot(game):
    """台帳の 1 枚（`board_json` から補助教師に要る欄だけ抜く）。1 手あたり約 0.2ms。

    盤面は動かさない。`power`／`n` は**場のキャラだけ**（リーダー・ステージは含めない）。
    """
    b = json.loads(game.board_json())
    ti = b.get("turn_info") or {}
    out = {"turn": int(ti.get("turn_count") or 0), "tp": ti.get("active_player_id"),
           "life": {}, "power": {}, "n": {}, "field": {}, "leader": {}, "names": {}}
    for nm, p in (b.get("players") or {}).items():
        field = ((p.get("zones") or {}).get("field")) or []
        leader = p.get("leader") or {}
        out["life"][nm] = int(p.get("life_count") or 0)
        out["power"][nm] = float(sum(int(c.get("power") or 0) for c in field))
        out["n"][nm] = float(len(field))
        out["field"][nm] = [c.get("uuid") for c in field]
        out["leader"][nm] = leader.get("uuid")
        names = {c.get("uuid"): c.get("name") for c in field}
        if leader.get("uuid"):
            names[leader["uuid"]] = leader.get("name")
        out["names"][nm] = names
    return out


def step_record(name, move, events):
    """台帳の 1 手（誰が・何を・どの uuid に／その適用が積んだ効果イベント）。

    `eff_src`＝効果イベントの**発生源カードの uuid**（Rust `push_effect_events` の
    `source_uuid`・WP `rs-replace-rest-fix`／計画 §20.8.4-2）。同名のカードが並んでも
    「どの枠が能力を発動したか」を取り違えない（旧実装は `card_name` 一致だった）。
    """
    payload = (move.get("payload") or {}) if move else {}
    eff, eff_src = {}, []
    for ev in events or ():
        if (ev.get("type") if isinstance(ev, dict) else None) != "EFFECT":
            continue
        pl = ev.get("player")
        eff[pl] = eff.get(pl, 0) + 1
        eff_src.append((pl, ev.get("source_uuid")))
    return {"actor": name,
            "at": (move.get("action_type") or payload.get("action_type")) if move else None,
            "uuid": (move.get("card_uuid") or payload.get("uuid")) if move else None,
            "eff": eff, "eff_src": eff_src}


def turn_segments(snaps, steps):
    """手を「そのターン（turn_count, 手番プレイヤー）」で区切る＝`[{tp, i0, i1, closed}]`。

    手 k のターンは**適用する前の盤面** `snaps[k]` で決める（`TURN_END` は自分のターンの手）。
    `closed`＝そのターンが終わったことが台帳から判る（後ろに別の区間がある）。最後の区間は
    「対局がその途中で終わった」＝`closed=False`。
    """
    segs = []
    for k in range(len(steps)):
        key = (snaps[k]["turn"], snaps[k]["tp"])
        if segs and segs[-1]["key"] == key:
            segs[-1]["i1"] = k
        else:
            segs.append({"key": key, "tp": snaps[k]["tp"], "i0": k, "i1": k})
    for j, s in enumerate(segs):
        s["closed"] = j + 1 < len(segs)
    return segs


def _count(steps, i0, i1, actor, ats):
    return sum(1 for k in range(i0, i1 + 1)
               if steps[k]["actor"] == actor and steps[k]["at"] in ats)


def _effects(steps, i0, i1, actor):
    return sum(steps[k]["eff"].get(actor, 0) for k in range(i0, i1 + 1))


def aux_from_ledger(snaps, steps, row_step, row_who):
    """台帳 →（`aux [D,10]`, `aux_tok [D,6,3]`, `aux_mask [D]`）。計画 §20.8.2 の正本。

    `snaps[k]`＝手 k を適用する**前**の盤面（`len(snaps) == len(steps)+1`）。行 d は
    `row_step[d]` 番目の手を決めた判断点で、視点は `row_who[d]`。

    区間の取り方（行の視点＝「自分」）:
      A＝**この行の後、相手の手番が 1 回終わるまで**＝行の手番以降で最初に来る「相手のターン」
         区間（行が相手ターンの中なら、その残り）。
      B＝A の後に来る最初の「自分のターン」区間。
    `aux_mask=1` は A も B も**終わったことが台帳から判る**ときだけ（対局が区間の途中で
    終わった行は 0＝損失に入れない）。

    値はすべて「起きたこと」＝符号なし（ライフは枚数・パワーは /10000）。**良し悪しは入れない。**
    """
    D = len(row_step)
    aux = np.zeros((D, AUX_DIM), np.float32)
    tok = np.zeros((D, AUX_TOK_SLOTS, AUX_TOK_DIM), np.float32)
    mask = np.zeros(D, np.int8)
    if not steps:
        return aux, tok, mask
    segs = turn_segments(snaps, steps)
    for d in range(D):
        t = int(row_step[d])
        me = row_who[d]
        opp = "p2" if me == "p1" else "p1"
        a = next((s for s in segs if s["tp"] == opp and s["i1"] >= t), None)
        if a is None:
            continue
        b = next((s for s in segs if s["tp"] == me and s["i0"] > a["i1"]), None)
        if b is None:
            continue
        a0, a1 = max(t, a["i0"]), a["i1"]
        b0, b1 = b["i0"], b["i1"]
        aux[d, 0] = max(0.0, snaps[a0]["life"][me] - snaps[a1 + 1]["life"][me])
        aux[d, 1] = _count(steps, a0, a1, opp, ATTACK_ATS)
        aux[d, 2] = _effects(steps, a0, a1, opp)
        aux[d, 3] = max(0.0, snaps[b0]["life"][opp] - snaps[b1 + 1]["life"][opp])
        aux[d, 4] = _count(steps, b0, b1, me, ATTACK_ATS)
        aux[d, 5] = _effects(steps, b0, b1, me)
        start = snaps[b0]                                  # 次の自分のターン開始時の盤面
        aux[d, 6] = start["power"][me] / AUX_POWER_SCALE
        aux[d, 7] = start["power"][opp] / AUX_POWER_SCALE
        aux[d, 8] = start["n"][me]
        aux[d, 9] = start["n"][opp]
        _fill_tok(tok[d], snaps, steps, a0, a1, me, opp, snaps[t])
        mask[d] = 1 if (a["closed"] and b["closed"]) else 0
    return aux, tok, mask


def _fill_tok(out, snaps, steps, a0, a1, me, opp, cur):
    """相手 6 枠（**行の時点**の相手 L ＋ 相手場 5）× [攻撃したか, 通したライフ枚数, 能力発動]。

    「通したライフ枚数」は攻撃宣言から**次の攻撃宣言（無ければ区間の終わり）まで**に自分が
    失ったライフ＝その攻撃に紐づく枚数（間に挟まった効果のぶんも同じ攻撃に寄る）。
    「能力を発動したか」は相手の `ACTIVATE_MAIN` の uuid 一致か、相手の EFFECT イベントの
    **発生源 uuid**（`source_uuid`）一致で立てる（§20.8.4-2 でイベントに欄を足した＝
    同名のカードが並んでも正しい枠だけが立つ。旧実装の `card_name` 一致は両方の枠を立てた）。
    どちらも「起きたこと」の記録で、良し悪しは含まない。
    """
    slots = [cur["leader"][opp]] + (list(cur["field"][opp]) + [None] * 5)[:5]
    pos = {u: j for j, u in enumerate(slots) if u}
    if not pos:
        return
    atk = [k for k in range(a0, a1 + 1)
           if steps[k]["actor"] == opp and steps[k]["at"] in ATTACK_ATS]
    for n_, k in enumerate(atk):
        j = pos.get(steps[k]["uuid"])
        if j is None:
            continue
        end = atk[n_ + 1] if n_ + 1 < len(atk) else a1 + 1
        out[j, 0] = 1.0
        out[j, 1] += max(0.0, snaps[k]["life"][me] - snaps[end]["life"][me])
    fired = set()
    for k in range(a0, a1 + 1):
        st = steps[k]
        if st["actor"] == opp and st["at"] == "ACTIVATE_MAIN":
            j = pos.get(st["uuid"])
            if j is not None:
                out[j, 2] = 1.0
        fired.update(su for pl, su in st["eff_src"] if pl == opp and su)
    for u in fired:
        j = pos.get(u)
        if j is not None:
            out[j, 2] = 1.0


# --- 両方向の ε 探索（dump v4・計画 §20.8.5）--------------------------------
#
# **既定 0＝1 bit も変わらない**。木も π も教師も変えず、`decide` が返した**後**に
# 「実対局へ出す手」だけを差し替える（入口は `driver.run_game(swap=…)`）。狙いは教材の対照
# ——「除去を打った／打たなかった」が棋譜に両方載るようにすること（§20.7.11 の穴）。
#
#   打つ側（--eps-play P）… 木が除去でない手を選んだのに候補に除去があれば、確率 P で
#                           その中の **N 最大**の 1 つへ差し替える → `forced=1`
#   保留側（--eps-hold H）… 木が除去を選んだら、確率 H で「除去でない候補の N 最大」へ
#                           差し替える（TURN_END も可）→ `forced=2`
#
# 差し替えた手は**原始手**で出す（箱の残り手順は捨てる＝その後の対話は既定解決）。
# 学習側（`n_rel_train`）は `forced` 列を**読まない**（教師は変えない・層別に使うだけ）。

#: 除去とみなす form（`deck_roles.FORMS`＝KO／bounce／deck／trash／lock／reduce）。
EPS_REMOVAL_FORMS = frozenset(DR_ROLES.FORMS)
#: 「除去を打つ手」とみなす action_type（箱は先頭原始手＝素の手で判定する）。
EPS_PLAY_ATS = ("PLAY", "ACTIVATE_MAIN")

# --- ε の対象ランダム化（符号化 v14・計画 §20.9 の D）------------------------
#
# 除去を**打つ**対照（`--eps-play`）は入ったが、「**誰に**撃つか」の対照は棋譜に無い
# （木は常に自分の読みどおりの対象を選ぶ）。対象の良し悪しを学べるようにするには、
# 除去を打った直後の**対象選択だけ**を一様に引いた行が要る。
#
#   forced=1 の直後 … その効果の最初の対象選択を**必ず**一様に引く（打つ側の ε は既に
#                     「木が選ばなかった手」を打っているので、対象まで木に戻す意味が無い）
#   --eps-target P  … 木が選んだ除去系の手の直後の対象選択も確率 P で一様に引く
#
# どちらも `forced=3` を立て、`forced_sig` に差し替えた手を残す。**π も sig も観測のまま**
# （教師は変えない）＝`n_rel_train` は forced 列を読まない。
EPS_TARGET_AT = "RESOLVE_EFFECT_SELECTION"


def target_choices(legal):
    """対象選択の候補（`selected_uuids` を 1 枚以上持つ `RESOLVE_EFFECT_SELECTION`）。

    「選ばない」（`selected_uuids` が空）は対象の対照にならないので外す（pure）。
    """
    return [i for i, mv in enumerate(legal or ())
            if (mv or {}).get("action_type") == EPS_TARGET_AT
            and ((mv.get("payload") or {}).get("selected_uuids") or ())]


def is_resolve_point(legal):
    """その判断点は**効果の解決の途中**か（合法手が全て `RESOLVE_EFFECT_SELECTION`）。

    除去を打った直後に来るのは対象選択とは限らない（任意確認・並び替え・複数段の効果）。
    「効果の解決が続いているあいだ」は腕を保ち、素の手（MAIN_ACTION・戦闘）が出た時点で
    降ろす＝**同じ効果の最初の対象選択**を捕まえる（pure）。
    """
    return bool(legal) and all((mv or {}).get("action_type") == EPS_TARGET_AT for mv in legal)


def eps_rng(game_seed, turn, seat, step):
    """1 判断点ぶんの乱数（seed・ターン・席・手数から決まる＝pure・並列度に依存しない）。

    `engine.search_seed` と同じ SplitMix64 の流儀。探索の乱数系統とは**別の salt** を混ぜる
    （ε の抽選が世界サンプルの並びを動かさないように）。
    """
    x = (int(game_seed) * 0x9E3779B97F4A7C15
         + int(turn) * 0xBF58476D1CE4E5B9
         + (1 if seat == "p2" else 0) * 0x94D049BB133111EB
         + int(step) * 0xD1342543DE82EF95
         + 0x45D9F3B3C5A4E7D1) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return random.Random(x ^ (x >> 31))


def box_base(mv):
    """箱 → 素の手（`SETUP_BOX` は `payload.base`）。他はそのまま返す（pure）。"""
    if (mv or {}).get("action_type") != "SETUP_BOX":
        return mv or {}
    return ((mv.get("payload") or {}).get("base")) or mv


def first_primitive(mv):
    """箱 → 先頭原始手（Rust `search::decide::don_box_first_primitive` の写し・pure）。

    `SETUP_BOX` は素の PLAY／ACTIVATE_MAIN、`DON_BOX` は付与なら ATTACH_DON・
    付与 0 枚で対象があれば ATTACK。箱でない手はそのまま。
    """
    mv = mv or {}
    at = mv.get("action_type")
    if at == "SETUP_BOX":
        return box_base(mv)
    if at != "DON_BOX":
        return mv
    p = mv.get("payload") or {}
    k = int(p.get("don_k") or 0)
    tids = list(p.get("target_ids") or ())
    if k <= 0 and tids:
        return {"kind": "game", "action_type": "ATTACK",
                "payload": {"uuid": p.get("uuid"), "target_ids": tids}}
    return {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": p.get("uuid")}}


def is_removal_move(mv, cids, db):
    """その手は「除去系」か＝素の手が PLAY／ACTIVATE_MAIN で、そのカードが除去の型を持つ。

    箱の中身も見る（`SETUP_BOX` の `payload.base`）。型は `deck_roles.classify`（分類の正本）。
    """
    base = box_base(mv)
    if (base or {}).get("action_type") not in EPS_PLAY_ATS:
        return False
    uuid = ((base.get("payload") or {}).get("uuid"))
    cid = cids.get(uuid) if uuid else None
    if not cid:
        return False
    master = db.get_card(cid)
    if master is None:
        return False
    return any(k.split(":", 1)[0] in EPS_REMOVAL_FORMS for k in DR_ROLES.classify(master))


def eps_swap(out, move, cids, db, eps_play, eps_hold, rng):
    """`(差し替えた手, forced)` を返す（差し替えなしなら `(move, 0)`）。pure に近い純関数。

    `out` は書き換えない（箱の残り手順を捨てるのは呼び出し側＝`_Recorder.swap`）。
    """
    if out.get("kind") != "main":
        return move, 0
    groups = out.get("groups") or []
    legal = (out.get("stats") or {}).get("legal") or []
    if len(groups) < 2:
        return move, 0                       # 差し替え先が無い
    reps = [legal[g["rep"]] for g in groups]
    removal = [is_removal_move(r, cids, db) for r in reps]
    chosen_removal = is_removal_move(move, cids, db)
    if not chosen_removal and eps_play > 0.0:
        pool = [i for i, f in enumerate(removal) if f]
        if pool and rng.random() < eps_play:
            best = max(pool, key=lambda i: groups[i]["n"])
            return first_primitive(reps[best]), 1
    elif chosen_removal and eps_hold > 0.0:
        pool = [i for i, f in enumerate(removal) if not f]
        if pool and rng.random() < eps_hold:
            best = max(pool, key=lambda i: groups[i]["n"])
            return first_primitive(reps[best]), 2
    return move, 0


def move_sig(mv):
    """Python `learned/plan.move_sig` と同じ鍵（Rust `search::decide::move_sig` の写し）。

    JSON にしたときの形（`[action_type, uuid, target_ids, selected_uuids, accepted]`）が
    旧版の `json.dumps(tuple)` と一致する＝`pol_sig`／`sig` 列の中身が変わらない。
    """
    p = mv.get("payload") or {}
    return [mv.get("action_type") or p.get("action_type"), p.get("uuid"),
            list(p.get("target_ids") or ()), list(p.get("selected_uuids") or ()),
            p.get("accepted")]


def _don_k(mv):
    return (mv.get("payload") or {}).get("don_k")


def _init_worker(sims, net, dirichlet_eps, temp_turns, decks=DEFAULT_DECKS, aux=True,
                 eps_play=0.0, eps_hold=0.0, search=None, eps_target=0.0):
    E.engine()
    # `search`＝探索の設定の上書き（§20.8.6・波 29〜: worlds／select_rule／q_min_frac／
    # root_prior_temp）。None／空＝serve 既定のまま＝歴代の波と同じ。
    _G["spec"] = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps,
                            temp_turns=temp_turns, eps_play=eps_play, eps_hold=eps_hold,
                            eps_target=eps_target,
                            prune_futile=E.GEN_PRUNE_FUTILE, **(search or {}))
    _G["db"] = D.load_db()
    _G["decks"] = decks
    _G["aux"] = bool(aux)


class _Recorder:
    """1 局ぶんの観測（`driver.run_game(observer=…)`）。盤面は動かさない。

    ε 探索（§20.8.5）を使うときは `swap` も `run_game` に渡す＝**観測の後**に呼ばれ、
    実対局へ出す手だけを差し替えて `forced` 列を立てる（π も `sig` も観測のまま）。
    """

    def __init__(self, seed=0, db=None, eps_play=0.0, eps_hold=0.0, eps_target=0.0):
        self.seed, self.db = seed, db
        self.eps_play, self.eps_hold = float(eps_play), float(eps_hold)
        self.eps_target = float(eps_target)
        #: ε の分母（「除去が合法だった main 行」の数）＝計測用。
        self.removal_legal_rows = 0
        #: 「次に来る対象選択を一様に引く」旗（§20.9 の D）。除去を打った直後だけ立つ。
        self._target_arm = False
        #: 直近の観測で引いた uuid → card_id（`swap` が使い回す＝往復を増やさない）。
        self._cids = None
        self.rows = {k: [] for k in ("scalars", "field", "card_idx", "who", "kind", "turn",
                                     "step", "sig", "pol_len", "pol_chosen", "tokens",
                                     "forced", "forced_sig", "pol_v0")}
        self.pol = {k: [] for k in ("n", "q", "k", "sig", "cid", "tcid", "si", "ti", "p")}
        # 補助教師の台帳（dump v4・§20.8.2）: `snaps[k]`＝手 k の直前の盤面・`steps[k]`＝手 k。
        self.snaps = []
        self.steps = []

    def post(self, game, name, turn, step, move, events):
        """`driver.run_game(post=…)`＝**手を適用した直後**に台帳を 1 枚積む（`step=-1` は初期盤面）。"""
        if step >= 0:
            self.steps.append(step_record(name, move, events))
        self.snaps.append(snapshot(game))

    def aux(self):
        """台帳 → 行ごとの補助教師（`aux`／`aux_tok`／`aux_mask`）。"""
        return aux_from_ledger(self.snaps, self.steps, self.rows["step"], self.rows["who"])

    def swap(self, game, name, turn, step, out, move):
        """`driver.run_game(swap=…)`＝**観測の後**に実対局へ出す手だけを差し替える（§20.8.5）。

        π（`pol_n`）も `sig` も `pol_chosen` も観測したまま＝**教師は変えない**。差し替えた行に
        `forced`（1=打つ側・2=保留側）を立てるだけ。箱の残り手順（`out["commit"]`）は捨てる
        ＝差し替え後の対話は既定解決。
        """
        if self.eps_play <= 0.0 and self.eps_hold <= 0.0 and self.eps_target <= 0.0:
            return move                        # 既定＝1 bit も変わらない（呼ぶだけ）
        legal = (out.get("stats") or {}).get("legal") or []
        # ① 対象のランダム化（§20.9 の D）: 除去を打った直後の**同じ効果の最初の**対象選択を
        #    一様に引く。対象選択は窓／箱コミットでも来る＝`kind` は問わない。効果の解決が
        #    続いているあいだ（合法手が全て RESOLVE）は腕を保ち、素の手が出たら降ろす。
        if self._target_arm:
            picks = target_choices(legal)
            if len(picks) >= 2:
                self._target_arm = False
                rng = eps_rng(self.seed, turn, name, step)
                new = legal[picks[rng.randrange(len(picks))]]
                self._mark_forced(3, new)
                out["commit"] = []              # 箱の残り手順（＝木が決めた対象）は捨てる
                return new
            if not is_resolve_point(legal):
                self._target_arm = False        # 効果の解決が終わった＝対象選択は来なかった
        cids = self._cids
        if out.get("kind") != "main" or cids is None or self.db is None:
            return move
        if any(is_removal_move(legal[g["rep"]], cids, self.db)
               for g in (out.get("groups") or [])):
            self.removal_legal_rows += 1
        rng = eps_rng(self.seed, turn, name, step)
        new, forced = eps_swap(out, move, cids, self.db, self.eps_play, self.eps_hold, rng)
        if forced:
            self._mark_forced(forced, new)
            out["commit"] = []                 # 木の選んだ箱の残り手順は捨てる
            # ② 強制した除去（forced=1）の直後の対象選択は**必ず**一様に引く。
            self._target_arm = (forced == 1)
            return new
        # ③ 木が自分で選んだ除去系の手は、確率 `eps_target` で対象だけ一様に引く。
        if self.eps_target > 0.0 and is_removal_move(move, cids, self.db) \
                and rng.random() < self.eps_target:
            self._target_arm = True
            out["commit"] = []                 # 対象を引き直すので箱の残り手順は使わない
        return move

    def _mark_forced(self, forced, new):
        """直近の行に `forced`（1=打つ側／2=保留側／3=対象）と差し替えた手の sig を立てる。"""
        if not self.rows["forced"]:
            return
        self.rows["forced"][-1] = forced
        # 差し替えた手そのもの（§20.8.9 の層別用・π／sig は木の選択のまま）。
        self.rows["forced_sig"][-1] = json.dumps(move_sig(new), ensure_ascii=False)

    def __call__(self, game, name, turn, step, out, move):
        self._cids = None
        # 符号化は decide **前**の状態＝この判断が見た盤面（Rust の decide は盤面を変えない）。
        enc = json.loads(game.encode(name))
        self.rows["scalars"].append(np.asarray(enc["scalars"], np.float32))
        self.rows["field"].append(
            np.asarray(enc["field"], np.float32).reshape(FIELD_SHAPE))
        ci = np.zeros(MAX_CI, np.int64)
        src = np.asarray(enc["card_idx"], np.int64)[:MAX_CI]
        ci[:len(src)] = src
        self.rows["card_idx"].append(ci)
        self.rows["who"].append(name)
        self.rows["forced"].append(0)          # ε で差し替えたら `swap` が上書きする
        self.rows["forced_sig"].append("")     # 同上（差し替えた手の move_sig・JSON）
        self.rows["pol_v0"].append(0.0)        # main で候補があれば下で上書きする（§20.6.1）
        self.rows["kind"].append(_KIND.get(out.get("kind"), 0))
        self.rows["turn"].append(int(turn))
        self.rows["step"].append(step)
        # 決定の同一性は**箱レベル**（原始手化と残り掘りの前）＝Rust が `sig`／`k` で返す。
        # `out["move"]` は実対局へ出す手なので、候補（groups）と突き合わせる鍵にはならない。
        sig = out.get("sig") if out.get("sig") is not None else move_sig(move)
        self.rows["sig"].append(json.dumps(sig, ensure_ascii=False))

        legal = (out.get("stats") or {}).get("legal") or []
        groups = out.get("groups") or []
        # 窓・コミットは訪問を配らない＝候補を持たない（旧版と同じく空）。
        if out.get("kind") != "main":
            groups = []
        self.rows["pol_len"].append(len(groups))
        self.rows["tokens"].append(
            np.asarray(enc["tokens"], np.float32).reshape(TOKENS_SHAPE))
        if not groups:
            self.rows["pol_chosen"].append(-1)
            return
        idx = json.loads(game.dump_index_json(name))
        cids, slots = idx["cids"], idx["slots"]
        self._cids = cids                      # `swap`（ε 探索）が使い回す
        # 効果の発生源（対象選択の候補行の主体・符号化 v14 の §20.9-A）。対象選択の候補が
        # 1 本も無ければ引かない（pending の JSON 化は 1 手あたりの往復に乗るので）。
        src_uuid = None
        if any((legal[g["rep"]] or {}).get("action_type") == NL.SELECT_AT for g in groups):
            src_uuid = (json.loads(game.pending_json()) or {}).get("source_card_uuid")
        # 改良方策 π' の材料（§20.6.1）: 根の事前分布を group の `idxs` で合算する。
        # `stats.P` は**根の平坦化（`root_prior_temp`）と Dirichlet 混合の後**の値＝
        # 生の P が要るときは `root_prior_temp=1`／`dirichlet_eps=0` で採るか、訓練側で
        # 生成役ネットを forward し直す（`n_rel_train --pi-teacher q_improved` の退避路）。
        pr = (out.get("stats") or {}).get("P")
        gp = [sum(float(pr[i]) for i in g["idxs"]) for g in groups] if pr \
            else [1.0 / len(groups)] * len(groups)
        chosen, k_sel = -1, out.get("k")
        for gi, g in enumerate(groups):
            rep = legal[g["rep"]]
            gsig = move_sig(rep)
            gk = _don_k(rep)
            # 配分箱は k 違いが同 sig（move_sig は don_k 非含有）＝ (sig, k) で照合
            if chosen < 0 and gsig == sig and gk == k_sel:
                chosen = gi
            self.pol["n"].append(float(g["n"]))
            self.pol["q"].append(float(g["q"]))
            self.pol["p"].append(gp[gi])
            self.pol["k"].append(-1 if gk is None else int(gk))
            self.pol["sig"].append(json.dumps(gsig, ensure_ascii=False))
            # 候補行の主体／対象（v14 の規則が正本＝`n_rel.cand_ids`）。素の手は今までどおり
            # uuid／target_ids[0]、対象選択だけ selected_uuids[0] と効果の発生源になる。
            # **dump は常に v14 の規則で書く**（生成役が v13 のネットでも、教材は v14 のネットが
            # 読むもの＝π は木のまま・行の見え方だけが新しい）。
            su, tu = NL.cand_ids(rep, src_uuid)
            self.pol["cid"].append(cids.get(su) or "")
            self.pol["tcid"].append((cids.get(tu) or "") if tu else "")
            self.pol["si"].append(slots.get(su, -1) if su else -1)
            self.pol["ti"].append(slots.get(tu, -1) if tu else -1)
        self.rows["pol_chosen"].append(chosen)
        # 根の価値の推定 v_mix＝訪問で重み付けた Q の平均（訪問 0 の根は 0・§20.6.1）。
        ntot = sum(float(g["n"]) for g in groups)
        self.rows["pol_v0"][-1] = (
            sum(float(g["n"]) * float(g["q"]) for g in groups) / ntot) if ntot > 0 else 0.0


def play_one(seed):
    """1 局を打って行を返す（勝敗が付いた局だけ・純正 z）。失敗は None。"""
    spec, db = _G["spec"], _G["db"]
    rec = _Recorder(seed=seed, db=db,
                    eps_play=getattr(spec, "eps_play", 0.0),
                    eps_hold=getattr(spec, "eps_hold", 0.0),
                    eps_target=getattr(spec, "eps_target", 0.0))
    aux_on = _G.get("aux", True)
    try:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2, kinds = D.build_pair(db, la, lb, seed, _G.get("decks", DEFAULT_DECKS),
                                     with_kinds=True)
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=MAX_STEPS,
                          observer=rec, post=rec.post if aux_on else None, swap=rec.swap)
    except Exception:                                          # noqa: BLE001  1 局の失敗で止めない
        return None
    winner, turn, steps, acts = res["winner"], res["turns"], res["steps"], res["acts"]
    if (winner is None or turn < 4 or steps >= MAX_STEPS or min(acts.values()) == 0):
        return None                                            # 純正 z＝勝敗が付いた対局のみ採用
    if not rec.rows["who"]:
        return None
    z = {"p1": 1.0 if winner == "p1" else -1.0}
    z["p2"] = -z["p1"]
    rows, pol = rec.rows, rec.pol
    # dump v4: 手番側デッキの除去の型（JSON・型を持たない経路は "{}"）
    kj = {"p1": json.dumps(kinds[0] or {}, ensure_ascii=False, separators=(",", ":")),
          "p2": json.dumps(kinds[1] or {}, ensure_ascii=False, separators=(",", ":"))}
    aux, aux_tok, aux_mask = rec.aux()          # dump v4 の追加列（§20.8.2・--no-aux なら全 0）
    # dump v3: tokens／scalars は float16・card_idx は int16 で持つ（cast するだけ・§18.5）
    out = {"tokens": np.array(rows["tokens"], np.float32).astype(DT_V3["tokens"]),
            "pol_si": np.array(pol["si"], np.int16),
            "pol_ti": np.array(pol["ti"], np.int16),
            "scalars": np.array(rows["scalars"], np.float32).astype(DT_V3["scalars"]),
            "field": np.array(rows["field"], np.float32),
            "card_idx": np.array(rows["card_idx"], np.int64).astype(DT_V3["card_idx"]),
            "z": np.array([z[w] for w in rows["who"]], np.float32),
            "who": np.array([0 if w == "p1" else 1 for w in rows["who"]], np.int8),
            "kind": np.array(rows["kind"], np.int8),
            "turn": np.array(rows["turn"], np.int16),
            "step": np.array(rows["step"], np.int32),
            "seed": np.full(len(rows["who"]), seed, np.int64),
            "sig": np.array(rows["sig"]),
            "pol_len": np.array(rows["pol_len"], np.int32),
            "pol_chosen": np.array(rows["pol_chosen"], np.int16),
            "pol_n": np.array(pol["n"], np.float32),
            "pol_q": np.array(pol["q"], np.float32),
            "pol_k": np.array(pol["k"], np.int16),
            "pol_sig": np.array(pol["sig"]) if pol["sig"] else np.array([], dtype="U1"),
            "pol_cid": np.array(pol["cid"]) if pol["cid"] else np.array([], dtype="U1"),
            "pol_tcid": np.array(pol["tcid"]) if pol["tcid"] else np.array([], dtype="U1"),
            "deck_kinds": np.array([kj[w] for w in rows["who"]]),
            # dump v4: ε 探索で手を差し替えた行の印（0=木のまま／1=打つ側／2=保留側・§20.8.5）
            "forced": np.array(rows["forced"], np.int8),
            "forced_sig": np.array(rows["forced_sig"]),
            # dump v4: 改良方策 π' の材料（§20.6.1）。教師そのものではない＝既定の訓練は読まない。
            "pol_p": np.array(pol["p"], np.float16),
            "pol_v0": np.array(rows["pol_v0"], np.float16),
            # 局メタ（npz には積まない・part の sidecar へ）
            "_meta": {"seed": seed, "leaders": [la, lb], "kinds": list(kinds),
                      "winner": winner, "turns": turn, "rows": len(rows["who"]),
                      # ε 探索の内訳（§20.8.5／§20.9 の D・ε=0 なら全部 0）
                      "forced": {"play": rows["forced"].count(1),
                                 "hold": rows["forced"].count(2),
                                 "target": rows["forced"].count(3)},
                      "removal_legal_rows": rec.removal_legal_rows}}
    if aux_on:
        out.update({"aux": aux.astype(np.float16), "aux_tok": aux_tok.astype(np.float16),
                    "aux_mask": aux_mask})
    return out


_ROW_KEYS = ("scalars", "field", "card_idx", "z", "who", "kind", "turn", "step",
             "seed", "sig", "pol_len", "pol_chosen")
_POL_KEYS = ("pol_n", "pol_q", "pol_k", "pol_sig", "pol_cid", "pol_tcid", "pol_p")
_TOK_KEYS = ("tokens", "pol_si", "pol_ti")            # NRel 用（v2 で追加・v3 も同じ列）
_V4_KEYS = ("deck_kinds", "forced", "forced_sig", "pol_v0")   # v4 で追加（§20.8／§20.8.5／§20.8.9／§20.6.1）
_AUX_KEYS = ("aux", "aux_tok", "aux_mask")            # 補助教師（v4 で追加・§20.8.2）


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=64,
                    help="生成 sims（純正 N ループは 32→64+ に引き上げ・2026-08-26 設計）")
    ap.add_argument("--net", default=None,
                    help="生成役のネット npz（既定=出荷既定）。旧 --n1-net/--neff-net の後継"
                         "（`neff:`／`n1:` 接頭辞も受ける）")
    ap.add_argument("--dirichlet-eps", type=float, default=0.25,
                    help="root priors への Dirichlet ノイズ（純正 AZ の自己対戦既定 0.25・0=無効）")
    ap.add_argument("--temp-turns", type=int, default=4,
                    help="この turn まではメイン窓を訪問分布 τ=1 でサンプリング（0=無効）")
    ap.add_argument("--shard-games", type=int, default=10)
    ap.add_argument("--dump-v2", action="store_true",
                    help="【廃止・受けるだけ】dump v3 が既定になった（2026-09-07・§18.5）。"
                         "分散生成の古い指示書がこの旗を付けたまま回っても壊れないように残す")
    ap.add_argument("--decks", default=DEFAULT_DECKS,
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "user"),
                    help="デッキの中身（既定 synth＝歴代の波と同じ規約）。"
                         "synth_roles=除去の型を色ごとに差し込む（教材の対照・§20.8.1）")
    ap.add_argument("--no-aux", action="store_true",
                    help="補助教師の列（aux／aux_tok／aux_mask・§20.8.2）を書かない＝dump v3 の"
                         "列だけにする。台帳（1 手 0.2ms の board_json）も積まない")
    ap.add_argument("--eps-play", type=float, default=0.0,
                    help="両方向 ε 探索の「打つ側」（§20.8.5）。木が除去でない手を選んだのに"
                         "候補に除去があれば、確率 P でその中の N 最大へ差し替える（既定 0＝無効）")
    ap.add_argument("--eps-hold", type=float, default=0.0,
                    help="同・「保留側」。木が除去を選んだら、確率 H で除去でない候補の"
                         "N 最大へ差し替える（既定 0＝無効）")
    ap.add_argument("--eps-target", type=float, default=0.0,
                    help="ε の対象ランダム化（§20.9 の D）。木が選んだ除去系の手の直後の"
                         "対象選択を確率 P で一様に引く（既定 0）。`--eps-play` で強制した"
                         "除去（forced=1）の直後は P に依らず必ず引く＝どちらも forced=3")
    # §20.8.6（波 29〜）: 探索の設定。省略＝serve 既定（worlds 1・visits・t=1）＝歴代の波と同じ。
    ap.add_argument("--worlds", type=int, default=None,
                    help="1 決定あたりの世界サンプル本数（§20.7.1・既定 1）")
    ap.add_argument("--select-rule", default=None, choices=("visits", "q_min_n"),
                    help="根で出す手の選び方（§20.5・既定 visits）")
    ap.add_argument("--q-min-frac", type=float, default=None,
                    help="q_min_n の訪問下限の割合（既定 0.125）")
    ap.add_argument("--root-prior-temp", type=float, default=None,
                    help="根の事前分布を P^(1/t) へ（既定 1.0・2.0 で平坦化）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    search = {k: v for k, v in (("worlds", args.worlds), ("select_rule", args.select_rule),
                                ("q_min_frac", args.q_min_frac),
                                ("root_prior_temp", args.root_prior_temp)) if v is not None}

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.dump_v2:
        print("[note] --dump-v2 は廃止（既定が dump v3・列は同じで dtype だけ半分）", flush=True)
    keys = _ROW_KEYS + _POL_KEYS + _TOK_KEYS + _V4_KEYS + (() if args.no_aux else _AUX_KEYS)
    buf = {k: [] for k in keys}
    games = []                                         # part の sidecar（対局メタ）
    shard = n_rows = n_drop = n_main = 0
    initargs = (args.sims, args.net, args.dirichlet_eps, args.temp_turns, args.decks,
                not args.no_aux, args.eps_play, args.eps_hold, search, args.eps_target)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=initargs) as pool:
        done = 0
        for r in pool.imap_unordered(play_one,
                                     [args.seed_base + i for i in range(args.games)]):
            done += 1
            if r is None:
                n_drop += 1
            else:
                games.append(r.pop("_meta"))
                for k in buf:
                    buf[k].append(r[k])
                n_rows += len(r["z"])
                n_main += int((r["kind"] == 0).sum())
            if done % args.shard_games == 0 or done == args.games:
                if buf["z"]:
                    path = os.path.join(args.out, f"n_record_{shard:05d}.npz")
                    tmp = os.path.join(args.out, f".n_record_{shard:05d}.tmp.npz")
                    np.savez_compressed(tmp, **{k: np.concatenate(buf[k]) for k in buf})
                    os.replace(tmp, path)
                    shard += 1
                    buf = {k: [] for k in keys}
                print(f"  {done}/{args.games}局 行{n_rows}（main {n_main}） 棄却{n_drop}"
                      f" {time.time()-t0:.0f}s", flush=True)
    forced = {"play": sum(g["forced"]["play"] for g in games),
              "hold": sum(g["forced"]["hold"] for g in games),
              "target": sum(g["forced"].get("target", 0) for g in games)}
    removal_legal = sum(g["removal_legal_rows"] for g in games)
    # part の sidecar: 対局メタ（seed・両席のリーダー・両席の型）＝層別の集計用（§20.8.1-4）
    with open(os.path.join(args.out, "meta_games.json"), "w") as f:
        json.dump({"decks": args.decks, "seed_base": args.seed_base, "games": games},
                  f, ensure_ascii=False)
    with open(os.path.join(args.out, "meta_n_record.json"), "w") as f:
        json.dump({"games": args.games, "rows": n_rows, "main_rows": n_main,
                   "dropped": n_drop, "sims": args.sims, "decks": args.decks,
                   "enc_version": ENC_VERSION_V14,
                   "dump_version": 3 if args.no_aux else DUMP_VERSION,
                   "aux": not args.no_aux, "seed_base": args.seed_base,
                   "net": E.resolve_net(args.net), "engine": "rust",
                   "dirichlet_eps": args.dirichlet_eps,
                   "temp_turns": args.temp_turns,
                   # ε 探索（§20.8.5／§20.9 の D・既定 0）とその内訳
                   "eps_play": args.eps_play, "eps_hold": args.eps_hold,
                   "eps_target": args.eps_target,
                   "forced": forced, "removal_legal_rows": removal_legal,
                   # 探索の設定の上書き（§20.8.6・空＝serve 既定）
                   "search": search},
                  f, ensure_ascii=False)
    print("N_RECORD_DONE " + json.dumps({"rows": n_rows, "main_rows": n_main,
                                         "dropped": n_drop, "forced": forced,
                                         "removal_legal_rows": removal_legal}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
