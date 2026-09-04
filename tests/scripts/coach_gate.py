"""コーチゲート（v9 §4・mark_gate の後継）: レフェリー検証済みバンドで候補ネットを判定する。

mark_gate（v4/v5 の人間マーク述語）は真実源の登場で部分的に古くなった——実測で
g3@115「無意味な守りをしない」は**守るのが唯一の勝ち筋**（レフェリー: カウンター1/8勝ち・
素通しは捲り32世界でも0勝）、@33 は「どの攻撃でも勝つ」同価値圏だった。本ゲートは
人間述語でなく**真盤面レフェリーの同価値バンド（band-top プランの初手集合）**への所属で
判定する。人間マークはレフェリーで裏取りされた形で引き継がれる（ユーザ承認 2026-07-18）。

判定（mark_gate と同型・gen5 と候補を同条件で比較）:
  - 非退行: base が確実に打てていた点（base≥0.8）で chall が大きく落ちない（chall > base−0.4）
  - 改善: ヒット率合計が base 以上（レフェリー正解へ近づいたか＝進歩検出）
  PASS = 非退行 かつ 改善。

VERIFIED の各点は真盤面レフェリー実測（worlds/sims/日付を出典に明記）から採録。
`--regen` での自動再検証は将来項（現状は採録値が正・変更時はレフェリーを回して更新する）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/coach_gate.py \
    --challenger cand_value.npz,cand_policy.npz --seeds 5
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import mark_gate as MG
import replay_reeval as RE
from opcg_sim.src.core import cpu_ai

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# レフェリー検証済み決定点（真盤面・出典は各行コメント: 世界数/sims/実測日）。
# accept = 同価値バンド（band-top）プランの**初手**の (action_type, card) 集合。
# card=None は action_type のみで判定（PASS/TURN_END 等）。
VERIFIED = [
    # @33: 全攻撃系が 8/8 勝ち・バンド= bare Marco / 付与→リーダー / OP15-119 / 付与Zeus系
    #      （8世界 sims32 auto 2026-07-18）
    ("g3", 33, {("ATTACK", "PRB02-008"), ("ATTACH_DON", "OP11-041"),
                ("ATTACK", "OP15-119"), ("ATTACH_DON", "OP11-106")}),
    # @64: 素攻撃 ≈ 攻撃者へ付与→攻撃（12世界 sims32 正味1・2026-07-17）
    ("g3", 64, {("ATTACK", "PRB02-008"), ("ATTACH_DON", "PRB02-008")}),
    # @68: 付与→ゼウスで攻撃が断定勝ち（16世界 正味+3・素攻撃/リーダー付与はバンド外・2026-07-17）
    ("g3", 68, {("ATTACH_DON", "OP11-106")}),
    # @82（防御窓）: 素通し PASS が最良・EB03/105切りはライフ差でバンド外
    #      （プランスイープ 4世界＋root 6世界・2026-07-17）
    ("g3", 82, {("PASS", None)}),
    # @93: 展開（唯一の勝ち筋系）。root 6世界=OP16-056 1/6・sweep 4世界=OP15-119 系＝
    #      展開2種を許容・付与/攻撃はバンド外（2026-07-16/17）
    ("g3", 93, {("PLAY", "OP16-056"), ("PLAY", "OP15-119")}),
    # @115（防御窓）: OP16-056 カウンターが唯一の勝ち筋（8世界 1/8・捲り32世界でも守り側のみ勝ち・
    #      素通しは最下位・2026-07-18）＝旧 mark_gate「無意味な守りをしない」を反転
    ("g3", 115, {("SELECT_COUNTER", "OP16-056")}),
    # @137: 捲り筋はゼウス付与→ゼウス攻撃のみ（捲り16世界 1/16・他0・2026-07-17）
    ("g3", 137, {("ATTACH_DON", "OP11-106")}),
]

# --- VERIFIED v2（gen7 実対局マーク 2026-07-28・`mark_referee_verify.py` worlds=8 実測） ---
# 出典: tests/fixtures/replays/gen7_marks_20260728/（5局34マーク→真盤面復元＋ターン一致 17点→
# 裁定14点。捲り1/32勝ちの極薄点 @94 は不採録＝旧@137型の反省）。バンド外=改善ターゲット・
# バンド内=非退行ガードの両方を採録する（判定則 judge は共通＝base≥0.8 の点が退行しないこと）。
# 旧 g3（単一対局・gen4期）は --profile g3 で存続（診断用）。
_FIX2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fixtures", "replays", "gen7_marks_20260728")
REPLAYS_V2 = {
    "m1": os.path.join(_FIX2, "opcg_replay_2057134394987494995.json.gz"),   # ナミ(人) vs シャンクス(CPU)
    "m2": os.path.join(_FIX2, "opcg_replay_3806796710697874793.json.gz"),   # シャンクス(人) vs ナミ(CPU)
    "m4": os.path.join(_FIX2, "opcg_replay_6563214359889287880.json.gz"),   # ナミ(人) vs シャンクス(CPU)
    "m5": os.path.join(_FIX2, "opcg_replay_9195490382040907274.json.gz"),   # シャンクス(人) vs ナミ(CPU)
}
# --- v48 エネル調査用リプレイ（2026-08-09・自己対戦・裁定の裏取り対象） ---
# 出典: `game_replay_log.py --matchup nami:p_enel --sims 128`（gen13・Dirichlet なし＝決定論的に
# 再現できる）。**まだ検証済み点ではない**——ユーザ裁定（先行2ターン目に1コストの登場時ドローを
# 出して OP15-118 エネル(cost6)を掘る）をレフェリーで裏取りするための盤面として登録する。
# 裏が取れた点だけを VERIFIED へ昇格させる（v46 の教訓＝ゲートの点を動かす前に勝率で効くかを測る）。
_FIX48 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "fixtures", "replays", "enel_v48_20260809")
REPLAYS_V48 = {
    "e1": os.path.join(_FIX48, "enel_selfplay_991001.json.gz"),   # p1=エネル / p2=ナミ（ナミ勝ち）
    "e2": os.path.join(_FIX48, "enel_selfplay_991002.json.gz"),   # p1=ナミ / p2=エネル（ナミ勝ち）
}
# --- 人間の基準線（2026-08-10・ユーザがエネルを握って CPU ナミに勝った実対局） ---
# 出典: アプリの traced 対局（cpu_trace=true・seed 5703575646787553228）。人間 96手 / CPU 44手、
# **turn 10 でエネル（p1・人間）勝ち**。CPU 自己対戦のエネルは e1/e2 とも敗北しているので、
# 「このデッキは勝てる」ことの存在証明であり、打点しきい値・決定点の食い違い抽出の基準になる。
_FIXH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "fixtures", "replays", "enel_human_20260810")
REPLAYS_HUMAN = {
    "h1": os.path.join(_FIXH, "enel_human_5703575646787553228.json.gz"),  # p1=エネル(人) / p2=ナミ(CPU)
    # 見本棋譜（2026-09-04・ユーザ採取・設計フェーズの定性分析用 `docs/reports/2026-09-04_human_samples_qualitative.md`）。
    # 全て p1=人間・勝ち。相手 CPU は learned（当時の出荷既定）。ローは効果エンジンの不具合が
    # 残る対局＝手順の型は参考にするが、個々の効果解決を正解扱いしない（ユーザ注記）。
    "h2": os.path.join(os.path.dirname(_FIXH), "human_enel_vs_luffy_20260904",
                       "enel_vs_luffy_3840809972709220407.json.gz"),      # エネル vs ルフィ OP16-022（後攻・turn12勝ち）
    "h3": os.path.join(os.path.dirname(_FIXH), "human_law_vs_luffy_20260904",
                       "law_vs_luffy_7808696651703795638.json.gz"),       # ロー ST10-001 vs ルフィ OP16-022（先攻・turn11勝ち）
    "h4": os.path.join(os.path.dirname(_FIXH), "human_enel_vs_roger_20260904",
                       "enel_vs_roger_5764875644126075159.json.gz"),      # エネル vs ロジャー OP13-003（先攻・turn15勝ち）
    # 対照（殴り合い系リーダーを人間が握る・2026-09-04 採取）
    "h5": os.path.join(os.path.dirname(_FIXH), "human_roger_vs_luffy_20260904",
                       "roger_vs_luffy_8473732354491595481.json.gz"),     # ロジャー OP13-003 vs ルフィ OP16-022（後攻・turn14勝ち）
}
# **VERIFIED v3**（2026-07-30 再裁定・/tmp/mark_verify3.jsonl・worlds16）。旧 v2（13点・worlds8）は
# 効果対話の既定解決欠陥（`docs/reports/default_interaction_fix_20260730.md`＝捨て札が公開札を
# 捨てる／up-to 獲得を常時見送る）で PLAY 系プランの測定が汚染されていたため、修正後エンジンで
# 34マークを全点再裁定した（11/34 で裁定/accept が変化・旧表は同レポートと v18 レポートに保存）。
# 変数名は既存プローブ（prior_bound_probe / value_blind_probe）互換のため VERIFIED_V2 のまま。
VERIFIED_V2 = [
    # m1: CPU=シャンクス
    # m1@3 は 2026-07-30 に「判別不能」で取り下げ→**2026-08-03 再採録**: (1) ユーザ最終裁定＝
    # 非発動イワンコフ（6000×1 で条件不成立・手札 6→5 でバニラ2000 が湧くだけ）を出すのが問題、
    # (2) 修正済み評価（32世界 def_temp0.7・マージン記録）でウタが勝敗（7/6/4）・ライフ差
    # （−0.44/−0.69/−1.25）の両方で最上位＝旧「判別不能」は def_temp=0 測定器の盲点だった、
    # (3) gen10 の行動欠陥を実測（8seed でイワンコフ 5/ウタ 3）＝改善ターゲットとして機能する。
    # TURN_END は両指標最下位（序盤6枚手札はテンポ優先）のため band 外。
    ("m1", 3,  {("PLAY", "OP09-002")}),                                    # ウタを出す（非発動イワンコフは band 外）
    # m1@14 は accept を**素通しへ再反転**（2026-08-04・ユーザ指摘＋修正済み評価方法で実測）:
    # 攻撃7000 vs リーダー5000 で、止めるには **2枚必要**（2000単独では 7000 同値＝攻撃側勝ち・
    # 1000+2000 で 8000）。ライフ5 の入口でカード2枚を1ライフに換えるのは割に合わない
    # （素通しなら受けたライフが手札に入るので実収支は 3枚 vs 1ライフ）。
    # 32世界 def_temp0.7 実測: 素通し z=-0.688（最上位タイ・応答直後 手札6/ライフ4）に対し
    # 旧 accept の OP10-011 は -0.812、OP09-002 は -0.875 と**どちらも下位**。
    # なお **m1@15（1枚払った後）は逆に守り切るのが正**（下記）＝「入口では守らない／
    # 一度払ったら止め切る」の一貫した原理。
    ("m1", 14, {("PASS", None)}),
    # m1@15（m1@14 で 1000 を1枚払った後・現在 6000 で 7000 に届かない）: 32世界 def_temp0.7
    # 実測で OP10-011（2000→8000 で止め切る）z=-0.562 が明確に最上位・素通しは -0.938 で最下位
    # ＝裁定どおり。既に払った1枚を無駄にせず止め切る側が正しい。
    ("m1", 15, {("SELECT_COUNTER", "OP10-011")}),
    # m1@42 は**取り下げ**（2026-08-04・ユーザ裁定）: m4@12 と同じく**パワー2000 の
    # イワンコフにドン2枚を付与した状態**という局面前提そのものが不自然で、ここでの
    # 最善手を検証点にしても実プレイの参考にならない。
    # m1@94 は**取り下げ**（2026-08-04・ユーザ裁定）: m1@42 / m4@12 と同種で、**パワー2000 の
    # ウタにドン2枚を付与した状態**という局面前提が不自然。実プレイの参考にならない。
    # m2: CPU=ナミ
    # m2@12 は**取り下げ**（2026-08-04・ユーザレビュー）: 攻撃7000 vs 防御7000（既に
    # カウンター2000を投入済み）で、**あと1000を1枚足せば 8000 で守り切れる**局面＝
    # 「守れないから捨てる」m2@58 型ではない。収支は 2枚 vs 1ライフ（素通しならライフが
    # 手札に入るため）で人間でも判断が割れる領域。裁定を確定できるまで検証済み集合から外す。
    ("m2", 44, {("ATTACH_DON", "OP11-041")}),                              # リーダーへ付与（守り）
    ("m2", 58, {("PASS", None)}),                                          # ガード（accept は素通しのみに縮小）
    # m2@64 は**取り下げ**（2026-08-04・ユーザ裁定）: accept の Mr.3（OP16-056・パワー5000）で
    # 攻撃しても**相手リーダーは 7000 で届かない**（相手キャラはこの時点でアタック可能な
    # レスト状態のものが無い）＝攻撃する意味が無い局面で「攻撃が唯一の正解」とする裁定は誤り。
    # m2@66 は**取り下げ**（2026-08-08・ユーザ裁定「正解が無い系統」）: 実測が裁定と逆を向いた。
    # 旧裁定（2026-08-05）は「ターン終了までにナミ・リーダー・ロビンの攻撃と Mr.3 の効果起動を
    # 全て消化すること」（turn_all 形式）だったが、v45/v46 の計測で次が判明した。
    #  (1) CPU は**一度も早期に畳んでいない**（16seed 全てで TURN_END が唯一の合法手）。未消化は
    #      相手の2つの「このターン中」デバフ（OP09-001 シャンクス −1000 / OP10-018 カマクラ十草紙
    #      −2000・どちらも相手が対象を選ぶ）が自リーダーに集中し 7000→4000 となり、リーダー攻撃が
    #      どの標的にも届かなくなって `_prune_futile_attacks` が正しく落とすため。
    #  (2) **4件消化に勝率上の価値が無い**。レフェリー32世界（CRN・初手強制）で
    #      Mr.3起動→効果解決→リーダー攻撃 19/32・→ロビン攻撃 18/32 に対し、
    #      攻撃を先に置く2腕は 13〜14/32 でバンド外。リーダー先攻か否かは1勝差＝ノイズ。
    #  (3) つまり**ゲートは、より勝つ手順（Mr.3 先・3/4 消化）を不合格にし、より勝たない手順
    #      （リーダー先・4/4 消化）を通していた**。旧裁定の理由づけ「攻撃の順序はほぼ等価」も、
    #      当時のレフェリーが攻撃を含む複数手順を表現できなかった制約下の測定に基づく。
    #  (4) そもそも**ネットにこの分岐の区別が付いていない**。ply0 の探索（160sims・合法手7）は
    #      上位4候補の Q が 0.09 幅の団子（−0.389〜−0.477）で訪問も最上位 25〜34% と薄く散り、
    #      8seed の選択は リーダー攻撃4回 / Mr.3起動4回 の**ちょうど 50/50**＝誤りではなく未決定。
    #      この 0.09 幅を注入で動かすのは v42（摂動過大でアリーナ 0.378）を繰り返す領域。
    # 再裁定するなら基準は「Mr.3 の起動を攻撃より先に置く」だが、CPU は全 seed で起動済みで
    # 勝率上の誤りが観測されていないため、検証点としての識別力が無い。
    # 詳細は docs/reports/cpu_v45_m2at66_root_cause_20260808.md / cpu_v46_m2at66_winrate_20260808.md。
    # m4: CPU=シャンクス
    # m4@2 の accept はユーザ最終裁定（2026-08-03）: **イワンコフ出しが正解**。手札に 6000×2
    # （OP16-012×2）があり登場時効果が**発動する**（3枚引いて2枚捨て＝手札 5→5・エンジン実測）
    # ＝実質ノーコストで体と手札の入替が得られる。2026-08-02 の TURN_END/A&S&L への拡大裁定は
    # 「発動しない」誤認に基づくもので破棄。単手測定（32世界 def_temp0.7）でも3枝拮抗＝
    # イワンコフを band 外に置く根拠は無い。gen8〜10 は本点を打てている＝非退行ガードとして機能。
    # 「非発動イワンコフを咎める」課題は m1@3（6000×1・不成立・手札 6→5）が正しい標的。
    ("m4", 2,  {("PLAY", "ST30-004")}),
    # m4@8 は**取り下げ**（2026-08-04・ユーザレビュー）: 合法手7に対し accept 4＝
    # 「ターン終了/ウタ/リーダー付与以外なら何でも合格」でテストとしての識別力が無い
    # （gen8〜11 が一律 1.00 なのは実力でなく外すのが難しいだけ）。再裁定するならバンドを絞る。
    # m4@12 は**取り下げ**（2026-08-04・ユーザ裁定）: パワー2000 のイワンコフにドン2枚を
    # 付与した状態という**局面前提そのものが不自然**（人間側の指し手の産物）で、ここでの
    # 最善手を検証点にしても実プレイの参考にならない。
    # m5: CPU=ナミ
    # m5@7 accept を**ユーザ裁定で縮小**（2026-08-04）: 正解は「リーダーにドン3枚を付与して
    # リーダーで殴る」。ゼウス（OP11-106）の登場時は**自分のライフ1枚を手札に加えるのが条件**で、
    # 除去できるのは場のイワンコフ（パワー2000・脅威にならない）＝ライフ1枚の方が価値が高い。
    ("m5", 7,  {("ATTACH_DON", "OP11-041")}),                              # ナミへドン付与→リーダーで攻撃
    # --- h1: 人間基準線（エネル・2026-08-11 追加・ユーザ指示「今回問題だったエネルの盤面もゲートへ」） ---
    # h1@2（turn1）: ユーザ裁定（2026-08-10）＝サトリ（1コスト登場時 don!!-1 ドロー）で山の
    # 勝ち筋（OP15-118）を掘る。don!!-1 は翌ターンのリーダー効果でドンデッキから再装填される
    # ため実質無料（真正のターン跨ぎ経済）。**勝率レフェリーでは裁定不能**（v49 実測: 教師正本
    # 設定×32世界で掘り/無行動とも 0/32＝飽和・ロールアウト方策自身が掘った札を活かせない）＝
    # 裁定ベース accept。**注意**: v49 掘り注入（dig_inject_gen）の裁定と同族のため、注入後の
    # ネットにとって本点は「注入が転移したか」の確認点であり独立した実力証明ではない
    # （注入コーパス自体には h1@2 は入らない＝group0 は学習規約上 val 側）。
    ("h1", 2,  {("PLAY", "OP15-066")}),                                    # turn1 サトリで掘る
    # h1@35（turn5）: ユーザ裁定（2026-08-10）＝6コストのエネル OP15-118 を出す（パワー10000 が
    # 7000 しきい値対面の勝ち筋）。v48 の旗艦欠陥「出せない」は計器バグ#6（復元盤面の
    # JournaledList）による偽欠陥で、修正後は prior 1位（0.600）・探索も最多訪問＝人間と一致。
    # 本点はその**回帰ガード**（計器・探索・ネットのどれが壊れても検出される）。
    ("h1", 35, {("PLAY", "OP15-118")}),                                    # turn5 6cエネル着地
]


def hit(desc, accept):
    """decide の記述（action_type/card）が合格集合に入るか（pure）。"""
    at = desc.get("action_type")
    card = desc.get("card")
    return (at, card) in accept or (at, None) in accept


def turn_all_required(accept):
    """accept が「ターン内全消化」形式（{"turn_all": {(type, card), ...}}）なら必須集合を
    返し、従来の初手集合なら None を返す（pure・decide_rate のディスパッチ用）。"""
    if isinstance(accept, dict):
        req = accept.get("turn_all")
        return frozenset(req) if req else None
    return None


def turn_all_rate(eng, m0, name, required, seeds, sims, max_plies=24):
    """決定点から**自ターンの終わりまで**エンジンに指させ、TURN_END までに required の
    (action_type, card) を全て実行した割合（m2@66 型・2026-08-05 ユーザ裁定）。

    初手だけ見る decide_rate では「どれか1つ打てば合格」になり、「全ての攻撃と起動を
    使い切ってからターンを終える」という裁定を表せない。相手側の戦闘応答（カウンター窓・
    効果対象選択）も同じエンジンが指す（self-play と同じ規約＝gen12 では箱読み出しが処理）。
    sticky 世界線は seed ごとにリセット＝ターン内は serve と同じ一貫した世界で計画する。
    max_plies 到達時は「必須を消化済みか」で判定（終え方でなく消化を測る計器のため）。

    **2026-08-08 現在この形式の検証点は存在しない**（唯一だった m2@66 は v46 で取り下げ＝
    消化率が勝率と逆を向いていた）。機構は将来の再裁定に備えて残す。"""
    game = eng.game
    hit_n = 0
    for s in range(seeds):
        eng._world_seeds = {}
        getattr(eng, "_battle_plans", {}).clear()   # 入口コミットのプランも独立化（2026-08-15・同一盤面の反復 decide でシード1のプラン尾を返す測定汚染の修正）
        rng = np.random.default_rng(9100 + 97 * s)
        mgr = m0
        done = set()
        ok = False
        for _ply in range(max_plies):
            if game.is_terminal(mgr):
                ok = required <= done
                break
            actor_name = game.current_player(mgr)
            if actor_name is None:
                break
            actor = mgr.p1 if mgr.p1.name == actor_name else mgr.p2
            mv = eng.decide(mgr, actor, sims=sims, rng=rng)
            if mv is None:
                break
            try:
                d = cpu_ai._describe_move(mgr, mv) or {}
            except Exception:
                d = {"action_type": (mv or {}).get("action_type")}
            if actor_name == name:
                at = d.get("action_type")
                if at == "TURN_END":
                    ok = required <= done
                    break
                done.add((at, d.get("card")))
            nxt = game.apply(mgr, mv, actor_name)
            if nxt is None:
                break
            mgr = nxt
        else:
            ok = required <= done
        hit_n += 1 if ok else 0
    return hit_n / max(seeds, 1)


def decide_rate(eng, m0, actor, accept, seeds, sims):
    req = turn_all_required(accept)
    if req is not None:
        return turn_all_rate(eng, m0, actor if isinstance(actor, str) else actor.name,
                             req, seeds, sims)
    n = 0
    for s in range(seeds):
        eng._world_seeds = {}
        getattr(eng, "_battle_plans", {}).clear()   # 入口コミットのプランも独立化（2026-08-15・同一盤面の反復 decide でシード1のプラン尾を返す測定汚染の修正）
        mv = eng.decide(m0, actor, sims=sims, rng=np.random.default_rng(9100 + 97 * s))
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        if hit(d, accept):
            n += 1
    return n / max(seeds, 1)


def min_reliable_delta(seeds):
    """点別の命中率差が『測定ノイズでない』と言える最小幅（pure・2σ・最悪ケース p=0.5）。

    命中率は seeds 回のベルヌーイ試行＝SE ≤ 0.5/√n。2条件の差の SE は √2 倍なので
    2σ ≈ 1.414/√n。v22 実測（`docs/reports/coach_gate_variance_20260729.md`）で
    5seed（bar 0.63）では m4@8 の 0.60→0.20 が『退行』に見えたが、16seed（bar 0.35）では
    両者 0.38 で差が無かった。**この bar 未満の点別増減を『治った/壊れた』と書かない**。"""
    return 1.4142135623730951 / (seeds ** 0.5) if seeds > 0 else float("inf")


def judge(rows, regress_base=0.8, regress_drop=0.4):
    """点別 (base, chall) → (非退行OK, 改善OK, 退行リスト)（pure・mark_gate と同型の判定）。"""
    regressions = [(tag, i, b, c) for (tag, i, b, c) in rows
                   if b >= regress_base and c <= b - regress_drop]
    improve = sum(c for _t, _i, _b, c in rows) >= sum(b for _t, _i, b, _c in rows)
    return (not regressions), improve, regressions


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--baseline", default=None,
                    help="value.npz[,policy.npz]（既定=出荷既定＝現 gen7）")
    ap.add_argument("--seeds", type=int, default=16,
                    help="点ごとの decide 回数。**5 は分散が大きすぎる**（v22 実測: 5seed で "
                         "『退行』に見えた m4@8 が 16seed では差なし）。`min_reliable_delta` 参照")
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--profile", default="v2", choices=("v2", "g3", "all"),
                    help="v2=gen7実対局13点（既定）／g3=旧7点（gen4期・診断用）／all=両方")
    # --- 分散実行（2026-08-14・gen15 採用ゲートの高速化）: 点列をストライプ分割して
    # 子セッションへ配る。部分実行の PASS/FAIL は**参考値**（正式判定は全点集約後に
    # coach_gate.judge を再適用する）。--out jsonl が集約用の正本。
    ap.add_argument("--point-offset", type=int, default=0)
    ap.add_argument("--point-stride", type=int, default=1)
    ap.add_argument("--out", default="", help="点ごとの結果を jsonl 追記（分散集約用）")
    ap.add_argument("--chall-boxes", action="store_true",
                    help="挑戦者エンジンに箱化フルセット（macro_moves+defense_box+box_dialog+"
                         "戦闘箱設定）を適用する＝既定ON前の裁定13点非退行確認（2026-08-25）。"
                         "ネット自体は既定と同一でもよい（機構だけのA/B）")
    ARGS = ap.parse_args()
    CR.ARGS = argparse.Namespace(true_board=True)

    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    db = _load_db()

    def _eng(spec, boxes=False):
        kw = dict(macro_moves=True, defense_box=True, box_dialog=True,
                  box_battle=True, quiesce=True) if boxes else {}
        if not spec:
            return LearnedEngine(**kw)
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0],
                             policy_path=parts[1] if len(parts) > 1 else None, **kw)

    base_eng = _eng(ARGS.baseline)
    chall_eng = _eng(ARGS.challenger, boxes=ARGS.chall_boxes)

    points = {"v2": VERIFIED_V2, "g3": VERIFIED,
              "all": VERIFIED + VERIFIED_V2}[ARGS.profile]
    if ARGS.point_stride > 1 or ARGS.point_offset:
        points = points[ARGS.point_offset::ARGS.point_stride]
        print(f"点ストライプ: offset={ARGS.point_offset}/stride={ARGS.point_stride}"
              f" → {len(points)}点", flush=True)
    replays = {**MG.REPLAYS, **REPLAYS_V2, **REPLAYS_V48, **REPLAYS_HUMAN}
    CR.GAMES = {}
    rows = []
    for tag, i, accept in points:
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, i)
        if isinstance(built, str):
            # 真盤面（記録全手順の再実行）が再生不能な対局（例: h1＝人間対局は相手CPUの
            # カウンター選択を replayer が再現できない・既知の制限）は**フレーム復元へ
            # フォールバック**する。h系の裁定・v48/v49 の測定は全てフレーム復元が正本。
            rec, fbi, actions = CR.GAMES[tag]
            built = MG._restore(db, rec, fbi, actions, i)
            if isinstance(built, str) or built is None:
                print(f"{tag}@{i}: 復元不可（スキップ）: {built}")
                continue
            print(f"{tag}@{i}: 真盤面再生不能→フレーム復元で測定")
        m0, who = built
        name = who if isinstance(who, str) else who.name
        actor = m0.p1 if m0.p1.name == name else m0.p2
        b = decide_rate(base_eng, m0, actor, accept, ARGS.seeds, ARGS.sims)
        c = decide_rate(chall_eng, m0, actor, accept, ARGS.seeds, ARGS.sims)
        rows.append((tag, i, b, c))
        print(f"  {tag}@{i:<4} base={b:.2f} chall={c:.2f}  合格手={sorted(accept)}", flush=True)
        if ARGS.out:
            with open(ARGS.out, "a") as f:
                f.write(json.dumps({"tag": tag, "i": i, "base": b, "chall": c,
                                    "seeds": ARGS.seeds, "sims": ARGS.sims},
                                   ensure_ascii=False) + "\n")
    bar = min_reliable_delta(ARGS.seeds)
    sig = [(t, i, b, c) for t, i, b, c in rows if abs(c - b) >= bar]
    print(f"\n測定ノイズでないと言える差の下限（2σ・seeds={ARGS.seeds}）= {bar:.2f}")
    print(f"  この bar を超えた点: "
          + (", ".join(f"{t}@{i}({b:.2f}→{c:.2f})" for t, i, b, c in sig) if sig else "なし")
          + "  ← これ未満の増減は『治った/壊れた』と読まない")
    ok_nr, ok_imp, regs = judge(rows)
    print(f"\n改善: {'OK' if ok_imp else 'NG'}"
          f"（chall計 {sum(c for *_ , c in rows):.1f} vs base計 {sum(b for _t, _i, b, _c in rows):.1f}）")
    print(f"非退行: {'OK' if ok_nr else 'NG'} {[(t, i, b, c) for t, i, b, c in regs]}")
    verdict = "PASS" if (ok_nr and ok_imp) else "FAIL"
    print(f"COACH_GATE_RESULT {json.dumps({'verdict': verdict, 'points': len(rows)})}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
