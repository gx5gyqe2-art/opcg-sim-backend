"""学習型CPU（Gen2 value+policy + NN誘導MCTS）の本番エントリ。

docs/reports/cpu_rl_pilot_p3_results_20260630.md。P3本走で得た Gen2 ネット（自己対戦2世代・
製品L1+α-βに 0.925・かつ製品より高速）を実ゲームに配線する。返り値は `decide_guarded` と同一契約
（単一 move 辞書 or None）＝decide 経路にドロップイン可能。

- 葉価値 = 学習 value ネット、事前確率 = 学習 pointer policy、探索 = ノード型 PUCT MCTS。
- 不完全情報は探索ごとに1世界へ決定化（PIMC・チート防止）。net/vocab はプロセス内で1回だけロード。

**ネットの持ち方（perf計画 A3）**: `LearnedEngine` が1つのネットを**明示ハンドル**で保持する
（arena の net-vs-net＝新Gen vs 凍結Gen2 を同一プロセスで戦わせるため）。本番既定 CPU が通る
`decide_learned` は既定ネットの**プロセス共有シングルトン**（`_default_engine()`）を使う薄いラッパ
＝**挙動不変**（vocab/game はネット非依存なので複数エンジンで共有ロード可能）。
"""
import os
import weakref
from typing import Any, Dict, Optional

import numpy as np

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.learned.policy import PolicyScorer, state_context
from opcg_sim.src.learned.action import legal_action_matrix
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.learned import config as CFG
from opcg_sim.src.learned.config import (
    C_PUCT, SERVE_SIMS, SERVE_DIRICHLET_EPS, SERVE_STICKY_WORLD)
from opcg_sim.src.learned.mcts import (   # make/unmake版（唯一の探索実装。旧clone版は削除済み）
    TreeMCTS, clear_box_budget, in_battle, in_dialog, reset_box_budget,
    resolve_battle_inplace, resolved_branch_values)
# move_sig（手の同一性キー）は plan.py の定義が正（重複定義しない・箱コミット実行 2026-08-26）
from opcg_sim.src.learned.plan import _find_move, move_sig
from opcg_sim.src.utils.loader import CardLoader

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "learned")
_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")

# gen9 = gen8 と「v25 防御混合候補」の value 重み**線形補間 α=0.5**（v28・
# docs/reports/gen9_adoption_20260801.md）。候補側は修正済み効果解決エンジン上の密対面
# コーパス（nami:shanks 2,048局 244,544行・v6符号化）＋防御CF 438行で gen8 を追い学習した
# もの（ep2 lr2e-4・混合ラベル α=0.5）。純候補はコーチゲート v3 初 PASS（6.1 vs 4.7・
# 退行ゼロ）だがアリーナ 0.454 で、補間 α=0.5 が「コーチ 5.6 PASS × アリーナ 0.475
# CI[0.426,0.524]（parity 棄却されず）」の意思決定点（交換曲線は v28 レポート）。
# **昇格基準（wr≥0.55）では FAIL のままユーザ判断で採用**＝人間検証14点の改善を取り、
# 自己対戦で最大 ~2-3pp を許容する取引（2026-08-01 ユーザ決定）。
# policy は gen8 と同一の重みを v6 へ恒等温スタート拡張したバイナリ（v12 確定＝policy
# 微調整は有害のため凍結。value が v6=60スカラーのため ctx 幅を揃える必要がある——
# 版不整合は行動特徴が5列ずれて黙って壊れる: dense_finetune の 2026-07-31 修正参照）。
# gen10 = gen9 に「登場時オプションの実測特徴（符号化 v7・63スカラー）」を積んで追い学習
# （v30・docs/reports/cpu_v30_option_feature_20260802.md）。密対面コーパス（nami:shanks
# 512局 54,511行・v7・マーク局面シード0.25）を gen9 の v7 恒等拡張ネットで生成し、
# ep2 lr2e-4・混合ラベル α=0.5・**distill0.5**（gen9 への value アンカー＝一般対局の忘却抑制）で
# 追い学習。v7 特徴（`cpu_ai.onplay_option_scan`＝手札の各 ON_PLAY 持ちを make/unmake で
# 試し発火を実測・ドン非依存）が m4@2 型（同じカードの価値が手札構成で反転する点）の value を
# 初めて動かした（子盤面 value 差 -0.140→-0.106・representation-bound の緩和）。判定＝
# コーチゲート v3 PASS（5.0 vs 4.6・m5@7 獲得）× アリーナ 0.509（gen9 と同等・800局
# CI[0.476,0.542]・Elo+6.1）。**昇格基準（wr≥0.55）では FAIL のまま、アリーナ中立の純増＋
# v7 特徴を土台として確定するユーザ判断で採用**（2026-08-02。標的 m4@2 の完全解決は
# 次段のペア順位損失へ）。decide は v7 実測ぶん +77%（309→546ms・1秒予算内）。
# policy は gen9 と同一重みの v7 恒等拡張バイナリ（v12 確定＝policy 微調整は有害・value との
# 符号化版一致が必須）。符号化は v7 で net の feat_dim から自動判別。gen9 以前はリプレイ
# 再現・A/B・ロールバック用に同梱維持（レフェリー教師の錨は gen5 固定のまま）。
# gen11 = gen10 に「蒸留アンカー付き順位学習」を積み、さらに gen10 と **α=0.3 で線形補間**
# （v33・docs/reports/gen11_adoption_20260803.md）。符号化は **v8**（=v7 + 自場集約3。相手場
# （v5）と同じ関数の純対称化＝自場が生カウントのみだった非対称の解消。しきい値つき弱ボディ
# 特徴は汎用性のため設けない・ユーザ方針 2026-08-03）。
# 教師は `option_pair_gen` のカード単位ペア（160局・296群・1,328ペア）で、**v32 で確立した
# 2つの評価方法の修正**を含む: (1) ロールアウトの **def_temp=0.7**（argmax 防御は「手札を回して
# 即出しする枝」に偽の優位を与える＝温存カウンターが使われる世界が生成されない。m4@2 実測で
# イワンコフ 18/32→11/32 と偽優位が消失）、(2) **margin_blend ラベル**（勝敗 z ＋
# 0.25·clip(平均残ライフ差/4)＝拮抗群を勝ち方の質でタイブレーク）。
# 学習は `rank_finetune_anchored`（v33）＝順位ヒンジのバッチごとに「dense 一般盤面 16,000点で
# gen10 の予測へ引き戻す蒸留バッチ」を交互に流す。v32 は**アンカー無し順位ヒンジが順位を
# 上げるほど防御較正（m2@12/58 の「素通しが正」）を先に壊す**負の結果を3回再現しており、
# 錘で既存挙動を固定したまま順位だけ動かすのが本世代の核。残った m2@58 の劣化は gen10 との
# α=0.3 補間で回収した（gen9 と同じ交換曲線の使い方）。
# 判定＝コーチゲート v3 **PASS（6.9 vs 6.5・歴代最高）**・bar 超えは狙った m1@3 の改善
# （0.50→1.00＝**非発動イワンコフ**〈手札 6000×1 で条件不成立・手札 6→5 でバニラ 2000 が
# 湧くだけ〉を出さずウタを出す）のみで**退行ゼロ**、アリーナ 800局 0.4925 CI[0.463,0.522]
# ＝中立（Elo−5.2）。**昇格基準（wr≥0.55）では FAIL のままユーザ判断で採用**（2026-08-03・
# gen9/gen10 と同じ「人間検証点の改善 × 自己対戦中立」の取引）。
# policy は gen10 と同一重みの v8 恒等温スタート拡張バイナリ（v12 確定＝policy 微調整は有害・
# value との符号化版一致が必須＝版不整合は行動特徴列がずれて黙って壊れる）。符号化は net の
# feat_dim から自動判別。gen10 以前はリプレイ再現・A/B・ロールバック用に同梱維持
# （レフェリー教師の錨は gen5 固定のまま）。
# gen12 = gen11 と「防御窓CF順位学習の腕」の value 重み**線形補間 α=0.5**（v35・
# docs/reports/gen12_adoption_20260805.md）。腕側は `defense_cf_gen` の解決後盤面コーパス
# （nami:shanks 4セッション並列生成・19シャード 2,198行 584群・def_temp0.7・margin_blend）で
# gen11 を `rank_finetune_anchored`（全面アンカー scale1.5・δ0.3/lr8e-5/ep10）したもの。
# 純腕は自ターン判断 m1@3 を 1.00→0.62-0.69 へ実際に劣化させたため（独立2 seed 系列で再現）、
# gen9/gen11 と同じ交換曲線の使い方で α=0.5 に置いた（α=0.3/0.5 が両立点・0.7 以上で劣化）。
#
# **本世代の核は重みでなく探索の規約**（ユーザ整理 2026-08-05「バトルを一つの箱としてまとめて、
# 探索と value はバトルをするかどうか／バトルの結果どんな盤面・手札・ライフになるかのみを
# 判断する」）＝戦闘を1つの箱として畳み、**出口（解決後の盤面）だけを value で比べる**。
# 3機構を既定 ON にして初めて意味を持つ: `SERVE_QUIESCE`（葉が戦闘中なら解決してから評価）・
# `SERVE_BATTLE_READOUT`（実対局の防御窓は出口 value で選ぶ）・`TREE_BOX_BATTLE`（木の戦闘窓
# ノードも出口最良の1手へ畳む）。これで木・実対局・教師コーパスの解決規約が初めて一致した。
# 判定＝コーチゲート **6.19 vs 4.00（8点・退行ゼロ・ガード4点満点維持）**。bar 超えは
# m1@15（0.00→1.00＝「一度払ったら2000で止め切る」）・m2@44（0.00→0.62）・m5@7（0.00→0.56）
# の3点で、後2点は木の箱化が初めて動かした（攻撃の帰結が「相手手札−1」か「相手ライフ−1」の
# 具体的な出口として立ち上がるため。攻撃は必ず相手に損失を強いるので『止められる＝無駄』では
# ない——ユーザ指摘 2026-08-05）。アリーナ 800局 0.5212 CI[0.489,0.554] Elo+14.8＝中立
# （帯別 0.565/0.550/0.490/0.480）。**昇格基準（wr≥0.55）では FAIL のままユーザ判断で採用**
# （2026-08-05・gen9/gen10/gen11 と同じ「人間検証点の改善 × 自己対戦中立」の取引）。
# policy は gen11 と同一バイナリ（v12 確定＝policy 微調整は有害・value と符号化版 v8 が一致）。
# gen13 = gen12 に**戦闘出口専用 value ヘッド**（`ValueNet.EXIT_HEADS` の "battle"・残差 MLP
# hidden4）を載せ、人間裁定済み7点の注入順位ペア（`verified_inject_gen.py`・84ペア）で
# **ヘッドのみ**を学習したもの（v43・docs/reports/gen13_adoption_20260808.md）。
# 胴体と本体 value ヘッドは物理的に凍結＝**通常の葉評価・プラン評価は gen12 と bit 一致**で、
# 変わるのは戦闘箱の枝順位づけ（防御窓の読み出し・木の箱化・ターン箱の戦闘窓）だけ。
# 学習は margin0.03/lr1e-3/ep200 に**摂動を点の修正に必要な最小限へ絞り**、学習後に
# 残差ロジットを defcf 2198盤面上で中心化（順位を厳密に保存して一律バイアスだけ除去）＝
# 一般の戦闘出口への摂動は標準偏差 0.038（枝間マージン 0.02〜0.03 と同規模）。
# 前段の失敗2つがこの設計を決めた: v40=本体 value を全面順位学習→ゲート満点でもアリーナ
# 0.447 の有意退行（置き場所の誤り）。v42=同じ注入教師で margin0.2/hidden32 のヘッド→
# 摂動 std0.32 が7点の外でノイズとなりアリーナ 0.378 で中止（摂動過大）。
# 判定＝コーチゲート PASS（7.4 vs 6.6・bar 超えは狙った m1@14 0.00→1.00〈素通しが正の
# 交換レート裁定〉のみ・退行ゼロ。**ただし8点中7点は本候補の訓練データ＝ゲートは確認用**）
# × アリーナ 400ペア800局 0.5212 CI[0.494,0.549] Elo+14.8＝中立（シャード別 0.575/0.535/
# 0.495/0.480・追加の seed90000系63ペア込みでも 0.518）。**昇格基準（wr≥0.55）では FAIL の
# ままユーザ判断で採用**（2026-08-08・gen9〜12 と同じ「人間検証点の改善 × 自己対戦中立」の
# 取引）。ロールバックはヘッドを外すだけ（幅0にすれば gen12 と bit 一致＝過去世代より安全）。
# policy は gen12 と同一バイナリ（v12 確定＝policy 微調整は有害・符号化版 v8 一致）。
# gen14 = gen13 の value 本体を**符号化 v9**（ドンデッキ残2＋自デッキ残キャラ頂点2・恒等温
# スタート）へ拡張し、掘り裁定の注入順位ペア（`dig_inject_gen.py`・エネル席5群11ペア・
# 「登場時ドローで掘ってEND ＞ 無行動END」）で蒸留アンカー付き順位微調整（v33 機構・
# アンカー=リプレイ復元633一般盤面）したもの（v49・docs/reports/gen14_adoption_20260811.md）。
# **戦闘出口ヘッドと vocab は gen13 と bit 一致**（較正維持）・胴体の摂動は最大 0.006/重み・
# 一般盤面の予測摂動 std 0.069。判定＝コーチゲート 9/9 PASS（h1@2 掘り 0.00→1.00 が
# 2σ=0.35 超の実獲得・注入点はゲート7点と独立・退行ゼロ）× アリーナ 400ペア800局
# 0.5038 CI[0.473,0.534] Elo+2.6＝中立。**昇格基準（wr≥0.55）FAIL のままユーザ判断で採用**
# （2026-08-11・gen9〜13 と同じ「人間検証点の改善 × 自己対戦中立」の取引）。
# 既知の限界（v49 レポート§トリレンマ）: 非エネル席の低コスト展開マージンが平均 +0.09 傾く
# 漏れ（行動レベルの退行はゲート未検出）・サトリ移植プローブ ✗（掘りはドン経済でなく
# カード/文脈特徴に紐づく）＝50ペア規模の注入で安全に買えるのはエネル獲得のみ。
# ロールバックは既定を gen13 に戻すだけ。policy は gen13 と挙動同一（v9 温スタートの
# ゼロ拡張のみ・v12 確定＝policy 微調整は有害）。
# gen15 = **符号化 v12**（=v9 + リーダー物理要約24・リーサルΔ抜き）の本体に、裁定注入で
# 学習した**戦闘出口ヘッド**を載せ直したもの（gen15c・docs/reports/gen15_adoption_20260815.md）。
# 本世代は**歴代で初めてアリーナの昇格基準（wr≥0.55 かつ CI下限>0.50）を満たした**
# ＝225ペア450局 **0.5756 CI[0.533,0.619] Elo+52.9**・帯別 0.595/0.593/0.546/0.565 で
# 4帯すべて 0.5 超え（デッキ相性の偏りでない）。gen9〜gen14 は全て「人間検証点の改善 ×
# 自己対戦中立」の取引でユーザ判断採用だったため、自己対戦そのもので有意に強い世代は初。
# 寄与は4つの合わせ技:
#  (1) **リーダー物理要約24**（能力木→毎ターン率×自/相手・`leader_feat`・ユーザ提案 2026-08-14）
#      ＝接戦帯を支配するリーダー再帰効果が現行特徴に0ビットだった欠陥への処方。ID非依存で
#      新リーダーへ汎化する。符号化コストはカードID キャッシュで**実質ゼロ**。
#  (2) **B/G混合コーパス**（実デッキ 17,912行 ＋ 合成リーダー世界 8,048行〔card_idx=UNK〕）
#      ＝「IDが分からない時はリーダー物理要約で読む」経路を数百種の合成リーダーで教える。
#      実データのみでは未見リーダー（ハンニャバル）で過適合した（+0.684→+0.365）ものが
#      混合で +0.622 へ回復＝B系の存在意義が G系の数字で初めて実証された。
#  (3) **入口コミット**（`SERVE_BATTLE_COMMIT`・ユーザ決定 2026-08-15）＝防御プランを窓の
#      入口で1回立て以後は実行のみ。人間の意思決定（カウンターを切った後に考え直さない）に
#      合わせる構造で、「払い始めたら払い切る」が保証され防御窓の再解決も消えて 31% 高速化。
#  (4) **戦闘出口ヘッドの載せ直し**（v43 レシピ・margin0.03/hidden4/e64・defcf 1198盤面で中心化）。
# **v10 のリーサル距離Δ3列は意図的に外した**（v12 の要点）: エンジンで台本を再生する実測特徴で
# ~25ms/盤面あり、探索が1手で数百回符号化するため decide が **0.47s(v9) → 13.5s(v11)** と
# 本番予算1秒を28倍超過していた（2026-08-15 実測・アリーナが10分/ペアになって発覚）。
# Δ自体も v53 で両系とも転移せず効果未実証のままだったため、**出荷実績のある v9 系譜に
# 無料の24列だけを継ぐ**構成にした。gen15 の decide は **0.49s**（gen14 0.47s と同等）。
# 判定チェーン: レイテンシ 0.49s（新設の関門）→ ns2 中間帯 r **+0.709**（gen14 +0.455）→
# コーチゲート 20点 **PASS**（11.9 vs 11.1・退行ゼロ・m2@44 0.00→1.00 の獲得）→ アリーナ上記。
# **ゲートの限定**: 注入8点は本候補の訓練データ＝その点は確認用（v43 と同じ取引）。独立証拠は
# 非注入点の非退行とアリーナ。
# ロールバックは既定を gen14 に戻すだけ（符号化は net の入力次元から自動判別＝配線変更不要）。
# policy は gen14 と挙動同一（v9→v12 の恒等ゼロ拡張のみ・v12 確定＝policy 微調整は有害）。
_DEFAULT_VALUE = os.path.join(_MODELS, "gen15_value.npz")
_DEFAULT_POLICY = os.path.join(_MODELS, "gen15_policy.npz")

# vocab（カード語彙）と game（アダプタ）はネット非依存＝プロセス内で1回だけ作り全エンジンで共有する。
_SHARED: Dict[str, Any] = {}


def _shared_vocab_game():
    if not _SHARED:
        db = CardLoader(os.path.join(_DATA, "opcg_cards.json"))
        db.load()
        for cid in list(db.raw_db.keys()):
            db.get_card(cid)
        _SHARED["vocab"] = E.build_vocab(db)
        _SHARED["game"] = OPCGGame()
    return _SHARED["vocab"], _SHARED["game"]


def available() -> bool:
    """モデル重みが同梱されているか（未同梱環境ではフォールバックさせる）。"""
    return os.path.exists(_DEFAULT_VALUE)


def _value_fn(vnet, vocab, enc_version=1):
    """葉価値関数＝素の predict（純正AZ化 2026-08-25: 旧 aux 粘り項は補償層として削除）。"""
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab, version=enc_version)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(vnet.predict(batch)[0])
    return value


def _exit_head_value_fn(vnet, vocab, enc_version=1, kind="turn"):
    """**箱の出口専用**の価値関数（v39 ターン末 / v41 戦闘出口・`ValueNet.predict_exit`）。

    箱の階層ごとに較正を分ける（v38/v40 の学び）: 通常の葉評価＝既存ヘッド、箱を畳んだ出口＝
    その階層の専用ヘッド。ネットに該当ヘッドが無ければ `predict_exit` は既存ヘッドへ落ちる
    ＝従来と同値。aux 粘り項は掛けない（飽和域の減衰は「探索の葉の無差別」を解く発見的補正で、
    出口 z を直接教えたヘッドの較正を上書きする理由がない）。"""
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab, version=enc_version)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(vnet.predict_exit(batch, kind)[0])
    return value


def _priors_fn(pnet, vocab, enc_version=1):
    if pnet is None:
        return None
    def priors(state, legal):
        me = state.pending_actor_action()[0]
        ctx = state_context(state, me, vocab, version=enc_version)
        am = legal_action_matrix(state, legal, me)
        p = pnet.priors(ctx, am)
        return p if p.shape[0] == len(legal) else None
    return priors


def _net_enc_version(vnet) -> int:
    """ロード済み value ネットの入力次元から符号化世代（encoder version）を判別する。

    v1=Gen2 出荷ネット（scalars 14）・v2=リーダー付与ドン追加（scalars 16）。重み側の
    次元が真実源＝コードのデフォルトに依存しない（v2 ネットへ差し替えた時点で自動有効）。
    `vnet.feat_dim` は lead_slots（リーダー条件付け専用枠）を自動的に除外する＝LC net でも誤判定しない。
    """
    feat = vnet.feat_dim
    for v in E.known_versions():
        if feat == E.feature_dim(v):
            return v
    raise ValueError(f"value ネットの入力次元が未知（feat_dim={feat}）: encoder と重みの対応を確認")


def warm_start_value(vnet, from_version, to_version):
    """value ネットを from_version→to_version へ温スタート拡張する（append-only 前提・恒等保存）。

    増えたスカラー（末尾 append）ぶんのゼロ行を W1 に挿入するだけ＝拡張後の出力は from 版と恒等。
    版の知識はここ（`E.scalars_dim`）に集約＝ネットは offset だけ受け取る。任意の版差（v1→v2, v2→v3,
    v1→v3…）に同一コードで対応する。to<from（縮小）は append-only に反するため拒否。"""
    insert_at = E.scalars_dim(from_version)
    n_new = E.scalars_dim(to_version) - insert_at
    if n_new < 0:
        raise ValueError(f"温スタートは拡張方向のみ（from=v{from_version} → to=v{to_version} は縮小）")
    return vnet.expanded(insert_at, n_new)


def warm_start_policy(pnet, from_version, to_version):
    """policy ネットの温スタート拡張（`warm_start_value` と同契約・挿入 offset は ctx 末尾＝scalars_dim）。"""
    insert_at = E.scalars_dim(from_version)
    n_new = E.scalars_dim(to_version) - insert_at
    if n_new < 0:
        raise ValueError(f"温スタートは拡張方向のみ（from=v{from_version} → to=v{to_version} は縮小）")
    return pnet.expanded(insert_at, n_new)


class LearnedEngine:
    """1つの Gen2 ネット（value+policy）を明示ハンドルで保持し 1 手を決める。

    net-vs-net（arena で新Gen vs 凍結Gen2）用に、ネットを**席ごとに別インスタンス**で持てるようにする。
    `value_path`/`policy_path` 省略時は出荷 Gen2（`gen2_*.npz`）＝本番既定 CPU と同一。vocab/game は
    ネット非依存なので既定では共有ロード（`_shared_vocab_game`）を使う。
    """

    def __init__(self, value_path: Optional[str] = None, policy_path: Optional[str] = None,
                 vocab=None, game=None,
                 sims: Optional[int] = None, c_puct: Optional[float] = None,
                 quiesce: Optional[bool] = None,
                 box_battle: Optional[bool] = None,
                 don_margin: Optional[bool] = None,
                 macro_moves: Optional[bool] = None,
                 defense_box: Optional[bool] = None,
                 box_dialog: Optional[bool] = None,
                 box_commit: Optional[bool] = None,
                 dirichlet_eps: Optional[float] = None,
                 temp_turns: Optional[int] = None,
                 residual_dig: Optional[bool] = None,
                 residual_activate: Optional[str] = None):
        if vocab is None or game is None:
            svocab, sgame = _shared_vocab_game()
            vocab = vocab if vocab is not None else svocab
            game = game if game is not None else sgame
        # (C) マージン付与／候補生成の席別 seam（quiesce 等と同じ A/B 用）: 指定時は共有 game
        # でなく席専用の adapter を持つ＝同一プロセスの net-vs-net で席ごとに候補生成を変えられる。
        if (don_margin is not None or macro_moves is not None
                or defense_box is not None):
            from opcg_sim.src.learned.adapter import OPCGGame as _OG
            game = _OG(don_margin=don_margin, macro_moves=macro_moves,
                       defense_box=defense_box)
        self.vocab = vocab
        self.game = game
        # 探索つまみのエンジン別上書き（None=既定に従う・A/B 用の seam）。
        # **設定時は decide の呼び出し引数より優先**する＝席ごとに探索設定を変えた net-vs-net
        # （`search_config_probe.py`）で、ハーネス側が渡す sims を上書きできるようにするため。
        self.sims = sims
        self.c_puct = c_puct
        # 静止探索（None=config.SERVE_QUIESCE に従う）。**同一プロセスで席ごとに機構を変える**
        # ための seam＝「新機構の候補 vs 現行本番」を公平に1回で測る（グローバル定数を書き換える
        # 測り方だと両席に同時に効いてしまい、機構とネットの寄与が分離できない）。
        self.quiesce = quiesce
        # 木の中の箱化（None=config.TREE_BOX_BATTLE に従う・同上の席別 seam）。
        self.box_battle = box_battle
        # 対話箱（P3/P5・config.TREE_BOX_DIALOG のエンジン別上書き・None=config に従う）
        self.box_dialog = box_dialog
        # 箱コミット実行（2026-08-26・config.SERVE_BOX_COMMIT のエンジン別上書き・None=config）
        self.box_commit = box_commit
        # 生成の探索多様性（純正AZ 2026-08-27・serve 既定は両方無効＝挙動不変）:
        #  dirichlet_eps: root priors への Dirichlet ノイズ（None=config.SERVE_DIRICHLET_EPS=0）
        #  temp_turns: turn <= この値のメイン窓で argmax でなく訪問分布 π∝n（τ=1）から
        #              サンプリングする（0/None=無効）。AZ の自己対戦の探索はこの2つが担う
        #              ——無いと同じ線ばかり打ち、π 教師が argmax クローンに退化して飽和する。
        self.dirichlet_eps = dirichlet_eps
        self.temp_turns = temp_turns
        # 残ドン掘り（2026-09-02・対照生成の腕A・serve 既定 None=無効＝挙動不変）:
        # 木が**メイン窓で TURN_END を選んだ時だけ**、手札に「登場時にドンを戻してドローする
        # コスト1キャラ」があり場に空きがあれば、代わりにそれを出す。通常の手（大型・攻撃・
        # 付与）は全て木の判断のまま＝腕の違いは「捨てるはずだったドンで掘ったか」だけ。
        # 掘りの良否は勝敗（z）が決める＝人間裁定を教えない。条件は力学だけ（カードID・
        # リーダー名を含めない）。発火は `residual_dig_events` に1件ずつ記録し、事後に
        # 区分別（ターン・場のドン・ドンデッキ残）で勝敗を割り直せるようにする。
        self.residual_dig = residual_dig
        self.residual_dig_events: list = []
        # 残り起動（2026-09-02・対照生成の腕A2・serve 既定 None=無効＝挙動不変）:
        # 木が**メイン窓で TURN_END を選んだ時だけ**、リーダーの起動効果が未使用（合法手に
        # ACTIVATE_MAIN がある）で、その効果が「ドン‼デッキからドンを追加する」構造語を持てば、
        # 代わりに起動する（ドンデッキが空でも「レストのドンをキャラに付与」が効くので条件に
        # しない）。起動で開く**自分のキャラへの付与対話**は
        # 方針で付与先を決める（"low"＝このターン攻撃できるキャラのうち最低パワー＝攻撃本数を
        # 増やす・実プレイの筋／"high"＝最高パワー）。その後は木に制御を返す（増えたドンで
        # 追加の手・付与したキャラで攻撃するかは木の判断）。監査（`don_refund_audit`）で
        # c10 が起動を負に評価して一度も使わない盲点が出たため、勝敗で良否を測る腕。
        self.residual_activate = residual_activate
        self.residual_activate_events: list = []
        self._resact_pending = False      # 直前の decide で起動を返した（付与対話を方針で解く）
        # 方策 priors の注入口（純正Nループ④ 2026-08-26）: None=既定（self.pnet の G 系
        # priors）。設定時はその callable(state, legal)->np.array|None を全経路（木・窓の
        # 根畳み・コミット生成）で使う＝N 系ネットの方策チャネルを pnet の G 形式を経由せず
        # serve に繋ぐ seam（value の vnet 差し替えと対）。
        self.priors_override = None
        # ターン内 sticky 世界線の seed キャッシュ {(id(manager), turn, player): (weakref, seed)}（§_world_rng）。
        self._world_seeds: Dict[Any, Any] = {}
        # 箱コミット（選んだ箱の自分側の残り手順）{(id(manager), turn, player): (weakref, steps)}。
        # steps の要素は (a) move_sig タプル (b) ("__box__", sig, 残り回数)＝DON_BOX の
        # カウントダウン形。ターンが変われば key が変わる＝自然に失効する。
        self._commits: Dict[Any, Any] = {}
        self.vnet = ValueNet.load(value_path or _DEFAULT_VALUE)
        # 符号化は**ネット付属 vocab を最優先**（訓練時の card_id→idx を固定）。カードDBが増えても
        # 既存カードの idx がズレず（build_vocab は途中挿入でズレる・2026-07-15 実害）、ネットが
        # 知らない新カードは encode 側で UNK=0 に落ちる＝範囲外クラッシュも起きない。
        # vocab_ids の無い旧 npz のみ、共有 build_vocab（現行DBソート）へフォールバックする。
        # dict は ids 単位のプロセス内キャッシュで共有＝同一ネットのエンジン同士は同一オブジェクト
        # （net-vs-net の複数エンジン同居でも重複を作らない・従来の共有前提を保つ）。
        if getattr(self.vnet, "vocab_ids", None):
            ids = tuple(self.vnet.vocab_ids)
            hit = _SHARED.setdefault("net_vocabs", {}).get(ids)
            if hit is None:
                hit = _SHARED["net_vocabs"][ids] = E.vocab_from_ids(ids)
            self.vocab = hit
        # 符号化世代は重みの入力次元から自動判別（v1=出荷Gen2・v2=リーダー付与ドン特徴）。
        self.enc_version = _net_enc_version(self.vnet)
        pp = policy_path or _DEFAULT_POLICY
        self.pnet = PolicyScorer.load(pp) if os.path.exists(pp) else None
        if self.pnet is not None:
            from opcg_sim.src.learned.action import ACTION_DIM
            # 行動特徴は append-only で拡張される（v9: +カウンター値/対象=リーダー）。旧 net の
            # 行動次元は現 ACTION_DIM 以下でありうる（新列は PolicyScorer._fit_actions が切詰＝
            # 出力恒等）。ここでは「状態 ctx の世代一致」だけを検査する: 行動次元
            # = in_dim − feature_dim(世代) が (0, ACTION_DIM] を外れたら世代不一致。
            ad = int(self.pnet.in_dim) - E.feature_dim(self.enc_version)
            if not (0 < ad <= ACTION_DIM):
                raise ValueError(
                    f"value/policy の符号化世代が不一致（value=v{self.enc_version}, "
                    f"policy in_dim={self.pnet.in_dim}）: 同一世代の npz ペアを配置してください")

    def _world_rng(self, manager, name: str, rng) -> np.random.Generator:
        """ターン内 sticky な PIMC 決定化 rng を返す（SERVE_STICKY_WORLD）。

        1 ターンは「ドン付与→攻撃→…」の連続 decide で構成されるが、decide ごとに世界線
        （相手伏せ手札のサンプル）を引き直すと、付与を正当化した攻撃プランが次の decide の
        別世界で棄却され「付与だけして攻撃しない」無駄ドンが出る（マークレビュー F3）。
        同一 (game, turn, player) の間は決定化 seed を固定し、ターン内の計画を同一世界で
        一貫させる。seed は初回 decide の rng から引く＝global random の消費量は従来と同一
        （リプレイ決定論を保つ）。キャッシュは挿入順で刈る（同時進行ゲーム数 ≪ 上限）。
        """
        key = (id(manager), int(getattr(manager, "turn_count", 0) or 0), name)
        hit = self._world_seeds.get(key)
        # id() は解放後に再利用されうる＝別ゲームの stale seed を拾うと「同一 seed 対局の
        # プロセス間再現」が破れる。weakref で同一オブジェクトであることを検証して排除する。
        seed = hit[1] if (hit is not None and hit[0]() is manager) else None
        if seed is None:
            seed = int(rng.integers(0, 2 ** 63 - 1))
            if len(self._world_seeds) >= 256:
                for k in list(self._world_seeds)[:128]:
                    del self._world_seeds[k]
            try:
                self._world_seeds[key] = (weakref.ref(manager), seed)
            except TypeError:
                pass   # weakref 不可の manager は毎 decide 新世界（従来挙動に退化・安全側）
        # 毎 decide 新しい Generator を同一 seed から作る＝ターン内のどの decide でも
        # determinize の shuffle が同じ乱数列から始まる（公開情報の更新は盤面側から反映される）。
        return np.random.default_rng(seed)

    def _exit_value_fn(self, kind="turn"):
        """箱の出口の評価関数（該当の専用ヘッドを持つネットのみ・無ければ None＝従来どおり）。

        None を返す経路では呼び出し側が通常の value_fn を使う＝**既存の同梱ネットでは
        v39/v41 導入前と完全に同一の計算**（新ヘッドを持つ候補ネットでだけ挙動が変わる）。"""
        # vnet が None のエンジン（value_fn を差し替えて「決めているのは value か」を検査する
        # 経路）でも成立させる＝ヘッド無しと同じ従来経路へ落とす。
        if self.vnet is None or not self.vnet.has_exit_head(kind):
            return None
        return _exit_head_value_fn(self.vnet, self.vocab, self.enc_version, kind=kind)

    def _battle_value_fn(self):
        """戦闘箱の物差し（v41）。戦闘出口ヘッドが無ければ通常の葉評価と同一関数を返す。

        `_exit_value_fn` と違って None を返さない: 戦闘箱の呼び出し側（`resolved_branch_values`）
        は物差しを必ず1つ要求するため、ここで「ヘッドがあればそれ／無ければ本体 value」を
        1か所に閉じ込める（分岐を呼び出し側にばら撒かない）。"""
        return self._exit_value_fn("battle") or _value_fn(
            self.vnet, self.vocab, self.enc_version)

    def _priors(self):
        """priors 関数（seam）: priors_override があればそれ・無ければ pnet の G 系 priors。"""
        if self.priors_override is not None:
            return self.priors_override
        return _priors_fn(self.pnet, self.vocab, self.enc_version)

    def _window_choice(self, manager, name, det_rng):
        """**窓の根畳み**（純正AZ化 2026-08-25 の統一読み出し）: 窓（戦闘窓＝in_battle／
        （box_dialog 有効時の）効果対話窓＝in_dialog）では、決定化1世界の合法手を
        `resolved_branch_values`（出口盤面の value）で採点し argmax の1手を返す。

        これは木の `_expand` の箱畳み（TREE_BOX_BATTLE/TREE_BOX_DIALOG）が root ノードで
        行う計算と**同一の意味論**＝探索の迂回ではなく「木の root 畳みの高速版」（畳まれた
        root は単一辺になり訪問を配る意味が無いため、探索を回さず直接その1手を返す）。
        物差しも木と同一: 戦闘窓＝`_battle_value_fn()`（戦闘出口ヘッド）・対話窓＝本体 value
        （出口はメイン窓の通常盤面＝葉評価と同じ物差し・window_pred=in_dialog）。

        世界線は探索と同じ決定化（PIMC・sticky）。返す手は決定化クローン上の合法手
        （`TreeMCTS.run` と同一契約）。返り値は (move, root統計, 評価に使った盤面)＝
        トレースは呼び出し側で埋める。選べないときは (None, None, None) で full-tree へ委ねる。
        """
        battle = in_battle(manager)
        mgr = self.game.determinize(manager, name, det_rng)
        legal = self.game.legal_actions(mgr)
        if not legal:
            return None, None, None
        if len(legal) == 1:
            return legal[0], None, None
        if battle:
            vf, wp = self._battle_value_fn(), None
        else:
            vf, wp = _value_fn(self.vnet, self.vocab, self.enc_version), in_dialog
        vals = resolved_branch_values(
            self.game, mgr, name, legal, vf,
            self._priors(), window_pred=wp)
        ok = [i for i, v in enumerate(vals) if v is not None]
        if not ok:
            return None, None, None      # 全枝で解決に失敗＝従来の full-tree に任せる（安全側）
        best = max(ok, key=lambda i: vals[i])
        # トレース用の root 統計は探索と同じ形（N/Q）で作る: 各枝を1回ずつ解決して評価した、
        # という事実をそのまま N=1 に、判断の根拠である出口評価を Q に載せる。
        stats = {"legal": legal,
                 "N": np.array([1.0 if v is not None else 0.0 for v in vals]),
                 "Q": np.array([v if v is not None else -1.0 for v in vals], dtype=float)}
        return legal[best], stats, mgr

    # --- 箱コミット実行（ユーザ決定 2026-08-26「箱は選ぶ時だけ判断し、中身は機械実行」）------
    #
    # 従来は箱（DON_BOX 等）を選んでも先頭原始手1枚を返すだけで、次の decide で木を回し直す
    # ＝箱の中身で再判断していた。評価が正当化した継続と実行される継続がずれる事故クラス
    # （battle commit 0.320 事件・プラン半消化バグ〔docs/reports/2026-08-25_planbox_finding.md〕）
    # の根治として、**選んだ箱の自分側の残り手順を確定し、以後は機械実行**する。
    # 手順が実盤面で非合法化したら契約違反＝その箱のコミットを丸ごと破棄して通常の判断へ
    # （縮退して続きだけ拾わない＝箱単位で再入札）。相手の窓・外周は対象外。

    def _commit_key(self, manager, name):
        return (id(manager), int(getattr(manager, "turn_count", 0) or 0), name)

    def _store_commit(self, manager, name, steps):
        """残り手順をコミットする（空なら何もしない・上限64で古い半分を掃除）。"""
        if not steps:
            return
        if len(self._commits) >= 64:
            for k in list(self._commits)[:32]:
                del self._commits[k]
        try:
            self._commits[self._commit_key(manager, name)] = (weakref.ref(manager),
                                                              list(steps))
        except TypeError:
            pass   # weakref 不可の manager はコミットを持たない（毎 decide 判断＝従来挙動）

    def _trace_to_steps(self, tr, name):
        """解決トレース（(actor, move) 列）→ 自分側の手順（sig 列）。

        DON_BOX（配分箱/アタック箱）はカウントダウン形 ("__box__", sig, 総原始手数) に変換する
        （sig は don_k 非含有のため残回数はステップ側に持つ＝半消化バグ 2026-08-25 の再発防止。
        付与 k 枚＋攻撃形はさらに1回＝ATTACK で完走）。相手の手は含めない（相手の窓は対象外。
        実行中に相手の実選択が予測と割れれば sig 照合が失敗し、契約違反として箱ごと破棄される）。"""
        steps = []
        for actor, mv in tr:
            if actor != name:
                continue
            if mv.get("action_type") == "DON_BOX":
                p = mv.get("payload") or {}
                total = int(p.get("don_k") or 0) + (1 if p.get("target_ids") else 0)
                if total >= 1:
                    steps.append(("__box__", move_sig(mv), total))
            else:
                steps.append(move_sig(mv))
        return steps

    def _commit_apply_ok(self, manager, name, move):
        """コミット手の適用検証（観測中立: クローンにのみ適用・global random は保存/復元）。"""
        import random as _random
        rst = _random.getstate()
        try:
            return cpu_ai._apply_clone(manager, name, move, stop_at_select=True) is not None
        except Exception:
            return False
        finally:
            _random.setstate(rst)

    def _commit_step(self, manager, player, name, trace):
        """コミット済み手順の機械実行（decide 入口・窓の根畳み/木より前）。

        現在の legal（エンジンの候補生成そのまま）から先頭 sig を照合し、見つかれば原始手化
        して返す。照合失敗＝契約違反 → その key のコミットを**全破棄**して None（通常の判断へ）。"""
        key = self._commit_key(manager, name)
        hit = self._commits.get(key)
        if hit is None:
            return None
        # id() は解放後に再利用されうる（_world_rng と同じ理由）＝weakref で同一性を検証。
        if hit[0]() is not manager:
            del self._commits[key]
            return None
        steps = hit[1]
        move = None
        try:
            legal = self.game.legal_actions(manager) if steps else None
            if legal:
                st = steps[0]
                if isinstance(st, tuple) and len(st) == 3 and st[0] == "__box__":
                    # カウントダウン形: sig（don_k 非含有）が今の箱候補に残っていることを
                    # 契約として確認し、原始手は算術で導出する（付与が残っていれば
                    # ATTACH_DON・攻撃形の最終回は ATTACK）。legal の先頭一致に don_k を
                    # 委ねない＝k=0 箱が先に並ぶと攻撃が早出しされる。
                    _tag, sig, n = st
                    if _find_move(legal, sig) is not None:
                        if n <= 1:
                            steps.pop(0)
                        else:
                            steps[0] = ("__box__", sig, n - 1)
                        tgts = list(sig[2] or ())
                        if tgts and n <= 1:
                            move = {"kind": "game", "action_type": "ATTACK",
                                    "payload": {"uuid": sig[1], "target_ids": tgts}}
                        else:
                            move = {"kind": "game", "action_type": "ATTACH_DON",
                                    "payload": {"uuid": sig[1]}}
                else:
                    mv = _find_move(legal, st)
                    if mv is not None:
                        p = mv.get("payload") or {}
                        k = int(p.get("don_k") or 0)
                        if mv.get("action_type") == "DON_BOX" and k > 0:
                            # 防御的経路（生成側はカウントダウン形へ変換済み＝通常来ない）
                            total = k + (1 if p.get("target_ids") else 0)
                            if total > 1:
                                steps[0] = ("__box__", st, total - 1)
                            else:
                                steps.pop(0)
                        else:
                            steps.pop(0)
                        move = cpu_ai.don_box_first_primitive(mv)
        except Exception:
            move = None
        # 適用検証（2026-08-26 void 修正）: コミットの返す手——特にカウントダウンの合成
        # ATTACK/ATTACH_DON——は legal に実在しないため、通常経路が持つ「木で一度適用して
        # 失敗手を弾く」検証を通っていない。実盤面クローンで適用可能かを確認し、不可なら
        # 契約違反として箱ごと全破棄する（実例: 手札2枚捨てが必要なアタックのコスト不足が
        # そのまま実対局へ出て ACTION_EXCEPTION→void・arena_n3 seed292004）。
        if move is not None and not self._commit_apply_ok(manager, name, move):
            move = None
        if move is None or not steps:
            # 消化完了（空）または契約違反（照合失敗/例外/適用不能）＝どちらも key を畳む
            self._commits.pop(key, None)
        if move is not None and trace is not None:
            try:
                _fill_trace(trace, manager, player, move, None)
                trace["readout"] = "box_commit"
            except Exception:
                pass   # 分析失敗で対局を止めない
        return move

    def _commit_window_continuation(self, manager, name, world, move):
        """窓の根畳み（`_window_choice`）で選んだ枝の自分側継続をコミットする。

        選んだ枝を決定化クローンに適用し、評価（`resolved_branch_values`）と**同じ解決規約**
        （同じ value_fn・box_depth・CRN＝global random 保存/復元）で解決した trace の自分側
        手順を確定する＝評価が正当化した継続と実行される継続が同一になる（旧・入口コミットの
        一般形）。失敗はコミット無し（毎 decide 判断＝従来挙動へ退化・安全側）。"""
        import random as _random
        rst = _random.getstate()
        try:
            if in_battle(manager):
                vf, wp = self._battle_value_fn(), None
            else:
                vf, wp = _value_fn(self.vnet, self.vocab, self.enc_version), in_dialog
            nxt = self.game.apply(world, move, name)
            if nxt is None:
                return
            tr = []
            resolve_battle_inplace(
                self.game, nxt, self._priors(),
                value_fn=vf, box_depth=CFG.BOX_RESOLVE_DEPTH, window_pred=wp, trace=tr)
            self._store_commit(manager, name, self._trace_to_steps(tr, name))
        except Exception:
            pass   # コミット生成の失敗で手を止めない
        finally:
            _random.setstate(rst)

    def _commit_play_dialog(self, manager, name, det_rng, move, world=None):
        """木が PLAY / ACTIVATE_MAIN を選んだ時: 後続の**自分側**効果対話列をコミットする。

        決定化クローン（`world` 指定時はそれ＝テスト/計器用）に適用し、対話が開けば評価
        （木の対話箱畳み）と同じ解決規約（window_pred=in_dialog・本体 value・box_depth）で
        解決＝同じ結論になる。対話が無ければコミット無し。乱数は CRN 規約（保存→復元）で
        汚染しない。"""
        import random as _random
        rst = _random.getstate()
        try:
            if world is None:
                world = self.game.determinize(manager, name, det_rng)
            nxt = self.game.apply(world, move, name)
            if nxt is None or not in_dialog(nxt) or in_battle(nxt):
                return                       # 対話が無ければコミット無し
            tr = []
            resolve_battle_inplace(
                self.game, nxt, self._priors(),
                value_fn=_value_fn(self.vnet, self.vocab, self.enc_version),
                box_depth=CFG.BOX_RESOLVE_DEPTH, window_pred=in_dialog, trace=tr)
            self._store_commit(manager, name, self._trace_to_steps(tr, name))
        except Exception:
            pass   # コミット生成の失敗で手を止めない
        finally:
            _random.setstate(rst)

    def decide(self, manager, player, sims: int = SERVE_SIMS, c_puct: float = C_PUCT,
               rng=None, trace=None, record=None) -> Optional[Dict[str, Any]]:
        """このエンジンのネットで 1 手決定する（`decide_learned` と同一契約・同一探索）。

        `record`（dict）は棋譜ダンプ用の**観測専用**出力（純正Nループ 2026-08-26）:
        kind（main=木探索／window=窓の根畳み／commit=箱コミット機械実行）・sig（選択手の
        move_sig・DON_BOX は箱レベル＝原始手化前）・main では groups（等価マージ後の
        全候補 {sig, n=訪問合算, q}・`_merge_root_stats` と同一集計）と sims。
        trace と違い L1 第二意見を採らない＝生成コスト最小。挙動には一切影響しない。"""
        name = player.name
        # 戦闘箱の枝予算をこの decide のぶんだけ張り直す（config.BOX_BRANCH_BUDGET）。
        # 使い切ったら箱の評価をやめて policy 最良手へ退避する＝**decide が必ず戻る**。
        # 通常局面では発動しない余裕（実測最大の約4.4倍）を取ってある。
        # 予算は decide の中だけの話なので、抜けるときに必ず外す（下の finally）。
        reset_box_budget()
        try:
            return self._decide_inner(manager, player, name, sims, c_puct, rng, trace,
                                      record)
        finally:
            clear_box_budget()

    def _decide_inner(self, manager, player, name, sims, c_puct, rng, trace, record=None):
        if self.sims is not None:
            sims = self.sims           # エンジン別上書き（未設定=None で従来どおり引数/既定）
        if self.c_puct is not None:
            c_puct = self.c_puct
        # 箱コミット実行（2026-08-26）: コミット済みの残り手順があれば機械実行（窓の根畳み・
        # 木より前）。照合失敗＝契約違反はコミット全破棄で下の通常判断へ落ちる。
        use_commit = (CFG.SERVE_BOX_COMMIT if self.box_commit is None else self.box_commit)
        if use_commit:
            move = self._commit_step(manager, player, name, trace)
            if move is not None:
                if record is not None:
                    record["kind"] = "commit"
                    record["sig"] = move_sig(move)
                return move
        # numpy rng の種を **global random** から引く＝リプレイ種（routers が cpu_trace 時に random.seed）で
        # learned 対局も決定論再生できる。通常対局は global random 未 seed（プロセス由来）＝実質ランダム。
        if not isinstance(rng, np.random.Generator):
            import random as _random
            rng = np.random.default_rng(_random.getrandbits(64))
        det_rng = self._world_rng(manager, name, rng) if SERVE_STICKY_WORLD else rng
        # 窓の根畳み（純正AZ化 2026-08-25）: 窓では full-tree を回さず、木の root 畳み
        # （TREE_BOX_BATTLE/TREE_BOX_DIALOG が `_expand` で行う計算）と同一の意味論で
        # 出口 value 最良の1手を直接返す（`_window_choice` 参照）。メインフェーズ
        # （バトルをするか・どこを殴るか）は下の full-tree のまま＝変更しない。
        # 残り起動（腕A2）: 直前の decide で起動を返したなら、続く付与対話を方針で解く。
        # メイン窓に戻ったら（対話でない）フラグは落とす＝起動1回ぶんの対話列にだけ効く。
        if self._resact_pending:
            if in_dialog(manager) and not in_battle(manager):
                alt = self._residual_attach_move(manager, player)
                if alt is not None:
                    return alt
            else:
                self._resact_pending = False
        use_dialog = (CFG.TREE_BOX_DIALOG if self.box_dialog is None else self.box_dialog)
        if in_battle(manager) or (use_dialog and in_dialog(manager)):
            move, stats, ev_mgr = self._window_choice(manager, name, det_rng)
            if move is not None:
                # 箱コミット: 選んだ枝の自分側の戦闘内/対話内継続を確定（以後は機械実行）。
                if use_commit and ev_mgr is not None:
                    self._commit_window_continuation(manager, name, ev_mgr, move)
                if record is not None:
                    record["kind"] = "window"
                    record["sig"] = move_sig(move)
                if trace is not None:
                    try:
                        _fill_trace(trace, ev_mgr if ev_mgr is not None else manager,
                                    player, move, stats)
                        trace["readout"] = "window_resolved"
                    except Exception:
                        pass   # 分析失敗で対局を止めない
                return move
        mcts = TreeMCTS(self.game, value_fn=_value_fn(self.vnet, self.vocab, self.enc_version),
                        priors_fn=self._priors(),
                        c_puct=c_puct, n_sims=sims,
                        dirichlet_eps=(SERVE_DIRICHLET_EPS if self.dirichlet_eps is None
                                       else self.dirichlet_eps),
                        determinize_fn=lambda s, r: self.game.determinize(s, name, r), rng=det_rng,
                        quiesce=self.quiesce, box_battle=self.box_battle,
                        box_dialog=self.box_dialog,
                        battle_value_fn=self._battle_value_fn())
        move, _, legal = mcts.run(manager)
        # 同名カードの別実体（手札の複製等）は探索木で別 edge になり訪問数が分裂する。
        # 素の argmax(N) は分裂した等価手を系統的に不利にする（例: EB03-053×2 のカウンターが
        # 30.6%+30.6% に割れ、38.8% の PASS に負ける）ため、等価キーで訪問数を合算した
        # グループの argmax(N)（`_merge_root_stats` は n 降順＝先頭グループ）で選ぶ。
        # 訪問合算は**行動の同一性**の話であり補償層ではない＝必ず残す。旧 LCB 乗り換え
        # （二重ゲート `_select_root_group`）は純正AZ化（2026-08-25）で削除。
        stats = getattr(mcts, "last_stats", None)
        groups = None
        if stats and stats.get("legal"):
            groups = _merge_root_stats(manager, stats["legal"], stats["N"], stats["Q"])
            if groups:
                gi = 0
                # 生成の温度サンプリング: 序盤（turn <= temp_turns）は訪問分布から引く。
                # 選択の後段（箱コミット・record・原始手化）はサンプル結果に自然追従する。
                tt = self.temp_turns or 0
                if tt and int(getattr(manager, "turn_count", 0) or 0) <= tt:
                    ns = np.array([g["n"] for g in groups], dtype=np.float64)
                    if ns.sum() > 0:
                        gi = int(det_rng.choice(len(groups), p=ns / ns.sum()))
                move = stats["legal"][groups[gi]["rep"]]
        if move is None:
            move = legal[0] if legal else None
        # 棋譜ダンプ（観測専用）: 決定の同一性は箱レベル（原始手化前）の sig で記録する。
        # groups は選択と同じ集計（等価マージ後の訪問合算）＝方策ターゲットの生分布。
        if record is not None:
            record["kind"] = "main"
            record["sims"] = sims
            record["sig"] = move_sig(move) if move is not None else None
            # move_sig は don_k 非含有（コミットの残回数管理と同じ理由）だが、配分箱は
            # k 違い＝別候補（`_move_equiv_key` が don_k を含む）。ダンプ側で候補を
            # 区別できるよう k を並記する（None=DON_BOX 以外）。
            record["k"] = ((move.get("payload") or {}).get("don_k")
                           if isinstance(move, dict) else None)
            lg = stats["legal"] if stats else None
            record["groups"] = ([{"sig": move_sig(lg[g["rep"]]),
                                  "k": (lg[g["rep"]].get("payload") or {}).get("don_k"),
                                  "n": g["n"], "q": g["q"]}
                                 for g in groups] if groups else [])
        # 箱コミット（2026-08-26）: 木が選んだ箱の**自分側の残り手順**を確定する。
        #  - DON_BOX: 原始手順は payload から算術で確定＝カウントダウン形1要素（トレース不要。
        #    このdecideが先頭原始手を返すので残りは 総原始手数-1）
        #  - PLAY/ACTIVATE_MAIN: 後続の自分側効果対話列を評価と同じ解決規約で確定
        # 残ドン掘り（腕A・§__init__ residual_dig）: 木が TURN_END を選んだ時だけ差し替える。
        # 差し替えた PLAY はコミットしない（後続の対話は合法手が既定解決1手＝「払う」なので
        # 次 decide が機械的に払う）。
        dig_override = False
        if (self.residual_dig and move is not None and move.get("action_type") == "TURN_END"
                and not in_battle(manager) and not in_dialog(manager)):
            alt = self._residual_dig_move(manager, player)
            if alt is not None:
                move = alt
                dig_override = True
        # 残り起動（腕A2）: 掘りと同じ発火点（木が TURN_END を選んだ時）。起動を返したら
        # 続く付与対話は `_residual_attach_move` が方針で解く（フラグ）。コミットはしない。
        if (not dig_override and self.residual_activate and move is not None
                and move.get("action_type") == "TURN_END"
                and not in_battle(manager) and not in_dialog(manager)):
            alt = self._residual_activate_move(manager, player)
            if alt is not None:
                move = alt
                dig_override = True            # コミットを止める（同じ経路）
                self._resact_pending = True
        if use_commit and move is not None and not dig_override:
            at = move.get("action_type")
            if at == "DON_BOX":
                p = move.get("payload") or {}
                total = int(p.get("don_k") or 0) + (1 if p.get("target_ids") else 0)
                if total > 1:
                    self._store_commit(manager, name,
                                       [("__box__", move_sig(move), total - 1)])
            elif at in ("PLAY", "ACTIVATE_MAIN"):
                self._commit_play_dialog(manager, name, det_rng, move)
        # ドン箱（探索内部のマクロ手）は実対局へは先頭原始手 ATTACH_DON で出す＝
        # 記録/再生/API の行動空間を変えない（cpu_don_box_plan §2.1。コミット無しの
        # 箱は次 decide で箱候補が再計算され計画の続行/変更を選び直す）。
        move = cpu_ai.don_box_first_primitive(move)
        if trace is not None:
            try:
                _fill_trace(trace, manager, player, move, getattr(mcts, "last_stats", None))
            except Exception:
                pass   # 分析失敗で対局を止めない
        return move

    def _residual_activate_move(self, manager, player):
        """残り起動の差し替え手（腕A2）: 合法な ACTIVATE_MAIN のうちリーダー自身の起動で、
        その効果がドン追加の構造語を持ち、ドンデッキに残があれば返す（無ければ None）。
        合法手にあること＝未使用・条件成立・コスト充足（gamestate `_has_activatable_main`）。"""
        try:
            leader = getattr(player, "leader", None)
            if leader is None:
                return None
            if not _leader_has_don_ramp(getattr(leader, "master", None)):
                return None
            # ドンデッキ残の条件は**付けない**（2026-09-02 実測）: エネルのドンデッキは6枚で
            # turn3〜4 で尽きる。以後の起動は「追加」が空振りでも「レストのドン4枚までを
            # キャラに付与」が効く（払ったドンを +4000 に変える）＝実プレイの主用途。
            # 合法手にあること＝inert でない（gamestate `_has_activatable_main`）。
            # don_deck は events に残し、集計で「追加あり／付与のみ」を分ける。
            for mv in self.game.legal_actions(manager) or ():
                if mv.get("action_type") != "ACTIVATE_MAIN":
                    continue
                if (mv.get("payload") or {}).get("uuid") != getattr(leader, "uuid", None):
                    continue
                self.residual_activate_events.append({
                    "kind": "activate",
                    "turn": int(getattr(manager, "turn_count", 0) or 0),
                    "player": getattr(player, "name", None),
                    "leader": getattr(getattr(leader, "master", None), "card_id", None),
                    "don_active": len(getattr(player, "don_active", ()) or ()),
                    "don_rested": len(getattr(player, "don_rested", ()) or ()),
                    "don_deck": len(getattr(player, "don_deck", ()) or ()),
                    "field": len(getattr(player, "field", ()) or ()),
                    "policy": self.residual_activate,
                })
                return mv
        except Exception:
            return None
        return None

    def _residual_attach_move(self, manager, player):
        """起動で開いた**自分のキャラへの付与対話**を方針で解く。候補が自分の場のキャラで
        なければ None（通常の窓処理へ）。選んだ対象の RESOLVE 手を adapter の列挙から探し、
        無ければ既定 payload に selected_uuids を差し替えて作る。"""
        try:
            req = manager.get_pending_request() or {}
            sel = list(req.get("selectable_uuids") or [])
            if not sel:
                return None
            by_uuid = {getattr(c, "uuid", None): c for c in (getattr(player, "field", ()) or ())}
            if not all(u in by_uuid for u in sel):
                return None                          # 自分のキャラ以外が混ざる対話は触らない
            turn = int(getattr(manager, "turn_count", 0) or 0)
            cands = []
            for u in sel:
                c = by_uuid[u]
                try:
                    pw = int(c.get_power(True))
                except Exception:
                    pw = int(getattr(getattr(c, "master", None), "power", 0) or 0)
                can_atk = (turn > 2 and not getattr(c, "is_rest", False)
                           and not (getattr(c, "is_newly_played", False)
                                    and not c.has_keyword("速攻")))
                cands.append((u, pw, bool(can_atk)))
            target = _pick_attach_target(cands, self.residual_activate)
            if target is None:
                return None
            chosen = None
            for mv in self.game.legal_actions(manager) or ():
                p = mv.get("payload") or {}
                if (mv.get("action_type") == "RESOLVE_EFFECT_SELECTION"
                        and list(p.get("selected_uuids") or []) == [target]):
                    chosen = mv
                    break
            if chosen is None:
                payload = dict(manager.default_interaction_payload() or {})
                payload["selected_uuids"] = [target]
                payload["accepted"] = True
                chosen = {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                          "payload": payload}
            tc = by_uuid[target]
            self.residual_activate_events.append({
                "kind": "attach",
                "turn": turn,
                "player": getattr(player, "name", None),
                "card": getattr(getattr(tc, "master", None), "card_id", None),
                "power": next(pw for u, pw, _ in cands if u == target),
                "can_attack": next(ca for u, _, ca in cands if u == target),
                "n_cands": len(cands),
                "policy": self.residual_activate,
            })
            return chosen
        except Exception:
            return None

    def _residual_dig_move(self, manager, player):
        """残ドン掘りの差し替え手（腕A）: 合法な PLAY のうち `_is_dig_card` を満たす手札を
        先頭から1枚返す（無ければ None）。場の空き・アクティブドンの有無は合法手列挙が
        担保する（コスト充足は列挙側、場の上限はここで見る）。発火を events に記録する。"""
        try:
            if len(getattr(player, "don_active", ()) or ()) < 1:
                return None
            if len(getattr(player, "field", ()) or ()) >= 5:
                return None
            by_uuid = {getattr(c, "uuid", None): c for c in (getattr(player, "hand", ()) or ())}
            for mv in self.game.legal_actions(manager) or ():
                if mv.get("action_type") != "PLAY":
                    continue
                c = by_uuid.get((mv.get("payload") or {}).get("uuid"))
                master = getattr(c, "master", None)
                if c is None or not _is_dig_card(master):
                    continue
                leader = getattr(getattr(player, "leader", None), "master", None)
                self.residual_dig_events.append({
                    "turn": int(getattr(manager, "turn_count", 0) or 0),
                    "player": getattr(player, "name", None),
                    "card": getattr(master, "card_id", None),
                    "leader": getattr(leader, "card_id", None),
                    "don_active": len(getattr(player, "don_active", ()) or ()),
                    "don_rested": len(getattr(player, "don_rested", ()) or ()),
                    "don_deck": len(getattr(player, "don_deck", ()) or ()),
                    "hand": len(getattr(player, "hand", ()) or ()),
                    "field": len(getattr(player, "field", ()) or ()),
                })
                return mv
        except Exception:
            return None
        return None


DON_RAMP_MARK = "ドン!!デッキから"     # リーダー効果の「ドンデッキからドンを追加」構造語（‼は!!へ正規化）


def _leader_has_don_ramp(master) -> bool:
    """リーダーの起動効果がドンデッキからドンを追加するか（テキスト構造語・pure）。

    パーサは紫エネル OP15-058 の起動効果を空（effect ops []）で返すため op 構造では判定できず、
    能力の raw_text の構造語で見る（`dig_cf_breakdown.leader_has_don_ramp` と同じ規約）。"""
    try:
        if master is None:
            return False
        blob = " ".join((getattr(ab, "raw_text", "") or "")
                        for ab in (getattr(master, "abilities", ()) or ()))
        if not blob:
            blob = getattr(master, "effect_text", "") or ""
        return DON_RAMP_MARK in blob.replace("‼", "!!")
    except Exception:
        return False


def _pick_attach_target(cands, policy):
    """付与先の方針（pure）。cands=[(uuid, power, can_attack_this_turn)]。

    "low": このターン攻撃できるキャラのうち最低パワー（4枚付与で攻撃本数を増やす＝実プレイの
           筋・ユーザ 2026-09-02）。攻撃できるキャラが無ければ全体の最低パワー。
    "high": 最高パワー。同点は uuid 順で決定論。候補が無ければ None。"""
    if not cands:
        return None
    if policy == "high":
        return sorted(cands, key=lambda t: (-t[1], t[0]))[0][0]
    pool = [t for t in cands if t[2]] or list(cands)
    return sorted(pool, key=lambda t: (t[1], t[0]))[0][0]


def _is_dig_card(master) -> bool:
    """「登場時にドンを戻してカードを引く」コスト1キャラか（構造判定・カードID非依存・pure）。

    条件: CHARACTER・cost==1・ON_PLAY 能力の効果に DRAW があり、その能力のコストに
    RETURN_DON（ドン‼-X）がある。サトリ OP15-066・シュラ OP15-067 が該当。速攻や他能力の
    有無は見ない（腕Aの規則は「捨てるはずのドンで掘る」だけ＝順序の妙は探索の仕事）。"""
    try:
        from opcg_sim.src.models.effect_types import ActionType, TriggerType
        if master is None or getattr(getattr(master, "type", None), "name", None) != "CHARACTER":
            return False
        if getattr(master, "cost", None) != 1:
            return False
        for ab in (getattr(master, "abilities", ()) or ()):
            if getattr(ab, "trigger", None) is not TriggerType.ON_PLAY:
                continue
            if not _tree_has(getattr(ab, "effect", None), ActionType.DRAW):
                continue
            if _tree_has(getattr(ab, "cost", None), ActionType.RETURN_DON):
                return True
        return False
    except Exception:
        return False


def _tree_has(node, atype) -> bool:
    """効果木（GameAction / Sequence / Branch / Choice / list）に atype の action があるか（pure）。"""
    if node is None:
        return False
    if getattr(node, "type", None) is atype:
        return True
    # 効果木の子属性は n_eff_feat._walk と同じ集合（GameAction.sub_effect／Sequence.children・
    # effects／Choice.options／Branch.branches・then・else_effect・on_true・on_false）。
    for attr in ("sub_effect", "children", "effects", "options", "branches",
                 "then", "else_effect", "on_true", "on_false"):
        sub = getattr(node, attr, None)
        if isinstance(sub, (list, tuple)):
            if any(_tree_has(s, atype) for s in sub):
                return True
        elif sub is not None and _tree_has(sub, atype):
            return True
    if isinstance(node, (list, tuple)):
        return any(_tree_has(s, atype) for s in node)
    return False


_DEFAULT_ENGINE: Optional[LearnedEngine] = None


def _default_engine() -> LearnedEngine:
    """本番既定 CPU（出荷 Gen2）のプロセス共有シングルトンエンジン。"""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = LearnedEngine()
    return _DEFAULT_ENGINE


def _lazy_init():
    """後方互換のウォームアップ（既定エンジンを1回ロード）。perf 計測等が初回ロードを計測から外すのに使う。"""
    _default_engine()


def decide_learned(manager, player, sims: int = SERVE_SIMS, c_puct: float = C_PUCT,
                   rng=None, trace=None) -> Optional[Dict[str, Any]]:
    """学習型CPUの1手決定（本番既定 CPU 経路）。返り値は `decide_guarded` 互換（move 辞書 or None）。

    出荷 Gen2 のシングルトンエンジンへ委譲する薄いラッパ＝A3 のリファクタで**挙動不変**。
    `trace`（dict）が渡された時（cpu_trace ON）は、その手の分析を書き込む＝変な手の検証用ログ:
      chosen（選んだ手）/ value（選手の行動価値Q）/ candidates（訪問上位・visit%・Q）/
      l1_move（独立評価器L1の推奨手）/ l1_disagrees（L1と食い違うか）。
    分析は挙動に影響しない（例外は握り潰し、手は必ず返す）。
    """
    return _default_engine().decide(manager, player, sims=sims, c_puct=c_puct, rng=rng, trace=trace)


def _merge_root_stats(manager, legal, N, Q):
    """ルート合法手を挙動等価キー（`cpu_ai._move_equiv_key`）でグループ化し訪問数を合算する。

    返り値: [{"rep": 代表index(グループ内の**列挙順先頭**), "idxs": [...], "n": N合算, "q": N加重平均Q}]
    を n 降順（同数は legal 列挙順＝安定）で。等価手が無い局面では全グループが単独＝
    先頭グループの rep が従来の argmax(N) と一致し**挙動不変**。

    等価判定は card_id 基準＝リプレイ逆写像（`replay_runner._key`）と同じ同一視。場の複製
    （同名キャラで付与ドン数が違う等）は厳密には非等価だが、その残差はリプレイ側と同じ
    許容（R0 §5）に揃える。**代表は列挙順先頭**（2026-07-30）: 旧実装のグループ内 N 最大は
    探索ノイズで割れた訪問数の多い側＝card_id 記述子から再現不能で、録画時に2枚目の複製へ
    ATTACH_DON した手を再生（`resolve_recorded_action`＝記述子一致の先頭）が1枚目に写像し、
    以後の盤面が無音で分岐する実害が出た（seed9100 round-trip 実測）。等価前提の下で
    代表の選び方は挙動中立＝再現可能な先頭に固定する。
    """
    from opcg_sim.src.core import cpu_ai
    order, groups = [], {}
    for i, mv in enumerate(legal):
        k = cpu_ai._move_equiv_key(manager, mv)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    out = []
    for k in order:
        idxs = groups[k]
        n = float(sum(float(N[i]) for i in idxs))
        q = (sum(float(N[i]) * float(Q[i]) for i in idxs) / n) if n > 0 else 0.0
        rep = idxs[0]   # 列挙順先頭＝リプレイ逆写像と同じ実体（旧: N最大＝再現不能）
        out.append({"rep": rep, "idxs": idxs, "n": n, "q": q})
    out.sort(key=lambda g: -g["n"])   # sort は安定＝同数なら列挙順を保つ
    return out


def _fill_trace(trace, manager, player, chosen, stats):
    """トレース dict に learned の意思決定分析を書き込む（cpu_trace 時のみ呼ばれる）。"""
    from opcg_sim.src.core import cpu_ai
    import random as _random
    trace["difficulty"] = "learned"
    trace["turn"] = getattr(manager, "turn_count", None)
    trace["chosen"] = cpu_ai._describe_move(manager, chosen) if chosen else None
    # 対話種別（SEARCH_AND_SELECT / ARRANGE_DECK / CONFIRM_OPTIONAL 等）。無いと
    # 「ライフ追加の選択」か「底送りの順番」かがトレースから読めない。
    pend = manager.get_pending_request(with_request_id=False) or {}  # action だけ読む＝request_id 不要
    if pend.get("action"):
        trace["dialog"] = pend.get("action")
    # ① 自分の探索の内訳（等価手マージ後の訪問上位・visit%・行動価値Q）。decide の選択と
    #    同じ集計（`_merge_root_stats`）で出す＝「分裂した同名手が別行に出て PASS に負けて
    #    見える」ログ上の錯覚も消す。copies>1 は複製がマージされた印。
    if stats and stats.get("legal"):
        legal, N, Q = stats["legal"], stats["N"], stats["Q"]
        tot = float(N.sum()) or 1.0
        groups = _merge_root_stats(manager, legal, N, Q)
        trace["candidates"] = [{
            "move": cpu_ai._describe_move(manager, legal[g["rep"]]),
            "visit_pct": round(100.0 * g["n"] / tot, 1),
            "q": round(g["q"], 3),
            **({"copies": len(g["idxs"])} if len(g["idxs"]) > 1 else {}),
        } for g in groups[:5]]
        # 選んだ手の Q（＝net が見込む行動価値・所属グループの加重平均）。
        for g in groups:
            if any(legal[i] is chosen for i in g["idxs"]):
                trace["value"] = round(g["q"], 3)
                break
        # 手の監査（段1）が読む2つの信号。どちらも観測専用で選択には使わない。
        #  q_gap: 最良グループの Q と選んだ手の Q の差＝「箱の出口評価で見た損」。ほぼ 0 なら
        #         **迷っている**（＝反実仮想で測る価値がある容疑者）。
        #  policy_rank: 事前分布 P で並べた順位（1=policy の第一候補）。順位が低い手を探索が
        #         選んだ点は「policy が見落としている」か「探索が間違えた」かの分岐点。
        qs = sorted((g["q"] for g in groups), reverse=True)
        if qs and "value" in trace:
            # q_gap: 打った手が最良 Q からどれだけ下か（読み出しが Q 最良を選ばなかった量）。
            trace["q_gap"] = round(qs[0] - trace["value"], 3)
        if len(qs) >= 2:
            # q_margin: 1位と2位の差＝**迷いの深さ**。ほぼ 0 なら「どちらでもよい」と読んで
            # いる＝反実仮想で実際に差があるかを測る価値がある（q_gap は CPU がほぼ常に
            # 最良 Q を選ぶため中央値 0 で、迷いの指標にはならない）。
            trace["q_margin"] = round(qs[0] - qs[1], 3)
        P = stats.get("P")
        if P is not None:
            ranked = sorted(groups, key=lambda g: -sum(float(P[i]) for i in g["idxs"]))
            for r, g in enumerate(ranked, 1):
                if any(legal[i] is chosen for i in g["idxs"]):
                    trace["policy_rank"] = r
                    trace["policy_top"] = cpu_ai._describe_move(manager, legal[ranked[0]["rep"]])
                    break
    # ② 独立評価器 L1 の第二意見（分布外での net 系統誤差を拾う・evalは信じ過ぎない）。
    #    **観測はグローバル乱数を消費しない**（L1 は PIMC 等で global random を引きうる）。
    #    消費すると「トレースを採ると対局が変わる」＝観測が対象を変えてしまい、同じ seed の
    #    再生で決定点を復元できない（手の監査の段2は seed+決定番号で局面を復元する）。
    _rand_state = _random.getstate()
    try:
        clone = manager.clone()
        cp = clone.p1 if clone.p1.name == player.name else clone.p2
        l1 = cpu_ai.decide_guarded(clone, cp, "hard", _random.Random(0), {}, pimc_worlds=1)
        trace["l1_move"] = cpu_ai._describe_move(clone, l1) if l1 else None
        trace["l1_disagrees"] = bool(l1 and chosen and
                                     l1.get("action_type") != chosen.get("action_type"))
    except Exception:
        pass
    finally:
        _random.setstate(_rand_state)
