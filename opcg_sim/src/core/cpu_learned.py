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
import math
import os
import weakref
from typing import Any, Dict, Optional

import numpy as np

from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned.value_net import ValueNet
from opcg_sim.src.learned.policy import PolicyScorer, state_context
from opcg_sim.src.learned.action import legal_action_matrix
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.learned import config as CFG
from opcg_sim.src.learned.config import (
    C_PUCT, SERVE_SIMS, SERVE_DIRICHLET_EPS,
    SERVE_ROOT_SWITCH_MIN_FRAC, SERVE_ROOT_SWITCH_MIN_GAP, SERVE_STICKY_WORLD,
    AUX_TIE_DECAY, AUX_SAT_START, TERM_FLOOR, V4_TURNS_SCALE)
from opcg_sim.src.learned.mcts import (   # make/unmake版（唯一の探索実装。旧clone版は削除済み）
    TreeMCTS, in_battle, resolved_branch_values)
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
_DEFAULT_VALUE = os.path.join(_MODELS, "gen12_value.npz")
_DEFAULT_POLICY = os.path.join(_MODELS, "gen12_policy.npz")

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


def _aux_tie_scale(v: float, t_hat: float,
                   decay: float = AUX_TIE_DECAY, sat_start: float = AUX_SAT_START,
                   floor: float = TERM_FLOOR) -> float:
    """aux 粘り項（config.SERVE_AUX_TIEBREAK・v5 §4-1）: 飽和域の葉価値を予測残りターン t̂ で減衰する。

    v' = v · max(floor, 1 − decay·t̂·sat),  sat = clip((|v|−sat_start)/(1−sat_start), 0, 1)。
    終局の深さ減衰（TERM_DECAY・「速い勝ち＞遅い勝ち／遅い負け＞速い負け」）を「終局に届かない
    飽和した葉」へ拡張＝敗勢では本当に延命する手（t̂ が伸びる）を、優勢では速い勝ち（t̂ が短い）を
    選好する。非飽和域（|v| < sat_start）は sat=0 で恒等＝中間域の較正は不変。純関数（テスト対象）。"""
    a = abs(v)
    if a <= sat_start:
        return v
    sat = min((a - sat_start) / (1.0 - sat_start), 1.0)
    return v * max(floor, 1.0 - decay * max(t_hat, 0.0) * sat)


def _value_fn(vnet, vocab, enc_version=1, aux_tiebreak=None):
    """葉価値関数。aux_tiebreak=None は config.SERVE_AUX_TIEBREAK に従う（A/B 用に明示上書き可）。"""
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab, version=enc_version)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        use_aux = CFG.SERVE_AUX_TIEBREAK if aux_tiebreak is None else aux_tiebreak
        if not use_aux:
            return float(vnet.predict(batch)[0])
        pred, aux = vnet.predict_with_aux(batch)   # forward 1回を共有（二重計算しない）
        t_hat = float(aux[0]) * V4_TURNS_SCALE     # 正規化残りターン → ターン数
        return _aux_tie_scale(float(pred[0]), t_hat)
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
                 vocab=None, game=None, aux_tiebreak: Optional[bool] = None,
                 sims: Optional[int] = None, c_puct: Optional[float] = None,
                 root_frac: Optional[float] = None, root_gap: Optional[float] = None,
                 battle_readout: Optional[bool] = None, quiesce: Optional[bool] = None,
                 box_battle: Optional[bool] = None, turn_quiesce: Optional[bool] = None,
                 plan_readout: Optional[bool] = None):
        if vocab is None or game is None:
            svocab, sgame = _shared_vocab_game()
            vocab = vocab if vocab is not None else svocab
            game = game if game is not None else sgame
        self.vocab = vocab
        self.game = game
        # aux 粘り項のエンジン別上書き（None=config.SERVE_AUX_TIEBREAK に従う）。ON/OFF を
        # 同一プロセスで対戦させる A/B（net-vs-net arena）用＝本番既定は None。
        self.aux_tiebreak = aux_tiebreak
        # 探索つまみのエンジン別上書き（None=既定に従う・aux_tiebreak と同じ A/B 用の seam）。
        # **設定時は decide の呼び出し引数より優先**する＝席ごとに探索設定を変えた net-vs-net
        # （`search_config_probe.py`）で、ハーネス側が渡す sims を上書きできるようにするため。
        self.sims = sims
        self.c_puct = c_puct
        self.root_frac = root_frac      # root 読み出しの乗り換え条件（訪問比）
        self.root_gap = root_gap        # 同（Q 差・inf で従来の argmax(N)）
        # 戦闘窓の読み出し（None=config.SERVE_BATTLE_READOUT に従う・A/B 用の seam）。
        self.battle_readout = battle_readout
        # 静止探索（None=config.SERVE_QUIESCE に従う）。**同一プロセスで席ごとに機構を変える**
        # ための seam＝「新機構の候補 vs 現行本番」を公平に1回で測る（グローバル定数を書き換える
        # 測り方だと両席に同時に効いてしまい、機構とネットの寄与が分離できない）。
        self.quiesce = quiesce
        # 木の中の箱化（None=config.TREE_BOX_BATTLE に従う・同上の席別 seam）。
        self.box_battle = box_battle
        # ターン静止（None=config.SERVE_TURN_QUIESCE に従う・v37 の席別 seam）。
        self.turn_quiesce = turn_quiesce
        # プラン読み出し（None=config.SERVE_PLAN_READOUT に従う・v37② の席別 seam）。
        self.plan_readout = plan_readout
        # ターンプランのキャッシュ {(id(manager), turn, name, world_seed): steps or None}。
        # world_seed 込みのキー＝sticky 世界線と同じ寿命（外部が _world_seeds をリセットして
        # 新しい世界を引けばプランも立て直す。ゲート計測の seed 独立性がこれで保たれる）。
        self._turn_plans: Dict[Any, Any] = {}
        # ターン内 sticky 世界線の seed キャッシュ {(id(manager), turn, player): (weakref, seed)}（§_world_rng）。
        self._world_seeds: Dict[Any, Any] = {}
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

    def _battle_window_choice(self, manager, name, det_rng):
        """戦闘窓の読み出し（`SERVE_BATTLE_READOUT`）: 出口盤面の value で選ぶ。

        **戦闘を1つの箱として畳む**（ユーザ整理 2026-08-05）: カウンター/ブロッカー選択は
        「どの出口（解決後の盤面・手札・ライフ）になるか」で決まる局所判断で、箱の外の深い
        未来まで平均した root Q は判断を薄める（v35 実測: 出口評価は防御3類型を全て正しく
        順序づけるのに、探索後 Q は木の68%を占める『次の自ターン』の通常盤面＝旧レートに
        引き戻されて逆転する）。判断するのは葉評価と同じ value ネット自身であり、別系統の
        防御ロジックではない。

        世界線は探索と同じ決定化（PIMC・sticky）を使い、返す手は決定化クローン上の合法手
        （`TreeMCTS.run` と同一契約）。返り値は (move, root統計, 評価に使った盤面)＝
        トレースは呼び出し側で埋める。選べないときは (None, None, None) で従来の探索へ委ねる。
        """
        mgr = self.game.determinize(manager, name, det_rng)
        legal = self.game.legal_actions(mgr)
        if not legal:
            return None, None, None
        if len(legal) == 1:
            return legal[0], None, None
        vals = resolved_branch_values(
            self.game, mgr, name, legal,
            _value_fn(self.vnet, self.vocab, self.enc_version, aux_tiebreak=self.aux_tiebreak),
            _priors_fn(self.pnet, self.vocab, self.enc_version))
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

    def _plan_step(self, manager, name, det_rng, world_seed):
        """プラン読み出し（`SERVE_PLAN_READOUT`・v37②）: ターンに1回プランを立てて
        （K世界期待値・`plan.select_plan`）、以後はその手を1つずつ返す。

        次の手が実盤面で非合法になったら（想定外の応手＝計画が割れた）**その場で1回だけ
        再計画**する。プランが尽きたら TURN_END を返してターンを閉じる（プラン評価は
        「この列を打って閉じたターン末」の値なので、閉じるまでがプランの一部）。
        立案失敗（候補ゼロ等）は None を記憶し、このターンは従来の探索に委ねる。"""
        from opcg_sim.src.learned import plan as PL
        key = (id(manager), int(getattr(manager, "turn_count", 0) or 0), name, world_seed)
        if key not in self._turn_plans:
            if len(self._turn_plans) >= 64:
                for k in list(self._turn_plans)[:32]:
                    del self._turn_plans[k]
            vf = _value_fn(self.vnet, self.vocab, self.enc_version,
                           aux_tiebreak=self.aux_tiebreak)
            pf = _priors_fn(self.pnet, self.vocab, self.enc_version)
            steps, _diag = PL.select_plan(self.game, manager, name, vf, pf, det_rng)
            self._turn_plans[key] = list(steps) if steps else None
        steps = self._turn_plans[key]
        if steps is None:
            return None
        for attempt in (0, 1):
            legal = self.game.legal_actions(manager)
            while steps:
                mv = PL._find_move(legal, steps[0])
                if mv is not None:
                    steps.pop(0)
                    return mv
                steps.pop(0)       # 対象消滅などで非合法になった手は縮退（プラン評価と同じ規約）
            if not steps and attempt == 0:
                # プランが尽きた: ターンを閉じる（これもプランの一部）
                for cand in legal:
                    try:
                        d = cpu_ai._describe_move(manager, cand) or {}
                    except Exception:
                        d = {}
                    if d.get("action_type") == "TURN_END":
                        return cand
                # TURN_END が無い（対話中など）＝計画が割れた → 1回だけ再計画
                vf = _value_fn(self.vnet, self.vocab, self.enc_version,
                               aux_tiebreak=self.aux_tiebreak)
                pf = _priors_fn(self.pnet, self.vocab, self.enc_version)
                new_steps, _diag = PL.select_plan(self.game, manager, name, vf, pf, det_rng)
                if not new_steps:
                    self._turn_plans[key] = None
                    return None
                steps = self._turn_plans[key] = list(new_steps)
        return None

    def decide(self, manager, player, sims: int = SERVE_SIMS, c_puct: float = C_PUCT,
               rng=None, trace=None) -> Optional[Dict[str, Any]]:
        """このエンジンのネットで 1 手決定する（`decide_learned` と同一契約・同一探索）。"""
        name = player.name
        if self.sims is not None:
            sims = self.sims           # エンジン別上書き（未設定=None で従来どおり引数/既定）
        if self.c_puct is not None:
            c_puct = self.c_puct
        # numpy rng の種を **global random** から引く＝リプレイ種（routers が cpu_trace 時に random.seed）で
        # learned 対局も決定論再生できる。通常対局は global random 未 seed（プロセス由来）＝実質ランダム。
        if not isinstance(rng, np.random.Generator):
            import random as _random
            rng = np.random.default_rng(_random.getrandbits(64))
        det_rng = self._world_rng(manager, name, rng) if SERVE_STICKY_WORLD else rng
        # 戦闘窓は箱として畳んで出口評価で選ぶ（config.SERVE_BATTLE_READOUT）。メインフェーズ
        # （バトルをするか・どこを殴るか）は下の full-tree のまま＝変更しない。
        use_battle = (CFG.SERVE_BATTLE_READOUT if self.battle_readout is None
                      else self.battle_readout)
        if use_battle and in_battle(manager):
            move, stats, ev_mgr = self._battle_window_choice(manager, name, det_rng)
            if move is not None:
                if trace is not None:
                    try:
                        _fill_trace(trace, ev_mgr if ev_mgr is not None else manager,
                                    player, move, stats)
                        trace["readout"] = "battle_resolved"
                    except Exception:
                        pass   # 分析失敗で対局を止めない
                return move
        # プラン読み出し（v37②）: 自ターンのメイン判断は「プラン×K世界の期待値」で決める。
        # 対象は自ターン所有かつ非戦闘の判断のみ（防御窓は上の箱読み出し・相手ターンは対象外）。
        use_plan = (CFG.SERVE_PLAN_READOUT if self.plan_readout is None
                    else self.plan_readout)
        if use_plan and not in_battle(manager) and \
                getattr(getattr(manager, "turn_player", None), "name", None) == name:
            wkey = (id(manager), int(getattr(manager, "turn_count", 0) or 0), name)
            hit = self._world_seeds.get(wkey)
            wseed = hit[1] if hit is not None else None
            move = self._plan_step(manager, name, det_rng, wseed)
            if move is not None:
                if trace is not None:
                    try:
                        _fill_trace(trace, manager, player, move, None)
                        trace["readout"] = "turn_plan"
                    except Exception:
                        pass
                return move
        mcts = TreeMCTS(self.game, value_fn=_value_fn(self.vnet, self.vocab, self.enc_version,
                                                      aux_tiebreak=self.aux_tiebreak),
                        priors_fn=_priors_fn(self.pnet, self.vocab, self.enc_version),
                        c_puct=c_puct, n_sims=sims, dirichlet_eps=SERVE_DIRICHLET_EPS,
                        determinize_fn=lambda s, r: self.game.determinize(s, name, r), rng=det_rng,
                        quiesce=self.quiesce, box_battle=self.box_battle,
                        turn_quiesce=self.turn_quiesce)
        move, _, legal = mcts.run(manager)
        # 同名カードの別実体（手札の複製等）は探索木で別 edge になり訪問数が分裂する。
        # 素の argmax(N) は分裂した等価手を系統的に不利にする（例: EB03-053×2 のカウンターが
        # 30.6%+30.6% に割れ、38.8% の PASS に負ける）ため、等価キーで訪問数を合算した
        # グループから選ぶ。さらに読み出しは argmax(N) でなく LCB 乗り換え
        # （`_select_root_group`）＝訪問が貼り付いたまま Q で劣後した手を採らない。
        # 探索（TreeMCTS）自体は不変＝ルートの読み出しのみ補正。
        stats = getattr(mcts, "last_stats", None)
        if stats and stats.get("legal"):
            groups = _merge_root_stats(manager, stats["legal"], stats["N"], stats["Q"])
            if groups:
                kw = {}
                if self.root_frac is not None:
                    kw["min_frac"] = self.root_frac
                if self.root_gap is not None:
                    kw["min_gap"] = self.root_gap
                move = stats["legal"][_select_root_group(groups, **kw)["rep"]]
        if move is None:
            move = legal[0] if legal else None
        if trace is not None:
            try:
                _fill_trace(trace, manager, player, move, getattr(mcts, "last_stats", None))
            except Exception:
                pass   # 分析失敗で対局を止めない
        return move


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


def _select_root_group(groups, min_frac: float = SERVE_ROOT_SWITCH_MIN_FRAC,
                       min_gap: float = SERVE_ROOT_SWITCH_MIN_GAP):
    """root 読み出し: 最多訪問グループを基準に、二重ゲートを満たす代替の Q が上回れば乗り換える。

    素の argmax(N) は PUCT の訪問が prior／先行 Q に貼り付く性質上、探索後半に Q で逆転した
    代替を拾えない（g1@12: ATTACK 56%/q=-0.127 が ATTACH_DON 31%/q=-0.043 に選ばれる）。
    一方、低訪問の Q は PUCT の選択バイアスで**楽観方向に大きく歪む**（連続 decide の実測で
    +0.14〜+0.54・g2@20-23＝1/√n の悲観補正では不足）。そこで乗り換えは
      ① 訪問が競っている: n ≥ min_frac·n_top（浅い読みの楽観を除外）
      ② Q 差が明確:       q ≥ q_top + min_gap（同格ノイズでの乗り換えを除外）
    の両方を満たす代替に限る（該当複数なら最大 Q）。min_gap=inf で従来の argmax(N) に一致。
    較正は実対局2局×16人間マークへの回帰（mark_review2 §S1・`test_learned_root_readout.py`）。
    探索・トレース統計は不変＝読み出しのみ。

    `groups`: `_merge_root_stats` の返り値（n 降順・{"rep","idxs","n","q"}）。返り値は選んだグループ。
    """
    best = groups[0]
    if len(groups) == 1 or best["n"] <= 0 or not math.isfinite(min_gap):
        return best
    gate = max(1.0, min_frac * best["n"])
    bar = best["q"] + min_gap
    for g in groups[1:]:
        if g["n"] >= gate and g["q"] >= bar and g["q"] > best["q"]:
            best = g
            bar = best["q"]   # 以降はさらに高い Q のみ（該当複数なら最大 Q・n 降順で安定）
    return best


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
    # ② 独立評価器 L1 の第二意見（分布外での net 系統誤差を拾う・evalは信じ過ぎない）。
    try:
        clone = manager.clone()
        cp = clone.p1 if clone.p1.name == player.name else clone.p2
        l1 = cpu_ai.decide_guarded(clone, cp, "hard", _random.Random(0), {}, pimc_worlds=1)
        trace["l1_move"] = cpu_ai._describe_move(clone, l1) if l1 else None
        trace["l1_disagrees"] = bool(l1 and chosen and
                                     l1.get("action_type") != chosen.get("action_type"))
    except Exception:
        pass
