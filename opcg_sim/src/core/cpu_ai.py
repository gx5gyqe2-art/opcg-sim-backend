"""ルールモード CPU（AI）の意思決定エンジン（docs/SPEC.md §2.5.2）。

設計:
  - 合法手は `GameManager.get_legal_actions` を単一の真実源として用いる。
  - 各候補手を `GameManager.clone()` 上で適用し、`evaluate` で盤面優劣を採点して選ぶ。
    クローン上では自分側の効果対話を既定解決でドレインしてから採点する。
  - ステートレス（毎ステップ再計画）。ポーリング駆動でも desync に強い。

評価関数（J値理論ベース・docs/SPEC.md §2.5.2）:
  「J値 = 白の枚数 = デッキ残 + トラッシュ」を下げ、相手の J値を上げるゲーム、という
  Jin 氏「J値理論」に整合する形で盤面を採点する。J値を下げる = 黒（手札・ライフ・場・ステージ）
  にカードが多い状態なので、本評価は黒リソースの重み付き和を主軸に、理論の以下の機微を加える:
    - ライフの非線形価値（薄いほど 1 枚の限界価値が跳ね上がる＝45[J] ラインの危険）。
    - 手札のカウンター値（防御リソース＝相手の +1[J] をいなす力）。
    - 場のアクティブキャラ（＝将来のアタック＝相手の J値を上げる圧力）とブロッカー（最終防御）。
  KO・カウンター誘発・ハンデス等の「相手 +1[J]」は、相手側の枚数・パワーが下がることで自然に
  差分へ反映される（明示の J値項は黒リソースと相補で二重計上になるため置かない）。

難易度＝情報方針の 3 分化（API キー easy/normal/hard は維持し挙動を再定義・docs/SPEC.md §2.5.2）:
  easy(かんたん)  : 正直な 1-ply 貪欲（ミスなし）。評価は公開情報のみ（相手手札は枚数だけ）。
  normal(ふつう)  : 多 ply 先読み。公開情報のみ＋相手 min ノードは隠れ手札に依存する手
                    （手札からの登場・カウンター）を使わない保守モデルで応答（リーダー推測の土台・
                    §2.5.4 のテンプレ供給で想定手を補強）。
  hard(つよい)    : フルクローン多 ply 先読み（α-β ＋ ビーム）。相手手札も読む「最強」。
  いずれも `_search` の ply 割引付き winner 到達検出で最短リーサルを認識する。

公平性メモ: easy/normal は隠れ情報（相手手札の中身・裏向きライフ）を読まない（evaluate の
see_opp_hand=False ＋ 相手 min ノードの手札依存手を除外）＝チート防止。hard のみユーザ選択により
相手手札を読む別方針（docs/SPEC.md §2.2/§6 参照）。
"""
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import re

from ..models.enums import TriggerType
from . import journal
from .journal import JournaledList


def _env_int(name: str, default: int) -> int:
    """探索ノブを環境変数で上書き可能にする（Phase 1 切り分け実験＝horizon/beam 掃引用）。

    未設定・不正値は `default`＝従来挙動と完全同値（テスト/本番は env を立てない限り不変）。
    実験は OPCG_HARD_HORIZON 等を立てて別プロセスで掃引する（定数は import 時に確定）。
    """
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """float 版 `_env_int`（未設定・不正値は default＝従来挙動と完全同値）。"""
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

# ② make/unmake: 探索の 1-ply 採点（ビーム選別）を clone でなく「適用→採点→巻き戻し」で行い
# per-node の deepcopy を消す（探索コストの ~86%＝clone）。**非中断（resolver が parked でない
# 静止点）から適用する手にのみ適用**し、中断（複数段効果の途中）を再開する手は clone へフォールバック
# する（parked resolver 状態は現状 journaled 化が未完＝docs/SPEC.md §2.5.2 の boundary）。
# 探索結果（選ぶ手・評価値）は clone 方式と完全同一（内部最適化）＝tests/test_cpu_make_unmake.py で
# 等価を機械照合。万一の取りこぼし時に即無効化できるようモジュールフラグで保持する。
_USE_MAKE_UNMAKE = True

# ④ 着手順序の前回PV/killer 再利用（move ordering・docs/SPEC.md §2.5.3）。
# α-β は「良い手から先に試すほど早く β/α カットできる」＝探索ノードが減る。本最適化は探索木の各ノードで
# 「同じ ply で直近にカット（alpha>=beta）を起こした手＝killer」を覚えておき、**ビーム選別後の子集合の中で**
# その手を先頭へ寄せる（＝探索する子の集合は一切変えず、順序だけ入れ替える）。
# **挙動不変の理屈**: α-β のカットは値を変えない（刈られる枝は結果に影響し得ない）ので、予算が拘束しない
# フル探索では選ぶ手・深掘りスコアは順序に依らず**完全同値**（`tests/test_cpu_pv_order.py` で機械照合）。
# 予算拘束時のみ、カットで節約したノードが深さ（settle 回避）に回って結果が改善し得る＝強化（A/B で検証）。
# `_USE_MAKE_UNMAKE` と同じく**フラグで即時 OFF**（=従来の 1-ply スコア順）に戻せる内部最適化。
_USE_PV_ORDER = True
_KILLER_SLOTS = 2          # 各 ply で保持する killer 手の数（standard killer heuristic）

# ④粒度b: killer 表を**連続する decide 間で持ち越す**（`decide_guarded` が `mem["killers"]` を供給）。
# 配線は実装済みで挙動不変（reorder は集合不変＝予算非拘束なら値不変＝`tests/test_cpu_pv_order.py` で照合）だが、
# **実測で中立〜わずかに悪化**（中盤6局面・増量予算でノード -0.4%＝悪化／ply を 1 ずらす整流でも ±0%）。
# 原因＝killer は局面固有のヒントで、decide をまたぐと探索ルート（盤面）が変わり同 ply の局面が一致しない
# （単一探索内の兄弟部分木のような構造類似が無い）。よって**既定 OFF**＝粒度a のみ本番有効。将来 PV 継続
# （最善応手列を次手の同一局面ノードへ正しく対応付ける）で正の利得が出れば有効化を再検討する。
_USE_PV_CROSS_DECIDE = False

# 情報方針（フェア制約・docs/reports/cpu_strength_roadmap_20260622.md §0/§4 Phase -1）。
# 旧実装は decide で `see_opp_hand, opp_public_only = True, False` を**ハードコード**＝出荷 CPU がチート
# （相手手札/裏ライフを読む）だった。これを引数化し、**出荷デフォルトを fair に即切替**する（固定値撤廃）。
#   "fair"  = 相手手札の中身（カウンター値）を読まず、相手 min ノードでも隠れ手札依存手を使わない保守
#             モデル（see_opp_hand=False, opp_public_only=True）。＝出荷デフォルト。
#   "cheat" = 相手手札も読むフルクローン（see_opp_hand=True, opp_public_only=False）。旧 hard＝最強だが
#             不公平。凍結ベースライン/参考天井・診断用に明示指定で残す。
# 値は (see_opp_hand, opp_public_only) のタプルへ解決する。
_INFO_POLICIES = {"fair": (False, True), "cheat": (True, False)}
DEFAULT_INFO_POLICY = "fair"


def _resolve_info_policy(info_policy: str) -> Tuple[bool, bool]:
    try:
        return _INFO_POLICIES[info_policy]
    except KeyError:
        raise ValueError("unknown info_policy: %r (expected 'fair' or 'cheat')" % (info_policy,))




# 勝敗の決定値（L1・`cpu_eval_v2` が終端でこの符号を返す。探索の winner 割引もこの定数を使う）。
W_WIN = 1.0e9            # 勝敗

# J値（白＝デッキ残＋トラッシュ）の決定境界の立ち上げ枚数。デッキ切れ（J=0）でドロー不能＝敗北なので、
# 自デッキ残が危険域に入るほど非線形に減点する＝相手を削り切る／自滅ドローを避ける動機。L1（`cpu_eval_v2`）が
# この閾値を流用する（`DECK_DANGER - len(deck)` を敗北距離として加味）。
DECK_DANGER = 4          # この枚数以下から減点を立ち上げる

# 戦闘の閾値性: パワーは「最も硬い相手の防御（リーダー/場キャラ）を上回る」までが意味を持ち、
# それを超える分（過剰パワー・届かせる必要のないドン付与）はほぼ価値が無い。超過分はこの係数で減衰。
# これにより「パワーが届かない/過剰なドン付与」は静的にはほぼ無加点となり、実際に戦闘結果を変える
# 付与だけが多 ply 探索のライフ/KO 差分として価値化される。
W_POWER_OVERCAP = 0.1

_EPS = 1.0  # これ未満の改善ならターンを畳む（無限ループ防止＋無意味手の抑制）
# 「何もしない（ターンを畳む）」を一級の選択肢として常に比較する。行動はこの幅を超えて盤面を
# 改善する場合のみ採用し、無意味なキャラ展開・不利アタック・効かないドン付与を抑制する。
_ACT_MARGIN = 300.0
_DRAIN_LIMIT = 12        # クローン上で自分側対話を解決する最大回数

# 多 ply 先読みのパラメータ。clone（≈4-5ms）が支配的なので、1 手あたりのレイテンシ（≈1 秒）に
# 収まるよう、ルート手は 1-ply で事前選別し上位のみを深掘りする（選別＋均等予算で公平に採点）。
# 単ターン探索（docs/SPEC.md §2.5.2）。探索は「自分のターン1回」に限定し、葉の評価点を相手のターン開始
# （相手の MAIN_ACTION＝自分のターンが完全に解決した直後）に固定する。これで全候補が同じ静止点（同じ手番
# パリティ）で評価され、horizon／手番パリティによる「常に何かする」バイアスが消える（パスが不当に低く
# ならない）。自ターン内の自分のメイン手＋自分のアタックへの相手ブロック/カウンター応答は従来通り読む。
HARD_BEAM = _env_int("OPCG_HARD_BEAM", 3)  # 各ノードで展開する子の数（1-ply 評価上位 K・env 上書き可）
# E1（Phase3 ③・「〜されたら〜する」を厚く）: 相手 min ノード専用の広いビーム。max（自分）は最善手
# 1本に収束させたいので狭くてよいが、min（相手）の応手（ブロック/カウンター/除去応答/相手ターンの攻め手）
# は「相手が取りうる手」を多く残すほど contingency（相手がこう来たら、の枝）が厚くなり守り/攻めが堅実化
# する。max は従来どおり HARD_BEAM、min のみ HARD_OPP_BEAM に拡げる。計算増は HARD_PER_MOVE_BUDGET 内で
# 吸収（深さと横幅のトレードオフ）。A/B 自己対戦（asymmetric arena・席交互）で強さ×レイテンシ非退行を確認。
# 値決め（2026-06・hard 自己対戦 seed0〜）: beam=5 は深掘り clone 増で**1 秒目標を超過**（p95 ~1009ms /
# max ~1285ms / 1秒超あり）＝不採用。beam=4 は **1 秒以内に収まる**（p95 ~870ms / max ~978ms / 1秒超 0）。
HARD_OPP_BEAM = _env_int("OPCG_HARD_OPP_BEAM", 4)
HARD_ROOT_BEAM = _env_int("OPCG_HARD_ROOT_BEAM", 4)  # 深掘りするルート手の数（残りは 1-ply・env 上書き可）
# B-3（§2.5.3）: 1-ply ランクに関係なく深掘り集合へ**強制投入**する重要手クラス（ブロッカー設置・除去
# 候補・逆算リーサル/クロック手）の追加上限。ビーム拡幅（4→6-8）は置換表によるレイテンシ削減が前提
# （SPEC）のため本数を絞り、取りこぼし是正だけを先取りする（1 手あたり HARD_PER_MOVE_BUDGET の追加読み）。
HARD_FORCE_DEEPEN_CAP = 3
# 深掘り 1 手あたりのクローン上限（予算切れは settle で境界評価＝正しいのでここはレイテンシ予算）。
# 予算が**実深さの律速**: budget=150 では葉の到達深さが中央値 ~1 ターン・horizon=4 まで到達できる葉は
# 全体の ~4%（残りは予算切れ settle）と実測（hard 自己対戦・seed0）。一方で 1 手レイテンシは**思考手で
# mean ~210ms / 最大 ~465ms**（高速化の蓄積で当初想定 ~516ms より速い）＝1 秒目標に対して約 2 倍の余力。
# この余力を深さへ振るため budget を 150→300 に増量（**1 秒厳守の即効改善**・段階導入 Phase 1）。
# 実測スイープ（seed0・hard 自己対戦）の到達率/レイテンシ:
#   budget=150: 思考手 mean 210ms / 最大 465ms / 1秒超 0% / 4T到達 4.4%
#   budget=300: 思考手 mean 385ms / 最大 846ms / 1秒超 0% / 4T到達 17.6%（4 倍）  ← 採用
#   budget=500: 最大 1436ms / 1秒超 21%（1 秒を破るため不採用）
# さらなる深掘りは Phase 2（固定予算→壁時計デッドライン化＋_apply_passive_effects 最適化）で 1 秒以内
# のまま到達率を伸ばす。横展開を増やすより深さ予算が費用対効果が高い（settle は信頼度 0.9 で割引済み）。
HARD_PER_MOVE_BUDGET = _env_int("OPCG_HARD_PER_MOVE_BUDGET", 300)  # 深掘り1手あたり clone 上限（env 上書き可）

# Phase 4: 探索予算の per-decider 上書き（PIMC の予算按分＝K 世界で合計を一定に保つ用）。
# `set_budget_override(b)` で一時設定（評価アリーナ＝単一スレッドで安全）。
# None で env/既定（HARD_PER_MOVE_BUDGET）へ戻る＝本番/テストは未設定で従来同値。
_BUDGET_OVERRIDE = None


def set_budget_override(b):
    """深掘り予算を一時上書き（評価アリーナ用・None で既定へ）。"""
    global _BUDGET_OVERRIDE
    _BUDGET_OVERRIDE = None if b is None else max(1, int(b))


def _effective_budget() -> int:
    return _BUDGET_OVERRIDE if _BUDGET_OVERRIDE is not None else HARD_PER_MOVE_BUDGET


# 探索深さ／ply 上限の per-decide オーバーライド（L1 外の伸びしろ＝探索深さの A/B 用・set_*_override と同型）。
# 席別に horizon/max_ply を切替えて「より深く読む方」を同一ゲーム内ペアで測る（評価アリーナ＝単一スレッドで安全）。
# None で既定（HARD_HORIZON / HARD_MAX_PLY）へ戻る＝本番/テストは未設定で従来同値。
_HORIZON_OVERRIDE = None
_MAX_PLY_OVERRIDE = None


def set_search_override(horizon=None, max_ply=None):
    """探索 horizon／ply 上限を一時上書き（評価アリーナの深さA/B用・どちらも None で既定へ）。"""
    global _HORIZON_OVERRIDE, _MAX_PLY_OVERRIDE
    _HORIZON_OVERRIDE = None if horizon is None else max(1, int(horizon))
    _MAX_PLY_OVERRIDE = None if max_ply is None else max(1, int(max_ply))


def _effective_horizon() -> int:
    return _HORIZON_OVERRIDE if _HORIZON_OVERRIDE is not None else HARD_HORIZON


def _effective_max_ply() -> int:
    return _MAX_PLY_OVERRIDE if _MAX_PLY_OVERRIDE is not None else HARD_MAX_PLY


# Phase 4 本番配線: PIMC 既定世界数（OPCG_PIMC_WORLDS・既定 1＝休眠＝従来同値）。本番 Dockerfile で 4 を
# 設定して出荷 fair CPU を「PIMC K=4・予算按分」へ切替（OPCG_HARD_PER_MOVE_BUDGET=75 併用＝合計≈300＝
# 等倍計算量・1秒内・+53 Elo）。各 decide 系の pimc_worlds 既定値に使う＝env 未設定なら全経路で 1。
PIMC_WORLDS_DEFAULT = _env_int("OPCG_PIMC_WORLDS", 1)
HARD_DEPTH = 5             # ply 割引の基準（最短リーサル認識のテスト境界・winner 到達 ply の上限目安）
HARD_MAX_PLY = 52          # 総 ply の安全上限（horizon=4 で 4 ターン分の自由展開＋戦闘サブステップを賄う）
# 探索ホライズン（B2-lite・docs/SPEC.md §2.5.3）。深掘りで何ターン先まで読むか。horizon=4＝「自分のターン
# 完了→相手のターン→自分の次ターン→相手の次ターン→…」付近の静止点で評価＝相手の反撃＋自分の立て直し＋
# さらに相手の再反撃まで読む。②③B の高速化（~4.2x）で生じた余力を深さへ振り、horizon=3→4 に拡張。
# **A/B 自己対戦で検証**（horizon4 vs horizon3・both hard・席交互 60 局）: 35/60＝58.3%＝**+58 Elo**
# （両独立シード群とも >50%・戦術退行なし）＝深く読む方が強いことを実測で確認して採用（2026-06）。
# 横展開は重いので上位 K 手（HARD_ROOT_BEAM＋TURN_END）のみに適用し、非対象は採用しない（評価ホライズンの一貫性）。
HARD_HORIZON = _env_int("OPCG_HARD_HORIZON", 4)  # 探索ホライズン（何ターン先まで読むか・env 上書き可）
_SETTLE_LIMIT = 16         # 打ち切り時にターン境界へ整流する最大手数（戦闘サブステップ込み）

# 注（A2・2026-06）: 地平線越えの「毎ターン価値を生むエンジン能力」プレミアム（旧 W_RECUR_ENGINE／
# _recurring_engine／_RECUR_TRIGGERS）は、ablation A/B（フル vs 当該項なし・hard 42局）で勝率 0.452＝
# **正味マイナス（バイアス）**と判明したため撤去した。horizon=4 の探索が将来発動を結果盤面で既に拾えており、
# 静的プレミアムは二重計上で評価を歪めていた。経緯は docs/SPEC.md §2.5.3。


def _other(manager, name: str):
    return manager.p2 if manager.p1.name == name else manager.p1


def _player_by_name(manager, name: str):
    return manager.p1 if manager.p1.name == name else manager.p2


def _power_cap(opp) -> float:
    """相手側の最も硬い防御パワー（リーダー/場キャラのうち最大）。これを超える自パワーは戦闘で無駄。

    防御側のパワーは自分のターン中のみ乗る付与ドン!!分を含まない素の値で測る（is_my_turn=False）。
    アタックは「攻撃側パワー >= 防御側パワー」で連撃成立（リーダーへはライフ -1、レストキャラは KO）
    なので、相手の最硬防御を上回るだけのパワーが有効上限となる。
    """
    cap = 0.0
    units = ([opp.leader] if opp.leader is not None else []) + list(opp.field)
    for c in units:
        try:
            cap = max(cap, float(c.get_power(False)))
        except Exception:
            cap = max(cap, float(c.master.power or 0))
    return cap


def _effective_power(power: float, cap: float) -> float:
    """戦闘で意味を持つ有効パワー。上限 cap までは等価、超過分は強く減衰する（過剰パワーは無価値）。"""
    if power <= cap:
        return power
    return cap + (power - cap) * W_POWER_OVERCAP


# B-2（§2.5.3）: ドン!!付与の手生成を「意味ある配分」だけに絞る。
_DON_POWER = 1000.0   # アクティブドン!! 1 枚あたりのパワー上昇（自分のターン中のみ）
# 付与ドン条件【ドン!!×N】（半角!!／全角！！／‼／× の表記揺れ・前後スペースに耐性）。これを持つカードへの
# 付与は効果を開くため戦闘閾値に関わらず残す（過剰プルーニングで don 条件効果の起動を潰さない）。
_DON_COND_RE = re.compile(r'【\s*ドン\s*(?:!!|！！|‼)\s*[××xX]\s*\d+\s*】')


def _has_don_conditional(c) -> bool:
    """カードが付与ドン条件【ドン!!×N】の能力を持つか（テキスト判定・保守的）。"""
    m = getattr(c, "master", None)
    if m is None:
        return False
    return bool(_DON_COND_RE.search(getattr(m, "effect_text", "") or ""))


def _attach_don_meaningful(manager, actor_name: str, c) -> bool:
    """ドン!!付与の手が「意味ある配分」か（B-2・§2.5.3）。

    付与ドンのパワーは自分のターンのみ・付与先がこのターン実際に攻撃する体でなければ純損（アクティブドンを
    失うだけ）。意味があるのは:
      (A) 戦闘結果を変えられる: 付与先（このターン攻撃できる体）が現状では上回れない相手の防御パワー
          （リーダー/場キャラ）を、手持ちアクティブドンの範囲で**新たに上回れる**。既に最硬防御を上回る
          付与先（過剰=オーバーキャップ）や、全ドンを乗せても最低の未踏破防御に届かない付与先は無意味。
      (B) 付与ドン条件【ドン!!×N】を開ける: 付与で常在/起動効果が立つカードは戦闘閾値に関わらず残す。
    """
    actor = _player_by_name(manager, actor_name)
    budget = len(actor.don_active)
    if budget <= 0:
        return False
    # (B) 付与ドン条件を持つカードは保守的に残す（レスト/召喚酔いでも条件達成で効果が立ち得る）。
    if _has_don_conditional(c):
        return True
    # (A) 戦闘結果を変えうる付与のみ。付与先はこのターン攻撃できる体に限る（レスト/召喚酔いは今ターン出ない）。
    if getattr(c, "is_rest", False):
        return False
    if getattr(c, "is_newly_played", False) and not c.has_keyword("速攻"):
        return False
    try:
        p = float(c.get_power(True))
    except Exception:
        p = float(getattr(getattr(c, "master", None), "power", 0) or 0)
    reach_max = p + budget * _DON_POWER
    opp = _other(manager, actor_name)
    for u in ([opp.leader] if opp.leader is not None else []) + list(opp.field):
        try:
            tp = float(u.get_power(False))
        except Exception:
            tp = float(getattr(getattr(u, "master", None), "power", 0) or 0)
        if p < tp <= reach_max:   # 現状は上回れない（p<tp）が、付与で上回れる（tp<=reach_max）
            return True
    return False


def _prune_don_moves(manager, actor_name: str, moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ドン!!付与の手を「意味ある配分」だけに絞る（B-2・§2.5.3）。ATTACH_DON 以外は素通し。

    付与の手生成は付与先候補ごとに 1 手出るため、5000 未満/過剰への無意味な付与でビーム（HARD_BEAM=3）と
    探索予算を浪費していた。戦闘結果を変えうる付与＋付与ドン条件を開ける付与だけを残し、ビームを意味ある
    配分へ集中させる（手生成側の組合せ爆発の抑制）。意味ある付与が一つも無ければ ATTACH_DON は全て落ちる
    （TURN_END・ATTACK・PLAY 等は常に残る）。CPU の探索/方策のみで作用しエンジンの合法手列挙は変えない。
    """
    if not moves:
        return moves
    actor = _player_by_name(manager, actor_name)
    by_uuid = {}
    for u in ([actor.leader] if actor.leader is not None else []) + list(actor.field):
        uid = getattr(u, "uuid", None)
        if uid is not None:
            by_uuid[uid] = u
    out: List[Dict[str, Any]] = []
    for m in moves:
        if m.get("action_type") == "ATTACH_DON":
            uid = (m.get("payload") or {}).get("uuid")
            tgt = by_uuid.get(uid)
            if tgt is None or not _attach_don_meaningful(manager, actor_name, tgt):
                continue
        out.append(m)
    return out


def _attacker_has_on_attack(card) -> bool:
    """attacker が【アタック時】能力を持つか。持つ場合は対象を倒せ/貫けなくても発動自体が目的に
    なり得る（カタリーナ OP16-104 等）ため、無駄攻撃の除外対象から外す（保守的に残す）。"""
    for ab in getattr(getattr(card, "master", None), "abilities", ()) or ():
        if getattr(ab, "trigger", None) == TriggerType.ON_ATTACK:
            return True
    return False


def _prune_futile_attacks(manager, actor_name: str, moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """無駄攻撃（攻撃側の有効パワー < 対象の有効パワー＝KO も貫通もできない）を CPU 候補から除外する。

    キャラ攻撃で対象を KO できない／リーダー攻撃で素通り（ライフを取れない）攻撃は、攻撃者をレストに
    するだけで何も達成しない（しかも相手は防御不要なのでカウンターも強要できない）。にもかかわらず
    探索は「自ターンが続く＝攻め圧 `W_ATTACKER` ぶん」TURN_END より高く評価し、相手リーダーが防御効果で
    パワーを上げ自軍の小型が顔に届かない局面（OP11-041 ナミの【相手のアタック時】+2000 等）で、CPU が
    倒せないキャラへ無駄攻撃していた（2026-06-19 報告）。**現在の有効パワーで届かない攻撃**を落とす。
    届かせるためのドン付与は別手（ATTACH_DON）として残るので、付与→攻撃の貫通筋は損なわない。
    【アタック時】持ちは効果が目的になり得るため除外しない。CPU の探索/方策のみで作用しエンジンの
    合法手列挙は変えない（人間プレイは無駄攻撃も自由）。
    """
    if not moves:
        return moves
    actor = _player_by_name(manager, actor_name)
    opp = _other(manager, actor_name)
    atk_by_uuid = {}
    for u in ([actor.leader] if actor.leader is not None else []) + list(actor.field):
        uid = getattr(u, "uuid", None)
        if uid is not None:
            atk_by_uuid[uid] = u
    tgt_by_uuid = {}
    for u in ([opp.leader] if opp.leader is not None else []) + list(opp.field):
        uid = getattr(u, "uuid", None)
        if uid is not None:
            tgt_by_uuid[uid] = u

    def _pw(card, is_attacker):
        try:
            return float(card.get_power(is_attacker))
        except Exception:
            return float(getattr(getattr(card, "master", None), "power", 0) or 0)

    out: List[Dict[str, Any]] = []
    for m in moves:
        if m.get("action_type") == "ATTACK":
            payload = m.get("payload") or {}
            a = atk_by_uuid.get(payload.get("uuid"))
            tids = payload.get("target_ids") or []
            t = tgt_by_uuid.get(tids[0]) if tids else None
            if (a is not None and t is not None and not _attacker_has_on_attack(a)
                    and _pw(a, True) < _pw(t, False)):
                continue  # 無駄攻撃: 現在の有効パワーでは KO も貫通もできない
        out.append(m)
    return out


# ドン!!返却（ドン-N コスト）の追加減点（§2.5.3）。アクティブドンをドンデッキへ戻す手は、当面の盤面形成力
# （将来の手出し・ドン付与の上限）を下げるテンポ損。静的 eval の `W_DON_ACTIVE`(200) だけでは過小評価で、
# 序盤に 2 ドン戻して軽微な効果（万雷 OP15-078 のドロー+レスト等）を撃つ不自然手を招く。戻した正味枚数
# （= 手の後にドンデッキが増えた分。紫のドンランプ等で再追加され正味増えない手は対象外）×序盤係数で減点。
_W_DON_RETURN = 600.0
_DON_DECK_FULL = 10.0


def _don_return_penalty(manager, actor_name: str, child) -> float:
    """root 手で actor がアクティブドンをドンデッキへ正味で戻した量に応じた追加減点（>=0）。

    戻した枚数 = `child` の actor ドンデッキ − 現在の actor ドンデッキ（増分＝返却）。序盤（ドンデッキが
    多く残る＝伸び代が大きい）ほど重く、終盤は軽い。ドンデッキから場へ足す手（ランプ＝増やす手）や
    正味増減が無い手は 0。CPU の手選択のみで作用し eval/合法手列挙は変えない。
    """
    if child is None:
        return 0.0
    try:
        before = _player_by_name(manager, actor_name)
        after = _player_by_name(child, actor_name)
        returned = len(after.don_deck) - len(before.don_deck)
    except Exception:
        return 0.0
    if returned <= 0:
        return 0.0
    early = min(1.0, len(before.don_deck) / _DON_DECK_FULL)
    return returned * _W_DON_RETURN * early


def _don_return_penalty_vals(before_don: int, after_don: int) -> float:
    """`_don_return_penalty` の値版（make/unmake 経路用＝適用前後のドンデッキ枚数だけで同値計算）。"""
    returned = after_don - before_don
    if returned <= 0:
        return 0.0
    early = min(1.0, before_don / _DON_DECK_FULL)
    return returned * _W_DON_RETURN * early


def evaluate(manager, me_name: str, see_opp_hand: bool = True,
             out: Optional[Dict[str, Any]] = None) -> float:
    """`me_name` 視点の盤面評価（L1 単一系統＝`evaluate_base`）。高いほど自分有利。

    かつては学習価値（winprob）の葉ブレンドを被せる 2 層構成だったが、ブレンドは α=0（既定）で常に素通し＝
    Elo 中立と実測され、学習価値サブシステムごと撤去した（2026-06-28）。本関数は `evaluate_base` の別名。
    """
    return evaluate_base(manager, me_name, see_opp_hand=see_opp_hand, out=out)


def evaluate_base(manager, me_name: str, see_opp_hand: bool = True,
                  out: Optional[Dict[str, Any]] = None) -> float:
    """`me_name` 視点の L1 評価（学習ブレンドを含まない素のスコア・高いほど自分有利）。

    CPU 評価は単一系統（L1コア・`cpu_eval_v2.evaluate_v2`）に集約済み。winner 短絡を含む終端の符号も
    `evaluate_v2` 側が扱う（重複ロジックを置かない）。`see_opp_hand=False` のとき相手手札は枚数のみ評価する
    （中身＝カウンター値を読まない）。自分の手札は常に full。情報方針は呼び出し側（`decide`）が渡す。
    """
    from . import cpu_eval_v2
    return cpu_eval_v2.evaluate_v2(manager, me_name, see_opp_hand=see_opp_hand, out=out)


def _pending_keys():
    from . import action_api
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    return pending_props.get('PLAYER_ID', 'player_id'), pending_props.get('ACTION', 'action')


# 効果対象選択（KO/除去/バウンス/手札破壊/場溢れトラッシュ等）の対話アクション名。
# get_pending_request が SELECT_TARGET / FIELD_OVERFLOW_TRASH をこの fe_action に正規化する。
# 探索ではこの単一対象選択を「候補ごとの手」に分岐して最善対象を読み切る（docs/SPEC.md §2.5.2）。
_SELECT_ACTION = "SEARCH_AND_SELECT"
# 1 つの選択ノードで分岐する単一対象候補数の安全上限（クローン暴発防止。通常は盤面サイズ未満）。
HARD_SELECT_CAP = 8


def _drain_own_interactions(manager, actor_name: str, stop_at_select: bool = False) -> None:
    """クローン上で actor 側の効果対話を既定解決でドレインする（採点を安定させるため）。

    相手の意思決定（ブロック/カウンター等）は解決しない（相手に委ねる）。
    `stop_at_select=True` のとき、分岐対象の単一対象選択（_SELECT_ACTION）はドレインせず残す
    （探索側が候補ごとに分岐して最善対象を選ぶため・§2.5.2）。
    """
    from . import action_api
    for _ in range(_DRAIN_LIMIT):
        # 判定は軽量版（pid, action だけ）で行い、重い payload は実際にドレインするときだけ作る
        # （get_pending_request は毎回 selectable 構築＋uuid4 で重い・§2.5.2）。pending_actor_action は
        # get_pending_request と (pid, action)・副作用が一致（test_pending_actor_action_matches_full）。
        pa = manager.pending_actor_action()
        if not pa or pa[0] != actor_name:
            return
        action = pa[1]
        # メイン/マリガン/戦闘は「意思決定」なのでドレインしない（呼び出し側が1手として扱う）。
        if action in ("MAIN_ACTION", "MULLIGAN", "SELECT_BLOCKER", "SELECT_COUNTER"):
            return
        # 探索モードでは分岐可能な単一対象選択もドレインしない（探索ノードとして残す）。
        if stop_at_select and _selection_moves(manager, actor_name) is not None:
            return
        pending = manager.get_pending_request()  # ドレイン確定時のみフル payload を構築
        payload = manager.default_interaction_payload(pending)
        actor = _player_by_name(manager, actor_name)
        manager.action_events = []
        try:
            action_api.apply_game_action(manager, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            return


def _apply_move_inplace(board, actor_name: str, move: Dict[str, Any], stop_at_select: bool = False):
    """`board` に move を**その場で**適用し、actor 側の対話をドレインする（例外はそのまま送出）。

    clone 経路（`_apply_clone`）と make/unmake 経路（`_score_move_1ply`）の共通コア。
    """
    from . import action_api
    actor = _player_by_name(board, actor_name)
    if move["kind"] == "battle":
        action_api.apply_battle_action(board, actor, move["action_type"], move.get("card_uuid"))
    else:
        action_api.apply_game_action(board, actor, move["action_type"], move.get("payload", {}))
    _drain_own_interactions(board, actor_name, stop_at_select=stop_at_select)


def _apply_clone(manager, actor_name: str, move: Dict[str, Any], stop_at_select: bool = False):
    """move を新しいクローンへ適用し、actor 側の対話をドレインしたクローンを返す。

    シミュレーションが例外を出す手は None を返す（呼び出し側で除外する）。
    `stop_at_select=True` のとき、ドレインは分岐対象の単一対象選択で停止する（探索側が分岐する）。
    """
    clone = manager.clone()
    clone.action_events = []
    try:
        _apply_move_inplace(clone, actor_name, move, stop_at_select=stop_at_select)
    except Exception:
        return None
    return clone


def _mu_safe(manager) -> bool:
    """make/unmake を安全に使える局面か。

    **parked resolver 状態（中断再開）も journaled 化済み**（resolver の journaled `__setattr__`／
    `context`・`saved_targets` の JournaledDict 化／`execution_stack`・`saved_stack`・退避スタックの
    JournaledList 化／誘発 item の JournaledDict 化）＝中断を再開する手も「適用→巻き戻し→開始状態と
    完全一致」が成立する（`tests/test_journal.py` の parked round-trip が実プレイ全手で機械照合）。
    よって中断局面でも clone へ退避せず make/unmake できる（残 clone フォールバックの大半を解消）。
    """
    return _USE_MAKE_UNMAKE


def _score_move_1ply(manager, actor_name: str, move: Dict[str, Any], eval_name: str,
                     see_opp_hand: bool, stop_at_select: bool = False) -> Optional[float]:
    """move 適用後の 1-ply 評価値（`eval_name` 視点）を返す。失敗（例外）は None。

    `_mu_safe` な局面では **make/unmake**（manager をその場で適用→採点→巻き戻し＝clone 不要）、
    それ以外（中断再開）は従来どおり clone で採点する。**clone 方式と完全同値**（同じ適用・同じ
    evaluate を、複製の有無だけ替えて行う）。
    """
    if _mu_safe(manager):
        saved_events = manager.action_events
        with journal.transaction():
            manager.action_events = JournaledList()
            try:
                _apply_move_inplace(manager, actor_name, move, stop_at_select=stop_at_select)
            except Exception:
                return None
            val = evaluate(manager, eval_name, see_opp_hand=see_opp_hand)
        # action_events は transient バッファ（探索値に無関係）。txn 内の付け替えも巻き戻るが念のため復元。
        manager.action_events = saved_events
        return val
    clone = _apply_clone(manager, actor_name, move, stop_at_select=stop_at_select)
    if clone is None:
        return None
    return evaluate(clone, eval_name, see_opp_hand=see_opp_hand)


def _recurse_child(manager, actor_name: str, move: Dict[str, Any], search_fn) -> Optional[float]:
    """move を適用した子局面で `search_fn(child) -> float` を評価して返す。失敗は None。

    `_mu_safe` な局面では **make/unmake**（manager をその場で適用→`search_fn` で再帰→巻き戻し＝
    再帰中だけ manager が子状態・抜けると無傷に戻る）。それ以外は clone で子を作って再帰する。
    深い再帰は入れ子トランザクションで扱われ、各深さが順に巻き戻る（世代カウンタ）。
    """
    if _mu_safe(manager):
        saved_events = manager.action_events
        result: List[Optional[float]] = [None]
        with journal.transaction():
            manager.action_events = JournaledList()
            try:
                _apply_move_inplace(manager, actor_name, move, stop_at_select=True)
            except Exception:
                result[0] = None
            else:
                result[0] = search_fn(manager)
        manager.action_events = saved_events
        return result[0]
    child = _apply_clone(manager, actor_name, move, stop_at_select=True)
    if child is None:
        return None
    return search_fn(child)


def _simulate_and_eval(manager, actor_name: str, move: Dict[str, Any],
                       see_opp_hand: bool = True) -> float:
    """move をクローン上で適用し、actor 側の対話をドレインしてから評価する（1-ply）。"""
    val = _score_move_1ply(manager, actor_name, move, actor_name,
                           see_opp_hand=see_opp_hand)
    return float("-inf") if val is None else val


def _rank_select_candidates(manager, uuids: List[str], actor_name: str) -> List[str]:
    """選択候補 uuid を「CPU にとって選ぶ価値の高い順」に並べる。

    相手のカード（除去/弱体の対象）＝**脅威の大きい順**（パワー→コスト降順）に除去する。
    自分のカード（コスト/犠牲としての対象）＝**価値の小さい順**（パワー→コスト昇順）に差し出す。
    候補に対応するカードが見つからないものは末尾へ（順序のみのヒューリスティック）。"""
    def _pw(c):
        try:
            return c.get_power(False)
        except Exception:
            return getattr(getattr(c, "master", None), "power", 0) or 0
    def _cost(c):
        return getattr(getattr(c, "master", None), "cost", 0) or 0
    pairs = [(u, manager._find_card_by_uuid(u)) for u in uuids]
    found = [(u, c) for u, c in pairs if c is not None]
    missing = [u for u, c in pairs if c is None]
    if found and all(getattr(c, "owner_id", None) == actor_name for _u, c in found):
        found.sort(key=lambda uc: (_pw(uc[1]), _cost(uc[1])))            # 自分＝弱い順に差し出す
    else:
        found.sort(key=lambda uc: (-_pw(uc[1]), -_cost(uc[1])))          # 相手＝強い脅威から除去
    return [u for u, _c in found] + missing


def _selection_moves(manager, actor_name: str):
    """actor の対象選択／任意確認対話を RESOLVE 手として列挙する（無ければ None）。

    対象は `_SELECT_ACTION`（SELECT_TARGET/FIELD_OVERFLOW_TRASH を正規化したもの）と
    `CONFIRM_OPTIONAL`（任意コスト「〜できる：」／任意効果「〜してもよい」の発動可否）。
      - **単一対象（max==1）**: 候補ごとに 1 手へ分岐＝「どれを KO/除去/バウンス/手札破壊するか」を読む。
      - **多対象「N枚まで」（max>=2）**: 影響度順（`_rank_select_candidates`）に **min..max 枚の累積**選択を
        候補化し、探索に「何枚・どれを選ぶか」を読ませる（候補は max-min+1 手＝有界）。これにより
        『相手のコスト1以下のキャラ2枚までを KO』等の**有益な除去を 0 枚で取りこぼす**のを防ぐ
        （従来は max>=2 を既定解決＝最小数=0枚へ委ねていた）。
      - **任意確認（CONFIRM_OPTIONAL・can_skip）**: 発動する(accept)／しない(decline) の2手へ分岐。
        従来は `get_legal_actions` が既定(accept)の1手しか出さず、CPU は**任意コストを必ず払って**いた
        （例: ティーチ OP16-080 が相手のアタック時にトリガー1枚を捨ててアタック対象を変更＝リーダーが
        既に対象なら no-op なのに毎回カードを浪費）。両手を採点して、得なときだけ払う。
    """
    from . import action_api
    pending = manager.get_pending_request()
    if not pending:
        return None
    KEY_PID, KEY_ACTION = _pending_keys()
    if pending.get(KEY_PID) != actor_name:
        return None
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_UUIDS = props.get('SELECTABLE_UUIDS', 'selectable_uuids')
    KEY_CONSTRAINTS = props.get('CONSTRAINTS', 'constraints')
    KEY_SKIP = props.get('CAN_SKIP', 'can_skip')

    # 任意確認（任意コスト/任意効果の発動可否）: accept(発動) / decline(見送り) を採点させる。
    if pending.get(KEY_ACTION) == "CONFIRM_OPTIONAL" and bool(pending.get(KEY_SKIP, False)):
        base = manager.default_interaction_payload(pending)
        accept = dict(base); accept["accepted"] = True
        decline = dict(base); decline["accepted"] = False
        return [{"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": accept},
                {"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": decline}]

    # 並び替え/上下選択（ARRANGE_DECK）: 従来は既定解決1手のみ＝底送りの順番・scry の上/下を
    # 探索できなかった。全順列は爆発するため「どのカードを先頭（＝配置順の1枚目）にするか」の
    # 回転 × 上/下（allow_position 時）だけを候補化する。返り値は**既定解決を含む完全な集合**
    # （L1 経路は本関数の返りで合法手を置換するため）。既定の組は base と同一 payload にする＝
    # merged_search_actions のキー重複除去で二重 edge にならない（訪問数を分裂させない）。
    if pending.get(KEY_ACTION) == "ARRANGE_DECK":
        uuids = list(pending.get(KEY_UUIDS, []) or [])
        allow_pos = bool(pending.get("allow_position", False))
        allow_reorder = bool(pending.get("allow_reorder", False))
        if not uuids or not (allow_pos or (allow_reorder and len(uuids) >= 2)):
            return None
        base = manager.default_interaction_payload(pending)

        def _mk_arr(order, position):
            payload = dict(base)
            if order is not None:            # None＝既定の並び（base の selected をそのまま）
                payload["selected_uuids"] = list(order)
            if position is not None:
                payload["position"] = position
            return {"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": payload}

        orders: List[Optional[List[str]]] = [None]
        if allow_reorder and len(uuids) >= 2:
            # 先頭カードの回転のみ（uuids[0] 先頭＝既定の並びは None が代表）。
            orders += [[u] + [v for v in uuids if v != u] for u in uuids[1:HARD_SELECT_CAP]]
        # 既定 payload の position は "BOTTOM"（上下選択なしの効果は resolve 側が fixed を適用）。
        positions = ["BOTTOM", "TOP"] if allow_pos else [None]
        moves = [_mk_arr(o, p) for o in orders for p in positions]
        return moves if len(moves) >= 2 else None

    if pending.get(KEY_ACTION) != _SELECT_ACTION:
        return None
    uuids = list(pending.get(KEY_UUIDS, []) or [])
    constraints = pending.get(KEY_CONSTRAINTS) or {}
    try:
        min_n = int(constraints.get("min", 0))
    except (TypeError, ValueError):
        min_n = 0
    try:
        max_n = int(constraints.get("max", len(uuids)))
    except (TypeError, ValueError):
        max_n = len(uuids)
    if not uuids:
        return None
    base = manager.default_interaction_payload(pending)

    def _mk(sel):
        payload = dict(base)
        payload["selected_uuids"] = list(sel)
        return {"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": payload}

    # 単一対象選択: 候補ごとに分岐。
    if max_n == 1 and min_n <= 1:
        moves: List[Dict[str, Any]] = [_mk([uid]) for uid in uuids[:HARD_SELECT_CAP]]
        # 任意選択（min==0・スキップ可）なら「選ばない」も一級の候補にする。
        if min_n == 0 and bool(pending.get(KEY_SKIP, False)):
            moves.append(_mk([]))
        return moves

    # 多対象「N枚まで」: 影響度順に min..max 枚の累積選択を候補化（探索に枚数を選ばせる）。
    if max_n >= 2 and 0 <= min_n <= max_n:
        ranked = _rank_select_candidates(manager, uuids, actor_name)
        hi = min(max_n, len(ranked))
        lo = max(min_n, 0)
        moves = [_mk(ranked[:k]) for k in range(lo, hi + 1)]
        return moves or None
    return None


def merged_search_actions(manager, actor_name: str, base_moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """学習型CPU(MCTS)用の合法手併合。

    `manager.get_legal_actions` は効果選択対話（SELECT_TARGET/CONFIRM_OPTIONAL 等）に対して
    組合せ爆発回避のため「機械的既定解決」1手しか返さない（CONFIRM_OPTIONAL=必ず accept・
    SELECT_TARGET min0=必ず0枚）。これを MCTS に渡すと**選択そのものを探索できず**、
    任意効果を常に発動（OP16-080 リダイレクトを毎回浪費）・up-to効果を常に見送る
    （OP16-119 のライフ追加を絶対使わない）という配線起因の系統的悪手になる。

    L1 が探索する候補ごと／accept・decline の代替手（`_selection_moves`）を base_moves に併合し、
    学習 MCTS が各選択の結果局面を value ネットで評価できるようにする。選択対話でない局面
    （MAIN_ACTION 等）は `_selection_moves` が None を返すため base_moves をそのまま返す。
    """
    alts = _selection_moves(manager, actor_name)
    if not alts:
        return base_moves

    out = list(base_moves)
    seen = {_selection_merge_key(m) for m in out}
    for m in alts:
        k = _selection_merge_key(m)
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


def _selection_merge_key(move: Dict[str, Any]):
    """`merged_search_actions` の重複判定キー（uuid 基準・同一 payload の二重併合を防ぐ）。

    position を含める: TOP/BOTTOM だけが違う代替手を将来生成したとき、誤って同一視して
    間引かないため（従来キーは position 欠落＝潜在の取りこぼし地雷だった）。
    """
    p = move.get("payload") or {}
    su = p.get("selected_uuids")
    su = tuple(su) if isinstance(su, (list, tuple)) else su
    return (move.get("action_type"), su, p.get("accepted"), p.get("index"), p.get("position"))


def _consumes_hand_card(manager, actor_name: str, move: Dict[str, Any]) -> bool:
    """move が actor の手札のカードを使う手か（手札からの登場 PLAY・手札からのカウンター等）。

    公平モデル（opp_public_only）で相手 min ノードから除外するための判定。盤面カードを参照する
    手（ATTACK/ATTACH_DON/ACTIVATE_MAIN/SELECT_BLOCKER）は手札 uuid に一致しないため残る。
    """
    payload = move.get("payload") or {}
    uuid = payload.get("uuid") or move.get("card_uuid")
    if not uuid:
        return False
    actor = _player_by_name(manager, actor_name)
    return any(getattr(c, "uuid", None) == uuid for c in actor.hand)


def _settle_eval(manager, root_name: str, see_opp_hand: bool, ply: int = 0) -> float:
    """探索の打ち切り点を一定の静止点（相手のターン開始＝相手の MAIN_ACTION）へ整流してから評価する。

    予算/ply 上限での打ち切りでも「自分のターン途中／戦闘途中の甘い局面」で評価せず、葉と同じ静止点で
    採点する（horizon の抜け道＝自ターンや戦闘で粘って相手の反撃を地平線外へ追いやる、を塞ぐ）。整流は
    既定解決のみ:
      - root の MAIN → root の TURN_END でターン境界へ送る。
      - 戦闘応答（SELECT_BLOCKER/SELECT_COUNTER・どちら側でも）→ 既定 PASS で戦闘を解決。
      - その他の選択（どちら側でも）→ `default_interaction_payload` で既定解決。
    相手の MAIN_ACTION に到達したら停止（＝相手ターン開始の静止点）。`manager` はクローンなので破壊的に進めてよい。
    """
    from . import action_api
    KEY_PID, KEY_ACTION = _pending_keys()
    battle_actions = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    ACT_PASS = battle_actions.get('PASS', 'PASS')
    for _ in range(_SETTLE_LIMIT):
        if manager.winner is not None:
            break
        # 判定は軽量版（pid, action）で。フル payload は既定選択を解決する else 枝でだけ作る。
        pa = manager.pending_actor_action()
        if not pa:
            break
        pid, action = pa
        if pid != root_name and action == "MAIN_ACTION":
            break  # 相手のターン開始＝静止点に到達
        actor = _player_by_name(manager, pid)
        manager.action_events = []
        try:
            if action == "MAIN_ACTION":              # root の手番 → ターンを畳む
                action_api.apply_game_action(manager, actor, "TURN_END", {})
            elif action in ("SELECT_BLOCKER", "SELECT_COUNTER"):  # 戦闘応答 → 既定パスで解決
                action_api.apply_battle_action(manager, actor, ACT_PASS, None)
            else:                                     # その他の選択 → 既定解決
                pending = manager.get_pending_request()  # 既定解決時のみフル payload
                payload = manager.default_interaction_payload(pending)
                action_api.apply_game_action(manager, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            break
    # 整流の途中/結果で勝敗が確定していたら、_search の winner 検出と同じく **ply 割引**して返す（最短の
    # 止めを優先）。予算切れ settle で勝者を観測した長い手順が、winner 検出（W_WIN-ply）の直接の止めより
    # 高く（生 W_WIN で）見えてしまう不整合を防ぐ＝lethal 認識の ply 割引を一貫させる。
    if manager.winner is not None:
        return (W_WIN - ply) if manager.winner == root_name else -(W_WIN - ply)
    return evaluate(manager, root_name, see_opp_hand=see_opp_hand)


def _record_killer(killers: Optional[Dict[int, List[tuple]]], ply: int, move: Dict[str, Any]) -> None:
    """④ カット（alpha>=beta）を起こした手をこの ply の killer として記録（最近使用を先頭・上限 _KILLER_SLOTS）。

    `killers` が None（PV 順序 OFF／外部呼び出し）なら no-op＝従来挙動。記録は move の signature のみ
    （盤面キーは持たない＝置換表のような健全性リスク無し。衝突しても順序が変わるだけで誤った値は再利用しない）。
    """
    if killers is None:
        return
    sig = _move_sig(move)
    lst = killers.get(ply)
    if lst is None:
        killers[ply] = [sig]
    elif sig in lst:
        lst.remove(sig)
        lst.insert(0, sig)
    else:
        lst.insert(0, sig)
        if len(lst) > _KILLER_SLOTS:
            lst.pop()


def _pv_reorder(children: List[Tuple[float, Any]],
                kill_sigs: List[tuple]) -> List[Tuple[float, Any]]:
    """④ ビーム選別後の子集合 `children` を、killer 手が先頭に来るよう**安定**に並べ替える。

    集合（要素）は不変＝順序のみ変化（探索する子は変わらない）。killer 以外は元の best-first 順を保ち、
    killer 同士は `kill_sigs`（最近使用が先頭）の順に並べる。killer がビーム内に無ければ元のまま返す。
    """
    if not kill_sigs:
        return children
    rank = {s: i for i, s in enumerate(kill_sigs)}
    front: List[Tuple[float, Any]] = []
    rest: List[Tuple[float, Any]] = []
    for c in children:
        (front if _move_sig(c[1]) in rank else rest).append(c)
    if not front:
        return children
    front.sort(key=lambda c: rank[_move_sig(c[1])])  # 安定ソート＝同 rank は元順を保持
    return front + rest


def _search(manager, root_name: str, alpha: float, beta: float,
            budget: List[int], see_opp_hand: bool, opp_public_only: bool,
            ply: int = 0, start_turn: int = 0, horizon: int = 1,
            killers: Optional[Dict[int, List[tuple]]] = None) -> float:
    """ターン境界評価の α-β ＋ ビーム探索。`root_name` 視点の最善到達値を返す。

    **葉は `start_turn` から `horizon` ターン進んだ MAIN_ACTION（一定の静止点）に固定**する。
      - `horizon=1`: 相手のターン開始（自分のターン完全解決後）で評価（B1・相手ターンへは潜らない）。
      - `horizon=2`: 相手のターンを丸ごと（攻撃まで）読み、自分の次ターン開始で評価（B2-lite・守りの
        深読み）。相手の攻撃は相手 min・自分のブロック/カウンターは root max として読まれる。
    自ターン内（diff=0）の自分のメイン手は max、自分のアタックへの相手応答は min。全候補が同じ静止点で
    評価されるため、手番パリティ／horizon による「常に何かする」バイアス（パスの不当な低評価）が消える。
    探索木内で `winner` に到達した手順は ±(W_WIN − ply) でリーサル認識として機能する（ply 割引で最短の止め）。

    情報方針:
      - `see_opp_hand`     : 葉の評価で相手手札の中身（カウンター値）を読むか（hard=True / 他=False）。
      - `opp_public_only`  : 相手 min ノードで相手の隠れ手札に依存する手（PLAY/カウンター）を除外する保守
                             モデル（normal=True / hard=False）。
    """
    if manager.winner is not None:
        return (W_WIN - ply) if manager.winner == root_name else -(W_WIN - ply)

    # 探索は (手番, アクション) しか見ない（手は get_legal_actions から）。重い get_pending_request
    # ではなく軽量な pending_actor_action を使う（payload/uuid4 を作らない・副作用は一致・§2.5.2 B-1）。
    pa = manager.pending_actor_action()
    if not pa:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand)
    actor_name, pend_action = pa
    # 葉: start_turn から horizon ターン進んだ MAIN_ACTION（一定の静止点）で評価。
    if pend_action == "MAIN_ACTION" and (manager.turn_count - start_turn) >= horizon:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand)
    # 安全打ち切り: 予算/ply 上限。自分の手番途中ならターン境界へ整流してから評価（甘い途中評価を避ける）。
    if budget[0] <= 0 or ply >= _effective_max_ply():
        return _settle_eval(manager, root_name, see_opp_hand, ply)

    actor = _player_by_name(manager, actor_name)
    # 単一対象選択ノードは候補ごとに分岐（最善対象を読み切る）。それ以外は通常の合法手列挙。
    moves = _selection_moves(manager, actor_name)
    if moves is None:
        moves = manager.get_legal_actions(actor)
        moves = _prune_don_moves(manager, actor_name, moves)  # B-2: 無意味なドン付与を手生成段で除外
        moves = _prune_futile_attacks(manager, actor_name, moves)  # 倒せない/届かない無駄攻撃を除外
    if not moves:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand)
    is_max = (actor_name == root_name)

    # 公平モデル: 相手 min ノードでは相手の隠れ手札に依存する手を読まない（公開情報のみで応答）。
    if not is_max and opp_public_only:
        filtered = [m for m in moves if not _consumes_hand_card(manager, actor_name, m)]
        if filtered:
            moves = filtered

    # 子ノードを生成し、1-ply 評価でビーム選別（best-first で α-β の枝刈り効率を上げる）。
    # ② 1-ply 採点は make/unmake（_score_move_1ply）で行い、ビーム選別には**子クローンを保持しない**
    #    （(値, 手) だけ持つ）。深掘り対象（上位 HARD_BEAM）だけを後段で改めて clone し直して再帰する。
    #    これで「全候補ぶんの clone」→「ビーム HARD_BEAM 件の clone」に減らす（clone は探索コストの ~86%）。
    children: List[Tuple[float, Any]] = []
    for m in moves:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        v = _score_move_1ply(manager, actor_name, m, root_name,
                             see_opp_hand=see_opp_hand, stop_at_select=True)
        if v is None:
            continue
        children.append((v, m))
    if not children:
        return _settle_eval(manager, root_name, see_opp_hand, ply)
    children.sort(key=lambda x: x[0], reverse=is_max)
    # E1（Phase3 ③）: 自分(max)は最善へ収束＝HARD_BEAM、相手(min)は応手を広く残す＝HARD_OPP_BEAM。
    children = children[:(HARD_BEAM if is_max else HARD_OPP_BEAM)]
    # ④ PV/killer 順序付け（docs/SPEC.md §2.5.3）: ビーム選別**後**の集合の中で killer 手を先頭へ寄せて
    # α-β カットを早める（集合は不変＝予算非拘束なら値・選択は完全同値・予算拘束時のみ深く読めて改善）。
    if killers is not None:
        children = _pv_reorder(children, killers.get(ply, ()))

    if is_max:
        value = float("-inf")
        for _leaf, m in children:
            cv = _recurse_child(manager, actor_name, m,
                                lambda b, a=alpha, bt=beta: _search(b, root_name, a, bt,
                                    budget, see_opp_hand, opp_public_only, ply + 1,
                                    start_turn, horizon, killers))
            if cv is None:
                continue
            value = max(value, cv)
            alpha = max(alpha, value)
            if alpha >= beta:
                _record_killer(killers, ply, m)  # ④ この手がカットを起こした＝同 ply の killer に登録
                break
        return value
    else:
        value = float("inf")
        for _leaf, m in children:
            if alpha >= beta:
                break
            cv = _recurse_child(manager, actor_name, m,
                                lambda b, a=alpha, bt=beta: _search(b, root_name, a, bt,
                                    budget, see_opp_hand, opp_public_only, ply + 1,
                                    start_turn, horizon, killers))
            if cv is None:
                continue
            value = min(value, cv)
            beta = min(beta, value)
            if alpha >= beta:
                _record_killer(killers, ply, m)  # ④ この手がカットを起こした＝同 ply の killer に登録
                break
        return value


def _active_blocker_count(p) -> int:
    n = 0
    for c in p.field:
        try:
            if not getattr(c, "is_rest", False) and c.has_keyword("ブロッカー"):
                n += 1
        except Exception:
            pass
    return n


def _is_important_root_move(manager, name: str, move: Dict[str, Any], child) -> bool:
    """B-3（§2.5.3）: 1-ply ランクに関係なく深掘りすべき重要手か。`child` は move 適用後のクローン。

    重要クラス: ①除去候補（単一対象選択の RESOLVE）②ブロッカー設置（適用後に自分のアクティブブロッカーが
    増える）③逆算リーサル/クロック手（適用後に相手ライフが減る）。1-ply の甘い採点で守備 setup や止め手を
    取りこぼすのを防ぐ（child を再利用するので追加クローンは生じない＝深掘り予算のみ消費）。
    """
    if child is None:
        return False
    from . import action_api
    if move.get("action_type") == action_api.ACT_RESOLVE_SELECTION:
        return True
    # クロック/逆算リーサル手: 相手リーダーへのアタック（child は戦闘応答待ちで未だライフ未減なので
    # 適用後ライフ差では拾えない＝ターゲットで判定する）。
    opp = _other(manager, name)
    if move.get("action_type") == "ATTACK" and opp.leader is not None:
        target_ids = (move.get("payload") or {}).get("target_ids") or []
        if getattr(opp.leader, "uuid", None) in target_ids:
            return True
    me_now = _player_by_name(manager, name)
    me_next = _player_by_name(child, name)
    if _active_blocker_count(me_next) > _active_blocker_count(me_now):
        return True
    # 効果で即時に相手ライフが減る手（バーン等・戦闘を介さない）。
    if len(_other(child, name).life) < len(opp.life):
        return True
    return False


def _is_important_root_move_post(manager, name: str, move: Dict[str, Any],
                                 pre_block: int, pre_opp_life: int) -> bool:
    """`_is_important_root_move` の make/unmake 版（child=適用後 manager・適用前スカラを受け取る）。

    `_is_important_root_move` と**完全同値**（適用後の自ブロッカー数を pre_block と、適用後の相手
    ライフ枚数を pre_opp_life と比較する）。相手リーダー uuid は適用で不変なので post の opp を使える。
    """
    from . import action_api
    if move.get("action_type") == action_api.ACT_RESOLVE_SELECTION:
        return True
    opp = _other(manager, name)
    if move.get("action_type") == "ATTACK" and opp.leader is not None:
        target_ids = (move.get("payload") or {}).get("target_ids") or []
        if getattr(opp.leader, "uuid", None) in target_ids:
            return True
    me_next = _player_by_name(manager, name)
    if _active_blocker_count(me_next) > pre_block:
        return True
    if len(_other(manager, name).life) < pre_opp_life:
        return True
    return False


def _eval_root_move(manager, name: str, move: Dict[str, Any], see_opp_hand: bool):
    """ルート手の 1-ply 採点に必要な (評価値, ドン返却ペナルティ, 重要手フラグ) をまとめて返す。

    `_mu_safe` な静止点では **make/unmake**（適用→3 値を txn 内で採点→巻き戻し＝子クローン不保持）、
    それ以外は clone。**clone 方式と完全同値**（同じ evaluate／同じ penalty／同じ importance を、
    複製の有無だけ替えて算出）。適用失敗（例外）は None。"""
    me = _player_by_name(manager, name)
    pre_don = len(me.don_deck)
    pre_block = _active_blocker_count(me)
    pre_opp_life = len(_other(manager, name).life)
    if _mu_safe(manager):
        saved_events = manager.action_events
        res = [None]
        with journal.transaction():
            manager.action_events = JournaledList()
            try:
                _apply_move_inplace(manager, name, move, stop_at_select=True)
            except Exception:
                res[0] = None
            else:
                ev = evaluate(manager, name, see_opp_hand=see_opp_hand)
                pen = _don_return_penalty_vals(pre_don, len(_player_by_name(manager, name).don_deck))
                imp = _is_important_root_move_post(manager, name, move, pre_block, pre_opp_life)
                res[0] = (ev, pen, imp)
        manager.action_events = saved_events
        return res[0]
    child = _apply_clone(manager, name, move, stop_at_select=True)
    if child is None:
        return None
    ev = evaluate(child, name, see_opp_hand=see_opp_hand)
    pen = _don_return_penalty(manager, name, child)
    imp = _is_important_root_move(manager, name, move, child)
    return (ev, pen, imp)


# 深掘り同点手の 1-ply タイブレーク（§2.5.3）。寄与は _TIEBREAK_W×クランプ prelim（最大 ~0.005）で、
# 実の深掘り差（典型 >0.3＝パワー1段ぶん）には影響せず、**厳密同点のみ**を即時盤面（1-ply）で割る。
# prelim は勝ち(±W_WIN)で巨大化し得るため _TIEBREAK_CLAMP でクランプし、巨大値が実差を覆さないようにする。
_TIEBREAK_W = 1e-6
_TIEBREAK_CLAMP = 5000.0


def _scored_search(manager, name: str, moves: List[Dict[str, Any]],
                   see_opp_hand: bool, opp_public_only: bool,
                   collect: Optional[Dict[str, Any]] = None,
                   killer_state: Optional[Dict[int, List[tuple]]] = None
                   ) -> List[Tuple[float, Dict[str, Any]]]:
    """ルート手を 1-ply で事前選別し、上位 HARD_ROOT_BEAM 手だけを多 ply 先読みで深掘りする。

    全手で予算を共有すると先に列挙された手ほど深く読まれて採点が不公平になるため、
    深掘り対象には**手ごとに均等予算**（HARD_PER_MOVE_BUDGET）を与える。非対象は 1-ply スコアの
    まま残す。事前選別で作った子クローンを深掘りに再利用するので無駄なクローンは作らない。

    `collect`（任意・既定 None＝完全に無オーバーヘッド）が渡されると、regret ログ（検証基盤・§2.5.3）
    用に 1-ply 事前スコアと深掘りスコアを `move_sig -> score` の dict で記録する:
      collect["prelim"]={sig: 1-ply スコア}, collect["deep"]={sig: 深掘りスコア}。
    """
    # 1) 全ルート手を 1-ply で採点。② make/unmake（`_eval_root_move`）で子クローンを保持せず
    #    (評価値, ドン返却ペナルティ, 重要手フラグ) をまとめて算出する（深掘りは後段で再適用）。
    #    アクティブドンをドンデッキへ戻す手は将来の盤面形成力を下げるテンポ損なので追加減点（prelim/deep 双方へ）。
    prelim: List[Tuple[float, Dict[str, Any], bool]] = []
    pen_by_idx: Dict[int, float] = {}
    imp_by_idx: Dict[int, bool] = {}
    for idx, m in enumerate(moves):
        r = _eval_root_move(manager, name, m, see_opp_hand)
        if r is None:
            prelim.append((float("-inf"), m, False))
            pen_by_idx[idx] = 0.0
            imp_by_idx[idx] = False
            continue
        ev, pen, imp = r
        pen_by_idx[idx] = pen
        imp_by_idx[idx] = imp
        prelim.append((ev - pen, m, True))

    # 2) 1-ply 上位を深掘り対象に選ぶ。TURN_END（パスの基準線）は必ず深掘りし、ターン境界で正しく採点する
    #    （非対象の 1-ply スコアは自ターン途中の甘い値になり得るため、パスの基準だけは確実に整える）。
    order = sorted(range(len(prelim)), key=lambda i: prelim[i][0], reverse=True)
    deepen = set(order[:HARD_ROOT_BEAM])
    for i, (_s, m, ok) in enumerate(prelim):
        if ok and m.get("action_type") == "TURN_END":
            deepen.add(i)
    # B-3: 重要手クラス（除去候補・ブロッカー設置・逆算リーサル/クロック）を 1-ply ランクに関係なく
    # 強制投入する（上限 HARD_FORCE_DEEPEN_CAP・1-ply 上位順＝レイテンシを絞りつつ取りこぼしを是正）。
    forced = 0
    for i in order:
        if forced >= HARD_FORCE_DEEPEN_CAP:
            break
        if i in deepen:
            continue
        _s, m, ok = prelim[i]
        if ok and imp_by_idx.get(i, False):
            deepen.add(i)
            forced += 1

    # 3) 深掘り対象を horizon ターン先まで探索（ply=1 から＝早い勝ちを優先）し、**深掘り集合のみ**返す。
    #    非対象（1-ply の甘い値）を混ぜると評価ホライズンが不一致になり誤選択するため返さない。深掘り集合は
    #    1-ply 上位＋TURN_END なので最善手はここに含まれる。
    start_turn = manager.turn_count
    if collect is not None:
        collect.setdefault("prelim", {})
        collect.setdefault("deep", {})
        for s1, m, ok in prelim:
            collect["prelim"][_move_sig(m)] = s1
    # ④ killer 表（ply→直近カット手の signature 列）。深掘り対象のルート手で共有し、α-β カットを早める。
    # 各ルート手は均等予算で独立に深掘りするが、killer は ply 単位の汎用ヒント＝木をまたいで再利用してよい
    # （集合は変えず順序のみ＝予算非拘束なら値不変）。OFF（_USE_PV_ORDER=False）なら None＝従来挙動。
    #   - 粒度a（killer_state=None）: この探索内だけの一時表（decide が終われば破棄）。
    #   - 粒度b（killer_state あり）: 呼び出し側（`mem["killers"]`）が保持する表を使い、**連続する decide 間で
    #     再利用**する。盤面が 1 手進んだだけの次手でも同 ply の良手（相手の応手/自分の続き）は刺さりやすい。
    #     持ち越しても reorder は集合不変＝予算非拘束なら値不変なので安全（粒度a と同じ等価ゲートで担保）。
    if not _USE_PV_ORDER:
        killers: Optional[Dict[int, List[tuple]]] = None
    elif killer_state is not None:
        killers = killer_state
    else:
        killers = {}
    out: List[Tuple[float, Dict[str, Any]]] = []
    for i, (s1, m, ok) in enumerate(prelim):
        if ok and i in deepen:
            budget = [_effective_budget()]
            v = _recurse_child(manager, name, m,
                               lambda b: _search(b, name, float("-inf"), float("inf"),
                                   budget, see_opp_hand, opp_public_only, ply=1,
                                   start_turn=start_turn, horizon=_effective_horizon(),
                                   killers=killers))
            if v is None:  # 深掘りの適用失敗（pass-1 と整合・通常は起きない）
                continue
            v -= pen_by_idx.get(i, 0.0)  # ドン!!返却のテンポ損を深掘り値にも反映（prelim と一致）
            # 深掘り値が**同点**の手は 1-ply（即時盤面）スコアで割る＝有益な選択（相手キャラの除去等）が
            # 深掘りで washout（相手ターン中の自分の誘発除去は探索ホライズン内で価値が相殺され同点になり得る）
            # してもランダムタイブレークで取りこぼさない。寄与は ±_TIEBREAK_W*クランプ prelim（最大 ~0.005）で、
            # 実差（>0.005）には一切影響しない＝従来採点をほぼ完全に保ったまま厳密同点のみを是正する。
            v_ranked = v + _TIEBREAK_W * max(-_TIEBREAK_CLAMP, min(_TIEBREAK_CLAMP, s1))
            out.append((v_ranked, m))
            if collect is not None:
                collect["deep"][_move_sig(m)] = v
    if not out:  # 念のため（全候補がクローン失敗）: 1-ply スコアにフォールバック
        out = [(s1, m) for s1, m, _ok in prelim]
    return out


# 1 ターン内に CPU が取れる手の総数上限（無限ループの最終防壁＝終了保証）。
# 正当なターンが到達しない大きさ＝思考に干渉しない。起動効果の繰り返しはエンジンの正規ゲート
# （コスト充足・ターン使用回数）で自己制限するため、旧 REPEAT_CAP は撤去した（2026-06-27）。
TURN_ACTION_CAP = 60


def _move_sig(move: Dict[str, Any]) -> tuple:
    payload = move.get("payload") or {}
    return (move.get("action_type"), payload.get("uuid") or move.get("card_uuid"),
            tuple(payload.get("target_ids", []) or []))


# === CPU 思考トレース（Phase 1・診断/挙動改善用） =====================================
# すべて trace 指定時のみ作動し、本番（trace=None）には一切のオーバーヘッド・挙動変化を与えない。
# トレースの手記述は uuid（実行ごとに変わる）でなく **card_id 基準** にして再現性を確保する
# （同一 seed → 同一決定列の比較が card_id で安定して行える）。
TRACE_TOPN = 6  # トレースに残す候補手の上限（deep スコア降順の上位）


def _find_card(manager, uuid: Optional[str]):
    """uuid からカードインスタンスを全ゾーン横断で引く（トレース記述用・低速で可）。"""
    if not uuid:
        return None
    for p in (manager.p1, manager.p2):
        leader = getattr(p, "leader", None)
        stage = getattr(p, "stage", None)
        # field/hand/life/deck/trash に加え **temp_zone**（解決中の一時ゾーン）と **stage** も探索する。
        # これらが漏れると ACTIVATE_MAIN 等の手記述が card_id に解決できず uuid のまま残り、
        # card_id 基準の記録が再現不能になる（実対局リプレイ・R1 round-trip で検出した欠落）。
        zones = [getattr(p, z, None) or [] for z in ("field", "hand", "life", "deck", "trash", "temp_zone")]
        if leader is not None:
            zones.append([leader])
        if stage is not None:
            zones.append([stage])
        for zone in zones:
            for c in zone:
                if getattr(c, "uuid", None) == uuid:
                    return c
    return None


def _card_label(manager, uuid: Optional[str]) -> Optional[str]:
    """uuid を card_id（無ければ名前・最後に uuid）へ解決した、再現性のある手記述ラベル。"""
    if not uuid:
        return None
    c = _find_card(manager, uuid)
    if c is None:
        return uuid
    return getattr(c.master, "card_id", None) or getattr(c.master, "name", None) or uuid


def _describe_move(manager, move: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """手を card_id 基準の人間可読 dict に変換する（uuid 非依存＝再現性あり）。"""
    if not move:
        return None
    payload = move.get("payload") or {}
    uuid = payload.get("uuid") or move.get("card_uuid")
    d: Dict[str, Any] = {"action_type": move.get("action_type")}
    label = _card_label(manager, uuid)
    if label:
        d["card"] = label
    tids = payload.get("target_ids") or []
    if tids:
        d["targets"] = [_card_label(manager, t) for t in tids]
    # 効果対話（RESOLVE_EFFECT_SELECTION 等）の選択内容も card_id 基準で載せる＝同種の選択肢を
    # 一意に区別できる（実対局リプレイの再現性。載せないと bare {action_type} で複数手が同記述になる）。
    extra = payload.get("extra") or {}
    sel = payload.get("selected_uuids") or extra.get("selected_uuids") or []
    if sel:
        d["selected"] = [_card_label(manager, u) for u in sel]
    for k in ("index", "position"):
        v = payload.get(k)
        if v is None:
            v = extra.get(k)
        if v is not None:
            d[k] = v
    # 任意効果の「見送り」(accepted=False) を明示する。accept 側は既定（多くの選択 payload が
    # accepted=True を機械的に含む）ため出力しない＝旧録画（accepted 無し）と同キーで照合できる。
    # 載せないと CONFIRM_OPTIONAL の accept/decline が同一記述に潰れ、トレースで区別不能だった。
    acc = payload.get("accepted")
    if acc is None:
        acc = extra.get("accepted")
    if acc is False:
        d["accepted"] = False
    return d


def _move_equiv_key(manager, move: Optional[Dict[str, Any]]):
    """手の挙動等価キー（`_describe_move` と同じ card_id 基準の同一視）。

    同名カードの別実体（手札の複製など）は同キー＝等価とみなす。リプレイの逆写像
    （`replay_runner._key`）と同じ仮定で、学習CPUのルート訪問数マージが使う。
    """
    d = _describe_move(manager, move) or {}
    return (d.get("action_type"), d.get("card"), tuple(d.get("targets") or ()),
            tuple(d.get("selected") or ()), d.get("index"), d.get("position"),
            d.get("accepted"))


def _read_ahead_line(manager, root_name: str, see_opp_hand: bool, opp_public_only: bool,
                     start_turn: int, horizon: int,
                     max_steps: int = 12) -> List[Dict[str, Any]]:
    """貪欲 PV（読み筋）: 各手番で 1-ply 最善手（root=max / 相手=min）を辿った想定進行。

    `_search` の探索木そのものではなく、その縮約版（各ノードでビーム1）の「想定される線」。
    trace 指定時のみ呼ばれるためコストは問わない。`_search` 本体には一切触れない＝探索挙動は不変。
    """
    line: List[Dict[str, Any]] = []
    cur = manager
    KEY_PID, KEY_ACTION = _pending_keys()
    # 繰り返しガードは不要（起動メインはエンジンの正規ゲートで自己制限・REPEAT_CAP 撤去済み）。
    # PV は `max_steps` で有界。
    for _ in range(max_steps):
        if cur.winner is not None:
            line.append({"winner": cur.winner})
            break
        pa = cur.pending_actor_action()  # (pid, action) だけで足りる（軽量・§2.5.2）
        if not pa:
            break
        # 葉: start_turn から horizon ターン進んだ MAIN_ACTION（_search と同じ静止点）で打ち切る。
        if pa[1] == "MAIN_ACTION" and (cur.turn_count - start_turn) >= horizon:
            break
        actor_name = pa[0]
        is_max = (actor_name == root_name)
        moves = _selection_moves(cur, actor_name)
        if moves is None:
            actor = _player_by_name(cur, actor_name)
            moves = _prune_don_moves(cur, actor_name, cur.get_legal_actions(actor))
        if not is_max and opp_public_only:
            filt = [m for m in moves if not _consumes_hand_card(cur, actor_name, m)]
            if filt:
                moves = filt
        if not moves:
            break
        best = None
        for m in moves:
            child = _apply_clone(cur, actor_name, m, stop_at_select=True)
            if child is None:
                continue
            sc = evaluate(child, root_name, see_opp_hand=see_opp_hand)
            if best is None or (sc > best[0] if is_max else sc < best[0]):
                best = (sc, m, child)
        if best is None:
            break
        sc, m, child = best
        line.append({"turn": cur.turn_count, "actor": actor_name, "is_max": is_max,
                     "move": _describe_move(cur, m), "eval": round(sc, 1)})
        cur = child
    return line


def _fill_decision_trace(trace: Dict[str, Any], manager, name: str, difficulty: str,
                         moves: List[Dict[str, Any]], scored: List[Tuple[float, Dict[str, Any]]],
                         collect: Optional[Dict[str, Any]], chosen: Dict[str, Any], folded: bool,
                         see_opp_hand: bool, opp_public_only: bool,
                         include_read_ahead: bool = True) -> None:
    """`decide` の意思決定結果を診断トレース dict に書き込む（trace 指定時のみ）。

    記録内容: 選んだ手・畳み判定・上位候補（1-ply prelim ／深掘り deep スコア）・regret・
    選んだ手の結果盤面の J値成分内訳・読み筋（貪欲 PV）。

    `include_read_ahead=False`（ライブ/本番の軽量トレース）では、最も重い `read_ahead`（読み筋＝
    各手番で全合法手をクローンする貪欲 PV）を省く。候補スコア・regret・J値成分は探索が出した値の回収＋
    クローン1回でほぼ無コストなので残す。読み筋はオフライン解析（`cpu_replay.py`）でのみ採る。
    """
    sig2move = {_move_sig(m): m for m in moves}
    cands: List[Dict[str, Any]] = []
    regret = 0.0
    if collect:
        prelim = collect.get("prelim", {})
        deep = collect.get("deep", {})
        sigs = list(deep.keys()) if deep else list(prelim.keys())
        for sig in sigs:
            m = sig2move.get(sig)
            cands.append({
                "move": _describe_move(manager, m) if m is not None else None,
                "prelim": round(prelim[sig], 1) if sig in prelim else None,
                "deep": round(deep[sig], 1) if sig in deep else None,
            })

        def _rank(c):
            return c["deep"] if c["deep"] is not None else (
                c["prelim"] if c["prelim"] is not None else float("-inf"))
        cands.sort(key=_rank, reverse=True)
        if deep and prelim:
            deep_best = max(deep.values())
            greedy_sig = max(prelim, key=lambda s: prelim[s])  # 1-ply 貪欲が選ぶ手
            greedy_deep = deep.get(greedy_sig)
            if greedy_deep is not None:
                regret = max(0.0, deep_best - greedy_deep)
    else:  # easy: scored は (score, move) の 1-ply 採点そのもの
        for s, m in sorted(scored, key=lambda x: x[0], reverse=True):
            cands.append({"move": _describe_move(manager, m), "prelim": round(s, 1), "deep": None})

    trace["difficulty"] = difficulty
    trace["turn"] = manager.turn_count
    pa = manager.pending_actor_action()  # action だけ＝軽量
    if pa:
        trace["pending_action"] = pa[1]
    trace["chosen"] = _describe_move(manager, chosen)
    trace["folded"] = folded
    trace["regret"] = round(regret, 1)
    trace["candidates"] = cands[:TRACE_TOPN]

    # 選んだ手の結果盤面で J値成分内訳＋読み筋を採る（trace 専用クローン・探索には不参加）。
    child = _apply_clone(manager, name, chosen, stop_at_select=True)
    if child is not None:
        comp: Dict[str, Any] = {}
        total = evaluate(child, name, see_opp_hand=see_opp_hand, out=comp)
        comp["total"] = round(total, 1)
        trace["j_components"] = comp
        if include_read_ahead:
            trace["read_ahead"] = _read_ahead_line(
                child, name, see_opp_hand, opp_public_only,
                manager.turn_count, HARD_HORIZON)


def _determinize_opponent(manager, me_name: str, rng):
    """`manager` のクローンを返し、**相手の伏せ手札を相手の山札＋手札プールから再サンプリング**する（公平化）。

    自分（`me_name`）の手札・場・山札順は不変（自分の手は実物＝返すプランが実ゲームで合法）。相手の手札枚数は
    保存し、中身だけ「相手のライブラリ（山札＋現手札）からランダムに同数」へ差し替える＝公開情報と整合する
    “ありえる手”＝チート除去。PIMC（§2.5.8）の各世界サンプリングが使う。journal 非作動の top-level 前提。
    """
    clone = manager.clone()
    opp = clone.p2 if clone.p1.name == me_name else clone.p1
    pool = list(opp.hand) + list(opp.deck)
    if not pool:
        return clone
    rng.shuffle(pool)
    n_hand = len(opp.hand)
    new_hand, new_deck = pool[:n_hand], pool[n_hand:]
    opp.hand[:] = new_hand
    opp.deck[:] = new_deck
    return clone


def _pimc_scored(manager, name: str, moves: List[Dict[str, Any]], k_worlds: int, rng,
                 collect=None) -> List[Tuple[float, Dict[str, Any]]]:
    """Phase 2 PIMC（決定化・docs/reports/cpu_strength_roadmap_20260622.md §4 Phase 2）。

    フェア化（相手手札を読まない）は「相手はカウンターも登場もできない」超楽観モデルに退化し、止まる
    攻撃を通ると誤読する別種の歪みを生む（Phase 1 計測＝損失は探索深さでなく情報限界）。PIMC は隠れ情報を
    **K 通りの"ありえる手"でサンプリング**し、各サンプルを完全情報 α-β で採点→**世界平均**で手を選ぶ。
    実手札は一度も見ない（`_determinize_opponent` が相手のライブラリから再サンプル）ので**チートせず**、
    平均が相手の防御強度を確率的に正しく見積もる＝楽観バイアスを埋める。

    決定論: 各世界の rng は親 `rng` から決定的に派生（同一 seed→同一手＝テスト/自己対戦の契約を維持）。
    戻り値は `decide` と同形 `[(avg_score, move)]`（move は実 manager の合法手 dict＝そのまま適用可）。
    """
    agg: Dict[Any, List[Any]] = {}  # sig -> [score 合計, move]
    order: List[Any] = []           # 初出順（決定論的な scored 順）
    for _w in range(k_worlds):
        wr = random.Random(rng.randrange(1 << 30))
        world = _determinize_opponent(manager, name, wr)
        # 各世界は「サンプルした相手手札」を完全情報として読む（see_opp_hand=True）＝実手札ではない＝フェア。
        sc = _scored_search(world, name, moves, see_opp_hand=True, opp_public_only=False,
                            collect=None, killer_state=None)
        for s, m in sc:
            sig = _move_sig(m)
            e = agg.get(sig)
            if e is None:
                agg[sig] = [s, m]
                order.append(sig)
            else:
                e[0] += s
    # 除数が k_worlds なのは、各世界が同一のルート手集合（`_scored_search` は入力 moves と同一の
    # 全 root sig を返す）を採点するため＝全 sig が全世界に現れる前提。`_scored_search` の返却集合を
    # 部分集合化するなら、ここを「その sig が出た世界数」で割る形へ直す必要がある。
    scored = [(agg[sig][0] / k_worlds, agg[sig][1]) for sig in order]
    if collect is not None:  # トレース用（PIMC では prelim=deep=世界平均）
        collect.setdefault("prelim", {}); collect.setdefault("deep", {})
        for avg, m in scored:
            collect["deep"][_move_sig(m)] = avg
            collect["prelim"].setdefault(_move_sig(m), avg)
    return scored


def decide(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
           moves: Optional[List[Dict[str, Any]]] = None,
           trace: Optional[Dict[str, Any]] = None, trace_read_ahead: bool = True,
           killer_state: Optional[Dict[int, List[tuple]]] = None,
           info_policy: str = DEFAULT_INFO_POLICY,
           pimc_worlds: int = PIMC_WORLDS_DEFAULT) -> Optional[Dict[str, Any]]:
    """`player` が取るべき次の 1 手を返す（合法手が無ければ None）。

    α-β＋ビーム。**easy/normal は廃止**（最強の α-β＝hard と MCTS＝expert の2系統に集約。`difficulty`
    引数は互換のため残すが分岐しない）。情報方針は `info_policy`（既定 "fair"＝相手手札を読まない出荷
    デフォルト／"cheat"＝旧 hard・相手手札も読む。`_INFO_POLICIES` 参照）で切り替える。
    `moves` を渡すとその候補集合から選ぶ（ガード driver が絞り込んだ手を渡す用途）。
    `trace`（任意・既定 None）で意思決定の診断情報を書き込む。
    `killer_state`（任意・④粒度b）で α-β killer 表を連続 decide 間で持ち越す。
    """
    rng = rng or random
    if moves is None:
        moves = manager.get_legal_actions(player)
    # 最上位が対象選択なら候補ごとに展開して最善対象を読み切る。
    is_selection = False
    sel = _selection_moves(manager, player.name)
    if sel:
        moves = sel
        is_selection = True
    if not moves:
        return None
    if len(moves) == 1:
        if trace is not None:
            trace["difficulty"] = difficulty
            trace["turn"] = manager.turn_count
            trace["chosen"] = _describe_move(manager, moves[0])
            trace["forced"] = "only_move"
            trace["candidates"] = [{"move": _describe_move(manager, moves[0]),
                                    "prelim": None, "deep": None}]
            trace["regret"] = 0.0
        return moves[0]
    moves = _prune_don_moves(manager, player.name, moves)  # B-2: 無意味なドン付与をルートから除外
    moves = _prune_futile_attacks(manager, player.name, moves)  # 倒せない/届かない無駄攻撃を除外

    name = player.name
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # トレース時のみ collect を渡して 1-ply prelim ／深掘り deep スコアを回収する（regret/候補表示用）。
    collect = {} if trace is not None else None

    # 情報方針（既定 fair＝相手手札を読まない出荷デフォルト・"cheat" で旧 hard のフルクローン）。
    see_opp_hand, opp_public_only = _resolve_info_policy(info_policy)
    if is_selection:
        # 対象選択（自分の確定効果の対象/枚数決定）は**即時盤面(1-ply)**が信頼できる信号。
        # 多 ply 先読みは「相手のターン中に発火した自分の誘発除去」等で価値が washout/逆転し
        # （例『相手のコスト1以下を2枚までKO』で 0〜1 枚に取りこぼす）、深掘りはむしろ有害。
        if collect is not None:
            collect.setdefault("prelim", {}); collect.setdefault("deep", {})
        scored = []
        for m in moves:
            s = _simulate_and_eval(manager, name, m, see_opp_hand=see_opp_hand)
            scored.append((s, m))
            if collect is not None:
                collect["prelim"][_move_sig(m)] = s
                collect["deep"][_move_sig(m)] = s
    elif pimc_worlds >= 2:
        # Phase 2 PIMC: K 決定化世界の完全情報採点を世界平均（チートせず隠れ情報を確率的に補う）。
        # 情報方針に依らずフェア（実手札は見ない）＝決定化が情報を供給する。
        scored = _pimc_scored(manager, name, moves, pimc_worlds, rng, collect=collect)
    else:
        scored = _scored_search(manager, name, moves, see_opp_hand=see_opp_hand,
                                opp_public_only=opp_public_only,
                                collect=collect, killer_state=killer_state)
    # 同点はランダムタイブレーク（決定論にしたい場合は呼び出し側で seed 済み rng を渡す）。
    rng.shuffle(scored)
    best_score, best_move = max(scored, key=lambda x: x[0])

    # 「何もしない（ターンを畳む）」を一級の選択肢として比較する。非ターン終了手が end を
    # _ACT_MARGIN を超えて上回らなければターンを畳む＝無意味な展開・不利アタック・効かない
    # ドン付与（いずれも改修後の評価では end とほぼ同値）を採らない（進行保証も兼ねる）。
    chosen = best_move
    folded = False
    if end_move is not None and best_move is not end_move:
        end_score = next((s for s, m in scored if m is end_move), None)
        margin = _ACT_MARGIN
        if end_score is not None and best_score <= end_score + margin:
            chosen = end_move
            folded = True

    if trace is not None:
        # トレース構築は追加クローンを作り、その効果解決がグローバル random を消費し得る。
        # 採点後の RNG 状態を保存→復元し、トレース有無でゲーム進行が分岐しないようにする
        # （決定論再現の保証。トレースはあくまで観測であって進行に影響させない）。
        _rng_state = random.getstate()
        try:
            _fill_decision_trace(trace, manager, name, difficulty, moves, scored, collect,
                                 chosen, folded, see_opp_hand, opp_public_only,
                                 include_read_ahead=trace_read_ahead)
        finally:
            random.setstate(_rng_state)
    return chosen


def decide_with_regret(manager, player, difficulty: str = "hard",
                       rng: Optional[random.Random] = None,
                       out: Optional[Dict[str, Any]] = None,
                       info_policy: str = DEFAULT_INFO_POLICY
                       ) -> Tuple[Optional[Dict[str, Any]], float]:
    """`decide` と同じ手を返しつつ、**greedy regret**（崖エラーの安価な代理・検証基盤・§2.5.3）も返す。

    regret = deep_value(深掘り最善手) − deep_value(1-ply 貪欲が選ぶ手)。
      - deep_value は多 ply 先読みスコア（`_scored_search`）。
      - 1-ply 貪欲手 = 事前選別スコア最大の手（＝浅い読みなら選ぶ手）。常に深掘り集合に入る（prelim 1位）。
    深掘りが浅い読みより良い手を見つけた量＝「1-ply 先読みでは崖に落ちる」局面の信号。常に >= 0。
    easy（1-ply 貪欲）や分岐の無い局面、深掘りスコアが取れない場合は regret=0.0 を返す。

    `out`（任意・既定 None＝完全に無オーバーヘッド）が渡されると、**value-realization gap**（§2.5.3）の
    計測用に `out["chosen_deep"]`＝採用手の深掘りスコア・`out["deep_best"]`＝深掘り最善値を記録する
    （取れない＝単一手/easy/深掘り無し では未設定）。out=None 時は採点・返り値とも従来と完全同値。
    """
    rng = rng or random
    moves = manager.get_legal_actions(player)
    sel = _selection_moves(manager, player.name)
    if sel:
        moves = sel
    if not moves:
        return None, 0.0
    if len(moves) == 1:
        return decide(manager, player, difficulty, rng, moves=moves,
                      info_policy=info_policy), 0.0

    name = player.name
    moves = _prune_don_moves(manager, name, moves)  # B-2: ルート手集合を decide と一致させる（regret 整合）
    moves = _prune_futile_attacks(manager, name, moves)  # 倒せない/届かない無駄攻撃を除外（decide と一致）
    see_opp_hand, opp_public_only = _resolve_info_policy(info_policy)
    collect: Dict[str, Any] = {}
    _scored_search(manager, name, moves, see_opp_hand=see_opp_hand, opp_public_only=opp_public_only,
                   collect=collect)
    move = decide(manager, player, difficulty, rng, moves=moves, info_policy=info_policy)
    deep = collect.get("deep", {})
    prelim = collect.get("prelim", {})
    regret = 0.0
    if deep and prelim:
        deep_best = max(deep.values())
        greedy_sig = max(prelim, key=lambda s: prelim[s])  # 1-ply 貪欲が選ぶ手
        greedy_deep = deep.get(greedy_sig)
        if greedy_deep is not None:
            regret = max(0.0, deep_best - greedy_deep)
        if out is not None:
            out["deep_best"] = deep_best
            if move is not None:
                cd = deep.get(_move_sig(move))
                if cd is not None:
                    out["chosen_deep"] = cd
    return move, regret


def decide_guarded(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
                   mem: Optional[Dict[str, Any]] = None,
                   trace: Optional[Dict[str, Any]] = None, trace_read_ahead: bool = True,
                   info_policy: str = DEFAULT_INFO_POLICY,
                   pimc_worlds: int = PIMC_WORLDS_DEFAULT) -> Optional[Dict[str, Any]]:
    """ターン内メモリ `mem` を用いた終了保証つきの意思決定。

    `mem` は呼び出し側が対局ごとに保持する dict（ステートレスな /cpu/step でも CPU_GAMES に
    保持して渡す）。同一ターン内で **取った手の総数が TURN_ACTION_CAP を超えたら強制 TURN_END** とし、
    「効果に per-turn 制限が無い/付け忘れ」のカードでも CPU ターンが必ず終わることを最終保証する。

    かつては同一起動効果の繰り返しを REPEAT_CAP で除外していたが、根因（自己レストコストの
    `cost_optional` 誤判定＋`ref_id='self'` 充足の食い違い）をエンジン側で修正し、起動メインが
    エンジンの正規ゲート（コスト充足・ターン使用回数）で自己制限するようになったため撤去した
    （2026-06-27 計測で REPEAT_CAP 発火 0/2231）。`mem["killers"]`（探索の手順序）は存続。
    """
    rng = rng or random
    if mem is None:
        mem = {}
    if mem.get("turn") != manager.turn_count:
        mem["turn"] = manager.turn_count
        mem["total"] = 0
        mem["killers"] = {}   # ④粒度b: killer 表はターン内でのみ持ち越す（ターン跨ぎの古い表は破棄）

    moves = manager.get_legal_actions(player)
    if not moves:
        return None
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # 総数キャップ（最終的な終了保証）: 上限超過ならターンを畳む（畳めない＝対話中等なら通常選択）。
    if end_move is not None and mem.get("total", 0) >= TURN_ACTION_CAP:
        if trace is not None:
            trace["difficulty"] = difficulty
            trace["turn"] = manager.turn_count
            trace["chosen"] = _describe_move(manager, end_move)
            trace["forced"] = "turn_action_cap"
            trace["candidates"] = []
            trace["regret"] = 0.0
        return end_move

    # ④粒度b: killer 表を mem に保持して連続 decide 間で持ち越す（reorder は集合不変＝安全だが実測で
    # 中立〜微減＝既定 OFF。`_USE_PV_CROSS_DECIDE` 有効時のみ供給。OFF は killer_state=None＝粒度a のみ）。
    ks = mem.setdefault("killers", {}) if (_USE_PV_ORDER and _USE_PV_CROSS_DECIDE) else None
    move = decide(manager, player, difficulty, rng, moves=moves,
                  trace=trace, trace_read_ahead=trace_read_ahead, killer_state=ks,
                  info_policy=info_policy, pimc_worlds=pimc_worlds)
    if move is not None:
        mem["total"] = mem.get("total", 0) + 1
    return move


def plan_turn(manager, name: str, difficulty: str = "hard", rng=None,
              mem: Optional[Dict[str, Any]] = None,
              info_policy: str = DEFAULT_INFO_POLICY,
              pimc_worlds: int = PIMC_WORLDS_DEFAULT) -> List[Dict[str, Any]]:
    """Phase 3 ①（計画キャッシュ）: 相手の介入（ブロック/カウンター等）が入るまで、または TURN_END
    までの自分(`name`)の**連続行動列**をクローン上で計画する。

    クローン上で `decide_guarded` を逐次適用して列を作るため、本物の per-action 流（同じ `rng`/`mem` を
    渡す）と**ビット等価**になる: 介入の無い区切り内では各手番の (盤面, rng, mem) が完全一致するため、
    計画の手列・rng/mem 消費は per-action と同一（前倒しで全部計算→以降は replay という時間配分だけが違う）。
    「自分が連続で動ける区切り」（ターン開始や相手介入の直後）で 1 回計算してキャッシュし、以降の手番は
    キャッシュ参照で即時化する＝**待ちを 1 回に集約**する（体感最適化）。相手の応手で前提が崩れる介入点で
    区切るので、そこから先は実結果が出てから再計画する。

    戻り値: 行動 move dict のリスト（末尾は TURN_END か、相手介入の直前まで）。`mem`/`rng` は per-action と
    同じものを渡すと、計画適用後の状態が本物の逐次実行と一致する（呼び出し側で replay 時は decide を呼ばない）。
    """
    from . import action_api
    rng = rng or random
    clone = manager.clone()
    actions: List[Dict[str, Any]] = []
    for _ in range(TURN_ACTION_CAP + 8):  # 安全上限（decide_guarded のキャップと整合・暴走防止）
        pa = clone.pending_actor_action()
        if not pa or pa[0] != name:
            break  # 相手の手番/介入点（SELECT_BLOCKER/SELECT_COUNTER 等）＝区切り
        actor = _player_by_name(clone, name)
        mv = decide_guarded(clone, actor, difficulty, rng, mem=mem, info_policy=info_policy,
                            pimc_worlds=pimc_worlds)
        if mv is None:
            break
        actions.append(mv)
        if mv.get("kind") == "battle":
            action_api.apply_battle_action(clone, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(clone, actor, mv["action_type"], mv.get("payload", {}))
        if mv.get("action_type") == "TURN_END":
            break
    return actions


def decide_cached(manager, player, difficulty: str = "hard", rng=None,
                  mem: Optional[Dict[str, Any]] = None, cache: Optional[Dict[str, Any]] = None,
                  info_policy: str = DEFAULT_INFO_POLICY,
                  pimc_worlds: int = PIMC_WORLDS_DEFAULT) -> Optional[Dict[str, Any]]:
    """Phase 3 ① 配線: 計画キャッシュ付き decide（**本番の体感最適化専用**）。

    `cache` は対局ごとに保持する dict（`{"queue": [...残りの計画手...]}`）。
      - **キャッシュヒット**: 次の計画手が現局面で**合法**なら即返す（探索なし＝即時 replay）。
      - **キャッシュミス/前提崩れ**: `plan_turn` でセグメント（相手介入/TURN_END まで）を計画して
        キャッシュし、先頭手を返す。計画手が現局面で不正なら破棄して通常の `decide_guarded` へ。

    **合法性検証で常に安全**（rng がズレてもキャッシュ手が不正なら通常 decide に落ちる＝不正手は打たない）。
    ただし `plan_turn` のクローン適用と本番 replay の実適用でシャッフル等の rng 消費が前後しうるため
    **決定性は保証しない＝テスト/自己対戦は `decide_guarded` を使う**こと（本番は決定性不要）。
    """
    rng = rng or random
    if cache is None:
        cache = {}
    legal = manager.get_legal_actions(player)
    if not legal:
        return None
    legal_by_sig = {_move_sig(m): m for m in legal}

    q = cache.get("queue")
    if q:
        nxt_sig = _move_sig(q[0])
        if nxt_sig in legal_by_sig:
            cache["queue"] = q[1:]
            return legal_by_sig[nxt_sig]  # 現局面の move（uuid 整合）を返す
        cache["queue"] = None  # 前提崩れ＝破棄して再計画

    actions = plan_turn(manager, player.name, difficulty, rng, mem=mem, info_policy=info_policy,
                        pimc_worlds=pimc_worlds)
    if actions:
        first_sig = _move_sig(actions[0])
        if first_sig in legal_by_sig:
            cache["queue"] = actions[1:]
            return legal_by_sig[first_sig]
    # 計画が空/先頭不正＝安全側で通常 decide（rng/mem は plan_turn で進行済みのため二重進行に注意だが
    # 本番専用＝決定性非依存。guard は安全網なので軽微な前後は許容）。
    cache["queue"] = None
    return decide_guarded(manager, player, difficulty, rng, mem=mem, info_policy=info_policy,
                          pimc_worlds=pimc_worlds)
