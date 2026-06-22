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


# 評価重み（盤面 1000=パワー1段相当に正規化）
W_LIFE = 6000.0          # ライフ 1 枚の基礎価値（膝＝薄域。最重要）
# 高ライフ域（膝超）のライフ 1 枚の基礎価値（concave 化・§2.5.3）。OPCG ではライフは「序盤は能動的に
# 受けてカウンターを温存する」資源で、薄いほど 1 枚の限界価値が高い（設計意図）。従来は線形（全枚数に
# W_LIFE）で高ライフでも 1 枚＝6000＝満額だったため、自ライフ5枚の序盤でも1点を守るためにカウンターを
# 3枚浪費する過剰防御を招いた。膝（life_knee）までは W_LIFE 満額・膝超は W_LIFE_HIGH（< W_LIFE）に減額し、
# 「高ライフの 1 点は安い（受ける）／薄ライフの 1 点は高い（守る）」の正しいカーブにする。
W_LIFE_HIGH = 2500.0
W_LIFE_LOW = 4000.0      # 希少域（最初の 2 枚）への上乗せ＝非線形・45[J] ラインの危険
# C-3（§2.5.3）: ライフ薄域上乗せの膝位置（この枚数までを厚く守る）。自他で別カーブ＝既定 2、攻め寄りの
# 相手と対面する自ライフ（守備）のみ 3 へ上げてレース耐性を厚く見る（profile 無し＝両側 2＝従来同値）。
_LIFE_KNEE_DEFAULT = 2
_LIFE_KNEE_AGGRO_MATCHUP = 3
_AGGRO_MATCHUP_THRESHOLD = 0.6   # 相手プロファイルの aggro_lean がこれ以上で「攻め対面」と見なす
W_HAND = 700.0           # 手札 1 枚の基礎価値
W_COUNTER = 0.6          # 手札のカウンター値 1 点あたり（防御リソース）
# 注（A2・2026-06）: 「コスト低減の資源価値化」= 次ターン手出し可ボーナス（旧 W_HAND_PLAYABLE）は、
# ablation A/B（フル vs 当該項なし・hard 42局）で勝率 0.500＝**無寄与**（探索が結果盤面で既に拾える）と
# 判明したため撤去した。経緯は docs/SPEC.md §2.5.3。
W_FIELD_COUNT = 1500.0   # 場のキャラ 1 体の存在価値
W_STAGE_COUNT = 800.0    # ステージ 1 枚の存在価値（Phase1.5）。戦闘に出ない永続リソース＝キャラより軽い。
W_FIELD_POWER = 0.3      # 場の有効パワー（戦闘で意味を持つ上限までを線形評価。素点でなく閾値性を尊重）
W_DON_ACTIVE = 200.0     # アクティブドン!! 1 枚
W_BLOCKER = 1200.0       # ブロッカー 1 体（最終防御）
W_ATTACKER = 400.0       # 「このターン実際に攻撃できる」アクティブキャラ＝攻め圧（相手 +1[J] 機会）
W_WIN = 1.0e9            # 勝敗
W_LIFE_AGGRO_K = 0.5     # リーダー推測: 相手の攻め寄り度 1.0 で自分ライフ重視を最大 +50%（§2.5.4）

# J値（白＝デッキ残＋トラッシュ）の決定境界。デッキ切れ（J=0）でドロー不能＝敗北なので、
# 自デッキ残が危険域に入るほど非線形に減点する＝相手を削り切る／自滅ドローを避ける動機。
# 黒リソース（ライフ/手札/場）は白の相補なので素点は据え置き、ここでは境界の非線形分だけを足す。
W_DECK_DANGER = 1500.0   # 危険域のデッキ残 1 枚あたりの減点
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
# C-4（§2.5.3）: 打ち切り settle 葉の不確実性ディスカウント（信頼度）。settle は予算切れの局面を**既定解決**で
# 無理に静止点へ整流して採点する＝探索で確かめた値ではない。勝敗未確定の settle 値はこの係数で中立（盤面差
# 0＝互角）へ寄せ、既定解決の選択が評価を不当に振らせるのを抑える（既定解決の中立化）。1.0=従来（割引なし）。
_SETTLE_CONFIDENCE = 0.9

# 自デッキ勝ち筋プラン（cpu_self_plan・docs/SPEC.md §2.5.5）の評価項。プラン未指定（plan=None）では
# 一切作動せず現行挙動と完全同値（乗数は既定 1.0／逆算項は plan が無ければ 0）。normal/hard でのみ供給。
_LOW_IMPACT_POWER = 5000          # 素パワーがこの値未満＝素ではリーダー(5000)に打点が通らない置物
_RELEVANT_KEYWORDS = ("ブロッカー", "速攻", "ダブルアタック")  # これらを持つ体は「置物」扱いしない
_CLOSER_W = 2000.0                # 逆算リーサル: 相手を削り切れる本数を持つ盤面への加点
_NEAR_W = 200.0                   # 同上: 削り切るには足りないが届く攻撃 1 本あたりの軽い加点
_MILE_DMG_W = 1200.0             # マイルストーン: 想定クロックより相手ライフが先行して減っている分
_MILE_RES_W = 220.0             # マイルストーン: リソース差（手札＋場の枚数差）1 枚あたり
_J_SCHED_W = 200.0              # マイルストーン(J値版): スケジュール遵守度（実測 J値差 − 理想差）1 あたり（§2.5.5）

# 脅威/キーワード資産の価値（§2.5.6・対面プランのルールベース実現）。場のキャラが持つ「除去すべき
# 脅威性／温存すべき資産性」をカードデータから加点する。両側に対称適用するので、相手の脅威キャラは
# opp 側スコアを押し上げ→除去すると自分の評価が大きく上がる→① の単一対象探索が最善の脅威を狙う。
# ブロッカーは既に W_BLOCKER で計上済みなのでここには含めない。プラン供給時のみ作動（plan=None 不変）。
W_KW_DOUBLE = 1200.0     # ダブルアタック: リーダー打点が2倍＝攻め脅威
W_KW_RESIST = 900.0      # 効果耐性「KOされない」: 除去されにくい永続的な体
W_KW_RUSH = 250.0        # 速攻: 即時の攻め圧（攻撃タイミングで一部反映済みのため小さめ）
W_KW_BANISH = 300.0      # バニッシュ: KO時にライフ/トリガーを与えない攻め強化
W_KW_UNBLOCK = 900.0     # アンブロッカブル【ブロック不可】: ブロッカーで止められず確実にリーダーへ通る攻め脅威
_RESIST_CUE = "KOされない"
_KEYWORD_ASSETS = (("ダブルアタック", W_KW_DOUBLE), ("速攻", W_KW_RUSH), ("バニッシュ", W_KW_BANISH))

# 注（A2・2026-06）: 地平線越えの「毎ターン価値を生むエンジン能力」プレミアム（旧 W_RECUR_ENGINE／
# _recurring_engine／_RECUR_TRIGGERS）は、ablation A/B（フル vs 当該項なし・hard 42局）で勝率 0.452＝
# **正味マイナス（バイアス）**と判明したため撤去した。horizon=4 の探索が将来発動を結果盤面で既に拾えており、
# 静的プレミアムは二重計上で評価を歪めていた。経緯は docs/SPEC.md §2.5.3。

# C-2（§2.5.3）: テレグラフ致死の減点。葉（相手ターン開始＝相手の攻撃が目前）で「相手の次ターンの有効打点
# ≥ 自残ライフ」なら、受け切れず負ける telegraph として減点する。W_WIN(1e9) に対し十分小さい＝**本物の
# リーサル発見（±W_WIN）は決して上書きしない**＝引き分け帯で守り（ブロッカー温存・脅威除去・ライフ獲得）へ
# 寄せるだけ。プラン供給時のみ作動（plan=None 完全同値）。過剰防御を避けるため打点見積りは素パワー（保守的）。
W_TELEGRAPH_LETHAL = 6000.0

# B（楽観是正・§2.5.3）: settle（予算切れの打ち切り葉）は相手のターン開始で止めて**静的**に採点する＝
# 相手の反撃を読まない＝**動いた側に楽観バイアス**（殴られる直前でスナップショット）。読み切れなかった
# 葉に限り、相手の次ターンの「自分が受け切れない打点（届く攻撃本数 − 自アクティブブロッカー）」1 本あたり
# をこの重みで減点して楽観を是正する。致死（reach ≥ 自ライフ）は C-2 telegraph が `evaluate` 内で既に
# W_TELEGRAPH_LETHAL を計上済みなので**致死未満のみ**ここで扱い二重計上を避ける。保守的に素パワーで測る。
# プラン供給時かつ相手ターン静止点でのみ作動（plan=None / 自分の手番では完全に不作動＝従来同値）。
W_SETTLE_PRESSURE = 2500.0


def _is_unblockable(c, etext: str) -> bool:
    """このキャラ自身が【ブロック不可】（アンブロッカブル）か（バッチA-1・§2.5.6）。

    キーワード集合に "ブロック不可" は載らない（マスタ未格納）ため、**自前キーワードのテキスト**で判定する:
    自前は `【ブロック不可】（このカードはブロックされない）` のように直後にリマインダ括弧が付く。一方、他者へ
    付与する句は `…【ブロック不可】を得る` で括弧が直後に来ない＝区別できる（付与カードを誤検出しない）。
    付与で timed_keywords に載った場合は `has_keyword` でも拾う（将来の付与解決に追従）。
    """
    try:
        if c.has_keyword("ブロック不可"):
            return True
    except Exception:
        pass
    return "【ブロック不可】(" in (etext or "").replace("（", "(")


def _threat_value(c, atk_mult: float = 1.0, def_mult: float = 1.0) -> float:
    """場のキャラ 1 体の脅威/資産価値（キーワード＋効果耐性＋アンブロッカブル）。カードデータから算出（§2.5.6）。

    A-2: 攻撃的キーワード（ダブルアタック/速攻/バニッシュ/アンブロッカブル）は `atk_mult`、防御的キーワード
    （効果耐性「KOされない」）は `def_mult` でアーキタイプ依存にスケール（aggro=攻め重視／control=守り重視）。
    既定 1.0＝従来値（plan 無し・midrange は完全同値）。
    """
    atk = 0.0  # 攻撃的キーワード資産（atk_mult でスケール）
    for kw, w in _KEYWORD_ASSETS:
        try:
            if c.has_keyword(kw):
                atk += w
        except Exception:
            pass
    m = getattr(c, "master", None)
    etext = (getattr(m, "effect_text", "") or "") if m is not None else ""
    if _is_unblockable(c, etext):
        atk += W_KW_UNBLOCK
    deff = W_KW_RESIST if _RESIST_CUE in etext else 0.0  # 防御的キーワード資産（def_mult でスケール）
    return atk * atk_mult + deff * def_mult


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


def _is_low_impact(c) -> bool:
    """「効果なし・素パワー<5000・関連キーワード無し」の置物キャラか（プラン重みの割引対象）。

    効果（abilities もしくは効果テキスト）か、戦闘で意味を持つキーワード（ブロッカー/速攻/ダブル
    アタック）を持つ体は、たとえ低パワーでも置物扱いしない＝割引しない。
    """
    m = getattr(c, "master", None)
    if m is None:
        return False
    if getattr(m, "abilities", None) or (getattr(m, "effect_text", "") or "").strip():
        return False
    if any(c.has_keyword(k) for k in _RELEVANT_KEYWORDS):
        return False
    return (getattr(m, "power", 0) or 0) < _LOW_IMPACT_POWER


# 時間割引（独立トラック・§2.5.3）: 地平線外の盤面ポテンシャル（場の存在価値 W_FIELD_COUNT）は、残りターンで
# 使い切れて初めて価値になる。静的重みは時間非依存なので、**残りゲーム長が短い（レース終盤）ほど場の存在価値を
# 割り引く**。スコープは全項リスケールではなく**地平線外の盤面価値の割引**に限定（ライフ＝即時価値・逆算リーサル
# /クロック＝実レース進捗・ブロッカー＝即時防御・カウンターは割り引かない）。残ターン代理＝先に死ぬ側のライフ
# min(自,相手)。プラン供給時のみ作動（plan=None 完全同値）。
_TEMPO_FULL_TURNS = 4.0   # 残ターン代理がこれ以上で場の存在価値を満額評価（未満は線形に割引）
_TEMPO_FLOOR = 0.3        # 割引の下限（0 ライフでも存在価値を全否定しない＝盤面が無価値化して挙動が壊れるのを防ぐ）


def _board_tempo_factor(me, opp) -> float:
    """地平線外の盤面ポテンシャル（場の存在価値）の時間割引係数（§2.5.3）。

    残りゲーム長の代理＝先に死ぬ側のライフ `min(len(me.life), len(opp.life))`。これが短い（レース終盤）ほど
    場のキャラの「将来攻撃する潜在価値」は使い切れず、割引する。`_TEMPO_FULL_TURNS` 以上で満額（1.0）、未満で
    線形に下げ、`_TEMPO_FLOOR` で下限を打つ。両者対称（ゲーム長はどちらの盤面にも共通）。
    """
    turns_left = min(len(me.life), len(opp.life))
    return max(_TEMPO_FLOOR, min(1.0, turns_left / _TEMPO_FULL_TURNS))


def _incoming_reach(me, opp) -> int:
    """相手の次ターンに自リーダーへ届く**受け切れない**打点本数（>=0）。C-2 telegraph と B（settle 楽観是正）
    の共通土台。

    相手の次ターンは場が全アクティブ化するので `is_rest` は無視し、リーダー＋場のうち**素パワーが自リーダー
    に届く**体を数える（素パワー＝保守的＝don 付与での押し上げは数えない＝過剰防御を避ける）。自分の
    アクティブブロッカー数（各 1 本を止める）を控除して 0 で床打ちする。
    """
    try:
        my_leader_pw = float(me.leader.get_power(False)) if me.leader is not None else 5000.0
    except Exception:
        my_leader_pw = 5000.0
    reach = 0
    units = list(opp.field) + ([opp.leader] if opp.leader is not None else [])
    for c in units:
        try:
            pw = float(c.get_power(False))
        except Exception:
            pw = float(getattr(getattr(c, "master", None), "power", 0) or 0)
        if my_leader_pw > 0 and pw >= my_leader_pw:
            reach += 1
    return max(0, reach - _active_blocker_count(me))


def _telegraph_lethal(me, opp) -> bool:
    """C-2（§2.5.3）: 相手の次ターンの有効打点が自残ライフ以上で、受け切れず負ける telegraph か。

    `_incoming_reach`（届く打点本数 − 自アクティブブロッカー）≥ 自残ライフ なら telegraph 致死。
    """
    my_life = len(me.life)
    if my_life <= 0:
        return False
    return _incoming_reach(me, opp) >= my_life


def _own_life_knee(profile) -> int:
    """自ライフ（守備）の非線形膝位置（C-3・§2.5.3）。攻め寄りの相手と対面するときだけ膝を 3 へ上げる。

    対面想定は相手モデル `profile.aggro_lean`（リーダー推測・§2.5.4）から取る。profile 無し＝既定 2＝従来同値。
    """
    if profile is not None and getattr(profile, "aggro_lean", 0.0) >= _AGGRO_MATCHUP_THRESHOLD:
        return _LIFE_KNEE_AGGRO_MATCHUP
    return _LIFE_KNEE_DEFAULT


def _side_score(p, is_turn: bool, power_cap: float, include_counter: bool = True,
                hand_factor: float = 1.0, life_factor: float = 1.0,
                body_factor: float = 1.0, attacker_factor: float = 1.0,
                counter_factor: float = 1.0, threat_aware: bool = False,
                idle_don_factor: float = 1.0,
                threat_atk_mult: float = 1.0, threat_def_mult: float = 1.0,
                life_knee: int = _LIFE_KNEE_DEFAULT,
                field_count_factor: float = 1.0,
                engine_aware: bool = False,
                out: Optional[Dict[str, float]] = None) -> float:
    """1 プレイヤー側の素点（J値理論ベース：黒リソースの重み付き和＋白の境界リスク）。

    `power_cap` は対面の最硬防御パワー＝有効パワーの上限（`_effective_power`）。これにより
    「届かない/過剰なドン付与」が静的にはほぼ無加点となる。
    `include_counter=False` のとき手札はカウンター値を読まず枚数のみで評価する
    （相手手札の中身を見ない「公開情報のみ」の情報方針＝easy/normal 用）。
    `hand_factor`/`life_factor` はリーダー推測プロファイルによる手札防御価値・ライフ重視度の倍率
    （§2.5.4。プロファイル無し時は 1.0）。
    `life_knee`（C-3・§2.5.3）はライフ薄域上乗せ（`W_LIFE_LOW`）を立ち上げる枚数の上限＝非線形の膝位置。
    既定 2＝従来。攻め寄りの相手と対面する自ライフ（守備）は膝を 3 へ上げ、レース下での 3 枚目までを
    厚く守る（クロック側＝相手ライフは既定 2 のまま＝自他で別カーブ）。
    """
    score = 0.0
    # out（任意・既定 None＝完全に無オーバーヘッド）が渡されると、成分内訳を `out[キー] += 値` で
    # 記録する（CPU 思考トレース・診断用）。`score +=` 行は一切変えないので out=None 時の挙動・採点は
    # 従来と完全同値（ベースライン不変）。
    def _emit(k: str, v: float):
        if out is not None:
            out[k] = out.get(k, 0.0) + v

    # ライフ: 非線形（薄いほど 1 枚の限界価値が高い）。膝位置（life_knee）までは W_LIFE 満額・膝超は
    # W_LIFE_HIGH(<W_LIFE) に減額する concave カーブ＋膝までを W_LIFE_LOW で厚く上乗せ（二重の非線形）。
    # 高ライフの 1 点を満額評価して序盤に過剰防御（カウンター浪費）するのを防ぐ（§2.5.3）。
    life_n = len(p.life)
    near = min(life_n, life_knee)            # 薄域（膝以下）＝満額で厚く守る
    far = max(0, life_n - life_knee)         # 高ライフ域（膝超）＝安い（受ける）
    life_base = (near * W_LIFE + far * W_LIFE_HIGH) * life_factor
    score += life_base
    _emit("life", life_base)
    score += near * W_LIFE_LOW * life_factor
    _emit("life_low", near * W_LIFE_LOW * life_factor)

    # 手札: 枚数 ＋（公開方針でなければ）カウンター値（防御に回せる力＝相手の +1[J] を打ち消す資源）。
    score += len(p.hand) * W_HAND * hand_factor
    _emit("hand", len(p.hand) * W_HAND * hand_factor)
    if include_counter:
        for c in p.hand:
            try:
                score += (c.current_counter or 0) * W_COUNTER * counter_factor
                _emit("hand_counter", (c.current_counter or 0) * W_COUNTER * counter_factor)
            except Exception:
                pass

    # 白（J）の決定境界: 自デッキ残がデッキ切れ（J=0・ドロー不能＝敗北）へ近づくほど非線形に減点。
    score -= max(0, DECK_DANGER - len(p.deck)) * W_DECK_DANGER
    _emit("deck_danger", -max(0, DECK_DANGER - len(p.deck)) * W_DECK_DANGER)

    # ドン!!（アクティブ）。`is_turn=False`（自分の手番でない静止点＝葉）では、浮いたアクティブドンは
    # 防御に使えない（OPCG はドンを防御に付与できない）ので、プラン由来の `idle_don_factor`(<1.0) で
    # 減価する＝「両枝でクロック同値→ドンの床でタイブレーク→握る」という余剰ドン温存を断つ（B-1・§2.5.3）。
    # 自分の手番中（is_turn=True）は付与でパワーに変換できる生きた資源なので減価しない。plan 無し時は 1.0。
    don_score = len(p.don_active) * W_DON_ACTIVE
    if not is_turn and idle_don_factor != 1.0:
        don_score *= idle_don_factor
    score += don_score
    _emit("don", don_score)

    # 場のキャラ: 存在価値 ＋ 有効パワー ＋ ブロッカー（最終防御）＋ 攻め圧（実際に攻撃できる体のみ）。
    # 存在価値はプランで「効果なし低パワーの置物」のみ割り引く（body_factor）＝デッキ依存の置物許容度。
    for c in p.field:
        # 時間割引（§2.5.3）: 場の存在価値（地平線外の盤面ポテンシャル）は残りゲーム長で割り引く。
        ev = W_FIELD_COUNT * field_count_factor
        if body_factor != 1.0 and _is_low_impact(c):
            ev *= body_factor
        score += ev
        _emit("field_count", ev)
        # 脅威/キーワード資産（ダブルアタック・効果耐性・速攻・バニッシュ・アンブロッカブル）。両側対称・
        # プラン時のみ。A-2: 攻め/守りキーワードをアーキタイプ依存にスケール（aggro=攻め重視/control=守り重視）。
        if threat_aware:
            tv = _threat_value(c, threat_atk_mult, threat_def_mult)
            score += tv
            _emit("field_threat", tv)
        try:
            pw = c.get_power(is_turn)
        except Exception:
            pw = c.master.power or 0
        score += _effective_power(pw, power_cap) * W_FIELD_POWER
        _emit("field_power", _effective_power(pw, power_cap) * W_FIELD_POWER)
        if not c.is_rest:
            # 攻め圧は「このターン実際に攻撃できる体」に限る。自ターンの召喚酔い（速攻なし）は
            # 今ターン攻撃できないので加点しない＝意味のない小型展開で攻め圧を水増ししない。
            sick = getattr(c, "is_newly_played", False) and not c.has_keyword("速攻")
            if not (is_turn and sick):
                score += W_ATTACKER * attacker_factor
                _emit("attacker", W_ATTACKER * attacker_factor)
            if c.has_keyword("ブロッカー"):
                score += W_BLOCKER
                _emit("blocker", W_BLOCKER)

    # リーダーの可変状態（Phase1.5・②・§2.5.3 評価観点の穴埋め）: リーダーの**有効パワー**を戦闘強度として
    # 価値化する。リーダーは存在/能力はゲーム中**不変＝探索全ノードで定数＝手選択(argmax)に無影響**なので
    # 加点しない（存在 W_FIELD_COUNT・攻め圧 W_ATTACKER も定数のため除外）。一方リーダーの有効パワーは
    # **ドン!!付与/バフで変動する＝可変**。従来 `_side_score` はリーダーを一切読まず、リーダーへのドン付与
    # （大型アタックを作る筋）が静的には「アクティブドンを失うだけの純損」に見えていた。場のキャラと同じく
    # 有効パワー（対面最硬防御 `power_cap` までを線形・超過減衰）を `W_FIELD_POWER` で加点し、ドン付与/バフの
    # 戦闘価値（相手の硬い体を KO/トレードできる）を拾う。クロック（ライフ削り）は `_plan_progress` の reach が
    # 別軸で評価済み＝二重計上を避ける（こちらは戦闘 KO/トレード軸）。プラン供給時のみ作動（engine_aware＝
    # plan is not None）＝plan=None では一切作動せず現行挙動と完全同値。
    if engine_aware and getattr(p, "leader", None) is not None:
        try:
            lpw = p.leader.get_power(is_turn)
        except Exception:
            lpw = getattr(getattr(p.leader, "master", None), "power", 0) or 0
        lev = _effective_power(lpw, power_cap) * W_FIELD_POWER
        score += lev
        _emit("leader_power", lev)

    # ステージ（Phase1.5・§2.5.3 評価観点の穴埋め）: 場の永続リソース（`p.stage`）の**存在価値**を加点する。
    # ステージはキャラと違い攻撃/ブロックに出ない＝パワー/攻め圧/ブロッカーは持たないため存在価値のみ
    # （連続バフ＝聖地マリージョア等のコスト/パワー補正はキャラ側のパワー/コストに既に反映済み＝二重計上を避ける）。
    # 存在は時間割引（field_count_factor）。ステージは可変・非対称（プレイ/除去/張替で増減）＝手選択に効く。
    # プラン供給時のみ作動（engine_aware＝plan is not None）＝plan=None では一切作動せず現行挙動と完全同値。
    # （エンジンプレミアムは A2 で撤去＝撤去理由は §2.5.3。）
    if engine_aware:
        st = getattr(p, "stage", None)
        if st is not None:
            sv = W_STAGE_COUNT * field_count_factor
            score += sv
            _emit("stage_count", sv)
    return score


# C-1: 逆算リーサルの false lethal 抑制（§2.5.3）。隠れカウンター緩衝（power）を「攻撃 1 本を無効化し得る
# 回数」へ粗く換算する単位。典型カウンター（1000〜2000）で限界的なアタックが 1 本止まる、を 1 セーブと数える。
_COUNTER_SAVE_UNIT = 2000.0


def _plan_progress(manager, me, opp, is_my_turn: bool, plan, profile=None) -> float:
    """勝利状態からの逆算サブゴール項（§2.5.5）。プラン未指定なら 0。

    - 逆算リーサル: 「相手リーダーに打点が通るアクティブ体」を数え、相手ライフを削り切れる本数を
      持つ盤面を加点する（探索の最短リーサル認識を、非終端ノードでも“止めの形”へ誘導する）。
      C-1（§2.5.3）: reach 本数から**相手の可視ブロッカー数**（各 1 本を止める）と、**隠れカウンター
      緩衝**（`profile` の `defense_factor` 由来＝公開情報ベリーフで更新した推定 power を `_COUNTER_SAVE_UNIT`
      でセーブ回数化・相手手札枚数で上限）を控除し、**割引後 reach** で止め/接近を判定する（false lethal の
      soft 精度改善）。`profile` 無しは控除 0＝従来どおり（plan 単体テストは不変）。
    - マイルストーン: アグロ＝想定クロックより相手ライフが先行して減っているほど加点／コントロール＝
      **J値スケジュール遵守度**（実測 (相手J値−自分J値) が構成由来の理想差 `plan.delta_schedule[t]` を
      上回る分を加点。`delta_schedule` 空＝従来の手札＋場リソース差へフォールバック）。攻め寄り度
      aggro_lean で両者をブレンドする。J値は白＝デッキ残＋トラッシュの枚数のみ参照＝フェア。
    """
    if plan is None:
        return 0.0
    opp_life = len(opp.life)
    # 相手リーダーの素の防御パワー（自分の付与ドンは乗らない＝攻撃が通る閾値）。
    opp_leader_pw = 0.0
    if opp.leader is not None:
        try:
            opp_leader_pw = float(opp.leader.get_power(False))
        except Exception:
            opp_leader_pw = float(getattr(opp.leader.master, "power", 0) or 0)
    # 相手リーダーに打点が通る、今/将来攻撃できるアクティブ体の本数。
    reach = 0
    units = list(me.field) + ([me.leader] if me.leader is not None else [])
    for c in units:
        if getattr(c, "is_rest", False):
            continue
        sick = getattr(c, "is_newly_played", False) and not c.has_keyword("速攻")
        if is_my_turn and sick:
            continue
        try:
            pw = float(c.get_power(is_my_turn))
        except Exception:
            pw = float(getattr(getattr(c, "master", None), "power", 0) or 0)
        if opp_leader_pw > 0 and pw >= opp_leader_pw:
            reach += 1

    # C-1: 割引後 reach（可視ブロッカー＋隠れカウンターセーブを控除）で false lethal を抑制。
    visible_blockers = 0
    for c in opp.field:
        try:
            if not getattr(c, "is_rest", False) and c.has_keyword("ブロッカー"):
                visible_blockers += 1
        except Exception:
            pass
    counter_saves = 0
    if profile is not None and opp.hand:
        buf = _estimate_counter_buffer(profile, len(opp.hand), getattr(opp, "trash", None))
        counter_saves = min(len(opp.hand), int(buf // _COUNTER_SAVE_UNIT))
    discounted_reach = max(0, reach - visible_blockers - counter_saves)

    score = 0.0
    if opp_life > 0:
        if discounted_reach >= opp_life:
            score += plan.lethal_mult * _CLOSER_W      # 割引後でも削り切れる盤面＝止めの形
        else:
            score += plan.lethal_mult * _NEAR_W * discounted_reach
    # マイルストーン: クロック先行（aggro）と J値スケジュール遵守/リソース差（control）を aggro_lean でブレンド。
    init_life = int(getattr(getattr(opp.leader, "master", None), "life", 0) or 0) if opp.leader else 0
    expected = max(0.0, init_life - plan.clock_rate * max(0, manager.turn_count))
    aggro_comp = (expected - opp_life) * _MILE_DMG_W
    sched = getattr(plan, "delta_schedule", ())
    if sched:
        # J値スケジュール遵守度: 実測 (相手J値 − 自分J値) が当ターンの理想差を上回る分を加点。
        # J値＝白＝デッキ残＋トラッシュの**枚数のみ**参照（中身は読まない＝公開情報・フェア）。
        my_j = len(me.deck) + len(getattr(me, "trash", None) or [])
        opp_j = len(opp.deck) + len(getattr(opp, "trash", None) or [])
        idx = min(max(0, manager.turn_count), len(sched) - 1)
        control_comp = ((opp_j - my_j) - sched[idx]) * _J_SCHED_W
    else:
        # 従来: 手札＋場の枚数差（プラン未導出・NEUTRAL・単体テストの _PRESETS 構築時＝完全同値）。
        res_diff = (len(me.hand) + len(me.field)) - (len(opp.hand) + len(opp.field))
        control_comp = res_diff * _MILE_RES_W
    score += plan.milestone_mult * (plan.aggro_lean * aggro_comp + (1.0 - plan.aggro_lean) * control_comp)
    return score


def evaluate(manager, me_name: str, see_opp_hand: bool = True, profile=None, plan=None,
             out: Optional[Dict[str, Any]] = None) -> float:
    """`me_name` 視点の盤面優劣スコア（高いほど自分有利）。

    `see_opp_hand=False` のとき相手手札は枚数のみ評価する（中身＝カウンター値を読まない）。
    自分の手札は常に full。難易度の情報方針: easy/normal=False（公開のみ）/ hard=True（チート）。
    `profile`（リーダー推測の相手モデル・§2.5.4）があれば、相手手札の防御価値（defense_factor）と
    自分のライフ重視度（aggro_lean）を補正する（normal のみ供給）。
    `plan`（自デッキ勝ち筋プラン・§2.5.5）があれば、自分側の評価重み（置物の存在価値・カウンター
    温存・ライフ重視・攻め圧）を補正し、逆算リーサル/マイルストーン項を加える（normal/hard で供給）。
    plan=None では一切作動せず現行挙動と完全同値。
    """
    if manager.winner == me_name:
        if out is not None:
            out["winner"] = me_name
        return W_WIN
    if manager.winner is not None:
        if out is not None:
            out["winner"] = manager.winner
        return -W_WIN
    me = _player_by_name(manager, me_name)
    opp = _other(manager, me_name)
    is_my_turn = manager.turn_player.name == me_name
    # リーダー推測補正: 相手の攻め寄り度が高いほど自分のライフを厚く見る／相手手札の防御価値を倍率補正。
    life_factor = 1.0
    opp_hand_factor = 1.0
    if profile is not None:
        life_factor = 1.0 + W_LIFE_AGGRO_K * profile.aggro_lean
        if not see_opp_hand:  # 公開方針のときだけ構築推測で相手手札の防御価値を補う
            opp_hand_factor = profile.defense_factor
    # 自デッキ勝ち筋プラン補正（自分側のみ。相手側は相手モデルが担当）。
    body_factor = attacker_factor = counter_factor = 1.0
    idle_don_factor = 1.0
    threat_atk_mult = threat_def_mult = 1.0   # A-2: 脅威キーワードのアーキタイプ依存スケール（両側対称）
    if plan is not None:
        life_factor *= plan.life_mult
        body_factor = plan.vanilla_body_mult
        attacker_factor = plan.attacker_mult
        counter_factor = plan.counter_mult
        idle_don_factor = plan.idle_don_mult
        threat_atk_mult = getattr(plan, "threat_atk_mult", 1.0)
        threat_def_mult = getattr(plan, "threat_def_mult", 1.0)
    # 脅威/キーワード資産評価（§2.5.6）は両側対称に適用＝相手の脅威を押し上げ→① の単一対象探索が
    # 最善の除去対象を狙う。プラン供給時のみ作動（plan=None では一切作動せず現行挙動と完全同値）。
    threat_aware = plan is not None
    # 有効パワー上限は対面の最硬防御。自分のパワーは相手を、相手のパワーは自分を上回るまでが有効。
    my_cap = _power_cap(opp)
    opp_cap = _power_cap(me)
    # C-3: 自ライフ（守備）は攻め対面で膝を 3 へ（レース耐性重視）。クロック側＝相手ライフは既定 2 のまま
    # ＝自他で別カーブ。profile 無し＝両側既定 2＝従来同値。
    own_life_knee = _own_life_knee(profile)
    # 時間割引（§2.5.3）: 地平線外の盤面ポテンシャル（場の存在価値）を残りゲーム長で割り引く（両側共通の係数）。
    # プラン供給時のみ作動（plan=None＝1.0＝従来同値）。ライフ高（早期）は 1.0＝割引なし。
    tempo_factor = _board_tempo_factor(me, opp) if plan is not None else 1.0
    # C-2: テレグラフ致死の減点（相手ターン開始の静止点＝相手の攻撃が目前のときだけ・プラン供給時）。
    telegraph = 0.0
    if plan is not None and not is_my_turn and _telegraph_lethal(me, opp):
        telegraph = W_TELEGRAPH_LETHAL
    # 成分内訳の収集（out 指定時のみ・採点には一切影響しない）。
    out_me = {} if out is not None else None
    out_opp = {} if out is not None else None
    plan_progress = _plan_progress(manager, me, opp, is_my_turn, plan, profile)
    total = (_side_score(me, is_my_turn, my_cap, include_counter=True, life_factor=life_factor,
                         body_factor=body_factor, attacker_factor=attacker_factor,
                         counter_factor=counter_factor, threat_aware=threat_aware,
                         idle_don_factor=idle_don_factor,
                         threat_atk_mult=threat_atk_mult, threat_def_mult=threat_def_mult,
                         life_knee=own_life_knee,
                         field_count_factor=tempo_factor, engine_aware=threat_aware, out=out_me)
             - _side_score(opp, not is_my_turn, opp_cap, include_counter=see_opp_hand,
                           hand_factor=opp_hand_factor, threat_aware=threat_aware,
                           threat_atk_mult=threat_atk_mult, threat_def_mult=threat_def_mult,
                           field_count_factor=tempo_factor,
                           engine_aware=threat_aware, out=out_opp)
             + plan_progress
             - telegraph)
    if out is not None:
        out["me"] = {k: round(v, 1) for k, v in out_me.items()}
        out["opp"] = {k: round(v, 1) for k, v in out_opp.items()}
        out["plan_progress"] = round(plan_progress, 1)
        out["telegraph"] = round(-telegraph, 1)
    return total


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
                     see_opp_hand: bool, profile, plan, stop_at_select: bool = False) -> Optional[float]:
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
            val = evaluate(manager, eval_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
        # action_events は transient バッファ（探索値に無関係）。txn 内の付け替えも巻き戻るが念のため復元。
        manager.action_events = saved_events
        return val
    clone = _apply_clone(manager, actor_name, move, stop_at_select=stop_at_select)
    if clone is None:
        return None
    return evaluate(clone, eval_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)


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
                       see_opp_hand: bool = True, profile=None, plan=None) -> float:
    """move をクローン上で適用し、actor 側の対話をドレインしてから評価する（1-ply）。

    `profile`/`plan`（任意・既定 None＝従来同値）を渡すと評価にプラン補正を反映する
    （対象選択の 1-ply 採点でプラン依存項を一致させる用途）。"""
    val = _score_move_1ply(manager, actor_name, move, actor_name,
                           see_opp_hand=see_opp_hand, profile=profile, plan=plan)
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


# --- B-1(b) カウンター強要（推定カウンター応答・§2.5.3/§2.5.6） -----------------------------------
# normal の保守 min ノードは手札カウンターを全除外＝相手は決してカウンターしない。これだと「余剰ドンを
# 攻撃に振って相手のカウンターを強要する」価値が出ない。そこで**相手手札の中身は読まず**（フェア）、
# リーダー推測 profile（§2.5.4）のカウンター密度から相手の「推定カウンター緩衝(power)」を見積もり、
# 相手 min ノードに「緩衝内なら攻撃を防ぐ（手札 1 枚を消費）／緩衝超なら貫通」という応答を PASS と並べて
# 入れる。min が選ぶので、緩衝内に収まる盛りは無駄（相手が守り切る）・緩衝超の盛りは正の手（貫通）になる。
# hard は opp_public_only=False で実カウンターを既に読むため本モデルは作動しない（profile も渡さない）。
_COUNTER_HAND_EST = 4.0   # 守りに割けるカウンター札の見込み枚数（コミット想定の上限・power 換算の係数）


def _estimate_counter_buffer(profile, opp_hand_size: Optional[int] = None, opp_trash=None) -> float:
    """profile のカウンター密度＋**公開情報ベリーフ**から、相手が 1 防御ターンに積める総カウンター power
    を推定する（§2.5.3 belief update）。profile が無ければ 0。

    静的なテンプレ密度（`counter_avg`）に、対局中に**公開された情報だけ**で belief を更新する:
      - **手札枚数**（公開 count）: カウンターには手札が要る。コミット想定枚数を実手札枚数でキャップ
        （少手札＝緩衝縮小・0 枚＝0）。手札の*中身*は読まない＝フェア。
      - **トラッシュ**（公開）: 既に使われた（見えた）カウンター値ぶん、デッキ＋手札の残カウンター密度を
        割り引く（消費が進むほど緩衝が縮む）。
    引数省略時は静的既定（手札 `_COUNTER_HAND_EST` 枚・未消費）＝従来値（テスト/後方互換）。
    """
    if profile is None:
        return 0.0
    per_card = float(getattr(profile, "counter_avg", 0.0) or 0.0)
    if per_card <= 0.0:
        return 0.0
    # 手札枚数ベリーフ: コミット想定を実手札枚数でキャップ（未供給時は静的既定 _COUNTER_HAND_EST）。
    commit = _COUNTER_HAND_EST if opp_hand_size is None else float(max(0, min(int(opp_hand_size), int(_COUNTER_HAND_EST))))
    # トラッシュ消費ベリーフ: 見えた消費カウンター値ぶん残密度を割り引く。
    depletion = 1.0
    if opp_trash:
        n = max(1, int(getattr(profile, "n_cards", 0) or 0))
        total_counter = per_card * n   # テンプレ基準のデッキ全体の総カウンター power
        if total_counter > 0.0:
            seen = 0.0
            for c in opp_trash:
                m = getattr(c, "master", None)
                if m is not None:
                    seen += float(getattr(m, "counter", 0) or 0)
            depletion = max(0.0, 1.0 - seen / total_counter)
    return max(0.0, per_card * commit * depletion)


def _counter_needed(manager) -> Optional[float]:
    """現在の戦闘で相手が攻撃を防ぐ（防御側が攻撃側を上回る）のに必要なカウンター power。

    攻撃が既に通らない／戦闘が無い場合は None。`resolve_attack` と同じパワー計算を用いる
    （攻撃側=自ターンなら付与込み・防御側=素＋既存 counter_buff）。
    """
    ab = getattr(manager, "active_battle", None)
    if not ab or ab.get("attacker") is None or ab.get("target") is None:
        return None
    atk = ab["attacker"]; tgt = ab["target"]
    ao = ab.get("attacker_owner"); to = ab.get("target_owner")
    try:
        ap = float(atk.get_power(ao == manager.turn_player))
        tp = float(tgt.get_power(to == manager.turn_player)) + float(ab.get("counter_buff", 0) or 0)
    except Exception:
        return None
    needed = ap - tp + 1.0
    return needed if needed > 0 else None


def _apply_modeled_counter(manager, defender_name: str, needed: float):
    """推定カウンターを適用したクローンを返す（`counter_buff` を needed 加算＋手札 1 枚を資源消費＋PASS で解決）。

    相手手札の**中身は選ばない**（先頭 1 枚を消費＝枚数のみ＝公開情報・フェア）。手札が無い／戦闘が無いとき None。
    """
    from . import action_api
    clone = manager.clone()
    defender = _player_by_name(clone, defender_name)
    if not defender.hand or not getattr(clone, "active_battle", None):
        return None
    clone.active_battle["counter_buff"] = clone.active_battle.get("counter_buff", 0) + needed
    defender.hand.pop(0)   # カウンター札 1 枚の消費（枚数のみ＝公開情報。中身は参照しない＝フェア）
    battle_actions = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    ACT_PASS = battle_actions.get('PASS', 'PASS')
    clone.action_events = []
    try:
        action_api.apply_battle_action(clone, defender, ACT_PASS, None)
    except Exception:
        return None
    return clone


def _settle_discount(value: float, plan) -> float:
    """打ち切り settle 葉の不確実性ディスカウント（C-4・§2.5.3）。

    settle は予算切れの局面を**既定解決**で無理に静止点へ整流して評価する＝探索で確かめた値ではない。
    既定解決の選択が評価を不当に振らせないよう、勝敗未確定（非 lethal）の settle 値は信頼度
    `_SETTLE_CONFIDENCE` で中立（盤面差 0＝互角）へ寄せて過信を防ぐ（＝既定解決の中立化）。これで探索は
    「予算内で読み切れる線」をやや優先し、既定解決頼みの甘い/辛い見積りに賭けすぎない。lethal（|value| が
    W_WIN 近傍＝勝敗確定）は確定事象なので割り引かない。plan 無し（plan=None）は従来どおり割り引かない。
    """
    if plan is None:
        return value
    if abs(value) >= W_WIN - HARD_MAX_PLY:   # 勝敗確定（±(W_WIN-ply)）は割り引かない
        return value
    return value * _SETTLE_CONFIDENCE


def _settle_eval(manager, root_name: str, see_opp_hand: bool, profile, plan, ply: int = 0) -> float:
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
    val = evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    # B（楽観是正・§2.5.3）: settle は相手のターン開始で止めて静的採点する＝相手の反撃を読まない＝動いた側に
    # 楽観的。読み切れなかった葉に限り、相手の次ターンの「受け切れない打点本数」を保守的に減点して楽観を
    # 是正する（致死は C-2 telegraph が `evaluate` 内で計上済みなので**致死未満のみ**＝二重計上を避ける）。
    # 相手ターンの静止点（rectify 後に turn_player が相手）かつ plan 供給時のみ作動（plan=None 完全同値）。
    if plan is not None and manager.turn_player.name != root_name:
        me = _player_by_name(manager, root_name)
        reach = _incoming_reach(me, _other(manager, root_name))
        if 0 < reach < len(me.life):
            val -= reach * W_SETTLE_PRESSURE
    # C-4: 既定解決で整流した打ち切り葉は不確実なので信頼度で中立へ寄せる（plan 供給時のみ・plan=None 同値）。
    return _settle_discount(val, plan)


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
            profile=None, ply: int = 0, plan=None, start_turn: int = 0, horizon: int = 1,
            counter_budget: float = 0.0, killers: Optional[Dict[int, List[tuple]]] = None) -> float:
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
      - `profile`/`plan`   : リーダー推測の相手モデル（§2.5.4）／自デッキ勝ち筋プラン（§2.5.5/§2.5.6）。
    """
    if manager.winner is not None:
        return (W_WIN - ply) if manager.winner == root_name else -(W_WIN - ply)

    # 探索は (手番, アクション) しか見ない（手は get_legal_actions から）。重い get_pending_request
    # ではなく軽量な pending_actor_action を使う（payload/uuid4 を作らない・副作用は一致・§2.5.2 B-1）。
    pa = manager.pending_actor_action()
    if not pa:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    actor_name, pend_action = pa
    # 葉: start_turn から horizon ターン進んだ MAIN_ACTION（一定の静止点）で評価。
    if pend_action == "MAIN_ACTION" and (manager.turn_count - start_turn) >= horizon:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    # 安全打ち切り: 予算/ply 上限。自分の手番途中ならターン境界へ整流してから評価（甘い途中評価を避ける）。
    if budget[0] <= 0 or ply >= HARD_MAX_PLY:
        return _settle_eval(manager, root_name, see_opp_hand, profile, plan, ply)

    actor = _player_by_name(manager, actor_name)
    # 単一対象選択ノードは候補ごとに分岐（最善対象を読み切る）。それ以外は通常の合法手列挙。
    moves = _selection_moves(manager, actor_name)
    if moves is None:
        moves = manager.get_legal_actions(actor)
        moves = _prune_don_moves(manager, actor_name, moves)  # B-2: 無意味なドン付与を手生成段で除外
        moves = _prune_futile_attacks(manager, actor_name, moves)  # 倒せない/届かない無駄攻撃を除外
    if not moves:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
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
                             see_opp_hand=see_opp_hand, profile=profile, plan=plan, stop_at_select=True)
        if v is None:
            continue
        children.append((v, m))
    if not children:
        return _settle_eval(manager, root_name, see_opp_hand, profile, plan, ply)
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
                                    budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                    start_turn, horizon, counter_budget, killers))
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
        # B-1(b): 相手の推定カウンター応答（normal 保守 min・SELECT_COUNTER・緩衝内で攻撃を防げる場合のみ）。
        # PASS（=カウンターしない＝攻撃が通る）と並べて min が選ぶ＝相手にとって有利な方（=自分に不利な方）。
        # 緩衝を needed ぶん消費して深掘り。これで「緩衝超まで盛ると貫通＝余剰ドンを攻撃に振るのが正」になる。
        if (opp_public_only and profile is not None and counter_budget > 0
                and pend_action == "SELECT_COUNTER"):
            needed = _counter_needed(manager)
            if needed is not None and needed <= counter_budget:
                cc = _apply_modeled_counter(manager, actor_name, needed)
                if cc is not None:
                    value = min(value, _search(cc, root_name, alpha, beta,
                                               budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                               start_turn, horizon, counter_budget - needed, killers))
                    beta = min(beta, value)
        for _leaf, m in children:
            if alpha >= beta:
                break
            cv = _recurse_child(manager, actor_name, m,
                                lambda b, a=alpha, bt=beta: _search(b, root_name, a, bt,
                                    budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                    start_turn, horizon, counter_budget, killers))
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


def _eval_root_move(manager, name: str, move: Dict[str, Any], see_opp_hand: bool, profile, plan):
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
                ev = evaluate(manager, name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
                pen = _don_return_penalty_vals(pre_don, len(_player_by_name(manager, name).don_deck))
                imp = _is_important_root_move_post(manager, name, move, pre_block, pre_opp_life)
                res[0] = (ev, pen, imp)
        manager.action_events = saved_events
        return res[0]
    child = _apply_clone(manager, name, move, stop_at_select=True)
    if child is None:
        return None
    ev = evaluate(child, name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    pen = _don_return_penalty(manager, name, child)
    imp = _is_important_root_move(manager, name, move, child)
    return (ev, pen, imp)
    return False


# 深掘り同点手の 1-ply タイブレーク（§2.5.3）。寄与は _TIEBREAK_W×クランプ prelim（最大 ~0.005）で、
# 実の深掘り差（典型 >0.3＝パワー1段ぶん）には影響せず、**厳密同点のみ**を即時盤面（1-ply）で割る。
# prelim は勝ち(±W_WIN)で巨大化し得るため _TIEBREAK_CLAMP でクランプし、巨大値が実差を覆さないようにする。
_TIEBREAK_W = 1e-6
_TIEBREAK_CLAMP = 5000.0


def _scored_search(manager, name: str, moves: List[Dict[str, Any]],
                   see_opp_hand: bool, opp_public_only: bool,
                   profile=None, plan=None, collect: Optional[Dict[str, Any]] = None,
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
        r = _eval_root_move(manager, name, m, see_opp_hand, profile, plan)
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
    # B-1(b): 相手の推定カウンター緩衝（normal の保守 min ノードでのみ作動。hard は実カウンターを読むため 0）。
    # #3: 公開情報ベリーフ＝相手の生の手札枚数＋トラッシュの消費カウンターで緩衝を動的更新（§2.5.3）。
    opp = _other(manager, name)
    cbudget = _estimate_counter_buffer(profile, len(opp.hand), opp.trash) if opp_public_only else 0.0
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
            budget = [HARD_PER_MOVE_BUDGET]
            v = _recurse_child(manager, name, m,
                               lambda b: _search(b, name, float("-inf"), float("inf"),
                                   budget, see_opp_hand, opp_public_only, profile, ply=1, plan=plan,
                                   start_turn=start_turn, horizon=HARD_HORIZON, counter_budget=cbudget,
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


# 1 ターン内に CPU が取れる手の総数上限（暴走/無限ループの最終防壁）。
TURN_ACTION_CAP = 60
# 同一の起動効果/ドン付与をこのターン内に繰り返してよい回数の上限。
REPEAT_CAP = 3


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
        zones = [getattr(p, z, None) or [] for z in ("field", "hand", "life", "deck", "trash")]
        if leader is not None:
            zones.append([leader])
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
    return d


def _read_ahead_line(manager, root_name: str, see_opp_hand: bool, opp_public_only: bool,
                     profile, plan, start_turn: int, horizon: int,
                     max_steps: int = 12) -> List[Dict[str, Any]]:
    """貪欲 PV（読み筋）: 各手番で 1-ply 最善手（root=max / 相手=min）を辿った想定進行。

    `_search` の探索木そのものではなく、その縮約版（各ノードでビーム1）の「想定される線」。
    trace 指定時のみ呼ばれるためコストは問わない。`_search` 本体には一切触れない＝探索挙動は不変。
    """
    line: List[Dict[str, Any]] = []
    cur = manager
    KEY_PID, KEY_ACTION = _pending_keys()
    repeatable = {"ACTIVATE_MAIN", "ATTACH_DON"}
    counts: Dict[tuple, int] = {}   # decide_guarded と同じ繰り返しガード（無限起動効果で PV が膨れるのを防ぐ）
    cur_turn = cur.turn_count
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
        if cur.turn_count != cur_turn:   # ターンが変わったら繰り返しカウントをリセット
            cur_turn = cur.turn_count
            counts = {}
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
        # 繰り返しガード: 同一の起動効果/ドン付与を REPEAT_CAP 回使い切ったら候補から除外する。
        moves = [m for m in moves
                 if not (m.get("action_type") in repeatable and counts.get(_move_sig(m), 0) >= REPEAT_CAP)]
        if not moves:
            break
        best = None
        for m in moves:
            child = _apply_clone(cur, actor_name, m, stop_at_select=True)
            if child is None:
                continue
            sc = evaluate(child, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
            if best is None or (sc > best[0] if is_max else sc < best[0]):
                best = (sc, m, child)
        if best is None:
            break
        sc, m, child = best
        if m.get("action_type") in repeatable:
            counts[_move_sig(m)] = counts.get(_move_sig(m), 0) + 1
        line.append({"turn": cur.turn_count, "actor": actor_name, "is_max": is_max,
                     "move": _describe_move(cur, m), "eval": round(sc, 1)})
        cur = child
    return line


def _fill_decision_trace(trace: Dict[str, Any], manager, name: str, difficulty: str,
                         moves: List[Dict[str, Any]], scored: List[Tuple[float, Dict[str, Any]]],
                         collect: Optional[Dict[str, Any]], chosen: Dict[str, Any], folded: bool,
                         see_opp_hand: bool, opp_public_only: bool, profile, plan,
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
        total = evaluate(child, name, see_opp_hand=see_opp_hand, profile=profile, plan=plan, out=comp)
        comp["total"] = round(total, 1)
        trace["j_components"] = comp
        if include_read_ahead:
            trace["read_ahead"] = _read_ahead_line(
                child, name, see_opp_hand, opp_public_only, profile, plan,
                manager.turn_count, HARD_HORIZON)


def decide(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
           moves: Optional[List[Dict[str, Any]]] = None, plan=None,
           trace: Optional[Dict[str, Any]] = None, trace_read_ahead: bool = True,
           killer_state: Optional[Dict[int, List[tuple]]] = None,
           info_policy: str = DEFAULT_INFO_POLICY) -> Optional[Dict[str, Any]]:
    """`player` が取るべき次の 1 手を返す（合法手が無ければ None）。

    α-β＋ビーム。**easy/normal は廃止**（最強の α-β＝hard と MCTS＝expert の2系統に集約。`difficulty`
    引数は互換のため残すが分岐しない）。情報方針は `info_policy`（既定 "fair"＝相手手札を読まない出荷
    デフォルト／"cheat"＝旧 hard・相手手札も読む。`_INFO_POLICIES` 参照）で切り替える。
    `moves` を渡すとその候補集合から選ぶ（ガード driver が絞り込んだ手を渡す用途）。
    `plan` は自デッキ勝ち筋プラン（§2.5.5）。`trace`（任意・既定 None）で意思決定の診断情報を書き込む。
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
            s = _simulate_and_eval(manager, name, m, see_opp_hand=see_opp_hand, plan=plan)
            scored.append((s, m))
            if collect is not None:
                collect["prelim"][_move_sig(m)] = s
                collect["deep"][_move_sig(m)] = s
    else:
        scored = _scored_search(manager, name, moves, see_opp_hand=see_opp_hand,
                                opp_public_only=opp_public_only,
                                plan=plan, collect=collect, killer_state=killer_state)
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
        # A-2: 畳み判定マージンをアーキタイプ依存にスケール（aggro=小さく攻めを通す／control=大きく畳む）。
        margin = _ACT_MARGIN * (getattr(plan, "act_margin_mult", 1.0) if plan is not None else 1.0)
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
                                 chosen, folded, see_opp_hand, opp_public_only, None, plan,
                                 include_read_ahead=trace_read_ahead)
        finally:
            random.setstate(_rng_state)
    return chosen


def decide_with_regret(manager, player, difficulty: str = "hard",
                       rng: Optional[random.Random] = None, plan=None,
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
        return decide(manager, player, difficulty, rng, moves=moves, plan=plan,
                      info_policy=info_policy), 0.0

    name = player.name
    moves = _prune_don_moves(manager, name, moves)  # B-2: ルート手集合を decide と一致させる（regret 整合）
    moves = _prune_futile_attacks(manager, name, moves)  # 倒せない/届かない無駄攻撃を除外（decide と一致）
    see_opp_hand, opp_public_only = _resolve_info_policy(info_policy)
    collect: Dict[str, Any] = {}
    _scored_search(manager, name, moves, see_opp_hand=see_opp_hand, opp_public_only=opp_public_only,
                   plan=plan, collect=collect)
    move = decide(manager, player, difficulty, rng, moves=moves, plan=plan, info_policy=info_policy)
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
                   mem: Optional[Dict[str, Any]] = None, plan=None,
                   trace: Optional[Dict[str, Any]] = None, trace_read_ahead: bool = True,
                   info_policy: str = DEFAULT_INFO_POLICY) -> Optional[Dict[str, Any]]:
    """ターン内メモリ `mem` を用いた暴走防止つきの意思決定。

    `mem` は呼び出し側が対局ごとに保持する dict（ステートレスな /cpu/step でも CPU_GAMES に
    保持して渡す）。同一ターン内で:
      - 取った手の総数が TURN_ACTION_CAP を超えたら強制 TURN_END
      - 同じ起動効果/ドン付与を REPEAT_CAP 回行ったら候補から除外（イガラム等の無限ループ防止）
    これにより「効果に per-turn 制限が無い/付け忘れ」のカードでも CPU ターンが必ず終わる。
    """
    rng = rng or random
    if mem is None:
        mem = {}
    if mem.get("turn") != manager.turn_count:
        mem["turn"] = manager.turn_count
        mem["counts"] = {}
        mem["total"] = 0
        mem["killers"] = {}   # ④粒度b: killer 表はターン内でのみ持ち越す（ターン跨ぎの古い表は破棄）

    moves = manager.get_legal_actions(player)
    if not moves:
        return None
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # 総数キャップ: 上限超過ならターンを畳む（畳めない＝対話中等なら通常選択）。
    if end_move is not None and mem.get("total", 0) >= TURN_ACTION_CAP:
        if trace is not None:
            trace["difficulty"] = difficulty
            trace["turn"] = manager.turn_count
            trace["chosen"] = _describe_move(manager, end_move)
            trace["forced"] = "turn_action_cap"
            trace["candidates"] = []
            trace["regret"] = 0.0
        return end_move

    # 繰り返しキャップ: 起動効果/ドン付与の同一手を上限まで使い切ったら除外する。
    counts = mem.get("counts", {})
    repeatable = {"ACTIVATE_MAIN", "ATTACH_DON"}
    filtered = [m for m in moves
                if not (m.get("action_type") in repeatable and counts.get(_move_sig(m), 0) >= REPEAT_CAP)]
    if not filtered:
        filtered = [end_move] if end_move is not None else moves

    # ④粒度b: killer 表を mem に保持して連続 decide 間で持ち越す（reorder は集合不変＝安全だが実測で
    # 中立〜微減＝既定 OFF。`_USE_PV_CROSS_DECIDE` 有効時のみ供給。OFF は killer_state=None＝粒度a のみ）。
    ks = mem.setdefault("killers", {}) if (_USE_PV_ORDER and _USE_PV_CROSS_DECIDE) else None
    move = decide(manager, player, difficulty, rng, moves=filtered, plan=plan,
                  trace=trace, trace_read_ahead=trace_read_ahead, killer_state=ks,
                  info_policy=info_policy)
    if move is not None:
        sig = _move_sig(move)
        counts[sig] = counts.get(sig, 0) + 1
        mem["counts"] = counts
        mem["total"] = mem.get("total", 0) + 1
    return move


def plan_turn(manager, name: str, difficulty: str = "hard", rng=None,
              mem: Optional[Dict[str, Any]] = None, plan=None,
              info_policy: str = DEFAULT_INFO_POLICY) -> List[Dict[str, Any]]:
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
        mv = decide_guarded(clone, actor, difficulty, rng, mem=mem, plan=plan, info_policy=info_policy)
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
                  plan=None, info_policy: str = DEFAULT_INFO_POLICY) -> Optional[Dict[str, Any]]:
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

    actions = plan_turn(manager, player.name, difficulty, rng, mem=mem, plan=plan, info_policy=info_policy)
    if actions:
        first_sig = _move_sig(actions[0])
        if first_sig in legal_by_sig:
            cache["queue"] = actions[1:]
            return legal_by_sig[first_sig]
    # 計画が空/先頭不正＝安全側で通常 decide（rng/mem は plan_turn で進行済みのため二重進行に注意だが
    # 本番専用＝決定性非依存。guard は安全網なので軽微な前後は許容）。
    cache["queue"] = None
    return decide_guarded(manager, player, difficulty, rng, mem=mem, plan=plan, info_policy=info_policy)
