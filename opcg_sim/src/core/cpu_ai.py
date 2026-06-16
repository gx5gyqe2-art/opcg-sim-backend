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
import random
from typing import Any, Dict, List, Optional, Tuple
import re

# 評価重み（盤面 1000=パワー1段相当に正規化）
W_LIFE = 6000.0          # ライフ 1 枚の基礎価値（最重要）
W_LIFE_LOW = 4000.0      # 希少域（最初の 2 枚）への上乗せ＝非線形・45[J] ラインの危険
# C-3（§2.5.3）: ライフ薄域上乗せの膝位置（この枚数までを厚く守る）。自他で別カーブ＝既定 2、攻め寄りの
# 相手と対面する自ライフ（守備）のみ 3 へ上げてレース耐性を厚く見る（profile 無し＝両側 2＝従来同値）。
_LIFE_KNEE_DEFAULT = 2
_LIFE_KNEE_AGGRO_MATCHUP = 3
_AGGRO_MATCHUP_THRESHOLD = 0.6   # 相手プロファイルの aggro_lean がこれ以上で「攻め対面」と見なす
W_HAND = 700.0           # 手札 1 枚の基礎価値
W_COUNTER = 0.6          # 手札のカウンター値 1 点あたり（防御リソース）
# コスト低減の資源価値化（§2.5.3・コスト低減を潜在資源として軽く価値化）。`evaluate` は素では手札の
# コストを読まないため、コスト低減は「次ターン手出しできる脅威」という潜在資源として無価値だった。
# 安価な代理として「次ターンに手出しできる（コスト ≤ 次ターン見込みドン）」手札に小ボーナスを与え、
# 打てる脅威の期待値を軽く織り込む。`current_cost` は cost_buff/timed_cost を含む＝**コスト低減が
# そのまま手出し可否に効く**＝低減の資源価値が拾える。W_HAND(700) より十分小さい＝既存の枚数価値を
# 歪めない軽い上乗せ。プラン供給時のみ作動（plan=None では一切作動せず現行挙動と完全同値）。
W_HAND_PLAYABLE = 150.0  # 次ターンに手出しできる手札 1 枚あたりの潜在資源ボーナス
W_FIELD_COUNT = 1500.0   # 場のキャラ 1 体の存在価値
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
HARD_BEAM = 3              # 各ノードで展開する子の数（1-ply 評価上位 K）
HARD_ROOT_BEAM = 4         # 深掘りするルート手の数（残りは 1-ply スコアのまま。TURN_END は常に深掘り）
# B-3（§2.5.3）: 1-ply ランクに関係なく深掘り集合へ**強制投入**する重要手クラス（ブロッカー設置・除去
# 候補・逆算リーサル/クロック手）の追加上限。ビーム拡幅（4→6-8）は置換表によるレイテンシ削減が前提
# （SPEC）のため本数を絞り、取りこぼし是正だけを先取りする（1 手あたり HARD_PER_MOVE_BUDGET の追加読み）。
HARD_FORCE_DEEPEN_CAP = 3
HARD_PER_MOVE_BUDGET = 36  # 深掘り 1 手あたりのクローン上限（予算切れは settle で境界評価＝正しいので
                           # ここはレイテンシ予算。自分のターン1回を~1秒で読める範囲に抑える）
HARD_DEPTH = 5             # ply 割引の基準（最短リーサル認識のテスト境界・winner 到達 ply の上限目安）
HARD_MAX_PLY = 30          # 総 ply の安全上限（自ターンの自由展開＋戦闘サブステップが暴走しない belt-and-suspenders）
# 探索ホライズン（B2-lite・docs/SPEC.md §2.5.3）。深掘りで何ターン先まで読むか。horizon=2＝「自分のターン
# 完了→相手のターンを丸ごと（攻撃まで）→自分の次ターン開始」の静止点で評価＝相手の反撃に対する守り
# （ブロッカー/カウンター温存）を min/max で読む。横展開は重いので上位 K 手（HARD_ROOT_BEAM＋TURN_END）
# のみに適用し、非対象は採用しない（評価ホライズンの一貫性を保つ）。
HARD_HORIZON = 2
_SETTLE_LIMIT = 16         # 打ち切り時にターン境界へ整流する最大手数（戦闘サブステップ込み）

# 自デッキ勝ち筋プラン（cpu_self_plan・docs/SPEC.md §2.5.5）の評価項。プラン未指定（plan=None）では
# 一切作動せず現行挙動と完全同値（乗数は既定 1.0／逆算項は plan が無ければ 0）。normal/hard でのみ供給。
_LOW_IMPACT_POWER = 5000          # 素パワーがこの値未満＝素ではリーダー(5000)に打点が通らない置物
_RELEVANT_KEYWORDS = ("ブロッカー", "速攻", "ダブルアタック")  # これらを持つ体は「置物」扱いしない
_CLOSER_W = 2000.0                # 逆算リーサル: 相手を削り切れる本数を持つ盤面への加点
_NEAR_W = 200.0                   # 同上: 削り切るには足りないが届く攻撃 1 本あたりの軽い加点
_MILE_DMG_W = 1200.0             # マイルストーン: 想定クロックより相手ライフが先行して減っている分
_MILE_RES_W = 220.0             # マイルストーン: リソース差（手札＋場の枚数差）1 枚あたり

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

# C-2（§2.5.3）: テレグラフ致死の減点。葉（相手ターン開始＝相手の攻撃が目前）で「相手の次ターンの有効打点
# ≥ 自残ライフ」なら、受け切れず負ける telegraph として減点する。W_WIN(1e9) に対し十分小さい＝**本物の
# リーサル発見（±W_WIN）は決して上書きしない**＝引き分け帯で守り（ブロッカー温存・脅威除去・ライフ獲得）へ
# 寄せるだけ。プラン供給時のみ作動（plan=None 完全同値）。過剰防御を避けるため打点見積りは素パワー（保守的）。
W_TELEGRAPH_LETHAL = 6000.0


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


def _next_turn_don(p) -> int:
    """`p` の次ターンに使える見込みドン!! 枚数（コスト低減の資源価値化・§2.5.3）。

    次の自ターン開始では現在の全ドン（アクティブ＋レスト＋付与中）がアクティブに戻り、さらにドンデッキ
    から 2 枚（ターン1のみ 1 枚だが中盤評価では常に 2）が補充される。ドンデッキ残でキャップ＝盤面の真値。
    相手手札の中身は読まない＝ドン枚数は公開情報なのでフェア。
    """
    total = len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)
    return total + min(2, len(p.don_deck))


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


def _telegraph_lethal(me, opp) -> bool:
    """C-2（§2.5.3）: 相手の次ターンの有効打点が自残ライフ以上で、受け切れず負ける telegraph か。

    相手の次ターンは場が全アクティブ化するので `is_rest` は無視し、リーダー＋場のうち**素パワーが自リーダー
    に届く**体を数える（素パワー＝保守的＝don 付与での押し上げは数えない＝過剰防御を避ける）。自分の
    アクティブブロッカー数（各 1 本を止める）を控除し、割引後の打点本数 ≥ 自残ライフ なら telegraph 致死。
    """
    my_life = len(me.life)
    if my_life <= 0:
        return False
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
    reach -= _active_blocker_count(me)
    return reach >= my_life


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
                next_turn_don: Optional[int] = None) -> float:
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
    `next_turn_don`（コスト低減の資源価値化・§2.5.3）が与えられると、手札のうち「次ターン手出しできる
    （`current_cost` ≤ `next_turn_don`）」枚数に `W_HAND_PLAYABLE` を上乗せする＝コスト低減を潜在資源として
    軽く価値化する。手札の中身（コスト）を読むため `include_counter`（＝この手札を読んでよい側）のときだけ
    作動する＝相手手札の中身を読まないフェア性を保つ。None（plan 無し）では作動しない＝従来同値。
    """
    score = 0.0

    # ライフ: 非線形（薄いほど 1 枚の限界価値が高い）。膝位置（life_knee）までを厚く上乗せする。
    life_n = len(p.life)
    score += life_n * W_LIFE * life_factor
    score += min(life_n, life_knee) * W_LIFE_LOW * life_factor

    # 手札: 枚数 ＋（公開方針でなければ）カウンター値（防御に回せる力＝相手の +1[J] を打ち消す資源）。
    score += len(p.hand) * W_HAND * hand_factor
    if include_counter:
        for c in p.hand:
            try:
                score += (c.current_counter or 0) * W_COUNTER * counter_factor
            except Exception:
                pass
            # コスト低減の資源価値化: 次ターン手出しできる手札を潜在資源として軽く加点（plan 供給時のみ）。
            if next_turn_don is not None:
                try:
                    if c.current_cost <= next_turn_don:
                        score += W_HAND_PLAYABLE
                except Exception:
                    pass

    # 白（J）の決定境界: 自デッキ残がデッキ切れ（J=0・ドロー不能＝敗北）へ近づくほど非線形に減点。
    score -= max(0, DECK_DANGER - len(p.deck)) * W_DECK_DANGER

    # ドン!!（アクティブ）。`is_turn=False`（自分の手番でない静止点＝葉）では、浮いたアクティブドンは
    # 防御に使えない（OPCG はドンを防御に付与できない）ので、プラン由来の `idle_don_factor`(<1.0) で
    # 減価する＝「両枝でクロック同値→ドンの床でタイブレーク→握る」という余剰ドン温存を断つ（B-1・§2.5.3）。
    # 自分の手番中（is_turn=True）は付与でパワーに変換できる生きた資源なので減価しない。plan 無し時は 1.0。
    don_score = len(p.don_active) * W_DON_ACTIVE
    if not is_turn and idle_don_factor != 1.0:
        don_score *= idle_don_factor
    score += don_score

    # 場のキャラ: 存在価値 ＋ 有効パワー ＋ ブロッカー（最終防御）＋ 攻め圧（実際に攻撃できる体のみ）。
    # 存在価値はプランで「効果なし低パワーの置物」のみ割り引く（body_factor）＝デッキ依存の置物許容度。
    for c in p.field:
        ev = W_FIELD_COUNT
        if body_factor != 1.0 and _is_low_impact(c):
            ev *= body_factor
        score += ev
        # 脅威/キーワード資産（ダブルアタック・効果耐性・速攻・バニッシュ・アンブロッカブル）。両側対称・
        # プラン時のみ。A-2: 攻め/守りキーワードをアーキタイプ依存にスケール（aggro=攻め重視/control=守り重視）。
        if threat_aware:
            score += _threat_value(c, threat_atk_mult, threat_def_mult)
        try:
            pw = c.get_power(is_turn)
        except Exception:
            pw = c.master.power or 0
        score += _effective_power(pw, power_cap) * W_FIELD_POWER
        if not c.is_rest:
            # 攻め圧は「このターン実際に攻撃できる体」に限る。自ターンの召喚酔い（速攻なし）は
            # 今ターン攻撃できないので加点しない＝意味のない小型展開で攻め圧を水増ししない。
            sick = getattr(c, "is_newly_played", False) and not c.has_keyword("速攻")
            if not (is_turn and sick):
                score += W_ATTACKER * attacker_factor
            if c.has_keyword("ブロッカー"):
                score += W_BLOCKER
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
      手札＋場のリソース差で加点。攻め寄り度 aggro_lean で両者をブレンドする。
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
    # マイルストーン: クロック先行（aggro）とリソース差（control）を aggro_lean でブレンド。
    init_life = int(getattr(getattr(opp.leader, "master", None), "life", 0) or 0) if opp.leader else 0
    expected = max(0.0, init_life - plan.clock_rate * max(0, manager.turn_count))
    aggro_comp = (expected - opp_life) * _MILE_DMG_W
    res_diff = (len(me.hand) + len(me.field)) - (len(opp.hand) + len(opp.field))
    control_comp = res_diff * _MILE_RES_W
    score += plan.milestone_mult * (plan.aggro_lean * aggro_comp + (1.0 - plan.aggro_lean) * control_comp)
    return score


def evaluate(manager, me_name: str, see_opp_hand: bool = True, profile=None, plan=None) -> float:
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
        return W_WIN
    if manager.winner is not None:
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
    # コスト低減の資源価値化（§2.5.3）: 次ターン手出しできる手札を潜在資源として軽く加点（プラン供給時のみ）。
    # 自分側は常に手札を読むので供給。相手側は手札の中身（コスト）を読む hard（see_opp_hand）のときだけ供給
    # ＝相手手札の中身を読まない normal のフェア性を保つ（plan 無しは両側 None＝従来同値）。
    me_next_don = _next_turn_don(me) if plan is not None else None
    opp_next_don = _next_turn_don(opp) if (plan is not None and see_opp_hand) else None
    # C-2: テレグラフ致死の減点（相手ターン開始の静止点＝相手の攻撃が目前のときだけ・プラン供給時）。
    telegraph = 0.0
    if plan is not None and not is_my_turn and _telegraph_lethal(me, opp):
        telegraph = W_TELEGRAPH_LETHAL
    return (_side_score(me, is_my_turn, my_cap, include_counter=True, life_factor=life_factor,
                        body_factor=body_factor, attacker_factor=attacker_factor,
                        counter_factor=counter_factor, threat_aware=threat_aware,
                        idle_don_factor=idle_don_factor,
                        threat_atk_mult=threat_atk_mult, threat_def_mult=threat_def_mult,
                        life_knee=own_life_knee, next_turn_don=me_next_don)
            - _side_score(opp, not is_my_turn, opp_cap, include_counter=see_opp_hand,
                          hand_factor=opp_hand_factor, threat_aware=threat_aware,
                          threat_atk_mult=threat_atk_mult, threat_def_mult=threat_def_mult,
                          next_turn_don=opp_next_don)
            + _plan_progress(manager, me, opp, is_my_turn, plan, profile)
            - telegraph)


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
    KEY_PID, KEY_ACTION = _pending_keys()
    for _ in range(_DRAIN_LIMIT):
        pending = manager.get_pending_request()
        if not pending or pending[KEY_PID] != actor_name:
            return
        action = pending[KEY_ACTION]
        # メイン/マリガン/戦闘は「意思決定」なのでドレインしない（呼び出し側が1手として扱う）。
        if action in ("MAIN_ACTION", "MULLIGAN", "SELECT_BLOCKER", "SELECT_COUNTER"):
            return
        # 探索モードでは分岐可能な単一対象選択もドレインしない（探索ノードとして残す）。
        if stop_at_select and _selection_moves(manager, actor_name) is not None:
            return
        payload = manager.default_interaction_payload(pending)
        actor = _player_by_name(manager, actor_name)
        manager.action_events = []
        try:
            action_api.apply_game_action(manager, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            return


def _apply_clone(manager, actor_name: str, move: Dict[str, Any], stop_at_select: bool = False):
    """move を新しいクローンへ適用し、actor 側の対話をドレインしたクローンを返す。

    シミュレーションが例外を出す手は None を返す（呼び出し側で除外する）。
    `stop_at_select=True` のとき、ドレインは分岐対象の単一対象選択で停止する（探索側が分岐する）。
    """
    from . import action_api
    clone = manager.clone()
    actor = _player_by_name(clone, actor_name)
    clone.action_events = []
    try:
        if move["kind"] == "battle":
            action_api.apply_battle_action(clone, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(clone, actor, move["action_type"], move.get("payload", {}))
        _drain_own_interactions(clone, actor_name, stop_at_select=stop_at_select)
    except Exception:
        return None
    return clone


def _simulate_and_eval(manager, actor_name: str, move: Dict[str, Any],
                       see_opp_hand: bool = True) -> float:
    """move をクローン上で適用し、actor 側の対話をドレインしてから評価する（1-ply）。"""
    clone = _apply_clone(manager, actor_name, move)
    if clone is None:
        return float("-inf")
    return evaluate(clone, actor_name, see_opp_hand=see_opp_hand)


def _selection_moves(manager, actor_name: str):
    """actor の「単一対象選択」対話を候補ごとの RESOLVE 手として列挙する（無ければ None）。

    対象は `_SELECT_ACTION`（SELECT_TARGET/FIELD_OVERFLOW_TRASH を正規化したもの）かつ **最大1体**
    （max==1・min<=1）の選択に限る＝「どれを KO/除去/バウンス/手札破壊するか」。多対象（max>=2）や
    min>1 は組合せ爆発を避けて既定解決へ委ねる（None を返す）。これにより探索が最善の単一対象を読む。
    """
    from . import action_api
    pending = manager.get_pending_request()
    if not pending:
        return None
    KEY_PID, KEY_ACTION = _pending_keys()
    if pending.get(KEY_PID) != actor_name or pending.get(KEY_ACTION) != _SELECT_ACTION:
        return None
    props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_UUIDS = props.get('SELECTABLE_UUIDS', 'selectable_uuids')
    KEY_CONSTRAINTS = props.get('CONSTRAINTS', 'constraints')
    KEY_SKIP = props.get('CAN_SKIP', 'can_skip')
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
    # 単一対象選択のみ分岐（v1）。0/複数対象は既定解決に委ねる。
    if not uuids or max_n != 1 or min_n > 1:
        return None
    base = manager.default_interaction_payload(pending)
    moves: List[Dict[str, Any]] = []
    for uid in uuids[:HARD_SELECT_CAP]:
        payload = dict(base)
        payload["selected_uuids"] = [uid]
        moves.append({"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": payload})
    # 任意選択（min==0・スキップ可）なら「選ばない」も一級の候補にする。
    if min_n == 0 and bool(pending.get(KEY_SKIP, False)):
        payload = dict(base)
        payload["selected_uuids"] = []
        moves.append({"kind": "game", "action_type": action_api.ACT_RESOLVE_SELECTION, "payload": payload})
    return moves


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
        pending = manager.get_pending_request()
        if not pending:
            break
        pid = pending[KEY_PID]
        action = pending.get(KEY_ACTION)
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
                payload = manager.default_interaction_payload(pending)
                action_api.apply_game_action(manager, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            break
    # 整流の途中/結果で勝敗が確定していたら、_search の winner 検出と同じく **ply 割引**して返す（最短の
    # 止めを優先）。予算切れ settle で勝者を観測した長い手順が、winner 検出（W_WIN-ply）の直接の止めより
    # 高く（生 W_WIN で）見えてしまう不整合を防ぐ＝lethal 認識の ply 割引を一貫させる。
    if manager.winner is not None:
        return (W_WIN - ply) if manager.winner == root_name else -(W_WIN - ply)
    return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)


def _search(manager, root_name: str, alpha: float, beta: float,
            budget: List[int], see_opp_hand: bool, opp_public_only: bool,
            profile=None, ply: int = 0, plan=None, start_turn: int = 0, horizon: int = 1,
            counter_budget: float = 0.0) -> float:
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

    KEY_PID, KEY_ACTION = _pending_keys()
    pending = manager.get_pending_request()
    if not pending:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    actor_name = pending[KEY_PID]
    # 葉: start_turn から horizon ターン進んだ MAIN_ACTION（一定の静止点）で評価。
    if pending.get(KEY_ACTION) == "MAIN_ACTION" and (manager.turn_count - start_turn) >= horizon:
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
    if not moves:
        return evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    is_max = (actor_name == root_name)

    # 公平モデル: 相手 min ノードでは相手の隠れ手札に依存する手を読まない（公開情報のみで応答）。
    if not is_max and opp_public_only:
        filtered = [m for m in moves if not _consumes_hand_card(manager, actor_name, m)]
        if filtered:
            moves = filtered

    # 子ノードを生成し、1-ply 評価でビーム選別（best-first で α-β の枝刈り効率を上げる）。
    children: List[Tuple[float, Any]] = []
    for m in moves:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        child = _apply_clone(manager, actor_name, m, stop_at_select=True)
        if child is None:
            continue
        children.append((evaluate(child, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan), child))
    if not children:
        return _settle_eval(manager, root_name, see_opp_hand, profile, plan, ply)
    children.sort(key=lambda x: x[0], reverse=is_max)
    children = children[:HARD_BEAM]

    if is_max:
        value = float("-inf")
        for _leaf, child in children:
            value = max(value, _search(child, root_name, alpha, beta,
                                       budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                       start_turn, horizon, counter_budget))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        # B-1(b): 相手の推定カウンター応答（normal 保守 min・SELECT_COUNTER・緩衝内で攻撃を防げる場合のみ）。
        # PASS（=カウンターしない＝攻撃が通る）と並べて min が選ぶ＝相手にとって有利な方（=自分に不利な方）。
        # 緩衝を needed ぶん消費して深掘り。これで「緩衝超まで盛ると貫通＝余剰ドンを攻撃に振るのが正」になる。
        if (opp_public_only and profile is not None and counter_budget > 0
                and pending.get(KEY_ACTION) == "SELECT_COUNTER"):
            needed = _counter_needed(manager)
            if needed is not None and needed <= counter_budget:
                cc = _apply_modeled_counter(manager, actor_name, needed)
                if cc is not None:
                    value = min(value, _search(cc, root_name, alpha, beta,
                                               budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                               start_turn, horizon, counter_budget - needed))
                    beta = min(beta, value)
        for _leaf, child in children:
            if alpha >= beta:
                break
            value = min(value, _search(child, root_name, alpha, beta,
                                       budget, see_opp_hand, opp_public_only, profile, ply + 1, plan,
                                       start_turn, horizon, counter_budget))
            beta = min(beta, value)
            if alpha >= beta:
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


def _scored_search(manager, name: str, moves: List[Dict[str, Any]],
                   see_opp_hand: bool, opp_public_only: bool,
                   profile=None, plan=None, collect: Optional[Dict[str, Any]] = None
                   ) -> List[Tuple[float, Dict[str, Any]]]:
    """ルート手を 1-ply で事前選別し、上位 HARD_ROOT_BEAM 手だけを多 ply 先読みで深掘りする。

    全手で予算を共有すると先に列挙された手ほど深く読まれて採点が不公平になるため、
    深掘り対象には**手ごとに均等予算**（HARD_PER_MOVE_BUDGET）を与える。非対象は 1-ply スコアの
    まま残す。事前選別で作った子クローンを深掘りに再利用するので無駄なクローンは作らない。

    `collect`（任意・既定 None＝完全に無オーバーヘッド）が渡されると、regret ログ（検証基盤・§2.5.3）
    用に 1-ply 事前スコアと深掘りスコアを `move_sig -> score` の dict で記録する:
      collect["prelim"]={sig: 1-ply スコア}, collect["deep"]={sig: 深掘りスコア}。
    """
    # 1) 全ルート手を 1-ply で採点（子クローンは深掘りに再利用）。
    prelim: List[Tuple[float, Dict[str, Any], Any]] = []
    for m in moves:
        child = _apply_clone(manager, name, m, stop_at_select=True)
        if child is None:
            prelim.append((float("-inf"), m, None))
            continue
        prelim.append((evaluate(child, name, see_opp_hand=see_opp_hand, profile=profile, plan=plan), m, child))

    # 2) 1-ply 上位を深掘り対象に選ぶ。TURN_END（パスの基準線）は必ず深掘りし、ターン境界で正しく採点する
    #    （非対象の 1-ply スコアは自ターン途中の甘い値になり得るため、パスの基準だけは確実に整える）。
    order = sorted(range(len(prelim)), key=lambda i: prelim[i][0], reverse=True)
    deepen = set(order[:HARD_ROOT_BEAM])
    for i, (_s, m, child) in enumerate(prelim):
        if child is not None and m.get("action_type") == "TURN_END":
            deepen.add(i)
    # B-3: 重要手クラス（除去候補・ブロッカー設置・逆算リーサル/クロック）を 1-ply ランクに関係なく
    # 強制投入する（上限 HARD_FORCE_DEEPEN_CAP・1-ply 上位順＝レイテンシを絞りつつ取りこぼしを是正）。
    forced = 0
    for i in order:
        if forced >= HARD_FORCE_DEEPEN_CAP:
            break
        if i in deepen:
            continue
        _s, m, child = prelim[i]
        if _is_important_root_move(manager, name, m, child):
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
        for s1, m, child in prelim:
            collect["prelim"][_move_sig(m)] = s1
    out: List[Tuple[float, Dict[str, Any]]] = []
    for i, (s1, m, child) in enumerate(prelim):
        if child is not None and i in deepen:
            budget = [HARD_PER_MOVE_BUDGET]
            v = _search(child, name, float("-inf"), float("inf"),
                        budget, see_opp_hand, opp_public_only, profile, ply=1, plan=plan,
                        start_turn=start_turn, horizon=HARD_HORIZON, counter_budget=cbudget)
            out.append((v, m))
            if collect is not None:
                collect["deep"][_move_sig(m)] = v
    if not out:  # 念のため（全候補がクローン失敗）: 1-ply スコアにフォールバック
        out = [(s1, m) for s1, m, _c in prelim]
    return out


# 1 ターン内に CPU が取れる手の総数上限（暴走/無限ループの最終防壁）。
TURN_ACTION_CAP = 60
# 同一の起動効果/ドン付与をこのターン内に繰り返してよい回数の上限。
REPEAT_CAP = 3


def _move_sig(move: Dict[str, Any]) -> tuple:
    payload = move.get("payload") or {}
    return (move.get("action_type"), payload.get("uuid") or move.get("card_uuid"),
            tuple(payload.get("target_ids", []) or []))


def decide(manager, player, difficulty: str = "normal", rng: Optional[random.Random] = None,
           moves: Optional[List[Dict[str, Any]]] = None, profile=None, plan=None) -> Optional[Dict[str, Any]]:
    """`player` が取るべき次の 1 手を返す（合法手が無ければ None）。

    `moves` を渡すとその候補集合から選ぶ（ガード driver が絞り込んだ手を渡す用途）。
    `profile` はリーダー推測の相手モデル（§2.5.4・normal でのみ使用）。
    `plan` は自デッキ勝ち筋プラン（§2.5.5・normal/hard で使用。easy は素の 1-ply のまま）。
    """
    rng = rng or random
    if moves is None:
        moves = manager.get_legal_actions(player)
    # normal/hard: 最上位が単一対象選択なら候補ごとに展開して最善対象を読み切る（easy は既定解決のまま）。
    if difficulty != "easy":
        sel = _selection_moves(manager, player.name)
        if sel:
            moves = sel
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]
    moves = _prune_don_moves(manager, player.name, moves)  # B-2: 無意味なドン付与をルートから除外

    name = player.name
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # 難易度＝情報方針の 3 分化（docs/SPEC.md §2.5.2）:
    #   easy   : 正直な 1-ply 貪欲（ミスなし・公開情報のみ）。
    #   normal : 多 ply 先読み・公開情報のみ＋相手は隠れ手札を使わない保守モデル＋リーダー推測 profile。
    #   hard   : 多 ply 先読み・相手手札も読むフルクローン（最強・チート）。
    if difficulty == "easy":
        scored = [(_simulate_and_eval(manager, name, m, see_opp_hand=False), m) for m in moves]
    elif difficulty == "hard":
        scored = _scored_search(manager, name, moves, see_opp_hand=True, opp_public_only=False,
                                plan=plan)
    else:  # normal
        scored = _scored_search(manager, name, moves, see_opp_hand=False, opp_public_only=True,
                                profile=profile, plan=plan)
    # 同点はランダムタイブレーク（決定論にしたい場合は呼び出し側で seed 済み rng を渡す）。
    rng.shuffle(scored)
    best_score, best_move = max(scored, key=lambda x: x[0])

    # 「何もしない（ターンを畳む）」を一級の選択肢として比較する。非ターン終了手が end を
    # _ACT_MARGIN を超えて上回らなければターンを畳む＝無意味な展開・不利アタック・効かない
    # ドン付与（いずれも改修後の評価では end とほぼ同値）を採らない（進行保証も兼ねる）。
    if end_move is not None and best_move is not end_move:
        end_score = next((s for s, m in scored if m is end_move), None)
        # A-2: 畳み判定マージンをアーキタイプ依存にスケール（aggro=小さく攻めを通す／control=大きく畳む）。
        margin = _ACT_MARGIN * (getattr(plan, "act_margin_mult", 1.0) if plan is not None else 1.0)
        if end_score is not None and best_score <= end_score + margin:
            return end_move
    return best_move


def decide_with_regret(manager, player, difficulty: str = "normal",
                       rng: Optional[random.Random] = None, profile=None, plan=None
                       ) -> Tuple[Optional[Dict[str, Any]], float]:
    """`decide` と同じ手を返しつつ、**greedy regret**（崖エラーの安価な代理・検証基盤・§2.5.3）も返す。

    regret = deep_value(深掘り最善手) − deep_value(1-ply 貪欲が選ぶ手)。
      - deep_value は多 ply 先読みスコア（`_scored_search`）。
      - 1-ply 貪欲手 = 事前選別スコア最大の手（＝浅い読みなら選ぶ手）。常に深掘り集合に入る（prelim 1位）。
    深掘りが浅い読みより良い手を見つけた量＝「1-ply 先読みでは崖に落ちる」局面の信号。常に >= 0。
    easy（1-ply 貪欲）や分岐の無い局面、深掘りスコアが取れない場合は regret=0.0 を返す。
    """
    rng = rng or random
    moves = manager.get_legal_actions(player)
    if difficulty != "easy":
        sel = _selection_moves(manager, player.name)
        if sel:
            moves = sel
    if not moves:
        return None, 0.0
    if len(moves) == 1 or difficulty == "easy":
        return decide(manager, player, difficulty, rng, moves=moves, profile=profile, plan=plan), 0.0

    name = player.name
    moves = _prune_don_moves(manager, name, moves)  # B-2: ルート手集合を decide と一致させる（regret 整合）
    collect: Dict[str, Any] = {}
    if difficulty == "hard":
        _scored_search(manager, name, moves, see_opp_hand=True, opp_public_only=False,
                       plan=plan, collect=collect)
    else:  # normal
        _scored_search(manager, name, moves, see_opp_hand=False, opp_public_only=True,
                       profile=profile, plan=plan, collect=collect)
    move = decide(manager, player, difficulty, rng, moves=moves, profile=profile, plan=plan)
    deep = collect.get("deep", {})
    prelim = collect.get("prelim", {})
    regret = 0.0
    if deep and prelim:
        deep_best = max(deep.values())
        greedy_sig = max(prelim, key=lambda s: prelim[s])  # 1-ply 貪欲が選ぶ手
        greedy_deep = deep.get(greedy_sig)
        if greedy_deep is not None:
            regret = max(0.0, deep_best - greedy_deep)
    return move, regret


def decide_guarded(manager, player, difficulty: str = "normal", rng: Optional[random.Random] = None,
                   mem: Optional[Dict[str, Any]] = None, profile=None, plan=None) -> Optional[Dict[str, Any]]:
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

    moves = manager.get_legal_actions(player)
    if not moves:
        return None
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # 総数キャップ: 上限超過ならターンを畳む（畳めない＝対話中等なら通常選択）。
    if end_move is not None and mem.get("total", 0) >= TURN_ACTION_CAP:
        return end_move

    # 繰り返しキャップ: 起動効果/ドン付与の同一手を上限まで使い切ったら除外する。
    counts = mem.get("counts", {})
    repeatable = {"ACTIVATE_MAIN", "ATTACH_DON"}
    filtered = [m for m in moves
                if not (m.get("action_type") in repeatable and counts.get(_move_sig(m), 0) >= REPEAT_CAP)]
    if not filtered:
        filtered = [end_move] if end_move is not None else moves

    move = decide(manager, player, difficulty, rng, moves=filtered, profile=profile, plan=plan)
    if move is not None:
        sig = _move_sig(move)
        counts[sig] = counts.get(sig, 0) + 1
        mem["counts"] = counts
        mem["total"] = mem.get("total", 0) + 1
    return move
