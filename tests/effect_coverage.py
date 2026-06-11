"""実行カバレッジスクリプト。

全カードの全トリガータイプ能力を GameManager 上で発動し、
手動テストの優先順位付けに使う分類レポートを出力する。

分類:
  ERROR        : 例外発生 → エンジン修正が必要（最優先）
  INTERACTIVE  : 発動中にプレイヤー選択が発生 → 手動テスト必須リスト
  EXECUTED     : 盤面変化ありまたは action_events 記録あり・自動確認済み
  EXECUTED+WARN: EXECUTED かつ効果タイプと盤面変化の方向が不一致（要確認）
  NO_CHANGE    : 発動完了したが変化なし（条件未達 or OTHER の疑い）

has_other フラグ: AST 中に ActionType.OTHER が含まれる（effect_diagnostics の OTHER と同義）

実行:
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show INTERACTIVE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show ERROR
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --show NO_CHANGE
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --trigger ON_PLAY
    OPCG_LOG_SILENT=1 python tests/effect_coverage.py --card OP01-001
"""
import itertools
import os
import re
import sys
import traceback
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.models.models import CardInstance, CardMaster
from opcg_sim.src.models.enums import ActionType, CardType, TriggerType, Phase
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.utils.loader import CardLoader
from engine_helpers import make_master, make_player
from interactive_target_audit import audit_target

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "opcg_sim", "data",
)


# ---------------------------------------------------------------------------
# フィラーカード
# ---------------------------------------------------------------------------

_filler_master: Optional[CardMaster] = None


def _get_filler_master() -> CardMaster:
    global _filler_master
    if _filler_master is None:
        _filler_master = make_master(card_id="FILLER", name="フィラー", power=1000)
    return _filler_master


def _instances(n: int, owner: str) -> List[CardInstance]:
    m = _get_filler_master()
    return [CardInstance(m, owner) for _ in range(n)]


# ---------------------------------------------------------------------------
# テスト用ゲームステート
# ---------------------------------------------------------------------------

def _build_test_state(
    test_card: CardMaster,
    source_in_hand: bool = False,
) -> Tuple[GameManager, Player, Player, CardInstance]:
    """メインフェイズ・リソース豊富な状態のテストゲームを構築する。

    source_in_hand=True のときはテストカードを手札に置く（ON_PLAY 用）。
    """
    p1 = make_player("P1")
    p2 = make_player("P2")

    # P1: ドン 10 枚・手札 5 枚・トラッシュ 10 枚・デッキ 20 枚・ライフ 5 枚
    # フィールド 3 体（フィールド条件・コスト条件を満たす）
    p1.don_deck   = _instances(5,  "P1")
    p1.don_active = _instances(10, "P1")
    p1.hand       = _instances(5,  "P1")
    p1.trash      = _instances(10, "P1")
    p1.deck       = _instances(20, "P1")
    p1.life       = _instances(5,  "P1")
    p1.field      = _instances(3,  "P1")

    # P2: フィールド 3 体・手札 3 枚・ドン 5 枚・デッキ 20 枚・ライフ 5 枚
    p2.don_active = _instances(5,  "P2")
    p2.field      = _instances(3,  "P2")
    p2.hand       = _instances(3,  "P2")
    p2.deck       = _instances(20, "P2")
    p2.life       = _instances(5,  "P2")

    gm             = GameManager(p1, p2)
    gm.turn_player = p1
    gm.opponent    = p2
    gm.turn_count  = 2
    gm.phase       = Phase.MAIN

    source = CardInstance(test_card, "P1")
    if source_in_hand:
        p1.hand.append(source)
    elif test_card.type == CardType.LEADER:
        p1.leader = source
    elif test_card.type == CardType.STAGE:
        p1.stage = source
    else:
        p1.field.append(source)

    return gm, p1, p2, source


def _snap(p1: Player, p2: Player) -> tuple:
    return (
        len(p1.hand), len(p1.field), len(p1.trash), len(p1.deck), len(p1.life),
        len(p1.don_active) + len(p1.don_rested),
        len(p2.hand), len(p2.field), len(p2.trash), len(p2.deck), len(p2.life),
        1 if getattr(p1, "stage", None) else 0,
        1 if getattr(p2, "stage", None) else 0,
    )


_SNAP_KEYS = (
    "p1_hand", "p1_field", "p1_trash", "p1_deck", "p1_life", "p1_don",
    "p2_hand", "p2_field", "p2_trash", "p2_deck", "p2_life",
    "p1_stage", "p2_stage",
)


def _snap_diff(before: tuple, after: tuple) -> str:
    return ", ".join(
        f"{_SNAP_KEYS[i]}:{before[i]}→{after[i]}"
        for i in range(len(_SNAP_KEYS))
        if before[i] != after[i]
    )


# ---------------------------------------------------------------------------
# ステータススナップショット（H-1: 枚数に出ない変化の測定）
# ---------------------------------------------------------------------------

def _stat_snap(p1: Player, p2: Player) -> dict:
    """盤面カードの power/cost/keyword/flag/rest、手札カードの cost、ドンの
    active/rested 構成を uuid キーで記録する。

    ゾーン枚数スナップショットでは見えない BUFF / GRANT_KEYWORD / REST /
    COST_CHANGE 等の変化を検出可能にする。
    """
    out: dict = {}
    for p in (p1, p2):
        units = ([p.leader] if p.leader else []) + list(p.field)
        if getattr(p, "stage", None):
            units.append(p.stage)
        for c in units:
            out[c.uuid] = (
                c.get_power(True),
                c.current_cost,
                frozenset(c.current_keywords | c.timed_keywords),
                frozenset(c.flags | c.timed_flags),
                c.is_rest,
            )
        for c in p.hand:
            out[c.uuid] = (0, c.current_cost, frozenset(), frozenset(), False)
        out[f"__don__{p.name}"] = (
            len(p.don_active), len(p.don_rested), frozenset(), frozenset(), False,
        )
    return out


def _stat_changed(sb: dict, sa: dict, ignore=frozenset()) -> bool:
    """両スナップショットに存在する uuid のみ比較する（移動した/新規のカードは
    タプル表現が変わるため対象外。ignore はソースカード等の除外用）。"""
    keys = (sb.keys() & sa.keys()) - set(ignore)
    return any(sb[k] != sa[k] for k in keys)


def _zone_fingerprint(p1: Player, p2: Player) -> dict:
    """全カード uuid → 所在ゾーン名 のマップ。ゾーン枚数では相殺されて見えない
    「グロスのカード移動」を検出するための指紋（例: ドロー+手札→デッキの cost が
    枚数では純0でも、各カードの所在は変わる）。don も含める。"""
    out: dict = {}
    for p in (p1, p2):
        pref = p.name
        zones = {
            "hand": p.hand, "field": p.field, "trash": p.trash,
            "deck": p.deck, "life": p.life, "temp": getattr(p, "temp_zone", []),
            "don_active": p.don_active, "don_rested": p.don_rested,
            "don_attached": getattr(p, "don_attached_cards", []),
        }
        for zname, zone in zones.items():
            for c in zone:
                out[f"{pref}:{c.uuid}"] = zname
        if p.leader:
            out[f"{pref}:{p.leader.uuid}"] = "leader"
        if getattr(p, "stage", None):
            out[f"{pref}:{p.stage.uuid}"] = "stage"
    return out


def _moved(fb: dict, fa: dict) -> bool:
    """指紋 fb→fa でカードがゾーン移動した（=何か実行された）か。"""
    for k in fb.keys() | fa.keys():
        if fb.get(k) != fa.get(k):
            return True
    return False


def _stat_diff(sb: dict, sa: dict, ignore=frozenset()) -> str:
    parts = []
    for k in sorted(sb.keys() & sa.keys()):
        if k in ignore or sb[k] == sa[k]:
            continue
        b, a = sb[k], sa[k]
        sub = []
        if b[0] != a[0]:
            sub.append(f"pw{b[0]}→{a[0]}")
        if b[1] != a[1]:
            sub.append(f"co{b[1]}→{a[1]}")
        if b[2] != a[2]:
            added = sorted(a[2] - b[2]); removed = sorted(b[2] - a[2])
            sub.append("kw" + "+".join(added) + ("-" + "-".join(removed) if removed else ""))
        if b[3] != a[3]:
            added = sorted(a[3] - b[3]); removed = sorted(b[3] - a[3])
            sub.append("fl" + "+".join(added) + ("-" + "-".join(removed) if removed else ""))
        if b[4] != a[4]:
            sub.append("rest" if a[4] else "active")
        parts.append(f"{str(k)[:8]}:{','.join(sub)}")
        if len(parts) >= 5:
            parts.append("…")
            break
    return "; ".join(parts)


def _selection_issues(ia: dict) -> List[str]:
    """SELECT_TARGET 中断時、実際の選択候補をテキスト由来の制約と突き合わせる（H-4）。

    静的監査（audit_target: クエリ vs テキスト）に加え、実行時の候補カードが
    テキストの側/コスト上限/特徴と矛盾しないかを検証する。
    """
    cont  = ia.get("continuation") or {}
    query = cont.get("query")
    stack = cont.get("execution_stack") or []
    node  = stack[-1] if stack else None
    raw   = getattr(node, "raw_text", "") or ""
    if not raw or query is None:
        return []

    issues = [f"静的:{x}" for x in audit_target(raw, query)]

    r   = unicodedata.normalize("NFKC", raw)
    sel = set(ia.get("selectable_uuids") or [])
    cands = [c for c in ia.get("candidates", []) if not sel or c.uuid in sel]
    if cands:
        has_aite  = re.search(r"相手の[^。、]*?(キャラ|リーダー)", r) and "自分の" not in r
        has_jibun = re.search(r"自分の[^。、]*?(キャラ|リーダー)", r) and "相手の" not in r
        owners = {c.owner_id for c in cands}
        if has_aite and owners == {"P1"}:
            issues.append("実行時:相手対象だが候補が全て自分側")
        if has_jibun and owners == {"P2"}:
            issues.append("実行時:自分対象だが候補が全て相手側")
        m = re.search(r"コスト(\d+)以下", r)
        if m:
            cap  = int(m.group(1))
            over = sum(1 for c in cands if c.current_cost > cap)
            if over:
                issues.append(f"実行時:コスト{cap}以下指定だが上限超過候補 {over} 枚")
        traits = [t for t in re.findall(r"《([^》]+)》", r)
                  if t not in ("斬", "打", "特", "知", "活")]
        if traits and not any(set(c.master.traits) & set(traits) for c in cands):
            issues.append(f"実行時:特徴{traits}を持つ候補ゼロ")

    deduped: List[str] = []
    for x in issues:
        if x not in deduped:
            deduped.append(x)
    return [f"«{raw[:30]}» {x}" for x in deduped]


def _smart_drain(
    gm: GameManager,
    limit: int = 30,
    choice_plan: Optional[List[int]] = None,
    record: Optional[dict] = None,
) -> Tuple[bool, int]:
    """インタラクションを賢㍖に解決する。

    - SELECT_TARGET: selectable_uuids から constraints.min 枚を選ぶ（空選択でなく有効な候補を渡す）
    - CHOICE: choice_plan から index を消費する（既定は先頭=0。H-3 のパス列挙用）
    - record に choices（遭遇した選択肢数）と select_issues（H-4 検証結果）を記録する
    - (stuck, 処理数) を返す
    """
    count = 0
    plan  = list(choice_plan or [])
    while gm.active_interaction and count < limit:
        ia          = gm.active_interaction
        player      = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        action_type = ia.get("action_type", "")

        if action_type == "SELECT_TARGET":
            if record is not None:
                iss = _selection_issues(ia)
                if iss:
                    record.setdefault("select_issues", []).extend(iss)
            candidates = ia.get("selectable_uuids") or [
                c.uuid for c in ia.get("candidates", [])
            ]
            cons     = ia.get("constraints") or {}
            min_req  = cons.get("min", 0)
            max_req  = cons.get("max", 1)
            # max が -1（=ALL/REMAINING、例「残りをデッキの下に置く」）の場合は全候補を選ぶ。
            # 従来は max(min,1)=1枚しか選ばず、残りの temp カードが取り残されて
            # TEMP リーク（デッキから消失）に見える測定アーティファクトを生んでいた。
            if max_req is not None and max_req < 0:
                n_select = len(candidates)
            else:
                n_select = max(min_req, 1) if candidates else 0
                if max_req:
                    n_select = min(n_select, max_req)
            selected = candidates[:n_select]
            payload  = {"selected_uuids": selected, "index": 0}
        elif action_type == "CHOICE":
            n_opt = len(ia.get("options") or []) or 2
            idx   = plan.pop(0) if plan else 0
            if record is not None:
                record.setdefault("choices", []).append(n_opt)
            payload = {"selected_uuids": [], "index": min(idx, n_opt - 1)}
        else:  # CONFIRM_OPTIONAL / DECLARE_COST 等は先頭の選戠肢
            payload = {"selected_uuids": [], "index": 0}

        try:
            gm.resolve_interaction(player, payload)
        except Exception:
            break
        count += 1
    return count >= limit, count


# ---------------------------------------------------------------------------
# AST ウォーク
# ---------------------------------------------------------------------------

def _walk(node):
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk(a)
    elif isinstance(node, Branch):
        yield from _walk(node.if_true)
        if node.if_false:
            yield from _walk(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk(o)


def _has_other_in(abilities) -> bool:
    for ab in abilities:
        for act in list(_walk(ab.effect)) + list(_walk(ab.cost)):
            if act and act.type == ActionType.OTHER:
                return True
    return False


# ---------------------------------------------------------------------------
# 方向性アサーション（改善第二値なし）
# ---------------------------------------------------------------------------

# ActionType 名 → (変化前 tuple, 変化後 tuple) から「期待方向」を返す関数
# snap インデックス: 0=p1_hand 1=p1_field 2=p1_trash 3=p1_deck 4=p1_life 5=p1_don
#                       6=p2_hand 7=p2_field 8=p2_trash 9=p2_deck 10=p2_life
# 方向チェックは「対象がどちら側か」に依存しない side-agnostic で書く。対象側を固定して
# 書くと、相手キャラを KO/デッキ送り/ハンデスする効果や自己バウンス等が軒並み誤検知になる
# （effect_coverage の汎用足場では対象側が一意に定まらないため）。
_DIRECTION: Dict[str, any] = {
    "DRAW":             lambda b, a: a[0] > b[0],                       # 自分の手札増
    "DISCARD":          lambda b, a: a[0] < b[0] or a[6] < b[6],        # どちらかの手札減
    "KO":               lambda b, a: a[1] < b[1] or a[7] < b[7],        # どちらかのフィールド減
    "BOUNCE":           lambda b, a: a[1] < b[1] or a[7] < b[7],        # どちらかのフィールド減
    "RAMP_DON":         lambda b, a: a[5] > b[5],                       # ドン増
    "RETURN_DON":       lambda b, a: a[5] < b[5],                       # ドン減
    "HEAL":             lambda b, a: a[4] > b[4] or a[10] > b[10],      # どちらかのライフ増
    "TRASH_FROM_DECK":  lambda b, a: a[3] < b[3] or a[9] < b[9],        # どちらかのデッキ減
    # PLAY_CARD: フィールド/ステージ増、または公開/サーチでデッキが減る派生も許容（look-and-play 系）
    "PLAY_CARD":        lambda b, a: (a[1] > b[1] or a[7] > b[7] or a[3] < b[3] or a[9] < b[9]
                                      or a[11] > b[11] or a[12] > b[12]),
    # DECK_BOTTOM: どちらかの場/手札減＋どちらかのデッキ増（「相手は自身の手札1枚を
    # デッキの下に置く」のような手札→デッキも正当）
    "DECK_BOTTOM":      lambda b, a: (a[1] < b[1] or a[7] < b[7] or a[0] < b[0] or a[6] < b[6])
                                     and (a[3] > b[3] or a[9] > b[9]),
    "TRASH_FROM_HAND":  lambda b, a: a[0] < b[0] or a[6] < b[6],        # どちらかの手札減
    "MOVE":             lambda b, a: b != a,                            # 何か変わればOK
}


def _card_keys(sb: dict, sa: dict, ignore):
    """両スナップショットに存在するカード uuid（ドン擬似キー除く）。"""
    return [k for k in (sb.keys() & sa.keys()) - set(ignore)
            if not str(k).startswith("__don__")]


def _any_card(sb, sa, ignore, fn) -> bool:
    return any(fn(sb[k], sa[k]) for k in _card_keys(sb, sa, ignore))


# ステータス差分（H-1）に基づく方向チェック。ゾーン枚数では検証できなかった
# BUFF / キーワード付与 / レスト切替 / コスト変更を検証する。
_STAT_DIRECTION: Dict[str, any] = {
    # BUFF はパワー以外に BLOCKER_DISABLE 等の status バリエーションを持つため
    # 「何らかのカードステータス変化」を期待方向とする
    "BUFF":           lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b != a),
    "BP_BUFF":        lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[0] != a[0]),
    "SET_BASE_POWER": lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[0] != a[0]),
    "SWAP_POWER":     lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[0] != a[0]),
    "COST_CHANGE":    lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[1] != a[1]),
    "COST_BUFF":      lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[1] != a[1]),
    "SET_COST":       lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[1] != a[1]),
    "GRANT_KEYWORD":  lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[2] != a[2]),
    "REST":           lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: not b[4] and a[4]),
    "ACTIVE":         lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: b[4] and not a[4]),
    "FREEZE":         lambda sb, sa, ig: _any_card(sb, sa, ig, lambda b, a: a[3] - b[3]),
    "REST_DON":       lambda sb, sa, ig: any(
        sa[k][0] < sb[k][0] for k in (sb.keys() & sa.keys()) if str(k).startswith("__don__")),
    # 両プレイヤーのドン総数（active+rested）減少/増加。ゾーンスナップは p1_don しか
    # 持たないため、相手ドンの返却/追加はここで判定する。
    "RETURN_DON":     lambda sb, sa, ig: any(
        sa[k][0] + sa[k][1] < sb[k][0] + sb[k][1]
        for k in (sb.keys() & sa.keys()) if str(k).startswith("__don__")),
    "RAMP_DON":       lambda sb, sa, ig: any(
        sa[k][0] + sa[k][1] > sb[k][0] + sb[k][1]
        for k in (sb.keys() & sa.keys()) if str(k).startswith("__don__")),
}


def _soft_assert(
    abilities,
    before: tuple,
    after: tuple,
    sb: Optional[dict] = None,
    sa: Optional[dict] = None,
    ignore=frozenset(),
) -> Optional[str]:
    """単一効果タイプの能力に限り、盤面変化の方向が期待値と逆の場合に警告を返す。

    ゾーン枚数（_DIRECTION）とカードステータス（_STAT_DIRECTION）の両面で判定し、
    どちらかが期待方向を満たせば OK とする。
    複合効果（DRAW+KO 等）は誤検知が多いためチェック対象外。
    変化なし（条件未達の可能性）は警告しない。
    """
    stat_moved = sb is not None and sa is not None and _stat_changed(sb, sa, ignore)
    if before == after and not stat_moved:
        return None  # 変化なし = 条件未達の可能性が高いためスキップ

    type_names = [
        act.type.name
        for ab in abilities
        for act in _walk(ab.effect)
        if act and act.type != ActionType.OTHER
    ]
    if not type_names:
        return None

    # 複数のタイプが混在する場合は誤検知が多いためチェックしない。
    # 「方向マップに無いタイプを無視」すると複合効果（HEAL+ダメージ、LOOK+手札追加等）の
    # 相殺で誤検知するため、能力の全アクションが単一タイプの場合のみ判定する。
    all_types = set(type_names)
    if len(all_types) != 1:
        return None
    unique_types = {n for n in all_types if n in _DIRECTION or n in _STAT_DIRECTION}
    if len(unique_types) != 1:
        return None

    (dominant,) = unique_types
    # ゾーン系アクション（_DIRECTION のみ）でゾーン枚数が一切動いていない場合は、
    # 同カードの別能力（PASSIVE等）によるステータス変化を誤って方向不一致と
    # 判定しない（対象不在の no-op + 測定ノイズ）。
    if dominant not in _STAT_DIRECTION and before == after:
        return None
    ok = False
    if dominant in _DIRECTION and _DIRECTION[dominant](before, after):
        ok = True
    if not ok and dominant in _STAT_DIRECTION and sb is not None and sa is not None:
        ok = _STAT_DIRECTION[dominant](sb, sa, ignore)
    if not ok:
        return f"WARN: {dominant} 期待だが方向不一致"
    return None


# ---------------------------------------------------------------------------
# 分類
# ---------------------------------------------------------------------------

_PRIORITY = {"ERROR": 4, "INTERACTIVE": 3, "EXECUTED": 2, "NO_CHANGE": 1}


@dataclass
class AbilityResult:
    card_id:   str
    name:      str
    trigger:   str    # TriggerType.name
    status:    str    # ERROR / INTERACTIVE / EXECUTED / NO_CHANGE
    has_other: bool = False
    detail:    str  = ""
    select_issues: str = ""   # H-4: 対話選択候補とテキストの矛盾
    choices:   tuple = ()     # H-3: 遭遇した CHOICE の選択肢数（パス列挙用）


def _outcome(
    gm: GameManager,
    p1: Player,
    before: tuple,
    card_id: str,
    name: str,
    trig: str,
    h_other: bool,
    abilities=None,
    sb: Optional[dict] = None,
    ignore=frozenset(),
    record: Optional[dict] = None,
    before_eff: Optional[tuple] = None,
    choice_plan: Optional[List[int]] = None,
) -> AbilityResult:
    """発動後のゲーム状態から結果を返す。インタラクションは賢㍖選戠で消化。

    before_eff: 登場アーティファクト控除後の基準スナップショット（H-2）。
    指定時は「効果由来の変化」をこの基準と比較して判定する。
    """
    if record is None:
        record = {}
    stuck, n_ia = _smart_drain(gm, choice_plan=choice_plan, record=record)
    after = _snap(p1, gm.p2)
    sa    = _stat_snap(p1, gm.p2) if sb is not None else None
    base  = before_eff if before_eff is not None else before
    stat_moved = sb is not None and _stat_changed(sb, sa, ignore)
    changed    = base != after or stat_moved
    # resolve_ability が登録するアクション履歴（REVEAL/SHUFFLE を EXECUTED に分類するため）
    has_events = bool(getattr(gm, "action_events", []))
    sel_iss = " / ".join(record.get("select_issues", []))
    chs     = tuple(record.get("choices", []))

    if stuck or gm.active_interaction:
        msg = gm.active_interaction.get("message", "") if gm.active_interaction else "stuck in drain loop"
        return AbilityResult(card_id, name, trig, "INTERACTIVE", h_other, msg, sel_iss, chs)
    if changed:
        diff = _snap_diff(base, after) or "(zone変化なし)"
        if sb is not None and stat_moved:
            sd = _stat_diff(sb, sa, ignore)
            if sd:
                diff = f"{diff}  |  stat: {sd}"
        warn = _soft_assert(abilities or [], base, after, sb, sa, ignore) if abilities else None
        detail = f"{diff}  |  {warn}" if warn else diff
        return AbilityResult(card_id, name, trig, "EXECUTED", h_other, detail, sel_iss, chs)
    if n_ia > 0:
        return AbilityResult(card_id, name, trig, "INTERACTIVE", h_other,
                             f"{n_ia} interaction(s) smart-resolved (no state change)", sel_iss, chs)
    if has_events:
        return AbilityResult(card_id, name, trig, "EXECUTED", h_other,
                             "ability fired (no visible state change)", sel_iss, chs)
    return AbilityResult(card_id, name, trig, "NO_CHANGE", h_other,
                         "no state change, no interaction", sel_iss, chs)


def _play_artifact(before: tuple, master: CardMaster) -> tuple:
    """登場行為そのものによるゾーン変化（手札→場/トラッシュ）を before に織り込む（H-2）。

    play_card_action はテストカードを手札から動かすため、効果が何もしなくても
    「手札-1/場+1」等の差分が残り、方向チェックを汚染していた（PLAY_ARTIFACT）。
    """
    art = list(before)
    art[0] -= 1  # p1_hand
    if master.type == CardType.EVENT:
        art[2] += 1  # p1_trash（イベントは解決後トラッシュへ）
    elif master.type == CardType.CHARACTER:
        art[1] += 1  # p1_field
    elif master.type == CardType.STAGE:
        art[11] = 1  # p1_stage スロット（既存ステージは置換でトラッシュへ行くが汎用盤面では空）
    return tuple(art)


def _test_on_play(
    master: CardMaster, h_other: bool, choice_plan: Optional[List[int]] = None,
) -> AbilityResult:
    """ON_PLAY: play_card_action を経由（全 ON_PLAY 能力を一括発動）。"""
    on_play_abs = [ab for ab in master.abilities
                   if (ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)) == "ON_PLAY"]
    try:
        gm, p1, p2, source = _build_test_state(master, source_in_hand=True)
    except Exception as e:
        return AbilityResult(master.card_id, master.name, "ON_PLAY", "ERROR", h_other, f"setup: {e}")

    before = _snap(p1, p2)
    sb     = _stat_snap(p1, p2)
    try:
        gm.play_card_action(p1, source)
    except Exception:
        return AbilityResult(master.card_id, master.name, "ON_PLAY", "ERROR", h_other,
                             traceback.format_exc(limit=2))

    return _outcome(gm, p1, before, master.card_id, master.name, "ON_PLAY", h_other,
                    on_play_abs, sb=sb, ignore=frozenset({source.uuid}),
                    before_eff=_play_artifact(before, master), choice_plan=choice_plan)


def _test_ability(
    master: CardMaster, ability, trig: str, h_other: bool,
    choice_plan: Optional[List[int]] = None,
) -> AbilityResult:
    """resolve_ability を直接呼んでテスト（ON_PLAY 以外の全トリガー）。

    TRIGGER（ライフ公開時）はソースが場ではなく手札（ライフ由来）にある状態が実態に
    近い。「このカードを登場させる」が field→field の見えない移動になり方向検査を
    誤判定しないよう、手札起点で発動する。
    """
    try:
        gm, p1, p2, source = _build_test_state(master, source_in_hand=(trig == "TRIGGER"))
    except Exception as e:
        return AbilityResult(master.card_id, master.name, trig, "ERROR", h_other, f"setup: {e}")

    before = _snap(p1, p2)
    sb     = _stat_snap(p1, p2)
    try:
        gm.resolve_ability(p1, ability, source)
    except Exception:
        return AbilityResult(master.card_id, master.name, trig, "ERROR", h_other,
                             traceback.format_exc(limit=2))

    return _outcome(gm, p1, before, master.card_id, master.name, trig, h_other,
                    [ability], sb=sb, choice_plan=choice_plan)


_MAX_PATHS = 8


def _run_paths(fn) -> AbilityResult:
    """CHOICE の全パスを列挙して最悪結果を返す（H-3、上限 _MAX_PATHS 実行）。

    fn(choice_plan) -> AbilityResult。初回（全て index=0）で遭遇した選択肢数から
    残りパスを列挙する。深さが実行ごとに変わる場合も choice_plan の余剰は無害。
    """
    first = fn(None)
    counts = first.choices
    if not counts or all(n <= 1 for n in counts):
        return first

    results = [first]
    all_paths = list(itertools.product(*[range(n) for n in counts]))
    for path in all_paths[1:_MAX_PATHS]:
        results.append(fn(list(path)))

    worst = max(results, key=lambda r: (
        _PRIORITY.get(r.status, 0),
        1 if "WARN" in (r.detail or "") else 0,
        1 if r.select_issues else 0,
    ))
    # 全パスの select_issues を統合する
    merged: List[str] = []
    for r in results:
        for x in (r.select_issues.split(" / ") if r.select_issues else []):
            if x and x not in merged:
                merged.append(x)
    if worst is not first and worst.status != first.status:
        worst.detail = f"[path={all_paths[results.index(worst)]}] {worst.detail}"
    worst.select_issues = " / ".join(merged)
    return worst


def classify(master: CardMaster) -> List[AbilityResult]:
    """カードの全能力をトリガータイプ別にテストし、各タイプの最悪結果を返す。"""
    if not master.abilities:
        return []

    h_other_all = _has_other_in(master.abilities)

    by_trigger: Dict[str, list] = defaultdict(list)
    for ab in master.abilities:
        k = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
        by_trigger[k].append(ab)

    results: List[AbilityResult] = []
    for trig, abilities in by_trigger.items():
        h_other = _has_other_in(abilities) or h_other_all

        if trig == "ON_PLAY":
            # play_card_action が全 ON_PLAY を発動するためまとめてテスト
            r = _run_paths(lambda plan: _test_on_play(master, h_other, choice_plan=plan))
        else:
            r = None
            for ability in abilities:
                ri = _run_paths(
                    lambda plan, _ab=ability: _test_ability(master, _ab, trig, h_other, choice_plan=plan))
                if r is None or _PRIORITY.get(ri.status, 0) > _PRIORITY.get(r.status, 0):
                    r = ri

        if r:
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

_LABELS = {
    "ERROR":       "--- ERROR ({n} 件) --- ← 要修正",
    "INTERACTIVE": "--- INTERACTIVE ({n} 件) --- ← 手動テスト優先リスト",
    "EXECUTED":    "--- EXECUTED ({n} 件) ---",
    "NO_CHANGE":   "--- NO_CHANGE ({n} 件) --- ← 条件未達 or OTHER の疑い",
}


def _print_section(items: List[AbilityResult], status: str) -> None:
    label = _LABELS.get(status, f"--- {status} ({{n}} 件) ---").format(n=len(items))
    print(label)
    if not items:
        print("  (なし)")
    else:
        for r in sorted(items, key=lambda x: (x.card_id, x.trigger)):
            other_mark = " [OTHER]" if r.has_other else ""
            warn_mark  = " [WARN]"  if r.detail and "WARN" in r.detail else ""
            print(f"  [{r.trigger:<20}] {r.card_id:<12}  {r.name}{other_mark}{warn_mark}")
            if r.detail:
                print(f"    └ {r.detail[:120]}")
    print()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run(
    show:        Optional[str] = None,
    card_filter: Optional[str] = None,
    trig_filter: Optional[str] = None,
) -> None:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()

    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    all_results: List[AbilityResult] = []
    skipped = 0
    total   = len(card_ids)

    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None:
            continue
        card_results = classify(master)
        if not card_results:
            skipped += 1
            continue
        for r in card_results:
            if trig_filter is None or r.trigger == trig_filter:
                all_results.append(r)

    sys.stderr.write(f"\r完了: {total} カード処理済み\n")

    # --- サマリ ---
    counts = Counter(r.status for r in all_results)
    warn_count = sum(1 for r in all_results if r.status == "EXECUTED" and "WARN" in r.detail)
    sel_count  = sum(1 for r in all_results if r.select_issues)
    print("=== 実行カバレッジ (能力単位) ===")
    print(f"  能力なし(SKIP)  : {skipped:4d}")
    for s in ("ERROR", "INTERACTIVE", "EXECUTED", "NO_CHANGE"):
        o    = sum(1 for r in all_results if r.status == s and r.has_other)
        note = f"  (うち OTHER フラグ {o} 件)" if o else ""
        print(f"  {s:<14}: {counts[s]:4d}{note}")
    if warn_count:
        print(f"  WARN 付き EXECUTED: {warn_count:4d}  ← 方向不一致の疑い（要確認）")
    if sel_count:
        print(f"  SELECT_MISMATCH : {sel_count:4d}  ← 選択候補とテキストの矛盾（H-4）")
    print()

    # --- トリガー別内訳 ---
    print("--- トリガー別内訳 ---")
    trig_status: Dict[str, Counter] = defaultdict(Counter)
    for r in all_results:
        trig_status[r.trigger][r.status] += 1
    for trig in sorted(trig_status):
        cs    = trig_status[trig]
        parts = [f"{s}={cs[s]}" for s in ("ERROR", "INTERACTIVE", "EXECUTED", "NO_CHANGE") if cs[s]]
        print(f"  {trig:<22}  {', '.join(parts)}")
    print()

    # --- 詳細リスト ---
    # --show 指定時はその分類のみ、デフォルトは ERROR と INTERACTIVE
    targets = [show] if show else ["ERROR", "INTERACTIVE"]
    for target in targets:
        items = [r for r in all_results if r.status == target]
        _print_section(items, target)

    # WARN 付き EXECUTED も常に表示（--show 指定なし or --show EXECUTED 時）
    if not show or show == "EXECUTED":
        warn_items = [r for r in all_results if r.status == "EXECUTED" and "WARN" in r.detail]
        if warn_items:
            print(f"--- WARN 付き EXECUTED ({len(warn_items)} 件) --- ← 方向不一致の疑い（要確認）")
            for r in sorted(warn_items, key=lambda x: (x.card_id, x.trigger)):
                print(f"  [{r.trigger:<20}] {r.card_id:<12}  {r.name}")
                print(f"    └ {r.detail[:120]}")
            print()

    # SELECT_MISMATCH（H-4）も常に表示
    if not show or show == "SELECT":
        sel_items = [r for r in all_results if r.select_issues]
        if sel_items:
            print(f"--- SELECT_MISMATCH ({len(sel_items)} 件) --- ← 選択候補とテキストの矛盾（H-4）")
            for r in sorted(sel_items, key=lambda x: (x.card_id, x.trigger)):
                print(f"  [{r.trigger:<20}] {r.card_id:<12}  {r.name}")
                print(f"    └ {r.select_issues[:160]}")
            print()


if __name__ == "__main__":
    show_opt = None
    card_opt = None
    trig_opt = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--show" and i + 1 < len(args):
            show_opt = args[i + 1]; i += 2
        elif args[i] == "--card" and i + 1 < len(args):
            card_opt = args[i + 1]; i += 2
        elif args[i] == "--trigger" and i + 1 < len(args):
            trig_opt = args[i + 1]; i += 2
        else:
            i += 1

    run(show=show_opt, card_filter=card_opt, trig_filter=trig_opt)
