"""実対局リプレイの再生側（R1・`docs/replay_verification_plan.md`）。

記録記述子（seed＋leaders＋decks＋first_player＋人間アクション列）から対局を**再構築・再生**する:
  - デッキは記録の card_id 列から復元（`Player` は deck を JournaledList へコピー＝記録は pre-shuffle 順。
    `random.seed(seed)`＋`start_game` で同一シャッフルを再現）。
  - **人間手番は記録アクションを注入**（R0 確定の (A) 決定論タイブレーク逆引き＝記述子に一致する合法手の
    列挙順先頭を採る。曖昧率 3.5〜4.5%・fan-out 小・場複製の残差は round-trip で検出）。
  - **CPU 手番は再 decide**（同一 seed から再計算＝一致が決定論の証明）。

対局ループは `game_driver.run_game` を共有し、人間席＝注入リゾルバ・CPU席＝learned/hard 席を差すだけ。
記録側ヘルパ `record_descriptor` は round-trip テスト用（実際の記録は `services/replay.py`＝API 側）。
"""
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from game_driver import run_game, make_seat, GameResult, InvariantError
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.models.models import CardInstance


# --- デッキ復元 --------------------------------------------------------------

def build_deck_from_ids(db, leader_id: Optional[str], card_ids: List[str], owner_id: str):
    """記録の card_id 列（pre-shuffle 順・複製あり）からデッキを復元する。"""
    leader = None
    if leader_id:
        lm = db.get_card(leader_id)
        if lm is not None:
            leader = CardInstance(lm, owner_id)
    cards: List[CardInstance] = []
    for cid in card_ids:
        m = db.get_card(cid)
        if m is None:
            raise ValueError(f"復元不能な card_id: {cid}")
        cards.append(CardInstance(m, owner_id))
    return leader, cards


# --- 人間アクションの逆写像（R0 確定の (A) タイブレーク） --------------------

def resolve_recorded_action(manager, actor, recorded: Dict[str, Any]):
    """記録記述子（`{action_type, card, targets}`・card_id 基準）を、現局面の合法手へ逆写像する。

    記述子に一致する合法手のうち**列挙順の先頭**を採る（録画・再生で列挙順が同一＝決定論なら安全）。
    手札複製は同一カード＝挙動等価。場複製の一部だけが分岐リスクで round-trip が検出する（R0 §5）。
    一致が無ければ None（＝再生不能。round-trip テストが検出）。
    """
    want = _key(recorded)
    for mv in manager.get_legal_actions(actor):
        if _key(cpu_ai._describe_move(manager, mv)) == want:
            return mv
    # 素の合法手に無い記述子＝決定層だけが持つ代替手（任意効果の decline: accepted=False や
    # up-to の見送り等）は merged_search_actions の展開候補から逆写像する（一致時のみ・挙動追加）。
    try:
        base = manager.get_legal_actions(actor)
        for mv in cpu_ai.merged_search_actions(manager, actor.name, base):
            if _key(cpu_ai._describe_move(manager, mv)) == want:
                return mv
    except Exception:
        pass
    return None


def _key(desc: Optional[Dict[str, Any]]):
    if not desc:
        return None
    # accepted: 任意効果の decline のみ False が載る（accept・旧録画は欠落=None で同キー＝互換）。
    return (desc.get("action_type"), desc.get("card"), tuple(desc.get("targets") or ()),
            tuple(desc.get("selected") or ()), desc.get("index"), desc.get("position"),
            desc.get("accepted"))


def _cpu_seat(difficulty: str, sims: int = 160):
    """再生・録画の CPU 席を作る（learned=Gen2 席／それ以外=L1 arena 席）。record/replay で共通。"""
    if difficulty == "learned":
        return make_seat(kind="learned", sims=sims)
    return make_seat(difficulty, kind="arena")


class _HumanReplaySeat:
    """記録された人間アクション列を順に注入する席（`seat(ctx)->move`）。"""

    def __init__(self, actions: List[Dict[str, Any]], resolver=None):
        self._actions = list(actions)
        self._i = 0
        self._resolve = resolver or resolve_recorded_action   # 既定＝従来（roundtrip 挙動不変）
        self.misses: List[Dict[str, Any]] = []   # 逆写像不能・列消尽を記録（round-trip 診断）

    def __call__(self, ctx):
        if self._i >= len(self._actions):
            self.misses.append({"reason": "actions_exhausted", "step": ctx.step})
            return None
        rec = self._actions[self._i]
        self._i += 1
        mv = self._resolve(ctx.manager, ctx.actor, rec)
        if mv is None:
            self.misses.append({"reason": "no_match", "step": ctx.step, "recorded": rec})
        return mv


def resolve_api_action(manager, actor, recorded: Dict[str, Any]):
    """API 実対局の記録語彙をエンジン合法手へ逆写像する（`state_at_action` 用の拡張リゾルバ）。

    標準リゾルバで一致しない API 固有の表現を追加で写像する:
      - `ATTACK_CONFIRM/攻撃者X`（対象欠落あり）→ エンジン `ATTACK/X`。対象は記録があればそれ、
        無ければ合法対象が一意ならそれ、複数なら相手リーダー優先（API の主経路）。
      - `RESOLVE_EFFECT_SELECTION`＝効果対話の人間解決。`get_legal_actions` は既定解決
        （`default_interaction_payload`＝min 件先頭選択）の**1手しか列挙しない**ため、人間の実選択
        （`selected` の card_id 列）は合法手列挙に現れない。pending request の候補
        （selectable_uuids）へ card_id を列挙順・重複消費で写像し、payload を直接構築する。
    合成録画の roundtrip（`resolve_recorded_action` 既定）には手を入れない＝挙動不変。"""
    mv = resolve_recorded_action(manager, actor, recorded)
    if mv is not None:
        return mv
    if recorded.get("action_type") == "RESOLVE_EFFECT_SELECTION":
        mv = _resolve_dialog_action(manager, actor, recorded)
        if mv is not None:
            return mv
    if recorded.get("action_type") == "ATTACK_CONFIRM":
        rec2 = dict(recorded); rec2["action_type"] = "ATTACK"
        mv = resolve_recorded_action(manager, actor, rec2)
        if mv is not None:
            return mv
        cands = []
        for m2 in manager.get_legal_actions(actor):
            d = cpu_ai._describe_move(manager, m2) or {}
            if d.get("action_type") == "ATTACK" and d.get("card") == recorded.get("card"):
                cands.append((m2, d))
        if len(cands) == 1:
            return cands[0][0]
        opp = manager.p2 if actor.name == manager.p1.name else manager.p1
        lid = getattr(getattr(opp.leader, "master", None), "card_id", None)
        for m2, d in cands:
            if lid and lid in (d.get("targets") or ()):
                return m2
        if cands:
            return cands[0][0]
    return None


def _resolve_dialog_action(manager, actor, recorded: Dict[str, Any]):
    """効果対話の記録（card_id 基準の `selected`／`index`／`accepted`）から解決 payload を直接構築する。

    既定 payload（`default_interaction_payload`）をベースに記録値で上書きするので、
    エンジンのハンドラが読む全キー（selected_uuids/index/accepted/position/declared_value）が
    常に揃う。card_id→uuid は pending の selectable 候補への**列挙順・重複消費**写像
    （同名複数候補は先頭から順に割り当て＝録画側 `_describe_move` と同じ card_id 同一視）。
    候補に無い card_id が要求されたら None（＝ルール分岐として miss 検出に回す）。"""
    pending = manager.get_pending_request()
    if not pending or pending.get("player_id") != actor.name:
        return None
    payload = dict(manager.default_interaction_payload(pending))
    want = list(recorded.get("selected") or [])
    if want:
        by_uuid = {c.get("uuid"): c.get("card_id")
                   for c in (pending.get("candidates") or []) if c.get("uuid")}
        pool = [(u, by_uuid.get(u) or cpu_ai._card_label(manager, u))
                for u in (pending.get("selectable_uuids") or [])]
        uuids: List[str] = []
        unmatched = 0
        for cid in want:
            for i, (u, label) in enumerate(pool):
                if label == cid:
                    uuids.append(u)
                    pool.pop(i)
                    break
            else:
                unmatched += 1
        if unmatched:
            # 録画側 `_card_label` が uuid フォールバックした候補（ドン!!・非公開ゾーンの札）は
            # 再生側 uuid と一致しない。残候補が**同一 card_id のみ**（挙動等価＝どれを選んでも
            # 同じ）のときに限り先頭から充当する。異種が混じる場合は特定不能＝miss に回す
            # （黙って誤対応させない）。
            labels = {label for _, label in pool}
            if len(labels) != 1 or len(pool) < unmatched:
                return None
            for _ in range(unmatched):
                uuids.append(pool.pop(0)[0])
        payload["selected_uuids"] = uuids
    else:
        payload["selected_uuids"] = []
    for k in ("index", "position"):
        if recorded.get(k) is not None:
            payload[k] = recorded[k]
    if recorded.get("accepted") is not None:
        payload["accepted"] = recorded["accepted"]
    return {"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION", "payload": payload}


# --- 真盤面の再構築（両席とも記録どおりに再実行して途中停止） -----------------

class _ManagerCapture:
    """run_game の manager 参照を捕まえる観測専用 observer（状態は一切変更しない）。"""

    def __init__(self):
        self.manager = None

    def on_decision(self, ctx, move):
        self.manager = ctx.manager


def state_at_action(db, descriptor: Dict[str, Any], upto: int,
                    first_player: Optional[str] = None):
    """記録済み対局を**両席とも記録アクションどおり**に再実行し、action index `upto` の直前で
    停止した真の GameManager を返す（(manager, actor_pid) or (None, 診断dict)）。

    フレーム復元（`replay_reeval._board_from_frame`）はパワー修正・一時効果などの内部状態を
    持たない（実測: デバフで 1000 のキャラが素の 7000 で復元される）。本関数は seed から
    デッキ構築→start_game→記録アクションを順に適用するので、その時点の**全内部状態**が正確。
    CPU 再 decide を一切しない（両席 scripted）＝現ビルドとの方策ドリフトの影響を受けない。
    ルール解決が録画時ビルドと変わっていた場合のみ miss（分岐）として検出・報告する。"""
    seed = int(descriptor["seed"])
    leaders = descriptor.get("leaders", {})
    decks = descriptor["decks"]

    def _deck_builder(_db, _seed):
        l_p1, c_p1 = build_deck_from_ids(_db, leaders.get("p1"), decks["p1"], "p1")
        l_p2, c_p2 = build_deck_from_ids(_db, leaders.get("p2"), decks["p2"], "p2")
        return l_p1, c_p1, l_p2, c_p2

    acts = descriptor.get("actions", [])
    seats = {pid: _HumanReplaySeat([a for a in acts if a.get("player") == pid],
                                   resolver=resolve_api_action)
             for pid in ("p1", "p2")}
    cap = _ManagerCapture()
    # API 実対局はコイントスが乱数を消費する＝"random" を渡して seed から再現（結果 "p1" を
    # 直接渡すと以後のシャッフル/ドローの乱数列がズレる）。合成録画は first_player_mode 保存値。
    fp = first_player if first_player is not None else \
        descriptor.get("first_player_mode") or "random"
    try:
        run_game(seed, db, seats=seats, deck_builder=_deck_builder, observers=[cap],
                 legal_moves="skip", invariants="raise", stop_after_decisions=upto,
                 first_player=fp)
    except InvariantError as e:
        misses = seats["p1"].misses + seats["p2"].misses
        return None, {"reason": "invariant", "step": getattr(e, "step", None), "misses": misses}
    misses = seats["p1"].misses + seats["p2"].misses
    if misses:
        return None, {"reason": "miss", "misses": misses}
    if cap.manager is None:
        return None, {"reason": "no_decision"}
    return cap.manager, acts[upto].get("player") if upto < len(acts) else None


# --- 再生本体 ----------------------------------------------------------------

def replay_from_descriptor(db, descriptor: Dict[str, Any], cpu_difficulty: Optional[str] = None,
                           sims: int = 160, observers=(),
                           first_player: Optional[str] = None) -> Dict[str, Any]:
    """記述子から実対局を再構築・再生し、勝敗・思考トレース・診断を返す。

    `descriptor`: {seed, first_player, cpu_player_id, leaders:{p1,p2}, decks:{p1,p2}, actions:[...]}。
    人間＝`cpu_player_id` 以外の席（記録アクション注入）、CPU＝`cpu_player_id`（再 decide）。
    `cpu_difficulty` 省略時は descriptor の difficulty（learned/hard）。
    `first_player`（コイントス再現）: 省略時は `descriptor["first_player_mode"]`（合成録画が保存）。
    API 実対局は CPU 対局＝常に "random"（`RealGame.tsx` は `vsCpu ? 'random' : …`）なので、
    API 記述子を食うときは `first_player="random"` を渡す（seed から coin toss を再現＝乱数列一致）。
    """
    seed = int(descriptor["seed"])   # API は 2^53 超対策で文字列化して返す（旧録画の int も可）
    fp_mode = first_player if first_player is not None else descriptor.get("first_player_mode")
    # cpu_player_id は API 実対局では**プレイヤー名**（"P2" 等）・合成録画では席キー（"p2"）。
    # run_game の席は常に "p1"/"p2"。API 慣習で CPU=player2＝席 "p2"（名前が p1/p2 ならそれを採用）。
    # 人間手の抽出は**名前 != cpu_name** で行う（席キーでなく記録の player 名で照合）。
    cpu_name = descriptor["cpu_player_id"]
    cpu_pid = cpu_name if cpu_name in ("p1", "p2") else "p2"
    human_pid = "p1" if cpu_pid == "p2" else "p2"
    diff = cpu_difficulty or descriptor.get("difficulty", "hard")
    leaders = descriptor.get("leaders", {})
    decks = descriptor["decks"]

    def _deck_builder(_db, _seed):
        l_p1, c_p1 = build_deck_from_ids(_db, leaders.get("p1"), decks["p1"], "p1")
        l_p2, c_p2 = build_deck_from_ids(_db, leaders.get("p2"), decks["p2"], "p2")
        return l_p1, c_p1, l_p2, c_p2

    human_actions = [a for a in descriptor.get("actions", [])
                     if a.get("player") != cpu_name]
    human_seat = _HumanReplaySeat(human_actions)
    seats = {human_pid: human_seat, cpu_pid: _cpu_seat(diff, sims)}

    # 人間手の逆写像失敗（場複製の分岐など・R0 §5 の残差）は run_game が NO_LEGAL_MOVE を上げる。
    # これは「再生できなかった局」＝**分岐検出**なので、クラッシュさせず結果に記録する
    # （真の再生成功＝reproduced True・misses 空。分岐＝reproduced False・misses 非空）。
    try:
        result: GameResult = run_game(seed, db, seats=seats, deck_builder=_deck_builder,
                                      observers=list(observers), legal_moves="skip", invariants="raise",
                                      first_player=fp_mode)
        return {
            "seed": seed, "reproduced": True, "winner": result.winner,
            "steps": result.steps, "turns": result.turns,
            "human_pid": human_pid, "cpu_pid": cpu_pid, "difficulty": diff,
            "misses": human_seat.misses,
        }
    except InvariantError as e:
        if not human_seat.misses:
            raise   # 人間手 miss でない＝真のインバリアント違反（再生機構の別バグ）はそのまま
        return {
            "seed": seed, "reproduced": False, "winner": None, "steps": None, "turns": None,
            "human_pid": human_pid, "cpu_pid": cpu_pid, "difficulty": diff,
            "misses": human_seat.misses, "stopped_at": e.step,
        }


# --- 記録側ヘルパ（round-trip テスト用の合成記述子生成） --------------------

class _RecordObserver:
    """1 対局を記述子形式（actions を card_id 基準で）へ記録する（テスト用の合成録画）。"""

    def __init__(self):
        self.actions: List[Dict[str, Any]] = []

    def on_decision(self, ctx, move):
        d = cpu_ai._describe_move(ctx.manager, move) or {}
        self.actions.append({"player": ctx.actor.name, "turn": ctx.turn, **d})


def _human_record_seat(rng):
    """合成「人間」席: **private rng** で多様な手を選ぶ（global random を消費しない＝実際の人間と同じ）。

    人間の decide が global random を消費すると、再生（注入＝消費なし）で CPU 側の乱数がずれる。
    private rng なら記録・再生ともグローバル乱数列が一致し、注入した人間手が録画と同一経路を辿る。
    カード/攻撃を能動的に選び、曖昧ケース（PLAY/ATTACK の同名複製）を踏ませる。
    """
    def seat(ctx):
        moves = ctx.manager.get_legal_actions(ctx.actor)
        if not moves:
            return None
        plays = [m for m in moves if m.get("action_type") == "PLAY"]
        attacks = [m for m in moves if m.get("action_type") == "ATTACK"]
        if plays and rng.random() < 0.55:
            return rng.choice(plays)
        if attacks and rng.random() < 0.6:
            return rng.choice(attacks)
        end = [m for m in moves if m.get("action_type") == "TURN_END"]
        if end and rng.random() < 0.3:
            return end[0]
        return rng.choice(moves)
    return seat


def record_descriptor(db, seed: int, deck_ids_builder, cpu_pid: str = "p2",
                      difficulty: str = "hard", sims: int = 160,
                      first_player: Optional[str] = None) -> Dict[str, Any]:
    """1 局を録画して記述子（seed/leaders/decks/actions）を作る（round-trip テスト用の合成録画）。

    人間席（`cpu_pid` 以外）＝private rng の合成人間（global random 非消費）、CPU席＝`difficulty`
    （learned=Gen2・`sims` 指定可）。`first_player`（"random"/"p1"/"p2"/None）でコイントス再現を試験できる
    （`first_player_mode` として記述子へ保存＝replay が同じモードで coin toss を再現する）。
    `deck_ids_builder(db, seed) -> (l1,c1,l2,c2)` で対局デッキを与える。
    """
    import random as _random
    built = deck_ids_builder(db, seed)
    l1, c1, l2, c2 = built
    human_pid = "p1" if cpu_pid == "p2" else "p2"

    def _fixed_builder(_db, _seed):
        return built

    rec = _RecordObserver()
    seats = {
        human_pid: _human_record_seat(_random.Random(seed * 7 + 1)),
        cpu_pid: _cpu_seat(difficulty, sims),
    }
    res: GameResult = run_game(seed, db, seats=seats,
                               deck_builder=_fixed_builder, observers=[rec],
                               legal_moves="skip", invariants="raise", first_player=first_player)
    return {
        "seed": seed, "first_player": None, "first_player_mode": first_player,
        "cpu_player_id": cpu_pid, "difficulty": difficulty,
        "leaders": {"p1": l1.master.card_id if l1 else None, "p2": l2.master.card_id if l2 else None},
        "decks": {"p1": [ci.master.card_id for ci in c1], "p2": [ci.master.card_id for ci in c2]},
        "actions": rec.actions,
        "_winner": res.winner, "_steps": res.steps, "_turns": res.turns,
    }
