"""リプレイ盤面フレーム → 記録 v5 `hidden`（分岐点シナリオの復元・`docs/rust_engine_plan.md` §20.1）。

入力は `/replay/frames`（`opcg_sim/api/services/replay.py`）が返す JSON 全体（`replay`／
`decisions`／`frames` を持つ dict＝リプレイビューアが読み込む形）。フレームは**両席の手札・場・
ライフ・トラッシュが card_id 込みで見える**完全情報の記録（`_frame_side`／`_compact_card`）なので、
そこから `opcg_engine.Game.from_hidden` が食える `hidden` を組み直せる。

**復元できるもの**（フレームにある動的状態）: 手札・場・ライフ・トラッシュ・ステージ・リーダー・
各カードの `attached_don`・レスト・ドン!!（active/rested/deck の枚数）・ターン・手番。
メインデッキ（山札）はフレームに中身が無い（`deck_count` だけ）ので、`replay.decks`（40〜50 枚の
card_id 列）から**見えているカードを引いた残り**を `seed` で shuffle して埋める。

**復元できないもの**（フレームに無い・記録 v5 の契約どおり）: 継続効果（`power_buff` 等のバフ・
`current_keywords`/`timed_*`）・中断（対話）スタック・誘発待ち行列・`ability_used_this_turn`。
シナリオの開始は必ずターンの始め（`turn_start_index` が探す `MAIN_ACTION` 待ちの最初のフレーム）
なので、継続効果は通常そこで切れており実害は小さい（§20.1）。
"""
import random
from collections import Counter
from typing import Any, Dict, List, Optional


class UnrestorableFrameError(ValueError):
    """指定フレームから `hidden` を組み直せない（対話や戦闘の途中・デッキの構成が合わない）。"""


def turn_start_index(payload: Dict[str, Any], turn: int) -> int:
    """`turn` の開始フレーム（`turn == turn` かつ `pending.action == "MAIN_ACTION"` の最初）の
    `payload["frames"]` 内 index。

    そのターンの最初の意思決定点＝ドロー・ドンフェイズが終わり、継続効果もまだ何も乗っていない
    地点（`frame_to_hidden` が復元の起点にする）。見つからなければ `ValueError`。
    """
    frames = payload.get("frames") or []
    for i, fr in enumerate(frames):
        if fr.get("turn") == turn and (fr.get("pending") or {}).get("action") == "MAIN_ACTION":
            return i
    raise ValueError(
        f"ターン{turn}の開始フレーム（MAIN_ACTION 待ち）が見つからない。別のターンを指定してください。")


def _card_record(card: Dict[str, Any], owner: str) -> Dict[str, Any]:
    """フレームのカード dict（`_CARD_KEEP` の形）→ 記録 v5 の `card_record`。

    フレームに無い欄（バフ・継続効果由来のフラグ）は復元できないので既定値で埋める。
    `keywords`（`to_dict` の `current_keywords | timed_keywords` 合成後）はまとめて
    `current_keywords` に戻す＝`timed_keywords` との内訳は失うが `keywords_union` は保たれる。
    """
    return {
        "card_id": card["card_id"],
        "uuid": card["uuid"],
        "owner_id": owner,
        "is_rest": bool(card.get("is_rest", False)),
        "is_newly_played": False,
        "attached_don": int(card.get("attached_don", 0) or 0),
        "is_face_up": bool(card.get("is_face_up", True)),
        "power_buff": 0,
        "cost_buff": 0,
        "passive_power": 0,
        "passive_power_override": None,
        "passive_counter": 0,
        "base_power_override": None,
        "base_cost_override": None,
        "negated": False,
        "ability_disabled": bool(card.get("ability_disabled", False)),
        "timed_power": 0,
        "timed_cost": 0,
        "current_keywords": sorted(set(card.get("keywords") or [])),
        "flags": ["FREEZE"] if card.get("is_frozen") else [],
        "timed_flags": [],
        "timed_keywords": [],
        "ability_used_this_turn": {},
    }


def _deck_card_record(card_id: str, owner: str, uid: str) -> Dict[str, Any]:
    """山札の未見カード（合成 uuid・裏向き）の `card_record`。"""
    return {
        "card_id": card_id,
        "uuid": uid,
        "owner_id": owner,
        "is_rest": False,
        "is_newly_played": False,
        "attached_don": 0,
        "is_face_up": False,
        "power_buff": 0,
        "cost_buff": 0,
        "passive_power": 0,
        "passive_power_override": None,
        "passive_counter": 0,
        "base_power_override": None,
        "base_cost_override": None,
        "negated": False,
        "ability_disabled": False,
        "timed_power": 0,
        "timed_cost": 0,
        "current_keywords": [],
        "flags": [],
        "timed_flags": [],
        "timed_keywords": [],
        "ability_used_this_turn": {},
    }


def _don_record(owner: str, uid: str, is_rest: bool = False,
                attached_to: Optional[str] = None) -> Dict[str, Any]:
    return {"uuid": uid, "owner_id": owner, "is_rest": is_rest,
            "attached_to": attached_to, "is_frozen": False}


def _synth_uuid(rng: random.Random, tag: str) -> str:
    """合成カード／ドン!!の内部 uuid（一意であればよく、実カードの uuid と衝突しない接頭辞を持つ）。"""
    return f"hb:{tag}:{rng.getrandbits(64):016x}"


def _player_hidden(seat: str, side: Dict[str, Any], deck_pool: List[str],
                   rng: random.Random) -> Dict[str, Any]:
    """1 席ぶんの `hidden.players.<seat>`。"""
    leader_raw = side.get("leader")
    stage_raw = side.get("stage")
    hand_raw = side.get("hand") or []
    field_raw = side.get("field") or []
    life_raw = side.get("life") or []
    trash_raw = side.get("trash") or []

    leader = _card_record(leader_raw, seat) if leader_raw else None
    stage = _card_record(stage_raw, seat) if stage_raw else None
    hand = [_card_record(c, seat) for c in hand_raw]
    field = [_card_record(c, seat) for c in field_raw]
    life = [_card_record(c, seat) for c in life_raw]
    trash = [_card_record(c, seat) for c in trash_raw]

    # メインデッキ（山札）: `replay.decks[seat]`（レコード全体・オモテのカードは持たない構築時
    # デッキリスト）から、見えているカード（ステージ・手札・場・ライフ・トラッシュ。**リーダーは
    # 別枠なので含めない**）を 1 枚ずつ引いた残りが山札の中身。
    visible = Counter()
    for c in ([stage_raw] if stage_raw else []) + hand_raw + field_raw + life_raw + trash_raw:
        visible[c["card_id"]] += 1
    pool = list(deck_pool)
    for card_id, n in visible.items():
        for _ in range(n):
            try:
                pool.remove(card_id)
            except ValueError:
                raise UnrestorableFrameError(
                    f"{seat}: 見えているカード '{card_id}' が構築デッキリストに無い"
                    "（山札の外から来たカード＝復元不能）。別のターンを指定してください。") from None
    deck_count = int(side.get("deck_count", 0) or 0)
    if len(pool) != deck_count:
        raise UnrestorableFrameError(
            f"{seat}: 山札の残り枚数が合わない（frame={deck_count}・"
            f"構築デッキから見えている分を引いた残り={len(pool)}）。別のターンを指定してください。")
    rng.shuffle(pool)
    deck = [_deck_card_record(cid, seat, _synth_uuid(rng, f"{seat}:deck:{i}"))
            for i, cid in enumerate(pool)]

    # ドン!!: active／rested の枚数はフレームのまま。付与中はカードごとの `attached_don` の合計。
    # 残りの合成カード（deck 分）はフレームの `don_deck`（残り枚数）をそのまま使う——リーダーの
    # ルール効果でドン!!デッキの総数が leader 依存で変わる（例: OP15-058「ドン‼デッキは6枚」）ので
    # 固定の合計値（10 枚）を仮定して逆算できない。
    don_attached: List[Dict[str, Any]] = []
    for card, rec in list(zip([leader_raw] if leader_raw else [], [leader] if leader else [])) + \
            list(zip(field_raw, field)):
        n = int(card.get("attached_don", 0) or 0)
        for i in range(n):
            don_attached.append(_don_record(
                seat, _synth_uuid(rng, f"{seat}:donatt:{rec['uuid']}:{i}"), attached_to=rec["uuid"]))
    don_active = [_don_record(seat, _synth_uuid(rng, f"{seat}:donact:{i}"))
                  for i in range(int(side.get("don_active", 0) or 0))]
    don_rested = [_don_record(seat, _synth_uuid(rng, f"{seat}:donrest:{i}"), is_rest=True)
                  for i in range(int(side.get("don_rested", 0) or 0))]
    don_deck = [_don_record(seat, _synth_uuid(rng, f"{seat}:dondeck:{i}"))
                for i in range(int(side.get("don_deck", 0) or 0))]

    return {
        "name": seat,
        "leader": leader,
        "stage": stage,
        "deck": deck,
        "hand": hand,
        "life": life,
        "field": field,
        "trash": trash,
        "temp_zone": [],
        "don": {"deck": don_deck, "active": don_active, "rested": don_rested, "attached": don_attached},
        "negate_onplay_until": 0,
        "restrictions": {},
    }


def frame_to_hidden(payload: Dict[str, Any], action_index: int, seed: int) -> Dict[str, Any]:
    """`payload["frames"][action_index]` から記録 v5 の `hidden` を組む。

    `action_index` は `payload["frames"]` 内の index（`turn_start_index` の戻り値）。
    フレームが対話や戦闘の途中（`pending.action != "MAIN_ACTION"` または `battle` あり）なら
    `ValueError`（別のターンを選ぶよう促す）。
    """
    frames = payload.get("frames") or []
    if not (0 <= action_index < len(frames)):
        raise ValueError(f"action_index={action_index} がフレーム範囲外（0..{len(frames) - 1}）")
    frame = frames[action_index]
    pending = frame.get("pending") or {}
    if pending.get("action") != "MAIN_ACTION":
        raise UnrestorableFrameError(
            "このフレームは効果対話の途中（pending="
            f"{pending.get('action')!r}）のため復元できません。別のターンを指定してください。")
    if frame.get("battle"):
        raise UnrestorableFrameError(
            "このフレームは戦闘の途中のため復元できません。別のターンを指定してください。")
    if frame.get("winner"):
        raise UnrestorableFrameError(
            "このフレームは既に決着している対局のため復元できません。別のターンを指定してください。")

    decks = ((payload.get("replay") or {}).get("decks")) or {}
    players = frame.get("players") or {}
    rng = random.Random(seed)
    p1 = _player_hidden("p1", players.get("p1") or {}, list(decks.get("p1") or []), rng)
    p2 = _player_hidden("p2", players.get("p2") or {}, list(decks.get("p2") or []), rng)

    active = frame.get("active")
    if active not in ("p1", "p2"):
        raise UnrestorableFrameError(f"フレームの手番（active={active!r}）が不明です。")

    manager = {
        "turn_count": int(frame.get("turn") or 0),
        "phase": frame.get("phase") or "MAIN",
        "turn_player": active,
        "winner": None,
        "active_battle": None,
        "turn_events": {},
        "mulligan_done": ["p1", "p2"],
        "setup_phase_pending": False,
        "turn_start_pending": False,
        "interaction_depth": 0,
        "pending_triggers": 0,
        "pending_end_of_turn": 0,
    }
    return {"manager": manager, "players": {"p1": p1, "p2": p2}}
