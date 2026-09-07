"""監査オラクル: 全カードの能力を汎用盤面で発動し、Rust 側の再生と照合する
（`docs/rust_engine_plan.md` §11.1 の「監査オラクル」・WP `rs-p3-resolver`）。

**問い**: Rust の効果解決は、同じカード・同じ盤面・同じ既定応答で Python と同じ盤面になるか。

手順は `tests/harness/full_card_audit.py` と**同じ**:
  1. `effect_coverage._build_test_state` の汎用盤面を組む（フィラー `FILLER` を詰めた両者の場・
     手札・デッキ・ライフ・トラッシュ・ドン!!）。
  2. 能力を 1 つ発動する（`ON_PLAY` は `play_card_action`、それ以外は `resolve_ability`）。
  3. `_smart_drain` と同じ既定応答で対話を消化する。
そのうえで **各段の盤面 dict（`pending_request` 込み・`request_id` 除外）を記録**し、
`opcg_engine.replay_audit(record)` の返す盤面と 1 段ずつ突き合わせる。

記録形式は v5 の `kind: "audit"`（§11.2／§11.8 #1）。汎用盤面は効果 JSON に無いカード
（`FILLER`）を使うので、その定義を `extra_masters` に同梱する（Rust 側は表へ足してから読む）。
`fire` 直後・各 `steps[i]` 直後に `shuffled`（その段で混ぜたデッキの持ち主）を持たせ、
Rust `replay_audit` が同じ位置で再同期する。

実行例:
    # Python 側の自己検査（記録だけ・Rust 不要）。全カードで例外 0 が受け入れ条件。
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py --self-check

    # Rust と照合（土台ハンドラだけを使うカードに絞る）
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py \\
      --action-types DRAW,DISCARD,KO,REST,ACTIVE,BUFF

    # 特定カードだけ
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_audit_replay.py --card-ids OP01-001,OP01-002

出力は `RS_AUDIT {...}` の 1 行:
    {"cards":.., "abilities":.., "match":.., "mismatch":.., "unimplemented":.., "first":{...}}
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import traceback

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import harness.effect_coverage as cov  # noqa: E402
from harness.engine_helpers import make_master  # noqa: E402
from opcg_sim.src.core.journal import JournaledList  # noqa: E402
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence  # noqa: E402
from opcg_sim.src.models.models import DonInstance  # noqa: E402
from opcg_sim.src.utils.loader import CardLoader  # noqa: E402
from opcg_sim.tools.export_effects_json import Stats, _encode  # noqa: E402

from rs_diff_replay import (  # noqa: E402
    DEFAULT_EFFECTS_PATH, RECORD_VERSION, ShuffleWatcher, board_dict, canon, first_diff,
    hidden_dict, load_masters, _norm, _strip_request_id,
)

try:                        # Rust 拡張は未導入でも動く（その場合は全件 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DATA = _os.path.join(_REPO_ROOT, "opcg_sim", "data")

# `_smart_drain` の上限（`effect_coverage._smart_drain` の既定と揃える）。
DRAIN_LIMIT = 30


# --- カード定義の書き出し（`export_effects_json.export` と同じ形）-------------------

def master_dict(master) -> dict:
    """`CardMaster` を効果 JSON と同じ形の dict にする（`extra_masters` 用）。"""
    stats = Stats()
    return {
        "card_id": master.card_id,
        "name": master.name,
        "type": master.type.name,
        "colors": [c.name for c in master.colors],
        "cost": master.cost,
        "power": master.power,
        "counter": master.counter,
        "attribute": master.attribute.name,
        "traits": list(master.traits),
        "life": master.life,
        "block_icon": master.block_icon,
        "keywords": sorted(master.keywords),
        "name_aliases": list(master.name_aliases),
        "effect_text": master.effect_text,
        "trigger_text": master.trigger_text,
        "abilities": [_encode(ab, stats) for ab in master.abilities],
    }


def _audit_leader_master():
    """汎用盤面の**共有リーダー定義**（`engine_helpers.make_player` の `L-001` の置き換え）。

    `make_player(name)` は `card_id="L-001"` のリーダーを `name=f"{name}リーダー"` で作るので、
    **同じ card_id が両者で違う名前**になる（"P1リーダー" と "P2リーダー"）。カード定義表は
    card_id で 1 枚を引くため、そのままでは片側の `name` が合わない。名前はここでは識別子
    としてしか使われない（`LEADER_NAME` 条件は実カード名を照合するので合成名は元から
    一致しない）ので、両者を 1 つの定義に揃える。
    """
    from opcg_sim.src.models.enums import CardType as _CardType
    return make_master(card_id="L-001", name="監査リーダー", type=_CardType.LEADER, life=5)


def extra_masters() -> list:
    """汎用盤面が使う「効果 JSON に無いカード定義」（`extra_masters`）。

    `_build_test_state` が使う合成カードは 2 種:
      - `FILLER`（`effect_coverage._get_filler_master`）＝両者の手札・デッキ・ライフ・
        トラッシュ・場に詰める中身
      - `L-001`（`engine_helpers.make_player`）＝両者のリーダー（[`_audit_leader_master`]）
    """
    return [
        master_dict(make_master(card_id="FILLER", name="フィラー", power=1000)),
        master_dict(_audit_leader_master()),
    ]


# --- ActionType の収集（`--action-types` の絞り込み用）-----------------------------

def _walk_actions(node):
    """効果木の全 `GameAction`（`sub_effect` も辿る）。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
        yield from _walk_actions(getattr(node, "sub_effect", None))
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk_actions(a)
    elif isinstance(node, Branch):
        yield from _walk_actions(node.if_true)
        yield from _walk_actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk_actions(o)


def card_action_types(master) -> set:
    """このカードの全能力が使う `ActionType` の名前集合（コスト句も含む）。"""
    names = set()
    for ab in (master.abilities or ()):
        for act in list(_walk_actions(ab.effect)) + list(_walk_actions(ab.cost)):
            t = getattr(act, "type", None)
            names.add(getattr(t, "name", None) or str(t))
    return names


# --- 記録 ---------------------------------------------------------------------

def _drain_and_record(gm, steps: list, watcher: "ShuffleWatcher | None" = None) -> None:
    """`effect_coverage._smart_drain` と**同じ既定応答**で対話を消化し、各段を記録する。

    分岐・払い出す payload は `_smart_drain` の逐語写し（挙動を変えるとオラクルの意味が
    変わるため）。違いは「各応答とその後の盤面を `steps` へ積む」ことだけ（v5: `shuffled` も）。
    """
    count = 0
    while gm.active_interaction and count < DRAIN_LIMIT:
        ia = gm.active_interaction
        player = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        action_type = ia.get("action_type", "")

        if action_type in ("SELECT_TARGET", "SELECT_RESOURCE"):
            candidates = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            cons = ia.get("constraints") or {}
            min_req = cons.get("min", 0)
            max_req = cons.get("max", 1)
            if max_req is not None and max_req < 0:
                n_select = len(candidates)
            else:
                n_select = max(min_req, 1) if candidates else 0
                if max_req:
                    n_select = min(n_select, max_req)
            payload = {"selected_uuids": candidates[:n_select], "index": 0}
        elif action_type == "CHOICE":
            n_opt = len(ia.get("options") or []) or 2
            payload = {"selected_uuids": [], "index": min(0, n_opt - 1)}
        else:  # CONFIRM_OPTIONAL / CONFIRM_TRIGGER / ARRANGE_DECK / DECLARE_COST 等
            payload = {"selected_uuids": [], "index": 0}

        try:
            gm.resolve_interaction(player, payload)
        except Exception:
            # `_smart_drain` は例外で打ち切る（記録もそこで止める＝Rust と同じ段数になる）。
            break
        owners = watcher.take() if watcher is not None else []
        steps.append({
            "payload": payload,
            "state": board_dict(gm),
            "shuffled": list(dict.fromkeys(owners)),
            "hidden": hidden_dict(gm),
        })
        count += 1


def _normalize_leaders(p1, p2) -> None:
    """両者の合成リーダー（`L-001`）を**同じ定義**に揃える（[`_audit_leader_master`] 参照）。

    検査対象カード自身がリーダーのときは `p1.leader` が実カードなので触らない。
    """
    shared = _audit_leader_master()
    for p in (p1, p2):
        leader = getattr(p, "leader", None)
        if leader is not None and leader.master.card_id == "L-001":
            leader.master = shared
            leader._refresh_keywords()


def _normalize_names(p1, p2) -> None:
    """汎用盤面のプレイヤー名を実対局と同じ `p1`／`p2` に揃える（カードの `owner_id` も）。

    `effect_coverage._build_test_state` は `make_player("P1")` を使うため、名前が
    **大文字**（"P1"/"P2"）になる。記録 v4 の `hidden.players` は `p1`/`p2` をキーにする契約
    （`GameManager` の実対局と同じ）なので、そのままでは Rust 側が読めない。
    名前は**識別子としてしか使われない**（`gm.p1.name == card.owner_id` の比較や dict のキー）
    ので、両者を揃えて小文字化するのは純粋な付け替え＝盤面の意味は変わらない。
    """
    for p, new_name in ((p1, "p1"), (p2, "p2")):
        p.name = new_name
    for p in (p1, p2):
        cards = list(p.hand) + list(p.field) + list(p.trash) + list(p.deck) \
            + list(p.life) + list(p.temp_zone)
        if p.leader:
            cards.append(p.leader)
        if getattr(p, "stage", None):
            cards.append(p.stage)
        for c in cards:
            c.owner_id = c.owner_id.lower()
        for zone in ("don_deck", "don_active", "don_rested", "don_attached_cards"):
            for d in getattr(p, zone, None) or ():
                d.owner_id = d.owner_id.lower()


def _normalize_dons(*players) -> None:
    """汎用盤面のドン!!ゾーンを**本物の `DonInstance`** に直す（枚数は変えない）。

    `effect_coverage._build_test_state` は `p.don_deck = _instances(5, "P1")` のように
    **`CardInstance` のフィラー**をドン!!ゾーンへ詰める。構造監査（枚数と例外だけを見る）では
    問題にならないが、盤面 dict の照合では致命的:
      - `don_active`／`don_rested` は `to_dict()` の結果がそのまま出る＝カード形の dict になる
      - `attached_to`／`is_frozen` を持たないので `hidden` の書き出しが `AttributeError` になる
    枚数・所有者は同じまま実体だけ差し替えるので、**盤面 dict の他の欄は 1 つも変わらない**
    （`don_deck_count` は枚数・`don_active`/`don_rested` は同数のドン!!）。
    """
    for p in players:
        for zone in ("don_deck", "don_active", "don_rested", "don_attached_cards"):
            items = getattr(p, zone, None) or []
            setattr(p, zone, JournaledList(DonInstance(owner_id=p.name) for _ in items))


def record_one(master, ability, ability_index: int, extra: list) -> dict:
    """1 能力ぶんの監査記録（記録 v5 の `kind: "audit"`）を作る。

    v5: `fire` 直後と各 `steps[i]` の応答直後に `shuffled`（その段で `random.shuffle` を
    呼んだデッキの持ち主）を持たせる（§11.8 #1）。`_build_test_state` 内のシャッフルは
    `setup.hidden` に既に畳まれているので、`watcher.manager` を立てた直後に捨てる
    （`rs_diff_replay.Recorder.on_start` と同じ規約）。
    """
    with ShuffleWatcher() as watcher:
        trig = ability.trigger.name if hasattr(ability.trigger, "name") else str(ability.trigger)
        if trig == "ON_PLAY":
            gm, p1, p2, src = cov._build_test_state(master, source_in_hand=True)
            _normalize_leaders(p1, p2)
            _normalize_names(p1, p2)
            _normalize_dons(p1, p2)
            watcher.manager = gm
            watcher.take()  # setup 中のシャッフルは setup.hidden に畳まれている
            setup = {"hidden": hidden_dict(gm), "state": board_dict(gm)}
            fire = {"kind": "play", "player": "p1", "source_uuid": src.uuid,
                    "ability_index": ability_index}
            gm.play_card_action(p1, src)
        else:
            gm, p1, p2, src = cov._build_test_state(master)
            _normalize_leaders(p1, p2)
            _normalize_names(p1, p2)
            _normalize_dons(p1, p2)
            watcher.manager = gm
            watcher.take()
            setup = {"hidden": hidden_dict(gm), "state": board_dict(gm)}
            fire = {"kind": "ability", "player": "p1", "source_uuid": src.uuid,
                    "ability_index": ability_index}
            gm.resolve_ability(p1, ability, src)
        fire["state"] = board_dict(gm)
        fire["shuffled"] = list(dict.fromkeys(watcher.take()))
        fire["hidden"] = hidden_dict(gm)

        steps: list = []
        _drain_and_record(gm, steps, watcher)
    return {
        "version": RECORD_VERSION,
        "kind": "audit",
        "card_id": master.card_id,
        "trigger": trig,
        "ability_index": ability_index,
        "extra_masters": extra,
        "setup": setup,
        "fire": fire,
        "steps": steps,
        "final": {"interactive": bool(gm.active_interaction)},
    }


# --- 照合 ---------------------------------------------------------------------

def compare(record: dict, replayed: dict) -> dict:
    """Rust の `replay_audit` 出力と記録を段ごとに突き合わせる。"""
    states = replayed.get("states")
    if not isinstance(states, list):
        return {"status": "bad_output", "detail": "replay_audit result has no 'states' list"}
    expected = [_strip_request_id(record["fire"]["state"])]
    expected += [_strip_request_id(s["state"]) for s in record["steps"]]
    got = [_strip_request_id(g) if isinstance(g, dict) else g for g in states]
    if len(got) != len(expected):
        return {"status": "mismatch", "step": min(len(got), len(expected)),
                "path": f"states[len {len(expected)}!={len(got)}]"}
    for i, (exp, act) in enumerate(zip(expected, got)):
        if _norm(exp) != _norm(act):
            return {"status": "mismatch", "step": i, "path": first_diff(canon(exp), canon(act))}
    return {"status": "match"}


def run_one(record: dict, effects_path: str) -> dict:
    if opcg_engine is None or not hasattr(opcg_engine, "replay_audit"):
        return {"status": "unimplemented", "detail": "opcg_engine.replay_audit is not available"}
    try:
        out = opcg_engine.replay_audit(json.dumps(record, ensure_ascii=False, default=str),
                                       effects_path)
    except NotImplementedError as e:
        return {"status": "unimplemented", "detail": str(e)}
    except ValueError as e:      # 契約違反（記録側のバグ）＝黙って通さない
        return {"status": "bad_payload", "detail": str(e)}
    try:
        replayed = json.loads(out)
    except (TypeError, ValueError) as e:
        return {"status": "bad_output", "detail": f"replay_audit returned non-JSON: {e}"}
    return compare(record, replayed)


# --- 入口 ---------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="全カード監査を Rust で再生して照合する（P3）")
    ap.add_argument("--card-ids", default=None,
                    help="カンマ区切りの card_id に絞る（既定: 全カード）")
    ap.add_argument("--action-types", default=None,
                    help="このセットに含まれる ActionType だけを使うカードに絞る"
                         "（例: DRAW,DISCARD,KO,REST,ACTIVE,BUFF）")
    ap.add_argument("--self-check", action="store_true",
                    help="Python 側の記録だけを行い、例外が出ないかを検査する（Rust 不要）")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS_PATH,
                    help="効果構造 JSON（Rust の CardMaster 表）")
    ap.add_argument("--dump", default=None, help="最初の記録をこのパスへ書き出す（開発用）")
    ap.add_argument("--verbose", action="store_true", help="能力ごとの判定を出す")
    args = ap.parse_args(argv)

    if not args.self_check:
        load_masters(args.effects)

    db = CardLoader(_os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    if args.card_ids:
        wanted = {c.strip() for c in args.card_ids.split(",") if c.strip()}
        card_ids = [c for c in card_ids if c in wanted]
    allowed = None
    if args.action_types:
        allowed = {t.strip() for t in args.action_types.split(",") if t.strip()}

    extra = extra_masters()
    totals = {"cards": 0, "abilities": 0, "match": 0, "mismatch": 0, "unimplemented": 0,
              "record_error": 0, "bad_payload": 0, "bad_output": 0}
    first = None
    dumped = False

    for i, cid in enumerate(card_ids, 1):
        if i % 300 == 0:
            sys.stderr.write(f"\r進行中: {i}/{len(card_ids)}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        if allowed is not None and not card_action_types(master).issubset(allowed):
            continue
        totals["cards"] += 1
        for index, ab in enumerate(master.abilities):
            totals["abilities"] += 1
            try:
                record = record_one(master, ab, index, extra)
            except Exception as e:  # noqa: BLE001 - 記録側の事故も集計に載せる
                totals["record_error"] += 1
                if first is None:
                    first = {"status": "record_error", "card_id": cid, "ability_index": index,
                             "detail": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()[-400:]}
                continue
            if args.dump and not dumped:
                with open(args.dump, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, default=str)
                dumped = True
            if args.self_check:
                totals["match"] += 1   # 記録できた＝自己検査は通過
                continue
            verdict = run_one(record, args.effects)
            status = verdict["status"]
            totals[status] = totals.get(status, 0) + 1
            if status != "match" and first is None:
                first = dict(verdict, card_id=cid, trigger=record["trigger"],
                             ability_index=index)
            if args.verbose and status != "match":
                print(f"[{cid}#{index} {record['trigger']}] {status} "
                      f"{str(verdict.get('detail') or verdict.get('path'))[:160]}")
    sys.stderr.write(f"\r完了: {len(card_ids)} カード処理済み\n")

    summary = {
        "cards": totals["cards"],
        "abilities": totals["abilities"],
        "match": totals["match"],
        "mismatch": totals["mismatch"],
        "unimplemented": totals["unimplemented"],
        "record_error": totals["record_error"],
        "bad_payload": totals["bad_payload"],
        "bad_output": totals["bad_output"],
        "mode": "self-check" if args.self_check else "replay",
        "action_types": sorted(allowed) if allowed else None,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "record_version": RECORD_VERSION,
        "first": first,
    }
    print("RS_AUDIT " + json.dumps(summary, ensure_ascii=False))
    return 1 if (totals["mismatch"] or totals["record_error"] or totals["bad_payload"]
                 or totals["bad_output"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
