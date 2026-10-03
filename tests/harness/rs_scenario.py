"""実カード ID で盤面を組み、Rust エンジンの `RsGame`（API と同じ口）を 1 手ずつ進めるための土台。

作り方: 必要なカードを山札に仕込んだ対局を `create_from_ids` で作りマリガンを流し、`hidden`（記録 v5）を
書き換えて（山札→手札/場/ステージ/ライフ/トラッシュへ移す・ドン!!の枚数・ターン）から
`Game.from_hidden` で組み直す。以後は API と同じ `apply_game_action`／`apply_battle_action`／
`get_pending_request`／`get_legal_actions` だけで進める＝実対局と同じ解決経路を通る。
"""
import copy
import json
import random

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame, load_engine

FILLER = "EB01-005"      # 能力なしのバニラ（コスト 1 想定・色は問わない）
LEADER_DEFAULT = "EB01-001"


class Scenario:
    def __init__(self, p1_leader=LEADER_DEFAULT, p2_leader=LEADER_DEFAULT, turn=5, seed=1):
        self.leaders = {"p1": p1_leader, "p2": p2_leader}
        self.turn = turn
        self.seed = seed
        # seat -> list of (card_id, zone, opts)
        self.plan = {"p1": [], "p2": []}
        self.don = {"p1": 5, "p2": 5}
        self.don_rested = {"p1": 0, "p2": 0}
        self.turn_player = "p1"
        self.life_n = {"p1": 5, "p2": 5}
        self.leader_don = {"p1": 0, "p2": 0}       # リーダーへ付与済みのドン!!の枚数

    def put(self, seat, card_id, zone, **opts):
        """zone: hand/field/stage/life/trash/deck_top。opts: rest, don(付与数), newly。"""
        self.plan[seat].append((card_id, zone, opts))
        return self

    def build(self):
        random.seed(self.seed)
        decks = {s: [c for c, _, _ in self.plan[s]] + [FILLER] * 50 for s in ("p1", "p2")}
        g = RsGame.create_from_ids("P1", "P2", self.leaders["p1"], decks["p1"],
                                   self.leaders["p2"], decks["p2"], first_player="p1")
        for _ in range(2):
            p = g.get_pending_request(False)
            g.apply_game_action(p["player_id"], "KEEP_HAND", {})
        h = copy.deepcopy(g.hidden())
        for seat in ("p1", "p2"):
            pl = h["players"][seat]
            # 手札・ライフ（開局で配られた分）は山札へ戻して空にする。
            pl["deck"] = pl["hand"] + pl["life"] + pl["deck"]
            pl["hand"], pl["life"] = [], []
            # ドン!!は全部ドン!!デッキへ戻してから配り直す。
            dn = pl["don"]
            for d in dn["active"] + dn["rested"] + dn["attached"]:
                d["is_rest"], d["attached_to"] = False, None
            dn["deck"] = dn["deck"] + dn["active"] + dn["rested"] + dn["attached"]
            dn["active"], dn["rested"], dn["attached"] = [], [], []
            for card_id, zone, opts in self.plan[seat]:
                idx = next(i for i, c in enumerate(pl["deck"]) if c["card_id"] == card_id)
                c = pl["deck"].pop(idx)
                if opts.get("rest"):
                    c["is_rest"] = True
                c["is_newly_played"] = bool(opts.get("newly"))
                if zone == "stage":
                    pl["stage"] = c
                elif zone == "deck_top":
                    pl["deck"].insert(0, c)
                else:
                    pl[{"hand": "hand", "field": "field", "life": "life", "trash": "trash"}[zone]].append(c)
                n_don = opts.get("don", 0)
                for _ in range(n_don):
                    d = pl["don"]["deck"].pop()
                    d["attached_to"] = c["uuid"]
                    pl["don"]["attached"].append(d)
                    c["attached_don"] += 1
            # ライフは未指定なら FILLER を 5 枚（先頭が上）。
            while len(pl["life"]) < self.life_n[seat]:
                pl["life"].append(pl["deck"].pop(next(i for i, c in enumerate(pl["deck"]) if c["card_id"] == FILLER)))
            for _ in range(self.leader_don[seat]):
                d = pl["don"]["deck"].pop()
                d["attached_to"] = pl["leader"]["uuid"]
                pl["don"]["attached"].append(d)
                pl["leader"]["attached_don"] += 1
            for _ in range(self.don[seat]):
                d = pl["don"]["deck"].pop()
                pl["don"]["active"].append(d)
            for _ in range(self.don_rested[seat]):
                d = pl["don"]["deck"].pop()
                d["is_rest"] = True
                pl["don"]["rested"].append(d)
            # 手札は FILLER で最低 1 枚（捨てる効果などの余地）。指定が無ければそのまま。
        h["manager"]["turn_count"] = self.turn
        h["manager"]["turn_player"] = self.turn_player
        h["manager"]["phase"] = "MAIN"
        engine = load_engine()
        game = engine.Game.from_hidden(json.dumps(h, ensure_ascii=False), "P1", "P2", self.seed)
        return Play(RsGame(game, "P1", "P2"))


class Play:
    """盤面を読み・1 手ずつ進める小さな補助。"""

    def __init__(self, g):
        self.g = g

    # --- 読み ---
    def board(self):
        return self.g.board()

    def pending(self):
        return self.g.get_pending_request(False)

    def zone(self, seat, name):
        b = self.board()["players"][seat]
        return b["zones"][name] if name in b["zones"] else b[name]

    def ids(self, seat, name):
        return [c["card_id"] for c in self.zone(seat, name)]

    def uuid(self, seat, name, card_id):
        for c in self.zone(seat, name):
            if c["card_id"] == card_id:
                return c["uuid"]
        raise KeyError((seat, name, card_id))

    def leader(self, seat):
        return self.board()["players"][seat]["leader"]

    def stage(self, seat):
        return self.board()["players"][seat].get("stage")

    def don_active(self, seat):
        return len(self.board()["players"][seat]["don_active"])

    def legal(self, pid):
        return self.g.get_legal_actions(pid)

    # --- 動かす ---
    def act(self, pid, kind, payload=None):
        return self.g.apply_game_action(pid, kind, payload or {})

    def play(self, pid, card_id, **kw):
        u = self.uuid(pid.lower(), "hand", card_id)
        return self.act(pid, "PLAY", {"card_id": u, **kw})

    def end_turn(self, pid):
        return self.act(pid, "TURN_END")

    def resolve(self, pid, uuids=None, **kw):
        return self.act(pid, "RESOLVE_EFFECT_SELECTION", {"selected_uuids": list(uuids or []), **kw})

    def battle(self, pid, kind, uuid=None):
        return self.g.apply_battle_action(pid, kind, uuid)

    def events(self):
        return self.g.action_events

    # --- 対話の応答 ---
    def action(self):
        pe = self.pending()
        return pe["action"] if pe else None

    def answer(self, accept=True, uuids=None, **kw):
        """今の中断要求（効果の確認・選択）へ、要求された側が応じる。"""
        pe = self.pending()
        return self.act(pe["player_id"], "RESOLVE_EFFECT_SELECTION",
                        {"selected_uuids": list(uuids or []), "accepted": accept, **kw})

    def pass_battle(self):
        """ブロッカー／カウンターの窓を全部パスして、バトルを解決させる。"""
        for _ in range(6):
            pe = self.pending()
            if pe and pe["action"] in ("SELECT_BLOCKER", "SELECT_COUNTER"):
                self.battle(pe["player_id"], "PASS")
            else:
                return

    def settle(self, pick=None, accept=True, answers=None, maxn=12):
        """メイン待ちに戻るまで、出た要求へ順に応じる。戻り値は `(席, 種別, メッセージ)` の列。

        `pick`: 選択要求で選ぶ uuid の優先（無ければ先頭）。`accept`: 確認への既定の答え。
        `answers`: `{メッセージの部分文字列: True/False}`（先に当たったものが勝つ）。
        """
        log = []
        for _ in range(maxn):
            pe = self.pending()
            if pe is None or pe["action"] == "MAIN_ACTION":
                break
            log.append((pe["player_id"], pe["action"], pe["message"]))
            kind = pe["action"]
            if kind in ("SELECT_BLOCKER", "SELECT_COUNTER"):
                self.battle(pe["player_id"], "PASS")
            elif kind in ("CONFIRM_OPTIONAL", "CONFIRM_TRIGGER"):
                ans = accept
                for key, val in (answers or {}).items():
                    if key in pe["message"]:
                        ans = val
                        break
                self.answer(ans)
            elif kind == "SEARCH_AND_SELECT" and pick:
                chosen = [u for u in pick if u in pe["selectable_uuids"]] or pe["selectable_uuids"]
                self.resolve(pe["player_id"], chosen[:1])
            else:
                self.g.apply_move(pe["player_id"], self.legal(pe["player_id"])[0])
        return log

    def hand_count(self, seat):
        return len(self.zone(seat, "hand"))

    def attack(self, pid, attacker_uuid, target_uuid):
        return self.act(pid, "ATTACK", {"uuid": attacker_uuid, "target_ids": [target_uuid]})

    def face_up_life(self, seat):
        return [c["card_id"] for c in self.zone(seat, "life") if c.get("is_face_up")]
