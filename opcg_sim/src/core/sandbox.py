import uuid
import random
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..models.models import CardInstance, DonInstance
from ..models.enums import Zone
from ..utils.logger_config import log_event

class SandboxManager:
    def __init__(self, p1_name: str = "P1", p2_name: str = "P2", room_name: str = "Custom Room"):
        self.game_id = str(uuid.uuid4())
        self.room_name = room_name
        self.created_at = datetime.now().isoformat()
        self.turn_count = 0
        self.active_player_id = "p1"
        self.status = "WAITING"
        self.state = {
            "p1": self._init_player(p1_name),
            "p2": self._init_player(p2_name)
        }
        self.ready_states = {"p1": False, "p2": False}

    def _init_player(self, name: str) -> Dict[str, Any]:
        return {
            "name": name,
            "life": [],
            "hand": [],
            "field": [],
            "trash": [],
            "deck": [],
            "don_deck": [DonInstance(owner_id=name) for _ in range(10)],
            "don_active": [],
            "don_rested": [],
            "don_attached": [], 
            "leader": None,
            "stage": None,
            "temp": []
        }

    def set_player_deck(self, pid: str, deck: List[CardInstance], leader: Optional[CardInstance]):
        if self.status != "WAITING": return
        p = self.state[pid]
        p["deck"] = deck
        p["leader"] = leader
        log_event("INFO", "sandbox.set_deck", f"Set deck for {pid}", player="system")

    def toggle_ready(self, pid: str):
        if self.status != "WAITING": return
        self.ready_states[pid] = not self.ready_states[pid]
        log_event("INFO", "sandbox.ready", f"Player {pid} ready: {self.ready_states[pid]}", player="system")

    def start_game(self):
        if self.status != "WAITING": return
        if not self.ready_states["p1"] or not self.ready_states["p2"]: return
        
        self.status = "PLAYING"
        self.turn_count = 1
        self.setup_initial_state()
        self.start_turn_process()
        log_event("INFO", "sandbox.start", "Sandbox game started", player="system")

    def setup_initial_state(self):
        for pid in ["p1", "p2"]:
            player = self.state[pid]
            random.shuffle(player["deck"])
            if player["leader"]:
                life_count = player["leader"].master.life
                for _ in range(life_count):
                    if player["deck"]:
                        player["life"].append(player["deck"].pop(0))
            for _ in range(5):
                if player["deck"]:
                    player["hand"].append(player["deck"].pop(0))

    def refresh_phase(self):
        pid = self.active_player_id
        p = self.state[pid]
        if p["leader"]: 
            p["leader"].is_rest = False
            p["leader"].attached_don = 0
        if p["stage"]: p["stage"].is_rest = False
        for c in p["field"]:
            c.is_rest = False
            c.attached_don = 0
        p["don_active"].extend(p["don_attached"])
        p["don_attached"] = []
        p["don_active"].extend(p["don_rested"])
        p["don_rested"] = []
        for d in p["don_active"]:
            d.is_rest = False
            d.attached_to = None

    def draw_phase(self):
        if self.turn_count == 1: return
        pid = self.active_player_id
        p = self.state[pid]
        if p["deck"]: p["hand"].append(p["deck"].pop(0))

    def don_phase(self):
        pid = self.active_player_id
        p = self.state[pid]
        add_count = 1 if self.turn_count == 1 else 2
        current_total = len(p["don_active"]) + len(p["don_rested"]) + len(p["don_attached"])
        actual_add = min(add_count, max(0, 10 - current_total))
        for _ in range(actual_add):
            if p["don_deck"]:
                don = p["don_deck"].pop(0)
                don.is_rest = False
                p["don_active"].append(don)

    def start_turn_process(self):
        self.refresh_phase()
        self.draw_phase()
        self.don_phase()

    def end_turn_process(self):
        self.active_player_id = "p2" if self.active_player_id == "p1" else "p1"
        self.turn_count += 1
        self.start_turn_process()

    def _find_card_location(self, card_uuid: str):
        for pid in ["p1", "p2"]:
            p_data = self.state[pid]
            if p_data["leader"] and p_data["leader"].uuid == card_uuid: return pid, "leader", -1
            if p_data["stage"] and p_data["stage"].uuid == card_uuid: return pid, "stage", -1
            lists = ["hand", "field", "trash", "life", "deck", "don_active", "don_rested", "don_attached", "temp", "don_deck"]
            for zone_name in lists:
                card_list = p_data[zone_name]
                for i, card in enumerate(card_list):
                    if card.uuid == card_uuid: return pid, zone_name, i
        return None, None, None

    def move_card(self, card_uuid: str, dest_pid: str, dest_zone: str, index: int = -1):
        src_pid, src_zone, src_idx = self._find_card_location(card_uuid)
        if not src_pid: return False
        p_src = self.state[src_pid]
        p_dest = self.state[dest_pid]
        cost_to_pay = 0
        if src_zone == "hand" and dest_zone == "field" and src_pid == dest_pid:
            c = p_src["hand"][src_idx]
            cost_to_pay = getattr(c, "cost", getattr(c.master, "cost", 0))
        if src_zone == "leader": card = p_src["leader"]; p_src["leader"] = None 
        elif src_zone == "stage": card = p_src["stage"]; p_src["stage"] = None
        else: card = p_src[src_zone].pop(src_idx)
        if hasattr(card, "owner_id"): card.owner_id = p_dest["name"]
        if hasattr(card, "is_rest"): card.is_rest = False
        if hasattr(card, "attached_don"): card.attached_don = 0
        if dest_zone == "leader": p_dest["leader"] = card
        elif dest_zone == "stage":
            if p_dest["stage"]: p_dest["trash"].append(p_dest["stage"])
            p_dest["stage"] = card
        else:
            dest_list = p_dest[dest_zone]
            if index == -1 or index >= len(dest_list): dest_list.append(card)
            else: dest_list.insert(index, card)
        if cost_to_pay > 0:
            active_dons = p_src["don_active"]; rest_dons = p_src["don_rested"]
            pay_amount = min(cost_to_pay, len(active_dons))
            for _ in range(pay_amount):
                don = active_dons.pop(0); don.is_rest = True; rest_dons.append(don)
        return True

    def attach_don(self, don_uuid: str, target_uuid: str):
        src_pid, src_zone, src_idx = self._find_card_location(don_uuid)
        if not src_pid or "don" not in src_zone: return False
        p = self.state[src_pid]
        if src_zone == "don_active": don_card = p["don_active"].pop(src_idx)
        elif src_zone == "don_rested": don_card = p["don_rested"].pop(src_idx)
        elif src_zone == "don_attached": don_card = p["don_attached"].pop(src_idx)
        else: return False
        t_pid, t_zone, t_idx = self._find_card_location(target_uuid)
        if not t_pid: p["don_active"].append(don_card); return False
        t_p = self.state[t_pid]
        target_card = t_p["leader"] if t_zone == "leader" else t_p["field"][t_idx] if t_zone == "field" else None
        if not target_card: p["don_active"].append(don_card); return False
        don_card.attached_to = target_uuid; don_card.is_rest = False; p["don_attached"].append(don_card); target_card.attached_don += 1
        return True

    def toggle_rest(self, card_uuid: str):
        src_pid, src_zone, src_idx = self._find_card_location(card_uuid)
        if not src_pid: return
        p_src = self.state[src_pid]
        card = p_src["leader"] if src_zone == "leader" else p_src["stage"] if src_zone == "stage" else p_src[src_zone][src_idx]
        if card: card.is_rest = not card.is_rest

    def process_action(self, req: Dict[str, Any]):
        act_type = req.get("action_type")
        if act_type == "MOVE_CARD": self.move_card(req["card_uuid"], req["dest_player_id"], req["dest_zone"], req.get("index", -1))
        elif act_type == "ATTACH_DON": self.attach_don(req["card_uuid"], req["target_uuid"])
        elif act_type == "TOGGLE_REST": self.toggle_rest(req["card_uuid"])
        elif act_type == "TURN_END": self.end_turn_process()
        elif act_type == "DRAW":
            pid = req.get("player_id", self.active_player_id)
            if self.state[pid]["deck"]: self.state[pid]["hand"].append(self.state[pid]["deck"].pop(0))
        elif act_type == "READY": self.toggle_ready(req.get("player_id"))
        elif act_type == "START": self.start_game()

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "room_name": self.room_name,
            "status": self.status,
            "ready_states": self.ready_states,
            "turn_info": {"turn_count": self.turn_count, "current_phase": "SANDBOX", "active_player_id": self.active_player_id},
            "players": {pid: self._player_to_dict(pid) for pid in ["p1", "p2"]}
        }

    def _player_to_dict(self, pid: str):
        p = self.state[pid]
        def fmt(card, face_up=True):
            if not card: return None
            d = card.to_dict()
            if face_up: d["is_face_up"] = True
            return d
        return {
            "player_id": p["name"], "name": p["name"],
            "life_count": len(p["life"]), "hand_count": len(p["hand"]),
            "don_deck_count": len(p["don_deck"]), "leader": fmt(p["leader"]), "stage": fmt(p["stage"]),
            "zones": {
                "field": [fmt(c) for c in p["field"]], "hand": [fmt(c) for c in p["hand"]],
                "life": [fmt(c, False) for c in p["life"]], "trash": [fmt(c) for c in p["trash"]],
                "deck": [fmt(c, False) for c in p["deck"]], "don_active": [fmt(d) for d in p["don_active"]],
                "don_rested": [fmt(d) for d in p["don_rested"]]
            }
        }
