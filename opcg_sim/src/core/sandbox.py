import uuid
import random
import traceback
from typing import Dict, List, Optional, Any
from ..models.models import CardInstance, DonInstance
from ..models.enums import Zone
from ..utils.logger_config import log_event

class SandboxManager:
    def __init__(self, p1_deck: List[CardInstance], p2_deck: List[CardInstance], p1_leader: Optional[CardInstance], p2_leader: Optional[CardInstance], p1_name: str = "P1", p2_name: str = "P2"):
        self.game_id = str(uuid.uuid4())
        self.turn_count = 1
        self.active_player_id = "p1"
        self.state = {
            "p1": self._init_player(p1_name, p1_deck, p1_leader),
            "p2": self._init_player(p2_name, p2_deck, p2_leader)
        }
        self.setup_initial_state()
        # ゲーム開始時に1ターン目を開始する
        self.start_turn_process()

    def _init_player(self, name: str, deck: List[CardInstance], leader: Optional[CardInstance]) -> Dict[str, Any]:
        return {
            "name": name,
            "life": [],
            "hand": [],
            "field": [],
            "trash": [],
            "deck": deck,
            "don_deck": [DonInstance(owner_id=name) for _ in range(10)],
            "don_active": [],
            "don_rested": [],
            "don_attached": [], 
            "leader": leader,
            "stage": None,
            "temp": []
        }

    def setup_initial_state(self):
        for pid in ["p1", "p2"]:
            player = self.state[pid]
            random.shuffle(player["deck"])
            # ライフ設定
            if player["leader"]:
                life_count = player["leader"].master.life
                for _ in range(life_count):
                    if player["deck"]:
                        player["life"].append(player["deck"].pop(0))
            # 初期手札5枚
            for _ in range(5):
                if player["deck"]:
                    player["hand"].append(player["deck"].pop(0))

    # --- ターン遷移ロジック ---

    def refresh_phase(self):
        """リフレッシュフェーズ: 全カードをアクティブにし、ドン!!を戻す"""
        pid = self.active_player_id
        p = self.state[pid]
        
        # 1. キャラクター、リーダー、ステージのアクティブ化 & ドン剥離
        if p["leader"]: 
            p["leader"].is_rest = False
            p["leader"].attached_don = 0
        if p["stage"]: 
            p["stage"].is_rest = False
        
        for c in p["field"]:
            c.is_rest = False
            c.attached_don = 0
            
        # 2. ドン!!の返却とアクティブ化
        # 付与されていたドンをアクティブに戻す
        p["don_active"].extend(p["don_attached"])
        p["don_attached"] = []
        
        # レスト状態のドンをアクティブに戻す
        p["don_active"].extend(p["don_rested"])
        p["don_rested"] = []
        
        # 全ドン!!の状態リセット
        for d in p["don_active"]:
            d.is_rest = False
            d.attached_to = None
            
        log_event("INFO", "sandbox.refresh", f"Refreshed all cards for {pid}", player="system")

    def draw_phase(self):
        """ドローフェーズ: カードを引く"""
        # 先攻1ターン目(turn_count=1)はドローなし
        if self.turn_count == 1:
            return

        pid = self.active_player_id
        p = self.state[pid]
        
        if p["deck"]:
            card = p["deck"].pop(0)
            p["hand"].append(card)
            log_event("INFO", "sandbox.draw", f"Player {pid} drew a card", player="system")

    def don_phase(self):
        """ドン!!フェーズ: ドン!!を追加する"""
        pid = self.active_player_id
        p = self.state[pid]
        
        # 先攻1ターン目は1枚、それ以外は2枚追加
        add_count = 1 if self.turn_count == 1 else 2
        
        # 場のドン!!は最大10枚まで
        # (refresh後なので active, rested, attached の合計を見る必要はないが、念のため全領域チェック)
        current_total = len(p["don_active"]) + len(p["don_rested"]) + len(p["don_attached"])
        limit = 10
        can_add = max(0, limit - current_total)
        actual_add = min(add_count, can_add)
        
        for _ in range(actual_add):
            if p["don_deck"]:
                don = p["don_deck"].pop(0)
                don.is_rest = False
                p["don_active"].append(don)
        
        log_event("INFO", "sandbox.don", f"Player {pid} added {actual_add} don!!", player="system")

    def start_turn_process(self):
        """ターン開始処理フロー"""
        log_event("INFO", "sandbox.turn_start", f"Turn {self.turn_count} started for {self.active_player_id}", player="system")
        self.refresh_phase()
        self.draw_phase()
        self.don_phase()

    def end_turn_process(self):
        """ターン終了処理: プレイヤー交代して次のターンへ"""
        log_event("INFO", "sandbox.turn_end", f"Turn {self.turn_count} ended for {self.active_player_id}", player="system")
        
        # プレイヤー交代
        self.active_player_id = "p2" if self.active_player_id == "p1" else "p1"
        self.turn_count += 1
        
        # 次のターンの開始処理を実行
        self.start_turn_process()

    # --- 汎用操作 ---

    def _find_card_location(self, card_uuid: str):
        for pid in ["p1", "p2"]:
            p_data = self.state[pid]
            if p_data["leader"] and p_data["leader"].uuid == card_uuid: return pid, "leader", -1
            if p_data["stage"] and p_data["stage"].uuid == card_uuid: return pid, "stage", -1
            
            lists = {
                "hand": p_data["hand"], "field": p_data["field"], "trash": p_data["trash"],
                "life": p_data["life"], "deck": p_data["deck"],
                "don_active": p_data["don_active"], "don_rested": p_data["don_rested"],
                "don_attached": p_data["don_attached"], "temp": p_data["temp"]
            }
            for zone_name, card_list in lists.items():
                for i, card in enumerate(card_list):
                    if card.uuid == card_uuid:
                        return pid, zone_name, i
        return None, None, None

    def move_card(self, card_uuid: str, dest_pid: str, dest_zone: str, index: int = -1):
        src_pid, src_zone, src_idx = self._find_card_location(card_uuid)
        if not src_pid: return False

        card = None
        p_src = self.state[src_pid]
        
        # 取り出し
        if src_zone == "leader":
            # Leaderは基本移動しないが、Sandboxならありえるかもしれない
            # ここでは複製せず移動とするなら、元の場所をNoneにする必要があるが、
            # Leader枠が空くのはまずいので、コピー移動にするか要検討。
            # 一旦、Leader/Stageは特別扱いせず移動可能にする。
            card = p_src["leader"]
            # p_src["leader"] = None # Leaderを消すとエラーになる箇所が多いので残すか、None許容するか
            pass 
        elif src_zone == "stage":
            card = p_src["stage"]
            p_src["stage"] = None
        else:
            src_list = p_src[src_zone]
            card = src_list.pop(src_idx)

        if not card: return False

        # ステータスリセット (移動時)
        if hasattr(card, "is_rest"): card.is_rest = False
        if hasattr(card, "attached_don"): card.attached_don = 0

        p_dest = self.state[dest_pid]
        
        # 配置
        if dest_zone == "leader":
            # Leaderの入れ替えは特殊だがSandboxなので上書き
            p_dest["leader"] = card
        elif dest_zone == "stage":
            old = p_dest["stage"]
            if old: p_dest["trash"].append(old)
            p_dest["stage"] = card
        else:
            dest_list = p_dest.get(dest_zone)
            if dest_list is None: return False 
            
            if index == -1 or index >= len(dest_list):
                dest_list.append(card)
            else:
                dest_list.insert(index, card)
        
        log_event("INFO", "sandbox.move", f"Moved {card.uuid} to {dest_pid}.{dest_zone}", player="system")
        return True

    def toggle_rest(self, card_uuid: str):
        src_pid, src_zone, src_idx = self._find_card_location(card_uuid)
        if not src_pid: return
        
        card = None
        p_src = self.state[src_pid]
        if src_zone == "leader": card = p_src["leader"]
        elif src_zone == "stage": card = p_src["stage"]
        else: card = p_src[src_zone][src_idx]
        
        if card:
            card.is_rest = not card.is_rest
            log_event("INFO", "sandbox.rest", f"Toggled rest for {card.uuid}", player="system")

    def process_action(self, req: Dict[str, Any]):
        act_type = req.get("action_type")
        if act_type == "MOVE_CARD":
            self.move_card(req["card_uuid"], req["dest_player_id"], req["dest_zone"], req.get("index", -1))
        elif act_type == "TOGGLE_REST":
            self.toggle_rest(req["card_uuid"])
        elif act_type == "TURN_END":
            self.end_turn_process()
        elif act_type == "DRAW":
            # 手動ドロー用
            pid = req.get("player_id", self.active_player_id)
            if self.state[pid]["deck"]:
                self.state[pid]["hand"].append(self.state[pid]["deck"].pop(0))

    def to_dict(self):
        return {
            "game_id": self.game_id,
            "mode": "sandbox",
            "turn_info": {
                "turn_count": self.turn_count,
                "current_phase": "SANDBOX",
                "active_player_id": self.active_player_id,
                "winner": None
            },
            "players": {
                pid: self._player_to_dict(pid) for pid in ["p1", "p2"]
            },
            "active_battle": None
        }

    def _player_to_dict(self, pid: str):
        p = self.state[pid]
        
        def fmt(card, face_up=True):
            if not card: return None
            d = card.to_dict()
            if face_up:
                d["is_face_up"] = True
            return d

        return {
            "player_id": p["name"], 
            "name": p["name"],
            "life_count": len(p["life"]),
            "hand_count": len(p["hand"]),
            "don_deck_count": len(p["don_deck"]),
            "don_active": [fmt(d) for d in p["don_active"]],
            "don_rested": [fmt(d) for d in p["don_rested"]],
            "leader": fmt(p["leader"]),
            "stage": fmt(p["stage"]),
            "zones": {
                "field": [fmt(c) for c in p["field"]],
                "hand": [fmt(c) for c in p["hand"]],
                "life": [fmt(c, False) for c in p["life"]],
                "trash": [fmt(c) for c in p["trash"]],
                "stage": fmt(p["stage"]),
                "deck": [fmt(c, False) for c in p["deck"]]
            }
        }
