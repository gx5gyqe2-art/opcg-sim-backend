import sys
import os
import json
import traceback
import math
import io
import shutil
from typing import List, Dict, Any, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from opcg_sim.src.core.gamestate import GameManager, Player, CardInstance
    from opcg_sim.src.models.models import CardMaster, DonInstance
    from opcg_sim.src.models.enums import CardType, Color, Attribute, TriggerType
    from opcg_sim.src.utils.loader import DataCleaner
    from opcg_sim.src.core.effects.parser import Effect
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class StdoutFilter(object):
    def __init__(self):
        self.terminal = sys.stdout

    def write(self, message):
        is_json_log = message.strip().startswith("{") and '"sessionId":' in message
        if not is_json_log:
            self.terminal.write(message)

    def flush(self):
        self.terminal.flush()

def create_mock_card(owner_id: str, def_data: Any) -> CardInstance:
    KEYWORD_MAP = {
        "DOUBLE_ATTACK": "ダブルアタック",
        "BANISH": "バニッシュ",
        "BLOCKER": "ブロッカー",
        "RUSH": "速攻"
    }

    name = "Unknown"
    cost = 1
    power = 5000
    counter = 1000
    color_val = Color.RED
    attr_val = Attribute.SLASH
    type_val = CardType.CHARACTER
    traits = []
    is_rest = False
    keywords = []
    text = ""
    
    if isinstance(def_data, str):
        name = def_data
    else:
        name = def_data.get("name", "Unknown")
        cost = def_data.get("cost", 1)
        power = def_data.get("power", 5000)
        counter = def_data.get("counter", 1000)
        
        c_str = def_data.get("color", "RED")
        if hasattr(Color, c_str): color_val = getattr(Color, c_str)
        
        a_str = def_data.get("attribute", "SLASH")
        if hasattr(Attribute, a_str): attr_val = getattr(Attribute, a_str)
        
        t_str = def_data.get("type", "CHARACTER")
        if hasattr(CardType, t_str): type_val = getattr(CardType, t_str)
        
        traits = def_data.get("traits", [])
        is_rest = def_data.get("is_rest", False)
        keywords = def_data.get("keywords", [])
        text = def_data.get("text", "")

    if text == "なし":
        text = ""

    converted_keywords = set()
    for k in keywords:
        converted_keywords.add(KEYWORD_MAP.get(k, k))

    master = CardMaster(
        card_id=f"MOCK-{name}",
        name=name,
        type=type_val,
        color=color_val,
        cost=cost,
        power=power,
        counter=counter,
        attribute=attr_val,
        traits=traits,
        effect_text=text,
        trigger_text="",
        life=0,
        keywords=converted_keywords
    )
    inst = CardInstance(master, owner_id)
    inst.is_rest = is_rest
    return inst

def setup_game_from_json(scenario: Dict) -> GameManager:
    p1 = Player("P1", [], None)
    p2 = Player("P2", [], None)
    gm = GameManager(p1, p2)
    gm.start_game()
    
    for pid, p_obj in [("p1", p1), ("p2", p2)]:
        p_data = scenario.get("setup", {}).get(pid, {})
        
        if "leader" in p_data:
            l_val = p_data["leader"]
            l_card = create_mock_card(p_obj.name, l_val)
            object.__setattr__(l_card.master, 'type', CardType.LEADER)
            object.__setattr__(l_card.master, 'life', 5)
            p_obj.leader = l_card

        for c_def in p_data.get("field", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.field.append(card)
            
        for c_def in p_data.get("hand", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.hand.append(card)
        
        for c_def in p_data.get("trash", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.trash.append(card)
            
        for c_def in p_data.get("deck", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.deck.append(card)
            
        for c_def in p_data.get("life", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.life.append(card)

        active_count = p_data.get("don_active", 0)
        p_obj.don_active = [DonInstance(p_obj.name) for _ in range(active_count)]

        rested_count = p_data.get("don_rested", 0)
        p_obj.don_rested = []
        for _ in range(rested_count):
            d = DonInstance(p_obj.name)
            d.is_rest = True
            p_obj.don_rested.append(d)
        
        total_in_play = active_count + rested_count
        deck_count = 10 - total_in_play
        if deck_count < 0: deck_count = 0
        p_obj.don_deck = [DonInstance(p_obj.name) for _ in range(deck_count)]

    return gm

def find_card_by_name(player: Player, name: str) -> CardInstance:
    for c in player.field + player.hand + player.life + player.trash:
        if c.master.name == name:
            return c
    return None

def run_scenario(scenario: Dict) -> Tuple[Dict, str]:
    result_report = {"id": scenario["id"], "title": scenario["title"], "passed": False, "details": []}
    
    capture_io = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = capture_io

    try:
        gm = setup_game_from_json(scenario)
        
        active_player_key = scenario.get("active_player", "p1")
        active_player = gm.p1 if active_player_key == "p1" else gm.p2
        
        if active_player_key == "p2":
            gm.turn_player = gm.p2
            gm.opponent = gm.p1
        
        source_name = scenario["source"]
        source_card = find_card_by_name(active_player, source_name)
        if not source_card:
            raise Exception(f"Source card '{source_name}' not found in {active_player_key.upper()} zones.")
        
        ability = None
        if "manual_action" in scenario:
            act = scenario["manual_action"]
            if act["type"] == "ATTACK":
                target_name = act["target"]
                target_card = None
                if target_name == "P2Leader": target_card = gm.p2.leader
                elif target_name == "P1Leader": target_card = gm.p1.leader
                
                if not target_card:
                     raise Exception(f"Manual Action Target '{target_name}' not found")
                
                gm.declare_attack(source_card, target_card)
                
                class DummyAbility:
                    trigger = None
                ability = DummyAbility()
        else:
            text = scenario["text"]
            effect_obj = Effect(text)
            if not effect_obj.abilities:
                raise Exception("Parser failed to extract abilities.")
            ability = effect_obj.abilities[0]

        expected_trigger = scenario.get("expected_trigger")
        if expected_trigger:
            actual_trigger = ability.trigger.name
            if actual_trigger != expected_trigger:
                result_report["details"].append(f"❌ Trigger Mismatch: Expected {expected_trigger}, Got {actual_trigger}")
                result_report["passed"] = False
                captured_log = capture_io.getvalue()
                sys.stdout = original_stdout
                return result_report, captured_log
            else:
                result_report["details"].append(f"✅ Trigger matched: {actual_trigger}")
        
        success = False
        try:
            if not hasattr(ability, 'trigger') or ability.trigger is not None:
                gm.resolve_ability(active_player, ability, source_card)
            
            interaction_steps = scenario.get("interaction", [])
            step_idx = 0
            
            loop_limit = 20
            while (gm.active_interaction or gm.get_pending_request()) and loop_limit > 0:
                loop_limit -= 1
                
                req = gm.active_interaction
                
                if not req:
                    pending = gm.get_pending_request()
                    if pending:
                        action_type = pending.get("action")
                        
                        if step_idx >= len(interaction_steps):
                            if action_type == "SELECT_BLOCKER":
                                gm.handle_block(None)
                                continue
                            elif action_type == "SELECT_COUNTER":
                                target_pid = pending.get("player_id")
                                target_p = gm.p1 if target_pid == gm.p1.name else gm.p2
                                gm.apply_counter(target_p, None)
                                continue
                        
                        req = {
                            "action_type": action_type,
                            "candidates": pending.get("candidates", []),
                            "can_skip": pending.get("can_skip", False)
                        }
                    else:
                        break

                if step_idx >= len(interaction_steps):
                    if req.get("can_skip"):
                        gm.resolve_interaction(active_player, {}) 
                    else:
                        raise Exception(f"Unexpected interaction required: {req.get('action_type')}")
                else:
                    step_input = interaction_steps[step_idx]
                    
                    if "verify_candidates" in step_input:
                        verify = step_input["verify_candidates"]
                        candidates = req.get("candidates", [])
                        
                        candidate_names = []
                        for c in candidates:
                            if isinstance(c, dict): candidate_names.append(c.get("name", "Unknown"))
                            else: candidate_names.append(c.master.name)
                        
                        for expected in verify.get("has_names", []):
                            if expected not in candidate_names:
                                raise Exception(f"Validation Error: Expected candidate '{expected}' not found. Candidates: {candidate_names}")

                    payload = {}
                    
                    if "select_cards" in step_input:
                        target_names = step_input["select_cards"]
                        selected_uuids = []
                        candidates = req.get("candidates", [])
                        
                        for t_name in target_names:
                            found = None
                            for c in candidates:
                                c_name = c.get("name") if isinstance(c, dict) else c.master.name
                                c_uuid = c.get("uuid") if isinstance(c, dict) else c.uuid
                                if c_name == t_name:
                                    found = c
                                    selected_uuids.append(c_uuid)
                                    break
                        payload["selected_uuids"] = selected_uuids
                    
                    if "use_blocker" in step_input and step_input["use_blocker"]:
                        blocker_name = step_input.get("blocker_card")
                        candidates = req.get("candidates", [])
                        found_blocker = None
                        for c in candidates:
                            c_name = c.get("name") if isinstance(c, dict) else c.master.name
                            if c_name == blocker_name:
                                c_uuid = c.get("uuid") if isinstance(c, dict) else c.uuid
                                found_blocker = gm._find_card_by_uuid(c_uuid)
                                break
                        if found_blocker:
                            gm.handle_block(found_blocker)
                            step_idx += 1
                            continue

                    if "select_option" in step_input:
                        payload["selected_option_index"] = step_input["select_option"]

                    if gm.active_interaction:
                        gm.resolve_interaction(active_player, payload)
                    elif req.get("action_type") == "SELECT_COUNTER":
                         target_pid = req.get("player_id") 
                         if not target_pid: target_pid = active_player.name
                         target_p = gm.p1 if target_pid == gm.p1.name else gm.p2
                         gm.apply_counter(target_p, None)

                    step_idx += 1
            
            success = True

        except ValueError as ve:
            result_report["details"].append(f"Caught Expected Error: {str(ve)}")
            success = False
        except Exception as e:
            result_report["details"].append(f"Runtime Error: {str(e)}")
            traceback.print_exc()
            success = False

        expect = scenario.get("expect", {})
        
        exp_success = expect.get("success")
        if exp_success is not None:
            if exp_success != success:
                result_report["details"].append(f"❌ Success Mismatch: Expected {exp_success}, Got {success}")
            else:
                result_report["details"].append(f"✅ Success matched: {success}")

        exp_msg = expect.get("error_msg_contains")
        if exp_msg:
            found_msg = any(exp_msg in d for d in result_report["details"])
            if found_msg:
                result_report["details"].append(f"✅ Error message contains '{exp_msg}'")
            else:
                result_report["details"].append(f"❌ Error message missing '{exp_msg}'")

        def check_prop(pid, p_obj, key, label):
            if key in expect:
                actual = 0
                if "don_deck_count" in key: actual = len(p_obj.don_deck)
                elif "don_active" in key: actual = len(p_obj.don_active)
                elif "hand_count" in key: actual = len(p_obj.hand)
                elif "deck_count" in key: actual = len(p_obj.deck)
                elif "life_count" in key: actual = len(p_obj.life)
                elif "trash_count" in key: actual = len(p_obj.trash)
                elif "field_count" in key: actual = len(p_obj.field)
                
                if actual == expect[key]:
                    result_report["details"].append(f"✅ {pid} {label}: {actual}")
                else:
                    result_report["details"].append(f"❌ {pid} {label}: Expected {expect[key]}, Got {actual}")

        check_prop("p1", gm.p1, "p1_hand_count", "Hand Count")
        check_prop("p1", gm.p1, "p1_deck_count", "Deck Count")
        check_prop("p1", gm.p1, "p1_life_count", "Life Count")
        check_prop("p1", gm.p1, "p1_trash_count", "Trash Count")
        check_prop("p1", gm.p1, "p1_field_count", "Field Count")
        check_prop("p1", gm.p1, "p1_don_active", "Don Active")
        check_prop("p1", gm.p1, "p1_don_deck_count", "Don Deck Count")

        check_prop("p2", gm.p2, "p2_hand_count", "Hand Count")
        check_prop("p2", gm.p2, "p2_deck_count", "Deck Count")
        check_prop("p2", gm.p2, "p2_life_count", "Life Count")
        check_prop("p2", gm.p2, "p2_trash_count", "Trash Count")
        check_prop("p2", gm.p2, "p2_field_count", "Field Count")

        if "p2_field_has" in expect:
            current_names = [c.master.name for c in gm.p2.field]
            for name in expect["p2_field_has"]:
                if name in current_names:
                     result_report["details"].append(f"✅ P2 Field has {name}")
                else:
                     result_report["details"].append(f"❌ P2 Field missing {name}. Current: {current_names}")

        if "p2_trash_has" in expect:
            current_names = [c.master.name for c in gm.p2.trash]
            for name in expect["p2_trash_has"]:
                if name in current_names:
                     result_report["details"].append(f"✅ P2 Trash has {name}")
                else:
                     result_report["details"].append(f"❌ P2 Trash missing {name}. Current: {current_names}")

        if "p1_field_has" in expect:
            current_names = [c.master.name for c in gm.p1.field]
            for name in expect["p1_field_has"]:
                if name in current_names:
                     result_report["details"].append(f"✅ P1 Field has {name}")
                else:
                     result_report["details"].append(f"❌ P1 Field missing {name}. Current: {current_names}")
        
        if "p1_hand_has" in expect:
            current_names = [c.master.name for c in gm.p1.hand]
            for name in expect["p1_hand_has"]:
                if name in current_names:
                     result_report["details"].append(f"✅ P1 Hand has {name}")
                else:
                     result_report["details"].append(f"❌ P1 Hand missing {name}. Current: {current_names}")
        
        if "p2_hand_has" in expect:
            current_names = [c.master.name for c in gm.p2.hand]
            for name in expect["p2_hand_has"]:
                if name in current_names:
                     result_report["details"].append(f"✅ P2 Hand has {name}")
                else:
                     result_report["details"].append(f"❌ P2 Hand missing {name}. Current: {current_names}")

        def verify_card_props(player_obj, check_list):
            for check in check_list:
                name = check["name"]
                target = find_card_by_name(player_obj, name)
                if target:
                    if "has_flag" in check:
                        flag = check["has_flag"]
                        if flag in target.flags:
                            result_report["details"].append(f"✅ {name} has flag {flag}")
                        else:
                            result_report["details"].append(f"❌ {name} missing flag {flag}")
                    if "power_buff" in check:
                        if target.power_buff == check["power_buff"]:
                            result_report["details"].append(f"✅ {name} power_buff is {check['power_buff']}")
                        else:
                            result_report["details"].append(f"❌ {name} power_buff mismatch. Exp {check['power_buff']}, Got {target.power_buff}")
                    if "cost" in check:
                        if target.current_cost == check["cost"]:
                             result_report["details"].append(f"✅ {name} cost is {check['cost']}")
                        else:
                             result_report["details"].append(f"❌ {name} cost mismatch. Exp {check['cost']}, Got {target.current_cost}")
                    if "is_rest" in check:
                        if target.is_rest == check["is_rest"]:
                             result_report["details"].append(f"✅ {name} is_rest is {check['is_rest']}")
                        else:
                             result_report["details"].append(f"❌ {name} is_rest mismatch. Exp {check['is_rest']}, Got {target.is_rest}")
                    if "has_keyword" in check:
                        kw = check["has_keyword"]
                        if kw in target.current_keywords:
                             result_report["details"].append(f"✅ {name} has keyword {kw}")
                        else:
                             result_report["details"].append(f"❌ {name} missing keyword {kw}")
                    if "attached_don" in check:
                        if target.attached_don == check["attached_don"]:
                             result_report["details"].append(f"✅ {name} attached_don is {check['attached_don']}")
                        else:
                             result_report["details"].append(f"❌ {name} attached_don mismatch. Exp {check['attached_don']}, Got {target.attached_don}")

        if "p2_field_check" in expect:
            verify_card_props(gm.p2, expect["p2_field_check"])
        
        if "p1_field_check" in expect:
            verify_card_props(gm.p1, expect["p1_field_check"])
        
        error_count = sum(1 for d in result_report["details"] if "❌" in d)
        result_report["passed"] = (error_count == 0)

    except Exception as e:
        result_report["passed"] = False
        result_report["details"].append(f"CRITICAL ERROR: {traceback.format_exc()}")

    captured_log = capture_io.getvalue()
    sys.stdout = original_stdout
    
    return result_report, captured_log

def main():
    sys.stdout = StdoutFilter()

    json_path = os.path.join(current_dir, "test_scenarios.json")
    if not os.path.exists(json_path):
        print(f"Scenario file not found: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return

    failed_logs_dir = os.path.join(current_dir, "failed_logs")
    if os.path.exists(failed_logs_dir):
        try:
            shutil.rmtree(failed_logs_dir)
        except Exception as e:
            print(f"Warning: Could not clear failed_logs dir: {e}")
    os.makedirs(failed_logs_dir, exist_ok=True)

    print(f"🚀 Running {len(scenarios)} Scenarios (JSON Mode)...\n")
    
    all_results = []
    failed_logs_collection = []
    
    passed_count = 0
    for s in scenarios:
        res, log_content = run_scenario(s)
        all_results.append(res)
        
        print(log_content, end="")
        
        status_icon = "✅" if res["passed"] else "❌"
        
        if res["passed"]:
            print(f"{status_icon} [{s['id']}] {s['title']}")
            passed_count += 1
        else:
            print(f"{status_icon} [{s['id']}] {s['title']}")
            print("    ▼ FAILURE DETAILS:")
            failure_details_text = ""
            for d in res["details"]:
                if "❌" in d or "Expected Error" in d:
                    line = f"      {d}\n"
                    print(line, end="")
                    failure_details_text += line
            print("-" * 50)
            
            scenario_json_str = json.dumps(s, indent=2, ensure_ascii=False)
            
            full_details = "\n".join(res['details'])
            
            combined_log = (
                f"=== ❌ FAILED SCENARIO: [{s['id']}] {s['title']} ===\n\n"
                f"--- [1. SCENARIO DEFINITION] ---\n{scenario_json_str}\n\n"
                f"--- [2. EXECUTION LOG] ---\n{log_content}\n"
                f"--- [3. RESULT REPORT] ---\n{full_details}\n\n"
                f"{'='*60}\n"
            )
            failed_logs_collection.append(combined_log)

    print(f"\nResult: {passed_count}/{len(scenarios)} Passed")

    if failed_logs_collection:
        print(f"\n💾 Saving {len(failed_logs_collection)} failed scenarios for AI analysis...")
        print(f"   Directory: {failed_logs_dir}")
        
        items_per_file = 5
        num_chunks = math.ceil(len(failed_logs_collection) / items_per_file)

        for i in range(num_chunks):
            chunk = failed_logs_collection[i * items_per_file : (i + 1) * items_per_file]
            filename = os.path.join(failed_logs_dir, f"failed_log_part_{i+1:03d}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                for log_block in chunk:
                    f.write(log_block)
            print(f"  -> Created: {filename}")
    else:
        print("\n🎉 No failures! No failed logs generated.")

    report_file_txt = os.path.join(current_dir, "test_report.txt")
    try:
        with open(report_file_txt, "w", encoding="utf-8") as f:
            f.write(f"Test Execution Report\nTotal: {len(scenarios)}, Passed: {passed_count}\n\n")
            for res in all_results:
                icon = "✅" if res["passed"] else "❌"
                f.write(f"{icon} {res['id']}: {res['title']}\n")
                for d in res['details']: f.write(f"    {d}\n")
                f.write("-" * 60 + "\n")
        print(f"📄 Report saved to: {report_file_txt}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")

if __name__ == "__main__":
    main()
