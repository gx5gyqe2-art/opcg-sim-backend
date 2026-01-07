import sys
import os
import json
import traceback
from typing import List, Dict, Any

# --- パス設定 ---
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

# ---------------------------------------------------------
# ロギングヘルパー: 標準出力をファイルにも保存する
# ---------------------------------------------------------
class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ---------------------------------------------------------
# ヘルパー: モック環境構築
# ---------------------------------------------------------
def create_mock_card(owner_id: str, def_data: Any) -> CardInstance:
    """JSON定義からCardInstanceを生成"""
    
    KEYWORD_MAP = {
        "DOUBLE_ATTACK": "ダブルアタック",
        "BANISH": "バニッシュ",
        "BLOCKER": "ブロッカー",
        "RUSH": "速攻"
    }

    if isinstance(def_data, str):
        name = def_data
        cost = 1
        traits = []
        is_rest = False
        keywords = []
        text = ""
    else:
        name = def_data.get("name", "Unknown")
        cost = def_data.get("cost", 1)
        traits = def_data.get("traits", [])
        is_rest = def_data.get("is_rest", False)
        keywords = def_data.get("keywords", [])
        text = def_data.get("text", "")

    converted_keywords = set()
    for k in keywords:
        converted_keywords.add(KEYWORD_MAP.get(k, k))

    master = CardMaster(
        card_id=f"MOCK-{name}",
        name=name,
        type=CardType.CHARACTER,
        color=Color.RED,
        cost=cost,
        power=5000,
        counter=1000,
        attribute=Attribute.SLASH,
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
    
    # プレイヤー状態のセットアップ
    for pid, p_obj in [("p1", p1), ("p2", p2)]:
        p_data = scenario.get("setup", {}).get(pid, {})
        
        # Leader Setup
        if "leader" in p_data:
            l_val = p_data["leader"]
            l_card = create_mock_card(p_obj.name, l_val)
            # CardMasterはfrozenなので強制書き換えでLeader属性を付与
            object.__setattr__(l_card.master, 'type', CardType.LEADER)
            object.__setattr__(l_card.master, 'life', 5)
            p_obj.leader = l_card

        # Field
        for c_def in p_data.get("field", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.field.append(card)
            
        # Hand
        for c_def in p_data.get("hand", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.hand.append(card)
        
        # Trash
        for c_def in p_data.get("trash", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.trash.append(card)
            
        # Deck
        for c_def in p_data.get("deck", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.deck.append(card)
            
        # Life
        for c_def in p_data.get("life", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.life.append(card)

        # Don Active
        active_count = p_data.get("don_active", 0)
        p_obj.don_active = [DonInstance(p_obj.name) for _ in range(active_count)]

        # Don Rested
        rested_count = p_data.get("don_rested", 0)
        p_obj.don_rested = []
        for _ in range(rested_count):
            d = DonInstance(p_obj.name)
            d.is_rest = True
            p_obj.don_rested.append(d)
        
        # Don Deck
        total_in_play = active_count + rested_count
        deck_count = 10 - total_in_play
        if deck_count < 0: deck_count = 0
        p_obj.don_deck = [DonInstance(p_obj.name) for _ in range(deck_count)]

    return gm

def find_card_by_name(player: Player, name: str) -> CardInstance:
    # 検索範囲をフィールドと手札に加えて、ライフとトラッシュも対象にする
    for c in player.field + player.hand + player.life + player.trash:
        if c.master.name == name:
            return c
    return None

# ---------------------------------------------------------
# テスト実行ロジック
# ---------------------------------------------------------
def run_scenario(scenario: Dict) -> Dict:
    result_report = {"id": scenario["id"], "title": scenario["title"], "passed": False, "details": []}
    
    try:
        # 1. セットアップ
        gm = setup_game_from_json(scenario)
        
        # 操作プレイヤーの切り替え (デフォルトはP1)
        active_player_key = scenario.get("active_player", "p1")
        active_player = gm.p1 if active_player_key == "p1" else gm.p2
        
        if active_player_key == "p2":
            gm.turn_player = gm.p2
            gm.opponent = gm.p1
        
        # 2. 効果発動元の特定
        source_name = scenario["source"]
        source_card = find_card_by_name(active_player, source_name)
        if not source_card:
            raise Exception(f"Source card '{source_name}' not found in {active_player_key.upper()} zones.")
        
        # 3. アクション実行 または テキストParse
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
                
                # ダミーAbility (後続のTrigger検証などをパスするため)
                class DummyAbility:
                    trigger = None
                ability = DummyAbility()
        else:
            text = scenario["text"]
            effect_obj = Effect(text)
            if not effect_obj.abilities:
                raise Exception("Parser failed to extract abilities.")
            ability = effect_obj.abilities[0]

        # トリガー検証
        expected_trigger = scenario.get("expected_trigger")
        if expected_trigger:
            actual_trigger = ability.trigger.name
            if actual_trigger != expected_trigger:
                result_report["details"].append(f"❌ Trigger Mismatch: Expected {expected_trigger}, Got {actual_trigger}")
                result_report["passed"] = False
                return result_report
            else:
                result_report["details"].append(f"✅ Trigger matched: {actual_trigger}")
        
        # 4. 効果解決 (Interaction処理含む)
        success = False
        try:
            if not hasattr(ability, 'trigger') or ability.trigger is not None:
                gm.resolve_ability(active_player, ability, source_card)
            
            # Interactionループ (Battleフェーズ進行も含む)
            interaction_steps = scenario.get("interaction", [])
            step_idx = 0
            
            loop_limit = 20
            # active_interaction または pending_request がある限り回す
            while (gm.active_interaction or gm.get_pending_request()) and loop_limit > 0:
                loop_limit -= 1
                
                req = gm.active_interaction
                
                # Active Interactionがない場合、Pending Requestをラップして処理
                if not req:
                    pending = gm.get_pending_request()
                    if pending:
                        action_type = pending.get("action")
                        
                        # シナリオ指定が尽きている場合、ブロック/カウンターは自動パス
                        if step_idx >= len(interaction_steps):
                            if action_type == "SELECT_BLOCKER":
                                gm.handle_block(None)
                                continue
                            elif action_type == "SELECT_COUNTER":
                                target_pid = pending.get("player_id")
                                target_p = gm.p1 if target_pid == gm.p1.name else gm.p2
                                gm.apply_counter(target_p, None)
                                continue
                        
                        # 処理対象としてラップ
                        req = {
                            "action_type": action_type,
                            "candidates": pending.get("candidates", []),
                            "can_skip": pending.get("can_skip", False)
                        }
                    else:
                        break

                # --- Interaction 処理 ---
                if step_idx >= len(interaction_steps):
                    if req.get("can_skip"):
                        gm.resolve_interaction(active_player, {}) # Pass
                    else:
                        raise Exception(f"Unexpected interaction required: {req.get('action_type')}")
                else:
                    step_input = interaction_steps[step_idx]
                    
                    # 候補の検証ロジック
                    if "verify_candidates" in step_input:
                        verify = step_input["verify_candidates"]
                        candidates = req.get("candidates", [])
                        
                        # candidatesは辞書(pending由来)かオブジェクト(interaction由来)か混在する可能性あり
                        candidate_names = []
                        for c in candidates:
                            if isinstance(c, dict): candidate_names.append(c.get("name", "Unknown"))
                            else: candidate_names.append(c.master.name)
                        
                        for expected in verify.get("has_names", []):
                            if expected not in candidate_names:
                                raise Exception(f"Validation Error: Expected candidate '{expected}' not found. Candidates: {candidate_names}")

                    payload = {}
                    
                    # カード選択
                    if "select_cards" in step_input:
                        target_names = step_input["select_cards"]
                        selected_uuids = []
                        candidates = req.get("candidates", [])
                        
                        # pendingの場合はUUIDsが別フィールドにあるが、ここではcandidatesから探す
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
                    
                    # ブロッカー使用 (manual_action風の指定がある場合)
                    if "use_blocker" in step_input and step_input["use_blocker"]:
                        blocker_name = step_input.get("blocker_card")
                        candidates = req.get("candidates", [])
                        found_blocker = None
                        for c in candidates:
                            c_name = c.get("name") if isinstance(c, dict) else c.master.name
                            if c_name == blocker_name:
                                # pending requestの場合はオブジェクトが必要なため、gmから探す必要あり
                                c_uuid = c.get("uuid") if isinstance(c, dict) else c.uuid
                                found_blocker = gm._find_card_by_uuid(c_uuid)
                                break
                        if found_blocker:
                            gm.handle_block(found_blocker)
                            step_idx += 1
                            continue # 次のループへ

                    # 選択肢 (Option)
                    if "select_option" in step_input:
                        payload["selected_option_index"] = step_input["select_option"]

                    if gm.active_interaction:
                        gm.resolve_interaction(active_player, payload)
                    elif req.get("action_type") == "SELECT_COUNTER":
                         # カウンターなどは現状手動で関数を呼ぶ必要がある（簡易実装）
                         target_pid = req.get("player_id") # pending requestはdict
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

        # 5. 検証 (Expectations)
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

        # 状態検証ヘルパー
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

    return result_report

def main():
    # ★追加: コンソール出力をファイルにも保存する設定
    log_path = os.path.join(current_dir, "full_execution_log.txt")
    sys.stdout = TeeLogger(log_path)
    print(f"📄 Full Execution Log will be saved to: {log_path}\n")

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

    print(f"🚀 Running {len(scenarios)} Scenarios (JSON Mode)...\n")
    
    # 全シナリオの結果を格納するリスト
    all_results = []
    
    passed_count = 0
    for s in scenarios:
        res = run_scenario(s)
        all_results.append(res)
        
        status_icon = "✅" if res["passed"] else "❌"
        print(f"{status_icon} [{s['id']}] {s['title']}")
        for d in res["details"]:
            print(f"    {d}")
        print("-" * 50)
        
        if res["passed"]: passed_count += 1

    print(f"\nResult: {passed_count}/{len(scenarios)} Passed")

    # --- レポート出力処理 ---
    report_file_txt = os.path.join(current_dir, "test_report.txt")
    report_file_json = os.path.join(current_dir, "test_report.json")

    try:
        # テキスト形式のレポート出力
        with open(report_file_txt, "w", encoding="utf-8") as f:
            f.write(f"Test Execution Report\n")
            f.write(f"Total Scenarios: {len(scenarios)}\n")
            f.write(f"Passed: {passed_count}\n")
            f.write(f"Failed: {len(scenarios) - passed_count}\n")
            f.write("=" * 60 + "\n\n")
            
            for res in all_results:
                status = "PASS" if res["passed"] else "FAIL"
                icon = "✅" if res["passed"] else "❌"
                f.write(f"{icon} [{status}] {res['id']}: {res['title']}\n")
                for d in res['details']:
                    f.write(f"    {d}\n")
                f.write("-" * 60 + "\n")
        
        print(f"\n📄 Text Report saved to: {report_file_txt}")

        # JSON形式のレポート出力
        with open(report_file_json, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total": len(scenarios),
                    "passed": passed_count,
                    "failed": len(scenarios) - passed_count
                },
                "results": all_results
            }, f, ensure_ascii=False, indent=2)
            
        print(f"📄 JSON Report saved to: {report_file_json}")

    except Exception as e:
        print(f"❌ Failed to save report files: {e}")

if __name__ == "__main__":
    main()
