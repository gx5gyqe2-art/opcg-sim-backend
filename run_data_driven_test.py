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
# ヘルパー: モック環境構築
# ---------------------------------------------------------
def create_mock_card(owner_id: str, def_data: Any) -> CardInstance:
    """JSON定義からCardInstanceを生成"""
    if isinstance(def_data, str):
        name = def_data
        cost = 1
        traits = []
        is_rest = False
    else:
        name = def_data.get("name", "Unknown")
        cost = def_data.get("cost", 1)
        traits = def_data.get("traits", [])
        is_rest = def_data.get("is_rest", False)

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
        effect_text="",
        trigger_text="",
        life=0
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
        
        # Don Deck (調整)
        deck_count = 10 - active_count
        p_obj.don_deck = [DonInstance(p_obj.name) for _ in range(deck_count)]

    return gm

def find_card_by_name(player: Player, name: str) -> CardInstance:
    # 検索範囲をフィールドと手札に限定（必要に応じて拡張）
    for c in player.field + player.hand:
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
            raise Exception(f"Source card '{source_name}' not found in {active_player_key.upper()} field/hand.")
        
        # 3. テキストのParse
        text = scenario["text"]
        effect_obj = Effect(text)
        if not effect_obj.abilities:
            raise Exception("Parser failed to extract abilities.")
        
        ability = effect_obj.abilities[0]

        # ▼ トリガー検証ロジックを追加
        expected_trigger = scenario.get("expected_trigger")
        if expected_trigger:
            actual_trigger = ability.trigger.name  # Enumのメンバー名 (例: "ON_PLAY")
            if actual_trigger != expected_trigger:
                result_report["details"].append(f"❌ Trigger Mismatch: Expected {expected_trigger}, Got {actual_trigger}")
                # トリガー不一致は即失敗扱いにする
                result_report["passed"] = False
                return result_report
            else:
                result_report["details"].append(f"✅ Trigger matched: {actual_trigger}")
        
        # 4. 効果解決 (Interaction処理含む)
        success = False
        try:
            gm.resolve_ability(active_player, ability, source_card)
            
            # Interactionループ
            interaction_steps = scenario.get("interaction", [])
            step_idx = 0
            
            loop_limit = 10
            while gm.active_interaction and loop_limit > 0:
                loop_limit -= 1
                req = gm.active_interaction
                
                if step_idx >= len(interaction_steps):
                    if req.get("can_skip"):
                        gm.resolve_interaction(active_player, {}) # Pass
                    else:
                        raise Exception(f"Unexpected interaction required: {req['action_type']}")
                else:
                    step_input = interaction_steps[step_idx]
                    
                    # 候補の検証ロジック
                    if "verify_candidates" in step_input:
                        verify = step_input["verify_candidates"]
                        candidates = req.get("candidates", [])
                        candidate_names = [c.master.name for c in candidates]
                        
                        for expected in verify.get("has_names", []):
                            if expected not in candidate_names:
                                raise Exception(f"Validation Error: Expected candidate '{expected}' not found. Candidates: {candidate_names}")

                        for unexpected in verify.get("missing_names", []):
                            if unexpected in candidate_names:
                                raise Exception(f"Validation Error: Unexpected candidate '{unexpected}' found. Candidates: {candidate_names}")
                        
                        if "count" in verify:
                            if len(candidates) != verify["count"]:
                                raise Exception(f"Validation Error: Candidate count mismatch. Expected {verify['count']}, Got {len(candidates)}")

                    payload = {}
                    
                    # カード選択
                    if "select_cards" in step_input:
                        target_names = step_input["select_cards"]
                        selected_uuids = []
                        candidates = req.get("candidates", [])
                        for t_name in target_names:
                            found = next((c for c in candidates if c.master.name == t_name), None)
                            if found: selected_uuids.append(found.uuid)
                        payload["selected_uuids"] = selected_uuids
                        
                    # 選択肢 (Option)
                    if "select_option" in step_input:
                        payload["selected_option_index"] = step_input["select_option"]

                    gm.resolve_interaction(active_player, payload)
                    step_idx += 1
            
            success = True

        except ValueError as ve:
            result_report["details"].append(f"Caught Expected Error: {str(ve)}")
            success = False
        except Exception as e:
            result_report["details"].append(f"Runtime Error: {str(e)}")
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
                
                # 具体的なキーを先に判定
                if "don_deck_count" in key: actual = len(p_obj.don_deck)
                elif "don_active" in key: actual = len(p_obj.don_active)
                
                # 一般的なキーは後にする
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

        if "p2_field_check" in expect:
            for check in expect["p2_field_check"]:
                name = check["name"]
                target = find_card_by_name(gm.p2, name)
                if target:
                    if "has_flag" in check:
                        flag = check["has_flag"]
                        if flag in target.flags:
                            result_report["details"].append(f"✅ {name} has flag {flag}")
                        else:
                            result_report["details"].append(f"❌ {name} missing flag {flag}")
        
        error_count = sum(1 for d in result_report["details"] if "❌" in d)
        result_report["passed"] = (error_count == 0)

    except Exception as e:
        result_report["passed"] = False
        result_report["details"].append(f"CRITICAL ERROR: {traceback.format_exc()}")

    return result_report

def main():
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
