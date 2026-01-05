import sys
import os
import json  # 変更: yaml -> json
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
        
        # Trash (追加)
        for c_def in p_data.get("trash", []):
            card = create_mock_card(p_obj.name, c_def)
            p_obj.trash.append(card)

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
        p1 = gm.p1
        
        # 2. 効果発動元の特定
        source_name = scenario["source"]
        source_card = find_card_by_name(p1, source_name)
        if not source_card:
            raise Exception(f"Source card '{source_name}' not found in P1 field/hand.")
        
        # 3. テキストのParse
        text = scenario["text"]
        effect_obj = Effect(text)
        if not effect_obj.abilities:
            raise Exception("Parser failed to extract abilities.")
        
        ability = effect_obj.abilities[0]
        
        # 4. 効果解決 (Interaction処理含む)
        success = False
        try:
            gm.resolve_ability(p1, ability, source_card)
            
            # Interactionループ
            interaction_steps = scenario.get("interaction", [])
            step_idx = 0
            
            loop_limit = 10
            while gm.active_interaction and loop_limit > 0:
                loop_limit -= 1
                req = gm.active_interaction
                
                if step_idx >= len(interaction_steps):
                    if req.get("can_skip"):
                        gm.resolve_interaction(p1, {}) # Pass
                    else:
                        raise Exception(f"Unexpected interaction required: {req['action_type']}")
                else:
                    step_input = interaction_steps[step_idx]
                    
                    # ▼▼▼ 追加: 候補の検証ロジック ▼▼▼
                    if "verify_candidates" in step_input:
                        verify = step_input["verify_candidates"]
                        candidates = req.get("candidates", [])
                        # CardInstanceから名前リストを抽出
                        candidate_names = [c.master.name for c in candidates]
                        
                        # 1. 含まれているべきカードのチェック (has_names)
                        for expected in verify.get("has_names", []):
                            if expected not in candidate_names:
                                raise Exception(f"Validation Error: Expected candidate '{expected}' not found. Candidates: {candidate_names}")

                        # 2. 含まれていてはいけないカードのチェック (missing_names)
                        for unexpected in verify.get("missing_names", []):
                            if unexpected in candidate_names:
                                raise Exception(f"Validation Error: Unexpected candidate '{unexpected}' found. Candidates: {candidate_names}")
                        
                        # 3. 候補数のチェック
                        if "count" in verify:
                            if len(candidates) != verify["count"]:
                                raise Exception(f"Validation Error: Candidate count mismatch. Expected {verify['count']}, Got {len(candidates)}")
                    # ▲▲▲ 追加ここまで ▲▲▲

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

                    gm.resolve_interaction(p1, payload)
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
        
        # 成功/失敗の期待値
        exp_success = expect.get("success")
        if exp_success is not None:
            if exp_success != success:
                result_report["details"].append(f"❌ Success Mismatch: Expected {exp_success}, Got {success}")
            else:
                result_report["details"].append(f"✅ Success matched: {success}")

        # エラーメッセージの検証
        exp_msg = expect.get("error_msg_contains")
        if exp_msg:
            found_msg = any(exp_msg in d for d in result_report["details"])
            if found_msg:
                result_report["details"].append(f"✅ Error message contains '{exp_msg}'")
            else:
                result_report["details"].append(f"❌ Error message missing '{exp_msg}'")

        # 状態検証
        if "p1_don_active" in expect:
            actual = len(p1.don_active)
            if actual == expect["p1_don_active"]:
                result_report["details"].append(f"✅ P1 Don Active: {actual}")
            else:
                result_report["details"].append(f"❌ P1 Don Active: Expected {expect['p1_don_active']}, Got {actual}")
        
        if "p1_don_deck_count" in expect:
            actual = len(p1.don_deck)
            if actual == expect["p1_don_deck_count"]:
                result_report["details"].append(f"✅ P1 Don Deck: {actual}")
            else:
                result_report["details"].append(f"❌ P1 Don Deck: Expected {expect['p1_don_deck_count']}, Got {actual}")

        if "p1_hand_count" in expect:
            actual = len(p1.hand)
            if actual == expect["p1_hand_count"]:
                result_report["details"].append(f"✅ P1 Hand Count: {actual}")
            else:
                result_report["details"].append(f"❌ P1 Hand Count: Expected {expect['p1_hand_count']}, Got {actual}")

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

# ---------------------------------------------------------
# メイン
# ---------------------------------------------------------
def main():
    # 変更: JSONファイルを読み込む
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
    
    passed_count = 0
    for s in scenarios:
        res = run_scenario(s)
        status_icon = "✅" if res["passed"] else "❌"
        print(f"{status_icon} [{s['id']}] {s['title']}")
        for d in res["details"]:
            print(f"    {d}")
        print("-" * 50)
        
        if res["passed"]: passed_count += 1

    print(f"\nResult: {passed_count}/{len(scenarios)} Passed")

if __name__ == "__main__":
    main()
