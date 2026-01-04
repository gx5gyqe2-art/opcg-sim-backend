import sys
import os
import json
import re
from collections import defaultdict

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if "opcg_sim" not in sys.path:
    sys.path.insert(0, project_root)

try:
    from opcg_sim.src.utils.loader import DataCleaner
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.models.enums import ActionType
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def load_cards():
    candidates = [
        os.path.join(current_dir, "opcg_sim", "data", "opcg_cards.json"),
        os.path.join(current_dir, "data", "opcg_cards.json"),
        "opcg_cards.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    print("Error: opcg_cards.json found.")
    return []

def find_pattern_b_actions(actions, found_list):
    for act in actions:
        if act.type == ActionType.OTHER and not act.then_actions:
            found_list.append(act.raw_text)
        if act.then_actions:
            find_pattern_b_actions(act.then_actions, found_list)

def classify_deep_dive(text):
    """
    UNCATEGORIZED (OTHER) として残ったものをさらに深掘り分類する
    """
    if not text: return "EMPTY"
    
    # 1. 発動制限・行動制限 (RESTRICTION系)
    # Parserの検知漏れの可能性が高い
    if "発動できない" in text:
        return "RESTRICTION: ACTIVATION (発動禁止)"
    if "アタックできない" in text:
        return "RESTRICTION: ATTACK (アタック禁止)"
    if "ブロックできない" in text or "ブロックされない" in text:
        return "RESTRICTION: BLOCK (ブロック禁止/不可)"
    if "加えられない" in text:
        return "RESTRICTION: ADD_TO_HAND (手札回収禁止)"

    # 2. コスト操作 (COST_CHANGE系)
    # "+" などの記号ゆらぎや、"コストXになる" などの固定化
    if "コスト" in text:
        if "+" in text or "＋" in text:
            return "COST: INCREASE (コスト加算)"
        if "なる" in text:
            return "COST: SET (コスト固定)"
        return "COST: OTHER (その他コスト関連)"

    # 3. パワー操作 (BUFF/DEBUFF系)
    # 複雑な条件でParserが漏らしたケース
    if "パワー" in text:
        if "なる" in text:
            return "POWER: SET (パワー固定)"
        if "倍" in text:
            return "POWER: MULTIPLY (パワー倍増)"
        return "POWER: OTHER (その他パワー関連)"

    # 4. バトル・耐性
    if "KOされない" in text:
        return "BATTLE: KO_IMMUNITY (KO耐性)"
    if "効果" in text and "受けない" in text:
        return "BATTLE: EFFECT_IMMUNITY (効果耐性)"
    
    # 5. ルール・特殊処理
    if "扱う" in text:
        return "RULE: TREAT_AS (名称/属性変更)"
    if "何枚でも" in text:
        return "RULE: DECK_BUILD (デッキ構築)"
    if "入れ替える" in text:
        return "RULE: SWAP (入れ替え)"
    if "やり直す" in text:
        return "RULE: RESTART (再実行)"
    
    # 6. ダメージ系
    if "ダメージ" in text:
        return "GAME: DAMAGE (ダメージ処理)"
    
    # 7. その他
    if "全て" in text or "すべて" in text:
        return "TARGET: ALL (全体対象の複雑な効果)"
    
    return "UNKNOWN: HARDCASE (要個別確認)"

def main():
    cards = load_cards()
    print(f"Loaded {len(cards)} cards. Deep diving into remaining UNCATEGORIZED effects...")
    
    registry = defaultdict(list)
    total_issues = 0

    for i, card in enumerate(cards):
        raw_text = card.get("効果(テキスト)") or card.get("effect_text") or ""
        raw_trigger = card.get("効果(トリガー)") or card.get("trigger_text") or ""
        
        # 正規化
        text = Effect(raw_text)._normalize(raw_text) if raw_text else ""
        trigger = Effect(raw_trigger)._normalize(raw_trigger) if raw_trigger else ""
        
        cid = card.get("number") or f"ID-{i}"
        name = card.get("name") or "Unknown"

        try:
            abilities = []
            if text: abilities.extend(Effect(text).abilities)
            if trigger: abilities.extend(Effect(trigger).abilities)
            
            failures = []
            for abi in abilities:
                find_pattern_b_actions(abi.actions, failures)
            
            if failures:
                unique_failures = list(set(failures))
                for fail_text in unique_failures:
                    # ここで深掘り分類
                    category = classify_deep_dive(fail_text)
                    registry[category].append({
                        "id": cid,
                        "name": name,
                        "text": fail_text
                    })
                    total_issues += 1

        except Exception:
            pass 

    # --- レポート出力 ---
    print("\n" + "="*60)
    print(f"DEEP DIVE REPORT (Remaining Issues: {total_issues})")
    print("="*60)
    
    sorted_categories = sorted(registry.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, items in sorted_categories:
        print(f"\n■ {category}: {len(items)} cases")
        for item in items[:3]: # サンプル3件
            print(f"  - [{item['id']} {item['name']}] ... {item['text']}")
        if len(items) > 3:
            print(f"    ... and {len(items)-3} more.")

if __name__ == "__main__":
    main()
