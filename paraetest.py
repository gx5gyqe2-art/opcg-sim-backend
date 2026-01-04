import sys
import os
import json
import re
from collections import defaultdict

# --- パス設定 ---
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
    """カードDBの読み込み"""
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
    """ActionType.OTHER かつ then_actions 無しを抽出"""
    for act in actions:
        if act.type == ActionType.OTHER and not act.then_actions:
            found_list.append(act.raw_text)
        if act.then_actions:
            find_pattern_b_actions(act.then_actions, found_list)

def classify_detailed(text):
    """
    未分類テキストを詳細にカテゴライズする
    """
    if not text: return "UNKNOWN"
    
    # 1. ルール・特殊処理系
    if "カード名" in text and "扱う" in text:
        return "RULE: NAME_CHANGE (名称変更)"
    if "デッキ" in text and "何枚でも" in text:
        return "RULE: DECK_BUILD (デッキ構築)"
    if "勝利する" in text:
        return "RULE: VICTORY (特殊勝利)"
    
    # 2. バトル・耐性系
    if "ブロックされない" in text:
        return "BATTLE: UNBLOCKABLE (ブロック不可)"
    if "KOされない" in text:
        return "BATTLE: KO_PROTECT (KO耐性)"
    if "効果" in text and "受けない" in text:
        return "BATTLE: IMMUNITY (効果耐性)"
    if "バトル" in text and "終了" in text:
        return "BATTLE: END_BATTLE (バトル強制終了)"
    
    # 3. パワー・ダメージ系
    if "元のパワー" in text:
        return "STAT: BASE_POWER (基本パワー変更)"
    if "ダメージ" in text and "与える" in text: # 注釈削除で消えなかったもの
        return "STAT: DAMAGE_DEAL (ダメージを与える)"
    
    # 4. トリッキーな移動・発動
    if "持ち主" in text and ("手札" in text or "デッキ" in text) and "終了時" in text:
        return "MOVE: BOUNCE_DELAYED (終了時バウンス)"
    if "入れ替える" in text:
        return "MOVE: SWAP (入れ替え)"
    if "発動する" in text:
        return "EFFECT: ACTIVATE (他効果の発動)"
    
    # 5. その他キーワード(注釈削除漏れチェック含む)
    if "ダブルアタック" in text: return "KEYWORD: DOUBLE (注釈漏れ?)"
    if "バニッシュ" in text: return "KEYWORD: BANISH (注釈漏れ?)"
    if "再起動" in text: return "KEYWORD: REBOOT (注釈漏れ?)"
    
    # 6. 前回修正したはずのもの(残存チェック)
    if "アタック" in text and "できない" in text: return "CHECK: ATTACK_DISABLE (修正漏れ)"
    if "無効" in text: return "CHECK: NEGATE (修正漏れ)"
    if "ライフ" in text: return "CHECK: LIFE (修正漏れ)"
    if "コスト" in text and ("-" in text or "下げる" in text): return "CHECK: COST (修正漏れ)"
    
    return "UNCATEGORIZED (完全未分類)"

def main():
    cards = load_cards()
    print(f"Loaded {len(cards)} cards. Analyzing remaining UNCATEGORIZED effects...")
    
    registry = defaultdict(list)
    total_issues = 0

    for i, card in enumerate(cards):
        raw_text = card.get("効果(テキスト)") or card.get("effect_text") or ""
        raw_trigger = card.get("効果(トリガー)") or card.get("trigger_text") or ""
        
        # 正規化(Parserと同じロジックを通す)
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
                    # ここで詳細分類を実行
                    category = classify_detailed(fail_text)
                    registry[category].append({
                        "id": cid,
                        "name": name,
                        "text": fail_text
                    })
                    total_issues += 1

        except Exception:
            pass 

    # --- レポート出力 ---
    print("\n" + "="*60)
    print(f"DETAILED ANALYSIS REPORT (Remaining Issues: {total_issues})")
    print("="*60)
    
    sorted_categories = sorted(registry.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, items in sorted_categories:
        print(f"\n■ {category}: {len(items)} cases")
        for item in items[:3]:
            print(f"  - [{item['id']} {item['name']}] ... {item['text']}")
        if len(items) > 3:
            print(f"    ... and {len(items)-3} more.")

if __name__ == "__main__":
    main()
