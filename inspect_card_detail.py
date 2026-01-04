import sys
import os
import json
import unicodedata

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if "opcg_sim" not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.models.enums import Player
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def normalize(text):
    if not text: return ""
    # NFKC正規化(半角・全角、濁点の結合などを統一)
    return unicodedata.normalize('NFKC', text)

def main():
    # データ読み込み
    data_path = os.path.join(current_dir, "opcg_sim", "data", "opcg_cards.json")
    if not os.path.exists(data_path):
        data_path = "opcg_cards.json" # カレントディレクトリも探す
    
    with open(data_path, "r", encoding="utf-8") as f:
        cards = json.load(f)
    
    print(f"Loaded {len(cards)} cards.")

    # ★検索キーワード(短めに設定してヒットさせる)★
    SEARCH_KEYWORD = "ロー" 
    
    # 検索実行
    candidates = []
    for c in cards:
        name = normalize(c.get("name") or c.get("名前") or "")
        if SEARCH_KEYWORD in name:
            candidates.append(c)
    
    if not candidates:
        print(f"No cards found matching '{SEARCH_KEYWORD}'")
        return

    print(f"\nFound {len(candidates)} candidates for '{SEARCH_KEYWORD}':")
    
    # 目的のカード(効果テキストがあるもの)を優先して探す
    target_card = None
    
    for i, c in enumerate(candidates):
        cid = c.get("card_id") or c.get("品番") or "?"
        name = c.get("name") or c.get("名前")
        text = normalize(c.get("effect_text") or c.get("効果(テキスト)") or "")
        
        print(f"  [{i}] ID:{cid} Name:{name} Text:{text[:20]}...")

        # 「登場させる」や「戻す」が含まれるローを自動選択してみる(トラファルガー・ローの典型的な効果)
        if not target_card and ("登場" in text or "戻" in text) and "ブロッカー" not in text: 
             target_card = c

    # 強制的にリストの6番目(トラファルガー・ロー)を解析する
    target_card = candidates[6] 

    print("\n" + "="*60)
    cid = target_card.get("card_id") or target_card.get("品番")
    name = target_card.get("name") or target_card.get("名前")
    print(f"🔍 INSPECTING: {name} ({cid})")
    print("="*60)
    
    raw_text = target_card.get("effect_text") or target_card.get("効果(テキスト)") or ""
    print(f"Raw Text: {raw_text}\n")
    
    # Parser実行
    effect = Effect(raw_text)
    
    print("--- PARSER RESULT ---")
    if not effect.abilities:
        print("❌ No abilities parsed!")
    
    for i, ability in enumerate(effect.abilities):
        print(f"\n[Ability {i+1}] Trigger: {ability.trigger}")
        
        def print_actions(actions, indent=2):
            spaces = " " * indent
            for j, act in enumerate(actions):
                print(f"{spaces}Step {j+1}: {act.type}")
                print(f"{spaces}  Raw: '{act.raw_text}'")
                
                if act.condition:
                    print(f"{spaces}  ❓ Condition: {act.condition.type} (Val:{act.condition.value})")
                
                if act.target:
                    t = act.target
                    print(f"{spaces}  🎯 Target Query:")
                    print(f"{spaces}     - Raw: '{t.raw_text}'")
                    print(f"{spaces}     - Zone: {t.zone}")
                    print(f"{spaces}     - Count: {t.count}")
                    # ★ここが一番重要:誰を対象にしているか★
                    print(f"{spaces}     - Player: {t.player}  <-- CHECK THIS!") 
                    if t.player.name == "SELF":
                        print(f"{spaces}       (⚠️ WARNING: Defaulted to SELF?)")
                else:
                    print(f"{spaces}  Target: None")
                
                if act.then_actions:
                    print(f"{spaces}  ⬇️ Then:")
                    print_actions(act.then_actions, indent + 4)

        print_actions(ability.actions)

if __name__ == "__main__":
    main()
