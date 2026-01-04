import sys
import os
import json
import re
from collections import defaultdict

# --- パス設定 (環境に合わせて調整) ---
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
    print("opcg_simフォルダが見える場所で実行してください。")
    sys.exit(1)

def load_cards():
    """カードDBの読み込み"""
    # 複数のパス候補を探査
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
    """
    再帰的にアクションツリーを探索し、Pattern B (中身のないOTHER) を探す
    """
    for act in actions:
        # Pattern B判定ロジック:
        # タイプがOTHER かつ 後続アクション(then_actions)が無い
        if act.type == ActionType.OTHER and not act.then_actions:
            found_list.append(act.raw_text)
        
        # 子要素があればさらに探索
        if act.then_actions:
            find_pattern_b_actions(act.then_actions, found_list)

def classify_failure(text):
    """
    解析失敗したテキストに含まれるキーワードから、修正カテゴリを推測する
    """
    if not text: return "UNKNOWN"
    
    # 優先度の高い順に判定
    if "ライフ" in text:
        return "LIFE_OPS (ライフ操作)"
    if "コスト" in text and ("-" in text or "下げる" in text):
        return "COST_CHANGE (コスト減少)"
    if "速攻" in text:
        return "KEYWORD: RUSH (速攻付与)"
    if "ブロッカー" in text:
        return "KEYWORD: BLOCKER (ブロッカー付与)"
    if "バニッシュ" in text:
        return "KEYWORD: BANISH (バニッシュ付与)"
    if "ダブルアタック" in text:
        return "KEYWORD: DOUBLE (ダブルアタック付与)"
    if "このターン中" in text and "パワー" in text:
        # パワー増減だがParserが拾えなかったケース(複雑な条件など)
        return "COMPLEX_BUFF (複雑なバフ)"
    if "入れ替える" in text:
        return "SWAP (入れ替え)"
    if "無効" in text:
        return "NEGATE (効果無効/バトル無効)"
    if "アタックできない" in text:
        return "ATTACK_RESTRICTION (アタック制限)"
    
    return "UNCATEGORIZED (その他未分類)"

def main():
    cards = load_cards()
    print(f"Loaded {len(cards)} cards. Searching for Pattern B (Unimplemented Effects)...")
    
    pattern_b_registry = defaultdict(list)
    total_failures = 0

    for i, card in enumerate(cards):
        # カード情報の取得
        raw_text = card.get("効果(テキスト)") or card.get("effect_text") or card.get("テキスト") or ""
        raw_trigger = card.get("効果(トリガー)") or card.get("trigger_text") or card.get("トリガー") or ""
        cid = card.get("number") or card.get("品番") or f"ID-{i}"
        name = card.get("name") or card.get("名前") or "Unknown"

        # 正規化
        text = DataCleaner.normalize_text(raw_text)
        trigger = DataCleaner.normalize_text(raw_trigger)
        
        full_text = text + (" / " + trigger if trigger else "")
        if not full_text.strip(): continue

        try:
            abilities = []
            if text: abilities.extend(Effect(text).abilities)
            if trigger: abilities.extend(Effect(trigger).abilities)
            
            # このカードに含まれる Pattern B (未実装部分) を抽出
            failures = []
            for abi in abilities:
                find_pattern_b_actions(abi.actions, failures)
            
            if failures:
                # 重複排除しつつ登録
                unique_failures = list(set(failures))
                for fail_text in unique_failures:
                    category = classify_failure(fail_text)
                    pattern_b_registry[category].append({
                        "id": cid,
                        "name": name,
                        "text": fail_text,
                        "full_text": full_text
                    })
                    total_failures += 1

        except Exception as e:
            pass # クラッシュするカードはここでは無視(別スクリプトで検知済み)

    # --- レポート出力 ---
    print("\n" + "="*60)
    print(f"PATTERN B DETECTION REPORT (Total Issues: {total_failures})")
    print("これらは『テキストはあるがプログラムが理解できず無視される効果』です")
    print("="*60)
    
    # 件数の多いカテゴリ順にソート
    sorted_categories = sorted(pattern_b_registry.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, items in sorted_categories:
        print(f"\n■ {category}: {len(items)} cases")
        # サンプルを5件表示
        for item in items[:5]:
            print(f"  - [{item['id']} {item['name']}] ... {item['text']}")
        if len(items) > 5:
            print(f"    ... and {len(items)-5} more.")

if __name__ == "__main__":
    main()
