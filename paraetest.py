import sys
import os
import json
import unicodedata
from typing import List

import importlib.util

# --- 外部ストレージ強制読み込みモード ---
current_dir = os.path.dirname(os.path.abspath(__file__))
opcg_sim_path = os.path.join(current_dir, "opcg_sim")

def import_direct(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None

try:
    print("モジュールを直接ファイルパスから読み込みます...")
    
    # 依存関係の順序で手動ロード (パスは環境に合わせて微調整が必要かもしれません)
    # 1. opcg_sim (パッケージ)
    import_direct("opcg_sim", os.path.join(opcg_sim_path, "__init__.py"))
    
    # 2. opcg_sim.src
    src_path = os.path.join(opcg_sim_path, "src")
    import_direct("opcg_sim.src", os.path.join(src_path, "__init__.py"))
    
    # 3. 必要なモジュール本体
    # parser.py
    parser_path = os.path.join(src_path, "core", "effects", "parser.py")
    parser_mod = import_direct("opcg_sim.src.core.effects.parser", parser_path)
    Effect = parser_mod.Effect
    
    # loader.py
    loader_path = os.path.join(src_path, "utils", "loader.py")
    loader_mod = import_direct("opcg_sim.src.utils.loader", loader_path)
    DataCleaner = loader_mod.DataCleaner
    
    print("読み込み成功！")

except Exception as e:
    print(f"強制読み込み失敗: {e}")
    print("解決策: フォルダごと 'On My iPhone' (ローカル) にコピーして実行してください。")
    sys.exit(1)
# --------------------------------------

# データパス
DATA_FILE = os.path.join(current_dir, "opcg_sim", "data", "opcg_cards.json")

# ... (以下 load_cards 関数など元のコード)

def load_cards():
# ... (以下、変更なし)
    if not os.path.exists(DATA_FILE):
        print(f"データファイルが見つかりません: {DATA_FILE}")
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def run_test():
    cards = load_cards()
    print(f"=== Parser 解析率テスト ===")
    print(f"対象ファイル: {DATA_FILE}")
    print(f"総カード数: {len(cards)}")
    print("-" * 40)

    total_cards_with_text = 0
    success_count = 0
    failed_cards = []

    # テキストがあるカードのみを対象にテスト
    for card in cards:
        # カラム名の揺らぎ吸収 (Loaderのロジックを簡易再現)
        raw_text = card.get("effect_text") or card.get("テキスト") or card.get("Text") or ""
        raw_trigger = card.get("trigger_text") or card.get("トリガー") or card.get("Trigger") or ""
        
        # 正規化
        text = DataCleaner.normalize_text(raw_text)
        trigger = DataCleaner.normalize_text(raw_trigger)
        
        # テキストが全くないカード(バニラ)は除外
        if not text and not trigger:
            continue
            
        total_cards_with_text += 1
        
        is_success = False
        abilities = []
        
        try:
            # メイン効果のパース
            if text:
                parser = Effect(text)
                abilities.extend(parser.abilities)
            
            # トリガー効果のパース
            if trigger:
                parser = Effect(trigger)
                abilities.extend(parser.abilities)

            # 解析成功の定義: 
            # 1. エラーが落ちない
            # 2. 何かしらのAbilityが生成されている
            # 3. アクションの中身が空でない (ActionType.OTHER だけでもOKとするか、厳密にするかは調整)
            if len(abilities) > 0:
                is_success = True
            
        except Exception as e:
            # エラー発生時は失敗とみなす
            is_success = False
            # print(f"Error on {card.get('number')}: {e}")

        if is_success:
            success_count += 1
        else:
            # 失敗したカードを記録
            failed_cards.append({
                "id": card.get("number") or card.get("品番"),
                "name": card.get("name") or card.get("名前"),
                "text": text
            })

    # 結果出力
    print(f"テキストありカード数: {total_cards_with_text}")
    print(f"解析成功数: {success_count}")
    print(f"解析失敗数: {total_cards_with_text - success_count}")
    print(f"解析率: {success_count / total_cards_with_text * 100:.2f}%")
    print("-" * 40)
    
    if failed_cards:
        print("\n=== 解析失敗カード (サンプル 10件) ===")
        for fc in failed_cards[:10]:
            print(f"ID: {fc['id']} | {fc['name']}")
            print(f"TEXT: {fc['text'][:50]}...")
            print("-" * 20)
            
    # 特定の複雑なテキストの構造チェック (デバッグ用)
    print("\n=== 構造解析デバッグ (複雑な効果のツリー確認) ===")
    complex_text = "自分のリーダーが特徴《ワノ国》を持つ場合、カードを1枚引く。その後、自分の手札1枚を捨てる。"
    print(f"Input: {complex_text}")
    try:
        debug_parser = Effect(complex_text)
        for i, abil in enumerate(debug_parser.abilities):
            print(f"Ability {i}:")
            for action in abil.actions:
                print_action_tree(action, indent=1)
    except Exception as e:
        print(f"Debug Parse Error: {e}")

def print_action_tree(action, indent=0):
    spaces = "  " * indent
    cond_str = f"[IF {action.condition.raw_text}] " if action.condition else ""
    print(f"{spaces}- {cond_str}{action.type.name} (Val:{action.value})")
    
    if action.then_actions:
        print(f"{spaces}  User Then:")
        for child in action.then_actions:
            print_action_tree(child, indent + 1)

if __name__ == "__main__":
    run_test()
