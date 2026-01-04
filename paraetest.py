import sys
import os
import json
import unicodedata

# パスの設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.utils.loader import DataCleaner
except ImportError:
    # パスが通らない場合の予備
    sys.path.append(os.path.join(os.path.dirname(__file__), "opcg-sim-backend"))
    try:
        from opcg_sim.src.core.effects.parser import Effect
        from opcg_sim.src.utils.loader import DataCleaner
    except ImportError:
        print("エラー: opcg_sim モジュールが見つかりません。")
        sys.exit(1)

# データパス
DATA_FILE = os.path.join("opcg_sim", "data", "opcg_cards.json")

def normalize_key(text: str) -> str:
    """キー照合用の強力な正規化 (NFKC)"""
    return unicodedata.normalize('NFKC', text)

def load_cards():
    candidates = [
        DATA_FILE,
        os.path.join(os.path.dirname(__file__), DATA_FILE),
        "opcg_cards.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"データファイル読み込み: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    print(f"警告: データファイルが見つかりません。")
    return []

def run_test():
    cards = load_cards()
    if not cards:
        return

    print(f"総カード数: {len(cards)}")
    print("-" * 40)

    total_cards_with_text = 0
    success_count = 0
    failed_cards = []

    # 探索対象のキー（正規化前）
    target_keys = [
        "effect_text", "テキスト", "Text", "text", "効果(テキスト)", 
        "trigger_text", "トリガー", "Trigger", "trigger", "効果(トリガー)"
    ]
    # 照合用に正規化セットを作成
    normalized_target_keys = {normalize_key(k) for k in target_keys}

    for card in cards:
        # カード側のキーも全て正規化して辞書を作り直す
        card_normalized = {normalize_key(k): v for k, v in card.items()}

        # テキストの抽出
        raw_text = ""
        raw_trigger = ""
        
        # 本文探索
        for k in ["effect_text", "テキスト", "Text", "text", "効果(テキスト)"]:
            nk = normalize_key(k)
            if nk in card_normalized and card_normalized[nk]:
                raw_text = card_normalized[nk]
                break
        
        # トリガー探索
        for k in ["trigger_text", "トリガー", "Trigger", "trigger", "効果(トリガー)"]:
            nk = normalize_key(k)
            if nk in card_normalized and card_normalized[nk]:
                raw_trigger = card_normalized[nk]
                break
        
        # 文字列の正規化
        text = DataCleaner.normalize_text(raw_text)
        trigger = DataCleaner.normalize_text(raw_trigger)
        
        # バニラ（テキストなし）はスキップ
        if not text and not trigger:
            continue
            
        total_cards_with_text += 1
        is_success = False
        abilities = []
        
        try:
            # 解析実行
            if text:
                parser = Effect(text)
                abilities.extend(parser.abilities)
            if trigger:
                parser = Effect(trigger)
                abilities.extend(parser.abilities)

            # 成功判定: 1つ以上のAbilityが生成されていること
            if len(abilities) > 0:
                is_success = True
                
        except Exception:
            is_success = False

        if is_success:
            success_count += 1
        else:
            # 失敗ログ用情報
            c_id = card_normalized.get(normalize_key("品番")) or card_normalized.get("number") or "?"
            c_name = card_normalized.get(normalize_key("名前")) or card_normalized.get("name") or "?"
            failed_cards.append({"id": c_id, "name": c_name, "text": text})

    # 結果表示
    print(f"テキストありカード数: {total_cards_with_text}")
    
    if total_cards_with_text > 0:
        rate = (success_count / total_cards_with_text) * 100
        print(f"解析成功数: {success_count}")
        print(f"解析失敗数: {total_cards_with_text - success_count}")
        print(f"解析率: {rate:.2f}%")
    else:
        print("テキストを持つカードが検出されませんでした。")

    print("-" * 40)
    
    if failed_cards:
        print(f"\n=== 解析失敗カード (サンプル {min(10, len(failed_cards))}件) ===")
        for fc in failed_cards[:10]:
            print(f"ID: {fc['id']} | {fc['name']}")
            print(f"TEXT: {fc['text'][:60]}...")
            print("-" * 20)

if __name__ == "__main__":
    run_test()
