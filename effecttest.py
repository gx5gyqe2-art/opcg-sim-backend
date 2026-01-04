import sys
import os
import json
import unicodedata
import traceback

# ---------------------------------------------------------
# Pythonista環境用 パス設定自動化ロジック
# ---------------------------------------------------------
# 現在のスクリプトのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# プロジェクトルートを探す（opcg_simフォルダがある場所を探す）
project_root = current_dir
if os.path.exists(os.path.join(current_dir, "opcg_sim")):
    project_root = current_dir
elif os.path.exists(os.path.join(current_dir, "..", "opcg_sim")):
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

# sys.path に追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project Root: {project_root}")

# ---------------------------------------------------------
# モジュールのインポート
# ---------------------------------------------------------
try:
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.utils.loader import DataCleaner
except ImportError as e:
    print("\n[!] モジュールのインポートに失敗しました。")
    print(f"エラー詳細: {e}")
    print("スクリプトを 'opcg-sim-backend' フォルダ内に置いて実行してください。")
    sys.exit(1)

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
DATA_FILE_NAME = "opcg_cards.json"
DATA_PATHS = [
    os.path.join(project_root, "opcg_sim", "data", DATA_FILE_NAME),
    os.path.join(project_root, "data", DATA_FILE_NAME),
    os.path.join(current_dir, DATA_FILE_NAME)
]

# キーの揺らぎ吸収用（NFKC正規化）
def normalize_key(text: str) -> str:
    if not text: return ""
    return unicodedata.normalize('NFKC', text)

def load_json_file():
    for path in DATA_PATHS:
        if os.path.exists(path):
            print(f"データファイル読み込み: {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[!] JSON読み込みエラー: {e}")
                return None
    print(f"[!] '{DATA_FILE_NAME}' が見つかりませんでした。")
    return None

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def run_test():
    print("\n" + "="*50)
    print(" OPCG Simulator - 全カード解析テスト")
    print("="*50)

    cards = load_json_file()
    if not cards:
        return

    print(f"総データ数: {len(cards)}")
    
    total_text_cards = 0
    success_count = 0
    failed_cards = []

    # 探索するキーの候補（正規化前）
    keys_text = ["effect_text", "テキスト", "Text", "text", "効果(テキスト)"]
    keys_trigger = ["trigger_text", "トリガー", "Trigger", "trigger", "効果(トリガー)"]
    
    # 照合用に正規化
    norm_keys_text = [normalize_key(k) for k in keys_text]
    norm_keys_trigger = [normalize_key(k) for k in keys_trigger]

    for i, card in enumerate(cards):
        # カードデータのキーも全て正規化して扱いやすくする
        c_norm = {normalize_key(k): v for k, v in card.items()}

        # IDと名前の取得（ログ用）
        c_id = c_norm.get(normalize_key("品番")) or c_norm.get("number") or f"Index-{i}"
        c_name = c_norm.get(normalize_key("名前")) or c_norm.get("name") or "Unknown"

        # テキスト抽出
        raw_text = ""
        for k in norm_keys_text:
            if k in c_norm and c_norm[k]:
                raw_text = c_norm[k]
                break
        
        # トリガー抽出
        raw_trigger = ""
        for k in norm_keys_trigger:
            if k in c_norm and c_norm[k]:
                raw_trigger = c_norm[k]
                break

        # クリーニング
        text = DataCleaner.normalize_text(raw_text)
        trigger = DataCleaner.normalize_text(raw_trigger)

        # テキストが何もないカード（バニラ）は対象外
        if not text and not trigger:
            continue

        total_text_cards += 1
        is_success = False
        error_msg = ""

        try:
            # 解析実行
            abilities = []
            if text:
                parser = Effect(text)
                abilities.extend(parser.abilities)
            if trigger:
                parser = Effect(trigger)
                abilities.extend(parser.abilities)

            # 何かしらの解析結果（Ability）が生成されていれば成功とみなす
            if len(abilities) > 0:
                is_success = True
            else:
                error_msg = "Abilityが生成されませんでした (空の結果)"

        except Exception as e:
            is_success = False
            error_msg = str(e)
            # traceback.print_exc() # 詳細デバッグ時はコメントアウト解除

        if is_success:
            success_count += 1
        else:
            failed_cards.append({
                "id": c_id,
                "name": c_name,
                "text": text,
                "error": error_msg
            })

    # 結果レポート
    print("-" * 50)
    print(f"テキストありカード: {total_text_cards} 枚")
    print(f"解析成功: {success_count} 枚")
    print(f"解析失敗: {total_text_cards - success_count} 枚")
    
    if total_text_cards > 0:
        rate = (success_count / total_text_cards) * 100
        print(f"解析成功率: {rate:.2f}%")
    else:
        print("解析対象が見つかりませんでした。")
    print("-" * 50)

    if failed_cards:
        print("\n=== [失敗カード サンプル (最大10件)] ===")
        for fc in failed_cards[:10]:
            print(f"ID: {fc['id']} | {fc['name']}")
            print(f"エラー: {fc['error']}")
            print(f"テキスト: {fc['text'][:60]}...")
            print("-" * 20)

if __name__ == "__main__":
    run_test()
