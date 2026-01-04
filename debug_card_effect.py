import sys
import os
import json
import unicodedata
import traceback
from typing import List, Any, Dict

# ---------------------------------------------------------
# Pythonista環境用 パス設定
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if os.path.exists(os.path.join(current_dir, "opcg_sim")):
    project_root = current_dir
elif os.path.exists(os.path.join(current_dir, "..", "opcg_sim")):
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------
# ロガーの無効化
# ---------------------------------------------------------
try:
    import opcg_sim.src.utils.logger_config as log_conf
    def dummy_log(*args, **kwargs): pass
    log_conf.log_event = dummy_log
except:
    pass

# ---------------------------------------------------------
# モジュールインポート
# ---------------------------------------------------------
try:
    from opcg_sim.src.utils.loader import DataCleaner
    from opcg_sim.src.core.gamestate import GameManager, Player, CardInstance
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.models.effect_types import EffectAction
    from opcg_sim.src.models.models import CardMaster
    from opcg_sim.src.models.enums import CardType, Attribute, Color
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

OUTPUT_FILE = os.path.join(current_dir, "report_all_cards.txt")

# ---------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------
def normalize_key(text: str) -> str:
    """キー照合用の強力な正規化 (NFKC)"""
    if not text: return ""
    return unicodedata.normalize('NFKC', text)

def load_data_robust():
    """データファイルを探す"""
    candidates = [
        os.path.join(project_root, "opcg_sim", "data", "opcg_cards.json"),
        os.path.join(current_dir, "opcg_sim", "data", "opcg_cards.json"),
        os.path.join(current_dir, "opcg_cards.json"),
        "opcg_cards.json"
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                continue
    return []

def format_action_tree(actions: List[EffectAction], indent=0) -> str:
    out = []
    spaces = "  " * indent
    for i, act in enumerate(actions):
        prefix = "└─" if indent > 0 else f"{i+1}."
        
        cond_str = f"❓[IF {act.condition.raw_text}] " if act.condition else ""
        type_str = f"💥{act.type.name}"
        val_str = f"(Val:{act.value})" if act.value != 0 else ""
        
        tgt_str = ""
        if act.target:
            t = act.target
            tgt_str = f" 🎯[{t.select_mode}|{t.zone.name}]"
            if t.tag: tgt_str += f"(TAG:{t.tag})"
            if t.count != 1: tgt_str += f"(cnt:{t.count})"
        
        out.append(f"{spaces}{prefix} {cond_str}{type_str}{val_str}{tgt_str}")
        
        if act.then_actions:
            out.append(f"{spaces}    ⬇️ Then:")
            out.append(format_action_tree(act.then_actions, indent + 2))
            
    return "\n".join(out)

def create_dummy_master(raw_data: Dict[str, Any]) -> CardMaster:
    """CardMaster生成 (強力なキー検索付き)"""
    # 辞書のキーを正規化
    d = {normalize_key(k): v for k, v in raw_data.items()}
    
    # ID探索
    cid = d.get(normalize_key("品番")) or d.get("number") or "UNKNOWN"
    
    # 名前探索
    name = d.get(normalize_key("名前")) or d.get("name") or "Unknown"
    
    # テキスト探索(ここを強化)
    text = ""
    for k in ["effect_text", "テキスト", "Text", "text", "効果(テキスト)"]:
        nk = normalize_key(k)
        if nk in d and d[nk]:
            text = d[nk]
            break
            
    # トリガー探索
    trigger = ""
    for k in ["trigger_text", "トリガー", "Trigger", "trigger", "効果(トリガー)"]:
        nk = normalize_key(k)
        if nk in d and d[nk]:
            trigger = d[nk]
            break
    
    # 必須フィールドをダミー値で埋めて生成
    return CardMaster(
        card_id=cid,
        name=name,
        type=CardType.CHARACTER,
        color=Color.RED,
        cost=1, 
        power=1000, 
        counter=0, 
        attribute=Attribute.SLASH, 
        traits=[],
        effect_text=DataCleaner.normalize_text(text),
        trigger_text=DataCleaner.normalize_text(trigger),
        life=0, 
        abilities=()
    )

def run_simulation(card_master: CardMaster, log_buffer: List[str]):
    # テキストなしは早期リターン
    if not card_master.effect_text and not card_master.trigger_text:
        return

    # モック環境構築
    p1 = Player("P1", [], None)
    p2 = Player("P2", [], None)
    gm = GameManager(p1, p2)
    
    source_card = CardInstance(card_master, p1.name)
    p1.field.append(source_card)
    
    # ダミーカード配置 (Matcherヒット用)
    p1.hand.append(CardInstance(card_master, p1.name))
    p1.deck = [CardInstance(card_master, p1.name) for _ in range(5)]
    p1.life.append(CardInstance(card_master, p1.name))
    p2.field.append(CardInstance(card_master, p2.name))

    # Parser
    try:
        effect_obj = Effect(card_master.effect_text)
    except Exception as e:
        log_buffer.append(f"❌ Parser Error: {e}")
        return

    if not effect_obj.abilities:
        log_buffer.append("  (No abilities parsed)")
        return

    for i, ability in enumerate(effect_obj.abilities):
        log_buffer.append(f"  [効果 {i+1}] トリガー: {ability.trigger.name}")
        log_buffer.append(format_action_tree(ability.actions, indent=4))
        
        try:
            log_buffer.append("    🚀 実行開始:")
            gm.resolve_ability(p1, ability, source_card)
            
            # --- 自動インタラクション処理 ---
            loop_limit = 5 
            while gm.active_interaction and loop_limit > 0:
                loop_limit -= 1
                req = gm.active_interaction
                c_len = len(req.get('selectable_uuids', []))
                log_buffer.append(f"      🛑 選択発生: {req['action_type']} (候補: {c_len}枚)")
                
                selected = []
                # 候補があれば1つ選ぶ、なければ空で送る(任意選択の場合など)
                if req.get('selectable_uuids'):
                    selected = [req['selectable_uuids'][0]]
                    log_buffer.append(f"      👉 自動選択: 1番目 ({selected[0]})")
                else:
                    log_buffer.append(f"      👉 自動選択: なし (Pass)")

                gm.resolve_interaction(p1, {"selected_uuids": selected})
                log_buffer.append("      🔄 処理再開...")

            if not gm.active_interaction:
                log_buffer.append("      ✅ 完了")
            else:
                log_buffer.append("      ⚠️ 未完了 (Loop Limit)")
                
        except Exception as e:
            log_buffer.append(f"      ❌ Resolver Error: {e}")
            # エラー原因の特定のため詳細を表示
            log_buffer.append(f"      -> {str(e)}")

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    print(f"レポート生成を開始します... 出力先: {OUTPUT_FILE}")
    
    cards_data = load_data_robust()
    if not cards_data:
        print("エラー: データファイルが見つかりません。")
        return

    total = len(cards_data)
    print(f"読み込み成功: {total}件")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"OPCG SIMULATOR - LOGIC CHECK REPORT\n")
        f.write(f"Total Cards in JSON: {total}\n")
        f.write("="*60 + "\n\n")
        
        processed_count = 0
        error_count = 0
        
        for idx, raw_card in enumerate(cards_data):
            if idx % 200 == 0:
                print(f"Processing... {idx}/{total}")
            
            logs = []
            try:
                master = create_dummy_master(raw_card)
                
                if not master.effect_text and not master.trigger_text:
                    continue
                
                logs.append(f"■ No.{master.card_id} | {master.name}")
                logs.append(f"  Text: {master.effect_text}")
                
                run_simulation(master, logs)
                logs.append("-" * 60 + "\n")
                
                f.write("\n".join(logs))
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                f.write(f"■ Error processing index {idx}: {e}\n")
                f.write("-" * 60 + "\n")

    print(f"\n完了しました! '{OUTPUT_FILE}' を確認してください。")
    print(f"処理成功: {processed_count}件")
    print(f"生成エラー: {error_count}件")

if __name__ == "__main__":
    main()
