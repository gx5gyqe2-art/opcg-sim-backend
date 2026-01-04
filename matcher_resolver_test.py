import sys
import os
import json
from typing import List, Dict, Any

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if "opcg_sim" not in sys.path:
    sys.path.insert(0, current_dir)

# --- ログキャプチャ用の準備 ---
captured_warnings = []

def capture_log_event(level_key, action, msg, **kwargs):
    """
    Matcherから送られてくる 'matcher.no_target' ログだけを捕まえる
    """
    if action == "matcher.no_target":
        payload = kwargs.get("payload", {})
        query = payload.get("query_raw", "unknown")
        captured_warnings.append(f"Query[{query}]")

# ロガーの設定を上書き
try:
    import opcg_sim.src.utils.logger_config as log_conf
    log_conf.log_event = capture_log_event
except ImportError:
    pass

try:
    from opcg_sim.src.core.gamestate import GameManager, Player, CardInstance
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.models.models import CardMaster
    from opcg_sim.src.models.enums import CardType, Attribute, Color
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

DATA_PATH = os.path.join(current_dir, "opcg_sim", "data", "opcg_cards.json")
REPORT_FILE = "report_full_execution.txt"

def load_cards():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    local_path = "opcg_cards.json"
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def create_dummy_target(owner_name: str, name: str, cost: int, power: int, traits: List[str], color: Color = Color.RED):
    master = CardMaster(
        card_id=f"DUMMY-{name}", name=name, type=CardType.CHARACTER, color=color,
        cost=cost, power=power, counter=1000, attribute=Attribute.SLASH,
        traits=traits, effect_text="", trigger_text="", life=0
    )
    return CardInstance(master, owner_name)

def setup_rich_board(gm: GameManager):
    """
    あらゆる効果の対象になりやすいように、多様なカードを配置する(強化版 v1.5)
    """
    # ★網羅的な特徴リスト
    all_traits = [
        "麦わらの一味", "赤髪海賊団", "白ひげ海賊団", "黒ひげ海賊団", "百獣海賊団", "ビッグ・マム海賊団",
        "ドンキホーテ海賊団", "ハートの海賊団", "キッド海賊団", "ファイアタンク海賊団", "フォクシー海賊団",
        "スリラーバーク海賊団", "九蛇海賊団", "クリーク海賊団", "クロネコ海賊団", "アーロン一味", 
        "太陽の海賊団", "新魚人海賊団", "バロックワークス", "クロスギルド", "革命軍", "海軍", 
        "世界政府", "インペルダウン", "CP", "王下七武海", "超新星", "四皇", "元海軍", "SWORD", 
        "五老星", "天竜人", "魚人族", "人魚族", "ミンク族", "巨人族", "小人族", "トンタッタ族",
        "手長族", "足長族", "ワノ国", "アラバスタ王国", "ドレスローザ", "空島", "シャンドラの戦士", 
        "エッグヘッド", "パンクハザード", "W7", "GC", "東の海", "西の海", "南の海", "北の海",
        "ジェルマ66", "ヴィンスモーク家", "アマゾン・リリー", "ドラム王国", "プロデンス王国", 
        "聖地マリージョア", "FILM", "ODYSSEY", "動物", "SMILE", "黒炭家", "赤鞘九人男", 
        "ホーミーズ", "音楽", "獄卒獣", "科学者", "B・W"
    ]
    
    colors = [Color.RED, Color.GREEN, Color.BLUE, Color.PURPLE, Color.BLACK, Color.YELLOW]

    # --- P1 (自分) のリーダー設定(★Step 1.5 修正箇所★) ---
    # リーダーカードを明示的に作成して配置する
    p1_leader_master = CardMaster(
        card_id="LEADER-001", name="多機能リーダー", type=CardType.LEADER, color=Color.RED,
        cost=5, power=5000, counter=0, attribute=Attribute.SLASH,
        traits=list(all_traits), # 全特徴を持たせる
        effect_text="", trigger_text="", life=5
    )
    gm.p1.leader = CardInstance(p1_leader_master, "P1")
    
    # 念のためP2(相手)にもリーダーを置いておく
    p2_leader_master = CardMaster(
        card_id="LEADER-002", name="相手リーダー", type=CardType.LEADER, color=Color.BLUE,
        cost=5, power=5000, counter=0, attribute=Attribute.STRIKE,
        traits=list(all_traits),
        effect_text="", trigger_text="", life=5
    )
    gm.p2.leader = CardInstance(p2_leader_master, "P2")

    # --- P2 (相手) の場に特殊なカードを配置 ---
    
    # 1. コスト0, パワー0の特殊個体
    c_zero = create_dummy_target("P2", "ZeroSpec", cost=0, power=0, traits=["海軍", "一般"])
    gm.p2.field.append(c_zero)

    # 2. 主要キャラ名を持つダミー(名称指定用)
    key_names = [
        "モンキー・D・ルフィ", "ポートガス・D・エース", "サボ", 
        "トラファルガー・ロー", "ユースタス・キッド", "キラー", "ベポ", 
        "ナミ", "サンジ", "ロロノア・ゾロ", "トニートニー・チョッパー",
        "プロメテウス", "ゼウス", "ヘラ", "カイドウ", "シャーロット・リンリン"
    ]
    for i, name in enumerate(key_names):
        c = create_dummy_target("P2", name, cost=3 + (i % 5), power=5000, traits=["超新星", "麦わらの一味", "四皇"])
        if i % 2 == 0: c.is_rest = True
        gm.p2.field.append(c)

    # 3. 汎用ダミー(コスト1〜10、特徴を分散)
    for i in range(1, 11): 
        start = (i * 5) % len(all_traits)
        end = start + 8 
        traits_subset = []
        if end > len(all_traits):
            traits_subset = all_traits[start:] + all_traits[:end-len(all_traits)]
        else:
            traits_subset = all_traits[start:end]
            
        c = create_dummy_target("P2", f"Enemy_C{i}", cost=i, power=i*1000, traits=traits_subset, color=colors[i%6])
        if i % 2 != 0: c.is_rest = True
        gm.p2.field.append(c)

    # --- P1 (自分) の場・手札・トラッシュ・ライフも強化 ---
    
    # P1場
    for i in range(1, 6):
        c = create_dummy_target("P1", f"Ally_C{i}", cost=i, power=i*1000, traits=all_traits[i:i+5], color=colors[i%6])
        gm.p1.field.append(c)
    
    # P1手札
    for name in ["モンキー・D・ルフィ", "トラファルガー・ロー", "パシフィスタ", "人造悪魔の実SMILE"]:
        c = create_dummy_target("P1", name, cost=3, power=3000, traits=all_traits)
        gm.p1.hand.append(c)
        
    # P1トラッシュ
    gm.p1.trash.append(create_dummy_target("P1", "TrashAce", 5, 6000, ["白ひげ海賊団", "スペード海賊団"]))
    gm.p1.trash.append(create_dummy_target("P1", "TrashPunk", 2, 3000, ["パンクハザード", "科学者"]))

    # P1ライフ
    gm.p1.life.append(create_dummy_target("P1", "LifeCard", 4, 5000, ["トリガー持ち"]))

    # ドン!!の設定
    for _ in range(5):
        d = opcg_sim.src.models.models.DonInstance(owner_id="P1")
        gm.p1.don_active.append(d)
    for _ in range(5):
        d = opcg_sim.src.models.models.DonInstance(owner_id="P1")
        d.is_rest = True
        gm.p1.don_rested.append(d)

    # 相手の手札
    gm.p2.hand.append(create_dummy_target("P2", "EnemyHand", 3, 3000, []))


# --- 依存関係解決 ---
import opcg_sim.src.models.models 

def run_test_for_card(card_data: Dict[str, Any]) -> str:
    captured_warnings.clear()

    text = card_data.get("effect_text") or card_data.get("効果(テキスト)") or ""
    if not text or text == "-": return "SKIP (No Text)"
    
    cid = card_data.get("card_id") or card_data.get("品番")
    name = card_data.get("name") or card_data.get("名前")

    gm = GameManager(Player("P1", [], None), Player("P2", [], None))
    setup_rich_board(gm)
    
    # 実行するカード自身
    master = CardMaster(
        card_id=cid, name=name, type=CardType.CHARACTER, color=Color.RED,
        cost=5, power=5000, counter=1000, attribute=Attribute.SLASH,
        traits=[], effect_text=text, trigger_text="", life=0
    )
    source_card = CardInstance(master, "P1")
    gm.p1.field.append(source_card)

    try:
        effect_obj = Effect(text)
    except:
        return "ERROR (Parse Failed)"

    if not effect_obj.abilities:
        return "SKIP (No Abilities Parsed)"

    ability = effect_obj.abilities[0]
    
    log_res = []
    try:
        gm.resolve_ability(gm.p1, ability, source_card)

        if gm.active_interaction:
            req = gm.active_interaction
            candidates = req.get("candidates", [])
            log_res.append(f"🟢 INTERACTION: {req['action_type']} (候補: {len(candidates)}枚)")
            if candidates:
                gm.resolve_interaction(gm.p1, {"selected_uuids": [candidates[0].uuid]})
                log_res.append("-> Auto-Selected")
            else:
                gm.resolve_interaction(gm.p1, {})
                log_res.append("-> Auto-Pass")
        else:
            log_res.append("⚪ NO_INTERACTION")

        if captured_warnings:
             log_res.append(f"⚠️ TARGET_NOT_FOUND: {', '.join(captured_warnings)}")

    except Exception as e:
        return f"🔴 RUNTIME ERROR: {str(e)}"

    return " | ".join(log_res)

def main():
    cards = load_cards()
    print(f"Loaded {len(cards)} cards. Starting Full Scenario Test (Step 1.5 Leader Fix)...")
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        for i, c in enumerate(cards):
            res = run_test_for_card(c)
            
            cid = c.get("card_id") or c.get("品番") or "?"
            name = c.get("name") or c.get("名前") or "?"
            line = f"[{i+1}] {cid} {name}: {res}"
            
            if i % 100 == 0: print(f"Processing... {i}/{len(cards)}")
            f.write(line + "\n")

    print(f"\nFinished! Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
