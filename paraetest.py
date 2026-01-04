import sys
import os
import json
import re
from collections import defaultdict
from typing import Optional, List

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if "opcg_sim" not in sys.path:
    sys.path.insert(0, project_root)

try:
    from opcg_sim.src.utils.loader import DataCleaner
    from opcg_sim.src.core.effects.parser import Effect
    from opcg_sim.src.models.enums import ConditionType, CompareOperator, Zone
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- 判定用キーワード定数 ---
KEYWORDS = {
    "CONDITION_INDICATORS": ["場合", "なら", "することで"],
    "TARGET_INDICATORS": ["選び", "対象とし", "キャラを", "枚を", "カードを"]
}

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
    print("Error: opcg_cards.json not found.")
    return []

class AnalysisStats:
    def __init__(self):
        # Condition統計
        self.cond_expected = 0      # キーワードから条件があると推測される数
        self.cond_detected = 0      # 実際にConditionオブジェクトが生成された数
        self.cond_types = defaultdict(int) # ConditionTypeごとの内訳
        self.cond_missed = []       # 解析漏れサンプル

        # Target統計
        self.target_expected = 0    # キーワードから対象選択があると推測される数
        self.target_detected = 0    # 実際にTargetオブジェクトが生成された数
        self.target_zones = defaultdict(int) # 対象Zoneごとの内訳
        self.target_missed = []     # 解析漏れサンプル

    def add_condition_result(self, raw_text, condition):
        # テキストによる期待値判定
        has_indicator = any(k in raw_text for k in KEYWORDS["CONDITION_INDICATORS"])
        
        if has_indicator:
            self.cond_expected += 1
            if condition and condition.type != ConditionType.NONE:
                self.cond_detected += 1
                self.cond_types[condition.type.name] += 1
            else:
                self.cond_missed.append(f"[{raw_text}] -> Missed")
        elif condition and condition.type != ConditionType.NONE:
            # キーワードがないのにConditionが生成された場合(暗黙的な条件など)
            self.cond_types[condition.type.name] += 1

    def add_target_result(self, raw_text, target):
        # ターゲット選択を含まないアクションタイプを除外するロジックが必要だが
        # ここでは簡易的にキーワードベースでチェック
        has_indicator = any(k in raw_text for k in KEYWORDS["TARGET_INDICATORS"])
        
        # 明らかに対象を取らないアクション(ドローなど)を除外したい場合はここでフィルタ
        
        if has_indicator:
            self.target_expected += 1
            if target:
                self.target_detected += 1
                zone_name = target.zone.name if hasattr(target, 'zone') else "UNKNOWN"
                self.target_zones[zone_name] += 1
            else:
                self.target_missed.append(f"[{raw_text}] -> Missed")

def analyze_actions(actions, stats: AnalysisStats):
    for act in actions:
        # Conditionの検証
        stats.add_condition_result(act.raw_text, act.condition)
        
        # Targetの検証
        stats.add_target_result(act.raw_text, act.target)

        # 再帰探索
        if act.then_actions:
            analyze_actions(act.then_actions, stats)

def main():
    cards = load_cards()
    print(f"Loaded {len(cards)} cards. Analyzing Conditions and Targets...")
    
    stats = AnalysisStats()

    for i, card in enumerate(cards):
        raw_text = card.get("効果(テキスト)") or card.get("effect_text") or ""
        raw_trigger = card.get("効果(トリガー)") or card.get("trigger_text") or ""
        
        texts = []
        if raw_text: texts.append(Effect(raw_text)._normalize(raw_text))
        if raw_trigger: texts.append(Effect(raw_trigger)._normalize(raw_trigger))
        
        cid = card.get("number") or f"ID-{i}"
        
        for text in texts:
            if not text: continue
            try:
                abilities = Effect(text).abilities
                for abi in abilities:
                    analyze_actions(abi.actions, stats)
            except Exception:
                pass

    # --- レポート出力 ---
    print("\n" + "="*60)
    print("PARSER COMPONENT ANALYSIS REPORT")
    print("="*60)
    
    # 1. Condition Report
    cond_rate = (stats.cond_detected / stats.cond_expected * 100) if stats.cond_expected else 0
    print(f"\n■ CONDITION Analysis")
    print(f"  Expected (Text-based): {stats.cond_expected}")
    print(f"  Detected (Parsed)    : {stats.cond_detected}")
    print(f"  Coverage Rate        : {cond_rate:.1f}%")
    
    print("  [Type Breakdown]")
    for k, v in sorted(stats.cond_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {k}: {v}")
        
    if stats.cond_missed:
        print(f"  [Top 5 Missed Samples]")
        for s in stats.cond_missed[:5]:
            print(f"    {s}")

    # 2. Target Report
    target_rate = (stats.target_detected / stats.target_expected * 100) if stats.target_expected else 0
    print(f"\n■ TARGET Analysis")
    print(f"  Expected (Text-based): {stats.target_expected}")
    print(f"  Detected (Parsed)    : {stats.target_detected}")
    print(f"  Coverage Rate        : {target_rate:.1f}%")
    
    print("  [Zone Breakdown]")
    for k, v in sorted(stats.target_zones.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {k}: {v}")

    if stats.target_missed:
        print(f"  [Top 5 Missed Samples]")
        for s in stats.target_missed[:5]:
            print(f"    {s}")

if __name__ == "__main__":
    main()
