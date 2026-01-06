import os

# --- ファイルパス ---
path_gamestate = os.path.join("opcg_sim", "src", "core", "gamestate.py")
path_resolver = os.path.join("opcg_sim", "src", "core", "effects", "resolver.py")
path_parser = os.path.join("opcg_sim", "src", "core", "effects", "parser.py")

def apply_fix(path, start_marker, end_marker, new_content):
    """
    start_marker から end_marker の手前までを new_content に置き換える
    """
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    start_idx = content.find(start_marker)
    if start_idx == -1:
        # すでに修正済みかもしれないので、主要キーワードをチェック
        if new_content.strip()[:20] in content: # 簡易チェック
             print(f"ℹ️ Code seems already updated in {os.path.basename(path)}")
        else:
             print(f"⚠️ Start marker not found in {os.path.basename(path)}")
             print(f"   Marker: '{start_marker.strip()}'")
        return

    # end_marker が None の場合は、start_marker の行だけ置換（parser用）
    if end_marker is None:
        # 行全体を置換
        end_idx = content.find("\n", start_idx)
        if end_idx == -1: end_idx = len(content)
        
        # インデントを維持するために元の行を取得
        original_line = content[start_idx:end_idx]
        indent = original_line[:len(original_line) - len(original_line.lstrip())]
        
        # 置換
        new_full_content = content[:start_idx] + new_content + content[end_idx:]
    else:
        # ブロック置換
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            print(f"⚠️ End marker not found in {os.path.basename(path)}")
            return
        
        # 置換実行
        new_full_content = content[:start_idx] + new_content + content[end_idx:]

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_full_content)
    print(f"✅ Fixed {os.path.basename(path)}")


# 1. GAMESTATE FIX (Double Attack & Banish)
gs_start = "def resolve_attack(self):"
gs_end = "def check_victory(self):" # 次のメソッド
gs_new = """def resolve_attack(self):
        if not self.active_battle:
            return

        attacker = self.active_battle["attacker"]
        target = self.active_battle["target"]
        attacker_owner = self.active_battle["attacker_owner"]
        target_owner = self.active_battle["target_owner"]
        counter_buff = self.active_battle.get("counter_buff", 0)

        is_my_turn = (attacker_owner == self.turn_player)
        is_target_turn = (target_owner == self.turn_player)
        
        attacker_pwr = attacker.get_power(is_my_turn)
        target_pwr = target.get_power(is_target_turn) + counter_buff
        
        log_event("DEBUG", "game.resolve_attack_pre", f"Attacker: {attacker.master.name}({attacker_pwr}) vs Target: {target.master.name}({target_pwr})", player=attacker_owner.name if attacker_owner else "system")
        
        if target == target_owner.leader:
            if attacker_pwr >= target_pwr:
                # ダブルアタック & バニッシュ判定
                damage_amount = 2 if "ダブルアタック" in attacker.current_keywords else 1
                is_banish = "バニッシュ" in attacker.current_keywords

                log_event("INFO", "game.damage_step", f"Dealing {damage_amount} damage (Banish: {is_banish})", player=attacker_owner.name)

                for _ in range(damage_amount):
                    if target_owner.life:
                        life_card = target_owner.life.pop(0)
                        dest_zone = Zone.TRASH if is_banish else Zone.HAND
                        self.move_card(life_card, dest_zone, target_owner)
                        log_event("INFO", "game.damage_life", f"{target_owner.name} takes damage to {dest_zone.name}", player=target_owner.name)
                    else:
                        self.winner = attacker_owner.name
                        log_event("INFO", "game.victory", f"{attacker_owner.name} wins the game", player=attacker_owner.name)
                        break
        else:
            if attacker_pwr >= target_pwr:
                self.move_card(target, Zone.TRASH, target_owner)
                log_event("INFO", "game.unit_ko", f"{target.master.name} was KO'd", player=target_owner.name)
        
        target.reset_turn_status()
        self.active_battle = None
        self.phase = Phase.MAIN
        self.check_victory()

    """

# 2. RESOLVER FIX (Set Cost)
res_start = "elif action.type == ActionType.SET_COST:"
res_end = "elif action.type == ActionType.SHUFFLE:" # 次のブロック
res_new = """elif action.type == ActionType.SET_COST:
        for t in targets:
            diff = action.value - t.current_cost
            t.cost_buff += diff
            log_event("INFO", "effect.set_cost", f"Set {t.master.name} cost to {action.value}", player=player.name)

    """

# 3. PARSER FIX (Bounce Target)
# 特定のif文の行だけを置き換える
par_start = "if act_type in [ActionType.KO, ActionType.DEAL_DAMAGE, ActionType.REST, ActionType.ATTACK_DISABLE, ActionType.FREEZE]:"
par_new = "if act_type in [ActionType.KO, ActionType.DEAL_DAMAGE, ActionType.REST, ActionType.ATTACK_DISABLE, ActionType.FREEZE, ActionType.MOVE_TO_HAND]:"


if __name__ == "__main__":
    print("🚀 Starting FINAL Force Fix...")
    apply_fix(path_gamestate, gs_start, gs_end, gs_new)
    apply_fix(path_resolver, res_start, res_end, res_new)
    apply_fix(path_parser, par_start, None, par_new)
    print("✨ Finished. Please run the test again.")
