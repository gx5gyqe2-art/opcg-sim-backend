import os
import re

# --- File Paths ---
path_matcher = os.path.join("opcg_sim", "src", "core", "effects", "matcher.py")
path_runner = "run_data_driven_test.py"

def fix_matcher():
    """matcher.py: '持ち主'の誤検知と'コストXにする'の誤フィルタリングを修正"""
    print(f"Checking {path_matcher}...")
    if not os.path.exists(path_matcher):
        print(f"❌ File not found: {path_matcher}")
        return

    with open(path_matcher, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Player判定ロジックの修正
    # 「持ち主」が含まれていても、それが「持ち主の手札/デッキ/etc」という移動先指定の場合は対象プレイヤー判定に使わない
    old_player_logic = """    if _nfc(ParserKeyword.EACH_OTHER) in tgt_text: tq.player = Player.ALL
    elif _nfc(ParserKeyword.OWNER) in tgt_text: tq.player = Player.OWNER
    elif _nfc(ParserKeyword.OPPONENT) in tgt_text: tq.player = Player.OPPONENT
    elif _nfc(ParserKeyword.SELF) in tgt_text or _nfc(ParserKeyword.SELF_REF) in tgt_text: tq.player = Player.SELF"""

    new_player_logic = """    if _nfc(ParserKeyword.EACH_OTHER) in tgt_text: tq.player = Player.ALL
    elif _nfc(ParserKeyword.OPPONENT) in tgt_text: tq.player = Player.OPPONENT
    elif _nfc(ParserKeyword.OWNER) in tgt_text: 
        # "持ち主の[領域]" という表現は移動先を示すことが多いため、選択モードとしては無視する
        is_dest = False
        for suffix in ["の手札", "のデッキ", "のライフ", "のトラッシュ"]:
            if _nfc(ParserKeyword.OWNER + suffix) in tgt_text:
                is_dest = True
                break
        
        if not is_dest:
            tq.player = Player.OWNER
        elif _nfc(ParserKeyword.OPPONENT) in tgt_text:
            tq.player = Player.OPPONENT
        else:
            # デフォルトに戻す（通常は自分だが、文脈による）
            tq.player = default_player
            
    elif _nfc(ParserKeyword.SELF) in tgt_text or _nfc(ParserKeyword.SELF_REF) in tgt_text: tq.player = Player.SELF"""

    # 2. Cost判定ロジックの修正
    # 「にする」が続く場合はフィルタとして扱わない
    old_cost_logic_start = "m_c = re.search(_nfc(ParserKeyword.COST + r'[^+\-\d]?(\d+)\D?(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)"
    
    new_cost_logic = """    # Cost
    # [^+\-\d]? ensures we don't match "+2" or "-2" as part of the number prefix
    m_c = re.search(_nfc(ParserKeyword.COST + r'[^+\-\d]?(\d+)\D?(' + ParserKeyword.BELOW + r'|' + ParserKeyword.ABOVE + r')?'), tgt_text)
    if m_c:
        # Extra check: ensure match start isn't preceded by + or -
        start_idx = m_c.start()
        prefix_context = tgt_text[max(0, start_idx-1):start_idx]
        
        # Extra check: ensure match end isn't followed by "にする" (SET_COST action)
        end_idx = m_c.end()
        post_match = tgt_text[end_idx:]
        is_set_action = _nfc("にする") in post_match[:5]

        if prefix_context not in ['+', '-', '\\u2212', '\\u2010'] and not is_set_action:
            val = int(m_c.group(1))
            if m_c.group(2) == _nfc(ParserKeyword.ABOVE): tq.cost_min = val
            else: tq.cost_max = val"""

    # 置換実行
    updated = False
    
    # Player部分を置換（空白やインデントの違いを吸収するため、特徴的な部分で検索）
    if "elif _nfc(ParserKeyword.OWNER) in tgt_text: tq.player = Player.OWNER" in content:
        content = content.replace(old_player_logic, new_player_logic)
        updated = True
        print("✅ Patched matcher.py: Player detection logic")
    
    # Cost部分を置換（m_cの検索行から次のブロック手前までを置換するのは難しいので、m_c定義行を目印にする）
    # ここはブロック全体を置き換える
    cost_block_pattern = r"# Cost[\s\S]+?else: tq.cost_max = val"
    match = re.search(cost_block_pattern, content)
    if match:
        content = content.replace(match.group(0), new_cost_logic)
        updated = True
        print("✅ Patched matcher.py: Cost filter logic")

    if updated:
        with open(path_matcher, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print("⚠️ matcher.py patterns not found (already updated?)")


def fix_runner_battle_flow():
    """run_data_driven_test.py: マニュアルアクション後のバトル進行ロジックを追加"""
    print(f"Checking {path_runner}...")
    if not os.path.exists(path_runner):
        print(f"❌ File not found: {path_runner}")
        return

    with open(path_runner, "r", encoding="utf-8") as f:
        content = f.read()

    # manual_action処理の後に、バトル進行ループを挿入する
    # 目印: Interactionループの前あたり
    
    target_marker = 'gm.resolve_interaction(active_player, payload)'
    
    # 既存のInteractionループの中に、バトルフェーズ進行ロジックを注入したいが、
    # 構造上、Interactionループの外側（manual_actionの直後）で処理する方が安全か、
    # あるいはInteractionループ内で active_interaction が無い場合も回すか。
    
    # 最も簡単なのは、manual_actionブロックの最後に「解決まで回す」コードを入れること
    # しかし manual_action は if "manual_action" in scenario: の中にある
    
    # run_scenario 関数内の manual_action 処理ブロックを探す
    manual_block_end = "ability = DummyAbility()"
    
    extra_logic = """
                # バトル進行自動化: 決着がつくまでブロック/カウンターをパスする（シナリオで指定がない限り）
                # Interactionループで処理させるため、ここでは何もしないが、
                # Interactionループの終了条件や処理を拡張する必要がある。
    """
    
    # 実は run_data_driven_test.py の while gm.active_interaction ループは
    # active_interaction がある間しか回らない。
    # バトル中は active_interaction が（PendingMessageとして）出るはずだが、
    # gm.active_interaction プロパティには入っていない（gm.get_pending_request()で取る設計）。
    
    # 現行の run_data_driven_test.py は gm.active_interaction しか見ていないのが欠陥。
    # gm.get_pending_request() もチェックするように修正が必要。
    
    loop_start = "while gm.active_interaction and loop_limit > 0:"
    new_loop_start = "while (gm.active_interaction or gm.get_pending_request()) and loop_limit > 0:"
    
    if loop_start in content:
        content = content.replace(loop_start, new_loop_start)
        print("✅ Patched run_data_driven_test.py: Loop condition extended")
        
    # さらに、ループ内で pending_request を active_interaction として扱う処理を追加
    req_logic = "req = gm.active_interaction"
    new_req_logic = """
                if not gm.active_interaction:
                    # Pending RequestをInteractionとしてラップする
                    pending = gm.get_pending_request()
                    if pending:
                        # 自動処理可能なフェーズかチェック
                        action_type = pending.get("action")
                        player_id = pending.get("player_id")
                        target_p = gm.p1 if player_id == "P1" else gm.p2
                        
                        # ブロック/カウンターの要求であれば、シナリオ指定がない限りパスする
                        # シナリオのinteractionステップが残っていればそちらに従う
                        
                        if step_idx >= len(interaction_steps):
                            # ステップが尽きている -> 自動パス
                            if action_type == "SELECT_BLOCKER":
                                gm.handle_block(None)
                                continue
                            elif action_type == "SELECT_COUNTER":
                                gm.apply_counter(target_p, None)
                                continue
                        
                        # ステップが残っている場合、active_interactionとして偽装して後続処理に任せる
                        req = {
                            "action_type": action_type,
                            "candidates": [], # 必要なら埋める
                            "can_skip": pending.get("can_skip", False)
                        }
                        # 次の処理へ（reqを使う）
                    else:
                        break # 何もなければ終了
                else:
                    req = gm.active_interaction
    """
    
    # req = gm.active_interaction を置換
    # インデント調整が必要
    pattern = r"                req = gm\.active_interaction"
    if re.search(pattern, content):
        # 置換後のインデントを合わせる
        replacement = new_req_logic.replace("\n", "\n                ")
        # 最初の改行のインデントを除去
        replacement = replacement.replace("                \n", "\n") 
        
        # 正規表現ではなく単純置換で行く（インデントが崩れやすいため注意）
        content = content.replace("                req = gm.active_interaction", new_req_logic.strip().replace("\n", "\n                "))
        print("✅ Patched run_data_driven_test.py: Added battle phase progression logic")
    else:
        print("⚠️ run_data_driven_test.py loop body not matched (check indentation)")

    with open(path_runner, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    print("🚀 Starting Logic Bug Fixes...")
    fix_matcher()
    fix_runner_battle_flow()
    print("✨ Updates completed.")
