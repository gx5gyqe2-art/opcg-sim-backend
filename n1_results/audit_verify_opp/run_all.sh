#!/bin/sh
# WP audit-verify-opp: 前の監査の確かめ済みの点を、新しい確かめ（今のターンの残りを両席とも出荷の設定）と旧い形（手を打った側だけ）で打ち直す
D=$PWD/n1_results/audit_verify_opp
NET=opcg_sim/data/learned/nrel_r13d.npz
# 1. blunder-audit（2026-10-23）の損の候補 12 点
python tests/scripts/blunder_audit.py verify --workers 4 --out n1_results/blunder_audit --verify-file $D/blunder_verify_both.jsonl >> $D/blunder_verify_both.log 2>&1
python tests/scripts/blunder_audit.py verify --workers 4 --out n1_results/blunder_audit --verify-mover-only --verify-file $D/blunder_verify_mover.jsonl > $D/blunder_verify_mover.log 2>&1
# 2. 守りの再監査（2026-10-24 overflow_trigger §3）の確かめ後の見逃し 6 点（甲 6・うち乙 2）
python tests/scripts/defense_audit.py verify --workers 4 --net $NET --points $D/defense_points6.jsonl --out $D --verify-file defense_verify_both.jsonl > $D/defense_verify_both.log 2>&1
python tests/scripts/defense_audit.py verify --workers 4 --net $NET --points $D/defense_points6.jsonl --out $D --verify-mover-only --verify-file defense_verify_mover.jsonl > $D/defense_verify_mover.log 2>&1
# 3. 参考: reaudit-wr（2026-10-24）の blunder の損の候補 3 点
python tests/scripts/blunder_audit.py verify --workers 3 --out n1_results/reaudit_wr/blunder --verify-file $D/reaudit_blunder_verify_both.jsonl > $D/reaudit_blunder_verify_both.log 2>&1
python tests/scripts/blunder_audit.py verify --workers 3 --out n1_results/reaudit_wr/blunder --verify-mover-only --verify-file $D/reaudit_blunder_verify_mover.jsonl > $D/reaudit_blunder_verify_mover.log 2>&1
echo ALL_DONE > $D/ALL_DONE
