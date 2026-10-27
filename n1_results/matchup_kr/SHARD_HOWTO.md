# WP `matchup-kr` の局のシャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.63（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/matchup_kr.py`（`play --shard i/8`）。

## 分け方
- 局は 200（局の番号 0〜199・seed 4241000＋番号・番号 < 100 は紫カイドウ先攻、≥ 100 は後攻・紫カイドウは常に p1）。
- 先攻と後攻を交互に並べた列（0, 100, 1, 101, …）の**位置 mod 8 == i** の局がシャード i（各 25 局・先攻 12〜13・後攻 12〜13）。
- シャード 0 はこのセッション（`claude/matchup-kr`）が回している。
- 所要の目安: 出荷の設定（1,280 回・世界 4 本）で 4 本並べると 1 局 6〜8 分 → **1 シャード 40〜50 分**。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）

cd /home/user/opcg-sim-backend
git fetch origin claude/matchup-kr
git checkout -B claude/matchup-kr-p$I origin/claude/matchup-kr

# Rust エンジンの wheel（maturin が無ければ入れる）
pip install -q maturin
make rust-develop

mkdir -p n1_results/matchup_kr
OPCG_LOG_SILENT=1 nohup python tests/scripts/matchup_kr.py play --workers 4 --shard $I/8 \
  > n1_results/matchup_kr/play_shard$I.log 2>&1 &
```

- 1 行目に `PLAY shard=I/8 todo 25 ...` が出れば正しく始まっている。
- 1 局終わるごとに `[k/25] idx=... kaido_won=...` の行が増える。25 行で終わり。9 分以内の待ちを繰り返して見張る。
- 再開できる（途中で止まっても同じコマンドで、終えた局は飛ばす）。

## 終わったら

```bash
git add n1_results/matchup_kr/games_shard$I.jsonl n1_results/matchup_kr/play_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "matchup-kr: play shard $I/8 (25 games)"
git push -u origin claude/matchup-kr-p$I
```

- push 先は **`claude/matchup-kr-p$I`**（`claude/matchup-kr` には push しない）。コードは変えない＝ゲートは不要（成果物の jsonl だけ）。
- `RESULT.json` は不要（コーディネータ側の台帳で受け取る場合は `{"shard": I, "games": 25, "branch": "claude/matchup-kr-p$I"}` を同じコミットに添える）。
