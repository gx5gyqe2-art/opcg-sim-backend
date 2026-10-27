# WP `matchup-5` の局のシャード（1〜15）の回し方

仕様: `docs/search_efficiency_plan.md` §7.65（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/matchup_5.py`（`play --shard i/16`）。

## 分け方
- 5 対面 × 120 局 ＝ 600 局。対面ごとに局の番号 0〜119（seed は対面の帯の先頭＋番号・番号 < 60 は視点の側の先攻、≥ 60 は後攻・視点の側は常に p1）。
  帯: 紫エネル対青ロックス 4251000〜・黒ルフィ対赤黒サボ 4252000〜・青ロックス ミラー 4253000〜・緑ミホーク対赤黒サボ 4254000〜・青ロックス対赤黒サボ 4255000〜。
- **同じ対面の先攻の局 j と後攻の局 60＋j を 1 組**にし、組の列（対面の順 × j）の**位置 mod 16 == i** の組がシャード i。
  ＝ どのシャードも**先攻と後攻がちょうど半々**（シャード 0〜11 は 38 局＝19 組、12〜15 は 36 局＝18 組）で、5 対面が 6〜8 局ずつ混ざる。
- シャード 0 はこのセッション（`claude/matchup-5`）が回している。
- 所要の目安: 出荷の設定（1,280 回・世界 4 本）で 4 本並べると 1 局 6〜8 分 → **1 シャード 60〜75 分**。

## 手順（シャード i ＝ 1〜15 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜15）

cd /home/user/opcg-sim-backend
git fetch origin claude/matchup-5
git checkout -B claude/matchup-5-p$I origin/claude/matchup-5

# Rust エンジンの wheel（maturin が無ければ入れる）
pip install -q maturin
make rust-develop

mkdir -p n1_results/matchup_5
OPCG_LOG_SILENT=1 nohup python tests/scripts/matchup_5.py play --workers 4 --shard $I/16 \
  > n1_results/matchup_5/play_shard$I.log 2>&1 &
```

- 1 行目に `PLAY shard=I/16 todo 38 (first 19) ...`（シャード 12〜15 は `todo 36 (first 18)`）が出れば正しく始まっている。
- 1 局終わるごとに `[k/38] mu=... idx=... me_won=...` の行が増える。38（か 36）行で終わり。9 分以内の待ちを繰り返して見張る。
- 再開できる（途中で止まっても同じコマンドで、終えた局は飛ばす）。

## 終わったら

```bash
git add n1_results/matchup_5/games_shard$I.jsonl n1_results/matchup_5/play_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "matchup-5: play shard $I/16"
git push -u origin claude/matchup-5-p$I
```

- push 先は **`claude/matchup-5-p$I`**（`claude/matchup-5` には push しない）。コードは変えない＝ゲートは不要（成果物の jsonl だけ）。
- `RESULT.json` を添えるなら `{"shard": I, "games": 38, "branch": "claude/matchup-5-p$I"}`（12〜15 は 36）を同じコミットに。
