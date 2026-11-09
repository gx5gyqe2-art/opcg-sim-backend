# WP `matchup-5` の打ち比べのシャード（1〜17）の回し方

仕様: `docs/search_efficiency_plan.md` §7.65 の手順 3（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/matchup_5.py`（`judge --shard i/18`）。

## 分け方
- 比べる点は `n1_results/matchup_5/points_pool.jsonl`（6 対面それぞれ最大 40 点・項目ごとに偏らないよう局の記録だけで決まる順位で選んだ点・CPU の手と記事の手。勝負の決まった局面〔評価 ±0.9 超〕は除いた）。
  局（`games_shard*.jsonl`）と点の表は `claude/matchup-5` に push 済み＝**シャードは判定だけ**。
- 表の**番号 mod 18 == i** の点がシャード i。シャード 0 はこのセッション（`claude/matchup-5`）が回す。
- 1 点 16 通り（公平な配り直し・今のターンの残りは両席とも出荷の設定、その先は審判 r27・160 回）。
- 所要の目安: 4 本並べで 1 点あたり約 15 分 ÷ 4 → **1 シャード（点の数は 1 行目に出る）40〜60 分**。

## 手順（シャード i ＝ 1〜17 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜17）

cd /home/user/opcg-sim-backend
git fetch origin claude/matchup-5
git checkout -B claude/matchup-5-j$I origin/claude/matchup-5

pip install -q maturin
make rust-develop

OPCG_LOG_SILENT=1 nohup python tests/scripts/matchup_5.py judge --k 16 --workers 4 --shard $I/18 \
  > n1_results/matchup_5/judge_shard$I.log 2>&1 &
```

- 1 行目に `JUDGE shard=I/18 todo N ...`（N は 0 より大きい）が出れば正しく始まっている（0 ならブランチの取り方が違う）。
- 点ごとに `[k/N] mu=... idx=... item=... diff=...` の行が増える。9 分以内の待ちを繰り返して見張る。再開できる。

## 終わったら

```bash
git add n1_results/matchup_5/judge_points_shard$I.jsonl n1_results/matchup_5/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "matchup-5: judge shard $I/18"
git push -u origin claude/matchup-5-j$I
```

- push 先は **`claude/matchup-5-j$I`**。コードは変えない＝ゲートは不要。
