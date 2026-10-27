# WP `matchup-kr` の打ち比べのシャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.63 の手順 3（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/matchup_kr.py`（`judge --shard i/8`）。

## 分け方
- 比べる点は `n1_results/matchup_kr/points_pool.jsonl`（局の記録だけで決まる順位で選んだ点・CPU の手と記事の手）。
  局（`games*.jsonl`）と点の表は `claude/matchup-kr` に push 済み＝**シャードは判定だけ**。
- 表の**番号 mod 8 == i** の点がシャード i。シャード 0 はこのセッション（`claude/matchup-kr`）が回す。
- 1 点 16 通り（公平な配り直し・今のターンの残りは両席とも出荷の設定、その先は審判 r27・160 回）。
- 所要の目安: 4 本並べで 1 点あたり約 15 分 ÷ 4 → **1 シャード（7〜8 点）30〜40 分**。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）

cd /home/user/opcg-sim-backend
git fetch origin claude/matchup-kr
git checkout -B claude/matchup-kr-j$I origin/claude/matchup-kr

pip install -q maturin
make rust-develop

OPCG_LOG_SILENT=1 nohup python tests/scripts/matchup_kr.py judge --k 16 --workers 4 --shard $I/8 \
  > n1_results/matchup_kr/judge_shard$I.log 2>&1 &
```

- 1 行目に `JUDGE shard=I/8 todo 7（か 8） ...` が出れば正しく始まっている（0 ならブランチの取り方が違う）。
- 点ごとに `[k/n] idx=... item=... diff=...` の行が増える。9 分以内の待ちを繰り返して見張る。再開できる。

## 終わったら

```bash
git add n1_results/matchup_kr/judge_points_shard$I.jsonl n1_results/matchup_kr/judge_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "matchup-kr: judge shard $I/8"
git push -u origin claude/matchup-kr-j$I
```

- push 先は **`claude/matchup-kr-j$I`**。コードは変えない＝ゲートは不要。
