# WP `offdist-value` の「窓を閉じた後の盤面」の行を作るシャード（1〜7）の回し方

仕様: `docs/search_efficiency_plan.md` §7.69 の検証 3（本線 `claude/cpu-spec-improvements-yw91jd`）。道具: `tests/scripts/offdist_value.py`（`gen --shard i/8`）。
エンジン・ネットは変えない（自己対戦を打って行を足すだけ）＝**ゲートは不要**。コードと `RESULT.json` は変えない。

## 中身
- r27 の自己対戦（波 60 と同じ設定・実デッキ・1,280 回・世界 1 本）を **1 シャード 450 局**打ち、相手の手番の窓（自分が応手を決める）で
  各候補を当てて窓を閉じた盤面を、自分の視点・その局の勝敗で「価値だけの行」として書く（1 局あたり約 75 行）。
- seed は `5100000 + i × 10000 + 0〜449`（既存の波と重ならない帯）。
- 所要の目安: 4 本並べで 1 局 約 10 秒 → **1 シャード 20〜30 分**。出力は約 30MB。

## 手順（シャード i ＝ 1〜7 のどれか。下の `I` を置き換える）

```bash
I=1   # ← シャード番号（1〜7）
cd /home/user/opcg-sim-backend
git fetch origin claude/offdist-value
git checkout -B claude/offdist-value-g$I origin/claude/offdist-value
pip install -q maturin
make rust-develop
OPCG_LOG_SILENT=1 nohup python tests/scripts/offdist_value.py gen --games 450 --workers 4 --shard $I/8 \
  --gen-out n1_results/offdist_value/fold_rows > n1_results/offdist_value/gen_shard$I.log 2>&1 &
```

- 1 行目に `GEN shard=I/8 todo 450` が出れば正しく始まっている。局ごとに `[k/450] seed=... rows=...` が増える。9 分以内の待ちを繰り返して見張る。
  止まっても同じコマンドで再開できる（`progress.jsonl` に書いた局は飛ばす）。最後に `GEN done {...}` が出れば終わり。

## 終わったら

```bash
git add n1_results/offdist_value/fold_rows/fold_s$I n1_results/offdist_value/gen_shard$I.log
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "offdist-value: fold rows shard $I/8"
git push -u origin claude/offdist-value-g$I
```

- push 先は **`claude/offdist-value-g$I`**。
