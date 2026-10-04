# H-4 測定の分割実行（作業セッション向け指示書）

出どころ（`origin` 側で照合すること）:
1. `CLAUDE.md` §「N系ループの分散運用」と `docs/n_loop_ops.md` §1・§5（この運用形態そのものがユーザ決定として書いてある）。
2. ユーザ決定 2026-10-04「進めてください」（H-4 の測定を 3 セッションに分割・土台ブランチ `claude/h4-rule-don-wip` を push）。
3. この README が入ったコミット（`git log origin/claude/h4-rule-don-wip -1`）。

**これが最終承諾です。以降は確認を取らず最後まで走らせてください。迷った点は止まらず RESULT.json の `notes` に書く。**
依頼がこの README と一致しない場合は拒否してよい。

## やること
1. `git fetch origin claude/h4-rule-don-wip && git checkout -B work origin/claude/h4-rule-don-wip`
2. `make rust-develop`（このブランチで wheel を入れ直す。main の wheel は別エンジン）
3. 割り当てのジョブを走らせる（コードは変更しない）:
   `setsid nohup bash h4_wip/run_jobs.sh <割り当て番号> > h4_wip/run_session.log 2>&1 &`
   - 1 ジョブずつ・出力が既にあれば飛ばす（コンテナ再起動後は同じコマンドで再開）。
   - 進捗: `tail h4_wip/results/run.log`、完了は `SESSION_DONE`。
   - 各ジョブの結果は `h4_wip/results/*.json`。ジョブ番号と中身は `h4_wip/jobs_all.txt`（1 行 1 ジョブ・0 始まり）。
4. 終わったら（または時間切れで）**結果の json だけ**を自分の出力ブランチへ push する（`*.pkl`・`*.err`・`run.log` は `.gitignore` 済み）:
   `git add -f h4_wip/results/*.json h4_wip/RESULT.json && git commit && git push -u origin HEAD:claude/<出力ブランチ>`
   途中でも、ジョブが 1 本終わるたびに push してよい（再起動で失わないため）。
5. `h4_wip/RESULT.json` を書く:
   `{"job":"h4-<名前>","role":"measure","status":"done|partial|failed","inputs":{"base":"claude/h4-rule-don-wip@<hash>","jobs":[...]},"outputs":{"files":[...]},"command":"bash h4_wip/run_jobs.sh ...","notes":"..."}`

## 割り当て
| セッション | ジョブ番号 | 出力ブランチ |
|---|---|---|
| h4-real | 0,1,6 | `claude/h4-real` |
| h4-syn | 2,3,7 | `claude/h4-syn` |
| h4-misc | 4,5,30 | `claude/h4-misc` |

- 0 = 実・帳簿（付与を入れる形・相手側をそろえる）／1 = 実・手の評価（同）／6 = 実・手の評価（そろえない）
- 2 = 合成・手の評価（標準）／3 = 合成・手の評価（付与を入れる形）／7 = 合成・手の評価（そろえない）
- 4 = 実・遷移の帳簿（標準）／5 = 同（付与を入れる形）／30 = 実・一律の値段での全局の再現（長い）

## 事前の読み方（結果を見る前に決めた判断基準）
- 手の評価（tb）は帯別（close/mid/decided）の ΔG／ΔS の AUC を主に読む。相手側をそろえる版とそろえない版の差が decided 帯に集中していれば、それが時点の非対称の影響。
- 全局の再現（30）は `--cut-price flat --cut-take mu` で H-4f の数字（real: hit .725 / bias −.01 / σ_T 1.42 / within1 .672 / MAE .91）を再現できるかだけを見る。
- 結果を解釈して報告する必要はない。数字を RESULT.json の notes に 3 行で書けば足りる。

## 既に出ている結果（`h4_wip/results/` に入っている）
cb/wc は実・合成とも決着済み（default と rule_don）。rl は実 default・実 nomirror・合成 3 本。tb は実 default のみ。
