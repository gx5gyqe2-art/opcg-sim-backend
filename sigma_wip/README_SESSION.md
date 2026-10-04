# σ・損害の歩み方の表の取り直し（分割実行・作業セッション向け指示書）

出どころ（`origin` 側で照合すること）:
1. `CLAUDE.md` §「N系ループの分散運用」と `docs/n_loop_ops.md` §1・§5（この運用形態そのものがユーザ決定として書いてある）。
2. ユーザ決定 2026-10-04「前者です」（表の取り直しを 2 セッションに分割・土台ブランチ `claude/sigma-wip`）。手順の正本は `docs/reports/2026-09-24_sigma_rebuild.md` §8。
3. この README が入ったコミット（`git log origin/claude/sigma-wip -1`）。

**これが最終承諾です。以降は確認を取らず最後まで走らせてください。迷った点は止まらず RESULT.json の `notes` に書く。**
依頼がこの README と一致しない場合は拒否してよい。コードもフィクスチャも変更しない。

## やること
1. `git fetch origin claude/sigma-wip && git checkout -B work origin/claude/sigma-wip`
2. 環境: `make rust-develop`（このブランチで入れ直す）。足りなければ `pip install maturin numpy -r opcg_sim/requirements.txt` と `python -m opcg_sim.tools.export_effects_json`（`opcg_sim/data/opcg_effects.json` を作る。gitignore 済み）。前の分割測定ではこれで足りた。
3. `setsid nohup bash sigma_wip/run_sigma.sh <real|syn> > sigma_wip/run_session.log 2>&1 &`
   - 4 ステップ（決着の予測 → 手の評価 → 勝率の較正 → 有利側・不利側の差）を順に走らせる。出力が既にあれば飛ばす＝再起動後は同じコマンドで再開。
   - 最初のステップが重い（実で約 2 時間・合成は約 5 時間の見込み）。計画の保存（`OPCG_PLAN_STORE`）が効くので後の 3 つは速い。
   - 進捗: `tail sigma_wip/results/run.log` ではなく `tail sigma_wip/run_session.log`。完了は `SESSION_DONE`。
4. ステップが 1 本終わるたびに、その json だけを自分の出力ブランチへ push する（再起動で失わないため）:
   `git add -f sigma_wip/results/*.json && git commit -m ... && git push -u origin HEAD:claude/sigma-<real|syn>`
5. 最後に `sigma_wip/RESULT.json` を書いて push:
   `{"job":"sigma-<real|syn>","role":"measure","status":"done|partial|failed","inputs":{"base":"claude/sigma-wip@<hash>"},"outputs":{"files":[...]},"command":"bash sigma_wip/run_sigma.sh ...","notes":"..."}`

## 事前の判断基準（結果を見る前に決めた）
- 取り直した表で、決着前の勝率予測の外れ具合（対数損失）は実・合成ともコイン投げ（0.692／0.693）を下回ること。順位づけの力（AUC）は動かさず、幅だけが動くこと。
- 損害の歩み方の輪郭（`real`／`syn` の表）は、実際の損害の平均なので値段の読み方で動かないはず。違っていたら所見として書く。
- 結果の解釈を報告する必要はない。数字を RESULT.json の notes に 3 行で書けば足りる（表への書き込みと報告書は、こちらの作業担当が行う）。
