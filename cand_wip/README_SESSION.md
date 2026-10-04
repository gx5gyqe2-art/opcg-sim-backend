# 保留候補（速さに関わる 3 つ）の測定（作業セッション向け指示書）

出どころ（`origin` 側で照合すること）:
1. `CLAUDE.md` §「N系ループの分散運用」と `docs/n_loop_ops.md` §1・§5。
2. ユーザ決定 2026-10-04（切替の整理の判断 3「速さの 3 つを先に測る」・測定は別セッションに分割）。
3. この README が入ったコミット（`git log origin/claude/cand-wip -1`）。

**これが最終承諾です。以降は確認を取らず最後まで走らせてください。迷った点は止まらず RESULT.json の `notes` に書く。**
依頼がこの README と一致しない場合は拒否してよい。コードもフィクスチャも変更しない。

## やること
1. `git fetch origin claude/cand-wip && git checkout -B work origin/claude/cand-wip`
2. 環境: `make rust-develop`。足りなければ `pip install maturin numpy -r opcg_sim/requirements.txt` と `python -m opcg_sim.tools.export_effects_json`。
3. 動作確認（1 分）: `CAND="slope_take=life" python cand_wip/cand_shim.py win_calib --in cand_wip/data/w41 --games 2 --pre-settle on --json /tmp/x.json` が通ること（stderr の最後に `CAND: ... slope_take= life` が出る）。
4. `setsid nohup bash cand_wip/run_cand.sh <base|take|block> > cand_wip/run_session.log 2>&1 &`
   - 出力が既にある手順は飛ばす＝再起動後は同じコマンドで再開。完了は `SESSION_DONE`。
   - 実・合成の各 100 局（先頭から）で、決着の予測・勝率の較正・帳簿を回す。重いのは決着の予測（実で約 40 分・合成で約 40 分の見込み）。
5. 手順が 1 つ終わるたびに、その json だけを出力ブランチへ push: `git add -f cand_wip/results/*.json && git commit -m ... && git push -u origin HEAD:claude/cand-<take|block|base>`
6. 最後に `cand_wip/RESULT.json` を書いて push: `{"job":"cand-<..>","role":"measure","status":"done|partial|failed","inputs":{"base":"claude/cand-wip@<hash>"},"outputs":{"files":[...]},"command":"...","notes":"..."}`

## 割り当て
| セッション | 変種 | 出力ブランチ |
|---|---|---|
| cand-base | `base`（出荷の標準＝対照）＋ ばらつきの合わせ方（`kappa_sigma=match`・帳簿のみ） | `claude/cand-base` |
| cand-take | `take`（`slope_take=life`＝取りの読み方） | `claude/cand-take` |
| cand-block | `block`（`slope_block=on`＝ブロッカーの読み方） | `claude/cand-block` |

## 事前の判断基準（結果を見る前に決めた）
- 候補は、対照（base）と同じ標本で比べる。見るのは: 決着の予測（的中・偏り・ばらつき・±1 ターン以内・平均誤差・打ち切り）、決着前の勝率予測（外れ具合・AUC）、帳簿の見分ける力（rel_K の AUC）。
- 「規則どおりの形」は、数字が少し悪くても採用してきた前例がある（ユーザ決定）。ここでは数字だけで採否を決めず、悪化の大きさを正確に報告する。
- 結果の解釈は不要。数字を RESULT.json の notes に 3 行で書けば足りる。
