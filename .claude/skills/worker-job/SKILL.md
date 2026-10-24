---
name: worker-job
description: コーディネータから「ジョブ <名前> の番号 k を回して」と頼まれた作業セッション（アリーナのシャード・教材の生成のシャード・訓練＋評価帯・監査のシャード）が使う。実行役 opcg_sim/loop/worker_job.py を nohup で回し、ALL_DONE まで 9 分以内の待ちを繰り返して見張り、結果ブランチへの push と RESULT.json までを確かめる。コードは変えない。
---

# 定型の作業を回す（worker-job）

やることは**ジョブ定義（`opcg_sim/loop/jobs/<名前>.json`）と番号 k を実行役に渡して、終わるまで見張る**だけ。
seed・ブランチ名・コマンドの引数は全部ジョブ定義と実行役が決める（正本 `docs/n_loop_ops.md` §7.42・監査は §7.45）。
**自分でコマンドを組み立てたり、引数を書き換えたりしない。**

## 決まり

- **コードは変えない・PR は作らない・マージしない・コミットにモデル名を入れない**（コミットは実行役がする）。
- **質問で止まらない**。迷った点・おかしな点は最後に `RESULT.json` の notes に書いて（下の「止まったとき」）完走する。
- **完走（`~/worker_job.out` の最後の行が `ALL_DONE`）までターンを終えない**。待ちに入るとコンテナが止まり、回しているプロセスも消える。
- 同じコンテナで他の重い処理を並べない。

## 手順

1. 起動（指示文の 2 行。`<名前>` と `<k>` は指示文のとおり）:

   ```bash
   cd ~ && ([ -d opcg-sim-backend ] || git clone -q https://github.com/gx5gyqe2-art/opcg-sim-backend.git) && cd ~/opcg-sim-backend \
     && git fetch -q origin claude/cpu-spec-improvements-yw91jd && git show FETCH_HEAD:opcg_sim/loop/worker_job.py > ~/worker_job.py
   nohup python ~/worker_job.py --job <名前> --k <k> > ~/worker_job.out 2>&1 &
   ```

   実行役は本線から取り出した 1 ファイル（`~/worker_job.py`）で動く＝作業中にリポジトリのブランチが替わっても本体は変わらない。
   最初に出どころの照合（CLAUDE.md の「N系ループの分散運用」と台帳の節が本線に在ること）をして、無ければ `WORKER_JOB_ERROR` で止まる。
   その前に作業の木（`~/opcg-sim-backend`）を確かめ、壊れたクローン（途中で止まった clone・origin が違う）なら
   `作業の木を取り直す: …` を出して自分で取り直す（何もしない・§7.46）。
   何をするかを先に見たいときは同じ行に `--dry-run` を付ける（git・ビルド・実行をしない）。

2. 見張り（**9 分以内**の待ちを、`ALL_DONE` か `WORKER_JOB_ERROR` か `PUSH_FAILED（最後の` が出るまで繰り返し呼ぶ）:

   ```bash
   timeout 540 bash -c 'until grep -qE "^(ALL_DONE|WORKER_JOB_ERROR|PUSH_FAILED（最後)" ~/worker_job.out; do sleep 30; done'; tail -n 3 ~/worker_job.out
   ```

   生きているかは `ps -eo pid,etime,cmd | grep "[w]orker_job.py"`。本体のログは `~/<ジョブ名>.log`（アリーナ）・`~/n51_w<K>.log`（生成）・
   `~/train_*.log`（訓練）・`~/audit-*.log`（監査）など、`~/worker_job.out` の `$ …` の行に書いてある。区切りごとに `保存 partial: push 済み` が出る。
   **監査**（ジョブの名前が `audit_*`）も回し方は同じ。道具（`defense_audit.py`／`blunder_audit.py`）のサブコマンドを
   定義の順に回し（`run` → `verify` → `summarize` など）、1 つの段が長いので**段の途中でも定義の間隔（既定 20 分）ごとに
   `保存 partial` が出る**。済んだ段は `n1_results/<…>/worker_steps.json` に書かれ、再開すると飛ばす（`run`／`play` の途中なら
   道具が終えた局を飛ばして続きから）。シャードの要約（`summary.json`）はそのシャードの分だけ＝全体の集計はコーディネータがつなぐ。

3. 完了: `ALL_DONE` は push の後に origin の先端が手元と同じことを確かめてから出る。`ALL_DONE` が出たら `git -C ~/opcg-sim-backend log -1 --stat | head` と `RESULT.json` の `status`（`done`）を確かめ、
   結果ブランチ名と RESULT.json の metrics を 3 行ほどで報告してターンを終える。

## 再開

- **コンテナが再起動した・プロセスが消えた**（`ps` に無いのに ALL_DONE が無い）: 手順 1 の 2 行を**そのままもう一度**回す。
  実行役は origin の結果ブランチから続きを取り戻し、終えた区切りを飛ばす（手元にだけ在った途中結果も残す）。
- 同じジョブを 2 本同時に回さない（`ps` で確かめてから起動する）。

## 止まったとき

- `WORKER_JOB_ERROR …`: 実行役が `RESULT.json`（`status: "error"`・notes にエラー）を書いて push を試みている。
  **ネットワークの一時的な失敗**（fetch・clone・pip の接続エラー）なら手順 1 をもう 1 回だけ回す。
  それ以外（`require`／出どころの照合の失敗・ビルドの失敗・本体の Traceback・教材の個数が合わない）は**直さずに**、
  `~/worker_job.out` の最後の 30 行と本体のログの最後の 30 行を報告して止める（コードやジョブ定義を変えない）。
- `PUSH_FAILED（n 回目）`: 実行役が自分で 4 回まで待ってやり直す（何もしない）。`届いていない（origin の … は …）` は
  push が成功で返ったのに origin の結果ブランチの先端が手元のコミットでない型（§7.46）。扱いは同じ。
- `WORKER_JOB_ERROR 今のブランチが …`: 結果ブランチ以外に居る＝実行役の外で誰かがブランチを替えた。直さずに報告して止める。
  `PUSH_FAILED（最後の push が …）` で終わったら、数分おいて手順 1 をもう 1 回だけ回す（冪等＝溜まった分をまとめて押す）。
  それでも通らなければ、その出力と `git -C ~/opcg-sim-backend status | head` を報告して止める。
- 迷った点（遅い・void が多い・ログに警告）は止めずに完走し、報告に書く。RESULT.json の notes に足したいときは、
  ALL_DONE の後に notes だけを書き換えて `git add RESULT.json && git -c user.email=g.x5gyqe2@gmail.com -c user.name=worker commit -m "notes" && git push -u origin <結果ブランチ>` する。
