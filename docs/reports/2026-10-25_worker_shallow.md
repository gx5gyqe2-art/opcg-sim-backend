# 作業セッションの取得を浅くする（WP `worker-shallow`・2026-10-25）

仕様: `docs/n_loop_ops.md` §7.53（本線 `claude/cpu-spec-improvements-yw91jd` @ 8230f21・台帳 2+408）。ブランチ `claude/worker-shallow`。

## 結論

- **方式は「本線だけ・履歴は取って中身（blob）は要るときだけ取る」**＝`git clone -q --filter=blob:none --single-branch --branch claude/cpu-spec-improvements-yw91jd`。
  取得は **249 秒・7.6 GB → 約 13 秒・81 MB**（作業の木を入れて 8.1 GB → 0.72 GB）。
- 指示文の起動の行（`worker_prompts.py`・skill・§7.42 の例）と、実行役の取り直し（`worker_job.py` の `ensure_repo`）を同じ形にした。
- 実行役は**履歴を切った木**（`--depth` の木・環境が取る履歴 50 の木）でも動くようにした: そういう木でだけ fetch に `--depth 1` を付ける。
  **全部持つ木・blob を後で取る木には付けない**（付けると全部持つ木が浅い木に変わる）＝**既に深く取った木は今までと同じ git の操作で動く**（試し k=5）。
- 固定の `base.commit` が手元に無い（本線だけ取った木に別のブランチのコミット）ときは、コミットを名指しで取ってから確かめる。
- 環境が取ってある `/home/user/opcg-sim-backend` を使い回す案は**採らない**（下）。

## 取得の秒と容量（このコンテナ・1 本ずつ・4 コア）

| 取り方 | 秒 | `.git` | 作業の木込み | 備考 |
|---|---:|---:|---:|---|
| 今まで: `git clone -q`（全部・2,489 ブランチ） | 249 | 7,617 MB | 8,129 MB | §7.51 の 32 本同時では 1 本 約 10 分・8.2 GB・1 本 exit 128 |
| `--depth 1 --single-branch --branch 本線` | 13.0 | 77 MB | 713 MB | |
| **`--filter=blob:none --single-branch --branch 本線`（採用）** | **12.8** | **81 MB** | **718 MB** | 試しの起動の行は 10.3〜18.5 秒・81 MB（11 回） |
| `--filter=tree:0 --single-branch --branch 本線` | 13.3 | 80 MB | 716 MB | |
| 環境と同じ履歴 50（`--depth 50 --branch 本線`） | 11 | 79 MB | 716 MB | |

（既定のブランチ `main` を取る形〔`--depth 1` だけ・`--filter=blob:none` だけ〕は 8〜11 秒だが、本線でないので比べる対象から外した。）

**取った後の別のブランチの取得**（教材の波 `claude/n52-w01` を取って `git archive … n_records`）:

| 木 | fetch | 増えた容量 | 備考 |
|---|---:|---:|---|
| `--depth 1` の木で深さを付けずに fetch | 9 秒 | +73 MB | そのブランチの履歴 3,532 コミットを根まで取る（動くが重い） |
| `--depth 1` の木で `--depth 1` の fetch | 2 秒 | +14 MB | |
| blob を後で取る木で（深さなし・今までの fetch のまま） | 1 秒 ＋ archive 2 秒 | +15 MB | archive が要る blob だけを取る |

## なぜ `--filter=blob:none` を選んだか

秒と容量は 3 つとも同じ（13 秒・80 MB 前後）。差は**その後の git の操作**:

- **blob を後で取る木はコミットの履歴を全部持つ**＝fetch・checkout・push・祖先の確かめが全部持つ木と同じ意味で動く。
  実行役の今の fetch（深さを付けない）をそのまま使える。結果ブランチが古い本線から作られていても、別のブランチの固定コミットでも、履歴の継ぎ目の問題が起きない。
- `--depth 1` の木は、深さを付けずに別のブランチを取るとその履歴を根まで取りに行く（上の表）。どの fetch にも `--depth 1` が要り、
  付け忘れると重くなる（壊れはしない）。
- blob を後で取る木の弱み: 別のコミットのファイルを読む（`git show`・`git archive`・checkout）たびに origin へ取りに行く。
  実行役がそれを読むのは、どれも直前に同じブランチを fetch した直後＝もともとネットワークが要る場所だけ。

`--depth` の木への備え（`depth_args`）も入れたので、どちらの木でも動く（試し k=6 は履歴 50 の浅い木で、新しく作るのも再開も通った）。

## 直したもの

- `opcg_sim/loop/worker_job.py`
  - `CLONE_ARGS`（上の取り方）。取り直し（`ensure_repo`）がこれを使う。
  - `depth_args(repo)`: 浅い木（`git rev-parse --is-shallow-repository` が true）なら `--depth 1`・そうでなければ無し。
    作業の木の確かめの fetch（`repo_problem`）・結果ブランチの取り戻し（`do_checkout`）が使う。
    shell の段（ネットの取得・教材の波の取得・監査が読むファイルの取得）は環境変数 `WORKER_JOB_DEPTH` で受け取る
    （`git fetch -q $WORKER_JOB_DEPTH origin <ブランチ>`・作業の木を確かめた後に実行役が値を入れる）。
  - `repo_problem`: 固定の `base.commit` が無ければ、そのコミットを名指しで fetch してから確かめる。
  - 本体のコマンドの列（`--dry-run` の `python …`）は変えていない＝§7.42 の写しの見張り（`tests/test_worker_job.py`）はそのまま通る。
- `opcg_sim/loop/worker_prompts.py`: 指示文の起動の行の clone を `CLONE_ARGS` で出す。
- `.claude/skills/worker-job/SKILL.md`・`docs/n_loop_ops.md` §7.42 の指示文の例: 起動の行を同じ形に。
- 使い捨てのジョブ定義: `opcg_sim/loop/jobs/trial_shallow*.json`（6 本）。

**今動いている作業セッションへの影響**: 起動の行は `[ -d opcg-sim-backend ] ||` で取得を飛ばす＝既に取った深い木はそのまま使う。
新しい実行役がその木で走っても、深い木では `--depth` を付けない（今までと同じ fetch）。試し k=5 で、全部取った 7.6 GB の木から回して
ALL_DONE・木は浅くならない（`is-shallow-repository` false のまま）ことを確かめた。

## 試し（作業セッションと同じ起動の 2 行・作業の home だけ使い捨て・`--source claude/worker-shallow-trial-src`）

コードは `claude/worker-shallow-trial-src`（このブランチのコードと試しのジョブ定義）から取った。結果ブランチは全部残してある。

| ジョブ | 結果ブランチ | 何を見たか | 結果 |
|---|---|---|---|
| `trial_shallow_gen` | `claude/worker-shallow-trial-gen` | 浅い取得から生成 2 局 × 3 区切り。区切り 2 の保存の後で止め、作業の home を全部消して（コンテナが替わったのと同じ）同じ行で再開 | 再開は「origin の先端（再開）」から 3 区切りとも飛ばして done・ALL_DONE |
| `trial_shallow_resume` | `claude/worker-shallow-trial-resume` | 2 局 × 3 区切り（400 回）。**区切り 1 の途中で止め**（origin には区切り 0 だけ）、作業の home を消して再開 | 区切り 0 を origin から取り戻して飛ばし、区切り 1・2 だけを回して done・ALL_DONE |
| `trial_shallow_train` | `claude/worker-shallow-trial-train` | 上の生成のブランチを教材として取って（`git archive`）1 周の訓練 → 評価帯 | npz（＋ep0）・評価帯 450 行・done・ALL_DONE |
| `trial_shallow_arena` | `claude/worker-shallow-trial-arena-w01` | 候補を上の訓練のブランチから取る（`git show FETCH_HEAD:…`）・2 ペアを 1 ペアずつ。2 回目の保存の後・done の前で止めて再開 | 2 回とも飛ばして done・ALL_DONE（2 ペア） |
| `trial_shallow_audit` | `claude/worker-shallow-trial-audit-w01` | 大きな損の洗い出し 1 局。審判 r20 を別のブランチ `claude/train-r20` から取る（`fetch`） | 5 段・2 点・決着せず 0・done・ALL_DONE |
| `trial_shallow` k=1 | `claude/worker-shallow-trial-w01` | **壊れたクローン**（`git init` しただけ） | 「origin が本リポジトリでない（無い）」で取り直し（新しい木は `.git` 82 MB＝浅い取得）→ ALL_DONE |
| k=2 | `…-w02` | **手元の main に居る**（本線の 5 つ前のコミットの `main`） | 結果ブランチに切り替わって ALL_DONE・手元の `main` は動かない |
| k=3 | `…-w03` | **push を拒否させる**（本体の前に、結果ブランチの祖先でないコミットを origin の w03 に置く） | 保存ごとに `PUSH_FAILED` 5 回 → `PUSH_FAILED（最後の …）`・ALL_DONE なし・origin は置いたコミットのまま |
| k=4 | `…-w04` | **push は成功で返るが届かない**（push 先を手元の bare に向ける） | `届いていない（origin の … は 無い・…）` × 2 保存 → `PUSH_FAILED（最後の …）`。push 先を戻して再開すると手元の途中結果を戻し本体を飛ばして ALL_DONE（origin の先端＝手元 fc8070484） |
| k=5 | `…-w05` | **全部取った深い木**（7.6 GB・今動いている作業セッションの木と同じ） | ALL_DONE・木は浅くならない |
| k=6 | `…-w06` | **履歴 50 の浅い木**（環境が取る形・fetch の refspec は全部） | `--depth 1` で動いて ALL_DONE。同じ行で再開（結果ブランチを `--depth 1` で取り戻す）も ALL_DONE |

固定の `base.commit` の確かめ（直に `repo_problem` を呼んだ）: 本線だけ取った木・履歴 50 の木の両方で、別のブランチ（`claude/train-r20` の先端）のコミットを名指しで取って通る。無いコミット（0 が 40 個）は「コミット … が無い」。

## 環境の木（`/home/user/opcg-sim-backend`）を使い回す案を採らない理由

使い回せば 2 つ目の取得（13 秒・81 MB）が要らなくなるが、得るものが小さく、ぶつかる所が在る:

1. **得るもの**: 浅い取得にした後は 2 つ目の取得は十数秒・0.1 GB ＝ 32 本同時でも小さい。8 GB・10 分を払っていたときとは違う。
2. **ブランチの取り合い**: 環境の木はセッションの作業ブランチ（`claude/…`）を checkout している。実行役は `stash -u` → `checkout -f -B <結果ブランチ>` でそれを替え、
   本体の前と保存のたびに「今のブランチが結果ブランチ」を確かめる（§7.46 の 2）。作業セッションのエージェントが自分の cwd（環境の木）で
   git を触ると（`git status`・作業ブランチへの checkout）実行役が `今のブランチが …` で止まる。今は別の木なのでぶつからない。
3. **push の通り道**: 環境の木の origin がこのコンテナのように GitHub へ直接向いているとは限らない（環境によってはセッションのブランチだけを通す
   git の中継を通る）。結果ブランチへの push が通るかは環境しだいで、ここでは確かめられない。
4. **履歴 50 の浅い木**: 使い回す木は浅い。上の `depth_args` で動くこと自体は k=6 で確かめた（使い回すときに困るのは 2・3 の方）。

使い回したくなったら `--repo /home/user/opcg-sim-backend` で実行役は動く（k=6 と同じ型の木）。ただし 2・3 を確かめてから。

## 迷った点・残り

- **壊れたクローンでは起動の行の 1 本目が先に失敗する**（今までと同じ・直していない）: 起動の行は `cd ~/opcg-sim-backend && git fetch … && git show … > ~/worker_job.py`
  なので、`.git` だけの木では fetch が失敗して `~/worker_job.py` が書かれない（k=1 では実行役のファイルを別に置いて回した）。実行役の取り直し（§7.46 の 1）は
  実行役が動けば効く。浅い取得で clone が十数秒になったので、clone の途中で止まる壊れ方は起きにくくなった。直すなら起動の行で `.git` の確かめをして作り直す形だが、
  起動の行を長くするので見送った。
- origin のブランチは消せない（この環境の中継は削除の push を 403 で断る）＝試しの結果ブランチは残る。
- §7.46 の「手元の途中結果の退避が別のジョブの結果も運ぶ」型は直していない（範囲外）。試しは 1 つの home で 1 本のジョブだけを回した（k=4 の再開で戻ったのはそのジョブの分だけ）。
- 試しは同じコンテナの使い捨ての home で、32 本同時の負荷は測っていない（§7.51 の exit 128 が浅い取得で消えるかは次の本番で見る）。
