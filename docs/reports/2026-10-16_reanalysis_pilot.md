# 再分析で π の教材を作り直す・第 1 段「器とパイロット」（2026-10-16）

起点: `claude/theory-map` @ 7e86c5e3。
出発点: ユーザ決定 2026-10-16「(a) 再分析で教材を作り直す」（司令塔経由・`docs/cpu_theory_gap.md` 台帳 2+232）。
理論の診断フェーズは区切った。理論は π の教材に使わない。

先例 `rs-q-pi`（2026-09-10・`2026-09-10_q_pi_judgment.md`）は、同じ 160 sims の記録で教師の式だけを π' に替え、r4 と同じ強さ（0.5182・不採用）だった。
**今回の新しい要素は探索の深さ（4 倍）と点の選び方で、式ではない**（訓練は `n_rel_train.py --pi-teacher q_improved` のまま）。

読み方: **【実行】**は測った数、**【コード】**はコードから読んだこと、**【推測】**は読み。**§1 は数を 1 つも見る前に書いて push した。**

## 0. まとめ

（§2 以降を測った後に書く）

---

## 1. 器・点・予告（数を見る前に固定）

### 1.1 器 `opcg_sim/loop/reanalyze.py`（生成の仲間・`theory_map.py playdeep` とは独立）

**入力**: 記録の dir（dump v4／enc 13〜14 の npz・`meta_n_record.json`・`meta_games.json`）・`--mult`（既定 4）・`--shard i/n`・`--games`（先頭から何局）・`--per-game`（既定 12）・`--out`・`--net`（meta のネットの置き場が手元に無いときの上書き）。

**再生**: 記録と同じ seed を、`meta_n_record.json` の設定を全部使って打ち直す。
- 使う設定: sims・dirichlet_eps・temp_turns・`search`（worlds・select_rule・q_min_frac・root_prior_temp）・`eps_play`／`eps_hold`／`eps_target`・decks。
- ε の差し替えは記録と同じ `record_gen._Recorder`（観測とその後の `swap`）で行う。`playdeep` は ε を渡していないので、訓練の波（ε 3%）では分岐する。
- main の各行（候補 2 つ以上）で、記録の `sig` と候補の数を照合する。ずれたら、その局は**その点から先を捨てて** `diverged` に数える（`playdeep` と同じ）。

**選んだ点の深い探索**: `game.fork().decide(...)` を、sims × mult・雑音なし（Dirichlet 0）・温度なし・ε なし・同じ worlds と同じ探索の設定で回す。
- 記録の候補の並び（`pol` の順＝再生の `out["groups"]` の順）に、束ごとの訪問数 n・値 q と、根の価値 v0（訪問で重み付けた q の平均・`record_gen` と同じ式）を取る。
- 合法手の集合が違う点、または記録の束が深い探索の束に無い点は `unmapped` に数えて落とす。

**出力**: `n_rel_train.py --in` にそのまま渡せる dump。
- 再分析した main 行だけを、記録の全列をコピーして書く。
- **`pol_n`／`pol_q`／`pol_v0` だけを深い探索の値で差し替える**。`pol_p` は記録のまま（P_net）。
- 波 29 の dump には `pol_p`・`pol_v0` の列が無い（rs-q-pi より前の波）。`pol_v0` は足し、`pol_p` は足さない（訓練は `--warm-start` の forward で補う）。
- `meta_n_record.json` は記録の meta に `reanalyzed: {mult, points, games, diverged, unmapped, selection, record_dir, engine_commit}` を足す。`meta_games.json` は再分析した局だけ。
- 点ごとの jsonl（規則・記録と深い探索の n／q・秒）を計測用に別に書く。

**点の選び方**（局ごと・上限 `--per-game 12`・優先順）:
1. ターン 1〜3 の main 行は全部。
2. 候補に「ターンを終える」と「攻撃」が両方在る行。
3. エネル（OP15-058）がリーダーの席の行は全部。
4. 残りは乱数で埋める（乱数の種は局の seed）。

- 各行は最初に当たった規則に属する。優先順に上限まで取る。規則の中で上限を超えるときは、乱数で選ぶ。
- 再生は、局の最後に選んだ点まで進めて止める。
- `selection` に規則ごとの数を記録する。

**テスト**（最小限・1 本）: fixture の記録（`tests/fixtures/f_identity/rec`）で、差し替えの段を確かめる。
- 元の列が全部コピーされること。
- `pol_n`／`pol_q`／`pol_v0` だけが変わること。
- `--mult 1`・雑音なしで、記録と同じ候補の集合に対応が付くこと。
- `docs/TEST_SPEC.md` §2 に 1 行足す。

### 1.2 エンジン（記録を作ったコミットの wheel で再生する）

- **波 29 w01**: `RESULT.json` の `commit` 欄＝**0be4a48f3**（§20.8.6 の波 29 の生成指示書の本線）。ブランチ `claude/n29-w01` の生成の親もこのコミット。
- **波 32 w01**: `RESULT.json` に code 欄が無く、ブランチの親（84a162aef・`n1-results`）は符号化 v14 より前のコード。記録は enc 14 なので、生成に使ったコードはこの親ではない。
  - 波 32 を出した本線のコミット **0af60ccda**（2026-09-11 15:23「Dispatch wave 32 …」・生成 ee2efc0a4 は 17:12）を使う。
  - 符号化 v14 の 3d042a63c から 2026-09-12 まで、`rust/opcg_engine/src`・`opcg_sim/` に触ったコミットは無い（`git log 3d042a63c..origin/claude/arena-b-w01 --until …`）。だから波 30〜32 のコードは 3d042a63c と同じ。
- ネット: 波 29＝そのコミットの `opcg_sim/data/learned/nrel_r3.npz`。波 32＝`claude/train-r5:n1_results/nrel_r5.npz`（meta の `net` は生成機の置き場なので `--net` で渡す）。
- 各コミットの worktree で wheel を作り、その worktree を cwd にして回す（器は worktree へ写す）。

### 1.3 パイロット

- 波 29 w01 と波 32 w01 で各 **40 局**（先頭の 40 seed）・`--mult 4`・上の選び方。

**測るもの**:
- **再生率**: 最後に選んだ点まで、記録と同じ手を再現した局の割合。
- diverged／unmapped の数。
- 1 点の秒数と、1 局の再生の秒数（深い探索の秒を除く）。
- **深い探索の最善が、記録の探索（160 sims）の最善と違う点の割合**。最善は訪問数の最大。全体と規則 (1)(2)(3)(4) 別。
- π'（`q_improved`・`c_scale 0.3`・`n_min 1`・`c_visit 50` の既定）のエントロピーを、次の 3 つと並べる。
  - 記録の訪問分布
  - 深い探索の訪問分布
  - 記録の (n, q) で作った π'
  - P_net は記録の `pol_p` を使う（無い波 29 は一様）。

### 1.4 予告（司令塔・数を見る前に固定）

- 再生率 ≥ 0.9（正しいコミットなら）。
- 深い最善 ≠ 記録の最善 25〜45%。
- 1 点 1〜2 秒＋再生 1 局 15 秒。
- **再生率が 0.9 を割ったら本走に進まない**。原因（コミット違い・ε・乱数の種）を 1 段だけ追って報告する。代替案（今のエンジンで新しい base 波を作り、同じ走りの中で再分析）はユーザ判断なので、勝手に始めない。

### 1.5 限界

- 深い探索の値は 4 倍の探索の中の値で、真の値ではない。
- π' の式は訪問数の最大（深い探索では約 4 倍）に比例して鋭くなる【コード】（`(c_visit + max N) × c_scale`）。深い探索の π' は、記録の π' より鋭くなるはず。
- 波 29 には `pol_p` が無い。

---
