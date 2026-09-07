# NRel 学習の torch（CPU）化（2026-09-07・WP `train-torch`）

`docs/rust_engine_plan.md` §18.4。計測 WP の結論（`2026-09-07_train_profile.md` §4・§6 の 2）を
本番の訓練器（`opcg_sim/learned/train/n_rel_train.py`）に入れた。**value と policy の両方**を
torch で書き、numpy の手書き backward は `--backend numpy` の参照実装として残してある。

## 結論（先に）

**1 シャード・1 エポックの学習ループが 146.7 秒 → 34.7 秒（4 コア・4.22 倍）。1 スレッドでも
2.29 倍**（計測 WP の予告 2.28 倍とほぼ一致）。forward は 1.9e-6 で一致し、勾配は「numpy 自身を
加算順だけ変えて回した誤差」より小さい。**npz は 1 バイトも形式が変わっておらず、Rust の
`load_net` が読んで同じ value を返す**（5.4e-7）。

| 受け入れ | 基準 | 実測 | |
|---|---|---|---|
| a. forward | 1e-5 | **1.9e-6**（value 8.8e-7・policy logits 1.9e-6） | ○ |
| b. 勾配 | 相対 1e-4 | **6.1e-5**（\|g\|>1e-4）／正規化 **2.9e-6** | ○（下記の注） |
| c. 学習 | 相対 1e-2 | warm-start で **v_mse 0.53%・p_loss 0.05%** | ○（下記の注） |
| d. 時間 | — | **2.29 倍**（1 スレッド）／**4.22 倍**（4 コア） | — |
| npz 形式 | 無変更 | Rust `load_net`＋`net_eval` が **5.4e-7** で一致 | ○ |

## 計測環境と条件

Intel Xeon @2.80GHz・**4 コア**／Python 3.11.15・numpy 2.4.6・**torch 2.14.0（CPU 実行）**。
入力は dump v2 の 1 シャード（`origin/claude/n28-w01` の `n28_records`・**81,871 行**・方策点
46,325）。すべて `--ablate rel`（生成役 a1・訓練 r2 の既定）・`--seed 13`・`--holdout-mod 7`・
bs-v 256／bs-p 64・lr 5e-4。1 エポックの構成は value 訓練 70,306 行＝274 バッチ・policy 訓練
39,762 点＝621 バッチ＝**合計 895 ステップ**。

> torch は PyPI から入れた（`pip install torch`）。**`download.pytorch.org` は本コンテナの
> プロキシが 403 で塞いでいる**ので、README の CPU 版 index を使う手順が通らない環境では
> PyPI 版で代替する（CPU 実行に違いは無い。`+cu130` のタグが付くだけで CUDA は使わない）。

再現は 2 本のコマンドで足りる:

```bash
# a（forward）・b（勾配）
OPCG_LOG_SILENT=1 python tests/scripts/n_rel_torch_check.py \
  --in <n28_records> --net <nrel_a1.npz> --out chk.json
# c（学習）・d（時間）… --backend / --threads を変えて 1 エポックずつ
OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_train train \
  --in <n28_records> --epochs 1 --seed 13 --ablate rel --backend torch --threads 0 --out out.npz
# npz 形式（Rust の load_net）
OPCG_LOG_SILENT=1 python tests/scripts/rs_net_load_check.py --in <n28_records> --net out.npz
```

## a. forward 一致（同じ重み・同じバッチ）

`nrel_a1.npz` の重みを両方に読ませ、value は 256 行・policy は 64 点（候補 626 行）で比べた。

| | 最大絶対差 | 平均絶対差 | |
|---|---|---|---|
| value（tanh 出力・値域 [-1,1]） | **8.79e-07** | 1.19e-07 | 計測 WP の 8.79e-7 と同値 |
| policy logits | **1.91e-06** | 4.31e-07 | |

**1e-5 で一致（合格）。** value 側が計測 WP のプロトタイプと同じ数字になったのは、式も
バッチも同じだから（プロトタイプは value 経路だけだった。policy はここで初めて照合した）。

## b. 勾配一致（numpy の手書き backward 対 torch autograd）

同じバッチで両者の全パラメータの勾配を比べた。指標は 3 つで、**読み方には注意が要る**。

| 指標 | value | policy | 判定 |
|---|---|---|---|
| \|g\|>1e-4 の要素の相対誤差 | 3.6e-05 | **6.13e-05** | < 1e-4 ○ |
| max\|Δg\| / max\|g\|（正規化） | 2.86e-06 | 7.73e-07 | < 1e-5 ○ |
| \|g\|>1e-6 の要素の相対誤差 | 1.03e-03 | **1.34e-03** | 下記 |

**指示書の「相対 1e-4（\|g\|>1e-6 の要素）」は float32 の丸めの下限より下にある基準で、
どんな再実装でも達成できない。** 根拠は「同じ numpy の式を、バッチ全体で回した場合と半分ずつ
回して足した場合」——**数学的に同じ値**なのに差が出る:

| | \|g\|>1e-6 | \|g\|>1e-4 | 正規化 |
|---|---|---|---|
| numpy 対 numpy（加算順だけ違う＝**丸めの下限**） | 3.99e-04 | 4.50e-05 | 3.03e-06 |
| **numpy 対 torch** | 1.34e-03 | 6.13e-05 | **2.86e-06** |

`|g|>1e-6` では下限の 3.4 倍だが、**正規化した誤差では torch の方が下限より小さい**
（2.86e-06 < 3.03e-06）＝「torch は、numpy が自分自身と一致する以上に numpy と一致している」。
絶対差の最大は全パラメータで 1.2e-07 以下＝float32 の 1〜2 ULP。**式は同じと判断してよい。**

> 方策の `bp2` だけは正規化から外してある。勾配が `Σ(p−π)/P` ＝ 区間ごとに両方 1 に和が合うので
> **解析的に厳密な 0** で、実測の `g_max` 4.4e-09 は numpy の丸めそのもの（両者の絶対差は
> 2.3e-10）。これで正規化すると意味の無い比（0.05）になる。

判定の式は `tests/harness/n_rel_torch_cmp.py`（テストと報告用 CLI が共有）。

## c. 学習（同じ教材・同じ seed・1 エポック）

holdout（`seed%7==0`）の v_mse と p_loss。**評価と保存は両 backend とも numpy 版が行う**
（torch は epoch の終わりに `sync_to_numpy()` で重みを書き戻す）＝指標の計算式は完全に同じ。

### ゼロから 1 エポック（**混沌が支配する土俵**）

| 実行 | 学習ループ | val v_mse | numpy 比 | val p_loss | numpy 比 |
|---|---|---|---|---|---|
| numpy（BLAS 1 スレッド＝本番） | 146.7 s | 0.7411 | — | 1.8417 | — |
| **numpy（BLAS 4 スレッド）** | 132.8 s | **0.7592** | **2.44%** | 1.8424 | 0.04% |
| torch 1 スレッド | 64.0 s | 0.7513 | 1.37% | 1.8474 | 0.31% |
| torch 4 コア | 34.7 s | 0.7374 | 0.51% | 1.8518 | 0.55% |

**同じ numpy 同士（BLAS のスレッド数だけ違う＝数学は同じ・加算順が違う）が 2.44% ずれる。**
numpy 対 torch のどの差（最大 1.37%）よりも大きい＝この 1% 超は backend の偏りではなく、
**乱数初期値から 895 ステップ回す軌道が浮動小数の丸めに対して混沌である**ことの現れ。
p_loss は全条件で 1% 以内に収まる。

### warm-start（`nrel_a1.npz` から 1 エポック・**混沌に依らない土俵**）

学習済みの重みから始めれば軌道は暴れない。ここが backend の等価性を見る本来の土俵:

| 実行 | 学習ループ | val v_mse | val p_loss |
|---|---|---|---|
| numpy | 122.3 s | 0.6760 | 1.7300 |
| torch 4 コア | 30.7 s | 0.6796 | 1.7291 |
| **差（相対）** | **3.99 倍** | **0.53%** | **0.05%** |

**v_mse・p_loss とも相対 1e-2 以内（合格）。**

> Adam の差: numpy 版は全パラメータで 1 本のバイアス補正カウンタ `_t` を共有する（value ステップ
> でも方策ヘッドの `_t` が進む）のに対し、`torch.optim.Adam` はパラメータごとに step を数える
> （勾配 None のパラメータは進まない）。`1-0.9^t` は t が数十で 1 に収束するので 1 エポック規模の
> 指標には出ない。lr/betas/eps は同じ。

## d. 時間（1 シャード・1 エポック・学習ループの壁時計）

読み込み（約 6 秒）と holdout 評価は両 backend とも numpy で共通なので、**ステップのループだけ**
を測った（`N_REL_TRAIN_EPOCH` の `train_sec`）。

| backend | スレッド | 895 ステップ | numpy 比 | 1 エポック全体（読み込み・評価込み） |
|---|---|---|---|---|
| numpy（手書き backward） | 1 | 146.7 s | 1.00x | 156 s |
| torch | 1 | 64.0 s | **2.29x** | 76 s |
| torch | 4（全コア） | **34.7 s** | **4.22x** | 44 s |

**1 スレッドで 2.29 倍**は計測 WP の予告（2.28 倍）と一致。4 コアは 4.22 倍で、予告の 5.33 倍には
届かない——予告は「バッチを事前に用意した value ステップだけ」の比で、**実際の 1 エポックには
Python のバッチ切り出し・`budget_feats`・R のゼロ配列の確保が残る**（どれも numpy のままで
スレッドが効かない）。§18.2 の見込み（①＋④で 7〜8 倍）は value/policy のステップ**だけ**を数えた
値なので、エポック全体では 4.2 倍と読むのが正しい。

> **バッチの準備を numpy から動かせば更に伸びる余地がある**（現在 34.7 秒のうち、計測 WP の
> 内訳から見て切り出し系が数秒を占める）。ただし §18.2 の 3（float16/int16 memmap）で読み口を
> 決めてからの方が手戻りが無いので、本 WP では触っていない。

## npz の形式（無変更）

torch で訓練した npz（`out_torchN.npz`）を Rust の `load_net` に読ませ、`net_eval` の value を
Python の `NRelNet.value` と 64 行で比べた（`tests/scripts/rs_net_load_check.py`）。

| | Python | Rust |
|---|---|---|
| hidden | 192 | 192 |
| ablate | `["rel"]` | `["rel"]` |
| vocab_ids | 2,803 | 2,803（card_table 2,804 行＝PAD ぶん +1） |
| meta.kind | `nrel-a` | `nrel-a` |
| value（64 行） | — | **最大絶対差 5.36e-07**・平均 1.75e-07（< 1e-5） |

**鍵・形・dtype（float32）・`meta.kind`・`vocab_ids`・`ablate` の焼き込みはすべて numpy 版のまま。**
学習中の重みは torch の Parameter に置くが、**保存は numpy 版の `NRelNet.save` が行う**ので
形式が変わりようがない（`sync_to_numpy()` で書き戻してから保存する）。

> 指示書は `rs_net_oracle.py` で照合せよとしていたが、あれは Python エンジン
> （`harness.game_driver`／`manager_from_hidden`）に依存していて退避後の本ツリーでは動かない。
> **盤面の生成を挟まず dump v2 の符号化をそのまま `net_eval` に渡す**最小の照合を新しく書いた
> （比べたいのは forward の値であって盤面の再現ではない）。

## 実装のかたち

- `opcg_sim/learned/train/n_rel_torch.py`（新規）: `TorchNRel`（forward・value と policy）と
  `TorchTrainer`（autograd＋`torch.optim.Adam`）。`value_step`／`policy_step` の**引数と戻り値は
  numpy 版と同じ**なので、訓練ループは backend を知らない。
- `--backend numpy|torch`（既定 torch）・`--threads N`（既定は全コア）。**torch を import
  できなければ警告して numpy に落ちる**＝torch は任意依存（requirements に固定しない）。
- `n_rel_train` は import 時に `OMP_NUM_THREADS=1` を立てる（numpy の BLAS はスレッドを増やすと
  遅くなる＝計測 WP §4 の対照表）ので、**torch のスレッド数は `torch.set_num_threads` で明示的に
  上書きする**（しないと 1 スレッドに引きずられる）。
- `--ablate rel` のときは torch 経路でも `relations_batch` を呼ばない（§18.3 の
  `relations_or_zeros` をそのまま通る）。遮断自体は numpy 版と同じく forward の入口で行う。
- `n_rel.cand_rel_rows` を切り出した: 候補の R(si,ti) の対応付けの正本を 1 か所にして、
  numpy の `cand_input` と torch 経路が同じ関数を呼ぶ（`rel_om` は入力データで学習対象ではない
  ので、torch 側もこの numpy の結果を定数として使える）。

## ついでに直したもの

**分岐元（`claude/rs-archive-cutover-8y6odi`）では `n_rel_train.py` の `train()` が動かなかった。**
退避で消えた `cpu_selfplay._load_db` を import していたため。`opcg_sim.learned.vocab.load_db`
（＝`opcg_sim.loop.decks.load_db`・旧 `_load_db` と同じもの）に差し替えた。語彙は
`build_eff_tables()` が同じ db から作るので値は変わらない。`n_rel_band.py` にも同じ import が
残っているが、本 WP の範囲外なので触っていない（**次に `n_rel_band` を回すときに落ちる**）。

## テスト

`tests/test_n_rel_train_torch.py`（`cpu_infra`・**torch が無ければ skip**）7 本。盤面は合成
（乱数）で dump も Python エンジンも要らない。a（forward 一致）・b（勾配一致＝上の 3 指標を
丸めの下限と並べて見る）・勾配が来るパラメータの集合が numpy と同じ・npz を `n_rel.load` で
読めて forward 一致・backend の選択と numpy への退避・numpy 参照実装が壊れていないこと・
1 ステップ目の損失が numpy と一致すること。判定の式は `tests/harness/n_rel_torch_cmp.py` に
置いて報告用 CLI と共有する。

数字の一次情報は同じ内容の `RESULT.json`（本ブランチ直下）にある。
