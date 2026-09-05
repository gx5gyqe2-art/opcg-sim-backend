# N系 学習ループの分散運用（正本・2026-09-01 ユーザ決定）

複数の Claude セッションで「教材収集 → 学習 → 検証」を回すための取り決め。
**コーディネータ**（本セッション）は指示書を書き、成果物を回収し、判定する。実行は全て
**独立した作業セッション**が行う。理由: コーディネータのコンテナは数十分ごとに巻き戻り
（2026-09-01 に1日で 19 回・35 分の訓練が 4 回連続で消失）、長時間ジョブを完走できない。

## 1. 原則

1. **やり取りは全て origin のブランチ経由**。チャット本文・ローカルファイル・/tmp は
   受け渡しに使わない（消える）。ブランチが唯一の正本。
2. **作業セッションはコードを変更しない**。指定ブランチを checkout して実行し、成果物を
   自分の出力ブランチに push するだけ。PR も作らない。
3. **成果物には必ず `RESULT.json` を添える**（機械可読・§5）。コーディネータはこれを読んで
   回収する＝チャットの報告文を解析しない。
4. **判断基準は実行前に決めて指示書に書く**（結果を見てから解釈を作らない）。
5. **seed 帯はコーディネータが台帳で払い出す**（§6）。作業セッションが勝手に選ばない。
6. 成果物のシャードを取捨選択しない。全部揃って初めて判定になる。

## 2. 役割と受け渡し

| 役割 | 入力（コーディネータが指示書に書く） | 出力（作業セッションが push する） |
|---|---|---|
| **生成** | 生成役ネットの取得元／波番号 W／シャード N／seed 帯／局数 | `claude/n{W}-w{NN}`: `n{W}_records/`・`n{W}_gen.log`・`RESULT.json` |
| **訓練** | 起点ネット／π 波の一覧／z 波の一覧／エポック・lr／出力名 | `claude/train-c{N}`: `n1_results/neff_net_c{N}.npz`・訓練ログ・`RESULT.json` |
| **検証** | 候補ネット／基準ネット／シャード N／seed 帯／ペア数／条件 | `claude/arena-{tag}-w{NN}`: `n1_results/arena_{tag}_dist/*.jsonl`・`RESULT.json` |
| **コーディネータ** | — | `claude/n1-results`（ネット・台帳の正本）／`docs/reports/`（判定の記録）／本書 §6 の seed 台帳 |

コーディネータの回収手順は役割ごとに固定（§4）。回収したネット・台帳は `claude/n1-results`
に取り込み、以後はそこが正本になる（作業セッションの出力ブランチは保全のため消さない）。

## 3. 1 サイクルの流れ

```
[生成] 波 W を 8 シャード並列（各 960 局・4〜5 時間）
   ↓ claude/n{W}-w01..08
[コーディネータ] 回収・meta 確認（games/rows/dropped・seed 衝突なし）・データ窓を決める
   ↓ 指示書（訓練）
[訓練] warm-start で 2 エポック（約 70 分・RSS ~11GB）
   ↓ claude/train-c{N}
[コーディネータ] 回収・n1-results へ取り込み・評価帯で参考値・指示書（検証）
   ↓ 指示書（検証）
[検証] 8 シャード並列（各条件 24 ペア・25〜35 分）
   ↓ claude/arena-c{N}-w01..08
[コーディネータ] arena_merge で合算・判定・レポート・昇格なら生成役を切り替え
```

データ窓・昇格条件・評価帯の規約は `CLAUDE.md`「学習データ窓の運用」「アリーナ昇格」に従う。
必要局数の根拠と分散判定の規約は `docs/reports/2026-09-01_distributed_arena.md`。

## 4. コーディネータの回収チェックリスト

**生成**: 8 ブランチ揃う → 各 `meta_n_record.json` の `games`=960・`seed_base` が台帳どおり →
`dropped` が数件以内 → `RESULT.json` の status=done。1 本でも欠けたら訓練に進まない。

**訓練**: `RESULT.json` の inputs（起点・π・z）が指示書と一致 → `epochs`/`best_ep`/val 指標を
記録 → ネットを `claude/n1-results` の `n1_results/` へ取り込む → 評価帯で参考値を測る。

**検証**: 8 ブランチ揃う → 各台帳 24 行 → `arena_merge.py` で条件ごとに合算 →
`dup_seeds`=0・`void` 率 ≤2% を確認 → 判定 → 台帳を `claude/n1-results` へ取り込む →
`docs/reports/` に記録。昇格なら次の指示書（生成）の生成役を新ネットにする。

## 5. RESULT.json の形式

作業セッションは出力ブランチの直下に置く。最低限これだけ（追加は自由）:

```json
{"job": "train-c10", "role": "train", "status": "done",
 "inputs": {"warm_start": "claude/n1-results:n1_results/neff_net_c8_e2.npz",
            "pi_waves": [16, 17, 18, 19], "z_waves": [14, 15, 16, 17, 18, 19],
            "epochs": 2, "lr": 2e-4},
 "outputs": {"net": "n1_results/neff_net_c10.npz"},
 "metrics": {"val_v_mse": 0.5185, "val_v_sign": 0.814, "pi_top1": 0.634, "best_ep": 0},
 "command": "PYTHONPATH=tests python tests/scripts/n_eff_train.py train ...",
 "notes": "ep1 はやや悪化"}
```

`status` は `done` / `partial`（局数・ペア数が指示に満たない）/ `failed`。`partial` と
`failed` は理由を `notes` に書く。コーディネータは `done` 以外を回収しない（やり直しを指示）。

## 6. seed 帯の台帳（コーディネータが払い出す・追記のみ）

規約: 生成 = `2000000 + W*10000 + N*1000`（W=波番号・N=シャード 1..8）。
検証 = 100000 番台〜（対戦ごとに 8 シャード分・条件ごとに別帯）。**既出の帯は再利用しない**。

| 用途 | 帯 | 状態 |
|---|---|---|
| 生成 波1〜18 | 2010000〜2188000 | 使用済み |
| 生成 波19 | 2191000〜2198000 | 使用済み（w05 は打ち切り＝75シャード約750局・meta無し・そのまま使う） |
| 生成 波20 | 2201000〜2208000 | 使用済み（2026-09-03 回収・7,680局・棄却5・生成役 c10＝era5。方策一致で c10 0.688 > c8 0.649 を確認） |
| 生成 波21 | 2211000〜2218000 | 使用済み（2026-09-03 回収・**w02 打ち切り（ユーザ決定 2026-09-03・そのまま使う）＝7シャード 6,720局**・棄却5・生成役 c10＝era5。方策一致 w01 で c10 0.678 > c8 0.635） |
| 生成 波22 | 2221000〜2228000 | 使用済み（2026-09-03 回収・**w02 打ち切り（ユーザ決定 2026-09-03・そのまま使う）＝7シャード 6,720局**・棄却5・生成役 c10＝era5。方策一致 w01 で c10 0.672 > c8 0.621。波20〜22 は同一 era＝π 窓は3波まとめて使う） |
| 生成 波23（**dump v2**・NRel P4） | 2231000〜2238000 | 使用済み（2026-09-05 回収・8 シャード 7,680 局・1,079,753 行・棄却 7・生成役 c10＝era5・sims 128・dump_version 2／enc_version 13・コード 0102748 以降） |
| 生成 波24（**dump v2**・NRel P4） | 2241000〜2248000 | 使用済み（2026-09-05 回収・8 シャード 7,680 局・1,084,300 行・棄却 6・生成役 c10＝era5・sims 128・dump_version 2／enc_version 13。波23 と同時に 16 セッションで生成〔ユーザ決定 2026-09-04・初物のためコーディネータが指示書をフルで書いた例外〕） |
| 生成 波25（**dump v2**・NRel P4） | 2251000〜2258000 | 払い出し済み・未実行（r1 の初回結果を見てから要否を決める） |
| 訓練 r1（NRel Stage A・初回・`n_rel_train.py`） | —（seed 帯不要・holdout は seed%7==0） | 指示済み（2026-09-05。入力: π＋z＝波23〜24 の dump v2（旧形式の波は使えない＝z の追加窓なし）・warm-start なし・epochs 2・lr 5e-4・hidden 192・bs 256/64。出力: `claude/train-r1` の `n1_results/nrel_r1.npz`＋log＋RESULT.json。判断は評価帯（新帯・後述）→h1〜h6 一致率→レイテンシ→アリーナ r1 vs c10（帯 281000〜/291000〜）の順） |
| 検証 c5a〜c9（単一セッション期） | 101000〜120000 | 使用済み |
| 検証 c9 vs c8 分散 | ランダム 201000〜208000／ミラー 211000〜218000 | 使用済み |
| 参考 c10 vs G15（出荷既定・単一セッション24ペア×2条件） | ランダム 121000／ミラー 122000 | 使用済み（2026-09-02） |
| 検証 c10 vs c8 分散 | ランダム 221000〜228000／ミラー 231000〜238000 | 使用済み（2026-09-02・主0.589/副0.568＝昇格・era5開始） |
| 検証 c11 vs c10 分散 | ランダム 241000〜248000／ミラー 251000〜258000 | 使用済み（2026-09-03 判定: **非昇格**・主 0.539 [0.485,0.593]／副 0.503 [0.459,0.546]・void 0・`docs/reports/2026-09-03_c11_judgment.md`。生成役は c10 のまま・N系ループはここで反復停止＝設計フェーズへ。指示内容: 訓練 train-c11: warm c10・π=波20〜22・z=波16〜19→OOM で 17〜19 に縮小（実績・val v_mse 0.6052/v_sign 0.768/pi_top1 0.681・best ep0・n1-results 5f882b0）。評価帯参考値 v_mse（w01+w03 holdout・c8_e2/c10/c11）: 旧帯 波16〜18 0.5615/0.5423/0.5399・新帯 波20〜22 0.6318/0.6214/0.6130＝世代順に単調改善・判断基準 2条件とも wr≥0.55 かつ CI下限>0.50。リーダー入替は従来どおり席のみ＝`cand_leaders` 記録あり） |
| 検証 c10 vs G15 正式（分散・出荷既定切替の確認） | ランダム 261000〜268000／ミラー 271000〜278000 | 使用済み（2026-09-03 判定: **2条件とも達成**・主 0.633 [0.580,0.686]／副 0.589 [0.541,0.637]・void 0・`docs/reports/2026-09-03_c10_vs_g15_formal.md`。指示内容: 候補 c10=`neff:`・基準 G15=明示パスの value,policy ペア〔既定が c10 に変わっても基準が動かない〕・8シャード×24ペア×2条件・判断基準 2条件とも wr≥0.55 かつ CI下限>0.50） |
| 検証 r1 vs c10 分散（予定） | ランダム 281000〜288000／ミラー 291000〜298000 | 予約（r1 の評価帯・一致率・レイテンシを見てから起動） |
| 対照実験 残り起動（腕A2=c10+residual_activate low vs 腕B=c10・紫リーダー×掘り差し込み合成デッキ `--leaders purple --decks synth_dig`・8シャード×24ペア） | 311000〜318000 | 使用済み（2026-09-03・全体 0.523 [0.471,0.576]・発火 37/384 局のみで判定不能。`high` 方針の 321000〜328000 は予約のまま） |
| 対照実験 残ドン掘り（腕A=c10+residual_dig vs 腕B=c10・紫リーダー×掘り差し込み合成デッキ `--leaders purple --decks synth_dig`・8シャード×24ペア） | 301000〜308000 | 使用済み（2026-09-02・全体 0.464 [0.411,0.516]＝基準未達・発火ペア 0.438・掘りは報われず＝教材に混ぜない。`docs/reports/2026-09-02_residual_dig_cf.md`） |

## 7. 指示書テンプレート

オーケストレータ用（セッションを生成して配る側）と、作業セッション用（配られる側）の
2 層。`{ }` はコーディネータが埋める。

### 7.1 訓練（1 セッション）

```
学習CPUのネットを1本訓練してください。コードは変更せず、下記を実行して結果ブランチを
push するだけです。PR は作らないでください。メモリは 12GB 以上必要です。

bash で以下を順に実行:

pip install numpy
cd ~ && git clone https://github.com/gx5gyqe2-art/opcg-sim-backend.git
cd opcg-sim-backend
git fetch origin {code_branch}:work && git checkout work
git fetch origin claude/n1-results:res
git show res:{warm_start_path} > ~/start.npz
for N in {all_waves}; do
  for w in 01 02 03 04 05 06 07 08; do
    git fetch -q origin claude/n${N}-w$w:tmpw
    mkdir -p ~/n${N}_wave/w$w
    git archive tmpw n${N}_records | tar -x -C ~/n${N}_wave/w$w
    git branch -qD tmpw
  done
done
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/n_eff_train.py train \
  --in {pi_dirs} \
  --z-in {z_dirs} \
  --epochs {epochs} --lr {lr} --warm-start ~/start.npz --out ~/neff_net_{name}.npz \
  2>&1 | tee ~/train_{name}.log

git checkout -B claude/train-{name} res
mkdir -p n1_results && cp ~/neff_net_{name}.npz n1_results/ && cp ~/train_{name}.log .
（RESULT.json を §5 の形式で書く。metrics はログ末尾の val 行から転記）
git add -f n1_results/neff_net_{name}.npz train_{name}.log RESULT.json
git -c user.email=g.x5gyqe2@gmail.com -c user.name=worker commit -m "train {name}"
git push -u origin claude/train-{name}

補足: 所要は約 70 分。OOM で落ちたら z の波を古い方から1つ減らして再実行し、
RESULT.json の notes に減らした旨を書いてください。
```

`{pi_dirs}` は `~/n16_wave/w*/n16_records ~/n17_wave/w*/n17_records ...` の形、
`{z_dirs}` も同様（π に含まれない波だけ）。`{all_waves}` は両方の和。

### 7.2 検証（8 セッション・オーケストレータ用）

`docs/reports/2026-09-01_distributed_arena.md` の実行ブロックを正とする。可変部は
候補・基準のネット取得元、`{tag}`、seed 帯（§6 から払い出し）。各セッションは
`claude/arena-{tag}-w{NN}` に台帳と RESULT.json を push する。

### 7.3 生成（8 セッション・オーケストレータ用）

**生成の指示書はコーディネータが書かない**（ユーザ決定 2026-09-02・生成担当セッションが
運用を把握しているため任せる）。コーディネータが渡すのは**生成役ネットの所在と波番号**だけ。
seed 帯は規約から自動で決まる。完了は生成側から共有され、コーディネータは §4 の手順で回収する。

**dump v2（NRel・2026-09-04〜）**: 生成コマンドに `--dump-v2` を足す（符号化 v13＋トークン状態＋候補の
枠 index・meta の `dump_version`=2）。生成役・波番号・seed 帯は従来どおり。旧形式（v1）の波は NRel の
教材にならない。

## 8. コーディネータが守ること

- 指示書を出したら、**その指示書の内容（入力・帯・判断基準）を本書 §6 と
  `docs/reports/` に先に記録する**（巻き戻りで指示内容を失わないため）。
- 回収は必ず origin から fetch して行う（ローカルの残骸を信用しない）。
- 判定は `arena_merge.py` の出力をそのまま使う（手計算で置き換えない）。
- 訓練・検証の結果がどうであれ、ネットと台帳は `claude/n1-results` に残す（不採用でも消さない）。
