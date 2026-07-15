# v5 本走 実行 runbook・セッション用プロンプト（2026-07-13）

v5 本走（分散マルチセッション）の起動手順と、各セッションへ貼るプロンプト。v4 運用（`v4_adoption_20260712.md`）の
枝式アクター/ラーナー方式を流用。**全セッションはブランチ `claude/cpu-spec-improvements-yw91jd`**（v5 実装が入った枝）で
起動すること（main には未マージ＝main で起動すると v4 コードになり enc-version 4 が使えない）。

## 0. 前提・パラメータ

- 符号化: **v4**（`--enc-version 4`）。種は gen4（v3）からの温スタート（恒等）。
- ネット形状ガード（種取り違え防止・任意だが推奨）: `OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116`（gen4 の形状）。
- 枝: net=`claude/v5-net`、data=`claude/v5-data-w1..w3`（ワーカー数ぶん）。
- v5 新レバーの初期パラメータ（v4 実績とデータ分布を見て調整）:
  - `--mark-seed-frac 0.15`（マーク局面シード比・§4-2）
  - `--l1-mix 0.25`（L1-hard 混合・v4 と同じ）
  - `--distill-weight 0.1`（忘却抑制・v4 教師アンカー・§4-4b）
  - `--aux-weight 0.25`（残りターン補助・v4 と同じ）

## 1. 種付け（司令塔・1回だけ）

```bash
cd <repo> && git checkout claude/cpu-spec-improvements-yw91jd
# 種ネット生成（gen4→v4 温スタート・恒等自己検証つき）
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/v5_seed_net.py --enc-version 4 --out /tmp/v5seed
# net枝＋data枝3本を種付け
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/pd_setup.py \
  --net-branch claude/v5-net --seed-dir /tmp/v5seed \
  --data-branches claude/v5-data-w1,claude/v5-data-w2,claude/v5-data-w3
# → "PD_SETUP_DONE" を確認
```

## 2. ワーカー（generator）セッション — 共通プロンプト（wN を各自 1..3 に変える）

> あなたは v5 本走の generator ワーカー **wN** です。ブランチ `claude/cpu-spec-improvements-yw91jd` で作業します。
> 以下を実行し、自己対戦データを生成し続けてください。プロセスはループなので、**落ちたら同じコマンドで再起動**して
> ください（再開は data枝の meta から自動）。コンテナのアイドル回収を避けるため、**10分ごとに軽い自己チェック**
> （プロセス生存・最新ログ行）を行い、止まっていたら再起動してください。異常（連続 push FAIL・例外ループ）が続く場合のみ
> 報告してください。
>
> ```bash
> cd <repo> && git fetch origin claude/cpu-spec-improvements-yw91jd && git checkout claude/cpu-spec-improvements-yw91jd
> OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
> OPCG_PD_NET_BRANCH=claude/v5-net OPCG_PD_DATA_BRANCH=claude/v5-data-wN \
> OPCG_PD_WT=/tmp/pd-wN OPCG_LOG_SILENT=1 PYTHONPATH=tests \
> python tests/scripts/pd_gen.py --enc-version 4 --sims 160 --games 32 --workers 3 \
>   --l1-mix 0.25 --mark-seed-frac 0.15 --dirichlet-eps 0.15
> ```
>
> メモリが厳しい場合は `--workers 2` / `--games 24` へ下げてよい（v4 で有効だった対処）。

## 3. ラーナー（learner）セッション — 1本だけ

> あなたは v5 本走の learner（net枝の単独 writer）です。ブランチ `claude/cpu-spec-improvements-yw91jd` で作業します。
> 以下を実行し、data枝から新鮮バッチを集めて学習し net枝へ push し続けてください。**落ちたら同じコマンドで再起動**
> （consumed/pending は manifest 永続＝二重学習しない）。10分ごとに自己チェックし、止まっていたら再起動してください。
>
> ```bash
> cd <repo> && git fetch origin claude/cpu-spec-improvements-yw91jd && git checkout claude/cpu-spec-improvements-yw91jd
> OPCG_P3_LEAD_SLOTS=2 OPCG_P3_EFF_DIM=116 \
> OPCG_PD_NET_BRANCH=claude/v5-net \
> OPCG_PD_DATA_BRANCHES=claude/v5-data-w1,claude/v5-data-w2,claude/v5-data-w3 \
> OPCG_PD_WT=/tmp/pd-learn OPCG_LOG_SILENT=1 PYTHONPATH=tests \
> python tests/scripts/pd_learn.py --enc-version 4 --lr 2e-4 --aux-weight 0.25 \
>   --distill-weight 0.1 --games-per-update 128 --max-staleness 3
> ```
>
> 各 round のログ `vmse=…/… aux±…T` を残してください（aux±T が時計学習の直接指標）。

## 4. 監視・評価（司令塔・定期）

net枝の checkpoint を凍結して評価する。マイルストーン（cum≈1k/2k/5k…）ごとに:

```bash
# 現在の net を取得
git fetch origin claude/v5-net && git show origin/claude/v5-net:p3ckpt/value.npz > /tmp/v5_value.npz
git show origin/claude/v5-net:p3ckpt/policy.npz > /tmp/v5_policy.npz

# (a) マーク回帰ゲート（v5 profile・baseline=gen4）
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_gate.py \
  --challenger /tmp/v5_value.npz,/tmp/v5_policy.npz --profile v5 --seeds 5

# (b) 対 v4 アリーナ（退行監視）＋対 L1 多様（絶対強度）
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/harness/cpu_arena.py arena-paired \
  --challenger learned --baseline learned --pairs 24   # challenger=v5(既定エンジン差し替え)・baseline=gen4

# (c) 時計誤差の対面別分解（aux の質）
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/clock_error_by_leader.py --batch <data枝のbatch.npz>

# (d) ピーク自動アラート（評価系列 JSONL を追記して再実行）
#   1行 = {"round":R, "mark_improved":M, "arena_wr":W}
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/peak_alert.py --evals /tmp/v5_evals.jsonl
```

**凍結判断**: `peak_alert` が「ピーク通過」を報せた round（＝best 複合スコア）を凍結候補にする。
v5 の主判定は **arena（対v4 非劣＋対L1 強度）＋本走後の再プレイ再マーク**（mark_gate は @82 改善＋非退行の補助）。

## 5. 採用（本走後）

ピーク round の value/policy を `gen5_{value,policy}.npz` として同梱・`cpu_learned` の既定を切替・
adoption レポート作成・SPEC/TEST_SPEC 追従。**そこまで到達したら、v5 準備〜採用を1本の PR にまとめる**
（本セッションの `claude/cpu-spec-improvements-yw91jd` 上の全コミット＋採用差分）。
