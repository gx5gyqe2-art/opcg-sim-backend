# 方式B（プロセス分離）: CPython で FastAPI/配信スタックを動かし、CPU 探索（decide）だけを
# 同梱した PyPy ランタイムのワーカープロセスで実行する（~2.1x・docs/reports/pypy_*）。
# 配信スタック（pydantic-core/grpcio）は PyPy 非互換のため CPython 側のみに置く（Phase0 で確認）。
FROM pypy:3.11-slim AS pypy

FROM python:3.11-slim

# PyPy ランタイムを CPython イメージへ同梱（探索ワーカー専用。配信スタックは載せない）。
COPY --from=pypy /opt/pypy /opt/pypy
RUN ln -sf /opt/pypy/bin/pypy3 /usr/local/bin/pypy3

WORKDIR /app
COPY opcg_sim/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# カードを事前パースしてキャッシュを焼き込む（起動時の全件パース~1.8sを回避）。
# COPY 後に実行するので、カードDB/コードが変われば自動で再生成される。
# エンジンは stdlib-only なので CPython/PyPy 双方が同一キャッシュを読む。
RUN python -m opcg_sim.tools.build_card_cache

ENV PORT=8080
# 探索オフロードを既定で有効化（無効化は OPCG_PYPY_WORKER=0 でインプロセス実行へ即ロールバック）。
ENV OPCG_PYPY_WORKER=1
# 体感最適化（Phase 3）。① 計画キャッシュは本番有効（イベントループ内・同期＝単一スレッドで安全）。
#   OPCG_PLAN_CACHE=1 : ① 計画キャッシュ（セグメントを1回計画→以降は即時 replay。手・盤面・勝敗は不変）
# ⑥ ポンダリング（OPCG_PONDER / OPCG_PONDER_SPEC）は **間欠クラッシュ（並行バグ）のため無効化**:
#   `_ponder_plan`/`_speculate_plan` は `asyncio.to_thread`（本物のOSスレッド）で探索系を走らせるが、
#   差分巻き戻し journal（`opcg_sim/src/core/journal.py`）は **プロセス共有のグローバル状態**（`_active`/
#   `_mut_count`・「探索は単一スレッド前提」）。バックグラウンドスレッドが探索で `transaction()` を開いた瞬間に
#   メインスレッドが新しい CardInstance を生成（`master` 初回セット）すると、そのセットがポンダリング側 journal
#   に誤記録され、rollback が live カードから `master` を pop ＝ `CardInstance has no attribute 'master'` で
#   間欠クラッシュ/フリーズ（ワーカー温まると探索が別プロセスへ出て窓が消える＝「時間が経つと治る」）。
#   止血としてポンダリングを無効化（体感最適化のみ＝手・盤面・勝敗は不変）。journal のスレッド安全化後に再検討。
ENV OPCG_PLAN_CACHE=1
ENV OPCG_PONDER=0
ENV OPCG_PONDER_SPEC=0
CMD ["sh", "-c", "uvicorn opcg_sim.api.app:app --host 0.0.0.0 --port $PORT"]
