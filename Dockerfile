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
# 体感最適化（Phase 3）。決定性は維持＝decide の決定的結果を前倒しするだけ・合法性ゲートで安全。
#   OPCG_PLAN_CACHE=1 : ① 計画キャッシュ（セグメントを1回計画→以降は即時 replay。手・盤面・勝敗は不変）
#   OPCG_PONDER=1     : ⑥-a 先行計画（人間の TURN_END 直後に次手番計画を前倒し）
#   OPCG_PONDER_SPEC=1: ⑥-b 投機ポンダリング（人間の MAIN 中に「今エンドしたら」を先回り計算＝体感の本命）
# ⑥ は一度 **間欠クラッシュ（並行バグ）**で無効化していたが、根本原因（差分巻き戻し journal の状態が
# プロセス共有グローバルで、ポンダリングの asyncio.to_thread と競合し live カードから属性が pop される）を
# **journal のスレッドローカル化**で解消済み（各スレッドの記録が互いに漏れない）＋`_ponder_plan` は live 盤面を
# メインスレッドで clone してから to_thread へ渡す（deepcopy 競合の防止）。並行回帰テスト
# `tests/test_journal_concurrency.py` でガード。よって再有効化。即ロールバックは各フラグを 0 に。
ENV OPCG_PLAN_CACHE=1
ENV OPCG_PONDER=1
ENV OPCG_PONDER_SPEC=1
CMD ["sh", "-c", "uvicorn opcg_sim.api.app:app --host 0.0.0.0 --port $PORT"]
