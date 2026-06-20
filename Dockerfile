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
CMD ["sh", "-c", "uvicorn opcg_sim.api.app:app --host 0.0.0.0 --port $PORT"]
