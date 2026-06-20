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
# 体感最適化（Phase 3）を本番で有効化（決定性は維持＝decide の決定的結果を前倒しするだけ・合法性ゲートで安全）。
#   OPCG_PLAN_CACHE=1 : ① 計画キャッシュ（セグメントを1回計画→以降は即時 replay）
#   OPCG_PONDER=1     : ⑥-a 先行計画（人間の TURN_END 直後に次手番計画を前倒し＝TURN_END→初回ポーリングの隙間を埋める）
#   OPCG_PONDER_SPEC=1: ⑥-b 投機ポンダリング（人間の MAIN 中に「今エンドしたら」を先回り計算）
# 体感の本命は ⑥-b: 実プレイでは TURN_END 直後にすぐポーリングが来る（思考の隙間が無い）ため ⑥-a 単独では
# `/cpu/step` がタスクを待ってフル待ちになりがち。人間が実際に考えている MAIN 中（⑥-b）に先回りして初めて
# CPU 手番の待ちが消える（自己対戦ベンチは TURN_END 後に思考時間を仮定したため ⑥-a を過大評価していた）。
# トレードオフ: ⑥-b は人間アクションごとに投機を焼き直すためワーカー負荷が増える（当たり率 ~30%）。単一の
# CPU 対戦では許容。即ロールバックは各フラグを 0/未設定に。テスト/自己対戦は env 非依存（同期 decide）で決定性維持。
ENV OPCG_PLAN_CACHE=1
ENV OPCG_PONDER=1
ENV OPCG_PONDER_SPEC=1
CMD ["sh", "-c", "uvicorn opcg_sim.api.app:app --host 0.0.0.0 --port $PORT"]
