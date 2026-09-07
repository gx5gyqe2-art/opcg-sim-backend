# ルール・効果・探索は Rust（`rust/opcg_engine`・PyO3 wheel）、配信と API は CPython。
# 方式B（PyPy ワーカーへ decide をオフロード）は Rust 化で不要になったので撤去した
# （2026-09-07・docs/rust_engine_plan.md §16.3-A3）。
#
# Rust エンジンの wheel をビルドする段。実行イメージには **wheel だけ** を COPY する
# ＝コンパイラ/ソース/target は載せない。
# suite（trixie）は**実行段の python:3.11-slim と揃える**こと（2026-09-06 時点で
# python:3.11-slim = 3.11-slim-trixie）。ずれると拡張が要求する glibc が実行段に無く、
# import 時に "GLIBC_x.yz not found" で落ちる。
FROM rust:slim-trixie AS rustbuild
# maturin は Python 拡張のビルドに Python を要る（abi3 の tag 決定・wheel 生成）。
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages maturin==1.15.0
WORKDIR /build
COPY rust/opcg_engine /build/opcg_engine
# abi3-py311 なので出力は cp311-abi3（実行段の Python 3.11 で読める）。
# --compatibility linux: manylinux の glibc タグ付けを省く（ビルド段と実行段は同じ Debian 系）。
RUN cd /build/opcg_engine && maturin build --release --compatibility linux --out /wheels

FROM python:3.11-slim

WORKDIR /app
COPY opcg_sim/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Rust エンジンの wheel（ビルド段の成果物のみ）。コンパイラ・cargo registry は持ち込まない。
COPY --from=rustbuild /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

COPY . /app

# カードを事前パースしてキャッシュを焼き込む（起動時の全件パース~1.8sを回避）。
# COPY 後に実行するので、カードDB/コードが変われば自動で再生成される。
RUN python -m opcg_sim.tools.build_card_cache

# Rust エンジンが読む効果構造 JSON（約 8MB・生成物なので git 管理外）を焼き込む。
# 無い場合は API の起動時に生成されるが、コンテナ起動のたびに ~10 秒かかるのでここで作る。
RUN python -m opcg_sim.tools.export_effects_json --out opcg_sim/data/opcg_effects.json

ENV PORT=8080
# Cloud Logging 費用対策: 効果処理ごとの盤面ダンプ JSON（resolver._log_execution_report /
# _log_failure_snapshot）を本番では全停止する。CPU 探索（PIMC×予算按分）が resolve_ability を
# 1手あたり数百回通るため、これが無効だと数MB級/手のログが stdout→Cloud Logging へ流入し課金が嵩む。
# フロント配信（to_dict / action_events）はこの print と別系統なので表示・通信には一切影響しない。
ENV OPCG_LOG_SILENT=1
# 体感最適化（計画キャッシュ・ポンダリング・投機）と PyPy ワーカー、L1 の探索ノブ
# （OPCG_PIMC_WORLDS／OPCG_HARD_PER_MOVE_BUDGET）は撤去した（2026-09-07・§16.3-A3／§16.3-8）。
# いずれも「Python の decide が秒オーダーだったこと」への手当てで、Rust の decide（数十 ms）では
# 意味が無い。ロールバックは tag `py-engine-final` を checkout する。
# --no-access-log: リクエスト毎の uvicorn アクセスログを停止（Cloud Run のリクエストログとも重複するため）。
CMD ["sh", "-c", "uvicorn opcg_sim.api.app:app --host 0.0.0.0 --port $PORT --no-access-log"]
