# 例: opcg_sim/api/app.py の起動時処理
from ..core.effects.catalog import load_generated_effects

# アプリ起動時に1回だけ実行
load_generated_effects()
