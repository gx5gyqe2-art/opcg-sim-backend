# 効果定義の主役は parser.py（日本語テキスト→Ability 変換）。
# LLM生成データ(generated_effects.json)は精度が低いため、ランタイムでは使用しない。
# 必要になった場合は catalog.load_generated_effects() を明示的に呼び出すこと。
