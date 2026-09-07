# opcg-sim-backend

ワンピースカードゲーム シミュレータのバックエンド（FastAPI + 独自ルールエンジン）。
ルールモード（公式ルール自動進行：ソロ／オンライン対戦／CPU 対戦）とフリーモード（自由操作）を提供する。

## ドキュメント

文書は種別（仕様＝正本 / 報告＝時点記録）で分類している。索引は [`docs/README.md`](docs/README.md)。

- [`docs/SPEC.md`](docs/SPEC.md) — システム仕様（アーキテクチャ・コアルール・オンライン対戦・CPU 対戦・効果システム）
- [`docs/TEST_SPEC.md`](docs/TEST_SPEC.md) — テスト仕様（戦略・スイート・効果検証ハーネス・品質ゲート）
- [`docs/parser_v2.md`](docs/parser_v2.md) — カード効果パーサ設計
- [`docs/leader_specs/`](docs/leader_specs/README.md) — 全137リーダーの個別仕様

## クイックスタート

```bash
make test    # テスト（全カード構造不変条件・挙動ベースラインも含む。コマンドの正本は Makefile）
```

## 学習（CPU のネットを訓練するとき）

対戦・API を動かすだけなら不要。**PyTorch は任意依存**で、入っていれば NRel の訓練器が
torch（CPU）経路を使い、無ければ numpy の手書き backward に落ちる（どちらでも同じ npz が出る）。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版（軽い）
# 上の index に到達できない環境では PyPI 版でよい（CPU 実行に違いは無い）: pip install torch

OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.n_rel_train train \
  --in "<dump v2 のディレクトリ（glob 可）>" --epochs 2 --ablate rel --out nrel.npz
#   --backend torch|numpy   既定 torch（import できなければ警告して numpy）
#   --threads N             torch のスレッド数（既定 0＝全コア）
```

1 シャード 1 エポックの実測は numpy 146.7 秒 → torch 34.7 秒（4 コア・4.22 倍）。
数字と等価性の照合は [`docs/reports/2026-09-07_train_torch.md`](docs/reports/2026-09-07_train_torch.md)。
