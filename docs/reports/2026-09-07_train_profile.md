# NRel 学習の時間・メモリ内訳（2026-09-07・計測 WP `train-profile`）

`docs/rust_engine_plan.md` §18.1。**全学習は回していない**——1 シャード・固定 200 バッチの部分計測で、
全体は「1 行あたり × 台帳の行数」で外挿する。学習コード（`tests/scripts/n_rel_train.py`）と
エンジンは無変更。計測用の複製は `tests/scripts/train_profile.py`、torch プロトタイプは
`tests/scripts/train_profile_torch.py`（どちらも新規）。

（本文は計測結果で確定。以下 TBD）
