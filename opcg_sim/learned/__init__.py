"""学習側（numpy）の正本——**ネットの定義と符号化の仕様**（計画 `docs/rust_engine_plan.md` §17）。

    encoder.py         符号化 v12/v13（盤面 → scalars/field/card_idx）と語彙
    n_rel_feat.py      トークン状態 S（22 枠 × S_DIM）と関係 R
    n_eff.py           効果構造の符号化（カード表 5 種・候補特徴 F_CAND）
    n_rel.py           NRel（N 系 v3）の forward／backward と npz の入出力
    effect_features.py／leader_feat.py   カード・リーダーの静的特徴
    config.py          探索と学習が共有する既定値（C_PUCT・sims・データ窓）
    train/             訓練・評価帯・holdout の CLI（`python -m opcg_sim.learned.train.xxx`）

**エンジンを import しない**（models だけに依存する）＝ここは「数値の仕様」であって
ルールでも探索でもない。serve の forward は Rust（`rust/opcg_engine/src/net/`）が持ち、
本モジュール群は**訓練とその検算**の正本として残る（2026-09-07・第 2 段
`rs-archive-cutover` で `opcg_sim/src/learned/` から移した）。
"""
