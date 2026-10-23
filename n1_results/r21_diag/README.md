# r21 診断（§7.38）

報告 `docs/reports/2026-10-23_r21_diag.md`・`RESULT.json`（ブランチ直下）。

- `align.json` — 照合（`tests/scripts/r21_teacher_align.py`）の全点の表（ずれ 0）。
- `teacher_compare.json` — 教師（q_improved）の性質を r11a の教材（波 32 の再分析）と並べた。
- `teacher_prior_noise.json` — π' の P に記録の `pol_p`（Dirichlet 後）と深い読みの雑音なしの P を使った差。
- `train_r21{b..g}.log` — 訓練の記録。ネットは `n1_results/nrel_r21{b..g}{,.ep0,.ep1}.npz`。
- `band_final_rean.{json,log}` — 再分析の点の holdout（波 49＋50・5,484 点）で r13d／r11a／r20／r21／r21b〜g。
- `band_final_w50.{json,log}` — 波 50 の holdout で同じ 10 本。
- `band_diag_*.json`・`band_diag2_*.json` — 途中の帯（final と同じ値の部分集合）。
