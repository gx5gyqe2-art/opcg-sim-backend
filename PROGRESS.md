# PROGRESS（claude/diag-survival-split・課題①の大きい塊の切り分け）

- [x] 起点 d8228c89 を diag-late-winner に合流・器（`m2.resolve`＝守る側の計算の診断の解き直し・`OPCG_M2_PROBE` のときだけ）
- [x] 時計の行に守る側の計算の入力（`din`）と局の鍵（seed・t・who・own_left）を足す
- [x] 実 w41・合成 w39+w42 の計器（①報告の要約と一致）
- [x] `tests/scripts/survival_split.py`・`survival_split_sum.py`: 打ち切りつきの較正・段ごとの 2×2・出どころ・今のターンの解き直し
- [x] `make test` green（cargo 473 passed/1 ignored・pytest 622 passed/12 skipped）
- [x] 報告 `docs/reports/2026-10-08_survival_split.md`・RESULT.json・push
