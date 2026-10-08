# PROGRESS（claude/fix-death-median・作業セッションの再開用）

指示: 時計 τ = min(τ_歩き, τ_med)（① 報告 §6.1）を既定に。報告 `docs/reports/2026-10-08_death_median.md`。

作業場所: scratchpad（/tmp/claude-0/-home-user-opcg-sim-backend/475426a6-90e5-56da-a74d-345dc017aea3/scratchpad）
- data/sigma_wip/data/{w39,w41,w42}・before/（d8228c8 の git archive）・before_site/（その wheel）
- bin/run.sh <tree> <site> <tag> tool:set ...（probe rl kv wc tb × real syn）→ runs/<tag>/

## 段
- [x] 1. 式と予告（報告 §1・§2）をコミット
- [ ] 2. 実装（outer.rs::death_median/clock_tau・rd_run・drv_cb・SOLVER_VERSION rd-speed-8）
- [ ] 3. 前の計測（runs/before: probe rl kv wc tb）
- [ ] 3. 後の計測（runs/after）
- [ ] 4. make theory-table → check → 後の器を表の後で
- [ ] 5. 不変量
- [ ] 6. golden 作り直し
- [ ] 7. 報告
- [ ] 8. make test → push・RESULT.json
