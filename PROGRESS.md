# PROGRESS（claude/fix-death-median・作業セッションの再開用）

指示: 時計 τ = min(τ_歩き, τ_med)（① 報告 §6.1）を既定に。報告 `docs/reports/2026-10-08_death_median.md`。

作業場所: scratchpad（/tmp/claude-0/-home-user-opcg-sim-backend/475426a6-90e5-56da-a74d-345dc017aea3/scratchpad）
- data/sigma_wip/data/{w39,w41,w42}・before/（d8228c8 の git archive）・before_site/（その wheel）
- bin/run.sh <tree> <site> <tag> tool:set ...（probe rl kv wc tb × real syn）→ runs/<tag>/

## 段
- [x] 1. 式と予告（報告 §1・§2）をコミット
- [x] 2. 実装 f3a0bb5c（outer.rs::death_median/clock_tau・rd_run・drv_cb・SOLVER_VERSION rd-speed-8）
- [x] 3. 前の計測（runs/before: probe rl kv wc tb）
- [x] 3. 後の計測（runs/after）
- [x] 4. make theory-table → check → 後の器を表の後で
- [x] 5. 不変量
- [x] 6. golden 作り直し
- [x] 7. 報告
- [x] 8. make test → push・RESULT.json

## メモ
- after/（f3a0bb5c の git archive）・after_site/（その wheel）。比較: python bin/cmp_clocks.py runs/before/probe.real.clocks.jsonl runs/after/probe.real.clocks.jsonl
- 所要の比較は最後に冷たいディスクで probe:real を前後とも単独で回し直す（前の probe:real は cargo test と重なった）
- clippy の ops.rs:278 collapsible_match は既存（新しい clippy）・make test には入らない
- 前: probe real/syn・rl real 済（rl syn は 2h 越えで止めた→再実行要）。後: probe real 済（的中 .7099・不変量 OK・動いた時計 4761/7094）
- 鎖: after probe:syn rl:real 実行中 → 次 before rl:syn wc:real → before wc:syn kv:real kv:syn → before tb:real tb:syn → table → after rl/kv/wc/tb
- 前は全部済（probe rl wc kv tb × real syn）。後: probe real/syn・rl real 済（rl real は前とビット同じ）
- 表: sigma 段済（sigma_rel_whole.theory だけ動いた 実 .387→.4695 合成 .3875→.4459）。w_bar 段: tb_real 済・tb_syn は 2h で止まった→再開中（table/all.out）
- 次: 表をコミット → after2（表入りの木の archive＋after wheel）で wc kv tb rl:syn → 所要の比較（cb real 前後を冷たく単独）→ 報告 → make test → push
- 完了: 報告・RESULT.json・make test green（cargo 474/1 ignored・pytest 622/12 skipped）
