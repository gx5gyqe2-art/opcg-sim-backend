# Switch inventory for the theory instruments (read-only audit, tip f58d6b2e)

Audited tree: `origin/claude/cpu-spec-improvements-yw91jd` @ `f58d6b2e` (throwaway worktree, nothing edited). Nothing here deletes anything.
Companion: `rust_design.md` (kernel port). Line numbers are for that commit. Ledger = `docs/cpu_theory_gap.md`; `docs/reports/*` are immutable snapshots and are never in a to-edit list.

## 0. How to read this / what is and is not covered

* **Covered**: every module-level `*_MODE(S)` / flag / frozenset / tuple-of-strings switch in `tests/scripts/*.py` and `tests/harness/*.py` that selects between formulas of the theory instruments (found by AST scan, 84 switch variables or CLI-only switches, below), every `choices=` argparse flag (175 occurrences, all map to the variables below or to measurement-arm tuples), every env toggle (`os.environ`/`getenv`) under `tests/scripts`, `tests/harness`, `opcg_sim`, and the pending N-4 switches that exist only on local branch `n4-work`.
* **Not switches (left out of the counts)**: `SLOPES` (output labels), `REMOVAL_OBSERVABLE` (capability constant in `exit_ledger.py`), measurement-arm tuples (`mu_branch.CBAR_MODES/PLAY_KIND`, `price_scaling.CONF_BANDS/SPLITS`, `nu_measure.BAND_MODES/SCHEMES`, `win_calib --scale`, `*_BANDS` analysis axes), `EX_STATE_BUDGET` (a numeric semantic parameter that is part of every cache key), engine/serve flags in `opcg_sim/learned/config.py` and `opcg_sim/loop/engine.py` (search behaviour, not theory instruments), and thread-count env vars. `opcg_sim` contains **no** theory-formula switch.
* **Env toggles for the instruments**: only `OPCG_PLAN_STORE` (plan_store.py:149, KEEP, infrastructure). The `MIRROR=0` environment variable that help text, ledger (L2385) and the H-4 report (L4) name **does not exist in any code** (`grep environ` finds nothing in theory_bridge/relative_ledger/transition_ledger); the real switch is `--mirror off` / `theory_bridge.MIRROR_ME = False`. Fix the three texts or implement it; either way it is a documentation defect, not a deletion candidate.
* **Evidence quality** is graded per row: `U` = verbatim user decision found (date and quote in the cited ledger/report line); `A` = statement only in agent-written ledger/report/code comment; `C` = code comment only. No row has a verbatim *user instruction to keep an old form* except T150 (section 8) and, arguably, N-4's `DECK_COUNTER_MODE=printed` on `n4-work` (the commit message attributes 'printed stays as a switch' to the 2026-10-04 decision). The retention of old forms is a **process rule** (ledger section 0.7 step 2, user instruction 2026-09-17; section 9.2 header) enforced by one **identity test** (section 4).
* **Line counts are estimates** (AST spans of switch definitions, setters, argparse calls, `if` branches that mention the switch, plus hand-identified helper functions that exist only for the dead value; tests = spans of test functions that mention a non-default literal on the same line as the switch name, so a lower bound on touched tests and an upper bound on lines that can simply be deleted because many of those functions also assert defaults). Treat them as +-25%.
* I could not run the full suite or the tools (shared machine, instructions); the byte-identity procedure in section 10 is therefore a plan, not a result.

## 1. Result in numbers

| class | variables with at least one value of the class | non-default values |
|---|---|---|
| KEEP-baseline | 7 | 8 |
| KEEP-live | 13 | 16 |
| KEEP-scaffold | 5 | 6 |
| DELETE-dead | 63 | 101 |
| DELETE-after-port | 1 | 1 |
| **total inventoried** | **84 variables** (5 of them have values of two different classes) | **132 values** |

DELETE-dead by wave (value counts): A=12, B=57, C=29, U=3.

* **KEEP-baseline (7 variables)**: `CUT_PRICE_MODE=flat`, `CUT_TAKE_MODE=mu`, `THETA_HAND_MODE=cuttable_forced`, `MIRROR_ME=False`, `DEFENDER_POWER_MODE=token`, `GUARD_S_COST_MODE=curve`, `F_PRICING_FIXES=none` - exactly the six old values that `test_switches_off_match_recorded_base_outputs_of_079e73b8` names, plus MIRROR (H-4). Plus, **not on the tip**: `DECK_COUNTER_MODE=printed` (n4-work). **One lever decides 3 of the 7**: if the user retires the 079e73b8 identity test and its three fixtures (`tests/fixtures/f_identity/*`), `F_PRICING_FIXES=none`, `GUARD_S_COST_MODE=curve` and `DEFENDER_POWER_MODE=token` lose their only enforcer and become DELETE; `CUT_PRICE_MODE`/`CUT_TAKE_MODE`/`THETA_HAND_MODE`/`MIRROR_ME` stay baselines because live measurement jobs still use them (section 4).
* **KEEP-live (13 variables)**: undecided candidates (`SLOPE_BLOCK_MODE=on`, `SLOPE_TAKE_MODE=life`, `THETA_SIDE_MODE=symmetric`, `KAPPA_SIGMA_MODE=match`, `ATTACK_ABILITY_MODE`, `PASSIVE_BODY_MODE`, T18-gated `--theta-mode`/`--nu-targets`/`SEQ_MODE`, `D_MODE=theory`), live measurement settings (`PRE_SETTLE_MODE`, `FLOW_PRICING/LEDGER_FLOW_PRICING`) and one stale candidate (`RATE_DECAY_MODE=ko`, unmeasured since the T128 fix). Plus, not on the tip: `SEARCH_VALUE_MODE=joint` (awaiting N-5).
* **KEEP-scaffold (5)**: `SURV_MODE=once`, `W_MODE=flat`, `CBAR_MODE=loose`, `OPTION_MODE=off` are forced by the suite-wide autouse fixture in `tests/conftest.py:43-79`; `SPEED_MEMO=False` is the oracle of `test_speed_memo`. They are dead in production but deleting them means rewriting ~100 tests' closed-form arithmetic - not part of this effort.
* **DELETE-dead: 101 values in 63 variables** (wave A kernel-touching 12 values, B clean 57, C heavy 29, U contradicts an explicit keep 3). **Estimated removal** (section 7 has the per-wave split): code ~1,440 lines in `tests/scripts` plus ~540 lines of whole dead scripts (`purse_race.py` 215, `bias_budget.py` 321); tests ~3,080 lines touched (upper bound; +214 for whole `test_bias_budget.py`); 47 of the ~65 rows of ledger section 9.2; 61 TEST_SPEC lines; 43 lines of `game_theory.md`.
* **DELETE-after-port: 1** (`EX_LAYER_COUNT=False`), plus the optimized Python solver functions themselves (not switches) if the user chooses to remove them after parity: list in `rust_design.md` section 9.

### Top findings that change the plan

1. **A single test is the real keeper of the old values**: `tests/test_attack_passive_wiring.py:754-796` (+ `tests/fixtures/f_identity/base_*_079e73b8.json`, 88 KB) re-runs three tools with exactly six old values set and demands byte equality with outputs recorded at commit 079e73b8. Each adoption wave since (N-2, 2b, N-3, H-4) has *edited that test to name the old value* (`TB.set_guard_s_cost_mode('curve')`, `T.set_defender_power_mode('token')`, `CP.set_cut_price_mode('flat'); CP.set_cut_take_mode('mu')`, `_CB.set_theta_hand_mode('cuttable_forced')`, `EV.set_f_pricing_fixes('none')`). Delete those values and the test dies; that needs an explicit user decision (section 11, D1).
2. **The old baselines were used as recently as today, and a measurement that depends on stores is running**: the H-4 flat-price full-corpus reproduction (`--cut-price flat --cut-take mu`, `claude/h4-misc` job 30) finished 2026-10-04 and `h4_tables.py` still expects its `cb_flatreal_rule_don.json`; `claude/sigma-wip` is running the sigma re-measure (two sessions, real and synthetic) with default switches **and `OPCG_PLAN_STORE`**. **The plan store key contains a hash of every file in `tests/scripts/`** (plan_store.py:40-47 `source_digest`, :120-128 `key_of`): *any* edit of `tests/scripts` - including a pure switch deletion - makes every stored plan cold. Do not start deleting until the sigma-table runs are merged, or first narrow the key to the solver's own source (rust_design.md section 6).
3. **`MIRROR=0` is documentation-only** (see section 0). Three texts promise an env var that was never implemented.
4. **bias_budget.py (321 lines) + test_bias_budget.py (214) are already stale**: the tool refuses to run in the shipped configuration (`static_target()` requires `THETA_RETURN_MODE=off`, `RACE_MODE=static`, `THETA_HAND_PLACE=stock`, `RATE_DECAY_MODE=off`; the shipped `THETA_RETURN_MODE` is `untap` since C-5c). It runs only with `--theta-return off` (the docstring L28-33 already admits 'not default since C-5c'); deleting `THETA_RETURN_MODE=off` leaves only `--no-require-static`, where the decomposition is no longer an identity - treat tool+test as one deletion.
5. **Ledger section 9.2 is stale against the code** on at least `THETA_RETURN_MODE` (table says default `off`, code `untap`), `THETA_HAND_MODE` (first row says `cuttable`; later rows say `rule_don`), and `EX_LAYER_COUNT` has no row. Any deletion PR must rewrite those rows anyway (section 9).
6. **The kernel dependency**: 6 switches are read *inside* the hot path (`RATE_T1_MODE`, `RATE_RUSH_MODE`, `SLOPE_EFFECT_MODE`, `RATE_DON_PAY`, `RATE_RAMP`, and the `fixed` field of `rule_don_purse`) and 2 gate the entry (`DON_PURSE_MODE != all`, `RATE_DON_MODE == purse`). Deleting their dead values *before* the port removes 12 dead values (11 branches) that the Rust code and its parity matrix would otherwise carry (wave A).

## 2. Classification rules used

* **KEEP-baseline** = the immediately previous default of an adoption on/after 2026-10-01 (N-3, 2b, H-4, MIRROR, N-4) **or** a value named by the 079e73b8 identity test; *and* the ledger/report says it reproduces older numbers. Evidence in section 4.
* **KEEP-live** = candidate with no close/adopt decision, or a decision that explicitly parks it with a follow-up (T18-gated, 'recommended but held', F-5 'stay as switches'), or a live measurement setting.
* **KEEP-scaffold** = non-default value forced onto the whole suite by `tests/conftest.py`, or a bit-identity oracle.
* **DELETE-dead** = alternative that was rejected or superseded by >= 2 later adoption waves, with no dependants except tests/scripts that exist to reproduce it. Waves: **A** touches the kernel (delete before the port); **B** clean (<~150 test lines, no fixture/σ dependence); **C** heavy or dependency-bound (σ-table fixtures `harm_profile.json`, >150 test lines, other tools depend); **U** conflicts with an explicit user 'keep the code' decision.
* **DELETE-after-port** = dead only when the Python solver is removed.
* 'confirm' in a note = my classification rests on `A`/`C` evidence only; ask before deleting.

## 3. Master table

Columns: `class(wave)` per value-group; `kernel` = does the Rust port read/gate it (`rust_design.md`); `est` = est. removable code lines / lower-bound test lines for the DELETE groups of that variable; `refs` = test files (count of name mentions) / ledger lines / TEST_SPEC lines / # reports. Full reference index in Appendix A.

| # | variable (file:line) | default | values and class(wave) | kernel | est code / tests | refs: tests / ledger / TEST_SPEC / reports | evidence |
|---|---|---|---|---|---|---|---|
| 1 | `THETA_HAND_MODE` (`crossing_bridge.py:106`) | `rule_don` | `cuttable_forced` -> **KB**<br>`rule_don_purse` -> **DD(A)**<br>`rule` -> **DD(C)**<br>`cuttable_seq` -> **DD(C)**<br>`cuttable_cx`, `cuttable`, `count`, `quality`, `play`, `guard` -> **DD(C)** | Y-gate (rule_don_purse reads actx['fixed']) | 205 / 343 | crossing_bridge:63, cut_price:6, relative_ledger:4, two_curves_state:1, attack_passive_wiring:1 / L316/337/338/398/521+ / L294/549/597 / 12 | cuttable_forced: previous default 2026-09-20..10-04; byte-identical to N-3 commit 74a51363; pinned by the 079e73b8 identity test; rule_don_purse: never measured (H-4 report L65/L68 'priority low, not needed for adoption'); needs actx['fixed'] plumbing in the kernel; rule: H-4 DP without attacker attach; never measured; VALUE only - the code path (`_rule_hand_term`, `rule_guard_plan`) stays as the fallback of rule_don; 5 test functions select it via the setter; cuttable_seq: T158/H-1: adoption 'pending H-2' (ledger L521), H-2 never ran, superseded by H-4 - confirm; cuttable_cx/cuttable/count/quality/play/: T76 rejected (ledger 0.6 'remaining 2: not entered'); cuttable = default 09-17..09-20; cx = T99 over-trimmed, fixed by T100; >=3 generations old |
| 2 | `THETA_HAND_PART` (`crossing_bridge.py:112`) | `(dict keyed by THETA_HAND_MODE)` | `dtotal`, `dh`, `dg` -> **DD(C)** | - | 4 / 11 | crossing_bridge:7 / L316 / L294/549/597 / 2 | dtotal/dh/dg: per-card price parts of count/quality/play/guard; dies with THETA_HAND_MODE legacy values |
| 3 | `SLOPE_MODE` (`crossing_bridge.py:130`) | `hand` | `board` -> **DD(B)** | - | 10 / 0 | crossing_bridge:4 / L337/2134/2135/2338 / L294/597 / 2 | board: T77 user decision 2026-09-17 'change 1' made hand default; board = pre-T77 numbers |
| 4 | `THETA_HAND_BLOCKER_MODE` (`crossing_bridge.py:188`) | `on` | `off` -> **DD(B)** | - | 21 / 24 | crossing_bridge:8 / L1365/1390/2341/2349 / L294/597 / 4 | off: T106; user decision 2026-09-19 'fix 2 to the correct form'; off = hand blockers not counted (wrong by rule) |
| 5 | `THETA_DON_MODE` (`crossing_bridge.py:222`) | `rule` | `off`, `blocker` -> **DD(B)** | - | 12 / 92 | crossing_bridge:12 / L1373/2343 / L294/597 / 3 | off/blocker: T110, user 2026-09-19 'fix it'; off/blocker = the wrong don budgets |
| 6 | `RATE_DON_MODE` (`crossing_bridge.py:293`) | `flow` | `off` -> **DD(A)**<br>`purse` -> **DD(A)** | Y-gate (attacker_ctx) | 28 / 47 | don_walk:6, crossing_bridge:4 / L1093/1288/1307/1316/2355 / L296/597 / 7 | off: pre-T114 walk growth flow*(j-1); ledger L1307 uses it to reproduce older numbers (>=3 generations old); same lines as purse, so removed with it; purse: T114 'not adopted'; attacker_ctx returns None for it (kernel never sees it) - dies to simplify the gate |
| 7 | `RATE_DON_PAY` (`crossing_bridge.py:296`) | `True` | `False` -> **DD(A)** | Y-read (rules_sched.left_of) | 1 / 0 | rd_speed:1 / L1288 / - / 2 | False: T114 design-first version (do not charge the drawn card's don); refuted - upper bound only |
| 8 | `RATE_RAMP` (`crossing_bridge.py:299`) | `0.0` | `nonzero` -> **DD(A)** | Y-read (rate_at sched path) | 4 / 0 | don_walk:2 / - / - / 3 | nonzero: placebo-only linear ramp (T114 refutation control) |
| 9 | `THETA_BODY_MODE` (`crossing_bridge.py:454`) | `blockers` | `all`, `attackable`, `none` -> **DD(C)** | - | 38 / 155 | crossing_bridge:27, kappa_vector:1, rate_tracking:1 / L318/332/997/1043/1051+ / L294/549/597 / 8 | all/attackable/none: T82/T83/T97: all+attackable adopted-then-reverted by user 2026-09-18 ('make it theoretically correct'); none = T129 bundle, closed 'shipped is best'. harm_profile.json keys sigma_t/w_bar still carry attackable/all |
| 10 | `THETA_RETURN_MODE` (`crossing_bridge.py:516`) | `untap` | `off` -> **DD(C)** | - | 17 / 64 | bias_budget:7, crossing_bridge:6, theta_return:6, relative_ledger:1 / L512/2354 / L294/300/328/597 / 5 | off: C-5c user 2026-09-25 'a'; off = 5-tuple state path in kappa_vector/relative_ledger/transition_ledger; bias_budget.py already refuses the default config - confirm |
| 11 | `EX_LAYER_COUNT` (`crossing_bridge.py:1871`) | `True` | `False` -> **DP** | Y (kernel always counts layers) | - | - / - / - / 0 | False: RD-speed 'stepwise shrink only' fallback; no test toggles it (comment claims tests compare both) |
| 12 | `THETA_HAND_PLACE` (`crossing_bridge.py:2349`) | `stock` | `shield` -> **DD(B)** | - | 46 / 46 | crossing_bridge:6, bias_budget:2 / L1070/1404/2350 / L328/597/609 / 5 | shield: T102: cost small, direction split, 'kept as switch'; superseded by THETA_HAND_WINDOW=horizon (T116, user 2026-09-20) - confirm |
| 13 | `THETA_HAND_WINDOW` (`crossing_bridge.py:2364`) | `horizon` | `off`, `fixpoint` -> **DD(B)** | - | 36 / 0 | crossing_bridge:3, don_walk:2 / L1289/1307/1323 / L597 / 3 | off/fixpoint: T116 user 2026-09-20 '3 all'; fixpoint measured worse than horizon; off = pre-T116 |
| 14 | `THETA_SIDE_MODE` (`crossing_bridge.py:2441`) | `legacy` | `symmetric` -> **KL** | - | - | crossing_bridge:8, relative_ledger:2 / L928/956 / L323/567/597 / 2 | symmetric: T133/T134: 'kept pending recommendation' (ledger L928); interacts with MIRROR=off baseline |
| 15 | `SLOPE_BLOCK_MODE` (`crossing_bridge.py:2830`) | `off` | `on` -> **KL** | - | - | crossing_bridge:10 / L323/1051/1058/2358 / L294/597 / 5 | on: T92 rule-correct form, numbers slightly worse; ledger says 're-enter in a pair later' (L323) |
| 16 | `RATE_THROUGH_MODE` (`crossing_bridge.py:2879`) | `off` | `cut`, `cut_block` -> **DD(B)** | - | 59 / 33 | crossing_bridge:7 / - / L597 / 1 | cut/cut_block: T131: built, measured worse on 4/4 metrics in both records; closed with T132 'stop improving A' |
| 17 | `SLOPE_TAKE_MODE` (`crossing_bridge.py:2950`) | `const` | `life` -> **KL** | - | - | crossing_bridge:10, relative_ledger:2 / L398/929/943 / L323/567/597 / 1 | life: T134 (3): recommended 'adopt' (ledger L398), not adopted; needed by --d-mode theory A/B |
| 18 | `SLOPE_HAND_MODE` (`crossing_bridge.py:3019`) | `flow` | `stock` -> **DD(B)** | - | 33 / 22 | crossing_bridge:6 / L322/1073/1462/2356/2357 / L294/597 / 4 | stock: T93 user 2026-09-18 'make it default'; stock = counts a stock as income (wrong) |
| 19 | `SCHED_T1_MODE` (`crossing_bridge.py:3142`) | `game` | `walk` -> **DD(B)** | - | 10 / 34 | crossing_bridge:10 / L398/877/888 / L294/598 / 2 | walk: T152 user 2026-09-24 '1 is default'; walk = control for pre-T152 numbers |
| 20 | `RACE_MODE` (`crossing_bridge.py:3303`) | `static` | `net`, `deck`, `deck_shield` -> **DD(B)** | - | 54 / 94 | crossing_bridge:11, bias_budget:5 / L324/325/398/928/1401+ / L294/328/597 / 10 | net/deck/deck_shield: T90/T91/T104: deck refuted both records (ledger L398), deck_shield 'right in rule but inert', net lets play style into the formula |
| 21 | `RATE_WALK_MODE` (`crossing_bridge.py:3347`) | `grow` | `flat` -> **DD(B)** | - | 24 / 22 | crossing_bridge:7 / L321/1457/2356 / L294/597 / 4 | flat: T94 user 2026-09-18; flat = pre-T94 |
| 22 | `RATE_DECAY_MODE` (`crossing_bridge.py:3370`) | `off` | `ko` -> **KL** | N (read by model_horizon, which stays in Python) | - | crossing_bridge:9, bias_budget:2 / L1092/1094/1450/2355 / L294/328/597 / 3 | ko: T95; was a dead switch from T114 to T128 (ledger L1092) and 'not re-measured in the current config' (stale candidate: re-measure or delete) |
| 23 | `RATE_RUSH_MODE` (`crossing_bridge.py:3395`) | `on` | `off` -> **DD(A)** | Y-read (rules_steps) | 14 / 37 | crossing_bridge:12 / L1365/1400/2341/2348/2349 / L294/597 / 5 | off: T103 user 2026-09-19 'fix 2'; rule-correct 'on' is default |
| 24 | `RATE_T1_MODE` (`crossing_bridge.py:3409`) | `on` | `off` -> **DD(A)** | Y-read (rate_at, rules_sched) | 18 / 74 | crossing_bridge:11, rd_speed:3, kappa_vector:1 / L1364/1365/1398/2341/2347+ / L293/294/570/597 / 8 | off: T103 user 2026-09-19; used by test_rd_speed only as a sample switch for store-key sensitivity |
| 25 | `SLOPE_EFFECT_MODE` (`crossing_bridge.py:3428`) | `hand` | `off`, `on` -> **DD(A)** | Y-read (rules_steps eff_on) | 36 / 22 | crossing_bridge:10 / L1073/1365/1381/1393/2345+ / L597 / 6 | off/on: T105/T108 user 2026-09-19 '1 default'; off/on = earlier steps |
| 26 | `DON_PURSE_MODE` (`crossing_bridge.py:3469`) | `all` | `off`, `one` -> **DD(A)**<br>`race` -> **DD(A)** | Y-gate (attacker_ctx returns None unless 'all') | 199 / 46 | crossing_bridge:8, don_walk:3 / L509/938/948/1366/1377+ / L294/296/592/597 / 6 | off/one: T109 user 2026-09-19; earlier purse forms; race: T111 'not adopted' + E-1 audit: its race_alloc is the only double count; E track ended by user 'let's stop' (ledger L945-949) |
| 27 | `PRE_SETTLE_MODE` (`crossing_bridge.py:3878`) | `off` | `on`, `game` -> **KL** | - | - | crossing_bridge:11, win_calib:11, pre_settle_asymmetry:4, rd_speed:4, settle_cond_floor:2 / L507/585/845 / L293/294/303/304/323+ / 8 | on/game: sigma-table re-measure runs `win_calib --pre-settle on` (claude/sigma-wip run_sigma.sh) |
| 28 | `OPP_CLOCK_MODE` (`crossing_bridge.py:3920`) | `mirror` | `prev_start` -> **DD(B)** | - | 20 / 20 | crossing_bridge:7, rd_speed:1 / L398/405/845/867/875 / L294/598/607/608 / 4 | prev_start: T151 user 2026-09-24 'as recommended'; prev_start = control for pre-T151 numbers |
| 29 | `CUT_PRICE_MODE` (`cut_price.py:74`) | `joint` | `flat` -> **KB**<br>`joint_slice`, `joint_theta`, `joint_floor` -> **DD(B)** | Y-input (kernel supports avg and flat prices only) | 14 / 0 | cut_price:2, relative_ledger:1, attack_passive_wiring:1, speed_memo:1 / L2382 / L286/294/563 / 2 | flat: previous default (<=2026-10-01); `--cut-price flat --cut-take mu` named in N-3 report L12 and H-4 report L5; flat full-corpus reproduction done on claude/h4-misc job 30 (2026-10-04); 079e73b8 identity test; joint_slice/joint_theta/joint_floor: diagnostic/control arms of the 2026-09-30 review (code comment L66-70); joint_slice raises in cut_card_price for rule* |
| 30 | `CUT_TAKE_MODE` (`cut_price.py:78`) | `gbar` | `mu` -> **KB** | Y-input | - | cut_price:4, speed_memo:4, relative_ledger:1, attack_passive_wiring:1 / L2383 / - / 2 | mu: same adoption as CUT_PRICE_MODE (ledger L2383); pinned by the 079e73b8 identity test |
| 31 | `PRICING_FIXES (L)` (`effect_value.py:917`) | `all (7 names)` | `legacy`, `-name subsets` -> **DD(C)** | - | 3 / 0 | effect_value:4 / - / - / 0 | legacy/-name subsets: L user 2026-09-26 '3a'; legacy = pre-L values, bound by test_effect_value; not in the 079e73b8 test (it was recorded after L) |
| 32 | `F_PRICING_FIXES (F)` (`effect_value.py:1026`) | `all (6 names)` | `none/legacy`, `subsets` -> **KB** | - | - | attack_passive_wiring:3, effect_value:1 / L2379 / - / 1 | none/legacy/subsets: F-5 user 2026-09-30; `none` is bit-identical to 079e73b8; pinned by test_attack_passive_wiring identity test + tests/fixtures/f_identity |
| 33 | `SEARCH_PRICE_MODE` (`effect_value.py:2320`) | `plan` | `sel` -> **DD(B)** | - | 10 / 21 | search_price:5 / L345/2143/2366 / L547/549/557/564/613 / 6 | sel: T68 user 2026-09-17 'replace'; sel = pre-T68 |
| 34 | `PLAY_NOW_MODE` (`effect_value.py:2348`) | `hand` | `full` -> **DD(B)** | - | 9 / 31 | search_price:6 / L343/2141/2368 / L547/549/557/613 / 3 | full: T70 user 2026-09-17; full = pre-T70 |
| 35 | `COST_AFFORD_MODE` (`effect_value.py:2449`) | `check` | `off` -> **DD(C)** | - | 8 / 197 | search_price:29 / L516/517/519/2373 / L289 / 5 | off: D-2..D-4 user 2026-09-25; 11 test functions (~200 lines) in test_search_price pin 'off' algebra |
| 36 | `FLOW_PRICING / LEDGER_FLOW_PRICING` (`effect_value.py:204,219`) | `option / exercise` | `exercise (decide)`, `option (ledger)` -> **KL** | - | - | - / - / - / 0 | exercise (decide)/option (ledger): T58 two-role convention (decide=option, count=exercise); both values are live by design |
| 37 | `INFLOW_MODE` (`hand_plan.py:187`) | `on` | `off` -> **DD(B)** | - | 12 / 72 | hand_plan:3, guard_hand_cost:2, hand_joint:1 / L343/2141/2369 / L549/557/560/613 / 3 | off: T70 user 2026-09-17; off = static v |
| 38 | `COND_CLOCK_MODE` (`hand_plan.py:304`) | `on` | `off` -> **DD(B)** | - | 9 / 52 | hand_plan:5 / L340/2138/2370 / L549/557/560/613 / 2 | off: T73 user 2026-09-17; off = state-now for all t |
| 39 | `ATTACK_REST_MODE` (`kappa_vector.py:134`) | `return` | `off`, `body` -> **DD(C)** | - | 13 / 66 | kappa_vector:10, attack_axis_by_result:8, theta_return:6 / L510/512 / L298/300/326 / 3 | off/body: C-5c user 2026-09-25; code comment says body 'kept as switch' (agent text); needs 7-tuple state - tests in test_kappa_vector/test_theta_return |
| 40 | `D_MODE` (`kappa_vector.py:157`) | `curve` | `theory` -> **KL**<br>`clock`, `curve_scaled` -> **DD(C)** | - | 79 / 302 | kappa_vector:28, transition_ledger:18, relative_ledger:15, attack_axis_by_result:8, theta_return:4 / L398/933 / L298/323/566/567/571 / 13 | theory: the only D reading that moves with speed-side switches (SLOPE_TAKE=life A/B used it, ledger L943); clock/curve_scaled: T49/T75 earlier readings; 34 test functions mention D_MODE - confirm which values they pin |
| 41 | `LETHAL_HAND_MODE` (`lethal_rule.py:92`) | `actual` | `share` -> **DD(B)** | - | 26 / 13 | lethal_rule:4, settlement_split:2 / - / L534/588 / 2 | share: T117 old convention |
| 42 | `LETHAL_STOP_MODE` (`lethal_rule.py:95`) | `max` | `econ` -> **DD(B)** | - | 13 / 11 | lethal_rule:4, settlement_split:2 / L491 / L588 / 3 | econ: T130 rule variant, not adopted |
| 43 | `LETHAL_LIFE_MODE` (`lethal_rule.py:98`) | `draw` | `off` -> **DD(B)** | - | 15 / 38 | lethal_rule:8, settlement_split:2 / - / L588 / 2 | off: T117 old convention |
| 44 | `AVG_COUNTER_MODE` (`lethal_rule.py:103`) | `rules` | `printed` -> **DD(B)** | - | 16 / 19 | lethal_rule:11 / L507 / L322 / 1 | printed: B-fix adopted 2026-09-25 ('switch to it'); NOTE same kind of baseline as DECK_COUNTER_MODE=printed which the user kept on 2026-10-04 - decide both together |
| 45 | `HAND_MEAS_MODE` (`price_realised.py:147`) | `quality` | `count` -> **DD(B)** | - | 9 / 13 | price_realised:5 / L344/2142/2367 / L557/560/613 / 3 | count: T69 user 2026-09-17 |
| 46 | `FREEZE_YARDSTICK` (`price_realised.py:176`) | `False` | `True` -> **DD(B)** | - | 3 / 0 | - / - / - / 0 | True: F-4 measurement option (F closed 2026-09-30) |
| 47 | `ATTACK_SPLIT` (`price_realised.py:212`) | `False` | `True` -> **DD(B)** | - | 6 / 0 | - / - / - / 0 | True: F-4 output option |
| 48 | `SEQ_MODE` (`shadow_forbid.py:104`) | `off` | `attack_any` -> **KL**<br>`attack`, `attack_le`, `attack_le_delta` -> **DD(C)** | - | 10 / 173 | t18_arena:14, shadow_forbid:12, theory_trace:1 / L665/669/754/757/821+ / L312/313/315/316/578+ / 10 | attack_any: T155b/T156: the T18 pilot reading; T18 'on hold, re-open after theory is organized'; attack/attack_le/attack_le_delta: T144/T147a intermediate readings superseded by attack_any; CLAUDE.md:221 uses `--seq attack_le` as a wording example |
| 49 | `GUARD_G_MODE` (`theory_bridge.py:137`) | `delta` | `paid`, `zero` -> **DD(B)** | - | 8 / 24 | theory_bridge:5, guard_hand_cost:1 / L327/351/1497/2125/2149+ / L549 / 5 | paid/zero: only meaningful with LEDGER_HARM_MODE=price (ledger L2332: 'normally unused') |
| 50 | `GUARD_COST_MODE` (`theory_bridge.py:147`) | `spent` | `formula`, `spent_all` -> **DD(B)** | - | 23 / 0 | - / L328/349/2126/2147/2334+ / L549 / 3 | formula/spent_all: same ledger-form bundle (T64/T86) |
| 51 | `LAST_TURN_MODE` (`theory_bridge.py:152`) | `keep` | `drop` -> **DD(B)** | - | 12 / 0 | - / L1556/2132 / L549 / 1 | drop: T80 diagnostic arm |
| 52 | `LEDGER_HARM_MODE` (`theory_bridge.py:192`) | `realised` | `price` -> **DD(B)** | - | 26 / 22 | theory_bridge:4 / L327/515/893/1494/2125+ / L272/549 / 6 | price: T87 user 2026-09-18 'default must be the correct one'; price = old ledger writing |
| 53 | `GUARD_PRICE_MODE` (`theory_bridge.py:219`) | `max_attack` | `all_attacks` -> **DD(B)** | - | 13 / 26 | theory_bridge:4 / L328/2126/2333 / L272/549 / 2 | all_attacks: T86, 'held'; third mismatch unresolved; only with price ledger |
| 54 | `ATTACH_LEDGER_MODE` (`theory_bridge.py:234`) | `in_attack` | `increment` -> **DD(B)** | - | 15 / 15 | theory_bridge:6, two_curves:3 / L329/2127/2335 / L272/549 / 2 | increment: T85 user 2026-09-18 'after fixing it' |
| 55 | `PLAY_BOOK_MODE` (`theory_bridge.py:250`) | `next` | `now` -> **DD(B)** | - | 13 / 33 | theory_bridge:7 / L330/397/1511/2128/2336 / L272/549 / 2 | now: T84 user 2026-09-18 'make default' |
| 56 | `GUARD_AFFORD_MODE` (`theory_bridge.py:311`) | `rule` | `lenient` -> **DD(B)** | - | 8 / 47 | guard_hand_cost:12 / L527/2375 / L284/603 / 1 | lenient: G-2 user 2026-09-26: a rule bug-fix; lenient = the bug |
| 57 | `GUARD_S_COST_MODE` (`theory_bridge.py:404`) | `joint` | `curve` -> **KB**<br>`hand` -> **DD(B)** | - | 8 / 85 | guard_hand_cost:11, hand_joint:7, theory_bridge:4, attack_passive_wiring:1 / L527/530/2376 / L272/284/285/549/562+ / 2 | curve: N-2 user 2026-09-26; ledger L530 'old curve via --guard-s-cost curve'; pinned by 079e73b8 identity test; hand: G-2/G-3 intermediate form (user 2026-09-26); 6 test functions in test_guard_hand_cost |
| 58 | `DECISION_ROW_MODE` (`theory_bridge.py:754`) | `main` | `any` -> **DD(B)** | - | 8 / 18 | decision_rows:9 / L518/519/2374 / L288 / 3 | any: D-5 user 2026-09-25 |
| 59 | `MIRROR_ME` (`theory_bridge.py:840`) | `True` | `False` -> **KB** | - | - | relative_ledger:2 / L533/2385 / L294 / 1 | False: H-4 user 2026-10-04 'adopt both'; report L4: `--mirror off`; ledger L533/L2385 'keep byte-identity to 74a51363 by naming the old form'. NOTE the env var MIRROR=0 named in help/ledger/report is NOT implemented anywhere |
| 60 | `SIGMA_FROM_CURVE` (`theory_bridge.py:1034`) | `True` | `False` -> **DD(B)** | - | 6 / 0 | theory_bridge:1 / L2353 / L272/549 / 1 | False: T97 borrowed sigma 1.0 |
| 61 | `SILENT_MODES (--silent)` (`theory_bridge.py:124`) | `zero` | `exclude` -> **DD(B)** | - | 0 / 0 | - / - / - / 0 | exclude: P2 sensitivity (2026-09-14) |
| 62 | `TAKE_MODE` (`theory_order.py:100`) | `lethal` | `const`, `by_life` -> **DD(B)** | - | 9 / 18 | theory_order:5, crossing_bridge:2 / L350/523/2148/2364 / L541/549/555/613 / 7 | const/by_life: T63 user 2026-09-16 '3 only': by_life rejected, const = pre-T63 |
| 63 | `THETA_MODES (--theta-mode)` (`theory_order.py:170`) | `const` | `board`, `max` -> **KL** | - | - | - / - / - / 0 | board/max: T20 'formula decided, judgement awaits T18' (ledger L2184); T18 on hold |
| 64 | `NU_TARGET_MODES (--nu-targets)` (`theory_order.py:179`) | `leader` | `board` -> **KL** | - | - | - / - / - / 0 | board: T20 b: same T18-gated status |
| 65 | `NU_MODE` (`theory_order.py:222`) | `pair` | `base` -> **DD(B)** | - | 9 / 43 | theory_order:5, nu_calib:4, attack_passive_wiring:1 / L239/240/2172/2174/2312+ / L267/540/541/542/543+ / 3 | base: T37 user 2026-09-15 |
| 66 | `SURV_MODE` (`theory_order.py:232`) | `geo` | `once` -> **KS(S)** | - | - | theory_order:10, conftest:3, attack_passive_wiring:2 / L353/2151/2361 / L541/549/555/557/558+ / 11 | once: tests/conftest.py autouse pins once for the whole suite |
| 67 | `SPEED_MEMO` (`theory_order.py:262`) | `True` | `False` -> **KS(S)** | - | - | speed_memo:6 / L532/2387 / L287 / 1 | False: bit-identity oracle of test_speed_memo (like rule_don_ref for RD-speed) |
| 68 | `W_MODE` (`theory_order.py:344`) | `curve` | `flat`, `clock` -> **KS(S)** | - | - | theory_order:5, conftest:3, kappa_vector:1, theory_bridge:1 / L235/335/339/1273/1282+ / L326/541/549/597/604 / 12 | flat/clock: tests/conftest.py autouse pins flat; clock = T49 reading |
| 69 | `CLOCK_HAND_MODE` (`theory_order.py:364`) | `off` | `on` -> **DD(C)** | - | 14 / 46 | theory_order:5, conftest:3 / L336/2134/2389 / L262/541 / 2 | on: T78 'regressed, waiting for an estimator'; premise voided by 0.05 (complete information); only feeds the superseded W_MODE=clock path - confirm |
| 70 | `W_ERR_MODE` (`theory_order.py:423`) | `rel` | `abs` -> **DD(B)** | - | 26 / 0 | settle_cond_floor:11, don_walk:8, theory_order:3, crossing_bridge:1 / L1289/1307/1328 / L549 / 3 | abs: T118 user 2026-09-20 '3 all'; abs gives log-loss 1.3 vs 0.67 |
| 71 | `W_MOVER_MODE` (`theory_order.py:480`) | `half` | `off` -> **DD(B)** | - | 14 / 53 | theory_order:10, settle_cond_floor:8 / L405/845/867/875 / L264/598/602/607/608 / 3 | off: T151 user 2026-09-24 |
| 72 | `SIGMA_FLOOR_MODE` (`theory_order.py:511`) | `off` | `on` -> **DD(C)** | - | 13 / 60 | settle_cond_floor:15 / - / L305 / 1 | on: K-2/K-3 candidate superseded by K-5 `whole`; but harm_profile.json carries sigma_rel_floor and the sigma re-measure is running - do after the merge |
| 73 | `SETTLE_COND_MODE` (`theory_order.py:574`) | `whole` | `off`, `on` -> **DD(C)** | - | 23 / 123 | settle_cond_floor:25, theory_order:3 / L528/2377 / L305 / 1 | off/on: K-5 user 2026-09-26 'a'; on = K-1 (worse); off = pre-K; same sigma-table dependency |
| 74 | `KAPPA_SIGMA_MODE` (`theory_order.py:697`) | `abs` | `match` -> **KL** | - | - | relative_ledger:9 / L1241 / L323/567/600 / 4 | match: T122: correctness fix, effect tiny, 'default kept' (ledger L1241) - undecided |
| 75 | `ACTIVATE_USES_STATE` (`theory_order.py:763`) | `True` | `False` -> **DD(B)** | - | 1 / 0 | theory_order:5 / L2169/2327 / L262 / 1 | False: T42 'sensitivity switch' back to 2026-09-15 valuation |
| 76 | `CBAR_MODE` (`theory_order.py:775`) | `strict` | `loose` -> **KS(S)** | - | - | crossing_bridge:15, theory_order:10, attack_passive_wiring:4, conftest:3 / L352/971/2150/2362 / L264/294/541/555/558+ / 13 | loose: tests/conftest.py autouse pins loose (closed algebra c(1000)=1.00) |
| 77 | `DEFENDER_POWER_MODE` (`theory_order.py:928`) | `rule` | `token` -> **KB** | - | - | hand_plan:5, attack_passive_wiring:1 / L532/2386 / - / 1 | token: N-3 remaining 2b, 2026-10-01 (ledger L532/L2386); pinned by the 079e73b8 identity test |
| 78 | `PLAY_COST_MODE` (`theory_order.py:1013`) | `state` | `flat` -> **DD(B)** | - | 3 / 29 | theory_order:8 / L726/2168/2326 / L541 / 2 | flat: T43 2026-09-16 |
| 79 | `ATTACK_DON_COST_MODE` (`theory_order.py:1172`) | `off` | `opportunity`, `misalloc`, `misalloc_play` -> **DD(U)** | - | 10 / 163 | theory_order:25 / L748 / L601 / 5 | opportunity/misalloc/misalloc_play: T150: user 2026-09-23 'as recommended' = closed as NOT adopted, default off, **'keep opportunity/misalloc/misalloc_play in the code'** (ledger L833-834) - deletion contradicts that decision unless re-confirmed |
| 80 | `ATTACK_ABILITY_MODE` (`theory_order.py:1223`) | `off` | `on` -> **KL** | - | - | attack_passive_wiring:21 / L531/2380 / L265 / 1 | on: F-2; F-5 user 2026-09-30 'adopt the fix set'; the two switches 'stay as switches' with a listed hole inventory (ledger L531) |
| 81 | `PASSIVE_BODY_MODE` (`theory_order.py:1228`) | `off` | `on` -> **KL** | - | - | attack_passive_wiring:22 / L531/2381 / L265 / 1 | on: F-3a; same status as ATTACK_ABILITY_MODE |
| 82 | `ATTACK_DON_MODE` (`theory_order.py:1773`) | `don` | `bare` -> **DD(B)** | - | 9 / 14 | theory_order:5, don_ledger:2 / L2166/2325 / L291/541 / 1 | bare: T45 2026-09-16 |
| 83 | `OPTION_MODE` (`theory_order.py:1832`) | `dist` | `off` -> **KS(S)** | - | - | theory_order:5, conftest:3, attack_passive_wiring:1 / L2165/2324/2391 / L541 / 3 | off: tests/conftest.py autouse pins off ('closed algebra of attack terms') |
| 84 | `BOUNDARY_MODE` (`transition_ledger.py:98`) | `off` | `rules`, `draw_untap`, `draw`, `untap`, `don` -> **DD(B)** | - | 20 / 71 | transition_ledger:13 / - / L565 / 6 | rules/draw_untap/draw/untap/don: T124/T125: boundary speed terms tried and retracted ('no speed term could be assigned to the boundary') |

Not on the tip (local branch `n4-work`, 3 commits ahead of 74f0d0b9, i.e. *behind* the RD-speed commits): `search_price.SEARCH_VALUE_MODE` (`legacy` default; `joint` = KEEP-live, awaiting N-5; ledger row at n4-work:cpu_theory_gap.md L2387) and `search_price.DECK_COUNTER_MODE` (default `rules` since commit 47f6a3e6, user decision 2026-10-04; `printed` = **KEEP-baseline**, the commit title says 'printed stays as a switch'; docs row L2388 on that branch still says default `printed` - doc drift to fix when merging). `tests/test_search_value_joint.py` ratchets the defaults. Also `h4_tables.py:63` lists table labels `noplan`/`spdavg` that exist in no code path ('not measured, low priority' in the H-4 report): delete the two labels (wave B, 2 lines).

## 4. KEEP-baseline: user-decision evidence

No verbatim *instruction to keep an old form* from the user was found for any of these. What exists is (i) the adoption decision itself (user quote), (ii) agent text in the adoption report/ledger/code comment promising reproducibility, (iii) a test that enforces it. The standing rule behind all of them: **ledger section 0.7 step 2 (user instruction 2026-09-17 'put the improvement flow in order'): 'old forms are kept as a switch (`*_MODE`)'**, and the section 9.2 header: **'whenever a default changes, the old form is kept as a switch (to compare past numbers)'**.

| baseline (value) | adoption it preserves | user decision (quote/date) | where reproducibility is promised | what enforces/uses it today |
|---|---|---|---|---|
| `CUT_PRICE_MODE=flat`, `CUT_TAKE_MODE=mu` | N-3 (`joint`+`gbar`) | 2026-10-01 '切り替える規則どおりの N-3 の読みへ今切り替える（数字が落ちても・理論はまだ使っていない）' (ledger L532; `reports/2026-10-01_n3_joint_default.md` L5) | `reports/2026-10-01_n3_joint_default.md` L12 '旧の既定（flat＋mu）は `--cut-price flat --cut-take mu` で再現できる'; `cut_price.py:66-78`; ledger L2382-2383 | 079e73b8 identity test (`test_attack_passive_wiring.py:760`); **flat full-corpus H-4 reproduction (done 2026-10-04) on `claude/h4-misc` job 30** (`reports/2026-10-04_h4_rule_don.md` L5, L72; `h4_tables.py:29,72-79`); `tests/test_cut_price.py` |
| `THETA_HAND_MODE=cuttable_forced` | H-4 (`rule_don`) | 2026-10-04 'both recommendations adopted' (`reports/2026-10-04_h4_rule_don.md` L3; ledger L533) | `crossing_bridge.py:97-99` ('`--theta-hand cuttable_forced` reproduces it, same as N-3 74a51363'); ledger L533 'N-3 の 74a51363 とバイト一致は旧を明示して保つ'; ledger L2384 | 079e73b8 identity test (`:763`); `h4_tables.py:63` columns; `test_relative_ledger.py:437` docstring; 10 literal uses in tests |
| `MIRROR_ME=False` (`--mirror off`) | H-4 mirror default | same 2026-10-04 decision | report L4 '鏡なしは `--mirror off`（または環境変数 `MIRROR=0`）'; ledger L2385 | `relative_ledger.py:606`, `transition_ledger.py:556`, `h4_tables.py` `rule_don_nomirror` columns. **env var does not exist** |
| `DEFENDER_POWER_MODE=token` | N-3 'remaining 2b' (fix of the defender's leader power) | 2026-10-01 (N-3 bundle) | ledger L532, L2386 | 079e73b8 identity test (`:758`); `test_hand_plan.py` (5 refs) |
| `GUARD_S_COST_MODE=curve` | N-2 (`joint`) | 2026-09-26 'N-1/N-2 adopt (a)' (ledger L530) | ledger L530 '旧 `curve` は `--guard-s-cost curve`' | 079e73b8 identity test (`:755`) |
| `F_PRICING_FIXES=none` (`--f-pricing-fixes none`/`legacy`) | F-5 (all 6 fixes) | 2026-09-30 'F の直しの集合を既定に' (ledger L531; `reports/2026-09-30_f_pricing_fixes_adoption.md` L3) | ledger L531 '`none`＝`legacy` で 079e73b8 の値付けとビット一致・同一性の試験は旧の集合を明示して…' | 079e73b8 identity test (`:754`) + `tests/fixtures/f_identity/base_*_079e73b8.json` |
| (n4-work) `DECK_COUNTER_MODE=printed` | N-4 deck counter events by rule | 2026-10-04 (commit `47f6a3e6`: 'user decision 2026-10-04; printed stays as a switch') | same commit; test `test_search_value_joint.py` docstring '旧 `printed` は切替で再現' | ratchet `_DEFAULTS == ('legacy','rules')` on that branch |

Sibling inconsistency to settle together: `lethal_rule.AVG_COUNTER_MODE=printed` is the same kind of old-default (B-fix, 2026-09-25) but nobody said to keep it; I classified it DELETE-dead (wave B). If the user keeps `DECK_COUNTER_MODE=printed` on principle, keep `AVG_COUNTER_MODE=printed` too (3 lines + 19 test lines).

## 5. KEEP-live (undecided or parked on purpose)

| variable | value | why live | what decides it |
|---|---|---|---|
| `THETA_SIDE_MODE` | `symmetric` | T133: 'recommendation kept pending' (ledger L928, T134 'each 1-3'); the only form that makes `threshold_of_me` use the same formula as the opponent side when MIRROR is off | user adoption |
| `SLOPE_TAKE_MODE` | `life` | T134 (3): recommended 'adopt' (ledger L398), orthogonal to the others; **`--d-mode theory` A/B depends on it** | user adoption |
| `SLOPE_BLOCK_MODE` | `on` | T92: rule-correct, numbers slightly worse, 'enter again as a pair after the hand term became flow' (ledger L323); T129 bundle used it with `RATE_THROUGH=cut_block` and was closed (then `on` alone is still the rule-correct form) | user adoption (same family as `RATE_T1`/`RATE_RUSH`/`THETA_HAND_BLOCKER`, which the user adopted for rule-correctness at a cost in numbers) |
| `KAPPA_SIGMA_MODE` | `match` | T122 correctness fix, effect +0.004/+0.001 AUC, 'default kept' (ledger L1241) | user adoption |
| `RATE_DECAY_MODE` | `ko` | T95; the switch did not work from T114 to T128; after the fix 'not re-measured in the current configuration' (ledger L1092, L1450). **Stale**: either re-measure at the H-4 baseline or delete | user |
| `ATTACK_ABILITY_MODE`, `PASSIVE_BODY_MODE` | `on` | F-5 (2026-09-30): rejected for now, explicitly 'stay as switches' with an inventory of remaining holes (`reports/2026-09-30_f_pricing_fixes_adoption.md` L103-112; ledger L531). 14+13 test functions in `test_attack_passive_wiring.py` | F-3b follow-up |
| `THETA_MODES` (`--theta-mode`), `NU_TARGET_MODES` (`--nu-targets`) | `board`, `max` / `board` | T20: 'formula decided, judgement awaits T18' (ledger L2184, game_theory L3116-3118); T18 is on hold (ledger 'T18 を据え置く') | T18 re-open |
| `SEQ_MODE` | `attack_any` (and `off`) | the T18 pilot reading (T155b/T156); T18 on hold | T18 re-open |
| `D_MODE` | `theory` | the only D reading that moves with speed-side switches (used for the SLOPE_TAKE=life A/B, ledger L943) | with `SLOPE_TAKE_MODE` |
| `PRE_SETTLE_MODE` | `on`, `game` | `claude/sigma-wip:sigma_wip/run_sigma.sh` runs `win_calib.py --pre-settle on` and `pre_settle_asymmetry.py`; T151-3 `game` is part of the K-series calibration | the sigma-table merge, then K-series follow-ups |
| `FLOW_PRICING`, `LEDGER_FLOW_PRICING` | `option` / `exercise` | T58 two-role convention - both values are used in the same run by design | not a candidate |
| (n4-work) `SEARCH_VALUE_MODE` | `joint` | N-4 switch only; 'adoption is the user's call', N-5 (pair with the leaving side) pending | N-5 |

## 6. KEEP-scaffold

`tests/conftest.py:43-79` (`_theory_option_off`, autouse for **every** test) sets `OPTION_MODE=off`, `W_MODE=flat`, `SURV_MODE=once`, `CBAR_MODE=loose`, `CLOCK_HAND_MODE=off` (the last equals the default). The closed-form arithmetic of ~100 tests (`test_theory_order` 91 functions, `test_crossing_bridge` 119, ...) is written against those values. `SPEED_MEMO=False` is the other half of `test_speed_memo.py` (bit-identity oracle). Recommendation: leave them; if the user ever wants them gone the clean way is to flip each *default-pinned* test to a local fixture first (one PR per switch). Est. cost if deleted anyway: ~330 test lines rewritten + 5 switch removals ~60 code lines.

## 7. DELETE plan (nothing is deleted now)

Common to every wave: (1) **freeze first** - `git tag theory-switches-final` (and push branch `claude/theory-switches-final`, because this environment's proxy rejects tag pushes - precedent: `py-engine-final` / `claude/py-engine-final` in CLAUDE.md) on the commit before wave A, and say in ledger section 9.2 + the 'compare older numbers' pointers that old forms are reproduced from that tag. (2) keep the **echo keys** of deleted switches in tool output as constants (section 10). (3) update `_SHIPPED` in `test_crossing_bridge.py:31-77` and `test_the_shipped_defaults_are_the_ones_we_decided` when a switch leaves the tuple. (4) never edit `docs/reports/*`.

### Wave A - before the port (kernel-touching): 12 values in 8 variables

| item | code to remove (crossing_bridge.py unless noted, line spans at f58d6b2e) | tests to delete/rewrite | est |
|---|---|---|---|
| `THETA_HAND_MODE=rule_don_purse` | `speed_plan_of` L1581-1601; `attacker_ctx` L1690-1694; `_rule_don_masks` `fixed` L1956,1966-1967,1982-1983; `_mask_firsts` L1994,2005-2006; `_rule_don_solve` L2172,2194-2195; `RULE_DON_MODES` L1557 becomes `('rule_don',)` and its ~15 reader sites in `kappa_vector.py:481-498`, `relative_ledger.py:312-354,615`, `theory_bridge.py:848-915,1893`, `win_calib.py:252`, `cut_price.py:430` become `== 'rule_don'` | `test_crossing_bridge.py::test_rule_don_purse_keeps_the_speed_sides_split…` (~18 lines, L851-868); `test_rd_speed.py:86-89` (the 15% `fixed` branch of `_rand_problem` - **keep the RNG draw** or the 40 random problems change); `rule_don_ref.py` `fixed` lines (the ref is verbatim - edit consciously) | ~50 code / ~40 test |
| `RATE_T1_MODE=off` | consts+comment L3396-3409; `set_rate_t1_mode` L3763-3768; argparse L5175-5177; `rate_at` L3792-3793 (make unconditional); `rules_sched` L1818 `skip1`; readers `kappa_vector.py` (2), `relative_ledger.py` (1), `theory_bridge.py` (1) | 3 functions in `test_crossing_bridge` (~74 lines); `test_rd_speed.py:255` uses it as the sample switch for store-key sensitivity -> use another solver-read global (e.g. `RATE_DON_PAY`... also deleted; use `EX_STATE_BUDGET` or `SLOPE_FLOOR`) | ~18 / ~74 |
| `RATE_RUSH_MODE=off` | L3391-3395; setter L3771-3776; argparse L5178-5180; readers `rules_steps` L1730, `attacker_ctx` L1662, `seat_slope_*` (3 sites); `rule_don_ref.py` (1) | 2 functions (~37 lines) | ~14 / ~37 |
| `SLOPE_EFFECT_MODE=off,on` | L3425-3428; setter L3431-3436; argparse L5163-5166; readers `rules_steps` L1734, `attacker_ctx` L1664-1665, `seat_slope_terms`/`seat_slope_sched` (7 if-sites), `kappa_vector.py` (2) | 1 function (~22 lines) | ~36 / ~22 |
| `RATE_DON_PAY=False` | L296; `set_rate_don_mode(pay=)` L302-313; argparse L5192-5194; `rules_sched.left_of` L1799; `seat_slope_sched` (4 ternaries); `bias_budget.py:301`; `rule_don_ref.py` (1); `plan_store` names list in `test_rd_speed.py:277` | none | ~10 / ~0 |
| `RATE_RAMP!=0` | L299; `rate_at` L3796/L3818; `set_rate_don_mode(ramp=)`; argparse; `bias_budget.py:303` | `test_don_walk.py` (2 refs) | ~6 / ~10 |
| `DON_PURSE_MODE=off,one,race` + `RATE_DON_MODE=off,purse` | consts/comments L3462-3469, L285-293; setters L3472-3477, L302-313; argparse L5167-5170, L5189-5191; `attacker_ctx` gate L1616; `seat_slope_terms` (13 if-sites) / `seat_slope_sched` (3); race family: `race_alloc` L3673-3705, `purse_pareto` L3611-3641, `_pareto` L3587-3608, `race_margin` L3644-3647, `choose_by_race` L3650-3662, `opp_board_slope` L3665-3670, `theta_groups` L3724-3747, `free_cuttable_g` L3716-3721, `HP_hand_items` L3708-3713 (~150 lines); `collect()` race branches L4044-4259,4431-4434 (E-1 audit: only double count in the pipeline); **whole script `purse_race.py` (215)**; `don_ledger.py` (3 refs), `bias_budget.py` | `test_crossing_bridge` race/purse tests (~46 lines), `test_don_walk.py` (6+3 refs) | ~250 code + 215 script / ~90 test |

Wave A total: ~350 code lines + the 215-line script; tests ~270 lines (upper bound). Risk to default outputs: **medium** - these edit the default code path where flags collapse (`rate_at`, `rules_steps`, `rules_sched`, `seat_slope_sched`, `collect`). Verify with section 10 steps 1-4 after each variable (not after the whole wave).

### Wave B - clean dead alternatives (57 values)

| variable | dead values | est code / tests | note |
|---|---|---|---|
| `SLOPE_MODE` | `board` | 10 / 0 | T77 user decision 2026-09-17 'change 1' made hand default; board = pre-T77 numbers |
| `THETA_HAND_BLOCKER_MODE` | `off` | 21 / 24 | T106; user decision 2026-09-19 'fix 2 to the correct form'; off = hand blockers not counted (wrong by rule) |
| `THETA_DON_MODE` | `off`, `blocker` | 12 / 92 | T110, user 2026-09-19 'fix it'; off/blocker = the wrong don budgets |
| `THETA_HAND_PLACE` | `shield` | 46 / 46 | T102: cost small, direction split, 'kept as switch'; superseded by THETA_HAND_WINDOW=horizon (T116, user 2026-09-20) - c |
| `THETA_HAND_WINDOW` | `off`, `fixpoint` | 36 / 0 | T116 user 2026-09-20 '3 all'; fixpoint measured worse than horizon; off = pre-T116 |
| `RATE_THROUGH_MODE` | `cut`, `cut_block` | 59 / 33 | T131: built, measured worse on 4/4 metrics in both records; closed with T132 'stop improving A' |
| `SLOPE_HAND_MODE` | `stock` | 33 / 22 | T93 user 2026-09-18 'make it default'; stock = counts a stock as income (wrong) |
| `SCHED_T1_MODE` | `walk` | 10 / 34 | T152 user 2026-09-24 '1 is default'; walk = control for pre-T152 numbers |
| `RACE_MODE` | `net`, `deck`, `deck_shield` | 54 / 94 | T90/T91/T104: deck refuted both records (ledger L398), deck_shield 'right in rule but inert', net lets play style into t |
| `RATE_WALK_MODE` | `flat` | 24 / 22 | T94 user 2026-09-18; flat = pre-T94 |
| `OPP_CLOCK_MODE` | `prev_start` | 20 / 20 | T151 user 2026-09-24 'as recommended'; prev_start = control for pre-T151 numbers |
| `CUT_PRICE_MODE` | `joint_slice`, `joint_theta`, `joint_floor` | 14 / 0 | diagnostic/control arms of the 2026-09-30 review (code comment L66-70); joint_slice raises in cut_card_price for rule* |
| `SEARCH_PRICE_MODE` | `sel` | 10 / 21 | T68 user 2026-09-17 'replace'; sel = pre-T68 |
| `PLAY_NOW_MODE` | `full` | 9 / 31 | T70 user 2026-09-17; full = pre-T70 |
| `INFLOW_MODE` | `off` | 12 / 72 | T70 user 2026-09-17; off = static v |
| `COND_CLOCK_MODE` | `off` | 9 / 52 | T73 user 2026-09-17; off = state-now for all t |
| `LETHAL_HAND_MODE` | `share` | 26 / 13 | T117 old convention |
| `LETHAL_STOP_MODE` | `econ` | 13 / 11 | T130 rule variant, not adopted |
| `LETHAL_LIFE_MODE` | `off` | 15 / 38 | T117 old convention |
| `AVG_COUNTER_MODE` | `printed` | 16 / 19 | B-fix adopted 2026-09-25 ('switch to it'); NOTE same kind of baseline as DECK_COUNTER_MODE=printed which the user kept o |
| `HAND_MEAS_MODE` | `count` | 9 / 13 | T69 user 2026-09-17 |
| `FREEZE_YARDSTICK` | `True` | 3 / 0 | F-4 measurement option (F closed 2026-09-30) |
| `ATTACK_SPLIT` | `True` | 6 / 0 | F-4 output option |
| `GUARD_G_MODE` | `paid`, `zero` | 8 / 24 | only meaningful with LEDGER_HARM_MODE=price (ledger L2332: 'normally unused') |
| `GUARD_COST_MODE` | `formula`, `spent_all` | 23 / 0 | same ledger-form bundle (T64/T86) |
| `LAST_TURN_MODE` | `drop` | 12 / 0 | T80 diagnostic arm |
| `LEDGER_HARM_MODE` | `price` | 26 / 22 | T87 user 2026-09-18 'default must be the correct one'; price = old ledger writing |
| `GUARD_PRICE_MODE` | `all_attacks` | 13 / 26 | T86, 'held'; third mismatch unresolved; only with price ledger |
| `ATTACH_LEDGER_MODE` | `increment` | 15 / 15 | T85 user 2026-09-18 'after fixing it' |
| `PLAY_BOOK_MODE` | `now` | 13 / 33 | T84 user 2026-09-18 'make default' |
| `GUARD_AFFORD_MODE` | `lenient` | 8 / 47 | G-2 user 2026-09-26: a rule bug-fix; lenient = the bug |
| `GUARD_S_COST_MODE` | `hand` | 8 / 85 | G-2/G-3 intermediate form (user 2026-09-26); 6 test functions in test_guard_hand_cost |
| `DECISION_ROW_MODE` | `any` | 8 / 18 | D-5 user 2026-09-25 |
| `SIGMA_FROM_CURVE` | `False` | 6 / 0 | T97 borrowed sigma 1.0 |
| `SILENT_MODES (--silent)` | `exclude` | 0 / 0 | P2 sensitivity (2026-09-14) |
| `TAKE_MODE` | `const`, `by_life` | 9 / 18 | T63 user 2026-09-16 '3 only': by_life rejected, const = pre-T63 |
| `NU_MODE` | `base` | 9 / 43 | T37 user 2026-09-15 |
| `W_ERR_MODE` | `abs` | 26 / 0 | T118 user 2026-09-20 '3 all'; abs gives log-loss 1.3 vs 0.67 |
| `W_MOVER_MODE` | `off` | 14 / 53 | T151 user 2026-09-24 |
| `ACTIVATE_USES_STATE` | `False` | 1 / 0 | T42 'sensitivity switch' back to 2026-09-15 valuation |
| `PLAY_COST_MODE` | `flat` | 3 / 29 | T43 2026-09-16 |
| `ATTACK_DON_MODE` | `bare` | 9 / 14 | T45 2026-09-16 |
| `BOUNDARY_MODE` | `rules`, `draw_untap`, `draw`, `untap`, `don` | 20 / 71 | T124/T125: boundary speed terms tried and retracted ('no speed term could be assigned to the boundary') |

Wave B total: ~710 code lines, ~1160 test lines (upper bound). Risk to default outputs: **low** per switch (the removed branch is non-default) *except* (a) the echo keys (section 10) and (b) the ledger-form bundle `LEDGER_HARM_MODE`/`GUARD_G`/`GUARD_COST`/`GUARD_PRICE`/`ATTACH_LEDGER`/`PLAY_BOOK` (`theory_bridge.py:124-312`), which share `realised_harm`, `guard_step` and `play_deferred` code with the default branch - delete them in one PR and run the full identity check.

### Wave C - dead, but dependency-bound (29 values)

| variable | dead values | blocker | est code / tests |
|---|---|---|---|
| `THETA_HAND_MODE` | `rule`, `cuttable_seq`, `cuttable_cx`, `cuttable`, `count`, `quality`, `play`, `guard` | H-4 DP without attacker attach; never measured; VALUE only - the code path (`_rule_hand_term`, `rule_guard_plan`) stays as the fallback of rule_don; 5 | 205 / 343 |
| `THETA_HAND_PART` | `dtotal`, `dh`, `dg` | per-card price parts of count/quality/play/guard; dies with THETA_HAND_MODE legacy values | 4 / 11 |
| `THETA_BODY_MODE` | `all`, `attackable`, `none` | T82/T83/T97: all+attackable adopted-then-reverted by user 2026-09-18 ('make it theoretically correct'); none = T129 bundle, closed 'shipped is best'.  | 38 / 155 |
| `THETA_RETURN_MODE` | `off` | C-5c user 2026-09-25 'a'; off = 5-tuple state path in kappa_vector/relative_ledger/transition_ledger; bias_budget.py already refuses the default confi | 17 / 64 |
| `PRICING_FIXES (L)` | `legacy`, `-name subsets` | L user 2026-09-26 '3a'; legacy = pre-L values, bound by test_effect_value; not in the 079e73b8 test (it was recorded after L) | 3 / 0 |
| `COST_AFFORD_MODE` | `off` | D-2..D-4 user 2026-09-25; 11 test functions (~200 lines) in test_search_price pin 'off' algebra | 8 / 197 |
| `ATTACK_REST_MODE` | `off`, `body` | C-5c user 2026-09-25; code comment says body 'kept as switch' (agent text); needs 7-tuple state - tests in test_kappa_vector/test_theta_return | 13 / 66 |
| `D_MODE` | `clock`, `curve_scaled` | T49/T75 earlier readings; 34 test functions mention D_MODE - confirm which values they pin | 79 / 302 |
| `SEQ_MODE` | `attack`, `attack_le`, `attack_le_delta` | T144/T147a intermediate readings superseded by attack_any; CLAUDE.md:221 uses `--seq attack_le` as a wording example | 10 / 173 |
| `CLOCK_HAND_MODE` | `on` | T78 'regressed, waiting for an estimator'; premise voided by 0.05 (complete information); only feeds the superseded W_MODE=clock path - confirm | 14 / 46 |
| `SIGMA_FLOOR_MODE` | `on` | K-2/K-3 candidate superseded by K-5 `whole`; but harm_profile.json carries sigma_rel_floor and the sigma re-measure is running - do after the merge | 13 / 60 |
| `SETTLE_COND_MODE` | `off`, `on` | K-5 user 2026-09-26 'a'; on = K-1 (worse); off = pre-K; same sigma-table dependency | 23 / 123 |

Extra whole-file candidates tied to wave C: `tests/scripts/bias_budget.py` (321) + `tests/test_bias_budget.py` (214) (already unrunnable in the shipped config); parts of `test_theta_return.py` (210) and `test_don_walk.py` (203). Sigma-table dependency: `tests/fixtures/harm_profile.json` carries `sigma_t`/`w_bar` for `attackable`/`all`, `sigma_rel_floor`, and `sigma_rel` (non-whole) - **do not edit the fixture or `summarise` output keys until `claude/sigma-wip` results are merged**; after that, drop the dead keys *with an explicit, listed diff* (they are inputs of dead switches only).

### Wave U - contradicts an explicit decision (1 variable, 3 values)

`ATTACK_DON_COST_MODE` = `opportunity`, `misalloc`, `misalloc_play` (`theory_order.py:1171-1180`, `attack_don_cost`, `don_misalloc`, `foregone_play_value`, `_attach_total_forced`, `play_price_of` split; wiring in 5 scripts (`crossing_bridge`, `theory_bridge`, `relative_ledger`, `two_curves`, `attack_response`) via `ctx['hand']`). Ledger L833-834: **user decision 2026-09-23 'それでお願いします' = close T150 as not adopted, default `off`, 'opportunity/misalloc/misalloc_play are kept in the code'**. Est. ~10 lines of switch plumbing + the functions (not counted: shared with `PLAY`), 12 test functions / ~160 lines. Do not delete without an explicit re-decision (section 11, D2).

### DELETE-after-port

`EX_LAYER_COUNT=False` (L1871, read at L1921): the kernel always counts layers; the stepwise-only branch only exists in the Python solver. No test toggles it today (the comment at L1869-1870 claims the tests compare both - they do not; `test_rd_speed.py` only checks `_ex_count_layers` sizes). Remove together with the Python solver (rust_design.md section 9).

### Totals

| | code lines (tests/scripts) | whole scripts | test lines touched (upper bound) | ledger 9.2 rows | TEST_SPEC lines |
|---|---|---|---|---|---|
| A | ~350 | 215 (`purse_race.py`) | ~270 | 13 | 7 |
| B | ~710 | - | ~1160 | 32 | 41 |
| C | ~380 | 321 (`bias_budget.py`) | ~1500 (+214 whole `test_bias_budget.py`) | 9 | 22 |
| U | ~10 | - | ~160 | 0 | 1 |
| **all DELETE** | **~1440** | **~540** | **~3080 (about 2,000 if functions that also assert defaults are only edited, not deleted)** | **47 distinct rows of ~65** | **61 distinct lines** |

Docs to update for any deletion: ledger section 9.2 (rows above) and the stale-row fixes of section 9 of this file; ledger section 0.7 step 2 and section 9.2 header ('old forms are kept as switches' -> 'old forms are reproduced from the freeze tag'); TEST_SPEC rows (line numbers in Appendix A); `docs/game_theory.md` (43 lines of 'compare older numbers with --X' pointers; do not rewrite history, add one 'see freeze tag' note per section); CLAUDE.md line 221 (the example `--seq attack_le` in the decision-format rule: replace by a surviving switch name, e.g. `--seq attack_any`) and one added sentence under 'Theory phase' pointing at the freeze tag. 162 body lines of the ledger mention deleted-switch names in narrative; they are history and stay.

## 8. Deletions that would contradict an explicit decision or standing rule

1. **T150 family (`ATTACK_DON_COST_MODE`)** - explicit user decision 2026-09-23 to *keep the code* (ledger L833-834). Needs re-decision.
2. **Ledger 0.7 step 2 / 9.2 header** - user-sanctioned process ('keep the old form as a switch'). The 2026-10-04 'delete switches that are no longer needed' decision supersedes it for specific switches; amend both lines in the same PR so the ledger does not contradict itself.
3. **The 079e73b8 identity test** - deleting any of its six baseline values forces retiring the test and `tests/fixtures/f_identity/base_*` (the only byte-level guard of `price_realised`/`theory_bridge`/`transition_ledger` outputs). Do not retire it without replacing it by a newer identity baseline recorded at the freeze tag (section 10).
4. **N-4 `DECK_COUNTER_MODE=printed`** (user decision 2026-10-04, on `n4-work`) - keep; and decide `AVG_COUNTER_MODE=printed` the same way.
5. **F-5 'two switches stay'** (`ATTACK_ABILITY_MODE`, `PASSIVE_BODY_MODE`) and **T18-gated** switches - classified KEEP-live; deleting them would also contradict 'T18 is on hold, re-measure after theory is organised'.
6. **Agent-written 'kept as switch' comments** on `ATTACK_REST_MODE=body` (`kappa_vector.py:146`) and `THETA_HAND_PLACE=shield` (T102): not user decisions; both superseded by user-adopted forms (C-5c, T116). Delete after one confirmation.
7. **Measurement users**: `claude/sigma-wip` (running; default switches + plan store) and the finished `claude/h4-misc` job 30 (`flat`). Deleting the *code that produced a running job's output* is harmless for the job, but merging its results needs `h4_tables.py` (references `flat`, `cuttable_forced`, `rule`, `rule_don_purse`, `noplan`, `spdavg`) and `summarise` keys to still exist.

## 9. Documentation drift found while auditing (fix in the same PRs)

* Ledger 9.2 row `THETA_RETURN_MODE` (L2354): says default `off` 'held'; code default is `untap` since C-5c (2026-09-25). Row `THETA_HAND_MODE` (L2344): says default `cuttable`; later rows (L2384) say `rule_don`. No row for `EX_LAYER_COUNT`, `RATE_DON_PAY`, `RATE_RAMP`, `FLOW_PRICING`.
* `MIRROR=0` env var: help text `theory_bridge.py:1782`, ledger L2385, `reports/2026-10-04_h4_rule_don.md` L4, `test_relative_ledger.py:437` - not implemented.
* `crossing_bridge.py:1869-1870` says tests compare `EX_LAYER_COUNT` on/off; no test does.
* `n4-work` ledger row `DECK_COUNTER_MODE` still says default `printed` while code and test say `rules` (commit 47f6a3e6 changed code/test only).
* `h4_tables.py:63` lists `noplan`/`spdavg` arms that exist in no code.
* `CLAUDE.md:221` uses `--seq attack_le` as an example of a switch name not to use.

## 10. Keeping default outputs byte-identical - how to verify

**What changes bytes even if the formulas do not**: tools echo the active switch values into their JSON (`stats` in `crossing_bridge.collect()` L4053-4082: `theta_hand, slope_mode, theta_body, slope_block, rate_through, race, slope_hand, rate_walk, rate_decay, rate_rush, rate_t1, slope_effect, don_purse, rate_don, rate_don_pay, rate_ramp, theta_don, theta_hand_blocker, theta_return, theta_hand_place, theta_hand_window` and conditionally `cut_price`; `theory_bridge` stats L1089-1113 and L1871-1886: `w_mode, clock_hand, play_book, attach_ledger, guard_price, ledger_harm, flow_pricing, guard_g, guard_afford_requested, guard_s_cost, search_price, attack_ability, passive_body, decision_rows, surv_mode, nu_mode, cbar_mode, take_mode, guard_cost, play_now, inflow, cond_clock, cut_price`; `price_realised` L279-281,612-618: `search_price, hand_meas, play_now, cost_afford, inflow, cond_clock, flow_pricing`; `relative_ledger` L491-492: `d_mode, kappa_sigma_mode, attack_rest_mode`; `transition_ledger` L472-473: `d_mode, boundary_mode, attack_rest_mode`; `kappa_vector` L753: `w_err_mode`; `win_calib` L209: `pre_settle`). **Rule: when a switch variable is deleted, keep its echo key with the constant default value** (or list the removed keys as an intended, reviewed diff). Same for `summarise` keys fed by deleted modes (`sigma_rel_floor`).

**Procedure**

1. Before wave A, at the freeze tag: for each tool in {`crossing_bridge`, `theory_bridge`, `relative_ledger`, `transition_ledger`, `price_realised`, `win_calib --pre-settle on`, `pre_settle_asymmetry`} run `OPCG_LOG_SILENT=1 python tests/scripts/<tool>.py --in <records> --out base/<tool>.json` on (a) `tests/fixtures/f_identity/rec` (2 real games; only price_realised/theory_bridge/transition_ledger have recorded baselines today - **crossing_bridge, relative_ledger and win_calib have none**) and (b) a 30-game slice of the w41/w39 data (the sigma branch ships them under `sigma_wip/data/`). Commit (a) as `tests/fixtures/default_identity/base_<tool>_<sha>.json` so it is a gate, keep (b) outside the repo.
2. After **each** switch removal (not each wave): re-run the same commands; compare with `python -c` that loads both JSONs, pops `seconds`/`elapsed`/`rule_stats`/`ex_speed_stats` timing-only fields, and asserts `==` (the pattern in `test_attack_passive_wiring.py:790-796`). Any diff is a bug in the removal, not a finding.
3. `OPCG_PLAN_STORE`: run once cold and once warm with a store created *at the freeze tag with the narrowed key* (rust_design.md section 6) - the second run must be byte-identical to the first; this also proves the narrowed key survives the deletion.
4. Existing gates that must stay green untouched: `test_the_shipped_defaults_are_the_ones_we_decided` (only the tuple loses names), the 079e73b8 test (unless D1), `test_rd_speed.py`, `test_speed_memo.py`, `make golden-audit` is irrelevant (engine untouched).
5. grep audit per removal: `git grep -nE '\bNAME\b|set_name|--flag'` over `tests/ opcg_sim/ docs/ Makefile` must return only history (`docs/reports/`) and the freeze-tag note.

**Verification I could not do**: running any tool or the suite (shared machine). All numbers in sections 1 and 7 are static estimates.

## 11. Decisions the user has to take (draft, plain language - for the parent to relay in the CLAUDE.md format)

**D1 - Do we retire the 'old numbers can be reproduced from the current code' guarantee?** Options: (a) freeze a tag/branch now, delete the old forms, retire the 079e73b8 test and replace it by a new identity baseline recorded at the freeze point (my recommendation); (b) keep the six baseline values and the test, delete only the other ~100 dead values (smaller win, the kernel still carries `flat` pricing only as an input); (c) keep everything. Why it came up: six old values are pinned by one byte-identity test that each adoption wave edited, and the same values were used by `claude/h4-misc` job 30 (finished today) and are still referenced by `h4_tables.py`. Without a decision the 6 baselines and 100 dead values stay as they are and the Rust parity matrix has to cover `flat` and the old don-purse forms. Caveat: after (a), reproducing a 2026-09 number needs `git checkout theory-switches-final`.
**D2 - The T150 family**: confirm that the 2026-09-23 decision 'keep opportunity/misalloc/misalloc_play in the code' no longer holds. Delete (my recommendation, ~170 lines incl. tests) or keep.
**D3 - Parked candidates**: do any of `SLOPE_TAKE_MODE=life`, `THETA_SIDE_MODE=symmetric`, `SLOPE_BLOCK_MODE=on`, `KAPPA_SIGMA_MODE=match`, `RATE_DECAY_MODE=ko`, `ATTACK_ABILITY/PASSIVE_BODY` get adopted or closed? Each stays costing ~15 code + ~50 test lines and a place in the parity matrix of any later change. Recommendation: decide the three that touch the speed side first (they are cheap to measure with the sigma/plan-store set-up) and close `RATE_DECAY_MODE` unless re-measured.
**D4 - Counter-value baselines**: keep both `DECK_COUNTER_MODE=printed` and `AVG_COUNTER_MODE=printed`, or neither.
D1 gates the size of everything else; D2-D4 are independent of each other and of the port. Wave A (kernel-touching) does not need D1 except for `RATE_*` readers inside the 079e73b8 test path - it does not read them, so wave A can start as soon as the sigma results are merged and the freeze tag exists.

## Appendix A - reference index (auto-generated, all inventoried variables)

`tests` = number of mentions of the variable/setter name per test file (no line numbers, files in `tests/test_*.py` and `tests/conftest.py`); `ledger` = lines of `docs/cpu_theory_gap.md`; `TEST_SPEC` lines; `game_theory`/`measurement` lines; reports = number of `docs/reports/*.md` files mentioning the name or its CLI flag (never edited); `flags` = CLI flags.

| variable | CLI flags | tests (mentions) | ledger lines | TEST_SPEC lines | game_theory / measurement | reports |
|---|---|---|---|---|---|---|
| `THETA_HAND_MODE` | `--theta-hand` | crossing_bridge:63, cut_price:6, relative_ledger:4, two_curves_state:1, attack_passive_wiring:1 | 316/337/338/398/521/533/1288/1307+7 | 294/549/597 | 206/216/2568/2576 | 12 |
| `THETA_HAND_PART` | - | crossing_bridge:7 | 316 | 294/549/597 | - | 2 |
| `SLOPE_MODE` | `--slope-mode` | crossing_bridge:4 | 337/2134/2135/2338 | 294/597 | 2567 | 2 |
| `THETA_HAND_BLOCKER_MODE` | `--theta-hand-blocker` | crossing_bridge:8 | 1365/1390/2341/2349 | 294/597 | 137/161 | 4 |
| `THETA_DON_MODE` | `--theta-don` | crossing_bridge:12 | 1373/2343 | 294/597 | - | 3 |
| `RATE_DON_MODE` | `--rate-don` | don_walk:6, crossing_bridge:4 | 1093/1288/1307/1316/2355 | 296/597 | - | 7 |
| `RATE_DON_PAY` | - | rd_speed:1 | 1288 | - | - | 2 |
| `RATE_RAMP` | `--rate-ramp` | don_walk:2 | - | - | - | 3 |
| `THETA_BODY_MODE` | `--theta-body` | crossing_bridge:27, kappa_vector:1, rate_tracking:1 | 318/332/997/1043/1051/2129/2130/2337 | 294/549/597 | 2500/2518 | 8 |
| `THETA_RETURN_MODE` | `--theta-return` | bias_budget:7, crossing_bridge:6, theta_return:6, relative_ledger:1 | 512/2354 | 294/300/328/597 | - | 5 |
| `EX_LAYER_COUNT` | - | - | - | - | - | 0 |
| `THETA_HAND_PLACE` | `--theta-hand-place` | crossing_bridge:6, bias_budget:2 | 1070/1404/2350 | 328/597/609 | 189 | 5 |
| `THETA_HAND_WINDOW` | `--theta-hand-window` | crossing_bridge:3, don_walk:2 | 1289/1307/1323 | 597 | - | 3 |
| `THETA_SIDE_MODE` | `--theta-side` | crossing_bridge:8, relative_ledger:2 | 928/956 | 323/567/597 | - | 2 |
| `SLOPE_BLOCK_MODE` | `--slope-block` | crossing_bridge:10 | 323/1051/1058/2358 | 294/597 | 299 | 5 |
| `RATE_THROUGH_MODE` | `--rate-through` | crossing_bridge:7 | - | 597 | - | 1 |
| `SLOPE_TAKE_MODE` | `--slope-take` | crossing_bridge:10, relative_ledger:2 | 398/929/943 | 323/567/597 | - | 1 |
| `SLOPE_HAND_MODE` | `--slope-hand` | crossing_bridge:6 | 322/1073/1462/2356/2357 | 294/597 | 279/287 | 4 |
| `SCHED_T1_MODE` | `--sched-t1` | crossing_bridge:10 | 398/877/888 | 294/598 | - | 2 |
| `RACE_MODE` | `--race` | crossing_bridge:11, bias_budget:5 | 324/325/398/928/1401/1476/2123/2349+1 | 294/328/597 | 181/302 | 10 |
| `RATE_WALK_MODE` | `--rate-walk` | crossing_bridge:7 | 321/1457/2356 | 294/597 | 270/277 | 4 |
| `RATE_DECAY_MODE` | `--rate-decay` | crossing_bridge:9, bias_budget:2 | 1092/1094/1450/2355 | 294/328/597 | 262 | 3 |
| `RATE_RUSH_MODE` | `--rate-rush` | crossing_bridge:12 | 1365/1400/2341/2348/2349 | 294/597 | 136/180 | 5 |
| `RATE_T1_MODE` | `--rate-t1` | crossing_bridge:11, rd_speed:3, kappa_vector:1 | 1364/1365/1398/2341/2347/2349 | 293/294/570/597 | 136/179 | 8 |
| `SLOPE_EFFECT_MODE` | `--slope-effect` | crossing_bridge:10 | 1073/1365/1381/1393/2345/2346/2349 | 597 | 135/145/169 | 6 |
| `DON_PURSE_MODE` | `--don-purse` | crossing_bridge:8, don_walk:3 | 509/938/948/1366/1377/2342/2344 | 294/296/592/597 | - | 6 |
| `PRE_SETTLE_MODE` | `--pre-settle` | crossing_bridge:11, win_calib:11, pre_settle_asymmetry:4, rd_speed:4, settle_cond_floor:2 | 507/585/845 | 293/294/303/304/323/567/597/598+4 | - | 8 |
| `OPP_CLOCK_MODE` | `--opp-clock` | crossing_bridge:7, rd_speed:1 | 398/405/845/867/875 | 294/598/607/608 | - | 4 |
| `CUT_PRICE_MODE` | `--cut-price` | cut_price:2, relative_ledger:1, attack_passive_wiring:1, speed_memo:1 | 2382 | 286/294/563 | - | 2 |
| `CUT_TAKE_MODE` | `--cut-take` | cut_price:4, speed_memo:4, relative_ledger:1, attack_passive_wiring:1 | 2383 | - | - | 2 |
| `PRICING_FIXES (L)` | - | effect_value:4 | - | - | - | 0 |
| `F_PRICING_FIXES (F)` | - | attack_passive_wiring:3, effect_value:1 | 2379 | - | - | 1 |
| `SEARCH_PRICE_MODE` | `--search-price` | search_price:5 | 345/2143/2366 | 547/549/557/564/613 | 454/2449 | 6 |
| `PLAY_NOW_MODE` | `--play-now` | search_price:6 | 343/2141/2368 | 547/549/557/613 | 458/2609/564 | 3 |
| `COST_AFFORD_MODE` | `--cost-afford` | search_price:29 | 516/517/519/2373 | 289 | - | 5 |
| `FLOW_PRICING / LEDGER_FLOW_PRICING` | - | - | - | - | - | 0 |
| `INFLOW_MODE` | `--inflow` | hand_plan:3, guard_hand_cost:2, hand_joint:1 | 343/2141/2369 | 549/557/560/613 | 2610 | 3 |
| `COND_CLOCK_MODE` | `--cond-clock` | hand_plan:5 | 340/2138/2370 | 549/557/560/613 | - | 2 |
| `ATTACK_REST_MODE` | `--attack-rest` | kappa_vector:10, attack_axis_by_result:8, theta_return:6 | 510/512 | 298/300/326 | - | 3 |
| `D_MODE` | `--d-mode` | kappa_vector:28, transition_ledger:18, relative_ledger:15, attack_axis_by_result:8, theta_return:4 | 398/933 | 298/323/566/567/571 | - | 13 |
| `LETHAL_HAND_MODE` | `--hand` | lethal_rule:4, settlement_split:2 | - | 534/588 | - | 2 |
| `LETHAL_STOP_MODE` | `--stop` | lethal_rule:4, settlement_split:2 | 491 | 588 | - | 3 |
| `LETHAL_LIFE_MODE` | `--life` | lethal_rule:8, settlement_split:2 | - | 588 | - | 2 |
| `AVG_COUNTER_MODE` | `--avg-counter` | lethal_rule:11 | 507 | 322 | - | 1 |
| `HAND_MEAS_MODE` | `--hand-meas` | price_realised:5 | 344/2142/2367 | 557/560/613 | 457/2457/562 | 3 |
| `FREEZE_YARDSTICK` | - | - | - | - | - | 0 |
| `ATTACK_SPLIT` | - | - | - | - | - | 0 |
| `SEQ_MODE` | `--seq` | t18_arena:14, shadow_forbid:12, theory_trace:1 | 665/669/754/757/821/911 | 312/313/315/316/578/580/581 | - | 10 |
| `GUARD_G_MODE` | `--guard-g` | theory_bridge:5, guard_hand_cost:1 | 327/351/1497/2125/2149/2332/2363 | 549 | 2463 | 5 |
| `GUARD_COST_MODE` | `--guard-cost` | - | 328/349/2126/2147/2334/2365 | 549 | 2426/2465 | 3 |
| `LAST_TURN_MODE` | `--last-turn` | - | 1556/2132 | 549 | - | 1 |
| `LEDGER_HARM_MODE` | `--ledger-harm` | theory_bridge:4 | 327/515/893/1494/2125/2331/2332 | 272/549 | 326/2463 | 6 |
| `GUARD_PRICE_MODE` | `--guard-price` | theory_bridge:4 | 328/2126/2333 | 272/549 | 2465 | 2 |
| `ATTACH_LEDGER_MODE` | `--attach-ledger` | theory_bridge:6, two_curves:3 | 329/2127/2335 | 272/549 | 345/2472 | 2 |
| `PLAY_BOOK_MODE` | `--play-book` | theory_bridge:7 | 330/397/1511/2128/2336 | 272/549 | 351/2485 | 2 |
| `GUARD_AFFORD_MODE` | `--guard-afford` | guard_hand_cost:12 | 527/2375 | 284/603 | - | 1 |
| `GUARD_S_COST_MODE` | `--guard-s-cost` | guard_hand_cost:11, hand_joint:7, theory_bridge:4, attack_passive_wiring:1 | 527/530/2376 | 272/284/285/549/562/603 | - | 2 |
| `DECISION_ROW_MODE` | `--decision-rows` | decision_rows:9 | 518/519/2374 | 288 | - | 3 |
| `MIRROR_ME` | `--mirror` | relative_ledger:2 | 533/2385 | 294 | - | 1 |
| `SIGMA_FROM_CURVE` | - | theory_bridge:1 | 2353 | 272/549 | - | 1 |
| `SILENT_MODES (--silent)` | - | - | - | - | - | 0 |
| `TAKE_MODE` | `--take-mode` | theory_order:5, crossing_bridge:2 | 350/523/2148/2364 | 541/549/555/613 | 2415/2420/2421 | 7 |
| `THETA_MODES (--theta-mode)` | - | - | - | - | - | 0 |
| `NU_TARGET_MODES (--nu-targets)` | - | - | - | - | - | 0 |
| `NU_MODE` | `--nu-mode` | theory_order:5, nu_calib:4, attack_passive_wiring:1 | 239/240/2172/2174/2312/2323 | 267/540/541/542/543/544/549/553+6 | 1559/1560 | 3 |
| `SURV_MODE` | `--surv-mode` | theory_order:10, conftest:3, attack_passive_wiring:2 | 353/2151/2361 | 541/549/555/557/558/559/560/611+1 | 422/2397/2401/2976 | 11 |
| `SPEED_MEMO` | - | speed_memo:6 | 532/2387 | 287 | - | 1 |
| `W_MODE` | `--w-mode` | theory_order:5, conftest:3, kappa_vector:1, theory_bridge:1 | 235/335/339/1273/1282/2137/2162/2339+1 | 326/541/549/597/604 | 44/395/2585 | 12 |
| `CLOCK_HAND_MODE` | `--clock-hand` | theory_order:5, conftest:3 | 336/2134/2389 | 262/541 | 2558 | 2 |
| `W_ERR_MODE` | - | settle_cond_floor:11, don_walk:8, theory_order:3, crossing_bridge:1 | 1289/1307/1328 | 549 | - | 3 |
| `W_MOVER_MODE` | `--w-mover` | theory_order:10, settle_cond_floor:8 | 405/845/867/875 | 264/598/602/607/608 | - | 3 |
| `SIGMA_FLOOR_MODE` | `--sigma-floor` | settle_cond_floor:15 | - | 305 | - | 1 |
| `SETTLE_COND_MODE` | `--settle-cond` | settle_cond_floor:25, theory_order:3 | 528/2377 | 305 | - | 1 |
| `KAPPA_SIGMA_MODE` | `--kappa-sigma` | relative_ledger:9 | 1241 | 323/567/600 | - | 4 |
| `ACTIVATE_USES_STATE` | - | theory_order:5 | 2169/2327 | 262 | - | 1 |
| `CBAR_MODE` | `--cbar-mode` | crossing_bridge:15, theory_order:10, attack_passive_wiring:4, conftest:3 | 352/971/2150/2362 | 264/294/541/555/558/559/560 | 2406 | 13 |
| `DEFENDER_POWER_MODE` | `--defender-power` | hand_plan:5, attack_passive_wiring:1 | 532/2386 | - | - | 1 |
| `PLAY_COST_MODE` | - | theory_order:8 | 726/2168/2326 | 541 | - | 2 |
| `ATTACK_DON_COST_MODE` | - | theory_order:25 | 748 | 601 | - | 5 |
| `ATTACK_ABILITY_MODE` | `--attack-ability` | attack_passive_wiring:21 | 531/2380 | 265 | - | 1 |
| `PASSIVE_BODY_MODE` | `--passive-body` | attack_passive_wiring:22 | 531/2381 | 265 | - | 1 |
| `ATTACK_DON_MODE` | - | theory_order:5, don_ledger:2 | 2166/2325 | 291/541 | - | 1 |
| `OPTION_MODE` | - | theory_order:5, conftest:3, attack_passive_wiring:1 | 2165/2324/2391 | 541 | - | 3 |
| `BOUNDARY_MODE` | `--boundary` | transition_ledger:13 | - | 565 | - | 6 |

