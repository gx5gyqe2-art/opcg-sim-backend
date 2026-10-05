# Rust kernel for the `rule_don` defender model - design (read-only prep, tip f58d6b2e)

Audited tree: `origin/claude/cpu-spec-improvements-yw91jd` @ `f58d6b2e`. Python references are `tests/scripts/crossing_bridge.py` (CB) unless noted; `REF` = `tests/harness/rule_don_ref.py` (500 lines, pre-speed solver, verbatim from `7bce9082`). Companion: `switch_inventory.md` (which switches die before/after this port).

Nothing in the repo was edited. I ran only small, single-core, `nice 19` measurements on the 8 frames of `tests/fixtures/rd_speed_frames.json` (each < 0.3 s) and a 400k-value `round()` comparison between Python and a 20-line Rust program (section 4.1). I did not run `make test`, the tools, or any 120/88-solve corpus (those are not in the repo; section 6.2).

## 1. Summary and recommendations

1. **Boundary**: move the whole *cold solve* below `rule_don_solve` into Rust: defender DP, state-count layers + horizon selection, `_rule_don_masks`/`rules_steps`/knapsack, plan enumeration with dedupe and scoring, `rules_sched`, `walk_crossing` (sched path of `tau_grow`). Keep in Python: the two caches and the plan store, `model_horizon`, `attacker_ctx`, `attack_value` (called back for the attach-gain table), plan-dict assembly (`a_time`, `a_turn`, `theta_parts`, ...), `RULE_STATS`. API = one handle object per attacker context + one call per problem, **native PyO3 types, no JSON** (section 3.4).
2. **Exactness is achievable and testable**: every sum has a fixed order, every compare a fixed tolerance. 24 order/tie-sensitive places are listed in section 4.2. Three Python semantics need explicit Rust helpers: `round(x, n)` (shown bit-equal to `format!("{:.n}")`+parse on 400k values incl. exact ties), banker's `round(float)`, and `sum()` (naive on the project's Python 3.11; **3.12+ switched to compensated summation and would silently change the Python reference** - pin or replace by explicit loops).
3. **Where**: new module `rust/opcg_engine/src/theory/` in the existing crate and wheel (no new crate/dependency: the crate policy allows only `miniz_oxide` beyond PyO3/serde_json; write a 20-line Fx-style hasher). Build/gate as today: `make rust-develop`, `make test` = `cargo test` + pytest. Add a version handshake because `make test` does not rebuild the wheel.
4. **Staging** (advice that differs from "DP first, then sched"): **Stage 0** value-preserving clean-ups incl. the *kernel-touching switch deletions* of wave A (`switch_inventory.md`); **Stage 1** `Defender` (DP + budget + counter sets); **Stage 2a** enumeration + `rules_sched` + walk + horizon loop taking the `steps` that Python's `rules_steps` still builds; **Stage 2b** `rules_steps`/knapsack in Rust *after* the M-2 "speed in two stages" model change settles (that is the part most likely to change). **Stage 3** burn-in, delete the optimized Python solver.
5. **Expected speed-up** (cProfile shares on the 8 fixture frames, modelled as DP 20x, everything else that moves 50x, `attack_value` callbacks unchanged, +4% Python glue): stage 1 alone 1.4-7x; stage 2 total **3.6-9x depending on frame mix, about 8.5x on the DP 61% / sched 29% mix the user quoted** (stage 1 alone: 2.4x). The ceiling is set by the Python callbacks to `attack_value` (`_attach_gain`, 2-21% of solve time) and by `attacker_ctx`/marshalling; a module-level, argument-keyed memo of `_attach_gain` across attacker contexts (pure function of `x, k, olp, theta, mu, blockers`; bit-exact by construction) is the cheap Stage-0 lever (observation, not checked whether intended: `attacker_ctx.price` calls `attack_value` with the raw `mu` argument while `_attach_gain` uses `actx['mu']` = `cut_card_price(mu)`, so under `joint` the two do not evaluate the same function and cannot share results).
6. **Effort**: 13-18 working days for one engineer (details in section 8). **Risks**: second copy of the model while H-4 evolves, stale wheel, plan-store key, 3.12 `sum()`, hash-collision-safe memo, (section 7).
7. **Python solver fate** (section 9): keep the optimized Python solver as `OPCG_RD_KERNEL=py` only through the burn-in (one full measurement wave), then delete it (~700 lines) and keep **only `REF`**, made self-contained, as the oracle and the `=ref` debug path. Recommended because a third copy is the worst of the options and the optimized Python solver would be neither oracle nor production.

## 2. Anatomy of the cold solve (measured on the 8 fixture frames, f58d6b2e)

```
rule_don_solve (CB L1876-1945)                       [cache probes; PLAN_STORE; cut_context_key assert]
 |- _rule_don_masks (L1948-1986)  once per solve      -> for each play-set (<= 2^n_cand, cost <= budget): rules_steps (L1718-1780)
 |     rules_steps: nsteps = jmax = 30 steps x knapsack purse_plan_witness (L1560-1578) + _attach_gain (L1703) -> attack_value callbacks
 |- model_horizon (L1844-1858) -> h0                  [tau_grow, non-sched path]
 `- attempts h = h0, ... : memo = {} ; limit = EX_STATE_BUDGET (h > 1)          (L1911-1937)
      `- _rule_don_solve (L2136-2243, turns = h)
           for mask in masks: r0 = solve(first_of(zeros)); visit(ks) nested loops over attach counts, dedupe by sig
              solve(xf) = rule_guard_plan_ex (L1167) -> _rule_guard_plan_ex (L1324-1547)   <- the DP ("within")
              rules_sched (L1783-1831), walk_crossing (L1838) -> tau_grow (L3821), score tuple, strict "<"
      on _ModelBudget: _ex_fit_horizon (L2017) -> _ex_count_layers (L2035-2133)  (sets of states per layer, no values)
```

Measured shares of `rule_don_solve` cumulative time (cProfile, so Python-call-heavy parts are inflated equally on both sides of the comparison):

| frame | solve s | defender DP | rules_sched+walk | rules_steps (in masks) | of which `_attach_gain` | masks total |
|---|---|---|---|---|---|---|
| real#50 | 0.153 | 94.8% | 0.8% | 3.8% | 2.4% | 3.9% |
| real#54 | 0.272 | 74.9% | 5.4% | 17.2% | 10.5% | 17.4% |
| real#104 | 0.093 | 42.9% | 31.8% | 14.9% | 7.2% | 15.1% |
| real#119 | 0.074 | 35.5% | 34.7% | 18.5% | 9.1% | 18.7% |
| syn#3 | 0.206 | 42.9% | 16.5% | 34.2% | 21.0% | 34.7% |
| syn#27 | 0.127 | 49.1% | 13.7% | 31.2% | 18.7% | 31.8% |
| syn#23 | 0.040 | 93.4% | 1.5% | 4.0% | 2.1% | 4.2% |
| syn#24 | 0.030 | 88.2% | 3.8% | 5.7% | 3.1% | 5.9% |

DP unit costs (unique memo states, fixed horizon, cProfile on): **6.4-8.4 memo lookups per state**, 12.8-46 us/state all-in (13 us when the DP dominates; plan-heavy frames pay more per state for `rules_sched` at 123-144 us per call and 83-197 plans per frame). The fixtures are the small problems the tests need; production problems with budget 300,000 states have a higher DP share (the user's 61% / 29% split).

## 3. (a) The exact boundary

### 3.1 What moves, what stays

| Python (CB line) | Rust | stage | note |
|---|---|---|---|
| `_ex_prep` L1267, `_norm_seq` L1238, `_seq_prep` L1255 | `defender::Prep` | 1 | pure; `_SEQ_NORM/_SEQ_PREP` caches become per-call interning |
| `_ex_counter_sets` L1288-1321 | `defender::counter_sets` (per-attempt cache keyed `(x_id, hand_id, dl_id)`) | 1 | result order = lexicographic over kinds (kind 0 = largest counter value first), preserved |
| `_rule_guard_plan_ex` L1324-1547 (`within`, `turn`, `next_turn`, `comb`, `better`) | `defender::Defender::{solve, within, ...}` | 1 | budget via `Result<_, BudgetExceeded>` |
| `rule_guard_plan_ex` L1167-1213 wrapper | stays; builds a `Defender` or calls `guard_plan_ex` | 1 | `_RULE_EX_CACHE`/`_RULE_EX_MEMO` stay in Python for the un-budgeted path (pure caches) |
| `_ex_count_layers` L2035-2133, `_mask_firsts` L1989, `_ex_fit_horizon` L2017 | `layers.rs` sharing the *same* successor function as the DP | 2a | the Python duplicates the transition logic in two places; Rust should have one `successors()` used by both (this removes a drift risk) |
| attempts loop of `rule_don_solve` L1902-1944 | `solve::run` | 2a | |
| `_rule_don_solve` L2136-2243 (`visit`, `sig` dedupe, `solve`, `score`) | `plans.rs` | 2a | |
| `rules_sched` L1783-1831, `_tab`, `walk_crossing` L1838 + `tau_grow`/`rate_at` sched path L3779-3855 | `sched.rs` | 2a | `tau_grow`'s other parameters (shield, refill, step, ko_p, board/stock/flow) are not reachable from here (all 0) |
| `_rule_don_masks` L1948, `rules_steps` L1718, `purse_plan_witness` L1560 | `steps.rs` | 2b | `_attach_gain` stays a Python callback |
| `rule_don_solve` outer: `_RULE_DON_CACHE`, `PLAN_STORE`, `cut_context_key` assert, plan dict assembly | **stays** | - | `a_time`, `a_turn`, `theta_parts`, `attach*`, `rest`, `arrive`, `draw_types` stay Python (Q1 readers' definitions are the next thing to change) |
| `model_horizon` L1844 (needs `tau_grow` general path, `nu_meas_of`, `RATE_DECAY_MODE`) | **stays**; passes `h0` | - | one call per solve, ~15 float ops |
| `attacker_ctx` L1604, `hand_groups`, `attack_value`, `life_types_of`, `cut_*price`, `_rule_don_term`, `RULE_STATS` | **stay** | - | |

### 3.2 Kernel-supported configuration (what the kernel reads from global switches)

The Python code reads these globals inside the moved region: `RATE_T1_MODE` (rate_at L3792, rules_sched L1818), `RATE_RUSH_MODE` (rules_steps L1730), `SLOPE_EFFECT_MODE` (rules_steps L1734), `RATE_DON_PAY` (rules_sched L1799), `RATE_RAMP` (rate_at L3796), `PWR_EPS`, `SLOPE_FLOOR`, `RACE_CAP`, `DELTA`, `EX_STATE_BUDGET`, `EX_LAYER_COUNT`; and `actx["fixed"]` (`rule_don_purse`). The gate `attacker_ctx` L1616 returns `None` unless `DON_PURSE_MODE == "all"` and `RATE_DON_MODE != "purse"`.

* **Recommended**: delete the dead values of these switches *before* porting (wave A of `switch_inventory.md`, 12 values). The kernel then takes **no mode flags**, only numbers, and the parity matrix has one configuration.
* **If not deleted before the port**: the handle carries a frozen snapshot `(rush_on, eff_on, pay, t1_on, ramp)`, the Python wrapper falls back to Python when any is non-default, and the handle cache key includes the snapshot.
* Prices (`lam`, `lam_net`, `mu`, `olp`, `mlp`) arrive as numbers in `actx` (`_prices_of`), so `CUT_PRICE_MODE`/`CUT_TAKE_MODE` need no kernel support beyond *input values*. `cut_card_price()` raises for `joint_slice`; unaffected.

### 3.3 Inputs and outputs

Per attacker context (built once, cached by `actx["key"]`, like `_RULE_DON_CACHE` already assumes that key identifies the whole actx):

```
Ctx { att1_x: Vec<f64>            // float(att1[q][1]); slot ids are irrelevant to the DP
      later_x: Vec<f64>           // board_x of rules_steps
      cand: Vec<{cost:i64, atk:f64, eff:f64, rush:f64, bx:Option<f64>, is_rush:bool}>   // cand[i] = (cost, parts, bx, rush)
      budget:i64, kmax:i64, jmax:i64 (=nsteps), ds:Vec<f64>, a_tab, ar_tab, e_tab, flow: Vec<f64>
      lead_bare, chars_bare, lam, lam_net, mu, olp, mlp, delta: f64
      rest_blk: Vec<f64>, no_attack_now: bool
      gain: GainSource            // trait; PyO3 impl calls back _attach_gain(actx, x, k); cargo tests use a table
    }
```

Per problem: `cards_d: Vec<(f64,f64)>`, `don_d: f64`, `blk: Vec<f64>`, `life: i64 (already rounded)`, `turns: Option<i64>`, `life_types`, `draw_types: Vec<(f64,f64,f64)>`, `arrive: Vec<f64>`, `h0: Option<i64>` (from Python `model_horizon`), `budget: Option<usize>`, `layer_count: bool`, plus `nu: Vec<(m,nu)>` for every distinct margin of `blk ∪ rest ∪ arrive` computed by Python with `nu_meas_of(m + olp, mlp)` (exact, avoids porting `NU_MEAS`).

Output tuple: `(cut, stopped, best_score_parts, res{cut, stopped, alive, prevented, harms, theta, nu_all}, atk, rush, eff, incr, paid, play, k, xs_first, later_seq(raw, unsorted), value, tau, sched, harm_steps, h, h0, stats{attempt_fail, attempt_skipped, count_calls, count_fallback, n_states_per_attempt})`. The Python wrapper rebuilds the *dict in the same key order* (`repr` equality in `test_rd_speed` depends on dict order: atk, rush, eff, incr, paid, play, k, xs_first, later_seq, alive, value, tau, harm_steps, sched, theta, cut, stopped, then theta_parts, a_time, a_turn, attach_lead, attach, rest, arrive, draw_types, horizon, horizon0).

**Price context and deck composition (what the kernel does *not* see).** The kernel never touches `CUT_PRICER`, `CUT_PRICER_KEY`, `CUT_TAKE_CARD` or `cut_price.CutView`. Their whole effect reaches the model through `lam_net` and `mu`, which are already inside `actx` (built by `attacker_ctx` under `CP.defending(view)`; `lam` = `LAM` and the leader powers `olp`/`mlp` are unaffected): `lam_net` (= `cut_take_price(mu)` = `THETA*mu + H_LIFE_TO_HAND*(mu - CUT_TAKE_CARD)` under `joint`+`gbar`, `THETA*mu` under `flat`+`mu`), `mu` (= `cut_card_price(mu)` = `view.price(1, mu)` = the reservation average `gbar` under `joint`, `MU` under `flat`), plus `actx["cp"]` = `cut_context_key()` = `(CUT_PRICER_KEY, CUT_TAKE_CARD)` which `rule_don_solve` asserts equal to the live context (L1885) and which `PlanStore.key_of` hashes. All three stay in Python; the handle cache key must include `actx["cp"]` (it is inside `actx["key"]` already, L1679-1683). `joint_slice` (non-`avg` view) makes `cut_card_price` raise before the kernel is reached. Deck composition enters only as (i) `life_types`/`draw_types` tuples `((counter, don, prob), ...)` from `life_types_of(deck_ids, h)` (probabilities `h*k/n`, sorted by `(counter, cost)`) and (ii) the tables `a_tab`, `ar_tab`, `e_tab`, `flow` from `deck_refill` inside `attacker_ctx` - all plain float lists.

**Semantics refresher for the branches the DP folds** (so a reviewer can map E5-E8 to rules): at each declared attack the defender chooses (1) *take it* - if `lf > 0` one life card moves to hand with probability `p_i` per life-card type (kind index `type_ix[i]`), probability `p_none` that it is unusable, and the harm vector's first entry gets `+lam_net`; if `lf == 0` the defender dies and the branch returns the terminal harm `(lam - lam_net)*L0 + sum nu(blockers)`; (2) *block* with an active blocker (`x >= m - eps` kills it, otherwise it survives resting; value `+1.0` prevented, `+nu(m)` harm if killed); (3) *counter* with a minimal set of hand cards whose sum exceeds the excess (`+1.0` prevented, `+nc` cut cards, `+1.0` stopped, `+mu*nc` harm); the attacker picks the next declared attack that is worst for the defender. At the end of a turn (`rem` empty) blockers return and, if draw types exist, the defender draws one card (probabilities `dprob`, `pd_none` for a dead draw), `alive += 1.0`, and a leading `0.0` is prepended to the harm vector.

### 3.4 Marshalling cost and traps

* **Native types, not JSON.** The crate's other entry points pass JSON strings; for this kernel the float round trip must be exact. `serde_json` parses floats best-effort unless the `float_roundtrip` feature is on (it is not in `Cargo.toml`); PyO3's `f64`/`Vec<f64>` extraction is exact. Do not reuse the JSON convention here.
* `actx` holds Python lists/tuples/dicts (`cand[i][1]` is a dict). The handle extracts them once per distinct `actx["key"]` (~200 numbers, tens of microseconds); a solve then passes ~10 small lists (~10-20 us). Both are noise next to a >=100 us Rust solve; small solves are absorbed by `_RULE_DON_CACHE` anyway.
* The handle must be **excluded from the plan-store key** (`PlanStore.key_of` drops `_gain`; add the handle key the same way) and must not be pickled.
* `numpy` scalars inside lists extract fine as `f64`/`i64`; `bool` is accepted as an int: pass bools as `bool` typed arguments.
* Gain callback: `_attach_gain` is lazy and memoized *per actx* (`actx["_gain"]`, key `(round(x,3), k)`; first writer wins). Rust must call it lazily for exactly the `(x,k)` pairs Python would (not a superset), via `GainSource`. Guard: if two distinct board `x` share `round(x,3)` but differ, fall back to Python (powers are integers, so this should never fire).
* The core algorithm is pure Rust with no PyO3 types (trait-based `GainSource`), so `cargo test --no-default-features` (extension-module off) can run all of it.
* GIL: with a callback the section must hold the GIL; the pure DP (`Defender`) can `Python::detach` like `Game.decide` does (py_game.rs:407). Not needed for the first version.

### 3.5 Memo and state encoding (what has to be injective)

Python key (CB L1422-1543): `(ctx, t, rem, hand, ready, rested, pend, dl, lf)` with `ctx` = id of `(kinds, types, dtypes, seq, don, L0, cap, lam, lam_net, mu, olp, mlp)`, which within one attempt varies only through `seq` (the later-turn attack lists). Rust key = fixed byte array `[ctx:u16][t:u8][rem: sorted value-ids, 0-padded][hand counts per kind][ready][rested][pend: counts per distinct margin][dl-id:u8][lf:u8]` (~60 bytes), hashed with an in-crate Fx-style hasher and **compared in full** (no 64-bit-hash-only identity: a collision would silently corrupt a value or a budget count). Value = `{prev, cut, alive, stopped}` plus an offset/len into a `harms: Vec<f64>` arena (variable length, <= cap entries). Distinct attack values / margins / `dl` values are interned per attempt (`-0.0` canonicalized to `0.0` before interning, because Python tuple equality treats them equal and the **state count must match**).
Capability limits (return `Unsupported(reason)` -> Python fallback, never a wrong answer): kinds > 12, |rem| > 16, margins > 8, any hand count > 255, `cap` > 64, `dl` ids > 255.

## 4. (b) Exactness plan

### 4.1 Language-level rules

* IEEE-754 binary64 only, default rounding, **no fast-math, no `mul_add`/FMA, no reassociation, no `.sum()` shorthand** (explicit left-to-right loops keep the order visible; Python 3.11 `sum()` is a left fold starting at the int 0, so its first add is `0 + a`, which is exact). rustc does not contract `a*b+c` by default, and CPython performs each float op separately, so x86_64 and aarch64 agree.
* **`round(x, 9|12|3)`** = `format!("{:.*}", n, x).parse::<f64>()`. Verified here: 400,000 values (uniform, dyadic ties `k/1024` which are exact ties at 9 digits, 5e-10 perturbations, negatives) all bit-equal to Python (script and Rust program: `/tmp/claude-0/-home-user/5d50a018-367f-553d-bd42-6904065cc396/scratchpad/rustprep_work/rr/`). Put this in `theory::numeric::py_round` with a unit test that replays a 1e6-value vector file generated by Python.
* **`round(float)` (no digits)** = banker's rounding (`int(max(0, round(float(life))))`, `purse_plan_witness`'s `round(float(budget))`, `left_of`'s `int(round(...))`): `f64::round_ties_even` (stable since Rust 1.77; the crate declares `rust-version = "1.75"` - bump it or hand-write).
* `math.ceil`, `int()` truncation -> `f64::ceil`, `as i64` after the same guards.
* **`max(a, b)` / `min`**: Python returns `a` unless `b > a`. Use `py_max/py_min`; `f64::max` is unspecified for `0.0` vs `-0.0`.
* Tuple comparison for the score `(int, float, float, int)` uses `==` then `<` per element: implement lexicographic compare with `==` (so `-0.0 == 0.0`).
* **Python 3.12+ `sum()`**: since 3.12 `sum()` over floats uses compensated (Neumaier) summation. The project is on 3.11 (`pyproject` requires >= 3.11, Dockerfile `python:3.11-slim`, this machine 3.11.15). The kernel's Rust sums are naive, i.e. bit-equal to 3.11 only. Pin it: add `assert sys.version_info < (3, 12)` to the parity test **or** rewrite the kernel's `sum(...)` calls in the Python reference/optimized solver as explicit loops (no value change on 3.11).

### 4.2 Every place where order or ties matter (CB line numbers at f58d6b2e)

| # | where | what must match |
|---|---|---|
| E1 | `_ex_prep` L1270-1273 | `p_none = max(0.0, 1.0 - sum(p ...))`, same for `pd_none`; `sum` over `types` in given order; types with `p <= 0` dropped before |
| E2 | L1339 | `nu_all = sum(nu(m) for m in blk0 + rest0 + arr0)`: concatenation of three descending-sorted tuples, **not** one sorted list |
| E3 | L1340, L1547 | `theta = lam * L0 + nu_all` (zero case) / `lam * L0 + mu * cut + nu_all` (`L0` int -> float) |
| E4 | L1470 | `kill = (lam - lam_net) * float(L0) + sum(nu(m) for m in ready + rested + pend)` (order ready, rested, pend) |
| E5 | L1474-1504 | receive branch: accumulators `prev, cut, alive, st` updated per type in index order then the `p_none` term last; `hh` zero-extended; `hh[j] += p * r4[j]`; then `hh[0] = hh[0] + lam_net` (or `(0.0 + lam_net,)` if empty) |
| E6 | L1401-1411, L1436-1439 | `comb` over draw types, same pattern, `pd_none` term last |
| E7 | L1514-1519, L1528-1529 | block / counter candidates: `(r0 + 1.0, r1, r2, r3, harms with h0 + v)` and `(r0 + 1.0, r1 + nc, r2, r3 + 1.0, ...)`, `v = nu(m)` / `mu * nc` (`nc` int) |
| E8 | L1445-1447 | empty `rem`: `(r0, r1, r2 + 1.0, r3, (0.0,) + r4)` |
| E9 | L1376-1397 `better` | absolute tolerance `FEQ = 1e-9` in four stages (prevented max, cut min, alive max, then **harms elementwise with zero padding**, smaller earlier entry wins) |
| E10 | L1520-1522, L1530-1532 | update rule is `d = cand0 - best0; if d > FEQ or (not d < -FEQ and better(cand, best))` - tolerance makes it order-dependent: **candidate order must be the Python order**: receive, then ready blockers (descending, skip equal-to-previous), then counter sets in `_ex_counter_sets` order |
| E11 | L1534-1535 | attacker minimizes: `if best_att is None or better(best_att, best_def): best_att = best_def`, over distinct `x` in ascending `rem` order (skip `x == prev_x`) |
| E12 | L1296-1316 counter sets | `need = float(x) + 1000.0 - PWR_EPS`; `s = s + n * kval` accumulated over kinds in order; minimality `s - kvals[q] >= need` for used kinds; `dc > dl + 1e-9` prune (monotone, so prune timing is irrelevant, order of results is not); key `round(dl - dc, 9)`; first occurrence wins |
| E13 | L1349, L1328 | `don = round(float(don), 9)`; `L0 = int(max(0, round(float(life))))`; `cap = sum(cnt0) + 2*L0 + 2` if no horizon |
| E14 | L1344, L1259 | `hits_f` = ascending sorted, filtered `x >= -PWR_EPS`; `seq` entries filtered the same, `last_hit`, `repeat_hits` as defined |
| E15 | `rules_steps` L1729-1779 | `d = float(ds[min(step, len(ds)) - 1])`; groups order: remaining cards ascending, then attach groups per on-board body (`board_x` then bodies in insertion order); option `(k, {"attach": gain - k*delta})` only if `> 0.0`; `fb = base + on_val + sum(rush prices) + sum(gains)` left to right; `paid = float(sum(costs) + sum(ks))`; `remaining.remove` order |
| E16 | `purse_plan_witness` L1560-1578 | `n = int(max(0, round(float(budget))))`; `val = atk + eff + attach + attach_lead` (missing keys 0.0); strict `best[b-c] + val > nb[b] + 1e-12`; reproduce the pick path with per-layer choice arrays (equivalent to the Python copying of `pick` lists) |
| E17 | `_rule_don_masks` L1948-1986 | `p_atk/p_eff/p_rush` sums in play order; `cap_x = max([s_cnt + min(L0, max(0, n_h1-1)) * max_t] + blk)`; `caps = min(kmax, b, max(0, int(ceil((cap_x - x)/1000.0 - 1e-9))))`; mask order 0..2^n_cand ascending |
| E18 | `rules_sched` L1783-1831 | `left_of`: `int(round(max(0.0, d - paid) if RATE_DON_PAY else d))` clamped to the table; `_tab` clamps; per `j`: if `skip1` (`no_attack_now` and `RATE_T1_MODE == on`) and `j == 1` the entry is `0.0`; else `v = harms[j-1]` if `j <= len(harms)` else `steps[j-1].fb`, then **`v += seq_l[q]` for q in 0..2j-3 in order** (O(n^2) is required: prefix sums are not bit-equal), then `+= tj[0]`, `+= tj[2]` |
| E19 | `walk_crossing`/`tau_grow` L3821-3855, `rate_at` L3792-3796 | loop `j = 1..30`; `add = 0.0` if `RATE_T1_MODE == on and j0 + j - 1 <= 1` else `sched[min(max(1,j), n)-1] + RATE_RAMP * (j-1)`; `need = theta + r*j + used + step` with `r = used = step = 0.0`; `if f + add >= need: short = max(0.0, need - f); return float(j-1) + (short/add if add > SLOPE_FLOOR else 1.0)`; `f += add`; end `float(cap)` |
| E20 | `_rule_don_solve` L2186-2217 | `incr = (h[0] if h else 0.0) - h1_bare`; `val = p_atk + p_eff + incr + float(flow[max(0, budget - paid)])`; `sig = (tuple(sorted(xf)), paid)` dedupe (first kept); `score = (int(ceil(round(tau, 9) - 1e-9)), round(alive, 9), -round(val, 12), paid)`; replace only if strictly `<`; enumeration order `k` ascending, nested per attacker in `att1` order |
| E21 | L2225-2238 | `a_time = th / tau if tau > 1e-12 else max(float(sched[0]) if sched else 0.0, SLOPE_FLOOR)` (can stay in Python) |
| E22 | `_ex_fit_horizon` L2017-2032 | `fit = max(1, min(fit, h_fail))`; `tot` compared with `>` against the limit |
| E23 | attempts L1911-1937 | limit applies for `h > 1` only; count-layers tried at most once per solve (`tried_count`); `fit >= h` or `None` -> `count_fallback` and `h - 1` |
| E24 | `_ex_count_layers` L2035-2133 | reachable-state sets per layer (identical transition set as the DP, same `lf > 0` guards, same repeat/last_hit cut, `nxt is None` at the horizon end); early exit when `total > lim` |

Order-independence that Rust may exploit (all verified by reading): the DP value of a state is a pure function of the state (no alpha-beta, no pruning), so Rust may evaluate children in any internal order **as long as the fold orders E5-E11 are kept**; the *set* of reachable states (hence the budget decision) is independent of DFS order.

### 4.3 Budget counting, per-layer counts, horizon choice

* Python counts `len(memo)` after inserting each finished state (`within` post-order) and raises when `> lim`. An attempt fails **iff the number of distinct reachable states exceeds `lim`**, whatever the order, because states are only added. Rust: count inserts, abort as soon as `> lim` (saves time; the decision is the same).
* `h == 1` attempts run without a limit (`limit = None`): guarantees termination. Un-budgeted calls (`turns` given, `EX_STATE_BUDGET is None`) use a call-scoped memo in Rust (Python's global `_RULE_EX_MEMO` is a pure cache).
* The memo is **shared across all roots (plans) of an attempt**, not per root; keys carry `ctx`, so roots with equal normalized later-sequences share states. Rust must keep one memo per attempt.
* Layer counts: `sum(sizes[:h]) == len(memo of an attempt with horizon h)` is already a test (`test_layer_count_equals_the_states_each_horizon_actually_creates`); Rust reuses the same assertion against its own DP (`n_states` per attempt exposed) and against the Python `_ex_count_layers`.
* Selection rule to port verbatim: first attempt at `h0 = model_horizon(...)` (computed in Python, passed in); on failure, if `EX_LAYER_COUNT` and not yet tried and `h-1 > 1`: `fit = fit_horizon(..., h_fail=h, lim)`; `h = fit` if `fit < h` else `h - 1`; subsequent failures step by 1. Stats `attempt_fail/attempt_skipped/count_calls/count_fallback` are returned (not part of the plan).
* Since the Rust DP is cheap, an alternative is to drop layer counting (just retry `h-1`) - the decisions are identical (pure function of the problem) - but each failed attempt then costs up to `lim` states (about 0.3 s at 1 us/state). Keep the count; it shares code with the DP.

## 5. (c) Placement in `rust/opcg_engine`, build, gate

* **Crate layout** (no new crate, no new dependency):
  `src/theory/mod.rs` (kernel version const, re-exports), `numeric.rs` (`py_round`, `bankers_round`, `py_max/min`, `naive_sum`), `defender.rs` (prep, state/memo, DP, counter sets), `layers.rs`, `steps.rs`, `plans.rs`, `sched.rs`, `solve.rs` (attempt loop), `pyapi.rs` (`#[pyclass] RdCtx`, `#[pyfunction] rd_solve`, `Defender` pyclass, `rd_kernel_version`, `rd_py_round` for tests), `tests_*.rs`. Register in `lib.rs` next to `m.add_class::<py_game::Game>()` (lib.rs:402). Hasher: 20-line Fx variant inside `defender.rs` (crate policy `docs/rust_engine_plan.md` §12.4-1 forbids new dependencies).
* **Features**: none needed. PyO3 0.29 `abi3-py311` works for `#[pyclass]` with native `Vec<f64>`. The core must not name PyO3 types so `cargo test --no-default-features` can exercise it.
* **Build / install** (CI-less local gate): `make rust-develop` = `maturin develop --release` (venv) or `maturin build --release --compatibility linux --out target/wheels && pip install --force-reinstall --no-deps target/wheels/*.whl`; `make rust-test` = `cargo test --no-default-features`; `make test` = `rust-test` + pytest (`-m "not slow and not legacy"`) and **does not rebuild the wheel**. Release profile has `lto = true, codegen-units = 1`; ~3k lines add seconds. The Dockerfile builds the same wheel, so the research kernel would ship in the production image (harmless, ~100 KB); if that is unwanted, gate `theory` behind a default-on cargo feature and build the image with `--no-default-features`.
* **Stale wheel guard**: `rd_kernel_version() -> (u32 api, str src_hash)`; Python holds `RD_KERNEL_EXPECTED_API`. Tools: mismatch or missing function -> warn once and use Python (or hard-fail with `OPCG_RD_KERNEL=rs`). `make test` parity tests: **fail, not skip**, when the wheel is stale (same convention as the golden gate: CLAUDE.md "無いと golden ゲートは skip ではなく fail する").
* **How `make test` covers it**: (i) `cargo test`: numeric helpers vs recorded Python vectors; counter-set enumeration; DP on hand-computed tiny cases; budget/layer equivalence; Rust-only replay of the recorded corpus against Python-recorded **float bit patterns** (hex u64, so Rust never has to reproduce Python's `repr` formatting; no Python at runtime, ~seconds); (ii) pytest: Python reference vs Rust through PyO3 on random problems and fixtures (below). Tests are `cpu_infra` like `test_rd_speed.py` (excluded from `make test-fast`, included in `make test`). Budget: +20-30 s on the current ~122 s gate (keep the REF-vs-Rust comparisons to ~120 problems; the big corpus is verified against recorded bit patterns).
* Clippy (`make rust`) is not in `make test`; keep the module clippy-clean anyway (`-D warnings`).

## 6. (d) Parity test strategy and the fallback switch

### 6.1 Layers

| level | what | where |
|---|---|---|
| L0 | numeric helpers vs Python vectors (round 9/12/3, banker's, py_max, naive sum, ties) | cargo + pytest through `rd_py_round` |
| L1 | `counter_sets(x, hand, dl)` equal to `_ex_counter_sets` (order included) on random kinds/hands | pytest |
| L2 | DP: `rule_guard_plan_ex` result dict `repr`-equal on random problems (40 existing `_rand_problem` + 400 new, budgets None/25/40/300/3000, explicit horizons 1-4), **and** Rust `n_states` == Python `len(memo)` and == `sum(_ex_count_layers[:h])` | pytest |
| L3 | layer counts and `fit` horizon equal on frames and random | pytest |
| L4 | `rule_don_solve` `repr`-equal to `REF.solve` and to the optimized Python on: the 8 frames at budgets 300000 and 150 (existing), random problems, **the 120 real / 88 synthetic captured solves**, `fixed` problems until deleted | pytest + recorded bit patterns in cargo |
| L5 | whole-tool JSON identity on 30 games (real and synthetic): run `crossing_bridge`, `theory_bridge`, `relative_ledger`, `transition_ledger`, `win_calib` with `OPCG_RD_KERNEL=py` and `=rs`, same `OPCG_PLAN_STORE`-less cold run; compare JSON after popping timing fields | script, run by hand at each stage and before deleting Python; not in `make test` (minutes) |
| L6 | fuzz CLI `tests/scripts/rd_kernel_fuzz.py --n 20000 --seed S` (random problems incl. larger hands to reach 10^5 states) py vs rs, for overnight runs | not in the gate |

The existing `test_rd_speed.py` assertions stay and are run **three-way** (Rust, optimized Python, REF) until stage 3.

### 6.2 The 120/88 captured corpora

Not found in the repo at f58d6b2e (only the 8-frame `rd_speed_frames.json`). To be committed as `tests/fixtures/rd_kernel_corpus_{real,syn}.jsonl.gz` (est. 2-5 KB per solve before compression, < 1 MB total): per line `{src, args:[cards, don, blk, life, actx(minus handle/_gain), turns, life_types, draw_types, arrive], cp:[pricer_key, take_card], budget, expected: result flattened to (path, value) pairs with floats stored as 16-hex-digit bit patterns, plus sha1(repr(result)) for the pytest side}` with `_tup` conversions as in `test_rd_speed._frames()`. A capture hook `OPCG_RD_CAPTURE=<dir>` in a thin wrapper (test-harness side, not in `rule_don_solve`) samples solves from a normal tool run (the `PlanStore` cannot be used: its keys are hashes, the inputs are not recoverable). Add `make rd-golden` (like `make golden-audit`) to regenerate the expected bit patterns from REF when the model is changed on purpose, with the same "review the diff" rule as the engine goldens.

### 6.3 Fallback / A-B switch

Environment variable **`OPCG_RD_KERNEL`**, read once in `crossing_bridge` (and exposed as `CB.RD_KERNEL`):

* `rs` - Rust kernel; **error** if the wheel is missing/stale (used by the parity tests and the L5 runs);
* `py` - current optimized Python solver (default until burn-in ends; after burn-in the default flips to `auto`);
* `auto` - Rust if present and fresh, else `py`; silent fallback is logged once on stderr;
* `both` - run Rust and Python, assert `repr` equal (raise with both plans on mismatch), return the Python result; used in measurement sessions during burn-in (cost: Python speed);
* `ref` - the pre-speed `REF.solve` (debug only; very slow).
The mode never enters tool output; stderr gets one summary line (calls, hits, fallbacks) like `plan_store.report()`. Unsupported shapes (section 3.5) fall back per call and are counted in `RULE_STATS["rd_kernel_fallback"]`.

### 6.4 Plan store, fingerprints

`PlanStore.key_of` hashes `SOLVER_VERSION`, **the digest of every `tests/scripts/*.py`**, a snapshot of solver-read globals, the whole `actx`, prices, budget. Consequences for this project:

* The Rust kernel needs its own identity in the key: `KERNEL_FP = (api, src_hash)` where `src_hash` is computed by `build.rs` over `src/theory/**` (so a Rust edit invalidates stored plans). During burn-in use `engine_tag` = `py` or `rs:<KERNEL_FP>` in the key so a kernel bug cannot poison Python-made entries (and vice versa); when `both` passes on the full corpora, drop the tag from the key and bump `SOLVER_VERSION` to `rd-speed-2` (stores become cold once, deliberately) so entries are shared across engines.
* **Narrow the source digest** from "all of `tests/scripts`" to "the solver's own source": the functions in `_SOLVER_FUNCS` plus everything `solver_reads` already traces. Reason: today any edit under `tests/scripts` (including deleting an unrelated dead switch, `switch_inventory.md` section 1 finding 2) makes every store cold - the sigma re-measure (`claude/sigma-wip`, hours per step) uses a store. Do this in Stage 0, before the first deletion, and re-verify with a cold/warm identity run.
* `test_the_solver_version_is_bumped_when_the_solver_changes` fingerprints Python function sources; extend it with the Rust `src_hash` once `src/theory` exists (a Rust edit without a `SOLVER_VERSION` bump must fail the test).
* The handle (`actx["_rs"]` or module-level dict) is excluded from `key_of` (like `_gain`).

## 7. (e) Risks

| # | risk | why | mitigation |
|---|---|---|---|
| R1 | **Second copy of the model drifts while H-4 evolves** | open items in the H-4 report: mirror reading time, M-2 "speed in two stages", sigma/harm-profile re-measure, slower time readers (1.4x) | split so the volatile parts stay Python (plan assembly, `a_time`, `model_horizon`; `rules_steps` until M-2 lands); mark every contract point with the same `EXACT-n` tag in `REF` and in Rust (grep-able, E1-E24 above); change protocol = edit REF first, `make rd-golden`, then Rust, then green parity; extend the `SOLVER_VERSION` bump rule to either implementation |
| R2 | Python-object-heavy inputs | `actx` is a dict of lists/tuples/dicts; tests copy it with `dict(ax)` and set `ax["fixed"]`/`ax["key"]` by hand | handle cache keyed by `actx["key"]` (same assumption as `_RULE_DON_CACHE`); frozen flag snapshot in the handle key; shape limits with fallback |
| R3 | plan-store key | see 6.4 | engine tag then kernel fingerprint; narrow digest first |
| R4 | float semantics | `round`, banker's, `max`, `-0.0`, `sum()` on 3.12+ | section 4.1 helpers + L0 tests; pin Python < 3.12 in the parity test |
| R5 | stale wheel / skew | `make test` does not rebuild | version handshake; parity tests fail on skew |
| R6 | state identity drift changes the budget decision | a different canonicalization (`-0.0`, interning) changes `len(memo)` by a few states, flipping a cut | L2/L3 compare counts exactly; full-key comparison in the memo |
| R7 | hash collisions | 64-bit hash as identity | full key equality; no probabilistic keys |
| R8 | memory | 300k states x harms vector (<= 30 f64) can reach ~80 MB | arena per attempt, freed between attempts; cap on `cap` |
| R9 | recursion depth | DFS depth ~ cap x (len(rem)+1) <= ~400 frames | fixed limits (section 3.5); run in a thread with explicit stack if the limits are raised |
| R10 | debuggability | a Rust-only wrong answer is invisible | `both` mode in every measurement session until burn-in ends; `n_states` and per-attempt stats exposed |
| R11 | build/ship | adds research code to the production wheel | cargo feature if wanted (section 5) |
| R12 | test time | REF is slow | recorded bit patterns for the large corpus; REF only on ~120 problems |
| R13 | `EX_SPEED_STATS`/`RULE_STATS` disclosure | outputs/stderr read them | return the same counters; `plan_cut`/`plan_n` counting stays in `_rule_don_term` (Python) |
| R14 | the DP value is exact but **ties inside `better`** depend on `FEQ=1e-9` | tolerance makes comparisons non-transitive | replicate candidate order exactly (E10), test with generated tie-heavy problems (all-equal prices, equal blockers) |

## 8. Effort, staged plan, expected speed-up

### 8.1 Plan

| stage | content | gate | days |
|---|---|---|---|
| 0 | (a) freeze tag `theory-switches-final` (+ branch); (b) wave-A switch deletions (kernel-touching dead values) with per-variable identity checks; (c) narrow the plan-store digest; (d) capture hook + corpora + `rd_kernel_fuzz.py` + recorded bit patterns; (e) cross-actx memo for `_attach_gain` keyed on `(round(x,3), k, olp, theta_p, mu, tuple(blk_a))` (value-preserving; measure the hit rate first) | identity runs (`switch_inventory.md` section 10); `test_rd_speed` | 3-4 (after the sigma merge) |
| 1 | `numeric.rs`, `defender.rs` (prep, memo, DP, counter sets, budget), `Defender` pyclass, Python glue in `_rule_guard_plan_ex` behind `OPCG_RD_KERNEL`; L0-L3 tests | L0-L3 + `test_rd_speed` three-way | 4-6 |
| 2a | `layers.rs`, attempts loop, `plans.rs` (visit/sig/score), `sched.rs`; Python still supplies `steps` from `rules_steps` | L4 + L5 on 30 games | 4-5 |
| 2b | `steps.rs` + knapsack witness + `GainSource` callback; do after M-2 settles | L4 + L5 | 2-3 |
| 3 | burn-in with `both` over a full measurement wave (the next 300/900-game runs); flip default to `auto`; delete optimized Python (section 9); ledger/TEST_SPEC rows; version bump `rd-speed-2` | full gate; L5 | 2 |
| **total** | | | **13-18** |

### 8.2 Expected speed-up, with reasoning

Model (fractions are shares of `rule_don_solve` cumulative time from the table in section 2): `T1 = (1 - DP) + DP/20 + g`, `T2 = DP/20 + (sched + steps_without_gain + other)/50 + gain + g`, `g` = 4% of solve time for Python glue outside the solve (`attacker_ctx`, the user's "97% of time is in `rule_don_solve`" leaves 3%) plus ~1% marshalling/dict assembly. The 20x for the DP comes from the measured 6.4-8.4 lookups and ~13 us per unique state in Python vs ~0.5-1 us in Rust (hash of a ~60-byte key 30-40 ns, ~8 probes, a few float ops, no allocation); the 50x for `rules_sched`/knapsack/enumeration is loop code (O(30^2) adds per plan, 30 knapsack steps per play set) that Rust runs in microseconds instead of 125-145 us per `rules_sched` call. `_attach_gain` keeps its Python cost (callbacks).

| frame (DP / sched+walk / steps w/o gain / gain / other, % of solve) | stage 1 only | stage 2 | limiting term |
|---|---|---|---|
| real#50 (94.8 / 0.8 / 1.4 / 2.4 / 0.5) | 7.2x | 8.9x | glue + gain |
| real#54 (74.9 / 5.4 / 6.7 / 10.5 / 2.3) | 3.0x | 5.4x | gain 10.5% |
| real#104 (42.9 / 31.8 / 7.7 / 7.2 / 10.2) | 1.6x | 7.0x | gain + glue |
| real#119 (35.5 / 34.7 / 9.4 / 9.1 / 11.1) | 1.4x | 6.3x | gain + glue |
| syn#3 (42.9 / 16.5 / 13.2 / 21.0 / 5.9) | 1.6x | 3.6x | gain 21% |
| syn#27 (49.1 / 13.7 / 12.5 / 18.7 / 5.4) | 1.7x | 3.9x | gain 18.7% |
| syn#23 (93.4 / 1.5 / 1.9 / 2.1 / 0.9) | 6.5x | 9.2x | glue + gain |
| syn#24 (88.2 / 3.8 / 2.6 / 3.1 / 2.0) | 4.9x | 8.6x | glue + gain |
| user's quoted mix (61 / 29 / ~2 / ~5 / ~3 incl. glue) | 2.4x | ~8.5x | gain + glue |

Hence "another 5-10x" is realistic, ~9x at best on DP-heavy problems and 3.6-4x on the synthetic plan-heavy ones, until `attack_value` is no longer called through Python. Two ways past that ceiling: port `attack_value` (theory_order, ~2.5k lines, a separate project) or memoize `_attach_gain` across attacker contexts (Stage 0e; useful only if the hit rate is high - measure first). Production problems (budget 300,000 states) have a higher DP share than these fixtures, which moves them toward the DP-heavy rows.

## 9. Fate of the Python solver (decision requested)

Options: (A) keep both Python implementations forever (three copies); (B) **recommended**: keep the optimized Python solver only through burn-in, then delete it and keep `REF` (self-contained) + Rust; (C) delete Python now.

* Python code that dies in (B) (CB, line spans at f58d6b2e): `_ex_prep` 19, `_norm_seq` 15, `_seq_prep` 10, `_ex_counter_sets` 34, `_rule_guard_plan_ex` 224, `_rule_don_masks` 39, `_mask_firsts` 26, `_ex_fit_horizon` 16, `_ex_count_layers` 99, `_rule_don_solve` 108, `rules_sched` 49, `_tab` 2, the attempts loop inside `rule_don_solve` (~35), `EX_LAYER_COUNT`, `_SEQ_*`/`_RULE_EX_*` caches, ~700 lines in total, and the `_SOLVER_FUNCS` fingerprint list in `test_rd_speed.py:296-299` shrinks accordingly. `rules_steps` (63), `purse_plan_witness` (19), `_attach_gain` (13) die with stage 2b.
* What must be made self-contained for `REF` to survive: `REF` imports `crossing_bridge` and `_sync()`s every non-owned global (REF L27-34), so it silently uses CB's **`rules_steps`, `walk_crossing`, `model_horizon`, `_attach_gain`, `_tab`, `_prices_of`, `purse_plan_witness`, `nu_meas_of`, `rate_at/tau_grow`**. Before deleting any of them from CB, copy them verbatim into `REF` (about 190 lines including `rate_at`/`tau_grow`, which other tools still use, so copy rather than move) and keep `attack_value` as the only external callback. Otherwise the oracle changes together with the code it is supposed to check.
* Why not (A): every H-4/M-2 model change is made three times and compared pairwise; REF is the only one that is "obviously the model", Rust the only one that is fast. Why not (C): no oracle during the first weeks. Why (B): one full measurement wave in `both` mode is the burn-in the user's workflow already has (the 300/900-game runs).
* The `ref` mode stays as the human-readable debugging path for a single problem.

## 10. Decisions for the user (draft in plain language, for the parent to relay)

**P1 - Remove the optimized Python solver after the Rust version has matched it for one full measurement wave?** (a) yes, keep only the original slow reference as the check (recommended: two copies instead of three); (b) keep the optimized Python forever (three copies; changes cost three edits); (c) remove it right after the unit tests agree (no cross-check on real runs). It came up because the tests currently compare the optimized solver with the original, and a third implementation would make every model change a three-way edit. If undecided the work starts on (a) and nothing is deleted until the burn-in run is reviewed. Caveat: after (a), a machine without the Rust build can only run the slow reference.
**P2 - Where does the model's "tunable" part live?** Keep the speed-side assembly (`rules_steps`, plan fields such as the time-reader speed) in Python until the planned "speed in two stages" change is done, and port it afterwards (recommended, costs about two extra days later), or port everything now and redo the Rust part when the model changes. Independent of P1.
