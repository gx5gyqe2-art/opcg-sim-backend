//! 移植の段 2（葉）: `tests/scripts/theory_order.py` の葉——費用曲線・生存の重み・時計の傾きと勝率・トークンの読み。
//!
//! **値付けの核（`nu_of`・`attack_value(_don)`・`attack_stream`・`option_value`・`block_cost`）は段 3**（33 関数の輪）。
//! ここは輪に入らない関数だけ。各関数の注に元の名前を書く。大域の切替（`SURV_MODE` など）は**明示の引数**（E37）。
//! 段 3 以降が使う定数（`KO_P`・`DELTA`・`SC_*` ほか）も置く（`dead_code` を許す）。
//! 浮動小数の演算の順は Python と同じ（`a*b+c` を融合しない・`**` は libm の `pow`・`math.erf`/`erfc`/`exp` は libm）。

#![allow(dead_code)]

use super::numeric::{erf, erfc, exp, np_rint, pow, py_max, py_min, py_round, py_round_int, sqrt};

pub const MU: f64 = 0.0551;
pub const LAM: f64 = 0.1362;
pub const H_LIFE_TO_HAND: f64 = 0.89;
pub const KO_P: f64 = 0.289;
pub const PWR_EPS: f64 = 10.0;
pub const SAT_OVER_PWR: f64 = 2000.0;
pub const CBAR_CURVE: [(f64, f64); 5] = [(1000.0, 1.00), (2000.0, 1.28), (3000.0, 2.25), (4000.0, 2.78), (5000.0, 3.63)];
pub const CBAR_SLOPE: f64 = 0.66;
pub const KO_P_CURVE: [(f64, f64); 5] =
    [(2000.0, 0.2417), (4000.0, 0.2782), (6000.0, 0.3297), (8000.0, 0.2436), (f64::INFINITY, 0.1987)];
pub const R_TURNS: f64 = 4.128;
pub const DON_SHARE: f64 = 0.467;
pub const DELTA: f64 = 0.0277;
pub const LAM_BY_LIFE_0: f64 = 0.5;

/// `THETA = round((LAM - H_LIFE_TO_HAND * MU) / MU, 4)`
pub fn theta_const() -> f64 {
    py_round((LAM - H_LIFE_TO_HAND * MU) / MU, 4)
}

/// `SIGMA_D = math.sqrt(2.0) * SIGMA_TURN`（`SIGMA_TURN = 1.0`）
pub fn sigma_d_default() -> f64 {
    sqrt(2.0) * 1.0
}

/// `W_BAR = 0.5 / R_TURNS`
pub fn w_bar_default() -> f64 {
    0.5 / R_TURNS
}

// トークンの枠と列（`n_rel_feat.S_COLS`）
pub const S_POWER: usize = 0;
pub const S_COST: usize = 1;
pub const S_ATTACHED_DON: usize = 2;
pub const S_IS_REST: usize = 3;
pub const S_CAN_ATTACK: usize = 5;
pub const S_IS_BLOCKER: usize = 6;
pub const S_COUNTER: usize = 7;
pub const S_IS_CHAR: usize = 18;
pub const S_IS_EVENT: usize = 19;
pub const S_POWER_OPP_TURN: usize = 20;
pub const OWN_FIELD: std::ops::Range<usize> = 2..7;
pub const OPP_FIELD: std::ops::Range<usize> = 7..12;
pub const HAND: std::ops::Range<usize> = 12..22;
pub const SC_MY_LIFE: usize = 0;
pub const SC_OPP_LIFE: usize = 1;
pub const SC_MY_DON: usize = 2;
pub const SC_MY_HAND: usize = 6;
pub const SC_OPP_HAND: usize = 7;
pub const SC_IS_MY_TURN: usize = 11;
pub const SC_MY_LEADER_POWER: usize = 12;
pub const SC_OPP_LEADER_POWER: usize = 13;

/// 1 行のトークン（`[枠, 列]`・float32 を f64 に広げた値＝E26）。
#[derive(Clone, Debug)]
pub struct Tok {
    pub v: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Tok {
    pub fn new(v: Vec<f64>, rows: usize, cols: usize) -> Tok {
        assert_eq!(v.len(), rows * cols, "トークンの形");
        Tok { v, rows, cols }
    }
    #[inline]
    pub fn at(&self, s: usize, c: usize) -> f64 {
        self.v[s * self.cols + c]
    }
    /// 枠 `s` の全部の列。
    pub fn row(&self, s: usize) -> &[f64] {
        &self.v[s * self.cols..(s + 1) * self.cols]
    }
}

/// Python の `x or 0.0`（`0.0`／`-0.0` は偽＝正の 0 へ）。
#[inline]
pub fn or0(x: Option<f64>) -> f64 {
    match x {
        Some(v) if v != 0.0 => v,
        _ => 0.0,
    }
}

/// `theta_take(life, theta, mu, h)`（`life=None` は `theta`）。
pub fn theta_take(life: Option<f64>, theta: f64, mu: f64, h: f64) -> f64 {
    match life {
        None => theta,
        Some(l) => {
            if py_round_int(l) > 0 {
                theta
            } else {
                py_max(0.0, LAM_BY_LIFE_0 - h * mu) / mu
            }
        }
    }
}

/// `turn_weights(r_turns, ko_p, mode)`（`geo`＝`(1 − ko_p)^t`・`once`＝1）。
pub fn turn_weights(r_turns: f64, ko_p: f64, geo: bool) -> Vec<f64> {
    let s = if geo { 1.0 - ko_p } else { 1.0 };
    let mut out = Vec::new();
    let mut r = py_max(0.0, r_turns);
    let mut t: i32 = 1;
    while r > 1e-12 {
        let share = py_min(1.0, r);
        out.push(share * pow(s, t as f64));
        r -= share;
        t += 1;
    }
    out
}

/// `surv_turns`＝`float(sum(turn_weights(...)))`（Python 3.11 の `sum`＝0 から左へ）。
pub fn surv_turns(r_turns: f64, ko_p: f64, geo: bool) -> f64 {
    let mut s = 0.0;
    for w in turn_weights(r_turns, ko_p, geo) {
        s += w;
    }
    s
}

/// `cbar_of(v)`
pub fn cbar_of(v: f64) -> f64 {
    if v <= PWR_EPS {
        return 0.0;
    }
    let mut prev = 0.0;
    for (thr, cards) in CBAR_CURVE {
        if v <= thr + PWR_EPS {
            return cards;
        }
        prev = cards;
    }
    let over = (v - CBAR_CURVE[4].0) / 1000.0;
    prev + CBAR_SLOPE * over
}

/// `c_of(x, mode)`（`strict`＝`c̄(x + 1000)`・`loose`＝`c̄(max(x, 1000))`）。
pub fn c_of(x: f64, strict: bool) -> f64 {
    if x < -PWR_EPS {
        return 0.0;
    }
    if strict {
        cbar_of(x + 1000.0)
    } else {
        cbar_of(py_max(x, 1000.0))
    }
}

/// `slot_power(tok_row, slot)`＝`float(np.round(float(tok[s, 0]) * 1e4 / PWR_EPS) * PWR_EPS)`。
pub fn slot_power(tok: &Tok, slot: i64) -> Option<f64> {
    if slot < 0 || slot as usize >= tok.rows {
        return None;
    }
    Some(np_rint(tok.at(slot as usize, S_POWER) * 1e4 / PWR_EPS) * PWR_EPS)
}

/// `slot_don(tok_row, slot)`＝`int(round(float(tok[s, 2]) * 5.0))`。
pub fn slot_don(tok: &Tok, slot: i64) -> Option<i64> {
    if slot < 0 || slot as usize >= tok.rows {
        return None;
    }
    Some(py_round_int(tok.at(slot as usize, S_ATTACHED_DON) * 5.0))
}

/// `incoming_x(tok_row, don, mine)`（高い順・`don` は高い攻撃から 1 個 +1000 ずつ）。
pub fn incoming_x(tok: &Tok, don: f64, mine: Option<f64>) -> Vec<f64> {
    let mine = match mine {
        None => or0(slot_power(tok, 0)),
        Some(m) => m,
    };
    let mut pwr = vec![or0(slot_power(tok, 1))];
    for s in OPP_FIELD {
        if tok.at(s, S_IS_CHAR) > 0.5 {
            pwr.push(or0(slot_power(tok, s as i64)));
        }
    }
    // `pwr.sort(reverse=True)`（安定・浮動小数は `or 0.0` を通して符号つきの 0 が無い）
    pwr.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let nd = py_max(don, 0.0) as i64; // `int(max(don, 0))`（切り捨て）
    let n = pwr.len();
    for k in 0..nd.max(0) as usize {
        if n == 0 {
            break;
        }
        pwr[k % n] += 1000.0;
    }
    pwr.iter().map(|p| p - mine).collect()
}

/// `count_blockers(tok_row)`（自分の場のブロッカー）。
pub fn count_blockers(tok: &Tok) -> i64 {
    OWN_FIELD.filter(|&s| tok.at(s, S_IS_CHAR) > 0.5 && tok.at(s, S_IS_BLOCKER) > 0.5).count() as i64
}

/// `opp_chars_of(tok_row)`＝相手の場のキャラ `(パワー, ブロッカーか)`。
pub fn opp_chars_of(tok: &Tok) -> Vec<(f64, bool)> {
    OPP_FIELD
        .filter(|&s| tok.at(s, S_IS_CHAR) > 0.5)
        .map(|s| (or0(slot_power(tok, s as i64)), tok.at(s, S_IS_BLOCKER) > 0.5))
        .collect()
}

thread_local! {
    /// 候補 `OPCG_CLOCK_VALUE` の「ターンの途中の局面を規則どおりに読む」（`docs/reports/2026-10-10_clock_value.md` §1.3）。
    /// 既定は偽＝どの式も 1 ビットも変わらない。値段の前後の読みのあいだだけ真にする（`drv_tb` の別の核で）。
    static MID_TURN_RULES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 規則どおりの読みが入っているか（§1.3）。
pub fn mid_turn_rules() -> bool {
    MID_TURN_RULES.with(|c| c.get())
}

/// 規則どおりの読みを入れ切りする（戻り＝前の値）。
pub fn set_mid_turn_rules(on: bool) -> bool {
    MID_TURN_RULES.with(|c| c.replace(on))
}

/// §1.3 の 1: リーダーを今のターンの攻め手に数えるか（既定はいつも数える・規則どおりの読みでは `can_attack_now` で読む）。
pub fn leader_attacks_now(tok: &Tok) -> bool {
    !mid_turn_rules() || tok.at(0, S_CAN_ATTACK) > 0.5
}

/// §1.3 の 2: 段 2 以降・相手のターンのパワーから外す、このターンに付けたドンの分（既定は 0）。
pub fn later_don_off(tok: &Tok, s: usize) -> f64 {
    if mid_turn_rules() {
        tok.at(s, S_ATTACHED_DON) * 5.0 * 1000.0
    } else {
        0.0
    }
}

/// `own_attackers_of(tok_row, opp_leader_power)`（リーダー＋このターン攻撃できるキャラの `x`）。
pub fn own_attackers_of(tok: &Tok, olp: f64) -> Vec<f64> {
    let mut xs = if leader_attacks_now(tok) { vec![tok.at(0, S_POWER) * 1e4 - olp] } else { Vec::new() };
    for s in OWN_FIELD {
        if tok.at(s, S_IS_CHAR) > 0.5 && tok.at(s, S_CAN_ATTACK) > 0.5 {
            xs.push(tok.at(s, S_POWER) * 1e4 - olp);
        }
    }
    xs
}

/// `crossing_bridge.opp_attackers_of(tok, my_leader_power)`（相手のリーダー＋場のキャラ全部）。
pub fn opp_attackers_of(tok: &Tok, mlp: f64) -> Vec<f64> {
    let mut xs = vec![tok.at(1, S_POWER) * 1e4 - mlp];
    for s in OPP_FIELD {
        if tok.at(s, S_IS_CHAR) > 0.5 {
            xs.push(tok.at(s, S_POWER) * 1e4 - mlp);
        }
    }
    xs
}

/// `crossing_bridge._opp_active_blockers`／`_own_active_blockers`（キャラ・ブロッカー・レストでない）。
pub fn active_blockers(tok: &Tok, slots: std::ops::Range<usize>) -> i64 {
    slots
        .filter(|&s| tok.at(s, S_IS_CHAR) > 0.5 && tok.at(s, S_IS_BLOCKER) > 0.5 && tok.at(s, S_IS_REST) <= 0.5)
        .count() as i64
}

/// `ko_p_of(power, fallback)`（パワー別の山形・`None` は定数）。
pub fn ko_p_of(power: Option<f64>, fallback: f64) -> f64 {
    let p = match power {
        None => return fallback,
        Some(p) => p,
    };
    for (hi, v) in KO_P_CURVE {
        if p < hi {
            return v;
        }
    }
    KO_P_CURVE[4].1
}

/// `power_band_of(power, opp_leader_power)` → 0 `lt_leader`／1 `leader_to_sat`／2 `over_sat`。
pub fn power_band_of(power: f64, olp: f64) -> usize {
    let x = power - olp;
    if x < 0.0 {
        0
    } else if x < SAT_OVER_PWR {
        1
    } else {
        2
    }
}

pub const BAND_NAMES: [&str; 3] = ["lt_leader", "leader_to_sat", "over_sat"];
pub const SHIELD_TERM: [f64; 3] = [0.0597, 0.0409, 0.0271];

/// `shield_of(power, opp_leader_power)`
pub fn shield_of(power: f64, olp: f64) -> f64 {
    SHIELD_TERM[power_band_of(power, olp)]
}

/// `_attack_bound(is_leader, theta, mu, nu_target, blockers)`＋値段の文脈（`CUT_PRICER is not None` かつ
/// `CUT_TAKE_CARD`）。`take_card`＝文脈の中の `CUT_TAKE_CARD`（文脈の外・`None` なら旧の受ける値）。
pub fn attack_bound(is_leader: bool, theta: f64, mu: f64, nu_target: Option<f64>, blockers: &[(f64, f64)], take_card: Option<f64>) -> f64 {
    let mut take = if is_leader {
        theta * mu
    } else {
        match nu_target {
            Some(n) => n,
            None => theta * mu,
        }
    };
    if let Some(tc) = take_card {
        if is_leader {
            take += H_LIFE_TO_HAND * (mu - tc);
        }
    }
    let mut b = take;
    for &(_pb, nub) in blockers {
        b = py_min(b, nub);
    }
    b
}

/// `clock_scale(t_me, t_opp, mode)`
pub fn clock_scale(t_me: f64, t_opp: f64, mode: &str) -> f64 {
    let a = py_max(0.0, t_me);
    let b = py_max(0.0, t_opp);
    match mode {
        "sum" => a + b,
        "mean" => 0.5 * (a + b),
        "max" => py_max(a, b),
        "geo" => sqrt(a * b),
        _ => sqrt(a * a + b * b),
    }
}

/// `mover_shift(mover)`
pub fn mover_shift(mover: bool) -> f64 {
    if mover {
        0.5
    } else {
        0.0
    }
}

/// `whole_clock_scale(t)`＝`max(1.0, t)`
pub fn whole_clock_scale(t: f64) -> f64 {
    py_max(1.0, t)
}

/// `_upper(x, mu, sd)`＝`P(X > x)`・`X ~ N(mu, sd²)`。
pub fn upper(x: f64, mu: f64, sd: f64) -> f64 {
    if sd <= 0.0 {
        return if mu > x { 1.0 } else { 0.0 };
    }
    0.5 * erfc((x - mu) / (sd * sqrt(2.0)))
}

/// `whole_turn_race_prob(t_me, t_opp, sd_me, sd_opp, k0)`（K-1 の整数ターンの競争）。
pub fn whole_turn_race_prob(t_me: f64, t_opp: f64, sd_me: f64, sd_opp: f64, k0: i64) -> f64 {
    let t_me = py_max(0.0, t_me);
    let t_opp = py_max(0.0, t_opp);
    let sd_me = py_max(0.0, sd_me);
    let sd_opp = py_max(0.0, sd_opp);
    let k0 = k0.max(1);
    let opp_reaches_after = |k: i64| -> f64 {
        if k <= 1 {
            1.0
        } else {
            upper((k - 1) as f64, t_opp, sd_opp)
        }
    };
    let denom = if k0 <= 1 { 1.0 } else { upper((k0 - 1) as f64, t_me, sd_me) };
    if denom <= 0.0 {
        return opp_reaches_after(k0);
    }
    let mut num = 0.0;
    let mut k = k0;
    let mut tail_lo = denom;
    loop {
        let tail_hi = upper(k as f64, t_me, sd_me);
        let pk = tail_lo - tail_hi;
        let q = opp_reaches_after(k);
        num += pk * q;
        if tail_hi <= 0.0 || q <= 0.0 {
            break;
        }
        tail_lo = tail_hi;
        k += 1;
    }
    py_min(1.0, py_max(0.0, num / denom))
}

/// 時計の読みの大域（`W_ERR_MODE`・`SIGMA_REL`・`SIGMA_D`・`W_MODE`・`W_BAR`）。
#[derive(Clone, Debug)]
pub struct ClockCfg {
    pub w_err_rel: bool,
    pub sigma_rel: Option<f64>,
    pub sigma_d: f64,
    /// `W_MODE in ("clock", "curve")`
    pub w_slope: bool,
    pub w_bar: f64,
}

/// `prob_of_d(d, sigma_d, t_me, t_opp, scale_mode, mover)`
pub fn prob_of_d(cfg: &ClockCfg, d: f64, sigma_d: Option<f64>, t_me: Option<f64>, t_opp: Option<f64>, scale_mode: &str, mover: bool) -> f64 {
    let rel = if cfg.w_err_rel && sigma_d.is_none() { cfg.sigma_rel.zip(t_me).zip(t_opp) } else { None };
    if let Some(((sr, a), b)) = rel {
        if mover && scale_mode == "hyp" {
            return whole_turn_race_prob(a, b, sr * whole_clock_scale(a), sr * whole_clock_scale(b), 1);
        }
    }
    let d = d + mover_shift(mover);
    if let Some(((sr, a), b)) = rel {
        let sd = sr * clock_scale(a, b, scale_mode);
        if sd <= 0.0 {
            return 0.5;
        }
        return 0.5 * (1.0 + erf(d / (sd * sqrt(2.0))));
    }
    let sd = sigma_d.unwrap_or(cfg.sigma_d);
    0.5 * (1.0 + erf(d / (sd * sqrt(2.0))))
}

/// `w_of_d(d, sigma, mover)`
pub fn w_of_d(cfg: &ClockCfg, d: f64, sigma: Option<f64>, mover: bool) -> f64 {
    let sigma = sigma.unwrap_or(cfg.sigma_d);
    let z = (d + mover_shift(mover)) / sigma;
    exp(-0.5 * z * z) / (sigma * sqrt(2.0 * std::f64::consts::PI))
}

/// `state_factor(d, mode, t_me, t_opp, scale_mode)`（`mode=None` は `cfg.w_slope`）。
pub fn state_factor(cfg: &ClockCfg, d: f64, slope: bool, t_me: Option<f64>, t_opp: Option<f64>, scale_mode: &str) -> f64 {
    if !slope {
        return 1.0;
    }
    local_slope(cfg, d, t_me, t_opp, scale_mode) / cfg.w_bar
}

/// 局面の傾き `w(D)`（`state_factor` の分子）: 時計が渡れば幅は `max(σ_rel·s(τ_me, τ_opp), σ_D)`（勝率の幅・10-05 の `match`）、
/// 渡らなければ `σ_D`。
pub fn local_slope(cfg: &ClockCfg, d: f64, t_me: Option<f64>, t_opp: Option<f64>, scale_mode: &str) -> f64 {
    let mut sd = None;
    if let (Some(sr), Some(a), Some(b)) = (cfg.sigma_rel, t_me, t_opp) {
        let s = clock_scale(a, b, scale_mode);
        if s > 0.0 {
            sd = Some(py_max(sr * s, cfg.sigma_d));
        }
    }
    w_of_d(cfg, d, sd, false)
}

/// `board_theta(tok_row, life, my_don, don_share, fallback)`
pub fn board_theta(tok: &Tok, life: f64, my_don: f64, don_share: f64, fallback: f64, strict: bool) -> f64 {
    let xs = incoming_x(tok, py_round_int(my_don * don_share) as f64, None);
    let g = (xs.len() as i64 - py_round_int(life) - count_blockers(tok)).max(0);
    if g <= 0 || g as usize > xs.len() {
        return fallback;
    }
    let mut cs: Vec<f64> = xs.iter().map(|&x| c_of(x, strict)).collect();
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cs[g as usize - 1]
}

/// `theta_of(tok_row, life, my_don, mode, theta, don_share)`
pub fn theta_of(tok: &Tok, life: f64, my_don: f64, mode: &str, theta: f64, don_share: f64, strict: bool) -> f64 {
    if mode == "const" {
        return theta;
    }
    let b = board_theta(tok, life, my_don, don_share, theta, strict);
    if mode == "board" {
        return b;
    }
    py_max(theta, b)
}

/// `leader_power_opp_turn` のトークンだけの段（付与ドン無しのパワー）。続き（手番つきの継続効果）が要るかは
/// 呼ぶ側が `sc[11]`・`ci_row`・`idx2cid` で決める（`effect_value.continuous_self_mods` は段 3）。
pub fn leader_power_opp_turn_base(tok: &Tok, sc: Option<&[f64]>) -> f64 {
    let mut p = 0.0;
    if tok.cols > S_POWER_OPP_TURN {
        p = np_rint(tok.at(0, S_POWER_OPP_TURN) * 1e4 / PWR_EPS) * PWR_EPS;
    }
    if p <= 0.0 {
        if let Some(sc) = sc {
            p = np_rint(sc[SC_MY_LEADER_POWER] * 1e4 / PWR_EPS) * PWR_EPS;
        }
    }
    if p <= 0.0 {
        p = or0(slot_power(tok, 0));
    }
    p
}

/// `price_realised.nu_meas_of(power, opp_leader_power)`（帯ごとの実測 `ν`）。
pub fn nu_meas_of(power: f64, olp: f64) -> f64 {
    let x = power - olp;
    if x < -PWR_EPS {
        0.0690
    } else if x <= SAT_OVER_PWR + PWR_EPS {
        0.1503
    } else {
        0.2112
    }
}

/// `price_realised.side_nu_meas(tok, slots, opp_leader_power)`（帯は付与ドンを外した素のパワー）。
pub fn side_nu_meas(tok: &Tok, slots: std::ops::Range<usize>, olp: f64) -> f64 {
    let mut tot = 0.0;
    for s in slots {
        if tok.at(s, S_IS_CHAR) <= 0.5 {
            continue;
        }
        let pw = tok.at(s, S_POWER) * 1e4 - tok.at(s, S_ATTACHED_DON) * 5.0 * 1000.0;
        if pw < -PWR_EPS {
            continue;
        }
        tot += nu_meas_of(pw, olp);
    }
    tot
}

/// `price_realised.don_stock`／`don_attached`（`me`＝列 (2, 3, 14)・自分の場／`opp`＝(4, 5, 15)・相手の場）。
pub fn don_parts(sc: &[f64], tok: &Tok, me: bool) -> (f64, f64, f64, f64) {
    let (a, r, ld) = if me { (2, 3, 14) } else { (4, 5, 15) };
    let slots = if me { OWN_FIELD } else { OPP_FIELD };
    // `sum(gen)`＝0（int）から左へ
    let mut attached = 0.0;
    for s in slots {
        if tok.at(s, S_IS_CHAR) > 0.5 {
            attached += tok.at(s, S_ATTACHED_DON) * 5.0;
        }
    }
    (sc[a], sc[r], sc[ld], attached)
}

pub fn don_stock(sc: &[f64], tok: &Tok, me: bool) -> f64 {
    let (a, r, ld, att) = don_parts(sc, tok, me);
    a + r + ld * 5.0 + att
}

pub fn don_attached(sc: &[f64], tok: &Tok, me: bool) -> f64 {
    let (_a, _r, ld, att) = don_parts(sc, tok, me);
    ld * 5.0 + att
}

/// `price_realised.quality_correction(gains, mu)`＝`float(sum(float(g) - float(mu) for g in gains))`。
pub fn quality_correction(gains: &[f64], mu: f64) -> f64 {
    let mut s = 0.0;
    for g in gains {
        s += g - mu;
    }
    s
}
