//! **M-2 の大きさの計器**（2026-10-08・`docs/reports/2026-10-08_m2_probe.md`・診断の出力だけ）。
//!
//! 環境変数 `OPCG_M2_PROBE`（空でも `0` でもない値）があるときだけ動く。既定（変数なし）では何もしない＝**既定の出力は 1 ビットも
//! 変わらない**（値を作る経路には触らない: ここは計画と行の読みが出来上がった後に、その値を読み直して別の欄に書くだけ）。
//!
//! 測るもの（ユーザ決定 2026-10-07「M-2 の残りの大きさを先に測る」）:
//! * 歩きの段ごとの値（`sched::rules_sched`）を出どころに分ける——守る側の計算の段の損害（地平の内側）・地平の先の `fb`・
//!   攻め手が引く札の流入（`a_tab`／`ar_tab`）・効果（`e_tab` とその段に出した札の `eff`）。計画が無い時計（攻め手か守り手の手札が
//!   読めず旧来の速さに落ちる道）は全部「落ちる道」。
//! * 時計が耐久に届く段（交点）が地平の内か外か、交点までに積んだ損害の出どころの割合。
//! * 地平を縮めた検算: 同じ計画（出す札の組・付与の枚数）のまま守る側の計算を短い地平で解き直し、尾を地平の外の値付けで歩かせた
//!   時刻を、元の（長い地平の）時刻と比べる。**診断だけ**（採る計画も既定の値も変えない）。
//! * 診断の別の歩き（引く札を抜く・効果を抜く・地平の先を「受ける」で粗く値付けする）。**式の提案ではない**。
//!
//! 守る側の解き方の原文（`defender.rs`・`plans.rs`・`sched.rs` ほか `build.rs` の 8 ファイル）には触らない——同じ足し算をここに写す。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use super::super::defender::{Defender, DpErr, Input, Output};
use super::super::leaves_to as lt;
use super::super::numeric::bankers_round;
use super::super::plans::{adraw_of, Mask, SolveIn};
use super::super::sched::{self, tab, StepIn, Tables};
use super::obj::V;

static INIT: Once = Once::new();
static ON: AtomicBool = AtomicBool::new(false);

/// 計器が動くか（`OPCG_M2_PROBE`・1 度だけ読む）
pub fn on() -> bool {
    INIT.call_once(|| {
        let v = std::env::var("OPCG_M2_PROBE").unwrap_or_default();
        ON.store(!v.is_empty() && v != "0", Ordering::SeqCst);
    });
    ON.load(Ordering::SeqCst)
}

fn fl(xs: &[f64]) -> V {
    V::list(xs.iter().map(|&x| V::Float(x)).collect())
}

#[inline]
fn max0(v: f64) -> f64 {
    if v > 0.0 {
        v
    } else {
        0.0
    }
}

/// `sched::left_of` の写し
fn left_of(tb: &Tables, i: usize, paid: f64) -> i64 {
    let d = tb.ds[i.min(tb.ds.len()) - 1];
    let l = bankers_round(max0(d - paid)) as i64;
    let nl = tb.a_tab.len() as i64 - 1;
    l.min(nl).max(0)
}

/// `sched::terms` の写し `(速攻の分, 素の体の分, 効果)`
fn terms(tb: &Tables, li: i64, eff: f64) -> (f64, f64, f64) {
    let ar = tab(tb.ar_tab, li);
    (ar, max0(tab(tb.a_tab, li) - ar), tab(tb.e_tab, li) + eff)
}

/// 歩きの段ごとの出どころ（`rules_sched` と同じ項の分け方）: `(段の芯, 芯が守る側の計算か, 引く札の流入, 効果)`。
/// 芯は地平の内側なら `harms[j−1]`、外なら `steps[j−1].fb`。段の値 = 芯 + 流入 + 効果（足す順だけ違う）。
pub fn sched_parts(tb: &Tables, harms: &[f64], steps: &[StepIn], paid1: f64) -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>) {
    let n = steps.len();
    let mut tm: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for i in 1..=n {
        let paid = if i == 1 { paid1 } else { steps[i - 1].paid };
        tm.push(terms(tb, left_of(tb, i, paid), steps[i - 1].eff));
    }
    let nh = harms.len();
    let (mut base, mut dp, mut flow, mut eff) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for j in 1..=n {
        if tb.no_now && j == 1 {
            base.push(0.0);
            dp.push(j <= nh);
            flow.push(0.0);
            eff.push(0.0);
            continue;
        }
        base.push(if j <= nh { harms[j - 1] } else { steps[j - 1].fb });
        dp.push(j <= nh);
        if !tb.d_tab.is_empty() && j <= nh {
            // 候補（`OPCG_DRAWN_ATTACKERS`）: 地平の内の流入は守る側の計算の損害に入っている（`rules_sched` と同じ）
            flow.push(0.0);
            eff.push(tm[j - 1].2);
            continue;
        }
        let mut f = 0.0;
        for t in tm.iter().take(j - 1) {
            f += t.0 + t.1;
        }
        f += tm[j - 1].0;
        flow.push(f);
        eff.push(tm[j - 1].2);
    }
    (base, dp, flow, eff)
}

/// `plans::first_of` の写し（今のターンの攻撃の並び）
fn first_of(inp: &SolveIn, m: &Mask, ks: &[i64]) -> Vec<f64> {
    if inp.no_now {
        return Vec::new();
    }
    let n = inp.att1_x.len();
    let mut xf: Vec<f64> = (0..n).map(|q| inp.att1_x[q] + 1000.0 * ks[q] as f64).collect();
    if m.hits1.len() > n {
        xf.extend_from_slice(&m.hits1[n..]);
    }
    xf
}

/// 同じ計画（`mask`・`ks`）のまま、守る側の計算を地平 `h` で解き直す（`plans::attempt` の `solve` と同じ入力）
fn resolve(inp: &SolveIn, m: &Mask, ks: &[i64], h: i64, d: &mut Defender) -> Result<Output, DpErr> {
    let (lam, lam_net, mu, olp, mlp) = inp.prices;
    let xf = first_of(inp, m, ks);
    let ad = adraw_of(inp, m, m.cost + ks.iter().sum::<i64>(), h);
    d.solve(&Input {
        cards: inp.cards,
        don: inp.don,
        xs_first: &xf,
        seq: &m.later_seq,
        blk: inp.blk,
        life: inp.life,
        turns: Some(h),
        life_types: inp.life_types,
        draw_types: inp.draw_types,
        lam,
        lam_net,
        mu,
        olp,
        mlp,
        rest: inp.rest,
        arrive: inp.arrive,
        nu: inp.nu,
        eps: inp.eps,
        feq: inp.feq,
        adraw: &ad,
    })
}

/// 同じ計画・同じ地平で、守る側の計算を「倒れた段にだけ 1 を置く」値段で解き直す（診断だけ）。
/// 値段を `λ_net = μ = ν = 0`・`(λ − λ_net)·L0 = 1` にすると、段の損害は**その段で倒れる確率**になる
/// （受けた 1 本・切った札・ブロッカーの値は 0、とどめの段にだけ 1）。守り手の選び方（防いだ本数・切る枚数・生き延びるターン）は
/// 値段に依らないので同じ（同点の割り切りだけが損害の並びで決まる）。ライフ 0 では `(λ − λ_net)·0 = 0` で測れない＝`None`。
fn resolve_death(inp: &SolveIn, m: &Mask, ks: &[i64], h: i64) -> Option<Output> {
    let l0f = bankers_round(inp.life);
    if l0f < 1.0 {
        return None;
    }
    let (_lam, _lam_net, _mu, olp, mlp) = inp.prices;
    let nu0: Vec<(f64, f64)> = inp.nu.iter().map(|&(k, _)| (k, 0.0)).collect();
    let xf = first_of(inp, m, ks);
    let ad = adraw_of(inp, m, m.cost + ks.iter().sum::<i64>(), h);
    let mut d = Defender::new(None);
    d.solve(&Input {
        cards: inp.cards,
        don: inp.don,
        xs_first: &xf,
        seq: &m.later_seq,
        blk: inp.blk,
        life: inp.life,
        turns: Some(h),
        life_types: inp.life_types,
        draw_types: inp.draw_types,
        lam: 1.0 / l0f,
        lam_net: 0.0,
        mu: 0.0,
        olp,
        mlp,
        rest: inp.rest,
        arrive: inp.arrive,
        nu: &nu0,
        eps: inp.eps,
        feq: inp.feq,
        adraw: &ad,
    })
    .ok()
}

/// 守る側の計算の入力（採った計画・地平 `h`）を辞書に書く（`docs/reports/2026-10-08_survival_split.md`・診断だけ）。
/// Python の器が手を加えて `m2.resolve` で解き直す（実際の攻撃の並び・実際の守り手の手札を写す）。
fn input_v(inp: &SolveIn, m: &Mask, ks: &[i64], h: i64) -> V {
    let (lam, lam_net, mu, olp, mlp) = inp.prices;
    let pairs = |xs: &[(f64, f64)]| V::list(xs.iter().map(|&(a, b)| fl(&[a, b])).collect());
    let trips = |xs: &[(f64, f64, f64)]| V::list(xs.iter().map(|&(a, b, c)| fl(&[a, b, c])).collect());
    let mut kv = vec![
        (V::s("cards"), pairs(inp.cards)),
        (V::s("don"), V::Float(inp.don)),
        (V::s("xs_first"), fl(&first_of(inp, m, ks))),
        (V::s("seq"), V::list(m.later_seq.iter().map(|s| fl(s)).collect())),
        (V::s("blk"), fl(inp.blk)),
        (V::s("life"), V::Float(inp.life)),
        (V::s("turns"), V::Int(h)),
        (V::s("life_types"), trips(inp.life_types)),
        (V::s("draw_types"), trips(inp.draw_types)),
        (V::s("lam"), V::Float(lam)),
        (V::s("lam_net"), V::Float(lam_net)),
        (V::s("mu"), V::Float(mu)),
        (V::s("olp"), V::Float(olp)),
        (V::s("mlp"), V::Float(mlp)),
        (V::s("rest"), fl(inp.rest)),
        (V::s("arrive"), fl(inp.arrive)),
        (V::s("nu"), pairs(inp.nu)),
        (V::s("eps"), V::Float(inp.eps)),
        (V::s("feq"), V::Float(inp.feq)),
        (V::s("no_now"), V::Bool(inp.no_now)),
    ];
    // 候補（`OPCG_DRAWN_ATTACKERS`）の引く札の型（無ければ欄を作らない＝既定の出力は不変）
    let ad = adraw_of(inp, m, m.cost + ks.iter().sum::<i64>(), h);
    if !ad.is_empty() {
        kv.push((
            V::s("adraw"),
            V::list(ad.iter().map(|st| V::list(st.iter().map(|&(x, r, p)| fl(&[x, if r { 1.0 } else { 0.0 }, p])).collect())).collect()),
        ));
    }
    V::dict(kv)
}

/// **`m2.resolve`**（`OPCG_M2_PROBE` のときだけ `entry` が通す・診断だけ）: `input_v` の形の辞書で守る側の計算を 1 回解く。
/// `death` が真なら値段を「倒れた段にだけ 1」（`resolve_death` と同じ）に置き換える＝段の損害が倒れる確率。
pub fn resolve_call(a: &V) -> Result<V, String> {
    let fv = |k: &str| -> Vec<f64> { a.get(k).items().iter().map(|x| x.f()).collect() };
    let pv = |k: &str| -> Vec<(f64, f64)> { a.get(k).items().iter().map(|e| (e.items()[0].f(), e.items()[1].f())).collect() };
    let tv = |k: &str| -> Vec<(f64, f64, f64)> {
        a.get(k).items().iter().map(|e| (e.items()[0].f(), e.items()[1].f(), e.items()[2].f())).collect()
    };
    let cards = pv("cards");
    let seq: Vec<Vec<f64>> = a.get("seq").items().iter().map(|s| s.items().iter().map(|x| x.f()).collect()).collect();
    let (xf, blk, rest, arrive) = (fv("xs_first"), fv("blk"), fv("rest"), fv("arrive"));
    let (lt_, dt_) = (tv("life_types"), tv("draw_types"));
    let mut nu = pv("nu");
    let life = a.get("life").f();
    let death = a.has("death") && a.get("death").truthy();
    let (mut lam, mut lam_net, mut mu) = (a.get("lam").f(), a.get("lam_net").f(), a.get("mu").f());
    if death {
        let l0f = bankers_round(life);
        if l0f < 1.0 {
            return Ok(V::None);
        }
        lam = 1.0 / l0f;
        lam_net = 0.0;
        mu = 0.0;
        for e in nu.iter_mut() {
            e.1 = 0.0;
        }
    }
    // 候補（`OPCG_DRAWN_ATTACKERS`）の引く札の型（無ければ空＝今のまま）
    let ad: Vec<Vec<(f64, bool, f64)>> = if a.has("adraw") {
        a.get("adraw")
            .items()
            .iter()
            .map(|st| st.items().iter().map(|e| (e.items()[0].f(), e.items()[1].f() > 0.5, e.items()[2].f())).collect())
            .collect()
    } else {
        Vec::new()
    };
    let mut d = Defender::new(None);
    let r = d
        .solve(&Input {
            cards: &cards,
            don: a.get("don").f(),
            xs_first: &xf,
            seq: &seq,
            blk: &blk,
            life,
            turns: Some(a.get("turns").int()),
            life_types: &lt_,
            draw_types: &dt_,
            lam,
            lam_net,
            mu,
            olp: a.get("olp").f(),
            mlp: a.get("mlp").f(),
            rest: &rest,
            arrive: &arrive,
            nu: &nu,
            eps: a.get("eps").f(),
            feq: a.get("feq").f(),
            adraw: &ad,
        })
        .map_err(|e| format!("m2.resolve: {e:?}"))?;
    Ok(V::dict(vec![
        (V::s("harms"), fl(&r.harms)),
        (V::s("alive"), V::Float(r.alive)),
        (V::s("cut"), V::Float(r.cut)),
        (V::s("prevented"), V::Float(r.prevented)),
        (V::s("stopped"), V::Float(r.stopped)),
        (V::s("theta"), V::Float(r.theta)),
    ]))
}

/// 計画の辞書に足す診断の欄（`OPCG_M2_PROBE` のときだけ `rd_run` が呼ぶ）。
/// `take`＝地平の先の段の攻撃を 1 本ずつ「受ける」の値段で数えた粗い芯（段の攻撃のうち効く本数 × 受ける値段）。
#[allow(clippy::too_many_arguments)]
pub fn plan_fields(inp: &SolveIn, m: &Mask, step_hits: &[Vec<f64>], ks: &[i64], paid: i64, res: &Output, h_used: Option<i64>, take_price: f64) -> Vec<(V, V)> {
    let tb = &inp.tables;
    let (base, dp, flow, eff) = sched_parts(tb, &res.harms, &m.steps, paid as f64);
    let fb: Vec<f64> = m.steps.iter().map(|s| s.fb).collect();
    let take: Vec<f64> = step_hits
        .iter()
        .map(|hs| hs.iter().filter(|&&x| x >= -lt::PWR_EPS).count() as f64 * take_price)
        .collect();
    let mut out = vec![
        (V::s("m2_base"), fl(&base)),
        (V::s("m2_dp"), V::list(dp.iter().map(|&b| V::Bool(b)).collect())),
        (V::s("m2_flow"), fl(&flow)),
        (V::s("m2_eff"), fl(&eff)),
        (V::s("m2_fb"), fl(&fb)),
        (V::s("m2_take"), fl(&take)),
        (V::s("m2_nh"), V::Int(res.harms.len() as i64)),
    ];
    // 地平を縮めた検算（同じ計画・同じ攻撃の並び・守る側の計算だけ短い地平で解き直す）
    let h = h_used.unwrap_or(res.harms.len() as i64);
    let tail = sched::tail_of(tb, &m.steps);
    let tau_full = {
        let sc = sched::rules_sched(tb, &res.harms, &m.steps, &tail, paid as f64);
        sched::walk_crossing(tb, &sc, res.theta)
    };
    let mut d = Defender::new(None);
    let mut shorts = Vec::new();
    // 同じ地平で解き直すと元と同じになるか（検算の道の確かめ）
    let same = match resolve(inp, m, ks, h, &mut d) {
        Ok(r2) => r2 == *res,
        Err(_) => false,
    };
    for k in 1..=3i64 {
        let hs = h - k;
        if hs < 1 {
            break;
        }
        let Ok(r2) = resolve(inp, m, ks, hs, &mut d) else { continue };
        // A: 元の段の損害を短い地平で切り、先を `fb` で歩く（耐久は元のまま）
        let sc_a = sched::rules_sched(tb, &res.harms[..(hs as usize).min(res.harms.len())], &m.steps, &tail, paid as f64);
        let tau_a = sched::walk_crossing(tb, &sc_a, res.theta);
        // B: 短い地平で解き直した段の損害と耐久で歩く（地平を短くしたときにモデルが言う時刻）
        let sc_b = sched::rules_sched(tb, &r2.harms, &m.steps, &tail, paid as f64);
        let tau_b = sched::walk_crossing(tb, &sc_b, r2.theta);
        // C: 解き直した段の損害で、耐久は元のまま（歩きの差だけ）
        let tau_c = sched::walk_crossing(tb, &sc_b, res.theta);
        shorts.push(V::dict(vec![
            (V::s("k"), V::Int(k)),
            (V::s("h"), V::Int(hs)),
            (V::s("tau_a"), V::Float(tau_a)),
            (V::s("tau_b"), V::Float(tau_b)),
            (V::s("tau_c"), V::Float(tau_c)),
            (V::s("theta_b"), V::Float(r2.theta)),
            (V::s("alive_b"), V::Float(r2.alive)),
            (V::s("harms_b"), fl(&r2.harms)),
        ]));
    }
    // 倒れる段の分布（同じ計画・同じ地平）——交点の時計が「倒れる時刻の真ん中」ではなく「最後の分岐が倒れる時刻」を
    // 言っているかを見る（`docs/reports/2026-10-08_late_winner.md`）
    match resolve_death(inp, m, ks, h) {
        Some(rd) => {
            out.push((V::s("m2_pdeath"), fl(&rd.harms)));
            out.push((V::s("m2_alive_death"), V::Float(rd.alive)));
            out.push((V::s("m2_cut_death"), V::Float(rd.cut)));
        }
        None => out.push((V::s("m2_pdeath"), V::None)),
    }
    out.push((V::s("m2_in"), input_v(inp, m, ks, h)));
    out.push((V::s("m2_tau_full"), V::Float(tau_full)));
    out.push((V::s("m2_theta_full"), V::Float(res.theta)));
    out.push((V::s("m2_alive"), V::Float(res.alive)));
    out.push((V::s("m2_cut"), V::Float(res.cut)));
    out.push((V::s("m2_resolve_same"), V::Bool(same)));
    out.push((V::s("m2_short"), V::list(shorts)));
    out
}

/// 段の値の列（`seat_slope_sched` の 30 段）を歩く（`outer::tau_grow` の `sched` の経路の写し・盾と補充は 0）。
/// 戻り＝`(τ, 届いた段 J〔届かなければ 0〕, 届いた段の割合, 段ごとに足した量)`。
pub fn walk(theta: f64, sched: &[f64], j0: i64, step: f64, cap: f64, floor: f64) -> (f64, i64, f64, Vec<f64>) {
    let mut f = 0.0;
    let ncap = cap as i64;
    let mut adds = Vec::new();
    for j in 1..=ncap {
        let add = if j0 + j - 1 <= 1 || sched.is_empty() { 0.0 } else { sched[(j.max(1) as usize).min(sched.len()) - 1] };
        let need = theta + 0.0 * j as f64 + 0.0 + (if j >= 2 { step } else { 0.0 });
        if f + add >= need {
            let short = max0(need - f);
            let frac = if add > floor { short / add } else { 1.0 };
            adds.push(add * frac);
            return ((j - 1) as f64 + frac, j, frac, adds);
        }
        adds.push(add);
        f += add;
    }
    (cap, 0, 0.0, adds)
}
