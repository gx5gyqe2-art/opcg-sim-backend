//! WP `theory-input` 段 0（2026-11-05・`docs/search_efficiency_plan.md` §7.82）: **打つ側に見える情報だけ**から読む理論の列。
//!
//! 入力は記録の 1 行（`sc`・`tok`・`ci`＝その席の符号化＝ネットが見るものと同じ）と両席のデッキの一覧だけ。相手の手札は
//! 枚数しか見えないので、**相手のデッキからの見積もり**（`deck_hand_read`: デッキのカウンター札の割合に比例した手札の組・
//! 札 1 枚の価格は `μ × デッキの切れる札の割合`・ライフの札の型もデッキから）で置き換える。全部見える読み
//! （交点の橋が相手の自席ターンの行から相手の手札を読む形）はここでは使わない。
//!
//! 自分の時計は交点の橋と同じ `seat_row_body`（守る側の計算・耐久 Θ・速さ・段ごとの速さ）を自分の行で読み、相手の時計は
//! 同じ行を `mirror_view`（相手の手札は空＝見えない）で相手席から読む（`BridgeRead::taus` と同じ運び方・手札だけ違う）。
//! 列は `VIS_COLS`（正本はここ 1 か所・Python は `theory_rs.visible_cols` で同じ関数を呼ぶ）。
//!
//! 呼び出し: `tool="visible_cols"`・`in={"decks": [席 0, 席 1] | None, "rows": [行の番号] | None}`。
//! 戻り: `{"cols": [名前…], "rows": [[行の番号, [値…]], …]}`。既定の挙動（他の器）は触らない。

use super::super::leaves_to::{self as lt, DELTA};
use super::super::numeric::py_max;
use super::drv_cb::{seat_row_body, tau_theory_of, Gs, SeatD};
use super::ev::R;
use super::game::{is_own_turn, Game};
use super::obj::V;
use super::outer::{lp_or, r_clip, RACE_CAP};
use super::pd::D;
use super::rows::{self as rw, hr_v, next_turn_don, purse_plan, zones, Cfg};
use super::state::Core;

/// 列の名前（並びは sidecar・ネットの入力の並びと同じ＝変えない）。目盛り: ターン数は ÷10、他はそのまま（耐久・速さは勝率の単位）。
pub const VIS_COLS: [&str; 25] = [
    "tau_me", "tau_opp", "d_clock", "w_cross",
    "theta_me", "th_life_me", "th_hand_me", "th_body_me",
    "theta_opp", "th_life_opp", "th_hand_opp", "th_body_opp",
    "slope_me", "slope_opp",
    "r_hand_opp", "r_hand_me",
    "don_me", "don_opp", "hand_me", "hand_opp", "nu_me", "nu_opp",
    "next_me", "sched1_me", "sched1_opp",
];
// 段 0 の下見で落とした列（既定の読み＝守る側の計算の計画が在るときは構造的に 0 になる）: `stock_*`・`flow_*`・`eff_*`・`lead_*`
// （速さが計画の `a_time` 1 本にまとまる）と `board_*`（`slope_*` と同じ値）と `th_back_*`（計画が在れば 0）。

/// 相手のデッキからの手札の見積もり（`hr_v` の形）。`n_hand` 枚を、デッキのカウンター札の型（カウンター値・費用）の割合に
/// 比例して配る（最大剰余法＝決定的）。札 1 枚の価格は `μ × 切れる札の割合`（`hand_price_mean` の `part_rule=false` の形）。
pub fn deck_hand_read(c: &Core, deck: &[String], n_hand: i64, don: f64, mu: f64) -> V {
    let types = c.life_types_of(deck, Some(1.0)); // (counter, cost, 割合)
    let n = py_max(0.0, n_hand as f64);
    let mut alloc: Vec<(f64, f64, f64, i64)> = types.iter().map(|&(ctr, cost, fr)| {
        let e = n * fr;
        (ctr, cost, e - e.floor(), e.floor() as i64)
    }).collect();
    let target = (types.iter().map(|t| t.2).sum::<f64>() * n).round() as i64;
    let mut have: i64 = alloc.iter().map(|a| a.3).sum();
    let mut order: Vec<usize> = (0..alloc.len()).collect();
    order.sort_by(|&i, &j| alloc[j].2.partial_cmp(&alloc[i].2).unwrap().then(i.cmp(&j)));
    for &i in &order {
        if have >= target {
            break;
        }
        alloc[i].3 += 1;
        have += 1;
    }
    let mut cards = Vec::new();
    for a in &alloc {
        for _ in 0..a.3 {
            cards.push((a.0, a.1));
        }
    }
    let share = super::super::leaves_deck::cut_share(&c.t, deck);
    hr_v(mu * share, &cards, py_max(0.0, don), n_hand, &c.life_types_of(deck, None), &types, &[])
}

/// 席 `s` が手番 `t` の前に済ませた自席ターンの数（＝次の自席ターンの番号）。
fn turns_before(s: i64, t: i64) -> i64 {
    let mut n = 0;
    for tt in 1..t {
        if is_own_turn(s, tt) {
            n += 1;
        }
    }
    n
}


/// 1 行の列（`VIS_COLS` の順）。
/// 区間ごとの秒（`in.profile` のときだけ足す・費用の下見用）
#[derive(Default, Clone, Debug)]
pub struct Prof {
    pub hand: f64,
    pub me: f64,
    pub mirror: f64,
    pub opp: f64,
    pub next: f64,
}

pub fn visible_row(c: &mut Core, g: &Game, cfg: &Cfg, p: &V, k: usize, prof: &mut Prof) -> R<Vec<f64>> {
    let t0 = std::time::Instant::now();
    let r = &g.rows[k];
    let (w, t) = (r.who, r.turn);
    let mu = cfg.mu;
    let row = r.row();
    let sc = &r.sc;
    let tok = &r.tok;
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    let mlp = lp_or(sc[lt::SC_MY_LEADER_POWER]);
    let own = is_own_turn(w, t);
    let j_me = if own { rw::own_turn_index(t) } else { turns_before(w, t) };
    let j_opp = turns_before(1 - w, t);
    let decks = [super::drv_kv::deck_of(p, 0), super::drv_kv::deck_of(p, 1)];
    let dk_me = decks[w as usize].clone();
    let dk_opp = decks[(1 - w) as usize].clone();
    // 自分の手札の読み（全部見える）＝交点の橋の `g_self` と同じ作り
    let mut g_me = c.hand_price_mean(&row, mu, true, Some(sc[lt::SC_MY_DON]))?;
    if let Some(dk) = dk_me.as_deref() {
        g_me = c.with_life_types(&g_me, Some(dk));
    }
    g_me = c.with_hand_blocker(&g_me, &row)?;
    let hb_me = c.hand_blocker_nu(&row, olp)?;
    // 相手の手札の読み＝デッキからの見積もり（デッキが無ければ読み無し＝μ）
    let (oa, orr, oatt, _od) = zones(sc, tok, false);
    let g_opp = match dk_opp.as_deref() {
        Some(dk) if !dk.is_empty() => deck_hand_read(c, dk, sc[lt::SC_OPP_HAND] as i64, oa + orr + oatt, mu),
        _ => V::None,
    };
    prof.hand += t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let gs = Gs {
        g,
        cfg,
        p,
        turn_seq: [vec![t], vec![t]],
        turn_start: vec![((0, t), k), ((1, t), k)],
        turn_last: vec![((0, t), k), ((1, t), k)],
        z_of: Vec::new(),
        g_self: vec![((w, t), g_me.clone()), ((1 - w, t), g_opp.clone())],
        hb_self: vec![((w, t), hb_me), ((1 - w, t), 0.0)],
        has_seat_decks: decks.iter().any(|d| d.is_some()),
        decks,
    };
    let mut st = D::new();
    let me = seat_row_body(c, &gs, &mut st, w, t, j_me, &row, 0.0, 0, true)?;
    prof.me += t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let mrow = c.mirror_view(&row, None, None);
    prof.mirror += t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let op = seat_row_body(c, &gs, &mut st, 1 - w, t, j_opp, &mrow, 0.0, 0, true)?;
    prof.opp += t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    let rd = cfg.rate_decay_ko;
    let tau_me = py_max(0.0, tau_theory_of(&me, rd)).min(RACE_CAP);
    let tau_opp = py_max(0.0, tau_theory_of(&op, rd)).min(RACE_CAP);
    let d = tau_opp - tau_me;
    let wc = lt::prob_of_d(&cfg.clock, d, None, Some(tau_me), Some(tau_opp), "hyp", own);
    // 価格の和
    let g_me_v = rw::gf(&g_me);
    let r_opp_me = me.f("r_opp");
    let r_opp_opp = op.f("r_opp");
    let (ma, mr, matt, _md) = zones(sc, tok, true);
    let _ = (ma, mr, matt);
    let don_me = DELTA * next_turn_don(sc, tok);
    let don_opp = DELTA * (oa + orr + oatt + (_od).min(2.0));
    let nu_me = lt::side_nu_meas(tok, lt::OWN_FIELD, olp);
    let nu_opp = lt::side_nu_meas(tok, lt::OPP_FIELD, mlp);
    // 次の自分の手番に出せる体の価格（手札の組を次の手番のドンで）
    let next_me = {
        let rt = r_clip(sc[lt::SC_OPP_LIFE]);
        let theta = rw::theta_of(tok, sc[lt::SC_MY_LIFE], sc[lt::SC_MY_DON], "const", cfg.theta);
        let items = c.hand_items(tok, &r.ci, olp, rt)?;
        let blk = c.opp_blockers_of(tok, Some(mlp), lt::R_TURNS, theta, mu);
        let mut grp = c.hand_groups_p(&items, olp, theta, mu, mlp, rt, false);
        grp.extend(c.attach_groups(tok, olp, theta, mu, &blk));
        let pl = purse_plan(&grp, next_turn_don(sc, tok));
        pl.atk + pl.attach + pl.attach_lead + pl.eff
    };
    prof.next += t0.elapsed().as_secs_f64();
    let sched1 = |d: &SeatD| -> f64 {
        let sv = d.get("sched");
        if sv.truthy() && !sv.items().is_empty() { sv.items()[0].f() } else { 0.0 }
    };
    Ok(vec![
        tau_me / 10.0, tau_opp / 10.0, d / 10.0, wc,
        me.f("theta"), me.f("th_life"), me.f("th_hand"), me.f("th_body"),
        op.f("theta"), op.f("th_life"), op.f("th_hand"), op.f("th_body"),
        me.f("slope_theory"), op.f("slope_theory"),
        r_opp_me, r_opp_opp,
        don_me, don_opp, g_me_v * sc[lt::SC_MY_HAND], r_opp_me * sc[lt::SC_OPP_HAND], nu_me, nu_opp,
        next_me, sched1(&me), sched1(&op),
    ])
}

pub fn game(c: &mut Core, g: &Game, cfg: &Cfg, p: &V) -> R<V> {
    let inp = p.get("in");
    let rows: Vec<usize> = if inp.get("rows").is_none() { (0..g.n()).collect() } else { inp.get("rows").items().iter().map(|x| x.int() as usize).collect() };
    let mut out = Vec::with_capacity(rows.len());
    let mut prof = Prof::default();
    for k in rows {
        let v = visible_row(c, g, cfg, p, k, &mut prof)?;
        out.push(V::list(vec![V::Int(k as i64), V::list(v.into_iter().map(V::Float).collect())]));
    }
    let mut kv = vec![
        (V::s("cols"), V::list(VIS_COLS.iter().map(|s| V::s(s)).collect())),
        (V::s("rows"), V::list(out)),
    ];
    if inp.get("profile").truthy() {
        kv.push((V::s("prof"), V::dict(vec![
            (V::s("hand"), V::Float(prof.hand)), (V::s("me"), V::Float(prof.me)), (V::s("mirror"), V::Float(prof.mirror)),
            (V::s("opp"), V::Float(prof.opp)), (V::s("next"), V::Float(prof.next)),
        ])));
    }
    Ok(V::dict(kv))
}
