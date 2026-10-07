//! 移植の段 6（2026-10-07）: **局の駆動の入口**——Python が記録を 1 局ずつ読んで枠（`TheoryFrame`）を渡し、Rust が局の中の
//! 計算（各器の `collect` のループの本体）を全部して、行の表と書き換えた `stats` を返す。集計・ブートストラップ・AUC・較正・
//! JSON の書き出しは Python に残る（ユーザ決定 2026-10-06・判断 7）。
//!
//! 1 回の呼び出し＝`{"tool": 器, "g": 段 3 の文脈と切替, "cfg": 局の駆動の設定, "in": 局ごとの入力（デッキ・決着の旗…）,
//! "stats": 呼ぶ前の `stats`, "carry": 局をまたぐ値}`。戻り＝`{"stats", "out": 器ごとの行の表, "carry", "ev": RULE_STATS の増分,
//! "cs": 条件の計数の差分}`。覚え書きは `Core` の中で局をまたいで生きる（Python と同じ順で育つ）。

use super::super::leaves_to as lt;
use super::ev::R;
use super::game::Game;
use super::obj::V;
use super::rows::Cfg;
use super::state::Core;

pub fn cfg_of(v: &V) -> R<Cfg> {
    let cl = v.get("clock");
    let of = |x: &V| if x.is_none() { None } else { Some(x.f()) };
    let prof = v.get("prof");
    Ok(Cfg {
        theta: v.get("theta").f(),
        mu: v.get("mu").f(),
        theta_mode: v.get("theta_mode").str_or_empty(),
        clock: lt::ClockCfg {
            w_err_rel: cl.get("W_ERR_MODE").as_str() == Some("rel"),
            sigma_rel: of(cl.get("SIGMA_REL")),
            sigma_d: cl.get("SIGMA_D").f(),
            w_slope: matches!(cl.get("W_MODE").as_str(), Some("clock") | Some("curve")),
            w_bar: cl.get("W_BAR").f(),
        },
        sigma_rel: of(cl.get("SIGMA_REL")),
        side_symmetric: v.get("THETA_SIDE_MODE").as_str() == Some("symmetric"),
        rate_decay_ko: v.get("RATE_DECAY_MODE").as_str() == Some("ko"),
        prof: if prof.is_none() { None } else { Some(prof.items().iter().map(|x| x.f()).collect()) },
    })
}

/// 局の駆動を 1 局ぶん（`c.ctx` は呼ぶ前に Python の大域の値＝既定の文脈で入れる）
pub fn run(c: &mut Core, g: &Game, p: &V) -> R<V> {
    super::entry::apply_g(c, p.get("g"))?;
    c.outer.events.clear();
    let cs0 = c.cond_stats;
    let tool = p.get("tool").pystr();
    let cfg = cfg_of(p.get("cfg"))?;
    let r = match tool.as_str() {
        "kappa_vector" => super::drv_kv::game(c, g, &cfg, p),
        "transition_ledger" => super::drv_tl::game(c, g, &cfg, p),
        "relative_ledger" => super::drv_rl::game(c, g, &cfg, p),
        "price_realised" => super::drv_pr::game(c, g, &cfg, p),
        "crossing_bridge" => super::drv_cb::game(c, g, &cfg, p),
        "lethal_rule" => super::drv_lr::game(c, g, &cfg, p),
        "theory_bridge" => super::drv_tb::game(c, g, &cfg, p),
        "two_curves" | "two_curves_state" | "price_cands" | "live_turn" => super::drv_t18::game(c, g, &cfg, p, &tool),
        o => Err(format!("局の駆動が無い器: {o}")),
    };
    c.ctx = super::state::Ctx::default();
    let mut kv = r?.kv().to_vec();
    let ev = std::mem::take(&mut c.outer.events);
    if !ev.is_empty() {
        kv.push((V::s("ev"), V::list(ev.into_iter().map(|(k, v)| V::list(vec![V::s(&k), v])).collect())));
    }
    let cs = [c.cond_stats[0] - cs0[0], c.cond_stats[1] - cs0[1], c.cond_stats[2] - cs0[2]];
    kv.push((V::s("cs"), V::list(cs.iter().map(|&x| V::Int(x)).collect())));
    Ok(V::dict(kv))
}

/// 行ごとの関数（`win_calib.probs_of`・`pre_settle_asymmetry.rows_with_p` の `p`／`s`）——局をまたがない
pub fn rows_call(c: &mut Core, p: &V) -> R<V> {
    let _ = c;
    let cfg = cfg_of(p.get("cfg"))?;
    match p.get("fn").pystr().as_str() {
        "wc.probs_of" => {
            // `[TO.prob_of_d(d, t_me=tm, t_opp=to, scale_mode=scale_mode, mover=True) for (d, tm, to, _z) in rs]`
            let sm = p.get("scale_mode").pystr();
            let out: Vec<V> = p
                .get("rs")
                .items()
                .iter()
                .map(|r| {
                    let it = r.items();
                    V::Float(lt::prob_of_d(&cfg.clock, it[0].f(), None, Some(it[1].f()), Some(it[2].f()), &sm, true))
                })
                .collect();
            Ok(V::list(out))
        }
        "to.clock_scale" => {
            // 段 7: `mode`（省略時 `hyp`）
            let m = if p.get("mode").is_none() { "hyp".to_string() } else { p.get("mode").pystr() };
            let out: Vec<V> = p.get("rs").items().iter().map(|r| V::Float(lt::clock_scale(r.items()[0].f(), r.items()[1].f(), &m))).collect();
            Ok(V::list(out))
        }
        // 段 7（2026-10-07）: 器の集計が読む理論の式（Python の写しを消したので Rust で）
        "to.prob_of_d" => {
            // `[TO.prob_of_d(d, sigma_d=sigma_d, t_me=tm, t_opp=to, scale_mode=scale_mode, mover=mover) for (d, tm, to) in rs]`
            let sm = if p.get("scale_mode").is_none() { "hyp".to_string() } else { p.get("scale_mode").pystr() };
            let mover = p.get("mover").truthy();
            let sd = if p.get("sigma_d").is_none() { None } else { Some(p.get("sigma_d").f()) };
            let of = |x: &V| if x.is_none() { None } else { Some(x.f()) };
            let out: Vec<V> = p
                .get("rs")
                .items()
                .iter()
                .map(|r| {
                    let it = r.items();
                    V::Float(lt::prob_of_d(&cfg.clock, it[0].f(), sd, of(&it[1]), of(&it[2]), &sm, mover))
                })
                .collect();
            Ok(V::list(out))
        }
        "to.whole_clock_scale" => {
            let out: Vec<V> = p.get("rs").items().iter().map(|x| V::Float(lt::whole_clock_scale(x.f()))).collect();
            Ok(V::list(out))
        }
        "cb.tau_from_profile" => {
            // `[CB.tau_from_profile(theta, j, prof, scale, r, shield, shield_rate, refill, step) for … in rs]`
            let prof: Vec<f64> = p.get("prof").items().iter().map(|x| x.f()).collect();
            let out: Vec<V> = p
                .get("rs")
                .items()
                .iter()
                .map(|r| {
                    let a = r.items();
                    V::Float(super::rows::tau_from_profile(a[0].f(), a[1].int(), &prof, a[2].f(), a[3].f(), a[4].f(), a[5].f(), a[6].f(), a[7].f()))
                })
                .collect();
            Ok(V::list(out))
        }
        o => Err(format!("行の関数が無い: {o}")),
    }
}
