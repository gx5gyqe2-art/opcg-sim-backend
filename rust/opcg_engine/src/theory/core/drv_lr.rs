//! 段 6: `lethal_rule._iter_declared_games` の 1 局ぶん（`settled_map` が読む決着の旗）。行の読み: `lethal_of_row`
//! （`attach_don`・`life_cards_as_counters`・`max_stops`）・`avg_counter`（`AVG_COUNTER_MODE=rules`・席ごとに覚える）。
//! 戻り＝`{"rows": [[w, t, declared], …]}`（手札が読めない行は `False`）。`avg_cache` は局をまたいで `carry` で持つ
//! （鍵は `(seed, 席)`＝値は札の並びだけで決まる）。

use super::super::leaves_deck as ld;
use super::super::leaves_to::{self as lt, Tok};
use super::drv_kv::deck_of;
use super::ev::R;
use super::game::{is_own_turn, Game};
use super::obj::V;
use super::outer::{lp_or, r_clip};
use super::rows::Cfg;
use super::state::Core;

/// `lethal_of_row(sc, tok, with_don, defender_counters, life_counter, defender_costs)` → 決着か
pub fn lethal_of_row(sc: &[f64], tok: &Tok, with_don: bool, counters: &[f64], life_counter: f64, costs: &[f64]) -> bool {
    let olp = lp_or(sc[lt::SC_OPP_LEADER_POWER]);
    let life = sc[lt::SC_OPP_LIFE];
    let mut xs = lt::own_attackers_of(tok, olp);
    if with_don {
        xs = ld::attach_don(&xs, sc[lt::SC_MY_DON]);
    }
    let n_block = lt::active_blockers(tok, lt::OPP_FIELD) as usize;
    let mut xs_sorted = xs.clone();
    xs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let xs_through0: Vec<f64> = if n_block > 0 { xs_sorted.iter().skip(n_block).cloned().collect() } else { xs_sorted };
    let xs_through: Vec<f64> = xs_through0.into_iter().filter(|&x| x >= -lt::PWR_EPS).collect();
    let through = xs_through.len() as i64;
    let mut pool: Vec<f64> = counters.to_vec();
    pool.extend(ld::life_cards_as_counters(life, life_counter));
    let mut cs: Vec<f64> = costs.to_vec();
    while cs.len() < pool.len() {
        cs.push(0.0);
    }
    let stops = ld::max_stops(&pool, &xs_through, Some(&cs), sc[4]);
    let hits = (through - stops).max(0);
    life >= 0.0 && hits as f64 >= life + 1.0
}

pub fn game(c: &mut Core, g: &Game, _cfg: &Cfg, p: &V) -> R<V> {
    let with_don = p.get("cfg").get("with_don").truthy();
    let seed = g.rows.first().map(|r| r.seed).unwrap_or(-1);
    let mut avg: Vec<(i64, f64)> = p.get("carry").get("avg").items().iter().map(|e| (e.items()[0].int(), e.items()[1].f())).collect();
    let mut turn_start: Vec<((i64, i64), usize)> = Vec::new();
    let mut turn_last: Vec<((i64, i64), usize)> = Vec::new();
    let mut turn_seq: [Vec<i64>; 2] = [Vec::new(), Vec::new()];
    for (n, r) in g.rows.iter().enumerate() {
        let (w, t) = (r.who, r.turn);
        if r.kind != 0 || !is_own_turn(w, t) {
            continue;
        }
        match turn_last.iter_mut().find(|(k, _)| *k == (w, t)) {
            Some(e) => e.1 = n,
            None => turn_last.push(((w, t), n)),
        }
        if !turn_start.iter().any(|(k, _)| *k == (w, t)) {
            turn_start.push(((w, t), n));
            turn_seq[w as usize].push(t);
        }
    }
    let get = |v: &[((i64, i64), usize)], k: (i64, i64)| v.iter().find(|(a, _)| *a == k).map(|e| e.1);
    let mut out = Vec::new();
    for w in 0..2i64 {
        for &t in &turn_seq[w as usize] {
            let r = &g.rows[get(&turn_start, (w, t)).unwrap()];
            let d = 1 - w;
            let prev: Vec<i64> = turn_seq[d as usize].iter().cloned().filter(|&tt| tt < t).collect();
            let Some(&pl) = prev.last() else {
                out.push(V::list(vec![V::Int(w), V::Int(t), V::Bool(false)]));
                continue;
            };
            let rd = &g.rows[get(&turn_last, (d, pl)).or_else(|| get(&turn_start, (d, pl))).unwrap()];
            let items = c.hand_items(&rd.tok, &rd.ci, lp_or(rd.sc[lt::SC_OPP_LEADER_POWER]), r_clip(rd.sc[lt::SC_OPP_LIFE]))?;
            let counters: Vec<f64> = items.iter().map(|it| it.get("counter").f()).collect();
            let costs: Vec<f64> = items.iter().map(|it| if it.get("event").truthy() { it.get("cost").f() } else { 0.0 }).collect();
            let lc = match avg.iter().find(|(k, _)| *k == d) {
                Some(e) => e.1,
                None => {
                    let dk = deck_of(p, d).ok_or_else(|| format!("seed {seed} 席 {d} のデッキが引けない"))?;
                    let v = ld::avg_counter(&c.t, &dk, true);
                    avg.push((d, v));
                    v
                }
            };
            let declared = lethal_of_row(&r.sc, &r.tok, with_don, &counters, lc, &costs);
            out.push(V::list(vec![V::Int(w), V::Int(t), V::Bool(declared)]));
        }
    }
    Ok(V::dict(vec![
        (V::s("rows"), V::list(out)),
        (V::s("carry"), V::dict(vec![(V::s("avg"), V::list(avg.iter().map(|(k, v)| V::list(vec![V::Int(*k), V::Float(*v)])).collect()))])),
    ]))
}

/// 段 7（2026-10-07）: 決着の定義（T136）の手計算の試験（旧 `tests/test_lethal_rule.py` から写した・規則から導ける値だけ）。
/// **決着 ⇔ 相手が最善で守っても通る本数 ≥ 相手の残りライフ ＋ 1**（「＋1」は規則: ライフ 0 で損害を受けたら負け）。
#[cfg(test)]
mod tests {
    use super::super::super::leaves_deck as ld;
    use super::super::super::leaves_to::{self as lt, Tok};
    use super::lethal_of_row;

    type Body = (f64, bool, bool, bool);

    /// `(sc, tok)`。`own`／`opp` は `(パワー, ブロッカー, レスト, 攻撃できる)` の列。
    fn row(opp_life: f64, don: f64, leader_power: f64, olp: f64, own: &[Body], opp: &[Body]) -> (Vec<f64>, Tok) {
        let mut sc = vec![0.0; 64];
        sc[lt::SC_OPP_LIFE] = opp_life;
        sc[lt::SC_MY_DON] = don;
        sc[lt::SC_OPP_LEADER_POWER] = olp / 1e4;
        let mut t = vec![0.0; 24 * 24];
        t[lt::S_POWER] = leader_power / 1e4;
        let mut put_side = |slots: std::ops::Range<usize>, bodies: &[Body]| {
            for (s, &(p, b, rest, a)) in slots.zip(bodies.iter()) {
                t[s * 24 + lt::S_IS_CHAR] = 1.0;
                t[s * 24 + lt::S_POWER] = p / 1e4;
                t[s * 24 + lt::S_IS_BLOCKER] = b as i64 as f64;
                t[s * 24 + lt::S_IS_REST] = rest as i64 as f64;
                t[s * 24 + lt::S_CAN_ATTACK] = a as i64 as f64;
            }
        };
        put_side(lt::OWN_FIELD, own);
        put_side(lt::OPP_FIELD, opp);
        (sc, Tok::new(t, 24, 24))
    }

    fn lethal(sc: &[f64], tok: &Tok, with_don: bool, counters: &[f64], costs: &[f64]) -> bool {
        lethal_of_row(sc, tok, with_don, counters, 0.0, costs)
    }

    const BODY: Body = (5000.0, false, false, true);

    #[test]
    fn the_plus_one_is_exact_at_every_life() {
        for l in 0..5usize {
            let (sc, tok) = row(l as f64, 0.0, 5000.0, 5000.0, &vec![BODY; l], &[]); // L+1 本（リーダー込み）
            assert!(lethal(&sc, &tok, false, &[], &[]), "L={l}");
            if l >= 1 {
                let (sc, tok) = row(l as f64, 0.0, 5000.0, 5000.0, &vec![BODY; l - 1], &[]); // L 本では足りない
                assert!(!lethal(&sc, &tok, false, &[], &[]), "L={l}");
            }
        }
    }

    #[test]
    fn life_zero_is_alive_and_a_negative_life_is_never_declared() {
        let (mut sc, tok) = row(0.0, 0.0, 5000.0, 5000.0, &[], &[]);
        assert!(lethal(&sc, &tok, false, &[], &[]));
        sc[lt::SC_OPP_LIFE] = -1.0;
        assert!(!lethal(&sc, &tok, false, &[], &[]));
    }

    #[test]
    fn one_counter_card_stops_the_leader() {
        let (sc, tok) = row(0.0, 0.0, 5000.0, 5000.0, &[], &[]);
        assert!(!lethal(&sc, &tok, false, &[1000.0], &[]));
    }

    #[test]
    fn an_active_blocker_takes_the_cheapest_attack_and_a_rested_one_does_not() {
        // リーダー 5000 ＋ 体 7000・ライフ 0・アクティブなブロッカー 1: リーダーが横取りされても 7000 が通る
        let atk = [(7000.0, false, false, true)];
        let (sc, tok) = row(0.0, 0.0, 5000.0, 5000.0, &atk, &[(3000.0, true, false, false)]);
        assert!(lethal(&sc, &tok, false, &[], &[]));
        // ライフ 1（2 本要る）: アクティブなブロッカーが 1 本取る → 足りない／レストのブロッカーは取れない → 2 本通る
        let (sc, tok) = row(1.0, 0.0, 5000.0, 5000.0, &atk, &[(3000.0, true, false, false)]);
        assert!(!lethal(&sc, &tok, false, &[], &[]));
        let (sc, tok) = row(1.0, 0.0, 5000.0, 5000.0, &atk, &[(3000.0, true, true, false)]);
        assert!(lethal(&sc, &tok, false, &[], &[]));
    }

    #[test]
    fn attacks_below_the_defending_leader_do_not_reach() {
        let (sc, tok) = row(0.0, 0.0, 3000.0, 5000.0, &[], &[]);
        assert!(!lethal(&sc, &tok, false, &[], &[]));
    }

    #[test]
    fn don_goes_to_the_cheapest_attack_four_per_body_and_raises_the_counter_needed() {
        assert_eq!(ld::attach_don(&[0.0, 2000.0], 5.0), vec![4000.0, 3000.0]);
        assert_eq!(ld::attach_don(&[0.0], 10.0), vec![4000.0]); // 1 体 4 枚まで・余りは捨てる
        // 超過 0 は 1000 で止まる。ドン 1 枚で超過 1000 → 2000 要る → 1000 の札では止まらない
        let (sc, tok) = row(0.0, 1.0, 5000.0, 5000.0, &[], &[]);
        assert!(!lethal(&sc, &tok, false, &[1000.0], &[]));
        assert!(lethal(&sc, &tok, true, &[1000.0], &[]));
    }

    #[test]
    fn max_stops_counts_the_most_attacks_the_hand_can_stop() {
        let inf = f64::INFINITY;
        // 超過 x を止めるには x + 1000 要る（同値は命中）: {0, 0, 3000} に {1000, 1000, 2000} → 2 本
        assert_eq!(ld::max_stops(&[1000.0, 1000.0, 2000.0], &[0.0, 0.0, 3000.0], None, inf), 2);
        assert_eq!(ld::max_stops(&[1000.0, 1000.0, 2000.0], &[0.0, 2000.0], None, inf), 2);
        assert_eq!(ld::max_stops(&[1000.0], &[-2000.0, -1000.0], None, inf), 0); // 通らない攻撃は数えない
        assert_eq!(ld::max_stops(&[0.0, 0.0, 0.0], &[0.0], None, inf), 0);
    }

    #[test]
    fn an_event_counter_needs_the_defenders_active_don() {
        // 札: キャラ 1000・イベント 4000（費用 1）。超過 2000 には 3000 要る＝イベントが要る
        assert_eq!(ld::max_stops(&[1000.0, 4000.0], &[2000.0], Some(&[0.0, 1.0]), 1.0), 1);
        assert_eq!(ld::max_stops(&[1000.0, 4000.0], &[2000.0], Some(&[0.0, 1.0]), 0.0), 0);
        assert_eq!(ld::max_stops(&[4000.0, 4000.0], &[2000.0, 2000.0], Some(&[1.0, 1.0]), 1.0), 1);
        assert_eq!(ld::max_stops(&[4000.0, 4000.0], &[2000.0, 2000.0], Some(&[1.0, 1.0]), 2.0), 2);
        // 行の予算は相手のアクティブなドン（`sc[4]`）
        let (mut sc, tok) = row(0.0, 0.0, 5000.0, 5000.0, &[], &[]);
        sc[4] = 0.0;
        assert!(lethal(&sc, &tok, false, &[4000.0], &[1.0]));
        sc[4] = 1.0;
        assert!(!lethal(&sc, &tok, false, &[4000.0], &[1.0]));
    }

    #[test]
    fn every_life_card_taken_becomes_a_counter_card() {
        assert_eq!(ld::life_cards_as_counters(4.0, 1000.0), vec![1000.0; 4]);
        assert!(ld::life_cards_as_counters(0.0, 1000.0).is_empty());
        assert!(ld::life_cards_as_counters(3.0, 0.0).is_empty());
        // リーダー ＋ 体 4・ライフ 4（5 本要る）・手札なし: ライフの札 4 枚（各 1000）で 4 本止まる → 決着しない
        let (sc, tok) = row(4.0, 0.0, 5000.0, 5000.0, &[BODY; 4], &[]);
        assert!(!lethal_of_row(&sc, &tok, false, &[], 1000.0, &[]));
        assert!(lethal_of_row(&sc, &tok, false, &[], 0.0, &[]));
    }
}
