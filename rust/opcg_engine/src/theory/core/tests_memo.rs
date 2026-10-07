//! 覚え書きの正確さの見張り（2026-10-07・`docs/reports/2026-10-07_memo_exact.md`）: **当たりの値＝解き直しの値**。
//!
//! 直した覚え書きごとに、旧い鍵（丸め・文脈の抜け）では**同じ鍵になる 2 つの入力**を温めた核で続けて解き、後の値が新しい核で
//! 解いた値とビットで同じことを確かめる（旧い鍵では最初の値が返って落ちる）。入力の違いが値を変えることも確かめる（見張りが空振りしない）。
//! 手札の読みの `JointValuer`（E52）と守る側の外側の `_RULE_DON_CACHE`（E61）は記録の golden の入力で、温めた物と新しい物を比べる。

use std::collections::HashMap;

use super::super::leaves_to::MU;
use super::super::pyval::gunzip;
use super::obj::{dset, from_pyval, V};
use super::outer::Actx;
use super::state::Core;
use super::to::{theta, ATTACK_DON_MAX, DELTA};

fn cores() -> (Core, Core) {
    super::tests_core::load_tables();
    (Core::new(), Core::new())
}

fn golden(name: &str) -> String {
    let p = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    String::from_utf8(gunzip(&std::fs::read(&p).unwrap())).unwrap()
}

/// golden の `fn` の最初の行の (a, g)
fn first_line(text: &str, f: &str) -> (V, V) {
    let line = text.lines().find(|l| l.contains(&format!("\"fn\":\"{f}\""))).unwrap_or_else(|| panic!("{f} の行が無い"));
    let (_n, p, _w, _cs) = super::tests_core::parse_line(line);
    (p.get("a").clone(), p.get("g").clone())
}

fn bits(x: f64) -> u64 {
    x.to_bits()
}

/// `_OPTION_CACHE`（旧: パワーを 100 単位・`R` を帯に丸めた鍵）
#[test]
fn option_value_hit_equals_fresh() {
    let (mut w, mut f) = cores();
    let (th, mu) = (theta(), MU);
    let first = w.option_value(5000.0, 5000.0, 3.0, th, mu, None, 0.289);
    let hit = w.option_value(5040.0, 5000.0, 3.2, th, mu, None, 0.289);
    let fresh = f.option_value(5040.0, 5000.0, 3.2, th, mu, None, 0.289);
    assert_eq!(bits(hit), bits(fresh));
    assert_ne!(bits(first), bits(fresh), "入力の違いが値を変えない（見張りが空振り）");
}

/// `_AVD_MEMO`（旧: 窓の `ḡ` を 12 桁に丸めた `CUT_PRICER_KEY`・`CUT_TAKE_CARD` はビットで鍵に入っていた＝窓が `ḡ` を両方に入れる
/// 局の駆動では実害が無かった・ここでは `CUT_TAKE_CARD` を共通にして `ḡ` だけ 12 桁より下で違える）
#[test]
fn attack_value_don_hit_equals_fresh_across_windows() {
    let (mut w, mut f) = cores();
    let (th, mu) = (theta(), MU);
    let (g1, g2) = (0.05, 0.05 * (1.0 + 4e-15));
    let set = |c: &mut Core, g: f64| {
        c.ctx.pricer = Some(g);
        c.ctx.pricer_key = V::tuple(vec![V::s("avg"), V::Float(super::super::numeric::py_round(g, 12))]);
        c.ctx.take_card = Some(0.05);
    };
    set(&mut w, g1);
    // キャラ狙い（体の損 1.0 は守る費用 c·ḡ より大きい＝値は ḡ に比例）
    let first = w.attack_value_don(6000.0, 5000.0, false, th, mu, Some(1.0), DELTA, ATTACK_DON_MAX, &[]);
    set(&mut w, g2);
    let hit = w.attack_value_don(6000.0, 5000.0, false, th, mu, Some(1.0), DELTA, ATTACK_DON_MAX, &[]);
    set(&mut f, g2);
    let fresh = f.attack_value_don(6000.0, 5000.0, false, th, mu, Some(1.0), DELTA, ATTACK_DON_MAX, &[]);
    assert_eq!(bits(hit), bits(fresh));
    assert_ne!(bits(first), bits(fresh), "ḡ の違いが値を変えない（見張りが空振り）");
}

/// `deck_refill._FLOW`（E39・旧: `round(olp, 1)`・θ／μ なし・窓の `ḡ` は 12 桁に丸めた鍵）
#[test]
fn a_of_hit_equals_fresh() {
    let (mut w, mut f) = cores();
    let text = golden("theory_core_golden.jsonl.gz");
    let (a, _g) = first_line(&text, "dr.a_of");
    let deck: Vec<String> = a.get("deck_ids").items().iter().map(|x| x.pystr()).collect();
    let mv = V::Float(MU);
    let (tv1, tv2) = (V::Float(theta()), V::Float(theta() * 1.5));
    // θ（旧い鍵に入らない）
    let first = w.a_of(&deck, 5000.0, Some(4.0), &tv1, &mv, false, true).unwrap();
    let hit = w.a_of(&deck, 5000.04, Some(4.0), &tv2, &mv, false, true).unwrap();
    let fresh = f.a_of(&deck, 5000.04, Some(4.0), &tv2, &mv, false, true).unwrap();
    assert_eq!(bits(hit), bits(fresh));
    assert_ne!(bits(first), bits(fresh), "θ の違いが値を変えない（見張りが空振り）");
    // 窓の ḡ（12 桁より下だけ違う 2 つの窓）
    let (g1, g2) = (0.05, 0.05 * (1.0 + 4e-15));
    let s = w.enter(Some(g1));
    let first = w.a_of(&deck, 5000.0, Some(4.0), &tv1, &mv, false, true).unwrap();
    w.leave(s);
    let s = w.enter(Some(g2));
    let hit = w.a_of(&deck, 5000.0, Some(4.0), &tv1, &mv, false, true).unwrap();
    w.leave(s);
    let s = f.enter(Some(g2));
    let fresh = f.a_of(&deck, 5000.0, Some(4.0), &tv1, &mv, false, true).unwrap();
    f.leave(s);
    assert_eq!(bits(hit), bits(fresh));
    assert_ne!(bits(first), bits(fresh), "ḡ の違いが値を変えない（見張りが空振り）");
}

/// `search_price._GAIN`（旧: 丸めた手札の読みの鍵 `_ctx_key`）
#[test]
fn card_gain_hit_equals_fresh() {
    let (mut w, _f) = cores();
    let text = golden("theory_core_golden.jsonl.gz");
    // 手札の札の値を 6 桁より下だけ・コストとカウンターを動かす（旧い鍵は札と `round(v, 6)` だけ＝コストもカウンターも持たない）
    let bump = |ctx: &V| {
        let items: Vec<V> = ctx
            .get("hand_items")
            .items()
            .iter()
            .map(|it| {
                let it = dset(&dset(it, "counter", V::Float(it.get("counter").f() + 2000.0)), "cost", V::Float(it.get("cost").f() + 3.0));
                match it.get("v") {
                    V::Float(x) => dset(&it, "v", V::Float(x + 3e-8)),
                    _ => it,
                }
            })
            .collect();
        dset(ctx, "hand_items", V::list(items))
    };
    let (mut n, mut moved) = (0, 0);
    for line in text.lines().filter(|l| l.contains("\"fn\":\"sp.card_gain\"")).take(40) {
        let (_n, p, _w, _cs) = super::tests_core::parse_line(line);
        let (a, g) = (p.get("a"), p.get("g"));
        let mut f = Core::new();
        super::entry::apply_g(&mut w, g).unwrap();
        super::entry::apply_g(&mut f, g).unwrap();
        let cid = a.get("cid").pystr();
        let ctx = a.get("ctx").clone();
        let ctx2 = bump(&ctx);
        let first = w.card_gain(&cid, &ctx).unwrap();
        let hit = w.card_gain(&cid, &ctx2).unwrap();
        let fresh = f.card_gain(&cid, &ctx2).unwrap();
        assert_eq!(bits(hit), bits(fresh), "{cid}");
        moved += (bits(first) != bits(fresh)) as usize;
        n += 1;
    }
    assert!(n >= 20 && moved > 0, "手札の違いが値を変えない（見張りが空振り）: {moved} / {n}");
}

/// 選択の分布 `_CACHE["sel"]`（旧: 最初に解いた文脈の分布を全部で使う）
#[test]
fn selection_dist_hit_equals_fresh_across_windows() {
    let (mut w, mut f) = cores();
    // 旧は最初の文脈の分布を全部で使った（`theory_bridge` は帳簿の値を `FLOW_PRICING=exercise` で読み直す）
    let first = w.selection_dist().unwrap();
    w.ctx.flow_exercise = true;
    let hit = w.selection_dist().unwrap();
    f.ctx.flow_exercise = true;
    let fresh = f.selection_dist().unwrap();
    let same = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same(&hit, &fresh));
    assert!(!same(&first, &fresh), "窓の違いが分布を変えない（見張りが空振り）");
}

/// 攻め手の財布の `_gain`（E60・旧: `(round(x, 3), k)`）
#[test]
fn attach_gain_hit_equals_fresh() {
    let (mut w, mut f) = cores();
    let text = golden("theory_outer_golden.jsonl.gz");
    let line = text.lines().find(|l| l.contains("\"fn\":\"cb.attacker_ctx\"") && !l.contains("\"r\":null")).unwrap();
    let l = super::tests_outer::parse_line(line);
    let d = from_pyval(&l.want);
    let mut ax_w = Actx::from_v(&d).unwrap();
    let mut ax_f = Actx::from_v(&d).unwrap();
    // 旧い鍵 `(round(x, 3), k)` は窓の値段の文脈を持たない: 窓 ḡ=0.05 で解いた値を ḡ=0.08 の窓でも返した
    let x = ax_w.att1.first().map(|p| p.1).unwrap_or(0.0);
    let s = w.enter(Some(0.05));
    let first = w.attach_gain(&mut ax_w, x, 1);
    w.leave(s);
    let s = w.enter(Some(0.08));
    let hit = w.attach_gain(&mut ax_w, x + 0.0002, 1);
    w.leave(s);
    let s = f.enter(Some(0.08));
    let fresh = f.attach_gain(&mut ax_f, x + 0.0002, 1);
    f.leave(s);
    assert_eq!(bits(hit), bits(fresh));
    assert_ne!(bits(first), bits(fresh), "x の違いが値を変えない（見張りが空振り）");
}

/// 手札の読みの `JointValuer`（E52・旧: 鍵は残った札だけ）: golden の `jv.value` を記録の順に温めた物で解き、新しい物と比べる
#[test]
fn joint_valuer_hits_equal_fresh_valuer() {
    let (mut c, _f) = cores();
    let text = golden("theory_core_golden.jsonl.gz");
    let mut hands: HashMap<String, V> = HashMap::new();
    let mut warm: HashMap<String, super::hj::JointValuer> = HashMap::new();
    let mut n = 0;
    for line in text.lines() {
        let (name, p, want, _cs) = super::tests_core::parse_line(line);
        if name == "tb.joint_valuer" {
            if let super::super::pyval::PyVal::Obj(id) = want {
                hands.insert(id, p.get("a").get("hand").clone());
            }
            continue;
        }
        if name != "jv.value" {
            continue;
        }
        let a = p.get("a");
        let id = match a.get("jv") {
            V::Obj(s) => s.to_string(),
            _ => continue,
        };
        let hand = hands[&id].clone();
        super::entry::apply_g(&mut c, p.get("g")).unwrap();
        let mut keep = 0u64;
        for i in a.get("keep").items() {
            keep |= 1 << i.int();
        }
        let jv = warm.entry(id).or_insert_with(|| c.joint_valuer_of(&hand));
        let hit = jv.value(&mut c, keep).unwrap();
        let mut fresh_jv = c.joint_valuer_of(&hand);
        let fresh = fresh_jv.value(&mut c, keep).unwrap();
        assert_eq!(bits(hit.0), bits(fresh.0), "{line}");
        assert_eq!(hit.1, fresh.1);
        c.ctx = Default::default();
        n += 1;
    }
    assert!(n > 100, "jv.value の行が少ない: {n}");
    // 読み直す物を、値段の窓の外で解いてから窓 ḡ の中で解く（旧い鍵は残った札だけ＝窓の外の値を返した）
    let mut moved = 0;
    for hand in hands.values() {
        let mut jv = c.joint_valuer_of(hand);
        if !jv.rereads() || jv.n == 0 {
            continue;
        }
        let full = (1u64 << jv.n) - 1;
        let first = jv.value(&mut c, full).unwrap().0;
        let s = c.enter(Some(0.02));
        let hit = jv.value(&mut c, full).unwrap().0;
        let fresh = c.joint_valuer_of(hand).value(&mut c, full).unwrap().0;
        c.leave(s);
        assert_eq!(bits(hit), bits(fresh));
        moved += (bits(first) != bits(fresh)) as usize;
    }
    assert!(moved > 0, "窓の違いが読み直す物の値を変えない（見張りが空振り）");
}

/// `_RULE_DON_CACHE`（E61）と計画: golden の守る側の計算を温めた核（覚え書きが育つ）と新しい核で解いて比べる
#[test]
fn rule_don_solve_hits_equal_fresh() {
    let (mut w, _f) = cores();
    let text = golden("theory_outer_golden.jsonl.gz");
    let mut n = 0;
    for line in text.lines().filter(|l| l.contains("\"fn\":\"cb.rule_don_solve\"") || l.contains("\"fn\":\"cb.rule_don_plan_for\"")) {
        let l = super::tests_outer::parse_line(line);
        let warm = super::entry::call_ev(&mut w, &l.name, &l.payload, false).map(|x| x.0);
        let mut f = Core::new();
        let fresh = super::entry::call_ev(&mut f, &l.name, &l.payload, false).map(|x| x.0);
        let (a, b) = (super::obj::to_pyval(&warm.unwrap()), super::obj::to_pyval(&fresh.unwrap()));
        assert!(a.same(&b), "{}: warm {a:?} ≠ fresh {b:?}", l.name);
        n += 1;
    }
    assert!(n > 40, "守る側の行が少ない: {n}");
    // 財布の表（`flow`・`a_tab`）を 12 桁より下だけ動かす（旧い鍵は財布の `key`＝12 桁に丸めた表で、この欄は変わらない）
    let line = text.lines().find(|l| l.contains("\"fn\":\"cb.rule_don_solve\"")).unwrap();
    let l = super::tests_outer::parse_line(line);
    let a = l.payload.get("a").clone();
    let d = a.get("actx").clone();
    let bump = |v: &V| V::list(v.items().iter().map(|x| V::Float(x.f() * (1.0 + 1e-13) + 1e-15)).collect());
    let d2 = dset(&dset(&d, "flow", bump(d.get("flow"))), "a_tab", bump(d.get("a_tab")));
    let p2 = dset(&l.payload, "a", dset(&a, "actx", d2));
    let first = super::obj::to_pyval(&super::entry::call_ev(&mut w, &l.name, &l.payload, false).unwrap().0);
    let hit = super::obj::to_pyval(&super::entry::call_ev(&mut w, &l.name, &p2, false).unwrap().0);
    let mut f = Core::new();
    let fresh = super::obj::to_pyval(&super::entry::call_ev(&mut f, &l.name, &p2, false).unwrap().0);
    assert!(hit.same(&fresh), "warm {hit:?} ≠ fresh {fresh:?}");
    assert!(!first.same(&fresh), "表の違いが計画を変えない（見張りが空振り）");
}
