//! 画面表示用の「継続中の状態」一覧（`Game.status_json`）。
//!
//! 盤面 dict（`board_json`）は golden の正本なので触らず、**表示専用の別出力**として
//! カードごとの状態（凍結・アタック不可・効果無効・KO 耐性・付与キーワード・パワー/コスト増減）と
//! 凍結ドン!!の枚数を出す。API（`opcg_sim/api/presenters.py`）が盤面の各カードへ合流させ、
//! フロントは状態が続く間だけ表示する（状態が消えれば次の盤面から消える）。
//!
//! 期間（`duration`）は `GameState.continuous` の該当エントリから取る（同じ状態が複数あれば
//! 長い方）。`flags` 由来（凍結・ブロッカー不可）と常在効果由来はエントリを持たないので固定値。

use serde_json::Value;

use crate::effects::ast::Duration;
use crate::model::{CardIdx, CardType, ContinuousKind, GameState, MasterTable, Obj, Seat};

/// 次のリフレッシュでアクティブにならない（`flags["FREEZE"]`）。
const DUR_NEXT_REFRESH: &str = "NEXT_REFRESH";
/// 常在効果（`apply_passive_effects` が毎回付け直す）＝条件が続く間。
const DUR_PASSIVE: &str = "PASSIVE";

/// `flags`（期間エントリを持たない）で表示する状態と、その期間。
const FLAG_STATUSES: &[(&str, &str)] = &[("FREEZE", DUR_NEXT_REFRESH), ("BLOCKER_DISABLED", "THIS_TURN")];

fn duration_name(d: Duration) -> &'static str {
    match d {
        Duration::Instant => "INSTANT",
        Duration::ThisTurn => "THIS_TURN",
        Duration::ThisBattle => "THIS_BATTLE",
        Duration::UntilNextTurnEnd => "UNTIL_NEXT_TURN_END",
        Duration::Permanent => "PERMANENT",
    }
}

/// 期間の長さの順位（同じ状態が複数あるとき長い方を見せる）。
fn duration_rank(d: Duration) -> u8 {
    match d {
        Duration::Instant => 0,
        Duration::ThisBattle => 1,
        Duration::ThisTurn => 2,
        Duration::UntilNextTurnEnd => 3,
        Duration::Permanent => 4,
    }
}

fn status(code: &str, duration: &str, expire_turn: Option<i32>) -> Value {
    let mut o = Obj::new();
    o.insert("code".into(), Value::from(code));
    o.insert("duration".into(), Value::from(duration));
    if let Some(t) = expire_turn {
        o.insert("expire_turn".into(), Value::from(t));
    }
    Value::Object(o)
}

impl GameState {
    /// 継続中の効果エントリのうち、`uuid` の `kind` で `name` に一致する最長のもの。
    fn longest_continuous(&self, uuid: &str, kind: ContinuousKind, name: &str) -> Option<(Duration, i32)> {
        self.continuous
            .iter()
            .filter(|e| e.target_uuid == uuid && e.kind == kind)
            .filter(|e| match kind {
                ContinuousKind::Flag => e.flag == name,
                ContinuousKind::Keyword => e.keyword == name,
                _ => false,
            })
            .map(|e| (e.duration, e.expire_turn))
            .max_by_key(|(d, t)| (duration_rank(*d), *t))
    }

    /// 期間付き状態 1 件（`lookup` で継続エントリを引き、`code` で出す）。
    fn timed_status(&self, uuid: &str, kind: ContinuousKind, lookup: &str, code: &str) -> Value {
        match self.longest_continuous(uuid, kind, lookup) {
            Some((d, t)) => status(
                code,
                duration_name(d),
                (d == Duration::UntilNextTurnEnd).then_some(t),
            ),
            // エントリが無い＝記録から復元した盤面など。期間は不明なのでターン内扱い。
            None => status(code, "THIS_TURN", None),
        }
    }

    /// 1 枚ぶんの状態。何も無ければ `None`（応答を小さく保つ）。
    fn card_status(&self, idx: CardIdx, masters: &MasterTable) -> Option<Value> {
        let c = self.card(idx);
        let m = masters.get(c.master);
        let mut statuses: Vec<Value> = Vec::new();

        for (flag, dur) in FLAG_STATUSES {
            if c.flags.iter().any(|f| f == flag) {
                statuses.push(status(flag, dur, None));
            }
        }
        for flag in &c.timed_flags {
            statuses.push(self.timed_status(&c.uuid, ContinuousKind::Flag, flag, flag));
        }
        // 付与キーワード: 期間付き（timed_keywords）と常在効果由来（current_keywords のうち
        // マスターに無いもの。常在の「アタックできない」もここに来る）。
        for kw in &c.timed_keywords {
            if !m.keywords.contains(kw) {
                statuses.push(self.timed_status(&c.uuid, ContinuousKind::Keyword, kw, &format!("KW:{kw}")));
            }
        }
        for kw in &c.current_keywords {
            if m.keywords.contains(kw) || c.timed_keywords.contains(kw) {
                continue;
            }
            let code = if kw == "ATTACK_DISABLE" { kw.clone() } else { format!("KW:{kw}") };
            statuses.push(status(&code, DUR_PASSIVE, None));
        }
        // パワー/コストの増減（付与ドン!!の +1000 は含めない＝相手ターン扱いで計算）。
        let power_mod = if matches!(m.ty, CardType::Leader | CardType::Character) {
            c.get_power(m, false) - m.power
        } else {
            0
        };
        let cost_mod = if matches!(m.ty, CardType::Character | CardType::Stage) {
            c.current_cost(m) - m.cost.max(0)
        } else {
            0
        };

        if statuses.is_empty() && power_mod == 0 && cost_mod == 0 {
            return None;
        }
        let mut o = Obj::new();
        o.insert("statuses".into(), Value::Array(statuses));
        o.insert("power_mod".into(), Value::from(power_mod));
        o.insert("cost_mod".into(), Value::from(cost_mod));
        Some(Value::Object(o))
    }

    /// `{"cards": {uuid: {statuses, power_mod, cost_mod}}, "players": {seat: {don_frozen}}}`。
    /// 対象は場に見えているカード（リーダー・キャラ・ステージ）だけ。
    pub fn status_json(&self, masters: &MasterTable) -> Value {
        let mut cards = Obj::new();
        let mut players = Obj::new();
        for seat in [Seat::P1, Seat::P2] {
            let p = self.player(seat);
            let on_board = p.leader.iter().chain(p.field.iter()).chain(p.stage.iter());
            for idx in on_board {
                if let Some(v) = self.card_status(*idx, masters) {
                    cards.insert(self.card(*idx).uuid.clone(), v);
                }
            }
            let don_frozen = p.don_rested.iter().filter(|d| self.don(**d).is_frozen).count();
            let mut po = Obj::new();
            po.insert("don_frozen".into(), Value::from(don_frozen));
            players.insert(seat.name().into(), Value::Object(po));
        }
        let mut out = Obj::new();
        out.insert("cards".into(), Value::Object(cards));
        out.insert("players".into(), Value::Object(players));
        Value::Object(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::actions::apply_action;
    use crate::effects::ast::{ActionType, GameAction};
    use crate::effects::{NodeRef, NodeRoot};
    use crate::journal::{DonBoolField, Session};
    use crate::testkit::{self, BoardBuilder, M_CHAR};

    fn run(s: &mut Session, masters: &MasterTable, action: &GameAction, target: CardIdx, value: i32) {
        apply_action(
            s, masters, Seat::P1, action, &NodeRef::root(0, NodeRoot::Effect),
            &crate::effects::refs_of(&[target]), value, None,
        )
        .expect("apply_action");
    }

    fn statuses(v: &Value, uuid: &str) -> Vec<(String, String, Option<i64>)> {
        v["cards"][uuid]["statuses"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|s| {
                        (
                            s["code"].as_str().unwrap().to_owned(),
                            s["duration"].as_str().unwrap().to_owned(),
                            s.get("expire_turn").and_then(Value::as_i64),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// 何も無い盤面はカードを出さない（応答を小さく保つ）。凍結ドン!!は 0。
    #[test]
    fn plain_board_has_no_card_entries() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        b.put_field(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let v = state.status_json(&masters);
        assert!(v["cards"].as_object().unwrap().is_empty(), "{v}");
        assert_eq!(v["players"]["p1"]["don_frozen"], 0);
    }

    /// 凍結＝次のリフレッシュまで・アタック不可（次のターン終了まで）・付与キーワード（このターン）・
    /// パワー増減が、期間つきで 1 枚にまとまって出る。
    #[test]
    fn reports_each_status_with_its_duration() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P1, M_CHAR);
        let (masters, state) = b.build();
        let mut s = Session::new(state);

        run(&mut s, &masters, &testkit::action(ActionType::Freeze, 0), c, 0);
        let mut atk = testkit::action(ActionType::AttackDisable, 0);
        atk.duration = Duration::UntilNextTurnEnd;
        run(&mut s, &masters, &atk, c, 0);
        let mut kw = testkit::action(ActionType::GrantKeyword, 0);
        kw.status = Some("速攻".to_string());
        kw.duration = Duration::ThisTurn;
        run(&mut s, &masters, &kw, c, 0);
        let mut buff = testkit::action(ActionType::Buff, 2000);
        buff.duration = Duration::ThisTurn;
        run(&mut s, &masters, &buff, c, 2000);

        let uuid = s.state().card(c).uuid.clone();
        let v = s.state().status_json(&masters);
        let got = statuses(&v, &uuid);
        assert!(got.contains(&("FREEZE".into(), "NEXT_REFRESH".into(), None)), "{got:?}");
        assert!(got.contains(&("ATTACK_DISABLE".into(), "UNTIL_NEXT_TURN_END".into(), Some(4))), "{got:?}");
        assert!(got.contains(&("KW:速攻".into(), "THIS_TURN".into(), None)), "{got:?}");
        assert_eq!(v["cards"][&uuid]["power_mod"], 2000);
    }

    /// 付与ドン!!の +1000 はパワー増減に含めない（効果による増減だけを見せる）。
    #[test]
    fn attached_don_is_not_a_power_modifier() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let c = b.put_field(Seat::P1, M_CHAR);
        b.attach_don(Seat::P1, c);
        let (masters, state) = b.build();
        assert!(state.status_json(&masters)["cards"].as_object().unwrap().is_empty());
    }

    /// レストの凍結ドン!!の枚数を席ごとに数える。
    #[test]
    fn counts_frozen_rested_don() {
        let mut b = BoardBuilder::new().turn(3, Seat::P1);
        let rested = b.dons(Seat::P2, "rested", 3);
        let (masters, state) = b.build();
        let mut s = Session::new(state);
        s.edit().set_don_bool(rested[0], DonBoolField::IsFrozen, true);
        s.edit().set_don_bool(rested[2], DonBoolField::IsFrozen, true);
        let v = s.state().status_json(&masters);
        assert_eq!(v["players"]["p2"]["don_frozen"], 2);
        assert_eq!(v["players"]["p1"]["don_frozen"], 0);
    }
}
