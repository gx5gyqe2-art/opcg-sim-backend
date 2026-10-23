//! 詰みの証明（`docs/search_efficiency_plan.md` §7.21・WP `rs-lethal-root`・MCTS-Solver の最も単純な形）。
//!
//! 監査の判定器（`tests/scripts/lethal_audit.py`〔ブランチ `claude/lethal-audit`〕の AND-OR の総当たり）の移植。
//! 自分の判断＝どれか 1 つ・相手の判断＝全部・このターンの終わりまで・置換表（自分の判断点だけ・ドン!!の個体を潰す）・
//! 攻撃の回数の上界による枝刈り。**健全性**: 自分の手の候補は絞る（見つかる勝ちが減るだけ）——
//!
//! - キャラへのアタックは外す（相手のライフを減らさない）。
//! - ドン!!の付与は「付けたユニットでそのまま攻撃する（か起動メインを使う）」形だけ（付けた直後の
//!   自分の判断点は、そのユニットの付与・攻撃・起動メインに限る）。レストのユニットへの付与は外す。
//! - 自分の効果対話は既定解決に加えて、確認の受ける／受けない・選択肢の各番号・場のカードを対象 1 枚ずつ
//!   （飛ばせるなら 0 枚）を足す。山札・手札・トラッシュから選ぶ対話（サーチ・捨てる）は既定解決だけ。
//!
//! **相手の応手は絞らない**: カウンター・ブロッカーは合法手の全部、確認は受ける／受けない、選択肢は全番号、
//! 対象の選択は選べる札の全部の組（枚数の制約の範囲・[`OPP_SELECT_MAX`] 組を超えたら証明しない）。
//! 相手の並べ替え（2 枚以上）は数え切れない＝証明しない。
//!
//! **配り方は公平に作る**（本当の配り方は読まない）: 決定の世界の引き直し（`fair`・知っている札は固定）
//! で [`LETHAL_RAND_DEALS`] 通り＋守りの強い配り方 2 通り（相手の知らない札のうち守りの値の大きい札を
//! 手札の枠へ先に入れる。自分の山札・裏向きのライフは `fair` と同じに混ぜる）。**全部の配り方で勝てる**
//! ときだけ「証明」とする（根では同じ最初の 1 手から）。打ち切りは**節点数**（時間ではない＝決定的）。

use crate::journal::{CardZone, Session};
use crate::effects::ast::TriggerType;
use crate::model::{CardIdx, CardType, GameState, MasterIdx, MasterTable, Seat};
use crate::rules::moves::Payload;
use crate::state::EngineError;
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::rng::{Pcg32SearchRng, SearchRng};
use super::Move;

/// 公平な引き直しの配り方の数（守りの強い 2 通りとは別）。
pub const LETHAL_RAND_DEALS: usize = 6;
/// 根の判定の節点の上限（全部の配り方・全部の候補の合計・opts の `lethal_nodes`）。
pub const LETHAL_ROOT_NODES: u64 = 20_000;
/// `root_reply` の応手の節点 1 つあたりの上限（opts の `lethal_reply_nodes`）。
pub const LETHAL_REPLY_NODES: u64 = 5_000;
/// `root_reply` の決定あたりの合計の上限（世界の本数で等分する・opts の `lethal_reply_total`）。
pub const LETHAL_REPLY_TOTAL: u64 = 40_000;
/// 相手の対象の選択で読む組の数の上限（超えたら証明しない＝相手の応手は絞らない）。
pub const OPP_SELECT_MAX: usize = 64;
/// 1 本の手順の深さの上限（監査の `MAX_DEPTH`）。
pub const LETHAL_MAX_DEPTH: usize = 300;

/// 計測用の切替（§7.30・WP `lethal-own-hidden`・環境変数 `OPCG_LETHAL_OWN`）:
/// `off`＝本線のまま／`cost`＝守りの強い配り方でカウンターイベントを相手の払えるドン!!の範囲でだけ数える／
/// `hidden`＝自分の隠れた札（山札・裏向きのライフ）を見る・引く手を含む手順は証明に使わない／`both`＝両方。
fn own_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("OPCG_LETHAL_OWN").as_deref() {
        Ok("cost") => 1,
        Ok("hidden") => 2,
        Ok("both") => 3,
        _ => 0,
    })
}

/// 切替（opts の `lethal_proof`・§7.21）。**既定 `Off`＝1 bit も変わらない**。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LethalProof {
    #[default]
    Off,
    /// 根で判定し、証明できれば最初の 1 手を返す（探索は回さない）。
    Root,
    /// `Root` に加えて、根の直後の相手の応手の節点で「この応手なら次にこちらの勝ちが証明できる」子を
    /// 相手から見て負け（-1）に固定する（[`super::mcts::TreeMcts`] の `lethal_reply`）。
    RootReply,
}

impl LethalProof {
    pub fn from_name(s: &str) -> Option<LethalProof> {
        match s {
            "off" => Some(LethalProof::Off),
            "root" => Some(LethalProof::Root),
            "root_reply" => Some(LethalProof::RootReply),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LethalProof::Off => "off",
            LethalProof::Root => "root",
            LethalProof::RootReply => "root_reply",
        }
    }
}

/// `root_reply` の木の中の証明の状態と実績（世界ごと・[`super::mcts::TreeMcts`] が持つ）。
#[derive(Debug, Clone)]
pub struct ReplyProof {
    pub me: Seat,
    /// 配り方の乱数の基（世界の seed から・節点と子の番号を混ぜて使う）
    pub seed: u64,
    /// 応手の節点 1 つあたりの節点の上限
    pub per_node: u64,
    /// この世界の残りの節点
    pub left: u64,
    /// 判定した応手の節点の数・子の数・証明できた子の数・使った節点
    pub checked: u64,
    pub children: u64,
    pub proven: u64,
    pub nodes: u64,
}

impl ReplyProof {
    pub fn new(me: Seat, seed: u64, per_node: u64, left: u64) -> ReplyProof {
        ReplyProof { me, seed, per_node, left, checked: 0, children: 0, proven: 0, nodes: 0 }
    }

    /// 応手の節点 `node` の各子（相手の応手 `legal`）について、その応手の後にこちらの勝ちが証明できるか。
    /// 子の盤面は木と同じ適用（`apply_move_inplace`・自分の選択は残す）で作り、読み終えたら巻き戻す。
    pub fn prove_children(
        &mut self,
        masters: &MasterTable,
        s: &mut Session,
        node: usize,
        legal: &[Move],
    ) -> Result<Vec<bool>, EngineError> {
        self.checked += 1;
        let mut budget = self.per_node.min(self.left);
        let mut out = vec![false; legal.len()];
        let opp = self.me.other();
        let rng_state = s.rng.snapshot();
        for (i, mv) in legal.iter().enumerate() {
            if budget == 0 {
                break;
            }
            self.children += 1;
            let seed = self.seed ^ ((node as u64) << 20 | i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let me = self.me;
            let r = s.discard_events(|s| {
                s.transaction(|s| -> Result<(bool, u64), EngineError> {
                    match super::apply::apply_move_inplace(s, masters, opp, mv, true) {
                        Ok(()) => prove(masters, s.state(), me, seed, budget),
                        Err(e @ EngineError::Unimplemented(_)) => Err(e),
                        Err(_) => Ok((false, 0)),
                    }
                })
            });
            s.rng.restore(&rng_state);
            let (ok, used) = r?;
            budget = budget.saturating_sub(used);
            self.left = self.left.saturating_sub(used);
            self.nodes += used;
            if ok {
                self.proven += 1;
                out[i] = true;
            }
        }
        Ok(out)
    }
}

/// 配り方の乱数の seed（決定の `search_seed` から、世界の引き直しと重ならない系列を作る）。
pub fn lethal_seed(search_seed: u64) -> u64 {
    search_seed ^ 0x1E7A_15EE_D0D1_E5EDu64
}

/// 判定の結果。
#[derive(Debug, Clone)]
pub struct LethalOut {
    /// "win"（全部の配り方で勝てる最初の 1 手が在る）／"no"（無い）／"unknown"（節点の上限・数え切れない応手で打ち切り）。
    pub status: &'static str,
    /// 勝てる最初の 1 手（`status == "win"` のときだけ）。
    pub mv: Option<Move>,
    /// 勝ち筋の長さ（最初の配り方で辿った手の数・両席）。
    pub line_len: usize,
    /// 配り方の数。
    pub deals: usize,
    pub nodes: u64,
    /// 計測用（§7.30・`Game.lethal_json` の `trace`）: 配り方ごとの手順（[`Solver::walk`]）。ふつうは `None`。
    pub lines: Option<Vec<Value>>,
}

/// 発動条件（§7.21 の 1）: 手番が自分で、自分が答える判断点（メイン、または自分の効果対話）で、**このターンに出せる
/// 攻撃の数**が相手のライフ＋1 以上（届きうる）。数え方は判定器の上界と同じ（[`Solver::bound_fails`]・自分の札に
/// 打点を増やしうる文言が 1 枚でも在れば届きうる扱い）。メインでは今アタックできるユニット（合法手）を数え、効果対話
/// では合法手に攻撃が出ないので、アクティブで攻撃できるユニット（レストでない・登場したターンなら速攻持ち）＋いま
/// 自分が攻撃中なら 1 を数える。
///
/// 指示書（§7.21 の 1）はメインだけだが、証明した手順の途中の自分の効果対話（アタック時・登場時の選択）を探索に任せる
/// と、探索が別の答えを選んで勝ちを手放す例が出た（打ち直しの監査・`2026-10-22_lethal_root.md` §4）ため対話にも広げた。
pub fn applies(s: &mut Session, masters: &MasterTable, me: Seat) -> bool {
    let st = s.state();
    if st.winner.is_some() || st.turn_player != me {
        return false;
    }
    let action = match crate::rules::pending::pending_actor_action(s) {
        Some((seat, action)) if seat == me => action,
        _ => return false,
    };
    if matches!(action, "MULLIGAN" | "SELECT_BLOCKER" | "SELECT_COUNTER") {
        return false;
    }
    let mut solver = Solver::new(masters, s.state(), me, 0);
    if action == MAIN {
        let Ok(legal) = crate::rules::legal::get_legal_moves(s, masters, me) else {
            return false;
        };
        return !solver.bound_fails(s.state(), &legal);
    }
    let st = s.state();
    let ready = ready_units(st, me) + st.active_battle.as_ref().is_some_and(|b| b.attacker_owner == me) as usize;
    match solver.attack_bound(st, ready) {
        None => true,
        Some(b) => b > st.player(me.other()).life.len(),
    }
}

/// アクティブで攻撃できるユニット（レストでない・登場したターンなら速攻持ち）の数（効果対話の中で合法手に
/// 攻撃が出ないときの「今アタックできるユニット」）。
fn ready_units(st: &GameState, me: Seat) -> usize {
    let p = st.player(me);
    p.leader
        .iter()
        .chain(&p.field)
        .filter(|c| {
            let card = st.card(**c);
            !card.is_rest && (!card.is_newly_played || card.current_keywords.iter().any(|k| k.starts_with("速攻")))
        })
        .count()
}

/// 発動条件の内訳（計測用・`Game.lethal_json` の `bound`）: 必要な攻撃の数（相手のライフ＋1）・今アタックできる
/// ユニットの数・上界（`null`＝上界なし）。
pub fn bound_report(s: &mut Session, masters: &MasterTable, me: Seat) -> Value {
    let mut solver = Solver::new(masters, s.state(), me, 0);
    let action = crate::rules::pending::pending_actor_action(s).map(|(_, a)| a).unwrap_or("");
    let ready = if action == MAIN {
        let legal = crate::rules::legal::get_legal_moves(s, masters, me).unwrap_or_default();
        distinct_attackers(&legal)
    } else {
        let st = s.state();
        ready_units(st, me) + st.active_battle.as_ref().is_some_and(|b| b.attacker_owner == me) as usize
    };
    let st = s.state();
    let bound = solver.attack_bound(st, ready);
    serde_json::json!({
        "need": st.player(me.other()).life.len() + 1, "ready": ready, "bound": bound,
        "unbounded_by": solver.unbounded_by,
    })
}

/// 合法手の攻撃者（重複を潰した数）。
fn distinct_attackers(moves: &[Move]) -> usize {
    let mut att: Vec<&str> = moves.iter().filter(|m| act_of(m) == "ATTACK").filter_map(mv_uuid).collect();
    att.sort_unstable();
    att.dedup();
    att.len()
}

// --- 守りの強さ（守りの強い配り方を作るためだけ）---------------------------------------------

/// 本文の「パワー+N」の N の最大（無ければ None）。
fn max_power_up(text: &str) -> Option<i32> {
    let mut best: Option<i32> = None;
    for (i, _) in text.match_indices("パワー+") {
        let digits: String = text[i + "パワー+".len()..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(v) = digits.parse::<i32>() {
            best = Some(best.map_or(v, |b| b.max(v)));
        }
    }
    best
}

/// 手札に在るときの守りの値（監査の `defense_value`）と、カウンターイベントか。
/// ブロッカーは同じ値の中で先にする（手札のブロッカーはこのターンには守れない＝並びの差だけ）。
fn defense(masters: &MasterTable, m: MasterIdx) -> (i64, bool) {
    let cm = masters.get(m);
    let mut v = cm.counter.max(0) as i64;
    let ev = cm.ty == CardType::Event && cm.effect_text.contains("【カウンター】");
    if ev {
        v = v.max(max_power_up(&cm.effect_text).unwrap_or(1500) as i64) + 500;
    }
    let blocker = cm.keywords.iter().any(|k| k == "ブロッカー");
    (v * 2 + blocker as i64, ev)
}

/// 守りの強い配り方（`mode` 0＝守りの値の大きい順に手札へ・1＝カウンターイベントを先に）。
///
/// 相手の手札＋山札＋裏向きのライフのうち自分の知らない枠（`fair` と同じ）の札を 1 つの山にし、
/// 無作為に混ぜてから守りの値で安定に並べ、上から手札の枠へ入れる。残りは混ぜて山札・ライフの枠へ。
/// 自分の側は `fair` と同じに混ぜる。
fn defensive_deal(
    state: &GameState,
    masters: &MasterTable,
    me: Seat,
    mode: u8,
    rng: &mut dyn SearchRng,
) -> Result<GameState, EngineError> {
    let opp = me.other();
    let mut out = state.clone();
    let unknown = |c: CardIdx| state.card(c).known_by & me.know_bit() == 0;
    let p = state.player(opp);
    let mut slots: Vec<(CardZone, usize)> = Vec::new();
    slots.extend(p.hand.iter().enumerate().filter(|(_, c)| unknown(**c)).map(|(i, _)| (CardZone::Hand, i)));
    slots.extend(p.deck.iter().enumerate().filter(|(_, c)| unknown(**c)).map(|(i, _)| (CardZone::Deck, i)));
    slots.extend(
        p.life
            .iter()
            .enumerate()
            .filter(|(_, c)| !state.card(**c).is_face_up && unknown(**c))
            .map(|(i, _)| (CardZone::Life, i)),
    );
    let at = |z: CardZone, i: usize| match z {
        CardZone::Hand => p.hand[i],
        CardZone::Deck => p.deck[i],
        _ => p.life[i],
    };
    let pool: Vec<CardIdx> = slots.iter().map(|&(z, i)| at(z, i)).collect();
    if !pool.is_empty() {
        let by_uuid = |order: Vec<String>| -> Vec<CardIdx> {
            order
                .iter()
                .filter_map(|u| pool.iter().copied().find(|c| &state.card(*c).uuid == u))
                .collect()
        };
        let uuids: Vec<String> = pool.iter().map(|c| state.card(*c).uuid.clone()).collect();
        let mut mixed = by_uuid(rng.shuffle_order(&uuids)?);
        let key = |v: i64, ev: bool| if mode == 1 { (-(ev as i64), -v) } else { (0, -v) };
        mixed.sort_by_key(|c| {
            let (v, ev) = defense(masters, state.card(*c).master);
            key(v, ev)
        });
        if own_mode() & 1 != 0 {
            // カウンターイベントは**相手が今払えるドン!!**（アクティブのドン!!）の範囲でだけ守りに数える（§7.30）:
            // 守りの値の順に見て、コストが残りの予算に収まるイベントだけ値を残し、残りは値 0（手札の枠に入れない）。
            // 旧い形は払えないイベントを値のまま手札の枠へ積み、本当の手札より弱い「守りの強い配り方」を作っていた。
            let mut budget = p.don_active.len() as i32;
            let eff: Vec<(CardIdx, (i64, i64))> = mixed
                .iter()
                .map(|c| {
                    let cm = masters.get(state.card(*c).master);
                    let (v, ev) = defense(masters, state.card(*c).master);
                    if ev {
                        if cm.cost <= budget {
                            budget -= cm.cost;
                            (*c, key(v, true))
                        } else {
                            (*c, key(0, false))
                        }
                    } else {
                        (*c, key(v, false))
                    }
                })
                .collect();
            let mut eff = eff;
            eff.sort_by_key(|(_, k)| *k);
            mixed = eff.into_iter().map(|(c, _)| c).collect();
        }
        let n_hand = slots.iter().filter(|(z, _)| *z == CardZone::Hand).count();
        let (hand, rest) = mixed.split_at(n_hand);
        let rest_uuids: Vec<String> = rest.iter().map(|c| state.card(*c).uuid.clone()).collect();
        let rest = if rest_uuids.is_empty() { Vec::new() } else { by_uuid(rng.shuffle_order(&rest_uuids)?) };
        let mut cards = hand.iter().copied().chain(rest);
        let po = out.player_mut(opp);
        for &(z, i) in &slots {
            let c = cards.next().expect("同じ枚数");
            match z {
                CardZone::Hand => po.hand[i] = c,
                CardZone::Deck => po.deck[i] = c,
                _ => po.life[i] = c,
            }
        }
    }
    // 自分の山札＋裏向きのライフ（`fair` の自分側と同じ）
    super::determinize::reshuffle_own_fair(&mut out, state, me, rng)?;
    Ok(out)
}

/// 配り方を作る（`fair` の引き直し `rand` 通り＋守りの強い 2 通り・守りの強い方を先に＝早く落ちる）。
pub fn make_deals(
    state: &GameState,
    masters: &MasterTable,
    me: Seat,
    seed: u64,
    rand: usize,
) -> Result<Vec<GameState>, EngineError> {
    let mut rng = Pcg32SearchRng::new(seed);
    let mut out = Vec::with_capacity(rand + 2);
    out.push(defensive_deal(state, masters, me, 0, &mut rng)?);
    out.push(defensive_deal(state, masters, me, 1, &mut rng)?);
    for _ in 0..rand {
        out.push(super::determinize::determinize_fair_with(state, me, &mut rng)?);
    }
    Ok(out)
}

// --- 判定器 ---------------------------------------------------------------------------------

/// 上限を超えた（判定不能）。
struct Abort;

/// 打点の回数を増やしうる文言（監査の `DANGER_WORDS`）。
const DANGER_WORDS: &[&str] = &[
    "速攻", "ダブルアタック", "アクティブに", "アクティブで", "ダメージ", "相手のライフ", "アタックできる",
    "もう一度", "レストにできない",
];
/// 他のカードを場に出す文言（監査の `SUMMON_WORDS`）。
const SUMMON_WORDS: &[&str] = &["登場させる", "登場できる"];

const MAIN: &str = "MAIN_ACTION";

struct Solver<'a> {
    masters: &'a MasterTable,
    me: Seat,
    turn: i32,
    opp_leader: String,
    max_nodes: u64,
    nodes: u64,
    /// 置換表（自分の判断点だけ・書くのは確定値だけ。相手の判断点にも引く案は節点 −0.5% で鍵の費用の方が大きかった＝§7.25 の報告）
    tt: HashMap<u64, bool>,
    /// 勝てた自分の判断点で選んだ手（勝ち筋の長さを数える）
    best: HashMap<u64, Move>,
    /// 札ごとの能力の読み（[`MasterInfo`]・上界のため）
    info: HashMap<MasterIdx, MasterInfo>,
    /// 直前の上界が「上界なし」になった札（計測用・`bound_report`）
    unbounded_by: Vec<String>,
    /// 自分の隠れた札（この配り方の根で自分の山札・裏向きのライフに在り、自分が知らない札・§7.30 の `hidden`）
    hidden: Vec<CardIdx>,
}

/// 上界のための札の読み（本文の文言は**解釈された能力の本文**（`Ability::raw_text`）だけを見る＝キーワードの
/// 説明文「(このカードは登場したターンにアタックできる)」等は能力ではないので数えない。エンジンが実行するのは
/// 解釈された能力だけ）。
#[derive(Debug, Clone, Default)]
struct MasterInfo {
    /// 能力ごとの (いつ働くか, 文言の旗)
    abilities: Vec<(TriggerType, u8)>,
    /// 【速攻】（「速攻:キャラ」も含む・上界では区別しない）
    rush: bool,
    /// 【ダブルアタック】
    double: bool,
}

const W_DANGER: u8 = 1;
const W_SUMMON: u8 = 2;
const W_COSTRED: u8 = 4;
const W_TRASH: u8 = 8;
const W_BOUNCE: u8 = 16;

impl MasterInfo {
    fn new(cm: &crate::model::CardMaster, abilities: &crate::effects::ast::AbilityTable) -> MasterInfo {
        let abilities = cm
            .ability_ids
            .iter()
            .filter_map(|id| abilities.get(*id))
            .map(|a| {
                let t = &a.raw_text;
                let mut w = 0u8;
                if DANGER_WORDS.iter().any(|x| t.contains(x)) {
                    w |= W_DANGER;
                }
                if SUMMON_WORDS.iter().any(|x| t.contains(x)) {
                    w |= W_SUMMON;
                }
                if t.contains("コスト-") {
                    w |= W_COSTRED;
                }
                if t.contains("トラッシュ") {
                    w |= W_TRASH;
                }
                if t.contains("手札に戻") {
                    w |= W_BOUNCE;
                }
                (a.trigger, w)
            })
            .collect();
        MasterInfo {
            abilities,
            rush: cm.keywords.iter().any(|k| k.starts_with("速攻")),
            double: cm.keywords.iter().any(|k| k == "ダブルアタック"),
        }
    }
}

/// 札の置き場（上界で「このターンに働きうる能力」を選ぶため）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Where {
    /// 場（リーダー・キャラ・ステージ）
    Board,
    /// 手札（と、手札に戻りうるトラッシュ・登場させうる山札）
    Hand,
}

/// この置き場の札の能力が、自分のこのターンの残りで働きうるか（自分の手番の MAIN 以降）。
/// 【登場時】は場の札ではもう働かない（手札に戻す文言が在れば働きうる）・【アタック時】は手札の札では
/// 速攻が無ければ働かない・相手のターン／ブロック時／カウンター／トリガー／ターン開始時／ターン終了時は働かない。
fn ability_live(t: TriggerType, at: Where, rush: bool, bounce: bool) -> bool {
    use TriggerType::*;
    match t {
        Trigger | Counter | OnBlock | OnOppAttack | OpponentTurn | OpponentAttack | OppTurnEnd | TurnStart | GameStart
        | OnOppPlay | TurnEnd => false,
        OnPlay => at == Where::Hand || bounce,
        OnAttack => at == Where::Board || rush,
        _ => true,
    }
}

fn act_of(mv: &Move) -> &str {
    mv.action_type().unwrap_or("")
}

/// 自分の手の並び（監査の `_my_order`・キャラへのアタックは先に外してある）。
fn my_order(m: &Move) -> u8 {
    match act_of(m) {
        "ATTACK" => 0,
        "ATTACH_DON" => 1,
        "PLAY" => 2,
        "ACTIVATE_MAIN" => 3,
        "TURN_END" => 9,
        _ => 5,
    }
}

/// 相手の手の並び（監査の `_opp_order`）。
fn opp_order(m: &Move) -> u8 {
    match act_of(m) {
        "SELECT_COUNTER" | "SELECT_BLOCKER" => 0,
        "PASS" => 2,
        _ => 1,
    }
}

fn mv_uuid(mv: &Move) -> Option<&str> {
    mv.payload_uuid()
}

impl<'a> Solver<'a> {
    fn new(masters: &'a MasterTable, state: &GameState, me: Seat, max_nodes: u64) -> Self {
        let opp_leader = state
            .player(me.other())
            .leader
            .map(|c| state.card(c).uuid.clone())
            .unwrap_or_default();
        Solver {
            masters,
            me,
            turn: state.turn_count,
            opp_leader,
            max_nodes,
            nodes: 0,
            tt: HashMap::new(),
            best: HashMap::new(),
            info: HashMap::new(),
            unbounded_by: Vec::new(),
            hidden: if own_mode() & 2 != 0 {
                let p = state.player(me);
                let bit = me.know_bit();
                p.deck
                    .iter()
                    .chain(p.life.iter().filter(|c| !state.card(**c).is_face_up))
                    .copied()
                    .filter(|c| state.card(*c).known_by & bit == 0)
                    .collect()
            } else {
                Vec::new()
            },
        }
    }

    /// 自分の隠れた札のどれかを見た・引いた（`known_by` に自分の bit が付いた）か（§7.30 の `hidden`）。
    /// 知識の印は journal が領域の出入り・LOOK・REVEAL で自動で付ける（`docs/SPEC.md` §2.5.9）。
    fn revealed_own_hidden(&self, st: &GameState) -> bool {
        let bit = self.me.know_bit();
        self.hidden.iter().any(|c| st.card(*c).known_by & bit != 0)
    }

    fn info_of(&mut self, m: MasterIdx) -> &MasterInfo {
        let masters = self.masters;
        self.info.entry(m).or_insert_with(|| MasterInfo::new(masters.get(m), &masters.abilities))
    }

    /// 札の能力の文言の旗（置き場で働きうる能力だけ・無効化された札は 0）。
    fn words(&mut self, st: &GameState, c: CardIdx, at: Where, bounce: bool) -> u8 {
        let card = st.card(c);
        if card.negated || card.ability_disabled {
            return 0;
        }
        let rush = card.current_keywords.iter().any(|k| k.starts_with("速攻"));
        let info = self.info_of(card.master);
        let rush = rush || info.rush;
        info.abilities.iter().filter(|(t, _)| ability_live(*t, at, rush, bounce)).fold(0u8, |w, (_, x)| w | x)
    }

    /// 置換表の鍵（盤面＋要求＋付与の続き）。監査は hidden の文字列（ドン!!の uuid を捨てる）を鍵にしたが、
    /// それは 1 回 0.4 ms かかる＝ここでは同じ中身を直接混ぜる（[`state_hash`]）。
    fn key(&self, s: &mut Session, action: &str, commit: Option<&str>) -> u64 {
        let _prof = crate::prof::enter(if action == MAIN { crate::prof::Cat::LethalKey } else { crate::prof::Cat::LethalKeyDialog });
        // 速いハッシュ（`fxhash`）＝鍵の中身は同じ・衝突の確率も同程度（§7.25 の 4・旧 `DefaultHasher` は 1 節点 12 µs で
        // 判定の時間の半分を占めていた）
        let mut hs = crate::fxhash::FxHasher::default();
        state_hash(s.state(), &mut hs);
        action.hash(&mut hs);
        if action != MAIN {
            // 中断の中身（どの効果のどの段か・候補）も混ぜる＝取り違えより取りこぼしの側に倒す
            interaction_hash(s.state(), &mut hs);
        }
        commit.hash(&mut hs);
        hs.finish()
    }

    /// 要求の手（合法手＋効果対話の分岐）。自分の対話は監査の `expand` と同じに絞る。相手の対話は
    /// 絞らない（[`Solver::opp_variants`]）＝数え切れなければ `Abort`（証明しない）。
    fn expand(&self, s: &mut Session, actor: Seat, action: &str) -> Result<Vec<Move>, Abort> {
        let legal = {
            let _prof = crate::prof::enter(crate::prof::Cat::Legal);
            crate::rules::legal::get_legal_moves(s, self.masters, actor).map_err(|_| Abort)?
        };
        if matches!(action, MAIN | "SELECT_COUNTER" | "SELECT_BLOCKER" | "MULLIGAN") || legal.is_empty() {
            return Ok(legal);
        }
        let _prof = crate::prof::enter(crate::prof::Cat::LethalExpand);
        let base = legal[0].clone();
        let Move::Game { at, payload } = &base else {
            return Ok(legal);
        };
        let pend = crate::rules::pending::get_pending_request(s, self.masters, false).unwrap_or(Value::Null);
        let payload = payload.to_json();
        let with = |k: &str, v: Value| {
            let mut p = payload.clone();
            if let Some(o) = p.as_object_mut() {
                o.insert(k.into(), v);
            }
            p
        };
        let mut variants = vec![payload.clone()];
        if actor != self.me {
            variants.extend(opp_variants(action, &pend, &with)?);
        } else if action.starts_with("CONFIRM") {
            variants.push(with("accepted", Value::Bool(false)));
        } else if action == "CHOICE" {
            let n = pend.get("options").and_then(Value::as_array).map_or(0, |a| a.len());
            for i in 0..n {
                variants.push(with("index", Value::from(i)));
            }
        } else if action != "ARRANGE_DECK" {
            let live = in_play(s.state());
            let uu: Vec<String> = pend
                .get("selectable_uuids")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|u| u.as_str().map(str::to_owned)).collect())
                .unwrap_or_default();
            for u in uu.into_iter().filter(|u| live.contains(u)) {
                variants.push(with("selected_uuids", Value::Array(vec![Value::String(u)])));
            }
            if pend.get("can_skip").and_then(Value::as_bool) == Some(true) {
                variants.push(with("selected_uuids", Value::Array(Vec::new())));
            }
        }
        // 同じ答えは 1 つに（JSON の文字列ではなく値で比べる＝同じ同値関係・文字列化の分だけ速い）
        let mut out: Vec<Move> = Vec::new();
        for v in variants {
            let dup = out.iter().any(|m| matches!(m, Move::Game { payload: Payload::Json(x), .. } if *x == v));
            if !dup {
                out.push(Move::Game { at, payload: Payload::Json(v) });
            }
        }
        Ok(out)
    }

    /// 自分の判断点の手の絞り込みと並び（監査と同じ）。`None`＝上界で詰みが無いと言えた。
    fn my_moves(&mut self, s: &mut Session, action: &str, commit: Option<&str>, moves: Vec<Move>) -> Option<Vec<Move>> {
        let leader = self.opp_leader.clone();
        let mut moves: Vec<Move> = moves
            .into_iter()
            .filter(|m| !(act_of(m) == "ATTACK" && m.first_target_id() != Some(leader.as_str())))
            .collect();
        if action == MAIN {
            let att: Vec<String> = moves
                .iter()
                .filter(|m| act_of(m) == "ATTACK")
                .filter_map(|m| mv_uuid(m).map(str::to_owned))
                .collect();
            moves = match commit {
                Some(c) => moves
                    .into_iter()
                    .filter(|m| mv_uuid(m) == Some(c) && matches!(act_of(m), "ATTACH_DON" | "ATTACK" | "ACTIVATE_MAIN"))
                    .collect(),
                None => moves
                    .into_iter()
                    .filter(|m| act_of(m) != "ATTACH_DON" || mv_uuid(m).is_some_and(|u| att.iter().any(|a| a == u)))
                    .collect(),
            };
            if commit.is_none() && self.bound_fails(s.state(), &moves) {
                return None;
            }
        }
        moves.sort_by_key(my_order);
        Some(moves)
    }

    /// このターンに自分が出せる攻撃（相手のライフを減らしうる機会）の数の上界（§7.25 の 1）。`None`＝上界なし。
    ///
    /// 数えるのは**今この手番で使える札**だけ: 場の札（リーダー・キャラ・ステージ）と、今のアクティブのドン!!で
    /// 払える手札。使える札の働きうる能力（[`ability_live`]）の本文に「登場させる」「コスト-」が在れば手札は
    /// 全部使えるとみなし、「トラッシュ」が在ればトラッシュの札も、「登場させる」が在れば山札の札も加える
    /// （旗が増えなくなるまで繰り返す）。使える札の働きうる能力の本文に打点の回数を増やしうる文言
    /// （[`DANGER_WORDS`]）が 1 つでも在れば上界なし。無ければ `ready`（今アタックできるユニット）＋手札から
    /// 登場させて速攻で殴れる札の数（払えるぶんだけ・手札が全部使えるなら全部）＋【ダブルアタック】の札の数。
    ///
    /// 旧い上界（WP `rs-lethal-root`）は「手札・場・トラッシュの札の本文のどこかに文言が在れば上界なし」で、
    /// 速攻の説明文「(…アタックできる)」や【トリガー】・【カウンター】の本文まで数えていたため、メインの 87% で
    /// 判定に入っていた（`docs/reports/2026-10-22_lethal_speed.md`）。
    fn attack_bound(&mut self, st: &GameState, ready: usize) -> Option<usize> {
        let _prof = crate::prof::enter(crate::prof::Cat::LethalBound);
        self.unbounded_by.clear();
        let p = st.player(self.me);
        let don = p.don_active.len() as i32;
        let board: Vec<CardIdx> = p.leader.iter().chain(&p.field).chain(p.stage.iter()).copied().collect();
        let payable = |st: &GameState, masters: &MasterTable, c: CardIdx| {
            let card = st.card(c);
            card.current_cost(masters.get(card.master)) <= don
        };
        let masters = self.masters;
        // 使える札の旗を、増えなくなるまで集める
        let mut flags = 0u8;
        loop {
            let free = flags & (W_SUMMON | W_COSTRED) != 0;
            let mut usable: Vec<(CardIdx, Where)> = board.iter().map(|c| (*c, Where::Board)).collect();
            usable.extend(p.hand.iter().filter(|c| free || payable(st, masters, **c)).map(|c| (*c, Where::Hand)));
            if flags & W_TRASH != 0 {
                usable.extend(p.trash.iter().filter(|c| free || payable(st, masters, **c)).map(|c| (*c, Where::Hand)));
            }
            if flags & W_SUMMON != 0 {
                usable.extend(p.deck.iter().map(|c| (*c, Where::Hand)));
            }
            let bounce = flags & W_BOUNCE != 0;
            let mut next = flags;
            for (c, at) in &usable {
                let w = self.words(st, *c, *at, bounce);
                next |= w;
                if w & W_DANGER != 0 && self.unbounded_by.len() < 8 {
                    let id = masters.get(st.card(*c).master).card_id.clone();
                    if !self.unbounded_by.contains(&id) {
                        self.unbounded_by.push(id);
                    }
                }
            }
            if next == flags {
                // 旗が増えない＝使える札の集合が確定
                if flags & W_DANGER != 0 {
                    return None;
                }
                // 速攻で殴れる手札（払えるぶんだけ）とダブルアタック
                let mut rush_costs: Vec<i32> = Vec::new();
                let mut double = 0usize;
                for (c, at) in &usable {
                    let card = st.card(*c);
                    let info = self.info_of(card.master).clone();
                    let kw = |k: &str| card.current_keywords.iter().any(|x| x == k);
                    if info.double || kw("ダブルアタック") {
                        double += 1;
                    }
                    if *at == Where::Hand
                        && masters.get(card.master).ty == CardType::Character
                        && (info.rush || card.current_keywords.iter().any(|k| k.starts_with("速攻")))
                    {
                        rush_costs.push(card.current_cost(masters.get(card.master)));
                    }
                }
                let rush = if free {
                    rush_costs.len()
                } else {
                    rush_costs.sort_unstable();
                    let mut left = don;
                    rush_costs
                        .iter()
                        .take_while(|c| {
                            left -= **c;
                            left >= 0
                        })
                        .count()
                };
                return Some(ready + rush + double);
            }
            flags = next;
        }
    }

    /// 攻撃の回数の上界で詰みが無いと言えるか（上界が相手のライフ＋1 に届かない）。
    fn bound_fails(&mut self, st: &GameState, moves: &[Move]) -> bool {
        let need = st.player(self.me.other()).life.len() + 1;
        self.attack_bound(st, distinct_attackers(moves)).is_some_and(|b| b < need)
    }

    fn solve(&mut self, s: &mut Session, commit: Option<&str>, depth: usize) -> Result<bool, Abort> {
        let st = s.state();
        if let Some(w) = st.winner {
            return Ok(w == self.me);
        }
        if st.turn_count != self.turn || st.turn_player != self.me {
            return Ok(false);
        }
        let Some((actor, action)) = crate::rules::pending::pending_actor_action(s) else {
            return Ok(false);
        };
        let mine = actor == self.me;
        let key = if mine { Some(self.key(s, action, commit)) } else { None };
        if let Some(k) = key {
            if let Some(&r) = self.tt.get(&k) {
                return Ok(r);
            }
        }
        self.nodes += 1;
        if depth > LETHAL_MAX_DEPTH || self.nodes > self.max_nodes {
            return Err(Abort);
        }
        let moves = self.expand(s, actor, action)?;
        let moves = if mine {
            match self.my_moves(s, action, commit, moves) {
                Some(m) => m,
                None => {
                    self.tt.insert(key.expect("mine"), false);
                    return Ok(false);
                }
            }
        } else {
            let mut m = moves;
            m.sort_by_key(opp_order);
            m
        };
        let mut res = !mine;
        let mut tried = 0;
        for mv in moves {
            // 子は盤面を複製せず、記録つきで適用して読み終えたら巻き戻す（`Session::isolated` と同じ見え方）
            s.begin();
            if apply(s, self.masters, actor, &mv).is_err() {
                let _prof = crate::prof::enter(crate::prof::Cat::LethalRollback);
                s.rollback();
                continue; // 分岐した対話の不正な選択などは手として数えない
            }
            tried += 1;
            let nxt = if mine && act_of(&mv) == "ATTACH_DON" { mv_uuid(&mv).map(str::to_owned) } else { None };
            // 自分の隠れた札を見る・引く手を含む手順は証明に使わない（`hidden`）: 自分の手なら負けの手として
            // 数え、相手の応手なら（その先は読めないので）こちらの負けとする＝健全側
            let r = if !self.hidden.is_empty() && self.revealed_own_hidden(s.state()) {
                Ok(false)
            } else {
                self.solve(s, nxt.as_deref(), depth + 1)
            };
            {
                let _prof = crate::prof::enter(crate::prof::Cat::LethalRollback);
                s.rollback();
            }
            let r = r?;
            if mine && r {
                res = true;
                self.best.insert(key.expect("mine"), mv);
                break;
            }
            if !mine && !r {
                res = false;
                break;
            }
        }
        if !mine && tried == 0 {
            res = false;
        }
        if let Some(k) = key {
            self.tt.insert(k, res);
        }
        Ok(res)
    }

    /// 勝ち筋の長さ（自分は記録した手・相手は並びの先頭で適用できる応手）。
    fn line_len(&self, state: &GameState, commit: Option<&str>, limit: usize) -> usize {
        let mut s = Session::new(state.clone());
        // 根の手が付与なら、その直後の自分の判断点は「付けたユニットで続ける」鍵（`commit`）で記録されている
        // （旧い形は `None` で引いて 1 手で止まっていた＝計測の欄だけの修正・打つ手は変わらない）
        let mut commit: Option<String> = commit.map(str::to_owned);
        let mut n = 0;
        while n < limit {
            let st = s.state();
            if st.winner.is_some() || st.turn_count != self.turn {
                break;
            }
            let Some((actor, action)) = crate::rules::pending::pending_actor_action(&mut s) else {
                break;
            };
            let mv = if actor == self.me {
                let k = self.key(&mut s, action, commit.as_deref());
                match self.best.get(&k) {
                    Some(m) => m.clone(),
                    None => break,
                }
            } else {
                let Ok(mut moves) = self.expand(&mut s, actor, action) else { break };
                moves.sort_by_key(opp_order);
                match moves.into_iter().find(|m| {
                    let mut c = Session::new(s.state().clone());
                    apply(&mut c, self.masters, actor, m).is_ok()
                }) {
                    Some(m) => m,
                    None => break,
                }
            };
            commit = if actor == self.me && act_of(&mv) == "ATTACH_DON" { mv_uuid(&mv).map(str::to_owned) } else { None };
            if apply(&mut s, self.masters, actor, &mv).is_err() {
                break;
            }
            n += 1;
        }
        n
    }

    /// 計測用（§7.30）: `state`（根の手を当てた後の盤面）から手順を 1 本辿って JSON の列にする。
    /// 自分の判断点は勝てた手（`best`）→ 無ければ `guide`（配り方 0 の勝ち筋の自分の手・同じ番目）→ 無ければ止まる。
    /// 相手の判断点は、各応手の後を読み直して**こちらが勝てない応手**（反証）が在ればそれ、無ければ並びの先頭。
    /// 各手に「自分の隠れた札（根で山札・裏向きのライフに在った自分の札）のうち、この判断点までに見えた札」を添える＝
    /// 配り方ごとに手順が分かれる点が、自分の隠れた札を見た直後かを読むため。
    fn walk(&mut self, state: &GameState, commit: Option<&str>, guide: &[Move], hidden: &[CardIdx], limit: usize) -> Vec<Value> {
        let mut s = Session::new(state.clone());
        let mut commit: Option<String> = commit.map(str::to_owned);
        let mut out = Vec::new();
        let mut my_n = 0usize;
        let mut seen_before: Vec<CardIdx> = Vec::new();
        for _ in 0..limit {
            let st = s.state();
            if let Some(w) = st.winner {
                out.push(serde_json::json!({"end": if w == self.me { "win" } else { "lose" }}));
                return out;
            }
            if st.turn_count != self.turn || st.turn_player != self.me {
                out.push(serde_json::json!({"end": "turn_end"}));
                return out;
            }
            let Some((actor, action)) = crate::rules::pending::pending_actor_action(&mut s) else {
                out.push(serde_json::json!({"end": "no_pending"}));
                return out;
            };
            let pend = if action == MAIN {
                Value::Null
            } else {
                crate::rules::pending::get_pending_request(&mut s, self.masters, false).unwrap_or(Value::Null)
            };
            // この判断点までに見えた自分の隠れた札（山札・裏向きのライフから出た札＋対話の候補に出た札）
            let seen = seen_hidden(s.state(), self.me, hidden, &pend);
            let newly: Vec<String> = seen
                .iter()
                .filter(|c| !seen_before.contains(c))
                .map(|c| self.masters.get(s.state().card(*c).master).card_id.clone())
                .collect();
            seen_before = seen;
            let mine = actor == self.me;
            let (mv, src) = if mine {
                let k = self.key(&mut s, action, commit.as_deref());
                match self.best.get(&k).cloned() {
                    Some(m) => (m, "best"),
                    None => match guide.get(my_n) {
                        Some(g) => (g.clone(), "guide"),
                        None => {
                            out.push(serde_json::json!({"actor": "me", "action": action, "stop": "no_win_move",
                                "message": pend.get("message").cloned().unwrap_or(Value::Null), "seen": newly}));
                            return out;
                        }
                    },
                }
            } else {
                let Ok(mut moves) = self.expand(&mut s, actor, action) else {
                    out.push(serde_json::json!({"actor": "opp", "action": action, "stop": "opp_unbounded"}));
                    return out;
                };
                moves.sort_by_key(opp_order);
                let mut pick: Option<(Move, &str)> = None;
                let mut hardest = 0u64;
                for m in moves {
                    s.begin();
                    if apply(&mut s, self.masters, actor, &m).is_err() {
                        s.rollback();
                        continue;
                    }
                    self.max_nodes = self.nodes + LETHAL_ROOT_NODES;
                    let before = self.nodes;
                    let r = self.solve(&mut s, None, 1);
                    s.rollback();
                    match r {
                        Ok(false) => {
                            pick = Some((m, "refute"));
                            break;
                        }
                        Ok(true) => {
                            // 全部の応手が負けなら、最も粘った応手（読んだ節点が最多）を手順に採る
                            let used = self.nodes - before;
                            if pick.is_none() || used > hardest {
                                hardest = used;
                                pick = Some((m, "any"));
                            }
                        }
                        Err(Abort) => {
                            pick = Some((m, "unknown"));
                            break;
                        }
                    }
                }
                match pick {
                    Some(p) => p,
                    None => {
                        out.push(serde_json::json!({"actor": "opp", "action": action, "stop": "no_reply"}));
                        return out;
                    }
                }
            };
            let mut row = describe_move(s.state(), self.masters, &mv);
            if let Some(o) = row.as_object_mut() {
                o.insert("actor".into(), Value::from(if mine { "me" } else { "opp" }));
                o.insert("action".into(), Value::from(action));
                o.insert("src".into(), Value::from(src));
                o.insert("seen".into(), Value::from(newly));
                if let Some(m) = pend.get("message") {
                    o.insert("message".into(), m.clone());
                }
            }
            if mine {
                my_n += 1;
            }
            commit = if mine && act_of(&mv) == "ATTACH_DON" { mv_uuid(&mv).map(str::to_owned) } else { None };
            if apply(&mut s, self.masters, actor, &mv).is_err() {
                if let Some(o) = row.as_object_mut() {
                    o.insert("stop".into(), Value::from("apply_failed"));
                }
                out.push(row);
                return out;
            }
            out.push(row);
        }
        out.push(serde_json::json!({"end": "limit"}));
        out
    }
}

/// 自分の隠れた札（山札＋裏向きのライフ・計測用）。
fn hidden_own(st: &GameState, me: Seat) -> Vec<CardIdx> {
    let p = st.player(me);
    p.deck.iter().copied().chain(p.life.iter().copied().filter(|c| !st.card(*c).is_face_up)).collect()
}

/// `hidden` のうち、今は山札・裏向きのライフに無い札（引いた・めくった・トラッシュした）と、対話の候補に出ている札。
fn seen_hidden(st: &GameState, me: Seat, hidden: &[CardIdx], pend: &Value) -> Vec<CardIdx> {
    let p = st.player(me);
    let listed: Vec<&str> = pend
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    hidden
        .iter()
        .copied()
        .filter(|c| {
            let in_deck = p.deck.contains(c);
            let in_life_down = p.life.contains(c) && !st.card(*c).is_face_up;
            !(in_deck || in_life_down) || listed.contains(&st.card(*c).uuid.as_str())
        })
        .collect()
}

/// 手の JSON に、手が指す札の `card_id` を添える（計測用）。
fn describe_move(st: &GameState, masters: &MasterTable, mv: &Move) -> Value {
    let j = mv.to_json();
    let mut uuids: Vec<String> = Vec::new();
    let mut push = |v: Option<&Value>| {
        if let Some(Value::String(u)) = v {
            uuids.push(u.clone());
        } else if let Some(Value::Array(a)) = v {
            uuids.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
        }
    };
    push(j.get("card_uuid"));
    if let Some(p) = j.get("payload") {
        push(p.get("uuid"));
        push(p.get("target_ids"));
        push(p.get("selected_uuids"));
    }
    let ids: Vec<String> = uuids
        .iter()
        .map(|u| {
            st.cards
                .iter()
                .find(|c| &c.uuid == u)
                .map(|c| masters.get(c.master).card_id.clone())
                .unwrap_or_else(|| "?".into())
        })
        .collect();
    serde_json::json!({"move": j, "cards": ids})
}

/// 相手の対話の全部の答え（§7.21 の健全性: 相手の応手は絞らない）。
fn opp_variants(action: &str, pend: &Value, with: &dyn Fn(&str, Value) -> Value) -> Result<Vec<Value>, Abort> {
    let mut out = Vec::new();
    if action.starts_with("CONFIRM") {
        out.push(with("accepted", Value::Bool(true)));
        out.push(with("accepted", Value::Bool(false)));
        return Ok(out);
    }
    if action == "CHOICE" {
        let n = pend.get("options").and_then(Value::as_array).map_or(0, |a| a.len());
        out.extend((0..n).map(|i| with("index", Value::from(i))));
        return Ok(out);
    }
    let uu: Vec<Value> = pend.get("selectable_uuids").and_then(Value::as_array).cloned().unwrap_or_default();
    if action == "ARRANGE_DECK" {
        // 並べ替えは数え切れない（2 枚以上なら証明しない）
        return if uu.len() <= 1 { Ok(out) } else { Err(Abort) };
    }
    let cons = pend.get("constraints");
    let max = cons.and_then(|c| c.get("max")).and_then(Value::as_u64).map_or(uu.len().min(1), |m| m as usize).min(uu.len());
    let mut min = cons.and_then(|c| c.get("min")).and_then(Value::as_u64).map_or(1, |m| m as usize).min(max);
    if pend.get("can_skip").and_then(Value::as_bool) == Some(true) {
        min = 0;
    }
    // 枚数が min..=max の全部の組（順は読まない）
    let n = uu.len();
    if n > 20 {
        return Err(Abort);
    }
    for mask in 0u32..(1u32 << n) {
        let k = mask.count_ones() as usize;
        if k < min || k > max {
            continue;
        }
        if out.len() >= OPP_SELECT_MAX {
            return Err(Abort);
        }
        let sel: Vec<Value> = (0..n).filter(|i| mask >> i & 1 == 1).map(|i| uu[i].clone()).collect();
        out.push(with("selected_uuids", Value::Array(sel)));
    }
    Ok(out)
}

fn in_play(st: &GameState) -> Vec<String> {
    let mut out = Vec::new();
    for seat in [Seat::P1, Seat::P2] {
        let p = st.player(seat);
        out.extend(p.leader.iter().chain(&p.field).chain(p.stage.iter()).map(|c| st.card(*c).uuid.clone()));
    }
    out
}

/// 盤面の鍵（監査の「hidden からドン!!の uuid を捨てたもの」と同じ中身）: 各ゾーンの並び・カードの可変の欄・
/// ドン!!は「どこに何枚」（どのドン!!かは区別しない）・ターンの欄・戦闘と誘発と継続効果（欄を直接混ぜる・
/// ふつうは空か小さい）。カードの番号（`CardIdx`）は複製で変わらない＝並びは番号で混ぜてよい。
fn state_hash(st: &GameState, h: &mut impl Hasher) {
    st.turn_count.hash(h);
    st.turn_player.hash(h);
    st.phase.hash(h);
    st.winner.hash(h);
    for seat in [Seat::P1, Seat::P2] {
        let p = st.player(seat);
        p.leader.hash(h);
        p.stage.hash(h);
        for z in [&p.hand, &p.field, &p.life, &p.trash, &p.deck, &p.temp_zone] {
            z.hash(h);
        }
        (p.don_deck.len(), p.don_active.len(), p.don_rested.len(), p.don_attached.len()).hash(h);
        p.negate_onplay_until.hash(h);
        p.restrictions.hash(h);
        p.granted_replacements.hash(h);
    }
    for c in &st.cards {
        // 可変の欄が全部既定（山札・トラッシュ・ライフの札のふつうの姿）なら 1 バイトで済ませる＝鍵の中身は同じ
        // （既定でない札だけ欄を全部混ぜる・印で区別する）。1 節点の鍵の時間の半分がこの輪だった（§7.25 の 4）。
        let plain = !c.is_rest
            && !c.is_newly_played
            && c.attached_don == 0
            && !c.is_face_up
            && !c.negated
            && !c.ability_disabled
            && c.power_buff == 0
            && c.cost_buff == 0
            && c.passive_power == 0
            && c.passive_power_override.is_none()
            && c.passive_counter == 0
            && c.base_power_override.is_none()
            && c.base_cost_override.is_none()
            && c.timed_power == 0
            && c.timed_cost == 0
            && c.current_keywords.is_empty()
            && c.flags.is_empty()
            && c.timed_flags.is_empty()
            && c.timed_keywords.is_empty()
            && c.ability_used_this_turn.is_empty();
        if plain {
            h.write_u8(0);
            continue;
        }
        h.write_u8(1);
        (c.is_rest, c.is_newly_played, c.attached_don, c.is_face_up, c.negated, c.ability_disabled).hash(h);
        (c.power_buff, c.cost_buff, c.passive_power, c.passive_power_override, c.passive_counter).hash(h);
        (c.base_power_override, c.base_cost_override, c.timed_power, c.timed_cost).hash(h);
        (&c.current_keywords, &c.flags, &c.timed_flags, &c.timed_keywords, &c.ability_used_this_turn).hash(h);
    }
    let mut don_kinds = [[0u16; 8]; 2];
    for d in &st.dons {
        let k = d.is_rest as usize | (d.is_frozen as usize) << 1 | (d.attached_to.is_some() as usize) << 2;
        don_kinds[(d.owner == Seat::P2) as usize][k] += 1;
    }
    don_kinds.hash(h);
    st.turn_events.hash(h);
    st.active_battle.hash(h);
    st.battle_triggers.hash(h);
    st.pending_triggers.hash(h);
    st.continuous.hash(h);
    st.pending_end_of_turn.hash(h);
    st.pending_extra_turn.hash(h);
}

/// 中断（対話）のスタックの鍵（旧 `format!("{:?}", interaction_stack)` と同じ中身を欄ごとに混ぜる・1 節点 14 µs → 1 µs 台）。
/// `Hash` を持たない欄（効果除去の置換の continuation）だけ `Debug` で混ぜる（在るときだけ・まれ）。
fn interaction_hash(st: &GameState, h: &mut impl Hasher) {
    st.interaction_stack.len().hash(h);
    for it in &st.interaction_stack {
        it.kind.hash(h);
        it.player.hash(h);
        it.message.hash(h);
        it.candidates.hash(h);
        it.selectable.hash(h);
        it.constraints.hash(h);
        it.can_skip.hash(h);
        it.source_card.hash(h);
        it.owner.hash(h);
        it.options.hash(h);
        (it.allow_position, it.allow_reorder).hash(h);
        it.intent.hash(h);
        match &it.continuation {
            None => h.write_u8(0),
            Some(c) => {
                h.write_u8(1);
                c.execution_stack.hash(h);
                c.context.hash(h);
                c.source_card.hash(h);
                c.node.hash(h);
                c.confirm_ability.hash(h);
                c.trigger_item.hash(h);
                c.arrange.hash(h);
                c.battle_ko.hash(h);
                c.rest_replace.hash(h);
                c.query.hash(h);
                if let Some(r) = &c.removal_replace {
                    format!("{r:?}").hash(h);
                }
            }
        }
    }
}

/// 実対局と同じ適用（`Game.apply_game_action`／`apply_battle_action`・対話はドレインしない）。
fn apply(s: &mut Session, masters: &MasterTable, actor: Seat, mv: &Move) -> Result<(), EngineError> {
    let _prof = crate::prof::enter(crate::prof::Cat::LethalApply);
    let at = mv.action_type().ok_or_else(|| EngineError::BadPayload("lethal: action_type が無い".into()))?;
    if mv.kind() == Some("battle") {
        crate::rules::actions::apply_battle_action(s, masters, actor, at, mv.card_uuid())
    } else {
        let payload = mv.payload().map(|p| p.to_json().into_owned()).unwrap_or_else(|| Value::Object(Default::default()));
        crate::rules::actions::apply_game_action(s, masters, actor, at, &payload)
    }
}

/// 1 つの配り方を読む（節点の上限は全部の配り方の合計＝この呼び出しの前までの使用量を引いた残り）。
fn solve_in(
    solver: &mut Solver,
    sess: &mut Session,
    used: u64,
    total: u64,
    first: Option<(&Move, Option<&str>)>,
    keep_child: &mut Option<GameState>,
    want_child: bool,
    keep_any: bool,
) -> Result<bool, Abort> {
    solver.max_nodes = solver.nodes + total.saturating_sub(used);
    let masters = solver.masters;
    let me = solver.me;
    sess.discard_events(|s| match first {
        None => solver.solve(s, None, 0),
        Some((mv, nxt)) => {
            s.begin();
            let r = if apply(s, masters, me, mv).is_err() {
                Ok(false)
            } else if !solver.hidden.is_empty() && solver.revealed_own_hidden(s.state()) {
                Ok(false)
            } else {
                let r = solver.solve(s, nxt, 1);
                if want_child && (matches!(r, Ok(true)) || keep_any) {
                    *keep_child = Some(s.state().clone());
                }
                r
            };
            s.rollback();
            r
        }
    })
}

/// 根の判定（§7.21 の 3）: 本当の盤面から自分の手の候補を作り、各候補について**全部の配り方**で
/// 勝てるかを読む（守りの強い配り方から）。最初に全部で勝てた手を返す。`max_nodes` は全部の合計。
pub fn solve_root(masters: &MasterTable, state: &GameState, me: Seat, deals: &[GameState], max_nodes: u64) -> LethalOut {
    solve_root_traced(masters, state, me, deals, max_nodes, None)
}

/// 計測用の指定（§7.30・`Game.lethal_json` の `trace`／`first`／`guide`）。
pub struct Trace<'a> {
    /// 根の候補をこの 1 手に限る（本当の配り方で「証明した手」を読み直すため）。
    pub first: Option<&'a Move>,
    /// 自分の判断点で勝てる手が無いときに辿る手（配り方 0 の勝ち筋の自分の手の列）。
    pub guide: &'a [Move],
}

/// [`solve_root`] の計測つき（`trace` が `Some` のとき: 候補を `first` に限り、全部の配り方を最後まで読み、
/// 配り方ごとの結果と手順（[`Solver::walk`]）を `lines` に返す）。`trace` が `None` なら [`solve_root`] と同一。
pub fn solve_root_traced(
    masters: &MasterTable,
    state: &GameState,
    me: Seat,
    deals: &[GameState],
    max_nodes: u64,
    trace: Option<Trace<'_>>,
) -> LethalOut {
    let _prof = crate::prof::enter_top(crate::prof::Cat::Search);
    let mut solvers: Vec<Solver> = deals.iter().map(|d| Solver::new(masters, d, me, max_nodes)).collect();
    let done = |status: &'static str, mv: Option<Move>, line_len: usize, solvers: &[Solver], lines: Option<Vec<Value>>| LethalOut {
        status,
        mv,
        line_len,
        deals: deals.len(),
        nodes: solvers.iter().map(|s| s.nodes).sum(),
        lines,
    };
    let mut real = Session::new(state.clone());
    let Some((actor, action)) = crate::rules::pending::pending_actor_action(&mut real) else {
        return done("no", None, 0, &solvers, None);
    };
    if actor != me || deals.is_empty() {
        return done("no", None, 0, &solvers, None);
    }
    let Ok(moves) = solvers[0].expand(&mut real, me, action) else {
        return done("unknown", None, 0, &solvers, None);
    };
    // 根の候補（上界の枝刈りは配り方ごとに子で効く＝根では使わない）
    let leader = solvers[0].opp_leader.clone();
    let mut moves: Vec<Move> = moves
        .into_iter()
        .filter(|m| !(act_of(m) == "ATTACK" && m.first_target_id() != Some(leader.as_str())))
        .collect();
    if action == MAIN {
        let att: Vec<String> =
            moves.iter().filter(|m| act_of(m) == "ATTACK").filter_map(|m| mv_uuid(m).map(str::to_owned)).collect();
        moves.retain(|m| act_of(m) != "ATTACH_DON" || mv_uuid(m).is_some_and(|u| att.iter().any(|a| a == u)));
    }
    moves.sort_by_key(my_order);
    if let Some(first) = trace.as_ref().and_then(|t| t.first) {
        let fj = first.to_json();
        moves.retain(|m| m.to_json() == fj);
    }
    let hidden = hidden_own(state, me);
    let mut sessions: Vec<Session> = deals.iter().map(|d| Session::new(d.clone())).collect();
    let mut unknown = false;
    'moves: for mv in moves {
        let nxt = if act_of(&mv) == "ATTACH_DON" { mv_uuid(&mv).map(str::to_owned) } else { None };
        let mut children: Vec<Option<GameState>> = vec![None; sessions.len()];
        let mut results: Vec<&'static str> = vec!["skip"; sessions.len()];
        let mut all = true;
        for i in 0..sessions.len() {
            let used: u64 = solvers.iter().map(|s| s.nodes).sum();
            let mut keep = None;
            let r = solve_in(&mut solvers[i], &mut sessions[i], used, max_nodes, Some((&mv, nxt.as_deref())), &mut keep, i == 0 || trace.is_some(), trace.is_some());
            children[i] = keep;
            match r {
                Ok(true) => results[i] = "win",
                Ok(false) => {
                    results[i] = "no";
                    all = false;
                    // 残りの配り方まで読むのは `first` を指定した読み直しだけ（候補を順に試す読みでは節点の予算を食い、
                    // 判定が変わってしまう）
                    if !trace.as_ref().is_some_and(|t| t.first.is_some()) {
                        continue 'moves;
                    }
                }
                Err(Abort) => {
                    results[i] = "unknown";
                    unknown = true;
                    all = false;
                    if !trace.as_ref().is_some_and(|t| t.first.is_some()) {
                        break 'moves;
                    }
                }
            }
        }
        let lines = trace.as_ref().map(|t| {
            let mut guide: Vec<Move> = t.guide.to_vec();
            let mut rows = Vec::new();
            for i in 0..sessions.len() {
                let line = match &children[i] {
                    Some(c) => solvers[i].walk(c, nxt.as_deref(), &guide, &hidden, 80),
                    None => Vec::new(),
                };
                // 配り方 0 の勝ち筋の自分の手を、後の配り方（本当の配り方）の案内に使う
                if i == 0 && guide.is_empty() {
                    guide = line
                        .iter()
                        .filter(|r| r.get("actor").and_then(Value::as_str) == Some("me"))
                        .filter_map(|r| r.get("move").cloned())
                        .map(Move::from_json)
                        .collect();
                }
                let ids = |cards: &[CardIdx]| -> Vec<String> {
                    cards.iter().map(|c| masters.get(deals[i].card(*c).master).card_id.clone()).collect()
                };
                let po = deals[i].player(me.other());
                let pm = deals[i].player(me);
                rows.push(serde_json::json!({"deal": i, "result": results[i], "nodes": solvers[i].nodes, "line": line,
                    "opp_hand": ids(&po.hand), "opp_life": ids(&po.life), "opp_deck_top": ids(&po.deck[..po.deck.len().min(5)]),
                    "my_deck_top": ids(&pm.deck[..pm.deck.len().min(5)]), "my_life": ids(&pm.life)}));
            }
            rows
        });
        if all {
            let line = 1 + children[0].as_ref().map_or(0, |c| solvers[0].line_len(c, nxt.as_deref(), 60));
            return done("win", Some(mv), line, &solvers, lines);
        }
        // `first` を指定した読み直しはその 1 手の結果（配り方ごとの手順つき）を返す。指定が無ければ次の候補へ
        if trace.as_ref().is_some_and(|t| t.first.is_some()) {
            return done(if unknown { "unknown" } else { "no" }, Some(mv), 0, &solvers, lines);
        }
        if unknown {
            break;
        }
    }
    done(if unknown { "unknown" } else { "no" }, None, 0, &solvers, None)
}

/// 盤面から（誰の番でも）このターンにこちらの勝ちが**全部の配り方で**証明できるか（§7.21 の 4 の応手の子）。
/// 戻り値は (証明できたか, 使った節点)。配り方は `state` から公平に作る（本当の配り方は読まない）。
pub fn prove(masters: &MasterTable, state: &GameState, me: Seat, seed: u64, max_nodes: u64) -> Result<(bool, u64), EngineError> {
    let deals = make_deals(state, masters, me, seed, LETHAL_RAND_DEALS)?;
    let mut solvers: Vec<Solver> = deals.iter().map(|d| Solver::new(masters, d, me, max_nodes)).collect();
    let mut ok = true;
    for (i, d) in deals.iter().enumerate() {
        let used: u64 = solvers.iter().map(|s| s.nodes).sum();
        let mut sess = Session::new(d.clone());
        let mut keep = None;
        match solve_in(&mut solvers[i], &mut sess, used, max_nodes, None, &mut keep, false, false) {
            Ok(true) => {}
            _ => {
                ok = false;
                break;
            }
        }
    }
    Ok((ok, solvers.iter().map(|s| s.nodes).sum()))
}

/// 根の判定の入口（§7.21 の 1・3）: 発動条件を満たせば配り方を作って根を判定する。満たさなければ `None`。
pub fn root_proof(
    masters: &MasterTable,
    state: &GameState,
    me: Seat,
    search_seed: Option<u64>,
    max_nodes: u64,
) -> Result<Option<LethalOut>, EngineError> {
    let mut s = Session::new(state.clone());
    if !applies(&mut s, masters, me) {
        return Ok(None);
    }
    let deals = make_deals(state, masters, me, lethal_seed(search_seed.unwrap_or(0)), LETHAL_RAND_DEALS)?;
    Ok(Some(solve_root(masters, state, me, &deals, max_nodes)))
}

#[cfg(test)]
#[path = "tests_lethal.rs"]
mod tests_lethal;
