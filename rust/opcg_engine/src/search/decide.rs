//! 1 手の決定（Python `core/cpu_learned.py::LearnedEngine.decide`／`_decide_inner` の移植）。
//!
//! | Rust | Python（正本） |
//! |---|---|
//! | [`decide`] | `LearnedEngine.decide`（枝予算を張って `_decide_inner`・抜けるとき外す） |
//! | [`commit_step`] | `_commit_step`（箱コミットの機械実行） |
//! | [`window_choice`] | `_window_choice`（窓の根畳み） |
//! | [`commit_window_continuation`] | `_commit_window_continuation` |
//! | [`commit_play_dialog`] | `_commit_play_dialog` |
//! | [`merge_root_stats`] | `_merge_root_stats`（等価手マージ） |
//! | [`move_sig`]／[`find_move`] | `learned/plan.py` の同名 |
//! | [`don_box_first_primitive`] | `cpu_ai.don_box_first_primitive` |
//! | [`residual_dig_move`]／[`residual_activate_move`]／[`residual_attach_move`] | 同名（腕 A／A2） |
//!
//! **エンジンをまたぐ状態**（Python は `LearnedEngine` のインスタンス辞書に持つ）は
//! [`DecideCarry`] で入出力する: 箱コミットの残り手順（`_commits`）と残り起動の
//! 付与待ちフラグ（`_resact_pending`）。ターン内 sticky 世界線（`_world_seeds`）は
//! **出目そのもの**が渡ってくる（[`crate::search::rng::SearchRng`]）ので Rust 側に状態は要らない。

use crate::journal::Session;
use crate::model::{CardIdx, GameState, MasterTable, Seat};
use crate::state::EngineError;
use serde_json::{json, Map, Value};

use super::mcts::{
    principal_variation_of, q_min_n_floor, select_index, Node, PvStep, RunOut, SelectRule, TreeMcts,
    C_PUCT, Q_MIN_FRAC, SERVE_SIMS,
};
use super::quiesce::{
    best_branch, in_battle, in_dialog, resolve_battle_inplace, resolved_branch_values, BoxBudget,
    Ctx, SearchState, TraceStep, Window, BOX_RESOLVE_DEPTH, QUIESCE_MAX_PLIES,
};
use super::rng::SearchRng;
use super::{apply, Move, SearchOptions};

/// 決定のつまみ（Python `config.py` の serve 既定 ＋ `LearnedEngine.__init__` の席別上書き）。
#[derive(Debug, Clone)]
pub struct DecideOptions {
    /// `SERVE_SIMS`
    pub sims: usize,
    /// `C_PUCT`
    pub c_puct: f64,
    /// `SERVE_DIRICHLET_EPS`
    pub dirichlet_eps: f64,
    /// `temp_turns`（0＝無効。turn <= これ のメイン窓で訪問分布からサンプルする）
    pub temp_turns: i32,
    /// §7.73 `gen-explore`: `temp_turns` より後のメイン窓でも、1 手ごとにこの確率で訪問分布から
    /// サンプルする（0＝無効＝乱数を引かない＝今までと 1 ビットも同じ）。
    pub temp_late: f64,
    /// `SERVE_BOX_COMMIT`
    pub box_commit: bool,
    /// `TREE_BOX_BATTLE`
    pub box_battle: bool,
    /// `TREE_BOX_DIALOG`
    pub box_dialog: bool,
    /// `SERVE_QUIESCE`
    pub quiesce: bool,
    /// `residual_dig`（腕 A）
    pub residual_dig: bool,
    /// `residual_activate`（腕 A2・"low"／"high"）
    pub residual_activate: Option<String>,
    /// 根で出す手の選び方（§20.5・既定＝訪問数最多＝今までの挙動）
    pub select_rule: SelectRule,
    /// `select_rule="q_min_n"` の訪問下限の割合（§20.5・既定 `Q_MIN_FRAC`＝sims/8）
    pub q_min_frac: f64,
    /// 根の事前分布の平坦化 `P^(1/t)`（§20.5・既定 1.0＝何もしない）
    pub root_prior_temp: f64,
    /// 1 回の decide で引く世界サンプルの本数（§20.7.1・既定 1＝今までどおり 1 本）。
    ///
    /// 2 以上なら K 本の世界で同じ sims の木を並列に回し、根の統計を束ねる
    /// （N は和・Q は N 重みの平均・P は世界 0 のもの）。[`DecideOptions::search_seed`] が
    /// 無いと世界 1 以降の乱数を作れないので 1 に落ちる（[`DecideOptions::effective_worlds`]）。
    pub worlds: usize,
    /// 世界の木を同時に回すスレッドの上限（§7.55・WP `rs-world-parallel`・既定 0）。
    ///
    /// 0＝世界 1 本に 1 スレッド（K 本が同時に走る＝今までどおり）。N≥1 なら同時に走る世界は
    /// 高々 N 本（世界 0 を回すこのスレッドを含む・1＝世界を 1 本ずつ順に回す）。どの値でも
    /// 世界 i は自分だけの新しいスレッドで回る（thread_local の状態を世界どうしで持ち越さない）
    /// ＝**打つ手と出力は 1 bit も変わらない**（変わるのは 1 手の時間とピークのコア数だけ）。
    pub world_threads: usize,
    /// 探索乱数の seed（`Game.decide` の `search_seed`）。世界 i の乱数は `seed + i`（§20.7.1）。
    pub search_seed: Option<u64>,
    /// 候補生成（`OPCGGame`）
    pub search: SearchOptions,
    /// `BOX_BRANCH_BUDGET`（`None`＝無制限）
    pub budget: Option<i64>,
    /// 全部見える探索（`search_efficiency_plan.md` §7.10・**測定専用**・既定 false）。
    ///
    /// true なら探索で世界を引く所（木・窓の根畳み・対話／準備箱のコミット）で引き直さず、
    /// 渡された本当の盤面（相手の手札・両者の山札の並び）をそのまま使う＝乱数も消費しない。
    /// 世界は 1 通りしか無いので `worlds > 1` は 1 として扱う（[`DecideOptions::effective_worlds`]）。
    /// 見えるのは探索の中だけ（選んだ手の後の進行は変えない）。ゲームとしては不正な読み。
    pub omniscient: bool,
    /// 世界サンプルの見える範囲（`search_efficiency_plan.md` §7.11・§7.15・既定 [`Reveal::Fair`]＝
    /// 自分に見えないものを混ぜ・知っている札は固定）。`Peek`／`OppHand` は測定用。`omniscient` とは
    /// 排他（`Fair` 以外との組は [`DecideOptions::validate`] がエラー）。
    pub reveal: Reveal,
    /// 木の中の相手に「こちらの手札」を見せない（`search_efficiency_plan.md` §7.14・**測定専用**・
    /// 既定 false）。true なら木の中で**相手が動く番のノードを展開するとき**（事前分布・葉の価値・
    /// 戦闘箱／対話箱の畳み）だけ、こちらの手札＋山札＋裏向きのライフを相手の視点で混ぜ直した複製で
    /// 評価する。合法手の生成と手の適用は本当の世界のまま（[`TreeMcts`](super::mcts::TreeMcts) の
    /// `opp_blind`）。混ぜ直しの乱数は世界ごとに `search_seed + i` から作る（[`opp_blind_seed`]）。
    pub opp_blind: bool,
    /// 詰みの証明（`search_efficiency_plan.md` §7.21・WP `rs-lethal-root`・既定 `Off`＝1 bit も変わらない）。
    /// `Root`: 自分のメインの判断点で、このターンに出せる攻撃の数が相手のライフ＋1 以上のとき、公平な配り方
    /// （[`super::lethal`]）の全部で勝てる最初の 1 手が在ればそれを返す（`kind="lethal"`・探索は回さない）。
    /// `RootReply`: 加えて、根の直後の相手の応手の節点で「その応手なら次にこちらの勝ちが証明できる」子を
    /// 相手から見て負けに固定する。配り方の乱数は決定の乱数とは別の系列・打ち切りは節点数＝決定的。
    pub lethal_proof: super::lethal::LethalProof,
    /// 根の判定の節点の上限（`lethal_nodes`・既定 [`super::lethal::LETHAL_ROOT_NODES`]）。
    pub lethal_nodes: u64,
    /// `RootReply` の応手の節点 1 つあたりの上限（`lethal_reply_nodes`）。
    pub lethal_reply_nodes: u64,
    /// `RootReply` の決定あたりの合計（`lethal_reply_total`・世界の本数で等分）。
    pub lethal_reply_total: u64,
    /// 旧い手の署名（`commit_sig_legacy`・WP `rs-commit-sig`・**測定専用**・既定 false＝修正後）。
    /// true なら箱の commit の積み・照合と複数世界の束ねで 2026-10-22 までの署名（`payload.uuid` だけ・
    /// 区別の欄なし＝戦闘の手の札が落ちる）を使う（[`move_sig_with`]）。本番アリーナの基準席用。
    pub commit_sig_legacy: bool,
    /// 守りの強い配り方の旧い形（`lethal_deal_legacy`・§7.33・WP `lethal-deal-cost`・測定用・既定 false）。
    /// 既定（false）はカウンターイベントを相手のアクティブのドン!!で払える枚数までだけ守りに数える。
    /// true は払えるかを見ずに積む旧い形（[`super::lethal::make_deals`]）。`lethal_proof=off` なら効かない。
    pub lethal_deal_legacy: bool,
    /// 見えていなかった札を選ぶ続きを捨てない旧い形（`commit_hidden_legacy`・§7.39・WP `rs-commit-hidden`・
    /// **測定専用**・既定 false＝修正後）。既定（false）は、箱の commit の続き（窓の根畳み・登場時の対話・
    /// 準備箱）を積むとき、**自分の判断点の対話の候補に、commit した時点で自分に見えていなかった札**
    /// （両者の山札・裏向きのライフ・相手の手札のうち `known_by` に自分の bit が無い札）が 1 枚でも在れば、
    /// その段から先を積まない（そこで続きが尽きて読み直す・[`trace_to_steps`]）。true は全部積む旧い形
    /// （引き直した世界の中身で選んだ札を本当の盤面で照合して打つ）。本番アリーナの基準席用。
    pub commit_hidden_legacy: bool,
    /// 対話・窓の続きを積む旧い形（`window_carry_legacy`・§7.42 の 1・WP `rs-window-read`・**測定専用**・既定 false）。
    /// 既定（false）は窓の根畳み（戦闘の応手・効果の対話・同時誘発の順）と登場／起動の後の対話が決めた
    /// 自分側の続き（効果の対話の段・戦闘の応手の段）を**積まず**、次の判断点で読み直す（窓の根畳み）。
    /// 付与 → 攻撃の箱（[`Step::Box`]）と準備箱の続きは今までどおり積む。true は 2026-10-23 までの形
    /// （続きは 1 段浅い読み〔`box_depth − 1`〕で決まる・`docs/reports/2026-10-23_commit_audit.md`）。
    pub window_carry_legacy: bool,
    /// 窓の根畳みで引く世界の本数（`window_worlds`・§7.42 の 2・既定 [`WINDOW_WORLDS`]）。
    /// 候補ごとの値を世界の平均で比べる（世界 0 は決定の乱数・世界 i≥1 は [`window_world_seed`]）。
    /// `search_seed` が無い経路・`omniscient` では 1。
    pub window_worlds: usize,
    /// 窓の世界を 1 本に戻す旧い形（`window_worlds_legacy`・§7.42 の 2・**測定専用**・既定 false）。
    pub window_worlds_legacy: bool,
    /// ライフが少ない戦闘の応手を静的な値で比べる旧い形（`window_tree_legacy`・§7.42 の 3・**測定専用**・既定 false）。
    /// 既定（false）は自分のライフが [`DecideOptions::window_tree_life`] 以下の戦闘の窓（ブロッカー・カウンター）で
    /// 候補が 2 つ以上なら、各候補を戦闘の終わりまで畳んだ盤面から短い木（`sims ÷ 候補の数`）を回し、
    /// その根の値で比べる（[`tree_branch_values`]）。
    pub window_tree_legacy: bool,
    /// 短い木で読む戦闘の窓の自分のライフの上限（`window_tree_life`・既定 1）。
    pub window_tree_life: i64,
    /// 短い木で読む窓の世界の本数の上限（`window_tree_worlds`・既定 [`WINDOW_TREE_WORLDS`]）。
    /// 木の費用は「世界の本数 × sims」なので、静的な値の窓の K とは別に抑える（実際は `min(K, これ)`）。
    pub window_tree_worlds: usize,
    /// 守りの窓で相手から見た詰みの証明を使う（`defense_proof`・§7.51・WP `rs-defense-proof`・**測定用**・既定 false）。
    /// true なら相手の手番の窓（戦闘の応手・効果の対話）で、自分のライフが [`DecideOptions::defense_proof_life`] 以下かつ
    /// 候補の値が平ら（差 ≤ [`DecideOptions::defense_proof_flat`]）か張り付き（全部 |値| ≥ [`DecideOptions::defense_proof_sat`]）の
    /// ときだけ、値の高い候補から順に、当てた後の盤面で**相手の側から**詰みの証明（[`super::lethal::prove_each`]・
    /// 配り方は [`DecideOptions::defense_proof_view`] の視点で公平に 8 通り）を回し、証明されなかった最初の候補を選ぶ
    /// （＝どの配り方でも相手が勝つ候補を外して値で選ぶ）。全部が証明されたら今の選び方のまま（[`defense_pick`]）。
    pub defense_proof: bool,
    /// 守りの証明の対象にする自分のライフの上限（`defense_proof_life`・既定 2）。
    pub defense_proof_life: i64,
    /// 「値が平ら」の差の上限（`defense_proof_flat`・既定 0.02・`2026-10-24_weak_points.md` §1.3）。
    pub defense_proof_flat: f64,
    /// 「値が張り付き」の絶対値の下限（`defense_proof_sat`・既定 0.85・同）。
    pub defense_proof_sat: f64,
    /// 候補 1 つの証明の節点の上限（`defense_proof_nodes`・既定 [`DEFENSE_PROOF_NODES`]）。
    pub defense_proof_nodes: u64,
    /// 1 つの窓の証明の節点の合計の上限（`defense_proof_total`・既定 [`DEFENSE_PROOF_TOTAL`]）。
    pub defense_proof_total: u64,
    /// 配り方を誰の視点で作るか（`defense_proof_view`・既定 [`DefenseView::Mine`]）。
    pub defense_proof_view: DefenseView,
}

/// 守りの証明の配り方の視点（§7.51）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefenseView {
    /// 自分の視点（既定）: 自分に見えない札（相手の手札・山札・裏向きのライフ・自分の山札・裏向きのライフ）を配り直し、
    /// 自分の手札は本物のまま（知っている札で読む）。攻め手（相手）は配り方ごとに全部見えている読み。
    #[default]
    Mine,
    /// 相手の視点（指示書 §7.51 の字面）: 相手の知らない札（自分の手札・山札・裏向きのライフ）を配り直し、
    /// 相手の札は窓の世界 0 で引いたまま（[`super::lethal::prove`] と同じ配り方）。
    Opp,
}

impl DefenseView {
    pub fn from_name(s: &str) -> Option<DefenseView> {
        match s {
            "mine" => Some(DefenseView::Mine),
            "opp" => Some(DefenseView::Opp),
            _ => None,
        }
    }
}

/// 守りの証明の候補 1 つの節点の上限の既定（§7.51・根の証明 [`super::lethal::LETHAL_ROOT_NODES`] と同じ桁）。
pub const DEFENSE_PROOF_NODES: u64 = 20_000;
/// 守りの証明の窓 1 つの節点の合計の上限の既定（候補 4 つぶん）。
pub const DEFENSE_PROOF_TOTAL: u64 = 80_000;

/// 窓の根畳みの世界の本数の既定（§7.42 の 2・WP `rs-window-read` で揺れと費用から決めた）。
/// blunder-audit の窓の 226 点で、探索の乱数だけを替えると手が変わる率は K=1/2/4/8/16/32/64 で
/// 19.5/14.6/11.5/5.8/4.4/3.1/1.8%。静的な値の窓は 1 本あたり数 ms なので 5% を切る最初の K を採った。
pub const WINDOW_WORLDS: usize = 16;
/// 短い木で読む窓の世界の本数の上限の既定（§7.42 の 3・費用は「この本数 × sims」）。
pub const WINDOW_TREE_WORLDS: usize = 2;

/// 窓の根畳みの世界 `i`（≥1）の乱数の seed（世界 0 は決定の乱数そのもの＝1 本なら今までと同じ）。
/// 木の世界 `search_seed + i` とは別の系列になるよう定数で混ぜる。
pub(crate) fn window_world_seed(search_seed: u64, i: usize) -> u64 {
    (search_seed ^ 0x57A1_D0C5_0FF5_EED5u64).wrapping_add(i as u64)
}

/// `opp_blind`（§7.14）の混ぜ直しの乱数の seed（世界 `i` の探索乱数 `search_seed + i` と同じ値から、
/// 別の系列になるよう定数で混ぜる＝世界の引き直し・Dirichlet の出目とは重ならない）。
pub(crate) fn opp_blind_seed(world_seed: u64) -> u64 {
    world_seed ^ 0x0B11_D5EE_D0FF_5EEDu64
}

/// 探索の世界サンプルで何を混ぜ直すか（§7.11・§7.15・opts JSON の鍵 `reveal`）。
///
/// **既定は `Fair`**（2026-10-21 ユーザ決定「全部公平の方が良さそう」・WP `rs-fair`）＝自分に見えない
/// ものを全部混ぜ、自分が知っている札（[`CardInstance::known_by`](crate::model::CardInstance::known_by)）
/// はその位置に固定する。`Peek` は 2026-10-21 までの既定（自分の山札の並びと裏向きのライフを本物のまま
/// 読む）で、`OppHand` と同じく**測定用**に残す。
///
/// | 値 | 相手の手札 | 相手の山札 | 相手の裏向きライフ | 自分の山札 | 自分の裏向きライフ | 知っている札 |
/// |---|---|---|---|---|---|---|
/// | `Fair`（鍵なし・`"fair"`） | 混ぜる（相手の 3 つで 1 つの山） | 混ぜる | 混ぜる | 混ぜる（自分の 2 つで 1 つの山） | 混ぜる | 固定 |
/// | `Peek`（`"peek"`） | 混ぜる（山札と 1 つの山） | 混ぜる | 本当のまま | 本当のまま | 本当のまま | 読まない |
/// | `OppHand`（`"opp_hand"`） | 本当のまま | 混ぜる | 本当のまま | 本当のまま | 本当のまま | 読まない |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reveal {
    #[default]
    Fair,
    Peek,
    OppHand,
}

impl Reveal {
    /// opts JSON の値から（知らない値は `None`＝呼び出し側がエラーにする）。
    pub fn from_name(s: &str) -> Option<Reveal> {
        match s {
            "fair" => Some(Reveal::Fair),
            "peek" => Some(Reveal::Peek),
            "opp_hand" => Some(Reveal::OppHand),
            _ => None,
        }
    }
}

/// 世界の作り方（`omniscient` と `reveal` の組・[`sample_world`] が読む）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorldView {
    omniscient: bool,
    reveal: Reveal,
}

impl DecideOptions {
    /// 実際に回す世界の本数（§20.7.1）。`search_seed` が無ければ世界 1 以降の乱数を作れない
    /// ＝1 本に落ちる（記録した出目で回すオラクル経路がこれに当たる）。
    /// 組み合わせの検査（§7.11: `omniscient` と `reveal` は排他。既定の `Fair` は「指定なし」と同じ扱い）。
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.omniscient && self.reveal != Reveal::Fair {
            return Err(EngineError::BadPayload(
                "decide: omniscient と reveal は同時に指定できない".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn world_view(&self) -> WorldView {
        WorldView { omniscient: self.omniscient, reveal: self.reveal }
    }

    /// 窓の根畳みで実際に引く世界の本数（§7.42 の 2）。旧い旗・`search_seed` 無し・`omniscient` は 1。
    pub fn effective_window_worlds(&self) -> usize {
        if self.window_worlds_legacy || self.search_seed.is_none() || self.omniscient {
            return 1;
        }
        self.window_worlds.max(1)
    }

    pub fn effective_worlds(&self) -> usize {
        if self.search_seed.is_none() || self.omniscient {
            return 1;
        }
        self.worlds.max(1)
    }
}

impl Default for DecideOptions {
    fn default() -> Self {
        DecideOptions {
            sims: SERVE_SIMS,
            c_puct: C_PUCT,
            dirichlet_eps: 0.0,
            temp_turns: 0,
            temp_late: 0.0,
            box_commit: true,
            box_battle: true,
            box_dialog: true,
            quiesce: true,
            residual_dig: false,
            residual_activate: None,
            select_rule: SelectRule::Visits,
            q_min_frac: Q_MIN_FRAC,
            root_prior_temp: 1.0,
            worlds: 1,
            world_threads: 0,
            search_seed: None,
            search: SearchOptions::default(),
            budget: Some(super::quiesce::BOX_BRANCH_BUDGET),
            omniscient: false,
            reveal: Reveal::Fair,
            opp_blind: false,
            lethal_proof: super::lethal::LethalProof::Off,
            lethal_nodes: super::lethal::LETHAL_ROOT_NODES,
            lethal_reply_nodes: super::lethal::LETHAL_REPLY_NODES,
            lethal_reply_total: super::lethal::LETHAL_REPLY_TOTAL,
            commit_sig_legacy: false,
            lethal_deal_legacy: false,
            commit_hidden_legacy: false,
            window_carry_legacy: false,
            window_worlds: WINDOW_WORLDS,
            window_worlds_legacy: false,
            window_tree_legacy: false,
            window_tree_life: 1,
            window_tree_worlds: WINDOW_TREE_WORLDS,
            defense_proof: false,
            defense_proof_life: 2,
            defense_proof_flat: 0.02,
            defense_proof_sat: 0.85,
            defense_proof_nodes: DEFENSE_PROOF_NODES,
            defense_proof_total: DEFENSE_PROOF_TOTAL,
            defense_proof_view: DefenseView::Mine,
        }
    }
}

/// 探索で使う世界を 1 つ作る（§7.10）。`omniscient` なら本当の盤面のクローン（乱数を引かない）、
/// そうでなければ `reveal`（§7.11・§7.15）の見える範囲に合わせて混ぜ直す（[`Reveal`] の表・既定は
/// `Fair`＝[`determinize_fair_with`](super::determinize::determinize_fair_with)）。
fn sample_world(
    state: &GameState,
    name: Seat,
    view: WorldView,
    rng: &mut dyn SearchRng,
) -> Result<GameState, EngineError> {
    if view.omniscient {
        return Ok(state.clone());
    }
    match view.reveal {
        Reveal::Fair => super::determinize::determinize_fair_with(state, name, rng),
        Reveal::Peek => super::determinize::determinize_with(state, name, rng),
        Reveal::OppHand => super::determinize::determinize_opp_deck_with(state, name, rng),
    }
}

/// 箱コミットの 1 手順（Python の `move_sig` タプル／`("__box__", sig, 残り回数)`）。
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// 素の手（`move_sig`）
    Sig(Value),
    /// DON_BOX のカウントダウン形（sig は don_k 非含有＝残回数はこちらが持つ）
    Box { sig: Value, left: i64 },
}

impl Step {
    pub fn to_json(&self) -> Value {
        match self {
            Step::Sig(sig) => json!({"kind": "sig", "sig": sig}),
            Step::Box { sig, left } => json!({"kind": "box", "sig": sig, "left": left}),
        }
    }

    pub fn from_json(v: &Value) -> Result<Step, EngineError> {
        let bad = || EngineError::BadPayload(format!("decide: 手順の形が違う（{v}）"));
        let sig = v.get("sig").cloned().ok_or_else(bad)?;
        match v.get("kind").and_then(Value::as_str) {
            Some("sig") => Ok(Step::Sig(sig)),
            Some("box") => Ok(Step::Box {
                sig,
                left: v.get("left").and_then(Value::as_i64).ok_or_else(bad)?,
            }),
            _ => Err(bad()),
        }
    }
}

/// decide をまたいで持ち越す状態（Python の `_commits`／`_resact_pending`）。
#[derive(Debug, Clone, Default)]
pub struct DecideCarry {
    /// 現在の (turn, seat) の残り手順。空＝コミット無し。
    pub commit: Vec<Step>,
    /// 直前の decide が残り起動を返した（続く付与対話を方針で解く）
    pub resact_pending: bool,
}

/// 決定の結果。
#[derive(Debug, Clone)]
pub struct DecideOut {
    pub mv: Option<Move>,
    /// 棋譜ダンプ用の**箱レベル**の `move_sig`（Python `record["sig"]`）。
    ///
    /// [`DecideOut::mv`] は「実対局へ出す手」＝配分箱は先頭原始手へ畳んだ後・残ドン掘り／
    /// 残り起動で差し替えた後なので、**決定の同一性**（訪問分布の候補と突き合わせる鍵）とは
    /// 別物になる。Python は `record` をその 2 つの前で採る＝ここも同じ位置で採る。
    pub sig: Option<Value>,
    /// 同じ位置の `don_k`（Python `record["k"]`。`move_sig` は don_k を含まないので並記が要る）。
    /// 窓・コミットでは `None`（Python も `record["k"]` を書かない）。
    pub k: Option<f64>,
    /// "main"／"window"／"commit"
    pub kind: &'static str,
    pub legal: Vec<Move>,
    pub n: Vec<f64>,
    pub q: Vec<f64>,
    /// 木のときだけ（窓・コミットは `None`）
    pub p: Option<Vec<f64>>,
    /// 等価手マージ後の候補（`record["groups"]` と同じ集計）
    pub groups: Vec<Group>,
    /// 選択規則の束ね（§20.7.8 の 4・準備箱の枝を card_id で 1 グループに）。箱が無ければ空。
    pub select_groups: Vec<SelectGroup>,
    pub carry: DecideCarry,
    pub budget_used: i64,
    pub budget_exhausted: u64,
    /// PV（主変化・§20.4）: kind=main のときだけ木から辿る（window／commit は空）。
    pub pv: Vec<PvStep>,
    /// 回した世界の本数（§20.7.1・1＝今までどおりの単一世界）。
    pub worlds: usize,
    /// 世界ごとの根の統計（`worlds>1` のときだけ・並びは世界 0 の `legal`）。
    pub per_world: Vec<WorldStats>,
    /// PV とコミットの継続を採った世界（`worlds>1` のときだけ・§20.7.1）。
    pub world_used: Option<usize>,
    /// 根の詰み判定の結果（§7.21・発動条件を満たして判定したときだけ）。`kind="lethal"` なら証明した。
    pub lethal: Option<super::lethal::LethalOut>,
    /// `RootReply` の木の中の証明の実績（全世界の和・応手の節点を判定したときだけ）。
    pub lethal_reply: Option<super::lethal::ReplyProof>,
    /// 窓の根畳みの読み方（§7.42・`kind="window"` で世界 K 本か短い木のときだけ）: (世界の本数, 短い木か)。
    pub window_read: Option<(usize, bool)>,
    /// 守りの証明の実績（§7.51・`defense_proof` で対象の窓だったときだけ・[`defense_pick`]）。
    pub defense_proof: Option<Value>,
    /// 温度サンプルの印（§7.73・サンプルを引いたときだけ）: (1＝序盤の温度／2＝後半の低確率の温度,
    /// 引いた手が貪欲に選ぶ手と違ったか)。
    pub temp_pick: Option<(u8, bool)>,
}

/// 1 つの世界の根の統計（§20.7.1・思考ログで「世界によって答えが割れたか」を読む）。
#[derive(Debug, Clone)]
pub struct WorldStats {
    /// その世界の乱数 seed（`search_seed + i`）
    pub seed: u64,
    /// 根の訪問数（**世界 0 の `legal` の並び**へ写したもの）
    pub n: Vec<f64>,
    /// 根の行動価値（同じ並び）
    pub q: Vec<f64>,
    /// その世界だけで選ぶならどの手か（`select_rule` を適用・世界 0 の `legal` の添字）
    pub best: Option<usize>,
    /// 世界 0 の `legal` に対応づけられなかった手の数（0＝並びが一致）
    pub unmapped: usize,
    /// 根の事前分布 P が世界 0 と違ったか（ネットの出力は世界に依らないはず＝通常 false）
    pub p_differs: bool,
}

/// `_merge_root_stats` の 1 グループ。
#[derive(Debug, Clone)]
pub struct Group {
    pub rep: usize,
    pub idxs: Vec<usize>,
    pub n: f64,
    pub q: f64,
}

/// 選択規則の束ね（§20.7.8 の 4）の 1 グループ。
///
/// 等価手マージ（[`merge_root_stats`]）の**上に重ねる**: 同じ card_id の `SETUP_BOX` の枝
/// （素の準備の手が候補に残っていればそれも）を 1 グループにし、それ以外は 1 手 1 グループ。
#[derive(Debug, Clone)]
pub struct SelectGroup {
    /// 束ねの鍵（`["setup", card_id]`／`["move", 等価グループの添字]`）
    pub key: Value,
    /// 束ねた等価グループ（[`merge_root_stats`] の並びの添字）
    pub idxs: Vec<usize>,
    /// 束ねた訪問数の和
    pub n: f64,
    /// 代表の手（`legal` の添字）＝グループ内で N が最大の枝
    pub rep: usize,
    /// 代表の Q
    pub q: f64,
}

// --- 手のキー（plan.move_sig／cpu_ai._move_equiv_key）-----------------------------

/// 手の署名（`[action_type, 札, target_ids, selected_uuids, accepted]`＋必要なら区別の欄）。
///
/// 箱の commit（窓の根畳み・登場時の対話・準備箱が決めた手順の続き）は各段をこの署名で合法手と照合し、
/// [`find_move`] は**署名が一致する最初の合法手**を取る＝署名に落ちた区別は再生で失われる
/// （WP `rs-commit-sig`・`docs/reports/2026-10-23_commit_sig.md`）。そのため:
/// - 札の欄は `payload.uuid`、無ければ最上位の `card_uuid`（戦闘の手 `SELECT_COUNTER`／`SELECT_BLOCKER`
///   は札を `card_uuid` に持つ・[`describe_move`] と同じ形）。
/// - 6 つ目の欄（区別の欄・dict）は、5 つでは割れない手どうしのときだけ付く:
///   `position`（並べ替えの上下・既定の `"BOTTOM"` 以外のとき）・`selects`（準備箱の対象選択が 2 段以上のとき・
///   段の区切りは `selected_uuids` の平らな並びでは落ちる）・`index`（選択肢の番号・0 以外のとき）。
///   付かない手の署名は今までと同じ 5 つ。
///
/// `DON_BOX` の `don_k` は入れない（残り回数は [`Step::Box`] が持つ）。
pub fn move_sig(mv: &Move) -> Value {
    move_sig_with(mv, false)
}

/// [`move_sig`] の、旧い署名（`legacy`＝`payload.uuid` だけ・区別の欄なし＝2026-10-22 までの形）も
/// 出せる版。旧い署名は**測定専用**（`DecideOptions::commit_sig_legacy`）。
pub fn move_sig_with(mv: &Move, legacy: bool) -> Value {
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let at = match mv.get("action_type") {
        Some(v) if !v.is_null() => v.clone(),
        _ => p.get("action_type").cloned().unwrap_or(Value::Null),
    };
    let card = match p.get("uuid") {
        Some(v) if legacy || !v.is_null() => v.clone(),
        _ if legacy => Value::Null,
        _ => mv.get("card_uuid").cloned().unwrap_or(Value::Null),
    };
    let mut sig = vec![
        at,
        card,
        p.get("target_ids").cloned().unwrap_or(Value::Array(vec![])),
        p.get("selected_uuids")
            .cloned()
            .unwrap_or(Value::Array(vec![])),
        p.get("accepted").cloned().unwrap_or(Value::Null),
    ];
    if !legacy {
        let mut ext = Map::new();
        if let Some(pos) = p.get("position").filter(|v| v.as_str() != Some("BOTTOM") && !v.is_null()) {
            ext.insert("position".into(), pos.clone());
        }
        if let Some(sel) = p.get("selects").filter(|v| v.as_array().is_some_and(|a| a.len() >= 2)) {
            ext.insert("selects".into(), sel.clone());
        }
        // 選択肢（CHOICE）の番号。探索の候補は既定の 0 だけだが、詰みの判定器の答えは番号で割れる。
        if let Some(i) = p.get("index").filter(|v| v.as_i64().is_some_and(|i| i != 0)) {
            ext.insert("index".into(), i.clone());
        }
        if !ext.is_empty() {
            sig.push(Value::Object(ext));
        }
    }
    Value::Array(sig)
}

/// Python `plan._find_move`。
pub fn find_move<'a>(legal: &'a [Move], sig: &Value) -> Option<&'a Move> {
    find_move_with(legal, sig, false)
}

/// [`find_move`] の、署名の形（[`move_sig_with`] の `legacy`）を選べる版。
pub fn find_move_with<'a>(legal: &'a [Move], sig: &Value, legacy: bool) -> Option<&'a Move> {
    legal.iter().find(|mv| move_sig_with(mv, legacy) == *sig)
}

/// Python `cpu_ai._find_card`（トレース記述用の全ゾーン横断・p1 → p2 の順）。
fn find_card(state: &GameState, uuid: &str) -> Option<CardIdx> {
    for seat in [Seat::P1, Seat::P2] {
        let p = state.player(seat);
        let zones = p
            .field
            .iter()
            .chain(p.hand.iter())
            .chain(p.life.iter())
            .chain(p.deck.iter())
            .chain(p.trash.iter())
            .chain(p.temp_zone.iter())
            .chain(p.leader.iter())
            .chain(p.stage.iter());
        for c in zones {
            if state.card(*c).uuid == uuid {
                return Some(*c);
            }
        }
    }
    None
}

/// Python `cpu_ai._card_label`（card_id → name → uuid）。
fn card_label(state: &GameState, masters: &MasterTable, uuid: Option<&str>) -> Value {
    let Some(uuid) = uuid.filter(|u| !u.is_empty()) else {
        return Value::Null;
    };
    match find_card(state, uuid) {
        None => Value::from(uuid),
        Some(c) => {
            let m = masters.get(state.card(c).master);
            if !m.card_id.is_empty() {
                Value::from(m.card_id.clone())
            } else if !m.name.is_empty() {
                Value::from(m.name.clone())
            } else {
                Value::from(uuid)
            }
        }
    }
}

/// Python `cpu_ai._describe_move`（手を card_id 基準の人間可読 dict へ・uuid 非依存＝再現性あり）。
///
/// 対戦 API の思考トレース（`services/replay._replay_record_action`）が録画に書く形。
/// 空の欄は**入れない**（Python も `if label:`／`if tids:` で出し分ける）＝旧録画と同じキー集合。
/// `selected_slots`（同名複製の曖昧性解消）は `pending` の `selectable_uuids` 内の位置で、
/// `RESOLVE_EFFECT_SELECTION` のときだけ載る。
pub fn describe_move(
    state: &GameState,
    masters: &MasterTable,
    mv: &Move,
    pending: Option<&Value>,
) -> Value {
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let extra = p.get("extra").unwrap_or(&null);
    let at = mv.get("action_type").cloned().unwrap_or(Value::Null);
    let mut d = Map::new();
    d.insert("action_type".into(), at.clone());
    if at.as_str() == Some("DON_BOX") {
        if let Some(k) = p.get("don_k").and_then(Value::as_f64) {
            d.insert("don_k".into(), Value::from(k as i64));
        }
    }
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    let label = card_label(state, masters, uuid);
    if !label.is_null() {
        d.insert("card".into(), label);
    }
    let labels = |a: &Vec<Value>| -> Value {
        Value::Array(
            a.iter()
                .map(|v| card_label(state, masters, v.as_str()))
                .collect(),
        )
    };
    if let Some(tids) = p
        .get("target_ids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
    {
        d.insert("targets".into(), labels(tids));
    }
    let sel = p
        .get("selected_uuids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .or_else(|| {
            extra
                .get("selected_uuids")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty())
        });
    if let Some(sel) = sel {
        d.insert("selected".into(), labels(sel));
        if at.as_str() == Some("RESOLVE_EFFECT_SELECTION") {
            if let Some(su) = pending
                .and_then(|pr| pr.get("selectable_uuids"))
                .and_then(Value::as_array)
            {
                let slots: Vec<i64> = sel
                    .iter()
                    .map(|u| {
                        su.iter()
                            .position(|x| x == u)
                            .map(|i| i as i64)
                            .unwrap_or(-1)
                    })
                    .collect();
                if !slots.is_empty() && slots.iter().all(|s| *s >= 0) {
                    d.insert("selected_slots".into(), Value::from(slots));
                }
            }
        }
    }
    for key in ["index", "position"] {
        let v = match p.get(key) {
            Some(v) if !v.is_null() => Some(v.clone()),
            _ => extra.get(key).filter(|v| !v.is_null()).cloned(),
        };
        if let Some(v) = v {
            d.insert(key.into(), v);
        }
    }
    // 任意効果の「見送り」だけを明示する（accept 側は既定＝旧録画と同キーで照合できる）。
    let acc = match p.get("accepted") {
        Some(v) if !v.is_null() => Some(v.clone()),
        _ => extra.get("accepted").filter(|v| !v.is_null()).cloned(),
    };
    if acc == Some(Value::Bool(false)) {
        d.insert("accepted".into(), Value::Bool(false));
    }
    Value::Object(d)
}

/// Python `cpu_ai._move_equiv_key`（`_describe_move` と同じ card_id 基準の同一視）。
///
/// 返り値は `[action_type, card, targets, selected, index, position, accepted, don_k]`。
/// `_describe_move` の `selected_slots`（同名複製の曖昧性解消）は等価キーが読まないので作らない。
pub fn move_equiv_key(state: &GameState, masters: &MasterTable, mv: &Move) -> Value {
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let extra = p.get("extra").unwrap_or(&null);
    let at = mv.get("action_type").cloned().unwrap_or(Value::Null);
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    let card = card_label(state, masters, uuid);
    let labels = |arr: Option<&Vec<Value>>| -> Value {
        match arr {
            None => Value::Array(vec![]),
            Some(a) => Value::Array(
                a.iter()
                    .map(|v| card_label(state, masters, v.as_str()))
                    .collect(),
            ),
        }
    };
    let tids = p.get("target_ids").and_then(Value::as_array);
    let targets = labels(tids.filter(|a| !a.is_empty()));
    let sel = p
        .get("selected_uuids")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .or_else(|| {
            extra
                .get("selected_uuids")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty())
        });
    let selected = labels(sel);
    let pick = |key: &str| -> Value {
        match p.get(key) {
            Some(v) if !v.is_null() => v.clone(),
            _ => match extra.get(key) {
                Some(v) if !v.is_null() => v.clone(),
                _ => Value::Null,
            },
        }
    };
    let accepted = match pick("accepted") {
        Value::Bool(false) => Value::Bool(false),
        _ => Value::Null, // Python は accepted=False のときだけ記述に載せる
    };
    let don_k = if at.as_str() == Some("DON_BOX") {
        match p.get("don_k") {
            Some(v) if !v.is_null() => Value::from(v.as_f64().unwrap_or(0.0) as i64),
            _ => Value::Null,
        }
    } else {
        Value::Null
    };
    Value::Array(vec![
        at,
        card,
        targets,
        selected,
        pick("index"),
        pick("position"),
        accepted,
        don_k,
    ])
}

/// Python `cpu_learned._merge_root_stats`（等価キーで訪問数を合算し n 降順・安定ソート）。
pub fn merge_root_stats(
    state: &GameState,
    masters: &MasterTable,
    legal: &[Move],
    n: &[f64],
    q: &[f64],
) -> Vec<Group> {
    let mut order: Vec<Value> = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, mv) in legal.iter().enumerate() {
        let k = move_equiv_key(state, masters, mv);
        match order.iter().position(|o| *o == k) {
            Some(g) => groups[g].push(i),
            None => {
                order.push(k);
                groups.push(vec![i]);
            }
        }
    }
    let mut out: Vec<Group> = groups
        .into_iter()
        .map(|idxs| {
            let total: f64 = idxs.iter().map(|i| n[*i]).sum();
            let qq = if total > 0.0 {
                idxs.iter().map(|i| n[*i] * q[*i]).sum::<f64>() / total
            } else {
                0.0
            };
            Group {
                rep: idxs[0],
                idxs,
                n: total,
                q: qq,
            }
        })
        .collect();
    // Python の `sort(key=lambda g: -g["n"])` は安定＝同数は列挙順のまま
    out.sort_by(|a, b| {
        b.n.partial_cmp(&a.n)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// 準備の手の action_type（素の手として候補に残りうるもの）。
const SETUP_ACTIONS: [&str; 2] = ["PLAY", "ACTIVATE_MAIN"];

/// 手の主体の card_id ラベル（`move_equiv_key` の 2 欄目と同じ引き方）。
fn move_card_label(state: &GameState, masters: &MasterTable, mv: &Move) -> Value {
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let uuid = p
        .get("uuid")
        .and_then(Value::as_str)
        .or_else(|| mv.get("card_uuid").and_then(Value::as_str));
    card_label(state, masters, uuid)
}

/// 選択規則の束ね（§20.7.8 の 4）。**箱が 1 つも無ければ空**を返す（＝既定は無影響）。
///
/// 同じ card_id の `SETUP_BOX` の枝（素の準備の手が候補に残っていればそれも）を 1 グループに
/// 束ね、それ以外は 1 手 1 グループ（等価手マージの結果をそのまま 1 グループにする）。
/// グループの N は和・代表は N 最大の枝・Q はその代表の Q。
pub fn select_groups(
    state: &GameState,
    masters: &MasterTable,
    legal: &[Move],
    groups: &[Group],
) -> Vec<SelectGroup> {
    let is_box = |mv: &Move| mv.action_type() == Some("SETUP_BOX");
    let mut boxed: Vec<Value> = Vec::new();
    for mv in legal.iter().filter(|m| is_box(m)) {
        let card = move_card_label(state, masters, mv);
        if !card.is_null() && !boxed.contains(&card) {
            boxed.push(card);
        }
    }
    if boxed.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<SelectGroup> = Vec::new();
    // グループごとの「今の代表の N」（代表は N 最大の枝・同点は先に出た方）。
    let mut rep_n: Vec<f64> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        let mv = &legal[g.rep];
        let at = mv.action_type().unwrap_or("");
        let card = move_card_label(state, masters, mv);
        let key = if (is_box(mv) || SETUP_ACTIONS.contains(&at)) && boxed.contains(&card) {
            json!(["setup", card])
        } else {
            json!(["move", gi]) // 1 手 1 グループ（既存の等価手マージのまま）
        };
        match out.iter().position(|s| s.key == key) {
            Some(k) => {
                out[k].idxs.push(gi);
                out[k].n += g.n;
                if g.n > rep_n[k] {
                    rep_n[k] = g.n;
                    out[k].rep = g.rep;
                    out[k].q = g.q;
                }
            }
            None => {
                out.push(SelectGroup { key, idxs: vec![gi], n: g.n, rep: g.rep, q: g.q });
                rep_n.push(g.n);
            }
        }
    }
    out
}

/// Python `cpu_ai.don_box_first_primitive`（DON_BOX → 先頭原始手）。
///
/// 準備箱（`SETUP_BOX`・§20.7.2）も同じ規約で先頭原始手（素の PLAY／ACTIVATE_MAIN）へ落とす。
pub fn don_box_first_primitive(mv: &Move) -> Move {
    if mv.action_type() == Some("SETUP_BOX") {
        return super::r#macro::setup_box_first_primitive(mv);
    }
    if mv.action_type() != Some("DON_BOX") {
        return mv.clone();
    }
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let k = p.get("don_k").and_then(Value::as_f64).unwrap_or(0.0) as i64;
    let tids = p.get("target_ids").and_then(Value::as_array);
    let uuid = p.get("uuid").cloned().unwrap_or(Value::Null);
    if k <= 0 {
        if let Some(t) = tids.filter(|a| !a.is_empty()) {
            return Move::from_json(json!({"kind": "game", "action_type": "ATTACK",
                          "payload": {"uuid": uuid, "target_ids": Value::Array(t.clone())}}));
        }
    }
    Move::from_json(json!({"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": uuid}}))
}

// --- 箱コミット -------------------------------------------------------------------

/// Python `_trace_to_steps`（解決トレース → 自分側の手順）。`legacy`＝旧い署名（[`move_sig_with`]・測定専用）。
///
/// `hidden`（WP `rs-commit-hidden`・[`hidden_at_commit`]）が在れば、自分の段の判断点の対話の候補に
/// commit した時点で見えていなかった札が 1 枚でも在る段で**止める**（その段と先を積まない）。
/// 続きはそこで尽き、次の判断点は読み直す（窓の根畳み・木）。`None` は全部積む旧い形。
fn trace_to_steps(trace: &[TraceStep], name: Seat, legacy: bool, hidden: Option<&[bool]>) -> Vec<Step> {
    let mut steps = Vec::new();
    for (actor, mv, cards) in trace {
        if *actor != name {
            continue;
        }
        if let Some(h) = hidden {
            if cards.iter().any(|c| h.get(*c as usize).copied().unwrap_or(false)) {
                break;
            }
        }
        if mv.action_type() == Some("DON_BOX") {
            let total = box_total(mv);
            if total >= 1 {
                steps.push(Step::Box {
                    sig: move_sig_with(mv, legacy),
                    left: total,
                });
            }
        } else {
            steps.push(Step::Sig(move_sig_with(mv, legacy)));
        }
    }
    steps
}

/// commit した時点で `me` に**見えていない札**の印（札の index ごと・WP `rs-commit-hidden`）。
///
/// 両者の山札・裏向きのライフ・相手の手札にあって、`known_by` に `me` の bit が無い札。公平な引き直し
/// （[`determinize_fair_with`](super::determinize::determinize_fair_with)）が混ぜる枠と同じ集合なので、
/// 本当の盤面でも引き直した世界でも同じ印になる（混ぜるのは札の位置だけで、札の index と印は動かない）。
/// 場・トラッシュ・表向きのライフ・自分の手札・一時領域（解決中に見ている札）は見えている。
pub(crate) fn hidden_at_commit(state: &GameState, me: Seat) -> Vec<bool> {
    let mut out = vec![false; state.cards.len()];
    let unknown = |c: CardIdx| state.card(c).known_by & me.know_bit() == 0;
    for seat in [me, me.other()] {
        let p = state.player(seat);
        let hand: &[CardIdx] = if seat == me { &[] } else { &p.hand };
        for &c in hand.iter().chain(p.deck.iter()) {
            if unknown(c) {
                out[c as usize] = true;
            }
        }
        for &c in &p.life {
            if !state.card(c).is_face_up && unknown(c) {
                out[c as usize] = true;
            }
        }
    }
    out
}

/// DON_BOX の総原始手数（付与 k 枚＋攻撃形なら 1）。
fn box_total(mv: &Move) -> i64 {
    // 根でしか呼ばれない読み＝手の JSON（`Move::json`）をそのまま読む（WP `rs-speed-d`）。
    let mvj = mv.json();
    let mv: &Value = &mvj;
    let null = Value::Null;
    let p = mv.get("payload").unwrap_or(&null);
    let k = p.get("don_k").and_then(Value::as_f64).unwrap_or(0.0) as i64;
    let has_t = p
        .get("target_ids")
        .and_then(Value::as_array)
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    k + i64::from(has_t)
}

/// Python `_commit_apply_ok`（クローンにのみ適用して合法性を確かめる）。
fn commit_apply_ok(state: &GameState, masters: &MasterTable, name: Seat, mv: &Move) -> bool {
    let mut s = Session::new(state.clone());
    matches!(apply::apply_move_inplace(&mut s, masters, name, mv, true), Ok(()))
}

/// Python `_commit_step`（コミット済み手順の機械実行）。
///
/// `steps` は**その場で書き換える**（Python も list を破壊的に更新する）。返り値が `None`
/// または `steps` が空になったら、呼び出し側はコミットを破棄する（契約違反／消化完了）。
/// `legacy`＝旧い署名で照合する（`steps` も旧い署名で積んだもの・測定専用・[`move_sig_with`]）。
pub fn commit_step(
    ctx: &Ctx,
    s: &mut Session,
    name: Seat,
    steps: &mut Vec<Step>,
    legacy: bool,
) -> Result<Option<Move>, EngineError> {
    if steps.is_empty() {
        return Ok(None);
    }
    // Python は本体を `try/except` で包み、候補生成が落ちたら「手を返さない」＝コミットを畳む。
    let legal = match ctx.legal_actions(s) {
        Ok(l) => l,
        Err(e @ EngineError::Unimplemented(_)) => return Err(e),
        Err(_) => return Ok(None),
    };
    let mut mv: Option<Move> = None;
    if !legal.is_empty() {
        match steps[0].clone() {
            Step::Box { sig, left } => {
                if find_move_with(&legal, &sig, legacy).is_some() {
                    if left <= 1 {
                        steps.remove(0);
                    } else {
                        steps[0] = Step::Box {
                            sig: sig.clone(),
                            left: left - 1,
                        };
                    }
                    let uuid = sig.get(1).cloned().unwrap_or(Value::Null);
                    let tgts = sig
                        .get(2)
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    mv = Some(Move::from_json(if !tgts.is_empty() && left <= 1 {
                        json!({"kind": "game", "action_type": "ATTACK",
                               "payload": {"uuid": uuid, "target_ids": tgts}})
                    } else {
                        json!({"kind": "game", "action_type": "ATTACH_DON",
                               "payload": {"uuid": uuid}})
                    }));
                }
            }
            Step::Sig(sig) => {
                if let Some(found) = find_move_with(&legal, &sig, legacy).cloned() {
                    let k = found
                        .payload()
                        .and_then(|p| p.don_k_f64())
                        .unwrap_or(0.0) as i64;
                    if found.action_type() == Some("DON_BOX") && k > 0
                    {
                        // 防御的経路（生成側はカウントダウン形へ変換済み＝通常来ない）
                        let total = box_total(&found);
                        if total > 1 {
                            steps[0] = Step::Box {
                                sig,
                                left: total - 1,
                            };
                        } else {
                            steps.remove(0);
                        }
                    } else {
                        steps.remove(0);
                    }
                    mv = Some(don_box_first_primitive(&found));
                }
            }
        }
    }
    // 適用検証（2026-08-26 void 修正）: 合成した ATTACK/ATTACH_DON は legal に実在しないので
    // 実盤面クローンで適用可能かを確かめる。不可なら契約違反＝箱ごと破棄。
    if let Some(m) = &mv {
        if !commit_apply_ok(s.state(), ctx.masters, name, m) {
            mv = None;
        }
    }
    Ok(mv)
}

// --- 窓の根畳み -------------------------------------------------------------------

struct WindowPick {
    mv: Move,
    legal: Vec<Move>,
    n: Vec<f64>,
    q: Vec<f64>,
    /// 評価に使った世界（コミット継続の生成に使う）。単一候補で即決したときは `None`。
    world: Option<GameState>,
    /// 引いた世界の本数と短い木で読んだか（§7.42・1 本の静的な値なら `None`）。
    read: Option<(usize, bool)>,
    /// 守りの証明の実績（§7.51・対象の窓だったときだけ）。
    defense: Option<Value>,
}

/// Python `_window_choice`（窓では木を回さず、出口 value 最良の 1 手を直接返す）。
///
/// WP `rs-window-read`（§7.42）で 2 つを足した（どちらも旧い旗で元の形に戻る）:
/// - **世界を K 本**（[`DecideOptions::effective_window_worlds`]）: 世界 0 は決定の乱数 `rng` で引き
///   （1 本なら今までと 1 bit も変わらない）、世界 i≥1 は [`window_world_seed`] の乱数で引く。
///   候補ごとの値は「値が出た世界」の平均（`n` はその世界の数）。候補の並びは世界 0 の列挙で、
///   他の世界は手の署名で突き合わせる。続き（旧い旗のとき）は選んだ候補の値が最大の世界から積む。
/// - **自分のライフが少ない戦闘の応手は短い木**（[`tree_branch_values`]）。
fn window_choice(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
) -> Result<Option<WindowPick>, EngineError> {
    let view = opts.world_view();
    let battle = real.active_battle.is_some();
    let world = sample_world(real, name, view, rng)?;
    let mut s = Session::new(world.clone());
    let legal = ctx.legal_actions(&mut s)?;
    if legal.is_empty() {
        return Ok(None);
    }
    if legal.len() == 1 {
        return Ok(Some(WindowPick {
            mv: legal[0].clone(),
            legal,
            n: Vec::new(),
            q: Vec::new(),
            world: None,
            read: None,
            defense: None,
        }));
    }
    let window = if battle { Window::Battle } else { Window::Dialog };
    let tree = !opts.window_tree_legacy && window_tree_applies(&mut s, name, opts.window_tree_life);
    let k = opts.effective_window_worlds();
    // 木で読む窓は世界の本数を別に抑える（費用は「本数 × sims」・静的な値の世界と混ぜて平均しない）。
    let k = if tree { k.min(opts.window_tree_worlds.max(1)) } else { k };
    if k <= 1 && !tree {
        // 今までどおり（世界 1 本・静的な値）。
        let vals = resolved_branch_values(
            ctx,
            &mut s,
            st,
            name,
            &legal,
            QUIESCE_MAX_PLIES,
            BOX_RESOLVE_DEPTH,
            window,
        )?;
        let Some(best) = best_branch(&vals) else {
            return Ok(None); // 全枝で解決に失敗＝full-tree へ委ねる（安全側）
        };
        let (best, defense) = defense_pick(ctx, real, &world, name, opts, &legal, &vals, best)?;
        return Ok(Some(WindowPick {
            mv: legal[best].clone(),
            legal,
            n: vals.iter().map(|v| f64::from(v.is_some())).collect(),
            q: vals.iter().map(|v| v.unwrap_or(-1.0)).collect(),
            world: Some(world),
            read: None,
            defense,
        }));
    }
    let seed0 = opts.search_seed.unwrap_or(0);
    let sigs: Vec<Value> = legal.iter().map(move_sig).collect();
    let m = legal.len();
    let mut sum = vec![0.0f64; m];
    let mut cnt = vec![0.0f64; m];
    // 世界ごとの値（世界 0 の添字に揃えた）と世界そのもの（続きを積む世界を選ぶのに使う）。
    let mut per_world: Vec<(GameState, Vec<Option<f64>>)> = Vec::with_capacity(k);
    for wi in 0..k {
        let (w, mut ws, wlegal) = if wi == 0 {
            (world.clone(), Session::new(world.clone()), legal.clone())
        } else {
            let mut r = super::rng::Pcg32SearchRng::new(window_world_seed(seed0, wi));
            let w = sample_world(real, name, view, &mut r)?;
            let mut ws = Session::new(w.clone());
            let wl = ctx.legal_actions(&mut ws)?;
            (w, ws, wl)
        };
        let vals = if tree {
            let sims = (opts.sims / m).max(1);
            let tseed = window_world_seed(seed0, wi) ^ 0x7EE5_EED0_0000_0001u64;
            tree_branch_values(ctx, &mut ws, st, name, &wlegal, window, sims, opts.c_puct, tseed)?
        } else {
            resolved_branch_values(
                ctx,
                &mut ws,
                st,
                name,
                &wlegal,
                QUIESCE_MAX_PLIES,
                BOX_RESOLVE_DEPTH,
                window,
            )?
        };
        // 世界 0 の添字へ（並びが同じなら恒等）。
        let same = wi == 0
            || (wlegal.len() == m && wlegal.iter().zip(&sigs).all(|(mv, sg)| move_sig(mv) == *sg));
        let mut aligned: Vec<Option<f64>> = vec![None; m];
        for (j, v) in vals.iter().enumerate() {
            let i = if same {
                Some(j)
            } else {
                let sg = move_sig(&wlegal[j]);
                sigs.iter().position(|x| *x == sg)
            };
            if let Some(i) = i {
                aligned[i] = *v;
            }
        }
        for (i, v) in aligned.iter().enumerate() {
            if let Some(v) = v {
                sum[i] += v;
                cnt[i] += 1.0;
            }
        }
        per_world.push((w, aligned));
    }
    let avg: Vec<Option<f64>> = sum
        .iter()
        .zip(&cnt)
        .map(|(s, c)| (*c > 0.0).then(|| s / c))
        .collect();
    let Some(best) = best_branch(&avg) else {
        return Ok(None);
    };
    let (best, defense) = defense_pick(ctx, real, &world, name, opts, &legal, &avg, best)?;
    // 続きを積む世界: 選んだ候補の値が最大の世界（同点は若い方）。
    let mut wbest = 0usize;
    for (wi, (_, v)) in per_world.iter().enumerate() {
        let cur = per_world[wbest].1[best];
        if let Some(x) = v[best] {
            if !matches!(cur, Some(c) if x <= c) {
                wbest = wi;
            }
        }
    }
    let world = per_world.swap_remove(wbest).0;
    Ok(Some(WindowPick {
        mv: legal[best].clone(),
        legal,
        n: cnt,
        q: avg.iter().map(|v| v.unwrap_or(-1.0)).collect(),
        world: Some(world),
        read: Some((k, tree)),
        defense,
    }))
}

/// 守りの窓の証明（§7.51・WP `rs-defense-proof`・[`DecideOptions::defense_proof`]）。
///
/// 対象（相手の手番・自分のライフ ≤ `defense_proof_life`・値の出た候補が 2 つ以上・平らか張り付き）なら、
/// 候補を値の高い順（同点は若い方＝[`best_branch`] と同じ）に、窓の世界 0（`world`・自分の視点で引いた世界）へ
/// 当てた盤面で**相手の側から**このターンの詰みを証明する（[`super::lethal::prove_each`]・配り方は
/// [`DefenseView`] の視点で公平に 8 通り〔[`super::lethal::make_deals`]〕・全部で相手が勝てば「証明」）。証明されなかった最初の候補を返す＝「どの配り方でも相手が勝つ候補を外して値で選ぶ」と同じ
/// （判定不能・節点の上限は「証明されない」側）。全部が証明されたら `best` のまま。対象外は `(best, None)`。
#[allow(clippy::too_many_arguments)]
fn defense_pick(
    ctx: &Ctx,
    real: &GameState,
    world: &GameState,
    name: Seat,
    opts: &DecideOptions,
    legal: &[Move],
    vals: &[Option<f64>],
    best: usize,
) -> Result<(usize, Option<Value>), EngineError> {
    if !opts.defense_proof || real.winner.is_some() || real.turn_player == name {
        return Ok((best, None));
    }
    if (real.player(name).life.len() as i64) > opts.defense_proof_life {
        return Ok((best, None));
    }
    let got: Vec<f64> = vals.iter().flatten().copied().collect();
    if got.len() < 2 {
        return Ok((best, None));
    }
    let hi = got.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lo = got.iter().copied().fold(f64::INFINITY, f64::min);
    let flat = hi - lo <= opts.defense_proof_flat;
    let sat = got.iter().all(|v| v.abs() >= opts.defense_proof_sat);
    if !flat && !sat {
        return Ok((best, None));
    }
    let mut order: Vec<usize> = (0..vals.len()).filter(|i| vals[*i].is_some()).collect();
    order.sort_by(|a, b| vals[*b].partial_cmp(&vals[*a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(b)));
    let opp = name.other();
    let seed0 = super::lethal::lethal_seed(window_world_seed(opts.search_seed.unwrap_or(0), 0)) ^ 0xDEF0_0000_0000_0000u64;
    let mut left = opts.defense_proof_total;
    let mut nodes = 0u64;
    let mut proven: Vec<usize> = Vec::new();
    let mut pick = None;
    let mut s = Session::new(world.clone());
    for (j, &i) in order.iter().enumerate() {
        let budget = opts.defense_proof_nodes.min(left);
        if budget == 0 {
            pick = Some(i);
            break;
        }
        let seed = seed0 ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mv = &legal[i];
        let r = s.discard_events(|s| {
            s.transaction(|s| -> Result<(bool, u64), EngineError> {
                match super::apply::apply_move_inplace(s, ctx.masters, name, mv, true) {
                    Ok(()) => {
                        let from = if opts.defense_proof_view == DefenseView::Opp { opp } else { name };
                        let deals = super::lethal::make_deals(
                            s.state(),
                            ctx.masters,
                            from,
                            seed,
                            super::lethal::LETHAL_RAND_DEALS,
                            opts.lethal_deal_legacy,
                        )?;
                        let (per, used) = super::lethal::prove_each(ctx.masters, &deals, opp, budget, true);
                        Ok((per.len() == deals.len() && per.iter().all(|r| *r == "win"), used))
                    }
                    Err(e @ EngineError::Unimplemented(_)) => Err(e),
                    Err(_) => Ok((false, 0)),
                }
            })
        });
        let (ok, used) = r?;
        nodes += used;
        left = left.saturating_sub(used);
        if !ok {
            pick = Some(i);
            break;
        }
        proven.push(i);
    }
    let chosen = pick.unwrap_or(best);
    let info = serde_json::json!({
        "flat": flat, "sat": sat, "proven": proven, "pick": chosen, "was": best,
        "all_proven": pick.is_none(), "nodes": nodes,
        "view": if opts.defense_proof_view == DefenseView::Opp { "opp" } else { "mine" },
    });
    Ok((chosen, Some(info)))
}

/// 短い木で読む戦闘の窓か（§7.42 の 3）: 戦闘中・判断するのが `name` のブロッカー／カウンターの選択・
/// `name` のライフ（枚数）が `life_max` 以下。候補の数は呼び出し側が見る。
fn window_tree_applies(s: &mut Session, name: Seat, life_max: i64) -> bool {
    if s.state().active_battle.is_none() {
        return false;
    }
    match crate::rules::pending::pending_actor_action(s) {
        Some((seat, action)) if seat == name => {
            (action == crate::rules::pending::ACT_SELECT_BLOCKER
                || action == crate::rules::pending::ACT_SELECT_COUNTER)
                && (s.state().player(name).life.len() as i64) <= life_max
        }
        _ => false,
    }
}

/// 各候補を戦闘の終わりまで畳んだ盤面から短い木を回し、その根の値（`name` から見た）で比べる（§7.42 の 3）。
///
/// 畳み方は [`resolved_branch_values`] と同じ（同じ窓・同じ深さ・同じ CRN）。出口盤面が終局なら ±1、
/// そうでなければその盤面（引いた世界の中＝木の中では引き直さない）から `sims` 回の木を回し、
/// 根の値 `Σ W ÷ Σ N`（根の手番から見た）を `name` から見た値に直す。Dirichlet は混ぜない。
#[allow(clippy::too_many_arguments)]
fn tree_branch_values(
    ctx: &Ctx,
    s: &mut Session,
    st: &mut SearchState,
    name: Seat,
    legal: &[Move],
    window: Window,
    sims: usize,
    c_puct: f64,
    seed: u64,
) -> Result<Vec<Option<f64>>, EngineError> {
    super::quiesce::resolved_branch_values_with(
        ctx,
        s,
        st,
        name,
        legal,
        QUIESCE_MAX_PLIES,
        BOX_RESOLVE_DEPTH,
        window,
        &mut |s, st| {
            if ctx.is_terminal(s) {
                return ctx.value(s, name);
            }
            let mut tree = TreeMcts::new(ctx, c_puct, sims, 0.0);
            let mut r = super::rng::Pcg32SearchRng::new(seed);
            tree.run_in_world(s.state().clone(), &mut r, st)?;
            let nodes = tree.into_nodes();
            let Some(root) = nodes.first() else {
                return ctx.value(s, name);
            };
            let n: f64 = root.n.iter().sum();
            let (Some(seat), true) = (root.to_move, n > 0.0) else {
                return ctx.value(s, name);
            };
            let v = root.w.iter().sum::<f64>() / n;
            Ok(if seat == name { v } else { -v })
        },
    )
}

/// Python `_commit_window_continuation`（選んだ枝の自分側継続をコミットする）。
///
/// `hidden_legacy`＝見えていなかった札を選ぶ段も積む旧い形（[`DecideOptions::commit_hidden_legacy`]）。
#[allow(clippy::too_many_arguments)]
fn commit_window_continuation(
    ctx: &Ctx,
    real: &GameState,
    world: &GameState,
    name: Seat,
    mv: &Move,
    st: &mut SearchState,
    legacy: bool,
    hidden_legacy: bool,
) -> Vec<Step> {
    let window = if real.active_battle.is_some() {
        Window::Battle
    } else {
        Window::Dialog
    };
    let hidden = (!hidden_legacy).then(|| hidden_at_commit(world, name));
    let mut s = Session::new(world.clone());
    if apply::apply_move_inplace(&mut s, ctx.masters, name, mv, true).is_err() {
        return Vec::new(); // Python: `nxt is None` → コミット無し
    }
    let mut trace: Vec<TraceStep> = Vec::new();
    let ok = resolve_battle_inplace(
        ctx,
        &mut s,
        st,
        window,
        QUIESCE_MAX_PLIES,
        true,
        BOX_RESOLVE_DEPTH,
        Some(&mut trace),
    );
    if ok.is_err() {
        return Vec::new(); // Python の `except Exception: pass`（コミット生成の失敗は手を止めない）
    }
    trace_to_steps(&trace, name, legacy, hidden.as_deref())
}

/// Python `_commit_play_dialog`（PLAY / ACTIVATE_MAIN の後続対話をコミットする）。
///
/// 世界サンプルをここで引く（単一世界の経路）。複数世界（§20.7.1）は木を回した世界を
/// 使い回す＝[`commit_play_dialog_in_world`] を直に呼ぶ（乱数はもう引かない）。
#[allow(clippy::too_many_arguments)]
fn commit_play_dialog(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    mv: &Move,
    view: WorldView,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
    legacy: bool,
    hidden_legacy: bool,
) -> Result<Vec<Step>, EngineError> {
    let world = sample_world(real, name, view, rng)?;
    Ok(commit_play_dialog_in_world(ctx, &world, name, mv, st, legacy, hidden_legacy))
}

/// [`commit_play_dialog`] の「世界を外から渡す」版（§20.7.1 の `worlds>1` が使う）。
fn commit_play_dialog_in_world(
    ctx: &Ctx,
    world: &GameState,
    name: Seat,
    mv: &Move,
    st: &mut SearchState,
    legacy: bool,
    hidden_legacy: bool,
) -> Vec<Step> {
    let hidden = (!hidden_legacy).then(|| hidden_at_commit(world, name));
    let mut s = Session::new(world.clone());
    if apply::apply_move_inplace(&mut s, ctx.masters, name, mv, true).is_err() {
        return Vec::new();
    }
    if !in_dialog(&mut s) || in_battle(&s) {
        return Vec::new(); // 対話が無ければコミット無し
    }
    let mut trace: Vec<TraceStep> = Vec::new();
    let ok = resolve_battle_inplace(
        ctx,
        &mut s,
        st,
        Window::Dialog,
        QUIESCE_MAX_PLIES,
        true,
        BOX_RESOLVE_DEPTH,
        Some(&mut trace),
    );
    if ok.is_err() {
        return Vec::new();
    }
    trace_to_steps(&trace, name, legacy, hidden.as_deref())
}

/// 準備箱（`SETUP_BOX`・§20.7.2）の「素の手より後」の手順をコミットへ積む。
///
/// `commit_play_dialog` と同じ流儀（世界サンプルの上で継続を打ち直し、自分の手だけ手順化）。
#[allow(clippy::too_many_arguments)]
fn commit_setup_box(
    ctx: &Ctx,
    real: &GameState,
    name: Seat,
    mv: &Move,
    view: WorldView,
    rng: &mut dyn SearchRng,
    legacy: bool,
    hidden_legacy: bool,
) -> Result<Vec<Step>, EngineError> {
    let world = sample_world(real, name, view, rng)?;
    let hidden = (!hidden_legacy).then(|| hidden_at_commit(&world, name));
    let mut s = Session::new(world);
    match super::r#macro::setup_box_continuation(&mut s, ctx.masters, name, mv) {
        Ok(trace) => Ok(trace_to_steps(&trace, name, legacy, hidden.as_deref())),
        Err(e @ EngineError::Unimplemented(_)) => Err(e),
        Err(_) => Ok(Vec::new()), // コミット生成の失敗は手を止めない（Python の `except: pass`）
    }
}

// --- 残ドン掘り／残り起動（腕 A・A2）------------------------------------------------

/// Python `cpu_learned.DON_RAMP_MARK`。
pub const DON_RAMP_MARK: &str = "ドン!!デッキから";

/// Python `_leader_has_don_ramp`（能力の raw_text の構造語）。
fn leader_has_don_ramp(masters: &MasterTable, master: crate::model::MasterIdx) -> bool {
    let m = masters.get(master);
    let mut blob: Vec<&str> = Vec::new();
    for id in &m.ability_ids {
        if let Some(ab) = masters.abilities.get(*id) {
            blob.push(ab.raw_text.as_str());
        }
    }
    let text = if blob.iter().all(|s| s.is_empty()) {
        m.effect_text.clone()
    } else {
        blob.join(" ")
    };
    text.replace('\u{203C}', "!!").contains(DON_RAMP_MARK)
}

/// Python `_tree_has`（効果木に `atype` の action があるか）。
fn tree_has(node: Option<&crate::effects::ast::EffectNode>, atype: crate::effects::ast::ActionType) -> bool {
    let mut acts = Vec::new();
    crate::encode::walk_all(node, &mut acts);
    acts.iter().any(|a| a.ty == atype)
}

/// Python `_is_dig_card`（登場時にドンを戻してドローするコスト 1 キャラ）。
fn is_dig_card(masters: &MasterTable, master: crate::model::MasterIdx) -> bool {
    use crate::effects::ast::{ActionType, TriggerType};
    let m = masters.get(master);
    if m.ty != crate::model::CardType::Character || m.cost != 1 {
        return false;
    }
    for id in &m.ability_ids {
        let Some(ab) = masters.abilities.get(*id) else {
            continue;
        };
        if ab.trigger != TriggerType::OnPlay {
            continue;
        }
        if !tree_has(ab.effect.as_ref(), ActionType::Draw) {
            continue;
        }
        if tree_has(ab.cost.as_ref(), ActionType::ReturnDon) {
            return true;
        }
    }
    false
}

/// Python `_residual_dig_move`（腕 A）。
fn residual_dig_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    legal: &[Move],
) -> Option<Move> {
    let state = s.state();
    let p = state.player(seat);
    if p.don_active.is_empty() || p.field.len() >= 5 {
        return None;
    }
    for mv in legal {
        if mv.action_type() != Some("PLAY") {
            continue;
        }
        let u = mv.payload()?.uuid();
        let Some(u) = u else { continue };
        let card = p.hand.iter().find(|c| state.card(**c).uuid == u);
        let Some(card) = card else { continue };
        if is_dig_card(ctx.masters, state.card(*card).master) {
            return Some(mv.clone());
        }
    }
    None
}

/// Python `_residual_activate_move`（腕 A2）。
fn residual_activate_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    legal: &[Move],
) -> Option<Move> {
    let state = s.state();
    let leader = state.player(seat).leader?;
    if !leader_has_don_ramp(ctx.masters, state.card(leader).master) {
        return None;
    }
    let leader_uuid = state.card(leader).uuid.clone();
    for mv in legal {
        if mv.action_type() != Some("ACTIVATE_MAIN") {
            continue;
        }
        if mv.payload_uuid() != Some(leader_uuid.as_str())
        {
            continue;
        }
        return Some(mv.clone());
    }
    None
}

/// Python `_pick_attach_target`（"low"＝攻撃できるキャラの最低パワー／"high"＝最高パワー）。
fn pick_attach_target(cands: &[(String, i32, bool)], policy: &str) -> Option<String> {
    if cands.is_empty() {
        return None;
    }
    if policy == "high" {
        let mut v: Vec<&(String, i32, bool)> = cands.iter().collect();
        v.sort_by(|a, b| (-a.1, &a.0).cmp(&(-b.1, &b.0)));
        return Some(v[0].0.clone());
    }
    let pool: Vec<&(String, i32, bool)> = {
        let atk: Vec<&(String, i32, bool)> = cands.iter().filter(|t| t.2).collect();
        if atk.is_empty() {
            cands.iter().collect()
        } else {
            atk
        }
    };
    let mut v = pool;
    v.sort_by(|a, b| (a.1, &a.0).cmp(&(b.1, &b.0)));
    Some(v[0].0.clone())
}

/// Python `_residual_attach_move`（起動で開いた自分のキャラへの付与対話を方針で解く）。
fn residual_attach_move(
    ctx: &Ctx,
    s: &mut Session,
    seat: Seat,
    policy: &str,
) -> Result<Option<Move>, EngineError> {
    let Some(req) = crate::rules::pending::get_pending_request(s, ctx.masters, false) else {
        return Ok(None);
    };
    let sel: Vec<String> = req
        .get("selectable_uuids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default();
    if sel.is_empty() {
        return Ok(None);
    }
    let turn = s.state().turn_count;
    let mut cands: Vec<(String, i32, bool)> = Vec::new();
    {
        let state = s.state();
        let field: Vec<CardIdx> = state.player(seat).field.clone();
        for u in &sel {
            let Some(c) = field.iter().find(|c| state.card(**c).uuid == *u) else {
                return Ok(None); // 自分のキャラ以外が混ざる対話は触らない
            };
            let card = state.card(*c);
            let pw = card.get_power(ctx.masters.get(card.master), true);
            // Python: `turn > 2 and not is_rest and not (is_newly_played and not 速攻)`
            let summoning_sick =
                card.is_newly_played && !crate::rules::has_keyword(state, *c, "速攻");
            let can_atk = turn > 2 && !card.is_rest && !summoning_sick;
            cands.push((u.clone(), pw, can_atk));
        }
    }
    let Some(target) = pick_attach_target(&cands, policy) else {
        return Ok(None);
    };
    let legal = ctx.legal_actions(s)?;
    for mv in &legal {
        if mv.action_type() != Some("RESOLVE_EFFECT_SELECTION") {
            continue;
        }
        let sel = mv.payload().and_then(|p| p.get("selected_uuids"));
        let su: Vec<&str> = sel
            .as_deref()
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if su == [target.as_str()] {
            return Ok(Some(mv.clone()));
        }
    }
    let pending = crate::rules::pending::get_pending_request(s, ctx.masters, false);
    let mut payload = crate::effects::interact::default_interaction_payload(
        s.state(),
        ctx.masters,
        pending.as_ref(),
    );
    let obj = payload.as_object_mut().map(std::mem::take).unwrap_or_default();
    let mut obj: Map<String, Value> = obj;
    obj.insert("selected_uuids".into(), json!([target]));
    obj.insert("accepted".into(), Value::Bool(true));
    Ok(Some(Move::from_json(json!({"kind": "game", "action_type": "RESOLVE_EFFECT_SELECTION",
                   "payload": Value::Object(obj)}))))
}

// --- 複数世界（§20.7.1・WP `rs-pimc-worlds`）----------------------------------------

/// 1 つの世界の探索結果。木（ノード列）と根の世界サンプルを持ち帰るのは、PV と
/// コミットの継続を「選んだ手の訪問が最も多い世界」から採るため（§20.7.1）。
struct WorldRun {
    /// この世界の乱数 seed（`search_seed + i`）
    seed: u64,
    run: RunOut,
    nodes: Vec<Node>,
    /// 根の世界サンプル（コミットの継続をこの盤面で作る）
    world: GameState,
    /// `root_reply` の木の中の証明の実績（§7.21・使ったときだけ）
    reply: Option<super::lethal::ReplyProof>,
}

/// 1 本の木を回す（つまみは全世界で同じ）。
#[allow(clippy::too_many_arguments)]
fn run_one_world(
    ctx: &Ctx,
    opts: &DecideOptions,
    name: Seat,
    world_seed: u64,
    world: GameState,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
    reply: bool,
) -> Result<(RunOut, Vec<Node>, Option<super::lethal::ReplyProof>), EngineError> {
    let mut tree = TreeMcts::new(ctx, opts.c_puct, opts.sims, opts.dirichlet_eps);
    tree.select_rule = opts.select_rule;
    tree.q_min_frac = opts.q_min_frac;
    tree.root_prior_temp = opts.root_prior_temp;
    if opts.opp_blind {
        tree.opp_blind = Some(super::mcts::OppBlind::new(name, opp_blind_seed(world_seed)));
    }
    if reply {
        // 決定あたりの合計を世界の本数で等分する（世界はスレッドで並ぶ＝分けておけば決定的）。
        let k = opts.effective_worlds().max(1) as u64;
        tree.lethal_reply = Some(super::lethal::ReplyProof::new(
            name,
            super::lethal::lethal_seed(world_seed),
            opts.lethal_reply_nodes,
            opts.lethal_reply_total / k,
            opts.lethal_deal_legacy,
        ));
    }
    let run = tree.run_in_world(world, rng, st)?;
    let reply = tree.lethal_reply.take();
    Ok((run, tree.into_nodes(), reply))
}

/// K 本の世界を引いて木を回す（§20.7.1）。
///
/// - 世界 0 は**呼び出し側の `rng`** で引き、**このスレッド**で回す＝`K=1` なら今までの
///   `TreeMcts::run` と 1 bit も変わらない（出目の消費順も同じ）。
/// - 世界 `i>=1` は `Pcg32SearchRng::new(search_seed + i)` を各スレッドで作る＝同じ seed なら
///   何度回しても同じ（スレッドの終わる順に依らない）。枝予算も世界ごとに独立に張る
///   （1 本の木から見た予算は `K=1` のときと同じ）。
fn run_worlds(
    ctx: &Ctx,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    st: &mut SearchState,
    reply: bool,
) -> Result<Vec<WorldRun>, EngineError> {
    let k = opts.effective_worlds();
    let seed0 = opts.search_seed.unwrap_or(0);
    let world0 = sample_world(state, name, opts.world_view(), rng)?;
    if k <= 1 {
        let (run, nodes, rp) = run_one_world(ctx, opts, name, seed0, world0.clone(), rng, st, reply)?;
        return Ok(vec![WorldRun { seed: seed0, run, nodes, world: world0, reply: rp }]);
    }
    // 世界 i>=1 を 1 本回す（呼んだスレッドの thread_local を使う＝必ず新しいスレッドから呼ぶ）。
    let world_job = |i: usize| -> Result<(WorldRun, i64, u64), EngineError> {
        let seed = seed0.wrapping_add(i as u64);
        let _prof = crate::prof::enter_top(crate::prof::Cat::Search);
        let _enc_cache = crate::encode::decide_cache_scope(ctx.masters);
        let _tok_cache = crate::net::nrel::tok_cache_scope();
        // 準備箱（§20.7.2）の状態は **thread_local**（`macro::SETUP`）＝新しい
        // スレッドでは既定（無効）で始まる。世界 0（このスレッド）と同じ候補列挙に
        // するため、世界ごとに張り直す（張らないと世界 1 以降だけ箱が出ず、根の
        // `legal` が世界 0 と食い違って `unmapped` に落ちる）。枝の計測は世界 0 の
        // ぶんだけが `boxes` に出る（他世界の計測はスレッドと共に捨てる）。
        super::r#macro::reset_setup_state(opts.search.setup_box, opts.search.select_branch_on());
        let mut r = super::rng::Pcg32SearchRng::new(seed);
        let mut wst = SearchState {
            budget: BoxBudget::new(opts.budget),
        };
        let world = sample_world(state, name, opts.world_view(), &mut r)?;
        let (run, nodes, rp) = run_one_world(ctx, opts, name, seed, world.clone(), &mut r, &mut wst, reply)?;
        Ok((WorldRun { seed, run, nodes, world, reply: rp }, wst.budget.used, wst.budget.exhausted))
    };
    let world_job = &world_job;
    let lost = || EngineError::BadPayload("decide: 世界の探索スレッドが panic した".into());
    // 世界 i を新しいスレッドで回して待つ（同時に走る本数を抑えるときの 1 単位）。
    let fresh = |i: usize| -> Result<(WorldRun, i64, u64), EngineError> {
        std::thread::scope(|s| s.spawn(|| world_job(i)).join().map_err(|_| lost())?)
    };
    // 同時に走る世界の上限（§7.55）。0 か K 以上なら全部を一度に（今までどおり）。
    let cap = if opts.world_threads == 0 { k } else { opts.world_threads.min(k) };
    let mut runs: Vec<WorldRun> = Vec::with_capacity(k);
    let mut extra_used = 0i64;
    let mut extra_exhausted = 0u64;
    let mut rest: Vec<Result<(WorldRun, i64, u64), EngineError>> = Vec::with_capacity(k - 1);
    if cap >= k {
        std::thread::scope(|scope| -> Result<(), EngineError> {
            let handles: Vec<_> = (1..k).map(|i| scope.spawn(move || world_job(i))).collect();
            // 世界 0 はこのスレッドで回す（スレッドは K-1 本＝K 本が同時に走る）。
            let (run, nodes, rp) = run_one_world(ctx, opts, name, seed0, world0.clone(), rng, st, reply)?;
            runs.push(WorldRun { seed: seed0, run, nodes, world: world0, reply: rp });
            for h in handles {
                rest.push(h.join().map_err(|_| lost())?);
            }
            Ok(())
        })?;
    } else {
        // 上限つき: 世界 1.. を次の番号から取り合う「取り手」を cap-1 本立て、このスレッドも
        // 世界 0 の後で取り手に加わる。取り手は世界ごとに新しいスレッドを立てて待つ
        // （thread_local を持ち越さない）。結果は番号の位置へ置く＝終わる順に依らない。
        type Slot = std::sync::Mutex<Option<Result<(WorldRun, i64, u64), EngineError>>>;
        let next = std::sync::atomic::AtomicUsize::new(1);
        let slots: Vec<Slot> = (1..k).map(|_| std::sync::Mutex::new(None)).collect();
        let take = || loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= k {
                break;
            }
            let res = fresh(i);
            *slots[i - 1].lock().unwrap_or_else(|e| e.into_inner()) = Some(res);
        };
        let w0 = std::thread::scope(|scope| -> Result<_, EngineError> {
            let takers: Vec<_> = (1..cap).map(|_| scope.spawn(&take)).collect();
            let w0 = run_one_world(ctx, opts, name, seed0, world0.clone(), rng, st, reply);
            take();
            for h in takers {
                h.join().map_err(|_| lost())?;
            }
            Ok(w0)
        })?;
        let (run, nodes, rp) = w0?;
        runs.push(WorldRun { seed: seed0, run, nodes, world: world0, reply: rp });
        for slot in slots {
            rest.push(slot.into_inner().unwrap_or_else(|e| e.into_inner()).ok_or_else(lost)?);
        }
    }
    for r in rest {
        let (wr, used, exhausted) = r?;
        extra_used += used;
        extra_exhausted += exhausted;
        runs.push(wr);
    }
    // 予算の実績だけ足す（`left` は世界 0 のものを残す＝この後のコミット生成の見え方を変えない）。
    st.budget.used += extra_used;
    st.budget.exhausted += extra_exhausted;
    Ok(runs)
}

/// 根の統計を束ねる（§20.7.1）。
///
/// N は和・Q は N 重みの平均（`W = Q*N` を足して和 N で割る）・P と `legal` の並びは世界 0。
/// 世界の `legal` は同じ盤面の同じ列挙なので普通は完全に一致するが、一致しない場合に備えて
/// [`move_sig`] で突き合わせる（対応が付かない手は捨て、[`WorldStats::unmapped`] に数える）。
///
/// 戻り値の 3 つ目は「世界 0 の添字 → その世界の添字」（PV の 1 手目を引くのに使う）。
#[allow(clippy::type_complexity)]
fn merge_worlds(
    runs: &[WorldRun],
    opts: &DecideOptions,
) -> (RunOut, Vec<WorldStats>, Vec<Vec<Option<usize>>>) {
    if runs.len() == 1 {
        // 単一世界は**そのまま**返す（束ねの算術を通さない＝1 bit も変わらない）。
        return (runs[0].run.clone(), Vec::new(), Vec::new());
    }
    let legal = runs[0].run.legal.clone();
    let m = legal.len();
    let sigs: Vec<Value> = legal.iter().map(|mv| move_sig_with(mv, opts.commit_sig_legacy)).collect();
    let mut n = vec![0.0f64; m];
    let mut w = vec![0.0f64; m];
    let mut per_world = Vec::with_capacity(runs.len());
    let mut back: Vec<Vec<Option<usize>>> = Vec::with_capacity(runs.len());
    let floor = q_min_n_floor(opts.sims, opts.q_min_frac);
    for (wi, r) in runs.iter().enumerate() {
        // その世界の添字 → 世界 0 の添字
        // 世界 0、または並びが世界 0 と完全に一致する世界は恒等写像（普通はこちら）。
        let same_order = wi == 0
            || (r.run.legal.len() == m
                && r.run.legal.iter().zip(&sigs).all(|(mv, s)| move_sig_with(mv, opts.commit_sig_legacy) == *s));
        let to_root: Vec<Option<usize>> = if same_order {
            (0..r.run.legal.len()).map(Some).collect()
        } else {
            r.run
                .legal
                .iter()
                .map(|mv| {
                    let sg = move_sig_with(mv, opts.commit_sig_legacy);
                    sigs.iter().position(|s| *s == sg)
                })
                .collect()
        };
        let mut wn = vec![0.0f64; m];
        let mut wq = vec![0.0f64; m];
        let mut bk: Vec<Option<usize>> = vec![None; m];
        let mut unmapped = 0usize;
        for (j, tgt) in to_root.iter().enumerate() {
            let Some(i) = *tgt else {
                unmapped += 1;
                continue;
            };
            let (nj, qj) = (r.run.n[j], r.run.q[j]);
            n[i] += nj;
            w[i] += qj * nj;
            wn[i] = nj;
            wq[i] = qj;
            bk[i] = Some(j);
        }
        let best = if r.run.legal.is_empty() {
            None
        } else {
            let bj = select_index(&r.run.n, &r.run.q, opts.select_rule, floor);
            to_root.get(bj).copied().flatten()
        };
        per_world.push(WorldStats {
            seed: r.seed,
            n: wn,
            q: wq,
            best,
            unmapped,
            p_differs: r.run.p != runs[0].run.p,
        });
        back.push(bk);
    }
    let q: Vec<f64> = n.iter().zip(&w).map(|(n, w)| w / n.max(1.0)).collect();
    // 束ねた統計の「訪問下限」は実効 sims（K×sims）で測る＝1 世界のときと同じ割合になる。
    let best = if legal.is_empty() {
        None
    } else {
        let bi = select_index(
            &n,
            &q,
            opts.select_rule,
            q_min_n_floor(opts.sims * runs.len(), opts.q_min_frac),
        );
        Some(legal[bi].clone())
    };
    let p = runs[0].run.p.clone();
    (RunOut { best, legal, n, q, p }, per_world, back)
}

// --- decide 本体 ------------------------------------------------------------------

/// Python `LearnedEngine.decide`／`_decide_inner`。
///
/// `state` は**実盤面**（世界サンプル前）。`rng` は世界サンプル・Dirichlet・温度の出目を出す。
pub fn decide(
    masters: &MasterTable,
    net: &crate::net::LoadedNet,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    carry: &DecideCarry,
) -> Result<DecideOut, EngineError> {
    let ctx = Ctx {
        masters,
        net,
        opts: opts.search.clone(),
        box_battle: opts.box_battle,
        box_dialog: opts.box_dialog,
        quiesce: opts.quiesce,
        quiesce_max_plies: QUIESCE_MAX_PLIES,
    };
    // 戦闘箱の枝予算をこの decide のぶんだけ張る（Python `reset_box_budget()`／`clear_box_budget()`）。
    opts.validate()?;
    let mut st = SearchState {
        budget: BoxBudget::new(opts.budget),
    };
    let out = decide_inner(&ctx, state, name, opts, rng, carry, &mut st);
    st.budget.clear();
    out
}

fn empty_out(kind: &'static str, mv: Option<Move>, carry: DecideCarry, st: &SearchState) -> DecideOut {
    DecideOut {
        sig: mv.as_ref().map(move_sig),
        k: None,
        mv,
        kind,
        legal: Vec::new(),
        n: Vec::new(),
        q: Vec::new(),
        p: None,
        groups: Vec::new(),
        select_groups: Vec::new(),
        carry,
        budget_used: st.budget.used,
        budget_exhausted: st.budget.exhausted,
        pv: Vec::new(),
        worlds: 1,
        per_world: Vec::new(),
        world_used: None,
        lethal: None,
        lethal_reply: None,
        window_read: None,
        defense_proof: None,
        temp_pick: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn decide_inner(
    ctx: &Ctx,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    carry: &DecideCarry,
    st: &mut SearchState,
) -> Result<DecideOut, EngineError> {
    let carry = carry.clone();
    let real = Session::new(state.clone());

    // ⓪ 詰みの証明（§7.21・既定 off）。根で証明できればその手を返す（持ち越しは捨てる）。証明できなければ
    // 探索へ戻す（決定の乱数は触っていない）。`root_reply` は発動条件を満たした決定だけ木の中でも証明する。
    use super::lethal::LethalProof;
    let lethal = if opts.lethal_proof != LethalProof::Off {
        super::lethal::root_proof(ctx.masters, state, name, opts.search_seed, opts.lethal_nodes, opts.lethal_deal_legacy)?
    } else {
        None
    };
    if let Some(l) = &lethal {
        if l.status == "win" {
            let mut out = empty_out("lethal", l.mv.clone(), DecideCarry::default(), st);
            out.lethal = lethal;
            return Ok(out);
        }
    }
    let reply = lethal.is_some() && opts.lethal_proof == LethalProof::RootReply;
    let mut out = decide_after_lethal(ctx, state, name, opts, rng, carry, st, real, reply)?;
    out.lethal = lethal;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn decide_after_lethal(
    ctx: &Ctx,
    state: &GameState,
    name: Seat,
    opts: &DecideOptions,
    rng: &mut dyn SearchRng,
    mut carry: DecideCarry,
    st: &mut SearchState,
    mut real: Session,
    reply: bool,
) -> Result<DecideOut, EngineError> {

    // ① 箱コミットの機械実行（窓の根畳み・木より前）
    if opts.box_commit && !carry.commit.is_empty() {
        let mut steps = carry.commit.clone();
        let mv = commit_step(ctx, &mut real, name, &mut steps, opts.commit_sig_legacy)?;
        match mv {
            Some(mv) => {
                carry.commit = if steps.is_empty() { Vec::new() } else { steps };
                return Ok(empty_out("commit", Some(mv), carry, st));
            }
            None => carry.commit = Vec::new(), // 契約違反／消化完了＝key を畳む
        }
    }

    // ② 残り起動の付与対話（腕 A2）
    if carry.resact_pending {
        if in_dialog(&mut real) && !in_battle(&real) {
            // Python は `residual_activate` が None でも呼ぶ（`_pick_attach_target` は
            // "high" 以外を「攻撃できるキャラの最低パワー」として扱う）＝同じにする。
            let policy = opts.residual_activate.as_deref().unwrap_or("");
            if let Some(mv) = residual_attach_move(ctx, &mut real, name, policy)? {
                return Ok(empty_out("main", Some(mv), carry, st));
            }
        } else {
            carry.resact_pending = false;
        }
    }

    // ③ 窓の根畳み（戦闘窓／対話箱が有効なら効果対話窓）
    // 安全側の守り（WP `rs-overflow-trigger`・§7.44 の (A)）: メインの判断点（`MAIN_ACTION`）は戦闘が
    // 残っていても窓の道に入らない＝木が合法手（原始手・箱）から選ぶ。窓は原始手を返す前提なので、
    // ここへ入るとメインの箱（`DON_BOX`）をそのまま返して適用で落ちる（規則の欠陥で戦闘が残った局）。
    let main_point = matches!(
        crate::rules::pending::pending_actor_action(&mut real),
        Some((_, "MAIN_ACTION"))
    );
    if (in_battle(&real) && !main_point) || (opts.box_dialog && in_dialog(&mut real)) {
        if let Some(pick) = window_choice(ctx, state, name, opts, rng, st)? {
            // §7.42 の 1: 窓の続きは効果の対話の段・戦闘の応手の段だけ＝既定は積まない（次の判断点で読み直す）。
            if opts.box_commit && opts.window_carry_legacy {
                if let Some(world) = &pick.world {
                    let steps =
                        commit_window_continuation(
                            ctx,
                            state,
                            world,
                            name,
                            &pick.mv,
                            st,
                            opts.commit_sig_legacy,
                            opts.commit_hidden_legacy,
                        );
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
            }
            return Ok(DecideOut {
                sig: Some(move_sig(&pick.mv)),
                k: None,
                mv: Some(pick.mv),
                kind: "window",
                legal: pick.legal,
                n: pick.n,
                q: pick.q,
                p: None,
                groups: Vec::new(),
                select_groups: Vec::new(),
                carry,
                budget_used: st.budget.used,
                budget_exhausted: st.budget.exhausted,
                pv: Vec::new(),
                worlds: 1,
                per_world: Vec::new(),
                world_used: None,
                lethal: None,
                lethal_reply: None,
                window_read: pick.read,
                defense_proof: pick.defense,
                temp_pick: None,
            });
        }
    }

    // ④ 木（§20.7.1: 世界を K 本引いて根で束ねる。K=1＝今までどおりの 1 本）
    let runs = run_worlds(ctx, state, name, opts, rng, st, reply)?;
    let reply = runs.iter().filter_map(|r| r.reply.clone()).reduce(|mut a, b| {
        a.checked += b.checked;
        a.children += b.children;
        a.proven += b.proven;
        a.nodes += b.nodes;
        a
    });
    let (run, per_world, back) = merge_worlds(&runs, opts);
    let worlds = runs.len();
    // 束ねた統計の実効 sims（K×sims）＝`q_min_n` の訪問下限はこれで測る。
    let eff_sims = opts.sims * worlds;
    let mut mv = run.best.clone();
    // 出す手の添字（世界 0 の `legal` 基準・PV とコミットの世界を選ぶのに使う）
    let mut chosen_idx = if run.legal.is_empty() {
        None
    } else {
        Some(select_index(
            &run.n,
            &run.q,
            opts.select_rule,
            q_min_n_floor(eff_sims, opts.q_min_frac),
        ))
    };
    let mut groups = Vec::new();
    let mut sel_groups: Vec<SelectGroup> = Vec::new();
    let mut temp_pick: Option<(u8, bool)> = None;
    if !run.legal.is_empty() {
        groups = merge_root_stats(state, ctx.masters, &run.legal, &run.n, &run.q);
        sel_groups = select_groups(state, ctx.masters, &run.legal, &groups);
        if !groups.is_empty() {
            // 出す手は**マージ後のグループ**から選ぶ（Python `_decide_inner` と同じ＝訪問数
            // 降順の先頭）。`select_rule="q_min_n"` はこの並びに対して「訪問下限を満たす
            // グループの中で Q 最大」を採る（下限を満たすグループが無ければ先頭＝訪問数最多）。
            let floor = q_min_n_floor(eff_sims, opts.q_min_frac);
            let gn: Vec<f64> = groups.iter().map(|g| g.n).collect();
            let gq: Vec<f64> = groups.iter().map(|g| g.q).collect();
            let gi = select_index(&gn, &gq, opts.select_rule, floor);
            let mut rep = groups[gi].rep;
            // §20.7.8 の 4: `q_min_n` のときだけ、準備箱の枝を card_id で束ねた
            // グループ（`select_groups`）に対して訪問下限を測る（`visits` は不変）。
            if opts.select_rule == SelectRule::QMinN && !sel_groups.is_empty() {
                let sn: Vec<f64> = sel_groups.iter().map(|g| g.n).collect();
                let sq: Vec<f64> = sel_groups.iter().map(|g| g.q).collect();
                rep = sel_groups[select_index(&sn, &sq, SelectRule::QMinN, floor)].rep;
            }
            // 生成の温度サンプリング（序盤は訪問分布から引く）
            let early = opts.temp_turns > 0 && state.turn_count <= opts.temp_turns;
            // §7.73: 序盤より後も、1 手ごとに確率 `temp_late` で訪問分布から引く。
            // `temp_late` が 0 なら乱数を引かない（既定の生成と 1 ビットも同じ）。
            let late = !early
                && opts.temp_late > 0.0
                && rng.choice(&[1.0 - opts.temp_late.min(1.0), opts.temp_late.min(1.0)])? == 1;
            if early || late {
                let ns: Vec<f64> = groups.iter().map(|g| g.n).collect();
                let total: f64 = ns.iter().sum();
                if total > 0.0 {
                    let probs: Vec<f64> = ns.iter().map(|n| n / total).collect();
                    let greedy = rep;
                    let gi = rng.choice(&probs)?;
                    rep = groups[gi].rep;
                    // 貪欲に選ぶ手と同じ候補（等価手マージ後のグループ）か
                    let same = groups[gi].rep == greedy || groups[gi].idxs.contains(&greedy);
                    temp_pick = Some((if early { 1 } else { 2 }, !same));
                }
            }
            mv = Some(run.legal[rep].clone());
            chosen_idx = Some(rep);
        }
    }
    if mv.is_none() {
        mv = run.legal.first().cloned();
        chosen_idx = if run.legal.is_empty() { None } else { Some(0) };
    }
    // PV とコミットの継続を採る世界（§20.7.1）: 選んだ手の訪問が最も多い世界（同点は若い方）。
    let world_used = if worlds > 1 {
        chosen_idx.map(|i| {
            (0..worlds).fold(0usize, |best, w| {
                if per_world[w].n[i] > per_world[best].n[i] {
                    w
                } else {
                    best
                }
            })
        })
    } else {
        None
    };
    let pv_world = world_used.unwrap_or(0);
    // PV（主変化・§20.4）: 8 手または葉まで。単一世界は root の argmax(N)（＝今までどおり）、
    // 複数世界は「束ねた統計で選んだ手」から辿る（その世界の argmax とは限らないため）。
    let pv_first = if worlds > 1 {
        chosen_idx.and_then(|i| back[pv_world][i])
    } else {
        None
    };
    let pv = principal_variation_of(&runs[pv_world].nodes, pv_first, 8);
    // 棋譜ダンプの鍵はここで採る（Python `_decide_inner` の `record` と同じ位置＝残ドン掘り／
    // 残り起動の差し替えと先頭原始手化の**前**＝訪問分布の候補と突き合わせられる箱レベル）。
    let rec_sig = mv.as_ref().map(move_sig);
    let rec_k = mv
        .as_ref()
        .and_then(|m| m.payload())
        .and_then(|p| p.don_k_f64());

    // ⑤ 残ドン掘り／残り起動（腕 A・A2）
    let mut dig_override = false;
    let is_turn_end = mv
        .as_ref()
        .and_then(|m| m.action_type())
        == Some("TURN_END");
    if is_turn_end && !in_battle(&real) && !in_dialog(&mut real) {
        let legal = ctx.legal_actions(&mut real)?;
        if opts.residual_dig {
            if let Some(alt) = residual_dig_move(ctx, &mut real, name, &legal) {
                mv = Some(alt);
                dig_override = true;
            }
        }
        if !dig_override && opts.residual_activate.is_some() {
            if let Some(alt) = residual_activate_move(ctx, &mut real, name, &legal) {
                mv = Some(alt);
                dig_override = true;
                carry.resact_pending = true;
            }
        }
    }

    // ⑥ 箱コミット（木が選んだ箱の自分側の残り手順を確定）
    if opts.box_commit && !dig_override {
        if let Some(m) = mv.clone() {
            match m.action_type() {
                Some("DON_BOX") => {
                    let total = box_total(&m);
                    if total > 1 {
                        carry.commit = vec![Step::Box {
                            sig: move_sig_with(&m, opts.commit_sig_legacy),
                            left: total - 1,
                        }];
                    }
                }
                // §7.42 の 1: 登場／起動の後の対話の続きは効果の対話の段だけ＝既定は積まない（読み直す）。
                Some("PLAY") | Some("ACTIVATE_MAIN") if opts.window_carry_legacy => {
                    // 複数世界は木を回した世界（PV と同じ）で継続を作る＝乱数を引き直さない。
                    let steps = if worlds > 1 {
                        commit_play_dialog_in_world(
                            ctx,
                            &runs[pv_world].world,
                            name,
                            &m,
                            st,
                            opts.commit_sig_legacy,
                            opts.commit_hidden_legacy,
                        )
                    } else {
                        commit_play_dialog(
                            ctx,
                            state,
                            name,
                            &m,
                            opts.world_view(),
                            rng,
                            st,
                            opts.commit_sig_legacy,
                            opts.commit_hidden_legacy,
                        )?
                    };
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
                // 準備箱（§20.7.2）: 素の手より後（対話 → 攻撃箱）を機械実行へ積む。
                Some("SETUP_BOX") => {
                    let steps = commit_setup_box(
                        ctx,
                        state,
                        name,
                        &m,
                        opts.world_view(),
                        rng,
                        opts.commit_sig_legacy,
                        opts.commit_hidden_legacy,
                    )?;
                    if !steps.is_empty() {
                        carry.commit = steps;
                    }
                }
                _ => {}
            }
        }
    }

    // ⑦ 箱は実対局へは先頭原始手で出す
    let mv = mv.map(|m| don_box_first_primitive(&m));
    Ok(DecideOut {
        mv,
        sig: rec_sig,
        k: rec_k,
        kind: "main",
        legal: run.legal,
        n: run.n,
        q: run.q,
        p: Some(run.p),
        groups,
        select_groups: sel_groups,
        carry,
        budget_used: st.budget.used,
        budget_exhausted: st.budget.exhausted,
        pv,
        worlds,
        per_world,
        world_used,
        lethal: None,
        lethal_reply: reply,
        window_read: None,
        defense_proof: None,
        temp_pick,
    })
}

#[cfg(test)]
#[path = "tests_fair.rs"]
mod tests_fair;

#[cfg(test)]
#[path = "tests_commit_hidden.rs"]
mod tests_commit_hidden;

#[cfg(test)]
#[path = "tests_window_read.rs"]
mod tests_window_read;

#[cfg(test)]
#[path = "tests_defense_proof.rs"]
mod tests_defense_proof;

#[cfg(test)]
#[path = "tests_overflow_trigger.rs"]
mod tests_overflow_trigger;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn move_sig_keys_on_action_uuid_targets_selection() {
        let a = json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":"u1","target_ids":["t1"]}});
        let b = json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":"u1","target_ids":["t2"]}});
        assert_ne!(move_sig(&Move::from_json((a).clone())), move_sig(&Move::from_json((b).clone())));
        // don_k は sig に入らない（残回数はステップ側が持つ）
        let c = json!({"action_type":"DON_BOX","payload":{"uuid":"u1","don_k":2}});
        let d = json!({"action_type":"DON_BOX","payload":{"uuid":"u1","don_k":3}});
        assert_eq!(move_sig(&Move::from_json((c).clone())), move_sig(&Move::from_json((d).clone())));
        // 効果選択は selected_uuids と accepted で割れる
        let e = json!({"action_type":"RESOLVE_EFFECT_SELECTION","payload":{"selected_uuids":["x"]}});
        let f = json!({"action_type":"RESOLVE_EFFECT_SELECTION","payload":{"selected_uuids":[]}});
        assert_ne!(move_sig(&Move::from_json((e).clone())), move_sig(&Move::from_json((f).clone())));
    }

    /// WP `rs-commit-sig`: 戦闘の手は札を最上位の `card_uuid` に持つ＝署名に札が入らないと、
    /// commit の続きは「署名が一致する最初の合法手」＝手札の先頭のカウンターを打つ。
    #[test]
    fn move_sig_splits_counter_cards() {
        let c1 = Move::battle("SELECT_COUNTER", Some("h1".into()));
        let c2 = Move::battle("SELECT_COUNTER", Some("h2".into()));
        let pass = Move::battle("PASS", None);
        assert_ne!(move_sig(&c1), move_sig(&c2));
        assert_eq!(move_sig(&c2)[1], json!("h2"));
        assert_eq!(move_sig(&pass)[1], Value::Null);
        // commit の続きを照合すると、選んだ札（2 枚目）が返る
        let legal = vec![c1.clone(), c2.clone(), pass.clone()];
        assert_eq!(find_move(&legal, &move_sig(&c2)).unwrap().card_uuid(), Some("h2"));
        // 旧い署名（測定専用）は札が落ちて 1 枚目を返す＝修正前の挙動
        assert_eq!(move_sig_with(&c1, true), move_sig_with(&c2, true));
        assert_eq!(
            find_move_with(&legal, &move_sig_with(&c2, true), true).unwrap().card_uuid(),
            Some("h1")
        );
    }

    #[test]
    fn move_sig_splits_blocker_cards() {
        let b1 = Move::battle("SELECT_BLOCKER", Some("f1".into()));
        let b2 = Move::battle("SELECT_BLOCKER", Some("f2".into()));
        assert_ne!(move_sig(&b1), move_sig(&b2));
        let legal = vec![b1.clone(), b2.clone(), Move::battle("PASS", None)];
        assert_eq!(find_move(&legal, &move_sig(&b2)).unwrap().card_uuid(), Some("f2"));
        assert_eq!(move_sig_with(&b1, true), move_sig_with(&b2, true));
    }

    /// 並べ替えの上下（`position`）と準備箱の段の区切り（`selects`）も署名で割れる。
    /// それ以外の手の署名は今までの 5 つのまま（旧い署名と同じ）。
    #[test]
    fn move_sig_ext_only_when_needed() {
        let base = json!({"selected_uuids":["a","b"],"index":0,"accepted":true,
                          "position":"BOTTOM","declared_value":0});
        let mut top = base.clone();
        top["position"] = json!("TOP");
        let bot = Move::from_json(json!({"kind":"game","action_type":"RESOLVE_EFFECT_SELECTION","payload":base}));
        let top = Move::from_json(json!({"kind":"game","action_type":"RESOLVE_EFFECT_SELECTION","payload":top}));
        assert_ne!(move_sig(&bot), move_sig(&top));
        assert_eq!(move_sig(&bot), move_sig_with(&bot, true));
        assert_eq!(move_sig(&top).as_array().unwrap().len(), 6);
        assert_eq!(move_sig_with(&bot, true), move_sig_with(&top, true));
        let sb = |sel: Value| {
            Move::from_json(json!({"kind":"game","action_type":"SETUP_BOX",
                "payload":{"uuid":"u","selects":sel.clone(),"selected_uuids":["a","b"]}}))
        };
        let one = sb(json!([["a","b"]]));
        let two = sb(json!([["a"],["b"]]));
        assert_ne!(move_sig(&one), move_sig(&two));
        assert_eq!(move_sig(&one).as_array().unwrap().len(), 5);
        let ch = |i: i64| Move::from_json(json!({"kind":"game","action_type":"RESOLVE_EFFECT_SELECTION",
            "payload":{"selected_uuids":[],"index":i,"accepted":true,"position":"BOTTOM","declared_value":0}}));
        assert_ne!(move_sig(&ch(0)), move_sig(&ch(1)));
        assert_eq!(move_sig(&ch(0)), move_sig_with(&ch(0), true));
        let atk = Move::game("ATTACK", super::super::Payload::Attack { uuid: "u".into(), target: "t".into() });
        assert_eq!(move_sig(&atk), move_sig_with(&atk, true));
    }

    #[test]
    fn don_box_first_primitive_expands_the_head() {
        let attach = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":2}});
        assert_eq!(
            don_box_first_primitive(&Move::from_json((attach).clone())).to_json()["action_type"],
            Value::from("ATTACH_DON")
        );
        // k=0 のアタック箱は ATTACK をそのまま出す
        let atk = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":0,"target_ids":["t"]}});
        let out = don_box_first_primitive(&Move::from_json((atk).clone()));
        assert_eq!(out.to_json()["action_type"], Value::from("ATTACK"));
        assert_eq!(out.to_json()["payload"]["target_ids"], json!(["t"]));
        // 付与つきの攻撃形はまず付与
        let both = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":1,"target_ids":["t"]}});
        assert_eq!(
            don_box_first_primitive(&Move::from_json((both).clone())).to_json()["action_type"],
            Value::from("ATTACH_DON")
        );
        // DON_BOX 以外は素通し
        let end = json!({"action_type":"TURN_END","payload":{}});
        assert_eq!(don_box_first_primitive(&Move::from_json((end).clone())), Move::from_json(end));
    }

    #[test]
    fn trace_to_steps_keeps_only_my_moves_and_counts_boxes() {
        let mine = json!({"action_type":"DON_BOX","payload":{"uuid":"u","don_k":2,"target_ids":["t"]}});
        let plain = json!({"action_type":"PASS","payload":{}});
        let trace = vec![
            (Seat::P1, Move::from_json(mine.clone()), Vec::new()),
            (Seat::P2, Move::from_json(plain.clone()), Vec::new()),
            (Seat::P1, Move::from_json(plain.clone()), Vec::new()),
        ];
        let steps = trace_to_steps(&trace, Seat::P1, false, None);
        assert_eq!(steps.len(), 2);
        match &steps[0] {
            Step::Box { left, .. } => assert_eq!(*left, 3), // 付与 2 ＋ 攻撃 1
            other => panic!("箱の形でない: {other:?}"),
        }
        assert_eq!(steps[1], Step::Sig(move_sig(&Move::from_json((plain).clone()))));
    }

    /// 箱コミットの残回数: 付与 k 枚＋攻撃形は「付与 → … → 攻撃」で消化し、
    /// 実盤面の候補から sig が消えたら（契約違反）1 手も返さない。
    #[test]
    fn box_countdown_shrinks_and_ends_with_the_attack() {
        let (masters, state) = crate::testkit::BoardBuilder::new().build();
        let net = crate::net::LoadedNet {
            weights: Default::default(),
            tab: Vec::new(),
            vocab: Default::default(),
            statics: Default::default(),
        };
        let ctx = Ctx {
            masters: &masters,
            net: &net,
            opts: Default::default(),
            box_battle: true,
            box_dialog: true,
            quiesce: true,
            quiesce_max_plies: QUIESCE_MAX_PLIES,
        };
        let mut s = Session::new(state);
        // `commit_step` は現在の合法手を引くが、この盤面では候補が出ない（＝契約違反）。
        // 残り手順が消化されないこと（`None`）だけを見る＝箱ごと破棄する側の分岐。
        let sig = move_sig(&Move::from_json(json!({"action_type": "DON_BOX",
                                   "payload": {"uuid": "u", "target_ids": ["t"]}})));
        let mut steps = vec![Step::Box { sig: sig.clone(), left: 2 }];
        assert!(commit_step(&ctx, &mut s, Seat::P1, &mut steps, false).unwrap().is_none());

        // カウントダウンの算術そのもの（Python `_commit_step` の箱の分岐と同じ規則）:
        // 残り n>1 は付与・n<=1 かつ対象ありは攻撃。
        let head = |left: i64| -> Move {
            let uuid = sig.get(1).cloned().unwrap();
            let tgts = sig.get(2).and_then(Value::as_array).cloned().unwrap();
            Move::from_json(if !tgts.is_empty() && left <= 1 {
                json!({"kind":"game","action_type":"ATTACK",
                       "payload":{"uuid":uuid,"target_ids":tgts}})
            } else {
                json!({"kind":"game","action_type":"ATTACH_DON","payload":{"uuid":uuid}})
            })
        };
        assert_eq!(head(2).to_json()["action_type"], Value::from("ATTACH_DON"));
        assert_eq!(head(1).to_json()["action_type"], Value::from("ATTACK"));
        // 生成側（木が箱を選んだ直後）の残回数＝総原始手数 − 1。
        let box_move = json!({"action_type":"DON_BOX",
                              "payload":{"uuid":"u","don_k":2,"target_ids":["t"]}});
        assert_eq!(box_total(&Move::from_json((box_move).clone())), 3);
    }

    /// 等価手マージ: 同名カードの別実体（card_id が同じ）は 1 グループへ畳み、
    /// 訪問数を合算して n 降順（同数は列挙順）に並べる。代表は**グループ内の先頭**。
    #[test]
    fn merge_root_stats_folds_equivalent_copies() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        let a = b.put_hand(Seat::P1, M_CHAR);
        let c = b.put_hand(Seat::P1, M_CHAR); // 同じカードの 2 枚目
        let (masters, state) = b.build();
        let ua = state.card(a).uuid.clone();
        let uc = state.card(c).uuid.clone();
        let play = |u: &str| Move::from_json(json!({"kind":"game","action_type":"PLAY","payload":{"uuid":u}}));
        let end = Move::from_json(json!({"kind":"game","action_type":"TURN_END","payload":{}}));
        let legal = vec![play(&ua), end.clone(), play(&uc)];
        let n = [30.0, 38.0, 30.0];
        let q = [0.5, -0.5, 0.1];
        let groups = merge_root_stats(&state, &masters, &legal, &n, &q);
        assert_eq!(groups.len(), 2, "同名の 2 枚は 1 グループ");
        // 合算 60 の PLAY が 38 の TURN_END を上回る（素の argmax(N) なら負けていた）
        assert_eq!(groups[0].n, 60.0);
        assert_eq!(groups[0].rep, 0, "代表は列挙順の先頭");
        assert_eq!(groups[0].idxs, vec![0, 2]);
        // Q は訪問加重平均
        assert!((groups[0].q - 0.3).abs() < 1e-12);
        assert_eq!(groups[1].n, 38.0);
    }

    /// 別実体でも card_id が違えば別グループ（等価キーは card_id 基準）。
    #[test]
    fn merge_root_stats_keeps_different_cards_apart() {
        use crate::testkit::{BoardBuilder, M_BLOCKER, M_CHAR};
        let mut b = BoardBuilder::new();
        let a = b.put_hand(Seat::P1, M_CHAR);
        let c = b.put_hand(Seat::P1, M_BLOCKER);
        let (masters, state) = b.build();
        let play = |u: &str| Move::from_json(json!({"kind":"game","action_type":"PLAY","payload":{"uuid":u}}));
        let legal = vec![
            play(&state.card(a).uuid.clone()),
            play(&state.card(c).uuid.clone()),
        ];
        let groups = merge_root_stats(&state, &masters, &legal, &[1.0, 2.0], &[0.0, 0.0]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].rep, 1, "n 降順");
    }

    /// DON_BOX は `don_k` が違えば別挙動＝別グループ（`_move_equiv_key` の最後の欄）。
    #[test]
    fn merge_root_stats_splits_boxes_by_don_k() {
        let (masters, state) = crate::testkit::BoardBuilder::new().build();
        let bx = |k: i64| Move::from_json(json!({"kind":"game","action_type":"DON_BOX",
                                 "payload":{"uuid":"u","target_ids":[],"don_k":k}}));
        let legal = vec![bx(1), bx(2), bx(1)];
        let groups = merge_root_stats(&state, &masters, &legal, &[1.0, 5.0, 2.0], &[0.0; 3]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].n, 5.0); // k=2
        assert_eq!(groups[1].idxs, vec![0, 2]); // k=1 は 2 本合算
    }

    /// 温度サンプル: 訪問数の分布から index を引く（累積分布 → searchsorted）。
    #[test]
    fn temperature_samples_from_the_visit_distribution() {
        use super::super::rng::choice_from_uniform;
        let probs = [0.25, 0.5, 0.25];
        assert_eq!(choice_from_uniform(&probs, 0.0), 0);
        assert_eq!(choice_from_uniform(&probs, 0.3), 1);
        assert_eq!(choice_from_uniform(&probs, 0.8), 2);
        assert_eq!(choice_from_uniform(&probs, 0.999), 2);
    }

    /// 付与先の方針: "low"＝このターン攻撃できるキャラの最低パワー・"high"＝最高パワー。
    #[test]
    fn attach_policy_picks_low_or_high() {
        let cands = vec![
            ("a".to_string(), 5000, false),
            ("b".to_string(), 3000, true),
            ("c".to_string(), 7000, true),
        ];
        assert_eq!(pick_attach_target(&cands, "low").as_deref(), Some("b"));
        assert_eq!(pick_attach_target(&cands, "high").as_deref(), Some("c"));
        // 攻撃できるキャラが無ければ全体の最低パワー
        let rest = vec![("a".to_string(), 5000, false), ("b".to_string(), 3000, false)];
        assert_eq!(pick_attach_target(&rest, "low").as_deref(), Some("b"));
        assert_eq!(pick_attach_target(&[], "low"), None);
    }

    // --- PV（§20.4・思考ログ）------------------------------------------------------

    /// テスト用のゼロ重みネット（forward は全て 0＝priors 一様・value 0）。vocab は空＝
    /// 全カードが PAD 行（index 0）を指す（盤面差は読まないが、形は正しいので木は普通に回る）。
    /// `principal_variation` の検証に要るのは「木が壊れず回ること」だけなので十分。
    pub(in crate::search) fn zero_net() -> crate::net::LoadedNet {
        use crate::encode::{Vocab, ABILITY_DIM, MAX_AB, R_DIM, STATS_DIM};
        use crate::net::nrel::{CardStatics, D_AB};
        use crate::net::{Mat, NRelWeights, D_C, D_PIN, D_R, D_T, D_X, D_Z};
        let mat = |rows: usize, cols: usize| Mat::new(rows, cols, vec![0.0f32; rows * cols]);
        let hidden = 8usize;
        let weights = NRelWeights {
            wa: mat(ABILITY_DIM, D_AB),
            ba: vec![0.0; D_AB],
            wt: mat(D_X, D_T),
            bt: vec![0.0; D_T],
            wr: mat(2 * D_T + R_DIM, D_R),
            br: vec![0.0; D_R],
            wc: mat(2 * D_T + R_DIM, D_C),
            bc: vec![0.0; D_C],
            w1: mat(D_Z, hidden),
            b1: vec![0.0; hidden],
            w2: mat(hidden, 64),
            b2: vec![0.0; 64],
            wv: mat(64, 1),
            bv: vec![0.0; 1],
            wp1: mat(D_PIN, 64),
            bp1: vec![0.0; 64],
            wp2: mat(64, 1),
            bp2: vec![0.0; 1],
            hidden,
            ablate: Default::default(),
            meta_json: String::new(),
            vocab_ids: Vec::new(),
            enc_version: crate::encode::ENC_VERSION,
            deck_v14: false,
        };
        let vocab = Vocab::from_ids(&[]);
        let tables = crate::encode::EffTables {
            n: 1,
            stats: vec![0.0; STATS_DIM],
            ab: vec![0.0; MAX_AB * ABILITY_DIM],
            abm: vec![0.0; MAX_AB],
            pwr: vec![0.0; 1],
            isl: vec![0.0; 1],
        };
        let tab = crate::net::card_table(&weights, &tables).expect("zero net: card_table");
        let statics = CardStatics { pwr: vec![0.0], isl: vec![0.0], ret_don: vec![0.0] };
        crate::net::LoadedNet { weights, tab, vocab, statics }
    }

    /// PV の先頭は「木が返した手」と一致し、長さは 8 以下（§20.4 の受け入れ）。
    #[test]
    fn pv_head_matches_the_returned_move_and_is_bounded() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR); // コスト2・3000（PLAY と TURN_END の 2 候補ができる）
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions { sims: 8, ..DecideOptions::default() };
        let mut rng = crate::search::Pcg32SearchRng::new(1);
        let carry = DecideCarry::default();
        let out = decide(&masters, &net, &state, Seat::P1, &opts, &mut rng, &carry).unwrap();
        assert_eq!(out.kind, "main");
        assert!(!out.pv.is_empty(), "main の決定は少なくとも 1 手の PV を持つ");
        assert!(out.pv.len() <= 8, "PV は 8 手まで");
        assert_eq!(out.mv.as_ref(), Some(&out.pv[0].mv), "PV の先頭は返した手と同じ");
        assert_eq!(out.pv[0].seat, Seat::P1);
    }

    // --- 探索の設定（§20.5・WP `rs-search-a`）------------------------------------

    /// 既定値を**明示して渡しても**出力は 1 bit も変わらない（serve・生成・アリーナは無影響）。
    #[test]
    fn spelling_out_the_defaults_changes_nothing() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let carry = DecideCarry::default();
        let base = DecideOptions { sims: 8, ..DecideOptions::default() };
        let spelled = DecideOptions {
            sims: 8,
            select_rule: SelectRule::Visits,
            q_min_frac: Q_MIN_FRAC,
            root_prior_temp: 1.0,
            ..DecideOptions::default()
        };
        let a = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &base,
            &mut crate::search::Pcg32SearchRng::new(7),
            &carry,
        )
        .unwrap();
        let c = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &spelled,
            &mut crate::search::Pcg32SearchRng::new(7),
            &carry,
        )
        .unwrap();
        assert_eq!(a.sig, c.sig);
        assert_eq!(a.n, c.n);
        assert_eq!(a.q, c.q);
        assert_eq!(a.p, c.p);
        assert_eq!(a.pv.len(), c.pv.len());
    }

    /// `select_rule="q_min_n"` は「訪問下限を満たすグループの中で Q 最大」を出す
    /// （decide の配線＝根の集計と選択が同じ並びを見ていること）。
    #[test]
    fn q_min_n_selects_the_group_with_the_best_q_above_the_floor() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions {
            sims: 8,
            select_rule: SelectRule::QMinN,
            ..DecideOptions::default()
        };
        let mut rng = crate::search::Pcg32SearchRng::new(7);
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &opts,
            &mut rng,
            &DecideCarry::default(),
        )
        .unwrap();
        assert_eq!(out.kind, "main");
        assert!(out.groups.len() >= 2, "PLAY と TURN_END の 2 候補は出るはず");
        let gn: Vec<f64> = out.groups.iter().map(|g| g.n).collect();
        let gq: Vec<f64> = out.groups.iter().map(|g| g.q).collect();
        let want = select_index(&gn, &gq, SelectRule::QMinN, q_min_n_floor(8, Q_MIN_FRAC));
        let rep = out.groups[want].rep;
        assert_eq!(out.sig.as_ref(), Some(&move_sig(&out.legal[rep])));
        // 選んだグループは下限を満たす手の中で Q 最大（下限を満たす手が居る盤面）。
        let floor = q_min_n_floor(8, Q_MIN_FRAC);
        assert!(gn.iter().any(|n| *n >= floor));
        for (i, n) in gn.iter().enumerate() {
            if *n >= floor {
                assert!(gq[want] >= gq[i], "Q 最大でない: {:?} / {:?}", gn, gq);
            }
        }
    }

    /// `root_prior_temp` は根の P だけを丸め、和は 1 のまま（返り値の `stats.P`）。
    #[test]
    fn root_prior_temp_keeps_the_root_prior_a_distribution() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        let net = zero_net();
        let opts = DecideOptions {
            sims: 8,
            root_prior_temp: 2.0,
            ..DecideOptions::default()
        };
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &opts,
            &mut crate::search::Pcg32SearchRng::new(7),
            &DecideCarry::default(),
        )
        .unwrap();
        let p = out.p.expect("kind=main は P を返す");
        assert!(p.len() >= 2);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "根の P の和が 1 でない: {s}");
        assert!(p.iter().all(|x| *x >= 0.0));
    }

    /// kind=window（戦闘窓の根畳み）のとき PV は空（§20.4）。
    #[test]
    fn pv_is_empty_when_kind_is_window() {
        use crate::model::ActiveBattle;
        use crate::testkit::{BoardBuilder, M_CHAR};
        // カウンターの段（2026-10-24 まではフェイズ MAIN のまま戦闘を置いていた＝規則の欠陥で戦闘が残った
        // 盤面と同じ形。今は MAIN の判断点で窓の道に入らない〔§7.44 の (A)〕ので段を明示する）。
        let mut b = BoardBuilder::new().phase(crate::model::Phase::BattleCounter);
        let atk = b.put_field(Seat::P1, M_CHAR);
        let (masters, mut state) = b.build();
        let target = state.player(Seat::P2).leader.expect("p2 leader");
        // p2 は手札・場が空＝ブロッカー無し・カウンター無し＝合法手は PASS 1 つだけ
        // （`window_choice` の legal.len()==1 分岐＝ネットの forward すら要らない）。
        state.active_battle = Some(ActiveBattle {
            attacker: atk,
            target,
            attacker_owner: Seat::P1,
            target_owner: Seat::P2,
            counter_buff: 0,
        });
        let net = zero_net();
        let opts = DecideOptions::default();
        let mut rng = crate::search::Pcg32SearchRng::new(2);
        let carry = DecideCarry::default();
        let out = decide(&masters, &net, &state, Seat::P2, &opts, &mut rng, &carry).unwrap();
        assert_eq!(out.kind, "window");
        assert!(out.pv.is_empty(), "window の決定に PV は無い");
    }

    // --- 複数世界（§20.7.1・WP `rs-pimc-worlds`）------------------------------------

    /// 世界を引ける盤面（相手に手札と山札がある＝`determinize` が実際に混ぜる）。
    fn worlds_board() -> (crate::model::MasterTable, GameState) {
        use crate::testkit::{BoardBuilder, M_BLOCKER, M_CHAR, M_EVENT};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR); // PLAY と TURN_END の 2 候補
        b.dons(Seat::P1, "active", 2);
        // 相手の伏せ札（手札 2・山札 3）＝世界サンプルの pool が 5 枚
        b.put_hand(Seat::P2, M_CHAR);
        b.put_hand(Seat::P2, M_EVENT);
        b.put_deck(Seat::P2, M_BLOCKER);
        b.put_deck(Seat::P2, M_CHAR);
        b.put_deck(Seat::P2, M_EVENT);
        b.build()
    }

    fn worlds_opts(worlds: usize, seed: Option<u64>) -> DecideOptions {
        DecideOptions { sims: 8, worlds, search_seed: seed, ..DecideOptions::default() }
    }

    /// `worlds=1`（明示・既定どちらも）は今までの単一世界と **1 bit も変わらない**。
    /// `search_seed` を渡しても変わらない（世界 0 は呼び出し側の rng で引くため）。
    #[test]
    fn worlds_one_is_bit_identical_to_the_default() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let run = |opts: &DecideOptions| {
            decide(
                &masters,
                &net,
                &state,
                Seat::P1,
                opts,
                &mut crate::search::Pcg32SearchRng::new(11),
                &carry,
            )
            .unwrap()
        };
        let base = run(&DecideOptions { sims: 8, ..DecideOptions::default() });
        for opts in [worlds_opts(1, None), worlds_opts(1, Some(11)), worlds_opts(4, None)] {
            // `worlds=4` でも `search_seed` が無ければ 1 本に落ちる（＝既定と同じ）。
            let got = run(&opts);
            assert_eq!(got.mv, base.mv);
            assert_eq!(got.sig, base.sig);
            assert_eq!(got.n, base.n);
            assert_eq!(got.q, base.q);
            assert_eq!(got.p, base.p);
            assert_eq!(got.pv.len(), base.pv.len());
            assert_eq!(got.worlds, 1);
            assert!(got.per_world.is_empty(), "単一世界は per_world を持たない");
            assert!(got.world_used.is_none());
        }
    }

    /// `worlds=K`: 根の訪問数の合計が `K * sims`・`per_world` が K 本・
    /// 世界 0 の統計は単一世界で回したときと同じ（seed の派生が seed+i である証拠）。
    #[test]
    fn worlds_k_sums_visits_and_keeps_world_zero() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let single = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &worlds_opts(1, Some(11)),
            &mut crate::search::Pcg32SearchRng::new(11),
            &carry,
        )
        .unwrap();
        let k = 4usize;
        let out = decide(
            &masters,
            &net,
            &state,
            Seat::P1,
            &worlds_opts(k, Some(11)),
            &mut crate::search::Pcg32SearchRng::new(11),
            &carry,
        )
        .unwrap();
        assert_eq!(out.kind, "main");
        assert_eq!(out.worlds, k);
        assert_eq!(out.per_world.len(), k, "per_world の数は K");
        let total: f64 = out.n.iter().sum();
        assert_eq!(total, (k * 8) as f64, "根の訪問の和は K×sims");
        // 世界 0 は単一世界のときと同じ木（乱数も同じ列）＝統計が一致する。
        assert_eq!(out.per_world[0].n, single.n);
        assert_eq!(out.per_world[0].q, single.q);
        assert_eq!(out.p, single.p, "P は世界 0 のもの");
        // 各世界の訪問の和も sims（＝K 本とも同じ sims で回っている）。
        for (i, w) in out.per_world.iter().enumerate() {
            let n: f64 = w.n.iter().sum();
            assert_eq!(n, 8.0, "世界 {i} の訪問の和");
            assert_eq!(w.seed, 11 + i as u64, "世界 {i} の seed は search_seed+i");
            assert_eq!(w.unmapped, 0, "世界 {i} の legal が世界 0 と対応しない");
            assert!(!w.p_differs, "世界 {i} の P が世界 0 と違う");
        }
        // PV とコミットを採った世界は「選んだ手の訪問が最も多い世界」。
        let used = out.world_used.expect("worlds>1 は world_used を持つ");
        assert!(used < k);
        assert!(!out.pv.is_empty());
        assert_eq!(out.mv.as_ref(), Some(&out.pv[0].mv), "PV の先頭は返した手と同じ");
    }

    /// 同じ seed で 2 回回せば同じ出力（スレッドの終わる順に依らない＝再現性）。
    #[test]
    fn worlds_are_reproducible_for_the_same_seed() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        let run = || {
            decide(
                &masters,
                &net,
                &state,
                Seat::P1,
                &worlds_opts(8, Some(5)),
                &mut crate::search::Pcg32SearchRng::new(5),
                &carry,
            )
            .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.mv, b.mv);
        assert_eq!(a.n, b.n);
        assert_eq!(a.q, b.q);
        assert_eq!(a.world_used, b.world_used);
        assert_eq!(a.per_world.len(), b.per_world.len());
        for (x, y) in a.per_world.iter().zip(&b.per_world) {
            assert_eq!(x.seed, y.seed);
            assert_eq!(x.n, y.n);
            assert_eq!(x.q, y.q);
            assert_eq!(x.best, y.best);
        }
        for (x, y) in a.pv.iter().zip(&b.pv) {
            assert_eq!(x.mv, y.mv);
            assert_eq!(x.n, y.n);
        }
    }

    /// 世界を同時に回すスレッドの上限（§7.55・`world_threads`）は打つ手も出力も変えない:
    /// 世界 2 本・4 本 × 上限 0（世界ごとに 1 本）・1・2・3・4・8 で `DecideOut` が丸ごと一致する
    /// （`Debug` の文字列で比べる＝f64 も bit 単位・PV・per_world・予算の実績まで）。
    #[test]
    fn world_threads_do_not_change_the_output() {
        let (masters, state) = worlds_board();
        let net = zero_net();
        let carry = DecideCarry::default();
        for worlds in [2usize, 4] {
            let run = |threads: usize| {
                let opts = DecideOptions { world_threads: threads, ..worlds_opts(worlds, Some(7)) };
                let out = decide(
                    &masters,
                    &net,
                    &state,
                    Seat::P1,
                    &opts,
                    &mut crate::search::Pcg32SearchRng::new(7),
                    &carry,
                )
                .unwrap();
                format!("{out:?}")
            };
            let base = run(0);
            assert!(base.contains(&format!("worlds: {worlds}")), "世界 {worlds} 本で回っていない");
            for threads in [1usize, 2, 3, 4, 8] {
                assert_eq!(run(threads), base, "worlds={worlds} world_threads={threads}");
            }
        }
    }

    /// 束ねの算術そのもの（N は和・Q は N 重みの平均）を [`merge_worlds`] で直に見る。
    #[test]
    fn merge_worlds_sums_n_and_weights_q_by_n() {
        let legal = vec![
            Move::from_json(json!({"action_type":"PLAY","payload":{"uuid":"a"}})),
            Move::from_json(json!({"action_type":"TURN_END","payload":{}})),
        ];
        let mk = |seed: u64, n: Vec<f64>, q: Vec<f64>| WorldRun {
            seed,
            run: RunOut {
                best: Some(legal[0].clone()),
                legal: legal.clone(),
                n,
                q,
                p: vec![0.5, 0.5],
            },
            nodes: Vec::new(),
            world: crate::testkit::sample_state(),
            reply: None,
        };
        let runs = vec![
            mk(0, vec![6.0, 2.0], vec![1.0, -1.0]),
            mk(1, vec![2.0, 6.0], vec![0.0, 0.5]),
        ];
        let opts = DecideOptions { sims: 8, worlds: 2, ..DecideOptions::default() };
        let (merged, per_world, back) = merge_worlds(&runs, &opts);
        assert_eq!(merged.n, vec![8.0, 8.0]);
        // (6*1.0 + 2*0.0)/8 = 0.75 ／ (2*-1.0 + 6*0.5)/8 = 0.125
        assert_eq!(merged.q, vec![0.75, 0.125]);
        assert_eq!(merged.p, vec![0.5, 0.5], "P は世界 0 のもの");
        // 訪問数最多（既定 visits）は同点＝添字が小さい方。
        assert_eq!(merged.best, Some(legal[0].clone()));
        assert_eq!(per_world.len(), 2);
        assert_eq!(per_world[0].best, Some(0));
        assert_eq!(per_world[1].best, Some(1));
        assert_eq!(back[1], vec![Some(0), Some(1)], "並びが同じなら恒等写像");
    }

    // --- 全部見える探索（`search_efficiency_plan.md` §7.10・WP `rs-omniscient`）---------------

    /// 並びを引いた回数を数える rng（中身は PCG32）。世界サンプルを引いたかどうかの証拠。
    struct CountingRng {
        inner: crate::search::Pcg32SearchRng,
        shuffles: usize,
    }

    impl SearchRng for CountingRng {
        fn shuffle_order(&mut self, pool: &[String]) -> Result<Vec<String>, EngineError> {
            self.shuffles += 1;
            self.inner.shuffle_order(pool)
        }
        fn dirichlet(&mut self, alpha: f64, n: usize) -> Result<Vec<f64>, EngineError> {
            self.inner.dirichlet(alpha, n)
        }
        fn choice(&mut self, probs: &[f64]) -> Result<usize, EngineError> {
            self.inner.choice(probs)
        }
    }

    fn omni_run(
        masters: &crate::model::MasterTable,
        state: &GameState,
        opts: &DecideOptions,
    ) -> (DecideOut, usize) {
        let net = zero_net();
        let mut rng = CountingRng { inner: crate::search::Pcg32SearchRng::new(11), shuffles: 0 };
        let out =
            decide(masters, &net, state, Seat::P1, opts, &mut rng, &DecideCarry::default()).unwrap();
        (out, rng.shuffles)
    }

    fn assert_same_out(got: &DecideOut, base: &DecideOut) {
        assert_eq!(got.kind, base.kind);
        assert_eq!(got.mv, base.mv);
        assert_eq!(got.sig, base.sig);
        assert_eq!(got.legal, base.legal);
        assert_eq!(got.n, base.n);
        assert_eq!(got.q, base.q);
        assert_eq!(got.p, base.p);
        assert_eq!(got.pv.len(), base.pv.len());
        assert_eq!(got.worlds, base.worlds);
    }

    /// (a) `omniscient=false` を明示しても既定と 1 bit も変わらない（世界も今までどおり引く）。
    #[test]
    fn omniscient_false_is_bit_identical_to_the_default() {
        let (masters, state) = worlds_board();
        let base_opts = DecideOptions { sims: 8, ..DecideOptions::default() };
        assert!(!base_opts.omniscient, "既定は false");
        let (base, base_shuffles) = omni_run(&masters, &state, &base_opts);
        let (got, got_shuffles) =
            omni_run(&masters, &state, &DecideOptions { omniscient: false, ..base_opts.clone() });
        assert_same_out(&got, &base);
        assert_eq!(got_shuffles, base_shuffles);
        assert!(base_shuffles >= 1, "相手の伏せ札がある盤面では既定は世界を引く");
    }

    /// (b) 相手の手札＋山札が空の盤面では、引き直す物が無い＝true でも false と同一。
    #[test]
    fn omniscient_matches_default_when_the_opponent_has_no_hidden_cards() {
        use crate::testkit::{BoardBuilder, M_CHAR};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.dons(Seat::P1, "active", 2);
        let (masters, state) = b.build();
        assert!(state.player(Seat::P2).hand.is_empty() && state.player(Seat::P2).deck.is_empty());
        let base_opts = DecideOptions { sims: 8, ..DecideOptions::default() };
        let (base, base_shuffles) = omni_run(&masters, &state, &base_opts);
        let (got, got_shuffles) =
            omni_run(&masters, &state, &DecideOptions { omniscient: true, ..base_opts });
        assert_same_out(&got, &base);
        assert_eq!((base_shuffles, got_shuffles), (0, 0), "pool が空なら並びは引かない");
    }

    /// (c) true なら世界を引く関数（並び）を一度も呼ばない＝探索は本当の盤面で読む。
    /// `worlds>1`（`search_seed` あり）でも 1 本として扱う。
    #[test]
    fn omniscient_never_samples_a_world() {
        let (masters, state) = worlds_board();
        for worlds in [1usize, 4] {
            let opts = DecideOptions { omniscient: true, ..worlds_opts(worlds, Some(11)) };
            assert_eq!(opts.effective_worlds(), 1);
            let (out, shuffles) = omni_run(&masters, &state, &opts);
            assert_eq!(out.kind, "main");
            assert_eq!(shuffles, 0, "worlds={worlds}: 世界サンプルを引いた");
            assert_eq!(out.worlds, 1);
            assert!(out.per_world.is_empty());
        }
        // 引き直さない世界＝本当の盤面そのもの（`sample_world` の契約）。
        let mut rng = CountingRng { inner: crate::search::Pcg32SearchRng::new(3), shuffles: 0 };
        let view = DecideOptions { omniscient: true, ..DecideOptions::default() }.world_view();
        let world = sample_world(&state, Seat::P1, view, &mut rng).unwrap();
        assert_eq!(rng.shuffles, 0);
        assert_eq!(world.player(Seat::P2).hand, state.player(Seat::P2).hand);
        assert_eq!(world.player(Seat::P2).deck, state.player(Seat::P2).deck);
        assert_eq!(world.player(Seat::P1).deck, state.player(Seat::P1).deck);
    }

    // --- 見える範囲を分けた探索（`search_efficiency_plan.md` §7.11・WP `rs-reveal`）---------------

    /// 自分（P1）: 手札 1・山札 5・ライフ 4（位置 1 が表向き）・場 1／相手（P2）: 手札 2・山札 4・
    /// ライフ 3（位置 2 が表向き）。
    fn reveal_board() -> (crate::model::MasterTable, GameState) {
        use crate::testkit::{BoardBuilder, M_BLOCKER, M_CHAR, M_EVENT};
        let mut b = BoardBuilder::new();
        b.put_hand(Seat::P1, M_CHAR);
        b.put_field(Seat::P1, M_BLOCKER);
        b.dons(Seat::P1, "active", 2);
        for m in [M_CHAR, M_EVENT, M_BLOCKER, M_CHAR, M_EVENT] {
            b.put_deck(Seat::P1, m);
        }
        for i in 0..4 {
            let c = b.put_life(Seat::P1, M_CHAR);
            b.card_mut(c).is_face_up = i == 1;
        }
        b.put_hand(Seat::P2, M_CHAR);
        b.put_hand(Seat::P2, M_EVENT);
        for m in [M_BLOCKER, M_CHAR, M_EVENT, M_CHAR] {
            b.put_deck(Seat::P2, m);
        }
        for i in 0..3 {
            let c = b.put_life(Seat::P2, M_EVENT);
            b.card_mut(c).is_face_up = i == 2;
        }
        b.build()
    }

    fn reveal_view(reveal: Reveal) -> WorldView {
        DecideOptions { reveal, ..DecideOptions::default() }.world_view()
    }

    fn sorted(mut v: Vec<CardIdx>) -> Vec<CardIdx> {
        v.sort_unstable();
        v
    }

    /// 裏向きのライフだけ（位置つき）・表向きのライフだけ（位置つき）。
    fn life_split(st: &GameState, seat: Seat) -> (Vec<CardIdx>, Vec<(usize, CardIdx)>) {
        let life = &st.player(seat).life;
        let down = life.iter().copied().filter(|c| !st.card(*c).is_face_up).collect();
        let up = life
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, c)| st.card(*c).is_face_up)
            .collect();
        (down, up)
    }

    /// 公開領域（場・トラッシュ・ステージ・リーダー・一時領域・ドン!!）は両席とも本当のまま。
    fn assert_public_kept(w: &GameState, s: &GameState) {
        for seat in [Seat::P1, Seat::P2] {
            let (a, b) = (w.player(seat), s.player(seat));
            assert_eq!(a.field, b.field);
            assert_eq!(a.trash, b.trash);
            assert_eq!(a.stage, b.stage);
            assert_eq!(a.leader, b.leader);
            assert_eq!(a.temp_zone, b.temp_zone);
            assert_eq!(a.don_active, b.don_active);
            assert_eq!(a.don_deck, b.don_deck);
        }
    }

    /// (a) `reveal` の既定は `fair`（§7.15）＝既定の世界は `determinize_fair_with` と同じ rng の出目で
    /// 1 bit も同じ・decide も明示の `Fair` と同一。`peek` は 2026-10-21 までの既定（`determinize_with`）と
    /// 1 bit も同じ。
    #[test]
    fn reveal_default_is_fair_and_peek_is_the_old_default() {
        let (masters, state) = reveal_board();
        assert_eq!(DecideOptions::default().reveal, Reveal::Fair);
        for seed in 0..8u64 {
            let mut r1 = crate::search::Pcg32SearchRng::new(seed);
            let mut r2 = crate::search::Pcg32SearchRng::new(seed);
            let a = sample_world(&state, Seat::P1, DecideOptions::default().world_view(), &mut r1).unwrap();
            let b = super::super::determinize::determinize_fair_with(&state, Seat::P1, &mut r2).unwrap();
            assert_eq!(a, b);
            let mut r3 = crate::search::Pcg32SearchRng::new(seed);
            let mut r4 = crate::search::Pcg32SearchRng::new(seed);
            let c = sample_world(&state, Seat::P1, reveal_view(Reveal::Peek), &mut r3).unwrap();
            let d = super::super::determinize::determinize_with(&state, Seat::P1, &mut r4).unwrap();
            assert_eq!(c, d);
            // peek は自分の山札・裏向きのライフを本物のまま読む。
            assert_eq!(c.player(Seat::P1), state.player(Seat::P1));
            assert_eq!(c.player(Seat::P2).life, state.player(Seat::P2).life);
        }
        let base_opts = DecideOptions { sims: 8, ..DecideOptions::default() };
        let (base, base_shuffles) = omni_run(&masters, &state, &base_opts);
        let (got, got_shuffles) =
            omni_run(&masters, &state, &DecideOptions { reveal: Reveal::Fair, ..base_opts });
        assert_same_out(&got, &base);
        assert_eq!(got_shuffles, base_shuffles);
    }

    /// (b) `opp_hand`: 相手の手札は本当のまま・相手の山札だけ混ざる（自分の物・相手のライフは動かない）。
    #[test]
    fn reveal_opp_hand_keeps_the_hand_and_shuffles_the_deck() {
        let (_m, state) = reveal_board();
        let mut deck_moved = false;
        for seed in 0..16u64 {
            let mut rng = CountingRng { inner: crate::search::Pcg32SearchRng::new(seed), shuffles: 0 };
            let w = sample_world(&state, Seat::P1, reveal_view(Reveal::OppHand), &mut rng).unwrap();
            assert_eq!(rng.shuffles, 1, "相手の山札だけで 1 回");
            let (o, t) = (w.player(Seat::P2), state.player(Seat::P2));
            assert_eq!(o.hand, t.hand, "相手の手札は本当のまま");
            assert_eq!(o.life, t.life, "相手のライフは本当のまま");
            assert_eq!(sorted(o.deck.clone()), sorted(t.deck.clone()));
            deck_moved |= o.deck != t.deck;
            assert_eq!(w.player(Seat::P1), state.player(Seat::P1), "自分の物は動かない");
            assert_public_kept(&w, &state);
        }
        assert!(deck_moved, "16 本の種のどれかで相手の山札の並びが変わる");
        // 山札が空なら乱数を消費しない。
        let mut s2 = state.clone();
        s2.player_mut(Seat::P2).deck.clear();
        let mut rng = CountingRng { inner: crate::search::Pcg32SearchRng::new(1), shuffles: 0 };
        let w = sample_world(&s2, Seat::P1, reveal_view(Reveal::OppHand), &mut rng).unwrap();
        assert_eq!((rng.shuffles, &w), (0, &s2));
    }

    /// (c) `fair`: 相手の手札＋山札＋裏向きライフ、自分の山札＋裏向きライフをそれぞれ 1 つの山として
    /// 混ぜる。表向きのライフ（位置と中身）・自分の手札・公開領域と各ゾーンの枚数は保つ。
    #[test]
    fn reveal_fair_shuffles_everything_hidden_from_me() {
        let (_m, state) = reveal_board();
        let (mut my_deck_moved, mut my_life_moved, mut opp_cross) = (false, false, false);
        for seed in 0..32u64 {
            let mut rng = CountingRng { inner: crate::search::Pcg32SearchRng::new(seed), shuffles: 0 };
            let w = sample_world(&state, Seat::P1, reveal_view(Reveal::Fair), &mut rng).unwrap();
            assert_eq!(rng.shuffles, 2, "相手の山と自分の山で 2 回");
            assert_public_kept(&w, &state);
            for seat in [Seat::P1, Seat::P2] {
                let (a, t) = (w.player(seat), state.player(seat));
                assert_eq!(
                    (a.hand.len(), a.deck.len(), a.life.len()),
                    (t.hand.len(), t.deck.len(), t.life.len()),
                    "枚数は保つ"
                );
                assert_eq!(life_split(&w, seat).1, life_split(&state, seat).1, "表向きのライフは動かない");
            }
            // 自分: 手札は本当のまま・山札＋裏向きライフは同じ集合の並べ替え。
            let (a, t) = (w.player(Seat::P1), state.player(Seat::P1));
            assert_eq!(a.hand, t.hand);
            let pool = |st: &GameState| {
                let mut v = st.player(Seat::P1).deck.clone();
                v.extend(life_split(st, Seat::P1).0);
                sorted(v)
            };
            assert_eq!(pool(&w), pool(&state));
            my_deck_moved |= a.deck != t.deck;
            my_life_moved |= life_split(&w, Seat::P1).0 != life_split(&state, Seat::P1).0;
            // 相手: 手札＋山札＋裏向きライフが同じ集合の並べ替え。
            let opool = |st: &GameState| {
                let p = st.player(Seat::P2);
                let mut v = p.hand.clone();
                v.extend(p.deck.iter().copied());
                v.extend(life_split(st, Seat::P2).0);
                sorted(v)
            };
            assert_eq!(opool(&w), opool(&state));
            let true_life = life_split(&state, Seat::P2).0;
            opp_cross |= w.player(Seat::P2).hand.iter().any(|c| true_life.contains(c));
        }
        assert!(my_deck_moved && my_life_moved, "自分の山札・裏向きライフが混ざる");
        assert!(opp_cross, "相手の裏向きライフが手札と 1 つの山で混ざる");
    }

    /// `opp_hand`／`fair` の decide が回る（`worlds=4`＋seed でも）・同じ種なら同じ手。
    #[test]
    fn reveal_modes_decide_and_are_deterministic() {
        let (masters, state) = reveal_board();
        for reveal in [Reveal::OppHand, Reveal::Fair, Reveal::Peek] {
            for worlds in [1usize, 4] {
                let opts = DecideOptions { reveal, ..worlds_opts(worlds, Some(5)) };
                let (a, _) = omni_run(&masters, &state, &opts);
                let (b, _) = omni_run(&masters, &state, &opts);
                assert_eq!(a.kind, "main");
                assert_same_out(&a, &b);
            }
        }
    }

    /// (d) `omniscient` と `reveal`（既定の `fair` 以外）の両方はエラー（decide も opts JSON も・JSON では
    /// `"fair"` の明示もエラー＝`tests_search::decide_options_json_reads_reveal`）。
    #[test]
    fn omniscient_and_reveal_together_is_an_error() {
        let (masters, state) = reveal_board();
        let net = zero_net();
        for reveal in [Reveal::OppHand, Reveal::Peek] {
            let opts = DecideOptions { sims: 8, omniscient: true, reveal, ..DecideOptions::default() };
            assert!(opts.validate().is_err());
            let mut rng = crate::search::Pcg32SearchRng::new(1);
            let r = decide(&masters, &net, &state, Seat::P1, &opts, &mut rng, &DecideCarry::default());
            assert!(matches!(r, Err(EngineError::BadPayload(_))));
        }
        let ok = DecideOptions { omniscient: true, ..DecideOptions::default() };
        assert!(ok.validate().is_ok(), "omniscient 単独は可");
    }

    // --- 木の中の相手にこちらの手札を見せない（`search_efficiency_plan.md` §7.14・WP `rs-oppblind`）---

    /// `opp_blind` で decide を回し、相手ノードの展開の記録（`mcts::BLIND_TRACE`）を持ち帰る。
    fn blind_run(
        masters: &crate::model::MasterTable,
        state: &GameState,
        opts: &DecideOptions,
    ) -> (DecideOut, crate::search::mcts::BlindTrace) {
        crate::search::mcts::BLIND_TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
        let (out, _) = omni_run(masters, state, opts);
        let trace = crate::search::mcts::BLIND_TRACE.with(|t| t.borrow_mut().take()).unwrap();
        (out, trace)
    }

    fn blind_opts(sims: usize) -> DecideOptions {
        DecideOptions { sims, opp_blind: true, search_seed: Some(7), ..DecideOptions::default() }
    }

    /// (a) `opp_blind` の既定は false・明示の false は既定と 1 bit も同じ（相手ノードの記録も出ない）。
    #[test]
    fn opp_blind_false_is_bit_identical_to_the_default() {
        let (masters, state) = reveal_board();
        assert!(!DecideOptions::default().opp_blind);
        for sims in [8usize, 64] {
            let base_opts = DecideOptions { sims, search_seed: Some(7), ..DecideOptions::default() };
            let (base, base_shuffles) = omni_run(&masters, &state, &base_opts);
            let (got, trace) =
                blind_run(&masters, &state, &DecideOptions { opp_blind: false, ..base_opts.clone() });
            assert_same_out(&got, &base);
            assert!(trace.is_empty(), "既定では相手ノードを混ぜ直さない");
            let (_, got_shuffles) = omni_run(&masters, &state, &base_opts);
            assert_eq!(got_shuffles, base_shuffles, "探索の乱数の消費も同じ");
        }
    }

    /// (b) true: 相手ノードの評価に渡る盤面は、こちら（P1）の手札＋山札＋裏向きライフだけが同じ集合の
    /// 並べ替えで、相手（P2）の側と公開領域・各ゾーンの枚数・表向きのライフは本当の世界と同一。
    /// こちらの手札は（一般に）本当の手札と違う。
    #[test]
    fn opp_blind_evaluates_opponent_nodes_on_a_reshuffled_copy_of_my_hidden_cards() {
        let (masters, state) = reveal_board();
        let (_, trace) = blind_run(&masters, &state, &blind_opts(96));
        assert!(!trace.is_empty(), "相手の番のノードが展開される");
        let mut hand_differs = false;
        for (real, w, _, _) in &trace {
            assert_eq!(w.player(Seat::P2), real.player(Seat::P2), "相手の側は触らない");
            assert_public_kept(w, real);
            let (a, t) = (w.player(Seat::P1), real.player(Seat::P1));
            assert_eq!(
                (a.hand.len(), a.deck.len(), a.life.len()),
                (t.hand.len(), t.deck.len(), t.life.len()),
                "枚数は保つ"
            );
            assert_eq!(life_split(w, Seat::P1).1, life_split(real, Seat::P1).1, "表向きのライフは動かない");
            let pool = |st: &GameState| {
                let p = st.player(Seat::P1);
                let mut v = p.hand.clone();
                v.extend(p.deck.iter().copied());
                v.extend(life_split(st, Seat::P1).0);
                sorted(v)
            };
            assert_eq!(pool(w), pool(real));
            hand_differs |= a.hand != t.hand;
            // 混ぜ直し以外（手番・フェイズ・戦闘・中断）は同じ。
            assert_eq!(w.turn_player, real.turn_player);
            assert_eq!(w.active_battle.is_some(), real.active_battle.is_some());
        }
        assert!(hand_differs, "こちらの手札が本当の手札と違う複製が出る");
    }

    /// (c) true でも合法手は本当の世界から作り、展開の後の本当の盤面は展開前と同一（木の降下・手の
    /// 適用は本当の世界のまま）。根の合法手も既定と同じ。
    #[test]
    fn opp_blind_keeps_legal_moves_and_the_real_world() {
        let (masters, state) = reveal_board();
        let opts = blind_opts(96);
        let (out, trace) = blind_run(&masters, &state, &opts);
        let net = zero_net();
        let ctx = Ctx {
            masters: &masters,
            net: &net,
            opts: opts.search.clone(),
            box_battle: opts.box_battle,
            box_dialog: opts.box_dialog,
            quiesce: opts.quiesce,
            quiesce_max_plies: QUIESCE_MAX_PLIES,
        };
        super::super::r#macro::reset_setup_state(opts.search.setup_box, opts.search.select_branch_on());
        for (real, w, legal, after) in &trace {
            assert_eq!(after, real, "展開は本当の世界を変えない");
            let mut s = Session::new(real.clone());
            assert_eq!(&ctx.legal_actions(&mut s).unwrap(), legal, "合法手は本当の世界から");
            assert_ne!(w, real, "評価だけが複製の上");
        }
        let (base, _) = omni_run(&masters, &state, &DecideOptions { opp_blind: false, ..opts.clone() });
        assert_eq!(out.legal, base.legal, "根の合法手は同じ");
    }

    /// (d) 同じ seed なら決定的（`worlds=1`／`4`・`reveal`・`omniscient` との併用も）。seed を変えると
    /// 混ぜ直しが変わる。
    #[test]
    fn opp_blind_is_deterministic_and_combines_with_other_views() {
        let (masters, state) = reveal_board();
        let combos = [
            DecideOptions { worlds: 1, ..blind_opts(32) },
            DecideOptions { worlds: 4, ..blind_opts(32) },
            DecideOptions { reveal: Reveal::Fair, ..blind_opts(32) },
            DecideOptions { reveal: Reveal::OppHand, ..blind_opts(32) },
            DecideOptions { omniscient: true, ..blind_opts(32) },
        ];
        for opts in &combos {
            let (a, ta) = blind_run(&masters, &state, opts);
            let (b, tb) = blind_run(&masters, &state, opts);
            assert_eq!(a.kind, "main");
            assert_same_out(&a, &b);
            let ws = |t: &Vec<(GameState, GameState, Vec<Move>, GameState)>| {
                t.iter().map(|x| x.1.clone()).collect::<Vec<_>>()
            };
            assert_eq!(ws(&ta), ws(&tb), "混ぜ直しの盤面も同じ");
        }
        let (_, t7) = blind_run(&masters, &state, &blind_opts(96));
        let (_, t8) =
            blind_run(&masters, &state, &DecideOptions { search_seed: Some(8), ..blind_opts(96) });
        let hands = |t: &Vec<(GameState, GameState, Vec<Move>, GameState)>| {
            t.iter().map(|x| x.1.player(Seat::P1).hand.clone()).collect::<Vec<_>>()
        };
        assert_ne!(hands(&t7), hands(&t8), "seed が違えば混ぜ直しも違う");
        // JSON の鍵。
        let j = |x: &str| serde_json::from_str::<serde_json::Value>(x).unwrap();
        let read = |x: &str| super::super::decide_opts_and_carry(&j(x)).map(|(o, _)| o.opp_blind).unwrap();
        assert!(!read("{}"));
        assert!(!read(r#"{"opp_blind": false}"#));
        assert!(read(r#"{"opp_blind": true}"#));
        assert!(read(r#"{"opp_blind": true, "omniscient": true}"#), "omniscient と併用できる");
    }
}
