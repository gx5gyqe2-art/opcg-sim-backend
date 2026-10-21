//! 符号化 v13 の**型契約**（P4・`docs/rust_engine_plan.md` §12）。WP `rs-p4-encode` が本体を入れる。
//!
//! Python の対応: `learned/encoder.py::encode(version=13)`（scalars 123・field 10×8・card_idx 24）、
//! `learned/n_rel_feat.py::encode_rel`（22 枠トークン: 状態 S20・関係 R5・追加列 29）、
//! `learned/n_eff.py::build_eff_tables`／`ability_vector`（カード表の元: STATS 16・能力 4×167）。
//! 数値は Python と同じ式・同じ正規化（`rs_encode_oracle.py` で 1e-6 一致を見る）。
//!
//! 語彙（`vocab_ids`＝npz の `vocab_ids`・0=PAD/UNK）はネット側の資産なので、符号化は
//! `card_idx` を「vocab 上の index」で出す（`Vocab` を引数に取る）。

#![allow(dead_code)]

pub mod cardtab;
pub mod leader;
pub mod scalars;
pub mod tokens;

use crate::effects::ast::{EffectNode, GameAction};
use crate::journal::Session;
use crate::model::{GameState, MasterTable, Seat};
use crate::state::EngineError;
use std::collections::HashMap;

/// `encoder.scalars_dim(14)`＝94（v12）＋33（`n_rel_feat.EXTRA_DIM`）。
///
/// **符号化 v14**（2026-09-11・`docs/rust_engine_plan.md` §20.9）は v13 に列を**末尾へ足した**
/// だけ（S 20→22・EXTRA 29→33）。v13 のネット（r3／a1）は新しい列の重みを 0 で埋めて読む
/// （[`crate::net::load_npz`]）＝出力が v13 と 1 bit も変わらない。
pub const D_SC: usize = 127;
/// v13 の scalars（94＋29）。npz の pad の位置を決めるのに要る。
pub const D_SC_V13: usize = 123;
pub const MAX_FIELD: usize = 5;
pub const MAX_HAND: usize = 10;
/// `encoder.PER_CHAR`＝[cost, power, is_rest, attached_don]＋キーワード 4。
pub const PER_CHAR: usize = 8;
/// `encoder.encode` の `card_idx` 長（v3 以降）: 自L・相L・自場5・相場5・手札10・ステージ2。
pub const N_CARD_IDX: usize = 24;
/// `n_rel_feat.N_TOK`＝22（card_idx の先頭 22 枠と同じ並び）。
pub const N_TOK: usize = 22;
pub const N_OWN: usize = 16;
pub const N_OPP: usize = 6;
pub const S_DIM: usize = 22;
/// v13 のトークン状態 S（20 列）。npz の pad の位置を決めるのに要る。
pub const S_DIM_V13: usize = 20;
pub const R_DIM: usize = 5;
pub const EXTRA_DIM: usize = 33;
/// v13 のグローバル追加列（29）。
pub const EXTRA_DIM_V13: usize = 29;
/// 符号化の世代（`n_rel.NR_ENC_VERSION`）。npz の meta `enc_version` と突き合わせる。
pub const ENC_VERSION: u32 = 14;
pub const ENC_VERSION_V13: u32 = 13;
/// `n_eff.STATS_DIM`（stats 8＋印字キーワード 8）／`MAX_AB`／`ABILITY_DIM`（23＋62×2＋4＋8＋7＋1）。
pub const STATS_DIM: usize = 16;
pub const MAX_AB: usize = 4;
pub const ABILITY_DIM: usize = 167;
/// 登場時スキャン（v7）の列位置＝`encoder.SCALARS_V6`（`n_rel.ONPLAY_COLS`）。
pub const ONPLAY_COLS: [usize; 3] = [67, 68, 69];
/// `n_rel_feat.GAP_SAT`（「届かない／該当なし」の飽和値）。
pub const GAP_SAT: f64 = 1.5;

/// `n_rel_feat._walk`／`leader_feat._walk` の木の歩き（**全ての子を辿る**）。
///
/// `n_eff._walk` は属性名が違うため Sequence／Branch の中へ入らない（`cardtab.rs` を参照）。
/// こちらは Sequence の `actions`・Choice の `options`・Branch の `if_true`/`if_false` を辿る。
pub fn walk_all<'a>(node: Option<&'a EffectNode>, out: &mut Vec<&'a GameAction>) {
    let Some(node) = node else { return };
    match node {
        EffectNode::Action(a) => {
            out.push(a);
            walk_all(a.sub_effect.as_deref(), out);
        }
        EffectNode::Sequence(items) => {
            for a in items {
                walk_all(Some(a), out);
            }
        }
        EffectNode::Branch {
            if_true, if_false, ..
        } => {
            walk_all(if_true.as_deref(), out);
            walk_all(if_false.as_deref(), out);
        }
        EffectNode::Choice { options, .. } => {
            for a in options {
                walk_all(Some(a), out);
            }
        }
    }
}

/// 引くだけの表（中身の並びを外へ出さない表）に使う軽いハッシュ（FNV-1a）。
///
/// std の既定（SipHash）は探索の 1 回の符号化・priors で数十〜百回回って重い。表の中身は
/// 同じ＝引いた値は 1 bit も変わらない（並びに意味を持たせる表には使わない）。
#[derive(Clone, Copy)]
pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl std::hash::Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        self.0 = h;
    }
}

/// [`Fnv`] の表。
pub type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<Fnv>>;

/// 語彙: card_id → index（0=PAD/UNK）。npz の `vocab_ids`（index 1..）から作る。
#[derive(Debug, Clone, Default)]
pub struct Vocab {
    pub ids: Vec<String>,
    pub index: FastMap<String, u32>,
}

impl Vocab {
    pub fn from_ids(ids: &[String]) -> Vocab {
        let mut index = FastMap::default();
        for (i, id) in ids.iter().enumerate() {
            index.insert(id.clone(), (i + 1) as u32); // Python `vocab_from_ids`: 1 始まり・0 は PAD
        }
        Vocab { ids: ids.to_vec(), index }
    }
    pub fn idx(&self, card_id: &str) -> u32 {
        *self.index.get(card_id).unwrap_or(&0)
    }
}

/// `n_eff.build_eff_tables` の 5 表（行＝vocab index・行 0 は PAD）。
#[derive(Debug, Clone, Default)]
pub struct EffTables {
    pub n: usize,
    /// [n × STATS_DIM]
    pub stats: Vec<f32>,
    /// [n × MAX_AB × ABILITY_DIM]
    pub ab: Vec<f32>,
    /// [n × MAX_AB]
    pub abm: Vec<f32>,
    pub pwr: Vec<f32>,
    pub isl: Vec<f32>,
}

/// 1 視点の符号化（Python `encode`＋`encode_rel` の出力を平らにしたもの）。
#[derive(Debug, Clone, Default)]
pub struct Encoding {
    /// `scalars` [D_SC]
    pub scalars: Vec<f32>,
    /// `field` [2*MAX_FIELD × PER_CHAR]
    pub field: Vec<f32>,
    /// `card_idx` [N_CARD_IDX]（vocab index・0=PAD）
    pub card_idx: Vec<u32>,
    /// トークン状態 S [N_TOK × S_DIM]（`encode_rel` の `tok`）
    pub tok: Vec<f32>,
    /// 関係 R own→opp [N_OWN × N_OPP × R_DIM]（`rel_om`・`skip_relations` のときは空）
    pub rel_om: Vec<f32>,
    /// 関係 R own→own [N_OWN × N_OWN × R_DIM]（`rel_oo`＝「i の減算で k のしきい値が届く」組）。
    ///
    /// **契約の注記（WP `rs-p4-encode`）**: 当初のコメントは `[N_OPP × N_OWN × R_DIM]` だったが、
    /// Python `n_rel_feat.relations_from_tokens` は `np.zeros((N_OWN, N_OWN, R_DIM))` を返す。
    /// 型（`Vec<f32>`）は変えず、寸法の記述だけ Python に合わせた。
    pub rel_oo: Vec<f32>,
    /// 追加列 [EXTRA_DIM]（scalars の末尾 29 と同じ値。方策の予算特徴が読む）
    pub extra: Vec<f32>,
}

/// 符号化の指定（ネットの ablate と同じ語）。
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// `with_relations=False`（a1）。R を計算しない＝`rel_*` は 0。
    pub skip_relations: bool,
    /// `skip_onplay=True`（ablate onplay）。登場時スキャンを省き列を 0 にする。
    pub skip_onplay: bool,
}

/// `n_eff.build_eff_tables(db, vocab)`。実体は [`cardtab::build_eff_tables`]。
pub fn build_eff_tables(masters: &MasterTable, vocab: &Vocab) -> Result<EffTables, EngineError> {
    Ok(cardtab::build_eff_tables(masters, vocab))
}

/// Python `encoder.encode(manager, me_name, vocab, version=13)`＋`n_rel_feat.encode_rel(...)`。
///
/// `_leader_act_avail` が読む合法手は Python と同じ `get_legal_actions`（探索用の枝刈り前）。
/// 登場時スキャン（v7）は `rules`/`effects` で PLAY を実際に適用して
/// Python `cpu_ai.onplay_option_scan` と同じ判定で数える。
///
/// 引数が `&GameState`（契約）なので、合法手列挙と登場時スキャンが要る **一時的な
/// 書き換え**は複製した [`Session`] の上で行い、呼び出し側の盤面は 1 bit も変えない
/// （Python は make/unmake で同じことをする＝巻き戻し後は完全一致）。
pub fn encode(
    state: &GameState,
    masters: &MasterTable,
    vocab: &Vocab,
    me: Seat,
    opts: &EncodeOptions,
) -> Result<Encoding, EngineError> {
    let mut session = Session::new(state.clone());
    scalars::encode(&mut session, masters, vocab, me, opts)
}

// --- decide 1 回のあいだのマスター単位のキャッシュ（`docs/reports/2026-10-17_game_speed.md` §2）-------------
//
// 符号化は 1 回ごとに `ProfileCache`／`LeaderCache`（マスター単位の静的な要約）を空から作っていた。
// 中身はマスター（カードの定義）だけの関数なので、**同じ decide の中**では使い回せる（decide の間、
// マスター表は不変参照で借りられたまま＝書き換わらない）。decide の外の符号化は今までどおり毎回空から作る。

type DecideCaches = (usize, tokens::ProfileCache, leader::LeaderCache);

thread_local! {
    static DECIDE_CACHE: std::cell::RefCell<Option<DecideCaches>> = const { std::cell::RefCell::new(None) };
}

/// decide（とその世界のスレッド）の入口で張る。落とすと元へ戻す。
pub struct DecideCacheScope {
    prev: Option<DecideCaches>,
}

pub fn decide_cache_scope(masters: &MasterTable) -> DecideCacheScope {
    let key = masters as *const MasterTable as usize;
    let prev = DECIDE_CACHE.with(|c| {
        c.borrow_mut()
            .replace((key, tokens::ProfileCache::default(), leader::LeaderCache::default()))
    });
    DecideCacheScope { prev }
}

impl Drop for DecideCacheScope {
    fn drop(&mut self) {
        let prev = self.prev.take();
        DECIDE_CACHE.with(|c| *c.borrow_mut() = prev);
    }
}

/// 符号化 1 回ぶんのキャッシュ（decide の中なら張ったものを借り、落とすと返す）。
pub(crate) struct EncCaches {
    pub profiles: tokens::ProfileCache,
    pub leaders: leader::LeaderCache,
    key: Option<usize>,
}

impl EncCaches {
    pub(crate) fn take(masters: &MasterTable) -> EncCaches {
        let key = masters as *const MasterTable as usize;
        let got = DECIDE_CACHE.with(|c| {
            let mut b = c.borrow_mut();
            match b.as_mut() {
                Some((k, p, l)) if *k == key => Some((std::mem::take(p), std::mem::take(l))),
                _ => None,
            }
        });
        match got {
            Some((profiles, leaders)) => EncCaches { profiles, leaders, key: Some(key) },
            None => EncCaches {
                profiles: tokens::ProfileCache::default(),
                leaders: leader::LeaderCache::default(),
                key: None,
            },
        }
    }
}

impl Drop for EncCaches {
    fn drop(&mut self) {
        let Some(key) = self.key else { return };
        let (p, l) = (std::mem::take(&mut self.profiles), std::mem::take(&mut self.leaders));
        DECIDE_CACHE.with(|c| {
            if let Some((k, cp, cl)) = c.borrow_mut().as_mut() {
                if *k == key {
                    *cp = p;
                    *cl = l;
                }
            }
        });
    }
}

/// [`encode`] と同じ値を、盤面を複製せずに `s` の上で作る（[`Session::isolated`]・探索の葉の評価が使う）。
pub fn encode_in(
    s: &mut Session,
    masters: &MasterTable,
    vocab: &Vocab,
    me: Seat,
    opts: &EncodeOptions,
) -> Result<Encoding, EngineError> {
    s.isolated(|s| scalars::encode(s, masters, vocab, me, opts))
}

// --- プロセス内の語彙とカード表（`lib.rs` の `set_vocab`／`encode_state` が使う）-----------
//
// 語彙（`vocab_ids`）はネット側の資産（npz）なので、符号化側は npz を読まない
// （§12.5 の決定: `set_vocab(ids_json)` で JSON として受け取る＝`rs-p4-net` と独立）。

use std::sync::{Arc, Mutex, OnceLock};

static VOCAB: OnceLock<Mutex<Option<Arc<Vocab>>>> = OnceLock::new();
static EFF_TABLES: OnceLock<Mutex<Option<Arc<EffTables>>>> = OnceLock::new();

fn vocab_slot() -> &'static Mutex<Option<Arc<Vocab>>> {
    VOCAB.get_or_init(|| Mutex::new(None))
}

fn eff_slot() -> &'static Mutex<Option<Arc<EffTables>>> {
    EFF_TABLES.get_or_init(|| Mutex::new(None))
}

fn lock_poisoned(what: &str) -> EngineError {
    EngineError::BadPayload(format!("encode: {what} のロックが壊れている"))
}

/// 語彙を差し替える（カード表のキャッシュも捨てる）。戻り値は語彙の件数。
pub fn set_vocab(ids: Vec<String>) -> Result<usize, EngineError> {
    let n = ids.len();
    let vocab = Vocab::from_ids(&ids);
    *vocab_slot().lock().map_err(|_| lock_poisoned("vocab"))? = Some(Arc::new(vocab));
    *eff_slot().lock().map_err(|_| lock_poisoned("eff_tables"))? = None;
    Ok(n)
}

/// 設定済みの語彙（未設定なら `None`）。
pub fn current_vocab() -> Result<Option<Arc<Vocab>>, EngineError> {
    Ok(vocab_slot()
        .lock()
        .map_err(|_| lock_poisoned("vocab"))?
        .clone())
}

/// 設定済み語彙のカード表（初回だけ計算してプロセス内に持つ）。
pub fn current_eff_tables(masters: &MasterTable) -> Result<Arc<EffTables>, EngineError> {
    if let Some(t) = eff_slot()
        .lock()
        .map_err(|_| lock_poisoned("eff_tables"))?
        .clone()
    {
        return Ok(t);
    }
    let vocab = current_vocab()?.ok_or_else(|| {
        EngineError::BadPayload(
            "encode: 語彙が未設定。opcg_engine.set_vocab(json.dumps(vocab_ids)) を先に呼ぶこと"
                .into(),
        )
    })?;
    let table = Arc::new(cardtab::build_eff_tables(masters, &vocab));
    *eff_slot().lock().map_err(|_| lock_poisoned("eff_tables"))? = Some(table.clone());
    Ok(table)
}
