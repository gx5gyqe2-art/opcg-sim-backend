//! 効果構造の**型契約**（P3・`docs/rust_engine_plan.md` §11.2）。
//!
//! Python の `opcg_sim/src/models/effect_types.py`（`Ability`／`EffectNode`＝`GameAction`・`Sequence`・
//! `Branch`・`Choice`／`Condition`／`TargetQuery`／`ValueSource`）と `enums.py`（`ActionType`・
//! `TriggerType`・`ConditionType`・`CompareOperator`・`Zone`・`Player`）を 1:1 で写す。入力は
//! `opcg_sim/tools/export_effects_json.py` の出力（enum は**名前**文字列・None は null・set はソート list）。
//!
//! 原則（§6）: 未知の enum 名・未知のキーは読み込み時に `BadPayload`（黙って捨てない）。
//! 型は append-only。ロジック（読込 `loader.rs`・対象 `matcher.rs`・条件 `cond.rs`・値 `value.rs`・
//! 実行 `resolver.rs`…）は WP が入れる。
//!
//! Python 側のエイリアス（`DAMAGE = DEAL_DAMAGE`・`DEBUFF = BUFF`）は JSON には出ない（`.name` は
//! 正規名を返す）ので Rust には持たない。

#![allow(dead_code)]

/// Python `ActionType`（62 メンバー・エイリアス除く）。カード DB で使われるのは 45 種（§8.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    Ko,
    Rest,
    Active,
    Freeze,
    Lock,
    DisableAbility,
    GrantEffect,
    MoveCard,
    DeckBottom,
    Draw,
    Discard,
    TrashFromDeck,
    Look,
    LookLife,
    Reveal,
    Shuffle,
    PlayCard,
    LifeRecover,
    FaceUpLife,
    BpBuff,
    SetBasePower,
    CostBuff,
    AttachDon,
    RestDon,
    FreezeDon,
    RampDon,
    ReturnDon,
    NegateEffect,
    SwapPower,
    Keyword,
    LifeManipulate,
    CostChange,
    GrantKeyword,
    AttackDisable,
    ExecuteMainEffect,
    Victory,
    ExtraTurn,
    RuleProcessing,
    Restriction,
    PreventRest,
    DeckTop,
    SetCost,
    DeclareCost,
    DealDamage,
    SelectOption,
    Select,
    PassiveEffect,
    PreventLeave,
    ReplaceEffect,
    MoveAttachedDon,
    ModifyDonPhase,
    RedirectAttack,
    OrderLife,
    ExecuteEvent,
    Other,
    MoveToHand,
    Trash,
    Buff,
    ActiveDon,
    Bounce,
    Move,
    Heal,
}

impl ActionType {
    /// Python の `ActionType.name`。
    pub fn name(self) -> &'static str {
        use ActionType::*;
        match self {
            Ko => "KO",
            Rest => "REST",
            Active => "ACTIVE",
            Freeze => "FREEZE",
            Lock => "LOCK",
            DisableAbility => "DISABLE_ABILITY",
            GrantEffect => "GRANT_EFFECT",
            MoveCard => "MOVE_CARD",
            DeckBottom => "DECK_BOTTOM",
            Draw => "DRAW",
            Discard => "DISCARD",
            TrashFromDeck => "TRASH_FROM_DECK",
            Look => "LOOK",
            LookLife => "LOOK_LIFE",
            Reveal => "REVEAL",
            Shuffle => "SHUFFLE",
            PlayCard => "PLAY_CARD",
            LifeRecover => "LIFE_RECOVER",
            FaceUpLife => "FACE_UP_LIFE",
            BpBuff => "BP_BUFF",
            SetBasePower => "SET_BASE_POWER",
            CostBuff => "COST_BUFF",
            AttachDon => "ATTACH_DON",
            RestDon => "REST_DON",
            FreezeDon => "FREEZE_DON",
            RampDon => "RAMP_DON",
            ReturnDon => "RETURN_DON",
            NegateEffect => "NEGATE_EFFECT",
            SwapPower => "SWAP_POWER",
            Keyword => "KEYWORD",
            LifeManipulate => "LIFE_MANIPULATE",
            CostChange => "COST_CHANGE",
            GrantKeyword => "GRANT_KEYWORD",
            AttackDisable => "ATTACK_DISABLE",
            ExecuteMainEffect => "EXECUTE_MAIN_EFFECT",
            Victory => "VICTORY",
            ExtraTurn => "EXTRA_TURN",
            RuleProcessing => "RULE_PROCESSING",
            Restriction => "RESTRICTION",
            PreventRest => "PREVENT_REST",
            DeckTop => "DECK_TOP",
            SetCost => "SET_COST",
            DeclareCost => "DECLARE_COST",
            DealDamage => "DEAL_DAMAGE",
            SelectOption => "SELECT_OPTION",
            Select => "SELECT",
            PassiveEffect => "PASSIVE_EFFECT",
            PreventLeave => "PREVENT_LEAVE",
            ReplaceEffect => "REPLACE_EFFECT",
            MoveAttachedDon => "MOVE_ATTACHED_DON",
            ModifyDonPhase => "MODIFY_DON_PHASE",
            RedirectAttack => "REDIRECT_ATTACK",
            OrderLife => "ORDER_LIFE",
            ExecuteEvent => "EXECUTE_EVENT",
            Other => "OTHER",
            MoveToHand => "MOVE_TO_HAND",
            Trash => "TRASH",
            Buff => "BUFF",
            ActiveDon => "ACTIVE_DON",
            Bounce => "BOUNCE",
            Move => "MOVE",
            Heal => "HEAL",
        }
    }

    pub const ALL: &'static [ActionType] = &[
        ActionType::Ko, ActionType::Rest, ActionType::Active, ActionType::Freeze, ActionType::Lock,
        ActionType::DisableAbility, ActionType::GrantEffect, ActionType::MoveCard, ActionType::DeckBottom,
        ActionType::Draw, ActionType::Discard, ActionType::TrashFromDeck, ActionType::Look,
        ActionType::LookLife, ActionType::Reveal, ActionType::Shuffle, ActionType::PlayCard,
        ActionType::LifeRecover, ActionType::FaceUpLife, ActionType::BpBuff, ActionType::SetBasePower,
        ActionType::CostBuff, ActionType::AttachDon, ActionType::RestDon, ActionType::FreezeDon,
        ActionType::RampDon, ActionType::ReturnDon, ActionType::NegateEffect, ActionType::SwapPower,
        ActionType::Keyword, ActionType::LifeManipulate, ActionType::CostChange, ActionType::GrantKeyword,
        ActionType::AttackDisable, ActionType::ExecuteMainEffect, ActionType::Victory, ActionType::ExtraTurn,
        ActionType::RuleProcessing, ActionType::Restriction, ActionType::PreventRest, ActionType::DeckTop,
        ActionType::SetCost, ActionType::DeclareCost, ActionType::DealDamage, ActionType::SelectOption,
        ActionType::Select, ActionType::PassiveEffect, ActionType::PreventLeave, ActionType::ReplaceEffect,
        ActionType::MoveAttachedDon, ActionType::ModifyDonPhase, ActionType::RedirectAttack,
        ActionType::OrderLife, ActionType::ExecuteEvent, ActionType::Other, ActionType::MoveToHand,
        ActionType::Trash, ActionType::Buff, ActionType::ActiveDon, ActionType::Bounce, ActionType::Move,
        ActionType::Heal,
    ];

    pub fn from_name(s: &str) -> Option<ActionType> {
        ActionType::ALL.iter().copied().find(|t| t.name() == s)
    }
}

/// Python `TriggerType`（JSON は `.name`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerType {
    OnPlay,
    OnAttack,
    OnBlock,
    OnKo,
    ActivateMain,
    TurnEnd,
    OppTurnEnd,
    TurnStart,
    OnOppAttack,
    Trigger,
    Counter,
    Rule,
    Passive,
    YourTurn,
    OpponentTurn,
    OpponentAttack,
    OnDamageDealtToLife,
    OnLifeDecrease,
    OnLeave,
    OnEventPlay,
    OnOppPlay,
    OnRest,
    GameStart,
    Unknown,
}

impl TriggerType {
    pub fn name(self) -> &'static str {
        use TriggerType::*;
        match self {
            OnPlay => "ON_PLAY",
            OnAttack => "ON_ATTACK",
            OnBlock => "ON_BLOCK",
            OnKo => "ON_KO",
            ActivateMain => "ACTIVATE_MAIN",
            TurnEnd => "TURN_END",
            OppTurnEnd => "OPP_TURN_END",
            TurnStart => "TURN_START",
            OnOppAttack => "ON_OPP_ATTACK",
            Trigger => "TRIGGER",
            Counter => "COUNTER",
            Rule => "RULE",
            Passive => "PASSIVE",
            YourTurn => "YOUR_TURN",
            OpponentTurn => "OPPONENT_TURN",
            OpponentAttack => "OPPONENT_ATTACK",
            OnDamageDealtToLife => "ON_DAMAGE_DEALT_TO_LIFE",
            OnLifeDecrease => "ON_LIFE_DECREASE",
            OnLeave => "ON_LEAVE",
            OnEventPlay => "ON_EVENT_PLAY",
            OnOppPlay => "ON_OPP_PLAY",
            OnRest => "ON_REST",
            GameStart => "GAME_START",
            Unknown => "UNKNOWN",
        }
    }

    pub const ALL: &'static [TriggerType] = &[
        TriggerType::OnPlay, TriggerType::OnAttack, TriggerType::OnBlock, TriggerType::OnKo,
        TriggerType::ActivateMain, TriggerType::TurnEnd, TriggerType::OppTurnEnd, TriggerType::TurnStart,
        TriggerType::OnOppAttack, TriggerType::Trigger, TriggerType::Counter, TriggerType::Rule,
        TriggerType::Passive, TriggerType::YourTurn, TriggerType::OpponentTurn, TriggerType::OpponentAttack,
        TriggerType::OnDamageDealtToLife, TriggerType::OnLifeDecrease, TriggerType::OnLeave,
        TriggerType::OnEventPlay, TriggerType::OnOppPlay, TriggerType::OnRest, TriggerType::GameStart,
        TriggerType::Unknown,
    ];

    pub fn from_name(s: &str) -> Option<TriggerType> {
        TriggerType::ALL.iter().copied().find(|t| t.name() == s)
    }
}

/// Python `ConditionType`（42 メンバー）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConditionType {
    LifeCount,
    HandCount,
    TrashCount,
    FieldCount,
    FieldCostSum,
    LifeHandSum,
    EventThisTurn,
    CharKoedThisTurn,
    LifeCountCompare,
    HandCountCompare,
    TurnCount,
    HasTrait,
    HasAttribute,
    HasUnit,
    HasDon,
    IsRested,
    DonCount,
    DeckCount,
    LeaderName,
    LeaderTrait,
    LeaderColor,
    Context,
    And,
    Or,
    Not,
    Other,
    None,
    TurnLimit,
    Generic,
    SourceState,
    FieldAllTrait,
    HasCharacter,
    LeaderAttribute,
    RestedCount,
    PrevAction,
    DonCountCompare,
    LeaderState,
    FieldCountCompare,
    RevealedCardTrait,
    OpponentRemoval,
    DeclaredCostMatch,
    LifeCountBoth,
}

impl ConditionType {
    pub fn name(self) -> &'static str {
        use ConditionType::*;
        match self {
            LifeCount => "LIFE_COUNT",
            HandCount => "HAND_COUNT",
            TrashCount => "TRASH_COUNT",
            FieldCount => "FIELD_COUNT",
            FieldCostSum => "FIELD_COST_SUM",
            LifeHandSum => "LIFE_HAND_SUM",
            EventThisTurn => "EVENT_THIS_TURN",
            CharKoedThisTurn => "CHAR_KOED_THIS_TURN",
            LifeCountCompare => "LIFE_COUNT_COMPARE",
            HandCountCompare => "HAND_COUNT_COMPARE",
            TurnCount => "TURN_COUNT",
            HasTrait => "HAS_TRAIT",
            HasAttribute => "HAS_ATTRIBUTE",
            HasUnit => "HAS_UNIT",
            HasDon => "HAS_DON",
            IsRested => "IS_RESTED",
            DonCount => "DON_COUNT",
            DeckCount => "DECK_COUNT",
            LeaderName => "LEADER_NAME",
            LeaderTrait => "LEADER_TRAIT",
            LeaderColor => "LEADER_COLOR",
            Context => "CONTEXT",
            And => "AND",
            Or => "OR",
            Not => "NOT",
            Other => "OTHER",
            None => "NONE",
            TurnLimit => "TURN_LIMIT",
            Generic => "GENERIC",
            SourceState => "SOURCE_STATE",
            FieldAllTrait => "FIELD_ALL_TRAIT",
            HasCharacter => "HAS_CHARACTER",
            LeaderAttribute => "LEADER_ATTRIBUTE",
            RestedCount => "RESTED_COUNT",
            PrevAction => "PREV_ACTION",
            DonCountCompare => "DON_COUNT_COMPARE",
            LeaderState => "LEADER_STATE",
            FieldCountCompare => "FIELD_COUNT_COMPARE",
            RevealedCardTrait => "REVEALED_CARD_TRAIT",
            OpponentRemoval => "OPPONENT_REMOVAL",
            DeclaredCostMatch => "DECLARED_COST_MATCH",
            LifeCountBoth => "LIFE_COUNT_BOTH",
        }
    }

    pub const ALL: &'static [ConditionType] = &[
        ConditionType::LifeCount, ConditionType::HandCount, ConditionType::TrashCount,
        ConditionType::FieldCount, ConditionType::FieldCostSum, ConditionType::LifeHandSum,
        ConditionType::EventThisTurn, ConditionType::CharKoedThisTurn, ConditionType::LifeCountCompare,
        ConditionType::HandCountCompare, ConditionType::TurnCount, ConditionType::HasTrait,
        ConditionType::HasAttribute, ConditionType::HasUnit, ConditionType::HasDon, ConditionType::IsRested,
        ConditionType::DonCount, ConditionType::DeckCount, ConditionType::LeaderName,
        ConditionType::LeaderTrait, ConditionType::LeaderColor, ConditionType::Context, ConditionType::And,
        ConditionType::Or, ConditionType::Not, ConditionType::Other, ConditionType::None,
        ConditionType::TurnLimit, ConditionType::Generic, ConditionType::SourceState,
        ConditionType::FieldAllTrait, ConditionType::HasCharacter, ConditionType::LeaderAttribute,
        ConditionType::RestedCount, ConditionType::PrevAction, ConditionType::DonCountCompare,
        ConditionType::LeaderState, ConditionType::FieldCountCompare, ConditionType::RevealedCardTrait,
        ConditionType::OpponentRemoval, ConditionType::DeclaredCostMatch, ConditionType::LifeCountBoth,
    ];

    pub fn from_name(s: &str) -> Option<ConditionType> {
        ConditionType::ALL.iter().copied().find(|t| t.name() == s)
    }
}

/// Python `CompareOperator`（JSON は `.name`: EQ/NEQ/GT/LT/GE/LE/HAS）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOperator {
    Eq,
    Neq,
    Gt,
    Lt,
    Ge,
    Le,
    Has,
}

impl CompareOperator {
    pub fn from_name(s: &str) -> Option<CompareOperator> {
        Some(match s {
            "EQ" => CompareOperator::Eq,
            "NEQ" => CompareOperator::Neq,
            "GT" => CompareOperator::Gt,
            "LT" => CompareOperator::Lt,
            "GE" => CompareOperator::Ge,
            "LE" => CompareOperator::Le,
            "HAS" => CompareOperator::Has,
            _ => return None,
        })
    }
}

/// Python `Player`（効果の主語: SELF/OPPONENT/OWNER/ALL）。`model::Seat` とは別物（実行時に解決する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlayerRef {
    SelfP,
    Opponent,
    Owner,
    All,
}

impl PlayerRef {
    pub fn from_name(s: &str) -> Option<PlayerRef> {
        Some(match s {
            "SELF" => PlayerRef::SelfP,
            "OPPONENT" => PlayerRef::Opponent,
            "OWNER" => PlayerRef::Owner,
            "ALL" => PlayerRef::All,
            _ => return None,
        })
    }
}

/// Python `Zone`（効果の参照先。`model::Zone` より広い: DON_DECK／COST_AREA／ANY を含む）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZoneRef {
    Field,
    Hand,
    Deck,
    Trash,
    Life,
    DonDeck,
    CostArea,
    Temp,
    Any,
}

impl ZoneRef {
    /// Python の `Zone.name`（`action_history` の `dest` に出す名前）。
    pub fn name(self) -> &'static str {
        match self {
            ZoneRef::Field => "FIELD",
            ZoneRef::Hand => "HAND",
            ZoneRef::Deck => "DECK",
            ZoneRef::Trash => "TRASH",
            ZoneRef::Life => "LIFE",
            ZoneRef::DonDeck => "DON_DECK",
            ZoneRef::CostArea => "COST_AREA",
            ZoneRef::Temp => "TEMP",
            ZoneRef::Any => "ANY",
        }
    }

    pub fn from_name(s: &str) -> Option<ZoneRef> {
        Some(match s {
            "FIELD" => ZoneRef::Field,
            "HAND" => ZoneRef::Hand,
            "DECK" => ZoneRef::Deck,
            "TRASH" => ZoneRef::Trash,
            "LIFE" => ZoneRef::Life,
            "DON_DECK" => ZoneRef::DonDeck,
            "COST_AREA" => ZoneRef::CostArea,
            "TEMP" => ZoneRef::Temp,
            "ANY" => ZoneRef::Any,
            _ => return None,
        })
    }
}

/// `GameAction.duration`（"INSTANT"／"THIS_TURN"／"THIS_BATTLE"／"UNTIL_NEXT_TURN_END"／"PERMANENT"）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Duration {
    Instant,
    ThisTurn,
    ThisBattle,
    UntilNextTurnEnd,
    Permanent,
}

impl Duration {
    pub fn from_name(s: &str) -> Option<Duration> {
        Some(match s {
            "INSTANT" => Duration::Instant,
            "THIS_TURN" => Duration::ThisTurn,
            "THIS_BATTLE" => Duration::ThisBattle,
            "UNTIL_NEXT_TURN_END" => Duration::UntilNextTurnEnd,
            "PERMANENT" => Duration::Permanent,
            _ => return None,
        })
    }
}

/// Python `TargetQuery`（全フィールド・既定値は Python と同じ）。
#[derive(Debug, Clone, PartialEq)]
pub struct TargetQuery {
    /// Python は `Zone | List[Zone]`。単一なら要素 1 の Vec。既定 `[FIELD]`。
    pub zone: Vec<ZoneRef>,
    pub player: PlayerRef,
    pub card_type: Vec<String>,
    pub traits: Vec<String>,
    pub attributes: Vec<String>,
    pub colors: Vec<String>,
    pub names: Vec<String>,
    pub cost_min: Option<i32>,
    pub cost_max: Option<i32>,
    pub cost_max_dynamic: Option<String>,
    pub power_min: Option<i32>,
    pub power_max: Option<i32>,
    pub power_sum_max: Option<i32>,
    pub min_attached_don: Option<i32>,
    pub is_face_up: Option<bool>,
    pub lacks_trigger: Option<String>,
    pub is_rest: Option<bool>,
    pub count: i32,
    pub is_up_to: bool,
    pub count_dynamic: Option<String>,
    /// "CHOOSE" ほか（文字列のまま。matcher が解釈）。
    pub select_mode: String,
    pub save_id: Option<String>,
    pub ref_id: Option<String>,
    pub chooser: Option<PlayerRef>,
    /// Python は set。ソート済み list で持つ。
    pub flags: Vec<String>,
    pub is_vanilla: bool,
    pub is_strict_count: bool,
    pub is_unique_name: bool,
    pub exclude_ids: Vec<String>,
    pub exclude_names: Vec<String>,
    pub raw_text: String,
}

/// Python `TargetQuery` の dataclass 既定値（`zone=FIELD`／`player=SELF`／`count=1`／
/// `select_mode="CHOOSE"`／他は空・None・False）。`cond.rs` の `HAS_TRAIT` 等が
/// 「`condition.target` が無いときに合成するクエリ」で使う。
impl Default for TargetQuery {
    fn default() -> TargetQuery {
        TargetQuery {
            zone: vec![ZoneRef::Field],
            player: PlayerRef::SelfP,
            card_type: Vec::new(),
            traits: Vec::new(),
            attributes: Vec::new(),
            colors: Vec::new(),
            names: Vec::new(),
            cost_min: None,
            cost_max: None,
            cost_max_dynamic: None,
            power_min: None,
            power_max: None,
            power_sum_max: None,
            min_attached_don: None,
            is_face_up: None,
            lacks_trigger: None,
            is_rest: None,
            count: 1,
            is_up_to: false,
            count_dynamic: None,
            select_mode: "CHOOSE".to_owned(),
            save_id: None,
            ref_id: None,
            chooser: None,
            flags: Vec::new(),
            is_vanilla: false,
            is_strict_count: false,
            is_unique_name: false,
            exclude_ids: Vec::new(),
            exclude_names: Vec::new(),
            raw_text: String::new(),
        }
    }
}

impl TargetQuery {
    /// Python `"X" in query.flags`。
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
}

/// Python `ValueSource`。
#[derive(Debug, Clone, PartialEq)]
pub struct ValueSource {
    pub base: i32,
    /// "PREV_ACTION_COUNT"／"COUNT_QUERY"／"REFERENCE_POWER"／"COUNT_REFERENCE"／"REFERENCE_BASE_POWER"
    pub dynamic_source: Option<String>,
    pub multiplier: i32,
    pub divisor: i32,
    pub ref_id: Option<String>,
    pub count_query: Option<Box<TargetQuery>>,
}

/// Python `ValueSource` の dataclass 既定値（`base=0`／`multiplier=1`／`divisor=1`）。
/// **`derive(Default)` は使えない**（`multiplier`／`divisor` が 0 になり評価が変わる）。
impl Default for ValueSource {
    fn default() -> ValueSource {
        ValueSource {
            base: 0,
            dynamic_source: None,
            multiplier: 1,
            divisor: 1,
            ref_id: None,
            count_query: None,
        }
    }
}

/// `Condition.value`（Python は「何でも入る」欄）。
///
/// **契約の追補（WP `rs-p3-core`・append-only）**: 当初の契約は `int | str | ValueSource` だったが、
/// 実データ（`opcg_effects.json`）の `Condition.value` は **tuple / list / dict / bool** も取る
/// （`EVENT_THIS_TURN`=("名前",N)・`FIELD_ALL_TRAIT`=("特徴",contains)・`LEADER_TRAIT`=["A","B"]・
/// `OPPONENT_REMOVAL`／`REVEALED_CARD_TRAIT`=dict）。既存の 3 種は意味を変えず、表現できない値を
/// 黙って落とさないために変種を足す。
///
/// exporter（`export_effects_json.py::_encode`）は **tuple も list も JSON 配列**にするので、
/// Rust 側では両者を区別できない。現行 DB では区別が要る型（`EVENT_THIS_TURN`／`SOURCE_STATE`／
/// `LEADER_STATE`／`FIELD_ALL_TRAIT`／`HAS_CHARACTER`＝Python が `isinstance(v, tuple)` で分岐する型）
/// の値は**全て tuple**であり、list を取る型（`LEADER_NAME`／`LEADER_TRAIT`）は Python 側が
/// `(list, tuple)` の両方を受けるため、`List` を「Python の tuple」として扱って一致する
/// （`cond.rs` の各分岐に注記あり）。
#[derive(Debug, Clone, PartialEq)]
pub enum CondValue {
    Null,
    Bool(bool),
    Int(i32),
    Str(String),
    /// Python の tuple／list（exporter はどちらも JSON 配列にする）。
    List(Vec<CondValue>),
    /// Python の dict（キー順は JSON の出現順＝exporter が `sort_keys` で書いた昇順）。
    Dict(Vec<(String, CondValue)>),
    Source(ValueSource),
}

impl CondValue {
    /// Python `isinstance(value, int)`（`bool` は除く＝現行 DB に bool 単体の値は無い）。
    pub fn as_int(&self) -> Option<i32> {
        match self {
            CondValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            CondValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[CondValue]> {
        match self {
            CondValue::List(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&[(String, CondValue)]> {
        match self {
            CondValue::Dict(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    /// dict の 1 キー（Python `val.get(key)`）。
    pub fn dict_get(&self, key: &str) -> Option<&CondValue> {
        self.as_dict()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// Python `Condition`。
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub ty: ConditionType,
    pub target: Option<TargetQuery>,
    pub player: PlayerRef,
    pub operator: CompareOperator,
    pub value: CondValue,
    pub args: Vec<Condition>,
    pub raw_text: String,
}

/// Python `GameAction`。
#[derive(Debug, Clone, PartialEq)]
pub struct GameAction {
    pub ty: ActionType,
    pub target: Option<TargetQuery>,
    pub value: ValueSource,
    pub duration: Duration,
    pub status: Option<String>,
    pub destination: Option<ZoneRef>,
    pub is_rest: Option<bool>,
    /// "TOP" | "BOTTOM"
    pub dest_position: Option<String>,
    pub raw_text: String,
    pub sub_effect: Option<Box<EffectNode>>,
    pub is_optional: bool,
    pub delay: Option<String>,
    pub face_up: Option<bool>,
}

/// Python `EffectNode` の 4 種（`effect_node_from_dict` の判別: actions→Sequence／
/// condition+if_true→Branch／options→Choice／type→GameAction）。
#[derive(Debug, Clone, PartialEq)]
pub enum EffectNode {
    Action(GameAction),
    Sequence(Vec<EffectNode>),
    Branch {
        condition: Option<Condition>,
        if_true: Option<Box<EffectNode>>,
        if_false: Option<Box<EffectNode>>,
    },
    Choice {
        message: String,
        options: Vec<EffectNode>,
        option_labels: Vec<String>,
        player: PlayerRef,
    },
}

/// Python `Ability`。
#[derive(Debug, Clone, PartialEq)]
pub struct Ability {
    pub trigger: TriggerType,
    pub condition: Option<Condition>,
    pub cost: Option<EffectNode>,
    pub effect: Option<EffectNode>,
    pub raw_text: String,
    pub cost_optional: bool,
}

/// 全能力の表。`model::CardMaster.ability_ids` はここへの index（カード内の順序＝Python の
/// `master.abilities` の順序を保つ。`ability_used_this_turn` のキー＝カード内 index）。
#[derive(Debug, Default, Clone)]
pub struct AbilityTable {
    pub abilities: Vec<Ability>,
}

impl AbilityTable {
    pub fn get(&self, id: u32) -> Option<&Ability> {
        self.abilities.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.abilities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.abilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_names_roundtrip() {
        for t in ActionType::ALL {
            assert_eq!(ActionType::from_name(t.name()), Some(*t));
        }
        for t in TriggerType::ALL {
            assert_eq!(TriggerType::from_name(t.name()), Some(*t));
        }
        for t in ConditionType::ALL {
            assert_eq!(ConditionType::from_name(t.name()), Some(*t));
        }
        assert_eq!(ActionType::ALL.len(), 62);
        assert_eq!(ConditionType::ALL.len(), 42);
        assert_eq!(TriggerType::ALL.len(), 24);
    }

    #[test]
    fn effects_json_uses_only_known_names() {
        // opcg_effects.json が手元にあるときだけ（生成物・git 管理外）。
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../opcg_sim/data/opcg_effects.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        let mut unknown = Vec::new();
        // 判別: Ability は `cost_optional` を持つ／GameAction・Condition は `raw_text` を持つ
        // （カード本体の `type`（LEADER 等）や TargetQuery の `lacks_trigger` は検査対象外）。
        fn walk(v: &serde_json::Value, unknown: &mut Vec<String>) {
            match v {
                serde_json::Value::Object(o) => {
                    if o.contains_key("cost_optional") {
                        if let Some(serde_json::Value::String(t)) = o.get("trigger") {
                            if TriggerType::from_name(t).is_none() {
                                unknown.push(format!("trigger:{t}"));
                            }
                        }
                    }
                    if o.contains_key("raw_text") {
                        if let Some(serde_json::Value::String(t)) = o.get("type") {
                            let is_cond = o.contains_key("args") || o.contains_key("operator");
                            let known = if is_cond {
                                ConditionType::from_name(t).is_some()
                            } else {
                                ActionType::from_name(t).is_some()
                            };
                            if !known {
                                unknown.push(format!("{}:{t}", if is_cond { "cond" } else { "action" }));
                            }
                        }
                    }
                    for x in o.values() {
                        walk(x, unknown);
                    }
                }
                serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, unknown)),
                _ => {}
            }
        }
        walk(&doc["cards"], &mut unknown);
        unknown.sort();
        unknown.dedup();
        assert!(unknown.is_empty(), "unknown enum names in effects JSON: {unknown:?}");
    }
}
