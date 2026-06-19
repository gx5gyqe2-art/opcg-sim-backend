from dataclasses import dataclass, field
from typing import List, Optional, Any, Set, Dict, Tuple
import uuid
import os
import json
import copy as _copy
from .enums import CardType, Color, Attribute, ActionType, Phase, Player
from .effect_types import Ability
from ..core import journal
from ..core.journal import JournaledSet, JournaledDict, record_attr

def load_shared_constants():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "shared_constants.json"))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            pass
    else:
        pass
    return {}

CONST = load_shared_constants()

if not CONST:
    # フォールバック
    CONST = {
        "CARD_PROPERTIES": {
            "UUID": "uuid", 
            "CARD_ID": "card_id",
            "NAME": "name", 
            "POWER": "power", 
            "COUNTER": "counter",
            "ATTRIBUTE": "attribute",
            "ATTACHED_DON": "attached_don", 
            "IS_REST": "is_rest", 
            "OWNER_ID": "owner_id"
        }
    }


# opcg_sim/src/models/models.py

@dataclass(frozen=True)
class CardMaster:
    card_id: str
    name: str
    type: CardType
    colors: List[Color] # 変更: color -> colors (リスト化)
    cost: int
    power: int
    counter: int
    attribute: Attribute
    traits: List[str]
    effect_text: str
    trigger_text: str
    life: int
    block_icon: str = ""
    keywords: Set[str] = field(default_factory=set)
    abilities: Tuple[Ability, ...] = field(default_factory=tuple)
    # 「ルール上、このカードはカード名を「X」（と「Y」）としても扱う」(EB04-038 ロシナンテ&ロー、
    # そげキング=ウソップ 等) の別名。RULE_PROCESSING は実行時 no-op のため、名前照合は
    # ここに保持した別名を all_names/matches_name 経由で参照して解決する。
    name_aliases: Tuple[str, ...] = field(default_factory=tuple)

    def __deepcopy__(self, memo):
        """カード定義は不変（frozen＝実行時状態は CardInstance 側のみ）なので、
        deepcopy（CPU 先読みの GameManager.clone 等）では複製せず共有する。
        効果木（abilities）まで毎回 deepcopy する重さを避けるための最適化。"""
        memo[id(self)] = self
        return self

    @property
    def all_names(self) -> List[str]:
        """カードが名乗る全カード名（本来名＋ルール上の別名）。"""
        return [self.name, *self.name_aliases]

    def matches_name(self, query_name: str, partial: bool = False) -> bool:
        """query_name が本来名または別名のいずれかに一致するか。
        partial=True なら query_name が各名の部分文字列であれば一致（テキスト準拠の
        「「X」を含む」「「X」がKOされる」等の包含判定に合わせる）。"""
        if partial:
            return any(query_name in n for n in self.all_names)
        return query_name in self.all_names

    def to_dict(self):
        return {
            "uuid": self.card_id,
            "name": self.name,
            "type": self.type.name if hasattr(self.type, "name") else str(self.type),
            "color": [c.value for c in self.colors] if self.colors else [], # 変更: リスト内の各色の値を出力
            "cost": self.cost,
            "power": self.power,
            "counter": self.counter,
            "attributes": [self.attribute.value] if hasattr(self.attribute, "value") else [],
            "text": self.effect_text,
            "traits": self.traits,
            "life": self.life,
            "trigger_text": self.trigger_text,
            "block_icon": self.block_icon
        }



@dataclass
class CardInstance:
    master: CardMaster
    owner_id: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_rest: bool = False
    is_newly_played: bool = False
    attached_don: int = 0
    is_face_up: bool = False
    power_buff: int = 0
    cost_buff: int = 0
    # PASSIVE/YOUR_TURN 由来のパワー修正の再計算レイヤ。_apply_passive_effects が
    # 毎回 0 にリセットして再適用する（power_buff に加えると再計算のたびに累積する）。
    passive_power: int = 0
    # PASSIVE 由来のパワー上書きの再計算レイヤ。即時効果の base_power_override を
    # 再計算リセットから守るために分離する（即時の同値パワー効果が消えないように）。
    passive_power_override: Optional[int] = None
    # PASSIVE 由来のカウンター値修正（「手札の…はカウンター+2000になる」）。
    # _apply_passive_effects が毎回リセットして再適用する。
    passive_counter: int = 0
    base_power_override: Optional[int] = None
    # 「このターン中、コスト0にする」等のコスト絶対値セット。base_power_override と対称で、
    # _apply_passive_effects ではリセットせず reset_turn_status のみで失効させる。
    base_cost_override: Optional[int] = None
    current_keywords: Set[str] = field(default_factory=JournaledSet)
    flags: Set[str] = field(default_factory=JournaledSet)
    negated: bool = False
    ability_disabled: bool = False
    ability_used_this_turn: Dict[int, int] = field(default_factory=JournaledDict)
    # ContinuousEffectManager 専用。reset_turn_status ではクリアしない
    # （ターン境界を跨いで存続する期間付き効果を保持するため）。
    timed_power: int = 0
    timed_flags: Set[str] = field(default_factory=JournaledSet)
    # 期間付きのコスト増減（「このターン中、コスト-N」等）。cost_buff は
    # _apply_passive_effects で毎回 0 にリセットされるため、期間付き分はここに保持する。
    timed_cost: int = 0
    # 効果で付与されたキーワード（【ブロッカー】等）。current_keywords は
    # _apply_passive_effects で master のコピーに毎回リセットされるため、付与分は
    # ここに保持して消えないようにする。失効は ContinuousEffectManager が管理。
    timed_keywords: Set[str] = field(default_factory=JournaledSet)

    def __setattr__(self, name, value):
        # 差分巻き戻し（journal.transaction 中のみ記録）。不活性時はグローバル 1 読みで素通り。
        if journal._active is not None:
            record_attr(self, name, self.__dict__)
        object.__setattr__(self, name, value)

    def __post_init__(self):
        if not self.uuid:
            self.uuid = str(uuid.uuid4())
        self._refresh_keywords()

    def __deepcopy__(self, memo):
        """高速 deepcopy（CPU 先読みの GameManager.clone が支配的コスト＝この複製。§2.5.2）。

        汎用 deepcopy は __dict__ を内省して全フィールドを再帰コピーするため重い。CardInstance の
        フィールドは **スカラ ＋ プリミティブ（str/int）だけを要素に持つ set/dict** と、不変共有の
        `master`（CardMaster.__deepcopy__ が self を返す）に限られる。よって set/dict は浅コピーで
        独立な深コピーになり、スカラ/master はそのまま共有してよい。想定外の可変属性のみ安全側で
        汎用 deepcopy にフォールバックする（正しさを保ったまま再帰・内省コストを排除）。
        """
        new = CardInstance.__new__(CardInstance)
        memo[id(self)] = new
        nd = new.__dict__
        for k, v in self.__dict__.items():
            t = type(v)
            # set/dict は journaled 型を維持して複製する（root clone 上で make/unmake を効かせるため）。
            if t is set or t is JournaledSet:
                nd[k] = JournaledSet(v)
            elif t is dict or t is JournaledDict:
                nd[k] = JournaledDict(v)
            elif v is None or t is int or t is str or t is bool or t is float or t is CardMaster:
                nd[k] = v
            else:
                nd[k] = _copy.deepcopy(v, memo)
        return new

    def _refresh_keywords(self):
        if self.ability_disabled:
            self.current_keywords = JournaledSet()
            return
        self.current_keywords = JournaledSet(self.master.keywords)
        for ability in self.master.abilities:
            if not hasattr(ability, 'actions'):
                continue
            for action in ability.actions:
                if action.type == ActionType.KEYWORD:
                    keyword_val = getattr(action, 'details', None)
                    if keyword_val:
                        self.current_keywords.add(keyword_val)

    @property
    def is_effect_negated(self) -> bool:
        """このカードの効果が無効化されているか。
        ability_disabled（同ターン内のフラグ）に加え、継続効果（timed_flags の
        "EFFECTS_DISABLED"）も見る。後者は reset_turn_status でクリアされず、
        「このターン中」/「次の相手のターン終了時まで」の無効化を途中のアクションで
        解除されずに維持する（OP09-093 等）。"""
        return self.ability_disabled or ("EFFECTS_DISABLED" in self.timed_flags)

    def has_keyword(self, keyword: str) -> bool:
        """カードが指定キーワードを持つか（本来のキーワード＋効果で付与された分）。
        効果が無効化されている場合はキーワード能力も持たない。"""
        if self.is_effect_negated:
            return False
        return keyword in self.current_keywords or keyword in self.timed_keywords

    def get_power(self, is_my_turn: bool) -> int:
        if self.master.type not in [CardType.LEADER, CardType.CHARACTER]:
            return 0
        override = (self.base_power_override if self.base_power_override is not None
                    else self.passive_power_override)
        base = override if override is not None else self.master.power
        buff = self.power_buff + self.timed_power + self.passive_power
        # 付与ドン!!のパワー(+1000/枚)は自分のターン中のみ適用する。
        # （マーカー自体は相手ターンも残るが、パワー上昇は自ターンだけ）
        don_power = (self.attached_don * 1000) if is_my_turn else 0
        return base + buff + don_power

    @property
    def current_counter(self) -> int:
        """カウンター値（基礎値＋効果による修正）。apply_counter が参照する。"""
        return (self.master.counter or 0) + self.passive_counter

    @property
    def current_cost(self) -> int:
        base = self.base_cost_override if self.base_cost_override is not None else self.master.cost
        result = base + self.cost_buff + self.timed_cost
        return max(0, result)

    def reset_turn_status(self, keep_don: bool = False, clear_usage: bool = False):
        # NOTE: 本メソッドは「このターン中」の一時効果（パワー/コスト/フラグ等）の解除に
        #   使われ、ターン境界だけでなく戦闘終了時(gamestate の battle 終了)や領域移動時
        #   (move_card)などターン途中でも頻繁に呼ばれる。そのため【ターン1回】の使用回数
        #   (ability_used_this_turn)はここで無条件にクリアしてはならない（クリアすると
        #   戦闘のたびにカウンタが戻り、ターン1回効果が複数回使えてしまう）。
        #   使用回数のリセットは clear_usage=True を明示した呼び出し（ターン境界、及び
        #   カードが場を離れて新規状態になる領域移動）でのみ行う。
        self.power_buff = 0
        self.cost_buff = 0
        self.base_power_override = None
        self.passive_power_override = None
        self.base_cost_override = None
        self.negated = False
        self.ability_disabled = False
        self.flags.clear()
        if clear_usage:
            self.ability_used_this_turn.clear()
        # keep_don=True のときは付与ドン!!を維持する（相手ターン開始時の状態リセットでは
        # 付与ドン!!を剥がさず、自分の次のリフレッシュフェイズまで残す）。
        if not keep_don:
            self.attached_don = 0
        self.is_newly_played = False
        self._refresh_keywords()

    def to_dict(self, is_my_turn: bool = True):
        props = CONST.get('CARD_PROPERTIES', {})
        return {
            props.get('UUID', 'uuid'): self.uuid,
            props.get('CARD_ID', 'card_id'): self.master.card_id,
            props.get('NAME', 'name'): self.master.name,
            # 付与ドン!!のパワーは自分のターン中のみ反映する（相手ターンは加算しない）。
            props.get('POWER', 'power'): self.get_power(is_my_turn=is_my_turn),
            props.get('COUNTER', 'counter'): self.master.counter,
            props.get('ATTRIBUTE', 'attribute'): self.master.attribute.value,
            props.get('COST', 'cost'): self.current_cost,
            props.get('TRAITS', 'traits'): list(self.master.traits),
            props.get('TEXT', 'text'): self.master.effect_text,
            props.get('TYPE', 'type'): self.master.type.value,
            props.get('IS_REST', 'is_rest'): self.is_rest,
            props.get('IS_FACE_UP', 'is_face_up'): self.is_face_up,
            props.get('ATTACHED_DON', 'attached_don'): self.attached_don,
            props.get('OWNER_ID', 'owner_id'): self.owner_id,
            props.get('KEYWORDS', 'keywords'): list(self.current_keywords | self.timed_keywords),
            props.get('TRIGGER_TEXT', 'trigger_text'): self.master.trigger_text or '',
            props.get('ABILITY_DISABLED', 'ability_disabled'): self.ability_disabled,
            props.get('IS_FROZEN', 'is_frozen'): 'FREEZE' in self.flags,
        }

@dataclass
class DonInstance:
    owner_id: str
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_rest: bool = False
    attached_to: Optional[str] = None
    # 「次の相手のリフレッシュフェイズでアクティブにならない」(OP07-026 ドン側)。
    # refresh_all がレストのドン!!をアクティブに戻す際、これが立っていれば1回スキップし
    # フラグを下ろす（キャラの flags["FREEZE"] と同じ1回限りのフリーズ）。
    is_frozen: bool = False

    def __setattr__(self, name, value):
        if journal._active is not None:
            record_attr(self, name, self.__dict__)
        object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        """高速 deepcopy（CardInstance と同趣旨・§2.5.2）。全フィールドがスカラなので直接複製する。"""
        new = DonInstance.__new__(DonInstance)
        memo[id(self)] = new
        new.__dict__.update(self.__dict__)
        return new

    def to_dict(self):
        props = CONST.get('CARD_PROPERTIES', {})
        # ドン!!返却の選択 UI 等で状態を区別できるよう、名前に付与中/レストを併記する。
        if self.attached_to:
            display_name = "ドン!!(付与中)"
        elif self.is_rest:
            display_name = "ドン!!(レスト)"
        else:
            display_name = "ドン!!"
        return {
            props.get('UUID', 'uuid'): self.uuid,
            props.get('OWNER_ID', 'owner_id'): self.owner_id,
            props.get('IS_REST', 'is_rest'): self.is_rest,
            "attached_to": self.attached_to,
            props.get('CARD_ID', 'card_id'): "DON",
            # 【追加】CardSchemaのバリデーションを通すためのダミー値
            props.get('NAME', 'name'): display_name,
            props.get('TYPE', 'type'): "DON",
            props.get('ATTRIBUTE', 'attribute'): "Special",
            props.get('POWER', 'power'): 0,
            props.get('COST', 'cost'): 0,
            props.get('COUNTER', 'counter'): 0,
            props.get('TRAITS', 'traits'): [],
            props.get('TEXT', 'text'): "",
            props.get('IS_FACE_UP', 'is_face_up'): True,
            props.get('ATTACHED_DON', 'attached_don'): 0,
            props.get('KEYWORDS', 'keywords'): []
        }
