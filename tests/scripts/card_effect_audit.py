"""カード効果の監査（新カード追加・既存カード点検）＝ skill `card-effect-audit` の機械部分。

カード DB（`opcg_sim/data/opcg_cards.json`）を丸ごと差し替えたときに、各カードの効果が
「パーサで解釈でき・エンジンが読み込め・本文の句がひとつ残らず実行経路に乗る」かを機械的に
洗い出し、**意味のレビュー（Claude が本文と解析結果を読み比べる）**のための資料を作る。
レビューの結果は確認済み台帳（`tests/fixtures/card_audit_ledger.json`）へ記録する。

    python tests/scripts/card_effect_audit.py scan                  # 台帳で未確認／変化したカードだけ
    python tests/scripts/card_effect_audit.py scan --all            # 全走査（既存カードの点検）
    python tests/scripts/card_effect_audit.py scan --cards OP01-016,EB03-012
    python tests/scripts/card_effect_audit.py scan --set OP15       # 品番の接頭辞で絞る
    python tests/scripts/card_effect_audit.py show OP01-016         # レビュー資料を表示
    python tests/scripts/card_effect_audit.py record OP01-016 --verdict ok
    python tests/scripts/card_effect_audit.py record OP09-081 --verdict ng --note "常在が登場時扱い"
    python tests/scripts/card_effect_audit.py status                # 台帳の集計

**検査の層**（どれも「正しさの証明」ではなく、レビューの優先度づけと見落とし防止の網）:

1. 読み込み   … パーサ出力（効果 JSON）を Rust が読めるか。未知の効果種別があるとここで落ちる
                 ＝**エンジン未実装**の検出（`LOAD_FAIL`）。
2. パーサ     … 未実装句（`PARSE_OTHER`）・ルール外のレガシー解析（`PARSER_LEGACY`）・
                 効果オラクル（`ORACLE_*`＝ターン1回／〜まで の取りこぼし）。
3. 本文照合   … 本文の文字を、解析結果の各ノードが持つ `raw_text` で覆えるか。覆えなかった
                 断片＝**本文の句が解析結果に無い**（`UNCOVERED_TEXT`）。【】見出しは
                 トリガー種別・条件・キーワードと対応させる（`TAG_*`・未知の見出し＝新しい仕組み）。
4. 実行       … Rust の効果プローブ（`opcg_engine.effect_probe`）で能力を発動し、
                 条件を真に固定・選べるだけ選ぶ・`Choice` は全選択肢・`Branch` の else は
                 条件を偽に固定、で回す。解析結果の各アクションに対応するイベントが出たかを
                 照合する（`ACTION_NOT_FIRED`／`PROBE_ERROR`）。

**台帳の指紋**: `text_sha`（カード本文）・`ast_sha`（解析結果）・`probe_sha`（プローブの
イベント列）。確認後に本文が変われば `TEXT_CHANGED`、パーサ／エンジンの変更で解析結果や
挙動が変われば `IMPL_CHANGED` になり、再確認の対象へ戻る。
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

ROOT = _bootstrap.ROOT
CARD_DB = os.path.join(ROOT, "opcg_sim", "data", "opcg_cards.json")
CARD_DB_REL = "opcg_sim/data/opcg_cards.json"
LEDGER = os.path.join(ROOT, "tests", "fixtures", "card_audit_ledger.json")
DEFAULT_OUT = os.path.join("/tmp", "card_effect_audit")
LEDGER_VERSION = 1

# ---------------------------------------------------------------------------
# 指摘の種類（重さ: E=実装の欠陥が確実／W=要確認／I=参考）
# ---------------------------------------------------------------------------

FLAG_LEVEL = {
    "LOAD_FAIL": "E",          # 効果 JSON を Rust が読めない（未知の効果種別＝エンジン未実装）
    "PARSE_OTHER": "E",        # 未実装句（ActionType.OTHER＝実行時に何もしない）
    "NO_ABILITY": "E",         # 本文があるのに能力が 1 つも無い
    "PROBE_ERROR": "E",        # 発動・解決中にエンジンが例外
    "UNKNOWN_TAG": "E",        # 見たことのない【】見出し＝新しい仕組みの可能性
    "ORACLE": "E",             # 効果オラクル（ターン1回／〜まで の取りこぼし・OTHER）
    "UNCOVERED_TEXT": "W",     # 本文の句が解析結果のどのノードにも対応しない
    "ACTION_NOT_FIRED": "W",   # 解析結果のアクションがプローブで一度も実行されない
    "TAG_TRIGGER_MISMATCH": "W",  # 【登場時】等の見出しに対応する能力トリガーが無い
    "TAG_CONTEXT_MISSING": "W",   # 【ターン1回】【ドン!!×N】等に対応する条件が無い
    "KEYWORD_MISSING": "E",    # 【ブロッカー】等が keywords にも付与効果にも無い（＝どこも処理しない）
    "DRAIN_LIMIT": "W",        # 対話が上限まで続いた（無限ループの疑い）
    "PARSER_LEGACY": "I",      # 合成ルールに当たらずレガシー解析へ落ちた句（結果は要確認）
    "FIRED_EMPTY": "I",        # 実行されたが対象 0 枚（汎用盤面に該当カードが無いだけのことが多い）
    "DEFERRED": "I",           # 置換効果の中身など、後の出来事で実行される（プローブでは確かめない）
}

# 見出し → 対応すべきトリガー種別（いずれか）。
TAG_TRIGGERS = {
    "登場時": {"ON_PLAY"},
    "起動メイン": {"ACTIVATE_MAIN"},
    "メイン": {"ACTIVATE_MAIN"},
    "トリガー": {"TRIGGER"},
    "アタック時": {"ON_ATTACK"},
    "カウンター": {"COUNTER"},
    "KO時": {"ON_KO"},
    "ブロック時": {"ON_BLOCK"},
    "相手のアタック時": {"ON_OPP_ATTACK"},
    "自分のターン終了時": {"TURN_END"},
}
# 見出し → トリガー種別か条件種別のどちらかで表現されていればよいもの。
TAG_CONTEXTS = {
    "自分のターン中": ({"YOUR_TURN"}, {"CONTEXT"}),
    "相手のターン中": ({"OPPONENT_TURN"}, {"CONTEXT"}),
    "ターンN回": (set(), {"TURN_LIMIT"}),
    "ドン!!×N": (set(), {"HAS_DON"}),
}
# 見出し → キーワード（`keywords` か、付与効果の本文に現れればよい）。
TAG_KEYWORDS = {"ブロッカー", "速攻", "ダブルアタック", "バニッシュ", "速攻:キャラ", "ブロック不可"}

# プローブのイベント名 ↔ 解析結果のアクション種別のずれ（実行しているがイベント名が違う／
# イベントを出さずに対話だけで済ませる種別）。照合で「実行された」とみなす根拠にする。
EVENT_ALIASES: Dict[str, set] = {}
# イベントを自分では出さず、別能力の中身を実行するもの（実行されれば何かしらイベントが出る）。
ANY_EVENT_EVIDENCE = {"EXECUTE_MAIN_EFFECT", "EXECUTE_EVENT"}
# 中身（sub_effect）を登録するだけで、実行は後の出来事に任せるもの。
DEFERRING_ACTIONS = {"REPLACE_EFFECT"}
INTERACTION_EVIDENCE = {
    "DECK_BOTTOM": {"ARRANGE_DECK"},
    "ORDER_LIFE": {"ARRANGE_DECK"},
    "SELECT": {"SELECT_TARGET"},
    "DECLARE_COST": {"DECLARE_COST"},
}

# 本文照合で無視する断片（接続語・助詞・記号だけの残り）。
_IGNORABLE = re.compile(
    r"^(?:その後|そうした場合|場合|以下から1つを選ぶ|ことができる|できる|"
    r"し|させ|て|き|り|を|は|が|の|か|と|に|で|も|、|。|:|・|/|,|\.|\s)+$"
    # 「キャラかドン!!1枚までを、レストにする」はキャラ側とドン側の 2 アクションに割れ、
    # 共有する述語が残る。
    r"|^を、?(?:レストに|アクティブに)する。?(?:\s*/|その後、?)?$"
)
# 残った断片の前後に付く接続部分（「引き、」の「き、」・「。その後、」等）。
_EDGE_HEAD = re.compile(r"^(?:[、。:・/,\s]|その後、?|[きしてり]、)+")
_EDGE_TAIL = re.compile(r"(?:[、。:・/,\s]|その後、?)+$")
_TURN_LIMIT_PHRASE = re.compile(r"^(?:このキャラは)?ターンに?1回、?$")
_OPP_CHOICE_PHRASE = re.compile(r"^:?相手は以下から1つを選ぶ。?(?:\s*/\s*・)?$")
_TRIGGER_PHRASE = re.compile(r"(時、?|時か|発動できる。?)$")
_TAG_RE = re.compile(r"【([^】]*)】")
_PAREN_RE = re.compile(r"\([^()]*\)")


def nfkc(s: Optional[str]) -> str:
    return unicodedata.normalize("NFKC", s or "")


def sha(obj: Any) -> str:
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:16]


def heading_tags(text: str) -> List[re.Match]:
    """見出しとして置かれた【】だけを返す（文頭・空白・「/」・別の見出しの直後）。

    「【トリガー】を持つ」「【メイン】効果を発動する」「相手が【ブロッカー】を発動した時」の
    ような文中の参照は見出しではない（照合では本文の一部として扱う）。
    """
    out = []
    for m in _TAG_RE.finditer(text):
        before = text[:m.start()].rstrip()
        if not before or before[-1] in "/】・。)":
            out.append(m)
    return out


def tag_key(tag: str) -> str:
    """【ターン1回】→ ターンN回・【ドン!! ×2】→ ドン!!×N（数字と空白を潰す）。"""
    return re.sub(r"\s+", "", re.sub(r"\d+", "N", nfkc(tag)))


# ---------------------------------------------------------------------------
# 効果構造（効果 JSON の dict）の走査
# ---------------------------------------------------------------------------

def walk_nodes(node: Any) -> Iterable[dict]:
    """dict／list を再帰で辿り、`node` キーを持つ dict を全て返す。"""
    if isinstance(node, dict):
        if "node" in node:
            yield node
        for v in node.values():
            yield from walk_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk_nodes(v)


def flatten_actions(ability: dict) -> List[dict]:
    """能力のアクションを実行順に並べる。`path` は表示用の位置、`alt` は分岐／選択肢の内側か。"""
    out: List[dict] = []

    def rec(n: Any, path: str, alt: Optional[str]) -> None:
        if not isinstance(n, dict):
            return
        kind = n.get("node")
        if kind == "GameAction":
            # 「このターン終了時、〜」（delay）はターン終了まで実行されない。
            out.append({"path": path, "type": n.get("type"), "raw_text": n.get("raw_text") or "",
                        "alt": "deferred" if n.get("delay") else alt,
                        "target": n.get("target"), "node": n})
            if n.get("sub_effect"):
                deferred = "deferred" if n.get("type") in DEFERRING_ACTIONS else alt
                rec(n["sub_effect"], path + ".sub", deferred)
        elif kind == "Sequence":
            for i, a in enumerate(n.get("actions") or []):
                rec(a, f"{path}.{i}", alt)
        elif kind == "Branch":
            rec(n.get("if_true"), path + ".then", "then")
            rec(n.get("if_false"), path + ".else", "else")
        elif kind == "Choice":
            for i, o in enumerate(n.get("options") or []):
                rec(o, f"{path}.opt{i}", f"opt{i}")

    rec(ability.get("cost"), "cost", None)
    rec(ability.get("effect"), "effect", None)
    return out


def expected_paths(ability: dict, choice_path: List[int], force: bool) -> List[str]:
    """プローブの 1 経路で通るはずのアクションの `path`（`flatten_actions` と同じ表記）を実行順に。

    `Choice` は `choice_path` を先頭から 1 つずつ消費して選ぶ（尽きたら 0）、`Branch` は
    条件を `force` に固定したときの側だけを辿る。
    """
    out: List[str] = []
    pos = [0]

    def rec(n: Any, path: str) -> None:
        if not isinstance(n, dict):
            return
        kind = n.get("node")
        if kind == "GameAction":
            out.append(path)
            if n.get("sub_effect"):
                rec(n["sub_effect"], path + ".sub")
        elif kind == "Sequence":
            for i, a in enumerate(n.get("actions") or []):
                rec(a, f"{path}.{i}")
        elif kind == "Branch":
            if force:
                rec(n.get("if_true"), path + ".then")
            else:
                rec(n.get("if_false"), path + ".else")
        elif kind == "Choice":
            opts = n.get("options") or []
            k = choice_path[pos[0]] if pos[0] < len(choice_path) else 0
            pos[0] += 1
            if opts:
                k = min(max(k, 0), len(opts) - 1)
                rec(opts[k], f"{path}.opt{k}")

    rec(ability.get("cost"), "cost")
    rec(ability.get("effect"), "effect")
    return out


def has_else(ability: dict) -> bool:
    return any(n.get("node") == "Branch" and n.get("if_false") for n in walk_nodes(ability))


def condition_types(ability: dict) -> set:
    return {n.get("type") for n in walk_nodes(ability) if n.get("node") == "Condition"}


def node_texts(ability: dict) -> List[str]:
    """本文照合に使う断片（アクション・条件・対象の raw_text と、選択肢の見出し）。"""
    out = []
    for n in walk_nodes({k: v for k, v in ability.items() if k != "raw_text"}):
        if n.get("node") in ("GameAction", "Condition", "TargetQuery") and n.get("raw_text"):
            out.append(nfkc(n["raw_text"]))
        elif n.get("node") == "Choice":
            out.append(nfkc(n.get("message")))
            out.extend(nfkc(x) for x in n.get("option_labels") or [])
    return out


# ---------------------------------------------------------------------------
# 本文照合（本文の句が解析結果のどこかに対応しているか）
# ---------------------------------------------------------------------------

def card_text(card: dict) -> str:
    eff, trig = nfkc(card.get("effect_text")), nfkc(card.get("trigger_text"))
    if eff.strip() in ("なし", "None"):
        eff = ""
    return eff + ((" / " + trig) if trig else "")


def _cover(text: str, cov: List[bool], fragment: str, lo: int = 0, hi: Optional[int] = None) -> bool:
    """`text[lo:hi]` の中で `fragment` に当たる文字を覆う。当たったら True。

    断片はパーサが正規化した本文（「引き」→「引く」、見出しの除去など）なので、完全一致が
    無ければ `difflib` で 2 文字以上の一致ブロックを拾い、断片の半分以上が一致したときだけ覆う。
    """
    hi = len(text) if hi is None else hi
    f = _TAG_RE.sub(r"\1", fragment).strip()
    if not f:
        return False
    region = text[lo:hi]
    hits = [m.start() for m in re.finditer(re.escape(f), region)]
    if hits:
        for i in hits:
            for j in range(lo + i, lo + i + len(f)):
                cov[j] = True
        return True
    sm = difflib.SequenceMatcher(None, region, f, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
    if sum(b.size for b in blocks) < max(2, len(f) // 2):
        return False
    for b in blocks:
        for j in range(lo + b.a, lo + b.a + b.size):
            cov[j] = True
    return True


def _ability_span(text: str, raw: str) -> Optional[Tuple[int, int]]:
    """能力の raw_text が本文のどこにあるか（完全一致→あいまい一致の順）。"""
    raw = raw.strip()
    if not raw:
        return None
    i = text.find(raw)
    if i >= 0:
        return i, i + len(raw)
    sm = difflib.SequenceMatcher(None, text, raw, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
    if sum(b.size for b in blocks) < len(raw) // 2:
        return None
    return blocks[0].a, blocks[-1].a + blocks[-1].size


def uncovered_fragments(text: str, abilities: List[dict]) -> List[str]:
    """本文のうち、解析結果のどのノードにも覆われない部分を返す（見出し・注釈の括弧・接続語は除く）。

    能力ごとに本文上の範囲（能力の raw_text）を特定し、その能力のノードはその範囲の中でだけ
    照合する。同じ句が 2 つの能力にある（【登場時】と【トリガー】で同じドロー等）とき、
    片方の能力だけが持っていても両方が覆われてしまうのを防ぐ。
    """
    # 文中の参照（「【トリガー】を持つ」等）は括弧だけ外す（パーサの raw_text も外している）。
    heads = {m.start() for m in heading_tags(text)}
    text = _TAG_RE.sub(lambda m: m.group(0) if m.start() in heads else m.group(1), text)
    cov = [False] * len(text)
    for m in heading_tags(text) + list(_PAREN_RE.finditer(text)):
        for j in range(m.start(), m.end()):
            cov[j] = True
    for ab in abilities:
        span = _ability_span(text, nfkc(ab.get("raw_text")))
        lo, hi = span if span else (0, len(text))
        for f in node_texts(ab):
            if not _cover(text, cov, f, lo, hi) and span:
                _cover(text, cov, f)      # 範囲の取り違えに備えて本文全体でも探す
    masked = "".join("\x00" if cov[i] else ch for i, ch in enumerate(text))
    pieces = [p.strip() for p in re.split(r"\x00+", masked)]
    pieces = [p for p in pieces if p and not _IGNORABLE.match(p)]
    return [t for t in (_EDGE_TAIL.sub("", _EDGE_HEAD.sub("", p)) for p in pieces) if t]


# ---------------------------------------------------------------------------
# 1 カードの静的検査
# ---------------------------------------------------------------------------

def static_checks(card: dict, parser_stats: dict) -> Tuple[List[dict], List[str]]:
    """戻り値は (指摘のリスト, 誘発条件の句＝トリガー種別で表現されていて raw_text に出ない句)。"""
    flags: List[dict] = []
    abilities = card.get("abilities") or []
    text = card_text(card)
    if not text.strip():
        return flags, []
    heads = heading_tags(text)
    if not abilities:
        rest = text
        for m in sorted(heads + list(_PAREN_RE.finditer(text)), key=lambda m: -m.start()):
            rest = rest[:m.start()] + " " + rest[m.end():]
        leftover = [p for p in re.split(r"\s+", rest) if p and not _IGNORABLE.match(p)]
        if leftover:
            flags.append({"flag": "NO_ABILITY", "detail": f"本文があるのに能力が 0（{' '.join(leftover)[:80]}）"})

    for frag in parser_stats.get("fallback_other", []):
        flags.append({"flag": "PARSE_OTHER", "detail": frag})
    for frag in parser_stats.get("unmatched", []):
        if frag not in parser_stats.get("fallback_other", []):
            flags.append({"flag": "PARSER_LEGACY", "detail": frag})

    trigger_phrases: List[str] = []
    triggers = {ab.get("trigger") for ab in abilities}
    conds = set().union(*[condition_types(ab) for ab in abilities]) if abilities else set()
    granted = " ".join(nfkc(a["raw_text"]) for ab in abilities for a in flatten_actions(ab))
    for piece in uncovered_fragments(text, abilities):
        if _TRIGGER_PHRASE.search(piece):
            trigger_phrases.append(piece)
        elif piece in ("メイン", "カウンター") and any(
                a["type"] in ANY_EVENT_EVIDENCE for ab in abilities for a in flatten_actions(ab)):
            continue      # 「このカードの【カウンター】効果を発動する」の raw_text は種別名を落とす
        elif _TURN_LIMIT_PHRASE.match(piece) and "TURN_LIMIT" in conds:
            continue
        elif _OPP_CHOICE_PHRASE.match(piece) and any(
                n.get("node") == "Choice" and n.get("player") == "OPPONENT"
                for ab in abilities for n in walk_nodes(ab)):
            continue
        else:
            flags.append({"flag": "UNCOVERED_TEXT", "detail": piece})

    for m in heads:
        raw = m.group(1)
        key = tag_key(raw)
        if key in TAG_TRIGGERS:
            if not (TAG_TRIGGERS[key] & triggers):
                flags.append({"flag": "TAG_TRIGGER_MISMATCH",
                              "detail": f"【{raw}】に対応するトリガー {sorted(TAG_TRIGGERS[key])} が無い"
                                        f"（能力: {sorted(t for t in triggers if t)}）"})
        elif key in TAG_CONTEXTS:
            trig_ok, cond_ok = TAG_CONTEXTS[key]
            # 置換効果は raw_text の【相手のターン中】をエンジンが直接読む（`triggers.rs`）。
            in_replace = any(a["type"] in DEFERRING_ACTIONS and f"【{raw}】" in nfkc(a["raw_text"])
                             for ab in abilities for a in flatten_actions(ab))
            if not ((trig_ok & triggers) or (cond_ok & conds) or in_replace):
                flags.append({"flag": "TAG_CONTEXT_MISSING",
                              "detail": f"【{raw}】に対応するトリガー／条件 "
                                        f"{sorted(trig_ok | cond_ok)} が無い"})
        elif key in TAG_KEYWORDS:
            name = key.split(":")[0]
            if name not in (card.get("keywords") or []) and name not in granted:
                flags.append({"flag": "KEYWORD_MISSING",
                              "detail": f"【{raw}】が keywords にも付与効果にも無い"})
        else:
            flags.append({"flag": "UNKNOWN_TAG", "detail": f"【{raw}】（既知の見出しに無い）"})
    return flags, trigger_phrases


def parse_stats(cards: Dict[str, dict]) -> Dict[str, dict]:
    """カードごとに EffectParserV2 を通し直し、未対応句（レガシー解析・OTHER）を集める。"""
    from opcg_sim.src.effects.parser_v2 import EffectParserV2
    from opcg_sim.src.utils.loader import DataCleaner  # noqa: F401  正規化は CardLoader と同じ
    out = {}
    for cid, c in cards.items():
        p = EffectParserV2()
        if c.get("effect_text"):
            p.parse_card_text(c["effect_text"])
        if c.get("trigger_text"):
            p.parse_card_text(c["trigger_text"], as_trigger=True)
        out[cid] = {"unmatched": list(dict.fromkeys(p.unmatched)),
                    "fallback_other": list(dict.fromkeys(p.fallback_other))}
    return out


# ---------------------------------------------------------------------------
# 実行（Rust の効果プローブ）
# ---------------------------------------------------------------------------

MAX_PROBE_RUNS = 32


def _probe_runs(engine, cid: str, idx: int, trigger: str, force: bool) -> List[Tuple[list, dict]]:
    """選択肢（入れ子を含む）の全経路を回す。経路は「k 番目の CHOICE で何番を選ぶか」の列。

    1 回目は全て 0 番で回し、結果の `summary.choices`（各 CHOICE の選択肢数）から、まだ
    0 番しか試していない位置の別の番号を足した経路を幅優先で積む（上限 MAX_PROBE_RUNS）。
    """
    queue: List[list] = [[]]
    done: List[Tuple[list, dict]] = []
    while queue and len(done) < MAX_PROBE_RUNS:
        path = queue.pop(0)
        try:
            r = json.loads(engine.effect_probe(cid, trigger, idx, force, path))
        except Exception as e:  # noqa: BLE001  エンジンの例外もプローブの結果
            r = {"events": [], "summary": {"error": f"{type(e).__name__}: {e}", "choices": []}}
        done.append((path, r))
        counts = r["summary"].get("choices") or []
        for j in range(len(path), len(counts)):
            for k in range(1, counts[j]):
                queue.append(path + [0] * (j - len(path)) + [k])
    return done


def probe_ability(engine, cid: str, idx: int, ability: dict) -> dict:
    """1 能力を全経路で回し、アクションごとの実行状況を返す。

    経路は「条件を真に固定」× 選択肢の全経路（＋ else 側がある能力は「条件を偽に固定」）。
    """
    runs = _probe_runs(engine, cid, idx, ability.get("trigger"), True)
    if has_else(ability):
        runs += [(p, r, ) for p, r in _probe_runs(engine, cid, idx, ability.get("trigger"), False)]
    actions = flatten_actions(ability)
    status = ["DEFERRED" if a["alt"] == "deferred" else "NOT_FIRED" for a in actions]
    errors, run_records = [], []
    drain_limit = False
    for path, r in runs:
        s = r["summary"]
        force = s.get("force_condition")
        if s.get("error"):
            errors.append(f"force={force} choices={path}: {s['error']}")
        drain_limit |= bool(s.get("drain_limit_hit"))
        interactions = set(s.get("interactions") or [])
        events = [e for stage in r["events"] for e in stage if e.get("type") == "EFFECT"]
        run_records.append({"force": force, "choices": path,
                            "interactions": s.get("interactions") or [],
                            "events": [{k: e.get(k) for k in ("action", "targets", "value", "success", "dest")
                                        if k in e} for e in events]})
        # この経路で通るはずのアクションだけを、実行順に貪欲に照合する
        # （同じ種別が続く能力でも 1 イベント＝1 アクション）。
        expected = set(expected_paths(ability, path, force is not False))
        p = 0
        for i, a in enumerate(actions):
            if status[i] == "DEFERRED" or a["path"] not in expected:
                continue
            if a["type"] in ANY_EVENT_EVIDENCE:
                if events:
                    status[i] = "FIRED"
                continue
            names = {a["type"]} | EVENT_ALIASES.get(a["type"], set())
            j = next((k for k in range(p, len(events)) if events[k].get("action") in names), None)
            if j is not None:
                p = j + 1
                ev = events[j]
                hit = bool(ev.get("targets")) or a.get("target") is None or ev.get("value")
                if hit:
                    status[i] = "FIRED"
                elif status[i] != "FIRED":
                    status[i] = "FIRED_EMPTY"
            elif status[i] == "NOT_FIRED" and INTERACTION_EVIDENCE.get(a["type"], set()) & interactions:
                status[i] = "FIRED_VIA_INTERACTION"
    return {"actions": [{"path": a["path"], "type": a["type"], "raw_text": a["raw_text"],
                         "alt": a["alt"], "status": st} for a, st in zip(actions, status)],
            "errors": errors, "drain_limit": drain_limit, "runs": run_records}


def dynamic_checks(engine, card: dict) -> Tuple[List[dict], List[dict]]:
    flags, per_ability = [], []
    for idx, ab in enumerate(card.get("abilities") or []):
        res = probe_ability(engine, card["card_id"], idx, ab)
        per_ability.append(res)
        for e in res["errors"]:
            flags.append({"flag": "PROBE_ERROR", "ability": idx, "detail": e[:300]})
        if res["drain_limit"]:
            flags.append({"flag": "DRAIN_LIMIT", "ability": idx, "detail": "対話が上限まで続いた"})
        for a in res["actions"]:
            if a["status"] == "NOT_FIRED":
                flags.append({"flag": "ACTION_NOT_FIRED", "ability": idx,
                              "detail": f"{a['type']} 「{a['raw_text']}」"})
            elif a["status"] in ("FIRED_EMPTY", "DEFERRED"):
                flags.append({"flag": a["status"], "ability": idx,
                              "detail": f"{a['type']} 「{a['raw_text']}」"})
    return flags, per_ability


# ---------------------------------------------------------------------------
# 台帳
# ---------------------------------------------------------------------------

def load_ledger() -> dict:
    if not os.path.exists(LEDGER):
        return {"version": LEDGER_VERSION, "cards": {}}
    with open(LEDGER, encoding="utf-8") as f:
        return json.load(f)


def save_ledger(ledger: dict) -> None:
    ledger["cards"] = dict(sorted(ledger["cards"].items()))
    with open(LEDGER, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def text_fingerprint(card: dict) -> str:
    return sha({"name": card.get("name"), "type": card.get("type"),
                "effect_text": nfkc(card.get("effect_text")), "trigger_text": nfkc(card.get("trigger_text"))})


def ledger_state(entry: Optional[dict], fp: dict) -> str:
    """台帳と現在の指紋から状態を決める。"""
    if not entry:
        return "UNREVIEWED"
    if entry.get("text_sha") != fp["text_sha"]:
        return "TEXT_CHANGED"
    if fp.get("ast_sha") and entry.get("ast_sha") != fp["ast_sha"]:
        return "IMPL_CHANGED"
    if fp.get("probe_sha") and entry.get("probe_sha") != fp["probe_sha"]:
        return "IMPL_CHANGED"
    return "VERIFIED_OK" if entry.get("verdict") == "ok" else "VERIFIED_NG"


# ---------------------------------------------------------------------------
# 効果 JSON の生成と Rust への読み込み
# ---------------------------------------------------------------------------

def build_effects(out_dir: str, card_db: str) -> str:
    path = os.path.join(out_dir, "effects.json")
    env = dict(os.environ, OPCG_LOG_SILENT="1")
    subprocess.run([sys.executable, "-m", "opcg_sim.tools.export_effects_json",
                    "--card-db", card_db, "--out", path],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, env=env)
    return path


def load_engine(effects_path: str, cards: Dict[str, dict]):
    """Rust へ読み込む。失敗したら (None, 原因と該当カード) を返す。"""
    try:
        import opcg_engine
    except ImportError as e:
        raise SystemExit(f"Rust 拡張 opcg_engine が無い。`make rust-develop` で入れること（{e}）")
    try:
        opcg_engine.load_masters(effects_path)
        return opcg_engine, None
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        names = re.findall(r"unknown \w+ '([^']+)'", msg)
        culprits = []
        for cid, c in cards.items():
            blob = json.dumps(c.get("abilities") or [], ensure_ascii=False)
            if any(f'"{n}"' in blob for n in names):
                culprits.append(cid)
        return None, {"error": msg, "unknown_names": names, "cards": culprits}


# ---------------------------------------------------------------------------
# 対象の選び方・前回 DB との差分
# ---------------------------------------------------------------------------

def base_db_diff(cards: Dict[str, dict], base_ref: str) -> dict:
    """git の `base_ref` にある DB と比べた追加・本文変更・削除。"""
    try:
        raw = subprocess.run(["git", "show", f"{base_ref}:{CARD_DB_REL}"], cwd=ROOT, check=True,
                             capture_output=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"base": base_ref, "error": "base の DB を読めない"}
    old = {}
    for row in json.loads(raw):
        cid = row.get("number")
        if cid:
            old[cid] = sha({"t": nfkc(row.get("効果(テキスト)")), "g": nfkc(row.get("効果(トリガー)")),
                            "n": row.get("name")})
    now = {}
    with open(CARD_DB, encoding="utf-8") as f:
        for row in json.load(f):
            cid = row.get("number")
            if cid:
                now[cid] = sha({"t": nfkc(row.get("効果(テキスト)")), "g": nfkc(row.get("効果(トリガー)")),
                                "n": row.get("name")})
    return {"base": base_ref,
            "added": sorted(set(now) - set(old)),
            "removed": sorted(set(old) - set(now)),
            "changed": sorted(c for c in set(now) & set(old) if now[c] != old[c])}


# ---------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------

def _load_cards(out_dir: str, card_db: str) -> Tuple[str, Dict[str, dict]]:
    os.makedirs(out_dir, exist_ok=True)
    path = build_effects(out_dir, card_db)
    with open(path, encoding="utf-8") as f:
        return path, json.load(f)["cards"]


def cmd_scan(args) -> int:
    effects_path, cards = _load_cards(args.out, args.card_db)
    ledger = load_ledger()
    diff = base_db_diff(cards, args.base)

    if args.cards:
        wanted = [c.strip() for c in args.cards.split(",") if c.strip()]
        missing = [c for c in wanted if c not in cards]
        if missing:
            print(f"[scan] DB に無いカード: {missing}", file=sys.stderr)
        targets = [c for c in wanted if c in cards]
    else:
        targets = sorted(cards)
        if args.set:
            targets = [c for c in targets if c.startswith(args.set)]

    engine, load_fail = load_engine(effects_path, cards)
    pstats = parse_stats({c: cards[c] for c in targets})
    oracle = _oracle_findings(args.card_db, set(targets))

    report: Dict[str, Any] = {}
    for cid in targets:
        card = cards[cid]
        flags, trigger_phrases = static_checks(card, pstats.get(cid, {}))
        for f in oracle.get(cid, []):
            flags.append({"flag": "ORACLE", "detail": f"{f['category']}: {f.get('detail', '')}"})
        per_ability: List[dict] = []
        if load_fail is not None:
            if cid in load_fail["cards"]:
                flags.append({"flag": "LOAD_FAIL", "detail": load_fail["error"][:300]})
        elif card.get("abilities"):
            dflags, per_ability = dynamic_checks(engine, card)
            flags.extend(dflags)
        fp = {"text_sha": text_fingerprint(card),
              "ast_sha": sha(card.get("abilities") or []),
              "probe_sha": sha([[r["runs"] for r in per_ability]]) if per_ability else None}
        state = ledger_state(ledger["cards"].get(cid), fp)
        report[cid] = {"card_id": cid, "name": card.get("name"), "type": card.get("type"),
                       "text": card_text(card), "keywords": card.get("keywords"),
                       "ledger": state, "fingerprint": fp, "flags": flags,
                       "trigger_phrases": trigger_phrases, "abilities": per_ability,
                       "has_effect": bool(card_text(card).strip())}

    if not args.cards and not args.all:
        # 既定（未確認／変化したカードだけ）。確認済みで指紋も同じものは外す。
        report = {c: r for c, r in report.items() if r["ledger"] != "VERIFIED_OK"}

    out_json = os.path.join(args.out, "report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "card_db": args.card_db, "effects": effects_path, "db_diff": diff,
                   "load_fail": load_fail, "cards": report}, f, ensure_ascii=False, indent=1)
    summary = _summary_md(report, diff, load_fail)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary if not args.quiet else summary.split("\n## ")[0])
    print(f"\n[scan] report: {out_json}")
    return 1 if load_fail else 0


def _oracle_findings(card_db: str, targets: set) -> Dict[str, List[dict]]:
    from effect_oracle import detect
    from opcg_sim.src.utils.loader import CardLoader
    db = CardLoader(card_db)
    db.load()
    out: Dict[str, List[dict]] = {}
    for f in detect(db):
        if f["card_id"] in targets and f["category"] in ("HAS_OTHER", "PER_TURN_LIMIT_GAP", "UP_TO_GAP"):
            out.setdefault(f["card_id"], []).append(f)
    return out


def _summary_md(report: Dict[str, dict], diff: dict, load_fail: Optional[dict]) -> str:
    lines = ["# カード効果監査サマリ", ""]
    if diff.get("error"):
        lines.append(f"- DB 差分（{diff['base']} 比）: {diff['error']}")
    else:
        lines.append(f"- DB 差分（{diff['base']} 比）: 追加 {len(diff['added'])}・本文変更 "
                     f"{len(diff['changed'])}・削除 {len(diff['removed'])}")
        for k, label in (("added", "追加"), ("changed", "本文変更"), ("removed", "削除")):
            if diff[k]:
                lines.append(f"  - {label}: {', '.join(diff[k][:60])}{' …' if len(diff[k]) > 60 else ''}")
    if load_fail:
        lines += ["", "## ⚠ Rust が効果 JSON を読めない（エンジン未実装の効果種別）", "",
                  f"- 未知の名前: {load_fail['unknown_names']}",
                  f"- 該当カード: {', '.join(load_fail['cards'])}",
                  f"- エラー: `{load_fail['error'][:300]}`",
                  "- この状態では実行検査（プローブ）は回らない。先にエンジン側へ実装すること。"]
    states = Counter(r["ledger"] for r in report.values())
    lines += ["", f"- 対象 {len(report)} 枚（効果あり {sum(r['has_effect'] for r in report.values())}）・"
                  f"台帳: " + "・".join(f"{k} {v}" for k, v in sorted(states.items()))]
    by_flag: Dict[str, List[str]] = {}
    for cid, r in report.items():
        for f in {f["flag"] for f in r["flags"]}:
            by_flag.setdefault(f, []).append(cid)
    lines += ["", "## 指摘の集計（E=欠陥が確実／W=要確認／I=参考）", "",
              "| 重さ | 指摘 | 枚数 | 例 |", "|---|---|---|---|"]
    order = sorted(by_flag, key=lambda k: ("EWI".index(FLAG_LEVEL.get(k, "I")), -len(by_flag[k])))
    for k in order:
        ex = ", ".join(sorted(by_flag[k])[:8])
        lines.append(f"| {FLAG_LEVEL.get(k, '?')} | {k} | {len(by_flag[k])} | {ex}{' …' if len(by_flag[k]) > 8 else ''} |")
    clean = [c for c, r in report.items() if r["has_effect"]
             and not any(FLAG_LEVEL.get(f["flag"]) in ("E", "W") for f in r["flags"])]
    lines += ["", f"- E/W の指摘なし（意味のレビューだけでよい）: {len(clean)} 枚",
              f"- 効果なし（バニラ）: {sum(not r['has_effect'] for r in report.values())} 枚", ""]
    return "\n".join(lines)


def _target_summary(t: Optional[dict]) -> str:
    if not t:
        return ""
    parts = [f"{t.get('player')}/{t.get('zone')}"]
    for k in ("card_type", "traits", "names", "exclude_names", "colors", "attributes", "flags"):
        if t.get(k):
            parts.append(f"{k}={t[k]}")
    for k in ("cost_min", "cost_max", "power_min", "power_max", "is_rest", "cost_max_dynamic"):
        if t.get(k) is not None:
            parts.append(f"{k}={t[k]}")
    parts.append(f"count={t.get('count')}{'まで' if t.get('is_up_to') else ''}")
    if t.get("select_mode") and t["select_mode"] != "CHOOSE":
        parts.append(f"mode={t['select_mode']}")
    return " ".join(parts)


def _value_summary(v: Optional[dict]) -> str:
    if not v:
        return ""
    s = str(v.get("base"))
    if v.get("dynamic_source"):
        s += f" dyn={v['dynamic_source']}×{v.get('multiplier')}/{v.get('divisor')}"
    return s


def _cond_summary(c: Optional[dict]) -> str:
    if not c:
        return ""
    if c.get("type") in ("AND", "OR", "NOT"):
        return f"{c['type']}(" + ", ".join(_cond_summary(a) for a in c.get("args") or []) + ")"
    return f"{c.get('type')}[{c.get('player')} {c.get('operator')} {c.get('value')}]「{c.get('raw_text') or ''}」"


def cmd_show(args) -> int:
    path = os.path.join(args.out, "report.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} が無い。先に scan を回すこと")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    with open(rep["effects"], encoding="utf-8") as f:
        cards = json.load(f)["cards"]
    for cid in args.card_ids:
        r = rep["cards"].get(cid)
        c = cards.get(cid)
        if not r or not c:
            print(f"### {cid}: report に無い（scan の対象外）\n")
            continue
        print(f"### {cid} {c['name']}（{c['type']}）台帳: {r['ledger']}")
        print(f"本文: {r['text']}")
        if c.get("keywords"):
            print(f"keywords: {c['keywords']}")
        for idx, ab in enumerate(c.get("abilities") or []):
            print(f"\n[能力{idx}] trigger={ab['trigger']} cost_optional={ab.get('cost_optional')}")
            print(f"  raw: {ab.get('raw_text')}")
            if ab.get("condition"):
                print(f"  条件: {_cond_summary(ab['condition'])}")
            probe = r["abilities"][idx] if idx < len(r["abilities"]) else {"actions": []}
            st = {a["path"]: a["status"] for a in probe.get("actions", [])}
            for a in flatten_actions(ab):
                n = a["node"]
                extra = []
                if n.get("destination"):
                    extra.append(f"→{n['destination']}{'/' + n['dest_position'] if n.get('dest_position') else ''}")
                if n.get("duration") and n["duration"] != "INSTANT":
                    extra.append(f"期間={n['duration']}")
                if n.get("is_optional"):
                    extra.append("任意")
                if n.get("status"):
                    extra.append(f"status={n['status']}")
                print(f"  - {a['path']:<16} {a['type']:<18} [{st.get(a['path'], '-')}] "
                      f"値={_value_summary(n.get('value'))} {' '.join(extra)}")
                print(f"      「{a['raw_text']}」")
                if a.get("target"):
                    print(f"      対象: {_target_summary(a['target'])}")
            for n in walk_nodes(ab.get("effect")):
                if n.get("node") == "Branch" and n.get("condition"):
                    print(f"  分岐条件: {_cond_summary(n['condition'])}")
            if args.events:
                for run in probe.get("runs", []):
                    print(f"  プローブ force={run['force']} choices={run['choices']} "
                          f"対話={run['interactions']}")
                    for e in run["events"]:
                        print(f"      {json.dumps(e, ensure_ascii=False)}")
        if r["trigger_phrases"]:
            print(f"\n誘発条件の句（トリガー種別で表現・raw_text 外）: {r['trigger_phrases']}")
        if r["flags"]:
            print("\n指摘:")
            for f in r["flags"]:
                ab = f" 能力{f['ability']}" if "ability" in f else ""
                print(f"  {FLAG_LEVEL.get(f['flag'], '?')} {f['flag']}{ab}: {f['detail']}")
        print()
    return 0


def cmd_list(args) -> int:
    """report の対象カードを条件で列挙する（レビューの分担用）。"""
    with open(os.path.join(args.out, "report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    rows = []
    for cid, r in rep["cards"].items():
        if not r["has_effect"] and not args.include_vanilla:
            continue
        levels = {FLAG_LEVEL.get(f["flag"]) for f in r["flags"]}
        if args.level and not (set(args.level) & levels):
            continue
        if args.flag and not any(f["flag"] == args.flag for f in r["flags"]):
            continue
        if args.state and r["ledger"] not in args.state.split(","):
            continue
        rows.append(cid)
    print(" ".join(rows) if args.oneline else "\n".join(rows))
    print(f"[list] {len(rows)} 枚", file=sys.stderr)
    return 0


def cmd_record(args) -> int:
    with open(os.path.join(args.out, "report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    ledger = load_ledger()
    today = datetime.date.today().isoformat()
    rows: List[dict] = []
    if args.from_file:
        # サブエージェントの判定（1 行 1 枚の JSONL: {"card_id","verdict","note"}）を一括で記録する。
        with open(args.from_file, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        if not args.verdict:
            raise SystemExit("--verdict（ok／ng）か --from を指定すること")
        ids = [c for arg in args.card_ids for c in arg.split(",") if c]
        rows = [{"card_id": c, "verdict": args.verdict, "note": args.note} for c in ids]
    errors = []
    for row in rows:
        cid, verdict = row.get("card_id"), row.get("verdict")
        r = rep["cards"].get(cid)
        if r is None:
            errors.append(f"{cid}: report に無い（scan の対象に入れてから記録すること）")
            continue
        if verdict not in ("ok", "ng"):
            errors.append(f"{cid}: verdict は ok／ng（{verdict!r}）")
            continue
        if verdict == "ok" and any(FLAG_LEVEL.get(f["flag"]) == "E" for f in r["flags"]) \
                and not (args.force and row.get("note")):
            errors.append(f"{cid}: E の指摘がある。誤検出と確かめたなら --force と note で理由を残す")
            continue
        entry = {"verdict": verdict, "reviewed": today, **r["fingerprint"]}
        if row.get("note"):
            entry["note"] = row["note"]
        ledger["cards"][cid] = entry
    ledger["version"] = LEDGER_VERSION
    save_ledger(ledger)
    print(f"[record] {len(rows) - len(errors)} 枚を記録 → {os.path.relpath(LEDGER, ROOT)}")
    for e in errors:
        print(f"[record] 記録しなかった: {e}", file=sys.stderr)
    return 1 if errors else 0


def cmd_status(args) -> int:
    ledger = load_ledger()
    path = os.path.join(args.out, "report.json")
    verdicts = Counter(e.get("verdict") for e in ledger["cards"].values())
    print(f"台帳: {len(ledger['cards'])} 枚（" + "・".join(f"{k} {v}" for k, v in sorted(verdicts.items())) + "）")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
        states = Counter(r["ledger"] for r in rep["cards"].values() if r["has_effect"])
        print(f"直近の scan（{rep['generated']}・対象 {len(rep['cards'])} 枚）の効果ありカード: "
              + "・".join(f"{k} {v}" for k, v in sorted(states.items())))
    ng = {c: e.get("note", "") for c, e in ledger["cards"].items() if e.get("verdict") == "ng"}
    if ng:
        print(f"\n未解決（ng）{len(ng)} 枚:")
        for c, n in ng.items():
            print(f"  {c}: {n}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="カード効果の監査（skill card-effect-audit の機械部分）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"作業ディレクトリ（既定 {DEFAULT_OUT}）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="検査を回して report.json／summary.md を作る")
    s.add_argument("--all", action="store_true", help="台帳で確認済みのカードも含めて全走査")
    s.add_argument("--cards", help="カード ID をカンマ区切りで")
    s.add_argument("--set", help="品番の接頭辞（例 OP15・EB04）")
    s.add_argument("--base", default="HEAD", help="DB 差分の比較元（git ref・既定 HEAD）")
    s.add_argument("--card-db", default=CARD_DB)
    s.add_argument("--quiet", action="store_true", help="サマリの表だけ省く")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("show", help="カードのレビュー資料（本文・解析結果・実行状況）")
    s.add_argument("card_ids", nargs="+")
    s.add_argument("--events", action="store_true", help="プローブのイベント列も出す")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("list", help="report の対象カードを条件で列挙")
    s.add_argument("--level", help="E／W／I の組み合わせ（例 EW）")
    s.add_argument("--flag", help="指摘の種類で絞る（例 UNCOVERED_TEXT）")
    s.add_argument("--state", help="台帳の状態で絞る（例 UNREVIEWED,TEXT_CHANGED）")
    s.add_argument("--include-vanilla", action="store_true")
    s.add_argument("--oneline", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("record", help="レビュー結果を台帳へ記録（指紋は直近の scan のもの）")
    s.add_argument("card_ids", nargs="*", help="カード ID（スペースかカンマ区切り）")
    s.add_argument("--verdict", choices=("ok", "ng"))
    s.add_argument("--note", help="ng の理由・E 指摘を誤検出と判断した理由など")
    s.add_argument("--from", dest="from_file",
                   help='判定の JSONL（1 行 1 枚: {"card_id","verdict","note"}）から一括で記録')
    s.add_argument("--force", action="store_true", help="E の指摘があっても ok で記録する")
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("status", help="台帳の集計")
    s.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
