"""結果ポストからの順位・リーダー抽出（LLM不使用・辞書マッチング、設計 §13）。

137 リーダー辞書（カードDB `種類=リーダー`）から OPCG コミュニティの通称に沿った
エイリアスを機械生成し、正規化済みテキストへ最長一致で当てる。純粋関数（DB 書き込みなし）で、
結果登録フォームへの候補サジェスト専用。確定は既存 `PUT /events/{id}/results`。

方針・限界は docs/design.md §13 を参照。画像のみ・独自表記は取れない → 手入力が受け皿。
"""
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

from ..resources import card_db

_LEADER_TYPE = unicodedata.normalize("NFC", "リーダー")

# エイリアスの種別ごとの信頼度（色付き・具体的なほど高い）。
_CONF = {"color_full": 0.95, "color_short": 0.9, "full": 0.8, "short": 0.7}
# 曖昧（同名複数で card_number を一意化できない）ときの信頼度。
_CONF_AMBIGUOUS = 0.4
# 種別の具体度ランク（重なり解消・タイブレークに使う）。
_SPEC_RANK = {"color_full": 3, "color_short": 2, "full": 1, "short": 0}


def _norm(s: str) -> str:
    """比較用正規化: NFKC → 空白・中黒・一般的な装飾記号を除去。"""
    s = unicodedata.normalize("NFKC", s)
    # 中黒（・/･）と空白を除去。名前・エイリアス・本文すべてに同じ正規化を掛ける。
    s = re.sub(r"[\s・･]", "", s)
    return s


def _short_name(name: str) -> str:
    """`・` 区切りの末尾要素（コミュニティの通称）。区切りが無ければ全体。"""
    parts = re.split(r"[・･]", name)
    return parts[-1] if parts else name


def _color_prefixes(colors: List[str]) -> List[str]:
    """色略称のプレフィックス。2 色は順序両方（黒黄 / 黄黒）を返す。"""
    if len(colors) == 1:
        return [colors[0]]
    if len(colors) == 2:
        return [colors[0] + colors[1], colors[1] + colors[0]]
    return ["".join(colors)]


class _AliasIndex:
    """正規化エイリアス → (card_number 集合, 最も具体的な種別)。プロセス内で一度だけ構築。"""

    def __init__(self) -> None:
        # alias_norm -> {"numbers": set, "kind": str}
        self.aliases: Dict[str, Dict[str, object]] = {}
        self._build()

    def _add(self, alias: str, number: str, kind: str) -> None:
        a = _norm(alias)
        if not a:
            return
        entry = self.aliases.get(a)
        if entry is None:
            self.aliases[a] = {"numbers": {number}, "kind": kind}
            return
        numbers: Set[str] = entry["numbers"]  # type: ignore[assignment]
        numbers.add(number)
        # 同一エイリアス文字列に別種別が来たら、より具体的な種別を残す。
        if _SPEC_RANK[kind] > _SPEC_RANK[str(entry["kind"])]:
            entry["kind"] = kind

    def _build(self) -> None:
        for number, item in card_db.raw_db.items():
            norm = {unicodedata.normalize("NFC", str(k)): v for k, v in item.items()}
            if unicodedata.normalize("NFC", str(norm.get("種類", ""))) != _LEADER_TYPE:
                continue
            name = str(norm.get("name", ""))
            if not name:
                continue
            color = str(norm.get("色", ""))  # "赤" または "赤/緑"
            colors = [c for c in color.split("/") if c]
            short = _short_name(name)

            self._add(name, number, "full")
            self._add(short, number, "short")
            for cp in _color_prefixes(colors):
                self._add(cp + name, number, "color_full")
                self._add(cp + short, number, "color_short")


_INDEX: Optional[_AliasIndex] = None


def _index() -> _AliasIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = _AliasIndex()
    return _INDEX


def _match_all(segment: str) -> List[Tuple[int, str, Optional[str], float]]:
    """正規化済みセグメントから重ならないエイリアス一致を位置順で返す。

    各要素 = (開始位置, エイリアス, card_number|None, confidence)。
    長いエイリアスを優先して重なりを解消する（`赤ゾロ` があれば `ゾロ` は取らない）。
    同名複数で一意化できなければ card_number=None（曖昧）。
    """
    idx = _index()
    occ: List[Tuple[int, str, Set[str], str]] = []
    for alias, entry in idx.aliases.items():
        pos = segment.find(alias)
        if pos >= 0:
            occ.append((pos, alias, entry["numbers"], str(entry["kind"])))  # type: ignore[arg-type]
    # 長いエイリアス優先で重なりを解消。
    occ.sort(key=lambda x: -len(x[1]))
    covered = [False] * len(segment)
    chosen: List[Tuple[int, str, Optional[str], float]] = []
    for pos, alias, numbers, kind in occ:
        end = pos + len(alias)
        if any(covered[pos:end]):
            continue
        for i in range(pos, end):
            covered[i] = True
        if len(numbers) == 1:
            number: Optional[str] = next(iter(numbers))
            conf = _CONF[kind]
        else:
            number = None
            conf = _CONF_AMBIGUOUS
        chosen.append((pos, alias, number, conf))
    chosen.sort(key=lambda x: x[0])
    return chosen


# 順位マーカー: 準優勝を優勝より先に（"準優勝" は "優勝" を含むため）。
_MARKER_RE = re.compile(
    r"準優勝"
    r"|優勝"
    r"|第?\s*([0-9]{1,2})\s*位"
    r"|(?:ベスト|ベースト|BEST|TOP|トップ)\s*([0-9]{1,2})",
    re.IGNORECASE,
)


def _marker_placement(m: "re.Match[str]") -> Tuple[int, bool]:
    """マッチした順位マーカーを (基準placement, グループか) に写像。"""
    text = m.group(0)
    if text.startswith("準優勝"):
        return 2, False
    if text.startswith("優勝"):
        return 1, False
    if m.group(1):  # N位
        return int(m.group(1)), False
    if m.group(2):  # ベストN / TOPN（グループ: 見つかったリーダーを N から連番）
        return int(m.group(2)), True
    return 1, False


class ExtractedEntry:
    """抽出候補 1 件。フロントの登録フォーム 1 行に対応。"""

    def __init__(self, placement: int, card_number: Optional[str],
                 leader_raw: str, confidence: float) -> None:
        self.placement = placement
        self.card_number = card_number
        self.leader_raw = leader_raw
        self.confidence = confidence


def extract_results(text: str) -> Tuple[List[ExtractedEntry], List[str]]:
    """本文から順位×リーダーの候補を抽出する。

    戻り値 = (候補リスト, 未マッチのヒント断片リスト)。
    - 順位マーカー（優勝/準優勝/N位/ベストN）の近傍にリーダー辞書を当てる。
    - マーカーが無ければ本文全体から優勝(1)を1件だけ試す。
    - placement で重複したら confidence の高い方を残し、placement 昇順で返す。
    """
    if not text or not text.strip():
        return [], []

    norm = _norm(text)
    markers = list(_MARKER_RE.finditer(norm))

    by_placement: Dict[int, ExtractedEntry] = {}
    unmatched: List[str] = []

    def consider(entry: ExtractedEntry) -> None:
        prev = by_placement.get(entry.placement)
        if prev is None or entry.confidence > prev.confidence:
            by_placement[entry.placement] = entry

    if not markers:
        # マーカー無し: 本文全体から最有力を優勝候補に。
        hits = _match_all(norm)
        if hits:
            _pos, alias, number, conf = hits[0]
            consider(ExtractedEntry(1, number, alias, conf))
        return _finalize(by_placement, unmatched)

    for i, m in enumerate(markers):
        base, is_group = _marker_placement(m)
        seg_start = m.end()
        seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(norm)
        segment = norm[seg_start:seg_end]
        hits = _match_all(segment)
        if not hits:
            unmatched.append(f"placement={base} 付近にリーダー未検出")
            continue
        if is_group:
            # ベストN 等: 見つかったリーダーを base から連番で割り当て。
            for offset, (_pos, alias, number, conf) in enumerate(hits):
                consider(ExtractedEntry(base + offset, number, alias, conf * 0.9))
        else:
            _pos, alias, number, conf = hits[0]
            consider(ExtractedEntry(base, number, alias, conf))

    return _finalize(by_placement, unmatched)


def _finalize(by_placement: Dict[int, ExtractedEntry],
              unmatched: List[str]) -> Tuple[List[ExtractedEntry], List[str]]:
    entries = [by_placement[p] for p in sorted(by_placement)]
    return entries, unmatched
