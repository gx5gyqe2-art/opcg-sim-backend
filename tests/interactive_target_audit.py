"""INTERACTIVE 対象の自動監査: 解釈済み TargetQuery をカードテキストと突き合わせ、
対象側(自分/相手)・コスト上限・枚数/まで・特徴 がテキストと食い違う候補を機械的に検出する。

実行: OPCG_LOG_SILENT=1 python tests/interactive_target_audit.py [--top N]
"""
import os, re, json, sys, unicodedata
os.environ.setdefault("OPCG_LOG_SILENT", "1")
sys.path.insert(0, os.path.dirname(__file__))
import conftest  # noqa: F401  (sys.path 設定 & google スタブ)
from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import GameAction, Sequence, Branch, Choice
from opcg_sim.src.models.enums import Player, Zone

def nf(s): return unicodedata.normalize("NFKC", s or "")

def walk(node):
    if node is None: return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions: yield from walk(a)
    elif isinstance(node, Branch):
        yield from walk(node.if_true); yield from walk(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options: yield from walk(o)

def _strip_modifiers(r):
    """matcher.parse_target と同じ従属節（対象側判定を汚す句）を除去する。

    H-7: 監査は raw_text の素の文字列を見るため、duration/chooser/コスト基準の
    「相手」を対象指定と誤認していた。matcher と同じ規則で除去して誤検知を消す。
    """
    # コスト基準の「(相手の/自分の/お互いの)ライフの枚数以下のコストを持つ」
    r = re.sub(r'(?:お互いの|相手の|自分の)?ライフの(?:合計)?枚数(?:分)?以下のコストを持つ', '', r)
    # 期間/タイミング句「(次の)相手の(ターン/エンドフェイズ)(終了時)(まで/中)」
    r = re.sub(r'(?:次の)?相手の(?:ターン|エンドフェイズ)(?:終了時)?(?:まで|中)', '', r)
    # 選択者句「相手が選(び/ぶ/んで)」・トリガー句「相手が…した時、」
    r = re.sub(r'相手が選(?:び|ぶ|んで)', '', r)
    r = re.sub(r'相手が[^、。]*した時、?', '', r)
    return r


def audit_target(raw, tq):
    """raw_text(原子句) と TargetQuery を突き合わせ、不一致の説明リストを返す。"""
    issues = []
    r0 = nf(raw)
    if not r0 or tq is None: return issues
    r = _strip_modifiers(r0)
    # 対象側: 「相手の(キャラ/リーダー)」なのに player=SELF など
    has_aite = bool(re.search(r"相手の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    has_jibun = bool(re.search(r"自分の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    side = getattr(tq.player, "name", None)
    # chooser（相手が選ぶ）が設定されている対象は選択者≠対象側なので側チェックを省く
    has_chooser = getattr(tq, "chooser", None) is not None
    # SOURCE 対象（このキャラ自身）の句では「相手の」は参照（同値パワー）・許可
    #（相手のキャラにもアタックできる）・保護（相手の効果でKOされない）であって対象側ではない。
    is_source = getattr(tq, "select_mode", None) == "SOURCE"
    # 「自分か相手の…」「お互いの…」の両側併記（scry の Choice 等）は側を一意に
    # 決められないので除外する（各 Choice 肢は片側を正しく指すが raw_text は全文）。
    both_sides = (has_aite and has_jibun) or bool(
        re.search(r"自分か相手|相手か自分|お互い", r))
    skip_side = has_chooser or is_source or both_sides
    if has_aite and not has_jibun and side == "SELF" and not skip_side:
        issues.append(f"側:相手指定だが SELF")
    if has_jibun and not has_aite and side == "OPPONENT" and not skip_side:
        issues.append(f"側:自分指定だが OPPONENT")
    # コスト上限
    m = re.search(r"コスト(\d+)以下", r)
    if m:
        want = int(m.group(1))
        if tq.cost_max != want and getattr(tq, "cost_max_dynamic", None) is None:
            issues.append(f"コスト上限:text {want} / tq {tq.cost_max}")
    # 枚数 / まで（「N枚につき」はスケーリング係数であって対象枚数ではないため除外）
    r_cnt = re.sub(r"[\d０-９]+枚につき", "", r)
    mc = re.search(r"(\d+)枚(まで)?", r_cnt)
    if mc:
        want = int(mc.group(1)); upto = bool(mc.group(2))
        if tq.count not in (want, -1):
            issues.append(f"枚数:text {want} / tq {tq.count}")
        # 「N枚まで」だが ALL（全体）指定の場合は is_up_to 不問
        if upto and not tq.is_up_to and tq.select_mode != "ALL":
            issues.append(f"まで:text有 / tq is_up_to False")
    # 特徴《X》（複数特徴の OR「《A》か《B》」はどちらか保持で可）
    raw_traits = [t for t in re.findall(r"《([^》]+)》", r)
                  if t not in ("斬", "打", "特", "知", "活")]
    if raw_traits and not any(t in tq.traits for t in raw_traits):
        issues.append(f"特徴:text《{'/'.join(raw_traits)}》 / tq {list(tq.traits)}")
    return issues

def run(top=60):
    data = json.load(open(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "src", "..", "data", "opcg_cards.json"))) \
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_cards.json")) else None
    data = json.load(open("opcg_sim/data/opcg_cards.json"))
    parser = EffectParserV2()
    flagged = []
    K = "効果(テキスト)"
    for c in data:
        t = c.get(K, "") or ""
        if not t.strip(): continue
        try:
            abils = parser.parse_card_text(t)
        except Exception:
            continue
        card_issues = []
        for ab in abils:
            for node in [ab.cost, ab.effect]:
                for ga in walk(node):
                    if ga.target is None: continue
                    iss = audit_target(ga.raw_text, ga.target)
                    if iss:
                        card_issues.append((ga.raw_text[:48], iss))
        if card_issues:
            flagged.append((c.get("number"), card_issues))
    print(f"=== INTERACTIVE 対象の自動監査: 疑い {len(flagged)} 枚 ===")
    # 不一致タイプ別集計
    from collections import Counter
    typ = Counter()
    for _, ci in flagged:
        for _, iss in ci:
            for x in iss: typ[x.split(":")[0]] += 1
    print("--- 不一致タイプ別 ---")
    for k, v in typ.most_common(): print(f"  {v:4d}  {k}")
    print(f"--- 上位 {top} 枚 ---")
    for num, ci in flagged[:top]:
        for raw, iss in ci[:2]:
            print(f"  {num}  [{' / '.join(iss)}]  «{raw}»")
    return flagged

if __name__ == "__main__":
    top = 60
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top")+1])
    run(top)
