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

def audit_target(raw, tq):
    """raw_text(原子句) と TargetQuery を突き合わせ、不一致の説明リストを返す。"""
    issues = []
    r = nf(raw)
    if not r or tq is None: return issues
    # 対象側: 「相手の(キャラ/リーダー)」なのに player=SELF など
    has_aite = bool(re.search(r"相手の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    has_jibun = bool(re.search(r"自分の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    side = getattr(tq.player, "name", None)
    if has_aite and not has_jibun and side == "SELF":
        issues.append(f"側:相手指定だが SELF")
    if has_jibun and not has_aite and side == "OPPONENT":
        issues.append(f"側:自分指定だが OPPONENT")
    # コスト上限
    m = re.search(r"コスト(\d+)以下", r)
    if m:
        want = int(m.group(1))
        if tq.cost_max != want and getattr(tq, "cost_max_dynamic", None) is None:
            issues.append(f"コスト上限:text {want} / tq {tq.cost_max}")
    # 枚数 / まで
    mc = re.search(r"(\d+)枚(まで)?", r)
    if mc:
        want = int(mc.group(1)); upto = bool(mc.group(2))
        if tq.count not in (want, -1):
            issues.append(f"枚数:text {want} / tq {tq.count}")
        if upto and not tq.is_up_to:
            issues.append(f"まで:text有 / tq is_up_to False")
    # 特徴《X》
    for trait in re.findall(r"《([^》]+)》", r):
        # 属性(斬/打/特/知/活)は除外
        if trait in ("斬", "打", "特", "知", "活"): continue
        if trait not in tq.traits:
            issues.append(f"特徴:text《{trait}》 / tq {list(tq.traits)}")
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
