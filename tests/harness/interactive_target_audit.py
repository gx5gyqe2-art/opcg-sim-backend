"""INTERACTIVE 対象の自動監査: 解釈済み TargetQuery をカードテキストと突き合わせ、
対象側(自分/相手)・コスト上限・枚数/まで・特徴 がテキストと食い違う候補を機械的に検出する。

実行: OPCG_LOG_SILENT=1 python tests/interactive_target_audit.py [--top N]
"""
import os, re, json, sys, unicodedata
os.environ.setdefault("OPCG_LOG_SILENT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
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


def _is_self_cost(tq):
    """このカード自身を対象にする句（SOURCE / ref=self の自己レスト等）。
    こうした句の raw_text はコスト節全文を共有し、特徴/枚数/コストは
    兄弟アクション側に乗るため、対象不一致の判定からは除外する。"""
    return (getattr(tq, "select_mode", None) == "SOURCE"
            or getattr(tq, "ref_id", None) == "self")


def audit_target(raw, tq):
    """単一 TargetQuery を監査する後方互換ラッパ（H-4 SELECT_MISMATCH 用）。"""
    if tq is None:
        return []
    return audit_group(raw, [tq])


def audit_group(raw, tqs):
    """raw_text を共有する TargetQuery 群（Choice/二択/二ティアの兄弟）を集約して監査する。

    制約（コスト上限/特徴/枚数）はいずれかの兄弟が満たせば不一致としない
    （各肢は raw_text 全文を持つが、制約は肢ごとに分かれて乗るため）。
    """
    issues = []
    r0 = nf(raw)
    if not r0 or not tqs: return issues
    r = _strip_modifiers(r0)
    # 自己コスト句（SOURCE/self）は raw を共有するだけなので対象判定から外す。
    eff = [tq for tq in tqs if not _is_self_cost(tq)]
    if not eff: return issues

    sides = {getattr(tq.player, "name", None) for tq in eff}
    has_aite = bool(re.search(r"相手の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    has_jibun = bool(re.search(r"自分の[^。]*?(キャラ|リーダー|ライフ|デッキ|手札|トラッシュ)", r))
    has_chooser = any(getattr(tq, "chooser", None) is not None for tq in eff)
    # 「相手は…自分の…できない」等、主題「相手は」がある句は対象側＝相手で正しい
    # （「自分の」はその文中の所有者参照であって対象側ではない: OP13-057）。
    aite_topic = bool(re.search(r"相手は", r))
    both_sides = (has_aite and has_jibun) or bool(
        re.search(r"自分か相手|相手か自分|お互い", r))
    skip_side = has_chooser or both_sides or aite_topic
    if not skip_side:
        if has_aite and not has_jibun and "OPPONENT" not in sides and sides <= {"SELF"}:
            issues.append("側:相手指定だが SELF")
        if has_jibun and not has_aite and "SELF" not in sides and sides <= {"OPPONENT"}:
            issues.append("側:自分指定だが OPPONENT")
    # コスト上限: text 中の各「コストN以下」が、いずれかの兄弟の cost_max(または動的)で
    # カバーされていれば可（二ティア「コスト6以下とコスト4以下」EB03-049 等）。
    cost_caps = {int(x) for x in re.findall(r"コスト(\d+)以下", r)}
    tq_caps = {tq.cost_max for tq in eff if tq.cost_max is not None}
    has_dyn = any(getattr(tq, "cost_max_dynamic", None) for tq in eff)
    if cost_caps and not has_dyn:
        missing = cost_caps - tq_caps
        if missing and not (tq_caps & cost_caps):
            issues.append(f"コスト上限:text {sorted(cost_caps)} / tq {sorted(tq_caps)}")
    # 枚数 /（「N枚につき」はスケーリング係数なので除外）
    r_cnt = re.sub(r"[\d０-９]+枚につき", "", r)
    wants = {int(x) for x in re.findall(r"(\d+)枚", r_cnt)}
    tq_counts = {tq.count for tq in eff}
    if wants and not (wants & tq_counts) and -1 not in tq_counts:
        issues.append(f"枚数:text {sorted(wants)} / tq {sorted(tq_counts)}")
    # 特徴《X》: text 中の各特徴がいずれかの兄弟の traits にあれば可。
    raw_traits = [t for t in re.findall(r"《([^》]+)》", r)
                  if t not in ("斬", "打", "特", "知", "活")]
    union_traits = set()
    for tq in eff: union_traits |= set(tq.traits)
    if raw_traits and not any(t in union_traits for t in raw_traits):
        issues.append(f"特徴:text《{'/'.join(raw_traits)}》 / tq {sorted(union_traits)}")
    return issues

def run(top=60):
    data = json.load(open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "opcg_sim", "src", "..", "data", "opcg_cards.json"))) \
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "opcg_sim", "data", "opcg_cards.json")) else None
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
        # raw_text を共有する対象群（Choice/二択/二ティアの兄弟）を集約して監査する。
        from collections import OrderedDict
        groups = OrderedDict()
        for ab in abils:
            for node in [ab.cost, ab.effect]:
                for ga in walk(node):
                    if ga.target is None: continue
                    groups.setdefault(ga.raw_text, []).append(ga.target)
        card_issues = []
        for raw, tqs in groups.items():
            iss = audit_group(raw, tqs)
            if iss:
                card_issues.append(((raw or "")[:48], iss))
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
