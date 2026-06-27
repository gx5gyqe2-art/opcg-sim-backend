"""効果価値プローブ（診断・dev専用／評価器には未統合）: 盤面の各キャラの起動メイン効果を
**実シミュレーション**で発火し、素材eval（evaluate_base）の増分Δを「効果価値」として数値化する。

方式1（決定ごと1回の実シミュレーション）の妥当性を、評価器へ統合する前に目視検証するための観測ツール。
自己対戦を回し、各意思決定で盤面の両陣営のキャラについて effect_impact を測ってログする。

  effect_impact(card, owner) =
     clone 上で資源を**現実的に**ステージ（自ターン文脈・現実的な次ターンのドン量）してから、card の各
     ACTIVATE_MAIN 能力を **条件/回数/コストを尊重して** resolve_ability で発火（条件付き効果は今の盤面で
     条件成立時のみ発火＝文脈的に正しい脅威価値）。対象は**実盤面の本物のみ**（注入なし＝対象が無ければ
     除去は 0＝今は脅威ゼロが正しい）。貪欲ドレイン後の evaluate_base(owner) の増分
     ＝ コスト差引後の純Δ（J単位＝evaluate_base と同単位で自動較正）。任意能力なので負Δは 0 にクランプ。

これで「毎ターン仕事するエンジン＝高Δ／登場時済みバニラ＝Δ≒0」が出るかを確認する。登場時(ON_PLAY)は
スコープ外（既に発動済みなので残価値は本体のみ）＝起動メインのみ測る。power/cost も併記して相関を見る。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/effect_value_probe.py --games 8 --real-decks --all-leaders
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.invariants import check_invariants
from opcg_sim.src.core.effects.resolver import EffectResolver
from opcg_sim.src.models.enums import TriggerType
from collect_value_data import _build_decks, _make_decider
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS

_DRAIN_LIMIT = 40
_NEXT_TURN_DON_GAIN = 2  # 次ターンに供給されるドン枚数（現実的なステージング上限の基準）
_DB = None
_CFG: Dict[str, Any] = {}


def _init_worker(cfg):
    global _DB, _CFG
    _DB = _load_db()
    _CFG = cfg


def _drain_pending(board, owner_name: str):
    """resolve_ability の中断（対象選択等）を **効果が空振りしないよう貪欲に**解消する（owner 側のみ・上限付き）。

    既定payloadは多くが0枚選択（デクライン）でKO/パンプ系がコストだけ払って空振りするため、`_selection_moves`
    の候補から「選択枚数が最大」の手（＝任意確認は accept・上限N選択はN枚）を選んで適用＝効果のポテンシャルを測る。
    """
    actor = board.p1 if board.p1.name == owner_name else board.p2
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")
    for _ in range(_DRAIN_LIMIT):
        if board.winner is not None:
            break
        pending = board.get_pending_request()
        if not pending or pending.get(pid_key) != owner_name:
            break
        board.action_events = []
        try:
            sel = cpu_ai._selection_moves(board, owner_name)
            if sel:
                # accept/最大枚数選択を優先（任意確認は accepted=True、上限選択は最多 uuid）。
                def _rank(mv):
                    pl = mv.get("payload", {})
                    return (1 if pl.get("accepted") else 0, len(pl.get("selected_uuids", []) or []))
                mv = max(sel, key=_rank)
                action_api.apply_game_action(board, actor, mv["action_type"], mv.get("payload", {}))
            else:
                payload = board.default_interaction_payload(pending)
                action_api.apply_game_action(board, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            break


def _stage_owner_turn(clone, owner, card):
    """オーナーが自分の次ターンに発火できる**現実的な**文脈へ整える（方式1のポテンシャル測定）。

    ターン文脈をオーナーに・リーダー/対象カードを起こす・使用回数リセット・ドンを「現実的な次ターン量」へ
    整える（場のドン active+rested を起こす＋次ターン供給ぶん +2、上限10）。**対象は実盤面の本物のみ**＝
    相手場に対象が無ければ除去は空振り（＝今は脅威ゼロ）として 0 に出るのが文脈的に正しい（注入はしない）。
    コストは実際に払われる（Δに自己較正で反映）。staged ドンは before/after で相殺。
    """
    clone.turn_player = owner
    if owner.leader is not None:
        owner.leader.is_rest = False
        try:
            owner.leader.ability_used_this_turn.clear()
        except Exception:
            pass
    # 現実的な次ターンのドン: 場のドン（active+rested）を起こし、次ターン供給ぶん +2 を予備から足す（上限10）。
    budget = min(10, len(owner.don_active) + len(owner.don_rested) + _NEXT_TURN_DON_GAIN)
    while owner.don_rested and len(owner.don_active) < budget:
        owner.don_active.append(owner.don_rested.pop())
    while owner.don_deck and len(owner.don_active) < budget:
        owner.don_active.append(owner.don_deck.pop())
    card.is_rest = False
    try:
        card.ability_used_this_turn.clear()
    except Exception:
        pass


def _action_types(node, out):
    """効果ツリー中の GameAction.type 名を収集（Sequence/Branch/Choice/sub_effect を再帰）。"""
    if node is None:
        return
    t = getattr(node, "type", None)
    if t is not None and hasattr(t, "name"):
        out.append(t.name)
    for attr in ("actions", "branches", "options", "sub_effect", "effect", "then", "otherwise"):
        v = getattr(node, attr, None)
        if isinstance(v, (list, tuple)):
            for x in v:
                _action_types(x, out)
        elif v is not None:
            _action_types(v, out)


def effect_impact(manager, card_uuid: str, owner_name: str):
    """card の起動メイン効果を実シミュレーションして (純Δ, 主ActionType) を返す（owner視点）。

    起動メイン能力が無ければ None（バニラ/登場時のみ＝測定対象外）。複数あれば最大Δの能力を採る。
    オーナー視点で資源をステージングしてから発火＝ポテンシャル価値（コスト込み・自己較正）を測る。
    """
    # 元カードの能力一覧（master は共有・immutable）。起動メインだけ対象。
    src = manager._find_card_by_uuid(card_uuid)
    if src is None:
        return None
    ams = [ab for ab in src.master.abilities if ab.trigger == TriggerType.ACTIVATE_MAIN]
    if not ams:
        return None
    best = None
    best_acts = ""
    for ab in ams:
        clone = manager.clone()
        owner = clone.p1 if clone.p1.name == owner_name else clone.p2
        cc = clone._find_card_by_uuid(card_uuid)
        if cc is None:
            continue
        try:
            _stage_owner_turn(clone, owner, cc)
            before = cpu_ai.evaluate_base(clone, owner_name, see_opp_hand=False)
            clone.action_events = []
            # 条件/回数/コストを尊重して発火（resolve_ability）。条件付き効果（「自分のライフ≤2なら」等）は
            # **今の盤面で条件が成立するときだけ**発火＝文脈的に正しい脅威価値を測る（高ライフ相手の条件付き
            # 除去は今は脅威ゼロ＝Δ0 が正解）。資源（自ターン・ドン）はステージ済みなのでコストは実払いされる。
            EffectResolver(clone).resolve_ability(owner, ab, cc)
            _drain_pending(clone, owner_name)
            after = cpu_ai.evaluate_base(clone, owner_name, see_opp_hand=False)
        except Exception:
            continue
        d = after - before
        if best is None or d > best:
            acts = []
            _action_types(ab.effect, acts)
            best = d
            best_acts = "+".join(dict.fromkeys(acts))  # 重複除去・順序維持
    if best is None:
        return None
    return (best, best_acts)


def probe_game(seed: int):
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(_CFG["difficulty"], 40, 2, _CFG["pimc"])
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    rows = []  # (card_id, name, power, cost, impact)
    step = 0
    probes = 0
    while m.winner is None and step < _CFG["max_steps"]:
        pending = m.get_pending_request()
        if not pending:
            break
        # 意思決定ノードでのみ（対話中はスキップ）プローブ。1局あたりの計測回数は budget で制限。
        if pending.get("action") == "MAIN_ACTION" and probes < _CFG["max_probes_per_game"]:
            for pl in (m.p1, m.p2):
                for c in list(pl.field) + ([pl.leader] if pl.leader else []):
                    res = effect_impact(m, c.uuid, pl.name)
                    if res is not None:
                        imp, acts = res
                        mm = c.master
                        is_leader = "LEADER" in str(getattr(mm, "type", ""))
                        rows.append((mm.card_id, mm.name, mm.power or 0,
                                     mm.cost or 0, round(imp, 1), is_leader, acts))
            probes += 1
        chosen = decide(m, m.p1 if m.p1.name == pending[pid_key] else m.p2)
        if chosen is None:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        m.action_events = []
        try:
            if chosen["kind"] == "battle":
                action_api.apply_battle_action(m, actor, chosen["action_type"], chosen.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, chosen["action_type"], chosen.get("payload", {}))
        except Exception:
            return rows
        if check_invariants(m):
            return rows
        step += 1
    return rows


def _one(seed):
    try:
        return probe_game(seed)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="効果価値プローブ（観測専用）")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--max-probes-per-game", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)

    cfg = {"difficulty": args.difficulty, "max_steps": args.max_steps, "real_decks": args.real_decks,
           "all_leaders": args.all_leaders, "pimc": args.pimc,
           "max_probes_per_game": args.max_probes_per_game}
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    seeds = [args.seed + g for g in range(args.games)]

    all_rows = []
    n_games = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, seeds), 1):
            if r is None:
                continue
            n_games += 1
            all_rows.extend(r)
            if i % 4 == 0:
                print(f"  {i}/{args.games} … probes={len(all_rows)}", flush=True)

    print(f"\n=== 効果価値プローブ: {n_games}局・計測 {len(all_rows)} 件 ===")
    if not all_rows:
        print("計測なし（起動メイン持ちが場に出なかった）")
        return 0
    chars = [r for r in all_rows if not r[5]]
    leaders = [r for r in all_rows if r[5]]
    print(f"内訳: キャラ {len(chars)} 件 / リーダー {len(leaders)} 件")

    def _dist(rows, label):
        if not rows:
            print(f"[{label}] なし")
            return
        imps = sorted(x[4] for x in rows)
        n = len(imps)
        pct = lambda p: imps[min(n - 1, int(p * n))]
        # 効果価値は max(0,Δ)＝任意能力（撃たなければ損しない）。生Δの負は「この文脈では非発火」。
        clamped = [max(0.0, v) for v in imps]
        zero = sum(1 for v in imps if abs(v) < 1.0)
        neg = sum(1 for v in imps if v < -1.0)
        pos = sum(1 for v in imps if v > 1.0)
        print(f"[{label}] 生Δ: min={imps[0]:.0f} 中央={pct(0.5):.0f} p75={pct(0.75):.0f} "
              f"p90={pct(0.9):.0f} max={imps[-1]:.0f} ／ 正Δ {pos}/{n} ({pos/n:.0%}) "
              f"・Δ0 {zero/n:.0%} ・負Δ {neg/n:.0%} ／ クランプ平均={sum(clamped)/n:.0f}")

    _dist(chars, "キャラ")
    _dist(leaders, "リーダー")

    # カード別の平均Δ（同一カードを集約・クランプ後）＝「どのカードが高効果価値か」を目視（キャラ優先）
    agg: Dict[str, List[Any]] = {}
    for cid, name, pw, cost, imp, is_ld, _acts in all_rows:
        a = agg.setdefault(cid, [name, pw, cost, 0.0, 0, is_ld])
        a[3] += max(0.0, imp)
        a[4] += 1
    cards = [(cid, a[0], a[1], a[2], a[3] / a[4], a[4], a[5]) for cid, a in agg.items()]
    cards.sort(key=lambda t: t[4], reverse=True)
    print(f"\n--- 効果価値 上位{args.top}（カード別クランプ平均Δ・L=リーダー） ---")
    print(f"{'card_id':12} {'pw':>5} {'cost':>4} {'価値':>7} {'n':>3} {'':2} name")
    for cid, name, pw, cost, avg, cnt, is_ld in cards[:args.top]:
        print(f"{cid:12} {pw:5d} {cost:4d} {avg:7.0f} {cnt:3d} {'L' if is_ld else ' ':2} {name}")

    # ActionType 別の測定可否＝「どの効果が測れて、どれが測れない（負/0Δ）か」を炙り出す。
    by_act: Dict[str, List[float]] = {}
    for r in all_rows:
        by_act.setdefault(r[6] or "(none)", []).append(r[4])
    print(f"\n--- ActionType 別の測定可否（生Δ平均・正Δ率・件数）＝測れない効果の特定 ---")
    print(f"{'actions':30} {'生Δ平均':>8} {'正Δ率':>6} {'負Δ率':>6} {'n':>4}")
    rowsa = []
    for act, vs in by_act.items():
        n = len(vs)
        avg = sum(vs) / n
        pos = sum(1 for v in vs if v > 1.0) / n
        neg = sum(1 for v in vs if v < -1.0) / n
        rowsa.append((act, avg, pos, neg, n))
    for act, avg, pos, neg, n in sorted(rowsa, key=lambda t: -t[4]):
        print(f"{act[:30]:30} {avg:8.0f} {pos:6.0%} {neg:6.0%} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
