"""レフェリー再ラベル・パイプライン（v9 フェーズ1・docs/cpu_v9_plan.md §2）。

外部アンカー学習の教師データ生成器。3段構成:
  1. **生成**: 両席 learned(gen5) の記録つき自己対戦（`record_selfplay_descriptor`）＝
     真盤面をいつでも再生できる記述子。
  2. **採掘**（生成と同じ1パスで観測・読み取り専用）: 効率盲点（合法手の1-ply後 value 差 < ε
     ＝ループが学べなかった無差別点）と飽和負け（value < −しきい＝捲りラベルの主戦場）を検出、
     1局あたり上限 K 点を選ぶ（少数を深く）。
  3. **ラベル**: 決定点を真盤面再生→プラン自動列挙→CRN 対照評価（飽和は捲りエスカレーション）
     → policy 教師＝同価値バンド上位プランの初手 multi-hot／value 教師＝z=2·wr−1（捲り率）。

出力は既存バッチスキーマ v2（pd_learn がそのまま消費可能）＋ meta（worker="ref"）。
教師ネットは gen5 固定＝学習が進んでもラベルはドリフトしない（外部の錨）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/referee_labeler.py \
    --games 4 --seed0 9000 --worlds 4 --sims 32 --out /tmp/refbatch
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import replay_runner as RR
import p3_loop as P
import rl_net as RN
import rl_encoder as E
import heldout_decks as HD
from az_policy import PolicyScorer, state_context
from opcg_action import legal_action_matrix
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from pd_batch_common import pack_policy
from opcg_sim.src.core import cpu_ai

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WINDOWS = ("MAIN_ACTION", "SELECT_COUNTER", "SELECT_BLOCKER")


class _MineObserver:
    """生成中の各決定点で採掘条件を観測する（読み取り専用・record と同順で index が揃う）。"""

    def __init__(self, game_root, vf, eps, sat, pf=None, disagree_margin=None):
        self.game_root, self.vf, self.eps, self.sat = game_root, vf, eps, sat
        self.pf = pf   # policy priors_fn(state, legal)→配列（反例採掘用・None で無効）
        self.disagree_margin = disagree_margin if disagree_margin is not None else eps
        self.cands = []   # (action_index, kind, metric, actor, pend種)
        self._i = 0

    TOP_M = 4   # blind 判定は上位M手の spread（v7 実測＝候補間の差 0.01〜0.05 が対象。
                # 全手 spread だと明確な悪手1つで閾値を超え @64 型を取り逃す・レビュー指摘#4）

    def on_decision(self, ctx, move):
        i = self._i
        self._i += 1
        m = ctx.manager
        pend = m.get_pending_request(with_request_id=False) or {}
        if pend.get("action") not in _WINDOWS:
            return
        name = ctx.actor.name
        try:
            v = self.vf(m, name)
        except Exception:
            return
        if v < self.sat:
            self.cands.append((i, "sat", float(v), name, pend.get("action")))
            return
        legal = self.game_root.legal_actions(m)
        if len(legal) < 3:
            return
        # 各手の1-ply後 value（legal と同順・不成立手は −∞ で除外扱い）。
        vals = []
        for mv in legal:
            child = self.game_root.apply(m, mv, name)
            vals.append(self.vf(child, name) if child is not None else -1e9)
        vals = np.asarray(vals, dtype=np.float64)
        top = np.sort(vals)[::-1][:self.TOP_M]
        if top[0] - top[-1] < self.eps:
            self.cands.append((i, "blind", float(top[0] - top[-1]), name, pend.get("action")))
            return
        # 反例採掘（disagree）: policy top1 が 1-ply value 最善と食い違い、かつ policy が推す手が
        # value で明確に劣る点。@82（policy=カウンター切る／value=温存）・@68（policy=TURN_END／
        # value=攻撃）型の「policy を矯正すべき点」を能動的に拾う。採掘はノイジーでよい
        # （最終ラベルはレフェリー＝真実源が付ける・採掘は「どこを見るか」だけ）。
        if self.pf is not None:
            try:
                pri = self.pf(m, legal)
            except Exception:
                pri = None
            if pri is not None and len(pri) == len(legal):
                vi = int(np.argmax(vals))
                pi = int(np.argmax(pri))
                loss = float(vals[vi] - vals[pi])
                if pi != vi and loss > self.disagree_margin:
                    self.cands.append((i, "disagree", loss, name, pend.get("action")))


# 採掘カテゴリと、metric の「優先向き」（昇順=小さいほど優先／降順=大きいほど優先）。
#   sat:   飽和負け＝value が低いほど優先（捲りラベルの主戦場・昇順）
#   disagree: policy 損失が大きいほど優先（反例＝policy 矯正の主戦場・降順）
#   blind: 1-ply spread が小さいほど優先（効率盲点・昇順）
_MINE_CATS = ("sat", "disagree", "blind")
_MINE_ASC = {"sat": True, "disagree": False, "blind": True}


def select_candidates(cands, max_per_game):
    """採掘候補から1局分の採点対象を選ぶ（pure）。

    **カテゴリ round-robin**（sat/disagree/blind を交互に採る）: どのカテゴリも飢えさせない
    （決着局終盤の sat 洪水で効率盲点/反例が枯れる構造の防止・レビュー指摘#3 の一般化）。
    各カテゴリは優先向き（`_MINE_ASC`）でソートし、上位から1つずつ回して max_per_game 個選ぶ。
    同種の隣接 index（同一ターンの連鎖）は間引く（index 差 < 2）。"""
    pools = {}
    for cat in _MINE_CATS:
        pools[cat] = sorted([c for c in cands if c[1] == cat],
                            key=lambda c: c[2], reverse=not _MINE_ASC[cat])
    active = [cat for cat in _MINE_CATS if pools[cat]]
    picked = []
    r = 0
    while len(picked) < max_per_game and any(len(pools[cat]) > r for cat in active):
        for cat in active:
            if len(picked) >= max_per_game:
                break
            if len(pools[cat]) > r:
                c = pools[cat][r]
                if not any(abs(c[0] - p[0]) < 2 for p in picked):
                    picked.append(c)
        r += 1
    return sorted(picked)


def plan_teacher_visit(legal_keys, entries, band):
    """プラン判定 → 合法手上の policy 教師分布（pure）。

    同価値バンド上位（best＋same_value）プランの**初手**へ均等に重みを置く multi-hot。
    バンド外プランの初手は 0＝「劣るプラン」を明示的に教える。どの初手も合法手に
    見つからなければ None（呼び出し側でスキップ＝黙って誤教師を作らない）。"""
    entries = sorted(entries, key=lambda e: (-e["wins"], -e["lifem"]))
    best = entries[0]
    tops = [e for e in entries if e is best or CR.same_value(best, e, band)]
    visit = np.zeros(len(legal_keys), dtype=np.float64)
    for e in tops:
        k0 = repr(e["keys"][0])
        for j, lk in enumerate(legal_keys):
            if repr(lk) == k0:
                visit[j] += 1.0
                break
    if visit.sum() <= 0:
        return None
    return visit / visit.sum()


def label_decision(db, game_root, game_serve, vf, pf, vocab, ev, desc, idx, log=print,
                   expect=None):
    """決定点 idx を真盤面再生してレフェリー教師サンプルを作る。

    `expect=(actor名, pend種)`（採掘時の観測）があれば再生結果と照合し、不一致は棄却＝
    採掘と再生の index ズレで**黙って別局面をラベルする**最悪モードを防ぐ（レビュー指摘#9）。
    返り値 (val_sample, pol_sample) or (None, None)（再生不能・強制手・教師構築不能）。
    val_sample = (enc, z)・pol_sample = (ctx, am, visit)。"""
    m0, who = RR.state_at_action(db, desc, idx)
    if m0 is None:
        return None, None
    pend = m0.get_pending_request(with_request_id=False) or {}
    if pend.get("action") not in _WINDOWS:
        return None, None
    if expect is not None and (who, pend.get("action")) != tuple(expect):
        log(f"    @{idx} 照合不一致（採掘{expect} vs 再生{(who, pend.get('action'))}）＝棄却")
        return None, None
    name = who
    plans = CR.enumerate_turn_plans(game_root, vf, m0, name, max_len=ARGS.plan_len,
                                    beam=ARGS.beam, max_plans=ARGS.max_plans, log=log)
    if len(plans) <= 1:
        return None, None
    entries = [{"label": ">".join(CR._step_label(d) for d in descs), "keys": keys}
               for keys, descs in plans]
    CR._eval_entries(entries, game_root, game_serve, vf, pf, m0, name, ARGS.worlds)
    entries.sort(key=lambda e: (-e["wins"], -e["lifem"]))
    n_w = ARGS.worlds
    if ARGS.comeback > 0 and entries[0]["wins"] <= 1:
        sub = entries[:min(6, len(entries))]
        CR._eval_entries(sub, game_root, game_serve, vf, pf, m0, name, ARGS.worlds * 4,
                         opp_temp=ARGS.comeback)
        sub.sort(key=lambda e: (-e["wins"], -e["lifem"]))
        entries = sub
        n_w = ARGS.worlds * 4
    legal = game_root.legal_actions(m0)
    legal_keys = []
    for mv in legal:
        try:
            legal_keys.append(cpu_ai._move_equiv_key(m0, mv))
        except Exception:
            legal_keys.append(None)
    visit = plan_teacher_visit(legal_keys, entries, ARGS.band)
    if visit is None:
        return None, None
    best = entries[0]
    z = 2.0 * (best["wins"] / max(best["ok"], 1)) - 1.0
    enc = E.encode(m0, name, vocab, version=ev)
    ctx = state_context(m0, name, vocab, version=ev)
    am = legal_action_matrix(m0, legal, name)
    log(f"    @{idx} {'捲り' if n_w > ARGS.worlds else ''}教師: "
        f"z={z:+.2f} 初手候補{int((visit > 0).sum())}/{len(legal)}  最良: {best['label']}")
    return (enc, z), (ctx, am, visit)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--sims-play", type=int, default=120, help="生成対局の decide sims")
    ap.add_argument("--sims", type=int, default=32, help="ラベル時ロールアウトの sims")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--comeback", type=float, default=0.7)
    ap.add_argument("--eps", type=float, default=0.10, help="効率盲点: 1-ply value 差 < ε")
    ap.add_argument("--sat", type=float, default=-0.8, help="飽和負け: value < これ")
    ap.add_argument("--max-per-game", type=int, default=4)
    ap.add_argument("--plan-len", type=int, default=4)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--max-plans", type=int, default=12)
    ap.add_argument("--out", default=None, help="batch.npz/meta.json の出力先ディレクトリ")
    ARGS = ap.parse_args()
    CR.ARGS = ARGS   # enumerate/rollout が sims 等を参照

    db = _load_db()
    vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
    pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    from opcg_sim.src.core.cpu_learned import _net_enc_version
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    vf = P.value_fn_of(vnet, vocab, ev)
    pf = P.priors_fn_of(pnet, vocab, ev)
    game_root = OPCGGame(prune_futile=False)
    game_serve = OPCGGame()
    ids = HD.deck_ids()

    def _deckb(_db, seed):
        l1, c1 = HD.build(_db, ids[seed % len(ids)], "p1")
        l2, c2 = HD.build(_db, ids[(seed + 1) % len(ids)], "p2")
        return l1, c1, l2, c2

    sinks = {"S": [], "F": [], "I": [], "Y": [], "Q": [], "T": []}
    pol = []
    n_labeled = 0
    for g in range(ARGS.games):
        seed = ARGS.seed0 + g
        miner = _MineObserver(game_root, vf, ARGS.eps, ARGS.sat, pf=pf)
        t0 = time.time()
        desc = RR.record_selfplay_descriptor(db, seed, _deckb, sims=ARGS.sims_play,
                                             first_player="random", observers=[miner])
        picked = select_candidates(miner.cands, ARGS.max_per_game)
        print(f"game {g + 1}/{ARGS.games} seed={seed}: {len(desc['actions'])}手 "
              f"候補{len(miner.cands)}→採掘{len(picked)} ({time.time() - t0:.0f}s)", flush=True)
        for idx, kind, metric, actor, pend_kind in picked:
            vs, ps = label_decision(db, game_root, game_serve, vf, pf, vocab, ev, desc, idx,
                                    expect=(actor, pend_kind))
            if vs is None:
                continue
            enc, z = vs
            sinks["S"].append(enc["scalars"]); sinks["F"].append(enc["field"])
            sinks["I"].append(enc["card_idx"])
            sinks["Y"].append(z); sinks["Q"].append(z); sinks["T"].append(np.nan)
            pol.append(ps)
            n_labeled += 1
    print(f"\nLABEL_RESULT: {ARGS.games}局 → 教師 {n_labeled} 決定", flush=True)
    if ARGS.out and n_labeled:
        os.makedirs(ARGS.out, exist_ok=True)
        arrays = {"scalars": np.stack(sinks["S"]), "field": np.stack(sinks["F"]),
                  "card_idx": np.stack(sinks["I"]),
                  "value": np.array(sinks["Y"], dtype=np.float32),
                  "q_root": np.array(sinks["Q"], dtype=np.float32),
                  "turns_left": np.array(sinks["T"], dtype=np.float32)}
        arrays.update(pack_policy(pol))
        np.savez_compressed(os.path.join(ARGS.out, "batch.npz"), **arrays)
        # worker はワーカー運用時に label_worker が w1 等へ上書きする（枝と consumed の単位）。
        # ref バッチの判別は source 欄が正（is_fresh の staleness 免除もこれを見る）。
        meta = {"worker": "ref", "batch_id": int(time.time()), "against_round": -1,
                "games": ARGS.games, "states": n_labeled, "schema_version": 2,
                "source": "referee_label", "worlds": ARGS.worlds, "comeback": ARGS.comeback,
                "miner": 3}   # 採掘条件の版: 3=sat/disagree/blind の3カテゴリround-robin
                              #（disagree=policy vs 1-ply value 最善の乖離＝反例採掘・v9.3）
        with open(os.path.join(ARGS.out, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"saved: {ARGS.out}/batch.npz ({n_labeled} states)", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
