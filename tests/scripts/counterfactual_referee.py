"""反実仮想レフェリー（root全数モード）: 1つの決定点で全選択肢を「同一世界で最後まで」打ち比べる。

docs/cpu_v7_plan.md の次段（教師CPU構想）の核。何万局の独立対局の平均（勝率）ではなく、
**同じ局面・同じ隠れ情報の世界で root の1手だけを変える対照実験**（CRN）により、
数個の世界線で選択の因果効果を測る:

  1. マーク局面を復元し、root の合法手を**枝刈りなしで全列挙**（等価手は探索と同じ規約でマージ済み）。
  2. 世界線 w=1..K: 相手の隠れ情報（手札等）を決定化で再サンプル＝「ありえた現実」を1つ固定。
     **同じ世界線を全 root 手で共有**（CRN・運の共通項を打ち消す）。
  3. 各 (root手, 世界線) で手を適用し、以降は両者とも同一エンジン（既定=出荷 gen5・固定教師）が
     終局まで打つ。探索の決定化は世界線から導出した sticky seed＝分岐間で可能な限り乱数を共有。
  4. 出力: 手ごとの勝ち数/K・ランキング・人間指摘方向との一致。差が分解能未満なら「同価値」。

レフェリーの性質: 教師ネットは**固定**（学習で漂流しない）＝v7 で確定した「オラクルが value の
ドリフトを継承する」問題を持たない外部の錨。ロールアウトは実対局同様の serve 設定（枝刈りON）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/counterfactual_referee.py \
    --marks g3:64,g3:68,g1:12,g3:82,g3:93,g1:16 --worlds 6 --sims 64
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import mark_gate as MG
import replay_reeval as RE
import replay_runner as RR
import p3_loop as P
import rl_net as RN
import rl_encoder as E
from az_policy import PolicyScorer
from az_mcts_tree import TreeMCTS
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAX_STEPS = 400


def _mark_table():
    t = {}
    t.update(MG.TARGETS); t.update(MG.GUARDS)
    t.update(MG.V5_TARGETS); t.update(MG.V5_GUARDS)
    return t


def rollout(game_serve, vf, pf, state, mover, world_seed, rng_seed):
    """state から終局まで両者同一エンジンで打つ（temp0・sticky世界線）。

    返り値 (winner, life_diff, end_turn): life_diff は mover 視点の残ライフ差（勝ち方の質・
    タイブレーク用）。end_turn は決着ターン（速い勝ち／粘る負けの判別用）。"""
    m = state
    world = {}
    steps = 0
    rng = np.random.default_rng(rng_seed)
    while m.winner is None and not game_serve.is_terminal(m) and steps < MAX_STEPS:
        name = game_serve.current_player(m)
        if name is None:
            break
        key = (int(getattr(m, "turn_count", 0) or 0), name)
        ds = world.get(key)
        if ds is None:
            ds = world[key] = int((world_seed * 1000003 + key[0] * 131 +
                                   (0 if name == "p1" else 7)) % (2 ** 63 - 1))
        mcts = TreeMCTS(game_serve, value_fn=vf, priors_fn=pf, c_puct=1.5, n_sims=ARGS.sims,
                        dirichlet_eps=0.0,
                        determinize_fn=lambda s, r, _d=ds, _n=name:
                            game_serve.determinize(s, _n, np.random.default_rng(_d)),
                        rng=rng)
        mv, N, legal = mcts.run(m)
        if mv is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, mv)
        except Exception:
            break
        steps += 1
    me = m.p1 if m.p1.name == mover else m.p2
    opp = m.p2 if m.p1.name == mover else m.p1
    life_diff = len(getattr(me, "life", []) or []) - len(getattr(opp, "life", []) or [])
    return m.winner, life_diff, int(getattr(m, "turn_count", 0) or 0)


def _restore_board(db, tag, i):
    """マーク局面の盤面を用意する。--true-board 時は記録全手順の再実行（`state_at_action`）＝
    パワー修正・一時効果込みの真盤面。既定はフレーム復元（従来挙動・公開情報のみ）。"""
    rec, fbi, actions = GAMES[tag]
    if ARGS.true_board:
        m, who = RR.state_at_action(db, rec, i, frames=fbi)
        if m is None:
            return f"true-board 再生不能: {who}"
        return m, who
    return MG._restore(db, rec, fbi, actions, i)


def referee_position(db, game_root, game_serve, vf, pf, tag, i, pred, worlds, log=print):
    built = _restore_board(db, tag, i)
    if isinstance(built, str):
        log(f"{tag}@{i}: 復元不可 ({built})"); return None
    m0, actor = built
    name = actor.name if hasattr(actor, "name") else actor
    legal = game_root.legal_actions(m0)   # 枝刈りなしの全列挙
    descs = []
    for mv in legal:
        try:
            d = cpu_ai._describe_move(m0, mv) or {}
        except Exception:
            d = {"action_type": (mv or {}).get("action_type")}
        descs.append(d)
    wins = np.zeros(len(legal))
    life = np.zeros(len(legal))     # mover 視点の残ライフ差の合計（勝ち方の質）
    turns = np.zeros(len(legal))    # 決着ターン合計（速い勝ちの判別・参考）
    outcomes = [dict() for _ in legal]   # 世界別勝敗（同価値バンド v2 の対判定用）
    t0 = time.time()
    for w in range(worlds):
        # 世界線 w: 隠れ情報を再サンプルした「ありえた現実」。全 root 手で共有（CRN）。
        world = game_serve.determinize(m0, name, np.random.default_rng(90000 + w * 97))
        for k, mv in enumerate(legal):
            child = game_serve.apply(world, mv, name)
            if child is None:
                continue
            winner, ld, et = rollout(game_serve, vf, pf, child, name,
                                     world_seed=90000 + w * 97, rng_seed=w * 7919 + k)
            outcomes[k][w] = (winner == name)
            if winner == name:
                wins[k] += 1
            life[k] += ld
            turns[k] += et
    lifem = life / max(worlds, 1)
    # 順位 = (勝ち数, 残ライフ差) の辞書式。飽和局面（全手同勝敗）でも「勝ち方の質」で判別する。
    score = wins * 1000 + lifem
    order = np.argsort(-score)
    human = np.array([bool(pred(d)) for d in descs])

    def _best(mask):
        if not mask.any():
            return None
        j = int(np.argmax(np.where(mask, score, -1e18)))
        return wins[j], lifem[j]
    bh, bn = _best(human), _best(~human)
    agree = bh is not None and (bn is None or (bh[0], round(bh[1], 3)) >= (bn[0], round(bn[1], 3)))
    wm = (bh[0] - bn[0]) if (bh and bn) else float("nan")
    lm = (bh[1] - bn[1]) if (bh and bn) else float("nan")
    log(f"\n=== {tag}@{i}（{len(legal)}手 × {worlds}世界・{time.time()-t0:.0f}s）"
        f" 人間一致={'○' if agree else '✗'}  margin: 勝ち{wm:+.0f}/{worlds}・ライフ{lm:+.2f} ===")
    top = order[0]
    top_e = {"outcomes": outcomes[top], "lifem": float(lifem[top])}
    for k in order:
        d = descs[k]
        mark = "◆人間" if human[k] else "  "
        # 同価値バンド v2（v8 柱B・対判定）: 最善との世界別勝敗の正味不一致 < 3 かつ
        # ライフ差 < band は ≈（同価値圏）。
        tie = "≈" if (k != top and same_value(
            top_e, {"outcomes": outcomes[k], "lifem": float(lifem[k])}, ARGS.band)) else " "
        log(f"  {mark}{tie} {wins[k]:.0f}/{worlds} L{lifem[k]:+.2f} T{turns[k]/max(worlds,1):.1f}"
            f"  {d.get('action_type')}"
            f"{'/' + str(d.get('card')) if d.get('card') else ''}")
    return {"mark": f"{tag}@{i}", "agree": bool(agree), "win_margin": wm, "life_margin": lm,
            "n_moves": len(legal)}


def _match_move(state, legal, step):
    """プラン1歩（'ACTION_TYPE:card' または 'ACTION_TYPE'）に合致する合法手を返す（無ければ None）。"""
    at, _, card = step.partition(":")
    for mv in legal:
        try:
            d = cpu_ai._describe_move(state, mv) or {}
        except Exception:
            continue
        if d.get("action_type") == at and (not card or d.get("card") == card):
            return mv
    return None


def same_value(best, e, band=0.5, min_discord=3):
    """同価値バンド v2（v8 柱B・対判定）: CRN＝全プランが同じ世界線を共有する構造を活かし、
    **同じ世界で勝敗が割れたペアの正味差**（符号検定風）で断定/同価値を分ける。

    素朴な「勝ち数差」は ±1 勝のノイズで断定が揺れる（@64 実測: 6世界=同値・12世界=素攻撃+1・
    16世界=付与+1 と符号が往復）。対判定なら運の共通項が消えた純粋な差だけが残る:
      - 正味不一致 (n10−n01) ≥ min_discord → 実差＝断定
      - それ未満 かつ 平均残ライフ差 < band → 同価値
      - それ未満 でも ライフ差 ≥ band → 勝ち方の質での序列（断定はしない・表示順のみ）
    best/e は {"outcomes": {world: bool}, "lifem": float}。"""
    b, o = best.get("outcomes") or {}, e.get("outcomes") or {}
    common = [w for w in b if w in o]
    n10 = sum(1 for w in common if b[w] and not o[w])
    n01 = sum(1 for w in common if o[w] and not b[w])
    if n10 - n01 >= min_discord:
        return False
    return abs(best.get("lifem", 0.0) - e.get("lifem", 0.0)) < band


def _match_move_by_key(state, legal, key):
    """equiv キー（`cpu_ai._move_equiv_key`）に一致する合法手を返す（自動列挙プランの再適用用）。"""
    for mv in legal:
        try:
            if cpu_ai._move_equiv_key(state, mv) == key:
                return mv
        except Exception:
            continue
    return None


def _step_label(d):
    s = f"{d.get('action_type')}"
    if d.get("card"):
        s += f":{d['card']}"
    if d.get("targets"):
        s += "→" + ",".join(str(t) for t in d["targets"])
    return s


def enumerate_turn_plans(game_root, vf, m0, name, max_len=4, beam=12, max_plans=16, log=print):
    """自ターン内のプラン（手順列）を自動列挙する（v8 柱A・docs/cpu_v8_plan.md §1）。

    プラン終端 = **手番が自分から離れた瞬間**（攻撃宣言→防御応答へ／TURN_END→相手ターンへ）。
    以降はロールアウトの領分、という境界を action_type の列挙でなく手番遷移で判定する
    （カード非依存・汎用）。
      - 等価除去: 同一手集合の並べ替え（sorted equivキー多重集合）は最初の順序のみ残す。
      - ビーム: 各深さの継続候補を value（固定 gen5）で評価し上位 beam 本のみ展開。
      - プラン上限 max_plans も value 順。**切り捨ては必ずログ**（無言の縮約禁止）。
    返り値: [(keys, descs)]（keys=equivキー列＝世界線への再適用用、descs=表示用）。
    """
    plans = []                      # (keys, descs, 終端盤面value)
    seen_sets = set()               # sorted(keys) の多重集合＝並べ替え等価の除去
    frontier = [([], [], m0)]
    for _depth in range(max_len):
        nxt = []
        for keys, descs, st in frontier:
            for mv in game_root.legal_actions(st):
                try:
                    d = cpu_ai._describe_move(st, mv) or {}
                    k = cpu_ai._move_equiv_key(st, mv)
                except Exception:
                    continue
                sig = tuple(sorted(map(repr, keys + [k])))
                if sig in seen_sets:
                    continue
                child = game_root.apply(st, mv, name)
                if child is None:
                    continue
                seen_sets.add(sig)
                if game_root.current_player(child) != name:
                    plans.append((keys + [k], descs + [d], vf(child, name)))
                else:
                    nxt.append((keys + [k], descs + [d], child))
        if not nxt:
            break
        if len(nxt) > beam:
            nxt.sort(key=lambda t: -vf(t[2], name))
            log(f"  [beam] 深さ{_depth + 1}: 継続 {len(nxt)}→{beam} 本に縮約")
            nxt = nxt[:beam]
        frontier = nxt
    else:
        if frontier:
            log(f"  [len] 長さ上限 {max_len} で未終端 {len(frontier)} 本を破棄")
    if len(plans) > max_plans:
        # 上限は**コミットメント（最終手＝攻撃宣言/TURN_END）別のラウンドロビン**で選ぶ。
        # 素朴な value 順は gen5 の盲点（付与→TURN_END を過大評価）でプラン一覧が埋まり、
        # 攻撃プランが全滅する（@64 実測）。レフェリーはネットの予断を**正す**計器なので、
        # 異なるコミットメントを必ず代表させ、同一コミットメント内の順位だけ value に任せる。
        groups: dict = {}
        for t in plans:
            groups.setdefault(repr(t[0][-1]), []).append(t)
        def _self_touch(t):
            # 準備手のうち最終コミット手と同じカードに触れる数（例: 攻撃者自身へのドン付与）。
            # 同長プランでは自己強化系を優先——「他ユニットに付与→攻撃」より「攻撃者に付与→攻撃」が
            # 比較の本命（equiv キーの card 欄＝構造のみ・カード非依存）。
            final_card = t[0][-1][1]
            return sum(1 for k in t[0][:-1] if k[1] is not None and k[1] == final_card)
        for g in groups.values():
            # 種内は**短いプラン優先**（素形→装飾の順・同長は自己強化→value）。value 単独だと
            # ネットの付与バイアスで素の攻撃/素の TURN_END が押し出される（@64 実測）。
            g.sort(key=lambda t: (len(t[0]), -_self_touch(t), -t[2]))
        order = sorted(groups.values(), key=lambda g: -g[0][2])
        kept = []
        r = 0
        while len(kept) < max_plans and any(len(g) > r for g in order):
            for g in order:
                if len(g) > r and len(kept) < max_plans:
                    kept.append(g[r])
            r += 1
        log(f"  [cap] プラン {len(plans)}→{len(kept)} 本に縮約"
            f"（コミットメント{len(groups)}種のラウンドロビン・種内は短さ→自己強化→value）")
        plans = kept
    return [(keys, descs) for keys, descs, _v in plans]


def plan_referee(db, game_root, game_serve, vf, pf, tag, i, plans, worlds,
                 band=0.5, log=print):
    """プランモード（教師CPU §ターンプラン）: 手順列を**固定して**適用→以降ロールアウト。

    root prefix 比較はロールアウト役の盲点（例: 付与後に攻撃しない）に後半の実行を委ねてしまい、
    「付与→攻撃」のようなプランの価値を系統的に取りこぼす（g3@64 のトレースで実証）。
    プラン全体を固定すれば、比較対象は純粋に「プラン間の因果差」になる。

    `plans`: 手指定リスト（'ACTION:card>...' 文字列）または "auto"（v8 柱A・自動列挙）。
    `band`: 同価値バンド（v8 柱B）。最善と勝ち数が同じ かつ ライフ差の差 < band の
    プランは「≈同価値」と申告し、バンド外の差だけを序列として断定する。"""
    built = _restore_board(db, tag, i)
    if isinstance(built, str):
        log(f"{tag}@{i}: 復元不可 ({built})"); return None
    m0, actor = built
    name = actor.name if hasattr(actor, "name") else actor
    if plans == "auto":
        auto = enumerate_turn_plans(game_root, vf, m0, name, max_len=ARGS.plan_len,
                                    beam=ARGS.beam, max_plans=ARGS.max_plans, log=log)
        entries = [{"label": ">".join(_step_label(d) for d in descs), "keys": keys}
                   for keys, descs in auto]
    else:
        entries = [{"label": p, "steps": p.split(">")} for p in plans]
    if not entries:
        log(f"{tag}@{i}: プラン0本"); return None
    for e in entries:
        e["wins"] = 0.0; e["life"] = 0.0; e["ok"] = 0; e["outcomes"] = {}
    for w in range(worlds):
        world = game_serve.determinize(m0, name, np.random.default_rng(90000 + w * 97))
        for n_e, e in enumerate(entries):
            m = world
            ok = True
            for step in (e.get("keys") or e["steps"]):
                legal = game_root.legal_actions(m)
                mv = (_match_move_by_key(m, legal, step) if "keys" in e else
                      _match_move(m, legal, step))
                if mv is None:
                    ok = False; break
                m = game_serve.apply(m, mv, name)
                if m is None:
                    ok = False; break
            if not ok:
                continue   # この世界ではプラン不成立（勝ち加算なし）
            winner, ld, _et = rollout(game_serve, vf, pf, m, name,
                                      world_seed=90000 + w * 97, rng_seed=w * 7919 + n_e)
            e["outcomes"][w] = (winner == name)
            if winner == name:
                e["wins"] += 1
            e["life"] += ld
            e["ok"] += 1
    for e in entries:
        e["lifem"] = e["life"] / max(e["ok"], 1)
    entries.sort(key=lambda e: (-e["wins"], -e["lifem"]))
    best = entries[0]
    log(f"\n=== プラン比較 {tag}@{i}（{len(entries)}プラン × {worlds}世界・band={band}）===")
    for e in entries:
        mark = "★" if e is best else ("≈" if same_value(best, e, band) else " ")
        miss = f" (不成立{worlds - e['ok']})" if e["ok"] < worlds else ""
        log(f"  {mark} {e['wins']:.0f}/{worlds} L{e['lifem']:+.2f}  {e['label']}{miss}")
    return entries


def main():
    global ARGS, GAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default="g3:64,g3:68,g1:12,g3:82,g3:93,g1:16",
                    help="tag:index のカンマ区切り")
    ap.add_argument("--plans", default=None,
                    help="プランモード: 'A|B' 形式（各プランは 'ACTION:card>ACTION:card' の手順列）"
                         "または 'auto'（v8 柱A・自ターン内プランの自動列挙）。"
                         "指定時は --marks の最初の1件でプラン同士を比較する")
    ap.add_argument("--band", type=float, default=0.5,
                    help="同価値バンド（v8 柱B）: 勝ち数同一かつライフ差の差がこれ未満は同価値と申告")
    ap.add_argument("--plan-len", type=int, default=4, help="自動列挙の手順長上限")
    ap.add_argument("--beam", type=int, default=12, help="自動列挙のビーム幅（value順）")
    ap.add_argument("--max-plans", type=int, default=16, help="自動列挙のプラン上限（value順）")
    ap.add_argument("--worlds", type=int, default=6, help="世界線数 K（CRN で全手に共有）")
    ap.add_argument("--sims", type=int, default=64, help="ロールアウト中の decide sims")
    ap.add_argument("--net", default=None,
                    help="value.npz[,policy.npz]（既定=出荷 gen5＝固定教師・ドリフトしない錨）")
    ap.add_argument("--true-board", action="store_true",
                    help="盤面をフレーム復元でなく記録全手順の再実行（真盤面＝パワー修正・"
                         "一時効果込み）で用意する")
    ARGS = ap.parse_args()

    db = _load_db()
    if ARGS.net:
        parts = ARGS.net.split(",")
        vnet = RN.ValueNet.load(parts[0])
        pnet = PolicyScorer.load(parts[1]) if len(parts) > 1 else None
    else:
        vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
        pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    vf = P.value_fn_of(vnet, vocab, ev)
    pf = P.priors_fn_of(pnet, vocab, ev)

    game_root = OPCGGame(prune_futile=False)   # root は全列挙
    game_serve = OPCGGame()                    # ロールアウトは serve 同等（config に従う）

    table = _mark_table()
    GAMES = {}
    results = []
    marks = []
    for spec in ARGS.marks.split(","):
        tag, i = spec.split(":"); i = int(i)
        marks.append((tag, i))
        if tag not in GAMES:
            raw = RE.load_replay_json(MG.REPLAYS[tag]); rec = raw.get("replay", raw)
            GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                          rec["actions"])
    if ARGS.plans:
        tag, i = marks[0]
        plans = "auto" if ARGS.plans == "auto" else ARGS.plans.split("|")
        plan_referee(db, game_root, game_serve, vf, pf, tag, i,
                     plans, ARGS.worlds, band=ARGS.band)
        return 0
    for tag, i in marks:
        pred = table[(tag, i)][1]
        r = referee_position(db, game_root, game_serve, vf, pf, tag, i, pred, ARGS.worlds)
        if r:
            results.append(r)
    n_ok = sum(1 for r in results if r["agree"])
    print(f"\nREFEREE_RESULT 一致 {n_ok}/{len(results)}: "
          + ", ".join(f"{r['mark']}={'○' if r['agree'] else '✗'}"
                      f"(勝{r['win_margin']:+.0f}/L{r['life_margin']:+.2f})" for r in results),
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
