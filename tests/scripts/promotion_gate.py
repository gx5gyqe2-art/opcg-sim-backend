"""昇格ゲート（v6 柱①）: candidate が現行 best に段階式 arena で勝った場合のみ昇格を PASS する CLI。

`docs/reports/v5_adoption_20260715.md` §4-1。v3/v4/v5 で3回再現した「ピーク一過性」（学習が進むと
ネットが劣化し、最新＝最強でなくなる）への構造的対策。learner は最新ネットを **candidate** に留め、
本ゲートに勝った場合のみ **best**（生成・出荷の採用元）を更新する＝run をいつ止めてもベストが残る。

段階式判定（24局監視 arena は CI±0.19 で判定不能＝v5 実測。判定だけ局数を張る）:
  - stage1: 12ペア=24局（席入替CRN）。**勝ち越し（>50%）で stage2 へ**、五分以下は即棄却
    （真に 55% の candidate が五分以下に沈む確率は ~31%＝次回ゲートで再挑戦できるので許容）。
  - stage2: +38ペア=累計100局。**累計勝率 ≥ 55% で昇格**（AlphaZero evaluator と同水準。
    CI下限>0.5（61/100）まで要求すると微改善が永遠に昇格できないため、比率しきい値にする）。

判定は純関数（stage1_decision / final_decision）＝ `tests/test_promotion_gate.py` が固定する。

実行例（単体・learner からは pd_learn --promote-every 経由で呼ばれる）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/promotion_gate.py \
    --candidate /tmp/cand_v.npz,/tmp/cand_p.npz            # best 未指定＝出荷既定(gen5)
出力: 最終行 `GATE_RESULT {json}`・exit 0=昇格 / 1=棄却。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import random
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

STAGE2_FRAC = 0.55   # 累計勝率がこの比率以上で昇格


def stage1_decision(wins: float, games: int) -> str:
    """stage1（少局数の粗いふるい）: 勝ち越しなら 'continue'、五分以下なら 'reject'。"""
    return "continue" if wins * 2 > games else "reject"


def final_decision(wins: float, games: int, frac: float = STAGE2_FRAC) -> bool:
    """最終判定: 累計勝率 ≥ frac で昇格（浮動小数の境界は昇格側に丸めない）。"""
    return wins + 1e-9 >= frac * games


def anchor_decision(wins: float, games: int, frac: float = 0.5) -> bool:
    """アンカー判定（v7・血統過適合の検出）: 固定アンカー（出荷 gen5 等）に**非退行**（勝率 ≥ frac）。

    v6 run の実測: 対best 連鎖で3段昇格した r99 が、祖先 gen5 との直接対戦で 0.33 と負け越した
    （閉じた血統内の「親に勝つ特化」が祖先への強さに転移しない＝じゃんけん構造）。昇格には
    「対best で勝ち越え」に加えて本判定を課し、血統内だけの見かけの前進を弾く。"""
    return wins + 1e-9 >= frac * games


# --- arena 実行（multiprocessing・席入替CRNペア）------------------------------
_G = {}


class PairTimeout(BaseException):
    """1ペアの実時間上限。**BaseException 派生**が要点で、エンジン/探索側の広い
    `except Exception` に食われて握り潰されないようにする。

    なぜ手数上限では足りないか（2026-08-16・b_p04 seed 904002）: 上限手数は「手」を数えるので、
    **1回の decide() から戻ってこない**（戦闘箱の枝が組合せ的に膨らむ）局面では一生増えない。
    実際に 1局面で 3時間48分 CPU を焼き続けた。実時間で切るしかない。"""


def _init_pool(cand_spec, best_spec, cand_kw=None, leaders_mode="fixed", decks="singleton",
               pair_timeout=0):
    """子プロセス初期化: DB とエンジン2体を1回だけロード（以後の全ペアで共有）。

    `cand_kw`（v35）: **候補席にだけ**渡す LearnedEngine のオプション（例
    `{"box_battle": True, "quiesce": True}`）。機構をグローバル定数で切り替えると
    両席に同時に効いてしまい「新機構つき候補 vs 現行本番」を測れないため、席別の seam を通す。
    未指定＝両席とも既定＝従来と同一挙動。"""
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    _G["leaders_mode"] = leaders_mode
    _G["decks"] = decks
    _G["pair_timeout"] = pair_timeout

    def eng(spec, **kw):
        if not spec:
            return LearnedEngine(**kw)   # 出荷既定（現 gen11）
        if spec.startswith("neff:"):
            # 効果構造符号化ネット（2026-08-27）: n_eff_gate 経由で注入。
            import n_eff_gate
            e = n_eff_gate.neff_engine(spec[5:])
            for k2, v2 in (kw or {}).items():
                setattr(e, k2, v2)
            return e
        if spec.startswith("n1:"):
            # N系ネット（純正Nループ④ 2026-08-26）: value+方策チャネルを n1_gate 経由で
            # 注入したエンジン。席別 seam（cand_kw）は注入後に属性で適用する
            # （LearnedEngine のコンストラクタ引数と同名の属性）。
            import n1_gate
            e = n1_gate.n1_engine(spec[3:])
            for k2, v2 in (kw or {}).items():
                setattr(e, k2, v2)
            return e
        parts = spec.split(",")
        return LearnedEngine(value_path=parts[0],
                             policy_path=parts[1] if len(parts) > 1 else None, **kw)
    _G["db"] = _load_db()
    _G["cand"] = eng(cand_spec, **(cand_kw or {}))
    _G["best"] = eng(best_spec)


# --- 対面の選び方（2026-08-15）------------------------------------------------
# 既定（`leader_deck_builder()`）は**両者ハンニャバル固定のミラー**で、歴代のアリーナ判定は
# 全てこの1対面だけで測られていた（実デッキは一度も対局していない）。ユーザ提案により
# **リーダーをランダム化**して汎化を測れるようにする。
#   fixed  … 従来（既定リーダーのミラー・歴代との地続き比較用）
#   random … 全リーダーからペアごとに2枚引く（**左右非対称**を許す）
#   real   … 実デッキ4リーダーの総当たり（出荷先の対面）
# **ペア内ではリーダー対を固定し、席とリーダーを入れ替える**（game a: cand=L1 / game b:
# cand=L2）＝リーダー相性の有利不利が相殺され、残るのは打ち回しの差だけになる。
REAL_LEADERS = ("OP11-041", "OP09-001", "OP15-058", "OP16-022")   # ナミ/シャンクス/エネル/黒黄ルフィ


def _leader_pool(db):
    if "leaders" not in _G:
        _G["leaders"] = sorted(cid for cid, _ in db.raw_db.items()
                               if (db.get_card(cid) is not None
                                   and getattr(db.get_card(cid).type, "name", "") == "LEADER"))
    return _G["leaders"]


def _leader_pair(db, seed, mode):
    """seed から決定論的にリーダー対を選ぶ（pure・pool は1回だけ構築）。"""
    if mode == "fixed":
        return None, None
    pool = REAL_LEADERS if mode == "real" else _leader_pool(db)
    if not pool:
        return None, None
    rng = random.Random(seed * 7919 + 13)
    return rng.choice(pool), rng.choice(pool)


def _play_pair(args):
    """1ペア＝同seedで**席とリーダーを入替**た2局。candidate の勝ち数(0..2)を返す。"""
    return _play_pair_detail(args)["score"]


def _play_pair_detail(args):
    """`_play_pair` の詳細版: 勝ち数に加えて**どの対面だったか**を返す。

    ランダムリーダー帯では「総合の勝率」だけ見ても、どのリーダーで強い/弱いかが分からない
    （ユーザ決定 2026-08-16: 対面を記録する）。台帳へ leaders を書けるよう、スコアと一緒に
    返す。判定側（勝ち数の集計）は score だけを見るので規約は不変。"""
    seed = args
    from cpu_arena import play_game
    from game_driver import leader_deck_builder
    mode = _G.get("leaders_mode", "fixed")
    la, lb = _leader_pair(_G["db"], seed, mode)
    if _G.get("decks") == "synth":
        # 中身もリーダーに合わせて合成する（deck_synth）。singleton builder は「ID順で色が
        # 合う最初の50枚・全部1枚ずつ・イベント0」という実在しない構築で、テーマ参照や
        # イベント/ステージを持つ効果が一度も盤面に乗らない＝歴代の判定の射程外だった。
        # デッキ内容は seed（=ペア）で決め、ペアの2局では**同じ中身のまま席だけ入替**える。
        from deck_synth import synth_deck_builder
        ab = synth_deck_builder(la, lb, seed=seed) if la else None
        ba = synth_deck_builder(lb, la, seed=seed) if la else None
    else:
        ab = leader_deck_builder(la, lb) if la else None      # game a: cand=la / best=lb
        ba = leader_deck_builder(lb, la) if la else None      # game b: best=lb→p1 なので入替
    if _G.get("pair_timeout"):
        import signal

        def _alarm(_sig, _frm):
            raise PairTimeout(f"pair exceeded {_G['pair_timeout']}s")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(_G["pair_timeout"]))
    try:
        a = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["cand"],
                      p2_engine=_G["best"], deck_builder=ab)
        b = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["best"],
                      p2_engine=_G["cand"], deck_builder=ba)
    except (Exception, PairTimeout) as e:
        # 対局がエンジン欠陥で成立しなかった（上限手数 MAX_STEPS / 実時間上限 等）。**1ペアの失敗で計測全体を
        # 落とさない**（ランダム対面では未知のループを踏むことがあり、シャードが丸ごと止まる）。
        # スコアは付けず void として台帳に残し、集計側で母数から外して**件数を明示**する
        # （黙って落とすと「全部測れた」ように見えてしまう）。対面も残すので後から再現できる。
        return {"seed": seed, "score": None, "leaders": [la, lb],
                "void": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        if _G.get("pair_timeout"):
            import signal as _sg
            _sg.alarm(0)
    wa = 1.0 if a["winner"] == "p1" else 0.0    # game a: 候補が la を握る
    wb = 1.0 if b["winner"] == "p2" else 0.0    # game b: 候補が lb を握る（席とリーダーを入替）
    # leaders=[la, lb] と games=[wa, wb] を対にして残すと、**どのリーダーを握って勝ったか**を
    # 後から集計できる（score だけだと2局の合計なのでリーダー別に割れない）。
    return {"seed": seed, "score": wa + wb, "leaders": [la, lb], "games": [wa, wb],
            "turns": [a.get("turns"), b.get("turns")]}


def run_stage(pool, seeds):
    wins = 0.0
    for w in pool.imap_unordered(_play_pair, seeds):
        wins += w
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--best", default="", help="value.npz[,policy.npz]（未指定＝出荷既定 gen5）")
    ap.add_argument("--pairs1", type=int, default=12, help="stage1 のペア数（局数はx2）")
    ap.add_argument("--pairs2", type=int, default=38, help="stage2 で追加するペア数")
    ap.add_argument("--frac", type=float, default=STAGE2_FRAC)
    ap.add_argument("--anchor", default=None,
                    help="固定アンカー value.npz[,policy.npz]（空文字=出荷既定 gen5）。指定時、"
                         "対best 通過後に candidate vs anchor を --anchor-pairs で測り、"
                         "非退行（勝率 ≥ --anchor-frac）でなければ昇格させない（v7・血統過適合の検出）")
    ap.add_argument("--anchor-pairs", type=int, default=12)
    ap.add_argument("--anchor-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=21000,
                    help="ペアseedの基点（学習roundを混ぜて呼び出し側が変える＝毎回同じ開幕で測らない）")
    args = ap.parse_args()

    t0 = time.time()
    pool = mp.Pool(args.workers, initializer=_init_pool, initargs=(args.candidate, args.best))
    result = {}
    try:
        wins = run_stage(pool, [args.seed_base + k for k in range(args.pairs1)])
        games = args.pairs1 * 2
        d1 = stage1_decision(wins, games)
        print(f"stage1: {wins}/{games} → {d1} ({time.time()-t0:.0f}s)", flush=True)
        promoted = False
        if d1 == "continue":
            wins += run_stage(pool, [args.seed_base + args.pairs1 + k for k in range(args.pairs2)])
            games += args.pairs2 * 2
            promoted = final_decision(wins, games, args.frac)
            print(f"stage2: 累計 {wins}/{games} (要{args.frac:.2f}) ({time.time()-t0:.0f}s)", flush=True)
    finally:
        pool.terminate(); pool.join()
    if promoted and args.anchor is not None:
        # アンカー段: 対best を超えた candidate だけが来る（稀）＝追加コストは昇格候補時のみ。
        pool = mp.Pool(args.workers, initializer=_init_pool, initargs=(args.candidate, args.anchor))
        try:
            aw = run_stage(pool, [args.seed_base + 500 + k for k in range(args.anchor_pairs)])
        finally:
            pool.terminate(); pool.join()
        ag = args.anchor_pairs * 2
        a_ok = anchor_decision(aw, ag, args.anchor_frac)
        print(f"anchor: {aw}/{ag} (要{args.anchor_frac:.2f}) → {'OK' if a_ok else 'NG=血統過適合'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        result.update(anchor_wins=aw, anchor_games=ag, anchor_ok=a_ok)
        promoted = promoted and a_ok
    result.update(promoted=promoted, wins=wins, games=games,
                  wr=round(wins / games, 4), stage1=d1, sec=round(time.time() - t0))
    print("GATE_RESULT " + json.dumps(result), flush=True)
    return 0 if promoted else 1


if __name__ == "__main__":
    _sys.exit(main())
