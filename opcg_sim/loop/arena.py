"""自己対戦アリーナ（席入替 CRN のペア対局・ペア水準 CI・帯設計・台帳）。

旧 `tests/scripts/arena_parallel.py`／`arena_gate.py`／`promotion_gate.py` の対局部と集計部を
1 か所へ集約したもの。**エンジンは Rust**（`opcg_sim.loop.driver`）だが、**判定規約は 1 文字も
変えていない**——歴代の台帳・seed 帯・報告と地続きに読めることが測定の値打ちなので:

- ペア＝同 seed で**席を入れ替えた 2 局**。候補は 2 局とも `la` を握る
  （game a: cand=p1=la／game b: cand=p2=la・相手は 2 局とも `lb`）。**入れ替わるのは席（先後）
  だけでリーダーは入れ替わらない**（2026-09-03 に台帳の発火イベントから判明した実態。総合勝率は
  la/lb がランダムなので不偏だが、ペア内でリーダー相性は相殺されない）。
- リーダー対は `seed*7919+13` から決定論的に引く（`decks.leader_pair`）。
- 勝率は**ペア水準**（0/0.5/1）で集計し 95% CI は正規近似（[`pair_level_ci`]）。
  2×pairs 局を独立 Bernoulli として扱う素朴 CI より狭い＝対照ペア設計の正しい区間。
- 成立しなかったペアは **void**（`score=None`）として台帳に残し、母数から外して**件数を必ず
  載せる**（黙って落とすと「全部測れた」ように見えてしまう）。
- 昇格は `wr >= frac` **かつ** ペア水準 95% CI 下限 > 0.50。
"""
import json
import math
import os
import signal
from typing import Any, Dict, List, Optional

from opcg_sim.loop import decks as D
from opcg_sim.loop import driver as DR
from opcg_sim.loop import engine as E

_G: Dict[str, Any] = {}


# --- 集計（pure）------------------------------------------------------------------

def elo_delta(win_rate: float) -> float:
    """勝率 → Elo 差（挑戦者 − ベースライン）。0.5→0・0.76→+200・0.24→−200。"""
    p = min(max(win_rate, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def pair_level_ci(pair_scores: List[float]) -> Dict[str, float]:
    """ペア単位スコア（{0,0.5,1}）の平均と 95% CI（正規近似）→ 勝率と Elo 区間。

    旧 `arena_parallel._pair_level_ci` と同一（判定規約の正本）。
    """
    n = len(pair_scores)
    mean = sum(pair_scores) / n
    var = sum((s - mean) ** 2 for s in pair_scores) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    return {"win_rate": mean, "lo": lo, "hi": hi,
            "elo": elo_delta(mean), "elo_lo": elo_delta(lo), "elo_hi": elo_delta(hi)}


def plan_bands(pairs: int, bands: int, seed_base: int, stride: int = 100000) -> List[List[int]]:
    """帯ごとに離れた seed 基点へ pairs を等分する（旧 `arena_gate.plan_bands`・pure）。"""
    per = pairs // bands
    rem = pairs - per * bands
    out = []
    for b in range(bands):
        n = per + (1 if b < rem else 0)
        base = seed_base + b * stride
        out.append(list(range(base, base + n)))
    return out


def final_decision(pair_scores: List[float], frac: float = 0.55):
    """本判定（pure）: 勝率 ≥ frac かつ**ペア水準 95% CI 下限 > 0.50**。"""
    ci = pair_level_ci(pair_scores)
    return bool(ci["win_rate"] >= frac and ci["lo"] > 0.50), ci


def stage1_decision(wins: float, games: int) -> str:
    """stage1（少局数の粗いふるい）: 勝ち越しなら 'continue'、五分以下なら 'reject'。"""
    return "continue" if wins * 2 > games else "reject"


def load_ledger(path: str) -> Dict[int, Optional[float]]:
    """台帳 jsonl → {seed: score}（`score=None` は void）。壊れた行は落とす＝黙って欠測にしない。"""
    done: Dict[int, Optional[float]] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sc = r.get("score")
                done[int(r["seed"])] = None if sc is None else float(sc)
    return done


def remaining_seeds(planned: List[int], done) -> List[int]:
    """計画 seed 列から消化済みを除いた残り（計画順を保つ・pure）。"""
    return [s for s in planned if s not in done]


def final_result(planned: List[int], done, frac: float = 0.55):
    """全ペア消化後の最終判定（旧 `arena_resume.final_result` と同規約・pure）。"""
    if any(s not in done for s in planned):
        return None
    valid = [s for s in planned if done[s] is not None]
    if not valid:
        return None
    ci = pair_level_ci([done[s] / 2.0 for s in valid])
    return {"pairs": len(valid), "games": 2 * len(valid),
            "void": len(planned) - len(valid),
            "wins": sum(done[s] for s in valid),
            "wr": round(ci["win_rate"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "elo": round(ci["elo"], 1),
            "promoted": bool(ci["win_rate"] >= frac and ci["lo"] > 0.50)}


# --- 対局（ワーカープロセス）--------------------------------------------------------

class PairTimeout(BaseException):
    """1 ペアの実時間上限。**BaseException 派生**が要点で、広い `except Exception` に
    食われて握り潰されないようにする（手数上限では「1 回の decide から戻らない」暴走を
    捕まえられない＝実時間で切るしかない・2026-08-16 の実害）。"""


def init_pool(cand_spec, best_spec, cand_kw=None, leaders_mode="fixed",
              decks="singleton", pair_timeout=0, sims=E.SERVE_SIMS):
    """子プロセス初期化: カード DB と席のつまみを 1 回だけ作る（以後の全ペアで共有）。

    `cand_kw` は**候補席にだけ**渡す探索のつまみ（`{"macro_moves": True}` 等）。機構を
    グローバルで切り替えると両席に効いて A/B にならないので、席別の seam を通す。
    **基準席は常に既定**（`sims` 以外は何も渡さない）＝比べているのは候補席の設定だけ。

    `cand_kw["sims"]`（`--cand-sims`）だけは `SeatSpec` の名前付き引数と衝突するので取り出す
    ＝候補席だけ sims を変えられる（基準席は `sims` のまま）。
    """
    E.engine()
    _G["leaders_mode"] = leaders_mode
    _G["decks"] = decks
    _G["pair_timeout"] = pair_timeout
    _G["db"] = D.load_db()
    kw = dict(cand_kw or {})
    _G["cand_opts"] = dict(kw)                 # 台帳に書く候補席の設定（既定なら空）
    cand_sims = int(kw.pop("sims", sims))
    _G["cand"] = E.SeatSpec(cand_spec or None, sims=cand_sims, **kw)
    _G["best"] = E.SeatSpec(best_spec or None, sims=sims)


def play_pair(seed: int) -> float:
    """1 ペア（同 seed で席を入替えた 2 局）の候補の勝ち数 0..2。void は None。"""
    return play_pair_detail(seed)["score"]


def play_pair_detail(seed: int) -> Dict[str, Any]:
    """`play_pair` の詳細版: 勝ち数に加えて**どの対面だったか**を返す（台帳行）。"""
    db, cand, best = _G["db"], _G["cand"], _G["best"]
    mode = _G.get("leaders_mode", "fixed")
    la, lb = D.leader_pair(db, seed, mode)
    decks = _G.get("decks", "singleton")
    if _G.get("pair_timeout"):
        def _alarm(_sig, _frm):
            raise PairTimeout(f"pair exceeded {_G['pair_timeout']}s")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(_G["pair_timeout"]))
    try:
        ab = D.build_pair(db, la, lb, seed, decks)          # game a: p1=cand=la / p2=best=lb
        ba = D.build_pair(db, lb, la, seed, decks)          # game b: p1=best=lb / p2=cand=la
        a = DR.run_game(seed, {"p1": cand, "p2": best}, *ab)
        b = DR.run_game(seed, {"p1": best, "p2": cand}, *ba)
        if a["winner"] is None or b["winner"] is None:
            raise DR.GameAborted(f"seed={seed} 決着せず（上限手数）")
    except (Exception, PairTimeout) as e:                    # noqa: BLE001  1 ペアで計測を止めない
        return _with_cand_opts({"seed": seed, "score": None, "leaders": [la, lb],
                                "void": f"{type(e).__name__}: {str(e)[:120]}"})
    finally:
        if _G.get("pair_timeout"):
            signal.alarm(0)
    wa = 1.0 if a["winner"] == "p1" else 0.0
    wb = 1.0 if b["winner"] == "p2" else 0.0
    return _with_cand_opts(
        {"seed": seed, "score": wa + wb, "leaders": [la, lb], "games": [wa, wb],
         "cand_leaders": [la, la], "turns": [a["turns"], b["turns"]]})


def _with_cand_opts(row: Dict[str, Any]) -> Dict[str, Any]:
    """台帳行へ候補席の設定を添える（**既定なら何も足さない**）。

    判定は設定に紐づく（同じネットでも探索のつまみが違えば別の測定）ので、行そのものに
    「どの設定で打ったか」を残す。既定（`--cand-*` を 1 つも付けない）実行では鍵ごと
    現れない＝**歴代の台帳と 1 バイトも変わらない**（既存の読み手を壊さない）。
    """
    opts = _G.get("cand_opts")
    if opts:
        row["cand_opts"] = dict(opts)
    return row


#: `--cand-*` フラグ → 席別の探索つまみ（旧 `arena_resume` の cand_kw と同じ意味）。
CAND_FLAGS = {
    "macro": dict(macro_moves=True, box_battle=True, quiesce=True),
    "defense_box": dict(defense_box=True),
    "dialog_box": dict(box_dialog=True),
    "boxes_all": dict(macro_moves=True, defense_box=True, box_dialog=True,
                      box_battle=True, quiesce=True),
    "box_commit": dict(box_commit=True),
    "no_box_commit": dict(box_commit=False),
    "residual_dig": dict(residual_dig=True),
}


def add_cand_args(ap) -> None:
    """候補席の seam フラグを argparse へ足す（`arena_shard`／`gate` が共有）。"""
    ap.add_argument("--cand-macro", action="store_true",
                    help="候補席だけマクロ手化 P1（配分箱＋戦闘の木内箱化・quiesce）")
    ap.add_argument("--cand-defense-box", action="store_true",
                    help="候補席だけ防御箱 v1（D1'/D2' 支配則の候補整形）")
    ap.add_argument("--cand-dialog-box", action="store_true",
                    help="候補席だけ対話箱（効果対話窓を出口 value 最良で畳む）")
    ap.add_argument("--cand-boxes-all", action="store_true",
                    help="候補席で箱化インフラ全部入り")
    ap.add_argument("--cand-box-commit", action="store_true",
                    help="候補席だけ箱コミット実行を明示 ON（既定 ON の上書き）")
    ap.add_argument("--cand-no-box-commit", action="store_true",
                    help="候補席だけ箱コミット実行を OFF（欠陥検出 A/B の OFF 側）")
    ap.add_argument("--cand-residual-dig", action="store_true",
                    help="候補席だけ残ドン掘り（腕 A・2026-09-02 の対照実験）")
    ap.add_argument("--cand-residual-activate", default=None, choices=("low", "high"),
                    help="候補席だけ残り起動（腕 A2）。付与対話を方針で解く")
    # --- 探索の設定そのもの（§20.5／§20.7・WP `rs-search-arena`）--------------------
    # ネットを固定して**設定だけ**を比べるための欄。`Game.decide` の opts と同名で、
    # 省略＝既定＝今までの挙動（opts に鍵が現れない）。基準席には一切渡らない。
    ap.add_argument("--cand-sims", type=int, default=None,
                    help="候補席だけ sims（省略＝`--sims`＝両席同じ）")
    ap.add_argument("--cand-select-rule", default=None, choices=("visits", "q_min_n"),
                    help="候補席だけ根の選択規則（§20.5・既定 visits）")
    ap.add_argument("--cand-q-min-frac", type=float, default=None,
                    help="候補席だけ q_min_n の訪問下限の割合（既定 0.125＝sims/8）")
    ap.add_argument("--cand-root-prior-temp", type=float, default=None,
                    help="候補席だけ根の事前分布の平坦化 P^(1/t)（§20.5・既定 1.0）")
    ap.add_argument("--cand-worlds", type=int, default=None,
                    help="候補席だけ世界サンプルの本数 K（§20.7.1・既定 1）。K 本ぶん"
                         "スレッドを使うので --workers を下げること")
    ap.add_argument("--cand-setup-box", action="store_true",
                    help="候補席だけ準備箱（§20.7.2・既定 OFF）")


#: `--cand-<flag>` → `Game.decide` の opts の欄（値をそのまま流す欄・省略＝既定）。
CAND_VALUE_FLAGS = {
    "cand_sims": "sims",
    "cand_select_rule": "select_rule",
    "cand_q_min_frac": "q_min_frac",
    "cand_root_prior_temp": "root_prior_temp",
    "cand_worlds": "worlds",
    "cand_residual_activate": "residual_activate",
}


def cand_kw_from_args(args) -> Optional[Dict[str, Any]]:
    """`add_cand_args` のフラグ → `init_pool(cand_kw=...)`。何も立っていなければ None。

    **None（省略）は載せない**＝既定の実行では `SeatSpec.opts` が今までと同じ
    `{"net", "sims"}` だけになる（記録も歴代の台帳と同じ形になる）。
    """
    kw: Dict[str, Any] = {}
    for flag, extra in CAND_FLAGS.items():
        if getattr(args, f"cand_{flag}", False):
            kw.update(extra)
    for attr, key in CAND_VALUE_FLAGS.items():
        value = getattr(args, attr, None)
        if value is not None:
            kw[key] = value
    if getattr(args, "cand_setup_box", False):
        kw["setup_box"] = True
    return kw or None
