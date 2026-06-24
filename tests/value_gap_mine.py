"""価値実現ギャップ採掘（人間ログ活用(b)・弱点採掘）（オフライン・dev専用・stdlib-only）。

ライブ採取（`api/app.py`＝終局済み traced 対局）の `value_samples`（ターン境界の両者視点 `{f,y}`・
producer 順＝boundary ごとに p1,p2）から、学習価値モデルの**予測勝率トラジェクトリ**を再構成し、
**value-realization gap** を抽出する:

  - comeback_depth = 1 − min(勝者の予測勝率)   …勝者が「最も負けに見えた」深さ（＝逆転の大きさ）
  - throwaway_peak  =     max(敗者の予測勝率)   …敗者が「最も勝ちに見えた」高さ（＝勝ちを捨てた場面）

gap が大きい局＝モデル/CPU の弱点現場（(A) データ補完の最優先候補・CPU 失着の診断点）。人間が CPU 相手に
逆転勝ちした局はここで上位に出る。自己対戦采取でも「現状の価値モデルが読み違える局面」を今すぐ surface できる。

推論は `cpu_value_model.predict_winprob`（同梱 `value_model.json`・推論の単一情報源）。読み取り専用。

実行例:
    python tests/value_gap_mine.py --in ./captures/ --top 10
    python tests/value_gap_mine.py --in game1.json game2.json --min-gap 0.3
"""
import argparse
import glob
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_value_model
from human_log_ingest import _find_value_samples, _valid_row


def realization_gap(value_samples: List[Dict[str, Any]],
                    predict: Optional[Callable[[List[float]], Optional[float]]] = None
                    ) -> Optional[Dict[str, Any]]:
    """1局の labeled value_samples（`{f,y}`・p1,p2 交互）から価値実現ギャップを算出（不能なら None）。

    勝者は y で判定（一方の視点だけ y=1）。予測勝率は注入可能（テスト用）＝既定は同梱モデル。
    """
    predict = predict or cpu_value_model.predict_winprob
    rows = [r for r in value_samples if _valid_row(r)]
    if len(rows) < 2:
        return None
    p1, p2 = rows[0::2], rows[1::2]          # boundary ごとに [p1, p2] で積まれる（turn_boundary_samples）
    n = min(len(p1), len(p2))
    if n < 1:
        return None
    p1, p2 = p1[:n], p2[:n]
    wp1 = [predict(r["f"]) for r in p1]
    wp2 = [predict(r["f"]) for r in p2]
    if any(w is None for w in wp1 + wp2):    # モデル未同梱/特徴長不一致＝採掘不能
        return None
    p1_won = p1[0]["y"] == 1
    win_traj, lose_traj = (wp1, wp2) if p1_won else (wp2, wp1)
    cb_turn = min(range(n), key=lambda i: win_traj[i])      # 勝者が最も負けに見えた境界
    tw_turn = max(range(n), key=lambda i: lose_traj[i])     # 敗者が最も勝ちに見えた境界
    comeback_depth = 1.0 - win_traj[cb_turn]
    throwaway_peak = lose_traj[tw_turn]
    return {
        "n_turns": n,
        "winner": "p1" if p1_won else "p2",                 # 採取順（manager.p1/p2）基準
        "comeback_depth": round(comeback_depth, 4),
        "comeback_turn": cb_turn,
        "throwaway_peak": round(throwaway_peak, 4),
        "throwaway_turn": tw_turn,
        "gap": round(max(comeback_depth, throwaway_peak), 4),
        "win_traj": [round(w, 3) for w in win_traj],
        "lose_traj": [round(w, 3) for w in lose_traj],
    }


def gap_from_dump(dump: Any, predict=None) -> Optional[Dict[str, Any]]:
    """采取エンベロープ（階層を問わず value_samples を内包）からギャップを算出。"""
    return realization_gap(_find_value_samples(dump), predict)


def _expand_inputs(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for p in inputs:
        paths.extend(sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p])
    return paths


def main(argv=None) -> int:
    import json
    ap = argparse.ArgumentParser(description="価値実現ギャップ採掘（人間ログ活用(b)）")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="采取 JSON/ディレクトリ（複数可）")
    ap.add_argument("--top", type=int, default=10, help="ギャップ上位 N 局を表示")
    ap.add_argument("--min-gap", type=float, default=0.0, help="この gap 未満は除外")
    args = ap.parse_args(argv)

    if cpu_value_model._load() is None:
        print("価値モデル未同梱（value_model.json なし）＝採掘不能"); return 1

    results = []
    for p in _expand_inputs(args.inputs):
        try:
            with open(p, "r", encoding="utf-8") as f:
                g = gap_from_dump(json.load(f))
        except (OSError, ValueError) as e:
            print(f"  skip {p}: {e}"); continue
        if g and g["gap"] >= args.min_gap:
            results.append((p, g))

    results.sort(key=lambda x: x[1]["gap"], reverse=True)
    print(f"=== value-realization gap 上位 {min(args.top, len(results))}/{len(results)} 局 ===")
    for p, g in results[:args.top]:
        print(f"  gap={g['gap']:.3f}  comeback_depth={g['comeback_depth']:.3f}@t{g['comeback_turn']} "
              f"throwaway_peak={g['throwaway_peak']:.3f}@t{g['throwaway_turn']}  "
              f"turns={g['n_turns']} winner={g['winner']}  {os.path.basename(p)}")
    if not results:
        print("  該当なし（終局済み traced 采取が無い/ギャップ閾値超えなし）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
