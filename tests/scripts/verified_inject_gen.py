"""裁定済み点の注入コーパス生成（v42・2026-08-07・ユーザ判断 (A)）。

`coach_gate.VERIFIED_V2` の各点について、root の合法手を**箱の規約で出口盤面まで解決**し、
裁定済み accept 集合に入る手の出口を勝ち（value=+1）・それ以外を負け（value=−1）として
1点＝1群の順位ペアコーパスを吐く。スキーマは defcf/plancf と同一なので
`exit_head_finetune.py` にそのまま流せる。

**なぜ注入なのか**（v41 実測 2026-08-07・`docs/reports/cpu_v41_battle_exit_head_20260807.md`）:
防御CFコーパス（584群775ペア）から学習した戦闘出口ヘッドは、gen12 の枝間マージン
0.02〜0.03 に対し標準偏差 0.23〜0.27 の摂動しか作れず、コーチゲート合計は 6.6 のまま
中身だけ入れ替わった（m1@14 獲得・m2@44 退行）。因果 z の粒度（worlds=4 で 0.5）が
マージンを覆い隠すため、**ロールアウト由来のラベルではこの解像度に届かない**。
裁定はレフェリー実測＋人間の最終判断なので、順位そのものは信頼できる最も鋭い信号になる。

**この教師を使うと何が犠牲になるか（重要・必ず読むこと）**: コーチゲート8点は
**この生成器の入力そのもの**なので、注入後のゲートは**独立した検査ではなくなる**
（自分の訓練データを測る）。ゲートは「注入が効いたか」の確認にしか使えず、
**採否の一次証拠はアリーナ（自己対戦勝率）へ移る**。この取引を承知したうえで
ユーザが選んだ道（2026-08-07）。ゲートの数字を「強くなった証拠」として引用しないこと。

**`--battle-only`（既定 ON・ヘッドの管轄と一致させる）**: 戦闘出口ヘッドが serve で評価するのは
**戦闘箱の出口盤面だけ**（`_battle_window_choice` / `TreeMCTS._expand` の箱化 /
`resolve_turn_inplace` の戦闘窓 / `plan._battle_box_step` はいずれも `in_battle` 下でしか
呼ばれない）。メインフェーズの root（m1@3/m2@44/m4@2/m5@7 等）の子盤面は戦闘を含まないので、
そのままでは**ヘッドが一度も見ない盤面**であり、そこへラベルを付けるのは管轄外の教師になる
（汎化を通じて本来の戦闘出口の較正を歪めるリスクだけが残る）。既定では戦闘窓の点のみを採る。
`--no-battle-only` で全点を採れるが、その場合は「近似の教師を混ぜている」ことを承知して使う。

制約:
  - accept が dict（`turn_all`＝ターン内全消化の系列基準・m2@66）の点は**除外**する。
    単一の root 手では合否が決まらず、勝ち/負けのラベルを付けられないため。
  - accept 手が1つも合法でない点、逆に全手が accept の点も除外（ペアが作れない）。
  - 出口解決は**土台ネットの物差し**（本体 value・`BOX_RESOLVE_DEPTH`）で行う。学習前は
    戦闘出口ヘッドが存在しないため、これが serve と一致する唯一の規約（defcf と同じ扱い）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/verified_inject_gen.py \\
    --out /tmp/vinj --repeat 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
import rl_encoder as E  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn, _priors_fn  # noqa: E402
from opcg_sim.src.learned.config import BOX_RESOLVE_DEPTH  # noqa: E402
from opcg_sim.src.learned.mcts import in_battle, resolve_battle_inplace  # noqa: E402

# 群IDの基点。実コーパス（seed_base 由来）と衝突しない十分大きな値を取る
# （併合して読むため。衝突すると別々の決定点が1群に混ざって順位が壊れる）。
GROUP_BASE = 900_000


def build_rows(eng, enc_version=8, repeat=1, box_depth=None, battle_only=True):
    """VERIFIED_V2 を走査して行辞書と診断リストを返す。"""
    if box_depth is None:
        box_depth = BOX_RESOLVE_DEPTH
    CR.ARGS = argparse.Namespace(true_board=True)
    db = _load_db()
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    vf = _value_fn(eng.vnet, eng.vocab, eng.enc_version)
    pf = _priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "group",
                            "q_root", "turns_left")}
    diag = []
    gi = 0
    for tag, idx, accept in CG.VERIFIED_V2:
        name_pt = f"{tag}@{idx}"
        if isinstance(accept, dict):
            diag.append({"point": name_pt, "skipped": "turn_all（系列基準は単一手で採点できない）"})
            continue
        if tag not in CR.GAMES:
            raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
            CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                             rec["actions"])
        built = CR._restore_board(db, tag, idx)
        if isinstance(built, str):
            diag.append({"point": name_pt, "skipped": f"復元不可: {built}"})
            continue
        m0, who = built
        name = who if isinstance(who, str) else who.name
        is_battle = in_battle(m0)
        if battle_only and not is_battle:
            diag.append({"point": name_pt, "skipped":
                         "戦闘窓でない（メインフェーズの子盤面は戦闘出口ヘッドの管轄外）"})
            continue
        legal = eng.game.legal_actions(m0)
        exits, labels, descs = [], [], []
        for mv in legal:
            child = eng.game.apply(m0, mv, name)
            if child is None:
                continue
            try:
                # 出口まで解決してから符号化する（defcf と同一規約＝train/serve skew の解消）。
                resolve_battle_inplace(eng.game, child, pf, value_fn=vf, box_depth=box_depth)
            except Exception:
                continue
            d = cpu_ai._describe_move(m0, mv) or {}
            exits.append(child)
            labels.append(1.0 if CG.hit(d, accept) else -1.0)
            descs.append((d.get("action_type"), d.get("card")))
        n_pos = int(sum(1 for y in labels if y > 0))
        if not exits or n_pos == 0 or n_pos == len(labels):
            diag.append({"point": name_pt, "skipped":
                         f"ペア不成立（合法{len(exits)}・accept{n_pos}）"})
            continue
        for r in range(repeat):
            g = GROUP_BASE + gi
            gi += 1
            for child, y in zip(exits, labels):
                enc = E.encode(child, name, eng.vocab, version=enc_version)
                rows["scalars"].append(enc["scalars"])
                rows["field"].append(enc["field"])
                rows["card_idx"].append(enc["card_idx"])
                rows["value"].append(y)
                rows["group"].append(g)
                rows["q_root"].append(np.nan)      # 勝敗単独ラベル（エコー遮断・defcf と同じ）
                rows["turns_left"].append(np.nan)  # 注入点には残りターンの実測が無い
        diag.append({"point": name_pt, "branches": len(exits), "accept": n_pos,
                     "groups": repeat, "battle": is_battle, "moves": descs})
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="シャードを書くディレクトリ")
    ap.add_argument("--base", default=None, help="出口解決に使う土台（既定＝出荷既定 gen12）")
    ap.add_argument("--enc-version", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1,
                    help="1点を何群に複製するか（順位ペアの重み。v40 は 4 を使った）")
    ap.add_argument("--shard", default="vinj_00000.npz")
    ap.add_argument("--battle-only", dest="battle_only", action="store_true", default=True,
                    help="戦闘窓の点だけを採る（既定・ヘッドの管轄と一致させる）")
    ap.add_argument("--no-battle-only", dest="battle_only", action="store_false",
                    help="メインフェーズの点も採る（近似の教師を混ぜることを承知で使う）")
    args = ap.parse_args()

    if args.base:
        parts = args.base.split(",")
        eng = LearnedEngine(value_path=parts[0],
                            policy_path=parts[1] if len(parts) > 1 else None)
    else:
        eng = LearnedEngine()

    rows, diag = build_rows(eng, enc_version=args.enc_version, repeat=args.repeat,
                            battle_only=args.battle_only)
    for d in diag:
        print(f"  {d.get('point'):<8} " +
              (f"skip: {d['skipped']}" if "skipped" in d
               else f"枝{d['branches']}（accept {d['accept']}）× {d['groups']}群"
                    f"{'  ←戦闘窓' if d.get('battle') else '  ←メイン（近似）'}"), flush=True)
    if not rows["value"]:
        print("注入コーパスが空（対象点がすべて除外された）")
        return 1

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.shard)
    np.savez(path,
             scalars=np.asarray(rows["scalars"], dtype=np.float32),
             field=np.asarray(rows["field"], dtype=np.float32),
             card_idx=np.asarray(rows["card_idx"], dtype=np.int64),
             value=np.asarray(rows["value"], dtype=np.float32),
             group=np.asarray(rows["group"], dtype=np.int64),
             q_root=np.asarray(rows["q_root"], dtype=np.float32),
             turns_left=np.asarray(rows["turns_left"], dtype=np.float32))
    res = {"shard": path, "boards": len(rows["value"]),
           "groups": int(len(set(rows["group"]))), "repeat": args.repeat,
           "battle_only": args.battle_only,
           "points_used": sum(1 for d in diag if "skipped" not in d),
           "points_skipped": sum(1 for d in diag if "skipped" in d)}
    print(f"VERIFIED_INJECT_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
