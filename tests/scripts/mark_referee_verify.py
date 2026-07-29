"""人間マークのレフェリー裏取り（v18・コーチゲート VERIFIED v2 の採録材料を作る）。

`tests/fixtures/replays/gen7_marks_20260728/` の実対局マーク（gen7 相手・5局34件）を
1点ずつ真盤面再生（`state_at_action`）し、レフェリー（**gen5 固定錨**・プラン自動列挙＋
同価値バンド CRN 判定＋捲りエスカレーション＝`referee_labeler`/`divergence_probe` と同一機構）で
裁く。出力は各点の**バンド上位プランの初手集合（accept 集合）**と、CPU が選んだ手・人間ノートの
突き合わせ。旧 VERIFIED（g3=単一対局・gen4期・7点）を置き換える v2 採録の一次資料。

裁定の3値:
  - cpu_out_band: CPU の手がバンド外＝マークをレフェリーが支持（VERIFIED v2 候補）
  - cpu_in_band : CPU の手もバンド内＝レフェリーは同価値と判定（人間ノートと不一致＝要目視）
  - no_plans    : プラン列挙が1本以下/再生不能/対象外ウィンドウ＝裏取り不能（診断のみ）

結果は `--out` へ JSONL 追記（1点1行・途中終了しても既存分は残る）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mark_referee_verify.py \
    --worlds 8 --out /tmp/mark_verify.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import gzip
import json
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import counterfactual_referee as CR
import replay_runner as RR
import p3_loop as P
import rl_net as RN
import rl_encoder as E
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from az_policy import PolicyScorer
from opcg_sim.src.core.cpu_learned import _net_enc_version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXDIR = os.path.join(REPO, "tests", "fixtures", "replays", "gen7_marks_20260728")
_WINDOWS = ("MAIN_ACTION", "SELECT_COUNTER", "SELECT_BLOCKER")
ARGS = None


def load_games(fixdir):
    """fixture ディレクトリ → [(marks_dict, replay_record)]（seed で対応付け）。"""
    out = []
    for mf in sorted(glob.glob(os.path.join(fixdir, "*marks*.json"))):
        md = json.load(open(mf))
        rf = os.path.join(fixdir, f"opcg_replay_{md['seed']}.json.gz")
        rec = json.load(gzip.open(rf)) if os.path.exists(rf) else \
            json.load(open(rf[:-3]))
        out.append((md, rec))
    return out


def first_move_desc(plans, key0_repr):
    """entries の初手 equiv キー（repr）→ 人間可読の初手記述子（plans の descs から引く）。"""
    for keys, descs in plans:
        if repr(keys[0]) == key0_repr and descs:
            d = descs[0] or {}
            return {"action_type": d.get("action_type"), "card": d.get("card")}
    return None


def hit(desc, accept):
    """coach_gate.hit と同じ判定: (action_type, card) が accept に入るか（card=None は種別のみ）。"""
    at, card = desc.get("action_type"), desc.get("card")
    return (at, card) in accept or (at, None) in accept


def adjudicate(game_root, game_serve, vf, pf, m0, name, log=print):
    """1点の裁定: プラン列挙＋CRN 評価＋捲りエスカレーション → (accept記述子リスト, entries要約, plans)。
    `divergence_probe.band_top_keys` と同一機構（そちらは repr キー、こちらは採録用に記述子も返す）。"""
    plans = CR.enumerate_turn_plans(game_root, vf, m0, name, max_len=ARGS.plan_len,
                                    beam=ARGS.beam, max_plans=ARGS.max_plans,
                                    log=lambda *a, **k: None)
    if len(plans) <= 1:
        return None, None, None
    entries = [{"label": "", "keys": keys} for keys, _descs in plans]
    CR._eval_entries(entries, game_root, game_serve, vf, pf, m0, name, ARGS.worlds)
    entries.sort(key=lambda e: (-e["wins"], -e["lifem"]))
    escalated = False
    if ARGS.comeback > 0 and entries[0]["wins"] <= 1:
        sub = entries[:min(6, len(entries))]
        CR._eval_entries(sub, game_root, game_serve, vf, pf, m0, name, ARGS.worlds * 4,
                         opp_temp=ARGS.comeback)
        sub.sort(key=lambda e: (-e["wins"], -e["lifem"]))
        entries = sub
        escalated = True
    best = entries[0]
    accept, rows = [], []
    for e in entries:
        in_band = e is best or CR.same_value(best, e, ARGS.band)
        d = first_move_desc(plans, repr(e["keys"][0]))
        rows.append({"first": d, "wins": e["wins"], "lifem": round(e.get("lifem", 0), 2),
                     "in_band": bool(in_band)})
        if in_band and d is not None:
            t = (d["action_type"], d["card"])
            if t not in accept:
                accept.append(t)
    return accept, {"escalated": escalated, "entries": rows}, plans


_G = {}


def _init_worker(args):
    """子プロセス初期化: DB・gen5錨ネット・fixture 群を1回だけロード（以後の全点で共有）。"""
    global ARGS
    ARGS = args
    CR.ARGS = args
    db = _load_db()
    vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
    pnet = PolicyScorer.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_policy.npz"))
    ev = _net_enc_version(vnet)
    vocab = E.vocab_from_ids(vnet.vocab_ids) if vnet.vocab_ids else E.build_vocab(db)
    _G.update(db=db, vf=P.value_fn_of(vnet, vocab, ev), pf=P.priors_fn_of(pnet, vocab, ev),
              game_root=OPCGGame(prune_futile=False), game_serve=OPCGGame(),
              games={md["seed"]: (md, rec) for md, rec in load_games(args.fixdir)})


def _verify_point(payload):
    """1点の裁定（子プロセス側）。返り値は JSONL 1行ぶんの dict。"""
    seed, idx = payload
    md, rec = _G["games"][seed]
    m = next(mm for mm in md["marks"] if mm["action_index"] == idx)
    t0 = time.time()
    row = {"seed": seed, "action_index": idx, "turn": m.get("turn"),
           "note": m.get("note"), "leaders": md["leaders"],
           "chosen": (m.get("decision") or {}).get("chosen") or m.get("action")}
    m0, who = RR.state_at_action(_G["db"], rec["replay"], idx, frames=rec.get("frames"))
    rec_turn = (rec["replay"]["actions"][idx] or {}).get("turn")
    if m0 is None:
        row.update({"verdict": "no_plans", "reason": "restore_fail"})
    elif rec_turn is not None and m0.turn_count != rec_turn:
        # 早停止/分岐の防壁: 復元盤面のターンが記録と食い違う＝別局面を黙って裁かない
        row.update({"verdict": "no_plans",
                    "reason": f"turn_mismatch:{m0.turn_count}!={rec_turn}"})
    else:
        pend = (m0.get_pending_request(with_request_id=False) or {})
        window = pend.get("action") or "MAIN_ACTION"
        row["window"] = window
        if window not in _WINDOWS:
            row.update({"verdict": "no_plans", "reason": f"window:{window}"})
        else:
            accept, summary, _plans = adjudicate(_G["game_root"], _G["game_serve"],
                                                 _G["vf"], _G["pf"], m0, who)
            if accept is None:
                row.update({"verdict": "no_plans", "reason": "plans<=1"})
            else:
                chosen = {"action_type": row["chosen"].get("action_type"),
                          "card": row["chosen"].get("card")}
                row.update({"verdict": ("cpu_in_band" if hit(chosen, accept)
                                        else "cpu_out_band"),
                            "accept": [list(a) for a in accept], **summary})
    row["sec"] = int(time.time() - t0)
    return row


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixdir", default=FIXDIR)
    ap.add_argument("--sims", type=int, default=32, help="レフェリーのロールアウト sims")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--comeback", type=float, default=0.7)
    ap.add_argument("--plan-len", type=int, default=4)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--max-plans", type=int, default=12)
    ap.add_argument("--only", default=None, help="seed末尾6桁:action_index で1点に絞る（smoke用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="JSONL 追記先")
    ARGS = ap.parse_args()
    CR.ARGS = ARGS

    done = set()
    if ARGS.out and os.path.exists(ARGS.out):
        for line in open(ARGS.out):
            r = json.loads(line)
            done.add((r["seed"], r["action_index"]))     # 再開: 裁定済みはスキップ
    points = []
    for md, _rec in load_games(ARGS.fixdir):
        for m in md["marks"]:
            key = (md["seed"], m["action_index"])
            if key in done:
                continue
            if ARGS.only and f"{md['seed'][-6:]}:{m['action_index']}" != ARGS.only:
                continue
            points.append(key)
    print(f"裁定対象: {len(points)}点（済み {len(done)}）", flush=True)

    t_all = time.time()
    n_out = n_in = n_skip = 0
    with mp.Pool(ARGS.workers, initializer=_init_worker, initargs=(ARGS,)) as pool:
        for row in pool.imap_unordered(_verify_point, points):
            v = row.get("verdict")
            n_out += v == "cpu_out_band"; n_in += v == "cpu_in_band"; n_skip += v == "no_plans"
            print(f"…{row['seed'][-6:]} @{row['action_index']} [{row.get('window', '-')}] → {v}"
                  f"{'（' + row.get('reason', '') + '）' if v == 'no_plans' else ''}"
                  f" {row['sec']}s  note={str(row.get('note'))[:40]}", flush=True)
            if ARGS.out:
                with open(ARGS.out, "a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    res = {"cpu_out_band": n_out, "cpu_in_band": n_in, "no_plans": n_skip,
           "sec": int(time.time() - t_all)}
    print(f"MARK_VERIFY_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
