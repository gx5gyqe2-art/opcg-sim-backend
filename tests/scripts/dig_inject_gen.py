"""掘り裁定の注入コーパス生成（v49・2026-08-10・`verified_inject_gen` の兄弟＝ターン末盤面版）。

**裁定（ユーザ 2026-08-10）**: 序盤（turn 1〜先行2ターン目）、余るドンがあるなら 1コストの
登場時ドローキャラを出して山の勝ち筋（OP15-118 等）を掘りに行く。turn 1 の don!!-1 は
翌ターンのリーダー効果でドンデッキから再装填されるため実質無料。

**なぜ勝率CFでなく注入か（v49 実測 2026-08-10）**: h1@2 の「掘ってEND vs 無行動END」を
教師正本設定（CR.rollout sims48 def_temp0.7）×32世界CRNで測ると**両腕 0/32 勝**＝ラベルが
完全飽和し掘りの信号ゼロ。さらにライフ差タイブレークは無行動側が上（−2.03 vs −2.44）＝
ロールアウト方策自身が掘った札を活かせないため、**勝率教師は逆向きに教える**
（ブートストラップ問題）。v42（`verified_inject_gen`）と同じく裁定そのものを教師にする。

各採取点（リプレイの自ターン・turn≤turn-max の最初のメイン判断）で:
  1. 合法手から**掘り手**を列挙: コスト≤cost-max のキャラ PLAY で、適用すると効果対話が
     立つもの（登場時が発火する＝カードID のハードコード無し・汎用判定）
  2. 掘り腕: PLAY → 対話は**受ける側**（accepted!=False を優先・選択は先頭）で解決 → TURN_END
     ＝ don!!-X の支払いと引いた1枚まで含んだ**ターン末盤面**（value=+1）
  3. 対照腕: 直接 TURN_END ＝無行動のターン末盤面（value=−1）
  4. 1点=1群。スキーマは defcf/plancf/optpair と同一＝`option_pair_finetune` がそのまま読む
     （順位ヒンジ＋蒸留アンカー rank_finetune_anchored・v33）

**この教師の射程と犠牲**: ラベルは人間裁定（±1）であり実測勝率ではない。注入点そのものは
以後「独立した検査」にならない（v42 と同じ取引）＝効果確認は h1@2 の3層分解（value Δ と
decide の選択）とコーチゲート非注入点の不変・アリーナ中立で行う。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/dig_inject_gen.py \\
    --replays h1,e1,e2 --turn-max 4 --enc-version 9 --out /tmp/diginj
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
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
import rl_encoder as E  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402
from opcg_game import OPCGGame  # noqa: E402
from opcg_sim.src.core import cpu_ai  # noqa: E402


def _desc(m, mv):
    try:
        return cpu_ai._describe_move(m, mv) or {}
    except Exception:
        return {}


def _apply_dialogs(gs, m, name, cap=12):
    """立った効果対話を「受ける側」で解決する（accepted!=False 優先・選択は列挙順先頭）。

    注意（2026-08-10 実測）: `get_pending_request` は通常メインフェーズでも `MAIN_ACTION` を
    返すため「pending が無くなるまで」では終了しない。**合法手が効果対話（RESOLVE）で
    ある間だけ**回す＝メイン判断に戻ったらその盤面を返す。"""
    for _ in range(cap):
        legal = gs.legal_actions(m)
        resolves = [mv for mv in legal
                    if _desc(m, mv).get("action_type") == "RESOLVE_EFFECT_SELECTION"]
        if not resolves:
            return m                      # 効果対話は終了＝メイン判断へ戻った
        pick = next((mv for mv in resolves if _desc(m, mv).get("accepted") is not False),
                    resolves[0])
        m2 = gs.apply(m, pick, name)
        if m2 is None:
            return None
        m = m2
    return None                           # cap 超過＝想定外の長い対話（安全側で棄却）


def _end_turn(gs, m, name):
    for mv in gs.legal_actions(m):
        if _desc(m, mv).get("action_type") == "TURN_END":
            return gs.apply(m, mv, name)
    return None


def _first_main_points(rec, acts, fbi, turn_max):
    """自ターン（turn≤turn_max）ごとの最初のメイン判断 index を両席ぶん返す。"""
    pts, seen = [], set()
    for i, a in enumerate(acts):
        t = int(a.get("turn", 0) or 0)
        if t <= 0 or t > turn_max:
            continue
        fr = fbi.get(i - 1)
        active = (fr or {}).get("active")
        pid = a.get("player")
        if active is not None and active != pid:
            continue                      # 相手ターン中の応答（防御窓）はメイン判断でない
        key = (pid, t)
        if key in seen:
            continue
        seen.add(key)
        pts.append(i)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default="h1,e1,e2", help="タグ（coach_gate 表）or JSON パス")
    ap.add_argument("--turn-max", type=int, default=4)
    ap.add_argument("--cost-max", type=int, default=2, help="掘り手と見なす PLAY のコスト上限")
    ap.add_argument("--leader", default="OP15-058",
                    help="この card_id のリーダー席だけを採る（裁定の射程＝既定は紫エネル）。"
                         "空文字で全席（未裁定席へ注入する場合は射程外の教師と承知して使う）")
    ap.add_argument("--enc-version", type=int, default=9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="diginj_00000.npz")
    args = ap.parse_args()

    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    db = _load_db()
    gr = OPCGGame(prune_futile=False)     # 掘り手の列挙は無枝刈り
    gs = OPCGGame()                       # 適用は serve 同等
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    eng = LearnedEngine()                 # vocab（card_id 焼き込み）だけ使う

    rows = {k: [] for k in ("scalars", "field", "card_idx", "value", "group")}
    diag = []
    gid = 0
    for tag in [t.strip() for t in args.replays.split(",") if t.strip()]:
        raw = RE.load_replay_json(table.get(tag, tag))
        rec = raw.get("replay", raw)
        acts = rec["actions"]
        fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
        leaders = rec.get("leaders") or {}
        for i in _first_main_points(rec, acts, fbi, args.turn_max):
            if args.leader and leaders.get(acts[i].get("player")) != args.leader:
                continue                  # 裁定の射程外の席は採らない
            built = MG._restore(db, rec, fbi, acts, i)
            if isinstance(built, str) or built is None:
                continue
            m0, actor = built
            name = actor.name if hasattr(actor, "name") else actor
            if name != acts[i].get("player"):
                continue
            # 掘り手の候補カード（カードIDハードコード無し）: 低コストキャラPLAY。
            # **注意（2026-08-10 実測）**: 復元は uuid を毎回振り直すため、m0 で列挙した
            # PLAY（payload に手札 uuid を持つ）を別復元へ適用すると黙って失敗する。
            # 候補はカードIDだけ m0 から取り、適用は**同じ復元盤面上で列挙し直して**行う。
            me0 = m0.p1 if m0.p1.name == name else m0.p2
            cost_by_card = {}
            for c in me0.hand:
                if getattr(getattr(c.master, "type", None), "name", "") == "CHARACTER":
                    cid = getattr(c.master, "card_id", None)
                    cost_by_card[cid] = int(getattr(c.master, "cost", 99) or 99)
            cand_cards = sorted({_desc(m0, mv).get("card") for mv in gr.legal_actions(m0)
                                 if _desc(m0, mv).get("action_type") == "PLAY"
                                 and cost_by_card.get(_desc(m0, mv).get("card"), 99) <= args.cost_max})
            digs = []
            for card in cand_cards:
                mb = MG._restore(db, rec, fbi, acts, i)
                if isinstance(mb, str):
                    continue
                mt, _ = mb
                mv = next((v for v in gr.legal_actions(mt)
                           if _desc(mt, v).get("action_type") == "PLAY"
                           and _desc(mt, v).get("card") == card), None)
                if mv is None:
                    continue
                mp = gs.apply(mt, mv, name)
                if mp is None or not (mp.get_pending_request(with_request_id=False) or {}):
                    continue              # 対話が立たない＝登場時が発火しない＝掘りでない
                md = _apply_dialogs(gs, mp, name)
                if md is None:
                    continue
                me_ = _end_turn(gs, md, name)
                if me_ is None:
                    continue
                digs.append((card, me_))
            if not digs:
                continue
            mb = MG._restore(db, rec, fbi, acts, i)
            if isinstance(mb, str):
                continue
            mpass = _end_turn(gs, mb[0], name)
            if mpass is None:
                continue
            for card, mend in digs:
                enc = E.encode(mend, name, eng.vocab, version=args.enc_version)
                for k in ("scalars", "field", "card_idx"):
                    rows[k].append(enc[k])
                rows["value"].append(1.0)
                rows["group"].append(gid)
            encp = E.encode(mpass, name, eng.vocab, version=args.enc_version)
            for k in ("scalars", "field", "card_idx"):
                rows[k].append(encp[k])
            rows["value"].append(-1.0)
            rows["group"].append(gid)
            diag.append({"tag": tag, "i": i, "turn": int(acts[i].get("turn", 0) or 0),
                         "player": name, "digs": [c for c, _ in digs]})
            gid += 1

    os.makedirs(args.out, exist_ok=True)
    arrays = {"scalars": np.array(rows["scalars"], np.float32),
              "field": np.array(rows["field"], np.float32),
              "card_idx": np.array(rows["card_idx"], np.int64),
              "value": np.array(rows["value"], np.float32),
              "group": np.array(rows["group"], np.int64)}
    np.savez_compressed(os.path.join(args.out, args.shard), **arrays)
    with open(os.path.join(args.out, "meta_diginj.json"), "w") as f:
        json.dump({"source": "dig_inject", "groups": gid, "rows": len(rows["value"]),
                   "enc_version": args.enc_version, "turn_max": args.turn_max,
                   "cost_max": args.cost_max, "diag": diag}, f, ensure_ascii=False, indent=1)
    print(f"DIG_INJECT_DONE 群={gid} 行={len(rows['value'])} out={args.out}")
    for d in diag:
        print(f"  {d['tag']}@{d['i']} turn{d['turn']} {d['player']}: 掘り {d['digs']}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
