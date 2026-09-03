"""P1/P2 支配ペアのターン出口教師（系統3・π非依存・A段・2026-08-20）。

段3裁定の原則 P1「ドン付与はアタック前・レスト済みへの付与は無意味」を、審判ロールアウト
**無し**のルールラベルで教える反実仮想コーパス。自己対戦ラベル（V^π）は「今のCPUが続きを
打てない」バイアスを継承する（原因分析 2026-08-19 §2: 価値逆転はπに対する正しい較正）ため、
π非依存の支配ペアで**逆転そのもの**を直すのが A段の設計。

各対（同一決定点＝group）:
  V1 … 浮ドン k を「死に先」（レスト済み・ドン条件なし・相手ターン常在なし）へ付与 vs
       同じ k を「アクティブな攻撃可能ユニット」へ付与（他は同一: 全アタッカーで攻撃→閉幕）
  V4 … 同じ攻撃者に対し「付与→攻撃」（正順・P1）vs「攻撃→付与」（監査 #1/#2 の実挙動）。
       付与後アタックは相手により多くのカウンター値を要求する＝ユーザ裁定を権威とする。
       レスト後の付与が列挙上可能な**場キャラ攻撃者のみ**（リーダーはレスト後に付与不可）。

ラベルは順位のみ（good=+0.5 / bad=−0.5・`build_rank_pairs` の δ=0.25 を確実に超える）。
実現は `plan.scripted_plan`・出口は `plan.execute_plan`＝**serve のプラン読み出しと同一規約**
（train/serve skew の予防・v38 の学び）。防御側の応手（戦闘箱）は現行ネットで解決されるが、
ラベルは順位ルールなので π 汚染しない（両側が同一の防御に会う）。

出力: `plandom_*.npz`（plancf と同スキーマの必須キーのみ）。学習は
  exit_head_finetune.py --head turn --globs "plandom_*.npz" --base gen14 ...
がそのまま読む。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/plan_dom_gen.py \
    --games 24 --seed-base 700000 --workers 6 --out /tmp/plandom
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import re

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import plan as PL

MAX_MAIN_PER_TURN = 6     # 1ターンに対を作る自席メイン判断の上限。レスト済みの自キャラ（V1 の
                          # 死に先）は攻撃後＝ターン後半にしか現れないため、先頭だけの採取だと
                          # V1 が枯れる（スモーク実測: 上限3で V1:V4 = 1:16）
K_ATTACH = 2              # 対で動かす付与ドンの枚数上限（1〜K）


# 【ドン!!×N】の閾値 N を取り出す（cpu_ai._DON_COND_RE のキャプチャ版・表記揺れ耐性は同一）
_DON_COND_N_RE = re.compile(r'【\s*ドン\s*(?:!!|！！|‼)\s*[××xX]\s*(\d+)\s*】')


def _effect_text(c):
    return getattr(getattr(c, "master", None), "effect_text", "") or ""


def _don_cond_max_n(c):
    """カードの【ドン!!×N】閾値の最大値（無ければ None・pure）。"""
    ns = [int(m.group(1)) for m in _DON_COND_N_RE.finditer(_effect_text(c))]
    return max(ns) if ns else None


def _attached_count(c):
    ad = getattr(c, "attached_don", None)
    return ad if isinstance(ad, int) else len(ad or [])


def dead_targets(player):
    """「死に付与先」の列挙（pure）。返り値 [(card, tag)]・tag は V1 の亜種名。

    - V1s（閾値達成済み・V1'・2026-08-20）: 【ドン!!×N】持ちでも **attached ≥ N なら追加付与は
      死に**（条件は既に開いており、レストのキャラへの+1000は自ターンの攻撃に使えない）。
      段3裁定 #1/#2 のドレーク（×1・レスト時常在）の2枚目以降がこの型＝原因分析§4で特定した
      (B)フィルタの穴と同形。「レストの場合」常在の文言も閾値達成後は追加付与を正当化しない。
    - V1（素の死に）: ドン条件なし・レスト/相手ターン常在の文言なしのレスト済みキャラ。
    どちらも「相手のターン」常在（チョッパー型＝守備的な付与に意味）を持つカードは除外。"""
    out = []
    for c in (getattr(player, "field", None) or []):
        if not getattr(c, "is_rest", False):
            continue
        t = _effect_text(c)
        if "相手のターン" in t:
            continue
        n = _don_cond_max_n(c)
        if n is not None:
            if _attached_count(c) >= n:
                out.append((c, "V1s"))
            continue                          # 閾値未達＝1枚目は条件を開く正当な付与
        if "レストの場合" in t:
            continue
        out.append((c, "V1"))
    # 閾値達成済み（V1s）を優先＝逆転の実測が濃い型（#1/#2）から教える
    out.sort(key=lambda ct: 0 if ct[1] == "V1s" else 1)
    return out


def active_attackers(player):
    """今アクティブで攻撃できる体（リーダー含む・召喚酔い除外・pure）。"""
    out = []
    lead = getattr(player, "leader", None)
    if lead is not None and not getattr(lead, "is_rest", False):
        out.append(lead)
    for c in (getattr(player, "field", None) or []):
        if getattr(c, "is_rest", False):
            continue
        if getattr(c, "is_newly_played", False) and not c.has_keyword("速攻"):
            continue
        out.append(c)
    return out


def dominance_pairs(manager, name):
    """この決定点で作れる支配ペア [(tag, intent_good, intent_bad)]（pure）。

    intent は plan.scripted_plan の抽象方針（("ATTACH"|"ATTACK", uuid) の列）。V1/V4 とも
    「両者の違いはドンの置き場所/順序だけ・攻撃対象と打ち納めは同一」になるよう組む。"""
    p = manager.p1 if getattr(manager.p1, "name", None) == name else manager.p2
    spare = len(getattr(p, "don_active", []) or [])
    if spare <= 0:
        return []
    deads = dead_targets(p)
    actives = active_attackers(p)
    if not actives:
        return []
    attacks = [("ATTACK", a.uuid) for a in actives]
    out = []
    k = min(spare, K_ATTACH)
    if deads:
        (d, dtag), a = deads[0], actives[0]
        out.append((f"{dtag}k{k}",
                    [("ATTACH", a.uuid)] * k + attacks,
                    [("ATTACH", d.uuid)] * k + attacks))
    # V4: レスト後も付与が列挙される場キャラ攻撃者のみ（リーダーはレスト後に付与不可）
    chars = [a for a in actives if a is not getattr(p, "leader", None)]
    if chars:
        a = chars[0]
        rest = [("ATTACK", x.uuid) for x in actives if x.uuid != a.uuid]
        out.append((f"V4k{k}",
                    [("ATTACH", a.uuid)] * k + [("ATTACK", a.uuid)] + rest,
                    [("ATTACK", a.uuid)] + [("ATTACH", a.uuid)] * k + rest))
    return out


class _Done(BaseException):
    pass


class _MainCap:
    """自席メイン判断（pending が MAIN_ACTION）の直前 manager を、ターンごとに上限つきで複製。"""

    def __init__(self, limit_decisions):
        self.limit = limit_decisions
        self.n = 0
        self.frames = []      # (decision_no, turn, seat, manager)
        self._per_turn = {}
        self._keys = cpu_ai._pending_keys()

    def on_decision_point(self, ctx):
        m = ctx.manager
        name = getattr(ctx.actor, "name", None)
        _kp, k_action = self._keys
        if (ctx.pending or {}).get(k_action) != "MAIN_ACTION":
            return
        if getattr(getattr(m, "turn_player", None), "name", None) != name:
            return
        turn = int(getattr(m, "turn_count", 0) or 0)
        key = (name, turn)
        if self._per_turn.get(key, 0) >= MAX_MAIN_PER_TURN:
            return
        self._per_turn[key] = self._per_turn.get(key, 0) + 1
        self.frames.append((self.n + 1, turn, name, m.clone()))

    def on_decision(self, ctx, move):
        self.n += 1
        if self.n > self.limit:
            raise _Done()


_G = {}


def _init(sims, enc_version):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    eng = CL.LearnedEngine(sims=sims,
                           # 教師生成は**原始手の全空間**が対象（レフェリー/計器と同じ原則・
                           # 2026-08-25 既定 ON 化に伴う明示化）: 箱化された候補では
                           # V1/V4/リーサル対の原始手探索が成立しない（実測 0 対）。
                           macro_moves=False, defense_box=False, box_dialog=False)
    _G["eng"] = eng
    _G["vf"] = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    _G["pf"] = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    _G["enc_version"] = enc_version or eng.enc_version


def _exit_enc(frame, name, sigs):
    """sig 列を execute_plan（serve と同一規約）で出口まで実行し符号化して返す。"""
    from opcg_sim.src.learned import encoder as E
    eng = _G["eng"]
    exit_mgr = PL.execute_plan(eng.game, frame.clone(), name, list(sigs),
                               _G["vf"], _G["pf"], battle_value_fn=None)
    return E.encode(exit_mgr, name, eng.vocab, version=_G["enc_version"])


def _run_game(job):
    seed, sims = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    db, eng = _G["db"], _G["eng"]
    la, lb = _leader_pair(db, seed, "random")
    cap = _MainCap(limit_decisions=200)
    seat = make_seat(kind="learned", want_trace=False, sims=sims, engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=200)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": []}

    rows = []      # (enc, z, group_local, tag)
    gi = 0
    for dec, turn, name, frame in cap.frames:
        for tag, good, bad in dominance_pairs(frame, name):
            try:
                sg = PL.scripted_plan(eng.game, frame.clone(), name, good, _G["vf"], _G["pf"])
                sb = PL.scripted_plan(eng.game, frame.clone(), name, bad, _G["vf"], _G["pf"])
                if not sg or not sb or sg == sb:
                    continue          # 実現で潰れた/同一化した対は捨てる（縮退＝支配が立たない）
                eg = _exit_enc(frame, name, sg)
                eb = _exit_enc(frame, name, sb)
            except BaseException:
                continue              # 生成の失敗で対局全体を落とさない
            if np.array_equal(eg["scalars"], eb["scalars"]) and \
                    np.array_equal(eg["field"], eb["field"]):
                continue              # 出口が同一＝順位を教えられない
            rows.append((eg, +0.5, gi, tag))
            rows.append((eb, -0.5, gi, tag))
            gi += 1
    return {"seed": seed, "error": None, "rows": rows, "frames": len(cap.frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=700000)
    ap.add_argument("--sims", type=int, default=160, help="局面採取の自己対戦（分布=本番仕様）")
    ap.add_argument("--enc-version", type=int, default=0, help="0=エンジンの符号化世代に従う")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard-size", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.seed_base + i, args.sims) for i in range(args.games)]
    buf, stats = [], {"pairs": 0, "V1": 0, "V1s": 0, "V4": 0, "errors": 0}
    shard = [0]

    def _flush(chunk):
        if not chunk:
            return
        arrays = {
            "scalars": np.stack([r[0]["scalars"] for r in chunk]).astype(np.float32),
            "field": np.stack([r[0]["field"] for r in chunk]).astype(np.float32),
            "card_idx": np.stack([r[0]["card_idx"] for r in chunk]).astype(np.int32),
            "value": np.array([r[1] for r in chunk], dtype=np.float32),
            "group": np.array([r[2] for r in chunk], dtype=np.int64),
        }
        np.savez_compressed(os.path.join(args.out, f"plandom_{shard[0]:05d}.npz"), **arrays)
        shard[0] += 1

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.sims, args.enc_version)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                stats["errors"] += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            # group はシャード内 local → seed 込みでグローバル化（マージ衝突の予防・plancf と同じ）
            gbase = res["seed"] * 1000
            for enc, z, g, tag in res["rows"]:
                buf.append((enc, z, gbase + g, tag))
                kind = tag.split("k")[0]
                stats[kind] = stats.get(kind, 0) + 0.5   # 対で1（行で0.5）
            stats["pairs"] += len(res["rows"]) // 2
            print(f"  seed {res['seed']}: 判断点{res.get('frames', 0)}・対 {len(res['rows']) // 2}",
                  flush=True)
            # 対（2行）がシャード境界で割れないよう偶数長で切る（group 単位の分割を保つ）
            while len(buf) >= args.shard_size:
                _flush(buf[:args.shard_size])
                buf = buf[args.shard_size:]
    _flush(buf)
    print(f"PLAN_DOM_DONE pairs={stats['pairs']} V1={int(stats.get('V1', 0))} "
          f"V1s={int(stats.get('V1s', 0))} V4={int(stats.get('V4', 0))} "
          f"errors={stats['errors']} shards={shard[0]} -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
