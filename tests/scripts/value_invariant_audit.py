"""価値ネットの**不変量監査**（オラクル不要の破れ探し・2026-08-15 ユーザ提案）。

なぜ要るか: 我々の検査は全て「誰かの判断」を基準にしている——ゲート＝人間の裁定、
アリーナ＝過去世代との対戦、ns2＝レフェリーのロールアウト。どれも**うちの系統が揃って
見落としている問題**には光が当たらない（gen13→14→15 は温スタートで血が繋がり、L1 も同じ
エンジン・同じ4デッキの上にいる＝盲点が相関する。実例: G14 のエネル盲点は訓練系譜に
エネル対面の多様性が無かったことが原因で、系統内の比較では見つからなかった）。

本器は**対戦相手も正解ラベルも使わない**。価値関数が満たすべき性質を主張し、破れを数える:

  1. **零和対称性**: 同じ盤面を自分視点と相手視点で評価したら符号が反転するはず。
     注意: 符号化は非対称（自分の手札は中身が見え相手は枚数のみ＝公平性契約）なので
     厳密な反転は期待できない。見るのは **平均バイアス**（`v(s,me)+v(s,opp)` の平均が0から
     ずれる＝席/手番バイアス。v47b のエネル席バイアスと同型の欠陥）と**順位相関**。
  2. **支配単調性**: 他が全く同じでこちらのライフ/手札/アクティブドン/パワーが増えた盤面は
     評価が下がってはいけない（逆に相手側が増えたら上がってはいけない）。破れ＝
     「資源が増えると悪く見える」という評価の破綻で、実害は温存/展開の判断に出る。

破れは**そのまま欠陥のカタログ**になる（誰の判断とも突き合わせていないので、
「今まで誰も気づいていない問題」を名指しできる唯一の道具）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_invariant_audit.py \\
    --boards 200 --out /tmp/invariant_audit.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
import replay_runner as RR  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402


def sample_boards(db, limit, stride=7):
    """リプレイ群から盤面を採取（tag, i, manager, me_name）。復元不能はスキップ。"""
    table = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    out = []
    for tag, path in sorted(table.items()):
        try:
            raw = RE.load_replay_json(path)
        except Exception:
            continue
        rec = raw.get("replay", raw)
        n = len(rec.get("actions") or [])
        for i in range(2, n, stride):
            try:
                built = RR.state_at_action(db, rec, i)
            except Exception:
                continue
            if not built:
                continue
            m, who = built
            name = who if isinstance(who, str) else getattr(who, "name", None)
            if name is None:
                continue
            out.append((tag, i, m, name))
            if len(out) >= limit:
                return out
    return out


def _players(m, name):
    me = m.p1 if m.p1.name == name else m.p2
    opp = m.p2 if m.p1.name == name else m.p1
    return me, opp


# --- 支配的な改変（**厳密に「他が全く同じ」**にする） -------------------------
# 重要（2026-08-15 の設計修正）: 初版は山札/ドンデッキから資源を移していたが、それは
# **デッキ残の減少という正当なコストを同時に課す**ので支配関係になっていない（ドン/パワーの
# 破れが「ドンデッキを消費したから下がった」＝正しい評価かもしれず、欠陥と区別できない）。
# ここでは資源を**無から足す**（コピー/新規生成）＝他の量は1ビットも動かさない。
# 代償として盤面はゲーム的にはやや非現実（総ドン11 等）になりうるが、評価関数への
# 主張「資源が増えて悪くなってはいけない」は成立する。分布外の可能性は解釈時の注意点として
# 記録する（破れの大きさが枝間マージン 0.02〜0.03 と同規模なら実害あり）。

def mut_life(m, name):
    """ライフを1枚増やす（既存カードの複製＝山札は不変）。"""
    me, _ = _players(m, name)
    src = (list(me.deck) or list(me.hand) or list(me.trash))
    if not src:
        return False
    me.life.append(copy.deepcopy(src[0]))
    return True


def mut_hand(m, name):
    """手札を1枚増やす（同上）。"""
    me, _ = _players(m, name)
    src = (list(me.deck) or list(me.hand) or list(me.trash))
    if not src:
        return False
    me.hand.append(copy.deepcopy(src[0]))
    return True


def mut_don(m, name):
    """アクティブドンを1つ増やす（新規生成＝ドンデッキは不変）。"""
    from opcg_sim.src.models.models import DonInstance
    me, _ = _players(m, name)
    me.don_active.append(DonInstance(owner_id=me.name))
    return True


def mut_power(m, name):
    """自分の場のキャラ1体のパワーを上げる（付与ドン+1・ドンデッキは不変）。"""
    me, _ = _players(m, name)
    if not me.field:
        return False
    c = me.field[0]
    try:
        c.attached_don = int(getattr(c, "attached_don", 0) or 0) + 1
    except Exception:
        return False
    return True


MUTATIONS = (("life", mut_life), ("hand", mut_hand), ("don", mut_don), ("power", mut_power))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=200)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--value", default="", help="value.npz[,policy.npz]（空=出荷既定）")
    ap.add_argument("--eps", type=float, default=1e-4,
                    help="この幅までの低下は数値誤差として見逃す")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from opcg_sim.src.core.cpu_learned import LearnedEngine
    from opcg_sim.src.learned import encoder as E
    parts = [p for p in args.value.split(",") if p]
    eng = LearnedEngine(value_path=parts[0] if parts else None,
                        policy_path=parts[1] if len(parts) > 1 else None)
    db = _load_db()

    def value_of(m, name):
        e = E.encode(m, name, eng.vocab, version=eng.enc_version)
        b = {"scalars": e["scalars"][None, :], "field": e["field"][None, :],
             "card_idx": np.asarray(e["card_idx"])[None, :]}
        return float(eng.vnet.predict(b)[0])

    boards = sample_boards(db, args.boards, args.stride)
    print(f"盤面 {len(boards)} 点で監査（enc_v={eng.enc_version}）", flush=True)

    zs, zsum = [], []
    viol = {f"{k}_{side}": [] for k, _ in MUTATIONS for side in ("me", "opp")}
    n_mut = {k: 0 for k in viol}
    for tag, i, m, name in boards:
        _, opp = _players(m, name)
        v_me, v_opp = value_of(m, name), value_of(m, opp.name)
        zs.append((v_me, v_opp))
        zsum.append(v_me + v_opp)
        for key, fn in MUTATIONS:
            for side, who in (("me", name), ("opp", opp.name)):
                try:
                    m2 = m.clone()
                except Exception:
                    continue
                if not fn(m2, who):
                    continue
                col = f"{key}_{side}"
                n_mut[col] += 1
                v2 = value_of(m2, name)     # 評価は常に name 視点
                # me 側の資源増 → 下がってはいけない／opp 側の資源増 → 上がってはいけない
                d = (v2 - v_me) if side == "me" else (v_me - v2)
                if d < -args.eps:
                    viol[col].append({"tag": tag, "i": i, "delta": round(d, 4)})

    a = np.array([x[0] for x in zs]), np.array([x[1] for x in zs])
    res = {"boards": len(boards), "enc_version": eng.enc_version,
           "zero_sum": {"mean_sum": round(float(np.mean(zsum)), 4),
                        "abs_mean_sum": round(float(np.mean(np.abs(zsum))), 4),
                        "corr": round(float(np.corrcoef(a[0], a[1])[0, 1]), 4)},
           "monotonicity": {}}
    print(f"\n零和対称性: 平均(v_me+v_opp)={res['zero_sum']['mean_sum']:+.4f} "
          f"（|平均|={res['zero_sum']['abs_mean_sum']:.4f}・相関={res['zero_sum']['corr']:+.4f}）")
    print("支配単調性（破れ＝資源が増えたのに評価が悪化）:")
    for col in sorted(viol):
        n, k = n_mut[col], len(viol[col])
        rate = k / n if n else float("nan")
        worst = sorted(viol[col], key=lambda r: r["delta"])[:3]
        res["monotonicity"][col] = {"checked": n, "violations": k, "rate": round(rate, 4),
                                    "worst": worst}
        print(f"  {col:<12} {k:>4}/{n:<4} = {rate:.3f}"
              + (f"  最悪 {worst[0]['delta']:+.3f} @{worst[0]['tag']}@{worst[0]['i']}" if worst else ""))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("\nINVARIANT_AUDIT_DONE " + json.dumps(
        {"boards": res["boards"],
         "worst_rate": max((v["rate"] for v in res["monotonicity"].values()
                            if v["checked"]), default=0.0)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
