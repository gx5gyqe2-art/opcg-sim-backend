"""自己対戦の棋譜ダンプ（純正 N ループの生データ・旧 `tests/scripts/n_record_gen.py`）。

**エンジンは Rust**（`opcg_sim.loop.driver`）になったが、**記録形式（npz の列）は 1 バイトも
変えていない**＝`n_rel_train.py`／`n_eff_train.py` は無変更で読める（計画 §16.3-A1）。

**設計**（旧版から不変）: AlphaZero の自己対戦棋譜に相当する生データを 1 本で採る——
  ①全判断点（main=木探索／window=窓の根畳み／commit=箱コミット機械実行）の符号化＋勝敗 z
  ②main 窓は**候補と訪問分布**（等価マージ後＝decide の選択と同じ集計）
採掘器（z 密教師・方策ターゲット）は別計器＝ダンプは生のまま保存する。

**規約**: ランダムリーダー×生成デッキ（`decks.leader_pair` は旧 `promotion_gate._leader_pair` と
同じ `seed*7919+13`）。候補生成は `prune_futile=GEN_PRUNE_FUTILE`（生成は枝刈りを外す＝v6 柱⑤）。
探索プロファイルは serve 既定（箱化一式＋箱コミット ON）＝実対局と同じ行動列。

**乱数**（旧版との違い・計画 §8.17 の申告 (2)）: 対局と探索は Rust の決定的な生成器で回す
（旧版は CPython の `random` と `np.random.Generator`）。seed → 対局は決定的だが、**同じ seed で
旧版と同じ局にはならない**（乱数系統が違う）。ルールが一致していることは golden 2 本が担保する。

実行例（分散: 子が --seed-base を分担）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.loop.record_gen \\
    --games 60 --seed-base 200000 --workers 4 --sims 64 --out n_records/part1

シャード npz の列（D=判断点行・K=main 候補の flatten）— 旧版と同一:
  scalars(D,·) field(D,·) card_idx(D,24)   … 符号化（判断点の状態・手番視点）
  z(D)                                     … 勝敗 ±1（手番視点・純正 AZ の素の z）
  who(D) kind(D: 0=main/1=window/2=commit) turn(D) step(D) seed(D)
  sig(D str)                               … 選択手の move_sig（JSON・箱レベル）
  pol_len(D) pol_chosen(D)                 … main の候補数／選択候補の slice 内 index（無ければ 0/-1）
  pol_n(K) pol_q(K) pol_k(K) pol_sig(K str)… 訪問合算 n・行動価値 q・don_k（-1=無し）・候補 move_sig
  pol_cid(K str) pol_tcid(K str)           … 候補の主体/第1対象の**カードID**
  tokens(D,22,·) pol_si(K) pol_ti(K)      … NRel 用（dump v2 で追加・v3 で dtype だけ変えた）
  deck_kinds(D str)                        … **dump v4 で追加**（§20.8.1-4）。手番側デッキの除去の
                                             型（`deck_roles` の JSON・`--decks synth_roles` 以外は
                                             `{}`）。層別の読み（除去率別・型別）に使う

**dump v3**（既定・2026-09-07・計画 §18.5）: 常駐の 9 割を占める 3 本の保存 dtype を半分にする——
`tokens`／`scalars` を **float16**・`card_idx` を **int16**（`pol_si`/`pol_ti` は v2 から int16）。
値は float32／int64 を **cast しただけ**で、**符号化（`enc_version`=13）も列の形も変えていない**。
学習側は `learned/train/dump_io.py` が波ごとに memmap の pack を作って読む（v2 の npz＝
float32／int64 も同じ関数で読める＝過去の波はそのまま使える）。v1／v2 を書く経路は無い。

**dump v4**（2026-09-10・計画 §20.8）: **既存の列は 1 バイトも変えず**、`deck_kinds(D str)` を足す
（`--decks synth_roles` で差し込んだ除去の型・型が無い経路では `{}`）。`dump_io.load_dump` は
知らない列を読まない＝v3 の波も v4 の波も同じ関数で読める。対局メタ（seed・両席のリーダー・
両席の型）は part ごとの sidecar `meta_games.json` にも書く（層別の集計用）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import multiprocessing as mp                                           # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

import numpy as np                                                     # noqa: E402

from opcg_sim.loop import decks as D                                   # noqa: E402
from opcg_sim.loop import driver as DR                                 # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402

MAX_STEPS = 400
MAX_CI = 24            # card_idx の PAD 長（G 系符号化の既定枠・列の形は据え置き）
ENC_VERSION = 12       # dump v1 の符号化世代（過去の波の meta にだけ残る）
ENC_VERSION_V2 = 13    # dump v2／v3（v12 ＋ n_rel_feat のグローバル追加 29・append-only）
DUMP_VERSION = 4       # v3 ＋ `deck_kinds` 列（既存の列は不変・§20.8）
#: `--decks` の既定（歴代の波は全て synth＝規約を変えない）。
DEFAULT_DECKS = "synth"

# v3 で dtype を落とす列（**float32／int64 を cast するだけ**＝符号化も列の形も変えない）。
# 常駐の 9 割はこの 3 本（tokens 1,760B／scalars 492B／card_idx 176B per 行）。float16 の丸めが
# forward に与える差は 1 バッチ最大 1.07e-4（`docs/reports/2026-09-07_train_profile.md` §3）。
DT_V3 = {"tokens": np.float16, "scalars": np.float16, "card_idx": np.int16}

# Rust の `encode` は平らな列で返す。**旧版と同じ形**へ戻してから積む（`n_rel_train` は
# `tokens.shape[1:]` をそのまま使う＝形が変われば読めない）。
FIELD_SHAPE = (10, 8)    # `encoder.MAX_FIELD*2` × `PER_CHAR`
TOKENS_SHAPE = (22, 20)  # `n_rel_feat` の 22 枠 × S_DIM

_KIND = {"main": 0, "window": 1, "commit": 2}
_G = {}


def move_sig(mv):
    """Python `learned/plan.move_sig` と同じ鍵（Rust `search::decide::move_sig` の写し）。

    JSON にしたときの形（`[action_type, uuid, target_ids, selected_uuids, accepted]`）が
    旧版の `json.dumps(tuple)` と一致する＝`pol_sig`／`sig` 列の中身が変わらない。
    """
    p = mv.get("payload") or {}
    return [mv.get("action_type") or p.get("action_type"), p.get("uuid"),
            list(p.get("target_ids") or ()), list(p.get("selected_uuids") or ()),
            p.get("accepted")]


def _don_k(mv):
    return (mv.get("payload") or {}).get("don_k")


def _init_worker(sims, net, dirichlet_eps, temp_turns, decks=DEFAULT_DECKS):
    E.engine()
    _G["spec"] = E.SeatSpec(net, sims=sims, dirichlet_eps=dirichlet_eps,
                            temp_turns=temp_turns, prune_futile=E.GEN_PRUNE_FUTILE)
    _G["db"] = D.load_db()
    _G["decks"] = decks


class _Recorder:
    """1 局ぶんの観測（`driver.run_game(observer=…)`）。盤面は動かさない。"""

    def __init__(self):
        self.rows = {k: [] for k in ("scalars", "field", "card_idx", "who", "kind", "turn",
                                     "step", "sig", "pol_len", "pol_chosen", "tokens")}
        self.pol = {k: [] for k in ("n", "q", "k", "sig", "cid", "tcid", "si", "ti")}

    def __call__(self, game, name, turn, step, out, move):
        # 符号化は decide **前**の状態＝この判断が見た盤面（Rust の decide は盤面を変えない）。
        enc = json.loads(game.encode(name))
        self.rows["scalars"].append(np.asarray(enc["scalars"], np.float32))
        self.rows["field"].append(
            np.asarray(enc["field"], np.float32).reshape(FIELD_SHAPE))
        ci = np.zeros(MAX_CI, np.int64)
        src = np.asarray(enc["card_idx"], np.int64)[:MAX_CI]
        ci[:len(src)] = src
        self.rows["card_idx"].append(ci)
        self.rows["who"].append(name)
        self.rows["kind"].append(_KIND.get(out.get("kind"), 0))
        self.rows["turn"].append(int(turn))
        self.rows["step"].append(step)
        # 決定の同一性は**箱レベル**（原始手化と残り掘りの前）＝Rust が `sig`／`k` で返す。
        # `out["move"]` は実対局へ出す手なので、候補（groups）と突き合わせる鍵にはならない。
        sig = out.get("sig") if out.get("sig") is not None else move_sig(move)
        self.rows["sig"].append(json.dumps(sig, ensure_ascii=False))

        legal = (out.get("stats") or {}).get("legal") or []
        groups = out.get("groups") or []
        # 窓・コミットは訪問を配らない＝候補を持たない（旧版と同じく空）。
        if out.get("kind") != "main":
            groups = []
        self.rows["pol_len"].append(len(groups))
        self.rows["tokens"].append(
            np.asarray(enc["tokens"], np.float32).reshape(TOKENS_SHAPE))
        if not groups:
            self.rows["pol_chosen"].append(-1)
            return
        idx = json.loads(game.dump_index_json(name))
        cids, slots = idx["cids"], idx["slots"]
        chosen, k_sel = -1, out.get("k")
        for gi, g in enumerate(groups):
            rep = legal[g["rep"]]
            gsig = move_sig(rep)
            gk = _don_k(rep)
            # 配分箱は k 違いが同 sig（move_sig は don_k 非含有）＝ (sig, k) で照合
            if chosen < 0 and gsig == sig and gk == k_sel:
                chosen = gi
            self.pol["n"].append(float(g["n"]))
            self.pol["q"].append(float(g["q"]))
            self.pol["k"].append(-1 if gk is None else int(gk))
            self.pol["sig"].append(json.dumps(gsig, ensure_ascii=False))
            su = gsig[1]
            tu = gsig[2][0] if gsig[2] else None
            self.pol["cid"].append(cids.get(su) or "")
            self.pol["tcid"].append((cids.get(tu) or "") if tu else "")
            self.pol["si"].append(slots.get(su, -1) if su else -1)
            self.pol["ti"].append(slots.get(tu, -1) if tu else -1)
        self.rows["pol_chosen"].append(chosen)


def play_one(seed):
    """1 局を打って行を返す（勝敗が付いた局だけ・純正 z）。失敗は None。"""
    spec, db = _G["spec"], _G["db"]
    rec = _Recorder()
    try:
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2, kinds = D.build_pair(db, la, lb, seed, _G.get("decks", DEFAULT_DECKS),
                                     with_kinds=True)
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2,
                          max_steps=MAX_STEPS, observer=rec)
    except Exception:                                          # noqa: BLE001  1 局の失敗で止めない
        return None
    winner, turn, steps, acts = res["winner"], res["turns"], res["steps"], res["acts"]
    if (winner is None or turn < 4 or steps >= MAX_STEPS or min(acts.values()) == 0):
        return None                                            # 純正 z＝勝敗が付いた対局のみ採用
    if not rec.rows["who"]:
        return None
    z = {"p1": 1.0 if winner == "p1" else -1.0}
    z["p2"] = -z["p1"]
    rows, pol = rec.rows, rec.pol
    # dump v4: 手番側デッキの除去の型（JSON・型を持たない経路は "{}"）
    kj = {"p1": json.dumps(kinds[0] or {}, ensure_ascii=False, separators=(",", ":")),
          "p2": json.dumps(kinds[1] or {}, ensure_ascii=False, separators=(",", ":"))}
    # dump v3: tokens／scalars は float16・card_idx は int16 で持つ（cast するだけ・§18.5）
    return {"tokens": np.array(rows["tokens"], np.float32).astype(DT_V3["tokens"]),
            "pol_si": np.array(pol["si"], np.int16),
            "pol_ti": np.array(pol["ti"], np.int16),
            "scalars": np.array(rows["scalars"], np.float32).astype(DT_V3["scalars"]),
            "field": np.array(rows["field"], np.float32),
            "card_idx": np.array(rows["card_idx"], np.int64).astype(DT_V3["card_idx"]),
            "z": np.array([z[w] for w in rows["who"]], np.float32),
            "who": np.array([0 if w == "p1" else 1 for w in rows["who"]], np.int8),
            "kind": np.array(rows["kind"], np.int8),
            "turn": np.array(rows["turn"], np.int16),
            "step": np.array(rows["step"], np.int32),
            "seed": np.full(len(rows["who"]), seed, np.int64),
            "sig": np.array(rows["sig"]),
            "pol_len": np.array(rows["pol_len"], np.int32),
            "pol_chosen": np.array(rows["pol_chosen"], np.int16),
            "pol_n": np.array(pol["n"], np.float32),
            "pol_q": np.array(pol["q"], np.float32),
            "pol_k": np.array(pol["k"], np.int16),
            "pol_sig": np.array(pol["sig"]) if pol["sig"] else np.array([], dtype="U1"),
            "pol_cid": np.array(pol["cid"]) if pol["cid"] else np.array([], dtype="U1"),
            "pol_tcid": np.array(pol["tcid"]) if pol["tcid"] else np.array([], dtype="U1"),
            "deck_kinds": np.array([kj[w] for w in rows["who"]]),
            # 局メタ（npz には積まない・part の sidecar へ）
            "_meta": {"seed": seed, "leaders": [la, lb], "kinds": list(kinds),
                      "winner": winner, "turns": turn, "rows": len(rows["who"])}}


_ROW_KEYS = ("scalars", "field", "card_idx", "z", "who", "kind", "turn", "step",
             "seed", "sig", "pol_len", "pol_chosen")
_POL_KEYS = ("pol_n", "pol_q", "pol_k", "pol_sig", "pol_cid", "pol_tcid")
_TOK_KEYS = ("tokens", "pol_si", "pol_ti")            # NRel 用（v2 で追加・v3 も同じ列）
_V4_KEYS = ("deck_kinds",)                            # v4 で追加（§20.8）


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=64,
                    help="生成 sims（純正 N ループは 32→64+ に引き上げ・2026-08-26 設計）")
    ap.add_argument("--net", default=None,
                    help="生成役のネット npz（既定=出荷既定）。旧 --n1-net/--neff-net の後継"
                         "（`neff:`／`n1:` 接頭辞も受ける）")
    ap.add_argument("--dirichlet-eps", type=float, default=0.25,
                    help="root priors への Dirichlet ノイズ（純正 AZ の自己対戦既定 0.25・0=無効）")
    ap.add_argument("--temp-turns", type=int, default=4,
                    help="この turn まではメイン窓を訪問分布 τ=1 でサンプリング（0=無効）")
    ap.add_argument("--shard-games", type=int, default=10)
    ap.add_argument("--dump-v2", action="store_true",
                    help="【廃止・受けるだけ】dump v3 が既定になった（2026-09-07・§18.5）。"
                         "分散生成の古い指示書がこの旗を付けたまま回っても壊れないように残す")
    ap.add_argument("--decks", default=DEFAULT_DECKS,
                    choices=("singleton", "synth", "synth_dig", "synth_roles"),
                    help="デッキの中身（既定 synth＝歴代の波と同じ規約）。"
                         "synth_roles=除去の型を色ごとに差し込む（教材の対照・§20.8.1）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.dump_v2:
        print("[note] --dump-v2 は廃止（既定が dump v3・列は同じで dtype だけ半分）", flush=True)
    keys = _ROW_KEYS + _POL_KEYS + _TOK_KEYS + _V4_KEYS
    buf = {k: [] for k in keys}
    games = []                                         # part の sidecar（対局メタ）
    shard = n_rows = n_drop = n_main = 0
    initargs = (args.sims, args.net, args.dirichlet_eps, args.temp_turns, args.decks)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                      initargs=initargs) as pool:
        done = 0
        for r in pool.imap_unordered(play_one,
                                     [args.seed_base + i for i in range(args.games)]):
            done += 1
            if r is None:
                n_drop += 1
            else:
                games.append(r.pop("_meta"))
                for k in buf:
                    buf[k].append(r[k])
                n_rows += len(r["z"])
                n_main += int((r["kind"] == 0).sum())
            if done % args.shard_games == 0 or done == args.games:
                if buf["z"]:
                    path = os.path.join(args.out, f"n_record_{shard:05d}.npz")
                    tmp = os.path.join(args.out, f".n_record_{shard:05d}.tmp.npz")
                    np.savez_compressed(tmp, **{k: np.concatenate(buf[k]) for k in buf})
                    os.replace(tmp, path)
                    shard += 1
                    buf = {k: [] for k in keys}
                print(f"  {done}/{args.games}局 行{n_rows}（main {n_main}） 棄却{n_drop}"
                      f" {time.time()-t0:.0f}s", flush=True)
    # part の sidecar: 対局メタ（seed・両席のリーダー・両席の型）＝層別の集計用（§20.8.1-4）
    with open(os.path.join(args.out, "meta_games.json"), "w") as f:
        json.dump({"decks": args.decks, "seed_base": args.seed_base, "games": games},
                  f, ensure_ascii=False)
    with open(os.path.join(args.out, "meta_n_record.json"), "w") as f:
        json.dump({"games": args.games, "rows": n_rows, "main_rows": n_main,
                   "dropped": n_drop, "sims": args.sims, "decks": args.decks,
                   "enc_version": ENC_VERSION_V2,
                   "dump_version": DUMP_VERSION, "seed_base": args.seed_base,
                   "net": E.resolve_net(args.net), "engine": "rust",
                   "dirichlet_eps": args.dirichlet_eps,
                   "temp_turns": args.temp_turns}, f, ensure_ascii=False)
    print("N_RECORD_DONE " + json.dumps({"rows": n_rows, "main_rows": n_main,
                                         "dropped": n_drop}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
