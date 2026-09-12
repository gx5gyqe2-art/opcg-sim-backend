"""方針ラベル（dump の行 → そのターンの「方針」）と、その sidecar（計画 §20.10・WP `rs-plan-aux`）。

ユーザ決定 2026-09-12「補助ヘッドに出すのが良いのかも」→「それでやってみましょうか」。
「この盤面ではどの方針が勝ちやすいか」を V(盤面, 方針) として学ばせる。入力も Rust も変えず、
**方針ごとの勝率ヘッド**（`n_rel.plan_head`・5 出力）を補助ヘッドとして足し、その行で実際に
打った方針のヘッドだけに z の勾配を流す（mask）。ラベルは既存の記録から後付けで作れる
（生成し直しは不要）＝シャードごとの sidecar `n_record_XXXXX.plan.npz` に `plan(D int8)` を書き、
`dump_io` が pack に取り込む。

**方針クラス（`PLAN_CLASSES`・行ごとに 1 つ・-1＝ラベル無し）**
  0 face    … 自席ターンの行で、そのターンにリーダー攻撃あり・キャラ攻撃も除去効果も無し
  1 board   … 同・キャラへの攻撃 or 除去効果（`deck_roles.classify` の型を持つ PLAY／ACTIVATE_MAIN）
               あり・リーダー攻撃なし
  2 mixed   … 同・両方あり
  3 take    … 相手ターンの行で、その相手ターンに自分のリーダーへの攻撃が 1 回以上あり、
               自分のライフが 1 枚以上減った（＝受けた）
  4 guard   … 同・攻撃はあったがライフが減らなかった（＝守った）
  -1        … 自席ターンで攻撃も除去も無い（develop／pass）・相手ターンでリーダー攻撃が無い・
               ターン 0（マリガン）・次の自席ターンが無くライフの増減が測れない相手ターン
攻撃の対象（リーダー／キャラ）は候補列（`pol_sig`×`pol_cid`/`pol_tcid`）から uuid→カード ID を
引いて判別する。ライフは `scalars[0]`（自席ターンの最初の main 行＝そのターン開始時の値）。
分類の正本はここ（`tests/scripts/plan_drift.py`／`race_state.py` の計器も同じ関数を使う）。

CLI（sidecar を書く・冪等・既にあれば飛ばす）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.learned.train.plan_labels --in ~/n32_wave/w*/n_records [--force]
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

PLAN_CLASSES = ("face", "board", "mixed", "take", "guard")
D_PLAN = len(PLAN_CLASSES)                 # 5（`n_rel.D_PLAN` と同じ数）
SIDECAR_SUFFIX = ".plan.npz"
#: sidecar を作るのに要る行の列と候補の列
ROW_COLS = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
POL_COLS = ("pol_sig", "pol_cid", "pol_tcid")


def sidecar_path(shard_path):
    """`.../n_record_00012.npz` → `.../n_record_00012.plan.npz`。"""
    base, _ext = os.path.splitext(shard_path)
    return base + SIDECAR_SUFFIX


def shard_files(d):
    from opcg_sim.learned.train.dump_io import shard_files as _sf
    return _sf(d)


class Cards:
    """カード ID → リーダーか／除去の型を持つか／ブロッカーか（マスター単位でキャッシュ）。"""

    def __init__(self, db=None):
        if db is None:
            from opcg_sim.loop import decks as D
            db = D.load_db()
        self.db = db
        self._t = {}

    def info(self, cid):
        if cid not in self._t:
            m = self.db.get_card(cid) if cid else None
            if m is None:
                self._t[cid] = None
            else:
                from opcg_sim.loop import deck_roles as DR_ROLES
                forms = {k.split(":", 1)[0] for k in DR_ROLES.classify(m)}
                self._t[cid] = {
                    "leader": getattr(getattr(m, "type", None), "name", "") == "LEADER",
                    "removal": bool(forms & set(DR_ROLES.FORMS)),
                    "blocker": "ブロッカー" in (getattr(m, "keywords", ()) or ())}
        return self._t[cid]


def move_class(sj, u2c, cards):
    """1 手（move_sig の JSON）→ face／board／develop／end／don／unknown／None。

    攻撃の単位は**対象付き DON_BOX の main 行**（箱の中の ATTACK 行は呼び出し側が数えない）。
    """
    at = sj[0]
    if at in ("ATTACK", "DON_BOX"):
        tg = sj[2][0] if sj[2] else None
        if tg is None:
            return "don" if at == "DON_BOX" else None
        inf = cards.info(u2c.get(tg))
        if inf is None:
            return "unknown"
        return "face" if inf["leader"] else "board"
    if at in ("PLAY", "ACTIVATE_MAIN"):
        inf = cards.info(u2c.get(sj[1]))
        if inf is not None and inf["removal"]:
            return "board"
        return "develop"
    if at == "TURN_END":
        return "end"
    return None


def turn_label(c):
    """自席ターンのクラス集計 → face／board／mixed／develop／pass。"""
    if c["face"] and c["board"]:
        return "mixed"
    if c["face"]:
        return "face"
    if c["board"]:
        return "board"
    if c["develop"]:
        return "develop"
    return "pass"


def turn_lean(c):
    """ターンの傾き＝リーダー攻撃 /（リーダー攻撃＋キャラ攻撃＋除去）。攻撃も除去も無ければ None。"""
    tot = c["face"] + c["board"]
    return (c["face"] / tot) if tot else None


def is_own_turn(who, turn):
    """ターン t が席 who の手番か（p1＝奇数ターン・p2＝偶数・ターン 0 はマリガン）。"""
    return turn >= 1 and ((turn % 2 == 1) == (who == 0))


def uuid_map(pol, L, ptr, idx):
    """対局の全候補から uuid → カード ID。"""
    u2c = {}
    for i in idx:
        k = int(L[i])
        for j in range(ptr[i], ptr[i] + k):
            sj = json.loads(pol["pol_sig"][j])
            if sj[1] and pol["pol_cid"][j]:
                u2c[sj[1]] = str(pol["pol_cid"][j])
            if sj[2] and pol["pol_tcid"][j]:
                u2c[sj[2][0]] = str(pol["pol_tcid"][j])
            if sj[3] and pol["pol_tcid"][j]:
                u2c[sj[3][0]] = str(pol["pol_tcid"][j])
    return u2c


def iter_shards(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=None, files=None):
    """シャード npz を 1 本ずつ開いて (path, rows, pol, extra) を返す（**1 波を丸ごと載せない**）。

    波 28／30／31 を一括で読むと 14GB を超えて OOM した（2026-09-12 実測）。シャードは対局単位で
    切られている（`record_gen` は `shard_games` 局ごとに丸ごと書く）ので、1 本ずつで集計できる。
    """
    files = list(files) if files is not None else [f for d in dirs for f in shard_files(d)]
    if not files:
        raise SystemExit("シャードが無い")
    for f in files:
        with np.load(f, allow_pickle=True) as dd:
            n = int(dd["z"].shape[0])
            rows = {k: (np.asarray(dd[k])[:n] if k in dd.files else np.zeros(n, np.int64))
                    for k in row_cols}
            pol = {k: np.asarray(dd[k]) for k in pol_cols}
            extra = extra_fn(dd, n) if extra_fn else None
        yield f, rows, pol, extra


def games_of(rows):
    """1 シャードの行 → (L, ptr, [idx …])。idx は対局ごと・手順どおりの行 index。"""
    L = rows["pol_len"].astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(L)]).astype(np.int64)
    order = np.lexsort((rows["step"], rows["seed"]))
    seeds = rows["seed"][order]
    bounds = np.flatnonzero(np.diff(seeds)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(seeds)]])
    return L, ptr, [order[s:e] for s, e in zip(starts, ends)]


def iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=None, files=None):
    """対局ごとに (rows, pol, extra, L, ptr, idx) を返す。"""
    for _f, rows, pol, extra in iter_shards(dirs, row_cols, pol_cols, extra_fn, files):
        L, ptr, games = games_of(rows)
        for idx in games:
            yield rows, pol, extra, L, ptr, idx


def turn_counts(rows, pol, L, ptr, idx, cards):
    """対局 → {(who, turn): Counter(face/board/develop)}（自席ターンの main 行だけ・箱の中は数えない）。

    戻り値の 2 つ目は uuid→カード ID・3 つ目は unknown（対象を引けなかった攻撃）の数。
    """
    u2c = uuid_map(pol, L, ptr, idx)
    turns = {}
    unknown = 0
    for i in idx:
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if not is_own_turn(w, t) or int(rows["kind"][i]) != 0:
            continue
        c = turns.setdefault((w, t), collections.Counter())
        cl = move_class(json.loads(rows["sig"][i]), u2c, cards)
        if cl == "unknown":
            unknown += 1
        elif cl in ("face", "board", "develop"):
            c[cl] += 1
    return turns, u2c, unknown


def label_game(rows, pol, life0, L, ptr, idx, cards):
    """対局 → 行ごとの方針クラス（int8・-1＝無し）。`life0`＝行ごとの scalars[0]（自分のライフ）。"""
    turns, _u2c, unknown = turn_counts(rows, pol, L, ptr, idx, cards)
    # 自席ターン開始時のライフ（最初の main 行）
    life_at = {}
    for i in idx:
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if is_own_turn(w, t) and int(rows["kind"][i]) == 0 and (w, t) not in life_at:
            life_at[(w, t)] = int(round(float(life0[i])))
    plan_of_turn = {}
    for (w, t), c in turns.items():
        lab = turn_label(c)
        plan_of_turn[(w, t)] = PLAN_CLASSES.index(lab) if lab in ("face", "board", "mixed") else -1
    # 相手ターン（相手の自席ターン t）の受け／守り: 自分の t−1 と t+1 の開始ライフの差
    guard_of = {}
    for w in (0, 1):
        ts = sorted(t for (ww, t) in life_at if ww == w)
        for a_t, b_t in zip(ts, ts[1:]):
            if b_t != a_t + 2:
                continue
            opp_key = (1 - w, a_t + 1)
            n_att = turns.get(opp_key, collections.Counter())["face"]
            if n_att <= 0:
                continue
            lost = life_at[(w, a_t)] - life_at[(w, b_t)]
            guard_of[(w, a_t + 1)] = PLAN_CLASSES.index("take" if lost >= 1 else "guard")
    out = np.full(len(idx), -1, np.int8)
    for n, i in enumerate(idx):
        w = int(rows["who"][i]); t = int(rows["turn"][i])
        if t < 1:
            continue
        if is_own_turn(w, t):
            out[n] = plan_of_turn.get((w, t), -1)
        else:
            out[n] = guard_of.get((w, t), -1)
    return out, unknown


def _life0(dd, n):
    return np.asarray(dd["scalars"])[:n, 0].astype(np.float32)


def build_sidecar(shard_path, cards, force=False):
    """1 シャードの sidecar を書く。戻り値は (書いたか, 行数, クラスの度数, unknown)。"""
    out = sidecar_path(shard_path)
    if os.path.exists(out) and not force:
        return False, 0, {}, 0
    plan = None
    unknown = 0
    for _f, rows, pol, life0 in iter_shards([], files=[shard_path], extra_fn=_life0):
        plan = np.full(len(rows["z"]), -1, np.int8)
        L, ptr, games = games_of(rows)
        for idx in games:
            lab, unk = label_game(rows, pol, life0, L, ptr, idx, cards)
            plan[idx] = lab
            unknown += unk
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, plan=plan, classes=np.array(PLAN_CLASSES))
    os.replace(tmp, out)
    cnt = {c: int((plan == k).sum()) for k, c in enumerate(PLAN_CLASSES)}
    cnt["none"] = int((plan < 0).sum())
    return True, len(plan), cnt, unknown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ（glob 展開済み）")
    ap.add_argument("--force", action="store_true", help="既にある sidecar も書き直す")
    args = ap.parse_args(argv)
    t0 = time.time()
    cards = Cards()
    tot = collections.Counter()
    n_written = n_skipped = n_rows = unknown = 0
    for d in args.src:
        for f in shard_files(d):
            wrote, n, cnt, unk = build_sidecar(f, cards, force=args.force)
            if wrote:
                n_written += 1; n_rows += n; unknown += unk
                tot.update(cnt)
            else:
                n_skipped += 1
    print(f"sidecar 書いた {n_written}・飛ばした {n_skipped}・行 {n_rows}・unknown 対象 {unknown}"
          f"・クラス {dict(tot)}（{time.time()-t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
