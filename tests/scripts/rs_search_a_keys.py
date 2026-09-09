"""WP `rs-search-a`（計画 §20.5）の「3 決定」を条件ごとに機械的に読む（判定は書かない）。

分析 #4（`docs/reports/2026-09-08_scenario_analysis_04.md`）が挙げた 3 つを、**決定番号ではなく
盤面と手の並びで**特定する（seed と sims が変わると決定番号がずれるため）:

1. `enel_t9_kaminosabaki_before_attack`（`enel_human_20260810_t9-9`）
   … seat の手の並びで「オーム(OP15-061) の最初の ATTACK」より前に「神の裁き(OP15-075) の PLAY」が
     あるか。人間はパンプしてから殴った（`true` が人間の線）。オームで殴らなかった run は
     `"no_oom_attack"`。
2. `enel_roger_t7_gammaknife`（`human_enel_vs_roger_20260904_t7-8`）
   … seat が「ガンマナイフ(OP05-077)」を PLAY したか（分析 #4 #7 は Q が上なのに訪問数で負けた）。
3. `enel_luffy_t4_kami_ko`（`human_enel_vs_luffy_20260904_t4-5`・WP `rs-setup-box`）
   … seat が「神の裁き(OP15-075)」を PLAY したときの **KO 対象の選択**が空でないか
     （＝相手の 1c ルフィを KO したか）。`RESOLVE_EFFECT_SELECTION` で相手のキャラを
     選んだ決定が神の裁きの PLAY より後にあれば `true`。打たなかった run は `"no_kami"`。
4. `doffy_t10_root_visited`（`human_doflamingo_vs_luffy_20260904_t10-11`）
   … 分岐点の**最初の** seat 側 kind=main 決定（turn=start_turn）で訪問された根の手の数と合法手の数。
   `doffy_t10_winner` … その run の勝敗（最後のフレームの winner）。
5. `enel_t9_winner`（`enel_human_20260810_t9-9`）… その run の勝敗（分析 #6 の「決着（T9・p1 勝ち）」）。
6. `box_vs_bare`（WP `rs-setup-box-2`・分析 #6 §3 の表と同じ定義）… **条件ごとの合計**（seed 別ではない）。
   箱のある seat 側 kind=main 決定について「箱を選んだ数／素の手を選んだ数／素の N > 箱の N 和 の枚数」。
   「枚」＝（決定 × card_id）の組で、素の準備の手と箱が**同じ決定に両方居る**もの。§20.7.6 で素の手を
   落とすと分母（`bare_gt_box` の m）が 0 に落ちる＝取り合いそのものが無くなったことの鍵。

出力は `{条件: {鍵: {seed: 値}}}`（`RESULT.json` の `key_decisions` にそのまま入る形）。
"""
import argparse
import glob
import json
import os
import sys

KAMI = "OP15-075"   # 神の裁き（イベント・リーダー +1000 か KO）
OOM = "OP15-061"    # オーム（キャラ 2000）
GAMMA = "OP05-077"  # ガンマナイフ（イベント・除去）

SCN_ENEL_T9 = "enel_human_20260810_t9-9"
SCN_ENEL_ROGER = "human_enel_vs_roger_20260904_t7-8"
SCN_ENEL_LUFFY = "human_enel_vs_luffy_20260904_t4-5"
SCN_DOFFY = "human_doflamingo_vs_luffy_20260904_t10-11"


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _seat_moves(res: dict) -> list:
    """seat（分岐点の手番）が出した手の並び（`decisions` の `chosen`・kind を問わない）。"""
    ds = res.get("decisions") or []
    seat = ds[0]["player"] if ds else None
    return [d.get("chosen") or {} for d in ds if d.get("player") == seat]


def _kami_before_oom_attack(res: dict):
    mv = _seat_moves(res)
    atk = next((i for i, m in enumerate(mv)
                if m.get("action_type") == "ATTACK" and m.get("card") == OOM), None)
    if atk is None:
        return "no_oom_attack"
    return any(m.get("action_type") == "PLAY" and m.get("card") == KAMI for m in mv[:atk])


def _played(res: dict, card: str) -> bool:
    return any(m.get("action_type") == "PLAY" and m.get("card") == card for m in _seat_moves(res))


def _kami_ko_target(res: dict):
    """神の裁きの KO 対象の選択が非空か（WP `rs-setup-box`・§20.7.2 の「KO する／しない」）。

    神の裁きを打った後の seat 側 `RESOLVE_EFFECT_SELECTION` のうち、**相手のカードを選んだ**
    ものがあれば `true`（自分への +1000 は自分のカードなので除く）。打たなければ `"no_kami"`。
    """
    ds = res.get("decisions") or []
    seat = ds[0]["player"] if ds else None
    opp_cards = _opponent_card_ids(res, seat)
    seen_kami = False
    for d in ds:
        if d.get("player") != seat:
            continue
        m = d.get("chosen") or {}
        if m.get("action_type") == "PLAY" and m.get("card") == KAMI:
            seen_kami = True
            continue
        if not seen_kami or m.get("action_type") != "RESOLVE_EFFECT_SELECTION":
            continue
        if any(c in opp_cards for c in (m.get("selected") or [])):
            return True
    return False if seen_kami else "no_kami"


def _opponent_card_ids(res: dict, seat) -> set:
    """相手の場・リーダーに居るカードの card_id（記述子は card_id 基準）。"""
    frames = res.get("frames") or []
    opp = "p2" if seat == "p1" else "p1"
    out = set()
    for fr in frames:
        side = ((fr.get("players") or {}).get(opp)) or {}
        for c in (side.get("field") or []):
            if c.get("card_id"):
                out.add(c["card_id"])
        leader = side.get("leader") or {}
        if leader.get("card_id"):
            out.add(leader["card_id"])
    return out


def _first_main_root(res: dict):
    """最初の seat 側 kind=main 決定の（訪問された根の手の数, 合法手の数, turn）。"""
    ds = res.get("decisions") or []
    seat = ds[0]["player"] if ds else None
    for d in ds:
        if d.get("player") != seat or d.get("kind") != "main":
            continue
        stats = d.get("legal_stats") or []
        if not stats:
            continue
        return (sum(1 for s in stats if (s.get("n") or 0) > 0), len(stats), d.get("turn"))
    return (None, None, None)


def _entry(stats: list, pred):
    """条件に合う `legal_stats` の中で訪問数最多のもの（無ければ None）。"""
    hits = [s for s in stats if pred(s.get("move") or {})]
    if not hits:
        return None
    e = max(hits, key=lambda s: (s.get("n") or 0))
    return {"n": e.get("n"), "q": (round(e["q"], 4) if e.get("q") is not None else None),
            "p": (round(e["p"], 4) if e.get("p") is not None else None),
            "move": e.get("move")}


def _is_play(card):
    return lambda m: m.get("action_type") == "PLAY" and m.get("card") == card


def _is_attack(card):
    """その card の「根の攻撃手」。攻撃は箱化されて `DON_BOX`（don_k・targets つき）で根に並ぶので
    素の `ATTACK`（ドンを付け終えた後の形）と両方を拾う。"""
    def pred(m):
        if m.get("card") != card:
            return False
        at = m.get("action_type")
        return at == "ATTACK" or (at == "DON_BOX" and m.get("targets"))
    return pred


def _seat_main(res: dict):
    ds = res.get("decisions") or []
    seat = ds[0]["player"] if ds else None
    return [d for d in ds
            if d.get("player") == seat and d.get("kind") == "main" and d.get("legal_stats")]


def _head_to_head(res: dict, preds: dict, rank: str):
    """`preds` の全部が合法手に居る seat 側 kind=main 決定のうち、`rank` の手が**最も訪問された**
    決定を、各手の N/Q/P つきで返す（同点は早い決定）。

    「最初の決定」ではなく「一番読まれた決定」を採る理由: 分岐点の #0 ではどちらも n=0 で
    情報が無い（P が低い手は序盤の決定では読まれない）。分析 #4 が見たのは「読まれた上で
    訪問数で負けた」決定（エネル T9 #14・ロジャー T7 #7）＝その手の訪問数が最大の決定。
    """
    best = None
    for d in _seat_main(res):
        stats = d["legal_stats"]
        got = {k: _entry(stats, pr) for k, pr in preds.items()}
        if any(v is None for v in got.values()):
            continue
        n = got[rank].get("n") or 0
        if best is not None and n <= best[0]:
            continue
        top = _entry(stats, lambda m: True)
        best = (n, {"action_index": d.get("action_index"), "turn": d.get("turn"),
                    "chosen": d.get("chosen"), "chosen_q": d.get("value"),
                    "top_by_n": top, **got})
    return best[1] if best else None


def _first_main_detail(res: dict):
    for d in _seat_main(res):
        stats = d["legal_stats"]
        top = sorted(stats, key=lambda s: -(s.get("n") or 0))[:3]
        return {"action_index": d.get("action_index"), "turn": d.get("turn"),
                "chosen": d.get("chosen"), "visited": sum(1 for s in stats if (s.get("n") or 0) > 0),
                "legal": len(stats),
                "top3": [{"move": s.get("move"), "n": s.get("n"),
                          "q": (round(s["q"], 4) if s.get("q") is not None else None),
                          "p": (round(s["p"], 4) if s.get("p") is not None else None)}
                         for s in top]}
    return None


SETUP_ACTIONS = ("PLAY", "ACTIVATE_MAIN")


def _box_vs_bare(res: dict, acc: dict) -> None:
    """箱と素の手の取り合いを 1 run ぶん `acc` へ足す（分析 #6 §3 の表と同じ定義）。

    - `decisions`＝箱（`SETUP_BOX`）が合法手に居る kind=main 決定の数（**両席**・両席とも同じ設定で打つ）
    - `chose_box`／`chose_bare`＝その決定で箱／素の準備の手（同じ card_id の箱があるもの）を選んだ数
      （選んだ手は `sig`＝**原始手化の前**の move_sig で見る＝箱を選んだ決定はここで分かる）
    - `bare_gt_box`＝（決定 × card_id）の組のうち「素の手の N 和 > 箱の N 和」の数／組の総数
    - `box_max_gt_bare`＝同じ組のうち「箱の N の最大 > 素の手の N 和」の数
    """
    for d in (res.get("decisions") or []):
        stats = d.get("legal_stats")
        if d.get("kind") != "main" or not stats:
            continue
        boxes = [s for s in stats if (s.get("move") or {}).get("action_type") == "SETUP_BOX"]
        if not boxes:
            continue
        acc["decisions"] += 1
        sig = d.get("sig") or [None]
        boxed_cards = {(s["move"] or {}).get("card") for s in boxes}
        if sig[0] == "SETUP_BOX":
            acc["chose_box"] += 1
        elif sig[0] in SETUP_ACTIONS and (d.get("chosen") or {}).get("card") in boxed_cards:
            acc["chose_bare"] += 1
        for card in sorted(c for c in boxed_cards if c is not None):
            bare = [s for s in stats
                    if (s.get("move") or {}).get("action_type") in SETUP_ACTIONS
                    and (s.get("move") or {}).get("card") == card]
            if not bare:
                continue  # 素の手が候補に居ない＝取り合いが起きない（§20.7.6 の狙い）
            mine = [s for s in boxes if (s["move"] or {}).get("card") == card]
            bare_n = sum(s.get("n") or 0 for s in bare)
            box_n = sum(s.get("n") or 0 for s in mine)
            acc["pairs"] += 1
            if bare_n > box_n:
                acc["bare_gt_box"] += 1
            if max((s.get("n") or 0) for s in mine) > bare_n:
                acc["box_max_gt_bare"] += 1


def _winner(res: dict):
    frames = res.get("frames") or []
    return (frames[-1] if frames else {}).get("winner")


def _seat_of(res: dict):
    ds = res.get("decisions") or []
    return ds[0]["player"] if ds else None


def keys_for(root: str) -> dict:
    out: dict = {}
    for cond_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(cond_dir):
            continue
        cond = os.path.basename(cond_dir)
        cur: dict = {}
        for scn, fn in (
            (SCN_ENEL_T9, lambda r: _kami_before_oom_attack(r)),
            (SCN_ENEL_ROGER, lambda r: _played(r, GAMMA)),
            (SCN_ENEL_LUFFY, lambda r: _kami_ko_target(r)),
        ):
            key = {SCN_ENEL_T9: "enel_t9_kaminosabaki_before_attack",
                   SCN_ENEL_ROGER: "enel_roger_t7_gammaknife",
                   SCN_ENEL_LUFFY: "enel_luffy_t4_kami_ko"}[scn]
            for path in sorted(glob.glob(os.path.join(cond_dir, scn, "r3_s*.frames.json"))):
                seed = os.path.basename(path).split(".")[0].split("_s")[-1]
                cur.setdefault(key, {})[f"s{seed}"] = fn(_load(path))
        for path in sorted(glob.glob(os.path.join(cond_dir, SCN_DOFFY, "r3_s*.frames.json"))):
            seed = os.path.basename(path).split(".")[0].split("_s")[-1]
            res = _load(path)
            visited, legal, turn = _first_main_root(res)
            cur.setdefault("doffy_t10_root_visited", {})[f"s{seed}"] = visited
            cur.setdefault("doffy_t10_root_legal", {})[f"s{seed}"] = legal
            cur.setdefault("doffy_t10_root_turn", {})[f"s{seed}"] = turn
            cur.setdefault("doffy_t10_winner", {})[f"s{seed}"] = _winner(res)
            cur.setdefault("doffy_t10_seat", {})[f"s{seed}"] = _seat_of(res)
            cur.setdefault("detail_doffy_t10_first_main", {})[f"s{seed}"] = _first_main_detail(res)
        # 3 決定の「前後」（同じ局面での N/Q/P の並び）と T9 の勝敗（決着）。
        for path in sorted(glob.glob(os.path.join(cond_dir, SCN_ENEL_T9, "r3_s*.frames.json"))):
            seed = os.path.basename(path).split(".")[0].split("_s")[-1]
            res = _load(path)
            cur.setdefault("detail_enel_t9_kami_vs_oom", {})[f"s{seed}"] = _head_to_head(
                res, {"kami": _is_play(KAMI), "oom_attack": _is_attack(OOM)},
                rank="oom_attack")
            cur.setdefault("enel_t9_winner", {})[f"s{seed}"] = _winner(res)
        # 箱と素の手の取り合い（条件ごとの合計・WP `rs-setup-box-2`）。
        acc = {"decisions": 0, "chose_box": 0, "chose_bare": 0,
               "pairs": 0, "bare_gt_box": 0, "box_max_gt_bare": 0}
        for path in sorted(glob.glob(os.path.join(cond_dir, "*", "r3_s*.frames.json"))):
            _box_vs_bare(_load(path), acc)
        if acc["decisions"]:
            cur["box_vs_bare"] = {**acc,
                                  "bare_gt_box_ratio": f"{acc['bare_gt_box']}/{acc['pairs']}"}
        for path in sorted(glob.glob(os.path.join(cond_dir, SCN_ENEL_ROGER, "r3_s*.frames.json"))):
            seed = os.path.basename(path).split(".")[0].split("_s")[-1]
            cur.setdefault("detail_enel_roger_t7_gamma", {})[f"s{seed}"] = _head_to_head(
                _load(path), {"gamma": _is_play(GAMMA)}, rank="gamma")
        if cur:
            out[cond] = cur
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = keys_for(args.root)
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
