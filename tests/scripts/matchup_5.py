"""対面記事 5 本でほかのデッキの進行も突き合わせる（WP `matchup-5`・`docs/search_efficiency_plan.md` §7.65・台帳 2+471）。

`matchup_kr.py`（紫カイドウ対青ロックス・§7.63）を 6 つの対面に一般化したもの（5 本の記事＋紫カイドウ対青ロックスの打ち直し）。
デッキは記事のデッキ（`tests/fixtures/decks/article_decks_20261027.json`）。読み取りだけの計測
（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。

対面（`docs/matchup_items/README.md` の表）。視点の側を常に **p1**。局の番号 i（0〜119）の seed は帯の先頭＋i、
i < 60 は視点の側の先攻・i ≥ 60 は後攻（6 対面 × 120 局＝720 局）。両席とも r27・出荷の設定（`blunder_audit.ship_spec`）。

1. ``play``: 局を打ち、全ての判断点で打った手・決め方・根の候補の上位 8 を記録する（`games_shard*.jsonl`）。
   シャードは「先攻の局と後攻の局の組」（同じ対面の i と i＋60）を単位に配る＝どのシャードも先攻と後攻が半々。
2. ``trace``: 全局を探索なしで当て直し、手ごとの盤面の要約と打った手のカードを書く（`trace_<対面>.jsonl`）。
3. ``items``: 記事の項目の突き合わせと、比べる点の洗い出し（`items_<対面>.jsonl`・`points_pool.jsonl`）。
4. ``judge``: 比べる点を 1 点 16 通りで打ち比べる（`judge_points*.jsonl`・`--shard i/n`）。
5. ``summary``: 対面ごとの集計と横断のまとめ（`summary.json`）。

    python tests/scripts/matchup_5.py play --workers 4 --shard 0/18
    python tests/scripts/matchup_5.py trace
    python tests/scripts/matchup_5.py items
    python tests/scripts/matchup_5.py judge --workers 4 --shard 0/18
    python tests/scripts/matchup_5.py summary

再開できる（終えた局・点は飛ばす）。
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "matchup_5")
NET = KR.NET
#: 対面 → (視点の側のデッキ, 相手のデッキ, seed の帯の先頭, 表の名前)
#: 記事のデッキ（ユーザが送った画像から書き起こし・`docs/matchup_items/README.md` の末尾の表）
DECKS = os.path.join("tests", "fixtures", "decks", "article_decks_20261027.json")
#: 対面 → (視点の側のデッキ, 相手のデッキ, seed の帯の先頭, 表の名前)
MU = {
    "enel_rocks": ("enel_1021", "rocks_enel_article_1021", 4251000, "紫エネル 対 青ロックス"),
    "bluffy_sabo": ("bluffy_1024", "sabo_bluffy_article_1024", 4252000, "黒ルフィ 対 赤黒サボ"),
    "rocks_mirror": ("rocks_fortress_1005", "rocks_fortress_1005", 4253000, "青ロックス ミラー"),
    "mihawk_sabo": ("mihawk_0917", "sabo_mihawk_article_0917", 4254000, "緑ミホーク 対 赤黒サボ"),
    "rocks_sabo": ("rocks_sabo_article_1015", "sabo_rocks_article_1015", 4255000, "青ロックス 対 赤黒サボ"),
    "kaido_rocks": ("kaido_1027", "rocks_kaido_article_1027", 4256000, "紫カイドウ 対 青ロックス"),
}
N_GAMES = 120
#: 判定の配り方の番号のずらし（blunder_audit 0〜2000 番台・hold_bias 3000・small_play 4000・matchup_kr 7000 と重ねない）。
OFFSET = 8000
_G = {}


def deck(name):
    x = json.load(open(DECKS, encoding="utf-8"))[name]
    cs = []
    for cid, n in x["cards"].items():
        cs += [cid] * int(n)
    assert len(cs) == 50, (name, len(cs))
    return x["leader"], cs


def game_setup(mu, idx):
    """(対面, 局の番号) → (seed, first_player)。視点の側は常に p1。"""
    return MU[mu][2] + idx, ("p1" if idx < N_GAMES // 2 else "p2")


def new_game(mu, idx):
    seed, first = game_setup(mu, idx)
    return DR.new_game(seed, deck(MU[mu][0]), deck(MU[mu][1]), first)


def replay(g, step):
    game = new_game(g["mu"], g["idx"])
    for s, mv in enumerate(g["moves"]):
        if s == step:
            break
        name = json.loads(game.pending_json())["player_id"]
        BA.apply(game, name, mv)
    return game


read_jsonl = BA.read_jsonl
read_glob = KR.read_glob


def load_games(out, mu=None):
    rows = read_glob(os.path.join(out, "games*.jsonl"), lambda g: (g["mu"], g["idx"]))
    return {k: v for k, v in rows.items() if mu is None or k[0] == mu}


def shard_of(a):
    return tuple(int(x) for x in a.shard.split("/")) if a.shard else (0, 1)


# ---------------------------------------------------------------- 1. play

def play_one(job):
    mu, idx = job
    E.engine()
    seed, first = game_setup(mu, idx)
    spec = BA.ship_spec(NET)
    pts, moves = [], []

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        own = game.turn_player() == name
        rows, top = KR.top_rows(out)
        ch = BA.chosen(out, rows) if rows else None
        pts.append({"step": step, "seat": name, "turn": turn, "own": own, "act": pend.get("action"), "kind": out.get("kind"),
                    "dtype": BA.dtype(pend, own, move), "n_cand": len(rows), "move": move,
                    "a_box": ch["mv"] if ch else None, "a_rank": rows.index(ch) if ch else None, "top": top,
                    "commit_left": len(out.get("commit") or [])})
        moves.append(move)

    t0 = time.perf_counter()
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, deck(MU[mu][0]), deck(MU[mu][1]), first_player=first, observer=obs)
        winner, turns, void = r["winner"], r["turns"], r["winner"] is None
    except DR.GameAborted as exc:
        winner, turns, void = None, None, str(exc)
    return {"mu": mu, "idx": idx, "seed": seed, "first": first, "me_first": first == "p1", "winner": winner,
            "me_won": None if winner is None else winner == "p1", "turns": turns, "void": void,
            "sec": round(time.perf_counter() - t0, 1), "points": pts, "moves": moves}


def play_order():
    """先攻と後攻の組の列（対面ごとに i と i＋60）。シャードは組の位置 mod n で配る。"""
    h = N_GAMES // 2
    return [((mu, j), (mu, h + j)) for mu in MU for j in range(h)]


def cmd_play(a):
    os.makedirs(a.out, exist_ok=True)
    si, sn = shard_of(a)
    gpath = os.path.join(a.out, f"games_shard{si}.jsonl" if a.shard else "games.jsonl")
    done = set(load_games(a.out))
    mus = set(a.mu.split(",")) if a.mu else set(MU)
    jobs = [j for n, pair in enumerate(play_order()) if n % sn == si for j in pair if j[0] in mus and j not in done]
    jobs = jobs[:a.games or None]
    print(f"PLAY shard={a.shard} todo {len(jobs)} (first {sum(1 for _, i in jobs if i < N_GAMES // 2)}) "
          f"(done {len(done)}) -> {gpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        for i, g in enumerate(pool.imap_unordered(play_one, jobs, chunksize=1)):
            with open(gpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] mu={g['mu']} idx={g['idx']} first={g['first']} turns={g['turns']} "
                  f"me_won={g['me_won']} sec={g['sec']} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 2. trace

def trace_one(g):
    E.engine()
    game = new_game(g["mu"], g["idx"])
    umap = {}
    rows = []
    h0 = json.loads(game.hidden_json())
    luid = {seat: h0["players"][seat]["leader"]["uuid"] for seat in ("p1", "p2")}
    for s, mv in enumerate(g["moves"]):
        h = json.loads(game.hidden_json())
        umap.update(KR.uuid_map(h))
        pend = json.loads(game.pending_json())
        name = pend["player_id"]
        man = h.get("manager") or {}
        r = {"step": s, "seat": name, "turn": int(game.turn_count or 0), "tp": game.turn_player(), "act": pend.get("action"),
             "msg": (pend.get("message") or "")[:60], "battle": bool(man.get("active_battle")), "st": KR.snap(h),
             "mv": KR.mv_sum(mv, umap)}
        if pend.get("action") in ("MAIN_ACTION",):
            legal = json.loads(game.legal_json(name))
            lg = [m for m in legal if m.get("action_type") in ("PLAY", "ACTIVATE_MAIN", "ATTACK")]
            r["legal"] = [KR.mv_sum(m, umap) for m in lg]
            if name == "p1":
                r["legal_raw"] = lg
        rows.append(r)
        BA.apply(game, name, mv)
        if game.winner() is not None:
            break
    return {"mu": g["mu"], "idx": g["idx"], "me_first": g["me_first"], "me_won": g["me_won"], "turns": g["turns"],
            "rows": rows, "umap": umap, "luid": luid}


def trace_path(a, mu):
    return os.path.join(a.trace_dir or a.out, f"trace_{mu}.jsonl")


def cmd_trace(a):
    games = load_games(a.out)
    for mu in MU:
        gs = [games[k] for k in sorted(games) if k[0] == mu]
        if not gs:
            continue
        path = trace_path(a, mu)
        with get_context("fork").Pool(a.workers) as pool, open(path, "w", encoding="utf-8") as f:
            for t in pool.imap(trace_one, gs, chunksize=2):
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print("TRACE", mu, len(gs), "->", path, flush=True)


# ---------------------------------------------------------------- 3. items

#: 記事の愛称 → カード番号（カード DB の名前とコストで引き当て・報告の表と同じ）。記事のデッキに無い札は持たない。
# 紫エネル（enel_1021）
E6 = {"OP15-118"}
OHM, SATORI, SHURA, PURIN = "OP15-061", "OP15-066", "OP15-067", "OP12-071"
HODEN, SABAKI, RAIJU, RAIRYU, BANRAI = "OP15-074", "OP15-075", "OP15-076", "OP15-077", "OP15-078"
LAW4, REIJU, PINK, GAMMA, GOMU_KAMINARI, KAMISAKE = "ST10-010", "OP12-063", "OP10-067", "OP05-077", "OP09-077", "OP13-076"
ENEL_ONE = {OHM, SATORI, SHURA, PURIN}
ENEL_REMOVAL = {SABAKI, RAIJU, RAIRYU, BANRAI, GAMMA, GOMU_KAMINARI, KAMISAKE}
# 青ロックス（rocks_*）
STROI, KYO, MARLON, STUSSY = "OP17-050", "OP17-045", "OP17-052", "OP17-054"
KAIDO4, JOHN, GLORI, LINLIN, NEWGATE, SHIKI, ROCKS10 = ("OP17-042", "OP17-044", "OP17-046", "OP17-049", "OP17-040",
                                                        "OP17-048", "OP17-118")
GANZUI, DREAM, BUCKIN = "OP17-043", "EB02-030", "OP08-051"
# 黒ルフィ（bluffy_1024）・赤黒サボ エルバフ型（sabo_*）
USOPP, GERZ, SANJI, ROBIN, BROOK, ZORO = "OP17-080", "OP17-081", "OP17-082", "OP17-087", "OP17-091", "OP17-095"
SAURO, LOKI, LUFFY8, CHOPPER, DOCK6, SUNNY = "OP17-089", "OP17-119", "OP17-093", "OP17-084", "OP15-088", "ST14-017"
LUFFY_TWO = {USOPP, GERZ, SANJI, ROBIN, BROOK, ZORO, "OP17-083", "ST31-003"}
# 緑ミホーク（mihawk_0917）
KANSEN, MIHAWK_L = "OP14-039", "OP14-020"
KIN1, PERONA1, OTAMA, DECKEN = "ST32-001", "OP12-034", "OP07-022", "OP06-033"
ODEN5, BONNEY5, KAWAMATSU, YASOPP5, LAW6, LAWBEPO, SHANKS10 = ("ST32-002", "OP12-118", "OP12-023", "OP17-031", "OP13-031",
                                                              "ST24-004", "OP17-022")
MH_ONE = {KIN1, PERONA1}
MH_FIVE = {ODEN5, BONNEY5, KAWAMATSU, YASOPP5}
# 紫カイドウ（kaido_1027・matchup_kr の番号を使う）
K6, Q1, Y3, H3, BK, RK, OK, L10, HIHO = KR.K6, KR.Q1, KR.Y3, KR.H3, KR.BK, KR.RK, KR.OK, KR.L10, KR.HIHO
ACCEL = [Q1, Y3, H3]

#: 項目 → (表の名前, 型)。型: resource＝ドン・リソースの読み／order＝展開の順番／target＝除去・攻撃の対象／
#: defense＝守り／lethal＝リーサル／mull＝マリガン・先後
TYPES = {"resource": "ドン・リソースの読み", "order": "展開の順番", "target": "除去・攻撃の対象", "defense": "守り",
         "lethal": "リーサル", "mull": "マリガン・先後"}
ITEMS = {
    "mull": ("マリガン（記事のキープ基準）", "mull"),
    "ideal": ("理想の展開（ターンごとに記事の札を出したか）", "order"),
    "atk_leader": ("攻撃の対象（キャラでなくリーダーを殴る）", "target"),
    "enel_kyo": ("凶を KO の除去の対象にしない", "target"),
    "enel_shiki": ("7シキが出たら最優先で処理する", "target"),
    "enel_hold": ("先攻 4 ターン目は除去を温存する", "resource"),
    "enel_banrai": ("万雷はカウンターとしても使う", "defense"),
    "luffy8_pick": ("8ルフィの展開先（ゲルズを避ける）", "order"),
    "loki_keep": ("相手のライフ 4 以上ではロキを殴らずに返す", "defense"),
    "shiki_keep": ("7シキを速攻で即レストにしない", "defense"),
    "shiki_rush": ("4 ターン目の 7シキは速攻でレストのキャラを取る", "target"),
    "mihawk_leader": ("リーダー効果はキャラを出してから使う", "resource"),
    "perona_t1": ("1ペローナを先攻 1 ターン目に出さない", "order"),
    "kr_leader": ("リーダー効果（ドンを戻して −2000）の時期", "resource"),
    "kr_k6bounce": ("6キングが返された後の組み立て", "order"),
    "kr_bk_keep": ("シキがいる局面でブロッカーカイドウを守りに残す", "defense"),
    "kr_rk_john": ("除去カイドウの対象（キャプテン・ジョン）", "target"),
    "kaido_banrai": ("万雷は序盤カウンターとして使う", "defense"),
}


def S(*xs):
    return [set([x]) if isinstance(x, str) else set(x) for x in xs]


def _hand(row):
    return row["st"][row["seat"]]["hand"]


def _opp_field(row):
    return row["st"]["p2" if row["seat"] == "p1" else "p1"]["field"]


def _my_field(row):
    return row["st"][row["seat"]]["field"]


def _enel_t1(row):
    """先 1・後 1: プリン・シュラ・6エネルが揃うならプリンだけ（シュラは次へ）、揃わないならサトリ＞ほかの 1 コスト。"""
    h = _hand(row)
    if PURIN in h and SHURA in h and any(c in h for c in E6):
        return S(PURIN)
    return S(SATORI, ENEL_ONE - {SATORI})


def _enel_t2(row):
    """先 2・後 2: 手札にシュラと 6エネルがあればシュラ（後 2 はシュラが無ければロー／レイジュ）。"""
    h = _hand(row)
    return S(SHURA) if SHURA in h and any(c in h for c in E6) else []


def _enel_s2(row):
    return _enel_t2(row) or S({LAW4, REIJU})


def _mirror_t3(row):
    """先 3: 相手が後 2 にキャプテン・ジョンを置いていたら 4カイドウ／グロリオーサ、でなければ 5リンリン。"""
    return S(KAIDO4, GLORI) if any(c[0] == JOHN for c in _opp_field(row)) else S(LINLIN)


def _luffy_f2(row):
    """先 2: 2ウソップ（次点 2サンジ・無ければ別の 2 コスト）。両方あれば 2サンジでウソップを捨てる。"""
    h = _hand(row)
    return S(SANJI) if SANJI in h and USOPP in h else S(USOPP, SANJI, LUFFY_TWO)


def _luffy_f3(row):
    """先 3: サニー号が出ていればドッキング6、無ければ 4サウロ。"""
    stage = row["st"][row["seat"]]["stage"]
    return S(DOCK6) if SUNNY in stage else S(SAURO, DOCK6)


def _luffy_s4(row):
    """後 4: 前のターンにロキが出ていれば 8ルフィ、でなければロキ。"""
    return S(LUFFY8) if any(c[0] == LOKI for c in _my_field(row)) else S(LOKI)


def _mh_f1(row):
    """先 1: 棺船か錦えもんがあるときだけ出す（ペローナは出さない）。"""
    return S(KANSEN, KIN1)


def _mh_s1(row):
    """後 1: 1ペローナ（ペローナと錦えもんが両方あるなら錦えもんだけ）。"""
    h = _hand(row)
    return S(KIN1) if KIN1 in h and PERONA1 in h else S(PERONA1, KIN1)


def _rs_f3(row):
    """青ロックス対赤黒サボの先 3: 相手の 2 コストが 2 枚以上ならガンズイ、でなければ 5リンリン（無ければもう一方）。"""
    n2 = sum(1 for c in _opp_field(row) if str(cname(c[0])[1]) == "2")
    return S(GANZUI, LINLIN) if n2 >= 2 else S(LINLIN, GANZUI)


ROCKS_MIRROR_IDEAL = {"first": {1: S(STROI), 2: S(KYO), 3: _mirror_t3, 4: S(SHIKI, NEWGATE), 5: S(SHIKI, STUSSY)},
                      "second": {1: S(KYO, STROI), 2: S(JOHN), 3: S(NEWGATE, LINLIN, GLORI), 4: S(SHIKI), 5: S(ROCKS10),
                                 6: S(ROCKS10, JOHN, STUSSY)},
                      "late": (6, S(ROCKS10))}
#: 赤黒サボ（エルバフ型）の理想の展開（青ロックス対赤黒サボの記事 §4: 2 コスト → 2 コスト 2 枚か 4サウロ → ロキ → 8ルフィ）
SABO_IDEAL = {"first": {1: S(LUFFY_TWO), 2: S(SAURO, LUFFY_TWO), 3: S(LOKI), 4: S(LUFFY8)},
              "second": {1: S(LUFFY_TWO), 2: S(SAURO, LUFFY_TWO), 3: S(LOKI), 4: S(LUFFY8)}}
#: 紫カイドウ（§7.63 の (3)・記事のデッキは 10リンリンを持つ）
KAIDO_IDEAL = {"first": {1: S(*ACCEL), 2: S(*ACCEL), 3: S(K6), 4: S(BK), 5: S(BK, OK, RK, L10)},
               "second": {1: S(*ACCEL), 2: S(*ACCEL), 3: S(K6), 4: S(BK), 5: S(BK, OK, RK, L10)},
               "late": (6, S(L10, OK, RK))}
#: 青ロックス側（紫カイドウの記事: 1〜2 シュトロイゼン・マーロン／ステューシー・凶、3 5リンリン・4 コスト 6000、5 シキ、6 10ロックス）
ROCKS_VS_KAIDO = {"first": {1: S({STROI, MARLON, STUSSY, KYO}), 2: S({STROI, MARLON, STUSSY, KYO}), 3: S(LINLIN, {KAIDO4, JOHN}),
                            5: S(SHIKI), 6: S(ROCKS10)},
                  "second": {1: S({STROI, MARLON, STUSSY, KYO}), 2: S({STROI, MARLON, STUSSY, KYO}), 3: S(LINLIN, {KAIDO4, JOHN}),
                             5: S(SHIKI), 6: S(ROCKS10)}}


def _kaido_mull(hand, first):
    nq = hand.count(Q1)
    if first:
        return (K6 in hand and (Q1 in hand or Y3 in hand)) or (HIHO in hand and any(x in hand for x in ACCEL)) or nq >= 2
    return K6 in hand or nq >= 2 or HIHO in hand


def _mh_mull(hand, first):
    n1 = sum(1 for c in hand if c in MH_ONE)
    big = MH_FIVE if first else MH_FIVE | {LAW6}
    return n1 >= 2 or (n1 >= 1 and any(c in hand for c in big))


#: 対面 → 設定。ideal は視点の側（p1）の理想の展開（自分の何ターン目 → 出す札の優先順。関数は盤面で決まる）・
#: opp_ideal は相手側（記事に書いてあれば）・mull は (手札, 先攻か) → 基準の内か（None＝測れない）。
CFG = {
    "enel_rocks": {
        "ideal": {"first": {1: _enel_t1, 2: _enel_t2, 3: S(E6), 4: S(E6), 5: S(REIJU, LAW4)},
                  "second": {1: _enel_t1, 2: _enel_s2, 3: S(E6), 4: S(REIJU, LAW4), 5: S(LAW4, REIJU)}},
        "opp_ideal": {"first": {1: S(KYO), 3: S(NEWGATE), 4: S(LINLIN, {MARLON, STUSSY}), 5: S(ROCKS10)},
                      "second": {1: S(KYO), 3: S(NEWGATE), 4: S(LINLIN, {MARLON, STUSSY}), 5: S(ROCKS10)}},
        "mull": lambda hand, first: any(c in hand for c in E6) and any(c in hand for c in ENEL_ONE),
        "items": ["mull", "ideal", "enel_kyo", "enel_shiki", "enel_hold", "enel_banrai"],
    },
    "bluffy_sabo": {
        "ideal": {"first": {1: S(SUNNY), 2: _luffy_f2, 3: _luffy_f3, 4: S(LOKI, DOCK6), 5: S(LUFFY8)},
                  "second": {1: S(USOPP, SANJI, LUFFY_TWO), 2: S(SAURO, ZORO), 3: S(LOKI, DOCK6), 4: _luffy_s4, 5: S(LUFFY8, LOKI)},
                  "late": (6, S(LUFFY8, LOKI))},
        "opp_ideal": SABO_IDEAL,
        "mull": lambda hand, first: USOPP in hand and ((DOCK6 in hand or LOKI in hand) if first else (SAURO in hand or LOKI in hand)),
        "items": ["mull", "ideal", "atk_leader", "luffy8_pick", "loki_keep"],
    },
    "rocks_mirror": {
        "ideal": ROCKS_MIRROR_IDEAL,
        "opp_ideal": ROCKS_MIRROR_IDEAL,
        "mull": lambda hand, first: (LINLIN in hand and any(c in hand for c in (SHIKI, GLORI, KAIDO4))) if first
        else (NEWGATE in hand and (JOHN in hand or SHIKI in hand)),
        "items": ["mull", "ideal", "shiki_keep"],
    },
    "mihawk_sabo": {
        "ideal": {"first": {1: _mh_f1, 2: S(PERONA1), 3: S(YASOPP5, ODEN5, BONNEY5), 4: S(LAW6), 5: S(LAW6)},
                  "second": {1: _mh_s1, 3: S(LAW6)},
                  "late": (6, S(SHANKS10))},
        "opp_ideal": SABO_IDEAL,
        "mull": _mh_mull,
        "items": ["mull", "ideal", "perona_t1", "atk_leader", "mihawk_leader"],
    },
    "rocks_sabo": {
        "ideal": {"first": {2: S(KYO, MARLON), 3: _rs_f3, 4: S(SHIKI), 5: S(SHIKI, STUSSY)},
                  "second": {1: S(KYO), 2: S(GLORI), 3: S(NEWGATE, GANZUI), 4: S(SHIKI), 5: S(ROCKS10)},
                  "late": (6, S(ROCKS10))},
        "opp_ideal": SABO_IDEAL,
        "mull": lambda hand, first: ((LINLIN in hand or GANZUI in hand) and SHIKI in hand) if first
        else (GLORI in hand and (NEWGATE in hand or GANZUI in hand)),
        "items": ["mull", "ideal", "shiki_rush"],
    },
    "kaido_rocks": {
        "ideal": KAIDO_IDEAL,
        "opp_ideal": ROCKS_VS_KAIDO,
        "mull": _kaido_mull,
        "items": ["mull", "ideal", "kr_leader", "kr_k6bounce", "kr_bk_keep", "kr_rk_john", "kaido_banrai"],
    },
}
DECIDED = 0.9


def cname(cid):
    return KR.cname(cid)


def is_char(cid):
    return cname(cid)[2] == "キャラクター"


def own_turn(t, seat, me_first):
    first = (seat == "p1") == me_first
    return (t + 1) // 2 if first else t // 2


def kv(p):
    """記録の点の、視点の側（p1）から見た評価（打った手の根の値）。"""
    if not p.get("top"):
        return None
    r = p["top"][p["a_rank"]] if p.get("a_rank") is not None and p["a_rank"] < len(p["top"]) else p["top"][0]
    return r["q"] if p["seat"] == "p1" else -r["q"]


def attack_of(mv):
    """手（原始手か箱）がアタック（の始まり）なら (アタックするカードの uuid, 対象の uuid)。"""
    if not mv:
        return None
    p = mv.get("payload") or {}
    at = mv.get("action_type")
    if at == "ATTACK" or (at == "DON_BOX" and p.get("target_ids")):
        return p.get("uuid"), (p.get("target_ids") or [None])[0]
    if at == "SETUP_BOX":
        return attack_of(p.get("base"))
    return None


def retarget(mv, tgt):
    """アタックの手の対象だけを差し替えた手。"""
    m = json.loads(json.dumps(mv))
    p = m.get("payload") or {}
    if m.get("action_type") == "SETUP_BOX":
        p["base"] = retarget(p["base"], tgt)
    else:
        p["target_ids"] = [tgt]
    return m


def ideal_for(table, ot, row):
    if table is None:
        return []
    w = table.get(ot)
    if w is None and table.get("late") and ot >= table["late"][0]:
        w = table["late"][1]
    if callable(w):
        w = w(row)
    return w or []


def analyse_game(g, t):
    """1 局の項目ごとの場面と、比べる点の候補。"""
    mu = g["mu"]
    cfg = CFG[mu]
    rows, pts, umap = t["rows"], g["points"], t["umap"]
    mf = g["me_first"]
    res = {"mu": mu, "idx": g["idx"], "me_first": mf, "won": g["me_won"], "turns": g["turns"], "void": bool(g["void"]),
           "scenes": [], "plays": {"p1": {}, "p2": {}}}
    cands = []

    def scene(item, agree, ot=None, **extra):
        res["scenes"].append(dict({"item": item, "agree": agree, "own_turn": ot}, **extra))

    def cand(item, s, b, why, ot=None):
        p = pts[s]
        cands.append({"mu": mu, "idx": g["idx"], "step": s, "item": item, "seat": "p1", "turn": p["turn"], "own_turn": ot,
                      "a": p["a_box"] or p["move"], "b": b, "why": why, "qa": kv(p)})

    def main_point_for(s, pred):
        """step s の原始手を始めた main の判断点（pred(箱) が真の、s 以前で最も近い点）。"""
        for q in range(s, -1, -1):
            p = pts[q]
            if p["seat"] != "p1":
                break
            if p["kind"] == "main" and pred(p["a_box"] or p["move"]):
                return q
        return None

    by_turn = {"p1": {}, "p2": {}}
    for r in rows:
        if r["tp"] in by_turn and r["turn"] >= 1:
            by_turn[r["tp"]].setdefault(r["turn"], []).append(r)
    # マリガン
    for r in rows:
        if r["act"] == "MULLIGAN" and r["seat"] == "p1":
            hand = r["st"]["p1"]["hand"]
            crit = cfg["mull"](hand, mf)
            keep = r["mv"]["at"] == "KEEP_HAND"
            res["mull"] = {"hand": hand, "crit": crit, "keep": keep}
            if crit is not None:
                scene("mull", keep == bool(crit), crit=bool(crit), keep=keep)
                if keep != bool(crit):
                    cand("mull", r["step"], {"kind": "game", "action_type": "KEEP_HAND" if crit else "MULLIGAN", "payload": {}},
                         "keep" if crit else "mulligan")
            break
    # 理想の展開（両席。比べる点は視点の側だけ）
    for seat in ("p1", "p2"):
        table = (cfg["ideal"] if seat == "p1" else cfg.get("opp_ideal")) or {}
        sub = table.get("first" if (seat == "p1") == mf else "second")
        if sub is not None and table.get("late"):
            sub = dict(sub, late=table["late"])
        res["ideal_" + seat] = {}
        for tn in sorted(by_turn[seat]):
            rs = by_turn[seat][tn]
            ot = own_turn(tn, seat, mf)
            plays = [r["mv"]["card"] for r in rs if r["seat"] == seat and r["mv"] and r["mv"]["at"] == "PLAY"]
            res["plays"][seat][ot] = plays
            mains = [r for r in rs if r["seat"] == seat and r["act"] == "MAIN_ACTION"]
            if not mains or sub is None:
                continue
            want = ideal_for(sub, ot, mains[0])
            if not want:
                continue
            legal = {}
            for r in mains:
                for k, m in enumerate(r.get("legal") or []):
                    if m["at"] == "PLAY" and m.get("card") not in legal and any(m.get("card") in w for w in want):
                        legal[m["card"]] = (r["step"], m["uuid"])
            tops = [w for w in want if any(c in legal for c in w)]
            if not tops:
                continue
            top = tops[0]
            agree = any(c in top for c in plays)
            agree_any = any(any(c in w for w in want) for c in plays)
            pick = next(c for c in legal if c in top)
            res["ideal_" + seat][ot] = {"want": pick, "agree": agree, "agree_any": agree_any}
            if seat == "p1":
                scene("ideal", agree, ot, want=pick, agree_any=agree_any)
                if not agree:
                    s0, uu = min((legal[c] for c in legal if c in top), key=lambda x: x[0])
                    b = {"kind": "game", "action_type": "PLAY", "payload": {"uuid": uu}}
                    if pts[s0]["kind"] == "main" and BA.key(pts[s0]["a_box"] or pts[s0]["move"]) != BA.key(b):
                        cand("ideal", s0, b, f"T{ot}:{pick}", ot)
    items = cfg["items"]
    opp_leader = t["luid"]["p2"]
    # 攻撃の対象（キャラでなくリーダー）
    if "atk_leader" in items:
        for r in rows:
            if not (r["seat"] == "p1" and r["tp"] == "p1" and r["mv"] and r["mv"]["at"] == "ATTACK"):
                continue
            att = r["mv"].get("uuid")
            to_leader = (r["mv"].get("tgt") or [None])[0] == umap.get(opp_leader)
            ot = own_turn(r["turn"], "p1", mf)
            scene("atk_leader", to_leader, ot, attacker=r["mv"].get("card"), target=(r["mv"].get("tgt") or [None])[0],
                  opp_life=r["st"]["p2"]["life"])
            if not to_leader:
                sp = main_point_for(r["step"], lambda m: (attack_of(m) or (None,))[0] == att)
                if sp is not None:
                    a = pts[sp]["a_box"] or pts[sp]["move"]
                    cand("atk_leader", sp, retarget(a, opp_leader), f"{r['mv'].get('card')}->leader", ot)
    # 紫エネル
    if mu == "enel_rocks":
        for i, r in enumerate(rows):
            if r["seat"] != "p1" or not r["mv"]:
                continue
            if r["mv"]["at"] == "PLAY" and r["mv"].get("card") in (SABAKI, GOMU_KAMINARI):
                nm = cname(r["mv"]["card"])[0]
                sel = next((x for x in rows[i + 1:i + 6] if x["seat"] == "p1" and x["act"] == "SEARCH_AND_SELECT"
                            and nm in x["msg"] and x["mv"].get("sel") and x["mv"]["sel"][0] != "?"
                            and x["mv"]["sel"][0] in [c[0] for c in x["st"]["p2"]["field"]]), None)
                if sel:
                    kyo_on = any(c[0] == KYO for c in sel["st"]["p2"]["field"])
                    if kyo_on:
                        scene("enel_kyo", sel["mv"]["sel"][0] != KYO, own_turn(r["turn"], "p1", mf), target=sel["mv"]["sel"][0])
            if r["mv"]["at"] == "SELECT_COUNTER" and r["mv"].get("card") == BANRAI:
                scene("enel_banrai", True, own_turn(r["turn"], "p1", mf), use="counter")
            if r["mv"]["at"] == "PLAY" and r["mv"].get("card") == BANRAI:
                scene("enel_banrai", False, own_turn(r["turn"], "p1", mf), use="main")
        seen_shiki = set()
        for tn, rs in sorted(by_turn["p1"].items()):
            ot = own_turn(tn, "p1", mf)
            mains = [r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"]
            if not mains:
                continue
            if mf and ot == 4:
                used = [r["mv"]["card"] for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY"
                        and r["mv"].get("card") in ENEL_REMOVAL]
                scene("enel_hold", not used, ot, used=used)
            # 7シキが出て最初の自分のターン（1 体につき 1 場面）
            sh0 = {c[4] for c in mains[0]["st"]["p2"]["field"] if c[0] == SHIKI} - seen_shiki
            seen_shiki |= sh0
            if sh0:
                nxt = next((r for r in rows if r["turn"] > tn), None)
                left = nxt is not None and not (sh0 & {c[4] for c in nxt["st"]["p2"]["field"]})
                scene("enel_shiki", left, ot)
    # 黒ルフィ
    if mu == "bluffy_sabo":
        n8 = 0
        for i, r in enumerate(rows):
            if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == LUFFY8:
                n8 += 1
                sel = next((x for x in rows[i + 1:i + 8] if x["seat"] == "p1" and x["act"] == "SEARCH_AND_SELECT"
                            and "ルフィ" in x["msg"]), None)
                if not sel:
                    continue
                pick = (sel["mv"].get("sel") or [None])[0]
                trash = sel["st"]["p1"]["trash"]
                good = [c for c in (USOPP, BROOK, ROBIN) if c in trash]
                if not good:
                    continue
                ot = own_turn(r["turn"], "p1", mf)
                scene("luffy8_pick", pick != GERZ, ot, pick=pick, nth=n8, opp_hand=len(sel["st"]["p2"]["hand"]))
                if pick == GERZ and pts[sel["step"]]["kind"] == "window":
                    alt = [x["mv"] for x in pts[sel["step"]]["top"]
                           if [umap.get(u) for u in (x["mv"].get("payload") or {}).get("selected_uuids") or []][:1]
                           and umap.get(((x["mv"].get("payload") or {}).get("selected_uuids") or [None])[0]) in good]
                    if alt:
                        cand("luffy8_pick", sel["step"], alt[0], "gerz", ot)
        for r in rows:
            if r["seat"] == "p1" and r["tp"] == "p1" and r["mv"] and r["mv"]["at"] == "ATTACK" and r["mv"].get("card") == LOKI:
                if r["st"]["p2"]["life"] < 4:
                    continue
                ot = own_turn(r["turn"], "p1", mf)
                att = r["mv"]["uuid"]
                scene("loki_keep", False, ot, opp_life=r["st"]["p2"]["life"])
                sp = main_point_for(r["step"], lambda m: (attack_of(m) or (None,))[0] == att)
                if sp is not None:
                    alt = [x["mv"] for x in pts[sp]["top"] if (attack_of(x["mv"]) or (None,))[0] != att]
                    if alt:
                        cand("loki_keep", sp, alt[0], "loki_attack", ot)
        # ロキが場にいて相手のライフ 4 以上の自分のターンで殴らなかった（＝記事どおり）場面も数える
        for tn, rs in sorted(by_turn["p1"].items()):
            m0 = next((r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"
                       and any(m["at"] == "ATTACK" and m.get("card") == LOKI for m in r.get("legal") or [])), None)
            if m0 and m0["st"]["p2"]["life"] >= 4 and not any(
                    r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "ATTACK" and r["mv"].get("card") == LOKI for r in rs):
                scene("loki_keep", True, own_turn(tn, "p1", mf), opp_life=m0["st"]["p2"]["life"])
    # 青ロックス（シキ）
    if mu in ("rocks_mirror", "rocks_sabo"):
        for tn, rs in sorted(by_turn["p1"].items()):
            ot = own_turn(tn, "p1", mf)
            played = [r for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == SHIKI]
            if not played:
                continue
            su = played[0]["mv"]["uuid"]
            att = [r for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "ATTACK" and r["mv"].get("uuid") == su]
            r10 = any(r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == ROCKS10 for r in rs)
            if mu == "rocks_mirror":
                if r10:
                    continue
                scene("shiki_keep", not att, ot)
                if att:
                    sp = main_point_for(att[0]["step"], lambda m: (attack_of(m) or (None,))[0] == su)
                    if sp is not None:
                        alt = [x["mv"] for x in pts[sp]["top"] if (attack_of(x["mv"]) or (None,))[0] != su]
                        if alt:
                            cand("shiki_keep", sp, alt[0], "shiki_rush", ot)
            else:
                if ot != 4:
                    continue
                # 速攻でレストのキャラを殴れた点（シキを出した後の最初のメイン）
                m1 = next((r for r in rs if r["step"] > played[0]["step"] and r["seat"] == "p1" and r["act"] == "MAIN_ACTION"
                           and any(m.get("action_type") == "ATTACK" and (m.get("payload") or {}).get("uuid") == su
                                   and (m.get("payload") or {}).get("target_ids") != [opp_leader] for m in r.get("legal_raw") or [])),
                          None)
                if m1 is None:
                    continue
                hit = [r for r in att if (r["mv"].get("tgt") or [None])[0] != umap.get(opp_leader)]
                scene("shiki_rush", bool(hit), ot, attacked=bool(att))
                if not hit and pts[m1["step"]]["kind"] == "main":
                    b = next(m for m in m1["legal_raw"] if m.get("action_type") == "ATTACK"
                             and (m.get("payload") or {}).get("uuid") == su and (m.get("payload") or {}).get("target_ids") != [opp_leader])
                    b = {"kind": "game", "action_type": "ATTACK", "payload": b.get("payload")}
                    if BA.key(pts[m1["step"]]["a_box"] or pts[m1["step"]]["move"]) != BA.key(b):
                        cand("shiki_rush", m1["step"], b, "shiki_char", ot)
    # 緑ミホーク（リーダー効果）
    if mu == "mihawk_sabo":
        for r in rows:
            if not (r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "ACTIVATE_MAIN" and r["mv"].get("card") == MIHAWK_L):
                continue
            ot = own_turn(r["turn"], "p1", mf)
            plays = [m for m in r.get("legal_raw") or [] if m.get("action_type") == "PLAY"
                     and is_char(umap.get((m.get("payload") or {}).get("uuid")))]
            scene("mihawk_leader", not plays, ot, could_play=[umap.get(m["payload"]["uuid"]) for m in plays])
            if plays and pts[r["step"]]["kind"] == "main":
                best = max(plays, key=lambda m: int(cname(umap.get(m["payload"]["uuid"]))[1] or 0))
                cand("mihawk_leader", r["step"], {"kind": "game", "action_type": "PLAY", "payload": best["payload"]}, "play_first", ot)
    # 緑ミホーク（1ペローナを先攻 1 ターン目に出さない・2 枚以上なら出してもよい）
    if mu == "mihawk_sabo" and mf:
        rs = by_turn["p1"].get(1, [])
        m0 = next((r for r in rs if r["seat"] == "p1" and r["act"] == "MAIN_ACTION"
                   and any(m["at"] == "PLAY" and m.get("card") == PERONA1 for m in r.get("legal") or [])), None)
        if m0 and _hand(m0).count(PERONA1) < 2:
            pl = [r for r in rs if r["seat"] == "p1" and r["mv"] and r["mv"]["at"] == "PLAY" and r["mv"].get("card") == PERONA1]
            scene("perona_t1", not pl, 1)
            if pl:
                pu = pl[0]["mv"]["uuid"]
                sp = main_point_for(pl[0]["step"], lambda m: (m or {}).get("action_type") == "PLAY"
                                    and (m.get("payload") or {}).get("uuid") == pu)
                if sp is not None:
                    alt = [x["mv"] for x in pts[sp]["top"] if (x["mv"].get("payload") or {}).get("uuid") != pu]
                    if alt:
                        cand("perona_t1", sp, alt[0], "perona_t1", 1)
    # 紫カイドウ（§7.63 の道具の項目をそのまま使う）
    if mu == "kaido_rocks":
        gk = dict(g, kaido_first=mf, kaido_won=g["me_won"])
        kr, kc = KR.analyse_game(gk, t)
        for x in kr["leader"]:
            if x["article"] is not None:
                scene("kr_leader", x["used"] == x["article"], x["own_turn"], rule=x["rule"], used=x["used"])
        for x in kr["k6_bounce"]:
            if x.get("next_own_turn"):
                scene("kr_k6bounce", x["agree"], x["next_own_turn"], replay_k6=x["replay_k6"])
        for x in kr["bk_vs_shiki"]:
            scene("kr_bk_keep", not x["attacked"], x["own_turn"])
        for x in kr["rk"]:
            if x["john"] and x["used"]:
                scene("kr_rk_john", x["target"] == KR.JOHN, x["own_turn"], target=x["target"])
        imap = {"5": "kr_leader", "6": "kr_k6bounce", "9": "kr_bk_keep", "10": "kr_rk_john"}
        for c in kc:
            if c["item"] in imap and c["b"] is not None:
                cands.append(dict(c, mu=mu, item=imap[c["item"]]))
        for r in rows:
            if r["seat"] != "p1" or not r["mv"]:
                continue
            d = r["st"]["p1"]["don"]
            early = d[0] + d[1] + d[2] <= 6
            if r["mv"]["at"] == "SELECT_COUNTER" and r["mv"].get("card") == BANRAI and early:
                scene("kaido_banrai", True, own_turn(r["turn"], "p1", mf), use="counter")
            if r["mv"]["at"] == "PLAY" and r["mv"].get("card") == BANRAI and early:
                scene("kaido_banrai", False, own_turn(r["turn"], "p1", mf), use="main")
    # 評価の推移と相手のターンの前後（厄介な札）
    res["values"] = [(p["step"], p["turn"], kv(p)) for p in pts if p["kind"] in ("main", "window") and kv(p) is not None]
    mains = [(p["step"], p["turn"], kv(p)) for p in pts if p["seat"] == "p1" and p["kind"] == "main"
             and p["dtype"] != "mulligan" and kv(p) is not None]
    res["opp_turns"] = []
    for tn in sorted(by_turn["p2"]):
        bef = [v for s_, t_, v in mains if t_ < tn]
        aft = [v for s_, t_, v in mains if t_ > tn]
        if not bef or not aft:
            continue
        cards = sorted({r["mv"]["card"] for r in by_turn["p2"][tn] if r["seat"] == "p2" and r["mv"] and r["mv"]["at"] == "PLAY"})
        res["opp_turns"].append({"turn": tn, "own_turn": own_turn(tn, "p2", mf), "before": bef[-1], "after": aft[0], "cards": cards})
    return res, cands


def _rank(mu, idx, step):
    return E.search_seed(int(MU[mu][2] + idx) ^ 0x3A7C_4D05, int(step), "p1")


def cmd_items(a):
    games = load_games(a.out)
    allc = []
    pool = []
    for mu in MU:
        tp = trace_path(a, mu)
        if not os.path.exists(tp):
            continue
        traces = {t["idx"]: t for t in read_jsonl(tp)}
        per, cands = [], []
        for k in sorted(games):
            if k[0] != mu or k[1] not in traces:
                continue
            r, c = analyse_game(games[k], traces[k[1]])
            per.append(r)
            cands += c
        with open(os.path.join(a.out, f"items_{mu}.jsonl"), "w", encoding="utf-8") as f:
            for r in per:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 比べる点: 項目ごとに局の記録だけで決まる順位で並べ、項目を順に回して上限まで（勝負の決まった局面は外す）
        by = {}
        for c in cands:
            if c["b"] is not None and (c["qa"] is None or abs(c["qa"]) < DECIDED):
                by.setdefault(c["item"], []).append(c)
        for k in by:
            by[k].sort(key=lambda c: _rank(c["mu"], c["idx"], c["step"]))
        quota = {"mull": 6}
        order = [it for it in CFG[mu]["items"] if it != "mull"] + ["mull"]
        used = {k: 0 for k in order}
        sel = []
        while len(sel) < a.cap and any(used[k] < min(len(by.get(k, [])), quota.get(k, 10 ** 6)) for k in order):
            for k in order:
                if len(sel) >= a.cap:
                    break
                if used[k] < min(len(by.get(k, [])), quota.get(k, 10 ** 6)):
                    sel.append(by[k][used[k]])
                    used[k] += 1
        pool += sel
        allc += cands
        print("ITEMS", mu, "games", len(per), "cands", {k: len(v) for k, v in by.items()},
              "decided", sum(1 for c in cands if c["b"] is not None and c["qa"] is not None and abs(c["qa"]) >= DECIDED),
              "pool", used, flush=True)
    ppath = os.path.join(a.out, "points_pool.jsonl")
    if os.path.exists(ppath) and not a.repool:
        ppath = os.path.join(a.out, "points_pool_latest.jsonl")
    with open(ppath, "w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "cands_all.jsonl"), "w", encoding="utf-8") as f:
        for c in allc:
            f.write(json.dumps({k: v for k, v in c.items() if k not in ("a", "b")}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 4. judge

def _init_judge(k):
    E.engine()
    BA._G["ship"] = BA.ship_spec(NET)
    _G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    _G["k"] = int(k)


def judge_point(job):
    g, c = job
    t0 = time.time()
    row = dict(c, ok=False, fail=None)
    try:
        game = replay(g, c["step"])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = f"replay: {type(exc).__name__}"
        return row
    pend = json.loads(game.pending_json())
    if pend.get("player_id") != "p1" or int(game.turn_count or 0) != c["turn"]:
        row["fail"] = "seat/turn"
        return row
    a, b = c["a"], c["b"]
    row["a_desc"], row["b_desc"] = BA.desc(game, a), BA.desc(game, b)
    p = g["points"][c["step"]]
    tot = sum(t["n"] for t in p["top"]) or 1.0

    def root(mv):
        hit = [t for t in p["top"] if BA.key(t["mv"]) == BA.key(mv)] or [t for t in p["top"] if BA.pkey(t["mv"]) == BA.pkey(mv)]
        return {"n": hit[0]["n"], "share": round(hit[0]["n"] / tot, 4), "q": hit[0]["q"], "p": hit[0].get("p")} if hit else None
    row["root_a"], row["root_b"] = root(a), root(b)
    wa, wb, errs = [], [], {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(MU[g["mu"]][2] + g["idx"], c["step"], OFFSET + i)
        ra, ea = BA.rollout_ship(game, "p1", a, ws, _G["ref"], BA._G["ship"], both=True)
        rb, eb = BA.rollout_ship(game, "p1", b, ws, _G["ref"], BA._G["ship"], both=True)
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    # paired は b − a ＝「記事の手 − CPU の手」
    row["ref"] = dict(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None)
    row["ok"] = row["ref"]["diff"] is not None
    if not row["ok"]:
        row["fail"] = "no_pairs"
    row["sec"] = round(time.time() - t0, 1)
    return row


def jkey(x):
    return (x["mu"], x["idx"], x["step"], x["item"])


def cmd_judge(a):
    games = load_games(a.out)
    pool = read_jsonl(os.path.join(a.out, "points_pool.jsonl"))
    si, sn = shard_of(a)
    jobs = [c for n, c in enumerate(pool) if n % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"judge_points_shard{si}.jsonl" if a.shard else "judge_points.jsonl")
    done = set(read_glob(os.path.join(a.out, "judge_points*.jsonl"), jkey))
    jobs = [(games[(c["mu"], c["idx"])], c) for c in jobs if jkey(c) not in done]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool_:
        for i, r in enumerate(pool_.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ref = r.get("ref") or {}
            print(f"[{i + 1}/{len(jobs)}] mu={r['mu']} idx={r['idx']} step={r['step']} item={r['item']} ok={r['ok']} "
                  f"diff={ref.get('diff')} ci={ref.get('ci')} sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 5. summary

rate, mean_ci, top_counts = KR.rate, KR.mean_ci, KR.top_counts
def summarize_mu(a, mu, J):
    per = read_jsonl(os.path.join(a.out, f"items_{mu}.jsonl"))
    ok = [r for r in per if not r["void"] and r["won"] is not None]
    S = {"name": MU[mu][3], "decks": MU[mu][:2], "games": len(per), "void": len(per) - len(ok),
         "winrate": {"all": rate([r["won"] for r in ok]), "first": rate([r["won"] for r in ok if r["me_first"]]),
                     "second": rate([r["won"] for r in ok if not r["me_first"]]),
                     "turns_mean": round(sum(r["turns"] for r in ok) / max(1, len(ok)), 2)}}
    # マリガン
    mm = [r for r in ok if r.get("mull")]
    S["mulligan"] = {}
    for tag, sub in (("first", [r for r in mm if r["me_first"]]), ("second", [r for r in mm if not r["me_first"]])):
        S["mulligan"][tag] = {"cpu_keep": rate([r["mull"]["keep"] for r in sub]),
                              "win_keep": rate([r["won"] for r in sub if r["mull"]["keep"]]),
                              "win_mull": rate([r["won"] for r in sub if not r["mull"]["keep"]])}
        if sub and sub[0]["mull"]["crit"] is not None:
            S["mulligan"][tag].update({
                "agree": rate([r["mull"]["keep"] == r["mull"]["crit"] for r in sub]),
                "crit_in": rate([r["mull"]["crit"] for r in sub]),
                "keep_when_in": rate([r["mull"]["keep"] for r in sub if r["mull"]["crit"]]),
                "keep_when_out": rate([r["mull"]["keep"] for r in sub if not r["mull"]["crit"]]),
                "win_in": rate([r["won"] for r in sub if r["mull"]["crit"]]),
                "win_out": rate([r["won"] for r in sub if not r["mull"]["crit"]])})
    # 理想の展開（両席）
    for seat in ("p1", "p2"):
        key = "ideal_" + seat
        S[key] = {}
        for tag, sub in (("first", [r for r in ok if r["me_first"] == (seat == "p1")]),
                         ("second", [r for r in ok if r["me_first"] != (seat == "p1")])):
            rows = {}
            for ot in range(1, 9):
                k = str(ot)
                app = [r[key][k] for r in sub if k in r[key]]
                reached = [r for r in sub if k in r["plays"][seat]]
                rows[k] = {"reached": len(reached), "applicable": len(app), "agree": rate([x["agree"] for x in app]),
                           "agree_any": rate([x["agree_any"] for x in app]), "want": top_counts([[x["want"]] for x in app], 3),
                           "win_agree": rate([r["won"] for r in sub if k in r[key] and r[key][k]["agree"]]),
                           "win_disagree": rate([r["won"] for r in sub if k in r[key] and not r[key][k]["agree"]]),
                           "plays_all": top_counts([r["plays"][seat][k] for r in reached]),
                           "plays_won": top_counts([r["plays"][seat][k] for r in reached if r["won"] == (seat == "p1")]),
                           "none_played": rate([not r["plays"][seat][k] for r in reached])}
            on = [r for r in sub if r[key] and all(r[key][k]["agree"] for k in r[key] if int(k) <= 5)]
            S[key][tag] = {"by_turn": rows, "on_track": rate([r in on for r in sub if r[key]]),
                           "win_on": rate([r["won"] for r in on]), "win_off": rate([r["won"] for r in sub if r[key] and r not in on])}
    # 項目ごと（場面の数・一致・一致した／しなかった局の勝率）
    S["items"] = {}
    for it in CFG[mu]["items"]:
        sc = [(r, x) for r in ok for x in r["scenes"] if x["item"] == it]
        S["items"][it] = {"name": ITEMS[it][0], "type": ITEMS[it][1], "scenes": len(sc),
                          "per_game": round(len(sc) / max(1, len(ok)), 3),
                          "agree": rate([x["agree"] for _, x in sc]),
                          "win_agree": rate([r["won"] for r, x in sc if x["agree"]]),
                          "win_disagree": rate([r["won"] for r, x in sc if not x["agree"]])}
    # 項目ごとの細目
    def sc_of(it):
        return [x for r in ok for x in r["scenes"] if x["item"] == it]
    det = {}
    if mu == "bluffy_sabo":
        det["luffy8_picks"] = top_counts([[x["pick"]] for x in sc_of("luffy8_pick") if x.get("pick")], 6)
        det["atk_targets"] = top_counts([[x["target"]] for x in sc_of("atk_leader") if x.get("target")], 6)
    if mu == "mihawk_sabo":
        det["atk_targets"] = top_counts([[x["target"]] for x in sc_of("atk_leader") if x.get("target")], 6)
        det["leader_by_turn"] = top_counts([[str(x["own_turn"])] for x in sc_of("mihawk_leader")], 8)
        det["leader_could_play"] = top_counts([x["could_play"] for x in sc_of("mihawk_leader")], 6)
    if mu == "enel_rocks":
        det["kyo_targets"] = top_counts([[x["target"]] for x in sc_of("enel_kyo")], 6)
        det["banrai_by_turn"] = {u: top_counts([[str(x["own_turn"])] for x in sc_of("enel_banrai") if x["use"] == u], 8)
                                 for u in ("counter", "main")}
        det["hold_used"] = top_counts([x["used"] for x in sc_of("enel_hold")], 6)
    S["detail"] = det
    # 厄介な札（相手の札が出たターンの前後で動いた視点の側の評価。同じ「何ターン目」の平均との差）
    OT = [x for r in ok for x in r["opp_turns"] if abs(x["before"]) < DECIDED]
    bt = {}
    for x in OT:
        bt.setdefault(min(x["own_turn"], 8), []).append(x["after"] - x["before"])
    base = {k: sum(v) / len(v) for k, v in bt.items()}
    oe = {}
    for x in OT:
        for c in x["cards"]:
            oe.setdefault(c, []).append(x["after"] - x["before"] - base[min(x["own_turn"], 8)])
    S["opp_cards"] = sorted([[c, cname(c)[0], len(v)] + [mean_ci(v)["mean"], mean_ci(v)["ci"]] for c, v in oe.items() if len(v) >= 8],
                            key=lambda x: x[3])
    # 判定
    Jm = [x for x in J if x["mu"] == mu]
    S["judge"] = {"points": len(Jm), "all": mean_ci([x["ref"]["diff"] for x in Jm]), "by_item": {}}
    for it in sorted({x["item"] for x in Jm}):
        sub = [x for x in Jm if x["item"] == it]
        S["judge"]["by_item"][it] = dict(mean_ci([x["ref"]["diff"] for x in sub]),
                                        article_better=sum(1 for x in sub if x["ref"]["ci"][0] is not None and x["ref"]["ci"][0] > 0),
                                        cpu_better=sum(1 for x in sub if x["ref"]["ci"][1] is not None and x["ref"]["ci"][1] < 0))
    S["judge"]["rows"] = sorted([{"idx": x["idx"], "step": x["step"], "item": x["item"], "why": x["why"], "own_turn": x.get("own_turn"),
                                  "a": x["a_desc"], "b": x["b_desc"], "diff": x["ref"]["diff"], "ci": x["ref"]["ci"],
                                  "wa": x["ref"]["wa"], "wb": x["ref"]["wb"], "root_a": x.get("root_a"), "root_b": x.get("root_b"),
                                  "qa": x.get("qa")} for x in Jm], key=lambda x: (x["item"], -(x["diff"] or 0)))
    cands = [c for c in read_jsonl(os.path.join(a.out, "cands_all.jsonl")) if c["mu"] == mu]
    S["cands"] = {it: {"all": len([c for c in cands if c["item"] == it]),
                       "decided": len([c for c in cands if c["item"] == it and c["qa"] is not None and abs(c["qa"]) >= DECIDED])}
                  for it in sorted({c["item"] for c in cands})}
    return S


def cmd_summary(a):
    J = [x for x in read_glob(os.path.join(a.out, "judge_points*.jsonl"), jkey).values() if x["ok"]]
    out = {"per_mu": {}}
    for mu in MU:
        if os.path.exists(os.path.join(a.out, f"items_{mu}.jsonl")):
            out["per_mu"][mu] = summarize_mu(a, mu, J)
    # 横断のまとめ: 項目ごとに 1 局あたりの「記事とずれた場面」の数 × 判定の平均の差 ＝ 1 局あたりの見込みの損
    rowsX = []
    for mu, S in out["per_mu"].items():
        for it, x in S["items"].items():
            j = S["judge"]["by_item"].get(it) or {}
            dis = (x["agree"]["n"] - x["agree"]["k"]) / max(1, S["games"] - S["void"]) if x["agree"]["n"] else 0.0
            rowsX.append({"mu": mu, "mu_name": S["name"], "item": it, "name": x["name"], "type": x["type"],
                          "scenes_per_game": x["per_game"], "agree": x["agree"]["rate"], "disagree_per_game": round(dis, 3),
                          "judged": j.get("n", 0), "diff": j.get("mean"), "ci": j.get("ci"),
                          "article_better": j.get("article_better", 0), "cpu_better": j.get("cpu_better", 0)})
    for r in rowsX:
        r["sig_article"] = bool(r["ci"] and r["ci"][0] is not None and r["ci"][0] > 0)
        r["sig_cpu"] = bool(r["ci"] and r["ci"][1] is not None and r["ci"][1] < 0)
        r["loss_per_game"] = round(r["disagree_per_game"] * r["diff"], 4) if r["diff"] is not None else None
    out["cross_items"] = sorted(rowsX, key=lambda r: -(r["loss_per_game"] or -9))
    ty = {}
    for r in rowsX:
        t = ty.setdefault(r["type"], {"type": r["type"], "name": TYPES[r["type"]], "items": 0, "sig_article": 0, "sig_cpu": 0,
                                      "disagree_per_game": [], "diffs": [], "loss_per_game": 0.0, "mus": set()})
        t["items"] += 1
        t["sig_article"] += int(r["sig_article"])
        t["sig_cpu"] += int(r["sig_cpu"])
        t["disagree_per_game"].append(r["disagree_per_game"])
        if r["diff"] is not None:
            t["diffs"] += [r["diff"]] * max(1, r["judged"])
            if r["sig_article"]:
                t["mus"].add(r["mu"])
        t["loss_per_game"] += max(0.0, r["loss_per_game"] or 0.0)
    for t in ty.values():
        t["mean_diff"] = round(sum(t["diffs"]) / len(t["diffs"]), 4) if t["diffs"] else None
        t["disagree_per_game"] = round(sum(t["disagree_per_game"]), 3)
        t["loss_per_game"] = round(t["loss_per_game"], 4)
        t["mus"] = sorted(t["mus"])
        del t["diffs"]
    out["cross_types"] = sorted(ty.values(), key=lambda t: -t["loss_per_game"])
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for mu, S in out["per_mu"].items():
        print(mu, S["games"], S["void"], S["winrate"]["first"]["rate"], S["winrate"]["second"]["rate"], flush=True)
    for t in out["cross_types"]:
        print(t["name"], t["loss_per_game"], t["sig_article"], t["mean_diff"], flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["play", "trace", "items", "judge", "summary", "story"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--mu", default=None)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--repool", action="store_true")
    a = ap.parse_args(argv)
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
