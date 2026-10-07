"""**理論の器の入力の準備**——記録の seed から両席のデッキを作り直し、補充の材料（切れる札の割合）を数える。
T91／T93・2026-09-18／段 7（2026-10-07）: 流入・効果の損害・除去の害の**式は Rust**（`rust/opcg_engine/src/theory/leaves_deck.rs`）
に移り、ここには記録とデッキ表から入力を作る部分だけが残る（ユーザ決定 2026-10-06／07: 合成デッキの作り直しは Python に残し
1 局ずつ Rust へ渡す）。

* **補充の割合**（`shares_by_seed`）＝席ごとの**切れる札の割合**（`r = μ × 割合` の材料・交点の橋の `refill`）。
  **切れる札の定義は符号化に合わせる**——印字カウンター > 0 **または**【カウンター】でパワーを上げるイベント
  （`n_rel_feat` の `counter_value` 欄と同じ式）。
* **両席のデッキ**（`decks_by_seed`）＝`meta_games.json` の `decks` のモードと各局の `leaders`・`seed` から
  `decks.build_pair` で決定論に作り直す（引かれた seed の分だけ・`_SeedMap`）。
* **探す能力の価格の席ごとのデッキ**（`record_decks`／`deck_for_seat`・T68・旧 `search_price`）＝最初の自席ターンの手札で検算する。
"""
import collections
import collections.abc
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned import n_rel_feat as NF  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402

_DB = {}
_SHARE = {}          # (leader_id, tuple(deck_ids)) は重いので id 列の署名でキャッシュ
_PAIR = {}           # (decks_mode, seed, la, lb) -> (share_p1, share_p2)
_DECKS = {}          # (decks_mode, seed, la, lb) -> (deck_ids_p1, deck_ids_p2)
_SEAT_DECKS = {}     # (seed, decks_mode) -> (deck_ids_p1, deck_ids_p2) または None（`deck_of`・旧 `search_price._DECKS`）
#: 手札の枠（`n_rel_feat` の枠の並び）
SLOT_HAND = slice(12, 22)


def db():
    if "db" not in _DB:
        _DB["db"] = D.load_db()
    return _DB["db"]


def is_cuttable(m):
    """その札が**切られうる**か。**符号化と同じ式**を使う（`n_rel_feat` の `counter_value` 欄・
    印字カウンター > 0 **または**【カウンター】でパワーを上げるイベント）＝`Θ` の手札項が読んでいる物と一致する。
    `m` はカード原本（`CardLoader.get_card`）。"""
    if float(getattr(m, "counter", 0) or 0) > 0:
        return True
    return float(NF.profile(m).get("counter_event") or 0.0) > 0.0


def cut_share(deck_ids):
    """デッキ（card_id の列・50 枚）の**切れる札の割合**。引けなかった札は分母から外す。"""
    key = tuple(deck_ids)
    if key in _SHARE:
        return _SHARE[key]
    d = db()
    n = c = 0
    for cid in deck_ids:
        m = d.get_card(cid)
        if m is None:
            continue
        n += 1
        c += 1 if is_cuttable(m) else 0
    out = (float(c) / n) if n else 0.0
    _SHARE[key] = out
    return out


def pair_decks(seed, decks_mode, leaders=(None, None)):
    """1 局の**両席のデッキ**（card_id の列・seed から決定論で作り直す）。"""
    la, lb = (leaders or (None, None))[:2]
    key = (decks_mode, int(seed), la, lb)
    if key in _DECKS:
        return _DECKS[key]
    (_l1, d1), (_l2, d2) = D.build_pair(db(), la, lb, int(seed), decks_mode)
    out = (tuple(d1), tuple(d2))
    _DECKS[key] = out
    return out


def pair_shares(seed, decks_mode, leaders=(None, None)):
    """1 局の**両席の切れる札の割合** `(p1, p2)`（デッキは seed から決定論で作り直す）。"""
    key = (decks_mode, int(seed), (leaders or (None, None))[0], (leaders or (None, None))[1])
    if key in _PAIR:
        return _PAIR[key]
    d1, d2 = pair_decks(seed, decks_mode, leaders)
    out = (cut_share(d1), cut_share(d2))
    _PAIR[key] = out
    return out


class _SeedMap(collections.abc.Mapping):
    """`{seed: fn(seed, mode, leaders)}` を**引かれた seed の分だけ**作る読み取り専用の対応表（2026-10-06・移植の段 0）。

    以前は `meta_games.json` の全局（合成の記録では 300 組）のデッキを先に作り直していた——`--limit-games 5` でも
    数秒かかる固定費。値は前と 1 ビットも変わらない: ある seed の値は「その seed が現れるディレクトリを順に見て
    `fn` が最初に成功したもの」（前の dict の作り方と同じ）、真偽は「どれか 1 局でも成功したか」、反復・`len` は
    前と同じ挿入順の dict を丸ごと作ってから返す。"""

    def __init__(self, dirs, fn):
        self._fn = fn
        self._occ = []                      # (seed, mode, leaders)・ディレクトリ順・局順
        self._by = {}                       # seed -> [(mode, leaders), ...]
        for d in dirs:
            p = os.path.join(d, "meta_games.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                meta = json.load(fh)
            mode = str(meta.get("decks") or "singleton")
            for g in meta.get("games", ()):
                s = int(g.get("seed"))
                lead = tuple(g.get("leaders") or (None, None))
                self._occ.append((s, mode, lead))
                self._by.setdefault(s, []).append((mode, lead))
        self._got = {}
        self._full = None

    def _value(self, s):
        if s not in self._got:
            res = (False, None, -1)
            for k, (mode, lead) in enumerate(self._by.get(s, ())):
                try:
                    res = (True, self._fn(s, mode, lead), k)
                    break
                except Exception:                            # noqa: BLE001
                    continue
            self._got[s] = res
        return self._got[s]

    def __getitem__(self, s):
        ok, v, _k = self._value(s)
        if not ok:
            raise KeyError(s)
        return v

    def _all(self):
        if self._full is None:
            out, seen = {}, {}
            for s, _mode, _lead in self._occ:        # 前の dict と同じ挿入順＝その seed が最初に成功した出現の位置
                k = seen.get(s, 0)
                seen[s] = k + 1
                if s in out:
                    continue
                ok, v, k_ok = self._value(s)
                if ok and k == k_ok:
                    out[s] = v
            self._full = out
        return self._full

    def __iter__(self):
        return iter(self._all())

    def __len__(self):
        return len(self._all())

    def __bool__(self):
        return any(self._value(s)[0] for s in self._by)


def _by_seed(dirs, fn):
    """記録のディレクトリ群 → `{seed: fn(seed, mode, leaders)}`（`meta_games.json` を読んで作り直す）。

    `who=0`＝p1・`who=1`＝p2（記録の規約）。同じ seed が複数のディレクトリに在れば先勝ち。
    **引かれた seed の分だけ作る**（`_SeedMap`）——値は全部を先に作っていたときと同じ。"""
    return _SeedMap(dirs, fn)


def shares_by_seed(dirs):
    """記録のディレクトリ群 → `{seed: (切れる札の割合 p1, p2)}`（T91 の `r` の材料）。"""
    return _by_seed(dirs, pair_shares)


def decks_by_seed(dirs):
    """記録のディレクトリ群 → `{seed: (デッキ p1, デッキ p2)}`（T93 の流入 `a` の材料）。"""
    return _by_seed(dirs, pair_decks)


def record_decks(dirs):
    """記録ディレクトリ → {seed: (decks モード, [リーダー p1, p2])}。無ければ空。"""
    out = {}
    for d in dirs:
        d = os.path.expanduser(d)
        mode = None
        try:
            with open(os.path.join(d, "meta_n_record.json"), encoding="utf-8") as fh:
                mode = json.load(fh).get("decks")
        except (OSError, ValueError):
            pass
        try:
            with open(os.path.join(d, "meta_games.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            mode = mode or m.get("decks")
            for g in m.get("games") or []:
                out[int(g["seed"])] = (mode, list(g.get("leaders") or [None, None]))
        except (OSError, ValueError):
            pass
    return out


def deck_of(seed, mode, leaders):
    """seed からその局の両席のデッキ（card_id の並び）を復元する。復元できなければ `None`。"""
    key = (int(seed), str(mode))
    if key in _SEAT_DECKS:
        return _SEAT_DECKS[key]
    try:
        db_ = D.load_db()
        la, lb = (list(leaders) + [None, None])[:2]
        (l1, d1), (l2, d2) = D.build_pair(db_, la, lb, int(seed), str(mode))[:2]
        _SEAT_DECKS[key] = (list(d1), list(d2))
    except Exception:
        _SEAT_DECKS[key] = None
    return _SEAT_DECKS[key]


def deck_for_seat(seed, mode, leaders, who, hand_cids):
    """席 `who`（0/1）のデッキ。**復元が記録と合うか**を手札で検算する（手札の札がデッキの並びに全部在るか）。
    合わなければ `None`＝その局の探す能力の価格は従来の `sel(k)` に落ちる。"""
    d = deck_of(seed, mode, leaders)
    if d is None or who not in (0, 1):
        return None
    deck = d[int(who)]
    cnt = collections.Counter(deck)
    for c in hand_cids:
        if cnt[c] <= 0:
            return None
        cnt[c] -= 1
    return list(deck)


def hand_ids(ci_row, idx2cid):
    """行の主の手札の card_id の並び（空の枠は落とす）。"""
    out = []
    for v in np.asarray(ci_row)[SLOT_HAND]:
        cid = idx2cid.get(int(v))
        if cid:
            out.append(str(cid))
    return out
