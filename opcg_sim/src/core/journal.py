"""差分巻き戻し（make/unmake）基盤 — CPU 先読みの clone コストを消す②最適化の PoC 部品。

`GameManager.clone()`（= `copy.deepcopy`）は CPU 探索コストの ~86% を占める（docs/SPEC.md §2.5.2,
プロファイル: 1 decide あたり ~430 clone、各 ~2.8ms）。各ノードで盤面全体（両者デッキ 40 枚超を含む
80+ カード）を複製しているが、1 手が実際に触る状態はごく一部（カードの領域移動・レスト・ドン付与・
継続効果の数件）にすぎない。そこで「**適用 → 評価 → 巻き戻し**」で 1 手分の変更だけを記録・復元すれば
複製を丸ごと省ける。

## 設計（不活性時は完全 no-op）

- `transaction()` の中だけ記録が有効。外（通常プレイ・既存テスト）では各 hook は
  `_active is None` を 1 回読むだけで素通りし、挙動・コストともに変化しない。
- **属性の書き換え**は対象クラスの `__setattr__` が旧値を記録する（スカラ・参照の付け替え）。
- **コンテナの in-place 変更**（list.append / set.add / dict[k]=v 等）は、対象を
  `JournaledList/Set/Dict` にしておけば、そのトランザクション内で**最初に変更された瞬間に中身を
  まるごと 1 回スナップショット**して記録する（以後の変更は追加記録不要）。新規に代入された
  プレーンなコンテナは __setattr__ が「属性ごと」旧コンテナへ戻すので journaled でなくても安全
  （= トランザクションを跨いで残る、その場変更されるコンテナだけ journaled 型であればよい）。

## 巻き戻しの正しさ

各記録は逆操作。`rollback()` は記録を**逆順**に再生し、再生中は `_active=None` にして二重記録を防ぐ。
入れ子（探索の再帰）は世代カウンタ `gen`（単調増加 int）で扱う: コンテナは「現 active 世代で
スナップショット済みか」を `_jgen` で判定する。内側トランザクションは必ず外側より大きい gen を持つ
ため、巻き戻し後に外側で再変更されれば再スナップショットされる（取りこぼし無し）。

検証用に `deep_diff()` を同梱する。`適用→巻き戻し` 後の盤面が開始時の deepcopy と完全一致するかを
照合し、journaled 化の取りこぼし（未カバーの in-place 変更）を機械的に検出する（PoC のゲート）。
"""
import threading
from contextlib import contextmanager

# --- スレッドローカル状態（並行安全）-----------------------------------------
# journal は make/unmake 探索の差分記録に使う。本番ではポンダリングが OS スレッド（asyncio.to_thread）でも
# 探索を走らせ得るため、状態をプロセス共有グローバルにすると「あるスレッドの transaction 中に別スレッドが
# 生きたオブジェクトの属性を初回セット→その set が誤記録され rollback で live から属性が消える」並行バグになる
# （`CardInstance has no attribute 'master'`）。そこで active/gen/mut を**スレッドローカル**にし、各スレッドの
# 記録が互いに漏れないようにする（単一スレッド時は従来と完全同値）。
#   active     : 現在の StateJournal（None = 不活性 = 記録しない）
#   gen_counter: トランザクションごとの単調増加世代（スレッド内）
#   mut_count  : 盤面変更の単調増加カウンタ（Phase2・継続効果再計算の dirty-flag 用）。journaled な全変更
#                （record_attr ＝属性 / コンテナの _touch）と探索開始（top-level transaction 入場）で +1。
#                不活性時（正常プレイ・active is None）は一切増えず作動しない＝従来同値。
class _ThreadState(threading.local):
    def __init__(self):
        self.active = None
        self.gen_counter = 0
        self.mut_count = 0


_TL = _ThreadState()


def __getattr__(name):
    """後方互換（PEP 562）: 外部は `journal._active`/`_mut_count`/`_gen_counter` をモジュール属性として
    読む（models.py・gamestate.py 等）。現スレッドのスレッドローカル値を返す（読み取り専用）。ホットパス
    （`CardInstance.__setattr__` 等）は `journal._TL.active` を直接読んで関数呼び出しのオーバーヘッドを避ける。"""
    if name == "_active":
        return _TL.active
    if name == "_mut_count":
        return _TL.mut_count
    if name == "_gen_counter":
        return _TL.gen_counter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 巻き戻しエントリのタグ
_ATTR = 0   # (obj, name, old_value)      → object.__setattr__(obj, name, old)
_POP = 1    # (obj, name)                 → obj.__dict__.pop(name, None)
_LIST = 2   # (lst, old_items)            → lst[:] = old_items（スライス代入で全置換）
_SET = 3    # (s, old_items)              → s 全置換
_DICT = 4   # (d, old_items)              → d 全置換


class StateJournal:
    __slots__ = ("gen", "_undo")

    def __init__(self):
        _TL.gen_counter += 1
        self.gen = _TL.gen_counter
        self._undo = []

    def rollback(self):
        """記録を逆順に再生して開始時点へ戻す（呼び出し側が _active を退避済みの前提）。"""
        u = self._undo
        while u:
            e = u.pop()
            tag = e[0]
            if tag == _ATTR:
                object.__setattr__(e[1], e[2], e[3])
            elif tag == _POP:
                e[1].__dict__.pop(e[2], None)
            elif tag == _LIST:
                list.__setitem__(e[1], slice(None), e[2])
            elif tag == _SET:
                s = e[1]
                set.clear(s)
                set.update(s, e[2])
            else:  # _DICT
                d = e[1]
                dict.clear(d)
                dict.update(d, e[2])


@contextmanager
def transaction():
    """このブロック内の状態変更を記録し、退出時に**必ず巻き戻す**（make/unmake の unmake）。

    入れ子可。退出時は記録の再生中だけ `_active=None` にして二重記録を防ぎ、親トランザクション
    （あれば）を復帰させる。
    """
    prev = _TL.active
    if prev is None:
        # 探索開始（top-level）。正常プレイ中は mut_count が凍結するため、ここで +1 して
        # 直前の正常プレイによる盤面変更を跨いだ dirty-flag の取り残しを無効化する（健全性）。
        _TL.mut_count += 1
    j = StateJournal()
    _TL.active = j
    try:
        yield j
    finally:
        _TL.active = None       # 再生中は記録しない
        j.rollback()
        _TL.active = prev       # 親（または None）へ復帰


def is_active() -> bool:
    return _TL.active is not None


# --- 属性書き換えの記録（対象クラスの __setattr__ から呼ぶ）-------------------
def record_attr(obj, name, d):
    """`obj.name` を書き換える直前に旧値（無ければ削除）を記録する。`d` は obj.__dict__。"""
    j = _TL.active
    if j is None:
        return
    _TL.mut_count += 1
    if name in d:
        j._undo.append((_ATTR, obj, name, d[name]))
    else:
        j._undo.append((_POP, obj, name))


# --- journaled コンテナ -------------------------------------------------------
# いずれも「現 active 世代で未スナップショットなら中身を 1 回だけ控える」方式。
# 不活性時（_active is None）は素の組み込み型と完全に同じ挙動・コスト。

class JournaledList(list):
    _jgen = 0

    def _touch(self):
        j = _TL.active
        if j is not None:
            _TL.mut_count += 1
            if self._jgen != j.gen:
                self._jgen = j.gen
                j._undo.append((_LIST, self, self[:]))

    def append(self, x):
        self._touch(); list.append(self, x)

    def extend(self, it):
        self._touch(); list.extend(self, it)

    def insert(self, i, x):
        self._touch(); list.insert(self, i, x)

    def remove(self, x):
        self._touch(); list.remove(self, x)

    def pop(self, i=-1):
        self._touch(); return list.pop(self, i)

    def clear(self):
        self._touch(); list.clear(self)

    def sort(self, *a, **k):
        self._touch(); list.sort(self, *a, **k)

    def reverse(self):
        self._touch(); list.reverse(self)

    def __setitem__(self, i, v):
        self._touch(); list.__setitem__(self, i, v)

    def __delitem__(self, i):
        self._touch(); list.__delitem__(self, i)

    def __iadd__(self, it):
        self._touch(); return list.__iadd__(self, it)

    def __imul__(self, n):
        self._touch(); return list.__imul__(self, n)


class JournaledSet(set):
    _jgen = 0

    def _touch(self):
        j = _TL.active
        if j is not None:
            _TL.mut_count += 1
            if self._jgen != j.gen:
                self._jgen = j.gen
                j._undo.append((_SET, self, set(self)))

    def add(self, x):
        self._touch(); set.add(self, x)

    def discard(self, x):
        self._touch(); set.discard(self, x)

    def remove(self, x):
        self._touch(); set.remove(self, x)

    def pop(self):
        self._touch(); return set.pop(self)

    def clear(self):
        self._touch(); set.clear(self)

    def update(self, *a):
        self._touch(); set.update(self, *a)

    def difference_update(self, *a):
        self._touch(); set.difference_update(self, *a)

    def intersection_update(self, *a):
        self._touch(); set.intersection_update(self, *a)

    def symmetric_difference_update(self, *a):
        self._touch(); set.symmetric_difference_update(self, *a)

    def __ior__(self, o):
        self._touch(); return set.__ior__(self, o)

    def __iand__(self, o):
        self._touch(); return set.__iand__(self, o)

    def __isub__(self, o):
        self._touch(); return set.__isub__(self, o)

    def __ixor__(self, o):
        self._touch(); return set.__ixor__(self, o)


class JournaledDict(dict):
    _jgen = 0

    def _touch(self):
        j = _TL.active
        if j is not None:
            _TL.mut_count += 1
            if self._jgen != j.gen:
                self._jgen = j.gen
                j._undo.append((_DICT, self, dict(self)))

    def __setitem__(self, k, v):
        self._touch(); dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        self._touch(); dict.__delitem__(self, k)

    def pop(self, *a):
        self._touch(); return dict.pop(self, *a)

    def popitem(self):
        self._touch(); return dict.popitem(self)

    def clear(self):
        self._touch(); dict.clear(self)

    def update(self, *a, **k):
        self._touch(); dict.update(self, *a, **k)

    def setdefault(self, k, default=None):
        self._touch(); return dict.setdefault(self, k, default)


# --- 検証ユーティリティ（テスト専用。production 経路では使わない）-------------
def deep_diff(a, b, _path="", _seen=None):
    """2 つの状態を再帰比較し、最初の相違パス（str）を返す。一致なら None。

    `CardMaster`（不変・共有）は id 同一性で比較する。サイクル（continuous.gm 後方参照等）は
    訪問済み id ペアで打ち切る。`適用→巻き戻し` 後の盤面と開始時 deepcopy の完全一致を確かめ、
    journaled 化の取りこぼしを検出する PoC ゲート。
    """
    if _seen is None:
        _seen = set()
    if a is b:
        return None
    ta = type(a)
    # CardMaster は共有不変 → 同一 id のみ一致扱い（中身は辿らない）
    from ..models.models import CardMaster
    if isinstance(a, CardMaster) or isinstance(b, CardMaster):
        return None if a is b else f"{_path}: CardMaster identity differs"
    # プリミティブ（journaled 型は基底でディスパッチするため exact type は見ない）
    if a is None or ta in (int, str, bool, float):
        return None if a == b else f"{_path}: {a!r} != {b!r}"
    key = (id(a), id(b))
    if key in _seen:
        return None
    _seen.add(key)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{_path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = deep_diff(x, y, f"{_path}[{i}]", _seen)
            if d:
                return d
        return None
    if isinstance(a, (set, frozenset)):
        return None if a == b else f"{_path}: set {a ^ b} differs"
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return f"{_path}: dict vs {type(b).__name__}"
        if a.keys() != b.keys():
            return f"{_path}: dict keys {set(a) ^ set(b)} differ"
        for k in a:
            d = deep_diff(a[k], b[k], f"{_path}[{k!r}]", _seen)
            if d:
                return d
        return None
    # enum 等（__dict__ を持たない値オブジェクト）
    da = getattr(a, "__dict__", None)
    db = getattr(b, "__dict__", None)
    if da is None or db is None:
        return None if a == b else f"{_path}: {a!r} != {b!r}"
    if type(a).__name__ != type(b).__name__:
        return f"{_path}: type {type(a).__name__} != {type(b).__name__}"
    # `_passive_mc`（Phase2・継続効果再計算の dirty-flag キャッシュ）はロールバック対象外
    # （object.__setattr__ で更新・盤面状態でない）。両者から除外して比較する。
    if (da.keys() - {"_passive_mc"}) != (db.keys() - {"_passive_mc"}):
        return f"{_path}: attrs {set(da) ^ set(db)} differ"
    for k in da:   # 挿入順（決定的）で反復
        if k == "_passive_mc":
            continue   # dirty-flag キャッシュは盤面状態でない＝比較対象外
        if k == "gm" or k.startswith("__"):   # 後方参照はサイクル管理に任せる
            pass
        d = deep_diff(da[k], db[k], f"{_path}.{k}", _seen)
        if d:
            return d
    return None
