"""**RD-speed（2026-10-04）**: 攻め手の計画（`crossing_bridge.rule_don_solve`）の**ディスクの覚え書き**——同じ作業の中で
別の器（交点の橋・線形の橋・帳簿・勝ち筋の較正）や同じ器の測り直しが、同じ局面の計画を解き直さないため。

**値は 1 ビットも変えない**: 覚えるのは `rule_don_solve` の戻り値そのもの（pickle は浮動小数を厳密に往復する）。鍵は
**計算に入る全部**の内容のハッシュ:

* 問題の入力の全部（守る側の札・ドン・ブロッカー・ライフ・地平・ライフの札／引く札の分布・手札から出るブロッカー・
  攻め手の財布 `actx` の全項目〔覚え書きの `_gain` だけ除く〕）を `repr`（浮動小数は最短の厳密な表記）で。
* 値段の文脈（`cut_context_key`・守り手の窓が有るか）と計算の予算 `EX_STATE_BUDGET`。
* **解き方の版** `crossing_bridge.SOLVER_VERSION`（解き方を変えたら上げる・テストが解き方の関数の原文と Rust の核の
  原文の指紋で見張る）と **`tests/scripts/*.py` 全部の原文のハッシュ**（どれか 1 文字でも変われば別の鍵＝古い値は返らない）と
  **Rust の核の版**（`rd_kernel.kernel_tag()`＝入口の版と `src/theory` の原文のハッシュ・第 3 段から必ず入る）。
* 解き方が**読みうる大文字の大域**（切替・定数）の今の値——`rule_don_solve` から呼ばれうる関数をモジュールをまたいで
  たどって集める（`solver_reads`・多めに拾う側）。解き方が読まない切替（`PRE_SETTLE_MODE` 等）は入れない＝器ごとに
  違う切替を立てても同じ問題は同じ鍵（器をまたいで共有できる）。`__main__` として走っても import されても同じ鍵。

使い方: 環境変数 `OPCG_PLAN_STORE=<ディレクトリ>` を付けて器を走らせる（`crossing_bridge` が import 時に開く）。
既定は無し（何も書かない）。中身は SQLite 1 ファイル（`plans.sqlite`・値は zlib＋pickle）。消せば冷たい実行に戻る。
"""
import glob
import hashlib
import inspect
import os
import pickle
import sqlite3
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCALARS = (bool, int, float, str, type(None))


def _scalar_like(v, depth=0):
    if isinstance(v, _SCALARS):
        return True
    if depth < 3 and isinstance(v, (tuple, frozenset)):
        return all(_scalar_like(x, depth + 1) for x in v)
    return False


def source_digest(root=None):
    """`tests/scripts/*.py` の原文の全部のハッシュ（名前順）。"""
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(root or _HERE, "*.py"))):
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def _code_names(code):
    out = set(code.co_names)
    for c in code.co_consts:
        if inspect.iscode(c):
            out |= _code_names(c)
    return out


def _label(g):
    return os.path.basename(g.get("__file__") or "") or str(g.get("__name__"))


def solver_reads(fn):
    """`fn` から呼ばれうる関数を（モジュールをまたいで）たどり、読まれうる**大文字の大域**の一覧
    `[(ファイル名, 名前, その大域の辞書)]`。関数の原文に出てくる名前（`co_names`・入れ子の関数も）を、その関数の
    大域と、そこで参照されるモジュールの属性の両方で引く（多めに拾う側に倒す）。"""
    seen, out, stack = set(), {}, [fn]
    while stack:
        f = inspect.unwrap(stack.pop())                      # Rust 化の継ぎ目（`rd_kernel`）の包みは元の関数に戻して読む
        if f in seen:
            continue
        seen.add(f)
        g = f.__globals__
        names = _code_names(f.__code__)
        srcs = [g] + [vars(g[n]) for n in names if n in g and inspect.ismodule(g[n])]
        for n in names:
            for src in srcs:
                if n not in src:
                    continue
                v = src[n]
                if inspect.isfunction(v):
                    stack.append(v)
                elif n[:1].isupper() and n.upper() == n and not inspect.ismodule(v) and n not in _VOLATILE:
                    out[(_label(src), n)] = src
    return sorted((lab, n, src) for (lab, n), src in out.items())


def _val(v):
    if _scalar_like(v):
        return repr(v)
    if isinstance(v, dict) and len(v) <= 64 and all(_scalar_like(a) and _scalar_like(b) for a, b in v.items()):
        return repr(sorted(v.items(), key=repr))
    return "<%s>" % type(v).__name__                         # 関数・オブジェクト: 有るか（型）だけ（中身は入力の側で鍵に入る）


def globals_snapshot(reads):
    """`solver_reads` の大域の今の値（ファイル名・名前つき）。モジュールの名前（`__main__` か否か）に依らない。"""
    return repr([(lab, n, _val(src.get(n))) for lab, n, src in reads])


#: 実行中に数が増えるだけの数え先と、この覚え書き自身（値に入らない）——鍵に入れると毎回別の鍵になって覚え書きが効かない
#: （`STATS` は `rd_kernel` の呼び出しの数え先＝解き方が `rd_kernel` の関数を呼ぶので、たどると拾われる）
_VOLATILE = {"RULE_STATS", "EX_SPEED_STATS", "PLAN_STORE", "STATS"}


class PlanStore:
    """`rule_don_solve` の結果のディスクの覚え書き（`get`／`put`）。"""

    def __init__(self, path, cb):
        os.makedirs(path, exist_ok=True)
        self.path = os.path.join(path, "plans.sqlite")
        self.cb = cb
        self.db = sqlite3.connect(self.path, timeout=60.0, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS plans (k TEXT PRIMARY KEY, v BLOB)")
        self.src = source_digest()
        self.reads = solver_reads(cb.rule_don_solve)
        self.hits = self.misses = self.puts = 0
        self._glob_cache = None


    def key_of(self, cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive):
        cb = self.cb
        ax = sorted((k, v) for k, v in actx.items() if k != "_gain")
        import cut_price as CP
        # **Rust 化・第 3 段**: 解くのは Rust の核だけ——核の版（入口の版と `src/theory` の原文のハッシュ）を鍵に入れる
        # （核を直せば別の鍵＝古い値は返らない）
        tag = getattr(cb, "RD_KERNEL_TAG", None)
        tag = tag() if tag is not None else "rs:none"
        body = repr((cb.SOLVER_VERSION, self.src, globals_snapshot(self.reads),
                     list(cards_d or ()), don_d, list(blk or ()), life, turns, tuple(life_types or ()),
                     tuple(draw_types or ()), tuple(arrive or ()), ax, cb.cut_context_key(),
                     CP.active() is not None, cb.EX_STATE_BUDGET, tag))
        return hashlib.sha256(body.encode()).hexdigest()

    def get(self, key):
        row = self.db.execute("SELECT v FROM plans WHERE k = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return pickle.loads(zlib.decompress(row[0]))

    def put(self, key, out):
        blob = zlib.compress(pickle.dumps(out, protocol=4), 6)
        self.db.execute("INSERT OR REPLACE INTO plans (k, v) VALUES (?, ?)", (key, blob))
        self.puts += 1

    def report(self):
        return {"plan_store": self.path, "hits": self.hits, "misses": self.misses, "puts": self.puts}


def open_from_env(cb):
    """`OPCG_PLAN_STORE` が有れば開いて `crossing_bridge.PLAN_STORE` に置く（無ければ何もしない）。"""
    path = os.environ.get("OPCG_PLAN_STORE")
    if not path:
        return None
    st = PlanStore(path, cb)
    cb.PLAN_STORE = st
    import atexit
    atexit.register(lambda: (st.hits or st.misses) and print("plan_store: %r" % (st.report(),), file=sys.stderr))
    return st
