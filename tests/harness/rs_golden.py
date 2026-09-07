"""golden 照合の共通正規化（`docs/rust_engine_plan.md` §16.1）。

Rust 化のあと、`make test` の必須ゲートは「Python エンジンを動かして突き合わせる」のではなく
**記録済みの sha1（golden）と Rust の出力を突き合わせる**形にした。その sha1 を作る規約を
ここに 1 か所だけ置く（Rust 側の正本は `rust/opcg_engine/src/audit.rs` の `UuidCanon`）。

規約は 3 つ:

1. **uuid の別名化**: `CardInstance.uuid` は `uuid4()`＝実行のたびに変わる。正規化の walk
   （dict はキー昇順・list は並びのまま）で最初に出た順に `u0`,`u1`,… へ潰す。同じ uuid は
   同じ別名になるので「段をまたいだ同一性」は保たれ、値そのものには依存しなくなる。
2. **順序を持たない list のソート**: `keywords` は Python 側が set から作るのでプロセス毎に
   並びが変わる（`rs_diff_replay.canon` と同じ扱い）。
3. **キー順に依存しない JSON**: `json.dumps(..., sort_keys=True, ensure_ascii=False)` の
   UTF-8 バイト列を sha1 する。

[`Canon`] は **1 つの golden 項目（監査 1 能力・再生 1 局）で使い回す**こと。段ごとに作り直すと
uuid の別名が段ごとにリセットされ、「同じカードが次の段でも同じカードか」を見なくなる。
"""
import hashlib
import json
import os
import re
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 効果構造 JSON（`opcg_sim/tools/export_effects_json.py` の生成物・git 管理外・約 8MB）。
# Rust は起動時にこれを 1 度だけ読んで `CardMaster` 表を作る。
DEFAULT_EFFECTS_PATH = os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")

# golden 置き場（`tests/fixtures/rs_goldens/`）。
GOLDENS_DIR = os.path.join(_REPO_ROOT, "tests", "fixtures", "rs_goldens")
AUDIT_GOLDEN = os.path.join(GOLDENS_DIR, "audit.json")
REPLAY_GOLDEN_DIR = os.path.join(GOLDENS_DIR, "replay")


def ensure_effects_json(path: str = DEFAULT_EFFECTS_PATH) -> str:
    """効果構造 JSON を用意する（無ければ exporter を呼んで生成する）。

    生成は**プロセス固有の一時名**へ書いてから `os.replace` する。`-n auto`（xdist）で
    複数のワーカーが同時に生成しようとしても、共有の `.tmp` を奪い合って片方が落ちない。
    """
    if os.path.exists(path):
        return path
    tmp = f"{path}.{os.getpid()}.tmp"
    subprocess.run([sys.executable, "-m", "opcg_sim.tools.export_effects_json", "--out", tmp],
                   cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    os.replace(tmp, path)
    return path


def load_engine(effects_path: str = DEFAULT_EFFECTS_PATH):
    """Rust 拡張を読み込んでカード定義表を張る。

    **wheel が無ければ skip ではなく例外**（`make test` のゲートが黙って通らないように）。
    導入は `make rust-develop`（`docs/rust_engine_plan.md` §16.1）。
    """
    try:
        import opcg_engine
    except ImportError as e:  # pragma: no cover - 実行環境依存
        raise AssertionError(
            "Rust 拡張 opcg_engine が入っていない（golden ゲートは skip しない）。"
            "`make rust-develop` で導入すること。" + f" ({e})"
        ) from e
    opcg_engine.load_masters(ensure_effects_json(effects_path))
    return opcg_engine

# `uuid4()` の str 形（8-4-4-4-12 の 16 進）。カード ID（`OP01-001`）は当たらない。
UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)

# イベントログの表示ラベルに埋まる uuid の**頭 4 桁**（`effects/resolver.py` の
# `f"{name}({card.uuid[:4]})"`／`f"{name} ({uuid[:4]})"`）。文字列の途中なので別名化できず、
# Rust が自分で振った uuid とは当然食い違う。**両側で同じ形に潰す**（`(#)`）＝ラベルの
# 「どの実体か」は照合しない。対象の枚数・名前・アクション・成否・値は照合を続け、
# 盤面そのもの（`state`）は各段で完全に照合するので、取り違えは盤面側で捕まる。
UUID_HEAD_RE = re.compile(r"\(([0-9a-f]{4})\)")

# 順序を持たない list 欄（`rs_diff_replay._UNORDERED_LIST_KEYS` と同じ）。
UNORDERED_LIST_KEYS = frozenset({"keywords"})

# golden ファイル（`tests/fixtures/rs_goldens/`）の形式バージョン。正規化の規約を変えたら +1 し、
# `make golden` で作り直す（古い golden はテストが version 不一致で落とす＝黙って通らない）。
GOLDEN_VERSION = 1


def _dumps(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


class Canon:
    """uuid を出現順の別名へ潰しながら正規化 JSON を作る器（1 項目に 1 つ）。

    `alias_uuids=False` は別名化を切る（＝`rs_diff_replay.canon` と同じ「キー順正規化＋
    順序なし list のソート」だけ）。**再生 golden** はこちら: 記録に uuid が焼き込まれていて
    Rust も同じ値を返すので、別名化しても得は無く、合法手のような順序不問の列で
    「別名の付き方が並び順に依存する」余計な脆さだけが増える。**監査 golden** は Rust が
    自分で盤面を作る＝uuid が Python と違うので別名化が要る（既定）。
    """

    def __init__(self, alias_uuids: bool = True):
        self.alias_uuids = alias_uuids
        self._alias: dict = {}

    def alias_of(self, uuid: str) -> str:
        a = self._alias.get(uuid)
        if a is None:
            a = f"u{len(self._alias)}"
            self._alias[uuid] = a
        return a

    def normalize(self, value, key=None):
        """`value` の複製を返す（dict はキー昇順に walk＝別名の付き方を Rust と揃える）。"""
        if isinstance(value, dict):
            return {k: self.normalize(value[k], k) for k in sorted(value)}
        if isinstance(value, list):
            items = [self.normalize(v) for v in value]
            if key in UNORDERED_LIST_KEYS:
                items = sorted(items, key=_dumps)
            return items
        if self.alias_uuids and isinstance(value, str):
            if UUID_RE.match(value):
                return self.alias_of(value)
            return UUID_HEAD_RE.sub("(#)", value)
        return value

    def hash(self, value) -> str:
        """正規化 JSON の sha1（16 進小文字 40 桁）。"""
        return hashlib.sha1(_dumps(self.normalize(value)).encode("utf-8")).hexdigest()

    def hash_unordered(self, items) -> str:
        """順序を問わない列（合法手など）の sha1＝正規化してから並べ替えて取る。"""
        norm = [self.normalize(v) for v in (items or [])]
        norm.sort(key=_dumps)
        return hashlib.sha1(_dumps(norm).encode("utf-8")).hexdigest()


def strip_request_id(board: dict) -> dict:
    """`pending_request.request_id` を落とす（フロント専用の sha1＝照合の対象外）。"""
    b = dict(board)
    pr = b.get("pending_request")
    if isinstance(pr, dict):
        pr = dict(pr)
        pr.pop("request_id", None)
        b["pending_request"] = pr
    return b


def audit_hashes(states, events) -> list:
    """監査 1 能力の sha1 列（発動直後・各応答直後）。Rust `audit::golden_audit` と同じ規約。

    段 `i` のハッシュ対象は `{"events": …, "state": …}`＝盤面とイベントログの両方。uuid は
    別名化する（Rust は自分で盤面を作るので uuid の値は Python と違う）。
    """
    canon = Canon(alias_uuids=True)
    return [
        canon.hash({"events": ev or [], "state": strip_request_id(st)})
        for st, ev in zip(states, events)
    ]


def replay_hashes(states, legals, events, shuffled) -> dict:
    """再生 1 局の sha1 列（各行動後の盤面／各決定点の合法手／各行動のイベントログ）。

    合法手は順序を問わない（`rs_diff_replay.compare` と同じ）ので並べ替えてから取る。
    イベントはシャッフルを挟んだ段だけ `targets` を枚数に潰す（[`mask_shuffled_targets`]）。
    uuid は記録に焼き込まれていて Rust も同じ値を返すので別名化しない。
    """
    canon = Canon(alias_uuids=False)
    return {
        "states": [canon.hash(strip_request_id(s)) for s in states],
        "legal": [canon.hash_unordered(m) for m in legals],
        "events": [
            canon.hash(mask_shuffled_targets(e or [], sh))
            for e, sh in zip(events, shuffled)
        ],
    }


def mask_shuffled_targets(events, shuffled) -> list:
    """シャッフルを挟んだ段のイベントの `targets` を**枚数だけ**に潰す（照合の前処理）。

    再生の規約（計画 §6／§10.2 の 3）では Rust は `random.shuffle` を再現せず、その段の並びは
    行動が終わってから記録の `hidden` で取り直す。よって「混ぜた直後に引いた／見た」カードの
    実体は Python と一致しようがない（盤面は再同期で一致する）。枚数・アクション・成否・値は
    照合を続け、実体の識別子だけを外す。`shuffled` が空の段（大多数）は何も変えない。
    """
    if not shuffled:
        return events
    out = []
    for e in (events or []):
        e = dict(e)
        if "targets" in e:
            e["targets"] = f"<{len(e['targets'] or [])} targets after shuffle>"
        out.append(e)
    return out
