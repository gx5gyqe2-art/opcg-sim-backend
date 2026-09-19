"""除去の「型」の分類と差し込み（`opcg_sim/loop/deck_roles.py`・2026-09-10・計画 §20.8.1）の契約。

標準（マーカー無し・常時実行）: 教材のデッキを組む器なので、壊れると**生成する棋譜そのもの**が
変わる（不正なデッキ＝同名 5 枚・50 枚でない・死に札だらけ）。ゲームプレイの正しさに直結する。

守る性質:
  1. **分類**（`classify`）が実カードで期待どおり: KO／bounce（手札へ）／deck（山札・ライフへ）／
     trash（直接トラッシュ）／lock（レスト等）／reduce（パワー・コスト減少）を、**相手の盤面**
     （zone=FIELD）に撃つときだけ数える（手札を捨てさせる・ライフを触るは除去ではない）。
     **バウンスは `ActionType.BOUNCE`**（`MOVE_TO_HAND` は別名）＝2026-09-09 の実測が「相手対象の
     バウンス 0 種」になった原因はここ（`n_rel_feat._REMOVAL_OPS` が BOUNCE を持たない）。
  2. **型 × 色**の表が DB から再計算できる（手書きの表を持たない）＝組めない型は組めないと出る。
  3. `inject_roles` が**決定論**（同じ (leader, seed) で同じデッキ）・**同名 4 枚以内**・50 枚・
     率 0 は元のデッキと**同一**・**死に札監査とカウンター枚数を悪化させない**。
  4. `decks.build_pair` の既定（singleton／synth／synth_dig）は**戻り値も中身も不変**で、
     `synth_roles` だけが型を返す。
  5. dump の `deck_kinds` 列（v4）: `record_gen` が手番側の型を JSON で載せ、`dump_io` は
     **v3 の波も v4 の波も**同じ関数で読める（知らない列は読まない）。
"""
import collections
import json
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import deck_roles as R
from opcg_sim.loop import deck_synth as DS
from opcg_sim.loop import decks as D

#: 実カードの期待値（form が 1 つに定まる代表・効果文は `classify` の docstring 参照）。
CASES = {
    "EB01-016": {"KO:ACTIVATE_MAIN:c0-2"},          # 【起動メイン】相手のレストのコスト1以下をKO
    "OP03-048": {"bounce:ON_PLAY:c0-2"},            # 【登場時】相手のコスト5以下を持ち主の手札へ
    "EB02-027": {"deck:ON_PLAY:c3-5"},              # 【登場時】相手のパワー1000以下をデッキの下へ
    "OP07-091": {"trash:ON_ATTACK:c3-5"},           # 【アタック時】相手のコスト2以下をトラッシュへ
    "EB01-015": {"lock:ON_PLAY:c0-2"},              # 【登場時】相手のコスト2以下をレストにする
}
#: 除去ではないもの（相手の**手札**を捨てさせるだけ・**ライフ**を触るだけ）は型を持たない。
NOT_REMOVAL = ("OP01-114",     # 【登場時】ドン!!-1: 相手は自身の手札1枚を捨てる
               "EB03-057")     # 【KO時】相手のライフの上から1枚までを、トラッシュに置く


@pytest.fixture(scope="module")
def db():
    return D.load_db()


# --- 1. 分類 ---------------------------------------------------------------
def test_classify_forms_on_real_cards(db):
    for cid, want in CASES.items():
        got = R.classify(db.get_card(cid))
        assert got == want, f"{cid}: {sorted(got)} != {sorted(want)}"


def test_classify_bounce_is_action_type_bounce(db):
    """「手札に戻す」は `BOUNCE`（`MOVE_TO_HAND` は別名）＝両方を bounce として数える。"""
    from opcg_sim.src.models.effect_types import ActionType
    assert R._FORM_OF_OP[ActionType.BOUNCE] == "bounce"
    assert R._FORM_OF_OP[ActionType.MOVE_TO_HAND] == "bounce"
    n = sum(1 for cid in db.raw_db
            if (c := db.get_card(cid)) is not None
            and any(k.startswith("bounce:") for k in R.classify(c)))
    assert n >= 20, f"相手対象のバウンスが {n} 種（実測 23 種・0 なら分類が落ちている）"


def test_classify_key_shape_and_vocabulary(db):
    forms = collections.Counter()
    for cid in db.raw_db:
        c = db.get_card(cid)
        if c is None:
            continue
        for k in R.classify(c):
            f, s, b = k.split(":")
            assert f in R.FORMS and s in R.SOURCES and b in R.BANDS, k
            forms[f] += 1
    assert set(forms) == set(R.FORMS), f"出ない form がある: {set(R.FORMS) - set(forms)}"


def test_hand_and_life_effects_are_not_removal(db):
    """相手の手札／ライフだけを触る効果は「盤面の除去」ではない（zone=FIELD 限定）。"""
    for cid in NOT_REMOVAL:
        c = db.get_card(cid)
        if c is None:
            continue
        assert not R.classify(c), f"{cid}: {sorted(R.classify(c))}"


# --- 2. 型 × 色のテンプレート表 ---------------------------------------------
def test_color_table_is_recomputed_from_db(db):
    """§20.7.11 の読み（コードで再計算する・手書きの表を持たない）。"""
    t = R.color_table(db)
    assert set(t) == {"BLACK", "BLUE", "GREEN", "PURPLE", "RED", "YELLOW"}
    assert all(n in R.TEMPLATE_NAMES for v in t.values() for n in v)
    # 「減少 → KO」は黒・赤・紫（＋黄）で組めて、青・緑では組めない
    assert "reduce_then_ko" in t["BLACK"] and "reduce_then_ko" in t["RED"]
    assert "reduce_then_ko" not in t["BLUE"] and "reduce_then_ko" not in t["GREEN"]
    assert "deck_out" in t["BLUE"] and "deck_out" not in t["RED"]     # 青は山札送り
    assert "lock_and_attack" in t["GREEN"]                            # 緑は疑似除去
    # 組める＝その型のカードが MIN_KIND_CARDS 種以上ある
    for col, names in t.items():
        pool = R.color_pool(db, {col})
        for t_ in R.buildable(pool):
            assert t_["name"] in names
            for p in t_["parts"]:
                assert len(R.part_cards(pool, p)) >= R.MIN_KIND_CARDS


def test_kind_matches_wildcards():
    assert R.kind_matches("KO:ON_PLAY:c3-5", "KO:*:c3-5")
    assert R.kind_matches("KO:ON_PLAY:c3-5", "*:ON_PLAY:*")
    assert not R.kind_matches("KO:ON_PLAY:c3-5", "lock:*:*")
    assert not R.kind_matches("KO:ON_PLAY:c3-5", "KO:*:c6+")


# --- 3. 差し込み -----------------------------------------------------------
LEADERS = ("OP11-041", "OP09-001", "OP15-058", "OP16-022")   # 実デッキ 4 リーダー
SEEDS = (0, 1, 2, 3, 4, 5)


def _inject(db, lid, seed):
    leader, cards = DS.synth_deck(db, lid, seed=seed, owner="p1")
    out, kinds = R.inject_roles(db, leader, cards, "p1", seed=seed)
    return leader, cards, out, kinds


def test_inject_is_deterministic(db):
    for lid in LEADERS:
        a = _inject(db, lid, 3)
        b = _inject(db, lid, 3)
        assert [DS._master(x).card_id for x in a[2]] == [DS._master(x).card_id for x in b[2]]
        assert a[3] == b[3]


def test_inject_keeps_deck_legal(db):
    for lid in LEADERS:
        for seed in SEEDS:
            leader, _base, out, kinds = _inject(db, lid, seed)
            assert len(out) == DS.DECK_SIZE
            cnt = collections.Counter(DS._master(x).name for x in out)
            assert max(cnt.values()) <= DS.MAX_COPIES
            lm = DS._master(leader)
            assert all(DS._legal_for(lm, DS._master(x)) for x in out)
            assert 0 <= kinds["rate"] <= max(R.RATES)


def test_rate_zero_returns_the_same_deck(db):
    """率 0 のデッキは元の合成デッキと**同一**（対照の基準・1/6 の割合で出る）。"""
    seen = 0
    for lid in D.leader_pool(db)[:40]:
        leader, cards = DS.synth_deck(db, lid, seed=0, owner="p1")
        out, kinds = R.inject_roles(db, leader, cards, "p1", seed=0)
        if kinds["rate"] == 0:
            seen += 1
            assert [DS._master(x).card_id for x in out] == [DS._master(x).card_id for x in cards]
            assert kinds["injected"] == [] and kinds["n"] == 0
    assert seen, "率 0 のデッキが 1 つも出なかった（RATES から 0 が消えている）"


def test_inject_does_not_break_audit_or_counters(db):
    """死に札監査（`deck_synth.audit_deck`）とカウンター枚数を**悪化させない**。"""
    for lid in D.leader_pool(db)[:30]:
        for seed in (0, 1):
            leader, base, out, kinds = _inject(db, lid, seed)
            if kinds["n"] == 0:
                continue
            bm = [DS._master(x) for x in base]
            om = [DS._master(x) for x in out]
            assert DS.audit_deck(leader, out)["dead_rate"] <= DS.audit_deck(leader, base)["dead_rate"]
            assert R._counter_cards(om) >= min(DS.MIN_COUNTER_CARDS, R._counter_cards(bm))


def test_injected_kinds_are_actually_in_the_deck(db):
    for lid in LEADERS:
        for seed in SEEDS:
            _leader, base, out, kinds = _inject(db, lid, seed)
            after = set(R.deck_kinds(out))
            assert set(kinds["injected"]) <= after
            assert set(kinds["native"]) == set(R.deck_kinds(base))
            assert set(kinds) == {"injected", "native", "rate", "colors", "n",
                                  "templates", "reduced"}
            json.dumps(kinds, ensure_ascii=False)          # 記録は JSON で載る


def test_injection_adds_removal_where_the_base_deck_had_none(db):
    """差し込みの目的＝教材の対照。硬い除去 0 のデッキが減ること（実測 56→26 / 286）。"""
    hard = ("KO", "bounce", "deck", "trash")
    before = after = 0
    for lid in D.leader_pool(db)[:60]:
        _leader, base, out, _k = _inject(db, lid, 1)
        before += 0 if any(k.split(":")[0] in hard for k in R.deck_kinds(base)) else 1
        after += 0 if any(k.split(":")[0] in hard for k in R.deck_kinds(out)) else 1
    assert after < before, f"除去 0 のデッキが減っていない（{before} → {after}）"


# --- 4. build_pair の契約 ---------------------------------------------------
def test_build_pair_defaults_are_unchanged(db):
    for mode in ("singleton", "synth", "synth_dig"):
        pair = D.build_pair(db, "OP11-041", "OP09-001", 7, mode)
        assert len(pair) == 2 and len(pair[0][1]) == D.DECK_SIZE
        p1, p2, kinds = D.build_pair(db, "OP11-041", "OP09-001", 7, mode, with_kinds=True)
        assert (p1, p2) == pair and kinds == (None, None)


def test_build_pair_synth_roles(db):
    p1, p2, kinds = D.build_pair(db, "OP11-041", "OP09-001", 7, "synth_roles", with_kinds=True)
    assert len(p1[1]) == len(p2[1]) == D.DECK_SIZE
    assert all(isinstance(k, dict) and "rate" in k for k in kinds)
    base = D.build_pair(db, "OP11-041", "OP09-001", 7, "synth")
    assert (p1, p2) != base or all(k["n"] == 0 for k in kinds)


# --- 5. dump の deck_kinds 列（v4） -----------------------------------------
def _fake_shard(rng, n, with_kinds):
    """`dump_io` が読む最小の列（v3）＋（`with_kinds` なら）v4 の `deck_kinds`。"""
    from opcg_sim.learned.n_rel import N_TOK
    kind = rng.integers(0, 3, n).astype(np.int8)
    out = {
        "scalars": rng.standard_normal((n, 123)).astype(np.float16),
        "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, 40, (n, 24)).astype(np.int16),
        "tokens": rng.standard_normal((n, N_TOK, 20)).astype(np.float16),
        "z": rng.choice([-1.0, 1.0], n).astype(np.float32),
        "who": rng.integers(0, 2, n).astype(np.int8), "kind": kind,
        "turn": rng.integers(1, 12, n).astype(np.int16),
        "step": np.arange(n, dtype=np.int32), "seed": np.arange(n, dtype=np.int64),
        "sig": np.array(["[]"] * n), "pol_len": np.zeros(n, np.int32),
        "pol_chosen": np.full(n, -1, np.int16),
        "pol_n": np.zeros(0, np.float32), "pol_q": np.zeros(0, np.float32),
        "pol_k": np.zeros(0, np.int16), "pol_sig": np.array([], dtype="U1"),
        "pol_cid": np.array([], dtype="U1"), "pol_tcid": np.array([], dtype="U1"),
        "pol_si": np.zeros(0, np.int16), "pol_ti": np.zeros(0, np.int16),
    }
    if with_kinds:
        out["deck_kinds"] = np.array([json.dumps({"injected": ["KO:ON_PLAY:c3-5"], "rate": 10},
                                                 ensure_ascii=False)] * n)
    return out


def test_dump_io_reads_v3_and_v4(tmp_path):
    """v4（`deck_kinds` 付き）も v3（無し）も同じ `load_dump` で読める＝過去の波はそのまま。"""
    from opcg_sim.learned.train import dump_io as DIO
    rng = np.random.default_rng(3)
    dirs = []
    for name, wk in (("w_v3", False), ("w_v4", True)):
        d = tmp_path / name
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(d / "n_record_00000.npz", **_fake_shard(rng, 12, wk))
        dirs.append(str(d))
    V, _P, _C = DIO.load_dump(dirs, {}, with_policy=True, cache_dir=str(tmp_path / "cache"))
    assert len(V["z"]) == 24
    with np.load(os.path.join(dirs[1], "n_record_00000.npz"), allow_pickle=True) as d_:
        assert "deck_kinds" in d_.files
        assert json.loads(str(d_["deck_kinds"][0]))["rate"] == 10
    with np.load(os.path.join(dirs[0], "n_record_00000.npz"), allow_pickle=True) as d_:
        assert "deck_kinds" not in d_.files


def test_record_gen_declares_the_v4_column():
    from opcg_sim.loop import record_gen as G
    # v4 の行ごとの追加列は 4 本（`deck_kinds`＝§20.8.1／`forced`＝ε 探索・§20.8.5／`forced_sig`＝
    # 差し替えた手の move_sig・§20.8.9／`pol_v0`＝根の価値の推定・改良方策 π' の材料・§20.6.1）。
    # 候補ごとの `pol_p` は `_POL_KEYS`。
    assert G.DUMP_VERSION == 4 and G._V4_KEYS == ("deck_kinds", "forced", "forced_sig", "pol_v0")
    assert "pol_p" in G._POL_KEYS
    assert G.DEFAULT_DECKS == "synth"           # 既定の波の規約は変えない


@pytest.mark.cpu_infra
def test_record_gen_writes_deck_kinds(tmp_path):
    """生成器を in-process で 1 局回して `deck_kinds` 列が載ることを確かめる（sims 4）。"""
    from opcg_sim.loop import record_gen as G
    G._G.clear()
    G._init_worker(4, None, 0.25, 4, "synth_roles")
    try:
        for seed in range(920001, 920008):
            r = G.play_one(seed)
            if r is None:
                continue
            n = len(r["z"])
            assert r["deck_kinds"].shape == (n,)
            ks = [json.loads(str(s)) for s in r["deck_kinds"]]
            assert all(set(k) >= {"rate", "injected", "native"} for k in ks)
            assert len({s for s in r["deck_kinds"]}) <= 2      # 両席ぶんの 2 通りだけ
            assert r["_meta"]["seed"] == seed and len(r["_meta"]["kinds"]) == 2
            return
        pytest.fail("7 seed で 1 局も決着しなかった（生成器の前提が崩れている）")
    finally:
        G._G.clear()
