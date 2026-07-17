"""防御側プラン化（v8・カウンター/ブロッカー窓のプラン列挙と実プラン復元）。

攻撃側プランと同じ終端規約（「手番が自分から離れる手で終わる」）が防御にもそのまま成立する:
PASS＝戦闘解決で手番が離れる／カウンター連打＝窓が続く限り自分の手番＝プラン継続。
g3@82（人間マーク「切るなら105・EB03温存」の実測点・SELECT_COUNTER 窓）で:
  1. 列挙が防御の選択肢を尽くす: 素通し(PASS)・各カウンター単発・重ね切り
  2. 実際の手（EB03-053 で切る→PASS）が記録から2手プランとして復元される
ロールアウトなし（構造のみ＝高速）。基盤健全性＝cpu_infra。
"""
import argparse
import os

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

pytestmark = pytest.mark.cpu_infra

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def setup():
    import sys
    sys.path.insert(0, os.path.join(REPO, "tests", "scripts"))
    import counterfactual_referee as CR
    import mark_gate as MG
    import replay_reeval as RE
    import p3_loop as P
    import rl_net as RN
    import rl_encoder as E
    from opcg_game import OPCGGame
    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import _net_enc_version

    CR.ARGS = argparse.Namespace(sims=16, plan_len=4, beam=12, max_plans=12,
                                 band=0.5, true_board=True)
    db = _load_db()
    vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
    vf = P.value_fn_of(vnet, E.vocab_from_ids(vnet.vocab_ids), _net_enc_version(vnet))
    game_root = OPCGGame(prune_futile=False)
    raw = RE.load_replay_json(MG.REPLAYS["g3"]); rec = raw.get("replay", raw)
    fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
    CR.GAMES = {"g3": (rec, fbi, rec["actions"])}
    m0, who = CR._restore_board(db, "g3", 82)
    assert not isinstance(m0, str)
    assert m0.get_pending_request().get("action") == "SELECT_COUNTER"
    return CR, game_root, vf, m0, who, rec, fbi


def test_defense_enumeration_covers_counter_options(setup):
    """素通し(PASS)・105単発・EB03単発・重ね切りが全て列挙される。"""
    CR, game_root, vf, m0, who, _rec, _fbi = setup
    plans = CR.enumerate_turn_plans(game_root, vf, m0, who, max_len=4, beam=12, max_plans=12)
    labels = {">".join(CR._step_label(d) for d in descs) for _k, descs in plans}
    assert "PASS" in labels
    assert "SELECT_COUNTER:OP15-105>PASS" in labels
    assert "SELECT_COUNTER:EB03-053>PASS" in labels
    assert any(l.count("SELECT_COUNTER") == 2 for l in labels), "重ね切りプランが無い"
    for _k, descs in plans:
        assert descs[-1].get("action_type") == "PASS", "防御プランの終端は戦闘解決(PASS)"


def test_defense_actual_plan_reconstruction(setup):
    """@82 の実際の手（EB03-053 で切る→PASS）が記録から復元される。"""
    CR, game_root, _vf, m0, who, rec, fbi = setup
    import coach_sweep as CS
    keys, descs = CS.actual_plan_keys(game_root, m0, who, rec["actions"], 82, fbi)
    assert keys is not None and len(keys) == 2
    assert descs[0].get("action_type") == "SELECT_COUNTER"
    assert descs[0].get("card") == "EB03-053"
    assert descs[1].get("action_type") == "PASS"
