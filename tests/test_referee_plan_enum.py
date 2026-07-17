"""ターンプラン自動列挙（v8 柱A・`counterfactual_referee.enumerate_turn_plans`）と
実プラン復元（柱C・`coach_sweep.actual_plan_keys`）。

g3@64 の真盤面（人間マークの実測点）で:
  1. 列挙が比較の本命を必ず含む: 素の攻撃・攻撃者自身への付与→攻撃・素の TURN_END。
     素朴な value 順の縮約は gen5 の付与バイアスでこれらを落とした（実測）＝
     コミットメント別ラウンドロビン＋種内「短さ→自己強化→value」の回帰を固定する。
  2. 全プランが終端規約（手番が自分から離れる手で終わる）を満たす。
  3. 実プラン復元が記録の実際の手（素の ATTACK:PRB02-008）と一致する。
ロールアウトは回さない（列挙・復元の構造のみ＝高速）。基盤健全性＝cpu_infra。
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

    CR.ARGS = argparse.Namespace(sims=16, plan_len=4, beam=12, max_plans=16,
                                 band=0.5, true_board=True)
    db = _load_db()
    vnet = RN.ValueNet.load(os.path.join(REPO, "opcg_sim", "data", "learned", "gen5_value.npz"))
    vf = P.value_fn_of(vnet, E.vocab_from_ids(vnet.vocab_ids), _net_enc_version(vnet))
    game_root = OPCGGame(prune_futile=False)
    raw = RE.load_replay_json(MG.REPLAYS["g3"]); rec = raw.get("replay", raw)
    fbi = {f.get("action_index"): f for f in raw.get("frames") or []}
    CR.GAMES = {"g3": (rec, fbi, rec["actions"])}
    m0, who = CR._restore_board(db, "g3", 64)
    assert not isinstance(m0, str)
    return CR, game_root, vf, m0, who, rec, fbi


@pytest.fixture(scope="module")
def plans(setup):
    CR, game_root, vf, m0, who, _rec, _fbi = setup
    logs = []
    out = CR.enumerate_turn_plans(game_root, vf, m0, who, max_len=4, beam=12,
                                  max_plans=16, log=logs.append)
    return CR, out, logs


def test_enumeration_includes_核心プラン(plans):
    """素の攻撃・攻撃者自身への付与→攻撃・素の TURN_END が必ず列挙に残る（縮約バイアスの回帰）。"""
    CR, out, _logs = plans
    labels = {">".join(CR._step_label(d) for d in descs) for _k, descs in out}
    assert "TURN_END" in labels
    assert "ATTACK:PRB02-008→OP16-060" in labels
    assert "ATTACH_DON:PRB02-008>ATTACK:PRB02-008→OP16-060" in labels


def test_enumeration_terminal_and_caps(plans):
    """全プランが終端規約を満たし、上限超過の縮約は無言でなくログに申告される。"""
    CR, out, logs = plans
    assert 0 < len(out) <= 16
    for keys, descs in out:
        assert len(keys) == len(descs) <= 4
        assert descs[-1].get("action_type") in ("ATTACK", "TURN_END")
    assert any("[cap]" in s for s in logs), "縮約したのにログが無い（無言の縮約）"


def test_actual_plan_reconstruction(setup):
    """@64 の実際の手（素の ATTACK:PRB02-008）が記録から1手プランとして復元される。"""
    CR, game_root, _vf, m0, who, rec, fbi = setup
    import coach_sweep as CS
    keys, descs = CS.actual_plan_keys(game_root, m0, who, rec["actions"], 64, fbi)
    assert keys is not None
    assert len(keys) == 1
    assert descs[0].get("action_type") == "ATTACK"
    assert descs[0].get("card") == "PRB02-008"
