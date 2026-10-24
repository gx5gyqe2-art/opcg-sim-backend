"""定型の作業の実行役（`opcg_sim/loop/worker_job.py`・WP `worker-jobs`・`docs/n_loop_ops.md` §7.42）の写しの見張り。

§7.41 の手書きの指示書（波 51 の生成・r22／r22o の訓練と評価帯・r22／r22o の判定アリーナ）を写したジョブ定義
（`opcg_sim/loop/jobs/*.json`）が、`--dry-run` で手書きと同じコマンド（`record_gen`／`n_rel_train`・`n_rel_band`／
`arena_shard` の引数の列）を出すことを見張る＝値を変えない移植の一致の見張り（CLAUDE.md）。
期待値は §7.41 の bash をここへ写したもの（テストは `.md` を読まない決まり）。手書きの指示書をもう使わなくなったら消す。
"""
import json
import os

import pytest

pytestmark = pytest.mark.cpu_infra       # 作業の段取りの見張り（対戦の動きに関わらない）

from opcg_sim.loop import worker_job as W

JOBS = os.path.join(os.path.dirname(W.__file__), "jobs")
H = "/H"                                    # `~` の置き場（手書きの ~/ に当たる）
N13D = "opcg_sim/data/learned/nrel_r13d.npz"


def _plan(name, k):
    spec = json.load(open(os.path.join(JOBS, f"{name}.json")))
    return W.build_plan(spec, k, home=H, repo=f"{H}/opcg-sim-backend")


def _gen_expected(k):
    """§7.41 生成の本文（ブロック 2 の for J）。BASE＝2550000 + ceil(k/2)×1000 + (k が偶数なら 500)。"""
    K = f"{k:02d}"
    BASE = 2550000 + -(-k // 2) * 1000 + (500 if k % 2 == 0 else 0)
    out = []
    for J in range(4):
        out.append(f"python -m opcg_sim.loop.record_gen --games 120 --seed-base {BASE + 120 * J} --workers 4 "
                   "--sims 1280 --worlds 1 --budget 0 --select-rule q_min_n --q-min-frac 0.125 --root-prior-temp 2.0 "
                   "--eps-play 0.03 --eps-hold 0.03 --decks real --shard-games 10 "
                   f"--net opcg_sim/data/learned/nrel_r13d.npz --out n_records/n51_w{K}_p{J}".split())
    return out


def _arena_expected(C, k):
    KK, B = f"{k:02d}", k - 1
    BM, BR, BS = 3361000, 3381000, 3401000
    head = f"python -m opcg_sim.loop.arena_shard --candidate {H}/nrel_{C}.npz --baseline {N13D}".split()
    return [head + f"--pairs 175 --max-pairs 175 --bands 1 --seed-base {BM + B * 1000} --workers 4 --leaders random --decks real "
                   f"--out arena_{C}/{C}_real_w{KK}.jsonl".split(),
            head + f"--pairs 175 --max-pairs 175 --bands 1 --seed-base {BR + B * 1000} --workers 4 "
                   f"--out arena_{C}/{C}_mirror_w{KK}.jsonl".split(),
            head + f"--pairs 175 --max-pairs 175 --bands 1 --seed-base {BS + B * 1000} --workers 4 --leaders random --decks synth_roles2 "
                   f"--out arena_{C}/{C}_synth2_w{KK}.jsonl".split()]


def _train_expected():
    P51, P50, Z49 = (f"{H}/n{n}/n_records/n{n}_w*_p[0-3]" for n in (51, 50, 49))
    COMMON = ("--ablate rel --aux-weight 0.1 --lr 2e-4 --epochs 2 --pi-teacher q_improved --pi-steps-mult 1.0 --fast "
              f"--save-each-epoch --warm-start {N13D}").split()
    tr = ["python", "-m", "opcg_sim.learned.train.n_rel_train", "train"]
    band = (["python", "-m", "opcg_sim.learned.train.n_rel_band", "--in", P51, "--nrel", N13D]
            + [f"{H}/nrel_{x}.npz" for x in ("r22o", "r22", "r22o.ep0", "r22o.ep1", "r22.ep0", "r22.ep1")]
            + ["--policy", "--holdout-mod", "7", "--out", f"{H}/band_r22.json"])
    return [tr + ["--in", P51] + COMMON + ["--out", f"{H}/nrel_r22.npz"],
            tr + ["--in", P50, "--z-in", Z49] + COMMON + ["--out", f"{H}/nrel_r22o.npz"],
            band]


@pytest.mark.parametrize("k", [1, 2, 17, 32])
def test_gen_n51_matches_741(k):
    assert W.run_commands(_plan("gen_n51", k)) == _gen_expected(k)


@pytest.mark.parametrize("C", ["r22", "r22o"])
@pytest.mark.parametrize("k", [1, 9, 16])
def test_arena_r22_matches_741(C, k):
    P = _plan(f"arena_{C}", k)
    assert W.run_commands(P) == _arena_expected(C, k)
    assert P.branch == f"claude/arena-{C}-w{k:02d}"
    # 候補は訓練のブランチから取ってくる（手書きの git show r22net:n1_results/nrel_{C}.npz）
    assert any(f"FETCH_HEAD:n1_results/nrel_{C}.npz > {H}/nrel_{C}.npz" in st.show() for st in P.steps)


def test_train_r22_matches_741():
    P = _plan("train_r22", 1)
    assert W.run_commands(P) == _train_expected()
    assert P.branch == "claude/train-r22"
    # 教材の取得: 波 51 は 32 本・波 50／49 は 16 本ずつ（手書きの for k in $(seq -w …)）
    fetched = [st.show() for st in P.steps if "git archive FETCH_HEAD" in st.show()]
    assert len(fetched) == 64
    assert "claude/n51-w32 " in fetched[31] and "claude/n49-w16 " in fetched[-1]


def test_gen_branch_and_seed_range():
    """§7.41 の seed 帯 2551000〜2566500（k＝1〜32）とブランチ名。"""
    seeds = [int(W.run_commands(_plan("gen_n51", k))[0][6]) for k in range(1, 33)]
    assert seeds[0] == 2551000 and seeds[-1] == 2566500 and len(set(seeds)) == 32
    assert _plan("gen_n51", 7).branch == "claude/n51-w07"
    with pytest.raises(SystemExit):
        _plan("gen_n51", 33)
