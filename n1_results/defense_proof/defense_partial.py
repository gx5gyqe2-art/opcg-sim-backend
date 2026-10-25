"""手が分かれた局の守りの監査を、分かれた手から先だけ打つ（WP `rs-defense-proof`・1 回きり・`rs-window-tree-blocker` の形）。

`defense_audit.py run` と同じ局（同じ設定・同じ seed）を打つが、`diverge_<B>.jsonl` の最初に手が分かれた点（step）より前の
判断点は判定しない（その局面と手は 2 つの形で同じ＝監査も同じ）。自分のメインの最後の点の控え（`main_point`／`arm_end`）は
前から取る（終わり方の判定の材料）。判定は step ≥ 分かれた点だけ。入れない形と入れた形（`--defense-proof`）の両方をこれで打つ。

  python defense_partial.py <B> run --seeds <file> --workers 4 --net <r13d> --out <dir> [--defense-proof]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "scripts"))
import defense_audit as DA  # noqa: E402

B = int(sys.argv.pop(1))
D = os.path.dirname(os.path.abspath(__file__))
DIV = {}
for d in map(json.loads, open(os.path.join(D, f"diverge_{B}.jsonl"))):
    if d["diverged"]:
        DIV[d["seed"]] = min(p["step"] for p in d["probes"] if not p["same"])


class PartialAudit(DA.Audit):
    def __init__(self, seed, a):
        super().__init__(seed, a)
        self.from_step = DIV[seed]

    def response_point(self, game, name, turn, step, move, pend):
        if step >= self.from_step:
            super().response_point(game, name, turn, step, move, pend)

    def end_point(self, game, opp, turn, step):
        if step >= self.from_step:
            super().end_point(game, opp, turn, step)


DA.Audit = PartialAudit

if __name__ == "__main__":
    DA.main()
