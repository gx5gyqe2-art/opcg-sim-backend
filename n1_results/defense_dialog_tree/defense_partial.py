"""手が分かれた局の守りの監査を、分かれた手から先だけ打つ（WP `rs-defense-dialog-tree`・1 回きり・`n1_results/defense_proof/defense_partial.py` の形）。

`defense_audit.py run` と同じ局（同じ設定・同じ seed）を打つが、`diverge_4462000.jsonl` の旗 <MODE>（life1／all）で最初に手が分かれた点
（step）より前の判断点は判定しない（その局面と手は 2 つの形で同じ＝監査も同じ）。自分のメインの最後の点の控えは前から取る。
入れない形の数は既存の監査（`n1_results/reaudit_r29f4/defense_all/`）を step ≥ 分かれた点で絞って使う（決定的に同じ）。

  python defense_partial.py <MODE> run --seeds <file> --workers 4 --out <dir> --defense-dialog-tree <MODE>
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "scripts"))
import defense_audit as DA  # noqa: E402

MODE = sys.argv.pop(1)
D = os.path.dirname(os.path.abspath(__file__))
DIV = {d["seed"]: d["first"][MODE] for d in map(json.loads, __import__("gzip").open(os.path.join(D, "diverge_4462000.jsonl.gz"), "rt"))
       if d["first"][MODE] is not None}


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
