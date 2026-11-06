"""WP reaudit-r29f4（§7.84）: `tests/scripts/blunder_audit.py` を出荷の r29f4 に向けて回す薄い入口（器は変えない）。

器の定数だけを差し替える: 出荷の席・確かめの出荷の側＝r29f4（`E.DEFAULT_NET`）、
審判 1（器の鍵 "r13d"）＝r29f4（160 回）、審判 2（器の鍵 "r20"）＝r28x4（160 回・ロールバック先）。
出力の鍵名は器のまま（r13d＝審判 1・r20＝審判 2 と読む）。
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import blunder_audit as BA  # noqa: E402
NET = "opcg_sim/data/learned/nrel_r29f4.npz"
BA.NET = NET
BA.REFS = {"r13d": NET, "r20": "opcg_sim/data/learned/nrel_r28x4.npz"}
# 器の recheck_point は `_G["verify_both"]`（verify の初期化だけが置く）を出力の札に写すだけ＝recheck では意味が無い。
# 置かれていないと KeyError で落ちる（§7.50 の取り込みの残り）ので、器を変えずにここで札を置く（recheck は出荷の側を使わない）。
BA._G.setdefault("verify_both", "n/a(recheck)")
if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "play" and "--net" not in argv:
        argv += ["--net", NET]
    sys.exit(BA.main(argv))
