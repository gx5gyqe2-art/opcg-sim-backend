"""L5 の比べ方: py と rs の crossing_bridge の JSON を、時間の欄（最上位の `seconds` だけ）を除いて 1 バイトで比べる
（ファイルのバイト列から `"seconds": <数>` を除いたものをそのまま比べる＝書き直しをしない）。

    python rust2a/compare_l5.py <py.json> <rs.json>
"""
import json
import sys

import re

PAT = re.compile(rb'"seconds": [0-9.eE+-]+')


def main(a, b):
    ra, rb = open(a, "rb").read(), open(b, "rb").read()
    sa, sb = PAT.findall(ra), PAT.findall(rb)
    assert len(sa) == len(sb) == 1, (sa, sb)
    ja, jb = PAT.sub(b'"seconds": X', ra), PAT.sub(b'"seconds": X', rb)
    same = ja == jb
    print("dropped:", sa, sb)
    print("bytes: %d vs %d  identical=%s" % (len(ja), len(jb), same))
    if not same:
        for i, (x, y) in enumerate(zip(ja, jb)):
            if x != y:
                print("first diff at byte", i, ja[max(0, i - 200):i + 200])
                print("                  ", jb[max(0, i - 200):i + 200])
                break
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
