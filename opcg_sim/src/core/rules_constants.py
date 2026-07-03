"""ルール上の定数（core 各層と actions ディスパッチが共有）。

`gamestate` へ import されると循環（actions → gamestate）を招くため、定数のみを持つ葉モジュールに
切り出す。`gamestate` からは再エクスポートで従来の参照互換を保つ。
"""

# 自己制限（self_cannot）の制限キー。parser が RULE_PROCESSING + status=これらで生成し、
# actions のプレイヤーレベル・ハンドラが player.restrictions に記録、各アクション地点で enforce する。
SELF_RESTRICTION_KEYS = {
    "CANNOT_PLAY_FROM_HAND",      # 手札からカードをプレイできない
    "CANNOT_PLAY_CHARACTER",      # キャラ(カード)を登場できない（min_cost で「コストN以上」に限定可）
    "CANNOT_DRAW_BY_EFFECT",      # 自分の効果でカードを引くことができない
    "CANNOT_LIFE_TO_HAND",        # 自分の効果でライフを手札に加えられない
    "CANNOT_ATTACK_LEADER",       # リーダーにアタックできない
    "CANNOT_ACTIVATE_DON",        # キャラの効果でドン‼をアクティブにできない
}
