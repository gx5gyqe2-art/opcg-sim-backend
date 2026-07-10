"""バッチ式アクター/ラーナー分離の共通部品（postdistill 追い学習の並列化・司令塔 2026-07-10）。

docs/reports/batched_selfplay_design_20260710.md。オンライン自己対戦の直列フィードバックループを
「凍結net で並列生成 → まとめて学習 → 更新 → 繰り返し」のラウンド制に組み替える。生成は独立作業に
なるので複数セッション（別コンテナ）で完全並列化できる（蒸留v2の並列生成と同じ形）。

協調は git のみ:
- **net枝**（learner 単独writer）: `p3ckpt/{value.npz, policy.npz, manifest.json}`。manifest に
  `round`（学習ラウンド）・`cum_games`・`consumed`（各generatorの最終消費 batch_id）。
- **data枝**（generator ごとに1本・そのgeneratorが単独writer）: `p3data/{batch.npz, meta.json}`。
  meta に `batch_id`（generator内で単調増加）・`against_round`（生成に使ったnetのround）・`games`。

このモジュールは **git 非依存の純粋ロジック**（鮮度フィルタ・消費更新・リングバッファ）だけを持つ＝
単体テスト可能。git 入出力は pd_gen.py / pd_learn.py 側に置く。
"""
import numpy as np


def is_fresh(meta, consumed, current_round, max_staleness):
    """このバッチを学習に採用すべきか。

    採用条件（両方満たす）:
      1. 未消費: meta.batch_id が consumed[wid] より新しい（重複学習を防ぐ）。
      2. 十分新鮮: against_round >= current_round - max_staleness（古すぎる off-policy データを捨てる）。
    返り値: ("accept" | "stale" | "seen")。
    """
    wid = meta["worker"]
    if meta["batch_id"] <= consumed.get(wid, -1):
        return "seen"
    if meta["against_round"] < current_round - max_staleness:
        return "stale"
    return "accept"


def plan_consumption(metas, consumed, current_round, max_staleness):
    """複数 generator の meta 群 → 採用リスト・スキップ理由の内訳。

    metas: [meta, ...]（各 generator の最新 batch）。
    返り値: (accepted[list[meta]], skipped[dict wid->reason])。
    """
    accepted, skipped = [], {}
    for meta in metas:
        verdict = is_fresh(meta, consumed, current_round, max_staleness)
        if verdict == "accept":
            accepted.append(meta)
        else:
            skipped[meta["worker"]] = verdict
    return accepted, skipped


def update_consumed(consumed, accepted):
    """採用した meta で consumed（wid->最終batch_id）を更新した新 dict を返す。"""
    out = dict(consumed)
    for meta in accepted:
        out[meta["worker"]] = max(out.get(meta["worker"], -1), meta["batch_id"])
    return out


def updates_for(new_games, games_per_update, max_updates):
    """1波の新規 games 数 → その波で回すべき学習ラウンド数（勾配パスの倍率）。

    **薄まり防止の核**（docs/reports/batched_selfplay_design_20260710.md §「off-policy 逓減」）:
    並列で K 本ぶんのデータが一度に来ても、1波1ラウンドだと「1局あたりの勾配露出」が K 分の1に薄まる。
    新規局数に比例して学習回数をスケール＝games:updates 比を直列(1バッチ:1ラウンド)と一致させる。
    - games_per_update: この局数ごとに1ラウンド（既定=1バッチの games＝K=1で従来と同一・後方互換）。
    - max_updates: 暴発防止の上限（1波で回す最大ラウンド数）。
    返り値は 1..max_updates の整数（新規0でも最低1）。
    """
    if games_per_update <= 0:
        return 1
    n = int(round(new_games / games_per_update))
    return max(1, min(int(max_updates), max(1, n)))


def ring_append(buf, new_arrays, cap):
    """リプレイバッファ（dict of np arrays）へ連結し末尾 cap 件に切る（相関緩和・忘却対策）。

    buf/new_arrays は同じキー集合の dict。buf が空（None）なら new をそのまま切って返す。
    """
    keys = list(new_arrays.keys())
    if not buf:
        merged = {k: np.asarray(new_arrays[k]) for k in keys}
    else:
        merged = {k: np.concatenate([buf[k], new_arrays[k]]) for k in keys}
    n = len(merged[keys[0]])
    if n > cap:
        merged = {k: merged[k][-cap:] for k in keys}
    return merged
