# 事象報告: アリーナ計測 b_p04 が seed 904002 で停止する（戦闘箱の組合せ爆発）

以下をそのまま別セッションへ貼って調査・修正を依頼できます。

---

## 依頼

`claude/cpu-spec-improvements-yw91jd`（HEAD = `fd04ec9`）で、アリーナ計測が**特定の局面で1回の
`decide()` から戻ってこなくなる**。シャードが丸ごと止まるため計測が進まない。原因の切り分けと
対処方針の判断をお願いしたい。

## 再現条件

```bash
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/arena_resume.py \
  --candidate opcg_sim/data/learned/gen15_value.npz,opcg_sim/data/learned/gen15_policy.npz \
  --baseline "opcg_sim/data/learned/gen14_value.npz,opcg_sim/data/learned/gen14_policy.npz" \
  --pairs 30 --bands 3 --seed-base 904000 --max-pairs 10 --workers 4 \
  --leaders random --decks synth --out /home/user/arena_b_p04.jsonl
```

- 環境: Python 3.11.15 / numpy 2.4.6 / 4 CPU（コンテナに numpy が未導入だったため `pip install numpy` で導入した。それ以外の環境変更なし）
- **seed 904002 で停止**。決定論なので撃ち直しても同じ場所で止まる（2回の独立実行で再現、同一 seed）。
- 台帳に書けたのは 2/30 ペアのみ。スコアは2回の実行で完全一致（下記）＝決定論自体は壊れていない。

```json
{"seed": 904000, "score": 0.0, "leaders": ["OP02-002", "OP15-001"], "games": [0.0, 0.0], "turns": [10, 11]}
{"seed": 904001, "score": 1.0, "leaders": ["ST22-001", "OP05-002"], "games": [1.0, 0.0], "turns": [11, 11]}
```

## 症状

- ワーカー4本中3本はアイドル（各 CPU 約12分でタスク待ち）。残り1本が **3時間48分ずっと 99.9% CPU**。
- `py-spy dump` を複数回取るとスタックが毎回変化する ⇒ **デッドロックではなく純粋に計算が終わらない**。
- 例外は上がらないので、`fd04ec9` で入れた **void 化では救えない**。`MAX_STEPS` は手数のカウンタで、
  1回の `decide()` の中に閉じ込められている間は進まない。
- `pool.imap` は入力順に書き出すため、904003 以降が完了していても 904002 が詰まっている間は
  台帳に flush されない ⇒ **シャード全体がブロックされる**。

## スタック（py-spy、内側から）

```
resolve_battle_inplace  (opcg_sim/src/learned/mcts.py:117)   box_depth=0, max_plies=12, name="p2"
resolved_branch_values  (opcg_sim/src/learned/mcts.py:157)
resolve_battle_inplace  (opcg_sim/src/learned/mcts.py:104)   box_depth=1
resolved_branch_values  (opcg_sim/src/learned/mcts.py:157)   box_depth=1
_expand                 (opcg_sim/src/learned/mcts.py:378)
_simulate               (opcg_sim/src/learned/mcts.py:478)
_descend_journal        (opcg_sim/src/learned/mcts.py:450)
_simulate               (opcg_sim/src/learned/mcts.py:492)   ... ×4段
run                     (opcg_sim/src/learned/mcts.py:289)
decide                  (opcg_sim/src/core/cpu_learned.py:668)
_learned                (game_driver.py:189)
run_game                (game_driver.py:374)                 seed=904002, prev_turn=11
play_game               (cpu_arena.py:175)                   seed=904002
_play_pair_detail       (tests/scripts/promotion_gate.py:142) seed=904002
```

フレームの locals（抜粋）:

- `legal` は **`SELECT_COUNTER` の列挙**（カウンター候補が多数並ぶ）
- `resolved_branch_values` の `vals: [-0.9201670319632933, -0.9201670319632933]`
  ⇒ **区別のつかない同値の枝**に計算を費やしている
- `saved_events` に `{"type": "COUNTER", "player": "p2", "card_name": "アブサロム", ...}`

## 分析（要確認）

相互再帰 `resolved_branch_values` ↔ `resolve_battle_inplace` の**深さ自体は有界**に見える:

- `resolve_battle_inplace` は `resolved_branch_values(..., box_depth=box_depth - 1)` と減らす（mcts.py:104）
- `resolved_branch_values` は `resolve_battle_inplace(..., box_depth=box_depth)` と**据え置き**（mcts.py:157）
- 1往復で 1 減るので `BOX_RESOLVE_DEPTH = 1`（config.py:195）なら 1→1→0→0 で止まる

したがって**無限再帰ではなく、幅（branching）の爆発**だと考えられる。1つの箱の中で

- 各 ply（`QUIESCE_MAX_PLIES = 12`, config.py:88）× `len(legal)` 本の枝
- 各枝がさらに `resolve_battle_inplace` で最大 12 ply 進む

を `_expand` のたびに、しかも MCTS のシミュレーション回数ぶん繰り返す。カウンターを大量に
持つ手札（turn 11・`--decks synth` のランダム対面）で `len(legal)` が膨らむと現実的な時間で
終わらなくなる、という筋。上記 `vals` が同値である点も、**判別に寄与しない枝**へコストを
払っている傍証。

> なお `mcts.py:370` 付近のコメントは「カウンターの組合せに費やしていた訪問がメイン判断へ回る」
> と、まさにこの組合せコストを畳む意図を述べている。意図どおりに効いていない局面がある可能性。

## 判断してほしいこと

1. **枝刈りを入れるか**（エンジン修正）: `resolved_branch_values` で同値枝の重複排除、
   `len(legal)` の上限、あるいは箱の中の探索に時間/ノード予算を持たせる。
   ゲート・アリーナの既存結果との互換性（挙動ベースラインの再生成要否）も併せて判断が必要。
2. **計測側に時間上限を入れるか**（ハーネス修正）: 1ペアに wall-clock 上限を設け、超過したら
   `fd04ec9` の void と同じ扱いで台帳に残して次へ進む。計測条件
   （seed-base / pairs / leaders / decks / candidate / baseline）は不変のまま、シャードが
   1局面で全損するのを防げる。`imap` のヘッドブロッキングも併せて要検討
   （`imap_unordered` + seed 明記で書き出す等）。
3. **b_p04 シャードをどうするか**: 現状 2/30。上記いずれかが入るまで進められない。

## 補足

- 計測ワーカー側では**エンジン・テストのコードは一切変更していない**。計測条件も未変更。
- 停止中のプロセスは診断のため生かしてある（必要なら追加の py-spy 取得が可能）。
- 旧フォーマットの台帳（`score` のみ・`leaders` なし）は `/home/user/arena_b_p04.jsonl.oldfmt.bak` に退避済み。
