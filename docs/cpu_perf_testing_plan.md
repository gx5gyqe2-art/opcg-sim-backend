# 計画: CPU 性能テスト運用＋arena の learned(Gen2) 対応

既定 CPU が learned(Gen2) になったのに、強度計器（arena/arena_parallel）は L1(hard) 前提のまま。
**本番既定 CPU の強さを測り・退行を止める運用**を、arena の learned 対応と合わせて整える。**本書は計画（未実装）**。
実装完了後に TEST_SPEC §5（品質ゲート）へ吸収する。

## 1. 現状（調査済み・2026-07-04）

- **強度A/B は L1 専用**: `cpu_arena.play_game` は両席 `make_seat(kind="arena")`＝L1(`decide_guarded`)固定。
  `arena`/`arena_paired`/`arena_parallel.paired_play` もこれを回すので **Gen2 の勝率・Elo を測れない**。
- **learned 席は存在する**: `game_driver.make_seat(kind="learned")`（PR-D3）。ただし play_game が使っていない。
- **learned のノブが L1 と別系統**: arena 席は info_policy/CRN rng/pimc/budget/search/coeffs（すべて L1 概念）。
  learned は sims/c_puct（MCTS）＝オーバーライドの意味が違う。
- **【最大の壁】ネットがプロセス共有シングルトン**: `cpu_learned._STATE` が固定パス `gen2_*.npz` を1組だけ
  キャッシュ（`decide_learned` は net を引数で選べない）。**1プロセス内で2つのネットを同居できない**＝
  「新Gen vs 凍結Gen2」の net-vs-net が今の構造では組めない。
- **regret/realize・monotonicity は L1 固有**（本計画の対象外。learned 版は別設計＝§6-4）。

## 2. 測定モードと実現性

| モード | 用途 | ネット数 | 実現性 |
|---|---|---|---|
| **learned vs L1(hard)** | Gen2 の**絶対強度アンカー**（「出荷 hard に何 Elo 勝つか」＝安定した連続監視点） | 1 | 席を差すだけ＝**容易**（A1） |
| **learned vs learned（新Gen vs 凍結Gen2）** | **昇格判定**（net 更新で強くなった/退行してないか） | 2 | シングルトン破壊が必要＝**要リファクタ**（A3） |
| **同一ネット・sims振り** | 思考時間（sims）の伸びしろ A/B | 1 | 席の sims を席別化＝容易（A1 に相乗り） |
| コード版 A/B（MCTS/決定化の変更） | 推論コード変更の強度影響 | 1（同net） | git チェックアウト A/B が基本（in-process 不要） |

→ **絶対アンカー（learned vs L1）を先に安く回せるようにし（A1）**、運用を固め（A2）、**net-vs-net の昇格ゲート（A3）**は
シングルトン解消とセットで後追いする。

## 3. 技術課題

1. **play_game の席をエンジン非依存に**: 現在 difficulty 文字列→arena 席固定。`difficulty=="learned"` を learned 席へ、
   かつ learned のノブ（sims/c_puct）を席別に渡せるようにする。`make_seat(kind="learned", sims=…)` は既にあるので
   play_game/paired_play の**席生成の分岐**を足すのが主。
2. **決定論・CRN の意味**: L1 の `separate_policy_rng`（CRN）は learned では numpy rng 分離に相当。learned の rng は
   global random 由来（PR-D2）なので、席別 CRN は「席ごとに派生 numpy Generator を渡す」で表現する
   （`make_seat(kind="learned")` に rng 注入口を足す）。対照ペア（antithetic）は seed 入替で従来どおり。
3. **並列 arena のネット読み込み**: `arena_parallel` はワーカープロセスごとに DB/net をロード。learned は
   `_lazy_init` が1回ロード＝learned vs L1 なら各ワーカー1ネットで OK。net-vs-net は各ワーカーが2ネット必要（§課題4）。
4. **【A3 の核】シングルトン解消**: `cpu_learned` に**インスタンス API**を足す
   （例 `LearnedEngine(value_path, policy_path)` ＝ネットを明示ハンドルで保持し `decide(manager, player, …)` を持つ）。
   既存 `decide_learned`（シングルトン・本番既定 CPU 経路）は薄いラッパとして温存＝本番挙動不変。
   これで arena が「凍結Gen2ハンドル vs 新Genハンドル」を同一プロセスで戦わせられる。
5. **凍結ベースラインの固定**: 出荷 Gen2＝`gen2_value.npz`/`gen2_policy.npz` を**ハッシュで固定**し、
   昇格時のみ更新＋レポート記録。arena の baseline はこのハッシュのネットを指す。

## 4. 実装計画（PR 分割）

| PR | 内容 | ゲート |
|---|---|---|
| **A1** | play_game/paired_play を **learned 席対応**（difficulty="learned"・sims 席別）。learned vs L1 と sims 振りを測れる。`cpu_arena arena-paired --challenger learned` 相当の CLI | learned vs L1 が決着・同一 seed で再現（`test_cpu_arena` に learned 対戦の機械健全性ケース追加） |
| **A2** | **`perf_gate.py`（運用ワンコマンド）**: 凍結ベースライン（npz ハッシュ固定）に対する Gen2 の勝率→Elo・ペア単位 CI・latency・（learned の Q ではなく）決着率/手数の非退行を1コマンドで PASS/FAIL 出力。`--quick`（少ペア）/`--full`（本走） | 自己対戦で PASS/FAIL が安定・TEST_SPEC §5 へ運用追記 |
| **A3** | **net-vs-net**: `cpu_learned` にインスタンス API（`LearnedEngine`）を足し（シングルトンはラッパで温存＝本番不変）、arena が「凍結Gen2 vs 新Gen」を戦わせる。昇格ゲート＝elo_lo>0（強い）or 非退行なら elo_hi>−15 | 同一ネット同士で ≈50%・凍結 vs 凍結が対称・全テスト |
| **A4（任意）** | learned 用の正しさ計器（Q値ギャップ＝learned版 regret、net 単調性）。regret/realize/monotonicity の learned アナログ | 別設計（本計画のスコープ外だが将来枠として明記） |

**実装順は A1→A2→A3**。A1（アンカー）で「Gen2 が hard にどれだけ勝つか」を先に可視化、A2 で運用化、
A3（シングルトン解消）で net 更新の昇格ゲートを開通。

## 5. 運用（A2 で定めるルール・前計画の「性能テスト運用案」を Gen2 中心へ）

- **凍結ベースライン = 出荷 Gen2**（npz ハッシュ固定）。L1(hard) は較正用の固定参照相手として併走（Elo の物差し）。
- **昇格条件（net 更新時）**: `perf_gate.py --full` で elo_lo>0（有意に強い）＋ latency 予算内（1手1秒）＋
  決着率/手数の非退行 ＋ held-out 実デッキで崩れない。非退行目的なら elo_hi>−15。
- **記録**: 結果は `docs/reports/` に日付スナップショット、リプレイ種を貼って再現可能に。
- **定期実行（任意）**: Routine で週次フルキャンペーン→結果サマリをチャット報告（CI に重すぎる強度測定を定期便化）。

## 6. 未解決の設計判断（着手前に確定したい）

1. **A1 の CLI 形**: 既存 `arena-paired` を拡張（`--challenger learned --challenger-sims 160`）か、新サブコマンド
   `perf`（learned 前提）を切るか。
2. **CRN の扱い**: learned の席別 numpy rng 分離をどこまで作り込むか（分散低減の価値 vs 複雑さ）。
3. **A3 のインスタンス API 範囲**: `LearnedEngine` を最小（value/policy/vocab/game を保持）に留めるか、
   sims/c_puct もハンドルに載せるか。本番 `decide_learned` の**挙動ビット不変**は必須ゲート。
4. **learned の正しさ計器（A4）**: regret（Q値ギャップ）・monotonicity（net）の learned アナログを作るか、
   arena の勝率だけで足りるとするか。
5. **CI 負荷**: learned の arena は MCTS で重い。`perf_gate --quick` は少ペア＋低 sims で有界化。本走は手動/定期。
6. **凍結の粒度**: baseline を npz ハッシュで固定するか、`gen2` のような tag で固定するか。

## 7. 前提（既に満たされている土台）

- `game_driver.make_seat(kind="learned")`（PR-D3）＝learned 席は存在。play_game が使うよう配線するのが A1 の主。
- learned の seed 再現（PR-D2）＝arena の決定論・対照ペアが learned でも成立する必要条件は充足。
- 共通ドライバ `run_game`（設計⑥）＝arena の対局ループは新規実装せず席を差し替えるだけ。
