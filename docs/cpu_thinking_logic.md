# CPU 思考ロジック詳細図（決定パイプライン・L1 単一系統）

> **種別: 仕様（正本）**。実装変更に追従して最新に保つ。詳細な散文仕様は [`SPEC.md` §2.5](SPEC.md)、
> 評価式 L1 の設計は [`reports/cpu_eval_redesign_card_currency_20260625.md`](reports/cpu_eval_redesign_card_currency_20260625.md)（点・履歴）を参照。
> 本書はその**全体フローを1枚で俯瞰する図**＝コードの構造（呼び出し経路・分岐・責務分担）を示す。

対象ソース: `opcg_sim/src/core/cpu_ai.py` / `cpu_eval_v2.py` / `effects/resolver.py` / `api/{app,decide_client}.py` / `tools/decide_worker.py`

---

## 0. 全体（本番の呼び出し経路）

```
 POST /api/game/cpu_step  (app.py)
        │
        ├─ plan_cache に手順キャッシュ有り？ ──Yes─→ _cached_cpu_move
        │     (replay/ponder/speculate で先読み済みの手列を再生・体感速度用)
        │            │ キャッシュmiss/前提崩れ
        │            ▼
        │      plan_segment ──→ decide_client.plan_segment ──→ cpu_ai.plan_turn
        │                                                          (1ターン分の手列を計画)
        └─No─→ decide_client.decide
                   │
                   ├─ OPCG_PYPY_WORKER=1 ? ─Yes→ Unix socket → PyPy worker
                   │                                  (decide_worker.py)
                   │                                       └→ cpu_ai.decide_guarded / plan_turn
                   └─No(or IPC失敗) ───────────────→ cpu_ai.decide_guarded  (インプロセス)

 ※ profile/plan 引数は撤去済み。worker IPC タプルも (manager,pid,difficulty,mem,rng,want_trace,read_ahead)。
 ※ plan_turn / decide_cached は decide_guarded を共有 mem で回す薄いループ（手列化）。
```

---

## 1. decide_guarded  ── 終了保証 + mem/killer 管理のみ（旧「暴走防止」を簡素化）

```
decide_guarded(manager, player, mem, ...)
   │
   ├─ mem のターン更新:  turn != turn_count なら  total=0 / killers={} にリセット
   │       (mem キーは turn / total / killers の3つだけ。旧 counts は撤去)
   │
   ├─ moves = get_legal_actions(player);   空 → return None
   ├─ end_move = TURN_END を探す
   │
   ├─【終了保証】 end_move あり かつ mem.total >= TURN_ACTION_CAP(=60)
   │        → 強制 TURN_END を返す   ◀── 無限ループの最終防壁（正当ターンは到達しない）
   │
   ├─ killer 表 ks = mem["killers"]  （_USE_PV_ORDER & _USE_PV_CROSS_DECIDE 時のみ・α-β手順序）
   │
   ├─ move = decide(manager, player, moves=moves, killer_state=ks, ...)   ◀── 本体へ
   │
   └─ move 採用なら  mem.total += 1   （← この加算が TURN_ACTION_CAP を駆動）
            return move

  ※ 旧 REPEAT_CAP（同一起動効果の3回除外）は撤去。
    起動効果の反復はエンジンのコストゲートが自己制限する（→ §6）。
```

---

## 2. decide  ── 候補生成 → 探索方式の分岐 → TURN_END fold

```
decide(manager, player, moves, ...)
   │
   ├─ 最上位が「対象選択」？  sel = _selection_moves()
   │     Yes → moves = sel ;  is_selection = True   (KO/除去/バウンス等の単一対象を候補展開)
   │
   ├─ moves 空 → None ;   moves 1個 → そのまま返す（forced）
   │
   ├─ 枝刈り:  _prune_don_moves   (無意味なドン付与を除外・B-2)
   │           _prune_futile_attacks (倒せない/届かない攻撃を除外)
   │
   ├─ 情報方針:  see_opp_hand, opp_public_only = _resolve_info_policy(info_policy)
   │              (既定 fair = 相手手札の中身を読まない / cheat = 読む※診断用)
   │
   ├─ 採点（3分岐）:
   │     ┌─ is_selection → 各候補を 1-ply 即時評価  _simulate_and_eval
   │     │     (確定効果の対象選びは深読みでwashout/逆転するので 1-ply が信頼信号)
   │     ├─ pimc_worlds>=2 → _pimc_scored   (隠れ情報を K 世界に決定化して平均・§5)
   │     └─ それ以外 → _scored_search        (α-β + ビーム・§3)
   │
   ├─ rng.shuffle(scored) → best_score, best_move = max(scored)
   │
   └─【TURN_END fold】 best_move ≠ end かつ  best_score <= end_score + _ACT_MARGIN
            → chosen = TURN_END（畳む）   ◀── 無意味な展開/不利攻撃/効かない付与を採らない
            else chosen = best_move
       (trace 指定時のみ _fill_decision_trace。rng 状態は保存→復元＝進行に影響させない)
       return chosen

  ※ 旧 plan の act_margin_mult は撤去 → margin は定数 _ACT_MARGIN。
```

---

## 3. _scored_search → _search  ── α-β + ビーム + ターン境界 settle

```
_scored_search(manager, name, moves)
   │
   ├─ 各 root 手を 1-ply prelim 採点（_score_move_1ply）
   ├─ prelim 上位 HARD_ROOT_BEAM 手 ＋ TURN_END を「深掘り対象」に選別
   ├─ 各深掘り手で _search(α-β) を回す（killer/PV 手順序で枝刈り効率化）
   └─ (prelim, deep) を返す  → decide が max を取る

_search(manager, root_name, alpha, beta, ply, budget, is_max...)
   │
   ├─【終端】winner 確定 → ±(W_WIN − ply)        （最短リーサルを ply 割引で優先）
   ├─【打ち切り】ply >= _effective_max_ply()  or  budget 切れ
   │        → _settle_eval（静止点へ整流して静的採点）
   │
   ├─ 子手を列挙 → ビーム: children.sort(reverse=is_max) で上位 K に剪定
   │        (max=高評価順 / min=低評価順 ＝ それぞれの手番側に最善な子を残す)
   │
   ├─ for child in beam:
   │      v = _search(child, ply+1, is_max 反転, budget-1 ...)   ◀── 再帰
   │      is_max:  best=max(best,v); alpha=max(alpha,v); if alpha>=beta: break  (βカット)
   │      min   :  best=min(best,v); beta =min(beta, v); if alpha>=beta: break  (αカット)
   │
   └─【葉】evaluate(node, root_name, see_opp_hand)   ◀── §4 へ

_settle_eval: TURN_END/既定解決で相手ターン開始まで整流 → 静的 evaluate。
              戦闘応答（SELECT_BLOCKER/SELECT_COUNTER・どちら側でも）は**既定 PASS で解決**。
              整流中に勝敗確定なら ±(W_WIN − ply)（ply 割引＝最短の止めを優先）。未確定なら静的 evaluate。
              ※ 旧 _SETTLE_CONFIDENCE/_settle_discount（楽観是正・plan 由来）／#4 settle_threat_penalty
                （Elo中立で撤去・2026-06-28）はいずれも削除済み＝settle は純粋な静止点採点のみ。
```

---

## 4. 評価ラッパー  ── evaluate → evaluate_base → L1

```
evaluate(manager, me_name, see_opp_hand)
   │   （旧・学習価値ブレンド層 _value_blend は Elo中立で撤去・2026-06-28＝evaluate は evaluate_base の別名）
   ▼
evaluate_base(manager, me_name, see_opp_hand)
   │   （手書きJ値・評価フラグ・profile・学習価値は全撤去。分岐なし）
   └─ return cpu_eval_v2.evaluate_v2(...)     ◀── 唯一の評価 = L1（§5）
```

---

## 5. L1 評価本体  cpu_eval_v2.evaluate_v2  ── 単一通貨「カード」

```
evaluate_v2(manager, me_name, see_opp_hand) -> float
   │
   ├─ winner == me  → +W_WIN     /   winner == 相手 → −W_WIN     （終端の符号はここで処理）
   │
   ├─ 共有状態（生存・圧力）を先に計算:
   │     my_clock  = _clock_of(me, opp)    ＝ 自分の毎ターン削り期待（有効パワー和/1000）
   │     opp_clock = _clock_of(opp, me)
   │     my_life, opp_life = max(len(life),1)
   │     γ_surv   = clamp(my_life/(opp_clock+ε), 0..1) ** V2_KAPPA      （あと何ターン展開を使えるか）
   │     amp      = 1 + V2_LAMBDA * _decay(opp_clock/my_life)           （圧力でカウンター価値を増幅）
   │     don_budget = len(don_active)
   │
   ├─ R_me  = R_life(me)  + R_board(me)  + R_hand(me, γ, amp, full=True)        + R_don(me)
   ├─ R_opp = R_life(opp) + R_board(opp) + R_hand(opp, γ', amp', full=see_opp_hand) + R_don(opp)
   ├─ Tele  = _telegraph(me→opp) − _telegraph(opp→me)
   │
   └─ return ( R_me − R_opp + Tele ) * V2_SCALE      （探索閾値 _ACT_MARGIN と同オーダー）


 ┌─ R_life ──────────────────────────────────────────────────────────────────┐
 │  near = min(life, knee=2) … 薄域は precious  (× V2_W_LIFE_PRECIOUS)         │
 │  far  = max(0, life-2)    … 厚域は安い        (× V2_W_LIFE_HIGH)            │
 │  − V2_W_DECK × max(0, DECK_DANGER − deck残)   ← デッキ切れ距離（同じ通貨）   │
 │  ＝ ダメージレースの主役（凹型＝薄いほど1枚が高い）                          │
 └────────────────────────────────────────────────────────────────────────────┘
 ┌─ R_board ─────────────────────────────────────────────────────────────────┐
 │  Σ 体ごと: V2_W_BODY × (有効パワー/1000)   ← トレード/制圧価値のみ          │
 │  レスト体は「自ターン終了が葉」のときだけ V2_REST_DISCOUNT で割引            │
 │  顔打点は持たない（殴った結果は探索が相手 R_life 減で拾う＝二重計上回避）    │
 └────────────────────────────────────────────────────────────────────────────┘
 ┌─ R_hand（1枚＝排他資源: 展開 or カウンター）───────────────────────────────┐
 │  自手札(full): 各札の Δ=dev−ctr を降順に、ドン予算分を展開(dev)・残りをctr  │
 │       dev_v = V2_W_DEV × γ_surv     ctr_v = V2_W_CTR × counter × amp        │
 │       （貪欲ナップサック＝dev コスト一律1.0で最適）                          │
 │  相手手札(中身不可視): 枚数 × max(dev,ctr) × V2_OPP_HAND_UPLIFT（上振れ補正）│
 └────────────────────────────────────────────────────────────────────────────┘
 ┌─ R_don ─┐   ┌─ Tele（リーサル番兵・防御控除はここ一箇所だけ）─────────────┐
 │ V2_W_DON│   │ reach = _clock_of(atk,def)                                  │
 │ × active│   │ eff = max(0, reach − (defのアクティブブロッカー数)          │
 │  don 枚 │   │                    − len(def.hand)*0.5 ※枚数ベース概算)     │
 └─────────┘   │ return V2_W_TELE × min(eff, max(def_life,1))                │
              └──────────────────────────────────────────────────────────────┘
```

---

## 6. 暴走防止の責務分担（REPEAT_CAP 撤去後）

```
 ┌── 決定層（cpu_ai） ──────────────────────────────────────────────┐
 │  TURN_ACTION_CAP = 60   … 1ターン手数の最終防壁（終了保証）        │
 │       └ 正当ターンは到達しない＝思考に干渉しない（実測 0 発火）    │
 │  探索の各種上限: budget / _effective_max_ply / _SETTLE_LIMIT /     │
 │                  _DRAIN_LIMIT / read_ahead の max_steps            │
 └──────────────────────────────────────────────────────────────────┘
 ┌── エンジン層（resolver / gamestate）＝起動効果の自己制限 ─────────┐
 │  _has_activatable_main / resolve_ability の三条件:                 │
 │    ① 条件成立(condition)                                          │
 │    ② ターン使用回数 < 【ターン1回】等の上限                       │
 │    ③ コスト充足 _can_satisfy_node                                 │
 │         ・REST コストはアクティブ対象が必要（レスト済みは不可）    │
 │         ・ref_id='self' は **source 限定**（←今回修正）            │
 │  + パーサ: 起動メインの「源自身を消費するコスト」は必須            │
 │            (cost_optional=False ←今回修正)                         │
 │  ⇒ 自己レスト型の起動メインは 起動→レスト→再起動不可 で自然に1回   │
 │     （旧 REPEAT_CAP が覆い隠していた無限ループの根因を解消）       │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 7. データの流れ（1手決定のサマリ）

```
 盤面 manager + player
   → decide_guarded（終了保証チェック）
   → decide（候補生成・枝刈り・方式分岐）
   → _scored_search/_pimc_scored（α-β+ビーム / K世界平均）
   → _search 再帰（深さ・ビーム・αβカット）→ 葉/打ち切り
   → evaluate（=evaluate_base）→ evaluate_v2(L1)
   → スカラー評価値（R_me−R_opp+Tele）×scale
   → 逆伝播で root 手のスコア確定 → max → TURN_END fold → 採用手
```
