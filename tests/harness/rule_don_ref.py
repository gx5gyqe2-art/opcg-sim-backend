"""**RD-speed の比べる相手**: 速くする前（`7bce9082`・H-4 採用時）の `rule_don` の解き方の**原文そのまま**。

`test_rd_speed.py` が今の `crossing_bridge` の解き方と 1 バイトずつ比べる（同じ問題・同じ切替で `repr` が同じ）。
ここは直さない——解き方の値を意図して変えたときは、この写しも同じ変更で更新する（さもなくばテストが落ちる）。
関数の外の名前（`rules_steps`・`walk_crossing`・`model_horizon`・`nu_meas_of`・切替の値…）は呼ぶたびに
`crossing_bridge` から写す（`_sync`）＝切替を変えたテストでも同じ値を読む。覚え書きはこのモジュール専用。
"""
import math  # noqa: F401

import crossing_bridge as CB

_OWN = {"rule_guard_plan_ex", "_rule_guard_plan_ex", "rules_sched", "rule_don_solve", "_rule_don_solve",
        "_RULE_EX_CACHE", "_RULE_EX_SETS", "_RULE_EX_MEMO", "_RULE_EX_CTX", "_RULE_DON_CACHE", "_EX_USED",
        "_ModelBudget", "_sync", "clear", "solve", "CB", "math"}
_RULE_EX_CACHE = {}
_RULE_EX_SETS = {}
_RULE_EX_MEMO = {}
_RULE_EX_CTX = {}
_RULE_DON_CACHE = {}
_EX_USED = {"memo": None, "limit": None}


class _ModelBudget(Exception):
    pass


def _sync():
    g = globals()
    for k, v in vars(CB).items():
        if k not in _OWN and not k.startswith("__"):
            g[k] = v


def clear():
    for d in (_RULE_EX_CACHE, _RULE_EX_SETS, _RULE_EX_MEMO, _RULE_EX_CTX, _RULE_DON_CACHE):
        d.clear()


def solve(*a, **k):
    """旧の `rule_don_solve`（切替の値は今の `crossing_bridge` のもの）。"""
    _sync()
    return rule_don_solve(*a, **k)


def rule_guard_plan_ex(cards, don, xs_first, xs_later, blk_margins, life, turns=None, life_types=(),
                       prices=None, later_seq=None, rest_blk=(), arrive_blk=(), draw_types=()):
    """**H-4e／H-4f**: 守る側の最善の守り（ブロッカーの割り当て・カウンター・**取られたライフの札を同じターンから使う**）。

    守備ターンの中は**攻撃が 1 本ずつ宣言される**（攻め手が順番を選ぶ・`declare_attack`）。守る側は宣言ごとに
    受ける／横取りする（アクティブなブロッカー）／カウンターを切る（手札＋それまでに入ったライフの札・イベントは
    使い残しのドンで払う）を選ぶ。受けたらライフが 1 枚手札に入る（`life_types` の分布・確率の残りは使えない札）。
    ライフ 0 で受けたら倒れる。守る側の目的は (防いだ本数の期待値 最大, 切る札 最小, 生き延びるターン 最大,
    損害を遅らせる)、攻め手は宣言の順番で同じ値を最小にする（ゼロ和）。

    **H-4f**: 攻撃の並びは**ターンごと**（`xs_first`＝今のターン・`later_seq[i]`＝i+2 ターン目・尽きたら最後を繰り返す。
    渡さなければ `xs_later` を毎ターン）。**レスト中のブロッカー**（`rest_blk`）は守る側の次のターンに戻り、
    **手札から出るブロッカー**（`arrive_blk`）も次のターンから居る（どちらも耐久の体の項に入る＝歩きの的と同じ）。
    **守る側も毎ターン 1 枚引く**（`draw_types`＝そのデッキの構成から `(カウンター値, 払うドン, 確率)`・攻め手が
    引く札を数えるのと対称・完全情報）。

    返すのは `{"cut", "stopped", "alive", "prevented", "harms", "theta", "nu_all"}`（期待値）。`harms[j]`＝**j 段目に積む
    損害の期待値**（`F` と同じ値段: 命中 `λ − h·μ`・切らせた札 `μ`・倒したブロッカー `ν`・**とどめの段は耐久の残り全部**
    ＝`(λ − (λ − h·μ)) × 最初のライフ ＋ ν × 残っている全てのブロッカー`・E2）。
    `theta`＝**この守りで耐久が数える量**＝`λ × ライフ ＋ μ × 切る札の期待値 ＋ ν × 全てのブロッカー`。
    倒れた枝では `Σ harms = theta` がちょうど成り立つ（とどめの段が残りを埋める）。"""
    pr = prices or {}
    lam = float(pr.get("lam", LAM)); lam_net = float(pr.get("lam_net", cut_take_price()))
    mu = float(pr.get("mu", cut_card_price())); olp = float(pr.get("olp", 5000.0)); mlp = float(pr.get("mlp", 5000.0))
    if later_seq is None:
        later_seq = (tuple(xs_later or ()),)
    seq = tuple(tuple(sorted(float(x) for x in s)) for s in later_seq) or ((),)
    key = (tuple(sorted((float(c), float(d)) for c, d in cards or ())), float(don),
           tuple(sorted(float(x) for x in xs_first or ())), seq,
           tuple(sorted(float(m) for m in blk_margins or ())), int(max(0, round(float(life)))),
           None if turns is None else int(turns), tuple(life_types or ()),
           round(lam, 9), round(lam_net, 9), round(mu, 9), round(olp, 3), round(mlp, 3),
           tuple(sorted(float(m) for m in rest_blk or ())), tuple(sorted(float(m) for m in arrive_blk or ())),
           tuple(draw_types or ()))
    budgeted = _EX_USED["memo"] is not None
    if not budgeted and key in _RULE_EX_CACHE:
        return _RULE_EX_CACHE[key]
    out = _rule_guard_plan_ex(cards, don, xs_first, seq, blk_margins, life, turns, life_types,
                              lam, lam_net, mu, olp, mlp, rest_blk, arrive_blk, draw_types)
    if budgeted:
        return out                         # 予算つきの試行では結果を共有の覚え書きに入れない（問題だけの関数に保つ）
    if len(_RULE_EX_CACHE) > 200000:
        _RULE_EX_CACHE.clear()
    _RULE_EX_CACHE[key] = out
    return out


def _rule_guard_plan_ex(cards, don, xs_first, seq, blk_margins, life, turns, life_types,
                        lam, lam_net, mu, olp, mlp, rest_blk=(), arrive_blk=(), draw_types=()):
    types = tuple((float(c), float(d), float(p)) for c, d, p in (life_types or ()) if float(p) > 0.0)
    p_none = max(0.0, 1.0 - sum(p for _c, _d, p in types))
    dtypes = tuple((float(c), float(d), float(p)) for c, d, p in (draw_types or ()) if float(p) > 0.0)
    pd_none = max(0.0, 1.0 - sum(p for _c, _d, p in dtypes))
    # 手札の状態は**札の種類（カウンター値, 払うドン）ごとの枚数**——同じ種類の札は入れ替えても同じ（厳密）。
    # 読めた手札の札・取られたライフから入った札・引いた札は同じ種類なら区別しない。
    kinds = sorted({(float(c), float(d)) for c, d in cards or () if float(c) > 0.0}
                   | {(c, d) for c, d, _p in types} | {(c, d) for c, d, _p in dtypes}, reverse=True)
    kind_ix = {kd: i for i, kd in enumerate(kinds)}
    NK = len(kinds)
    cnt0 = [0] * NK
    for c, d in cards or ():
        if float(c) > 0.0:
            cnt0[kind_ix[(float(c), float(d))]] += 1
    cnt0 = tuple(cnt0)
    type_ix = tuple(kind_ix[(c, d)] for c, d, _p in types)
    dtype_ix = tuple(kind_ix[(c, d)] for c, d, _p in dtypes)
    L0 = int(max(0, round(float(life))))
    cap = (sum(cnt0) + 2 * L0 + 2) if turns is None else int(max(0, turns))
    blk0 = tuple(sorted((float(m) for m in blk_margins or ()), reverse=True))
    rest0 = tuple(sorted((float(m) for m in rest_blk or ()), reverse=True))
    arr0 = tuple(sorted((float(m) for m in arrive_blk or ()), reverse=True))
    nu_of = {}

    def nu(m):
        if m not in nu_of:
            nu_of[m] = float(nu_meas_of(float(m) + olp, mlp))
        return nu_of[m]
    nu_all = sum(nu(m) for m in blk0 + rest0 + arr0)
    zero = {"cut": 0.0, "stopped": 0.0, "alive": 0.0, "prevented": 0.0, "harms": (), "theta": lam * L0 + nu_all,
            "nu_all": nu_all}
    if cap <= 0:
        return zero
    hits_f = tuple(sorted(float(x) for x in xs_first or () if float(x) >= -PWR_EPS))
    seq = tuple(tuple(x for x in s if x >= -PWR_EPS) for s in seq)
    last_hit = max([i for i, s in enumerate(seq) if s] + [-1])      # これより先の段に攻撃は無い（尽きたら最後を繰り返す）
    repeat_hits = bool(seq[-1])

    def hits_at(t):
        return hits_f if t == 0 else seq[min(t - 1, len(seq) - 1)]

    def any_hits_from(t):
        return repeat_hits or (t - 1) <= last_hit

    don = round(float(don), 9)
    NT = len(types)
    kinds_t = tuple(kinds)
    # 計画の列挙（`rule_don_solve`）は同じ守る側に対して何百回も呼ぶ——結果は下の文脈と状態だけで決まるので、
    # 呼び出しをまたいで覚える（値は 1 つも変わらない・速さのためだけ）。
    ctx_t = (kinds_t, types, dtypes, seq, don, L0, cap, lam, lam_net, mu, olp, mlp)
    ctx = _RULE_EX_CTX.setdefault(ctx_t, len(_RULE_EX_CTX))
    kid = _RULE_EX_CTX.setdefault(("kinds", kinds_t), len(_RULE_EX_CTX))
    set_cache = _RULE_EX_SETS
    if len(set_cache) > 400000:
        set_cache.clear()

    def counter_sets(x, hand, dl):
        """超過 `x` を止める**過不足の無い**札の組（種類ごとの枚数）→ `(新しい手札, 払ったドンの残り, 枚数)` の列。
        過不足が無い＝どの 1 枚を外しても足りない（切る枚数の最小を探す守る側は、余る組を選ぶ理由が無い）。"""
        ck = (kid, x, hand, dl)
        if ck in set_cache:
            return set_cache[ck]
        need = float(x) + 1000.0 - PWR_EPS
        res = {}
        use = [0] * NK

        def rec(i, s, dc):
            if dc > dl + 1e-9:
                return
            if i == NK:
                if s < need:
                    return
                if any(use[q] > 0 and s - kinds[q][0] >= need for q in range(NK)):
                    return
                nh = tuple(hand[q] - use[q] for q in range(NK))
                kk = (nh, round(dl - dc, 9))
                if kk not in res:
                    res[kk] = sum(use)
                return
            for n_ in range(hand[i] + 1):
                use[i] = n_
                rec(i + 1, s + n_ * kinds[i][0], dc + n_ * kinds[i][1])
            use[i] = 0
        rec(0, 0.0, 0.0)
        out = [(nh, dl2, nc) for (nh, dl2), nc in res.items()]
        set_cache[ck] = out
        return out

    memo = _EX_USED["memo"]
    lim = _EX_USED["limit"]
    if memo is None:                       # 窓・テスト・感度の測定: 呼び出しをまたいで覚える（予算は無い）
        memo = _RULE_EX_MEMO
        lim = None
        if len(memo) > 600000:
            memo.clear()
            set_cache.clear()
            _RULE_EX_CTX.clear()      # 文脈の番号も振り直す（覚えた値と一緒に捨てる）
            ctx = _RULE_EX_CTX.setdefault(ctx_t, 0)
            kid = _RULE_EX_CTX.setdefault(("kinds", kinds_t), 1)

    def better(a, b):
        """守る側の比較（期待値・同点は誤差で）。"""
        if b is None:
            return True
        if abs(a[0] - b[0]) > _FEQ:
            return a[0] > b[0]
        if abs(a[1] - b[1]) > _FEQ:
            return a[1] < b[1]
        if abs(a[2] - b[2]) > _FEQ:
            return a[2] > b[2]
        # 同じなら**損害を遅らせる方**（早い段の損害が小さい方・段の順に比べる）——守る側は歩きが耐久に届く時刻を
        # 遅らせたい（攻め手の目的 E4 の裏返し）。切る札の総数は同じなので耐久は変わらず、時刻だけが変わる。
        ha, hb = a[4], b[4]
        for q in range(max(len(ha), len(hb))):
            u = ha[q] if q < len(ha) else 0.0
            v = hb[q] if q < len(hb) else 0.0
            if abs(u - v) > _FEQ:
                return u < v
        return False

    def add_h(h, j, v):
        h = list(h) + [0.0] * max(0, j + 1 - len(h))
        h[j] += v
        return tuple(h)

    def comb(parts):
        """確率つきの結果の和。"""
        prev = cut = alive = st = 0.0
        harms = ()
        for p, r in parts:
            prev += p * r[0]; cut += p * r[1]; alive += p * r[2]; st += p * r[3]
            hh = list(harms) + [0.0] * max(0, len(r[4]) - len(harms))
            for j, v in enumerate(r[4]):
                hh[j] += p * v
            harms = tuple(hh)
        return (prev, cut, alive, st, harms)

    def turn(t, hand, lf, blk):
        """t 段目の始まり（ブロッカーは全部戻っている・ドンは満タン）。"""
        if t >= cap:
            return (0.0, 0.0, 0.0, 0.0, ())
        if t >= 1 and not any_hits_from(t):
            return (0.0, 0.0, float(cap - t), 0.0, ())       # 以後ずっと命中が無い
        return within(t, hits_at(t), hand, blk, (), (), don, lf)

    def next_turn(t, hand, lf, blk):
        """守る側のターン（**1 枚引く**・H-4f）を挟んで t+1 段目へ。"""
        if t + 1 >= cap or not dtypes:
            return turn(t + 1, hand, lf, blk)
        parts = []
        for di in range(len(dtypes)):
            nh = list(hand); nh[dtype_ix[di]] += 1
            parts.append((dtypes[di][2], turn(t + 1, tuple(nh), lf, blk)))
        if pd_none > 0.0:
            parts.append((pd_none, turn(t + 1, hand, lf, blk)))
        return comb(parts)

    def within(t, rem, hand, ready, rested, pend, dl, lf):
        key = (ctx, t, rem, hand, ready, rested, pend, dl, lf)    # `dl` は丸め済み
        if key in memo:
            return memo[key]
        if not rem:
            r = next_turn(t, hand, lf, tuple(sorted(ready + rested + pend, reverse=True)))
            out = (r[0], r[1], r[2] + 1.0, r[3], (0.0,) + tuple(r[4]))
            memo[key] = out
            if lim is not None and len(memo) > lim:
                raise _ModelBudget()
            return out
        best_att = None
        for x in sorted(set(rem)):                            # 攻め手が次に宣言する攻撃を選ぶ
            i = rem.index(x)
            rest_rem = rem[:i] + rem[i + 1:]
            best_def = None
            # 受ける
            if lf <= 0:
                kill = (lam - lam_net) * float(L0) + sum(nu(m) for m in ready + rested + pend)
                cand = (0.0, 0.0, 0.0, 0.0, (kill,))
            else:
                parts = []
                for ti in range(NT):
                    nh = list(hand); nh[type_ix[ti]] += 1
                    parts.append((types[ti][2], within(t, rest_rem, tuple(nh), ready, rested, pend, dl, lf - 1)))
                if p_none > 0.0:
                    parts.append((p_none, within(t, rest_rem, hand, ready, rested, pend, dl, lf - 1)))
                r = comb(parts)
                cand = (r[0], r[1], r[2], r[3], add_h(r[4], 0, lam_net))
            best_def = cand
            # 横取りする
            for bi, m in enumerate(ready):
                if bi > 0 and ready[bi - 1] == m:
                    continue
                nr = ready[:bi] + ready[bi + 1:]
                if x >= m - PWR_EPS:
                    r = within(t, rest_rem, hand, nr, rested, pend, dl, lf)
                    cand = (r[0] + 1.0, r[1], r[2], r[3], add_h(r[4], 0, nu(m)))
                else:
                    r = within(t, rest_rem, hand, nr, tuple(sorted(rested + (m,), reverse=True)), pend, dl, lf)
                    cand = (r[0] + 1.0, r[1], r[2], r[3], r[4])
                if better(cand, best_def):
                    best_def = cand
            # カウンターを切る
            for nh, dl2, nc in counter_sets(x, hand, dl):
                r = within(t, rest_rem, nh, ready, rested, pend, dl2, lf)
                cand = (r[0] + 1.0, r[1] + nc, r[2], r[3] + 1.0, add_h(r[4], 0, mu * nc))
                if better(cand, best_def):
                    best_def = cand
            # 攻め手は守る側の値を最小にする順番
            if best_att is None or better(best_att, best_def):
                best_att = best_def
        memo[key] = best_att
        if lim is not None and len(memo) > lim:
            raise _ModelBudget()
        return best_att

    # 今のターン: アクティブなブロッカーだけが横取りでき、レスト中のものと手札から出るものは次のターンから
    r = within(0, hits_f, cnt0, blk0, rest0, arr0, don, L0)
    prev, cut, alive, st, harms = r
    harms = tuple(harms)                                     # 道筋が覆う段の数だけ（倒れた段・地平まで）
    return {"cut": cut, "stopped": st, "alive": alive, "prevented": prev, "harms": harms,
            "theta": lam * L0 + mu * cut + nu_all, "nu_all": nu_all}


def rules_sched(harms, steps, actx, paid1):
    """**H-4f**: 歩きの段ごとの速さ（`seat_slope_sched` の代わり・`rule_don` 系）。

    段 `j` ＝ (守る側の計算が覆う段なら) その段の損害の期待値 `harms[j]`／(先は) その段の財布の `fb`
    ＋ **計算の外**: 引いた 1 枚（段 `i` に引いた札は速攻なら `i` から・素の体は `i+1` から・その段の残ったドン
    `d_i − paid_i` で払える札だけ・素殴り＝E6）＋ 引いた札の効果 ＋ その段に出した札の効果。
    **速攻は 1 回だけ**（出した速攻の体は `hits`／`fb` の中・F3）・`RATE_RUSH_MODE` に従う。局の最初の自席ターンは 0（T103）。"""
    ds = actx["ds"]
    a_tab, ar_tab, e_tab = actx.get("a_tab") or [0.0], actx.get("ar_tab") or [0.0], actx.get("e_tab") or [0.0]
    n = len(steps)

    def left(i):
        d = float(ds[min(i, len(ds)) - 1])
        paid = paid1 if i == 1 else steps[i - 1]["paid"]
        l_ = int(round(max(0.0, d - paid) if RATE_DON_PAY else d))
        return max(0, min(l_, len(a_tab) - 1))
    lefts = [left(i) for i in range(1, n + 1)]
    out = []
    for j in range(1, n + 1):
        if actx.get("no_attack_now") and j == 1 and RATE_T1_MODE == "on":
            out.append(0.0)
            continue
        v = float(harms[j - 1]) if j <= len(harms) else float(steps[j - 1]["fb"])
        for i in range(1, j + 1):
            li = lefts[i - 1]
            v += _tab(ar_tab, li)
            if i <= j - 1:
                v += max(0.0, _tab(a_tab, li) - _tab(ar_tab, li))
        v += _tab(e_tab, lefts[j - 1]) + float(steps[j - 1]["eff"])
        out.append(float(v))
    return out


def rule_don_solve(cards_d, don_d, blk, life, actx, turns=None, life_types=(), draw_types=(), arrive=()):
    """`_rule_don_solve` を地平 `⌈τ0⌉`（手札抜きの地平）から始め、守る側の計算の状態数が予算 `EX_STATE_BUDGET` を超えたら
    地平を 1 ターンずつ縮めてやり直す（縮めた回数を数える・計画の `horizon` に使った地平を残す）。`turns` を渡されたら
    その地平のまま（窓・テスト）。"""
    if tuple(actx.get("cp", (None, None))) != cut_context_key():
        raise RuntimeError("攻め手の財布（attacker_ctx）と守る側の計算が別の値段の文脈で作られている（同じ `CP.defending` の中で呼ぶ）")
    key = (tuple(sorted((float(c), float(d)) for c, d in cards_d or ())), float(don_d),
           tuple(sorted(float(m) for m in blk or ())), int(max(0, round(float(life)))),
           None if turns is None else int(turns), tuple(life_types or ()), tuple(draw_types or ()),
           tuple(sorted(float(m) for m in arrive or ())), actx["key"], EX_STATE_BUDGET, "w")
    if key in _RULE_DON_CACHE:
        return _RULE_DON_CACHE[key]
    if turns is not None:
        out = _rule_don_solve(cards_d, don_d, blk, life, actx, turns, life_types, draw_types, arrive)
    else:
        L0 = int(max(0, round(float(life))))
        h0 = model_horizon(actx, blk, L0, arrive)
        out = None
        for h in range(h0, 0, -1):
            if EX_STATE_BUDGET is None:
                _EX_USED["memo"], _EX_USED["limit"] = None, None
            else:
                _EX_USED["memo"], _EX_USED["limit"] = {}, (EX_STATE_BUDGET if h > 1 else None)
            try:
                out = _rule_don_solve(cards_d, don_d, blk, life, actx, h, life_types, draw_types, arrive)
            except _ModelBudget:
                continue
            finally:
                _EX_USED["memo"], _EX_USED["limit"] = None, None
            break
        out[2]["horizon"] = h
        out[2]["horizon0"] = h0
    if len(_RULE_DON_CACHE) > 100000:
        _RULE_DON_CACHE.clear()
    _RULE_DON_CACHE[key] = out
    return out


def _rule_don_solve(cards_d, don_d, blk, life, actx, turns, life_types=(), draw_types=(), arrive=()):
    """**H-4b／H-4e／H-4f**: 攻め手の最善の計画（今のターンに出す札の組 × 付与）に対する守る側の最善の守り。

    **2 ターン目からは攻め手もその段のドンで財布を解き直す**（`rules_steps`・F2）——今のターンの計画が変えるのは
    今のターンの攻撃と、残る手札（＝先の段で出せる札）だけ。**付けたドンはそのターンで戻る**。
    守る側は**レスト中のブロッカーが次のターンに戻り**（`actx["rest_blk"]`）、**手札のブロッカー**（`arrive`）も
    次のターンから居り、**毎ターン 1 枚引く**（`draw_types`・F4）＝歩きの的と守る側の計算の耐久が同じもの（F1）。

    攻め手の目的は (**歩きが耐久に届くターン**〔整数・端数は切り上げ〕 最小, 守る側が生き延びるターン数 最小,
    速さの値打ち 最大, 使うドン 最小)（E4）。

    返すのは `(切る枚数, 止める本数, 計画)`。計画は `play`・`k`・`paid`・`xs_first`・`later_seq`・`harm_steps`・
    `sched`（歩きの段ごとの速さ）・`theta`（守る側の計算の耐久＝歩きの的）・`theta_parts`＝(ライフ, 手札, 体)・
    `tau`（歩きが届く時刻）・**`a_time`＝耐久 ÷ 届く時刻**（時刻で読む器〔線形の橋・帳簿〕が受け取る速さ）・
    **`a_turn`＝今のターンの損害**（1 ターンで読む器〔速さの検算〕が受け取る速さ）・`alive`・`value`。"""
    key = (tuple(sorted((float(c), float(d)) for c, d in cards_d or ())), float(don_d),
           tuple(sorted(float(m) for m in blk or ())), int(max(0, round(float(life)))),
           None if turns is None else int(turns), tuple(life_types or ()), tuple(draw_types or ()),
           tuple(sorted(float(m) for m in arrive or ())), actx["key"])
    if key in _RULE_DON_CACHE:
        return _RULE_DON_CACHE[key]
    prices = _prices_of(actx)
    L0 = int(max(0, round(float(life))))
    rest = tuple(actx.get("rest_blk") or ())
    att1, cand = actx["att1"], actx["cand"]
    budget, kmax = actx["budget"], actx["kmax"]
    flow = actx.get("flow") or [0.0] * (budget + 1)
    nsteps = int(actx.get("jmax") or RACE_CAP)
    best = None
    n_c = len(cand)
    fixed = actx.get("fixed")
    no_now = bool(actx.get("no_attack_now"))
    max_t = max([0.0] + [float(c) for c, _d, _p in (life_types or ())])
    s_cnt = sum(float(c) for c, _d in cards_d or ())

    for mask in range(1 << n_c):
        play = [i for i in range(n_c) if mask >> i & 1]
        cost = sum(cand[i][0] for i in play)
        if cost > budget:
            continue
        if fixed is not None and tuple(play) != tuple(fixed[0]):
            continue                                        # `rule_don_purse`: 出す札は速さの側の計画そのまま
        b = budget - cost
        steps = rules_steps(actx, play, nsteps)
        later_seq = tuple(st["hits"] for st in steps[1:]) or ((),)
        hits1 = () if no_now else steps[0]["hits"]
        p_atk = sum(float(cand[i][1].get("atk", 0.0)) for i in play)
        p_eff = sum(float(cand[i][1].get("eff", 0.0)) for i in play)
        p_rush = sum(float(cand[i][1].get("rush", 0.0)) for i in play)

        def solve(ks):
            xf = () if no_now else (tuple(float(att1[q][1]) + 1000.0 * ks[q] for q in range(len(att1)))
                                    + tuple(hits1[len(att1):]))
            return xf, rule_guard_plan_ex(cards_d, don_d, xf, None, blk, life, turns, life_types, prices,
                                          later_seq=later_seq, rest_blk=rest, arrive_blk=arrive,
                                          draw_types=draw_types)
        r0 = solve([0] * len(att1))[1]
        h1_bare = r0["harms"][0] if r0["harms"] else 0.0
        # **厳密な縮約**: 付与は今のターンの攻撃だけを動かす。今のターンに守る側が使えるカウンターは手札 ＋ このターンに
        # 取るライフの札（攻撃の本数 − 1 枚まで）だけで、アクティブなブロッカーもこのターンの分だけ——それ以上のパワーは同じ。
        n_h1 = len(hits1)
        cap_x = max([s_cnt + min(L0, max(0, n_h1 - 1)) * max_t] + [float(m) for m in blk or ()])
        caps = [min(kmax, b, max(0, int(math.ceil((cap_x - float(x)) / 1000.0 - 1e-9)))) for _s, x in att1]
        if no_now:
            caps = [0] * len(att1)                          # 局の最初の自席ターンは攻撃できない＝付けても効かない
        if fixed is not None:
            caps = [min(kmax, fixed[1]) for _s, _x in att1]
        ks = [0] * len(att1)

        def visit(i, left):
            nonlocal best
            if i == len(att1):
                if fixed is not None and sum(ks) != fixed[1]:
                    return
                xf, res = solve(ks)
                paid = cost + sum(ks)
                h = res["harms"]
                incr = (h[0] if h else 0.0) - h1_bare
                val = p_atk + p_eff + incr + float(flow[max(0, budget - paid)])
                sched = rules_sched(h, steps, actx, float(paid))
                tau = walk_crossing(sched, res["theta"], actx)
                score = (int(math.ceil(round(tau, 9) - 1e-9)), round(float(res["alive"]), 9), -round(val, 12), paid)
                if best is None or score < best[0]:
                    best = (score, res, {"atk": p_atk, "rush": p_rush, "eff": p_eff, "incr": incr,
                                         "paid": float(paid), "play": tuple(play), "k": tuple(ks),
                                         "xs_first": tuple(xf), "later_seq": later_seq,
                                         "alive": res["alive"], "value": val, "tau": tau,
                                         "harm_steps": tuple(h), "sched": tuple(sched), "theta": res["theta"],
                                         "cut": res["cut"], "stopped": res["stopped"]})
                return
            for k in range(0, min(caps[i], left) + 1):
                ks[i] = k
                visit(i + 1, left - k)
            ks[i] = 0

        visit(0, b)
    res, plan = best[1], dict(best[2])
    lam = float(prices["lam"]); mu = float(prices["mu"])
    plan["theta_parts"] = (lam * L0, mu * float(res["cut"]), float(res["nu_all"]))
    tau = float(plan["tau"]); th = float(plan["theta"])
    sched = plan["sched"]
    # **Q1**: 時刻で読む器は「耐久 ÷ 届く時刻」（Θ/A がちょうど歩きの τ）、1 ターンで読む器は今のターンの損害
    plan["a_time"] = (th / tau) if tau > 1e-12 else max(float(sched[0]) if sched else 0.0, SLOPE_FLOOR)
    plan["a_turn"] = float(sched[0]) if sched else 0.0
    plan["attach_lead"] = 0.0
    plan["attach"] = 0.0
    plan["rest"] = rest
    plan["arrive"] = tuple(arrive or ())
    plan["draw_types"] = tuple(draw_types or ())
    out = (res["cut"], res["stopped"], plan)
    if len(_RULE_DON_CACHE) > 100000:
        _RULE_DON_CACHE.clear()
    _RULE_DON_CACHE[key] = out
    return out
