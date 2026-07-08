# 司令塔 引き継ぎ書 — 学習CPU 分散実験（2026-07-08）

新しいセッション（別コンテナ）が**司令塔（監視・測定・グラフ・判断）**を引き継ぐための文書。
「このリポジトリの `claude/opcg-cluster-learning` を checkout し、`docs/orchestrator_handoff.md` を読んで司令塔を引き継いで」で渡せる。

> **あなた（新司令塔）の役割**: 自分では重い自己対戦を走らせず、**別セッションで走っている各runを git 経由で監視**し、
> 節目で**強度測定**を行い、**リーダー数スケーリングのグラフ**を作り、**続行/打ち切りを判断**する。
> 測定は自分のコンテナのCPUで走るが、各訓練runは別コンテナなので競合しない。

---

## 1. 目的とこれまでの結論（背景）

**目的**: 出荷CPU（学習ネット v1・`opcg_sim/data/learned/gen2_*.npz`）を超える学習CPUを作る。
出荷v1の実力＝**対製品L1（α-β+PIMC4）で多様デッキ 0.833**（紫デッキ単体では 0.750）。

**確定している結論**（すべて実測）:
- **v1本走（成功・過去）**: 弱いSLネット（対L1 0.45）から**単一デッキ**で自己対戦 2世代(2万局)→ 出荷Gen2（対L1 0.925）。
- **v2本走（失敗）**: 出荷強ネットを温スタート＋137リーダーrotate で再訓練 → **劣化**（出荷v1に負け）。
  主因＝**出発点が既に天井**（強いネットを弱いsims40自己対戦で訓練すると引き戻される）＋**137希釈**。
- **案A（弱出発・97リーダー・進行中）**: v1と同じ**弱い出発点**に戻し、97リーダーで連続学習。Gen1(1万局)=対L1多様 **0.433**。
- **案B（色クラスタ・進行中）**: 6色に分けて**狭い分布**で並列学習（v1的な速い climb を狙う）。
- **リーダー数スケーリング（pool6/12/24/48・未起動）**: 色を交ぜてリーダー数を変え、「1リーダーあたり局数→強度」の曲線を描く実験。

**現アーキは「達成可能上限に近い」が繰り返し実証されている**（`docs/reports/cpu_strength_plan_20260628.md`）。
RL・sims増・eval微修正はいずれも純増を出せなかった。

---

## 2. 最新の重要結果（2026-07-08）

**紫クラスタを紫デッキで対L1測定**（2点目を 2026-07-08 に司令塔が追加）:
| ネット | 局/リーダー | 対L1（紫デッキ・sims160・pimc4・N=24） |
|---|---|---|
| 出荷v1（バー） | — | **0.750** [.551,.880] |
| 紫クラスタnet（cum 15,420） | 640 | **0.479** [.296,.668] |
| 紫クラスタnet（cum 19,620） | **813** | **0.542** [.351,.721]（13-11/24・所要72分） |

→ **狭いクラスタで濃く訓練しても L1 互角止まり・出荷v1に大きく届かず**。97汎化Gen1(0.433)から6倍濃くして+0.05のみ
＝**濃縮の効きが弱い＝希釈だけが原因ではなく、アーキ天井が本質**の可能性が濃厚。この方向は**打ち切り寄り**の暫定判断。
- **2点目（813局/リーダー）の追加所見**: 640→813（+27%濃縮）で 0.479→0.542。**+0.063 だが両点のCIは大きく重複＝傾きはほぼフラット**。
  出荷v1（0.750）には依然遠く、打ち切り寄り判断を補強。ただし測定1本=約72分と重く、N=24でCIが広い点は割り引く。
ただし各runはまだ climb 途中なので、**リーダー数スケーリングのグラフ（pool版）で傾きを確認してから最終判断**するのが筋。

**6色一括 対L1測定**（2026-07-08 司令塔が全色を同一条件で測定・各色は自分の色デッキ・sims160/pimc4・N=24）:
| 色 | N | 局/リーダー | 勝率 | 95%CI | 対L1 |
|---|---|---|---|---|---|
| 赤 | 31 | 664 | **0.542** | [.351,.721] | 互角圏 |
| 紫 | 24 | 813 | **0.542** | [.351,.721] | 互角圏 |
| 黒 | 22 | 835 | **0.542** | [.351,.721] | 互角圏 |
| 黄 | 24 | 640 | 0.375 | [.212,.573] | 及ばず |
| 緑 | 23 | 634 | 0.312 | [.164,.512] | 及ばず |
| 青 | 27 | 538 | **0.208** | [.092,.405] | 及ばず |

→ 訓練後は 赤/紫/黒＝0.542（L1互角）／黄/緑/青＝0.21〜0.38（L1劣後）。青が最弱。**ただし下のgen0切り分けで「色相性」説は棄却**。
- **総括**: 最良の赤/紫/黒でも 0.542 で **出荷v1(0.750)には全色届かず** → §10「打ち切り寄り」を全色で裏取り。
- **構造チェック（発散なし）**: 全色 value.npz は NaN/Inf=0・ノルムは gen0(L2=38)→80〜90 で約2倍に収束、局あたり重み移動は最も進んだ紫が最小
  ＝「発散」でなく「プラトーへの収束」。

**gen0切り分け＝出発点(共通弱Gen0)を各色デッキで対L1測定**（2026-07-08 司令塔・同一条件・N=24）:
| 色 | gen0(出発) | 訓練後 | Δ |
|---|---|---|---|
| 赤 | 0.292（最弱） | 0.542 | **+0.250 ↑** |
| 黄 | 0.542（最強） | 0.375 | −0.167 ↓ |
| 緑 | 0.438 | 0.312 | −0.126 ↓ |
| 青 | 0.417 | 0.208 | −0.209 ↓ |

→ **「弱群は色相性」説は棄却**。青/緑/黄のgen0は0.42〜0.54と健全で、**自己対戦が積極的に引き下げた**（劣化）。赤だけ最弱出発から+0.25で伸びた。
- **核心の像＝sims40自己対戦のアトラクター（平均回帰）**: 出発点で並べると（赤< 青< 緑< 黄）**最下位の赤だけ上昇・他は全部下降**。
  訓練は出発点がどこでも net を共通バンド(~0.4–0.54)へ引き寄せる。**gen0がバンド未満（赤）→上げ、超（黄/緑/青）→下げ**。
- **＝「ネットは自分の教師を超えられない」**: 教師信号＝sims40自己対戦の強さが評価バーL1(α-β+pimc4)と同等以下。汎用化＝多デッキ平均は
  「大半が既に天井以上」を平均するので天井へ引き下げられる。**v2失敗（強出発を弱自己対戦で全面劣化）を色ごとに再現**したもの。
- **レバーの含意（更新）**: 第一のボトルネックは**容量より「訓練時サーチ＝教師の弱さ(sims40)」**。効くのは**訓練simsを上げる/より強い探索・相手で自己対戦**。
  ただし過去「sims増は純増なし」との整合を要確認（訓練時か評価時か・範囲）。この gen0 所見は訓練時sims＝教師強度が本丸だと指す。
- 留保: 各色N=24でΔ単体はノイズ内。だが**4色を貫く「高い出発点ほど下がる」反相関は系統的**で頑健。紫/黒のgen0は未測（赤対照で十分と判断）。

---

## 3. 現在走っている全run（監視対象）※各cumは 2026-07-08 時点

| run | checkpoint枝 | 直近cum | 場所 |
|---|---|---|---|
| **A（97汎化）** | `claude/p3-checkpoints` | gen=1 cum≈15,240（**通算=+10,000**） | 元セッション（別コンテナ） |
| B: red | `claude/p3-cluster-red-checkpoints` | 17,400 | 各別コンテナ |
| B: green | `claude/p3-cluster-green-checkpoints` | 12,480 | |
| B: blue | `claude/p3-cluster-blue-checkpoints` | 12,360 | |
| B: purple | `claude/p3-cluster-purple-checkpoints` | 19,380 | |
| B: black | `claude/p3-cluster-black-checkpoints` | 15,480 | |
| B: yellow | `claude/p3-cluster-yellow-checkpoints` | 14,520（※停止しがち） | |
| pool6/12/24/48 | `claude/p3-pool{6,12,24,48}-checkpoints` | **0（未起動）** | — |

各runは連続モード（`--target/--max-shards` 実質無限）＝世代境界で止まらず、各shardで checkpoint枝へ force-push。

---

## 4. 資産一覧（ブランチ・ネット）

- **コード枝**: `claude/opcg-cluster-learning`（ハーネス改修入り。case A の元コードは `claude/opcg-cpu-decision-analysis-49solv`）。
  - `deckgen.all_leader_ids`: `OPCG_LEADER_COLORS`（色フィルタ）／`OPCG_LEADER_POOL_SIZE`（色バランス入れ子・POOL>COLORS>全97）／block_icon==1除外で97。
  - `p3_run.py`: `OPCG_P3_WT` / `OPCG_P3_BRANCH` で checkpoint worktree/枝を上書き（run隔離）。
- **弱Gen0**（全runの共通出発点）: 各checkpoint枝の `p3ckpt/gen0_value.npz`（SHA1 `92ae0c1f8a4e`）＝v1のSLネットをv2温スタートしたもの。`value.npz`と同一で開始・policyなし(uniform)。
- **アーカイブ**: `claude/p3-checkpoints-v1-archive`（v1本走の最終・出荷Gen2含む）／`claude/p3-checkpoints-v2-failed-archive`（失敗v2）。
- **出荷v1バー**: `opcg_sim/data/learned/gen2_*.npz`（本番同梱・v1エンコード）。
- **共通プロンプト**: `docs/cluster_training_prompt.md`（新ワーカーが空き色を拾って学習開始する用）。

---

## 5. cum_games の正規化（重要な落とし穴）

グラフ・比較では**「弱Gen0からの通算局数」に揃える**こと:
- **色クラスタ・pool版**（manifest `gen=0`＝リセット無し）: cum = 弱Gen0からの通算。**そのまま**。
- **A**（manifest `gen=1`＝Gen1スナップショットで一度 cum=0 にリセット済み）: 表示cum は「Gen1以降」。
  **通算 = 表示cum + 10,000**。
- 「1リーダーあたり局数」= 通算 ÷ リーダー数（A=97, red=31, green=23, blue=27, purple=24, black=22, yellow=24, pool=N）。

---

## 6. 監視・re-arm の手順

各runは別コンテナ。**司令塔は git の最終push時刻で死活を判断**（force-pushは~2.5分ごと）:
```bash
for b in cluster-red cluster-green cluster-blue cluster-purple cluster-black cluster-yellow pool6 pool12 pool24 pool48; do
  git fetch origin claude/p3-$b-checkpoints -q 2>/dev/null
  echo "$b cum=$(git show origin/claude/p3-$b-checkpoints:p3ckpt/manifest.json 2>/dev/null|python3 -c 'import sys,json;print(json.load(sys.stdin)["cum_games"])') last=$(git log -1 --format=%cr origin/claude/p3-$b-checkpoints)"
done
```
- 最終push **>10分**（cum不変）= その色は**停止**。**司令塔は別コンテナのrunを直接再起動できない**ので、
  該当色のセッションに「同じ起動コマンドで再実行」を促す（checkpointから自動継続）。
- **注意**: yellow は停止しがち。各ワーカーが自分で死活監視＋re-armする運用が望ましい（共通プロンプトに追記余地）。

---

## 7. 強度測定の手順（司令塔の主業務）

対象runの現netを凍結して、その分布で対L1測定。バー＝同条件の出荷v1。
```bash
# 例: 紫クラスタnetを取得
git fetch origin claude/p3-cluster-purple-checkpoints -q
git show origin/claude/p3-cluster-purple-checkpoints:p3ckpt/value.npz  > /tmp/x_value.npz
git show origin/claude/p3-cluster-purple-checkpoints:p3ckpt/policy.npz > /tmp/x_policy.npz
# 紫デッキで対L1（バーは出荷v1で同条件）
OPCG_LEADER_COLORS=紫 OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_vs_l1.py \
  --value-path /tmp/x_value.npz --policy-path /tmp/x_policy.npz --label purple-net \
  --rotate-leaders --sims 160 --pimc 4 --pairs 12
```
- pool版は `OPCG_LEADER_POOL_SIZE=<N>`（色指定の代わり）で対象プールに絞る。
- N=24(pairs=12)は速い代わりにCI広い。有望なら pairs 増でCIを締める。
- **軽い代替**: 各runの共通弱Gen0を凍結し、`p3_gate`系で net vs Gen0（sims40）を測ると速い（L1不要・相対climb）。ただし絶対バー比較は上のvs-L1。

---

## 8. リーダー数スケーリングのグラフ（アウトプット）

- **横軸 = 1リーダーあたり通算局数**（= 通算 ÷ N・対数）／**縦軸 = 強度**（共通の色バランス評価で。推奨は
  `OPCG_LEADER_POOL_SIZE=6`（各色1・全プールに内包）で固定した6リーダー評価にして全版を同一土俵で測る）。
- 各版（pool6/12/24/48 と A=97）を同じ軸に載せる。**1本の曲線に重なれば「強度は局/リーダーで決まる（希釈が本質）」／
  バラければリーダー数（容量）が効く**。
- 現状の暗示（紫の1点）: 濃くしても伸びが緩い＝**希釈以外（アーキ天井）が支配的**の可能性。グラフで確証する。

---

## 9. 未起動: pool版4つ の起動（フル4版でユーザ合意済み）

各版=別セッション。`git checkout claude/opcg-cluster-learning` 後、N=6/12/24/48 で:
```bash
OPCG_LEADER_POOL_SIZE=6 OPCG_P3_WT=/tmp/pool6-wt OPCG_P3_BRANCH=claude/p3-pool6-checkpoints \
OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/p3_run.py \
  --enc-version 2 --rotate-leaders --shard-games 60 --sims 40 --workers 4 --target 100000000 --max-shards 100000000
```
（枝はpristine弱Gen0にリセット済み・cum=0。赤汚染は修正済み。）

---

## 10. 判断基準・次の一手

- **成功条件**: いずれかの版が **対L1で出荷v1（多様0.833／色別0.7-0.8）を超える**。
- **打ち切り条件**: リーダー数スケーリング曲線が**寝て**、どの版も出荷v1に届かない見込みが立つ（＝アーキ天井確定）。
- **暫定**: 紫の1点は打ち切り寄り。だが**曲線を数点そろえてから**最終判断するのが誠実（pool版4つ起動→各節目で測定→作図）。
- グラフが「濃くすれば届く」形なら、**per-leader級の極端な濃縮**（≒v1条件）まで詰める価値を再評価。

---

## 11. 教訓・注意

- **長時間runは死活監視必須**（Aは監視の隙に2h停止した）。各runは自分でwatcher＋re-armするのが理想。
- **cum正規化**（§5）を忘れると軸がズレる。
- **測定は凍結スナップ**（value.npzをコピー）で行う＝訓練が進んでも測定は固定netで完結。
- **water-oil注意**: p3_vs_l1 は α-β評価器 vs NN-MCTS で完全フェアではない（参考値）。相対比較・推移で見る。
- マージ/PR はユーザ明示指示があるまで行わない（`CLAUDE.md`）。この実験群は研究用でまだ製品化しない。
