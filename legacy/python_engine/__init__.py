"""Rust 化する前の Python エンジン（2026-09-07 に `opcg_sim/src/` から退避・計画 §16.2-2）。

    core/     gamestate・engine（ターン進行/戦闘/誘発）・effects（resolver/continuous）・
              actions・journal・action_api・invariants・cpu_ai（L1・廃止）・cpu_eval_v2・
              cpu_learned・rs_bridge
    learned/  mcts・adapter・lethal・plan・policy・action・value_net（探索と serve の配線）
    tests/    このエンジンを直に叩くテスト（`legacy` マーカー）とハーネス・移行期の照合オラクル

**残したもの**（`opcg_sim/` 側）: `src/models`（型）・`src/effects`（パーサ）・`src/utils`・
`src/core/sandbox.py`（自由配置）・`tools`・`learned`（ネット定義と符号化仕様）・`loop`・`api`。

**依存の向き**: `legacy/` → `opcg_sim/`（型・パーサ・符号化仕様）は張ってよい。逆は 0。
符号化のうち**エンジン実測**を要る 3 か所（登場時スキャン・リーサル距離・条件列）は
[`install_hooks`] が `opcg_sim.learned.hooks` へ差し込む（opcg_sim 側は既定 None＝
その列を 0 にするだけで、legacy を import はしない）。
"""




def install_hooks() -> None:
    """符号化がエンジン実測を要る 3 か所を差す（`opcg_sim.learned.hooks`・冪等）。

    `encoder.encode` の登場時スキャン v7 とリーサル距離 v10、`n_rel_feat` の条件列。
    Python エンジンで符号化を再現する計器（旧ラインの実験 CLI）はこれを呼んでから使う。
    """
    from opcg_sim.learned import hooks
    from legacy.python_engine.core.cpu_ai import onplay_option_scan
    from legacy.python_engine.core.effects.resolver import EffectResolver
    from legacy.python_engine.learned.lethal import lethal_scan, MAX_TURNS
    hooks.install(onplay_scan=onplay_option_scan,
                  lethal_scan=(lethal_scan, MAX_TURNS),
                  resolver_factory=EffectResolver)
