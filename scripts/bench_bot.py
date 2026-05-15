#!/usr/bin/env python3
"""
scripts/bench_bot.py — BeamSearchBot 성능 벤치마크

사용법
------
  python scripts/bench_bot.py                  # 모든 preset 측정
  python scripts/bench_bot.py --config fast_play
  python scripts/bench_bot.py --pieces 20 --games 5

출력 항목
---------
  Config      : preset 이름
  N           : 총 측정 횟수 (game × pieces)
  Mean ms     : search() 평균 소요 시간 (ms)
  Median ms   : search() 중앙값 소요 시간 (ms)
  P95 ms      : 95번째 백분위수
  Max ms      : 최댓값
  nodes/call  : 평균 expanded_nodes
  depth       : 평균 searched_depth
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (스크립트 직접 실행 시)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.bag import SevenBag
from engine.state import GameState

from bot.adapter import BotEnvAdapter
from bot.beam_search_bot import BeamSearchBot
from bot.search_config import SearchConfig


# ── Preset 정의 ────────────────────────────────────────────────────────────────

PRESETS: dict[str, SearchConfig] = {
    "safe_default": SearchConfig.safe_default(),
    "fast_play":    SearchConfig.fast_play(),
    "deeper_play":  SearchConfig.deeper_play(),
    "tiny":         SearchConfig(beam_width=10, depth=2, time_budget_ms=None),
    "no_budget":    SearchConfig(beam_width=60, depth=4, time_budget_ms=None),
}


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def make_game(seed: int, n_pieces: int = 25) -> GameState:
    bag = SevenBag(seed=seed)
    state = GameState()
    for _ in range(n_pieces):
        state.enqueue(bag.pop())
    state.spawn_next()
    return state


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * pct / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ── 단일 config 벤치마크 ───────────────────────────────────────────────────────

def bench_config(
    config_name: str,
    config: SearchConfig,
    n_games: int,
    n_pieces: int,
    verbose: bool = False,
) -> dict:
    bot     = BeamSearchBot(config=config)
    adapter = BotEnvAdapter()

    times_ms:   list[float] = []
    nodes_list: list[int]   = []
    depth_list: list[int]   = []

    for seed in range(n_games):
        env = make_game(seed, n_pieces + 5)

        for _ in range(n_pieces):
            if adapter.is_terminal(env):
                break

            t0     = time.monotonic()
            result = bot.search(env)
            dt_ms  = (time.monotonic() - t0) * 1000

            times_ms.append(dt_ms)
            nodes_list.append(result.expanded_nodes)
            depth_list.append(result.searched_depth)

            if verbose:
                print(
                    f"  [{config_name}] seed={seed} "
                    f"action={result.best_action} "
                    f"depth={result.searched_depth} "
                    f"nodes={result.expanded_nodes} "
                    f"dt={dt_ms:.1f}ms"
                )

            if result.best_action is None:
                break
            env, _ = adapter.simulate_action(env, result.best_action)

    return {
        "n":          len(times_ms),
        "mean_ms":    statistics.mean(times_ms) if times_ms else 0.0,
        "median_ms":  statistics.median(times_ms) if times_ms else 0.0,
        "p95_ms":     percentile(times_ms, 95),
        "max_ms":     max(times_ms) if times_ms else 0.0,
        "nodes":      statistics.mean(nodes_list) if nodes_list else 0.0,
        "depth":      statistics.mean(depth_list) if depth_list else 0.0,
    }


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BeamSearchBot 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="all",
        help=f"측정할 preset 이름, 또는 'all'. 가능한 값: {list(PRESETS)}"
    )
    parser.add_argument("--pieces", type=int, default=10, help="게임당 배치할 피스 수")
    parser.add_argument("--games",  type=int, default=3,  help="반복 게임 수")
    parser.add_argument("--verbose", action="store_true", help="각 배치 상세 출력")
    args = parser.parse_args()

    if args.config == "all":
        configs_to_run = list(PRESETS.items())
    elif args.config in PRESETS:
        configs_to_run = [(args.config, PRESETS[args.config])]
    else:
        print(f"[ERROR] 알 수 없는 config: {args.config!r}")
        print(f"        가능한 값: {list(PRESETS)}")
        sys.exit(1)

    header = (
        f"{'Config':<20} {'N':>5} {'Mean ms':>9} {'Median':>9} "
        f"{'P95 ms':>9} {'Max ms':>9} {'nodes/call':>12} {'depth':>7}"
    )
    print(header)
    print("-" * len(header))

    for name, cfg in configs_to_run:
        stats = bench_config(name, cfg, args.games, args.pieces, args.verbose)
        budget_str = (
            f"(budget={cfg.time_budget_ms}ms)" if cfg.time_budget_ms else "(no budget)"
        )
        label = f"{name}"
        print(
            f"{label:<20} {stats['n']:>5} {stats['mean_ms']:>9.1f} "
            f"{stats['median_ms']:>9.1f} {stats['p95_ms']:>9.1f} "
            f"{stats['max_ms']:>9.1f} {stats['nodes']:>12.0f} "
            f"{stats['depth']:>7.1f}  {budget_str}"
        )


if __name__ == "__main__":
    main()
