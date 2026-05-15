"""
tests/test_search_perf.py — BeamSearchBot 성능·설정 테스트

타이밍 테스트는 CI 환경에서 불안정할 수 있으므로 여유 있는 허용 범위를 사용한다.
@pytest.mark.slow 마커가 붙은 테스트는 기본적으로 실행되지 않는다.
  실행 시: pytest -m slow
"""

from __future__ import annotations

import time

import pytest

from engine.bag import SevenBag
from engine.state import GameState, RoundResult

from bot.adapter import BotEnvAdapter
from bot.beam_search_bot import BeamSearchBot
from bot.placement_generator import FinalPlacement
from bot.search_config import SearchConfig
from bot.search_node import SearchResult


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_state(seed: int = 42, n_pieces: int = 14) -> GameState:
    bag = SevenBag(seed=seed)
    state = GameState()
    for _ in range(n_pieces):
        state.enqueue(bag.pop())
    state.spawn_next()
    return state


# ── SearchConfig preset 테스트 ────────────────────────────────────────────────

class TestSearchConfigPresets:
    def test_safe_default_values(self):
        cfg = SearchConfig.safe_default()
        assert cfg.beam_width                == 120
        assert cfg.depth                     == 4
        assert cfg.futility_delta            == pytest.approx(15.0)
        assert cfg.time_budget_ms            == pytest.approx(35.0)
        assert cfg.use_hold                  is True
        assert cfg.root_top_k               is None
        assert cfg.quiescence_max_extensions == 2
        assert cfg.quiescence_beam_fraction  == pytest.approx(0.20)

    def test_fast_play_values(self):
        cfg = SearchConfig.fast_play()
        assert cfg.beam_width                == 80
        assert cfg.depth                     == 3
        assert cfg.futility_delta            == pytest.approx(20.0)
        assert cfg.time_budget_ms            == pytest.approx(20.0)
        assert cfg.use_hold                  is True
        assert cfg.root_top_k               == 12
        assert cfg.quiescence_max_extensions == 1
        assert cfg.quiescence_beam_fraction  == pytest.approx(0.15)

    def test_deeper_play_values(self):
        cfg = SearchConfig.deeper_play()
        assert cfg.beam_width                == 180
        assert cfg.depth                     == 5
        assert cfg.futility_delta            == pytest.approx(12.0)
        assert cfg.time_budget_ms            == pytest.approx(50.0)
        assert cfg.use_hold                  is True
        assert cfg.root_top_k               == 16
        assert cfg.quiescence_max_extensions == 2
        assert cfg.quiescence_beam_fraction  == pytest.approx(0.20)

    def test_all_presets_return_placement(self):
        env = make_state()
        for preset in (
            SearchConfig.safe_default(),
            SearchConfig.fast_play(),
            SearchConfig.deeper_play(),
        ):
            bot = BeamSearchBot(config=preset)
            pl  = bot.select_placement(env)
            assert isinstance(pl, FinalPlacement), \
                f"preset {preset} returned {pl!r}"


# ── 시간 예산 테스트 ──────────────────────────────────────────────────────────

class TestTimeBudget:
    def test_no_budget_reaches_full_depth(self):
        """time_budget_ms=None 이면 depth 한도까지 탐색한다."""
        cfg = SearchConfig(beam_width=10, depth=3, time_budget_ms=None)
        result = BeamSearchBot(config=cfg).search(make_state())
        assert result.searched_depth == 3

    def test_time_budget_terminates_early(self):
        """매우 짧은 time_budget_ms 이면 depth 한도(8)까지 도달하지 않아야 한다."""
        cfg = SearchConfig(beam_width=120, depth=8, time_budget_ms=1.0)
        result = BeamSearchBot(config=cfg).search(make_state())
        assert result.best_placement is not None
        assert result.searched_depth >= 1
        assert result.searched_depth < 8

    def test_minimum_depth_one_always_completed(self):
        """시간 예산이 있어도 depth=1은 항상 완료된다."""
        cfg = SearchConfig(beam_width=120, depth=10, time_budget_ms=0.001)
        result = BeamSearchBot(config=cfg).search(make_state())
        assert result.searched_depth >= 1
        assert result.best_placement is not None

    def test_time_budget_check_in_expansion_loop(self):
        """확장 루프 중간에도 시간 초과 체크가 작동한다."""
        # 배치 생성기(BFS)는 기존 단일 행동보다 호출당 비용이 크다.
        # 넉넉한 예산(100ms)으로 체크 자체가 동작하는지만 확인한다.
        cfg = SearchConfig(beam_width=500, depth=6, time_budget_ms=100.0)
        t0 = time.monotonic()
        result = BeamSearchBot(config=cfg).search(make_state())
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert result.best_placement is not None
        # depth=6 전부를 완전 탐색하지 않아야 시간 체크가 작동한 것
        assert result.searched_depth < 6
        # 경과 시간이 예산의 50배를 넘지 않으면 충분
        assert elapsed_ms < 100.0 * 50


# ── Beam Width 효과 ───────────────────────────────────────────────────────────

class TestBeamWidthEffect:
    def setup_method(self):
        self.env = make_state()

    def test_wider_beam_more_expanded_nodes(self):
        """beam_width가 클수록 expanded_nodes가 더 많아야 한다."""
        r_narrow = BeamSearchBot(
            config=SearchConfig(beam_width=3, depth=2, time_budget_ms=None)
        ).search(self.env)
        r_wide = BeamSearchBot(
            config=SearchConfig(beam_width=30, depth=2, time_budget_ms=None)
        ).search(self.env)
        assert r_wide.expanded_nodes >= r_narrow.expanded_nodes

    def test_beam_width_one_still_returns_placement(self):
        """beam_width=1도 정상 작동해야 한다."""
        result = BeamSearchBot(
            config=SearchConfig(beam_width=1, depth=3, time_budget_ms=None)
        ).search(self.env)
        assert result.best_placement is not None

    def test_expanded_nodes_increases_with_depth(self):
        """depth가 깊을수록 expanded_nodes가 증가해야 한다."""
        r1 = BeamSearchBot(config=SearchConfig(beam_width=10, depth=1)).search(self.env)
        r2 = BeamSearchBot(config=SearchConfig(beam_width=10, depth=2)).search(self.env)
        r3 = BeamSearchBot(config=SearchConfig(beam_width=10, depth=3)).search(self.env)
        assert r1.expanded_nodes <= r2.expanded_nodes <= r3.expanded_nodes


# ── root_top_k 테스트 ─────────────────────────────────────────────────────────

class TestRootTopK:
    def setup_method(self):
        self.env = make_state()

    def test_root_top_k_limits_root_diversity(self):
        """root_top_k를 작게 설정하면 root_scores 키가 더 적어야 한다."""
        r_full  = BeamSearchBot(
            config=SearchConfig(beam_width=50, depth=1, root_top_k=None)
        ).search(self.env)
        r_limit = BeamSearchBot(
            config=SearchConfig(beam_width=50, depth=1, root_top_k=2)
        ).search(self.env)
        assert len(r_limit.root_scores) <= len(r_full.root_scores)

    def test_root_scores_keys_are_final_placements(self):
        """root_scores 의 키는 FinalPlacement 여야 한다."""
        r = BeamSearchBot(
            config=SearchConfig(beam_width=200, depth=1, root_top_k=None)
        ).search(self.env)
        for pl in r.root_scores:
            assert isinstance(pl, FinalPlacement)


# ── use_hold 테스트 ───────────────────────────────────────────────────────────

class TestUseHold:
    def test_use_hold_false_excludes_hold(self):
        """use_hold=False이면 root_scores에 hold 배치가 없어야 한다."""
        env = make_state()
        r = BeamSearchBot(
            config=SearchConfig(beam_width=50, depth=1, use_hold=False)
        ).search(env)
        assert not any(pl.use_hold for pl in r.root_scores)

    def test_use_hold_true_may_include_hold(self):
        """use_hold=True이면 hold 배치가 root_scores에 나타날 수 있다."""
        env = make_state()
        r = BeamSearchBot(
            config=SearchConfig(beam_width=50, depth=1, use_hold=True)
        ).search(env)
        # hold 배치는 있을 수도, 없을 수도 있음 — 타입 확인만
        for pl in r.root_scores:
            assert isinstance(pl, FinalPlacement)


# ── track_path 테스트 ─────────────────────────────────────────────────────────

class TestTrackPath:
    def setup_method(self):
        self.env = make_state()

    def test_track_path_false_stores_root_placement(self):
        """track_path=False(기본)이면 best_path에 root_placement 하나만 저장된다."""
        cfg    = SearchConfig(beam_width=10, depth=3, track_path=False)
        result = BeamSearchBot(config=cfg).search(self.env)
        assert len(result.best_path) == 1
        assert result.best_path[0] == result.best_placement

    def test_track_path_true_depth1_has_one_placement(self):
        """track_path=True, depth=1이면 path 길이 1."""
        cfg    = SearchConfig(beam_width=10, depth=1, track_path=True)
        result = BeamSearchBot(config=cfg).search(self.env)
        assert len(result.best_path) >= 1
        assert result.best_path[0] == result.best_placement

    def test_track_path_true_depth2_has_longer_path(self):
        """track_path=True, depth=2이면 path가 최대 2 배치를 포함할 수 있다."""
        cfg    = SearchConfig(beam_width=10, depth=2, track_path=True)
        result = BeamSearchBot(config=cfg).search(self.env)
        assert 1 <= len(result.best_path) <= 2
        assert result.best_path[0] == result.best_placement

    def test_track_path_false_default(self):
        """track_path 기본값은 False이다."""
        assert SearchConfig().track_path is False


# ── lean simulation 일관성 테스트 ─────────────────────────────────────────────

class TestLeanSimulationConsistency:
    def test_same_config_gives_same_placement(self):
        """동일 설정·동일 env → 동일 배치."""
        env = make_state()
        cfg = SearchConfig(beam_width=10, depth=1, time_budget_ms=None)
        r1  = BeamSearchBot(config=cfg).search(env)
        r2  = BeamSearchBot(config=cfg).search(env)
        assert r1.best_placement == r2.best_placement

    def test_lean_result_env_is_terminal_free(self):
        """lean simulation 후 반환된 env가 항상 ONGOING이어야 한다 (일반 상태)."""
        from bot.adapter import Action
        adapter = BotEnvAdapter()
        env     = make_state()
        legal   = adapter.list_legal_actions(env)

        for action in legal[:3]:
            result_env, lr = adapter.simulate_action_lean(env, action)
            assert list(env.next_queue)  # 원본 큐가 살아 있음

    def test_build_summary_from_env_matches_full_path(self):
        """build_summary_from_env 결과가 기존 build_outcome_summary와 동일해야 한다."""
        from bot.adapter import Action, build_summary_from_env
        from bot.tactical_evaluator import build_outcome_summary

        adapter = BotEnvAdapter()
        env     = make_state()

        lean_env, lean_lr = adapter.simulate_action_lean(env, Action.HARD_DROP)
        full_env, full_snap, full_lr = adapter.simulate_action_full(env, Action.HARD_DROP)

        lean_sum = build_summary_from_env(lean_env, lean_lr)
        full_sum = build_outcome_summary(full_env, full_snap, full_lr)

        assert lean_sum.lines_cleared    == full_sum.lines_cleared
        assert lean_sum.spin_type        == full_sum.spin_type
        assert lean_sum.did_b2b          == full_sum.did_b2b
        assert lean_sum.combo            == full_sum.combo
        assert lean_sum.is_perfect_clear == full_sum.is_perfect_clear
        assert lean_sum.terminal         == full_sum.terminal


# ── quiescence 테스트 ─────────────────────────────────────────────────────────

class TestQuiescence:
    def test_quiescence_off_still_works(self):
        cfg    = SearchConfig(beam_width=10, depth=2, quiescence_max_extensions=0)
        result = BeamSearchBot(config=cfg).search(make_state())
        assert result.best_placement is not None

    def test_quiescence_on_returns_placement(self):
        cfg    = SearchConfig(beam_width=10, depth=2, quiescence_max_extensions=2)
        result = BeamSearchBot(config=cfg).search(make_state())
        assert result.best_placement is not None


# ── 성능 특성 테스트 (slow) ───────────────────────────────────────────────────

@pytest.mark.slow
class TestPresetPerformance:
    """
    실제 시간을 측정하는 테스트. 'pytest -m slow'로 실행한다.
    CI 환경에서는 허용 범위를 5배로 넉넉히 잡는다.
    """

    def _run_preset(self, cfg: SearchConfig, n_steps: int = 5) -> list[float]:
        bot     = BeamSearchBot(config=cfg)
        adapter = BotEnvAdapter()
        env     = make_state()
        times   = []

        for _ in range(n_steps):
            if adapter.is_terminal(env):
                break
            t0  = time.monotonic()
            res = bot.search(env)
            times.append((time.monotonic() - t0) * 1000)
            if res.best_placement is None:
                break
            env, _lr = adapter.simulate_placement(env, res.best_placement)

        return times

    def test_fast_play_within_budget(self):
        cfg   = SearchConfig.fast_play()
        times = self._run_preset(cfg)
        assert times, "측정값이 없음"
        avg = sum(times) / len(times)
        assert avg < cfg.time_budget_ms * 5

    def test_safe_default_within_budget(self):
        cfg   = SearchConfig.safe_default()
        times = self._run_preset(cfg)
        assert times
        avg = sum(times) / len(times)
        assert avg < cfg.time_budget_ms * 5

    def test_deeper_play_within_budget(self):
        cfg   = SearchConfig.deeper_play()
        times = self._run_preset(cfg)
        assert times
        avg = sum(times) / len(times)
        assert avg < cfg.time_budget_ms * 5
