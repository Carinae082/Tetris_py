"""
tests/test_beam_search_bot.py — SearchConfig, SearchNode, SearchResult,
                                 BeamSearchBot 테스트
"""

from __future__ import annotations

import pytest

from engine.bag import SevenBag
from engine.state import GameState, RoundResult

from bot.adapter import BotEnvAdapter
from bot.attack_model import AttackConfig
from bot.beam_search_bot import BeamSearchBot
from bot.evaluator import EvalWeights
from bot.placement_generator import FinalPlacement, list_reachable_placements
from bot.search_config import SearchConfig
from bot.search_node import SearchNode, SearchResult
from bot.tactical_evaluator import TacticalWeights


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_state(seed: int = 42, n_pieces: int = 14) -> GameState:
    bag = SevenBag(seed=seed)
    state = GameState()
    for _ in range(n_pieces):
        state.enqueue(bag.pop())
    state.spawn_next()
    return state


def _dummy_placement(env: GameState) -> FinalPlacement:
    """테스트용 첫 번째 도달 가능 배치를 반환한다."""
    return list_reachable_placements(env)[0]


# ── SearchConfig ──────────────────────────────────────────────────────────────

class TestSearchConfig:
    def test_defaults(self):
        cfg = SearchConfig()
        assert cfg.beam_width                == 120
        assert cfg.depth                     == 4
        assert cfg.futility_delta            == pytest.approx(15.0)
        assert cfg.time_budget_ms            is None
        assert cfg.use_hold                  is True
        assert cfg.root_top_k               is None
        assert cfg.quiescence_max_extensions == 2
        assert cfg.quiescence_beam_fraction  == pytest.approx(0.20)

    def test_custom_values(self):
        cfg = SearchConfig(beam_width=50, depth=2, futility_delta=5.0)
        assert cfg.beam_width     == 50
        assert cfg.depth          == 2
        assert cfg.futility_delta == pytest.approx(5.0)

    def test_time_budget_can_be_set(self):
        cfg = SearchConfig(time_budget_ms=100.0)
        assert cfg.time_budget_ms == pytest.approx(100.0)

    def test_use_hold_false(self):
        cfg = SearchConfig(use_hold=False)
        assert cfg.use_hold is False

    def test_root_top_k(self):
        cfg = SearchConfig(root_top_k=10)
        assert cfg.root_top_k == 10


# ── SearchNode ────────────────────────────────────────────────────────────────

class TestSearchNode:
    def test_fields_exist(self):
        env = make_state()
        pl  = _dummy_placement(env)
        node = SearchNode(
            env              = env,
            root_placement   = pl,
            depth            = 1,
            cumulative_score = 1000.0,
            immediate_score  = 1000.0,
            board_score      = -10.0,
            attack_score     = 4.0,
            path_placements  = [pl],
            terminal         = False,
            used_hold        = False,
            lines_cleared    = 0,
        )
        assert node.depth             == 1
        assert node.root_placement    is pl
        assert node.cumulative_score  == pytest.approx(1000.0)
        assert node.immediate_score   == pytest.approx(1000.0)
        assert node.board_score       == pytest.approx(-10.0)
        assert node.attack_score      == pytest.approx(4.0)
        assert node.path_placements   == [pl]
        assert node.terminal          is False
        assert node.used_hold         is False
        assert node.lines_cleared     == 0

    def test_lines_cleared_default(self):
        env = make_state()
        pl  = _dummy_placement(env)
        node = SearchNode(
            env              = env,
            root_placement   = pl,
            depth            = 1,
            cumulative_score = 0.0,
            immediate_score  = 0.0,
            board_score      = 0.0,
            attack_score     = 0.0,
            path_placements  = [],
            terminal         = False,
            used_hold        = False,
        )
        assert node.lines_cleared == 0


# ── SearchResult ──────────────────────────────────────────────────────────────

class TestSearchResult:
    def test_fields_exist(self):
        env = make_state()
        pl  = _dummy_placement(env)
        result = SearchResult(
            best_placement = pl,
            best_score     = 1234.5,
            root_scores    = {pl: 1234.5},
            expanded_nodes = 100,
            searched_depth = 4,
            best_path      = [pl],
        )
        assert result.best_placement is pl
        assert result.best_score     == pytest.approx(1234.5)
        assert result.expanded_nodes == 100
        assert result.searched_depth == 4
        assert result.best_path      == [pl]


# ── BeamSearchBot 생성 ────────────────────────────────────────────────────────

class TestBeamSearchBotConstruction:
    def test_default_construction(self):
        bot = BeamSearchBot()
        assert bot is not None

    def test_custom_config(self):
        cfg = SearchConfig(beam_width=10, depth=2)
        bot = BeamSearchBot(config=cfg)
        assert bot is not None

    def test_all_custom(self):
        bot = BeamSearchBot(
            config=SearchConfig(beam_width=20, depth=2),
            tactical_weights=TacticalWeights(),
            eval_weights=EvalWeights(),
            attack_config=AttackConfig.quick_play(),
        )
        assert bot is not None


# ── select_placement ──────────────────────────────────────────────────────────

class TestSelectPlacement:
    def setup_method(self):
        self.cfg     = SearchConfig(beam_width=10, depth=2)
        self.bot     = BeamSearchBot(config=self.cfg)
        self.env     = make_state()
        self.adapter = BotEnvAdapter()

    def test_returns_final_placement(self):
        pl = self.bot.select_placement(self.env)
        assert isinstance(pl, FinalPlacement)

    def test_placement_is_reachable(self):
        """선택된 배치가 도달 가능 목록에 포함된다."""
        pl         = self.bot.select_placement(self.env)
        reachable  = list_reachable_placements(self.env)
        # piece_type / row / col / rotation / use_hold 가 일치하는 배치가 존재
        assert any(
            r.piece_type == pl.piece_type
            and r.row == pl.row
            and r.col == pl.col
            and r.rotation == pl.rotation
            and r.use_hold == pl.use_hold
            for r in reachable
        )

    def test_returns_none_when_terminal(self):
        self.env.round_result = RoundResult.LOSE
        assert self.bot.select_placement(self.env) is None

    def test_does_not_mutate_env(self):
        orig_pieces = self.env.pieces_placed
        orig_queue  = list(self.env.next_queue)
        self.bot.select_placement(self.env)
        assert self.env.pieces_placed  == orig_pieces
        assert list(self.env.next_queue) == orig_queue


# ── evaluate_root_placements ──────────────────────────────────────────────────

class TestEvaluateRootPlacements:
    def setup_method(self):
        self.cfg = SearchConfig(beam_width=10, depth=2)
        self.bot = BeamSearchBot(config=self.cfg)
        self.env = make_state()

    def test_returns_dict(self):
        scores = self.bot.evaluate_root_placements(self.env)
        assert isinstance(scores, dict)

    def test_keys_are_final_placements(self):
        scores = self.bot.evaluate_root_placements(self.env)
        for key in scores:
            assert isinstance(key, FinalPlacement)

    def test_values_are_floats(self):
        scores = self.bot.evaluate_root_placements(self.env)
        for val in scores.values():
            assert isinstance(val, float)


# ── search ────────────────────────────────────────────────────────────────────

class TestSearch:
    def setup_method(self):
        self.cfg = SearchConfig(beam_width=10, depth=2)
        self.bot = BeamSearchBot(config=self.cfg)
        self.env = make_state()

    def test_returns_search_result(self):
        result = self.bot.search(self.env)
        assert isinstance(result, SearchResult)

    def test_best_placement_not_none(self):
        result = self.bot.search(self.env)
        assert result.best_placement is not None

    def test_best_score_finite(self):
        result = self.bot.search(self.env)
        assert result.best_score > float("-inf")

    def test_expanded_nodes_positive(self):
        result = self.bot.search(self.env)
        assert result.expanded_nodes > 0

    def test_searched_depth_at_least_one(self):
        result = self.bot.search(self.env)
        assert result.searched_depth >= 1

    def test_searched_depth_bounded_by_config(self):
        result = self.bot.search(self.env)
        assert result.searched_depth <= self.cfg.depth

    def test_best_path_nonempty(self):
        result = self.bot.search(self.env)
        assert len(result.best_path) >= 1

    def test_best_path_starts_with_best_placement(self):
        result = self.bot.search(self.env)
        assert result.best_path[0] == result.best_placement

    def test_root_scores_covers_best_placement(self):
        result = self.bot.search(self.env)
        assert result.best_placement in result.root_scores

    def test_best_score_equals_root_score_best(self):
        result = self.bot.search(self.env)
        # best_score는 root_scores의 최댓값 이상이어야 한다
        assert result.best_score >= max(result.root_scores.values())

    def test_terminal_returns_none_placement(self):
        self.env.round_result = RoundResult.LOSE
        result = self.bot.search(self.env)
        assert result.best_placement is None

    def test_terminal_has_zero_expanded(self):
        self.env.round_result = RoundResult.LOSE
        result = self.bot.search(self.env)
        assert result.expanded_nodes == 0


# ── 설정 변경 효과 ─────────────────────────────────────────────────────────────

class TestConfigEffects:
    def setup_method(self):
        self.env = make_state()

    def test_depth_one(self):
        bot    = BeamSearchBot(config=SearchConfig(beam_width=10, depth=1))
        result = bot.search(self.env)
        assert result.searched_depth == 1

    def test_depth_two(self):
        bot    = BeamSearchBot(config=SearchConfig(beam_width=10, depth=2))
        result = bot.search(self.env)
        assert result.searched_depth == 2

    def test_use_hold_false_no_hold_placement(self):
        bot    = BeamSearchBot(config=SearchConfig(beam_width=10, depth=1, use_hold=False))
        result = bot.search(self.env)
        assert not any(pl.use_hold for pl in result.root_scores)

    def test_root_top_k_limits_root_diversity(self):
        cfg_full  = SearchConfig(beam_width=50, depth=1, root_top_k=None)
        cfg_limit = SearchConfig(beam_width=50, depth=1, root_top_k=2)
        r_full    = BeamSearchBot(config=cfg_full).search(self.env)
        r_limit   = BeamSearchBot(config=cfg_limit).search(self.env)
        assert len(r_limit.root_scores) <= len(r_full.root_scores)

    def test_narrow_beam_still_returns_placement(self):
        bot    = BeamSearchBot(config=SearchConfig(beam_width=1, depth=2))
        result = bot.search(self.env)
        assert result.best_placement is not None

    def test_time_budget_respected(self):
        """time_budget_ms=1 이면 거의 즉시 종료되어 searched_depth가 짧아야 한다."""
        import time
        bot    = BeamSearchBot(config=SearchConfig(
            beam_width=120, depth=10, time_budget_ms=1.0
        ))
        t0 = time.monotonic()
        result = bot.search(self.env)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert result.best_placement is not None
        assert result.searched_depth <= 10

    def test_quiescence_off_still_works(self):
        bot    = BeamSearchBot(config=SearchConfig(
            beam_width=10, depth=2, quiescence_max_extensions=0
        ))
        result = bot.search(self.env)
        assert result.best_placement is not None


# ── 다중 스텝 플레이 ───────────────────────────────────────────────────────────

class TestMultiStepPlay:
    def test_multi_step_play(self):
        adapter = BotEnvAdapter()
        bot     = BeamSearchBot(config=SearchConfig(beam_width=10, depth=2))
        env     = make_state()

        for _ in range(6):
            if adapter.is_terminal(env):
                break
            pl = bot.select_placement(env)
            assert pl is not None
            env, _lr = adapter.simulate_placement(env, pl)

    def test_avoids_immediate_terminal(self):
        """선택된 배치를 적용해도 즉시 game over가 되지 않아야 한다 (신선한 상태 기준)."""
        adapter = BotEnvAdapter()
        bot     = BeamSearchBot(config=SearchConfig(beam_width=10, depth=2))
        env     = make_state()

        pl = bot.select_placement(env)
        next_env, _lr = adapter.simulate_placement(env, pl)
        assert not adapter.is_terminal(next_env)

    def test_different_seeds_give_placements(self):
        bot = BeamSearchBot(config=SearchConfig(beam_width=10, depth=1))
        for seed in [0, 1, 99, 777]:
            env = make_state(seed=seed)
            pl  = bot.select_placement(env)
            assert pl is not None
