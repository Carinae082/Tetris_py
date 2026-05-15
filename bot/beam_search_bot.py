"""
bot/beam_search_bot.py — 빔 서치 봇 (fusion 스타일 합성 점수 + 도달 가능 배치 기반)

fusion 대비 변경 사항
---------------------
* 결정 단위: 원시 행동(Action) → 도달 가능한 최종 배치(FinalPlacement)
  - BFS 가 모든 도달 가능한 (row, col, rotation, 스핀 컨텍스트) 를 열거
  - simulate_placement 로 배치를 즉시 적용 (input_sequence 재생 불필요)
* 합성 점수 = board*1.0 + (cum_attack/√depth)*0.50
                        + (cum_chain/√depth)*0.15
                        + context*0.10
* B2B 체이닝: 로그 보너스 (_b2b_chaining_bonus)
* 콤보: log floor 보장 multiplier (_apply_combo_multiplier)
* 체인 가치: 1 - exp(-0.25 * combo)  (shape_chain_value)
* Context: CoachingState delta (fatality + obligation + surge + phase) + combo delta
* 깊이 스케일: cum_attack, cum_chain 을 min(√depth, 2.45) 로 나눔
* SearchNode 에 path_attack, path_chain, path_placements 추가

공개 API
--------
select_placement(env)         → FinalPlacement | None
evaluate_root_placements(env) → dict[FinalPlacement, float]
search(env)                   → SearchResult
"""

from __future__ import annotations

import math
import time

from engine.state import GameState, RoundResult

from .adapter import BotEnvAdapter, build_summary_from_env
from .attack_model import AttackConfig, calculate_immediate_attack
from .evaluator import EvalWeights, evaluate_board
from .placement_generator import FinalPlacement, list_reachable_placements
from .search_config import SearchConfig
from .search_node import SearchNode, SearchResult
from .tactical_evaluator import (
    TacticalWeights, _DEFAULT_TACTICAL,
    CoachingState, compute_coaching_state, coaching_context_bias, shape_context_modifier,
    shape_chain_value,
)
from .transposition_table import TranspositionTable

_NEG_INF            = float("-inf")
_QUIESCENCE_ATK_THR = 2.0   # attack_score 이 값 이상 → "활성 노드"


class BeamSearchBot:
    """
    빔 서치 기반 다중 배치(ply) 탐색 봇. fusion 스타일 합성 점수 사용.

    각 검색 노드는 "한 피스 배치 완료 후" 상태를 나타낸다.
    depth=N 이면 N번의 피스 배치를 미리 내다본다.
    결정 단위는 FinalPlacement (BFS 로 열거한 도달 가능 배치).
    """

    def __init__(
        self,
        config:           SearchConfig    | None = None,
        tactical_weights: TacticalWeights | None = None,
        eval_weights:     EvalWeights     | None = None,
        attack_config:    AttackConfig    | None = None,
    ) -> None:
        self._config   = config or SearchConfig()
        self._adapter  = BotEnvAdapter()
        self._tactical = tactical_weights
        self._eval     = eval_weights
        self._attack   = attack_config
        self._tt       = TranspositionTable()  # 탐색별 board_score 캐시

    # ── public API ────────────────────────────────────────────────────────────

    def select_placement(self, env: GameState) -> FinalPlacement | None:
        """최선 배치 하나를 반환한다. 도달 가능한 배치가 없으면 None."""
        return self.search(env).best_placement

    def evaluate_root_placements(self, env: GameState) -> dict[FinalPlacement, float]:
        """각 루트 배치의 최고 cumulative_score 를 반환한다."""
        return self.search(env).root_scores

    def search(self, env: GameState) -> SearchResult:
        """빔 서치를 수행하고 SearchResult 를 반환한다."""
        cfg = self._config

        # 게임 종료 또는 활성 미노 없음 → 즉시 반환
        if env.active is None or env.round_result != RoundResult.ONGOING:
            return SearchResult(
                best_placement=None, best_score=_NEG_INF,
                root_scores={}, expanded_nodes=0,
                searched_depth=0, best_path=[],
            )

        start_time: float | None = (
            time.monotonic() if cfg.time_budget_ms is not None else None
        )

        # 이전 탐색의 캐시 초기화 (탐색마다 fresh start)
        self._tt.clear()

        track_path = cfg.track_path

        # ── depth=1 빔 초기화 ────────────────────────────────────────────────
        beam, expanded_nodes = self._build_depth1_beam(env, track_path)

        if not beam:
            return SearchResult(
                best_placement=None, best_score=_NEG_INF,
                root_scores={}, expanded_nodes=0,
                searched_depth=0, best_path=[],
            )

        if cfg.root_top_k is not None and len(beam) > cfg.root_top_k:
            beam.sort(key=lambda n: n.cumulative_score, reverse=True)
            beam = beam[: cfg.root_top_k]

        beam = self._prune_and_truncate(beam, cfg.beam_width, cfg.futility_delta)
        searched_depth = 1

        # ── depth 2 ~ cfg.depth ──────────────────────────────────────────────
        for depth in range(2, cfg.depth + 1):
            if _time_exceeded(start_time, cfg.time_budget_ms):
                break

            all_children, n_expanded, interrupted = self._expand_beam_nodes(
                beam, track_path, start_time, cfg.time_budget_ms
            )
            expanded_nodes += n_expanded

            if not all_children:
                break

            beam = self._prune_and_truncate(
                all_children, cfg.beam_width, cfg.futility_delta
            )
            searched_depth = depth

            if interrupted:
                break

        # ── 정적 탐색(Quiescence) ─────────────────────────────────────────────
        if (
            cfg.quiescence_max_extensions > 0
            and beam
            and not _time_exceeded(start_time, cfg.time_budget_ms)
        ):
            beam, q_expanded = self._quiescence_search(beam, cfg, start_time, track_path)
            expanded_nodes += q_expanded

        # ── 결과 집계 ─────────────────────────────────────────────────────────
        if not beam:
            placements = list_reachable_placements(env)
            fallback   = placements[0] if placements else None
            return SearchResult(
                best_placement=fallback, best_score=_NEG_INF,
                root_scores={}, expanded_nodes=expanded_nodes,
                searched_depth=searched_depth,
                best_path=[fallback] if fallback else [],
            )

        best_node = max(beam, key=lambda n: n.cumulative_score)

        root_scores: dict[FinalPlacement, float] = {}
        for node in beam:
            pl = node.root_placement
            if pl not in root_scores or node.cumulative_score > root_scores[pl]:
                root_scores[pl] = node.cumulative_score

        return SearchResult(
            best_placement=best_node.root_placement,
            best_score=best_node.cumulative_score,
            root_scores=root_scores,
            expanded_nodes=expanded_nodes,
            searched_depth=searched_depth,
            best_path=list(best_node.path_placements),
        )

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _get_tactical(self) -> TacticalWeights:
        return self._tactical if self._tactical is not None else _DEFAULT_TACTICAL

    def _compute_composite(
        self,
        board_s:    float,
        cum_attack: float,
        cum_chain:  float,
        context_s:  float,
        depth:      int,
    ) -> float:
        """
        fusion assemble_composite + 깊이 스케일링.

        composite = board * board_weight
                  + (cum_attack / depth_factor) * attack_weight
                  + (cum_chain  / depth_factor) * chain_weight
                  + context * context_weight

        depth_factor = min(sqrt(depth), max_depth_factor)
        depth=1 → depth_factor=1.0 (스케일링 없음)
        """
        cfg          = self._config
        depth_factor = min(math.sqrt(max(depth, 1)), cfg.max_depth_factor)
        return (
            board_s * cfg.board_weight
            + (cum_attack / depth_factor) * cfg.attack_weight
            + (cum_chain  / depth_factor) * cfg.chain_weight
            + context_s   * cfg.context_weight
        )

    def _eval_components(self, summary, result_board) -> tuple[float, float, float]:
        """
        (board_s, attack_s, chain_s) 를 반환한다.
        context_s 는 _compute_context() 로 별도 계산한다.

        result_board : 시뮬레이션 결과 Board 객체. TT 캐시 키로 사용하며,
                       캐시 미스 시 board.col_heights 를 evaluate_board 에 전달한다.
        """
        board_hash = result_board.zobrist_hash
        board_s = self._tt.get(board_hash)
        if board_s is None:
            board_s = evaluate_board(
                summary.board, self._eval, result_board.col_heights
            )
            self._tt.set(board_hash, board_s)

        attack_s = calculate_immediate_attack(summary, self._attack)
        chain_s  = shape_chain_value(float(max(summary.combo, 0)))
        return board_s, attack_s, chain_s

    @staticmethod
    def _annotate_surge(
        summary,
        parent_b2b:   int,
        result_env:   GameState,
        lock_result,
    ) -> None:
        """
        fusion/attack.rs surge release 용 b2b_broken_from 을 summary에 주입한다.

        non-difficult 클리어(single/double/triple, no spin)가 긴 B2B 체인(Python b2b >= 3,
        Rust 환산 >= 4)을 끊었을 때 summary.b2b_broken_from = parent_b2b + 1 (Rust 스케일).
        """
        if (
            lock_result is not None
            and lock_result.lines_cleared > 0
            and parent_b2b >= 3          # Python b2b 3 → Rust b2b 4 (surge 임계값)
            and result_env.b2b < 0       # 체인이 끊김
        ):
            summary.b2b_broken_from = parent_b2b + 1  # Rust 스케일로 저장

    @staticmethod
    def _compute_context(
        prev_coaching:    CoachingState,
        prev_combo:       int,
        result_env:       GameState,
        lines_cleared:    int,
    ) -> tuple[float, CoachingState]:
        """
        fusion/search_expand.rs 의 context_mod 계산 포팅.

        1. result_env 에서 next CoachingState 를 계산한다.
        2. combo delta + coaching delta → shape_context_modifier → context_s.

        반환: (context_s, next_coaching)
        """
        heights   = result_env.board.col_heights
        max_h     = max(heights) if heights else 0
        # 클리어 후 남은 가비지 (부모 가비지 기준은 result_env 가 이미 반영)
        remaining_garbage = result_env.incoming.total_lines()

        next_coaching = compute_coaching_state(
            max_height       = max_h,
            spawn_blocked    = result_env.spawn_blocked,
            b2b              = result_env.b2b,
            pieces_placed    = result_env.pieces_placed,
            imminent_garbage = remaining_garbage,
            lines_cleared    = lines_cleared,
        )

        combo_context = float(max(result_env.combo + 1, 0)) - float(max(prev_combo + 1, 0))
        context_s = shape_context_modifier(
            combo_context + coaching_context_bias(prev_coaching, next_coaching)
        )
        return context_s, next_coaching

    def _build_depth1_beam(
        self,
        env: GameState,
        track_path: bool,
    ) -> tuple[list[SearchNode], int]:
        """루트 배치 전체를 빠르게 평가해 depth=1 빔을 구성한다."""
        adapter    = self._adapter
        cfg        = self._config
        tw         = self._get_tactical()
        placements = list_reachable_placements(env)
        beam: list[SearchNode] = []

        # 루트(배치 전) coaching state — depth-1 자식들의 delta 기준점
        root_heights  = env.board.col_heights
        root_coaching = compute_coaching_state(
            max_height       = max(root_heights) if root_heights else 0,
            spawn_blocked    = env.spawn_blocked,
            b2b              = env.b2b,
            pieces_placed    = env.pieces_placed,
            imminent_garbage = env.incoming.total_lines(),
            lines_cleared    = 0,
        )

        for placement in placements:
            if not cfg.use_hold and placement.use_hold:
                continue

            result_env, lock_result = adapter.simulate_placement(env, placement)
            summary = build_summary_from_env(result_env, lock_result)
            self._annotate_surge(summary, env.b2b, result_env, lock_result)

            if summary.terminal:
                composite     = tw.terminal_penalty
                board_s       = 0.0
                attack_s      = 0.0
                chain_s       = 0.0
                next_coaching = root_coaching
            else:
                board_s, attack_s, chain_s = self._eval_components(summary, result_env.board)
                lc = lock_result.lines_cleared if lock_result else 0
                context_s, next_coaching = self._compute_context(
                    root_coaching, env.combo, result_env, lc
                )
                composite = self._compute_composite(
                    board_s, attack_s, chain_s, context_s, depth=1
                )

            node = SearchNode(
                env              = result_env,
                root_placement   = placement,
                depth            = 1,
                cumulative_score = composite,
                immediate_score  = composite,
                board_score      = board_s,
                attack_score     = attack_s,
                terminal         = summary.terminal,
                used_hold        = placement.use_hold,
                lines_cleared    = summary.lines_cleared,
                # 루트는 track_path 여부에 무관하게 항상 [placement] 저장
                # → best_path 가 빈 리스트가 되지 않도록 보장
                path_placements  = [placement],
                path_attack      = attack_s,
                path_chain       = chain_s,
                coaching         = next_coaching,
            )
            beam.append(node)

        return beam, len(placements)

    def _expand_node(
        self,
        node: SearchNode,
        track_path: bool,
    ) -> list[SearchNode]:
        """
        노드의 env 에서 도달 가능한 배치를 모두 시뮬레이션해 자식 SearchNode 목록을 반환한다.
        누적 attack/chain 에 깊이 스케일링을 적용해 합성 점수를 계산한다.
        """
        if node.terminal:
            return []

        adapter    = self._adapter
        cfg        = self._config
        tw         = self._get_tactical()
        placements = list_reachable_placements(node.env)
        if not placements:
            return []

        child_depth = node.depth + 1
        children: list[SearchNode] = []

        for placement in placements:
            if not cfg.use_hold and placement.use_hold:
                continue

            result_env, lock_result = adapter.simulate_placement(node.env, placement)
            summary = build_summary_from_env(result_env, lock_result)
            self._annotate_surge(summary, node.env.b2b, result_env, lock_result)

            if summary.terminal:
                composite     = tw.terminal_penalty
                board_s       = 0.0
                attack_s      = 0.0
                chain_s       = 0.0
                cum_attack    = node.path_attack
                cum_chain     = node.path_chain
                next_coaching = node.coaching
            else:
                board_s, attack_s, chain_s = self._eval_components(summary, result_env.board)
                lc = lock_result.lines_cleared if lock_result else 0
                context_s, next_coaching = self._compute_context(
                    node.coaching, node.env.combo, result_env, lc
                )
                cum_attack = node.path_attack + attack_s
                cum_chain  = node.path_chain  + chain_s
                composite  = self._compute_composite(
                    board_s, cum_attack, cum_chain, context_s, depth=child_depth
                )

            if track_path:
                path = node.path_placements + [placement]
            else:
                path = [node.root_placement]

            child = SearchNode(
                env              = result_env,
                root_placement   = node.root_placement,
                depth            = child_depth,
                cumulative_score = composite,
                immediate_score  = composite,
                board_score      = board_s,
                attack_score     = attack_s,
                terminal         = summary.terminal,
                used_hold        = node.used_hold or placement.use_hold,
                lines_cleared    = summary.lines_cleared,
                path_placements  = path,
                path_attack      = cum_attack,
                path_chain       = cum_chain,
                coaching         = next_coaching,
            )
            children.append(child)

        return children

    def _expand_beam_nodes(
        self,
        beam: list[SearchNode],
        track_path: bool,
        start_time: float | None,
        budget_ms: float | None,
    ) -> tuple[list[SearchNode], int, bool]:
        """빔 전체를 확장한다. 시간 초과 시 조기 중단한다."""
        all_children: list[SearchNode] = []
        total = 0
        interrupted = False

        for node in beam:
            if _time_exceeded(start_time, budget_ms):
                interrupted = True
                break

            if node.terminal:
                all_children.append(node)
                continue

            children = self._expand_node(node, track_path)
            all_children.extend(children)
            total += len(children)

        return all_children, total, interrupted

    def _quiescence_search(
        self,
        beam: list[SearchNode],
        cfg: SearchConfig,
        start_time: float | None,
        track_path: bool,
    ) -> tuple[list[SearchNode], int]:
        """빔 상위 quiescence_beam_fraction 중 활성 노드를 추가 확장한다."""
        quiescence_size = max(1, int(len(beam) * cfg.quiescence_beam_fraction))
        q_beam = [
            n for n in beam[:quiescence_size]
            if not n.terminal and _is_exciting(n)
        ]

        total_expanded = 0

        for _ in range(cfg.quiescence_max_extensions):
            if not q_beam or _time_exceeded(start_time, cfg.time_budget_ms):
                break

            q_children: list[SearchNode] = []
            for node in q_beam:
                if _time_exceeded(start_time, cfg.time_budget_ms):
                    break
                children = self._expand_node(node, track_path)
                total_expanded += len(children)
                exciting = [c for c in children if _is_exciting(c)]
                q_children.extend(exciting)

            if not q_children:
                break

            q_children.sort(key=lambda n: n.cumulative_score, reverse=True)
            q_beam = q_children[:quiescence_size]

            beam_floor = beam[-1].cumulative_score if beam else _NEG_INF
            for node in q_beam:
                if node.cumulative_score > beam_floor:
                    beam.append(node)

            beam = _prune_truncate_static(beam, cfg.beam_width, cfg.futility_delta)

        return beam, total_expanded

    @staticmethod
    def _prune_and_truncate(
        nodes: list[SearchNode],
        beam_width: int,
        futility_delta: float,
    ) -> list[SearchNode]:
        return _prune_truncate_static(nodes, beam_width, futility_delta)


# ── 모듈 레벨 순수 함수 ───────────────────────────────────────────────────────

def _prune_truncate_static(
    nodes: list[SearchNode],
    beam_width: int,
    futility_delta: float,
) -> list[SearchNode]:
    """
    2단계 pruning:
      1단계: immediate_score 기준 futility 필터 (비-terminal 노드만)
      2단계: cumulative_score 내림차순 정렬 + beam_width 절단
    """
    if not nodes:
        return []

    term     = [n for n in nodes if     n.terminal]
    non_term = [n for n in nodes if not n.terminal]

    if non_term:
        best_imm  = max(n.immediate_score for n in non_term)
        threshold = best_imm - futility_delta
        non_term  = [n for n in non_term if n.immediate_score >= threshold]

    combined = non_term + term
    if not combined:
        return []

    combined.sort(key=lambda n: n.cumulative_score, reverse=True)
    return combined[:beam_width]


def _time_exceeded(start: float | None, budget_ms: float | None) -> bool:
    if start is None or budget_ms is None:
        return False
    return (time.monotonic() - start) * 1000 >= budget_ms


def _is_exciting(node: SearchNode) -> bool:
    """정적 탐색 확장 대상 판단: 라인 클리어 또는 강한 공격."""
    return (
        node.lines_cleared > 0
        or node.attack_score >= _QUIESCENCE_ATK_THR
    )
