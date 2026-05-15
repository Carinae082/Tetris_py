"""
bot/rl_trainer.py — Evolution Strategy 기반 가중치 강화학습

OpenAI ES 변형 (Salimans et al., 2017):
  - antithetic perturbation sampling (+ε / -ε 쌍)
  - rank normalization (fitness shaping)
  - weight bounds clamping
  - 백그라운드 스레드 실행 + on_generation 콜백

학습 대상 가중치 (14개):
  EvalWeights  11개 : holes, cell_coveredness, height, height_upper_half,
                      height_upper_quarter, bumpiness, bumpiness_sq,
                      row_transitions, well_depth, tsd_overhang, four_wide_well
  SearchConfig  3개 : attack_weight, chain_weight, context_weight
"""
from __future__ import annotations

import json
import os
import random
import threading
from dataclasses import dataclass
from typing import Callable

from collections import deque

from engine import SevenBag, GameState, RoundResult
from engine.mino import ActivePiece, MinoType, SPAWN, SHAPES
from engine.board import BOARD_COLS, BOARD_ROWS
from .adapter import BotEnvAdapter, build_summary_from_env
from .attack_model import calculate_immediate_attack, AttackConfig
from .evaluator import EvalWeights, evaluate_board
from .placement_generator import FinalPlacement, _PS, _soft_drop_one, _TRANSITIONS
from .search_config import SearchConfig


# ── 가중치 명세 ────────────────────────────────────────────────────────────────

WEIGHT_NAMES: list[str] = [
    "holes", "cell_coveredness", "height",
    "height_upper_half", "height_upper_quarter",
    "bumpiness", "bumpiness_sq", "row_transitions",
    "well_depth", "tsd_overhang", "four_wide_well",
    "attack_weight", "chain_weight", "context_weight",
]

WEIGHT_DEFAULTS: list[float] = [
    -4.0, -0.5, -0.2, -1.0, -5.0,
    -0.3, -0.1, -0.3,
     0.2,  6.0,  1.5,
     0.50, 0.15, 0.10,
]

WEIGHT_BOUNDS: list[tuple[float, float]] = [
    (-20.0,  0.0),  # holes
    ( -5.0,  0.0),  # cell_coveredness
    ( -2.0,  0.0),  # height
    ( -5.0,  0.0),  # height_upper_half
    (-20.0,  0.0),  # height_upper_quarter
    ( -3.0,  0.0),  # bumpiness
    ( -3.0,  0.0),  # bumpiness_sq
    ( -3.0,  0.0),  # row_transitions
    (  0.0,  3.0),  # well_depth
    (  0.0, 20.0),  # tsd_overhang
    (  0.0, 10.0),  # four_wide_well
    (  0.0,  2.0),  # attack_weight
    (  0.0,  1.0),  # chain_weight
    (  0.0,  1.0),  # context_weight
]

N_WEIGHTS   = len(WEIGHT_NAMES)
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_weights.json")


# ── 가중치 직렬화 ──────────────────────────────────────────────────────────────

def weights_to_objects(w: list[float]) -> tuple[EvalWeights, SearchConfig]:
    """가중치 벡터 → (EvalWeights, SearchConfig) 변환. SearchConfig는 실전용."""
    ew = EvalWeights(
        holes                = w[0],
        cell_coveredness     = w[1],
        height               = w[2],
        height_upper_half    = w[3],
        height_upper_quarter = w[4],
        bumpiness            = w[5],
        bumpiness_sq         = w[6],
        row_transitions      = w[7],
        well_depth           = w[8],
        tsd_overhang         = w[9],
        four_wide_well       = w[10],
    )
    cfg = SearchConfig(
        beam_width     = 800,
        depth          = 14,
        time_budget_ms = 35.0,
        use_hold       = True,
        attack_weight  = w[11],
        chain_weight   = w[12],
        context_weight = w[13],
    )
    return ew, cfg


def clamp_weights(w: list[float]) -> list[float]:
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(w, WEIGHT_BOUNDS)]


def save_weights(w: list[float], path: str = WEIGHTS_PATH) -> None:
    data = {name: val for name, val in zip(WEIGHT_NAMES, w)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_weights(path: str = WEIGHTS_PATH) -> list[float] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [float(data.get(name, default))
                for name, default in zip(WEIGHT_NAMES, WEIGHT_DEFAULTS)]
    except Exception:
        return None


# ── 훈련 설정 ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    population_size: int   = 16     # 세대당 개체 수 (짝수 권장, antithetic)
    sigma:           float = 0.15   # 초기 탐색 표준편차
    learning_rate:   float = 0.05   # 파라미터 업데이트 학습률
    sigma_decay:     float = 0.998  # 세대마다 sigma 감소율
    sigma_min:       float = 0.02   # sigma 하한
    n_pieces:        int   = 20     # 게임당 배치 피스 수 (BFS 기반 → 속도 절충)
    n_eval_games:    int   = 2      # 가중치 평가 반복 게임 수
    garbage_rate:    int   = 10     # N 피스마다 가비지 1줄 추가 (0=없음)
    max_generations: int   = 500    # 최대 세대 수


@dataclass
class GenerationResult:
    generation:   int
    best_fitness: float
    mean_fitness: float
    sigma:        float
    weights:      list[float]  # 현재 center weights


# ── 훈련 전용 Save-Restore BFS ────────────────────────────────────────────────
# input_sequence를 추적하지 않고, 전환 실패 시 복원이 필요 없는
# (try_move/try_rotate는 성공 시에만 상태를 변경) 특성을 이용해
# copy()를 방문하지 않은 새 상태에만 호출한다 → ~6× 감소.

def _bfs_training(board, start_piece: ActivePiece, use_hold: bool) -> list[FinalPlacement]:
    """
    Save-restore BFS: input_sequence 미추적, copy()는 새 상태에만 호출.
    킥·T-spin 완전 지원.
    """
    start = _PS(board, start_piece)
    visited: set = {start.key()}
    queue: deque[_PS] = deque([start.copy()])

    landings: list[FinalPlacement] = []
    landing_keys: set = set()

    while queue:
        cur = queue.popleft()

        if cur.is_grounded():
            lkey = cur.key()
            if lkey not in landing_keys:
                landing_keys.add(lkey)
                p = cur.active
                landings.append(FinalPlacement(
                    piece_type         = p.type,
                    row                = p.row,
                    col                = p.col,
                    rotation           = p.rotation,
                    use_hold           = use_hold,
                    last_was_rotation  = cur.last_was_rotation,
                    last_kick_index    = cur.last_kick_index,
                    last_kick_upgrades = cur.last_kick_upgrades,
                    input_sequence     = (),
                ))

        # 전환 전 상태 저장 (필드 5개만)
        p_saved  = cur.active
        lwr_s    = cur.last_was_rotation
        lki_s    = cur.last_kick_index
        lku_s    = cur.last_kick_upgrades
        lt_s     = cur.lock_timer

        for _, fn in _TRANSITIONS:
            if fn(cur):                      # in-place 변경 (성공 시에만)
                nkey = cur.key()
                if nkey not in visited:
                    visited.add(nkey)
                    queue.append(cur.copy()) # 새 상태만 복사
                # 성공 후 복원 (p_saved 참조 재할당, 새 객체 생성 불필요)
                cur.active             = p_saved
                cur.last_was_rotation  = lwr_s
                cur.last_kick_index    = lki_s
                cur.last_kick_upgrades = lku_s
                cur.lock_timer         = lt_s
            # 실패 시 복원 불필요 (fn이 상태를 변경하지 않음)

    return landings


# 스핀 판정이 의미 있는 피스 타입 (T/S/Z/J/L).
# O·I는 스핀 이득이 없으므로 직접 낙하로 빠르게 처리한다.
_SPIN_TYPES: frozenset[MinoType] = frozenset([
    MinoType.T, MinoType.S, MinoType.Z, MinoType.J, MinoType.L,
])
_SHAPES = SHAPES
_BOARD_COLS = BOARD_COLS


def _fast_drop_placements(board, ptype: MinoType, use_hold: bool) -> list[FinalPlacement]:
    """
    col_heights로 착지 행을 O(4)에 계산 — is_valid_position 루프 없음.
    스핀·킥 미지원 (O/I 피스 전용).
    """
    results: list[FinalPlacement] = []
    heights = board.col_heights
    for rot in range(4):
        shape = _SHAPES[ptype][rot]
        col_offsets = [dc for _, dc in shape]
        min_dc = min(col_offsets)
        max_dc = max(col_offsets)
        for col in range(-min_dc, _BOARD_COLS - max_dc):
            # 각 셀 (dr, dc): row + dr < BOARD_ROWS - heights[col+dc] 이어야 하므로
            # row = min(BOARD_ROWS - heights[col+dc] - 1 - dr) 가 최대 유효 행
            r = min(BOARD_ROWS - heights[col + dc] - 1 - dr for dr, dc in shape)
            if r < 0:
                continue
            results.append(FinalPlacement(
                piece_type         = ptype,
                row                = r,
                col                = col,
                rotation           = rot,
                use_hold           = use_hold,
                last_was_rotation  = False,
                last_kick_index    = -1,
                last_kick_upgrades = False,
                input_sequence     = (),
            ))
    return results


def _list_training_placements(env: GameState) -> list[FinalPlacement]:
    """
    훈련 전용: 스핀 피스(T/S/Z/J/L)는 BFS, 나머지(O/I)는 직접 낙하.
    합리적인 정확도와 속도 균형을 위해 하이브리드 방식을 사용한다.
    """
    if env.active is None or env.round_result != RoundResult.ONGOING:
        return []

    board = env.board
    p     = env.active

    def _get_placements(ptype: MinoType, use_hold: bool) -> list[FinalPlacement]:
        if ptype in _SPIN_TYPES:
            sp_row, sp_col = SPAWN[ptype]
            start_piece    = ActivePiece(ptype, sp_row, sp_col, 0)
            if not board.is_valid_position(start_piece):
                return []
            return _bfs_training(board, start_piece, use_hold)
        else:
            return _fast_drop_placements(board, ptype, use_hold)

    placements = _get_placements(p.type, False)

    if not env.hold_used:
        hold_type = env.hold if env.hold is not None else (
            env.next_queue[0] if env.next_queue else None
        )
        if hold_type is not None:
            placements.extend(_get_placements(hold_type, True))

    return placements


# ── 헤드리스 게임 실행 ─────────────────────────────────────────────────────────

_adapter = BotEnvAdapter()
_atk_cfg = AttackConfig.tetra_league()


def _run_one_game(eval_w: EvalWeights, cfg: TrainConfig, rng: random.Random) -> float:
    """
    단일 헤드리스 게임을 실행하고 APP(attack per piece)를 반환한다.

    Greedy depth-1 평가: 하이브리드 배치 생성(_list_training_placements) +
    즉시 점수(board_score + attack) 기준으로 최선 배치 선택.
    """
    bag = SevenBag()
    env = GameState()

    for _ in range(7):
        env.enqueue(bag.pop())
    env.spawn_next()

    total_attack = 0.0
    pieces_done  = 0

    for i in range(cfg.n_pieces):
        if env.round_result != RoundResult.ONGOING:
            return total_attack / max(pieces_done, 1) - 3.0

        if cfg.garbage_rate > 0 and i > 0 and i % cfg.garbage_rate == 0:
            env.add_incoming_garbage(1, rng.randint(0, 9))
            if env.apply_incoming_garbage():
                return total_attack / max(pieces_done, 1) - 3.0

        placements = _list_training_placements(env)
        if not placements:
            break

        best_score  = float("-inf")
        best_result = None
        best_lock   = None

        for placement in placements:
            result_env, lock_result = _adapter.simulate_placement(env, placement)
            if result_env.round_result != RoundResult.ONGOING:
                continue
            summary    = build_summary_from_env(result_env, lock_result)
            atk        = calculate_immediate_attack(summary, _atk_cfg)
            board_s    = evaluate_board(summary.board, eval_w, result_env.board.col_heights)
            score      = board_s + atk * 0.5
            if score > best_score:
                best_score  = score
                best_result = result_env
                best_lock   = lock_result

        if best_result is None:
            break

        summary = build_summary_from_env(best_result, best_lock)
        total_attack += calculate_immediate_attack(summary, _atk_cfg)
        env = best_result
        while len(env.next_queue) < 7:
            env.enqueue(bag.pop())
        pieces_done += 1

    heights = env.board.col_heights
    avg_h   = sum(heights) / len(heights)
    height_penalty = max(0.0, avg_h - 8.0) * 0.05

    return total_attack / max(pieces_done, 1) - height_penalty


def _evaluate_weights(w: list[float], cfg: TrainConfig, rng: random.Random) -> float:
    """가중치를 n_eval_games 게임으로 평가해 평균 APP를 반환한다."""
    ew, _ = weights_to_objects(w)
    total = sum(_run_one_game(ew, cfg, rng) for _ in range(cfg.n_eval_games))
    return total / cfg.n_eval_games


# ── Rank normalization ────────────────────────────────────────────────────────

def _rank_normalize(fitnesses: list[float]) -> list[float]:
    """순위 기반 정규화 → [-0.5, 0.5]."""
    n = len(fitnesses)
    order = sorted(range(n), key=lambda i: fitnesses[i])
    result = [0.0] * n
    for rank, orig_idx in enumerate(order):
        result[orig_idx] = rank / max(n - 1, 1) - 0.5
    return result


# ── OpenAI ES 트레이너 ─────────────────────────────────────────────────────────

class RLTrainer:
    """
    OpenAI ES 기반 Tetris bot 가중치 최적화.

    백그라운드 스레드에서 실행되며 on_generation 콜백으로 진행 상황을 보고한다.
    UI는 콜백을 통해 GenerationResult를 수신하고 화면을 갱신한다.
    """

    def __init__(
        self,
        config:          TrainConfig | None                        = None,
        initial_weights: list[float] | None                        = None,
        on_generation:   Callable[[GenerationResult], None] | None = None,
        seed:            int                                       = 42,
    ):
        self.cfg           = config or TrainConfig()
        self.weights       = list(initial_weights or load_weights() or WEIGHT_DEFAULTS)
        self.on_generation = on_generation
        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng          = random.Random(seed)
        self.best_weights  = self.weights[:]
        self.best_fitness  = float("-inf")
        self.generation    = 0

    def start(self) -> None:
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="RLTrainer")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def save(self, path: str = WEIGHTS_PATH) -> None:
        save_weights(self.best_weights, path)

    def _run(self) -> None:
        cfg   = self.cfg
        n     = N_WEIGHTS
        sigma = cfg.sigma
        rng   = self._rng

        for gen in range(cfg.max_generations):
            if self._stop_event.is_set():
                break

            half     = cfg.population_size // 2
            epsilons = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(half)]

            fitnesses_pos: list[float] = []
            fitnesses_neg: list[float] = []

            for eps in epsilons:
                if self._stop_event.is_set():
                    return
                w_pos = clamp_weights([self.weights[i] + sigma * eps[i] for i in range(n)])
                w_neg = clamp_weights([self.weights[i] - sigma * eps[i] for i in range(n)])
                fitnesses_pos.append(_evaluate_weights(w_pos, cfg, rng))
                fitnesses_neg.append(_evaluate_weights(w_neg, cfg, rng))

            if self._stop_event.is_set():
                return

            all_f  = fitnesses_pos + fitnesses_neg
            ranked = _rank_normalize(all_f)
            rp, rn = ranked[:half], ranked[half:]

            # ES 그래디언트 계산
            grad = [0.0] * n
            for i, eps in enumerate(epsilons):
                diff = rp[i] - rn[i]
                for j in range(n):
                    grad[j] += diff * eps[j]

            # 가중치 업데이트
            lr = cfg.learning_rate / (half * sigma)
            self.weights = clamp_weights([self.weights[j] + lr * grad[j] for j in range(n)])

            # best 갱신 (세대 내 최고 샘플)
            best_f = max(all_f)
            if best_f > self.best_fitness:
                self.best_fitness = best_f
                bi = all_f.index(best_f)
                if bi < half:
                    self.best_weights = clamp_weights(
                        [self.weights[j] + sigma * epsilons[bi][j] for j in range(n)]
                    )
                else:
                    bi2 = bi - half
                    self.best_weights = clamp_weights(
                        [self.weights[j] - sigma * epsilons[bi2][j] for j in range(n)]
                    )

            sigma = max(cfg.sigma_min, sigma * cfg.sigma_decay)
            self.generation = gen + 1

            if self.on_generation:
                self.on_generation(GenerationResult(
                    generation   = gen + 1,
                    best_fitness = best_f,
                    mean_fitness = sum(all_f) / len(all_f),
                    sigma        = sigma,
                    weights      = self.weights[:],
                ))
