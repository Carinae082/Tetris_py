"""
bot/search_node.py — 빔 서치 트리 노드 및 탐색 결과 정의
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engine.state import GameState
from .tactical_evaluator import CoachingState

if TYPE_CHECKING:
    from .placement_generator import FinalPlacement


@dataclass(slots=True)
class SearchNode:
    """
    빔 서치 트리의 단일 노드. slots=True로 메모리와 속성 접근 속도를 최적화한다.

    env              : 이 노드에서의 게임 환경 (배치 완료 후 다음 피스가 스폰된 상태)
    root_placement   : 루트에서 첫 번째로 적용한 FinalPlacement (최종 행동 선택에 사용)
    depth            : 루트로부터의 깊이 (1부터 시작)
    cumulative_score : 합성 점수 (board + 깊이 스케일된 attack/chain + context)
    immediate_score  : 이 노드의 합성 점수 (futility pruning에 사용)
    board_score      : evaluate_board 결과 (분리 저장)
    attack_score     : calculate_immediate_attack 결과 (분리 저장, quiescence 판단용)
    terminal         : 게임 종료 여부
    used_hold        : 이 경로에서 홀드를 사용했는지 여부
    lines_cleared    : 이 노드에서 클리어한 줄 수 (정적 탐색 판단에 사용)
    path_placements  : track_path=True 일 때 전체 경로; False 이면 [root_placement] 만 저장
    path_attack      : 루트에서 이 노드까지의 누적 공격력 (깊이 스케일링 전 원시값)
    path_chain       : 루트에서 이 노드까지의 누적 체인 가치 (깊이 스케일링 전 원시값)
    """

    env:               GameState
    root_placement:    Any           # FinalPlacement
    depth:             int
    cumulative_score:  float
    immediate_score:   float
    board_score:       float
    attack_score:      float
    terminal:          bool
    used_hold:         bool
    lines_cleared:     int          = 0
    path_placements:   list         = field(default_factory=list)
    path_attack:       float        = 0.0   # 누적 공격력 (raw)
    path_chain:        float        = 0.0   # 누적 체인 가치 (raw)
    coaching:          CoachingState = field(default_factory=CoachingState)  # 코칭 상태 (delta context 계산용)


@dataclass
class SearchResult:
    """
    BeamSearchBot.search()의 반환값.

    best_placement : 선택된 최선 FinalPlacement (도달 가능한 배치가 없으면 None)
    best_score     : 최선 노드의 cumulative_score
    root_scores    : 각 루트 FinalPlacement 에서 도달한 최고 cumulative_score
    expanded_nodes : 전체 탐색 중 확장(생성)된 노드 수
    searched_depth : 실제로 탐색한 최대 깊이
    best_path      : best_placement 에 해당하는 배치 경로
                     (track_path=False 면 [best_placement] 만 포함)
    """

    best_placement: Any | None    # FinalPlacement | None
    best_score:     float
    root_scores:    dict          # dict[FinalPlacement, float]
    expanded_nodes: int
    searched_depth: int
    best_path:      list          # list[FinalPlacement]
