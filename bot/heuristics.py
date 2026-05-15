"""
bot/heuristics.py — 스냅샷 기반 경량 평가 유틸리티

규칙
----
* 게임 규칙을 다시 구현하지 않는다.
* StateSnapshot만 입력으로 받으며, 환경 원본 객체에 접근하지 않는다.
* 각 함수는 독립적으로 사용 가능한 순수 함수다.
"""

from __future__ import annotations

from engine.board import BOARD_COLS, BOARD_ROWS

from .snapshot import StateSnapshot


def board_max_height(snapshot: StateSnapshot) -> int:
    """
    보드에서 가장 높은 블록의 높이를 반환한다.
    높이 = 바닥(row 39)부터 가장 높은 블록(row가 낮을수록 높음)까지의 셀 수.
    보드가 완전히 비어 있으면 0을 반환한다.
    """
    board = snapshot.board
    for row in range(BOARD_ROWS):          # 위(row 0)에서 아래(row 39)로 탐색
        if any(cell is not None for cell in board[row]):
            return BOARD_ROWS - row        # 해당 row에서 바닥까지의 거리
    return 0


def count_holes(snapshot: StateSnapshot) -> int:
    """
    보드에서 홀(구멍) 수를 센다.
    홀 = 해당 열에서 블록보다 아래에 있는 빈 셀.
    (위쪽에 블록이 하나라도 있으면 아래의 빈 셀은 홀로 간주)
    """
    board = snapshot.board
    holes = 0
    for col in range(BOARD_COLS):
        block_seen = False
        for row in range(BOARD_ROWS):     # 위에서 아래로
            if board[row][col] is not None:
                block_seen = True
            elif block_seen:              # 블록 아래의 빈 셀
                holes += 1
    return holes


def column_heights(snapshot: StateSnapshot) -> list[int]:
    """
    각 열의 높이(바닥부터 가장 높은 블록까지의 셀 수)를 10개 리스트로 반환한다.
    빈 열의 높이는 0이다.
    """
    board = snapshot.board
    heights: list[int] = []
    for col in range(BOARD_COLS):
        height = 0
        for row in range(BOARD_ROWS):    # 위에서 아래로 탐색
            if board[row][col] is not None:
                height = BOARD_ROWS - row
                break
        heights.append(height)
    return heights


def bumpiness(snapshot: StateSnapshot) -> int:
    """
    인접한 열 사이의 높이 차이 합계를 반환한다.
    값이 클수록 보드 표면이 울퉁불퉁하다.
    """
    heights = column_heights(snapshot)
    return sum(abs(heights[i] - heights[i + 1]) for i in range(BOARD_COLS - 1))
