from __future__ import annotations

from typing import Optional

from .mino import ActivePiece, MinoType

BOARD_COLS = 10
BOARD_ROWS = 40
VISIBLE_ROW_START = 20  # row 20~39 = 표시 영역, row 0~19 = 버퍼(숨김)

# None = 빈 셀, MinoType = 고정된 블록 (GARBAGE 포함)
Cell = Optional[MinoType]


# ── Zobrist 해시 키 (SplitMix64, fusion/transposition.rs 와 동일 시드) ──────────
_MASK64: int = (1 << 64) - 1
_ZOBRIST_SEED: int = 0x9E3779B97F4A7C15


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, z ^ (z >> 31)


def _generate_zobrist_keys() -> list[list[int]]:
    """keys[col][row] — BOARD_COLS × BOARD_ROWS 개의 64비트 난수."""
    state = _ZOBRIST_SEED
    keys: list[list[int]] = []
    for _ in range(BOARD_COLS):
        row_keys: list[int] = []
        for _ in range(BOARD_ROWS):
            state, val = _splitmix64(state)
            row_keys.append(val)
        keys.append(row_keys)
    return keys


# 모듈 로드 시 1회 생성 후 재사용
_ZOBRIST_KEYS: list[list[int]] = _generate_zobrist_keys()


def _compute_hash(grid: list[list]) -> int:
    h = 0
    keys = _ZOBRIST_KEYS
    for row in range(BOARD_ROWS):
        grid_row = grid[row]
        for col in range(BOARD_COLS):
            if grid_row[col] is not None:
                h ^= keys[col][row]
    return h


def _compute_col_heights(grid: list[list]) -> list[int]:
    heights = [0] * BOARD_COLS
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS):
            if grid[row][col] is not None:
                heights[col] = BOARD_ROWS - row
                break
    return heights


class Board:
    """
    10×40 그리드.
    - row 0 : 최상단 (버퍼)
    - row 39 : 최하단 (바닥)
    행이 증가할수록 아래쪽이다.
    """

    def __init__(self) -> None:
        self._grid: list[list[Cell]] = [
            [None] * BOARD_COLS for _ in range(BOARD_ROWS)
        ]
        # Zobrist 해시 캐시 (빈 보드 = 0)
        self._hash: int = 0
        self._hash_dirty: bool = False
        # 열 높이 캐시
        self._col_heights: list[int] = [0] * BOARD_COLS
        self._heights_dirty: bool = False

    @property
    def zobrist_hash(self) -> int:
        """보드 상태의 Zobrist 해시. 동일 보드면 항상 같은 값."""
        if self._hash_dirty:
            self._hash = _compute_hash(self._grid)
            self._hash_dirty = False
        return self._hash

    @property
    def col_heights(self) -> list[int]:
        """각 열의 높이 (캐시됨). 바닥에서 가장 높은 블록까지의 셀 수."""
        if self._heights_dirty:
            self._col_heights = _compute_col_heights(self._grid)
            self._heights_dirty = False
        return self._col_heights

    # ------------------------------------------------------------------
    # 읽기
    # ------------------------------------------------------------------

    def get(self, row: int, col: int) -> Cell:
        return self._grid[row][col]

    def is_in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS

    def is_cell_free(self, row: int, col: int) -> bool:
        return self.is_in_bounds(row, col) and self._grid[row][col] is None

    def is_valid_position(self, piece: ActivePiece) -> bool:
        """piece의 모든 셀이 범위 내에 있고 빈 공간인지 확인한다."""
        return all(self.is_cell_free(r, c) for r, c in piece.cells())

    # ------------------------------------------------------------------
    # 쓰기
    # ------------------------------------------------------------------

    def lock(self, piece: ActivePiece) -> None:
        """활성 미노를 보드에 고정한다."""
        keys = _ZOBRIST_KEYS
        for r, c in piece.cells():
            self._grid[r][c] = piece.type
            # 해시 증분 갱신 (dirty이면 스킵 — 나중에 전체 재계산)
            if not self._hash_dirty:
                self._hash ^= keys[c][r]
            # 높이 증분 갱신
            if not self._heights_dirty:
                new_h = BOARD_ROWS - r
                if new_h > self._col_heights[c]:
                    self._col_heights[c] = new_h

    def clear_lines(self) -> tuple[int, bool]:
        """
        완성된 줄을 제거하고, 빈 줄을 상단에 추가한다.
        반환값: (제거된 줄 수, 가비지 행이 포함됐는지 여부).
        """
        full_rows = [r for r in range(BOARD_ROWS) if all(self._grid[r])]
        clears_garbage = any(MinoType.GARBAGE in self._grid[r] for r in full_rows)
        for r in full_rows:
            del self._grid[r]
            self._grid.insert(0, [None] * BOARD_COLS)
        if full_rows:
            self._hash_dirty = True
            self._heights_dirty = True
        return len(full_rows), clears_garbage

    def add_garbage(self, lines: int, hole_col: int) -> None:
        """
        가비지 줄을 아래에서 삽입한다.
        - 기존 블록은 위로 밀린다 (최상단 행은 소멸).
        - hole_col 위치만 빈칸, 나머지는 GARBAGE로 채운다.
        """
        for _ in range(lines):
            garbage_row: list[Cell] = [MinoType.GARBAGE] * BOARD_COLS
            garbage_row[hole_col] = None
            self._grid.pop(0)   # 최상단 행 제거 (위로 밀기 효과)
            self._grid.append(garbage_row)
        if lines > 0:
            self._hash_dirty = True
            self._heights_dirty = True

    # ------------------------------------------------------------------
    # 디버그
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines: list[str] = []
        for r in range(VISIBLE_ROW_START, BOARD_ROWS):
            row_str = "".join(
                "." if cell is None else cell.value
                for cell in self._grid[r]
            )
            lines.append(f"{r:2d}|{row_str}|")
        return "\n".join(lines)
