from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class MinoType(Enum):
    I = "I"
    J = "J"
    L = "L"
    O = "O"
    S = "S"
    T = "T"
    Z = "Z"
    GARBAGE = "G"  # 가비지 블록 식별용


# SRS 회전 좌표: SHAPES[type][rotation] = [(row_offset, col_offset), ...]
# 기준점: 바운딩 박스의 (0,0) 좌상단
SHAPES: dict[MinoType, list[list[tuple[int, int]]]] = {
    MinoType.I: [
        [(1, 0), (1, 1), (1, 2), (1, 3)],  # 0 (스폰)
        [(0, 2), (1, 2), (2, 2), (3, 2)],  # 1 (CW)
        [(2, 0), (2, 1), (2, 2), (2, 3)],  # 2 (180)
        [(0, 1), (1, 1), (2, 1), (3, 1)],  # 3 (CCW)
    ],
    MinoType.J: [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    MinoType.L: [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
    MinoType.O: [
        [(0, 1), (0, 2), (1, 1), (1, 2)],  # O는 4회전 동일
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (1, 2)],
    ],
    MinoType.S: [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 1), (1, 2), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    MinoType.T: [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    MinoType.Z: [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
}

# 스폰 위치: 바운딩 박스 (row, col) 기준
# 10×40 보드에서 row 0~19 = 버퍼(숨김), row 20~39 = 표시 영역
# I는 4×4 바운딩 박스라 row=18에서 실제 셀은 row 19에 위치
SPAWN: dict[MinoType, tuple[int, int]] = {
    MinoType.I: (18, 3),
    MinoType.J: (19, 3),
    MinoType.L: (19, 3),
    MinoType.O: (19, 3),
    MinoType.S: (19, 3),
    MinoType.T: (19, 3),
    MinoType.Z: (19, 3),
}


@dataclass
class ActivePiece:
    type: MinoType
    row: int       # 바운딩 박스 좌상단 row (보드 절대 좌표)
    col: int       # 바운딩 박스 좌상단 col (보드 절대 좌표)
    rotation: int  # 0=스폰, 1=CW, 2=180, 3=CCW

    def cells(self) -> List[Tuple[int, int]]:
        """현재 회전 상태에서 점유하는 셀의 절대 (row, col) 목록."""
        return [
            (self.row + dr, self.col + dc)
            for dr, dc in SHAPES[self.type][self.rotation]
        ]

    @classmethod
    def spawn(cls, mino_type: MinoType) -> "ActivePiece":
        row, col = SPAWN[mino_type]
        return cls(type=mino_type, row=row, col=col, rotation=0)

    def __repr__(self) -> str:
        return f"ActivePiece({self.type.value} rot={self.rotation} @({self.row},{self.col}))"
