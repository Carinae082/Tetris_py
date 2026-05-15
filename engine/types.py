"""
engine/types.py — 공유 열거형 (순환 임포트 방지용 분리)
"""
from enum import Enum


class SpinType(Enum):
    NONE = "none"
    MINI = "mini"
    FULL = "full"
