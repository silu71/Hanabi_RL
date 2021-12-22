from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto


class Color(Enum):
    RED = "R"
    BLUE = "B"
    YELLOW = "Y"
    GREEN = "G"
    WHITE = "W"


class Rank(Enum):
    EMPTY = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    @property
    def next_rank(self) -> Optional["Rank"]:
        if self == Rank.FIVE:
            return None
        return list(Rank)[self.value + 1]


@dataclass
class Card:
    color: Color
    rank: Rank

    def __str__(self):
        return f"{self.color.value}{self.rank.value}"
