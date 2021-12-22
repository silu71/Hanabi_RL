from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering


class InvalidCardError(Exception):
    pass


class Color(Enum):
    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3
    WHITE = 4

    def __str__(self):
        return self.name[0]


@total_ordering
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

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value > other.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return str(self.value)


@dataclass
class Card:
    color: Color
    rank: Rank

    def __str__(self):
        return f"{self.color}{self.rank}"
