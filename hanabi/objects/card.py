from dataclasses import dataclass
from enum import Enum, auto


class Color(Enum):
    RED = auto()
    BLUE = auto()
    YELLOW = auto()
    GREEN = auto()
    WHITE = auto()


class Rank(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5


@dataclass
class Card:
    color: Color
    rank: Rank
