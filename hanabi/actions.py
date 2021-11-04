from typing import List
from dataclasses import dataclass
from hanabi.objects import Color, Rank


class Action:
    pass


@dataclass
class PlayCard(Action):
    played_card_index: int


@dataclass
class GetHintToken(Action):
    discard_card_index: int


@dataclass
class GiveColorHint(Action):
    player_index: int
    color: Color


@dataclass
class GiveRankHint(Action):
    player_index: int
    rank: Rank
