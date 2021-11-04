from typing import List
from dataclasses import dataclass
from hanabi.objects import Color, Rank


class Action:
    pass


@dataclass
class PlayCard(Action):
    played_card_index: int

    def __str__(self):
        return f"Play{self.played_card_index}"


@dataclass
class GetHintToken(Action):
    discard_card_index: int

    def __str__(self):
        return f"GetHint{self.discard_card_index}"


@dataclass
class GiveColorHint(Action):
    player_index: int
    color: Color

    def __str__(self):
        return f"ColorHintTo{self.player_index}For{self.color.value}"


@dataclass
class GiveRankHint(Action):
    player_index: int
    rank: Rank

    def __str__(self):
        return f"RankHintTo{self.player_index}For{self.rank.value}"
