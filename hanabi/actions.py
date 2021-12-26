from dataclasses import dataclass
from .objects import Color, Rank


class Action:
    pass


@dataclass
class PlayCard(Action):
    played_card_index: int

    def __str__(self):
        return f"PlayCard({self.played_card_index})"


@dataclass
class GetHintToken(Action):
    discard_card_index: int

    def __str__(self):
        return f"GetHintToken({self.discard_card_index})"


@dataclass
class GiveColorHint(Action):
    player_index: int
    color: Color

    def __str__(self):
        return f"GiveColorHint({self.player_index}, {self.color})"


@dataclass
class GiveRankHint(Action):
    player_index: int
    rank: Rank

    def __str__(self):
        return f"RankHintTo({self.player_index}, {self.rank})"
