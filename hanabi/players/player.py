from typing import List, Optional, Dict
from dataclasses import dataclass

from ..objects import Card, Color, Rank
from ..actions import Action


# @dataclass
# class CardHint:
#     color: Optional[Color] = None
#     rank: Optional[Rank] = None

# @dataclass
# class CardKnowledge:
#     color_possibilities: Optional[Color] = None

class CardKnowledge:
    def __init__(self):
        self.color_possibilities = {c: True for c in list(Color)}
        # exclude EMPTY rank
        self.rank_possibilities = {r: True for r in list(Rank)[1:]}

    def get_color_hint(self, positive: bool, color: Optional[Color]):
        if positive:
            # the color of this card is ***
            if color is not None:
                for c in list(Color):
                    if c != color:
                        self.color_possibilities[c] = False
        else:
            # the color of this card is not ***
            if color is not None:
                self.color_possibilities[color] = False

    def get_rank_hint(self, positive: bool, rank: Optional[int] = None):
        if positive:
            # the rank of this card is ***
            if rank is not None:
                for r in list(Rank)[1:]:
                    if r != rank:
                        self.rank_possibilities[r] = 0
        else:
            # the rank of this card is not ***
            if rank is not None:
                self.rank_possibilities[rank] = 0
    
    def __repr__(self):
        return f"CardKnowlege(color={self.color_possibilities}, rank={self.rank_possibilities}"
        # [self.color_possibilities[c] for c in list(Color)]
    
    # def must_be_color(self, color: Color):
    #     for c in list(Color):
    #         if (
    #             (c == color and not self.color_possibilities[c]) or
    #             (c != color and self.color_possibilities[c])
    #         ):
    #             return False

    #     return True

    # def must_be_rank(self, rank: Rank):
    #     for r in list(Rank):
    #         if (
    #             (r == rank and not self.rank_possibilities[r]) or
    #             (r != rank and self.rank_possibilities[r])
    #         ):
    #             return False

    #     return True


@dataclass
class PlayerObservation:
    deck_size: int
    other_player_knowledges: List[List[CardKnowledge]]
    other_player_hands: List[List[Card]]
    current_player_knowledges: List[CardKnowledge]
    num_failure_tokens: int
    num_hint_tokens: int
    tower_ranks: Dict[Color, Rank]
    discard_pile: List[Card]
    current_player_id: int


class Player:
    def __init__(self):
        self.hand: List[Card] = []
        self.card_knowledges: List[CardKnowledge] = []

    def draw_card(self, card: Card):
        self.hand.append(card)
        self.card_knowledges.append(CardKnowledge())

    def use_card(self, index: int) -> Card:
        if len(self.hand) <= index:
            raise RuntimeError()
        used_card = self.hand.pop(index)
        self.card_knowledges.pop(index)
        return used_card

    def has_color(self, color: Color) -> bool:
        return any([card.color == color for card in self.hand])

    def has_rank(self, rank: Rank) -> bool:
        return any([card.rank == rank for card in self.hand])

    def get_color_hint(self, color: Color):
        for card, knowledge in zip(self.hand, self.card_knowledges):
            positive = (card.color == color)
            knowledge.get_color_hint(positive, color)

    def get_rank_hint(self, rank: Rank):
        for card, knowledge in zip(self.hand, self.card_knowledges):
            positive = (card.rank == rank)
            knowledge.get_rank_hint(positive, rank)

    def choose_action(self, valid_actions: List[Action], observation: PlayerObservation) -> Action:
        raise NotImplementedError
