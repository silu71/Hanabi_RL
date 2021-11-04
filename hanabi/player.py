from typing import List, Optional
from dataclasses import dataclass
from hanabi.objects import Card, Deck, Color, Rank
from hanabi.hanabi_field import HanabiField

from .actions import Action


@dataclass
class CardObservation:
    color_hint: Optional[Color] = None
    rank_hint: Optional[Rank] = None


class Player:
    def __init__(self, index: int):
        self.index = index
        self.hand: List[Card] = []
        self.card_observation: List[CardObservation] = []

    def draw_card(self, deck: Deck):
        self.hand.append(deck.get_card())
        self.card_observation.append(CardObservation())

    def use_card(self, index: int) -> Card:
        if len(self.hand) <= index:
            raise RuntimeError()
        used_card = self.hand.pop(index)
        self.card_observation.pop(index)
        return used_card

    def has_color(self, color: Color) -> bool:
        return any([card.color == color for card in self.hand])

    def has_rank(self, rank: Rank) -> bool:
        return any([card.rank == rank for card in self.hand])

    def get_color_hint(self, color: Color):
        assert self.has_color(color)
        for card, observation in zip(self.hand, self.card_observation):
            if card.color == color:
                observation.color_hint = color

    def get_rank_hint(self, rank: Rank):
        assert self.has_rank(rank)
        for card, observation in zip(self.hand, self.card_observation):
            if card.rank == rank:
                observation.rank_hint = rank

    def choose_action(self, valid_action: List[Action]) -> Action:
        raise NotImplementedError
