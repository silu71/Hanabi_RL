from typing import List, Optional, Dict
from dataclasses import dataclass

from hanabi.objects import Card, Color, Rank
from hanabi.actions import Action


@dataclass
class CardHint:
    color: Optional[Color] = None
    rank: Optional[Rank] = None


@dataclass
class PlayerObservation:
    deck_size: int
    other_player_hints: List[List[CardHint]]
    other_player_hands: List[List[Card]]
    current_player_hints: List[CardHint]
    num_failure_tokens: int
    num_hint_tokens: int
    tower_ranks: Dict[Color, Rank]
    discard_pile: List[Card]
    current_player_id: int


class Player:
    def __init__(self):
        self.hand: List[Card] = []
        self.card_hints: List[CardHint] = []

    def draw_card(self, card: Card):
        self.hand.append(card)
        self.card_hints.append(CardHint())

    def use_card(self, index: int) -> Card:
        if len(self.hand) <= index:
            raise RuntimeError()
        used_card = self.hand.pop(index)
        self.card_hints.pop(index)
        return used_card

    def has_color(self, color: Color) -> bool:
        return any([card.color == color for card in self.hand])

    def has_rank(self, rank: Rank) -> bool:
        return any([card.rank == rank for card in self.hand])

    def get_color_hint(self, color: Color):
        assert self.has_color(color)
        for card, observation in zip(self.hand, self.card_hints):
            if card.color == color:
                observation.color = color

    def get_rank_hint(self, rank: Rank):
        assert self.has_rank(rank)
        for card, observation in zip(self.hand, self.card_hints):
            if card.rank == rank:
                observation.rank = rank

    def choose_action(self, valid_actions: List[Action], observation: PlayerObservation) -> Action:
        raise NotImplementedError
