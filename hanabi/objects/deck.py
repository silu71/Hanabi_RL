from typing import List
import random
from .card import Card, Color, Rank


class Deck:
    def __init__(self, max_rank: int = 5, colors: List[Color] = None):
        self.cards = []

        colors = colors or list(Color)
        for color in colors:

            for rank, num_cards in [(Rank.ONE, 3), (Rank.TWO, 2), (Rank.THREE, 2), (Rank.FOUR, 2), (Rank.FIVE, 1)][
                :max_rank
            ]:
                self.cards += [Card(color=color, rank=rank) for _ in range(num_cards)]

        random.shuffle(self.cards)

    def __len__(self):
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def get_card(self) -> Card:
        if self.is_empty():
            raise RuntimeError("The deck is empty!")
        return self.cards.pop(0)
