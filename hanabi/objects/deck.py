from typing import List
import random
from .card import Card, Color, Rank

DEFAULT_NUM_CARDS = {Rank.ONE: 3, Rank.TWO: 2, Rank.THREE: 2, Rank.FOUR: 2, Rank.FIVE: 1}


class Deck:
    def __init__(self, max_rank: int = 5, num_colors: int = 5):
        self.cards = []

        colors = Color.list(num_colors)
        ranks = Rank.list(max_rank)
        for color in colors:

            for rank in ranks:
                self.cards += [Card(color=color, rank=rank) for _ in range(DEFAULT_NUM_CARDS[rank])]

        random.shuffle(self.cards)

    def __len__(self):
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def get_card(self) -> Card:
        if self.is_empty():
            raise RuntimeError("The deck is empty!")
        return self.cards.pop(0)
