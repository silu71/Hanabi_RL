from typing import Optional
from .card import Card, Color, Rank
import numpy as np


def get_num_cards(rank: int, max_rank: int) -> int:
    if rank == 1:
        return 3
    elif rank == max_rank:
        return 1
    else:
        return 2


class Deck:
    def __init__(
        self,
        max_rank: int = 5,
        num_colors: int = 5,
        np_random: Optional[np.random.Generator] = None,
    ):
        self.cards = []

        colors = Color.list(num_colors)
        ranks = Rank.list(max_rank)
        for color in colors:
            for rank in ranks:
                num_cards = get_num_cards(rank=rank.value, max_rank=max_rank)
                self.cards += [Card(color=color, rank=rank) for _ in range(num_cards)]

        # shuffle cards
        if np_random is None:
            np_random = np.random.default_rng()

        perm = np_random.permutation(len(self.cards))
        self.cards = [self.cards[p] for p in perm]

    def __len__(self):
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def get_card(self) -> Card:
        if self.is_empty():
            raise RuntimeError("The deck is empty!")
        return self.cards.pop(0)
