import random
from .card import Card, Color, Rank


class Deck:
    def __init__(self):
        self.cards = []

        for color in Color:
            self.cards += [Card(color=color, rank=Rank.ONE) for _ in range(3)]
            self.cards += [Card(color=color, rank=Rank.TWO) for _ in range(2)]
            self.cards += [Card(color=color, rank=Rank.THREE) for _ in range(2)]
            self.cards += [Card(color=color, rank=Rank.FOUR) for _ in range(2)]
            self.cards += [Card(color=color, rank=Rank.FIVE) for _ in range(1)]

        random.shuffle(self.cards)

    def __len__(self):
        return len(self.cards)

    def is_empty(self) -> bool:
        return len(self.cards) == 0

    def get_card(self) -> Card:
        if self.is_empty():
            raise RuntimeError("The deck is empty!")
        return self.cards.pop(0)
