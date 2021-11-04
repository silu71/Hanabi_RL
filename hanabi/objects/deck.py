from typing import List
from .card import Card


class Deck:
    def __init__(self, cards: List[Card]):
        self.cards = cards
