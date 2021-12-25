from typing import Optional
from ..objects.card import Card, InvalidCardError, Color, Rank


class HanabiTower:
    def __init__(self, color: Color, max_rank: Rank = Rank.FIVE):
        self.color = color
        self.cards = [Card(color, Rank.EMPTY)]
        self.max_rank = max_rank

    @property
    def rank(self) -> Rank:
        return self.top_card.rank

    @property
    def top_card(self) -> Optional[Card]:
        return self.cards[-1]

    def is_able_to_add(self, card: Card) -> bool:

        if card.rank.value > self.max_rank.value:
            raise InvalidCardError(
                f"{Card} should not exist in this game setting where the max rank is {self.max_rank}."
            )

        if self.color != card.color:
            return False

        if self.top_card.rank.value + 1 != card.rank.value:
            return False

        return True

    def add_card(self, card: Card):
        if not self.is_able_to_add(card):
            raise RuntimeError("card type is mistaken")
        self.cards.append(card)

    def is_completed(self) -> bool:
        return self.top_card.rank == self.max_rank

    def __str__(self):
        return f"{self.color.value}: {[card.rank.value for card in self.cards[1:]]}"
