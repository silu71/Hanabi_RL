from hanabi.objects import Card
from hanabi.game_field.color_field import ColorField
from typing import List


class HanabiField:
    def __init__(self, color_fields: List[ColorField]):
        self.color_fields = {field.color: field for field in color_fields}

    def is_able_to_add(self, card: Card) -> bool:
        return self.color_fields[card.color].is_able_to_add(card)

    def add_card(self, card: Card):
        self.color_fields[card.color].add_card(card)
