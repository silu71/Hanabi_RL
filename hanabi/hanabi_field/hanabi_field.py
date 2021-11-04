from typing import Dict
from hanabi.objects import Card, Color
from hanabi.hanabi_field.color_field import ColorField


class HanabiField:
    def __init__(self):
        self.color_fields: Dict[Color, ColorField] = {
            color: ColorField(color) for color in list(Color)
        }

    def is_able_to_add(self, card: Card) -> bool:
        return self.color_fields[card.color].is_able_to_add(card)

    def add_card(self, card: Card) -> bool:
        self.color_fields[card.color].add_card(card)
        return self.color_fields[card.color].is_completed()

    def is_completed(self) -> bool:
        completed_list = [
            color_field.is_completed() for color_field in self.color_fields.values()
        ]
        return all(completed_list)

    def get_score(self) -> int:
        total_score = 0
        for color_field in self.color_fields.values():
            total_score = color_field.top_card.rank.value
        return total_score

    def __str__(self):
        string = ""
        for color_field in self.color_fields.values():
            string += str(color_field) + "\n"
        return string
