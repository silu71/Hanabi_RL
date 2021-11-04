from hanabi.objects import Card, Color, FailureTokensOnField
from hanabi.game_field import Colorfield
from typing import List

class Hanabifield:
    def __init__(self, colorfields: List[Colorfield]):
        self.colorfields = {field.color: field for field in colorfields}

    def is_able_to_add(self, card):
        return self.colorfields[card.color].is_able_to_add(card)


    def add_card(self, card):
        self.colorfields[card.color].add_card(card)
            
