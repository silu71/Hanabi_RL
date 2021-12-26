from typing import Dict, List
from ..objects import Card, Color, Rank
from .hanabi_tower import HanabiTower


class HanabiField:
    def __init__(self, max_rank: int = 5, num_colors: int = 5):
        colors = Color.list(num_colors)
        self.hanabi_towers: Dict[Color, HanabiTower] = {
            color: HanabiTower(color, max_rank=Rank(max_rank)) for color in colors
        }

    def is_able_to_add(self, card: Card) -> bool:
        return self.hanabi_towers[card.color].is_able_to_add(card)

    def add_card(self, card: Card) -> bool:
        self.hanabi_towers[card.color].add_card(card)
        return self.hanabi_towers[card.color].is_completed()

    def is_completed(self) -> bool:
        completed_list = [hanabi_tower.is_completed() for hanabi_tower in self.hanabi_towers.values()]
        return all(completed_list)

    def get_score(self) -> int:
        total_score = 0
        for hanabi_tower in self.hanabi_towers.values():
            total_score += hanabi_tower.rank.value
        return total_score

    def __str__(self):
        string = ""
        for hanabi_tower in self.hanabi_towers.values():
            string += str(hanabi_tower) + "\n"
        return string
