from typing import List
from hanabi.actions import Action
import random

from .player import Player


class RandomPlayer(Player):
    def choose_action(self, valid_actions: List[Action]) -> Action:
        return random.choice(valid_actions)
