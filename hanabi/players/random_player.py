from typing import List, Dict
from hanabi.actions import Action
import random

from .player import Player, PlayerObservation


class RandomPlayer(Player):
    def choose_action(self, valid_actions: List[Action], observation: PlayerObservation) -> Action:
        return random.choice(valid_actions)
