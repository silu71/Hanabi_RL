from typing import List
from ..actions import Action

from .player import Player, PlayerObservation


class HumanPlayer(Player):
    def choose_action(self, valid_actions: List[Action], observation: PlayerObservation) -> Action:
        chosen_action = None
        print("Valid actions")
        print([str(a) for a in valid_actions])
        while chosen_action is None:
            action = input("Input action: ")
            for a in valid_actions:
                if str(a) == action:
                    chosen_action = a
            if chosen_action is None:
                print("Invalid action!")
        return chosen_action
