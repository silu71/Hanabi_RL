from typing import List, Optional, Dict
from dataclasses import dataclass

from ..objects import Card, Color, Rank
from ..actions import Action


class CardKnowledge:
    def __init__(self, max_rank, num_colors):
        self._color_list = Color.list(num_colors)
        self._rank_list = Rank.list(max_rank)
        self.color_possibilities = {c: True for c in self._color_list}
        self.rank_possibilities = {r: True for r in self._rank_list}

    def get_color_hint(self, positive: bool, color: Color):
        if positive:
            # the color of this card is ***
            for c in self._color_list:
                if c != color:
                    self.color_possibilities[c] = False
        else:
            self.color_possibilities[color] = False

    def get_rank_hint(self, positive: bool, rank: Rank):
        if positive:
            # the rank of this card is ***
            for r in self._rank_list:
                if r != rank:
                    self.rank_possibilities[r] = False
        else:
            # the rank of this card is not ***
            self.rank_possibilities[rank] = False

    def __str__(self):
        string = ""

        for color, possible in self.color_possibilities.items():
            if possible:
                string += str(color)

        for rank, possible in self.rank_possibilities.items():
            if possible:
                string += str(rank)

        return string


@dataclass
class PlayerObservation:
    deck_size: int
    other_player_knowledges: List[List[CardKnowledge]]
    other_player_hands: List[List[Card]]
    player_knowledge: List[CardKnowledge]
    num_failure_tokens: int
    num_hint_tokens: int
    tower_ranks: Dict[Color, Rank]
    discard_pile: List[Card]
    current_player_id: int

    def __str__(self):
        string = ""
        string += f"deck_size: {self.deck_size}\n"
        string += "other_player_knowledges\n"
        for index, knowledge in enumerate(self.other_player_knowledges):
            string += f"player {index}: {[str(ck) for ck in knowledge]}\n"
        string += "other_player_hands\n"
        for index, hand in enumerate(self.other_player_hands):
            string += f"player {index}: {[str(c) for c in hand]}\n"
        string += f"player_knowledge: {[str(ck) for ck in self.player_knowledge]}\n"
        string += f"num_failure_tokens: {self.num_failure_tokens}\n"
        string += f"num_hint_tokens: {self.num_hint_tokens}\n"
        string += "tower_ranks\n"
        for color, rank in self.tower_ranks.items():
            string += f"{color}: {str(rank)}\n"
        string += f"discard_pile: {[str(c) for c in self.discard_pile]}\n"
        string += f"current_player_id: {self.current_player_id}"
        return string


class Player:
    def __init__(self):
        self.hand: List[Card] = []
        self.card_knowledges: List[CardKnowledge] = []
        self.max_rank = None
        self.num_colors = None

    def notify_game_info(self, max_rank: int, num_colors: int):
        self.max_rank = max_rank
        self.num_colors = num_colors

    def draw_card(self, card: Card):
        self.hand.append(card)
        self.card_knowledges.append(CardKnowledge(self.max_rank, self.num_colors))

    def use_card(self, index: int) -> Card:
        if len(self.hand) <= index:
            raise RuntimeError()
        used_card = self.hand.pop(index)
        self.card_knowledges.pop(index)
        return used_card

    def has_color(self, color: Color) -> bool:
        return any([card.color == color for card in self.hand])

    def has_rank(self, rank: Rank) -> bool:
        return any([card.rank == rank for card in self.hand])

    def get_color_hint(self, color: Color):
        for card, knowledge in zip(self.hand, self.card_knowledges):
            positive = (card.color == color)
            knowledge.get_color_hint(positive, color)

    def get_rank_hint(self, rank: Rank):
        for card, knowledge in zip(self.hand, self.card_knowledges):
            positive = (card.rank == rank)
            knowledge.get_rank_hint(positive, rank)

    def choose_action(self, valid_actions: List[Action], observation: PlayerObservation) -> Action:
        raise NotImplementedError
