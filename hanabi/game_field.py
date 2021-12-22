from typing import List, Dict
from dataclasses import dataclass
import logging

from hanabi.objects import (
    FailureTokensOnField,
    HintTokensOnField,
    Deck,
    Card,
    Color,
    Rank,
)
from hanabi.hanabi_field import HanabiField
from hanabi.players import Player, CardHint, PlayerObservation
from hanabi.actions import Action, PlayCard, GetHintToken, GiveColorHint, GiveRankHint


logger = logging.getLogger(__name__)


class InvalidActionError(Exception):
    pass


@dataclass
class FullState:
    deck_size: int
    player_hints: List[List[CardHint]]
    player_hands: List[List[Card]]
    num_failure_tokens: int
    num_hint_tokens: int
    tower_ranks: Dict[Color, Rank]
    current_player_id: int


class GameField:
    def __init__(self):

        self.deck = Deck()
        self.players: List[Player] = None
        self.failure_tokens = FailureTokensOnField()
        self.hint_tokens = HintTokensOnField(initial_num_hint_tokens=8, max_num_hint_tokens=8)
        self.hanabi_field = HanabiField()
        self.discard_pile: List[Card] = []

        self.turn = 0
        self.turn_since_deck_is_empty = 0

        self.current_player_id = None

    def reset(self):
        self.__init__()

    def distribute_cards(self, num_initial_cards: int = 5):
        for _ in range(num_initial_cards):
            for player in self.players:
                player.draw_card(self.deck.get_card())

    def get_valid_actions(self, player: Player) -> List[Action]:
        valid_actions = [PlayCard(card_index) for card_index in range(len(player.hand))]

        if self.hint_tokens.is_able_to_add_token() and len(player.hand) > 0:
            valid_actions += [GetHintToken(card_index) for card_index in range(len(player.hand))]

        if self.hint_tokens.is_able_to_use_token():

            for other_player in self.players:
                if other_player == player:
                    continue
                for color in list(Color):
                    if other_player.has_color(color):
                        valid_actions.append(GiveColorHint(player_index=other_player.index, color=color))
                for rank in list(Rank):
                    if other_player.has_rank(rank):
                        valid_actions.append(GiveRankHint(player_index=other_player.index, rank=rank))

        return valid_actions

    def is_terminal(self) -> bool:
        if self.failure_tokens.is_failed():
            return True

        if self.hanabi_field.is_completed():
            return True

        if self.turn_since_deck_is_empty == len(self.players):
            return True

        return False

    def get_current_full_state(self) -> FullState:
        return FullState(
            deck_size=len(self.deck),
            player_hints=[p.card_hints for p in self.players],
            player_hands=[p.hand for p in self.players],
            num_failure_tokens=self.failure_tokens.num_failure_tokens,
            num_hint_tokens=len(self.hint_tokens),
            tower_ranks={color: hanabi_tower.rank for color, hanabi_tower in self.hanabi_field.hanabi_towers.items()},
            current_player_id=self.current_player_id,
        )

    def get_current_player_observation(self) -> PlayerObservation:
        full_states = self.get_current_full_state()

        i = full_states.current_player_id
        return PlayerObservation(
            deck_size=full_states.deck_size,
            # Note that the index in this list is relative to current_player_id
            other_player_hints=full_states.player_hints[i + 1 :] + full_states.player_hints[:i],
            other_player_hands=full_states.player_hands[i + 1 :] + full_states.player_hands[:i],
            current_player_hints=full_states.player_hints[i],
            num_failure_tokens=full_states.num_failure_tokens,
            num_hint_tokens=full_states.num_hint_tokens,
            tower_ranks=full_states.tower_ranks,
            current_player_id=full_states.current_player_id,
        )

    def receive_action(self, player: Player, action: Action):
        if isinstance(action, PlayCard):
            card = player.use_card(action.played_card_index)
            if self.hanabi_field.is_able_to_add(card):
                color_field_is_completed = self.hanabi_field.add_card(card)
                if color_field_is_completed and self.hint_tokens.is_able_to_add_token():
                    self.hint_tokens.add_token()
            else:
                self.failure_tokens.add_token()
                self.discard_pile.append(card)
            player.draw_card(self.deck.get_card())

        elif isinstance(action, GetHintToken):
            if not self.hint_tokens.is_able_to_add_token():
                raise InvalidActionError("The number of hint tokens has reached the max.")

            discarded_card = player.use_card(action.discard_card_index)
            self.discard_pile.append(discarded_card)
            self.hint_tokens.add_token()
            player.draw_card(self.deck.get_card())
        elif isinstance(action, GiveColorHint):
            if action.player_index == self.current_player_id:
                raise InvalidActionError("You cannot give the hint to yourself.")

            if not self.hint_tokens.is_able_to_use_token():
                raise InvalidActionError("The number of hint tokens is empty.")

            self.hint_tokens.use_token()
            self.players[action.player_index].get_color_hint(action.color)
        elif isinstance(action, GiveRankHint):
            if action.player_index == self.current_player_id:
                raise InvalidActionError("You cannot give the hint to yourself.")
            if not self.hint_tokens.is_able_to_use_token():
                raise InvalidActionError("The number of hint tokens is empty.")
            self.hint_tokens.use_token()
            self.players[action.player_index].get_rank_hint(action.rank)
        else:
            raise InvalidActionError(f"Invalid action: {action}")

    def setup_game(self, players: List[Player]):
        self.reset()
        self.players = players
        self.distribute_cards()
        self.current_player_id = 0

    def auto_play(self):

        logging.info(self)
        max_num_rounds = (len(self.deck) + len(self.hint_tokens) + 1) // len(self.players) + 2

        for current_round in range(max_num_rounds):
            for current_player_id, player in enumerate(self.players):
                self.current_player_id = current_player_id

                valid_actions = self.get_valid_actions(player)
                action = player.choose_action(valid_actions, self.get_current_player_observation())
                logging.info(f"Player {current_player_id}'s action:", action)
                self.receive_action(player=player, action=action)

                self.turn += 1
                self.turn_since_deck_is_empty += int(self.deck.is_empty())

                logging.info(self)

                if self.is_terminal():
                    return self.hanabi_field.get_score()

    def __str__(self):
        string = ""
        string += "==============================\n"
        string += f"Deck: {len(self.deck)}" + "\n"
        string += f"Hint Tokens: [" + "○" * len(self.hint_tokens) + "]\n"
        string += f"Failure Tokens: [" + "●" * len(self.failure_tokens) + "]\n"
        string += "\n"

        string += "Hanabi Field:" + "\n"
        string += str(self.hanabi_field) + "\n"

        string += "Hand: \n"
        for player in self.players:
            string += f"Player {player.index}: \n"
            string += str([str(c) for c in player.hand]) + "\n"
            string += "\n"
        string += "==============================\n"

        return string
