from typing import List, Dict
from dataclasses import dataclass
import logging
import numpy as np

from .objects import (
    FailureTokensOnField,
    HintTokensOnField,
    Deck,
    Card,
    Color,
    Rank,
)
from .hanabi_field import HanabiField
from .players import Player, CardKnowledge, PlayerObservation
from .actions import Action, PlayCard, GetHintToken, GiveColorHint, GiveRankHint


logger = logging.getLogger(__name__)


class InvalidActionError(Exception):
    pass


@dataclass
class FullState:
    deck_size: int
    player_knowledges: List[List[CardKnowledge]]
    player_hands: List[List[Card]]
    num_failure_tokens: int
    num_hint_tokens: int
    tower_ranks: Dict[Color, Rank]
    discard_pile: List[Card]
    current_player_id: int


def abs_to_rel_player_index(player_index: int, other_player_index: int, num_players: int) -> int:
    return (other_player_index - player_index - 1) % num_players


def rel_to_abs_player_index(player_index: int, relative_other_index: int, num_players: int) -> int:
    return (relative_other_index + player_index + 1) % num_players


class GameEngine:
    def __init__(
        self,
        num_initial_cards: int = 5,
        num_initial_hint_tokens: int = 8,
        num_max_hint_tokens: int = 8,
        max_num_failure_tokens: int = 3,
        max_rank: int = 5,
        num_colors: int = 5,
    ):

        self.num_initial_cards = num_initial_cards
        self.num_initial_hint_tokens = num_initial_hint_tokens
        self.num_max_hint_tokens = num_max_hint_tokens
        self.max_num_failure_tokens = max_num_failure_tokens
        self.max_rank = max_rank
        self.num_colors = num_colors

        self.deck: Deck = None
        self.hanabi_field: HanabiField = None
        self.failure_tokens: FailureTokensOnField = None
        self.hint_tokens: HintTokensOnField = None
        self.discard_pile: List[Card] = None
        self.players: List[Player] = None
        self.current_player_id: int = None
        self.turn_since_deck_is_empty: int = None
        self._prev_action_info: tuple = None
        self.np_random: np.random.Generator = None

    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_id]

    def seed(self, seed: int):
        self.np_random = np.random.default_rng(seed)

    def reset(self):
        self.deck = Deck(max_rank=self.max_rank, num_colors=self.num_colors, np_random=self.np_random)
        self.hanabi_field = HanabiField(max_rank=self.max_rank, num_colors=self.num_colors)
        self.failure_tokens = FailureTokensOnField(self.max_num_failure_tokens)
        self.hint_tokens = HintTokensOnField(self.num_initial_hint_tokens, self.num_max_hint_tokens)
        self.discard_pile = []
        self.current_player_id = 0
        self.turn_since_deck_is_empty = 0
        self._prev_action_info = None

    def setup_game(self, players: List[Player]):
        self.reset()
        self.players = players
        for player in players:
            player.notify_game_info(self.max_rank, self.num_colors)

        self.distribute_cards()

    def distribute_cards(self):
        for _ in range(self.num_initial_cards):
            for player in self.players:
                player.draw_card(self.deck.get_card())

    def get_valid_actions(self, num_current_player_cards: int, current_player_index: int) -> List[Action]:
        valid_actions = [PlayCard(card_index) for card_index in range(num_current_player_cards)]

        if self.hint_tokens.is_able_to_add_token() and num_current_player_cards > 0:
            valid_actions += [GetHintToken(card_index) for card_index in range(num_current_player_cards)]

        if self.hint_tokens.is_able_to_use_token():

            for other_index, other_player in enumerate(self.players):
                if other_index == current_player_index:
                    continue
                relative_other_index = abs_to_rel_player_index(
                    player_index=current_player_index,
                    other_player_index=other_index,
                    num_players=len(self.players),
                )

                for color in Color.list(self.num_colors):
                    if other_player.has_color(color):
                        valid_actions.append(GiveColorHint(player_index=relative_other_index, color=color))
                for rank in Rank.list(self.max_rank):
                    if other_player.has_rank(rank):
                        valid_actions.append(GiveRankHint(player_index=relative_other_index, rank=rank))

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
            player_knowledges=[p.card_knowledges for p in self.players],
            player_hands=[p.hand for p in self.players],
            num_failure_tokens=self.failure_tokens.num_failure_tokens,
            num_hint_tokens=len(self.hint_tokens),
            tower_ranks={color: hanabi_tower.rank for color, hanabi_tower in self.hanabi_field.hanabi_towers.items()},
            discard_pile=self.discard_pile,
            current_player_id=self.current_player_id,
        )

    def get_all_players_observations(self) -> List[PlayerObservation]:
        full_states = self.get_current_full_state()

        observations = []
        for i in range(len(self.players)):
            relative_current_index = abs_to_rel_player_index(
                i, full_states.current_player_id, len(self.players)
            )
            observations.append(PlayerObservation(
                deck_size=full_states.deck_size,
                # Note that the index in this list is relative to current_player_id
                other_player_knowledges=full_states.player_knowledges[i + 1 :] + full_states.player_knowledges[:i],
                other_player_hands=full_states.player_hands[i + 1 :] + full_states.player_hands[:i],
                player_knowledge=full_states.player_knowledges[i],
                num_failure_tokens=full_states.num_failure_tokens,
                num_hint_tokens=full_states.num_hint_tokens,
                tower_ranks=full_states.tower_ranks,
                discard_pile=full_states.discard_pile,
                current_player_id=relative_current_index,
            ))

        return observations


    def get_current_player_observation(self) -> PlayerObservation:
        full_states = self.get_current_full_state()

        i = full_states.current_player_id
        relative_current_index = abs_to_rel_player_index(
            i, i, len(self.players)
        )
        return PlayerObservation(
            deck_size=full_states.deck_size,
            # Note that the index in this list is relative to current_player_id
            other_player_knowledges=full_states.player_knowledges[i + 1 :] + full_states.player_knowledges[:i],
            other_player_hands=full_states.player_hands[i + 1 :] + full_states.player_hands[:i],
            player_knowledge=full_states.player_knowledges[i],
            num_failure_tokens=full_states.num_failure_tokens,
            num_hint_tokens=full_states.num_hint_tokens,
            tower_ranks=full_states.tower_ranks,
            discard_pile=full_states.discard_pile,
            current_player_id=relative_current_index,
        )

    def get_current_valid_actions(self) -> List[Action]:
        current_player = self.players[self.current_player_id]
        return self.get_valid_actions(
            num_current_player_cards=len(current_player.hand), current_player_index=self.current_player_id
        )

    def receive_action(self, player: Player, action: Action):
        self.turn_since_deck_is_empty += int(self.deck.is_empty())

        if isinstance(action, PlayCard):
            card = player.use_card(action.played_card_index)
            if self.hanabi_field.is_able_to_add(card):
                color_field_is_completed = self.hanabi_field.add_card(card)
                if color_field_is_completed and self.hint_tokens.is_able_to_add_token():
                    self.hint_tokens.add_token()
            else:
                self.failure_tokens.add_token()
                self.discard_pile.append(card)
            if not self.deck.is_empty():
                player.draw_card(self.deck.get_card())

        elif isinstance(action, GetHintToken):
            if not self.hint_tokens.is_able_to_add_token():
                raise InvalidActionError("The number of hint tokens has reached the max.")

            discarded_card = player.use_card(action.discard_card_index)
            self.discard_pile.append(discarded_card)
            self.hint_tokens.add_token()
            if not self.deck.is_empty():
                player.draw_card(self.deck.get_card())

        elif isinstance(action, GiveColorHint):
            if not self.hint_tokens.is_able_to_use_token():
                raise InvalidActionError("The number of hint tokens is empty.")
            other_player_index = rel_to_abs_player_index(
                self.current_player_id, relative_other_index=action.player_index, num_players=len(self.players)
            )

            self.hint_tokens.use_token()
            self.players[other_player_index].get_color_hint(action.color)

        elif isinstance(action, GiveRankHint):
            if not self.hint_tokens.is_able_to_use_token():
                raise InvalidActionError("The number of hint tokens is empty.")
            other_player_index = rel_to_abs_player_index(
                self.current_player_id, relative_other_index=action.player_index, num_players=len(self.players)
            )

            self.hint_tokens.use_token()
            self.players[other_player_index].get_rank_hint(action.rank)
        else:
            raise InvalidActionError(f"Invalid action: {action}")

        self._prev_action_info = (self.current_player_id, action)
        self.current_player_id = (self.current_player_id + 1) % len(self.players)

    def auto_play(self):
        logging.info(self)
        max_num_rounds = (len(self.deck) + len(self.hint_tokens) + 1) // len(self.players) + 2

        for current_round in range(max_num_rounds):
            for current_player_id, player in enumerate(self.players):
                assert self.current_player_id == current_player_id

                valid_actions = self.get_valid_actions(
                    num_current_player_cards=len(player.hand), current_player_index=current_player_id
                )
                action = player.choose_action(valid_actions, self.get_current_player_observation())
                logging.info(f"Player {current_player_id}'s action: {action}")
                self.receive_action(player=player, action=action)

                logging.info(self)

                if self.is_terminal():
                    return self.hanabi_field.get_score()

    def __str__(self):
        string = ""

        if self._prev_action_info is not None:
            player_id, action = self._prev_action_info
            string += f"\nPlayer {player_id}'s action: {action}\n\n"

        string += "==============================\n"
        string += f"Deck: {len(self.deck)}" + "\n"
        string += f"Hint Tokens: [" + " o " * len(self.hint_tokens) + "]\n"
        string += f"Failure Tokens: [" + " x " * len(self.failure_tokens) + "]\n"
        string += "\n"

        string += "Hanabi Field" + "\n"
        string += str(self.hanabi_field)
        string += "\n"

        string += "Hand\n"
        for index, player in enumerate(self.players):
            string += f"Player {index}: "
            string += str([str(c) for c in player.hand]) + "\n"
        string += "\n"

        string += "Knowledge\n"
        for index, player in enumerate(self.players):
            string += f"Player {index}: "
            string += str([str(ck) for ck in player.card_knowledges]) + "\n"

        string += "=============================="

        return string
