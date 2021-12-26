from typing import Union, List
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


from .game_engine import GameEngine, Player, InvalidActionError, PlayerObservation, Card, CardKnowledge, Rank, Color
from .objects.deck import DEFAULT_NUM_CARDS
from .actions import Action, PlayCard, GetHintToken, GiveColorHint, GiveRankHint


class ObservationEncoder:
    def __init__(
        self,
        num_players: int,
        max_deck_size: int,
        num_initial_cards: int,
        num_max_hint_tokens: int,
        max_num_failure_tokens: int,
        max_rank: int,
        num_colors: int,
    ):

        self.num_players = num_players
        self.max_deck_size = max_deck_size
        self.num_initial_cards = num_initial_cards
        self.num_max_hint_tokens = num_max_hint_tokens
        self.max_num_failure_tokens = max_num_failure_tokens
        self.max_rank = max_rank
        self.num_colors = num_colors

        self._color_list = Color.list(num_colors)
        self._rank_list = Rank.list(max_rank)

    @property
    def encode_dim(self) -> int:

        deck_size_dim = 1
        other_player_knowledges_dim = self.player_hands_dim
        other_player_hands_dim = self.player_hands_dim
        current_player_knowledges_dim = self.hand_dim
        current_player_index_dim = self.num_players
        num_failure_tokens_dim = 1
        num_hint_tokens_dim = 1
        tower_ranks_dim = self.num_colors
        discard_pile_dim = self.num_colors * self.max_rank

        return (
            deck_size_dim
            + other_player_knowledges_dim
            + other_player_hands_dim
            + current_player_knowledges_dim
            + current_player_index_dim
            + num_failure_tokens_dim
            + num_hint_tokens_dim
            + tower_ranks_dim
            + discard_pile_dim
        )

    @property
    def hand_dim(self) -> int:
        return self.num_initial_cards * (self.num_colors + self.max_rank)

    @property
    def player_hands_dim(self) -> int:
        return (self.num_players - 1) * self.hand_dim

    def _encode_card(self, card: Card) -> np.ndarray:
        rank_array = np.zeros(self.max_rank)
        color_array = np.zeros(self.num_colors)

        rank_array[card.rank.value - 1] = 1
        color_array[card.color.value] = 1

        return np.concatenate((rank_array, color_array))
    
    def _encode_card_knowledge(self, card_knoledge: CardKnowledge) -> np.ndarray:
        rank_array = np.zeros(self.max_rank)
        color_array = np.zeros(self.num_colors)

        for r in self._rank_list:
            rank_array[r.value - 1] = int(card_knoledge.rank_possibilities[r])

        for c in self._color_list:
            color_array[c.value] = int(card_knoledge.color_possibilities[c])

        return np.concatenate((rank_array, color_array))

    def _encode_hand(self, hand: List[Card]) -> np.ndarray:
        return np.concatenate([self._encode_card(card) for card in hand])

    def _encode_hand_knowledge(self, hand_knowledge: List[CardKnowledge]) -> np.ndarray:
        return np.concatenate([self._encode_card_knowledge(card_knowledge) for card_knowledge in hand_knowledge])

    def _encode_player_hands(self, player_hands: List[List[Card]]) -> np.ndarray:
        return np.concatenate([self._encode_hand(hand) for hand in player_hands])

    def _encode_player_hand_knowledges(self, player_hand_knowledges: List[List[CardKnowledge]]) -> np.ndarray:
        return np.concatenate([self._encode_hand_knowledge(hand_knowledge) for hand_knowledge in player_hand_knowledges])

    def _encode_player_index(self, player_index: int) -> np.ndarray:
        array = np.zeros(self.num_players)
        array[player_index] = 1
        return array

    def encode(self, observation: PlayerObservation) -> np.ndarray:

        discard_pile_array = np.zeros(self.num_colors * self.max_rank)
        for card in observation.discard_pile:
            card_idx = (card.rank.value - 1) * self.num_colors + card.color.value
            discard_pile_array[card_idx] += 1 / DEFAULT_NUM_CARDS[card.rank]

        obs_array = np.concatenate(
            [
                np.array([observation.deck_size / self.max_deck_size]),
                self._encode_player_hand_knowledges(observation.other_player_knowledges),
                self._encode_player_hands(observation.other_player_hands),
                self._encode_hand_knowledge(observation.current_player_knowledges),
                self._encode_player_index(observation.current_player_id),
                np.array([observation.num_failure_tokens / self.max_num_failure_tokens]),
                np.array([observation.num_hint_tokens / self.num_max_hint_tokens]),
                np.array([
                    observation.tower_ranks[self._color_list[i]].value / self.max_rank
                    for i in range(self.num_colors)
                ]),
                discard_pile_array,
            ]
        )
        if len(obs_array) != self.encode_dim:
            print(self._encode_player_hand_knowledges(observation.other_player_knowledges).shape)
            print(self._encode_player_hands(observation.other_player_hands).shape)
            print(self._encode_hand_knowledge(observation.current_player_knowledges).shape)
            print(self._encode_player_index(observation.current_player_id))
            print(np.array([
                    observation.tower_ranks[self._color_list[i]].value / self.max_rank
                    for i in range(self.num_colors)
                ]).shape)
            print(discard_pile_array.shape)
            import pdb; pdb.set_trace()
        assert len(obs_array) == self.encode_dim
        assert (0 <= obs_array).all() and (obs_array <= 1).all()
        return obs_array


class ActionEncoder:
    def __init__(self, num_players: int, num_initial_cards: int, max_rank: int, num_colors: int):
        self.num_players = num_players
        self.num_initial_cards = num_initial_cards
        self.max_rank = max_rank
        self.num_colors = num_colors

        self._color_list = Color.list(num_colors)
        self._rank_list = Rank.list(max_rank)

    @property
    def num_actions(self) -> int:
        play_card_dim = self.num_initial_cards
        get_hint_token_dim = self.num_initial_cards
        give_color_hint_dim = (self.num_players - 1) * self.num_colors
        give_rank_hint_dim = (self.num_players - 1) * self.max_rank

        return play_card_dim + get_hint_token_dim + give_color_hint_dim + give_rank_hint_dim

    def encode(self, action: Action) -> int:
        if isinstance(action, PlayCard):
            return action.played_card_index
        elif isinstance(action, GetHintToken):
            return self.num_initial_cards + action.discard_card_index
        elif isinstance(action, GiveColorHint):
            return (
                self.num_initial_cards * 2
                + action.player_index * self.num_colors + action.color.value
            )
        elif isinstance(action, GiveRankHint):
            return (
                self.num_initial_cards * 2
                + (self.num_players - 1) * self.num_colors
                + action.player_index * self.max_rank + action.rank.value - 1
            )
        else:
            raise InvalidActionError(action)

    def decode(self, action_index: int) -> Action:
        if 0 <= action_index < self.num_initial_cards:
            return PlayCard(action_index)
        elif self.num_initial_cards <= action_index < self.num_initial_cards * 2:
            return GetHintToken(action_index - self.num_initial_cards)
        elif (
            self.num_initial_cards * 2
            <= action_index
            < self.num_initial_cards * 2 + (self.num_players - 1) * self.num_colors
        ):
            player_color_index = action_index - self.num_initial_cards * 2
            player_index, color_index = divmod(player_color_index, self.num_colors)
            return GiveColorHint(player_index=player_index, color=self._color_list[color_index])
        elif action_index < self.num_actions:
            player_rank_index = action_index - (self.num_initial_cards * 2 + (self.num_players - 1) * self.num_colors)
            player_index, rank_index = divmod(player_rank_index, self.max_rank)
            return GiveRankHint(player_index=player_index, rank=self._rank_list[rank_index])
        else:
            raise ValueError(f"Action index is out of range: {action_index}.")


class HanabiEnv(gym.Env):
    def __init__(
        self,
        num_players: int = 2,
        num_initial_cards: int = 5,
        num_initial_hint_tokens: int = 8,
        num_max_hint_tokens: int = 8,
        max_num_failure_tokens: int = 3,
        max_rank: int = 5,
        num_colors: int = 5,
        use_sparse_reward: bool = False,
    ):
        self.game_engine = GameEngine(
            num_initial_cards=num_initial_cards,
            num_initial_hint_tokens=num_initial_hint_tokens,
            num_max_hint_tokens=num_max_hint_tokens,
            max_num_failure_tokens=max_num_failure_tokens,
            max_rank=max_rank,
            num_colors=num_colors,
        )

        self.observation_encoder = ObservationEncoder(
            num_players=num_players,
            num_initial_cards=num_initial_cards,
            max_deck_size=self.game_engine.max_deck_size,
            num_max_hint_tokens=num_max_hint_tokens,
            max_num_failure_tokens=max_num_failure_tokens,
            max_rank=max_rank,
            num_colors=num_colors,
        )

        self.action_encoder = ActionEncoder(
            num_players=num_players, num_initial_cards=num_initial_cards, max_rank=max_rank, num_colors=num_colors,
        )

        # Actions are discrete integer values
        self.action_space = spaces.MultiDiscrete(
            [self.action_encoder.num_actions] * num_players
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(
                low=0, high=1,
                shape=(num_players, self.observation_encoder.encode_dim),
                dtype=np.float32
            ),
            spaces.Box(
                low=0, high=1,
                shape=(num_players, self.action_encoder.num_actions),
                dtype=np.float32
            )
        ))

        self.num_players = num_players
        self.use_sparse_reward = use_sparse_reward
        self._game_is_done = None
        self._prev_valid_actions = None

    def reset(self):
        players = [Player() for _ in range(self.num_players)]
        self.game_engine.setup_game(players)

        obs_all = self.game_engine.get_all_players_observations()
        obs_array = np.stack(
            [self.observation_encoder.encode(obs) for obs in obs_all],
            axis=0
        )

        self._game_is_done = False
        valid_actions = self.get_valid_actions()
        self._prev_valid_actions = valid_actions

        return (obs_array, valid_actions)

    def seed(self, seed: int = 1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def get_valid_actions(self) -> np.ndarray:
        current_player_id = self.game_engine.current_player_id
        valid_actions = self.game_engine.get_current_valid_actions()

        array = np.zeros((self.num_players, self.action_encoder.num_actions))
        for action in valid_actions:
            action_index = self.action_encoder.encode(action)
            array[current_player_id, action_index] = 1

        return array

    def step(self, actions: np.ndarray):

        if self._game_is_done:
            raise RuntimeError("Game is already done.")

        action_index = actions[self.game_engine.current_player_id]
        action = self.action_encoder.decode(action_index)

        if self._prev_valid_actions[self.game_engine.current_player_id, action_index] == 0:
            raise InvalidActionError()

        prev_score = self.game_engine.hanabi_field.get_score()
        self.game_engine.receive_action(player=self.game_engine.current_player, action=action)
        dense_reward = self.game_engine.hanabi_field.get_score() - prev_score

        obs_all = self.game_engine.get_all_players_observations()
        obs_array = np.stack(
            [self.observation_encoder.encode(obs) for obs in obs_all],
            axis=0
        )

        valid_actions = self.get_valid_actions()
        self._prev_valid_actions = valid_actions

        done = self.game_engine.is_terminal()

        if done:
            sparse_reward = self.game_engine.hanabi_field.get_score()
            self._game_is_done = True
        else:
            sparse_reward = 0

        if self.use_sparse_reward:
            reward = sparse_reward
        else:
            reward = dense_reward

        return (obs_array, valid_actions), reward, done, {}

    def render(self, mode="human"):
        pass
