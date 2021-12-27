from typing import Union, List, Dict
import numpy as np
import gym
from gym import spaces


from .game_engine import GameEngine, Player, InvalidActionError, PlayerObservation, Card, CardKnowledge, Rank, Color
from .objects.deck import DEFAULT_NUM_CARDS
from .actions import Action, PlayCard, GetHintToken, GiveColorHint, GiveRankHint


class ObservationEncoder:
    def __init__(
        self,
        num_players: int,
        num_initial_cards: int,
        num_max_hint_tokens: int,
        max_num_failure_tokens: int,
        max_rank: int,
        num_colors: int,
    ):

        self.num_players = num_players
        self.num_initial_cards = num_initial_cards
        self.num_max_hint_tokens = num_max_hint_tokens
        self.max_num_failure_tokens = max_num_failure_tokens
        self.max_rank = max_rank
        self.num_colors = num_colors

        self._color_list = Color.list(num_colors)
        self._rank_list = Rank.list(max_rank)

    @property
    def encode_dim(self) -> int:

        deck_size_dim = self.max_deck_size
        other_player_knowledges_dim = self.player_hands_dim
        other_player_hands_dim = self.player_hands_dim
        player_knowledge_dim = self.hand_dim
        num_failure_tokens_dim = self.max_num_failure_tokens
        num_hint_tokens_dim = self.num_max_hint_tokens
        tower_ranks_dim = self.num_colors * self.max_rank
        discard_pile_dim = self.num_colors * sum([DEFAULT_NUM_CARDS[r] for r in self._rank_list])
        current_player_id_dim = self.num_players

        return (
            deck_size_dim
            + other_player_knowledges_dim
            + other_player_hands_dim
            + player_knowledge_dim
            + num_failure_tokens_dim
            + num_hint_tokens_dim
            + tower_ranks_dim
            + discard_pile_dim
            + current_player_id_dim
        )

    @property
    def card_dim(self) -> int:
        return self.num_colors + self.max_rank

    @property
    def hand_dim(self) -> int:
        return self.num_initial_cards * self.card_dim

    @property
    def player_hands_dim(self) -> int:
        return (self.num_players - 1) * self.hand_dim

    @property
    def max_deck_size(self) -> int:
        num_all = self.num_colors * sum(DEFAULT_NUM_CARDS.values())
        num_distributed = self.num_initial_cards * self.num_players
        return num_all - num_distributed

    def encode(self, observation: PlayerObservation) -> np.ndarray:

        def encode_onehot(index: int, size: int) -> np.ndarray:
            assert 0 <= index < size
            array = np.zeros(size)
            array[index] = 1
            return array

        def encode_num(num: int, max_num: int) -> np.ndarray:
            assert 0 <= num <= max_num
            array = np.zeros(max_num)
            array[:num] = 1
            return array

        def encode_card(card: Card) -> np.ndarray:
            rank_array = encode_onehot(card.rank.value - 1, self.max_rank)
            color_array = encode_onehot(card.color.value, self.num_colors)
            return np.concatenate((rank_array, color_array))

        def encode_hand(hand: List[Card]) -> np.ndarray:
            array = []
            for i in range(self.num_initial_cards):
                if i < len(hand):
                    array.append(encode_card(hand[i]))
                else:
                    array.append(np.zeros(self.card_dim))

            return np.concatenate(array)

        def encode_card_knowledge(card_knowledge: CardKnowledge) -> np.ndarray:
            rank_array = np.zeros(self.max_rank)
            color_array = np.zeros(self.num_colors)

            for r in self._rank_list:
                rank_array[r.value - 1] = int(card_knowledge.rank_possibilities[r])

            for c in self._color_list:
                color_array[c.value] = int(card_knowledge.color_possibilities[c])

            return np.concatenate((rank_array, color_array))

        def encode_hand_knowledge(hand_knowledge: List[CardKnowledge]) -> np.ndarray:
            array = []
            for i in range(self.num_initial_cards):
                if i < len(hand_knowledge):
                    array.append(encode_card_knowledge(hand_knowledge[i]))
                else:
                    array.append(np.zeros(self.card_dim))

            return np.concatenate(array)

        def encode_tower_ranks(tower_ranks: Dict[Color, Rank]) -> np.ndarray:
            # import pdb; pdb.set_trace()
            array = []
            for i in range(self.num_colors):
                color = self._color_list[i]
                rank = tower_ranks[color]
                array.append(encode_num(rank.value, self.max_rank))

            return np.concatenate(array)

        def encode_discard_pile(discard_pile: List[Card]) -> np.ndarray:
            count_array = [[0 for j in range(self.num_colors)] for i in range(self.max_rank)]
            for card in discard_pile:
                count_array[card.rank.value - 1][card.color.value] += 1

            array = []
            for i in range(self.max_rank):
                rank = self._rank_list[i]
                rank_num_cards = DEFAULT_NUM_CARDS[rank]
                for j in range(self.num_colors):
                    array.append(encode_num(count_array[i][j], rank_num_cards))

            return np.concatenate(array)


        deck_size_array = encode_num(observation.deck_size, self.max_deck_size)
        other_player_knowledges_array = np.concatenate([
            encode_hand_knowledge(hand_knowledge)
            for hand_knowledge in observation.other_player_knowledges
        ])
        other_player_hands_array = np.concatenate([
            encode_hand(hand) for hand in observation.other_player_hands
        ])
        player_knowledge_array = encode_hand_knowledge(observation.player_knowledge)
        num_failure_tokens_array = encode_num(observation.num_failure_tokens, self.max_num_failure_tokens)
        num_hint_tokens_array = encode_num(observation.num_hint_tokens, self.num_max_hint_tokens)
        tower_ranks_array = encode_tower_ranks(observation.tower_ranks)
        discard_pile_array = encode_discard_pile(observation.discard_pile)
        current_player_id_array = encode_onehot(observation.current_player_id, self.num_players)

        obs_array = np.concatenate([
            deck_size_array,
            other_player_knowledges_array,
            other_player_hands_array,
            player_knowledge_array,
            num_failure_tokens_array,
            num_hint_tokens_array,
            tower_ranks_array,
            discard_pile_array,
            current_player_id_array,
        ])
        assert len(obs_array) == self.encode_dim
        assert (0 <= obs_array).all() and (obs_array <= 1).all()
        return obs_array

    def decode(self, obs_array: np.ndarray) -> PlayerObservation:

        def decode_card(card_array: np.ndarray) -> Card:
            rank_array = card_array[:self.max_rank]
            assert np.sum(rank_array > 0) == 1
            rank = self._rank_list[np.argmax(rank_array)]

            color_array = card_array[self.max_rank:]
            assert np.sum(color_array > 0) == 1
            color = self._color_list[np.argmax(color_array)]

            return Card(color=color, rank=rank)

        def decode_hand(hand_array: np.ndarray) -> List[Card]:
            hand = []

            for card_index in range(self.num_initial_cards):
                card_array = hand_array[self.card_dim * card_index : self.card_dim * (card_index + 1)]
                if np.all(card_array == 0):
                    # no more card
                    break

                card = decode_card(card_array)
                hand.append(card)

            return hand

        def decode_card_knowledge(card_knowledge_array: np.ndarray) -> CardKnowledge:
            card_knowledge = CardKnowledge(self.max_rank, self.num_colors)

            rank_array = card_knowledge_array[:self.max_rank]
            assert np.any(rank_array > 0)
            for r in self._rank_list:
                if rank_array[r.value - 1] == 0:
                    card_knowledge.get_rank_hint(positive=False, rank=r)

            color_array = card_knowledge_array[self.max_rank:]
            assert np.any(color_array > 0)
            for c in self._color_list:
                if color_array[c.value] == 0:
                    card_knowledge.get_color_hint(positive=False, color=c)

            return card_knowledge

        def decode_hand_knowledge(hand_knowledge_array: np.ndarray) -> List[CardKnowledge]:
            hand_knowledge = []

            for card_index in range(self.num_initial_cards):
                card_knowledge_array = hand_knowledge_array[self.card_dim * card_index : self.card_dim * (card_index + 1)]
                if np.all(card_knowledge_array == 0):
                    # no more card
                    break

                card_knowledge = decode_card_knowledge(card_knowledge_array)
                hand_knowledge.append(card_knowledge)

            return hand_knowledge

        def decode_tower_ranks(tower_ranks_array: np.ndarray) -> Dict[Color, Rank]:
            tower_ranks = {}
            for i in range(self.num_colors):
                color = self._color_list[i]
                r = int(np.sum(tower_ranks_array[self.max_rank * i : self.max_rank * (i + 1)]))
                rank = Rank(r)
                tower_ranks[color] = rank

            return tower_ranks

        def decode_discard_pile(discard_pile_array: np.ndarray) -> List[Card]:
            discard_pile = []
            offset = 0
            for i in range(self.max_rank):
                rank = self._rank_list[i]
                rank_num_cards = DEFAULT_NUM_CARDS[rank]
                for j in range(self.num_colors):
                    color = self._color_list[j]
                    num = int(np.sum(discard_pile_array[offset : offset + rank_num_cards]))
                    for _ in range(num):
                        discard_pile.append(Card(color=color, rank=rank))
                    offset += rank_num_cards

            return discard_pile


        offset = 0

        deck_size_dim = self.max_deck_size
        deck_size = int(np.sum(obs_array[offset : offset + deck_size_dim]))
        offset += deck_size_dim

        other_player_knowledges_dim = self.player_hands_dim
        other_player_knowledges_array = obs_array[offset : offset + other_player_knowledges_dim]
        other_player_knowledges = []
        for player_index in range(self.num_players - 1):
            hand_knowledge_array = other_player_knowledges_array[self.hand_dim * player_index : self.hand_dim * (player_index + 1)]
            hand_knowledge = decode_hand_knowledge(hand_knowledge_array)
            other_player_knowledges.append(hand_knowledge)
        offset += other_player_knowledges_dim

        other_player_hands_dim = self.player_hands_dim
        other_player_hands_array = obs_array[offset : offset + other_player_hands_dim]
        other_player_hands = []
        for player_index in range(self.num_players - 1):
            hand_array = other_player_hands_array[self.hand_dim * player_index : self.hand_dim * (player_index + 1)]
            hand = decode_hand(hand_array)
            other_player_hands.append(hand)
        offset += other_player_hands_dim

        player_knowledge_dim = self.hand_dim
        player_knowledge_array = obs_array[offset : offset + player_knowledge_dim]
        player_knowledge = decode_hand_knowledge(player_knowledge_array)
        offset += player_knowledge_dim

        num_failure_tokens_dim = self.max_num_failure_tokens
        num_failure_tokens_array = obs_array[offset : offset + num_failure_tokens_dim]
        num_failure_tokens = int(np.sum(num_failure_tokens_array))
        offset += num_failure_tokens_dim

        num_hint_tokens_dim = self.num_max_hint_tokens
        num_hint_tokens_array = obs_array[offset : offset + num_hint_tokens_dim]
        num_hint_tokens = int(np.sum(num_hint_tokens_array))
        offset += num_hint_tokens_dim

        tower_ranks_dim = self.num_colors * self.max_rank
        tower_ranks_array = obs_array[offset : offset + tower_ranks_dim]
        tower_ranks = decode_tower_ranks(tower_ranks_array)
        offset += tower_ranks_dim

        discard_pile_dim = self.num_colors * sum(DEFAULT_NUM_CARDS.values())
        discard_pile_array = obs_array[offset : offset + discard_pile_dim]
        discard_pile = decode_discard_pile(discard_pile_array)
        offset += discard_pile_dim

        current_player_id_dim = self.num_players
        curren_player_id_array = obs_array[offset : offset + current_player_id_dim]
        current_player_id = np.argmax(curren_player_id_array)
        offset += current_player_id_dim

        assert offset == self.encode_dim

        return PlayerObservation(
            deck_size=deck_size,
            other_player_knowledges=other_player_knowledges,
            other_player_hands=other_player_hands,
            player_knowledge=player_knowledge,
            num_failure_tokens=num_failure_tokens,
            num_hint_tokens=num_hint_tokens,
            tower_ranks=tower_ranks,
            discard_pile=discard_pile,
            current_player_id=current_player_id,
        )


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
        self.game_engine.seed(seed)

    def get_valid_actions(self) -> np.ndarray:
        current_player_id = self.game_engine.current_player_id
        valid_actions = self.game_engine.get_current_valid_actions()

        array = np.zeros((self.num_players, self.action_encoder.num_actions))
        for action in valid_actions:
            action_index = self.action_encoder.encode(action)
            array[current_player_id, action_index] = 1

        return array

    def step(self, action_indices: np.ndarray):

        if self._game_is_done:
            raise RuntimeError("Game is already done.")

        action_index = action_indices[self.game_engine.current_player_id]
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
        return str(self.game_engine)
