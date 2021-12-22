import gym
from gym import spaces
from gym.utils import seeding

from .game_engine import GameEngine, Player, InvalidActionError


class HanabiEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}

    # Enumeration of possible actions

    def __init__(self, num_players: int = 2, seed: int = 2):
        self.game_engine = GameEngine()
        self.players = [Player() for _ in range(num_players)]

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.action_set))

        self.observation_space = spaces.Box(low=0, high=10, shape=(100,), dtype="uint8")

        # Initialize the RNG
        self.seed(seed=seed)
        # Initialize the state
        self.reset()

    def reset(self):
        self.game_engine.setup_game(self.players)

    def seed(self, seed: int = 1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        valid_actions = self.game_engine.get_valid_actions()
        if action not in valid_actions:
            raise InvalidActionError()

        obs = self.game_engine.get_current_player_observation()
        done = self.game_engine.is_terminal()
        reward = 0
        if done:
            reward = self.game_engine.hanabi_field.get_score()

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass
