import numpy as np

from hanabi.game_engine import GameEngine
from hanabi.gym import ObservationEncoder
from hanabi.players import RandomPlayer


def test_if_encode_and_decode_is_same():
    game = GameEngine(num_colors=3)

    num_players = 2
    observation_encoder = ObservationEncoder(
        num_players=num_players,
        num_initial_cards=game.num_initial_cards,
        num_max_hint_tokens=game.num_initial_hint_tokens,
        max_num_failure_tokens=game.max_num_failure_tokens,
        max_rank=game.max_rank,
        num_colors=game.num_colors,
    )

    players = [RandomPlayer() for _ in range(num_players)]
    game.setup_game(players)


    def check(observation):
        encoded_observation = observation_encoder.encode(observation)
        decoded_observation = observation_encoder.decode(encoded_observation)
        reencoded_observation = observation_encoder.encode(decoded_observation)
        assert np.max(np.abs(encoded_observation - reencoded_observation)) < 1e-3


    for _ in range(100):
        for current_player_id, player in enumerate(game.players):
            assert game.current_player_id == current_player_id

            valid_actions = game.get_valid_actions(
                num_current_player_cards=len(player.hand), current_player_index=current_player_id
            )
            observation = game.get_current_player_observation()

            check(observation)

            action = player.choose_action(valid_actions, observation)
            game.receive_action(player=player, action=action)

            if game.is_terminal():
                players = [RandomPlayer() for _ in range(num_players)]
                game.setup_game(players)
                break

    print("ok")

