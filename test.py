from hanabi.game_engine import GameEngine
from hanabi.players import Player, RandomPlayer, NaiveRuleBasedPlayer



if __name__ == "__main__":
    game = GameEngine(num_colors=2, max_rank=4)
    game.seed(10)

    game.setup_game([
        NaiveRuleBasedPlayer(),
        NaiveRuleBasedPlayer(),
        # NaiveRuleBasedPlayer(),
        # NaiveRuleBasedPlayer(),
    ])

    print(game)

    terminal = False
    while not terminal:
        for current_player_id, player in enumerate(game.players):
            assert game.current_player_id == current_player_id

            valid_actions = game.get_valid_actions(
                num_current_player_cards=len(player.hand), current_player_index=current_player_id
            )
            observation = game.get_current_player_observation()
            observations = game.get_all_players_observations()
            # import pdb; pdb.set_trace()
            assert str(observation) == str(observations[current_player_id])
            action = player.choose_action(valid_actions, observation)
            # print(f"Player {current_player_id}'s action: {action}")
            game.receive_action(player=player, action=action)

            print(game)

            # observations = game.get_all_players_observations()
            # for i in range(len(game.players)):
            #     print(f"Player {i}" + (" *" if i == current_player_id else ""))
            #     print(observations[i])
            #     print()

            if game.is_terminal():
                print(f"score: {game.hanabi_field.get_score()}")
                terminal = True
                break
