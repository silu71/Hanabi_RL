from hanabi.gym import ActionEncoder


def test_if_encode_and_decode_is_same():
    action_encoder = ActionEncoder(num_players=3, num_initial_cards=5, max_rank=5, num_colors=5)

    for action_index in range(action_encoder.num_actions):
        action = action_encoder.decode(action_index)
        encoded_action_index = action_encoder.encode(action)
        print(action, encoded_action_index)
        assert encoded_action_index == action_index
