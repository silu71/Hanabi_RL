# from hanabi.gym import HanabiEnv
from tests.test_action_encoder import test_if_encode_and_decode_is_same as test_action
from tests.test_observation_encoder import test_if_encode_and_decode_is_same as test_obs


if __name__ == "__main__":
    test_action()
    test_obs()
