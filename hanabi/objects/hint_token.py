class HintTokensOnField:
    def __init__(self, initial_num_hint_tokens: int = 8, max_num_hint_tokens: int = 8):
        assert initial_num_hint_tokens >= 0
        assert max_num_hint_tokens >= 0
        assert initial_num_hint_tokens <= max_num_hint_tokens

        self.num_hint_tokens = initial_num_hint_tokens
        self.max_num_hint_tokens = max_num_hint_tokens

    def is_able_to_add_token(self) -> bool:
        return self.num_hint_tokens < self.max_num_hint_tokens

    def add_token(self):
        if self.num_hint_tokens >= self.max_num_hint_tokens:
            raise RuntimeError(
                f"The number of hint tokens will exceed the max number: "
                f"{self.num_hint_tokens} >= {self.max_num_hint_tokens}"
            )
        self.num_hint_tokens += 1

    def is_able_to_use_token(self) -> bool:
        return self.num_hint_tokens > 0

    def use_token(self):
        if self.num_hint_tokens <= 0:
            raise RuntimeError(
                f"The number of hint tokens is less than 0!: {self.num_hint_tokens}"
            )
        self.num_hint_tokens -= 1
