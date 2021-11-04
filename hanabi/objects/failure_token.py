class FailureTokensOnField:
    def __init__(self, max_num_failure_tokens: int = 3):
        self.num_failure_tokens = 0
        self.max_num_failure_tokens = max_num_failure_tokens

    def add_token(self):
        if self.num_failure_tokens > self.max_num_failure_tokens:
            raise RuntimeError(
                f"The rank of failure tokens exceed the max rank: "
                f"{self.num_failure_tokens} > {self.max_num_failure_tokens}"
            )
        self.num_failure_tokens += 1

    def is_failed(self) -> bool:
        return self.num_failure_tokens >= self.max_num_failure_tokens
