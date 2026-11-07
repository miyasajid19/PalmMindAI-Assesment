from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultiQueryGeneratorConfig:
    max_queries: int = 1
    temperature: float = 0.2
    max_output_tokens: int = 1500

    def __post_init__(self):
        if isinstance(self.max_queries, bool) or not isinstance(self.max_queries, int) or self.max_queries < 1:
            raise ValueError("max_queries must be greater than 0.")

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")

        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than 0.")
