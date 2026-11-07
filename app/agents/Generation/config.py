from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    temperature: float = 0.1
    max_output_tokens: int = 5000

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")

        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer.")
