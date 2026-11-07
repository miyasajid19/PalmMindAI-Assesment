from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentSummarizerConfig:

    summary_length: int = 500

    summary_style: str = "descriptive and technical"

    language: str = "English"

    temperature: float = 0.2

    max_output_tokens: int = 2000

    def __post_init__(self):

        if self.summary_length <= 0:
            raise ValueError(
                "summary_length must be greater than zero."
            )

        if not 0 <= self.temperature <= 2:
            raise ValueError(
                "temperature must be between 0 and 2."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if not self.summary_style.strip():
            raise ValueError(
                "summary_style cannot be empty."
            )

        if not self.language.strip():
            raise ValueError(
                "language cannot be empty."
            )