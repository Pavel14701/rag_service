"""Interface for LLM-based answer generation."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMGenerator(Protocol):
    """Abstract interface for a language model."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User's query with context.
            temperature: Sampling temperature (0-1).

        Returns:
            The generated text.

        """
        ...
