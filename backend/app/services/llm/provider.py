from abc import ABC, abstractmethod
from typing import List, Dict

class LLMProvider(ABC):
    @abstractmethod
    async def generate_answer(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """
        Takes a system prompt and a list of messages (history + new query),
        and returns the generated string answer.
        Messages should be formatted as: [{"role": "user"|"assistant", "content": "..."}]
        """
        pass
