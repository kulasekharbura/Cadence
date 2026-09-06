from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of texts and returns a list of embedding vectors.
        Should handle batching internally if necessary.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by this provider.
        """
        pass
