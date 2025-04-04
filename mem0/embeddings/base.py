from abc import ABC, abstractmethod
from typing import Literal, Optional
import numpy as np  # Import numpy

from mem0.configs.embeddings.base import BaseEmbedderConfig


class EmbeddingBase(ABC):
    """Initialized a base embedding class

    :param config: Embedding configuration option class, defaults to None
    :type config: Optional[BaseEmbedderConfig], optional
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        if config is None:
            self.config = BaseEmbedderConfig()
        else:
            self.config = config

    def _truncate_embedding_if_needed(self, embedding, target_dims=None):
        """
        Helper method to truncate embeddings to target dimensions if specified.
        Primarily used for Matryoshka-style embeddings.

        Args:
            embedding: The embedding vector(s) (list, numpy array).
            target_dims: Override dimensions to truncate to. If None, uses config.matryoshka_dims.

        Returns:
            Truncated embedding(s) if target_dims or config.matryoshka_dims is specified,
            otherwise the original embedding(s).
        """
        dims = target_dims or getattr(self.config, 'matryoshka_dims', None)

        if dims is None:
            return embedding

        if isinstance(embedding, np.ndarray):
            if len(embedding.shape) == 1:  # Single embedding
                return embedding[:dims]
            elif len(embedding.shape) > 1: # Batch of embeddings
                return embedding[:, :dims]
            else: # Unexpected shape
                return embedding
        elif isinstance(embedding, list):
            if not embedding: # Empty list
                return embedding
            # Check if it's a list of lists/arrays (batch)
            if isinstance(embedding[0], (list, np.ndarray)):
                return [list(emb)[:dims] for emb in embedding] # Ensure inner lists
            else:  # Single embedding as list
                return embedding[:dims]

        return embedding  # Return as-is if format not recognized or empty

    @abstractmethod
    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]]):
        """
        Get the embedding for the given text.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        pass
