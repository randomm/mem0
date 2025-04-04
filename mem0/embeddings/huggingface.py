from typing import Literal, Optional, List, Union

from sentence_transformers import SentenceTransformer
import numpy as np

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class HuggingFaceEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "multi-qa-MiniLM-L6-cos-v1"
        
        # Clean up the kwargs to avoid passing unsupported parameters to SentenceTransformer
        model_kwargs = self.config.model_kwargs or {}
        # Store truncation dimension for later use but don't pass it to SentenceTransformer
        self.truncate_dimension = model_kwargs.pop('truncate_dimension', None)
        
        self.model = SentenceTransformer(self.config.model, **model_kwargs)
        
        # Get the full dimension of the model
        full_dim = self.model.get_sentence_embedding_dimension()
        
        # If truncate_dimension is set, use it as embedding_dims
        if self.truncate_dimension is not None:
            if self.truncate_dimension > full_dim:
                raise ValueError(f"Truncate dimension ({self.truncate_dimension}) cannot be larger than the model's dimension ({full_dim})")
            self.config.embedding_dims = self.truncate_dimension
        else:
            self.config.embedding_dims = self.config.embedding_dims or full_dim

    def embed(self, text: Union[str, List[str]], memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """
        Get the embedding for the given text using Hugging Face.
        Supports matryoshka embedding truncation if truncate_dimension is set.

        Args:
            text (str or List[str]): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector(s).
        """
        # Generate full embeddings
        embeddings = self.model.encode(text, convert_to_numpy=True)
        
        # Apply truncation if specified using the base class helper
        # Pass the dimension stored during initialization
        embeddings = self._truncate_embedding_if_needed(embeddings, target_dims=self.truncate_dimension)
        
        # Convert numpy arrays to lists
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        elif isinstance(embeddings, list) and all(isinstance(item, np.ndarray) for item in embeddings):
            return [emb.tolist() for emb in embeddings]
        
        return embeddings
