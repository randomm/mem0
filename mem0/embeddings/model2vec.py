import logging
from typing import List, Literal, Optional

from mem0.configs.embeddings.base import BaseEmbedderConfig

from .base import EmbeddingBase

try:
    from model2vec import StaticModel
except ImportError:
    raise ImportError(
        "model2vec is not installed. Please install it using `pip install model2vec`"
    )


logger = logging.getLogger(__name__)


class Model2VecEmbedder(EmbeddingBase):
    """
    Initialize the Model2Vec Embedder.

    Args:
        config (BaseEmbedderConfig): Configuration object for the embedder.
    """

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        if not self.config.model:
            raise ValueError("Model name is required for Model2VecEmbedder.")

        try:
            self.model = StaticModel.from_pretrained(
                self.config.model, **self.config.model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load Model2Vec model '{self.config.model}': {e}")
            raise

        # Attempt to infer embedding dimension (Model2Vec doesn't explicitly store this)
        # We can infer it by encoding a dummy text and checking the length.
        try:
            dummy_embedding = self.model.encode("test")
            self._embedding_dims = len(dummy_embedding)
            if self.config.embedding_dims and self.config.embedding_dims != self._embedding_dims:
                logger.warning(
                    f"Configured embedding_dims ({self.config.embedding_dims}) "
                    f"does not match inferred dimension ({self._embedding_dims}) "
                    f"for model '{self.config.model}'. Using inferred dimension."
                )
            elif not self.config.embedding_dims:
                 self.config.embedding_dims = self._embedding_dims

        except Exception as e:
            logger.warning(f"Could not infer embedding dimension for '{self.config.model}': {e}")
            if not self.config.embedding_dims:
                 raise ValueError(
                     "Embedding dimension could not be inferred and was not provided in config."                     
                 )
            self._embedding_dims = self.config.embedding_dims # Trust config if inference fails

        logger.info(f"Initialized Model2VecEmbedder with model '{self.config.model}' (dims: {self._embedding_dims})")


    @property
    def embedding_dims(self) -> int:
        """
        Get the embedding dimensions.

        Returns:
            int: The number of dimensions in the embedding.
        """
        return self._embedding_dims

    def embed(
        self, text: str | List[str], memory_action: Optional[Literal["add", "search", "update"]] = None
    ) -> List[List[float]]:
        """
        Get the embedding for the given text.

        Args:
            text (str | List[str]): The text or list of texts to embed.
            memory_action (Optional[Literal["add", "search", "update"]], optional): Type of memory action (ignored by this embedder). Defaults to None.

        Returns:
            List[List[float]]: The embedding vector(s).
        """
        try:
            # model2vec expects list of strings or single string, returns np.ndarray or list of np.ndarray
            embeddings = self.model.encode(text)

            # Ensure output is always List[List[float]]
            if isinstance(text, str):
                # Single text input, encode returns a single numpy array
                return [embeddings.tolist()]
            else:
                # List of texts input, encode returns list of numpy arrays
                return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to embed text with Model2Vec model '{self.config.model}': {e}")
            raise 