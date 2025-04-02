from .base import BaseEmbedderConfig


class Model2VecEmbedderConfig(BaseEmbedderConfig):
    """
    Config for Model2Vec Embeddings.
    """

    provider: str = "model2vec" 