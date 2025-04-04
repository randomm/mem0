from .base import BaseEmbedderConfig


class JinaEmbedderConfig(BaseEmbedderConfig):
    """
    Config for Jina AI Embeddings.
    """
    provider: str = "jina" 