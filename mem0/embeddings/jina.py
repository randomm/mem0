import os
import requests
from typing import Literal, Optional, List

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase


class JinaEmbedding(EmbeddingBase):
    """
    Embedding provider for Jina AI embeddings API.
    Supports Jina embeddings v3 with scalar vectors.
    """
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        # Default model is jina-embeddings-v3
        self.config.model = self.config.model or "jina-embeddings-v3"
        # Default dimensions for jina-embeddings-v3 is 1024
        self.config.embedding_dims = self.config.embedding_dims or 1024

        # Get API key from config or environment
        self.api_key = self.config.api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key is required. Set it in the config or as JINA_API_KEY environment variable.")
        
        # Set API base URL with possible override
        self.base_url = self.config.jina_base_url or os.getenv("JINA_API_BASE") or "https://api.jina.ai/v1/embeddings"
        
        # Set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def embed(self, text: str, memory_action: Optional[Literal["add", "search", "update"]] = None) -> List[float]:
        """
        Get the embedding for the given text using Jina AI.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """
        text = text.replace("\n", " ")
        
        # Create request payload
        payload = {
            "model": self.config.model,
            "input": [text],
            "dimensions": self.config.embedding_dims
        }
        
        # Make API request
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Error from Jina API: {response.status_code}, {response.text}")
        
        response_data = response.json()
        return response_data["data"][0]["embedding"] 