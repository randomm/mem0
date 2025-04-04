from unittest.mock import Mock, patch

from mem0.configs.base import MemoryConfig
from mem0.graphs.configs import Neo4jConfig
from mem0.memory.main import Memory
from mem0.utils.factory import EmbedderFactory


def test_embedding_model_sharing():
    """Test that the embedding model is properly shared between Memory and MemoryGraph."""
    with patch("mem0.memory.graph_memory.Neo4jGraph"), \
         patch.object(EmbedderFactory, 'create') as mock_embedder_factory, \
         patch("mem0.memory.main.LlmFactory"), \
         patch("mem0.memory.main.VectorStoreFactory"), \
         patch("mem0.memory.main.SQLiteManager"), \
         patch("mem0.memory.main.capture_event"):
        
        # Create a mock embedder
        mock_embedder = Mock(name="MockEmbedder")
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        
        # Setup the mock factory to return our mock embedder
        mock_embedder_factory.return_value = mock_embedder
        
        # Create a proper Neo4j config
        neo4j_config = Neo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        
        # Create the memory config with graph store config
        config = MemoryConfig()
        config.graph_store.config = neo4j_config
        
        # Initialize Memory instance
        memory = Memory(config)
        
        # Verify the embedder factory was called once for the Memory instance
        assert mock_embedder_factory.call_count == 1
        
        # Verify both Memory and MemoryGraph use the same embedding model instance
        assert memory.embedding_model is memory.graph.embedding_model
        
        # The MemoryGraph should NOT have called the factory again
        assert mock_embedder_factory.call_count == 1 