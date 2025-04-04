from unittest.mock import Mock, patch

import pytest

from mem0.utils.factory import EmbedderFactory


@pytest.fixture
def reset_factory_cache():
    """Fixture to reset the EmbedderFactory cache between tests"""
    EmbedderFactory._instances = {}
    yield
    EmbedderFactory._instances = {}


def test_embedder_factory_caching(reset_factory_cache):
    """Test that EmbedderFactory properly caches instances with the same configuration"""
    # Mock the embedder creation to avoid actual model loading
    with patch("mem0.utils.factory.load_class") as mock_load_class:
        # Create a unique mock for each call to load_class
        mock_embedder_class = Mock()
        mock_instance1 = Mock(name="instance1")
        mock_instance2 = Mock(name="instance2")
        
        # Make the mock embedder class return different instances
        mock_embedder_class.side_effect = [mock_instance1, mock_instance2]
        mock_load_class.return_value = mock_embedder_class
        
        # Create an embedder with a specific configuration
        config = {"model": "test-model", "model_kwargs": {"device": "cpu"}}
        provider = "huggingface"
        
        # First call should create a new instance
        instance1 = EmbedderFactory.create(provider, config)
        
        # Second call with the same config should return the cached instance
        instance2 = EmbedderFactory.create(provider, config)
        
        # Verify both instances are the same object (from cache)
        assert instance1 is instance2
        
        # Verify the embedder class was instantiated only once
        assert mock_embedder_class.call_count == 1
        
        # Change the config slightly
        different_config = {"model": "test-model", "model_kwargs": {"device": "cuda"}}
        
        # This should create a new instance
        instance3 = EmbedderFactory.create(provider, different_config)
        
        # Verify instance3 is different from instance1
        assert instance1 is not instance3
        
        # The embedder class should be called twice (once for each unique config)
        assert mock_embedder_class.call_count == 2


def test_embedder_factory_different_providers(reset_factory_cache):
    """Test that EmbedderFactory creates different instances for different providers"""
    with patch("mem0.utils.factory.load_class") as mock_load_class:
        # Create unique mocks for each provider
        mock_embedder_class = Mock()
        mock_instance1 = Mock(name="hf_instance")
        mock_instance2 = Mock(name="openai_instance")
        
        # Make the mock embedder class return different instances
        mock_embedder_class.side_effect = [mock_instance1, mock_instance2]
        mock_load_class.return_value = mock_embedder_class
        
        config = {"model": "test-model"}
        
        # Create embedders with different providers but same config
        instance1 = EmbedderFactory.create("huggingface", config)
        instance2 = EmbedderFactory.create("openai", config)
        
        # Verify they are different instances
        assert instance1 is not instance2
        
        # Verify the embedder class was instantiated twice (once for each provider)
        assert mock_embedder_class.call_count == 2 