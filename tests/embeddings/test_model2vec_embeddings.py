import pytest

from mem0.configs.embeddings.model2vec import Model2VecEmbedderConfig
from mem0.embeddings.model2vec import Model2VecEmbedder

MODEL_NAME = "CISCai/jina-embeddings-v3-separation-distilled"
EXPECTED_DIMENSIONS = 256  # Based on the model name and typical usage


def test_model2vec_initialization():
    """Test initializing the Model2VecEmbedder."""
    config = Model2VecEmbedderConfig(model=MODEL_NAME, embedding_dims=EXPECTED_DIMENSIONS)
    try:
        embedder = Model2VecEmbedder(config=config)
        assert embedder is not None
        assert embedder.embedding_dims == EXPECTED_DIMENSIONS
        assert embedder.config.model == MODEL_NAME
    except ImportError:
        pytest.skip("model2vec not installed, skipping test")
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")


def test_model2vec_embedding_single():
    """Test embedding a single text."""
    config = Model2VecEmbedderConfig(model=MODEL_NAME, embedding_dims=EXPECTED_DIMENSIONS)
    try:
        embedder = Model2VecEmbedder(config=config)
        text = "This is a test sentence."
        embedding = embedder.embed(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 1  # Should return a list containing one embedding
        assert isinstance(embedding[0], list)
        assert len(embedding[0]) == EXPECTED_DIMENSIONS
        assert all(isinstance(x, float) for x in embedding[0])
    except ImportError:
        pytest.skip("model2vec not installed, skipping test")
    except Exception as e:
        pytest.fail(f"Embedding failed: {e}")


def test_model2vec_embedding_multiple():
    """Test embedding multiple texts."""
    config = Model2VecEmbedderConfig(model=MODEL_NAME, embedding_dims=EXPECTED_DIMENSIONS)
    try:
        embedder = Model2VecEmbedder(config=config)
        texts = ["First sentence.", "Second sentence is longer."]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == EXPECTED_DIMENSIONS
            assert all(isinstance(x, float) for x in emb)
    except ImportError:
        pytest.skip("model2vec not installed, skipping test")
    except Exception as e:
        pytest.fail(f"Embedding failed: {e}")

def test_model2vec_initialization_no_model():
    """Test initialization fails without a model name."""
    config = Model2VecEmbedderConfig()
    with pytest.raises(ValueError, match="Model name is required"): # Expect ValueError
        Model2VecEmbedder(config=config)

def test_model2vec_initialization_invalid_model():
    """Test initialization fails with an invalid model name."""
    config = Model2VecEmbedderConfig(model="invalid/model-name-that-does-not-exist")
    try:
        with pytest.raises(Exception): # Expect some exception from model loading
             Model2VecEmbedder(config=config)
    except ImportError:
        pytest.skip("model2vec not installed, skipping test") 