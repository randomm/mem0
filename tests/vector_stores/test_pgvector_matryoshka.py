import pytest
import numpy as np
import os
import uuid
import logging
from unittest.mock import patch, MagicMock

# Try importing dependencies and skip tests if unavailable
try:
    import psycopg2
    from psycopg2.extras import Json
    from mem0.configs.vector_stores.pgvector import PGVectorConfig
    from mem0.vector_stores.pgvector import PGVector
except ImportError:
    pytest.skip("psycopg2 or mem0 components not available, skipping pgvector tests", allow_module_level=True)

# Mock connection/cursor for tests without a live DB
@pytest.fixture
def mock_pg_connection():
    """Provides a mocked psycopg2 connection and cursor."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    # Mock fetchone to simulate table existence checks or other single-row results
    mock_cur.fetchone.return_value = None # Default to table not existing
    # Mock fetchall for search results
    mock_cur.fetchall.return_value = []
    
    with patch('psycopg2.connect', return_value=mock_conn):
        yield mock_conn, mock_cur


# --- Configuration Tests ---

def test_pgvector_config_validation_matryoshka():
    """Test PGVectorConfig validation for matryoshka_dims."""
    base_config = {
        "user": "test", "password": "test", "host": "localhost", "port": 5432
    }
    
    # Valid configuration
    config = PGVectorConfig(
        embedding_model_dims=1024, 
        matryoshka_dims=512, 
        **base_config
    )
    assert config.embedding_model_dims == 1024
    assert config.matryoshka_dims == 512
    
    # Valid: matryoshka_dims not provided
    config_no_matryoshka = PGVectorConfig(
        embedding_model_dims=1024, 
        **base_config
    )
    assert config_no_matryoshka.matryoshka_dims is None

    # Invalid: matryoshka_dims > embedding_model_dims
    with pytest.raises(ValueError, match="must be less than embedding_model_dims"):
        PGVectorConfig(embedding_model_dims=512, matryoshka_dims=1024, **base_config)
        
    # Invalid: matryoshka_dims == embedding_model_dims
    with pytest.raises(ValueError, match="must be less than embedding_model_dims"):
        PGVectorConfig(embedding_model_dims=512, matryoshka_dims=512, **base_config)

# --- Implementation Tests ---

@pytest.fixture
def pg_vector_matryoshka_instance(mock_pg_connection):
    """Provides a PGVector instance configured for matryoshka truncation."""
    mock_conn, mock_cur = mock_pg_connection
    # Simulate table not existing initially
    mock_cur.fetchone.side_effect = [None] # First check finds no table
    
    config = PGVectorConfig(
        dbname="test_db",
        collection_name="matryoshka_test_col",
        embedding_model_dims=1024,
        matryoshka_dims=512,
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        diskann=False,
        hnsw=False,
    )
    store = PGVector(**config.model_dump())
    return store, mock_cur

def test_table_creation_with_matryoshka(pg_vector_matryoshka_instance):
    """Test that the table is created with the truncated dimension."""
    store, mock_cur = pg_vector_matryoshka_instance
    
    # Check the CREATE TABLE statement execution call
    create_table_call = None
    for call in mock_cur.execute.call_args_list:
        if "CREATE TABLE matryoshka_test_col" in call[0][0]:
            create_table_call = call
            break
            
    assert create_table_call is not None, "CREATE TABLE statement not executed"
    # Verify the vector column uses the truncated dimension (512)
    assert "vector vector(512)" in create_table_call[0][0]
    # Verify the original dimension (1024) is NOT used for the column definition
    assert "vector(1024)" not in create_table_call[0][0]

def test_truncation_method_logic(pg_vector_matryoshka_instance):
    """Test the internal _maybe_truncate_embedding method directly."""
    store, _ = pg_vector_matryoshka_instance
    store.quantization_precision = None # Ensure quantization is off for this test
    store.matryoshka_dims = 512

    vectors_full = [[1.0] * 1024, [2.0] * 1024]
    truncated = store._maybe_truncate_embedding(vectors_full)
    
    assert len(truncated) == 2
    assert len(truncated[0]) == 512
    assert len(truncated[1]) == 512
    assert truncated[0] == [1.0] * 512
    assert truncated[1] == [2.0] * 512

def test_insert_truncates_vectors(pg_vector_matryoshka_instance):
    """Test that the insert method correctly truncates vectors before insertion."""
    store, mock_cur = pg_vector_matryoshka_instance
    store.quantization_precision = None # Ensure quantization is off

    vector_id = str(uuid.uuid4())
    full_vector = [1.5] * 1024
    payload = {"data": "test data"}

    store.insert(vectors=[full_vector], payloads=[payload], ids=[vector_id])

    # Find the execute_values call
    insert_call = None
    for call in mock_cur.method_calls:
        if call[0] == 'execute_values':
            insert_call = call
            break
            
    assert insert_call is not None, "execute_values not called for insert"
    
    # Check the data passed to execute_values
    inserted_data = insert_call[1][1] # Second argument to execute_values
    assert len(inserted_data) == 1
    assert inserted_data[0][0] == vector_id # Check ID
    assert len(inserted_data[0][1]) == 512 # Check truncated vector dimension
    assert inserted_data[0][1] == [1.5] * 512 # Check truncated vector content
    assert inserted_data[0][2] == json.dumps(payload) # Check payload

def test_search_truncates_query_vector(pg_vector_matryoshka_instance):
    """Test that the search method correctly truncates the query vector."""
    store, mock_cur = pg_vector_matryoshka_instance
    store.quantization_precision = None # Ensure quantization is off

    query_vector_full = [3.0] * 1024
    
    # Mock search result
    mock_cur.fetchall.return_value = [(str(uuid.uuid4()), 0.1, {"data": "result"})]
    
    store.search(query="test query", vectors=query_vector_full, limit=1)
    
    # Find the search execute call
    search_execute_call = None
    for call in mock_cur.execute.call_args_list:
        # Check for the specific SQL pattern used in search
        if "SELECT id, vector <=> %s::vector AS distance, payload" in call[0][0]:
            search_execute_call = call
            break
            
    assert search_execute_call is not None, "Search query execution not found"
    
    # Check the parameters passed to execute
    query_params = search_execute_call[1][0] # Second element of the tuple is params
    assert len(query_params) >= 1 # Should have at least the vector param
    passed_vector = query_params[0] # First param is the query vector
    assert isinstance(passed_vector, list)
    assert len(passed_vector) == 512 # Dimension should be truncated
    assert passed_vector == [3.0] * 512 # Content should be truncated

@patch('mem0.vector_stores.pgvector.logger')
def test_matryoshka_and_quantization_warning(mock_logger, mock_pg_connection):
    """Test that a warning is logged if both matryoshka and quantization are set."""
    mock_conn, mock_cur = mock_pg_connection
    mock_cur.fetchone.side_effect = [None] # Simulate table not existing
    
    config_dict = {
        "dbname": "test_db",
        "collection_name": "quant_matryoshka_test",
        "embedding_model_dims": 1024,
        "user": "test_user",
        "password": "test_password",
        "host": "localhost",
        "port": 5432,
        "diskann": False,
        "hnsw": True,
        "quantization": {"precision": "binary"},
        "matryoshka_dims": 512
    }
    
    # Patch quantize_embeddings to avoid ImportError if sentence-transformers not installed
    with patch('mem0.vector_stores.pgvector.quantize_embeddings', MagicMock()):
        config = PGVectorConfig(**config_dict)
        store = PGVector(**config.model_dump())

    mock_logger.warning.assert_called_once()
    assert "Quantization will take precedence" in mock_logger.warning.call_args[0][0]

# TODO: Add tests for end-to-end workflow with a mocked HuggingFaceEmbedder
# Requires mocking the embedder factory and the embedder itself. 