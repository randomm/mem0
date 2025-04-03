import logging
import os
import uuid

import pytest

# Try importing dependencies and skip tests if unavailable
try:
    import psycopg2

    from mem0.configs.vector_stores.pgvector import PGVectorConfig
    from mem0.vector_stores.pgvector import PGVector
except ImportError:
    pytest.skip("psycopg2 or mem0 components not available, skipping pgvector tests", allow_module_level=True)

try:
    import sentence_transformers.quantization  # noqa: F401
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Environment variable names for PostgreSQL connection
DB_USER = os.getenv("PG_USER", "test_user")
DB_PASSWORD = os.getenv("PG_PASSWORD", "test_password")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")
DB_NAME = os.getenv("PG_DBNAME", "test_db_mem0")

# Check if DB connection is possible
CONN_AVAILABLE = False
try:
    conn_test = psycopg2.connect(dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    conn_test.autocommit = True
    cur_test = conn_test.cursor()
    # Create test database if it doesn't exist
    cur_test.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if not cur_test.fetchone():
        cur_test.execute(f"CREATE DATABASE {DB_NAME}")
        logging.info(f"Created test database: {DB_NAME}")
    # Check for pgvector extension in the target database
    conn_target = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
    cur_target = conn_target.cursor()
    cur_target.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn_target.commit()
    cur_test.close()
    conn_test.close()
    cur_target.close()
    conn_target.close()
    CONN_AVAILABLE = True
    logging.info(f"Connection to PostgreSQL ({DB_HOST}:{DB_PORT}/{DB_NAME}) successful.")
except psycopg2.Error as e:
    logging.warning(f"PostgreSQL connection failed: {e}. Skipping pgvector tests that require DB connection.")
    CONN_AVAILABLE = False
except Exception as e:
     logging.warning(f"An unexpected error occurred during DB setup: {e}. Skipping pgvector tests.")
     CONN_AVAILABLE = False

# Test data
DIMENSIONS = 4 # Small dimension for testing
COLLECTION_NAME_FLOAT = "test_mem0_float_collection"
COLLECTION_NAME_BINARY = "test_mem0_binary_collection"

FLOAT_VECTORS = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
PAYLOADS = [{"meta": "payload1"}, {"meta": "payload2"}]
IDS = [str(uuid.uuid4()), str(uuid.uuid4())]

@pytest.fixture(scope="module")
def base_config():
    return {
        "user": DB_USER,
        "password": DB_PASSWORD,
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "embedding_model_dims": DIMENSIONS,
    }

@pytest.fixture
def float_vector_store(base_config):
    "Provides a PGVector instance configured for float vectors and cleans up after."""
    if not CONN_AVAILABLE:
        pytest.skip("DB connection not available")

    config_dict = {**base_config, "collection_name": COLLECTION_NAME_FLOAT, "quantization": None, "hnsw": False}
    config = PGVectorConfig(**config_dict)
    store = PGVector(**config.model_dump())
    yield store
    # Cleanup
    try:
        store.delete_col()
    except Exception as e:
        logging.error(f"Error cleaning up float collection: {e}")

@pytest.fixture
def binary_vector_store(base_config):
    "Provides a PGVector instance configured for binary vectors and cleans up after."""
    if not CONN_AVAILABLE:
        pytest.skip("DB connection not available")
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        pytest.skip("sentence-transformers not available for quantization")

    config_dict = {
        **base_config,
        "collection_name": COLLECTION_NAME_BINARY,
        "quantization": {"precision": "binary"},
        "hnsw": True # Test HNSW with binary
    }
    config = PGVectorConfig(**config_dict)
    store = PGVector(**config.model_dump())
    yield store
    # Cleanup
    try:
        store.delete_col()
    except Exception as e:
        logging.error(f"Error cleaning up binary collection: {e}")

# --- Test Functions --- #

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
def test_pgvector_init_float(float_vector_store):
    assert float_vector_store is not None
    assert float_vector_store.collection_name == COLLECTION_NAME_FLOAT
    assert COLLECTION_NAME_FLOAT in float_vector_store.list_cols()
    # Check if table uses VECTOR type (indirectly by checking index or table definition if needed)
    # For simplicity, we rely on insert/search working as expected for float

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
def test_pgvector_init_binary(binary_vector_store):
    assert binary_vector_store is not None
    assert binary_vector_store.collection_name == COLLECTION_NAME_BINARY
    assert COLLECTION_NAME_BINARY in binary_vector_store.list_cols()
    # Check if table uses BIT type (requires inspecting schema)
    try:
        binary_vector_store.cur.execute(
            "SELECT data_type FROM information_schema.columns WHERE table_name = %s AND column_name = 'vector'",
            (COLLECTION_NAME_BINARY,)
        )
        col_type = binary_vector_store.cur.fetchone()
        assert col_type and col_type[0] == 'bit varying' # pgvector uses 'bit varying' internally for BIT()
    except psycopg2.Error as e:
        pytest.fail(f"Failed to check column type: {e}")

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
def test_insert_and_search_float(float_vector_store):
    float_vector_store.insert(vectors=FLOAT_VECTORS, ids=IDS, payloads=PAYLOADS)

    # Search for vector similar to the first one
    query_vector = [1.1, 2.1, 3.1, 4.1]
    results = float_vector_store.search(query="test query", vectors=query_vector, limit=1)

    assert len(results) == 1
    assert results[0].id == IDS[0]
    assert results[0].payload == PAYLOADS[0]
    assert results[0].score < 0.1 # Cosine distance should be small

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
def test_insert_and_search_binary(binary_vector_store):
    # Insertion requires float vectors, quantization happens internally
    binary_vector_store.insert(vectors=FLOAT_VECTORS, ids=IDS, payloads=PAYLOADS)

    # Search requires float query vector
    query_vector = [1.1, 2.1, 3.1, 4.1] # Similar to first vector
    results = binary_vector_store.search(query="test query", vectors=query_vector, limit=1)

    assert len(results) == 1
    assert results[0].id == IDS[0]
    assert results[0].payload == PAYLOADS[0]
    assert results[0].score is not None
    # Hamming distance - exact value depends on quantization, expect low integer
    assert isinstance(results[0].score, float) and results[0].score >= 0
    logging.info(f"Binary search result score (Hamming): {results[0].score}")

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers not available")
def test_update_binary(binary_vector_store):
    # Insert initial data
    binary_vector_store.insert(vectors=[FLOAT_VECTORS[0]], ids=[IDS[0]], payloads=[PAYLOADS[0]])

    # Update vector and payload
    new_vector_float = [9.0, 9.0, 9.0, 9.0]
    new_payload = {"meta": "updated_payload"}
    binary_vector_store.update(vector_id=IDS[0], vector=new_vector_float, payload=new_payload)

    # Verify update
    retrieved = binary_vector_store.get(vector_id=IDS[0])
    assert retrieved is not None
    assert retrieved.id == IDS[0]
    assert retrieved.payload == new_payload

    # Verify search reflects update (search for the new vector)
    results = binary_vector_store.search(query="search new", vectors=new_vector_float, limit=1)
    assert len(results) == 1
    assert results[0].id == IDS[0]
    assert results[0].score == 0.0 # Hamming distance should be 0 for exact match

@pytest.mark.skipif(not CONN_AVAILABLE, reason="DB connection not available")
def test_delete_float(float_vector_store):
    float_vector_store.insert(vectors=[FLOAT_VECTORS[0]], ids=[IDS[0]], payloads=[PAYLOADS[0]])
    assert float_vector_store.get(IDS[0]) is not None
    float_vector_store.delete(vector_id=IDS[0])
    assert float_vector_store.get(IDS[0]) is None

# Test config validation
def test_pgvector_config_validation(base_config):
    # Valid binary config
    valid_binary_config = {**base_config, "collection_name": "vbc", "quantization": {"precision": "binary"}}
    PGVectorConfig(**valid_binary_config)

    # Valid ubinary config
    valid_ubinary_config = {**base_config, "collection_name": "vuc", "quantization": {"precision": "ubinary"}}
    PGVectorConfig(**valid_ubinary_config)

    # Invalid precision
    invalid_precision_config = {**base_config, "collection_name": "ipc", "quantization": {"precision": "int8"}}
    with pytest.raises(ValueError, match="Unsupported precision 'int8'"):
        PGVectorConfig(**invalid_precision_config)

    # Missing precision key
    missing_precision_config = {**base_config, "collection_name": "mpc", "quantization": {"other_key": "value"}}
    with pytest.raises(ValueError, match="'precision' key is required"):
        PGVectorConfig(**missing_precision_config) 