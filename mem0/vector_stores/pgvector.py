import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

try:
    import psycopg2
    from psycopg2.extras import Json, execute_values
except ImportError:
    raise ImportError("The 'psycopg2' library is required. Please install it using 'pip install psycopg2'.")

try:
    import pgvector.sqlalchemy  # noqa: F401
except ImportError:
    raise ImportError("The 'pgvector' library is required. Please install it using 'pip install pgvector'.")

try:
    from sentence_transformers.quantization import quantize_embeddings
except ImportError:
    # Fail silently if sentence-transformers not installed, quantization just won't work
    quantize_embeddings = None

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class PGVector(VectorStoreBase):
    """
    VectorStoreBase implementation using PostgreSQL with pgvector extension.

    This class handles connection, table/index creation, data insertion,
    updates, deletions, and similarity searches.
    It supports optional binary quantization using the sentence-transformers library.

    Attributes:
        collection_name (str): Name of the table used for storage.
        embedding_model_dims (int): Original dimension of the float embeddings.
        quantization_config (Optional[Dict]): Configuration for quantization.
        quantization_precision (Optional[str]): Precision type ('binary', 'ubinary') if quantized.
        use_diskann (bool): Whether DiskANN index is requested (ignored if quantized).
        use_hnsw (bool): Whether HNSW index is requested.
        conn: Active psycopg2 connection.
        cur: Active psycopg2 cursor.
    """
    def __init__(
        self,
        dbname,
        collection_name,
        embedding_model_dims,
        user,
        password,
        host,
        port,
        diskann,
        hnsw,
        quantization: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the PGVector database connection and ensures the collection exists.

        Args:
            dbname: Name of the PostgreSQL database.
            collection_name: Name for the table storing vectors and metadata.
            embedding_model_dims: The dimension of the *original* float vectors.
            user: Database username.
            password: Database password.
            host: Database host address.
            port: Database port number.
            diskann: If True, attempts to use DiskANN index (float only, requires extension).
            hnsw: If True, attempts to use HNSW index (float or quantized).
            quantization: Optional dict, e.g., {"precision": "binary"} to enable.

        Raises:
            ImportError: If required libraries (psycopg2, pgvector, sentence-transformers for quantization) are missing.
            ValueError: If quantization config is invalid.
            psycopg2.Error: If database connection fails.
        """
        self.collection_name = collection_name
        self.embedding_model_dims = embedding_model_dims
        self.quantization_config = quantization
        self.quantization_precision = quantization.get("precision") if quantization else None

        # Validate quantization config early
        if self.quantization_precision:
            if self.quantization_precision not in ["binary", "ubinary"]:
                # This check is also in config, but good to have here too
                raise ValueError(f"Unsupported quantization precision: {self.quantization_precision}")
            if not quantize_embeddings:
                 raise ImportError(
                    "'sentence-transformers' library is required for quantization. "
                    "Please install it using 'pip install sentence-transformers'"
                )
            # If quantization is enabled, HNSW is preferred if available, DiskANN likely won't work
            self.use_diskann = False # DiskANN unlikely to support BIT type
            self.use_hnsw = hnsw # User can still request HNSW
            logger.info(f"Quantization enabled ({self.quantization_precision}). DiskANN disabled.")
        else:
            # Use user-provided index prefs if not quantizing
            self.use_diskann = diskann
            self.use_hnsw = hnsw

        try:
            self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            self.cur = self.conn.cursor()
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

        self._ensure_collection_exists()
        
    def create_col(self, name=None, vector_size=None, distance=None):
        """
        Create a new collection.
        
        This is a placeholder to satisfy the VectorStoreBase abstract method requirement.
        PGVector handles collection creation in __init__ via _ensure_collection_exists.
        """
        logger.info("PGVector create_col called (no-op)")
        pass

    def _get_vector_type_and_ops(self) -> tuple[str, str, str]:
        """
        Determines SQL vector type, index operator class, and distance operator.

        Based on whether quantization is enabled in the configuration.

        Returns:
            A tuple containing (sql_vector_type, index_operator_class, distance_operator).
        """
        if self.quantization_precision:
            vector_type = f"BIT({self.embedding_model_dims})"
            index_opclass = "bit_hamming_ops"
            distance_operator = "<~>"
        else:
            vector_type = f"VECTOR({self.embedding_model_dims})"
            index_opclass = "vector_cosine_ops"
            distance_operator = "<=>"
        return vector_type, index_opclass, distance_operator

    def _ensure_collection_exists(self):
        """
        Checks if the required table exists, creates it and the index if not.

        Ensures the table schema (VECTOR or BIT) and index match the configuration.

        Raises:
            psycopg2.Error: If there are issues interacting with the database.
        """
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit() # Commit extension creation separately

            vector_type, index_opclass, _ = self._get_vector_type_and_ops()

            # Check if table exists
            self.cur.execute(
                "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s",
                (self.collection_name,)
            )
            table_exists = self.cur.fetchone() is not None

            if not table_exists:
                logger.info(f"Creating table '{self.collection_name}' with vector type {vector_type}.")
                create_table_sql = f"""
                    CREATE TABLE {self.collection_name} (
                        id UUID PRIMARY KEY,
                        vector {vector_type},
                        payload JSONB
                    );
                """
                self.cur.execute(create_table_sql)
                self.conn.commit() # Commit table creation
                self._create_index_if_needed(index_opclass)
            else:
                # TODO: Optionally verify existing schema matches config
                logger.debug(f"Table '{self.collection_name}' already exists.")
                self._create_index_if_needed(index_opclass) # Ensure index exists even if table does

        except psycopg2.Error as e:
            logger.error(f"Error during collection/index setup for '{self.collection_name}': {e}")
            self.conn.rollback() # Rollback any partial changes on error
            raise

    def _create_index_if_needed(self, index_opclass: str):
        """
        Creates HNSW or DiskANN index based on config if it doesn't exist.

        Logs warnings if requested index type isn't supported or available.

        Args:
            index_opclass: The appropriate index operator class (e.g.,
                vector_cosine_ops, bit_hamming_ops).
        """
        index_name = f"{self.collection_name}_idx"
        index_type = None
        if self.use_hnsw:
            index_type = "hnsw"
        elif self.use_diskann and not self.quantization_precision:
             # Check if vectorscale extension is installed only if DiskANN is viable
            self.cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'")
            if self.cur.fetchone():
                index_type = "diskann"
            else:
                logger.warning("DiskANN requested but 'vectorscale' extension not found. Will not create DiskANN index.")

        if not index_type:
            logger.debug("No specific index type (DiskANN/HNSW) requested or applicable.")
            return

        # Check if index exists
        self.cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (index_name,))
        if self.cur.fetchone():
            logger.debug(f"Index '{index_name}' already exists.")
            return

        logger.info(f"Creating index '{index_name}' using {index_type.upper()} with opclass {index_opclass}.")
        create_index_sql = f"""
            CREATE INDEX {index_name}
            ON {self.collection_name}
            USING {index_type} (vector {index_opclass});
        """
        try:
            self.cur.execute(create_index_sql)
            self.conn.commit() # Commit index creation
        except psycopg2.Error as e:
            logger.error(f"Failed to create index '{index_name}': {e}")
            self.conn.rollback() # Rollback failed index creation
            # Don't raise here, maybe index creation isn't critical

    def _maybe_quantize(self, vectors: List[List[float]]) -> np.ndarray | List[List[float]]:
        """
        Applies binary quantization to float vectors if configured.

        Args:
            vectors: A list of float vector lists.

        Returns:
            A numpy array of quantized vectors (int8 or uint8) if quantization is
            enabled, otherwise the original list of float vectors.
        """
        if self.quantization_precision:
            float_vectors_np = np.array(vectors, dtype=np.float32)
            logger.debug(
                f"Quantizing {len(float_vectors_np)} vectors to precision '{self.quantization_precision}'..."
            )
            quantized = quantize_embeddings(float_vectors_np, precision=self.quantization_precision)
            logger.debug(f"Quantized vectors dtype: {quantized.dtype}, shape: {quantized.shape}")
            # pgvector expects the numpy array directly for BIT type
            return quantized
        else:
            return vectors # Return original list for VECTOR type

    def insert(self, vectors: List[List[float]], payloads: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """
        Inserts vectors and payloads into the collection.

        Quantizes vectors before insertion if binary quantization is enabled.
        Uses ON CONFLICT DO NOTHING to ignore duplicates based on ID.

        Args:
            vectors: List of float vectors.
            payloads: Corresponding list of metadata dictionaries.
            ids: Corresponding list of unique string IDs.

        Raises:
            ValueError: If lists have mismatched lengths or are missing.
            psycopg2.Error: If database insertion fails.
        """
        if not vectors:
            logger.warning("Insert called with no vectors.")
            return
        if not ids or len(ids) != len(vectors):
            raise ValueError("IDs list must be provided and match the length of vectors.")
        if not payloads or len(payloads) != len(vectors):
            raise ValueError("Payloads list must be provided and match the length of vectors.")

        logger.info(f"Processing {len(vectors)} vectors for insertion into '{self.collection_name}'")
        processed_vectors = self._maybe_quantize(vectors)
        json_payloads = [json.dumps(payload) for payload in payloads]

        data_to_insert = list(zip(ids, processed_vectors, json_payloads))

        try:
            with self.conn.cursor() as cur: # Use context manager for cursor
                 # Use %s placeholder for all data types, including vectors (BIT or VECTOR)
                # pgvector's psycopg2 integration handles the type adaptation.
                execute_values(
                    cur,
                    f"INSERT INTO {self.collection_name} (id, vector, payload) VALUES %s ON CONFLICT (id) DO NOTHING",
                    data_to_insert,
                    template="(%s, %s, %s)",
                    page_size=100 # Optional: Adjust page size for large inserts
                )
            self.conn.commit()
            logger.info(f"Successfully inserted/ignored {len(data_to_insert)} vectors.")
        except psycopg2.Error as e:
            logger.error(f"Error during vector insertion: {e}")
            self.conn.rollback()
            raise

    def search(self, query: str, vectors: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[OutputData]:
        """
        Performs similarity search using the appropriate distance metric.

        Uses Hamming distance (<~>) for quantized (BIT) vectors and Cosine
        distance (<=>) for float (VECTOR) vectors.

        Args:
            query: Original query text (unused, for context).
            vectors: The float query vector.
            limit: Maximum number of results to return.
            filters: Dictionary for filtering based on payload keys.

        Returns:
            A list of OutputData objects containing ID, score (distance),
            and payload, ordered by ascending distance. Returns empty list on error.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                # Ensure filter values are strings for JSONB ->> operator
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        _, _, distance_operator = self._get_vector_type_and_ops()

        # pgvector handles float query vector vs BIT/VECTOR column with appropriate operator
        sql = f"""
            SELECT id, vector {distance_operator} %s AS distance, payload
            FROM {self.collection_name}
            {filter_clause}
            ORDER BY distance ASC
            LIMIT %s
        """

        # Query vector is always float32, pgvector handles comparison
        query_params = (vectors, *filter_params, limit)

        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, query_params)
                results = cur.fetchall()
            # Score is the distance (lower is better for Hamming/Jaccard/Cosine)
            return [OutputData(id=str(r[0]), score=float(r[1]), payload=r[2]) for r in results]
        except psycopg2.Error as e:
            logger.error(f"Error during vector search: {e}")
            self.conn.rollback() # Rollback if transaction was started implicitly
            return [] # Return empty list on error

    def delete(self, vector_id: str):
        """
        Deletes a vector by its ID.

        Args:
            vector_id: The UUID string of the vector to delete.

        Raises:
            psycopg2.Error: If the database deletion fails.
        """
        logger.info(f"Deleting vector '{vector_id}' from '{self.collection_name}'")
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.collection_name} WHERE id = %s", (vector_id,))
                deleted_count = cur.rowcount
            self.conn.commit()
            if deleted_count == 0:
                 logger.warning(f"Attempted to delete non-existent vector ID: {vector_id}")
        except psycopg2.Error as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            self.conn.rollback()
            raise

    def update(self, vector_id: str, vector: Optional[List[float]] = None, payload: Optional[Dict] = None):
        """
        Updates the vector and/or payload for a given ID.

        Quantizes the new vector if binary quantization is enabled.

        Args:
            vector_id: The UUID string of the vector to update.
            vector: The new float vector (optional).
            payload: The new metadata dictionary (optional).

        Raises:
            psycopg2.Error: If the database update fails.
        """
        if not vector and not payload:
            logger.warning(f"Update called for vector '{vector_id}' with no changes.")
            return

        logger.info(f"Updating vector '{vector_id}' in '{self.collection_name}'")
        updates = []
        params = []

        if vector:
            processed_vector = self._maybe_quantize([vector])[0]
            updates.append("vector = %s")
            params.append(processed_vector)
        if payload:
            updates.append("payload = %s")
            params.append(Json(payload))

        params.append(vector_id)
        update_sql = f"UPDATE {self.collection_name} SET {', '.join(updates)} WHERE id = %s"

        try:
            with self.conn.cursor() as cur:
                cur.execute(update_sql, tuple(params))
                updated_count = cur.rowcount
            self.conn.commit()
            if updated_count == 0:
                logger.warning(f"Attempted to update non-existent vector ID: {vector_id}")
        except psycopg2.Error as e:
            logger.error(f"Error updating vector {vector_id}: {e}")
            self.conn.rollback()
            raise

    def get(self, vector_id: str) -> Optional[OutputData]:
        """
        Retrieves the payload for a specific vector ID.

        Note: Does not retrieve the actual vector data for efficiency.

        Args:
            vector_id: The UUID string of the vector.

        Returns:
            An OutputData object with id and payload, or None if not found or on error.
        """
        try:
            with self.conn.cursor() as cur:
                 # Exclude the vector column itself from the result for efficiency
                cur.execute(f"SELECT id, payload FROM {self.collection_name} WHERE id = %s", (vector_id,))
                result = cur.fetchone()
            if not result:
                return None
            # Note: Vector is not included in the returned OutputData
            return OutputData(id=str(result[0]), score=None, payload=result[1])
        except psycopg2.Error as e:
            logger.error(f"Error getting vector {vector_id}: {e}")
            self.conn.rollback()
            return None

    def list_cols(self) -> List[str]:
        """
        Lists all tables in the 'public' schema of the connected database.

        Returns:
            A list of table names, or an empty list on error.
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                return [row[0] for row in cur.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Error listing collections: {e}")
            self.conn.rollback()
            return []

    def delete_col(self):
        """
        Deletes the entire collection table. This action is irreversible.

        Raises:
            psycopg2.Error: If the DROP TABLE command fails.
        """
        logger.warning(f"Deleting collection (table) '{self.collection_name}'. This is irreversible.")
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            self.conn.commit()
            logger.info(f"Collection '{self.collection_name}' deleted.")
        except psycopg2.Error as e:
            logger.error(f"Error deleting collection '{self.collection_name}': {e}")
            self.conn.rollback()
            raise

    def col_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the row count and estimated total size of the collection table.

        Returns:
            A dictionary with 'name', 'count', and 'size' (pretty string),
            or None if the table doesn't exist or on error.
        """
        try:
            with self.conn.cursor() as cur:
                # Get row count
                cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
                row_count = cur.fetchone()[0]

                # Get total size
                cur.execute("SELECT pg_size_pretty(pg_total_relation_size(%s::regclass))", (self.collection_name,))
                total_size = cur.fetchone()[0]

            return {"name": self.collection_name, "count": row_count, "size": total_size}
        except psycopg2.ProgrammingError:
            # Table might not exist
            logger.warning(f"Collection '{self.collection_name}' not found for col_info.")
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting collection info for '{self.collection_name}': {e}")
            self.conn.rollback()
            return None

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[List[OutputData]]:
        """
        Lists entries from the collection, optionally applying filters.

        Note: Does not retrieve the vector data itself.

        Args:
            filters: Dictionary for filtering based on payload keys.
            limit: Maximum number of entries to return.

        Returns:
            A list containing one list of OutputData objects (matching the
            format of other vector stores), or `[[]]` on error.
        """
        filter_conditions = []
        filter_params = []

        if filters:
            for k, v in filters.items():
                filter_conditions.append("payload->>%s = %s")
                filter_params.extend([k, str(v)])

        filter_clause = "WHERE " + " AND ".join(filter_conditions) if filter_conditions else ""

        query = f"""
            SELECT id, payload
            FROM {self.collection_name}
            {filter_clause}
            LIMIT %s
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (*filter_params, limit))
                results = cur.fetchall()
             # Return format matches other vector stores [[OutputData,...]]
            return [[OutputData(id=str(r[0]), score=None, payload=r[1]) for r in results]]
        except psycopg2.Error as e:
            logger.error(f"Error listing vectors: {e}")
            self.conn.rollback()
            return [[]] # Return empty list structure on error

    def __del__(self):
        """
        Ensures database connection and cursor are closed when the object is deleted.
        Logs errors if closing fails but does not raise exceptions.
        """
        if hasattr(self, "cur") and self.cur and not self.cur.closed:
            try:
                self.cur.close()
            except psycopg2.Error as e:
                 logger.error(f"Error closing cursor: {e}")
        if hasattr(self, "conn") and self.conn and not self.conn.closed:
            try:
                self.conn.close()
            except psycopg2.Error as e:
                logger.error(f"Error closing connection: {e}")
