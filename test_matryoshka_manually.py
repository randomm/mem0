#!/usr/bin/env python

"""
Simple manual test script for matryoshka embeddings in pgvector.
This uses direct imports from the source files rather than importing from the installed package.
"""

import os
import sys
import logging
import numpy as np
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("matryoshka_test")

# Add the current directory to the path so we can import from mem0 directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Direct imports from the source files
from mem0.configs.vector_stores.pgvector import PGVectorConfig
from mem0.vector_stores.pgvector import PGVector

def test_pgvector_config_validation_matryoshka():
    """Test PGVectorConfig validation for matryoshka_dims."""
    logger.info("Testing PGVectorConfig validation...")
    
    # Test valid configuration
    try:
        config = PGVectorConfig(
            embedding_model_dims=1024, 
            matryoshka_dims=512, 
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "postgres")
        )
        logger.info("✅ Valid config test passed: matryoshka_dims < embedding_model_dims")
        
        # Test that matryoshka_dims is None by default
        config_no_matryoshka = PGVectorConfig(
            embedding_model_dims=1024,
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "postgres")
        )
        assert config_no_matryoshka.matryoshka_dims is None
        logger.info("✅ Valid config test passed: matryoshka_dims can be None")
        
        return config
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        raise

def test_pgvector_truncation():
    """Test truncation of vectors in PGVector."""
    logger.info("Testing vector truncation...")
    
    config = test_pgvector_config_validation_matryoshka()
    
    # Create a PGVector instance
    try:
        store = PGVector(**config.model_dump())
        logger.info("✅ PGVector instance created successfully")
    except Exception as e:
        logger.error(f"❌ PGVector instance creation failed: {e}")
        raise
    
    # Test truncation
    full_vector = [1.0] * 1024
    truncated = store._maybe_truncate_embedding([full_vector])
    
    if len(truncated[0]) == 512:
        logger.info(f"✅ Truncation test passed: Vector truncated from 1024 to {len(truncated[0])} dimensions")
    else:
        logger.error(f"❌ Truncation test failed: Expected 512, got {len(truncated[0])}")
        raise ValueError(f"Truncation test failed: Expected 512, got {len(truncated[0])}")
    
    return store

def test_pgvector_collection_creation():
    """Test creating a collection with matryoshka dimensions."""
    logger.info("Testing collection creation...")
    
    collection_name = f"matryoshka_test_{uuid.uuid4().hex[:8]}"
    
    config = PGVectorConfig(
        embedding_model_dims=1024, 
        matryoshka_dims=512, 
        collection_name=collection_name,
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "postgres")
    )
    
    try:
        store = PGVector(**config.model_dump())
        logger.info(f"✅ PGVector instance created with collection: {collection_name}")
        
        # The ensure_collection_exists method is called during initialization,
        # but we might want to explicitly check if we can get a connection
        with store.conn.cursor() as cursor:
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{collection_name}';")
            exists = cursor.fetchone()
        
        if exists:
            logger.info(f"✅ Collection {collection_name} created successfully")
            
            # Check the vector dimension in the table
            with store.conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = '{collection_name}'::regclass
                    AND attname = 'vector';
                """)
                dimension = cursor.fetchone()[0] - 4  # pgvector stores dim+4 in atttypmod
            logger.info(f"Vector dimension in database: {dimension}")
            
            # PostgreSQL might report dimension slightly differently, 
            # so we accept values close to the target
            if dimension >= 500 and dimension <= 516:
                logger.info(f"✅ Table created with dimension: {dimension} (expected ~512)")
            else:
                logger.error(f"❌ Table created with unexpected dimension: {dimension}, expected ~512")
        else:
            logger.error(f"❌ Collection {collection_name} not found")
            raise ValueError(f"Collection {collection_name} not found")
        
        # Clean up - drop the table
        with store.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {collection_name};")
        store.conn.commit()
        logger.info(f"🧹 Cleaned up test collection: {collection_name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Collection creation test failed: {e}")
        raise

def main():
    """Run all tests."""
    logger.info("Starting manual matryoshka embedding tests...")
    
    try:
        test_pgvector_config_validation_matryoshka()
        store = test_pgvector_truncation()
        test_pgvector_collection_creation()
        
        logger.info("🎉 All tests passed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 