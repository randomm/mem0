"""
Example demonstrating the use of matryoshka embeddings with pgvector in mem0.
This example uses Jina embeddings v3 via Hugging Face and truncates dimensions
to save space and improve performance.

To run this example:
1. Ensure you have PostgreSQL with pgvector extension installed and running.
2. Set environment variables for your PostgreSQL connection:
   - POSTGRES_USER
   - POSTGRES_PASSWORD
   - POSTGRES_HOST (defaults to localhost)
   - POSTGRES_PORT (defaults to 5432)
   - POSTGRES_DB (defaults to mem0_example)
3. Install required libraries: pip install mem0ai psycopg2-binary sentence-transformers
4. Run the script: python examples/pgvector_matryoshka_embeddings.py
"""

import os
import logging
from mem0 import Memory
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Run matryoshka embedding example with pgvector."""
    
    # --- Configuration ---
    # Check for necessary environment variables
    pg_user = os.getenv("POSTGRES_USER")
    pg_password = os.getenv("POSTGRES_PASSWORD")
    if not pg_user or not pg_password:
        logging.error("Please set POSTGRES_USER and POSTGRES_PASSWORD environment variables.")
        return

    # Define target dimensions for truncation
    target_matryoshka_dims = 512
    original_dims = 1024
    collection_name = f"matryoshka_{target_matryoshka_dims}d_example"
    
    logging.info(f"Configuring Mem0 for pgvector with Matryoshka truncation to {target_matryoshka_dims} dimensions.")
    
    config = {
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "jinaai/jina-embeddings-v3", 
                "embedding_dims": original_dims,  # Embedder needs the original dimension
                "model_kwargs": {
                    "trust_remote_code": True  # Required for Jina embeddings model
                }
            }
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "user": pg_user,
                "password": pg_password,
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")), # Ensure port is int
                "dbname": os.getenv("POSTGRES_DB", "mem0_example"),
                "collection_name": collection_name,
                "embedding_model_dims": original_dims,  # Store needs original for validation
                "matryoshka_dims": target_matryoshka_dims, # Actual dimension used in DB
                "hnsw": True # Enable HNSW index for better performance
            }
        }
    }
    
    # --- Initialize Memory --- 
    try:
        logging.info("Initializing Memory...")
        # Clean up previous collection if it exists
        temp_memory_for_cleanup = Memory.from_config(config)
        if collection_name in temp_memory_for_cleanup.vector_store.list_cols():
            logging.warning(f"Deleting existing collection: {collection_name}")
            temp_memory_for_cleanup.vector_store.delete_col()
        del temp_memory_for_cleanup
        
        memory = Memory.from_config(config)
        logging.info(f"Memory initialized. Using collection: {collection_name}, Target Dims: {target_matryoshka_dims}")
    except Exception as e:
        logging.error(f"Failed to initialize Memory: {e}")
        return

    # --- Add Content --- 
    logging.info("\nAdding sample content...")
    sample_texts = [
        "Matryoshka Representation Learning (MRL) allows embeddings to be truncated.",
        "Jina AI's v3 embedding models leverage MRL for flexible dimensions.",
        "Storing lower-dimensional vectors saves space in vector databases like pgvector.",
        "Querying truncated embeddings can improve search speed.",
        "The trade-off is a potential decrease in retrieval accuracy at lower dimensions."
    ]
    
    # Base user ID for testing
    base_user_id = "matryoshka_tester"
    
    try:
        memory_ids = []  # To store added memory IDs
        for idx, text in enumerate(sample_texts):
            # Use idx to ensure uniqueness for each memory
            unique_text = f"Memory {idx+1}: {text}"
            
            # Use a unique user_id for each memory to prevent deduplication
            unique_user_id = f"{base_user_id}_{idx+1}"
            
            # Add the memory with unique text and user_id
            memory.add(unique_text, user_id=unique_user_id, metadata={"doc_id": f"doc_{idx+1}"})
            logging.info(f"Added: '{unique_text}' with user_id: {unique_user_id}")
            
            # Verify memory was added by searching for it
            verification = memory.search(unique_text[:20], user_id=unique_user_id, limit=1)
            if verification:
                logging.info(f"  ✓ Verified memory was stored successfully")
                memory_ids.append(unique_user_id)
            else:
                logging.warning(f"  ⚠ Could not verify memory was stored")
            
            # Count total memories for this user after addition
            all_memories = memory.search("Memory", user_id=unique_user_id, limit=100)
            logging.info(f"  Total memories for user '{unique_user_id}': {len(all_memories)}")
    except Exception as e:
        logging.error(f"Failed to add content: {e}")
        return
        
    # --- Perform Search --- 
    logging.info("\nPerforming similarity search for each user...")
    query = "What are the benefits of matryoshka embeddings?"
    
    all_results = []
    # Search for each user ID separately
    for user_id in memory_ids:
        try:
            results = memory.search(query, user_id=user_id, limit=3)
            logging.info(f"Search results for user '{user_id}':")
            if not results:
                logging.warning("No results found.")
            for idx, result in enumerate(results):
                # Extract content using various approaches
                if isinstance(result, dict):
                    result_id = result.get('id', 'N/A')
                    result_score = result.get('score', 0.0)
                    result_payload = result.get('payload', {})
                    result_content = result.get('content', '') or result.get('memory', '')
                    if not result_content and result_payload and 'data' in result_payload:
                        result_content = result_payload['data']
                else:
                    # Attempt to access attributes safely
                    result_id = getattr(result, 'id', 'N/A')
                    result_score = getattr(result, 'score', 0.0) 
                    result_payload = getattr(result, 'payload', {})
                    # Try multiple potential locations for the content
                    result_content = getattr(result, 'content', None)
                    if not result_content:
                        result_content = getattr(result, 'memory', None)
                    if not result_content and result_payload and 'data' in result_payload:
                        result_content = result_payload['data']
                    if not result_content:
                        result_content = str(result)
                
                logging.info(f"  Result {idx+1}: "
                            f"Score={result_score:.4f}, "
                            f"Content=\"{result_content}\"")
                all_results.append((result_score, result_content))
        except Exception as e:
            logging.error(f"Search failed for user '{user_id}': {e}")
            continue
    
    # Sort and display combined results
    if all_results:
        logging.info("\nTop 3 combined results across all users:")
        all_results.sort(key=lambda x: x[0])  # Sort by score (lower is better for similarity)
        for idx, (score, content) in enumerate(all_results[:3]):
            logging.info(f"  Combined Result {idx+1}: Score={score:.4f}, Content=\"{content}\"")
    else:
        logging.warning("No combined results found across any users.")

    # --- Verify Dimension in DB (Optional - requires direct DB query) ---
    # This part is for advanced verification and requires psycopg2
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=config['vector_store']['config']['dbname'],
            user=config['vector_store']['config']['user'],
            password=config['vector_store']['config']['password'],
            host=config['vector_store']['config']['host'],
            port=config['vector_store']['config']['port']
        )
        cur = conn.cursor()
        
        # Check how many records are in the table
        cur.execute(f"SELECT COUNT(*) FROM {collection_name};")
        record_count = cur.fetchone()[0]
        logging.info(f"Total records in database: {record_count}")
        
        # Get all record IDs to see what's stored
        cur.execute(f"SELECT id, payload->>'data' FROM {collection_name} LIMIT 10;")
        records = cur.fetchall()
        logging.info(f"Records in database:")
        for record in records:
            logging.info(f"  ID: {record[0]}, Data: {record[1][:50]}...")
        
        cur.execute(f"SELECT vector FROM {collection_name} LIMIT 1;")
        vector_from_db = cur.fetchone()[0]
        # pgvector returns vector as string '[1.2, 3.4,...]' or NumPy array depending on psycopg2 setup
        # Simple check based on string representation length is fragile.
        # Better check: Parse the string or use vector_dims if available.
        logging.info(f"Sample vector retrieved from DB (first few elements): {str(vector_from_db)[:100]}...")
        # Attempt to determine dimension (this might vary based on pgvector/psycopg2 versions)
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            cur.execute(f"SELECT vector_dims(vector) FROM {collection_name} LIMIT 1;")
            db_dims = cur.fetchone()[0]
            logging.info(f"Dimension reported by DB: {db_dims}")
            assert db_dims == target_matryoshka_dims, f"Dimension in DB ({db_dims}) doesn't match target ({target_matryoshka_dims})!"
        except Exception as dim_check_e:
            logging.warning(f"Could not directly verify vector dimension in DB: {dim_check_e}")
            
        cur.close()
        conn.close()
    except ImportError:
        logging.warning("psycopg2 not installed, skipping direct DB dimension verification.")
    except Exception as db_e:
        logging.error(f"Error during direct DB verification: {db_e}")

    logging.info("\nExample completed successfully!")

if __name__ == "__main__":
    main() 