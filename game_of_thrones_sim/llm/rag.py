import asyncio
import json
from pathlib import Path
from typing import Optional, List

import chromadb

# Placeholder for logger and Config, will be replaced by actual imports
class PrintLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg, exc_info=False): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

log = PrintLogger()

class MockConfig: # Used until actual Config is properly importable
    CHROMA_DIR = "./chroma_db_auto_gui_v3_rag_module" # Use a distinct dir for direct test
    BOOK_CHUNKS = "./data/got_chunks_rag_module.json" # Use a distinct file for direct test
    RAG_TOPK = 3

# Global Chroma client instance
_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None

def initialize_rag(config: object) -> bool:
    """
    Initializes the RAG system by ensuring embeddings and the Chroma client.
    Args:
        config: The application's configuration object (e.g., Config class instance).
    Returns:
        True if ChromaDB client was initialized (even if collection is new/empty),
        False if a critical error occurred during client initialization.
    """
    global _CHROMA_CLIENT, log

    # Ensure logger is properly set up
    try:
        from ..utils.logger import log as main_log
        log = main_log
    except ImportError:
        pass # Keep print logger if utils.logger is not yet available

    log.info("Initializing RAG: Checking for ChromaDB embeddings...")
    chroma_dir_path = Path(config.CHROMA_DIR)
    book_chunks_path = Path(config.BOOK_CHUNKS)

    try:
        chroma_dir_path.mkdir(parents=True, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(chroma_dir_path))
    except Exception as e:
        log.error(f"Failed to initialize ChromaDB PersistentClient at {chroma_dir_path}: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to initialize ChromaDB at {chroma_dir_path}. Check permissions/dependencies. RAG will be disabled.")
        _CHROMA_CLIENT = None
        return False # Critical failure

    try:
        _CHROMA_CLIENT.get_collection("got")
        log.info("Chroma collection 'got' already exists.")
    except Exception: # Exception means collection likely doesn't exist
        log.info("Chroma collection 'got' not found. Creating and populating...")
        try:
            coll = _CHROMA_CLIENT.create_collection("got")
            chunks: List[str] = []
            if not book_chunks_path.exists():
                log.warning(f"Lore file not found at {book_chunks_path}. RAG will be less effective.")
                print(f"Warning: Lore file '{book_chunks_path}' not found. Creating dummy file for RAG.")
                try:
                    # Ensure parent directory for dummy file exists
                    book_chunks_path.parent.mkdir(parents=True, exist_ok=True)
                    dummy_data = ["Winter is coming.", "The Targaryens ruled Westeros with dragons.", "The Lannisters always pay their debts."]
                    with book_chunks_path.open('w', encoding='utf-8') as f:
                        json.dump(dummy_data, f, indent=2)
                    log.info(f"Created dummy lore file at {book_chunks_path}.")
                    chunks = dummy_data
                except Exception as create_e:
                    log.error(f"Failed to create dummy lore file at {book_chunks_path}: {create_e}. RAG will be empty.")
            else:
                try:
                    with book_chunks_path.open('r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    log.info(f"Loaded {len(chunks)} lore chunks from {book_chunks_path}")
                except json.JSONDecodeError:
                    log.error(f"Failed to decode JSON from {book_chunks_path}. RAG will be empty.")
                except Exception as e:
                    log.error(f"Error reading lore file {book_chunks_path}: {e}. RAG will be empty.")

            if chunks:
                ids = [f"doc_{i}" for i in range(len(chunks))]
                # Ensure all chunks are strings
                doc_chunks = [str(chunk) if not isinstance(chunk, str) else chunk for chunk in chunks]
                
                # Batch add to Chroma
                batch_size = 100 # As in original code
                for i in range(0, len(doc_chunks), batch_size):
                    batch_ids = ids[i:i+batch_size]
                    batch_docs = doc_chunks[i:i+batch_size]
                    if batch_ids and batch_docs: # Ensure not empty
                        coll.add(ids=batch_ids, documents=batch_docs)
                log.info(f"Embedded {len(doc_chunks)} lore snippets into 'got' collection.")
            else:
                log.warning(f"Lore file {book_chunks_path} was empty or failed to load. RAG will have no data in 'got' collection.")
        except Exception as e:
            # This is not a client init failure, but collection creation/population failure.
            log.error(f"Error creating/populating Chroma 'got' collection: {e}", exc_info=True)
            print(f"Error setting up Chroma 'got' collection: {e}. RAG might be non-functional.")
            # _CHROMA_CLIENT is still valid, so return True, but RAG might be useless.

    return True # Chroma client itself is initialized

def get_chroma_client() -> Optional[chromadb.PersistentClient]:
    """Returns the initialized ChromaDB client instance."""
    return _CHROMA_CLIENT

async def query_rag(prompt: str, top_k: int) -> str:
    """
    Fetches top-K lore snippets relevant to the prompt using the initialized Chroma client.
    Args:
        prompt: The query prompt.
        top_k: The number of top results to retrieve.
    Returns:
        A string containing the concatenated relevant lore snippets, or an empty string if error/no results.
    """
    global _CHROMA_CLIENT, log

    if not _CHROMA_CLIENT:
        log.warning("Chroma client not available for RAG query.")
        return ""

    lore_content = ""
    try:
        # This part needs to be synchronous as chromadb client calls are typically sync.
        # We run it in an executor to make the outer function awaitable.
        def _query_chroma_sync(p: str, n: int) -> str:
            try:
                # Ensure collection exists before querying
                # This check might be redundant if initialize_rag guarantees collection creation,
                # but good for robustness if client is used independently.
                try:
                    coll = _CHROMA_CLIENT.get_collection("got")
                except Exception: # Collection might not exist if init failed post client creation
                    log.warning("Chroma collection 'got' not found during RAG query.")
                    return ""

                count = coll.count()
                if count == 0:
                    # log.debug("RAG query: 'got' collection is empty.") # Can be noisy
                    return ""

                n_results = min(n, count)
                res = coll.query(query_texts=[p], n_results=n_results)

                if res and res.get("documents") and res["documents"][0]:
                    # Filter out empty strings from results just in case
                    return "\n".join(filter(None, res["documents"][0]))
                return ""
            except Exception as e:
                log.warning(f"Synchronous Chroma query failed: {e}", exc_info=False)
                return ""

        loop = asyncio.get_running_loop()
        lore_content = await loop.run_in_executor(None, _query_chroma_sync, prompt, top_k)
        # log.debug(f"RAG query returned {len(lore_content)} chars for prompt: '{prompt[:50]}...'")
    except Exception as e:
        log.warning(f"RAG query execution failed: {e}", exc_info=False)

    return lore_content


if __name__ == '__main__':
    # Example Usage (requires chromadb installed)
    # This test will create a dummy chroma_db_auto_gui_v3_rag_module directory
    # and a dummy data/got_chunks_rag_module.json file in the current working directory.

    async def test_rag_system():
        global log
        try:
            from ..utils.logger import log as main_log # Adjust if testing from project root
            log = main_log
            log.info("Using main application logger for RAG test.")
        except ImportError:
            # Fallback for direct execution if utils.logger is not in python path
            # This might happen if you run `python game_of_thrones_sim/llm/rag.py` directly
            # without `python -m game_of_thrones_sim.llm.rag`
            print("INFO: Using print logger for RAG test. For full logging, run as module or ensure utils is in PYTHONPATH.")


        # Use MockConfig for testing if main Config is not available
        # In a real scenario, this would come from the main application config
        mock_config = MockConfig()
        
        # Create dummy data directory if it doesn't exist for the test
        Path(mock_config.BOOK_CHUNKS).parent.mkdir(parents=True, exist_ok=True)


        if initialize_rag(config=mock_config):
            log.info("RAG system initialized for test.")
            client = get_chroma_client()
            if client:
                log.info(f"Chroma client obtained: {type(client)}")
                try:
                    collections = client.list_collections()
                    log.info(f"Available collections: {[col.name for col in collections]}")
                    if any(c.name == "got" for c in collections):
                        got_coll = client.get_collection("got")
                        log.info(f"'got' collection count: {got_coll.count()}")
                except Exception as e:
                    log.error(f"Error inspecting Chroma client post-init: {e}")


                test_query = "What is Winterfell?"
                results = await query_rag(test_query, mock_config.RAG_TOPK)
                log.info(f"Query: '{test_query}'\nResults:\n{results or 'No results found.'}")

                test_query_2 = "Tell me about dragons."
                results_2 = await query_rag(test_query_2, mock_config.RAG_TOPK)
                log.info(f"Query: '{test_query_2}'\nResults:\n{results_2 or 'No results found.'}")
            else:
                log.error("Failed to get Chroma client after RAG initialization.")
        else:
            log.error("RAG system failed to initialize for test.")

    asyncio.run(test_rag_system())
