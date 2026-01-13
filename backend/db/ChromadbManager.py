import logging
import chromadb
import re
import threading

from contextlib import asynccontextmanager
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
        Singleton manager for ChromaDB client and collection caching
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._client = None
            self._initialized = True
            self._collections_cache = {}
            
            self._collections_lock = threading.Lock()
            self._client_lock = threading.Lock()

            logger.info("ChromaDBManager initialized")
    
    
    def get_client(self, db_path: str):
        """
            Get or create ChromaDB client with caching.
        """
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    logger.info(f"Creating new ChromaDB client at {db_path}")
                    try:
                        self._client = chromadb.PersistentClient(path=db_path)
                    except Exception as e:
                        logger.error(f"Error getting client from '{db_path}': {e}.\t Did you forget to initialize a chromadb instance in {db_path}?")
                        raise
        return self._client
    
    
    def get_collection(self, db_path: str, collection_name: str = "pat", ef_model_name:str = "all-mpnet-base-v2"):
        """
            Get collection from cached client
        """
        if collection_name not in self._collections_cache:
            with self._collections_lock:
                if collection_name not in self._collections_cache:
                    client = self.get_client(db_path)
                    
                    try:
                        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=ef_model_name)
                        collection = client.get_collection(
                            name=collection_name,
                            embedding_function=ef
                        )
                        
                        self._collections_cache[collection_name] = collection
                        logger.info(f"Collection '{collection_name}' cached with embedding function")
                    except Exception as e:
                        logger.error(f"Error getting collection '{collection_name}': {e}")
                        raise

        return self._collections_cache[collection_name]
    
    
    def close(self):
        """
            Close client and clear cache
            Must be done before initializing a new client
        """
        with self._client_lock:
            with self._collections_lock:
                self._client = None
                self._collections_cache.clear()
                logger.info("ChromaDB client closed and collections cache cleared")
