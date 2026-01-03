"""
Fetch reference examples from the ChromaDB database.
"""

import os
import json
import chromadb
from chromadb.utils import embedding_functions


# Configuration (matches query_db.py)
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pat"
MODEL_NAME = "all-mpnet-base-v2"


class ExampleFetcher:
    """
    Fetches reference examples from ChromaDB.
    Initializes the embedding model once and reuses it for all queries.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize the ChromaDB client and embedding function once.
        
        Args:
            base_dir: Base directory of the project (where chroma_db is located)
        """
        if base_dir is None:
            # Default to project root (3 levels up from this file)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        chroma_path = os.path.join(base_dir, CHROMA_PATH)
        
        # Initialize client and embedding function once
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )
    
    def get_examples(self, query: str, n_results: int = 3) -> str:
        """
        Query the database for similar palettes.
        
        Args:
            query: The text query to search for
            n_results: Number of results to return
        
        Returns:
            String containing formatted example palettes
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        output_lines = []
        if results['documents'] and results['documents'][0]:
            for i, result_doc in enumerate(results['documents'][0], start=1):
                data = json.loads(result_doc)
                output_lines.append(f"Palette {i}:")
                output_lines.append(f"Description: {data['description']}")
                for color in data['palette']:
                    output_lines.append(f"  - {color}")
                output_lines.append("")
        
        return "\n".join(output_lines)