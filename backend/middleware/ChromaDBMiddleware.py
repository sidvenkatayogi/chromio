import logging
import time
import json
from flask import g, jsonify, request
from functools import wraps

from errors import CustomAPIError
from db.ChromadbManager import ChromaDBManager

logger = logging.getLogger(__name__)

class RetrieveCollectionMiddleware:

    def __init__(
        self,
        db_path: str,
        collection_name: str = "pat",
        ef_model_name: str = "all-mpnet-base-v2"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.ef_model_name = ef_model_name
        
    
    def __call__(self, f):
        @wraps(f)
        def w(*args, **kwargs):
            start_time = time.time()
            manager = ChromaDBManager()
            
            try:
                g.chroma_collection = manager.get_collection(
                    self.db_path,
                    collection_name=self.collection_name,
                    ef_model_name=self.ef_model_name,
                )
                
                logger.info(
                    f"ChromaDB middleware: Collection '{self.collection_name}' "
                    f"loaded in {time.time() - start_time:.3f}s"
                )
            except Exception as e:
                logger.error(f"ChromaDB middleware error: {e}")
                raise CustomAPIError(
                    name="Database initialization failed",
                    message=str(e)
                )
        
            return f(*args, **kwargs)
        
        return w


class QueryCollectionMiddleware:
    
    def __init__(self, n_results: int):
        self.n_results = n_results
    
    
    def __call__(self, f):
        @wraps(f)
        def w(*args, **kwargs):
            start_time = time.time()
            
            collection = getattr(g, "chroma_collection", None)
            if collection is None:
                logger.error("Query middleware invoked without chroma_collection")
                raise CustomAPIError(
                    name="Chroma collection not initialized",
                    message="RetrieveCollectionMiddleware must run before querying",
                )

            # Extract query text
            query_text = self._extract_query()

            try:
                retrieval_results = collection.query(
                    query_texts=[query_text],
                    n_results=self.n_results
                )
                
                try:
                    output_lines = []
                    if retrieval_results['documents'] and retrieval_results['documents'][0]:
                        for j, result_doc in enumerate(retrieval_results['documents'][0], start=1):
                            data = json.loads(result_doc)
                            output_lines.append(f"Palette {j}:")
                            output_lines.append(f"Description: {data['description']}")
                            for color in data['palette']:
                                output_lines.append(f"  - {color}")
                            output_lines.append("")
                    
                    g.retrieved_examples = "\n".join(output_lines)

                except Exception as e:
                    logger.exception("ChromaDB query parsing failed")
                    raise CustomAPIError(
                        name="ChromaDB query parsing failed",
                        message=str(e),
                    )

                logger.info(
                    "ChromaDB query completed in %.3fs (n_results=%s)",
                    time.time() - start_time,
                    self.n_results,
                )

            except Exception as e:
                logger.exception("ChromaDB query failed")
                raise CustomAPIError(
                    name="ChromaDB query failed",
                    message=str(e),
                )

            return f(*args, **kwargs)

        return w

    
    def _extract_query(self):
        data = request.json_data or {}

        user_query = data.get('user_query', "<Empty Query>")
        return user_query