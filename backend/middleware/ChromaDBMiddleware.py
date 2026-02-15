import logging
import time
import json
from flask import current_app, g, jsonify, request
from functools import wraps

from errors import CustomAPIError

logger = logging.getLogger(__name__)


class QueryCollectionMiddleware:
    
    def __init__(self, n_results: int):
        self.n_results = n_results
    
    
    def __call__(self, f):
        @wraps(f)
        def w(*args, **kwargs):
            start_time = time.time()
            
            collection = current_app.extensions.get('chromadb_collection')
            if collection is None:
                logger.error("Query middleware: chromadb_collection not found on app")
                raise CustomAPIError(
                    name="Chroma collection not initialized",
                    message="ChromaDB collection was not initialized at startup",
                )

            # Extract query text
            query_text = self._extract_query()

            try:
                retrieval_results = collection.query(
                    query_texts=[query_text],
                    n_results=self.n_results
                )

                query_time = time.time()
                logger.info("[timing] ChromaDB query (embed + search): %.3fs", query_time - start_time)

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

                parse_time = time.time()
                logger.info("[timing] ChromaDB result parsing: %.3fs", parse_time - query_time)

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