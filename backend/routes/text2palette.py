from flask import g, Blueprint, request  # type: ignore

from controllers.text2palette import *
from middleware.ChromaDBMiddleware import RetrieveCollectionMiddleware, QueryCollectionMiddleware

text2palette_bp = Blueprint('text2palette', __name__)

@text2palette_bp.route('/', methods=['GET'])
@RetrieveCollectionMiddleware(db_path="db/chroma_db_hsl")
@QueryCollectionMiddleware(n_results=3)
def get_single_text_palette():
    data = request.json_data or {}

    user_query = data.get('user_query', "<Empty Query>")
    retrieved_examples = g.get('retrieved_examples', "<EXAMPLES NOT FOUND>")
    
    return generate_test_palette_from_query(user_query, retrieved_examples)