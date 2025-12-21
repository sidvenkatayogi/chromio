from flask import Blueprint, request  # type: ignore
from controllers.text2palette import *

text2palette_bp = Blueprint('text2palette', __name__)

@text2palette_bp.route('/', methods=['GET'])
def get_single_text_palette():
    data = request.json_data or {}

    user_query = data.get('user_query', "<Empty Query>")
    
    return generate_test_palette_from_query(user_query)