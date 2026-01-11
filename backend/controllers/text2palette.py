from flask import jsonify
from errors import BadRequestError

def generate_test_palette_from_query(user_query: str, retrieved_examples: str):
    res = {
        "msg": "Test palette generated from user query!",
        "user_query": user_query,
        "retrieved_examples": retrieved_examples,
        "palette": [
            "Word1(Hex1)",
            "Word2(Hex2)", 
            "Word3(Hex3)", 
            "Word4(Hex4)", 
            "Word5(Hex5)"
        ]
    }

    return jsonify(res), 200


__all__ = [
    'generate_test_palette_from_query'
]