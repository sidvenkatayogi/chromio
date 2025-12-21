from flask import Flask, jsonify # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore
from flask_restful import Api # type: ignore
from flask_cors import CORS # type: ignore

from middleware import register_middleware

from routes import *


api = Api()

def create_app():
    app = Flask(__name__)
    CORS(app, origins='*')

    api.init_app(app) # init app
    register_middleware(app) # middleware


    @app.route('/api/v1/', methods=['GET'])
    def handle_home():
        print("-------\tresponding...\t\t\t-------")
        return "Welcome to Chromio backend RESTful API!", 200
    
    # router
    app.register_blueprint(text2palette_bp, url_prefix='/api/v1/text2palette')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
