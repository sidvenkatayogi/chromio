from flask import request, jsonify
from errors import CustomAPIError

def register_middleware(app):

    @app.before_request
    def before_request_func():
        def parse_json():
            if request.is_json:
                request.json_data = request.get_json()
            else:
                request.json_data = None;
        
        parse_json()
        print("-------\tbefore request middleware\t-------")
    
    @app.after_request
    def after_request_func(response):
        # do error handler middleware and not found middleware
        if response.status_code == 404 and not response.is_json:
            response_data = {
                "error": "Not Found",
                "message": "The requested resource was not found"
            }
            response.set_data(json.dumps(response_data))
            response.content_type = "application/json" # no need to create new response object
        
        print("-------\tafter request middleware\t-------")
        return response

    @app.errorhandler(CustomAPIError)
    def handle_custom_api_error(e):
        print(f"-------\tAPI ERROR occurred: {str(e)}")
        
        response_data = {
            "error": e.name if hasattr(e, 'name') else "Internal Server Error",
            "message": str(e.message) if hasattr(e, 'message') else str(e),
            "payload": e.payload if hasattr(e, 'payload') else None
        }
        response = jsonify(response_data)
        response.status_code = e.status_code or 500
        return response
    
    @app.errorhandler(Exception)
    def handle_exceptions(e):
        # handle any thrown error
        print(f"-------\tError occurred: {str(e)}")
        
        # default internal server error
        response_data = {
            "error": e.name if hasattr(e, 'name') else "Internal Server Error",
            "message": str(e.description) if hasattr(e, 'description') else str(e)
        }
        response = jsonify(response_data)
        response.status_code = e.code or 500
        return response