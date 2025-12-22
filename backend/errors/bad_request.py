from .custom_api import CustomAPIError

class BadRequestError(CustomAPIError):
    status_code = 400

    def __init__(self, message="Bad Request", payload=None):
        super().__init__("Bad Request Error", message, self.status_code, payload)