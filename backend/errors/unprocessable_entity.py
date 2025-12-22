from .custom_api import CustomAPIError

class UnprocessableEntityError(CustomAPIError):
    status_code = 422

    def __init__(self, message="Unprocessable Entity", payload=None):
        super().__init__("Unprocessable Entity Error", message, self.status_code, payload)