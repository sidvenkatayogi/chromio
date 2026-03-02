from .custom_api import CustomAPIError

class UnauthorizedError(CustomAPIError):
    status_code = 401

    def __init__(self, message="Unauthorized", payload=None):
        super().__init__("Unauthorized Error", message, self.status_code, payload)