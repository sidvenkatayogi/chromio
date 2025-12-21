from .custom_api import CustomAPIError

class NotFoundError(CustomAPIError):
    status_code = 404

    def __init__(self, message="Not Found", payload=None):
        super().__init__("Not Found Error", message, self.status_code, payload)