class CustomAPIError(Exception):
    status_code = 500

    def __init__(self, name="Internal Server Error", message="Internal Server Error", status_code=None, payload=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.name = name
        
    def to_dict(self):
        rv = dict(message=self.message)
        if self.payload:
            rv['payload'] = self.payload
        return rv