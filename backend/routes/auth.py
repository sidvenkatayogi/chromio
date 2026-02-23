from flask import g, Blueprint, request  # type: ignore
from errors import BadRequestError

from models.auth import SignInRequest, SignUpRequest, OAuthRequest


auth_bp = Blueprint('auth', __name__)

@auth_bp.post("/signup")
def signup_route():
    try:
        pass
    except Exception as e:
        raise CustomAPIError(message=e.message, status_code=e.status)