from flask import g, Blueprint, request  # type: ignore
from errors import CustomAPIError, BadRequestError

from models.auth import SignInRequest, SignUpRequest, OAuthRequest
from db.SupaAuthManager import SupaAuthManager
from controllers.auth import *


auth_bp = Blueprint('auth', __name__)

@auth_bp.post("/signup")
def signup_route():
    """
    POST /auth/signup
    Body: { "email": "...", "password": "..." }

    Returns 201 with { user, tokens? }.
    When email confirmation is enabled, tokens will be absent and a message
    field explains that the user must confirm their email first.
    """
    
    try:
        user_data = request.json_data;
        email = user_data.get('email')
        password = user_data.get('password')
        
        return sign_up_controller(SignUpRequest(email=email, password=password), SupaAuthManager().get_client())
    except ValueError as e:
        raise BadRequestError(message=str(e))


@auth_bp.post("/signin")
def signin_route():
    """
    POST /auth/signin
    Body: { "email": "...", "password": "..." }

    Returns 200 with { user, tokens }.
    """
    
    try:
        user_data = request.json_data;
        email = user_data.get('email')
        password = user_data.get('password')
        
        return sign_in_controller(SignInRequest(email=email, password=password), SupaAuthManager().get_client())
    except ValueError as e:
        raise BadRequestError(message=str(e))


@auth_bp.post("/oauth/signin")
def oauth_signin_route():
    """
    POST /auth/oauth/signin
    Body: { "provider": "google", "redirect_to": "https://yourapp.com/dashboard" }

    Returns 200 with { provider, url } — the client must redirect the user's
    browser to `url` to complete the OAuth flow.
    """
    
    try:
        user_data = request.json_data;
        provider = user_data.get('provider', 'google')
        redirect_to = user_data.get('redirect_to', None)
        
        return oauth_sign_in_controller(OAuthRequest(provider=provider, redirect_to=redirect_to), SupaAuthManager().get_client())
    except ValueError as e:
        raise BadRequestError(message=str(e))



@auth_bp.get("/oauth/callback")
def oauth_callback_route():
    """
    GET /auth/oauth/callback?code=<one_time_code>

    Supabase / Google redirects the user here after they authorise access.
    Exchanges the code for JWT tokens and returns them as JSON.
    """
    
    code = request.args.get("code")
    if not code:
        raise BadRequestError(message="Missing 'code' query parameter.")

    return oauth_cb(code, SupaAuthManager().get_client())