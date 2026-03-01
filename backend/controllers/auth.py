from flask import jsonify  # type: ignore
from supabase_auth.errors import AuthApiError

from errors import CustomAPIError, UnauthorizedError, UnprocessableEntityError

def _session_to_tokens(session) -> dict:
    """Extract JWT-related fields from a GoTrue Session object."""
    return {
        "access_token": session.access_token,
        "refresh_token": session.refresh_token,
        "token_type": session.token_type,
        "expires_in": session.expires_in,
        "expires_at": session.expires_at,
    }

def _user_summary(user) -> dict:
    return {
        "id": str(user.id),
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }



def sign_up_controller(signup_request, client):
    try:
        response = client.sign_up({
            "email": signup_request.email,
            "password": signup_request.password
        })
    except AuthApiError as e:
        msg = getattr(e, "message", None) or str(e)
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 400
        if status == 422:
            raise UnprocessableEntityError(message=msg)
        raise CustomAPIError(message=msg, status_code=status)
    
    user = response.user
    session = response.session
    
    result = { 'user': _user_summary(user) }
    if session:
        result["tokens"] = _session_to_tokens(session)
    else:
        result["message"] = "Confirmation email sent. Please verify your email before signing in."
    
    return jsonify(result), 200


def sign_in_controller(signin_request, client):
    try:
        response = client.sign_in_with_password({
            "email": signin_request.email,
            "password": signin_request.password
        })
    except AuthApiError as e:
        msg = getattr(e, "message", None) or str(e)
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 400

        if status == 400:
            raise UnauthorizedError(message=msg)
        raise CustomAPIError(message=msg, status_code=status)
    
    user = response.user
    session = response.session
    result = {
        'user': _user_summary(user),
        'tokens': _session_to_tokens(session)
    }
    
    return jsonify(result), 200


def oauth_sign_in_controller(oauth_request, client, allowed_providers={"google"}):
    provider = oauth_request.provider.lower()
    if provider not in allowed_providers:
        raise CustomAPIError(
            message=f"Provider '{provider}' is not supported. "
                    f"Allowed: {', '.join(sorted(ALLOWED_PROVIDERS))}.",
            status_code=400,
        )
    
    try:
        response = client.sign_in_with_oauth(provider=provider)
    except AuthApiError as e:
        msg = getattr(e, "message", None) or str(e)
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 400
        if status == 400:
            raise UnauthorizedError(message=msg)
        raise CustomAPIError(message=msg, status_code=status)

    result = {
        'provider': provider,
        'url': response.url
    }
    
    return jsonify(result), 200


def oauth_cb(code, client):
    try:
        response = client.exchange_code_for_session(auth_code=code)
    except AuthApiError as e:
        msg = getattr(e, "message", None) or str(e)
        status = getattr(e, "status", None) or getattr(e, "status_code", None) or 400
        if status == 400:
            raise UnauthorizedError(message=msg)
        raise CustomAPIError(message=msg, status_code=status)
    
    user = response.user
    session = response.session
    result = {
        'user': _user_summary(user),
        'tokens': _session_to_tokens(session)
    }
    
    return jsonify(result), 200

__all__ = [
    "sign_up_controller",
    "sign_in_controller",
    "oauth_sign_in_controller",
    "oauth_cb"
]