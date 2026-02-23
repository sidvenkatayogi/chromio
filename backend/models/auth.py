from pydantic import BaseModel, EmailStr, field_validator


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class OAuthRequest(BaseModel):
    """Used when the client wants to kick off a Google OAuth flow."""
    provider: str = "google"
    redirect_to: str | None = None