"""Pydantic request/response schemas for Phase 3 auth endpoints."""

import uuid

from pydantic import BaseModel, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class RefreshIn(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    username: str

    model_config = {"from_attributes": True}


class KeyIn(BaseModel):
    provider: str = Field(pattern=r"^(gemini|openai|anthropic)$")
    api_key: str = Field(min_length=1)


class KeyOut(BaseModel):
    id: uuid.UUID
    provider: str

    model_config = {"from_attributes": True}
