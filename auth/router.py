"""Auth endpoints: register / login / refresh (Phase 3)."""

import jwt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from auth.schemas import LoginIn, RefreshIn, RegisterIn, TokenPair, UserOut
from auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    require_token_type,
    verify_password,
)
from database.repository import UserRepository
from dependencies import get_db_session
from exceptions import InvalidTokenTypeError

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201, response_model=UserOut)
async def register(
    body: RegisterIn,
    db: AsyncSession = Depends(get_db_session),
) -> UserOut:
    repo = UserRepository(db)
    if await repo.get_by_email(body.email):
        raise HTTPException(409, "email already registered")
    if await repo.get_by_username(body.username):
        raise HTTPException(409, "username already taken")
    user = await repo.create(
        email=body.email,
        username=body.username,
        hashed_password=hash_password(body.password),
    )
    return UserOut.model_validate(user)


@router.post("/login", response_model=TokenPair)
async def login(
    body: LoginIn,
    db: AsyncSession = Depends(get_db_session),
) -> TokenPair:
    user = await UserRepository(db).get_by_email(body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "invalid credentials")  # generic on purpose
    return TokenPair(
        access_token=create_access_token(str(user.id)),
        refresh_token=create_refresh_token(str(user.id)),
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshIn) -> TokenPair:
    try:
        claims = require_token_type(decode_token(body.refresh_token), expected="refresh")
    except (jwt.PyJWTError, InvalidTokenTypeError) as exc:
        raise HTTPException(401, "invalid or expired refresh token") from exc
    sub = claims["sub"]
    return TokenPair(
        access_token=create_access_token(sub),
        refresh_token=create_refresh_token(sub),
    )
