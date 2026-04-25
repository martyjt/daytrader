"""Argon2id password hashing wrappers.

Uses passlib's CryptContext so we can swap algorithms or tune cost without
changing call sites.
"""

from __future__ import annotations

from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        return _pwd_context.verify(password, password_hash)
    except ValueError:
        return False
