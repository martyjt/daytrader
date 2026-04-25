"""Fernet-based secret encryption for broker credentials and API keys."""

from __future__ import annotations

import functools

from cryptography.fernet import Fernet, InvalidToken

from .settings import get_settings


class SecretCodec:
    """Encrypt / decrypt small secrets at rest using Fernet (AES-128-CBC + HMAC).

    Key comes from ``APP_ENCRYPTION_KEY``. Generate one with::

        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    """

    def __init__(self, key: str | None = None) -> None:
        key = key or get_settings().app_encryption_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "APP_ENCRYPTION_KEY is empty. Set it in .env before using "
                "SecretCodec. Generate one with Fernet.generate_key()."
            )
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError(
                "Failed to decrypt secret — wrong APP_ENCRYPTION_KEY or corrupted data."
            ) from exc


@functools.lru_cache(maxsize=1)
def get_codec() -> SecretCodec:
    """Process-lifetime ``SecretCodec`` singleton."""
    return SecretCodec()


def reset_codec_cache() -> None:
    """Clear the cached codec (used by tests + the rotate-key admin script)."""
    get_codec.cache_clear()


async def assert_encryption_key_for_existing_secrets() -> None:
    """Fail loudly at startup if encrypted credentials exist but the key is unset.

    The check is *only* fatal when there's actually something to decrypt — fresh
    installs without broker credentials still work without configuring the key.
    """
    from sqlalchemy import func as sa_func
    from sqlalchemy import select

    from ..storage.database import get_session
    from ..storage.models import BrokerCredentialModel

    async with get_session() as session:
        count = (
            await session.execute(
                select(sa_func.count()).select_from(BrokerCredentialModel)
            )
        ).scalar_one()

    if count == 0:
        return

    key = get_settings().app_encryption_key.get_secret_value()
    if not key:
        raise RuntimeError(
            f"APP_ENCRYPTION_KEY is empty but {count} encrypted broker "
            "credential row(s) exist. Set APP_ENCRYPTION_KEY in .env to the "
            "key those rows were encrypted with, or delete the broker_credentials "
            "table to start fresh."
        )

    # Probe-decrypt one row so we fail at startup if the key is wrong, rather
    # than the first time someone clicks "test connection".
    async with get_session() as session:
        sample = (
            await session.execute(select(BrokerCredentialModel).limit(1))
        ).scalar_one_or_none()
    if sample is None:
        return
    try:
        get_codec().decrypt(sample.credential_data)
    except Exception as exc:
        raise RuntimeError(
            "APP_ENCRYPTION_KEY does not match the key used to encrypt the "
            "existing broker credentials. Either restore the original key or "
            "delete the broker_credentials table and re-enter your API keys."
        ) from exc
