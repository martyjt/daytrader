"""Fernet-based secret encryption for broker credentials and API keys."""

from __future__ import annotations

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
