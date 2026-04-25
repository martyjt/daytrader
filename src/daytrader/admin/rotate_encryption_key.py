"""Rotate ``APP_ENCRYPTION_KEY`` — re-encrypt every broker credential row.

Usage::

    python -m daytrader.admin.rotate_encryption_key --new-key <fernet-key>

The script:

1. Reads ``APP_ENCRYPTION_KEY`` from settings (the *old* key) and decrypts
   every row in ``broker_credentials``.
2. Re-encrypts each payload with the supplied ``--new-key`` inside a single
   transaction, so a partial failure rolls everything back.
3. Tells you to update ``.env`` and restart the app.

Generate a fresh Fernet key with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

This is a destructive-ish operation (irreversible without the new key); take
a database backup first.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from cryptography.fernet import Fernet
from sqlalchemy import select

from ..core.crypto import SecretCodec, get_codec, reset_codec_cache
from ..core.settings import get_settings
from ..storage.database import close_db, get_session, init_db
from ..storage.models import BrokerCredentialModel


async def _rotate(new_key: str, *, dry_run: bool) -> int:
    settings = get_settings()
    await init_db(settings.database_url)
    try:
        try:
            old_codec = get_codec()
        except RuntimeError as exc:
            print(f"ERROR: cannot read old key — {exc}", file=sys.stderr)
            return 1
        new_codec = SecretCodec(new_key)

        async with get_session() as session:
            rows = (
                await session.execute(select(BrokerCredentialModel))
            ).scalars().all()

            if not rows:
                print("No credentials to rotate.")
                return 0

            print(f"Rotating {len(rows)} row(s)...")
            for row in rows:
                plaintext = old_codec.decrypt(row.credential_data)
                row.credential_data = new_codec.encrypt(plaintext)

            if dry_run:
                print("Dry run — rolling back.")
                await session.rollback()
            else:
                await session.commit()
                print(
                    "Rotation committed. Update APP_ENCRYPTION_KEY in your .env "
                    "to the new key and restart the app."
                )
        reset_codec_cache()
        return 0
    finally:
        await close_db()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-key", required=True, help="The new Fernet key (urlsafe-base64)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decrypt + re-encrypt each row but roll back instead of committing.",
    )
    args = parser.parse_args()

    try:
        Fernet(args.new_key.encode())
    except Exception as exc:
        print(f"ERROR: --new-key is not a valid Fernet key: {exc}", file=sys.stderr)
        return 2

    return asyncio.run(_rotate(args.new_key, dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
