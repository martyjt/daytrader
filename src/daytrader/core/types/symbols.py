"""Symbol normalization across asset classes and venues."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

_QUOTE_HINTS = ("USDT", "USDC", "BUSD", "USD", "EUR", "GBP", "BTC", "ETH")


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    EQUITIES = "equities"
    FOREX = "forex"
    COMMODITIES = "commodities"
    FUTURES = "futures"


@dataclass(frozen=True, slots=True)
class Symbol:
    """A tradable instrument, normalized across venues.

    Examples:
        Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance")
        Symbol("AAPL", "USD", AssetClass.EQUITIES, "nasdaq")
    """

    base: str
    quote: str
    asset_class: AssetClass
    venue: str | None = None

    @property
    def canonical(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def key(self) -> str:
        """Fully qualified identifier suitable for logs, cache keys, DB rows."""
        venue = f"@{self.venue}" if self.venue else ""
        return f"{self.asset_class.value}:{self.canonical}{venue}"

    @classmethod
    def parse(
        cls,
        raw: str,
        asset_class: AssetClass = AssetClass.CRYPTO,
        venue: str | None = None,
    ) -> Symbol:
        """Parse a symbol from common string formats.

        Accepts: ``BTC/USDT``, ``BTC-USDT``, ``BTCUSDT``, ``AAPL``.
        """
        raw = raw.strip().upper()
        if "/" in raw:
            base, quote = raw.split("/", 1)
        elif "-" in raw:
            base, quote = raw.split("-", 1)
        elif asset_class == AssetClass.EQUITIES:
            base, quote = raw, "USD"
        else:
            base, quote = cls._split_smashed(raw)
        return cls(base=base, quote=quote, asset_class=asset_class, venue=venue)

    @staticmethod
    def _split_smashed(raw: str) -> tuple[str, str]:
        for hint in _QUOTE_HINTS:
            if raw.endswith(hint) and len(raw) > len(hint):
                return raw[: -len(hint)], hint
        raise ValueError(f"Cannot parse symbol {raw!r}: unknown quote currency")
