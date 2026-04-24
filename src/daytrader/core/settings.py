"""Application settings (env-driven) and YAML defaults."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Environment-driven settings. Reads ``.env`` then OS env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    app_secret_key: SecretStr = SecretStr("dev-insecure-change-me")
    app_encryption_key: SecretStr = SecretStr("")

    # Storage
    database_url: str = (
        "postgresql+asyncpg://daytrader:daytrader@localhost:5432/daytrader"
    )
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Phase 0 auth stub
    default_tenant_id: UUID = UUID("00000000-0000-0000-0000-000000000001")
    default_tenant_name: str = "default"
    default_user_email: str = "operator@local"

    # AG Grid Enterprise
    ag_grid_license_key: str = ""

    # Crypto (Phase 0)
    binance_api_key: SecretStr = SecretStr("")
    binance_api_secret: SecretStr = SecretStr("")
    binance_testnet: bool = True

    # Equities (Phase 2)
    alpaca_api_key: SecretStr = SecretStr("")
    alpaca_api_secret: SecretStr = SecretStr("")
    alpaca_paper: bool = True

    # Macro / news
    fred_api_key: SecretStr = SecretStr("")
    newsapi_key: SecretStr = SecretStr("")

    # Additional market-data sources (Phase 6+ expansion)
    alpha_vantage_api_key: SecretStr = SecretStr("")
    twelve_data_api_key: SecretStr = SecretStr("")

    # Exploration Agent scheduler (0 = disabled)
    exploration_schedule_hours: float = 0.0
    exploration_schedule_symbols: str = "BTC-USD"
    exploration_schedule_timeframe: str = "1d"
    exploration_schedule_lookback_days: int = 365

    # Regime watcher — keeps the Regime Badge fresh and fires alerts on change
    regime_refresh_minutes: float = 30.0
    regime_pulse_symbol: str = "BTC-USD"
    regime_pulse_timeframe: str = "1d"

    # Shadow Tournament scheduler (0 = disabled)
    shadow_schedule_hours: float = 0.0
    shadow_schedule_primary: str = "ema_crossover"
    shadow_schedule_candidates: str = "macd_signal,rsi_mean_reversion"
    shadow_schedule_symbol: str = "BTC-USD"
    shadow_schedule_timeframe: str = "1d"
    shadow_schedule_lookback_days: int = 180


class YamlConfig:
    """Layered YAML defaults (read-only). Env settings override these."""

    def __init__(self, path: Path | str = "config/default.yaml") -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = (
            yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            if self._path.exists()
            else {}
        )

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested lookup. ``get("risk", "per_trade", "default_stop_loss_atr_mult")``."""
        cur: Any = self._data
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


@functools.lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


@functools.lru_cache
def get_yaml_config() -> YamlConfig:
    return YamlConfig()
