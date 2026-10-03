from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", env_file=".env", extra="ignore")

    store_dir: Path = Field(default=Path("var/store"))
    min_roc_auc: float = 0.85
    min_recall: float = 0.70
    random_seed: int = 11
    serving_stage: str = "Production"
    log_level: str = "INFO"
    drift_psi_trigger: float = 0.25


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
