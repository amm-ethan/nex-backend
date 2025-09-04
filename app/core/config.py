from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Ignore extra fields from .env
    )

    # API Settings
    PROJECT_NAME: str = "NEX Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    ROOT_PATH: str = ""  # For proxy prefixes

    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Log File Locations
    ERROR_LOG_LOCATION: str = "storage/logs/errors"
    TRACE_LOG_LOCATION: str = "storage/logs/traces"

    MICROBIOLOGY_FILE_LOCATION: str = "app/data/microbiology.csv"
    TRANSFERS_FILE_LOCATION: str = "app/data/transfers.csv"

    # Environment
    ENVIRONMENT: str = "production"  # e.g., development, staging, production

    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        # Allow empty string from env to mean empty list
        if isinstance(v, str) and v == "":
            return []
        raise ValueError(v)


# Instantiate settings
settings = Settings()

for directory_path_str in [
    settings.ERROR_LOG_LOCATION,
    settings.TRACE_LOG_LOCATION,
]:
    directory = PROJECT_ROOT / directory_path_str
    directory.mkdir(parents=True, exist_ok=True)

error_log_directory = PROJECT_ROOT / settings.ERROR_LOG_LOCATION
trace_log_directory = PROJECT_ROOT / settings.TRACE_LOG_LOCATION
microbiology_file = PROJECT_ROOT / settings.MICROBIOLOGY_FILE_LOCATION
transfers_file = PROJECT_ROOT / settings.TRANSFERS_FILE_LOCATION

# Logger Config
logger_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(asctime)s %(levelname)s [%(module)s->%(funcName)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "verbose",
            "stream": "ext://sys.stdout",
        },
        "error_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "ERROR",
            "formatter": "verbose",
            "filename": str(error_log_directory / "error.log"),  # Use Path object
            "when": "D",
            "interval": 1,
            "backupCount": 7,  # Changed from 0
            "encoding": "utf-8",
            "delay": True,
        },
        "trace_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "verbose",
            "filename": str(trace_log_directory / "trace.log"),  # Use Path object
            "when": "D",
            "interval": 1,
            "backupCount": 7,  # Changed from 0
            "encoding": "utf-8",
            "delay": True,
        },
    },
    "loggers": {
        "": {
            "level": "DEBUG",
            "handlers": ["console", "error_file", "trace_file"],
            "propagate": False,  # Keeping as per original, user can reconsider
        },
    },
}
