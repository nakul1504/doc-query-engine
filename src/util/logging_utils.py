import contextvars
import logging
import logging.config
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="Duplicate Operation ID"
)

request_id_var = contextvars.ContextVar("request_id", default="default")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s",
        },
        "http": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        "http_handler": {
            "class": "logging.StreamHandler",
            "formatter": "http",
        },
        "console_simple": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console_simple"],
            "level": "INFO",
        },
        "uvicorn.access": {
            "handlers": ["http_handler", "console"],
            "level": "WARNING",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
        "application": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": True,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
}


class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[{request_id_var.get()}] : {msg}", kwargs


def get_logger(name):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(name)
    return RequestLoggerAdapter(logger, {})
