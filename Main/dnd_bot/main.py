"""Main entry point for the D&D 5e Discord bot."""

import asyncio
import sys
from pathlib import Path

import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dnd_bot.bot.client import create_bot
from dnd_bot.config import get_settings


def configure_logging():
    """Configure structured logging."""
    import logging

    settings = get_settings()

    # Set up stdlib logging first
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main():
    """Run the bot."""
    configure_logging()
    logger = structlog.get_logger()

    settings = get_settings()

    logger.info("starting_bot", model=settings.ollama_model)

    # Create and run bot
    bot = create_bot()

    try:
        bot.run(settings.discord_bot_token)
    except KeyboardInterrupt:
        logger.info("bot_interrupted")
    except Exception as e:
        logger.error("bot_error", error=str(e))
        raise


if __name__ == "__main__":
    main()
