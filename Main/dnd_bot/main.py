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
    """Configure structured logging to console and file."""
    import logging
    from pathlib import Path

    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Ensure log directory exists
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler for persistent logs
    file_handler = logging.FileHandler(log_dir / "bot.log", encoding="utf-8")
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[console_handler, file_handler],
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
    import argparse
    import os

    parser = argparse.ArgumentParser(description="D&D 5e Discord Bot")
    parser.add_argument(
        "--profile", "-p",
        help="Override the active profile (takes precedence over ACTIVE_PROFILE env var)",
    )
    args = parser.parse_args()

    # CLI flag overrides env var
    if args.profile:
        os.environ["ACTIVE_PROFILE"] = args.profile

    configure_logging()
    logger = structlog.get_logger()

    settings = get_settings()

    from .config import get_profile
    profile = get_profile()
    logger.info("starting_bot", profile=profile.name, narrator=f"{profile.narrator.provider}/{profile.narrator.model}", brain=f"{profile.brain.provider}/{profile.brain.model}")

    # Auto-start Fish Speech servers if the profile uses them
    from .voice.fish_manager import auto_start_fish_if_needed
    auto_start_fish_if_needed()

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
