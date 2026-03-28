"""Discord bot client setup and initialization."""

import discord
from discord.ext import commands
import structlog

from ..config import get_settings
from ..data import get_database, close_database, get_srd

logger = structlog.get_logger()


class DnDBot(commands.Bot):
    """The D&D 5e Discord bot."""

    def __init__(self):
        settings = get_settings()

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.voice_states = True  # For future TTS support

        super().__init__(
            intents=intents,
            auto_sync_commands=True,
        )

        self.settings = settings

    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(
            "bot_ready",
            user=str(self.user),
            guilds=len(self.guilds),
        )
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print(f"Connected to {len(self.guilds)} guild(s)")

        # Force sync commands to all guilds (guild-specific for instant availability)
        try:
            # py-cord 2.7+ uses register_commands / sync_commands differently
            if hasattr(self, 'sync_commands'):
                for guild in self.guilds:
                    await self.sync_commands(commands=None, guild_ids=[guild.id])
                    print(f"Synced commands to guild: {guild.name} ({guild.id})")
            else:
                # Newer py-cord auto-syncs with auto_sync_commands=True
                print("Auto-sync enabled, commands will sync automatically")
            if hasattr(self, 'pending_application_commands'):
                print(f"Total commands registered: {len(self.pending_application_commands)}")
        except Exception as e:
            print(f"Failed to sync commands: {e}")
            import traceback
            traceback.print_exc()

        print("------")

    async def on_connect(self):
        """Called when the bot connects to Discord."""
        logger.info("bot_connected")

    async def on_disconnect(self):
        """Called when the bot disconnects from Discord."""
        logger.warning("bot_disconnected")

    async def on_application_command_error(
        self, ctx: discord.ApplicationContext, error: Exception
    ):
        """Global error handler for application commands.

        Handles all unhandled exceptions from slash commands, including
        Discord API failures, timeouts, and unexpected errors.
        """
        # Unwrap the original exception if wrapped by discord.py
        original = getattr(error, "original", error)

        if isinstance(error, commands.CommandOnCooldown):
            msg = f"This command is on cooldown. Try again in {error.retry_after:.1f}s"
        elif isinstance(error, commands.MissingPermissions):
            msg = "You don't have permission to use this command."
        elif isinstance(original, TimeoutError):
            msg = "The AI took too long to respond. Please try again."
            logger.error(
                "command_timeout",
                command=ctx.command.name if ctx.command else "unknown",
            )
        else:
            msg = "An error occurred while processing your command. Please try again."
            logger.error(
                "command_error",
                command=ctx.command.name if ctx.command else "unknown",
                error=str(original),
                error_type=type(original).__name__,
            )

        # Try to respond — the interaction may already be responded to or expired
        try:
            if ctx.response.is_done():
                await ctx.followup.send(msg, ephemeral=True)
            else:
                await ctx.respond(msg, ephemeral=True)
        except discord.HTTPException:
            # Interaction expired or we can't respond — just log it
            logger.warning(
                "error_response_failed",
                command=ctx.command.name if ctx.command else "unknown",
                original_error=str(original),
            )

    async def setup_hook(self):
        """Called when the bot is setting up."""
        print("=== Bot Setup Starting ===")
        logger.info("bot_setup_starting")

        # Initialize database
        await get_database()
        logger.info("database_initialized")

        # Load SRD data
        get_srd()
        logger.info("srd_data_loaded")

        # Load cogs
        await self._load_cogs()

        logger.info("bot_setup_complete")

    async def _load_cogs(self):
        """Load all cog extensions."""
        cog_modules = [
            "dnd_bot.bot.cogs.admin",
            "dnd_bot.bot.cogs.character",
            "dnd_bot.bot.cogs.actions",
            "dnd_bot.bot.cogs.combat",
            "dnd_bot.bot.cogs.spells",
            "dnd_bot.bot.cogs.rest",
            "dnd_bot.bot.cogs.game",
            "dnd_bot.bot.cogs.campaign",
            "dnd_bot.bot.cogs.inventory",
        ]

        for cog in cog_modules:
            try:
                self.load_extension(cog)
                print(f"Loaded cog: {cog}")
                logger.info("cog_loaded", cog=cog)
            except Exception as e:
                print(f"FAILED to load cog {cog}: {e}")
                logger.error("cog_load_failed", cog=cog, error=str(e))
                import traceback
                traceback.print_exc()

    async def close(self):
        """Clean up resources when the bot shuts down."""
        logger.info("bot_shutting_down")
        await close_database()
        await super().close()


def create_bot() -> DnDBot:
    """Create and return a new bot instance."""
    bot = DnDBot()

    # Load cogs synchronously before bot starts
    cog_modules = [
        "dnd_bot.bot.cogs.admin",
        "dnd_bot.bot.cogs.character",
        "dnd_bot.bot.cogs.actions",
        "dnd_bot.bot.cogs.combat",
        "dnd_bot.bot.cogs.spells",
        "dnd_bot.bot.cogs.rest",
        "dnd_bot.bot.cogs.game",
        "dnd_bot.bot.cogs.campaign",
        "dnd_bot.bot.cogs.inventory",
    ]

    for cog in cog_modules:
        try:
            bot.load_extension(cog)
            print(f"Loaded cog: {cog}")
        except Exception as e:
            print(f"FAILED to load cog {cog}: {e}")
            import traceback
            traceback.print_exc()

    return bot
