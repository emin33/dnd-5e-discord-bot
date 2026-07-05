"""Immersion feature commands: TTS, image generation, and /imagine."""

import discord
from discord.ext import commands
import structlog

from ...data.repositories.immersion_repo import get_immersion_repo
from ...models.immersion import ImageFrequency

logger = structlog.get_logger()


class ImmersionCog(commands.Cog):
    """Commands for managing immersion features (TTS, images)."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    immersion = discord.SlashCommandGroup(
        name="immersion",
        description="Configure immersion features (TTS narration, scene images)",
    )

    @immersion.command(name="tts", description="Enable or disable TTS narration")
    @discord.option("enabled", description="Turn TTS on or off", type=bool)
    async def immersion_tts(self, ctx: discord.ApplicationContext, enabled: bool):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)
        settings.tts_enabled = enabled
        await repo.upsert_guild_settings(settings)

        state = "enabled" if enabled else "disabled"
        await ctx.respond(f"TTS narration **{state}** for this server.", ephemeral=True)
        logger.info("immersion_tts_toggled", guild=ctx.guild_id, enabled=enabled)

    @immersion.command(name="images", description="Enable or disable scene image generation")
    @discord.option("enabled", description="Turn images on or off", type=bool)
    async def immersion_images(self, ctx: discord.ApplicationContext, enabled: bool):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)
        settings.image_enabled = enabled
        await repo.upsert_guild_settings(settings)

        state = "enabled" if enabled else "disabled"
        await ctx.respond(f"Scene images **{state}** for this server.", ephemeral=True)
        logger.info("immersion_images_toggled", guild=ctx.guild_id, enabled=enabled)

    @immersion.command(name="frequency", description="Set how often scene images are generated")
    @discord.option(
        "mode",
        description="Image generation frequency",
        choices=[
            discord.OptionChoice("Every narration", "every"),
            discord.OptionChoice("On scene changes only", "scene_change"),
            discord.OptionChoice("On demand (/imagine only)", "on_demand"),
        ],
    )
    async def immersion_frequency(self, ctx: discord.ApplicationContext, mode: str):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)
        settings.image_frequency = ImageFrequency(mode)
        await repo.upsert_guild_settings(settings)

        labels = {
            "every": "every narration",
            "scene_change": "scene changes only",
            "on_demand": "on demand (/imagine only)",
        }
        await ctx.respond(
            f"Image frequency set to **{labels[mode]}**.",
            ephemeral=True,
        )

    @immersion.command(name="narrator_voice", description="Set the narrator TTS voice")
    @discord.option(
        "provider",
        description="TTS provider for narrator",
        choices=[
            discord.OptionChoice("Kokoro (local, recommended)", "kokoro"),
            discord.OptionChoice("Inworld (cloud)", "inworld"),
            discord.OptionChoice("OpenAI", "openai"),
            discord.OptionChoice("ElevenLabs", "elevenlabs"),
            discord.OptionChoice("Fish Speech (local)", "fish"),
        ],
    )
    @discord.option(
        "voice",
        description="Voice name/ID (e.g. 'af_heart' for Kokoro, 'onyx' for OpenAI). Leave empty for default.",
        required=False,
        default="",
    )
    async def immersion_narrator_voice(
        self, ctx: discord.ApplicationContext, provider: str, voice: str
    ):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)
        settings.narrator_tts_provider = provider
        settings.narrator_tts_voice = voice
        await repo.upsert_guild_settings(settings)

        await ctx.respond(
            f"Narrator voice set to **{voice}** ({provider}).",
            ephemeral=True,
        )

    @immersion.command(name="character_voice", description="Set the TTS provider for NPC/character dialogue")
    @discord.option(
        "provider",
        description="TTS provider for character voices",
        choices=[
            discord.OptionChoice("Fish Speech (local)", "fish"),
            discord.OptionChoice("Kokoro (local, fast)", "kokoro"),
            discord.OptionChoice("ElevenLabs (cloud)", "elevenlabs"),
            discord.OptionChoice("Same as narrator", ""),
        ],
    )
    async def immersion_character_voice(self, ctx: discord.ApplicationContext, provider: str):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)
        settings.character_tts_provider = provider
        await repo.upsert_guild_settings(settings)

        label = provider if provider else "same as narrator"
        await ctx.respond(
            f"Character voice provider set to **{label}**.",
            ephemeral=True,
        )

    @immersion.command(name="status", description="Show current immersion settings")
    async def immersion_status(self, ctx: discord.ApplicationContext):
        repo = await get_immersion_repo()
        settings = await repo.get_or_create_guild_settings(ctx.guild_id)

        freq_labels = {
            ImageFrequency.EVERY: "Every narration",
            ImageFrequency.SCENE_CHANGE: "Scene changes only",
            ImageFrequency.ON_DEMAND: "On demand (/imagine)",
        }

        embed = discord.Embed(
            title="Immersion Settings",
            color=discord.Color.purple(),
        )
        embed.add_field(
            name="TTS Narration",
            value="Enabled" if settings.tts_enabled else "Disabled",
            inline=True,
        )
        embed.add_field(
            name="Scene Images",
            value="Enabled" if settings.image_enabled else "Disabled",
            inline=True,
        )
        embed.add_field(
            name="Image Frequency",
            value=freq_labels.get(settings.image_frequency, settings.image_frequency.value),
            inline=True,
        )
        embed.add_field(
            name="Narrator Voice",
            value=f"{settings.narrator_tts_voice or '(default)'} ({settings.narrator_tts_provider})",
            inline=True,
        )
        char_provider = settings.character_tts_provider or "from catalog"
        embed.add_field(
            name="Character Voices",
            value=char_provider,
            inline=True,
        )

        await ctx.respond(embed=embed, ephemeral=True)

    @discord.slash_command(name="imagine", description="Generate a scene image from the current narrative")
    async def imagine(self, ctx: discord.ApplicationContext):
        """On-demand scene image generation."""
        await ctx.defer()

        from ...game.session import get_session_manager

        session_manager = get_session_manager()
        session = session_manager.get_session(ctx.channel_id)

        if not session:
            await ctx.followup.send("No active game session in this channel.", ephemeral=True)
            return

        try:
            from ...immersion.image_coordinator import generate_scene_image
            from ...game.scene.registry import get_scene_registry

            # Get scene context
            scene_registry = get_scene_registry(session.campaign_id, session.session_key)
            characters = [p.character for p in session.players.values() if p.character]

            image_bytes = await generate_scene_image(
                narrative=session.last_narrative or "The adventure continues...",
                scene_registry=scene_registry,
                characters=characters,
            )

            if image_bytes:
                import io
                file = discord.File(io.BytesIO(image_bytes), filename="scene.png")
                embed = discord.Embed(color=discord.Color.dark_gold())
                embed.set_image(url="attachment://scene.png")
                await ctx.followup.send(embed=embed, file=file)
            else:
                await ctx.followup.send("Could not generate scene image.", ephemeral=True)

        except Exception as e:
            logger.error("imagine_command_failed", error=str(e), exc_info=True)
            await ctx.followup.send(f"Image generation failed: {e}", ephemeral=True)


def setup(bot: commands.Bot):
    bot.add_cog(ImmersionCog(bot))
