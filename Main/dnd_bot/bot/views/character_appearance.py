"""Character appearance modal and portrait generation view.

Added as an optional step in the character creation wizard.
Players can describe their character's appearance (stored as character.description)
and optionally generate a portrait image.
"""

import asyncio
from typing import Optional, Callable, Awaitable

import discord
import structlog

logger = structlog.get_logger()


class AppearanceModal(discord.ui.Modal):
    """Modal for entering character appearance description."""

    def __init__(self, callback: Callable[[str], Awaitable[None]]):
        super().__init__(title="Character Appearance")
        self._callback = callback

        self.description_input = discord.ui.InputText(
            label="Describe your character's appearance",
            placeholder="A tall elf with silver hair and emerald eyes, wearing worn leather armor...",
            style=discord.InputTextStyle.paragraph,
            required=False,
            max_length=500,
        )
        self.add_item(self.description_input)

    async def callback(self, interaction: discord.Interaction):
        description = self.description_input.value or ""
        await interaction.response.defer()
        await self._callback(description)


class PortraitGenerationView(discord.ui.View):
    """View for generating and approving a character portrait."""

    def __init__(
        self,
        description: str,
        race: str,
        class_name: str,
        on_approve: Callable[[str, Optional[bytes]], Awaitable[None]],
        on_skip: Callable[[], Awaitable[None]],
    ):
        super().__init__(timeout=120)
        self.description = description
        self.race = race
        self.class_name = class_name
        self._on_approve = on_approve
        self._on_skip = on_skip
        self._current_image: Optional[bytes] = None
        self._interaction: Optional[discord.Interaction] = None

    @discord.ui.button(label="Generate Portrait", style=discord.ButtonStyle.primary)
    async def generate_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        self._interaction = interaction
        await interaction.response.defer()

        try:
            from ...immersion.image_factory import get_image_provider

            prompt = self._build_portrait_prompt()
            provider = get_image_provider()
            self._current_image = await provider.generate(prompt, size="1024x1024")

            if self._current_image:
                import io
                file = discord.File(io.BytesIO(self._current_image), filename="portrait.png")
                embed = discord.Embed(
                    title="Character Portrait",
                    description="Approve this portrait or retry for a new one.",
                    color=discord.Color.purple(),
                )
                embed.set_image(url="attachment://portrait.png")

                # Switch to approve/retry/skip buttons
                approve_view = _PortraitApproveView(
                    image_bytes=self._current_image,
                    on_approve=self._on_approve,
                    on_retry=self._retry_generation,
                    on_skip=self._on_skip,
                    description=self.description,
                )
                await interaction.followup.send(embed=embed, file=file, view=approve_view)
            else:
                await interaction.followup.send("Portrait generation failed. You can skip this step.", ephemeral=True)

        except Exception as e:
            logger.warning("portrait_generation_failed", error=str(e))
            await interaction.followup.send(
                f"Portrait generation unavailable: {e}\nYou can skip this step.",
                ephemeral=True,
            )

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary)
    async def skip_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self._on_skip()

    def _build_portrait_prompt(self) -> str:
        """Build a portrait generation prompt from character info."""
        parts = [
            "D&D character portrait, bust shot, facing forward",
            f"{self.race} {self.class_name}",
        ]
        if self.description:
            parts.append(self.description)
        parts.append("fantasy art style, detailed, dramatic lighting, clean background")
        return ", ".join(parts)

    async def _retry_generation(self):
        """Retry portrait generation."""
        if self._interaction:
            try:
                from ...immersion.image_factory import get_image_provider

                prompt = self._build_portrait_prompt()
                provider = get_image_provider()
                self._current_image = await provider.generate(prompt, size="1024x1024")

                if self._current_image:
                    import io
                    file = discord.File(io.BytesIO(self._current_image), filename="portrait.png")
                    embed = discord.Embed(
                        title="Character Portrait (Retry)",
                        description="Approve this portrait or retry again.",
                        color=discord.Color.purple(),
                    )
                    embed.set_image(url="attachment://portrait.png")

                    approve_view = _PortraitApproveView(
                        image_bytes=self._current_image,
                        on_approve=self._on_approve,
                        on_retry=self._retry_generation,
                        on_skip=self._on_skip,
                        description=self.description,
                    )
                    await self._interaction.followup.send(embed=embed, file=file, view=approve_view)

            except Exception as e:
                logger.warning("portrait_retry_failed", error=str(e))


class _PortraitApproveView(discord.ui.View):
    """Sub-view for approving/retrying a generated portrait."""

    def __init__(
        self,
        image_bytes: bytes,
        on_approve: Callable[[str, Optional[bytes]], Awaitable[None]],
        on_retry: Callable[[], Awaitable[None]],
        on_skip: Callable[[], Awaitable[None]],
        description: str,
    ):
        super().__init__(timeout=120)
        self._image_bytes = image_bytes
        self._on_approve = on_approve
        self._on_retry = on_retry
        self._on_skip = on_skip
        self._description = description

    @discord.ui.button(label="Approve", style=discord.ButtonStyle.success)
    async def approve_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self._on_approve(self._description, self._image_bytes)

    @discord.ui.button(label="Retry", style=discord.ButtonStyle.primary)
    async def retry_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        await self._on_retry()

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary)
    async def skip_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self._on_skip()
