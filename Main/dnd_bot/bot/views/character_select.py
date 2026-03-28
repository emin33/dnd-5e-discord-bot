"""Character selection view for joining campaigns."""

from typing import Callable, Awaitable, Optional
import discord

from ...models import Character


class CharacterSelectView(discord.ui.View):
    """View for selecting an existing character or creating a new one."""

    def __init__(
        self,
        existing_characters: list[Character],
        on_select_existing: Callable[[discord.Interaction, Character], Awaitable[None]],
        on_create_new: Callable[[discord.Interaction], Awaitable[None]],
        on_quick_join: Optional[Callable[[discord.Interaction], Awaitable[None]]] = None,
    ):
        super().__init__(timeout=300)
        self.existing_characters = existing_characters
        self.on_select_existing = on_select_existing
        self.on_create_new = on_create_new
        self.on_quick_join = on_quick_join

        # Add character select dropdown if there are existing characters
        if existing_characters:
            self._add_character_select()

    def _add_character_select(self):
        """Add the character selection dropdown."""
        options = [
            discord.SelectOption(
                label=char.name,
                value=char.id,
                description=f"Level {char.level} {char.race_index.title()} {char.class_index.title()}"[:100],
                emoji="🎭",
            )
            for char in self.existing_characters[:25]  # Discord limit
        ]

        select = discord.ui.Select(
            placeholder="Select an existing character...",
            options=options,
            row=0,
        )
        select.callback = self._on_character_selected
        self.add_item(select)

    async def _on_character_selected(self, interaction: discord.Interaction):
        """Handle character selection."""
        character_id = interaction.data["values"][0]
        character = next(
            (c for c in self.existing_characters if c.id == character_id),
            None
        )
        if character:
            await interaction.response.defer()
            self.stop()
            await self.on_select_existing(interaction, character)

    @discord.ui.button(
        label="Create New Character",
        style=discord.ButtonStyle.success,
        emoji="✨",
        row=1,
    )
    async def create_new_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Handle create new character button."""
        await interaction.response.defer()
        self.stop()
        await self.on_create_new(interaction)

    @discord.ui.button(
        label="Quick Join (Test)",
        style=discord.ButtonStyle.secondary,
        emoji="🧪",
        row=1,
    )
    async def quick_join_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Handle quick join with template character."""
        if not self.on_quick_join:
            await interaction.response.send_message(
                "Quick join not available.",
                ephemeral=True,
            )
            return
        await interaction.response.defer()
        self.stop()
        await self.on_quick_join(interaction)

    def get_embed(self, campaign_name: str) -> discord.Embed:
        """Build the selection embed."""
        embed = discord.Embed(
            title=f"Join {campaign_name}",
            color=discord.Color.blue(),
        )

        if self.existing_characters:
            embed.description = (
                "You have existing characters on this server!\n\n"
                "**Select a character** to use them in this campaign, or\n"
                "**Create a new character** to start fresh."
            )

            # List existing characters
            char_list = []
            for char in self.existing_characters[:10]:
                char_list.append(
                    f"**{char.name}** - Level {char.level} {char.race_index.title()} {char.class_index.title()}"
                )

            embed.add_field(
                name="Your Characters",
                value="\n".join(char_list),
                inline=False,
            )
        else:
            embed.description = (
                "Welcome, adventurer!\n\n"
                "Click **Create New Character** to build your hero for this campaign."
            )

        return embed
