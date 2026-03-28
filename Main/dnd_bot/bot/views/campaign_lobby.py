"""Campaign lobby view with join functionality."""

from typing import Callable, Awaitable, Optional
import discord

from ...models import Campaign


class CampaignLobbyView(discord.ui.View):
    """Persistent lobby view for a campaign with Join and Start Game buttons."""

    def __init__(
        self,
        campaign: Campaign,
        dm_id: int,
        on_join: Callable[[discord.Interaction, Campaign], Awaitable[None]],
        on_start: Callable[[discord.Interaction, Campaign], Awaitable[None]],
        players: Optional[list[int]] = None,
    ):
        # Persistent view - no timeout
        super().__init__(timeout=None)
        self.campaign = campaign
        self.dm_id = dm_id
        self.on_join = on_join
        self.on_start = on_start
        self.players = players or []

    def get_embed(self) -> discord.Embed:
        """Build the lobby embed."""
        embed = discord.Embed(
            title=f":scroll: {self.campaign.name}",
            description=self.campaign.description or "A new adventure awaits!",
            color=discord.Color.gold(),
        )

        embed.add_field(
            name=":crown: Dungeon Master",
            value=f"<@{self.dm_id}>",
            inline=True,
        )

        embed.add_field(
            name=":busts_in_silhouette: Players",
            value=str(len(self.players)) if self.players else "0",
            inline=True,
        )

        if self.players:
            player_mentions = [f"<@{p}>" for p in self.players[:10]]
            if len(self.players) > 10:
                player_mentions.append(f"_+{len(self.players) - 10} more_")
            embed.add_field(
                name="Joined",
                value="\n".join(player_mentions),
                inline=False,
            )

        embed.set_footer(text="Click 'Join Campaign' to create a character and join!")

        return embed

    @discord.ui.button(
        label="Join Campaign",
        style=discord.ButtonStyle.success,
        emoji="🎲",
        custom_id="campaign_join",
    )
    async def join_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Handle join button click."""
        await self.on_join(interaction, self.campaign)

    @discord.ui.button(
        label="Start Game",
        style=discord.ButtonStyle.primary,
        emoji="⚔️",
        custom_id="campaign_start",
    )
    async def start_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Handle start game button click - DM only."""
        if interaction.user.id != self.dm_id:
            await interaction.response.send_message(
                "Only the Dungeon Master can start the game!",
                ephemeral=True,
            )
            return
        await self.on_start(interaction, self.campaign)


# Track active campaign per guild (in-memory for now)
_active_campaigns: dict[int, str] = {}  # guild_id -> campaign_id


def set_active_campaign(guild_id: int, campaign_id: str) -> None:
    """Set the active campaign for a guild."""
    _active_campaigns[guild_id] = campaign_id


def get_active_campaign_id(guild_id: int) -> Optional[str]:
    """Get the active campaign ID for a guild."""
    return _active_campaigns.get(guild_id)


def clear_active_campaign(guild_id: int) -> None:
    """Clear the active campaign for a guild."""
    _active_campaigns.pop(guild_id, None)
