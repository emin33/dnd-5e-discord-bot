"""Admin and utility commands for the D&D 5e bot."""

import time
from datetime import datetime

import discord
from discord.ext import commands
import structlog

from ...data import get_srd

logger = structlog.get_logger()


class AdminCog(commands.Cog):
    """Admin and utility commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @discord.slash_command(name="ping", description="Check if the bot is alive and measure latency")
    async def ping(self, ctx: discord.ApplicationContext):
        """Simple ping command to check bot responsiveness."""
        start = time.perf_counter()
        await ctx.respond("Pong!", ephemeral=True)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        ws_latency_ms = self.bot.latency * 1000

        await ctx.edit(
            content=(
                f"Pong!\n"
                f"Response latency: `{latency_ms:.1f}ms`\n"
                f"WebSocket latency: `{ws_latency_ms:.1f}ms`"
            )
        )

    @discord.slash_command(name="status", description="Show bot status and statistics")
    async def status(self, ctx: discord.ApplicationContext):
        """Show current bot status."""
        srd = get_srd()

        # Count SRD data
        spell_count = len(srd.get_all("spells"))
        monster_count = len(srd.get_all("monsters"))
        class_count = len(srd.get_all("classes"))
        race_count = len(srd.get_all("races"))

        embed = discord.Embed(
            title="D&D 5e Bot Status",
            color=discord.Color.green(),
            timestamp=datetime.utcnow(),
        )

        embed.add_field(
            name="Bot Info",
            value=(
                f"Guilds: {len(self.bot.guilds)}\n"
                f"Latency: {self.bot.latency * 1000:.1f}ms"
            ),
            inline=True,
        )

        embed.add_field(
            name="SRD Data Loaded",
            value=(
                f"Spells: {spell_count}\n"
                f"Monsters: {monster_count}\n"
                f"Classes: {class_count}\n"
                f"Races: {race_count}"
            ),
            inline=True,
        )

        embed.set_footer(text="D&D 5e Discord Bot")

        await ctx.respond(embed=embed)

    @discord.slash_command(name="srd", description="Look up SRD content")
    async def srd_lookup(
        self,
        ctx: discord.ApplicationContext,
        category: discord.Option(
            str,
            "Category to search",
            choices=["spell", "monster", "class", "race", "condition", "equipment"],
            required=True,
        ),
        name: discord.Option(str, "Name to search for", required=True),
    ):
        """Look up content from the SRD."""
        await ctx.defer()

        srd = get_srd()

        # Map category to SRD category name
        category_map = {
            "spell": "spells",
            "monster": "monsters",
            "class": "classes",
            "race": "races",
            "condition": "conditions",
            "equipment": "equipment",
        }

        srd_category = category_map.get(category, category)
        results = srd.search(srd_category, name)

        if not results:
            await ctx.followup.send(f"No {category} found matching '{name}'.")
            return

        # Show first result
        item = results[0]
        embed = self._format_srd_embed(category, item)

        # If multiple results, mention them
        if len(results) > 1:
            other_names = ", ".join(r["name"] for r in results[1:5])
            embed.set_footer(text=f"Also found: {other_names}{'...' if len(results) > 5 else ''}")

        await ctx.followup.send(embed=embed)

    def _format_srd_embed(self, category: str, item: dict) -> discord.Embed:
        """Format an SRD item as a Discord embed."""
        name = item.get("name", "Unknown")
        desc = item.get("desc", [])

        # Combine description paragraphs
        description = "\n\n".join(desc) if isinstance(desc, list) else str(desc)

        # Truncate if too long
        if len(description) > 2000:
            description = description[:1997] + "..."

        embed = discord.Embed(
            title=name,
            description=description or "No description available.",
            color=self._get_category_color(category),
        )

        # Add category-specific fields
        if category == "spell":
            embed.add_field(name="Level", value=str(item.get("level", "?")), inline=True)
            embed.add_field(
                name="School",
                value=item.get("school", {}).get("name", "Unknown"),
                inline=True,
            )
            embed.add_field(
                name="Casting Time",
                value=item.get("casting_time", "Unknown"),
                inline=True,
            )
            embed.add_field(name="Range", value=item.get("range", "Unknown"), inline=True)
            embed.add_field(name="Duration", value=item.get("duration", "Unknown"), inline=True)
            if item.get("concentration"):
                embed.add_field(name="Concentration", value="Yes", inline=True)

        elif category == "monster":
            embed.add_field(name="Size", value=item.get("size", "Unknown"), inline=True)
            embed.add_field(name="Type", value=item.get("type", "Unknown"), inline=True)
            embed.add_field(
                name="CR",
                value=str(item.get("challenge_rating", "?")),
                inline=True,
            )
            embed.add_field(
                name="HP",
                value=f"{item.get('hit_points', '?')} ({item.get('hit_points_roll', '?')})",
                inline=True,
            )
            embed.add_field(name="AC", value=str(item.get("armor_class", [{}])[0].get("value", "?")), inline=True)

        elif category == "class":
            embed.add_field(name="Hit Die", value=f"d{item.get('hit_die', '?')}", inline=True)

        elif category == "race":
            embed.add_field(name="Speed", value=str(item.get("speed", "?")) + " ft", inline=True)
            embed.add_field(name="Size", value=item.get("size", "Unknown"), inline=True)

        elif category == "condition":
            pass  # Description is enough

        elif category == "equipment":
            if "cost" in item:
                cost = item["cost"]
                embed.add_field(
                    name="Cost",
                    value=f"{cost.get('quantity', '?')} {cost.get('unit', '')}",
                    inline=True,
                )
            if "weight" in item:
                embed.add_field(name="Weight", value=f"{item['weight']} lb", inline=True)

        return embed

    def _get_category_color(self, category: str) -> discord.Color:
        """Get a color for a category."""
        colors = {
            "spell": discord.Color.purple(),
            "monster": discord.Color.red(),
            "class": discord.Color.blue(),
            "race": discord.Color.green(),
            "condition": discord.Color.orange(),
            "equipment": discord.Color.gold(),
        }
        return colors.get(category, discord.Color.greyple())


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(AdminCog(bot))
