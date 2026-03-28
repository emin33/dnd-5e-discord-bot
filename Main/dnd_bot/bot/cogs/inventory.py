"""Inventory management commands."""

import discord
from discord.ext import commands
import structlog

from ...models import InventoryItem
from ...data.repositories import get_character_repo, get_inventory_repo
from ...data.srd import get_srd
from ...game.session import get_session_manager

logger = structlog.get_logger()


class InventoryCog(commands.Cog):
    """Inventory management commands."""

    inventory = discord.SlashCommandGroup(
        "inventory",
        "Manage your character's inventory",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.srd = get_srd()
        self.session_manager = get_session_manager()

    def _get_campaign_id(self, ctx: discord.ApplicationContext) -> str:
        """Get campaign ID from active session, or fall back to 'default'."""
        session = self.session_manager.get_session(ctx.channel_id)
        if session:
            return session.campaign_id
        return "default"

    async def _get_character(self, ctx: discord.ApplicationContext, campaign_id: str | None = None):
        """Get the user's character for the active session's campaign."""
        if campaign_id is None:
            campaign_id = self._get_campaign_id(ctx)

        char_repo = await get_character_repo()
        character = await char_repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            session = self.session_manager.get_session(ctx.channel_id)
            if session:
                await ctx.respond(
                    f"You don't have a character in the active campaign. Create one with `/character create`.",
                    ephemeral=True,
                )
            else:
                await ctx.respond(
                    f"No active game session. Start one with `/game start` or use `/inventory list` in a game channel.",
                    ephemeral=True,
                )
            return None

        return character

    async def _recalculate_ac(self, character, inv_repo) -> None:
        """Recalculate character AC from currently equipped armor/shield.

        Looks up each equipped item in SRD to get armor_class data,
        then uses Character.calculate_ac_from_equipment().
        """
        try:
            equipped_items = await inv_repo.get_equipped_items(character.id)
            armor_data = []
            for item in equipped_items:
                srd_item = self.srd.get_equipment(item.item_index) if item.item_index else None
                if srd_item and "armor_class" in srd_item:
                    armor_data.append(srd_item)

            new_ac = character.calculate_ac_from_equipment(armor_data)
            if new_ac != character.armor_class:
                character.armor_class = new_ac
                char_repo = await get_character_repo()
                await char_repo.update(character)
        except Exception as e:
            logger.warning("ac_recalculate_failed", character=character.name, error=str(e))

    @inventory.command(name="list", description="Show your inventory")
    async def inventory_list(
        self,
        ctx: discord.ApplicationContext,
    ):
        """List all items in inventory."""
        await ctx.defer()
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_all_items(character.id)
        currency = await inv_repo.get_currency(character.id)

        embed = discord.Embed(
            title=f":school_satchel: {character.name}'s Inventory",
            color=discord.Color.dark_gold(),
        )

        # Currency
        currency_parts = []
        if currency.platinum > 0:
            currency_parts.append(f"{currency.platinum} pp")
        if currency.gold > 0:
            currency_parts.append(f"{currency.gold} gp")
        if currency.electrum > 0:
            currency_parts.append(f"{currency.electrum} ep")
        if currency.silver > 0:
            currency_parts.append(f"{currency.silver} sp")
        if currency.copper > 0:
            currency_parts.append(f"{currency.copper} cp")

        embed.add_field(
            name=":coin: Currency",
            value=", ".join(currency_parts) if currency_parts else "No money",
            inline=False,
        )

        # Equipped items
        equipped = [i for i in items if i.equipped]
        if equipped:
            equipped_lines = []
            for item in equipped:
                line = f":crossed_swords: **{item.item_name}**"
                if item.attuned:
                    line += " (attuned)"
                equipped_lines.append(line)

            embed.add_field(
                name="Equipped",
                value="\n".join(equipped_lines),
                inline=False,
            )

        # Other items
        unequipped = [i for i in items if not i.equipped]
        if unequipped:
            item_lines = []
            for item in unequipped[:15]:  # Limit display
                qty = f"x{item.quantity}" if item.quantity > 1 else ""
                item_lines.append(f"• {item.item_name} {qty}")

            if len(unequipped) > 15:
                item_lines.append(f"_...and {len(unequipped) - 15} more items_")

            embed.add_field(
                name=f"Items ({len(unequipped)})",
                value="\n".join(item_lines),
                inline=False,
            )
        elif not equipped:
            embed.add_field(
                name="Items",
                value="_Your inventory is empty_",
                inline=False,
            )

        await ctx.respond(embed=embed)

    @inventory.command(name="add", description="Add an item to your inventory")
    async def inventory_add(
        self,
        ctx: discord.ApplicationContext,
        item_name: discord.Option(
            str,
            "Name or SRD index of the item",
            required=True,
        ),
        quantity: discord.Option(
            int,
            "Quantity to add",
            required=False,
            default=1,
            min_value=1,
            max_value=999,
        ),
    ):
        """Add an item to inventory."""
        await ctx.defer()
        character = await self._get_character(ctx)
        if not character:
            return

        # Try to find item in SRD
        item_index = item_name.lower().replace(" ", "-")
        srd_item = self.srd.get_equipment(item_index)

        if srd_item:
            display_name = srd_item.get("name", item_name)
        else:
            # Custom item
            display_name = item_name

        inv_repo = await get_inventory_repo()
        item = InventoryItem(
            character_id=character.id,
            item_index=item_index,
            item_name=display_name,
            quantity=quantity,
        )

        await inv_repo.add_item(item)

        embed = discord.Embed(
            title=":package: Item Added",
            description=f"Added **{quantity}x {display_name}** to {character.name}'s inventory.",
            color=discord.Color.green(),
        )

        if srd_item:
            # Show item details
            cost = srd_item.get("cost", {})
            if cost:
                embed.add_field(
                    name="Value",
                    value=f"{cost.get('quantity', 0)} {cost.get('unit', 'gp')}",
                    inline=True,
                )

            weight = srd_item.get("weight")
            if weight:
                embed.add_field(
                    name="Weight",
                    value=f"{weight} lb",
                    inline=True,
                )

        await ctx.respond(embed=embed)

        logger.info(
            "item_added",
            character=character.name,
            item=display_name,
            quantity=quantity,
        )

    @inventory.command(name="remove", description="Remove an item from your inventory")
    async def inventory_remove(
        self,
        ctx: discord.ApplicationContext,
        item_name: discord.Option(
            str,
            "Name of the item to remove",
            required=True,
        ),
        quantity: discord.Option(
            int,
            "Quantity to remove",
            required=False,
            default=1,
            min_value=1,
        ),
    ):
        """Remove an item from inventory."""
        await ctx.defer()
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_all_items(character.id)

        # Find matching item
        item_name_lower = item_name.lower()
        matching = [i for i in items if item_name_lower in i.item_name.lower()]

        if not matching:
            await ctx.respond(
                f"No item matching '{item_name}' found in your inventory.",
                ephemeral=True,
            )
            return

        item = matching[0]

        if item.equipped:
            await ctx.respond(
                f"Cannot remove **{item.item_name}** - it's currently equipped. Unequip it first.",
                ephemeral=True,
            )
            return

        await inv_repo.remove_item(item.id, quantity)

        await ctx.respond(
            f":wastebasket: Removed **{min(quantity, item.quantity)}x {item.item_name}** from your inventory."
        )

    @inventory.command(name="equip", description="Equip an item")
    async def inventory_equip(
        self,
        ctx: discord.ApplicationContext,
        item_name: discord.Option(
            str,
            "Name of the item to equip",
            required=True,
        ),
    ):
        """Equip an item."""
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_all_items(character.id)

        # Find matching item
        item_name_lower = item_name.lower()
        matching = [i for i in items if item_name_lower in i.item_name.lower() and not i.equipped]

        if not matching:
            await ctx.respond(
                f"No unequipped item matching '{item_name}' found.",
                ephemeral=True,
            )
            return

        item = matching[0]
        await inv_repo.equip_item(item.id)

        # Recalculate AC if armor/shield equipped
        await self._recalculate_ac(character, inv_repo)

        await ctx.respond(f":crossed_swords: Equipped **{item.item_name}**.")

    @inventory.command(name="unequip", description="Unequip an item")
    async def inventory_unequip(
        self,
        ctx: discord.ApplicationContext,
        item_name: discord.Option(
            str,
            "Name of the item to unequip",
            required=True,
        ),
    ):
        """Unequip an item."""
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_equipped_items(character.id)

        # Find matching equipped item
        item_name_lower = item_name.lower()
        matching = [i for i in items if item_name_lower in i.item_name.lower()]

        if not matching:
            await ctx.respond(
                f"No equipped item matching '{item_name}' found.",
                ephemeral=True,
            )
            return

        item = matching[0]
        await inv_repo.unequip_item(item.id)

        # Recalculate AC after unequip
        await self._recalculate_ac(character, inv_repo)

        if item.attuned:
            await inv_repo.unattune_item(item.id)
            await ctx.respond(f":package: Unequipped and ended attunement to **{item.item_name}**.")
        else:
            await ctx.respond(f":package: Unequipped **{item.item_name}**.")

    @inventory.command(name="attune", description="Attune to a magic item")
    async def inventory_attune(
        self,
        ctx: discord.ApplicationContext,
        item_name: discord.Option(
            str,
            "Name of the item to attune to",
            required=True,
        ),
    ):
        """Attune to a magic item."""
        await ctx.defer()
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_all_items(character.id)

        # Find matching item
        item_name_lower = item_name.lower()
        matching = [i for i in items if item_name_lower in i.item_name.lower()]

        if not matching:
            await ctx.respond(
                f"No item matching '{item_name}' found in your inventory.",
                ephemeral=True,
            )
            return

        item = matching[0]

        if item.attuned:
            await ctx.respond(
                f"You're already attuned to **{item.item_name}**.",
                ephemeral=True,
            )
            return

        # Check attunement limit (3 items normally)
        attuned_count = await inv_repo.get_attuned_count(character.id)
        if attuned_count >= 3:
            await ctx.respond(
                "You're already attuned to 3 items (the maximum). End attunement to another item first.",
                ephemeral=True,
            )
            return

        await inv_repo.attune_item(item.id)

        embed = discord.Embed(
            title=":sparkles: Attunement Complete",
            description=f"You spend a short rest focusing on **{item.item_name}**, forming a magical bond with it.",
            color=discord.Color.purple(),
        )
        embed.add_field(
            name="Attunement Slots",
            value=f"{attuned_count + 1}/3 used",
            inline=True,
        )

        await ctx.respond(embed=embed)

    @inventory.command(name="give", description="Give an item to another player")
    async def inventory_give(
        self,
        ctx: discord.ApplicationContext,
        recipient: discord.Option(
            discord.Member,
            "Player to give the item to",
            required=True,
        ),
        item_name: discord.Option(
            str,
            "Name of the item to give",
            required=True,
        ),
        quantity: discord.Option(
            int,
            "Quantity to give",
            required=False,
            default=1,
            min_value=1,
        ),
    ):
        """Give an item to another player."""
        if recipient.bot:
            await ctx.respond("Cannot give items to bots.", ephemeral=True)
            return

        if recipient.id == ctx.author.id:
            await ctx.respond("Cannot give items to yourself.", ephemeral=True)
            return

        # Get campaign from active session
        campaign_id = self._get_campaign_id(ctx)

        # Get both characters
        char_repo = await get_character_repo()

        giver = await char_repo.get_by_user_and_campaign(ctx.author.id, campaign_id)
        if not giver:
            await ctx.respond(
                "You don't have a character in the active campaign.",
                ephemeral=True,
            )
            return

        receiver = await char_repo.get_by_user_and_campaign(recipient.id, campaign_id)
        if not receiver:
            await ctx.respond(
                f"{recipient.display_name} doesn't have a character in the active campaign.",
                ephemeral=True,
            )
            return

        inv_repo = await get_inventory_repo()
        items = await inv_repo.get_all_items(giver.id)

        # Find matching item
        item_name_lower = item_name.lower()
        matching = [i for i in items if item_name_lower in i.item_name.lower() and not i.equipped]

        if not matching:
            await ctx.respond(
                f"No unequipped item matching '{item_name}' found in your inventory.",
                ephemeral=True,
            )
            return

        item = matching[0]

        if quantity > item.quantity:
            await ctx.respond(
                f"You only have {item.quantity}x {item.item_name}.",
                ephemeral=True,
            )
            return

        await inv_repo.transfer_item(item.id, giver.id, receiver.id, quantity)

        await ctx.respond(
            f":handshake: **{giver.name}** gave **{quantity}x {item.item_name}** to **{receiver.name}**."
        )

        logger.info(
            "item_transferred",
            from_character=giver.name,
            to_character=receiver.name,
            item=item.item_name,
            quantity=quantity,
        )

    @inventory.command(name="gold", description="Manage your gold")
    async def inventory_gold(
        self,
        ctx: discord.ApplicationContext,
        action: discord.Option(
            str,
            "Add or remove gold",
            choices=["add", "remove", "check"],
            required=True,
        ),
        amount: discord.Option(
            int,
            "Amount of gold",
            required=False,
            default=0,
            min_value=0,
        ),
    ):
        """Manage gold."""
        character = await self._get_character(ctx)
        if not character:
            return

        inv_repo = await get_inventory_repo()

        if action == "check":
            currency = await inv_repo.get_currency(character.id)
            embed = discord.Embed(
                title=f":coin: {character.name}'s Wealth",
                color=discord.Color.gold(),
            )
            embed.add_field(name="Platinum", value=str(currency.platinum), inline=True)
            embed.add_field(name="Gold", value=str(currency.gold), inline=True)
            embed.add_field(name="Electrum", value=str(currency.electrum), inline=True)
            embed.add_field(name="Silver", value=str(currency.silver), inline=True)
            embed.add_field(name="Copper", value=str(currency.copper), inline=True)
            embed.add_field(name="Total (in GP)", value=f"{currency.total_in_gold:.2f}", inline=True)
            await ctx.respond(embed=embed)
            return

        if amount <= 0:
            await ctx.respond("Please specify a positive amount.", ephemeral=True)
            return

        if action == "add":
            currency = await inv_repo.add_gold(character.id, amount)
            await ctx.respond(f":coin: Added **{amount} gp** to {character.name}'s purse. Total: {currency.gold} gp")
        else:  # remove
            success, currency = await inv_repo.remove_gold(character.id, amount)
            if success:
                await ctx.respond(f":coin: Removed **{amount} gp** from {character.name}'s purse. Total: {currency.gold} gp")
            else:
                await ctx.respond(
                    f"Insufficient funds. You only have {currency.total_in_gold:.2f} gp worth of currency.",
                    ephemeral=True,
                )


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(InventoryCog(bot))
