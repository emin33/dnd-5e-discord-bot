"""Discord UI views for combat action selection.

These views provide the button/menu interface for structured combat turns.
They work with CombatTurnCoordinator to execute actions and display results.
"""

from typing import Optional, Callable, Awaitable
import discord

from ...game.combat.actions import (
    CombatAction,
    CombatActionType,
    ActionResult,
    TurnContext,
    WeaponStats,
)
from ...game.combat.coordinator import CombatTurnCoordinator
from ...game.session import get_session_manager
from ...models import CombatState
from ..embeds.combat_embed import build_combat_over_embed
from .base import SafeView


async def _reject_interaction(interaction: discord.Interaction, message: str) -> None:
    """Ephemeral rejection used by the views' interaction_check gates."""
    try:
        await interaction.response.send_message(message, ephemeral=True)
    except Exception:
        pass


async def _disable_items_and_ack(view: discord.ui.View, interaction: discord.Interaction) -> None:
    """Disable every item on a view and acknowledge by editing the message.

    Called BEFORE any awaited game mutation so double-clicks land on dead
    components (audit P0-6). Falls back to a bare defer if the edit fails;
    the callers' one-shot flags remain the real guard either way.
    """
    for item in view.children:
        item.disabled = True
    try:
        await interaction.response.edit_message(view=view)
    except Exception:
        try:
            await interaction.response.defer()
        except Exception:
            pass


async def _replace_with_fresh_action_menu(
    view: discord.ui.View,
    parent: "CombatActionView",
    interaction: discord.Interaction,
) -> None:
    """Replace a dead-ended view with a fresh, unclaimed action menu.

    Shared by the no-castable-slot dead end (adversarial review,
    should-fix 3) and the on_error recovery hooks: a fresh CombatActionView
    resets the one-shot claim so the player can still take their turn.
    Tries the interaction's own message first, then the action menu's
    original message (the slot-level sub-view lives on an ephemeral
    followup that can't host the recovered menu).
    """
    view.stop()
    fresh = CombatActionView(
        coordinator=parent.coordinator,
        turn_context=parent.ctx,
        on_action_complete=parent.on_action_complete,
        on_turn_end=parent.on_turn_end,
        actor_user_id=parent.actor_user_id,
    )
    embed = fresh.get_embed()
    for message in (interaction.message, parent.message):
        if message is None:
            continue
        try:
            await message.edit(content=None, embed=embed, view=fresh)
            fresh.message = message
            return
        except Exception:
            continue


class CombatActionView(SafeView):
    """
    Main action menu for a combatant's turn.

    Shows available actions as buttons based on remaining resources.
    """

    def __init__(
        self,
        coordinator: CombatTurnCoordinator,
        turn_context: TurnContext,
        on_action_complete: Callable[[ActionResult], Awaitable[None]],
        on_turn_end: Callable[[], Awaitable[None]],
        actor_user_id: Optional[int] = None,
    ):
        super().__init__(timeout=300)  # 5 minute timeout
        self.coordinator = coordinator
        self.ctx = turn_context
        self.on_action_complete = on_action_complete
        self.on_turn_end = on_turn_end
        self.actor_user_id = actor_user_id
        self.message: Optional[discord.Message] = None
        # One-shot guard (audit P0-6): set when a click claims this turn so
        # double-clicks and stray buttons can't drive a second mutation.
        self._acted = False

        self._build_buttons()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Gate clicks: only the acting player, and only until the turn is claimed."""
        if (
            self.actor_user_id is not None
            and interaction.user
            and interaction.user.id != self.actor_user_id
        ):
            await _reject_interaction(
                interaction,
                f"Only {self.ctx.combatant_name}'s player can act on this turn.",
            )
            return False
        if self._acted:
            await _reject_interaction(
                interaction, "That turn is already being resolved."
            )
            return False
        return True

    async def disable_and_claim(self, interaction: discord.Interaction) -> bool:
        """Claim the turn and disable all buttons BEFORE any game mutation.

        Returns False (after an ephemeral notice) if another click already
        claimed this view — the caller must bail without touching combat
        state. The coordinator's per-channel turn lock is the hard backstop
        for anything that still slips through (audit P0-6).
        """
        if self._acted:
            await _reject_interaction(
                interaction, "That turn is already being resolved."
            )
            return False
        self._acted = True
        await _disable_items_and_ack(self, interaction)
        return True

    async def on_error_recover(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        """A failed callback must not leave the turn bricked: release the
        one-shot claim and re-enable the buttons. The coordinator's resource
        tracking + per-channel turn lock remain the guards against double
        execution (audit P0-6).
        """
        self._acted = False
        if self.is_finished():
            # End Turn stops the view before its callback runs — a stopped
            # view no longer dispatches, so hand out a fresh menu instead.
            await _replace_with_fresh_action_menu(self, self, interaction)
            return
        for child in self.children:
            child.disabled = False
        try:
            message = interaction.message or self.message
            if message is not None:
                await message.edit(view=self)
        except Exception:
            pass

    def _build_buttons(self):
        """Build action buttons based on available resources."""
        # Row 0: Primary actions (require action)
        if self.ctx.has_action:
            self.add_item(AttackButton(self, row=0))
            if self.ctx.available_spells or self.ctx.monster_actions:
                self.add_item(CastSpellButton(self, row=0))
            self.add_item(DashButton(self, row=0))
            self.add_item(DodgeButton(self, row=0))

        # Row 1: Secondary actions
        if self.ctx.has_action:
            self.add_item(DisengageButton(self, row=1))
            self.add_item(HelpButton(self, row=1))
            self.add_item(HideButton(self, row=1))

        # Row 2: Always available
        self.add_item(EndTurnButton(self, row=2))

    def get_embed(self) -> discord.Embed:
        """Build the turn status embed."""
        embed = discord.Embed(
            title=f"{self.ctx.combatant_name}'s Turn",
            color=discord.Color.green() if self.ctx.is_player else discord.Color.red(),
        )

        # HP and AC
        hp_bar = self._build_hp_bar(self.ctx.hp_current, self.ctx.hp_max)
        embed.add_field(
            name="Status",
            value=f"HP: {self.ctx.hp_current}/{self.ctx.hp_max} {hp_bar}\nAC: {self.ctx.armor_class}",
            inline=True,
        )

        # Resources
        resources = []
        if self.ctx.has_action:
            resources.append("Action")
        if self.ctx.has_bonus_action:
            resources.append("Bonus Action")
        if self.ctx.has_reaction:
            resources.append("Reaction")
        resources.append(f"{self.ctx.movement_remaining}ft Movement")

        embed.add_field(
            name="Available Resources",
            value="\n".join(resources) if resources else "None",
            inline=True,
        )

        # Conditions
        if self.ctx.conditions:
            embed.add_field(
                name="Conditions",
                value=", ".join(self.ctx.conditions),
                inline=False,
            )

        # Position
        if self.ctx.in_melee_with:
            embed.add_field(
                name="In Melee With",
                value=", ".join(self.ctx.in_melee_with),
                inline=False,
            )
        else:
            embed.add_field(name="Position", value="At Range", inline=False)

        # Concentration
        if self.ctx.is_concentrating and self.ctx.concentration_spell:
            embed.set_footer(text=f"Concentrating on: {self.ctx.concentration_spell}")

        return embed

    def _build_hp_bar(self, current: int, max_hp: int, length: int = 10) -> str:
        """Build a visual HP bar."""
        if max_hp <= 0:
            return ""
        ratio = max(0, min(1, current / max_hp))
        filled = int(ratio * length)
        empty = length - filled
        return f"[{'█' * filled}{'░' * empty}]"


class AttackButton(discord.ui.Button):
    """Button to initiate an attack."""

    def __init__(self, parent: CombatActionView, row: int = 0):
        super().__init__(
            label="Attack",
            style=discord.ButtonStyle.danger,
            emoji="⚔️",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Show target selection (navigation only — the mutation happens in
        # the sub-view, which carries its own double-click guard)
        view = TargetSelectionView(
            coordinator=self.parent.coordinator,
            turn_context=self.parent.ctx,
            action_type=CombatActionType.ATTACK,
            on_select=self._on_target_selected,
            on_cancel=self._on_cancel,
            actor_user_id=self.parent.actor_user_id,
            parent=self.parent,
        )
        await interaction.response.edit_message(
            embed=view.get_embed(),
            view=view,
        )

    async def _on_target_selected(
        self,
        interaction: discord.Interaction,
        target_id: str,
        weapon: Optional[WeaponStats] = None,
    ):
        action = CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=self.parent.ctx.combatant_id,
            target_ids=[target_id],
            weapon_index=weapon.name if weapon else None,
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)

    async def _on_cancel(self, interaction: discord.Interaction):
        await interaction.response.edit_message(
            embed=self.parent.get_embed(),
            view=self.parent,
        )


class CastSpellButton(discord.ui.Button):
    """Button to cast a spell."""

    def __init__(self, parent: CombatActionView, row: int = 0):
        super().__init__(
            label="Cast Spell",
            style=discord.ButtonStyle.primary,
            emoji="✨",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        """Show spell selection UI."""
        async def on_cancel(cancel_interaction: discord.Interaction):
            await cancel_interaction.response.edit_message(
                content=None,
                embed=self.parent.get_embed(),
                view=self.parent,
            )

        view = SpellSelectionView(
            parent=self.parent,
            on_cancel=on_cancel,
        )
        await interaction.response.edit_message(
            content="Choose a spell to cast:",
            embed=None,
            view=view,
        )


class DashButton(discord.ui.Button):
    """Button to take the Dash action."""

    def __init__(self, parent: CombatActionView, row: int = 0):
        super().__init__(
            label="Dash",
            style=discord.ButtonStyle.secondary,
            emoji="🏃",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Claim the turn and disable buttons BEFORE the game mutation so a
        # double-click can't execute twice (audit P0-6).
        if not await self.parent.disable_and_claim(interaction):
            return

        action = CombatAction(
            action_type=CombatActionType.DASH,
            combatant_id=self.parent.ctx.combatant_id,
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)


class DodgeButton(discord.ui.Button):
    """Button to take the Dodge action."""

    def __init__(self, parent: CombatActionView, row: int = 0):
        super().__init__(
            label="Dodge",
            style=discord.ButtonStyle.secondary,
            emoji="🛡️",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Disable-before-mutate double-click guard (audit P0-6)
        if not await self.parent.disable_and_claim(interaction):
            return

        action = CombatAction(
            action_type=CombatActionType.DODGE,
            combatant_id=self.parent.ctx.combatant_id,
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)


class DisengageButton(discord.ui.Button):
    """Button to take the Disengage action."""

    def __init__(self, parent: CombatActionView, row: int = 1):
        super().__init__(
            label="Disengage",
            style=discord.ButtonStyle.secondary,
            emoji="🚪",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Disable-before-mutate double-click guard (audit P0-6)
        if not await self.parent.disable_and_claim(interaction):
            return

        action = CombatAction(
            action_type=CombatActionType.DISENGAGE,
            combatant_id=self.parent.ctx.combatant_id,
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)


class HelpButton(discord.ui.Button):
    """Button to take the Help action."""

    def __init__(self, parent: CombatActionView, row: int = 1):
        super().__init__(
            label="Help",
            style=discord.ButtonStyle.secondary,
            emoji="🤝",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Show ally selection for Help (navigation only — the sub-view
        # carries the double-click guard)
        view = TargetSelectionView(
            coordinator=self.parent.coordinator,
            turn_context=self.parent.ctx,
            action_type=CombatActionType.HELP,
            on_select=self._on_target_selected,
            on_cancel=self._on_cancel,
            allies_only=True,
            actor_user_id=self.parent.actor_user_id,
            parent=self.parent,
        )
        await interaction.response.edit_message(
            embed=view.get_embed(),
            view=view,
        )

    async def _on_target_selected(
        self,
        interaction: discord.Interaction,
        target_id: str,
        weapon: Optional[WeaponStats] = None,
    ):
        action = CombatAction(
            action_type=CombatActionType.HELP,
            combatant_id=self.parent.ctx.combatant_id,
            target_ids=[target_id],
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)

    async def _on_cancel(self, interaction: discord.Interaction):
        await interaction.response.edit_message(
            embed=self.parent.get_embed(),
            view=self.parent,
        )


class HideButton(discord.ui.Button):
    """Button to take the Hide action."""

    def __init__(self, parent: CombatActionView, row: int = 1):
        super().__init__(
            label="Hide",
            style=discord.ButtonStyle.secondary,
            emoji="🙈",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Disable-before-mutate double-click guard (audit P0-6)
        if not await self.parent.disable_and_claim(interaction):
            return

        action = CombatAction(
            action_type=CombatActionType.HIDE,
            combatant_id=self.parent.ctx.combatant_id,
        )

        result = await self.parent.coordinator.execute_action(action)
        await self.parent.on_action_complete(result)


class EndTurnButton(discord.ui.Button):
    """Button to end the current turn."""

    def __init__(self, parent: CombatActionView, row: int = 2):
        super().__init__(
            label="End Turn",
            style=discord.ButtonStyle.primary,
            emoji="⏭️",
            row=row,
        )
        self.parent = parent

    async def callback(self, interaction: discord.Interaction):
        # Claim + disable before the turn-end mutation (audit P0-6): a
        # double-click here used to advance the turn twice, skipping a
        # combatant.
        if not await self.parent.disable_and_claim(interaction):
            return
        self.parent.stop()
        await self.parent.on_turn_end()


class TargetSelectionView(SafeView):
    """
    View for selecting a target for an action.

    Shows a dropdown of valid targets based on the action type.
    """

    def __init__(
        self,
        coordinator: CombatTurnCoordinator,
        turn_context: TurnContext,
        action_type: CombatActionType,
        on_select: Callable[[discord.Interaction, str, Optional[WeaponStats]], Awaitable[None]],
        on_cancel: Callable[[discord.Interaction], Awaitable[None]],
        allies_only: bool = False,
        actor_user_id: Optional[int] = None,
        parent: Optional[CombatActionView] = None,
    ):
        super().__init__(timeout=60)
        self.coordinator = coordinator
        self.ctx = turn_context
        self.action_type = action_type
        self.on_select = on_select
        self.on_cancel = on_cancel
        self.allies_only = allies_only
        self.actor_user_id = actor_user_id
        # Action menu that spawned this sub-view — used by on_error recovery
        # to hand out a fresh, unclaimed menu.
        self.parent = parent
        # One-shot guard (audit P0-6): a double-select must not fire twice
        self._acted = False

        self.selected_weapon: Optional[WeaponStats] = None

        self._build_view()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Gate clicks: only the acting player, and only until a target is picked."""
        if (
            self.actor_user_id is not None
            and interaction.user
            and interaction.user.id != self.actor_user_id
        ):
            await _reject_interaction(
                interaction,
                f"Only {self.ctx.combatant_name}'s player can act on this turn.",
            )
            return False
        if self._acted:
            await _reject_interaction(
                interaction, "That action is already being resolved."
            )
            return False
        return True

    async def on_error_recover(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        """The target select claims + stops this view BEFORE on_select runs,
        so releasing _acted alone can't revive it — hand out a fresh action
        menu instead (same recovery as the no-castable-slot dead end).
        """
        self._acted = False
        if self.parent is not None:
            await _replace_with_fresh_action_menu(self, self.parent, interaction)

    def _build_view(self):
        """Build the selection view."""
        # Weapon select for attacks (if player has multiple weapons)
        if self.action_type == CombatActionType.ATTACK and len(self.ctx.equipped_weapons) > 1:
            weapon_options = [
                discord.SelectOption(
                    label=w.name,
                    value=w.name,
                    description=f"{w.damage_dice} {w.damage_type}",
                )
                for w in self.ctx.equipped_weapons
            ]

            weapon_select = discord.ui.Select(
                placeholder="Choose weapon...",
                options=weapon_options,
                row=0,
            )
            weapon_select.callback = self._on_weapon_select
            self.add_item(weapon_select)

            # Target select on row 1
            target_row = 1
        else:
            # Use first weapon by default
            if self.ctx.equipped_weapons:
                self.selected_weapon = self.ctx.equipped_weapons[0]
            target_row = 0

        # Build target options
        targets = self._get_valid_targets()

        if targets:
            target_options = [
                discord.SelectOption(
                    label=t["name"],
                    value=t["id"],
                    description=t.get("description", ""),
                )
                for t in targets
            ]

            target_select = discord.ui.Select(
                placeholder="Choose target...",
                options=target_options[:25],
                row=target_row,
            )
            target_select.callback = self._on_target_select
            self.add_item(target_select)
        else:
            # No valid targets - show disabled placeholder
            no_target = discord.ui.Button(
                label="No valid targets available",
                style=discord.ButtonStyle.secondary,
                disabled=True,
                row=target_row,
            )
            self.add_item(no_target)

        # Cancel button
        cancel = discord.ui.Button(
            label="Cancel",
            style=discord.ButtonStyle.secondary,
            row=target_row + 1,
        )
        cancel.callback = self._on_cancel_click
        self.add_item(cancel)

    def _get_valid_targets(self) -> list[dict]:
        """Get list of valid targets for the action."""
        targets = []
        combat = self.coordinator.manager.combat

        for combatant in combat.combatants:
            if not combatant.is_active or not combatant.is_conscious:
                continue

            # Skip self
            if combatant.id == self.ctx.combatant_id:
                continue

            # Filter by ally/enemy
            is_ally = combatant.is_player == self.ctx.is_player

            if self.allies_only and not is_ally:
                continue
            if not self.allies_only and self.action_type == CombatActionType.ATTACK and is_ally:
                # Don't show allies as attack targets (unless PvP enabled)
                continue

            # Check zone validity for melee attacks
            description = ""
            if self.action_type == CombatActionType.ATTACK:
                in_melee = self.coordinator.zone_tracker.is_in_melee_with(
                    self.ctx.combatant_id, combatant.id
                )
                if in_melee:
                    description = "In melee"
                else:
                    description = "At range"

            hp_display = f"HP: {combatant.hp_current}/{combatant.hp_max}"

            targets.append({
                "id": combatant.id,
                "name": combatant.name,
                "description": f"{hp_display} | {description}"[:100] if description else hp_display,
            })

        return targets

    def get_embed(self) -> discord.Embed:
        """Build the target selection embed."""
        action_name = self.action_type.value.replace("_", " ").title()

        embed = discord.Embed(
            title=f"Select Target for {action_name}",
            description="Choose your target from the dropdown below.",
            color=discord.Color.blue(),
        )

        if self.selected_weapon:
            embed.add_field(
                name="Weapon",
                value=f"{self.selected_weapon.name} ({self.selected_weapon.damage_dice} {self.selected_weapon.damage_type})",
                inline=False,
            )

        return embed

    async def _on_weapon_select(self, interaction: discord.Interaction):
        weapon_name = interaction.data["values"][0]
        self.selected_weapon = next(
            (w for w in self.ctx.equipped_weapons if w.name == weapon_name),
            self.ctx.equipped_weapons[0] if self.ctx.equipped_weapons else None,
        )
        await interaction.response.edit_message(embed=self.get_embed())

    async def _on_target_select(self, interaction: discord.Interaction):
        target_id = interaction.data["values"][0]
        # Claim + disable BEFORE the game mutation runs in on_select, so a
        # double-select can't execute the action twice (audit P0-6).
        if self._acted:
            return
        self._acted = True
        await _disable_items_and_ack(self, interaction)
        self.stop()
        await self.on_select(interaction, target_id, self.selected_weapon)

    async def _on_cancel_click(self, interaction: discord.Interaction):
        self.stop()
        await self.on_cancel(interaction)


class SpellSelectionView(SafeView):
    """
    View for selecting a spell to cast during combat.

    Shows a dropdown of the player's available spells, filtered to
    combat-relevant casting times. After selection, routes to target
    selection and/or slot level selection before executing.
    """

    def __init__(
        self,
        parent: CombatActionView,
        on_cancel: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=60)
        self.parent = parent
        self.on_cancel_cb = on_cancel
        # One-shot guard (audit P0-6): every route into the actual cast
        # funnels through _execute_cast — it must never run twice.
        self._cast_executed = False

        self._build_view()

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Gate clicks: only the acting player, and only until a cast fires."""
        if (
            self.parent.actor_user_id is not None
            and interaction.user
            and interaction.user.id != self.parent.actor_user_id
        ):
            await _reject_interaction(
                interaction,
                f"Only {self.parent.ctx.combatant_name}'s player can act on this turn.",
            )
            return False
        if self._cast_executed:
            await _reject_interaction(
                interaction, "That spell is already being resolved."
            )
            return False
        return True

    async def on_error_recover(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        """A failed spell flow may have disabled everything before raising —
        same dead-end shape as should-fix 3, same recovery.
        """
        await self._restore_action_menu(interaction)

    def _build_view(self):
        """Build spell selection dropdown."""
        from ...game.magic.spellcasting import SpellcastingManager

        spell_mgr = SpellcastingManager()
        options = []

        for spell_index in self.parent.ctx.available_spells[:25]:
            info = spell_mgr.get_spell_info(spell_index)
            if not info:
                continue
            # Filter to combat-relevant casting times
            if info.casting_time not in ("1 action", "1 bonus action"):
                continue
            # Filter out leveled spells with no castable slot remaining —
            # offering one dead-ended the turn in a fully disabled view
            # (adversarial review, should-fix 3). Cantrips always castable.
            if info.level > 0 and not any(
                level >= info.level and remaining > 0
                for level, remaining in self.parent.ctx.spell_slots.items()
            ):
                continue
            level_label = f"Cantrip" if info.level == 0 else f"Level {info.level}"
            options.append(discord.SelectOption(
                label=info.name,
                value=info.index,
                description=f"{level_label} | {info.casting_time}",
            ))

        if options:
            spell_select = discord.ui.Select(
                placeholder="Choose a spell...",
                options=options[:25],
                row=0,
            )
            spell_select.callback = self._on_spell_select
            self.add_item(spell_select)
        else:
            # No valid spells — shouldn't happen but handle gracefully
            pass

        cancel = discord.ui.Button(
            label="Cancel",
            style=discord.ButtonStyle.secondary,
            row=1,
        )
        cancel.callback = self._on_cancel
        self.add_item(cancel)

    async def _on_spell_select(self, interaction: discord.Interaction):
        """Handle spell selection — route to target/slot selection."""
        from ...game.magic.spellcasting import SpellcastingManager, SpellType

        spell_index = interaction.data["values"][0]
        spell_mgr = SpellcastingManager()
        info = spell_mgr.get_spell_info(spell_index)

        if not info:
            await interaction.response.send_message("Spell not found.", ephemeral=True)
            return

        spell_type = spell_mgr.get_spell_type(info)

        # Determine if we need target selection
        needs_target = spell_type in (SpellType.ATTACK, SpellType.SAVE)
        is_healing = spell_type == SpellType.HEALING

        if needs_target or is_healing:
            # Show target selection, then proceed to slot/execute
            async def on_target_selected(target_interaction, target_id, _weapon):
                await self._proceed_to_slot_or_execute(
                    target_interaction, spell_index, info.level, [target_id]
                )

            target_view = TargetSelectionView(
                coordinator=self.parent.coordinator,
                turn_context=self.parent.ctx,
                action_type=CombatActionType.CAST_SPELL,
                on_select=on_target_selected,
                on_cancel=self.on_cancel_cb,
                allies_only=is_healing,
                actor_user_id=self.parent.actor_user_id,
                parent=self.parent,
            )
            await interaction.response.edit_message(
                content=f"Casting **{info.name}** — choose a target:",
                embed=None,
                view=target_view,
            )
        else:
            # Utility/self spell — no target needed. Disable the dropdown
            # BEFORE executing so a double-select can't cast twice
            # (audit P0-6).
            await _disable_items_and_ack(self, interaction)
            await self._proceed_to_slot_or_execute(
                interaction, spell_index, info.level, []
            )

    async def _proceed_to_slot_or_execute(
        self,
        interaction: discord.Interaction,
        spell_index: str,
        spell_level: int,
        target_ids: list[str],
    ):
        """After target selection, pick slot level or execute directly."""
        if spell_level == 0:
            # Cantrip — no slot needed, execute directly
            await self._execute_cast(interaction, spell_index, 0, target_ids)
            return

        # Find valid slot levels (>= spell level with remaining slots)
        valid_levels = []
        for level, remaining in self.parent.ctx.spell_slots.items():
            if level >= spell_level and remaining > 0:
                valid_levels.append(level)

        if not valid_levels:
            try:
                await interaction.followup.send(
                    "No spell slots available for this spell.", ephemeral=True
                )
            except Exception:
                pass
            # Recover instead of returning into a view whose every component
            # — including Cancel — is already disabled (adversarial review,
            # should-fix 3).
            await self._restore_action_menu(interaction)
            return

        if len(valid_levels) == 1:
            # Only one valid level — use it directly
            await self._execute_cast(interaction, spell_index, valid_levels[0], target_ids)
            return

        # Multiple valid levels — show slot selection
        slot_view = SlotLevelSelectionView(
            parent=self.parent,
            spell_index=spell_index,
            valid_levels=valid_levels,
            target_ids=target_ids,
            on_cast=self._execute_cast,
            on_cancel=self.on_cancel_cb,
        )
        try:
            await interaction.followup.send(
                content=f"Choose spell slot level (minimum {spell_level}):",
                view=slot_view,
                ephemeral=True,
            )
        except Exception:
            # Fallback: use lowest valid slot
            await self._execute_cast(interaction, spell_index, valid_levels[0], target_ids)

    async def _restore_action_menu(self, interaction: discord.Interaction) -> None:
        """Replace a dead-ended sub-view with a fresh, unclaimed action menu.

        Reached when the cast can no longer proceed (e.g. no castable slot
        remained) AFTER the sub-view disabled all of its components. A fresh
        CombatActionView resets the one-shot claim so the player can still
        take their turn (adversarial review, should-fix 3).
        """
        await _replace_with_fresh_action_menu(self, self.parent, interaction)

    async def _execute_cast(
        self,
        interaction: discord.Interaction,
        spell_index: str,
        slot_level: int,
        target_ids: list[str],
    ):
        """Construct CombatAction and execute through coordinator."""
        # One-shot guard (audit P0-6): target select, slot select, and the
        # direct path all funnel here — never execute the cast twice.
        if self._cast_executed:
            return
        self._cast_executed = True

        action = CombatAction(
            action_type=CombatActionType.CAST_SPELL,
            combatant_id=self.parent.ctx.combatant_id,
            target_ids=target_ids,
            spell_index=spell_index,
            slot_level=slot_level,
        )

        result = await self.parent.coordinator.execute_action(action)
        self.stop()
        await self.parent.on_action_complete(result)

    async def _on_cancel(self, interaction: discord.Interaction):
        self.stop()
        await self.on_cancel_cb(interaction)


class SlotLevelSelectionView(SafeView):
    """View for selecting which spell slot level to use when upcasting."""

    def __init__(
        self,
        parent: CombatActionView,
        spell_index: str,
        valid_levels: list[int],
        target_ids: list[str],
        on_cast: Callable,
        on_cancel: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=30)
        self.parent = parent
        self.spell_index = spell_index
        self.target_ids = target_ids
        self.on_cast = on_cast
        self.on_cancel_cb = on_cancel
        # One-shot guard (audit P0-6): a double-select must not cast twice
        self._acted = False

        options = [
            discord.SelectOption(
                label=f"Level {level} slot",
                value=str(level),
                description=f"{parent.ctx.spell_slots.get(level, 0)} remaining",
            )
            for level in valid_levels
        ]

        slot_select = discord.ui.Select(
            placeholder="Choose slot level...",
            options=options[:25],
            row=0,
        )
        slot_select.callback = self._on_select
        self.add_item(slot_select)

        cancel = discord.ui.Button(
            label="Cancel",
            style=discord.ButtonStyle.secondary,
            row=1,
        )
        cancel.callback = self._on_cancel
        self.add_item(cancel)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Gate clicks: only the acting player, and only until a slot is picked."""
        if (
            self.parent.actor_user_id is not None
            and interaction.user
            and interaction.user.id != self.parent.actor_user_id
        ):
            await _reject_interaction(
                interaction,
                f"Only {self.parent.ctx.combatant_name}'s player can act on this turn.",
            )
            return False
        if self._acted:
            await _reject_interaction(
                interaction, "That spell is already being resolved."
            )
            return False
        return True

    async def on_error_recover(
        self,
        error: Exception,
        item: discord.ui.Item,
        interaction: discord.Interaction,
    ) -> None:
        """The slot select claims + stops this view BEFORE the cast runs, so
        releasing _acted alone can't revive it — hand out a fresh action menu
        (this view lives on an ephemeral followup, so the helper falls back
        to the action menu's original message).
        """
        self._acted = False
        await _replace_with_fresh_action_menu(self, self.parent, interaction)

    async def _on_select(self, interaction: discord.Interaction):
        slot_level = int(interaction.data["values"][0])
        # Claim + disable BEFORE the cast executes (audit P0-6)
        if self._acted:
            return
        self._acted = True
        await _disable_items_and_ack(self, interaction)
        self.stop()
        await self.on_cast(interaction, self.spell_index, slot_level, self.target_ids)

    async def _on_cancel(self, interaction: discord.Interaction):
        self.stop()
        await self.on_cancel_cb(interaction)


class ActionResultEmbed:
    """Helper for building action result embeds."""

    @staticmethod
    def build(result: ActionResult, target_name: str = "") -> discord.Embed:
        """Build an embed displaying action results."""
        action_name = result.action.action_type.value.replace("_", " ").title()

        if result.success:
            color = discord.Color.green()
            title = f"{action_name} - Success!"
        else:
            color = discord.Color.red()
            title = f"{action_name} - Failed"

        embed = discord.Embed(title=title, color=color)

        # Error message
        if result.error:
            embed.description = f"**Error:** {result.error}"
            return embed

        # Attack roll details
        if result.attack_roll:
            roll = result.attack_roll
            roll_text = f"**{roll.total}**"

            if result.critical_hit:
                roll_text += " 💥 **CRITICAL HIT!**"
            elif result.critical_miss:
                roll_text += " 💀 **Critical Miss!**"
            else:
                hit_miss = "Hit!" if result.damage_dealt else "Miss"
                if result.target_ac:
                    roll_text += f" vs AC {result.target_ac} - {hit_miss}"

            embed.add_field(name="Attack Roll", value=roll_text, inline=False)

        # Damage
        if result.damage_dealt:
            for target_id, damage in result.damage_dealt.items():
                damage_text = f"**{damage}** {result.damage_type or ''} damage"

                if target_id in result.damage_resisted:
                    modifier = result.damage_resisted[target_id]
                    if modifier == "resistance":
                        damage_text += " (resisted!)"
                    elif modifier == "vulnerability":
                        damage_text += " (vulnerable!)"
                    elif modifier == "immunity":
                        damage_text += " (immune!)"

                embed.add_field(name="Damage", value=damage_text, inline=True)

        # Healing
        if result.healing_done:
            total_heal = sum(result.healing_done.values())
            embed.add_field(name="Healing", value=f"**{total_heal}** HP restored", inline=True)

        # Status effects
        if result.unconscious_targets:
            embed.add_field(
                name="Knocked Unconscious",
                value=", ".join(result.unconscious_targets),
                inline=False,
            )

        if result.killed_targets:
            embed.add_field(
                name="Killed",
                value=", ".join(result.killed_targets),
                inline=False,
            )

        if result.concentration_broken:
            embed.add_field(
                name="Concentration",
                value="Concentration broken!",
                inline=False,
            )

        # Zone changes
        if result.zone_changes:
            embed.add_field(
                name="Effects",
                value="\n".join(result.zone_changes),
                inline=False,
            )

        # Skill roll
        if result.skill_roll:
            embed.add_field(
                name="Skill Check",
                value=f"Rolled **{result.skill_roll.total}**",
                inline=True,
            )

        return embed


class NPCTurnView(SafeView):
    """
    Simple view for when NPCs go first in combat.

    Provides a button to run NPC turns automatically.

    No on_error_recover override: both button callbacks already release
    _busy and re-enable the buttons in their try/finally (adversarial
    review, blocker 1d) before the exception reaches SafeView.on_error,
    which only needs to log and notify.
    """

    def __init__(
        self,
        coordinator: CombatTurnCoordinator,
        channel: discord.TextChannel,
        on_turns_complete: Callable[[], Awaitable[None]],
    ):
        super().__init__(timeout=600)  # 10 minute timeout
        self.coordinator = coordinator
        self.channel = channel
        self.on_turns_complete = on_turns_complete
        # In-flight guard (audit P0-6): NPC turns have no owning player, so
        # there is no actor to restrict to — but clicks during a run are
        # rejected, and buttons disable BEFORE the slow multi-LLM-call turn.
        self._busy = False

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Anyone at the table may drive NPC turns, but only one run at a time."""
        if self._busy:
            await _reject_interaction(
                interaction, "NPC turns are already running."
            )
            return False
        return True

    async def _finish_combat(self) -> None:
        """Victory/defeat embed + teardown through the single owner
        (GameSessionManager.end_combat) — adversarial review, blocker 1d."""
        embed = build_combat_over_embed(self.coordinator.manager.combat)
        await self.channel.send(embed=embed)
        await get_session_manager().end_combat(self.channel.id)

    @discord.ui.button(
        label="Run NPC Turn",
        style=discord.ButtonStyle.danger,
        emoji="💀",
    )
    async def run_npc_turn(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Run the current NPC's turn."""
        current = self.coordinator.manager.combat.get_current_combatant()
        if (
            self.coordinator.manager.combat.state == CombatState.COMBAT_END
            or not current
            or current.is_player
        ):
            await interaction.response.send_message("No NPC turn to run.", ephemeral=True)
            return

        # Disable buttons and edit BEFORE the multi-LLM-call turn so the
        # whole slow window isn't re-clickable (audit P0-6).
        self._busy = True
        await _disable_items_and_ack(self, interaction)

        combat_over = False
        turn_ran = False
        try:
            # Run the NPC turn
            results = await self.coordinator.run_npc_turn(current)
            turn_ran = True

            for result in results:
                result_embed = ActionResultEmbed.build(result)
                await self.channel.send(embed=result_embed)

                # Narrate the result
                try:
                    narrative = await self.coordinator.narrate_result(result)
                    if narrative:
                        await self.channel.send(f"*{narrative}*")
                except Exception:
                    pass

            combat_over = self.coordinator.manager.combat.is_combat_over()
        finally:
            # Always release the view (adversarial review, blocker 1d): a
            # raised turn must not leave _busy stuck True with every button
            # dead.
            self._busy = False
            if combat_over:
                for item in self.children:
                    item.disabled = True
                self.stop()
            elif turn_ran:
                # Pre-hardening UX: this button stays used-up after one
                # turn, the "Run All" button comes back.
                for item in self.children:
                    if isinstance(item, discord.ui.Button):
                        item.disabled = False
                button.disabled = True
            else:
                # The turn raised — restore every button so the table can
                # retry instead of staring at a bricked view.
                for item in self.children:
                    item.disabled = False
            try:
                await interaction.message.edit(view=self)
            except Exception:
                pass

        if combat_over:
            # NPC-side victory (e.g. TPK): defeat/victory embed + teardown
            # through the single owner (adversarial review, blocker 1d).
            await self._finish_combat()
            return

        # Call completion callback
        await self.on_turns_complete()

    @discord.ui.button(
        label="Run All NPC Turns",
        style=discord.ButtonStyle.secondary,
        emoji="⚔️",
    )
    async def run_all_npc_turns(self, button: discord.ui.Button, interaction: discord.Interaction):
        """Run all consecutive NPC turns automatically."""
        # Disable buttons and edit BEFORE the long multi-turn run so the
        # whole slow window isn't re-clickable (audit P0-6).
        self._busy = True
        await _disable_items_and_ack(self, interaction)

        combat_over = False
        completed = False
        try:
            turns_run = 0
            max_turns = 10

            while turns_run < max_turns:
                if self.coordinator.manager.combat.state == CombatState.COMBAT_END:
                    # Ended elsewhere (teardown raced us) — nothing to run,
                    # the view is done (adversarial review, should-fix 2).
                    completed = True
                    self.stop()
                    return
                current = self.coordinator.manager.combat.get_current_combatant()
                if not current:
                    break
                if current.is_player:
                    break

                turns_run += 1
                await self.channel.send(f":skull: **{current.name}**'s turn...")

                results = await self.coordinator.run_npc_turn(current)

                for result in results:
                    result_embed = ActionResultEmbed.build(result)
                    await self.channel.send(embed=result_embed)

                    try:
                        narrative = await self.coordinator.narrate_result(result)
                        if narrative:
                            await self.channel.send(f"*{narrative}*")
                    except Exception:
                        pass

                # Check if combat is over
                if self.coordinator.manager.combat.is_combat_over():
                    combat_over = True
                    break
            completed = True
        finally:
            self._busy = False
            if not completed:
                # A raised turn must not leave the view bricked (adversarial
                # review, blocker 1d): restore the buttons for a retry.
                for item in self.children:
                    item.disabled = False
                try:
                    await interaction.message.edit(view=self)
                except Exception:
                    pass

        if combat_over:
            # NPC-side victory (e.g. TPK): defeat/victory embed + teardown
            # through the single owner (adversarial review, blocker 1d).
            self.stop()
            await self._finish_combat()
            return

        # Buttons were already disabled (and the message edited) before the
        # run; this view is used up.
        await self.on_turns_complete()
