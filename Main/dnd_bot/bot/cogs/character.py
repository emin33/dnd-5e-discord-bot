"""Character management commands."""

from typing import Optional
import discord
from discord.ext import commands
import structlog

from ...models import AbilityScore, AbilityScores, Skill
from ...data.srd import get_srd
from ...data.repositories.character_repo import get_character_repo
from ...game.character.creation import (
    AbilityScoreMethod,
    CharacterCreationState,
    get_creator,
    STANDARD_ARRAY,
)
from ...game.character.leveling import (
    get_leveling_manager,
    can_level_up,
    get_xp_progress,
    get_xp_for_next_level,
    get_asi_levels,
    XP_THRESHOLDS,
)
from ..views.character_creation import (
    AbilityScoreMethodView,
    AbilityAssignmentView,
    ClassSelectView,
    ConfirmCharacterView,
    NameModal,
    PointBuyView,
    RaceSelectView,
    SkillSelectView,
)
from ..embeds.character_sheet import (
    build_ability_roll_embed,
    build_character_sheet_embed,
    build_character_summary_embed,
)
from ..views.campaign_lobby import get_active_campaign_id

logger = structlog.get_logger()

# Store active creation wizards ((guild_id, user_id) -> state)
_creation_states: dict[tuple[int, int], CharacterCreationState] = {}


class CharacterCog(commands.Cog):
    """Character management commands."""

    character = discord.SlashCommandGroup("character", "Character management commands")

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.srd = get_srd()
        self.creator = get_creator()

    async def _get_campaign_id(self, guild_id: int, campaign_name: Optional[str] = None) -> Optional[str]:
        """Get the campaign ID for a guild, using active campaign or by name."""
        from ...data.repositories.campaign_repo import get_campaign_repo
        campaign_repo = await get_campaign_repo()

        if campaign_name:
            campaign = await campaign_repo.get_by_name_and_guild(campaign_name, guild_id)
            return campaign.id if campaign else None

        # Try active campaign first
        active_id = get_active_campaign_id(guild_id)
        if active_id:
            campaign = await campaign_repo.get_by_id(active_id)
            if campaign:
                return campaign.id

        # Fall back to first campaign
        campaigns = await campaign_repo.get_all_by_guild(guild_id)
        return campaigns[0].id if campaigns else None

    @character.command(name="create", description="Create a new character")
    async def create_character(
        self,
        ctx: discord.ApplicationContext,
        campaign: discord.Option(
            str,
            "Campaign name (leave empty to use the active campaign)",
            required=False,
            default=None,
        ),
    ):
        """Start the character creation wizard."""
        user_id = ctx.author.id
        guild_id = ctx.guild_id

        # Find the campaign
        from ...data.repositories.campaign_repo import get_campaign_repo
        campaign_repo = await get_campaign_repo()

        campaign_obj = None

        if campaign:
            # Look up by name if specified
            campaign_obj = await campaign_repo.get_by_name_and_guild(campaign, guild_id)
            if not campaign_obj:
                await ctx.respond(
                    f"No campaign named '{campaign}' found in this server.\n"
                    f"Use `/campaign create` to start a new campaign, or `/campaign list` to see existing ones.",
                    ephemeral=True,
                )
                return
        else:
            # Try to get the active campaign for this guild
            active_campaign_id = get_active_campaign_id(guild_id)
            if active_campaign_id:
                campaign_obj = await campaign_repo.get_by_id(active_campaign_id)

            # Fall back to getting any campaign for this guild
            if not campaign_obj:
                campaigns = await campaign_repo.get_all_by_guild(guild_id)
                if not campaigns:
                    await ctx.respond(
                        "No campaigns exist in this server yet!\n"
                        "Ask your DM to create one with `/campaign create <name>`, then you can create your character.",
                        ephemeral=True,
                    )
                    return
                campaign_obj = campaigns[0]

        campaign_id = campaign_obj.id

        # Check if user already has a character in this campaign
        repo = await get_character_repo()
        existing = await repo.get_by_user_and_campaign(user_id, campaign_id)
        if existing:
            await ctx.respond(
                f"You already have a character ({existing.name}) in **{campaign_obj.name}**. "
                f"Use `/character delete` first if you want to create a new one.",
                ephemeral=True,
            )
            return

        # Initialize creation state (keyed by guild+user for isolation)
        guild_id = ctx.guild_id or 0
        state = CharacterCreationState(user_id=user_id, campaign_id=campaign_id)
        _creation_states[(guild_id, user_id)] = state

        # Start with name entry modal directly - pass interaction to callback
        async def on_name_complete(interaction: discord.Interaction):
            await self._show_ability_method_from_modal(interaction)

        modal = NameModal(state, on_name_complete)
        await ctx.send_modal(modal)

    async def _show_ability_method_from_modal(self, interaction: discord.Interaction):
        """Show ability score method selection after modal submission."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_method_complete(btn_interaction: discord.Interaction):
            await self._show_ability_assignment_from_interaction(btn_interaction)

        view = AbilityScoreMethodView(state, on_method_complete)

        embed = discord.Embed(
            title="Choose Ability Score Method",
            description=f"Creating **{state.name}**\n\nHow would you like to determine ability scores?",
            color=discord.Color.blue(),
        )
        embed.add_field(
            name="Roll 4d6 Drop Lowest",
            value="Roll 4d6, drop the lowest die. Do this 6 times and assign to abilities.",
            inline=False,
        )
        embed.add_field(
            name="Standard Array",
            value="Use the fixed array: 15, 14, 13, 12, 10, 8",
            inline=False,
        )
        embed.add_field(
            name="Point Buy",
            value="Spend 27 points to customize your scores (8-15 range)",
            inline=False,
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_ability_assignment_from_interaction(self, interaction: discord.Interaction):
        """Show ability score assignment after method selection."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        if state.ability_method == AbilityScoreMethod.POINT_BUY:
            await self._show_point_buy_from_interaction(interaction)
        else:
            # Roll or Standard Array - show assignment view
            if state.ability_method == AbilityScoreMethod.ROLL:
                scores = state.ability_rolls.totals
                # Show roll results first
                embed = build_ability_roll_embed(
                    state.name,
                    state.ability_rolls.rolls,
                    state.ability_rolls.totals,
                )
                await interaction.followup.send(embed=embed, ephemeral=True)
            else:
                scores = STANDARD_ARRAY.copy()

            async def on_assignment_complete(assign_interaction: discord.Interaction):
                # Build AbilityScores from assignments
                state.final_abilities = AbilityScores(
                    strength=state.ability_assignments[AbilityScore.STRENGTH],
                    dexterity=state.ability_assignments[AbilityScore.DEXTERITY],
                    constitution=state.ability_assignments[AbilityScore.CONSTITUTION],
                    intelligence=state.ability_assignments[AbilityScore.INTELLIGENCE],
                    wisdom=state.ability_assignments[AbilityScore.WISDOM],
                    charisma=state.ability_assignments[AbilityScore.CHARISMA],
                )
                await self._show_race_select_from_interaction(assign_interaction)

            view = AbilityAssignmentView(state, scores, on_assignment_complete)
            await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    async def _show_point_buy_from_interaction(self, interaction: discord.Interaction):
        """Show point buy interface."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_point_buy_complete(pb_interaction: discord.Interaction):
            state.final_abilities = state.point_buy_state.to_ability_scores()
            await self._show_race_select_from_interaction(pb_interaction)

        view = PointBuyView(state, on_point_buy_complete)

        await interaction.followup.send(
            embed=view.get_embed(),
            view=view,
            ephemeral=True,
        )

    async def _show_race_select_from_interaction(self, interaction: discord.Interaction):
        """Show race selection."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_race_complete(race_interaction: discord.Interaction):
            await self._show_class_select_from_interaction(race_interaction)

        view = RaceSelectView(state, on_race_complete)

        embed = discord.Embed(
            title="Choose Your Race",
            description="Select a race for your character. Racial bonuses will be applied to your ability scores.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_class_select_from_interaction(self, interaction: discord.Interaction):
        """Show class selection."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_class_complete(class_interaction: discord.Interaction):
            await self._show_skill_select_from_interaction(class_interaction)

        view = ClassSelectView(state, on_class_complete)

        # Apply racial bonuses to abilities
        state.final_abilities = self.creator.apply_racial_bonuses(
            state.final_abilities,
            state.race_index,
        )

        race_data = self.srd.get_race(state.race_index)
        race_name = race_data["name"] if race_data else state.race_index.title()

        embed = discord.Embed(
            title="Choose Your Class",
            description=f"**{state.name}** the {race_name}\n\nSelect a class for your character.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_skill_select_from_interaction(self, interaction: discord.Interaction):
        """Show skill selection."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        class_data = self.srd.get_class(state.class_index)
        if not class_data:
            await interaction.followup.send("Error loading class data.", ephemeral=True)
            return

        # Get number of skills and available choices
        proficiency_choices = class_data.get("proficiency_choices", [])
        skill_choice = None
        for choice in proficiency_choices:
            from_data = choice.get("from", {})
            if from_data.get("option_set_type") == "options_array":
                # Check if these are skills
                options = from_data.get("options", [])
                if options and "skill" in str(options[0].get("item", {}).get("index", "")):
                    skill_choice = choice
                    break

        if not skill_choice:
            # No skill choices, skip to confirmation
            await self._show_confirmation_from_interaction(interaction)
            return

        num_skills = skill_choice.get("choose", 2)
        available_skills = []
        for opt in skill_choice.get("from", {}).get("options", []):
            item = opt.get("item", {})
            skill_index = item.get("index", "").replace("skill-", "")
            if skill_index:
                available_skills.append(skill_index)

        async def on_skills_complete(skill_interaction: discord.Interaction):
            await self._show_confirmation_from_interaction(skill_interaction)

        view = SkillSelectView(state, available_skills, num_skills, on_skills_complete)

        class_name = class_data.get("name", state.class_index.title())
        embed = discord.Embed(
            title="Choose Your Skills",
            description=f"As a {class_name}, choose {num_skills} skills to be proficient in.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_confirmation_from_interaction(self, interaction: discord.Interaction):
        """Show character confirmation."""
        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_confirm(confirm_interaction: discord.Interaction):
            await self._finalize_character_from_interaction(confirm_interaction)

        async def on_cancel(cancel_interaction: discord.Interaction):
            _creation_states.pop((getattr(interaction, 'guild_id', 0) or 0, user_id), None)
            await cancel_interaction.followup.send(
                "Character creation cancelled.",
                ephemeral=True,
            )

        view = ConfirmCharacterView(on_confirm, on_cancel)

        # Build preview embed from state
        race_data = self.srd.get_race(state.race_index)
        class_data = self.srd.get_class(state.class_index)
        race_name = race_data["name"] if race_data else state.race_index.title()
        class_name = class_data["name"] if class_data else state.class_index.title()

        embed = discord.Embed(
            title="Confirm Your Character",
            description=f"**{state.name}**\n{race_name} {class_name}",
            color=discord.Color.green(),
        )

        # Ability scores with racial bonuses applied
        abilities = state.final_abilities
        embed.add_field(
            name="Ability Scores",
            value=(
                f"STR: **{abilities.strength}** ({(abilities.strength - 10) // 2:+d})\n"
                f"DEX: **{abilities.dexterity}** ({(abilities.dexterity - 10) // 2:+d})\n"
                f"CON: **{abilities.constitution}** ({(abilities.constitution - 10) // 2:+d})\n"
                f"INT: **{abilities.intelligence}** ({(abilities.intelligence - 10) // 2:+d})\n"
                f"WIS: **{abilities.wisdom}** ({(abilities.wisdom - 10) // 2:+d})\n"
                f"CHA: **{abilities.charisma}** ({(abilities.charisma - 10) // 2:+d})"
            ),
            inline=True,
        )

        # Skills
        if state.skill_choices:
            skills_str = ", ".join(s.value.replace("-", " ").title() for s in state.skill_choices)
            embed.add_field(name="Skill Proficiencies", value=skills_str, inline=True)

        embed.set_footer(text="Click 'Create Character' to finalize or 'Cancel' to start over.")

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _finalize_character_from_interaction(self, interaction: discord.Interaction):
        """Create and save the character."""
        from ...game.character.starting_equipment import assign_starting_equipment

        user_id = interaction.user.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        try:
            # Create the character
            character = self.creator.build_character(state)

            # Save to database
            repo = await get_character_repo()
            await repo.create(character)

            # Assign starting equipment and gold
            equipment_result = await assign_starting_equipment(
                character.id,
                character.class_index,
            )

            # Clean up state
            _creation_states.pop((getattr(interaction, 'guild_id', 0) or 0, user_id), None)

            embed = build_character_sheet_embed(character)
            embed.title = f":tada: {character.name} Created!"
            embed.color = discord.Color.green()

            # Add starting equipment info
            if equipment_result["items"]:
                item_lines = [f"• {item['name']}" + (f" x{item['quantity']}" if item['quantity'] > 1 else "")
                              for item in equipment_result["items"][:8]]
                if len(equipment_result["items"]) > 8:
                    item_lines.append(f"_...and {len(equipment_result['items']) - 8} more items_")
                embed.add_field(
                    name=":school_satchel: Starting Equipment",
                    value="\n".join(item_lines),
                    inline=False,
                )

            if equipment_result["gold"]:
                embed.add_field(
                    name=":coin: Starting Gold",
                    value=f"{equipment_result['gold']} gp",
                    inline=True,
                )

            await interaction.followup.send(embed=embed)

            logger.info(
                "character_created",
                character_id=character.id,
                user_id=user_id,
                name=character.name,
                race=character.race,
                class_name=character.class_name,
            )

        except Exception as e:
            logger.error("character_creation_failed", error=str(e))
            await interaction.followup.send(
                f"Failed to create character: {e}",
                ephemeral=True,
            )

    async def _show_ability_assignment(self, ctx: discord.ApplicationContext):
        """Show ability score assignment or point buy."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        if state.ability_method == AbilityScoreMethod.POINT_BUY:
            await self._show_point_buy(ctx)
        else:
            # Roll or Standard Array - show assignment view
            if state.ability_method == AbilityScoreMethod.ROLL:
                scores = state.ability_rolls.totals
                # Show roll results
                embed = build_ability_roll_embed(
                    state.name,
                    state.ability_rolls.rolls,
                    state.ability_rolls.totals,
                )
                await ctx.interaction.followup.send(embed=embed, ephemeral=True)
            else:
                scores = STANDARD_ARRAY.copy()

            async def on_assignment_complete():
                # Build AbilityScores from assignments
                state.final_abilities = AbilityScores(
                    strength=state.ability_assignments[AbilityScore.STRENGTH],
                    dexterity=state.ability_assignments[AbilityScore.DEXTERITY],
                    constitution=state.ability_assignments[AbilityScore.CONSTITUTION],
                    intelligence=state.ability_assignments[AbilityScore.INTELLIGENCE],
                    wisdom=state.ability_assignments[AbilityScore.WISDOM],
                    charisma=state.ability_assignments[AbilityScore.CHARISMA],
                )
                await self._show_race_select(ctx)

            view = AbilityAssignmentView(state, scores, on_assignment_complete)

            embed = discord.Embed(
                title="Assign Ability Scores",
                description=f"Assign each score to an ability.\nAvailable: {', '.join(str(s) for s in sorted(scores, reverse=True))}",
                color=discord.Color.blue(),
            )

            await ctx.interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_point_buy(self, ctx: discord.ApplicationContext):
        """Show point buy interface."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_point_buy_complete():
            state.final_abilities = state.point_buy_state.to_ability_scores()
            await self._show_race_select(ctx)

        view = PointBuyView(state, on_point_buy_complete)

        await ctx.interaction.followup.send(
            embed=view.get_embed(),
            view=view,
            ephemeral=True,
        )

    async def _show_race_select(self, ctx: discord.ApplicationContext):
        """Show race selection."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_race_complete():
            await self._show_class_select(ctx)

        view = RaceSelectView(state, on_race_complete)

        embed = discord.Embed(
            title="Choose Your Race",
            description="Select a race for your character. Racial bonuses will be applied to your ability scores.",
            color=discord.Color.blue(),
        )

        await ctx.interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_class_select(self, ctx: discord.ApplicationContext):
        """Show class selection."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_class_complete():
            await self._show_skill_select(ctx)

        view = ClassSelectView(state, on_class_complete)

        # Apply racial bonuses to abilities
        state.final_abilities = self.creator.apply_racial_bonuses(
            state.final_abilities,
            state.race_index,
        )

        race_data = self.srd.get_race(state.race_index)
        race_name = race_data["name"] if race_data else state.race_index.title()

        embed = discord.Embed(
            title="Choose Your Class",
            description=f"**{state.name}** the {race_name}\n\nSelect a class for your character.",
            color=discord.Color.blue(),
        )

        # Show current ability scores with racial bonuses
        abilities = state.final_abilities
        embed.add_field(
            name="Ability Scores (with racial bonuses)",
            value=(
                f"STR: {abilities.strength} ({'+' if abilities.str_mod >= 0 else ''}{abilities.str_mod})\n"
                f"DEX: {abilities.dexterity} ({'+' if abilities.dex_mod >= 0 else ''}{abilities.dex_mod})\n"
                f"CON: {abilities.constitution} ({'+' if abilities.con_mod >= 0 else ''}{abilities.con_mod})\n"
                f"INT: {abilities.intelligence} ({'+' if abilities.int_mod >= 0 else ''}{abilities.int_mod})\n"
                f"WIS: {abilities.wisdom} ({'+' if abilities.wis_mod >= 0 else ''}{abilities.wis_mod})\n"
                f"CHA: {abilities.charisma} ({'+' if abilities.cha_mod >= 0 else ''}{abilities.cha_mod})"
            ),
            inline=False,
        )

        await ctx.interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_skill_select(self, ctx: discord.ApplicationContext):
        """Show skill proficiency selection."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        available_skills, choose_count = self.creator.get_skill_choices(state.class_index)

        if not available_skills or choose_count == 0:
            # Class has no skill choices (shouldn't happen, but handle it)
            await self._show_confirmation(ctx)
            return

        async def on_skills_complete():
            await self._show_confirmation(ctx)

        view = SkillSelectView(state, available_skills, choose_count, on_skills_complete)

        class_data = self.srd.get_class(state.class_index)
        class_name = class_data["name"] if class_data else state.class_index.title()

        embed = discord.Embed(
            title="Choose Skill Proficiencies",
            description=f"As a {class_name}, choose **{choose_count}** skills to be proficient in.",
            color=discord.Color.blue(),
        )

        await ctx.interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_confirmation(self, ctx: discord.ApplicationContext):
        """Show character confirmation."""
        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        async def on_confirm():
            await self._finalize_character(ctx)

        async def on_restart():
            state.reset()
            await ctx.interaction.followup.send(
                "Character creation reset. Use `/character create` to start over.",
                ephemeral=True,
            )
            _creation_states.pop((getattr(interaction, 'guild_id', 0) or 0, user_id), None)

        view = ConfirmCharacterView(state, on_confirm, on_restart)

        # Build preview
        race_data = self.srd.get_race(state.race_index)
        class_data = self.srd.get_class(state.class_index)
        race_name = race_data["name"] if race_data else state.race_index.title()
        class_name = class_data["name"] if class_data else state.class_index.title()

        abilities = state.final_abilities
        starting_hp = self.creator.calculate_starting_hp(state.class_index, abilities.con_mod)
        ac = self.creator.calculate_armor_class(abilities, state.class_index)

        embed = discord.Embed(
            title=f"Confirm: {state.name}",
            description=f"Level 1 {race_name} {class_name}",
            color=discord.Color.green(),
        )

        embed.add_field(
            name="Ability Scores",
            value=(
                f"STR: {abilities.strength} ({'+' if abilities.str_mod >= 0 else ''}{abilities.str_mod})\n"
                f"DEX: {abilities.dexterity} ({'+' if abilities.dex_mod >= 0 else ''}{abilities.dex_mod})\n"
                f"CON: {abilities.constitution} ({'+' if abilities.con_mod >= 0 else ''}{abilities.con_mod})\n"
                f"INT: {abilities.intelligence} ({'+' if abilities.int_mod >= 0 else ''}{abilities.int_mod})\n"
                f"WIS: {abilities.wisdom} ({'+' if abilities.wis_mod >= 0 else ''}{abilities.wis_mod})\n"
                f"CHA: {abilities.charisma} ({'+' if abilities.cha_mod >= 0 else ''}{abilities.cha_mod})"
            ),
            inline=True,
        )

        embed.add_field(
            name="Combat Stats",
            value=(
                f"HP: {starting_hp}\n"
                f"AC: {ac}\n"
                f"Speed: {self.creator.get_speed(state.race_index)} ft"
            ),
            inline=True,
        )

        if state.skill_choices:
            skills_text = ", ".join(
                s.value.replace("-", " ").title() for s in state.skill_choices
            )
            embed.add_field(
                name="Skill Proficiencies",
                value=skills_text,
                inline=False,
            )

        await ctx.interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _finalize_character(self, ctx: discord.ApplicationContext):
        """Create the character in the database."""
        from ...game.character.starting_equipment import assign_starting_equipment

        user_id = ctx.author.id
        state = _creation_states.get((getattr(interaction, 'guild_id', 0) or 0, user_id))
        if not state:
            return

        try:
            # Build and save character
            character = self.creator.build_character(state)
            repo = await get_character_repo()
            await repo.create(character)

            # Assign starting equipment and gold
            equipment_result = await assign_starting_equipment(
                character.id,
                character.class_index,
            )

            # Show final character sheet
            embed = build_character_sheet_embed(character)
            embed.title = f"Welcome, {character.name}!"
            embed.description = f"Your character has been created successfully!\n\n{embed.description}"

            # Add starting equipment info
            if equipment_result["items"]:
                item_lines = [f"• {item['name']}" + (f" x{item['quantity']}" if item['quantity'] > 1 else "")
                              for item in equipment_result["items"][:8]]
                if len(equipment_result["items"]) > 8:
                    item_lines.append(f"_...and {len(equipment_result['items']) - 8} more items_")
                embed.add_field(
                    name=":school_satchel: Starting Equipment",
                    value="\n".join(item_lines),
                    inline=False,
                )

            if equipment_result["gold"]:
                embed.add_field(
                    name=":coin: Starting Gold",
                    value=f"{equipment_result['gold']} gp",
                    inline=True,
                )

            await ctx.interaction.followup.send(embed=embed)

            logger.info(
                "character_created",
                character_id=character.id,
                user_id=user_id,
                name=character.name,
                race=character.race_index,
                class_=character.class_index,
            )

        except Exception as e:
            logger.error("character_creation_failed", error=str(e))
            await ctx.interaction.followup.send(
                f"Failed to create character: {str(e)}",
                ephemeral=True,
            )

        finally:
            _creation_states.pop((getattr(interaction, 'guild_id', 0) or 0, user_id), None)

    @character.command(name="sheet", description="View your character sheet")
    async def view_sheet(
        self,
        ctx: discord.ApplicationContext,
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """View your character sheet."""
        await ctx.defer()
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond(
                "No campaign found. Create one with `/campaign create`.",
                ephemeral=True,
            )
            return

        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign. Use `/character create` to make one.",
                ephemeral=True,
            )
            return

        embed = build_character_sheet_embed(character)
        await ctx.respond(embed=embed)

    @character.command(name="delete", description="Delete your character")
    async def delete_character(
        self,
        ctx: discord.ApplicationContext,
        confirm: discord.Option(
            bool,
            "Are you sure? This cannot be undone.",
            required=True,
        ),
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Delete your character."""
        if not confirm:
            await ctx.respond("Character deletion cancelled.", ephemeral=True)
            return

        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond(
                "No campaign found.",
                ephemeral=True,
            )
            return

        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign.",
                ephemeral=True,
            )
            return

        await repo.delete(character.id)

        await ctx.respond(
            f"**{character.name}** has been deleted. You can create a new character with `/character create`.",
            ephemeral=True,
        )

        logger.info(
            "character_deleted",
            character_id=character.id,
            user_id=ctx.author.id,
            name=character.name,
        )

    # ==================== XP & Leveling Commands ====================

    @character.command(name="xp", description="View your XP and level progress")
    async def view_xp(
        self,
        ctx: discord.ApplicationContext,
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """View XP status and level progress."""
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond("No campaign found.", ephemeral=True)
            return

        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign.",
                ephemeral=True,
            )
            return

        leveling = get_leveling_manager()
        status = leveling.get_xp_status(character)

        embed = discord.Embed(
            title=f"{character.name} - Experience",
            color=discord.Color.gold(),
        )

        embed.add_field(
            name="Current Level",
            value=str(character.level),
            inline=True,
        )

        embed.add_field(
            name="Total XP",
            value=f"{character.experience:,}",
            inline=True,
        )

        if status["is_max_level"]:
            embed.add_field(
                name="Status",
                value="Maximum level reached!",
                inline=True,
            )
        else:
            progress, needed = status["progress_to_next"], status["xp_needed_for_next"]
            progress_bar = self._make_progress_bar(progress, needed)

            embed.add_field(
                name=f"Progress to Level {character.level + 1}",
                value=f"{progress:,} / {needed:,} XP\n{progress_bar}",
                inline=False,
            )

            if status["can_level_up"]:
                embed.add_field(
                    name="Ready to Level Up!",
                    value="Use `/character levelup` to advance!",
                    inline=False,
                )
                embed.color = discord.Color.green()

        await ctx.respond(embed=embed)

    def _make_progress_bar(self, current: int, total: int, length: int = 20) -> str:
        """Create a text-based progress bar."""
        if total <= 0:
            return "[" + "=" * length + "]"
        filled = int((current / total) * length)
        empty = length - filled
        return "[" + "=" * filled + "-" * empty + "]"

    @character.command(name="xp_add", description="Add XP to a character (DM only)")
    async def add_xp(
        self,
        ctx: discord.ApplicationContext,
        amount: discord.Option(
            int,
            "Amount of XP to add",
            required=True,
            min_value=1,
        ),
        player: discord.Option(
            discord.Member,
            "Player to award XP to (leave empty for yourself)",
            required=False,
        ),
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Add XP to a character (for DMs or testing)."""
        await ctx.defer()
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond("No campaign found.", ephemeral=True)
            return

        target_user = player or ctx.author
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(target_user.id, campaign_id)

        if not character:
            await ctx.respond(
                f"{target_user.display_name} doesn't have a character in this campaign.",
                ephemeral=True,
            )
            return

        old_xp = character.experience
        leveling = get_leveling_manager()
        new_xp, ready_to_level = leveling.add_xp(character, amount)

        # Save to database
        await repo.update(character)

        embed = discord.Embed(
            title=f"XP Awarded to {character.name}",
            description=f"+{amount:,} XP",
            color=discord.Color.gold(),
        )

        embed.add_field(
            name="XP",
            value=f"{old_xp:,} → {new_xp:,}",
            inline=True,
        )

        embed.add_field(
            name="Level",
            value=str(character.level),
            inline=True,
        )

        if ready_to_level:
            embed.add_field(
                name="Level Up Available!",
                value=f"{character.name} can now level up to {character.level + 1}!",
                inline=False,
            )
            embed.color = discord.Color.green()

        await ctx.respond(embed=embed)

        logger.info(
            "xp_added",
            character_id=character.id,
            character_name=character.name,
            amount=amount,
            new_total=new_xp,
            can_level=ready_to_level,
        )

    @character.command(name="xp_set", description="Set a character's XP (DM only)")
    async def set_xp(
        self,
        ctx: discord.ApplicationContext,
        amount: discord.Option(
            int,
            "New XP value",
            required=True,
            min_value=0,
        ),
        player: discord.Option(
            discord.Member,
            "Player to modify (leave empty for yourself)",
            required=False,
        ),
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Set a character's XP directly (for DMs or testing)."""
        await ctx.defer()
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond("No campaign found.", ephemeral=True)
            return

        target_user = player or ctx.author
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(target_user.id, campaign_id)

        if not character:
            await ctx.respond(
                f"{target_user.display_name} doesn't have a character in this campaign.",
                ephemeral=True,
            )
            return

        old_xp = character.experience
        leveling = get_leveling_manager()
        new_xp, ready_to_level = leveling.set_xp(character, amount)

        # Save to database
        await repo.update(character)

        embed = discord.Embed(
            title=f"XP Set for {character.name}",
            color=discord.Color.blue(),
        )

        embed.add_field(
            name="XP",
            value=f"{old_xp:,} → {new_xp:,}",
            inline=True,
        )

        if ready_to_level:
            embed.add_field(
                name="Level Up Available!",
                value=f"Use `/character levelup` to advance.",
                inline=False,
            )

        await ctx.respond(embed=embed)

        logger.info(
            "xp_set",
            character_id=character.id,
            old_xp=old_xp,
            new_xp=new_xp,
        )

    @character.command(name="levelup", description="Level up your character")
    async def level_up_character(
        self,
        ctx: discord.ApplicationContext,
        take_average: discord.Option(
            bool,
            "Take average HP instead of rolling",
            required=False,
            default=True,
        ),
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Level up your character if you have enough XP."""
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond("No campaign found.", ephemeral=True)
            return

        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign.",
                ephemeral=True,
            )
            return

        if not can_level_up(character):
            if character.level >= 20:
                await ctx.respond(
                    f"{character.name} is already at maximum level (20).",
                    ephemeral=True,
                )
            else:
                next_xp = get_xp_for_next_level(character.level)
                await ctx.respond(
                    f"{character.name} needs {next_xp:,} XP to reach level {character.level + 1}. "
                    f"Current XP: {character.experience:,}",
                    ephemeral=True,
                )
            return

        # Perform level up
        leveling = get_leveling_manager()
        result = leveling.level_up_character(character, take_average)

        # Save to database
        await repo.update(character)

        # Build result embed
        embed = discord.Embed(
            title=f"Level Up! {character.name} is now Level {result.new_level}!",
            color=discord.Color.green(),
        )

        # HP gain
        hp_method = "average" if take_average else f"rolled {result.hp_rolled}"
        embed.add_field(
            name="Hit Points",
            value=f"+{result.hp_gained} HP ({hp_method})\nNew Max HP: {character.hp.maximum}",
            inline=True,
        )

        # Proficiency
        if result.new_proficiency_bonus != result.old_proficiency_bonus:
            embed.add_field(
                name="Proficiency Bonus",
                value=f"+{result.old_proficiency_bonus} → +{result.new_proficiency_bonus}",
                inline=True,
            )

        # Spell slots
        if result.new_spell_slots:
            slots_text = "\n".join(
                f"Level {lvl}: {count} slots"
                for lvl, count in sorted(result.new_spell_slots.items())
            )
            embed.add_field(
                name="Spell Slots",
                value=slots_text,
                inline=True,
            )

        # ASI
        if result.has_asi:
            embed.add_field(
                name="Ability Score Improvement!",
                value="You can increase your ability scores!\nUse `/character asi` to apply your improvement.",
                inline=False,
            )

        # Features
        if result.features_gained:
            embed.add_field(
                name="New Features",
                value="\n".join(f"• {f}" for f in result.features_gained),
                inline=False,
            )

        await ctx.respond(embed=embed)

        logger.info(
            "character_leveled_up",
            character_id=character.id,
            character_name=character.name,
            old_level=result.old_level,
            new_level=result.new_level,
            hp_gained=result.hp_gained,
        )

    @character.command(name="asi", description="Apply Ability Score Improvement")
    async def apply_asi(
        self,
        ctx: discord.ApplicationContext,
        ability1: discord.Option(
            str,
            "First ability to increase",
            required=True,
            choices=[
                discord.OptionChoice("Strength", "strength"),
                discord.OptionChoice("Dexterity", "dexterity"),
                discord.OptionChoice("Constitution", "constitution"),
                discord.OptionChoice("Intelligence", "intelligence"),
                discord.OptionChoice("Wisdom", "wisdom"),
                discord.OptionChoice("Charisma", "charisma"),
            ],
        ),
        increase1: discord.Option(
            int,
            "Amount to increase first ability (1 or 2)",
            required=True,
            choices=[
                discord.OptionChoice("+1", 1),
                discord.OptionChoice("+2", 2),
            ],
        ),
        ability2: discord.Option(
            str,
            "Second ability to increase (if splitting +1/+1)",
            required=False,
            choices=[
                discord.OptionChoice("Strength", "strength"),
                discord.OptionChoice("Dexterity", "dexterity"),
                discord.OptionChoice("Constitution", "constitution"),
                discord.OptionChoice("Intelligence", "intelligence"),
                discord.OptionChoice("Wisdom", "wisdom"),
                discord.OptionChoice("Charisma", "charisma"),
            ],
        ),
        campaign: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Apply an Ability Score Improvement (ASI) to your character."""
        await ctx.defer()
        campaign_id = await self._get_campaign_id(ctx.guild_id, campaign)
        if not campaign_id:
            await ctx.respond("No campaign found.", ephemeral=True)
            return

        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign.",
                ephemeral=True,
            )
            return

        # Validate ASI points (must total exactly 2)
        if increase1 == 2 and ability2 is not None:
            await ctx.respond(
                "If you increase one ability by 2, you cannot choose a second ability.",
                ephemeral=True,
            )
            return

        if increase1 == 1 and ability2 is None:
            await ctx.respond(
                "If you increase one ability by 1, you must choose a second ability to also increase by 1.",
                ephemeral=True,
            )
            return

        # Convert string to AbilityScore enum
        ability1_enum = AbilityScore[ability1.upper()]
        ability2_enum = AbilityScore[ability2.upper()] if ability2 else None
        increase2 = 1 if ability2 else 0

        # Check for score cap
        current1 = character.abilities.get_score(ability1_enum)
        if current1 + increase1 > 20:
            await ctx.respond(
                f"Cannot increase {ability1.title()} beyond 20. Current: {current1}",
                ephemeral=True,
            )
            return

        if ability2_enum:
            current2 = character.abilities.get_score(ability2_enum)
            if current2 + increase2 > 20:
                await ctx.respond(
                    f"Cannot increase {ability2.title()} beyond 20. Current: {current2}",
                    ephemeral=True,
                )
                return

        # Apply ASI
        leveling = get_leveling_manager()
        changes = leveling.apply_asi_to_character(
            character,
            ability1_enum,
            ability2_enum,
            increase1,
            increase2,
        )

        # Save to database
        await repo.update(character)

        # Build result
        embed = discord.Embed(
            title=f"ASI Applied to {character.name}",
            color=discord.Color.green(),
        )

        changes_text = "\n".join(
            f"{name}: {score}" for name, score in changes.items()
        )
        embed.add_field(
            name="New Ability Scores",
            value=changes_text,
            inline=False,
        )

        await ctx.respond(embed=embed)

        logger.info(
            "asi_applied",
            character_id=character.id,
            changes=changes,
        )


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(CharacterCog(bot))
