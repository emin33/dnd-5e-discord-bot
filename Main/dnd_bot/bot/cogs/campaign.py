"""Campaign management commands."""

import asyncio
import discord
from discord.ext import commands
import structlog

from ...models import Campaign, Character, AbilityScore, AbilityScores
from ...data.repositories import get_campaign_repo, get_character_repo
from ...data.srd import get_srd
from ...game.character.creation import (
    CharacterCreationState,
    get_creator,
    STANDARD_ARRAY,
    AbilityScoreMethod,
)
from ..views.base import SafeView
from ..views.campaign_lobby import (
    CampaignLobbyView,
    set_active_campaign,
)
from ..views.character_creation import (
    NameModal,
    AbilityScoreMethodView,
    AbilityAssignmentView,
    PointBuyView,
    RaceSelectView,
    ClassSelectView,
    SkillSelectView,
    ConfirmCharacterView,
)
from ..views.character_select import CharacterSelectView
from ..embeds.character_sheet import (
    build_ability_roll_embed,
    build_character_sheet_embed,
)

logger = structlog.get_logger()

# Track players who have joined each campaign (in-memory)
_campaign_players: dict[str, list[int]] = {}  # campaign_id -> [user_ids]
_campaign_players_lock = asyncio.Lock()

# campaign_id -> sourcebook_key chosen at creation, installed at session start.
# In memory beside the player list rather than on the campaign row, because
# the DURABLE record of which book a campaign plays is `campaign_sourcebook`,
# and that row is written by the install itself. This only has to survive the
# lobby. A restart between creating a campaign and starting it loses the
# choice, which is the right failure: nothing was installed, so nothing is
# half-done — the DM picks again.
_campaign_sourcebook: dict[str, str] = {}


async def _bound_sourcebook(campaign_id: str) -> str:
    """The book this campaign is bound to, surviving a restart.

    The in-memory choice covers the lobby; this covers everything after it.
    A campaign bound to more than one book (a base module plus a supplement)
    installs the first — install is per-key and idempotent, so the rest can
    follow without conflict.
    """
    try:
        from ...data.repositories.sourcebook_repo import get_sourcebook_repo

        keys = await (await get_sourcebook_repo()).sourcebook_keys_for_campaign(
            campaign_id
        )
        return keys[0] if keys else ""
    except Exception as e:
        logger.warning(
            "sourcebook_binding_lookup_failed",
            campaign_id=campaign_id, error=str(e),
        )
        return ""


async def _library() -> tuple[list, list[tuple[str, str]]]:
    """Every book on the shelf, plus the files that would not read.

    Scanning here rather than at boot means a DM can drop a book in the
    folder and see it without restarting the bot. Import is
    content-addressed, so re-scanning an unchanged shelf writes nothing.

    The rejects come back with the books because they are the answer to "why
    is my book not in the menu" — discarding them is how a one-line YAML
    error becomes an unexplained empty list.
    """
    from ...data.repositories.sourcebook_repo import get_sourcebook_repo
    from ...game.knowledge.sourcebook_library import available_books, scan_library

    repo = await get_sourcebook_repo()
    rejected: list[tuple[str, str]] = []
    try:
        rejected = (await scan_library(repo)).rejected
    except Exception as e:
        # A broken shelf must not take the command down; the DM can still
        # create an improvised campaign.
        logger.warning("sourcebook_scan_failed", error=str(e), exc_info=True)
    return await available_books(repo), rejected


async def _sourcebook_choices(ctx: discord.AutocompleteContext) -> list[str]:
    """Titles for the campaign-creation autocomplete."""
    try:
        typed = (ctx.value or "").casefold()
        books, _rejected = await _library()
        return [
            header.title for header in books
            if typed in header.title.casefold()
        ][:25]
    except Exception as e:
        logger.warning("sourcebook_autocomplete_failed", error=str(e))
        return []


async def _resolve_sourcebook(choice: str) -> tuple[str, list[tuple[str, str]]]:
    """A typed title (or key) -> ``(sourcebook_key, rejected_files)``.

    The key is "" when nothing matches, and the rejects travel with it so the
    caller can say WHY the library looks emptier than the folder does.
    Accepts a key directly, so the value survives if Discord ever hands back
    something other than the display title.
    """
    books, rejected = await _library()
    wanted = (choice or "").strip().casefold()
    if not wanted:
        return "", rejected
    for header in books:
        if wanted in (header.title.casefold(), header.sourcebook_key.casefold()):
            return header.sourcebook_key, rejected
    return "", rejected

# Serializes "Start Game" attempts so two near-simultaneous clicks can't both
# pass the `has_active_session` check before either creates the session.
_session_start_lock = asyncio.Lock()

# Track active character creation wizards ((guild_id, user_id) -> state)
_creation_states: dict[tuple[int, int], CharacterCreationState] = {}
_creation_states_lock = asyncio.Lock()


async def _add_campaign_player(campaign_id: str, user_id: int) -> bool:
    """Thread-safe add player to campaign. Returns False if already joined."""
    async with _campaign_players_lock:
        if campaign_id not in _campaign_players:
            _campaign_players[campaign_id] = []
        if user_id in _campaign_players[campaign_id]:
            return False
        _campaign_players[campaign_id].append(user_id)
        return True


class CampaignCog(commands.Cog):
    """Campaign management commands."""

    campaign = discord.SlashCommandGroup(
        "campaign",
        "Manage D&D campaigns",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.srd = get_srd()
        self.creator = get_creator()

    async def _handle_join(self, interaction: discord.Interaction, campaign: Campaign):
        """Handle a player joining the campaign.

        Caller (the lobby button) has already deferred the interaction, so we
        use `interaction.followup.send` rather than `response.send_message`.
        """
        user_id = interaction.user.id

        # Check if already joined with a character in this campaign
        char_repo = await get_character_repo()
        character = await char_repo.get_by_user_and_campaign(user_id, campaign.id)

        if character:
            await interaction.followup.send(
                f"You've already joined as **{character.name}**!",
                ephemeral=True,
            )
            return

        # Initialize campaign players list if needed
        if campaign.id not in _campaign_players:
            _campaign_players[campaign.id] = []

        # Show the character selection view
        await self._show_character_select(interaction, campaign)

        logger.info(
            "player_joining_campaign",
            campaign_id=campaign.id,
            user_id=user_id,
        )

    async def _show_character_select(self, interaction: discord.Interaction, campaign: Campaign):
        """Show the character selection view (existing characters or create new)."""
        user_id = interaction.user.id

        # Get all characters this user has in this guild (across all campaigns)
        char_repo = await get_character_repo()
        existing_characters = await char_repo.get_all_by_user_in_guild(
            user_id, interaction.guild_id
        )

        # Filter out any character already in this campaign
        existing_characters = [c for c in existing_characters if c.campaign_id != campaign.id]

        async def on_select_existing(select_interaction: discord.Interaction, character: Character):
            await self._use_existing_character(select_interaction, campaign, character)

        async def on_create_new(create_interaction: discord.Interaction):
            await self._start_character_wizard_from_select(create_interaction, campaign)

        async def on_quick_join(quick_interaction: discord.Interaction):
            await self._quick_join_template(quick_interaction, campaign)

        view = CharacterSelectView(
            existing_characters=existing_characters,
            on_select_existing=on_select_existing,
            on_create_new=on_create_new,
            on_quick_join=on_quick_join,
        )

        embed = view.get_embed(campaign.name)
        # Caller (lobby button) deferred — use followup, not response.
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _use_existing_character(
        self,
        interaction: discord.Interaction,
        campaign: Campaign,
        source_character: Character,
    ):
        """Copy an existing character to use in this campaign."""
        user_id = interaction.user.id
        char_repo = await get_character_repo()

        # Create a copy of the character for this campaign
        from ...models import Character, HitPoints, HitDice, DeathSaves
        import uuid

        new_character = Character(
            id=str(uuid.uuid4()),
            discord_user_id=user_id,
            campaign_id=campaign.id,
            name=source_character.name,
            race_index=source_character.race_index,
            class_index=source_character.class_index,
            subclass_index=source_character.subclass_index,
            level=source_character.level,
            experience=source_character.experience,
            background_index=source_character.background_index,
            abilities=source_character.abilities.model_copy(),
            armor_class=source_character.armor_class,
            speed=source_character.speed,
            initiative_bonus=source_character.initiative_bonus,
            hp=HitPoints(
                current=source_character.hp.maximum,
                maximum=source_character.hp.maximum,
                temporary=0,
            ),
            hit_dice=HitDice(
                die_type=source_character.hit_dice.die_type,
                total=source_character.hit_dice.total,
                remaining=source_character.hit_dice.total,  # Full hit dice
            ),
            death_saves=DeathSaves(),
            spellcasting_ability=source_character.spellcasting_ability,
            spell_slots=source_character.spell_slots.model_copy(),
            known_spells=source_character.known_spells.copy(),
            prepared_spells=source_character.prepared_spells.copy(),
            saving_throw_proficiencies=source_character.saving_throw_proficiencies.copy(),
            skill_proficiencies=source_character.skill_proficiencies.copy(),
            skill_expertise=source_character.skill_expertise.copy(),
            conditions=[],
        )

        await char_repo.create(new_character)

        # Add to players list
        if campaign.id not in _campaign_players:
            _campaign_players[campaign.id] = []
        if user_id not in _campaign_players[campaign.id]:
            _campaign_players[campaign.id].append(user_id)

        embed = build_character_sheet_embed(new_character)
        embed.title = f":tada: {new_character.name} Joins {campaign.name}!"
        embed.color = discord.Color.green()

        await interaction.followup.send(embed=embed)

        logger.info(
            "character_copied_to_campaign",
            source_character_id=source_character.id,
            new_character_id=new_character.id,
            campaign_id=campaign.id,
            user_id=user_id,
            name=new_character.name,
        )

    async def _quick_join_template(
        self,
        interaction: discord.Interaction,
        campaign: Campaign,
    ):
        """Create a template character and join instantly (for testing)."""
        import uuid
        import random
        from ...models import HitPoints, HitDice, DeathSaves, SpellSlots, Skill

        user_id = interaction.user.id
        char_repo = await get_character_repo()

        # Template: Level 1 Elf Ranger with standard array
        # Standard array assigned: DEX 15, WIS 14, CON 13, STR 12, INT 10, CHA 8
        base_abilities = AbilityScores(
            strength=12,
            dexterity=15,
            constitution=13,
            intelligence=10,
            wisdom=14,
            charisma=8,
        )

        # Apply elf racial bonuses (+2 DEX)
        final_abilities = self.creator.apply_racial_bonuses(base_abilities, "elf")

        # Random name from a list
        names = ["Thorn", "Willow", "Ash", "Brook", "Sage", "Fern", "Reed", "Storm", "Flint", "Ivy"]
        name = f"{random.choice(names)} (Test)"

        # Calculate stats
        con_mod = (final_abilities.constitution - 10) // 2
        starting_hp = 10 + con_mod  # Ranger hit die is d10

        character = Character(
            id=str(uuid.uuid4()),
            discord_user_id=user_id,
            campaign_id=campaign.id,
            name=name,
            race_index="elf",
            class_index="ranger",
            subclass_index=None,
            level=1,
            experience=0,
            background_index=None,
            abilities=final_abilities,
            armor_class=10 + final_abilities.dex_mod,  # Unarmored
            speed=30,
            initiative_bonus=final_abilities.dex_mod,
            hp=HitPoints(maximum=starting_hp, current=starting_hp, temporary=0),
            hit_dice=HitDice(die_type=10, total=1, remaining=1),
            death_saves=DeathSaves(),
            spellcasting_ability=AbilityScore.WISDOM,
            spell_slots=SpellSlots(),
            saving_throw_proficiencies=[AbilityScore.STRENGTH, AbilityScore.DEXTERITY],
            skill_proficiencies=[Skill.PERCEPTION, Skill.STEALTH, Skill.SURVIVAL],
        )

        await char_repo.create(character)

        # Assign starting equipment and gold
        from ...game.character.starting_equipment import assign_starting_equipment
        equipment_result = await assign_starting_equipment(
            character.id,
            character.class_index,
        )

        # Add to players list
        if campaign.id not in _campaign_players:
            _campaign_players[campaign.id] = []
        if user_id not in _campaign_players[campaign.id]:
            _campaign_players[campaign.id].append(user_id)

        embed = build_character_sheet_embed(character)
        embed.title = f":zap: {character.name} Quick-Joined {campaign.name}!"
        embed.color = discord.Color.orange()
        embed.set_footer(text="Template character created for testing")

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
            "character_quick_joined",
            character_id=character.id,
            campaign_id=campaign.id,
            user_id=user_id,
            name=character.name,
        )

    async def _start_character_wizard_from_select(
        self, interaction: discord.Interaction, campaign: Campaign
    ):
        """Start the character wizard after the select view (uses followup)."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0

        # Initialize creation state (keyed by guild+user for isolation)
        state = CharacterCreationState(user_id=user_id, campaign_id=campaign.id)
        _creation_states[(guild_id, user_id)] = state

        # Show name modal - but we need to send it differently since we already deferred
        # We'll show an embed with instructions and a button to open the modal
        async def on_name_complete(modal_interaction: discord.Interaction):
            await self._show_ability_method(modal_interaction, campaign)

        # Create a simple view with a button to open the name modal
        class StartCreationView(SafeView):
            def __init__(self, cog, campaign_ref, state_ref):
                super().__init__(timeout=300)
                self.cog = cog
                self.campaign_ref = campaign_ref
                self.state_ref = state_ref

            @discord.ui.button(label="Enter Character Name", style=discord.ButtonStyle.primary, emoji="📝")
            async def name_button(self, button: discord.ui.Button, btn_interaction: discord.Interaction):
                async def on_name_done(modal_interaction: discord.Interaction):
                    await self.cog._show_ability_method(modal_interaction, self.campaign_ref)

                modal = NameModal(self.state_ref, on_name_done)
                await btn_interaction.response.send_modal(modal)
                self.stop()

        view = StartCreationView(self, campaign, state)

        embed = discord.Embed(
            title="Create Your Character",
            description=f"Let's create a new character for **{campaign.name}**!\n\nClick the button below to begin.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _start_character_wizard(self, interaction: discord.Interaction, campaign: Campaign):
        """Start the character creation wizard for a player."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0

        # Initialize creation state (keyed by guild+user for isolation)
        state = CharacterCreationState(user_id=user_id, campaign_id=campaign.id)
        _creation_states[(guild_id, user_id)] = state

        # Start with name entry modal
        async def on_name_complete(modal_interaction: discord.Interaction):
            await self._show_ability_method(modal_interaction, campaign)

        modal = NameModal(state, on_name_complete)
        await interaction.response.send_modal(modal)

    async def _show_ability_method(self, interaction: discord.Interaction, campaign: Campaign):
        """Show ability score method selection."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        async def on_method_complete(btn_interaction: discord.Interaction):
            await self._show_ability_assignment(btn_interaction, campaign)

        view = AbilityScoreMethodView(state, on_method_complete)

        embed = discord.Embed(
            title="Choose Ability Score Method",
            description=f"Creating **{state.name}** for **{campaign.name}**\n\nHow would you like to determine ability scores?",
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

    async def _show_ability_assignment(self, interaction: discord.Interaction, campaign: Campaign):
        """Show ability score assignment after method selection."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        if state.ability_method == AbilityScoreMethod.POINT_BUY:
            await self._show_point_buy(interaction, campaign)
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
                await self._show_race_select(assign_interaction, campaign)

            view = AbilityAssignmentView(state, scores, on_assignment_complete)
            await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    async def _show_point_buy(self, interaction: discord.Interaction, campaign: Campaign):
        """Show point buy interface."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        async def on_point_buy_complete(pb_interaction: discord.Interaction):
            state.final_abilities = state.point_buy_state.to_ability_scores()
            await self._show_race_select(pb_interaction, campaign)

        view = PointBuyView(state, on_point_buy_complete)
        await interaction.followup.send(embed=view.get_embed(), view=view, ephemeral=True)

    async def _show_race_select(self, interaction: discord.Interaction, campaign: Campaign):
        """Show race selection."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        async def on_race_complete(race_interaction: discord.Interaction):
            await self._show_class_select(race_interaction, campaign)

        view = RaceSelectView(state, on_race_complete)

        embed = discord.Embed(
            title="Choose Your Race",
            description="Select a race for your character. Racial bonuses will be applied to your ability scores.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_class_select(self, interaction: discord.Interaction, campaign: Campaign):
        """Show class selection."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        async def on_class_complete(class_interaction: discord.Interaction):
            await self._show_skill_select(class_interaction, campaign)

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

    async def _show_skill_select(self, interaction: discord.Interaction, campaign: Campaign):
        """Show skill selection."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
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
                options = from_data.get("options", [])
                if options and "skill" in str(options[0].get("item", {}).get("index", "")):
                    skill_choice = choice
                    break

        if not skill_choice:
            await self._show_appearance(interaction, campaign)
            return

        num_skills = skill_choice.get("choose", 2)
        available_skills = []
        for opt in skill_choice.get("from", {}).get("options", []):
            item = opt.get("item", {})
            skill_index = item.get("index", "").replace("skill-", "")
            if skill_index:
                available_skills.append(skill_index)

        async def on_skills_complete(skill_interaction: discord.Interaction):
            await self._show_appearance(skill_interaction, campaign)

        view = SkillSelectView(state, available_skills, num_skills, on_skills_complete)

        class_name = class_data.get("name", state.class_index.title())
        embed = discord.Embed(
            title="Choose Your Skills",
            description=f"As a {class_name}, choose {num_skills} skills to be proficient in.",
            color=discord.Color.blue(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_appearance(self, interaction: discord.Interaction, campaign: Campaign):
        """Show appearance step (only if images enabled)."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        # Skip if images aren't enabled
        try:
            from ...config import get_profile
            profile = get_profile()
            if not profile.immersion.image_enabled:
                await self._show_confirmation(interaction, campaign)
                return
        except Exception:
            await self._show_confirmation(interaction, campaign)
            return

        from ..views.character_appearance import AppearanceModal

        async def on_appearance_done(description: str):
            state.description = description
            await self._show_confirmation(interaction, campaign)

        modal = AppearanceModal(on_appearance_done)

        view = discord.ui.View(timeout=120)

        async def show_modal(btn_interaction: discord.Interaction):
            await btn_interaction.response.send_modal(modal)

        async def skip(btn_interaction: discord.Interaction):
            await btn_interaction.response.defer()
            state.description = ""
            await self._show_confirmation(btn_interaction, campaign)

        describe_btn = discord.ui.Button(label="Describe Appearance", style=discord.ButtonStyle.primary)
        describe_btn.callback = show_modal
        skip_btn = discord.ui.Button(label="Skip", style=discord.ButtonStyle.secondary)
        skip_btn.callback = skip
        view.add_item(describe_btn)
        view.add_item(skip_btn)

        embed = discord.Embed(
            title="Character Appearance",
            description=(
                f"**{state.name}** the {state.race_index.title()} {state.class_index.title()}\n\n"
                "Describe your character's physical appearance. This will be used for "
                "scene image generation and your character sheet.\n\n"
                "*Example: A tall elf with silver hair and emerald eyes, wearing worn leather armor...*"
            ),
            color=discord.Color.purple(),
        )

        await interaction.followup.send(embed=embed, view=view, ephemeral=True)

    async def _show_confirmation(self, interaction: discord.Interaction, campaign: Campaign):
        """Show character confirmation."""
        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        async def on_confirm(confirm_interaction: discord.Interaction):
            await self._finalize_character(confirm_interaction, campaign)

        async def on_cancel(cancel_interaction: discord.Interaction):
            _creation_states.pop((guild_id, user_id), None)
            await cancel_interaction.followup.send(
                "Character creation cancelled. Click 'Join Campaign' to try again!",
                ephemeral=True,
            )

        view = ConfirmCharacterView(on_confirm, on_cancel)

        # Build preview embed from state
        race_data = self.srd.get_race(state.race_index)
        class_data = self.srd.get_class(state.class_index)
        race_name = race_data["name"] if race_data else state.race_index.title()
        class_name = class_data["name"] if class_data else state.class_index.title()

        desc = f"**{state.name}**\n{race_name} {class_name}"
        if state.description:
            desc += f"\n\n*{state.description[:200]}*"

        embed = discord.Embed(
            title="Confirm Your Character",
            description=desc,
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

    async def _finalize_character(self, interaction: discord.Interaction, campaign: Campaign):
        """Create and save the character."""
        from ...game.character.starting_equipment import assign_starting_equipment

        user_id = interaction.user.id
        guild_id = interaction.guild_id or 0
        state = _creation_states.get((guild_id, user_id))
        if not state:
            return

        try:
            # Create the character
            character = self.creator.build_character(state)

            # Set appearance description from wizard
            if state.description:
                character.description = state.description

            # Save to database
            char_repo = await get_character_repo()
            await char_repo.create(character)

            # Assign starting equipment and gold
            equipment_result = await assign_starting_equipment(
                character.id,
                character.class_index,
            )

            # Clean up state
            _creation_states.pop((guild_id, user_id), None)

            # Add to players list
            if campaign.id not in _campaign_players:
                _campaign_players[campaign.id] = []
            if user_id not in _campaign_players[campaign.id]:
                _campaign_players[campaign.id].append(user_id)

            embed = build_character_sheet_embed(character)
            embed.title = f":tada: {character.name} Joins {campaign.name}!"
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
                "character_created_from_lobby",
                character_id=character.id,
                campaign_id=campaign.id,
                user_id=user_id,
                name=character.name,
                race=character.race_index,
                class_name=character.class_index,
            )

        except Exception as e:
            logger.error("character_creation_failed", error=str(e), exc_info=True)
            await interaction.followup.send(
                f"Failed to create character: {e}",
                ephemeral=True,
            )

    async def _install_sourcebook(
        self,
        session,
        campaign: Campaign,
        sourcebook_key: str,
        interaction: discord.Interaction,
    ) -> tuple[str, str]:
        """Install the chosen book into the fresh session.

        Returns ``(authored_scene, title)`` for the opening narration, or
        ``("", "")`` when nothing was installed.

        The install itself lives in ``game.knowledge.sourcebook_library``
        because it is the only production path that seeds an authored world
        and therefore the thing most worth testing — and a cog cannot be
        imported without py-cord. What stays here is the genuinely Discord
        half: showing the DM why they did not get the world they asked for.
        """
        from ...data.repositories.sourcebook_repo import get_sourcebook_repo
        from ...game.knowledge.sourcebook_library import install_for_campaign

        outcome = await install_for_campaign(
            await get_sourcebook_repo(), session, campaign.id, sourcebook_key,
        )
        if outcome.error:
            # Reported, not raised. The party is already in a started
            # session; failing it to punish a bad book would cost them the
            # session too. They get an improvised world and a visible reason.
            await interaction.followup.send(
                f":warning: The sourcebook could not be installed "
                f"({outcome.error}). The adventure will start with an "
                "improvised world.",
                ephemeral=True,
            )
        return outcome.scene, outcome.title

    async def _handle_start(self, interaction: discord.Interaction, campaign: Campaign):
        """Handle the DM starting the game - actually starts the session.

        Caller (the lobby button) has already deferred. The has_active_session
        check + start_session must run under `_session_start_lock` so two
        concurrent "Start Game" clicks can't both pass the check and create
        duplicate sessions.
        """
        # Check database for characters in this campaign
        char_repo = await get_character_repo()
        characters = await char_repo.get_all_by_campaign(campaign.id)

        if not characters:
            await interaction.followup.send(
                "No players have joined yet! Wait for players to click 'Join Campaign'.",
                ephemeral=True,
            )
            return

        # Start the actual game session
        from ...game.session import get_session_manager

        session_manager = get_session_manager()

        async with _session_start_lock:
            # Check inside the lock so the result is consistent with the start call below.
            if session_manager.has_active_session(interaction.channel_id):
                await interaction.followup.send(
                    "A game session is already active in this channel!",
                    ephemeral=True,
                )
                return

            # Start the session with the actual campaign ID
            # Note: dm_user_id=None because the AI is the DM, not a human
            session = await session_manager.start_session(
                channel_id=interaction.channel_id,
                guild_id=interaction.guild_id,
                campaign_id=campaign.id,
                dm_user_id=None,  # AI is the DM, no human DM
            )

        # Install the authored world, if the DM chose one. After start_session
        # (the graph and world store exist only from there) and before the
        # opening narrative, which must describe the seeded room rather than
        # invent a competing one.
        authored_scene = ""
        book_title = ""
        chosen = _campaign_sourcebook.get(campaign.id) or await _bound_sourcebook(
            campaign.id
        )
        if chosen:
            authored_scene, book_title = await self._install_sourcebook(
                session, campaign, chosen, interaction,
            )
            if authored_scene:
                # Snapshot NOW, not after the first processed turn. The
                # install has already rewritten location, roster and facts;
                # a crash before turn 1 would otherwise lose the seeded world
                # entirely, and recovery has no snapshot to fall back to.
                try:
                    await session_manager._persist_world_snapshot(session)
                except Exception as e:
                    logger.warning(
                        "seeded_world_snapshot_failed",
                        campaign_id=campaign.id, error=str(e), exc_info=True,
                    )

        # Auto-join all players who have characters
        for character in characters:
            member = interaction.guild.get_member(character.discord_user_id)
            if member:
                await session_manager.join_session(
                    channel_id=interaction.channel_id,
                    user_id=character.discord_user_id,
                    user_name=member.display_name,
                    character=character,
                )

        # Build player list for embed
        player_lines = []
        for char in characters:
            member = interaction.guild.get_member(char.discord_user_id)
            name = member.display_name if member else "Unknown"
            player_lines.append(f"**{char.name}** ({name}) - L{char.level} {char.class_index.title()}")

        embed = discord.Embed(
            title=f":crossed_swords: {campaign.name} Begins!",
            description=campaign.world_setting or "A new adventure awaits...",
            color=discord.Color.green(),
        )

        embed.add_field(
            name=f"Players ({len(characters)})",
            value="\n".join(player_lines),
            inline=False,
        )

        if book_title:
            embed.add_field(name="Sourcebook", value=book_title, inline=False)

        embed.add_field(
            name="How to Play",
            value=(
                "Describe your actions in chat and the AI Dungeon Master will respond!\n"
                "Use `/game status` to see session info."
            ),
            inline=False,
        )

        # followup, NOT response: the lobby button already deferred (see this
        # method's docstring), so `response.send_message` raises
        # InteractionResponded -- after the session has started and, now, after
        # a sourcebook has installed. The opening narration below never ran.
        await interaction.followup.send(embed=embed)

        # Generate opening narrative from the AI DM
        try:
            from ...llm.brains.narrator import get_narrator
            from ...memory import get_memory_manager
            narrator = get_narrator()
            opening, opening_effects = await narrator.generate_opening(
                campaign_name=campaign.name,
                world_setting=campaign.world_setting,
                characters=characters,
                # Empty for an improvised campaign, which leaves the
                # invent-a-scene path exactly as it was.
                authored_scene=authored_scene,
            )

            # Process tool call effects (registers NPCs, assigns voices).
            # Discord context, so the session_key is f"discord:{channel_id}".
            if opening_effects:
                from ...game.scene.registry import get_scene_registry
                from ...llm.effects import EffectExecutor, EffectValidator
                _session_key = f"discord:{interaction.channel_id}"
                scene_registry = get_scene_registry(campaign.id, _session_key)
                # WITH the session: `_execute_add_npc` only routes through
                # `world_store.ensure_npc` when it has one, and that is the
                # single place a tool-path NPCState is minted. Sessionless,
                # every opening NPC became a SceneEntity with no `npc_id` --
                # unlinked from world state, which is precisely the
                # split-identity Stage C exists to prevent. With a sourcebook
                # it is worse: the book has already put its cast on the
                # roster, so an add_npc for one of them minted a twin.
                executor = EffectExecutor(
                    scene_registry=scene_registry, session=session,
                )
                # VALIDATED, like every other effect path. Executing raw let
                # the opening do things no later turn is allowed to -- most
                # sharply, put a canonically dead NPC on stage alive, which
                # is exactly what a sourcebook that authors a death invites
                # the narrator to do in its first paragraph.
                validator = EffectValidator(session=session)
                for effect in opening_effects:
                    logger.info(
                        "opening_effect",
                        type=effect.effect_type.value,
                        npc_name=effect.npc_name,
                        dialogue_indices=effect.dialogue_indices,
                        dialogue_emotions=effect.dialogue_emotions,
                    )
                    verdict = validator.validate(effect)
                    if not verdict.valid:
                        logger.warning(
                            "opening_effect_rejected",
                            type=effect.effect_type.value,
                            npc_name=effect.npc_name,
                            reason=verdict.rejection_reason,
                        )
                        continue
                    await executor.execute(effect)

            if opening:
                # Store opening in memory so future narrator calls have scene context
                memory = await get_memory_manager(campaign.id)
                memory.update_scene(opening)

                # Also add to message history so narrator has conversation context
                await memory.add_dm_response(content=opening, is_narration=True)

                narrator_embed = discord.Embed(
                    description=opening,
                    color=discord.Color.dark_gold(),
                )
                narrator_embed.set_author(name="The Story Begins...")
                await interaction.followup.send(embed=narrator_embed)

                # Fire immersion pipelines for opening narration
                import asyncio as _asyncio
                try:
                    from ...config import get_profile
                    from ...game.frontend import GameEvent

                    profile = get_profile()
                    imm = profile.immersion
                    if imm.tts_enabled or imm.image_enabled:
                        from ..frontends.discord_text import DiscordTextFrontend
                        from ...game.scene.registry import get_scene_registry as _get_sr
                        channel = interaction.channel
                        frontend = DiscordTextFrontend(channel)
                        _sr = _get_sr(campaign.id, f"discord:{interaction.channel_id}")
                        event = GameEvent.narrative_complete(
                            narrative=opening,
                            proposed_effects=opening_effects or [],
                            scene_entities=_sr.get_all() if _sr else [],
                            player_characters=characters,
                        )
                        if imm.tts_enabled:
                            _asyncio.create_task(frontend._generate_and_send_audio(event))
                        if imm.image_enabled:
                            _asyncio.create_task(frontend._generate_and_send_image(event))
                except Exception as imm_err:
                    logger.debug("opening_immersion_failed", error=str(imm_err), exc_info=True)

        except Exception as e:
            logger.warning("failed_to_generate_opening", error=str(e), exc_info=True)

        logger.info(
            "campaign_starting",
            campaign_id=campaign.id,
            player_count=len(characters),
        )

    async def _update_lobby(self, interaction: discord.Interaction, campaign: Campaign):
        """Update the lobby embed with current player count."""
        players = _campaign_players.get(campaign.id, [])

        view = CampaignLobbyView(
            campaign=campaign,
            dm_id=campaign.dm_user_id,
            on_join=self._handle_join,
            on_start=self._handle_start,
            players=players,
        )

        try:
            await interaction.message.edit(embed=view.get_embed(), view=view)
        except Exception:
            pass  # Message might not be editable

    @campaign.command(name="create", description="Create a new campaign")
    async def campaign_create(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Name for your campaign",
            required=True,
            max_length=100,
        ),
        description: discord.Option(
            str,
            "Brief description of the campaign",
            required=False,
            max_length=500,
        ),
        world_setting: discord.Option(
            str,
            "Description of the world/setting for the AI DM",
            required=False,
            max_length=1000,
        ),
        sourcebook: discord.Option(
            str,
            "Authored sourcebook to start from (optional)",
            required=False,
            autocomplete=discord.utils.basic_autocomplete(_sourcebook_choices),
        ),
    ):
        """Create a new campaign."""
        await ctx.defer()
        repo = await get_campaign_repo()

        # Resolve the book BEFORE creating anything. A typo'd title must not
        # leave a campaign created and silently bookless -- the DM asked for
        # an authored world and would not find out until the opening prose
        # invented one.
        chosen_key = ""
        if sourcebook:
            chosen_key, rejected = await _resolve_sourcebook(sourcebook)
            if not chosen_key:
                # Name the unreadable files. The scanner records WHICH file
                # and WHY, and discarding that left a DM whose book has one
                # bad line staring at "no sourcebook matches" with nothing
                # to act on.
                detail = ""
                if rejected:
                    detail = "\n\nUnreadable books in the library:\n" + "\n".join(
                        f"- `{name}`: {reason.splitlines()[0][:180]}"
                        for name, reason in rejected[:5]
                    )
                await ctx.respond(
                    f"No sourcebook matches '{sourcebook}'. "
                    "Leave it blank for an improvised world, or use the "
                    f"autocomplete to pick one.{detail}",
                    ephemeral=True,
                )
                return

        # Check if campaign with same name exists
        existing = await repo.get_by_name_and_guild(name, ctx.guild_id)
        if existing:
            await ctx.respond(
                f"A campaign named '{name}' already exists in this server.",
                ephemeral=True,
            )
            return

        # Create campaign
        campaign = Campaign(
            guild_id=ctx.guild_id,
            name=name,
            description=description,
            world_setting=world_setting or "A classic high fantasy world filled with magic, monsters, and adventure.",
            dm_user_id=ctx.author.id,
        )

        await repo.create(campaign)

        # Set as active campaign for this guild
        set_active_campaign(ctx.guild_id, campaign.id)

        # Initialize player list
        _campaign_players[campaign.id] = []
        if chosen_key:
            # DURABLE, not just in memory. The lobby can sit for days and a
            # restart in between used to drop the choice silently -- the DM
            # got an improvised campaign with no indication the book was
            # gone, and no command to pick it again. `campaign_sourcebook`
            # is the table that already answers "which book is this
            # campaign's"; binding here simply answers it earlier, and the
            # install is idempotent so binding twice costs nothing.
            _campaign_sourcebook[campaign.id] = chosen_key
            try:
                from ...data.repositories.sourcebook_repo import (
                    get_sourcebook_repo,
                )
                await (await get_sourcebook_repo()).bind_campaign(
                    campaign.id, chosen_key,
                )
            except Exception as e:
                logger.warning(
                    "sourcebook_prebind_failed",
                    campaign_id=campaign.id, error=str(e), exc_info=True,
                )

        # Create lobby view
        view = CampaignLobbyView(
            campaign=campaign,
            dm_id=ctx.author.id,
            on_join=self._handle_join,
            on_start=self._handle_start,
            players=[],
        )

        await ctx.respond(embed=view.get_embed(), view=view)

        logger.info(
            "campaign_created",
            campaign_id=campaign.id,
            name=name,
            guild_id=ctx.guild_id,
            dm_id=ctx.author.id,
        )

    @campaign.command(name="list", description="List all campaigns in this server")
    async def campaign_list(
        self,
        ctx: discord.ApplicationContext,
    ):
        """List all campaigns."""
        await ctx.defer()
        repo = await get_campaign_repo()
        campaigns = await repo.get_all_by_guild(ctx.guild_id)

        if not campaigns:
            await ctx.respond(
                "No campaigns in this server yet. Create one with `/campaign create`!",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=":books: Campaigns",
            description=f"Found {len(campaigns)} campaign(s) in this server.",
            color=discord.Color.blue(),
        )

        for campaign in campaigns[:10]:  # Limit to 10
            dm = ctx.guild.get_member(campaign.dm_user_id)
            dm_name = dm.display_name if dm else "Unknown"

            last_played = "Never"
            if campaign.last_played_at:
                days_ago = (discord.utils.utcnow() - campaign.last_played_at.replace(tzinfo=None)).days
                if days_ago == 0:
                    last_played = "Today"
                elif days_ago == 1:
                    last_played = "Yesterday"
                else:
                    last_played = f"{days_ago} days ago"

            value = f"DM: {dm_name}\nLast played: {last_played}"
            if campaign.description:
                value += f"\n_{campaign.description[:80]}..._" if len(campaign.description) > 80 else f"\n_{campaign.description}_"

            embed.add_field(
                name=f":crossed_swords: {campaign.name}",
                value=value,
                inline=False,
            )

        await ctx.respond(embed=embed)

    @campaign.command(name="info", description="Show detailed campaign information")
    async def campaign_info(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Campaign name",
            required=True,
        ),
    ):
        """Show campaign details."""
        await ctx.defer()
        repo = await get_campaign_repo()
        campaign = await repo.get_by_name_and_guild(name, ctx.guild_id)

        if not campaign:
            await ctx.respond(
                f"Campaign '{name}' not found. Use `/campaign list` to see available campaigns.",
                ephemeral=True,
            )
            return

        # Get characters in campaign
        char_repo = await get_character_repo()
        characters = await char_repo.get_all_by_campaign(campaign.id)

        embed = discord.Embed(
            title=f":scroll: {campaign.name}",
            description=campaign.description or "_No description_",
            color=discord.Color.gold(),
        )

        dm = ctx.guild.get_member(campaign.dm_user_id)
        embed.add_field(
            name="Dungeon Master",
            value=dm.mention if dm else "Unknown",
            inline=True,
        )

        embed.add_field(
            name="Created",
            value=campaign.created_at.strftime("%Y-%m-%d"),
            inline=True,
        )

        if campaign.last_played_at:
            embed.add_field(
                name="Last Played",
                value=campaign.last_played_at.strftime("%Y-%m-%d"),
                inline=True,
            )

        if characters:
            char_list = []
            for char in characters[:8]:
                member = ctx.guild.get_member(char.discord_user_id)
                player_name = member.display_name if member else "Unknown"
                char_list.append(f"**{char.name}** (L{char.level} {char.class_index.title()}) - {player_name}")

            embed.add_field(
                name=f"Characters ({len(characters)})",
                value="\n".join(char_list) or "None",
                inline=False,
            )
        else:
            embed.add_field(
                name="Characters",
                value="_No characters yet_",
                inline=False,
            )

        embed.add_field(
            name="World Setting",
            value=campaign.world_setting[:500] if campaign.world_setting else "_Default fantasy setting_",
            inline=False,
        )

        await ctx.respond(embed=embed)

    @campaign.command(name="settings", description="Update campaign settings (DM only)")
    async def campaign_settings(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Campaign name",
            required=True,
        ),
        world_setting: discord.Option(
            str,
            "New world setting description",
            required=False,
            max_length=1000,
        ),
        description: discord.Option(
            str,
            "New campaign description",
            required=False,
            max_length=500,
        ),
    ):
        """Update campaign settings."""
        await ctx.defer()
        repo = await get_campaign_repo()
        campaign = await repo.get_by_name_and_guild(name, ctx.guild_id)

        if not campaign:
            await ctx.respond(
                f"Campaign '{name}' not found.",
                ephemeral=True,
            )
            return

        # Check if user is the DM
        if campaign.dm_user_id != ctx.author.id and not ctx.author.guild_permissions.administrator:
            await ctx.respond(
                "Only the DM can modify campaign settings.",
                ephemeral=True,
            )
            return

        # Update fields
        changes = []
        if world_setting:
            campaign.world_setting = world_setting
            changes.append("World setting updated")

        if description:
            campaign.description = description
            changes.append("Description updated")

        if not changes:
            await ctx.respond(
                "No changes specified. Use the `world_setting` or `description` options.",
                ephemeral=True,
            )
            return

        await repo.update(campaign)

        embed = discord.Embed(
            title=":gear: Campaign Updated",
            description=f"**{campaign.name}**\n\n" + "\n".join(f":white_check_mark: {c}" for c in changes),
            color=discord.Color.green(),
        )

        await ctx.respond(embed=embed)

        logger.info(
            "campaign_updated",
            campaign_id=campaign.id,
            changes=changes,
        )

    @campaign.command(name="delete", description="Delete a campaign (DM only)")
    async def campaign_delete(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Campaign name to delete",
            required=True,
        ),
        confirm: discord.Option(
            bool,
            "Confirm deletion (this cannot be undone!)",
            required=True,
        ),
    ):
        """Delete a campaign."""
        await ctx.defer(ephemeral=True)
        if not confirm:
            await ctx.respond(
                "Set `confirm` to True to delete the campaign.",
                ephemeral=True,
            )
            return

        repo = await get_campaign_repo()
        campaign = await repo.get_by_name_and_guild(name, ctx.guild_id)

        if not campaign:
            await ctx.respond(
                f"Campaign '{name}' not found.",
                ephemeral=True,
            )
            return

        # Check if user is the DM or admin
        if campaign.dm_user_id != ctx.author.id and not ctx.author.guild_permissions.administrator:
            await ctx.respond(
                "Only the DM or a server administrator can delete a campaign.",
                ephemeral=True,
            )
            return

        await repo.delete(campaign.id)

        embed = discord.Embed(
            title=":wastebasket: Campaign Deleted",
            description=f"**{name}** has been permanently deleted.\nAll characters and session data have been removed.",
            color=discord.Color.red(),
        )

        await ctx.respond(embed=embed)

        logger.info(
            "campaign_deleted",
            campaign_id=campaign.id,
            name=name,
            deleted_by=ctx.author.id,
        )

    @campaign.command(name="transfer", description="Transfer DM role to another user")
    async def campaign_transfer(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Campaign name",
            required=True,
        ),
        new_dm: discord.Option(
            discord.Member,
            "The new DM for this campaign",
            required=True,
        ),
    ):
        """Transfer DM role to another user."""
        await ctx.defer()
        repo = await get_campaign_repo()
        campaign = await repo.get_by_name_and_guild(name, ctx.guild_id)

        if not campaign:
            await ctx.respond(
                f"Campaign '{name}' not found.",
                ephemeral=True,
            )
            return

        # Check if user is the current DM
        if campaign.dm_user_id != ctx.author.id and not ctx.author.guild_permissions.administrator:
            await ctx.respond(
                "Only the current DM can transfer the DM role.",
                ephemeral=True,
            )
            return

        if new_dm.bot:
            await ctx.respond(
                "Cannot transfer DM role to a bot.",
                ephemeral=True,
            )
            return

        campaign.dm_user_id = new_dm.id
        await repo.update(campaign)

        embed = discord.Embed(
            title=":crown: DM Role Transferred",
            description=f"**{campaign.name}**\n\n{new_dm.mention} is now the Dungeon Master!",
            color=discord.Color.gold(),
        )

        await ctx.respond(embed=embed)

        logger.info(
            "campaign_dm_transferred",
            campaign_id=campaign.id,
            old_dm=ctx.author.id,
            new_dm=new_dm.id,
        )


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(CampaignCog(bot))
