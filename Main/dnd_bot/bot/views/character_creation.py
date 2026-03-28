"""Discord UI views for character creation wizard."""

from typing import Optional, Callable, Awaitable
import discord

from ...models import AbilityScore, Skill
from ...data.srd import get_srd
from ...game.character.creation import (
    AbilityScoreMethod,
    CharacterCreationState,
    CharacterCreator,
    PointBuyState,
    STANDARD_ARRAY,
    get_creator,
)


class NameModal(discord.ui.Modal):
    """Modal for entering character name."""

    def __init__(self, state: CharacterCreationState, on_complete: Callable[[discord.Interaction], Awaitable[None]]):
        super().__init__(title="Name Your Character")
        self.state = state
        self.on_complete = on_complete

        self.name_input = discord.ui.InputText(
            label="Character Name",
            placeholder="Enter your character's name",
            min_length=2,
            max_length=64,
            required=True,
        )
        self.add_item(self.name_input)

    async def callback(self, interaction: discord.Interaction):
        self.state.name = self.name_input.value
        await interaction.response.defer()
        await self.on_complete(interaction)


class AbilityScoreMethodView(discord.ui.View):
    """View for selecting ability score generation method."""

    def __init__(
        self,
        state: CharacterCreationState,
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.state = state
        self.on_complete = on_complete
        self.message: Optional[discord.Message] = None

    @discord.ui.button(label="Roll 4d6 Drop Lowest", style=discord.ButtonStyle.primary, row=0)
    async def roll_method(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.state.ability_method = AbilityScoreMethod.ROLL
        creator = get_creator()
        self.state.ability_rolls = creator.roll_ability_scores()
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)

    @discord.ui.button(label="Standard Array", style=discord.ButtonStyle.secondary, row=0)
    async def standard_array(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.state.ability_method = AbilityScoreMethod.STANDARD_ARRAY
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)

    @discord.ui.button(label="Point Buy", style=discord.ButtonStyle.secondary, row=0)
    async def point_buy(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.state.ability_method = AbilityScoreMethod.POINT_BUY
        self.state.point_buy_state = PointBuyState()
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)


class AbilityAssignmentView(discord.ui.View):
    """View for assigning rolled/array scores to abilities one at a time."""

    ABILITIES = [
        ("Strength", AbilityScore.STRENGTH),
        ("Dexterity", AbilityScore.DEXTERITY),
        ("Constitution", AbilityScore.CONSTITUTION),
        ("Intelligence", AbilityScore.INTELLIGENCE),
        ("Wisdom", AbilityScore.WISDOM),
        ("Charisma", AbilityScore.CHARISMA),
    ]

    def __init__(
        self,
        state: CharacterCreationState,
        available_scores: list[int],
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.state = state
        self.available_scores = sorted(available_scores, reverse=True)
        self.on_complete = on_complete
        self.assignments: dict[AbilityScore, int] = {}
        self.current_score_index = 0
        self.message: Optional[discord.Message] = None

        # Add a single select for choosing which ability gets the current score
        self._add_ability_select()

    def _add_ability_select(self):
        """Add select menu for choosing ability."""
        options = [
            discord.SelectOption(
                label=name,
                value=ability.value,
                description=f"Assign {self.available_scores[self.current_score_index]} to {name}",
            )
            for name, ability in self.ABILITIES
            if ability not in self.assignments
        ]

        select = discord.ui.Select(
            placeholder=f"Assign {self.available_scores[self.current_score_index]} to which ability?",
            options=options,
            row=0,
        )
        select.callback = self._on_ability_selected
        self.add_item(select)

    def get_embed(self) -> discord.Embed:
        """Build the current assignment state embed."""
        current_score = self.available_scores[self.current_score_index]
        remaining = self.available_scores[self.current_score_index:]

        embed = discord.Embed(
            title="Assign Ability Scores",
            description=f"**Current score to assign: {current_score}**\n\nRemaining: {', '.join(str(s) for s in remaining)}",
            color=discord.Color.blue(),
        )

        # Show current assignments
        for name, ability in self.ABILITIES:
            if ability in self.assignments:
                score = self.assignments[ability]
                mod = (score - 10) // 2
                mod_str = f"+{mod}" if mod >= 0 else str(mod)
                embed.add_field(name=name, value=f"**{score}** ({mod_str})", inline=True)
            else:
                embed.add_field(name=name, value="_unassigned_", inline=True)

        return embed

    async def _on_ability_selected(self, interaction: discord.Interaction):
        """Handle ability selection."""
        ability_value = interaction.data["values"][0]
        ability = AbilityScore(ability_value)
        current_score = self.available_scores[self.current_score_index]

        # Assign the score
        self.assignments[ability] = current_score
        self.current_score_index += 1

        # Check if all abilities assigned
        if len(self.assignments) == 6:
            self.state.ability_assignments = self.assignments
            await interaction.response.defer()
            self.stop()
            await self.on_complete(interaction)
        else:
            # Update the view for next assignment
            self.clear_items()
            self._add_ability_select()
            await interaction.response.edit_message(embed=self.get_embed(), view=self)


class PointBuyView(discord.ui.View):
    """View for point buy ability score allocation."""

    def __init__(
        self,
        state: CharacterCreationState,
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=600)
        self.state = state
        self.on_complete = on_complete
        self.message: Optional[discord.Message] = None

    def get_embed(self) -> discord.Embed:
        """Build the point buy embed."""
        pb = self.state.point_buy_state
        embed = discord.Embed(
            title="Point Buy - Allocate Ability Scores",
            description=f"Points Remaining: **{pb.points_remaining}** / 27",
            color=discord.Color.blue(),
        )

        for ability in AbilityScore:
            score = pb.scores[ability]
            mod = (score - 10) // 2
            mod_str = f"+{mod}" if mod >= 0 else str(mod)
            embed.add_field(
                name=ability.name.title(),
                value=f"**{score}** ({mod_str})",
                inline=True,
            )

        return embed

    @discord.ui.button(label="STR +", style=discord.ButtonStyle.success, row=0)
    async def str_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.STRENGTH, 1, interaction)

    @discord.ui.button(label="STR -", style=discord.ButtonStyle.danger, row=0)
    async def str_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.STRENGTH, -1, interaction)

    @discord.ui.button(label="DEX +", style=discord.ButtonStyle.success, row=0)
    async def dex_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.DEXTERITY, 1, interaction)

    @discord.ui.button(label="DEX -", style=discord.ButtonStyle.danger, row=0)
    async def dex_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.DEXTERITY, -1, interaction)

    @discord.ui.button(label="CON +", style=discord.ButtonStyle.success, row=1)
    async def con_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.CONSTITUTION, 1, interaction)

    @discord.ui.button(label="CON -", style=discord.ButtonStyle.danger, row=1)
    async def con_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.CONSTITUTION, -1, interaction)

    @discord.ui.button(label="INT +", style=discord.ButtonStyle.success, row=1)
    async def int_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.INTELLIGENCE, 1, interaction)

    @discord.ui.button(label="INT -", style=discord.ButtonStyle.danger, row=1)
    async def int_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.INTELLIGENCE, -1, interaction)

    @discord.ui.button(label="WIS +", style=discord.ButtonStyle.success, row=2)
    async def wis_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.WISDOM, 1, interaction)

    @discord.ui.button(label="WIS -", style=discord.ButtonStyle.danger, row=2)
    async def wis_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.WISDOM, -1, interaction)

    @discord.ui.button(label="CHA +", style=discord.ButtonStyle.success, row=2)
    async def cha_up(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.CHARISMA, 1, interaction)

    @discord.ui.button(label="CHA -", style=discord.ButtonStyle.danger, row=2)
    async def cha_down(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self._modify(AbilityScore.CHARISMA, -1, interaction)

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.primary, row=3)
    async def confirm(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)

    async def _modify(self, ability: AbilityScore, delta: int, interaction: discord.Interaction):
        pb = self.state.point_buy_state
        if delta > 0:
            pb.increase(ability)
        else:
            pb.decrease(ability)

        await interaction.response.edit_message(embed=self.get_embed(), view=self)


class RaceSelectView(discord.ui.View):
    """View for selecting race."""

    def __init__(
        self,
        state: CharacterCreationState,
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.state = state
        self.on_complete = on_complete
        self.message: Optional[discord.Message] = None

        srd = get_srd()
        races = srd.get_all_races()

        options = [
            discord.SelectOption(
                label=race["name"],
                value=race["index"],
                description=f"Speed: {race.get('speed', 30)}ft"[:100],
            )
            for race in races
        ]

        select = discord.ui.Select(
            placeholder="Choose your race...",
            options=options[:25],  # Discord limit
        )
        select.callback = self._on_race_select
        self.add_item(select)

    async def _on_race_select(self, interaction: discord.Interaction):
        self.state.race_index = interaction.data["values"][0]
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)


class ClassSelectView(discord.ui.View):
    """View for selecting class."""

    def __init__(
        self,
        state: CharacterCreationState,
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.state = state
        self.on_complete = on_complete
        self.message: Optional[discord.Message] = None

        srd = get_srd()
        classes = srd.get_all_classes()

        options = [
            discord.SelectOption(
                label=cls["name"],
                value=cls["index"],
                description=f"Hit Die: d{cls.get('hit_die', 8)}"[:100],
            )
            for cls in classes
        ]

        select = discord.ui.Select(
            placeholder="Choose your class...",
            options=options[:25],
        )
        select.callback = self._on_class_select
        self.add_item(select)

    async def _on_class_select(self, interaction: discord.Interaction):
        self.state.class_index = interaction.data["values"][0]
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)


class SkillSelectView(discord.ui.View):
    """View for selecting skill proficiencies."""

    def __init__(
        self,
        state: CharacterCreationState,
        available_skills: list[str],
        choose_count: int,
        on_complete: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.state = state
        self.available_skills = available_skills
        self.choose_count = choose_count
        self.on_complete = on_complete
        self.message: Optional[discord.Message] = None

        options = [
            discord.SelectOption(
                label=skill.replace("-", " ").title(),
                value=skill,
            )
            for skill in available_skills
        ]

        select = discord.ui.Select(
            placeholder=f"Choose {choose_count} skills...",
            options=options[:25],
            min_values=choose_count,
            max_values=choose_count,
        )
        select.callback = self._on_skills_select
        self.add_item(select)

    async def _on_skills_select(self, interaction: discord.Interaction):
        skill_map = {s.value: s for s in Skill}
        self.state.skill_choices = [
            skill_map[val] for val in interaction.data["values"]
        ]
        await interaction.response.defer()
        self.stop()
        await self.on_complete(interaction)


class ConfirmCharacterView(discord.ui.View):
    """View for confirming character creation."""

    def __init__(
        self,
        on_confirm: Callable[[discord.Interaction], Awaitable[None]],
        on_cancel: Callable[[discord.Interaction], Awaitable[None]],
    ):
        super().__init__(timeout=300)
        self.on_confirm = on_confirm
        self.on_cancel = on_cancel
        self.message: Optional[discord.Message] = None

    @discord.ui.button(label="Create Character", style=discord.ButtonStyle.success)
    async def confirm(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self.on_confirm(interaction)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.defer()
        self.stop()
        await self.on_cancel(interaction)
