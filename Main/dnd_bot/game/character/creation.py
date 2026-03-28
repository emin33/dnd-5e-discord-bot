"""Character creation logic and ability score generation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid

from ...models import (
    AbilityScore,
    AbilityScores,
    Character,
    CharacterProficiency,
    DeathSaves,
    HitDice,
    HitPoints,
    Skill,
    SpellSlots,
)
from ...data.srd import get_srd
from ..mechanics.dice import DiceRoll, get_roller


class AbilityScoreMethod(str, Enum):
    """Methods for generating ability scores."""

    ROLL = "roll"  # 4d6 drop lowest, 6 times
    STANDARD_ARRAY = "standard_array"  # 15, 14, 13, 12, 10, 8
    POINT_BUY = "point_buy"  # 27 points to distribute


# Standard array values
STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]

# Point buy costs
POINT_BUY_COSTS = {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 7,
    15: 9,
}
POINT_BUY_TOTAL = 27
POINT_BUY_MIN = 8
POINT_BUY_MAX = 15


@dataclass
class AbilityScoreRolls:
    """Result of rolling ability scores."""

    rolls: list[DiceRoll]
    totals: list[int]

    def get_sorted_totals(self, descending: bool = True) -> list[int]:
        """Get totals sorted by value."""
        return sorted(self.totals, reverse=descending)


@dataclass
class PointBuyState:
    """State for point buy ability score allocation."""

    points_remaining: int = POINT_BUY_TOTAL
    scores: dict[AbilityScore, int] = field(
        default_factory=lambda: {ability: 8 for ability in AbilityScore}
    )

    def get_cost(self, score: int) -> int:
        """Get the point cost for a score."""
        return POINT_BUY_COSTS.get(score, 0)

    def get_total_cost(self) -> int:
        """Get total points spent."""
        return sum(self.get_cost(score) for score in self.scores.values())

    def can_increase(self, ability: AbilityScore) -> bool:
        """Check if an ability can be increased."""
        current = self.scores[ability]
        if current >= POINT_BUY_MAX:
            return False
        cost_diff = self.get_cost(current + 1) - self.get_cost(current)
        return cost_diff <= self.points_remaining

    def can_decrease(self, ability: AbilityScore) -> bool:
        """Check if an ability can be decreased."""
        return self.scores[ability] > POINT_BUY_MIN

    def increase(self, ability: AbilityScore) -> bool:
        """Increase an ability score. Returns True if successful."""
        if not self.can_increase(ability):
            return False
        current = self.scores[ability]
        cost_diff = self.get_cost(current + 1) - self.get_cost(current)
        self.scores[ability] = current + 1
        self.points_remaining -= cost_diff
        return True

    def decrease(self, ability: AbilityScore) -> bool:
        """Decrease an ability score. Returns True if successful."""
        if not self.can_decrease(ability):
            return False
        current = self.scores[ability]
        cost_diff = self.get_cost(current) - self.get_cost(current - 1)
        self.scores[ability] = current - 1
        self.points_remaining += cost_diff
        return True

    def to_ability_scores(self) -> AbilityScores:
        """Convert to AbilityScores model."""
        return AbilityScores(
            strength=self.scores[AbilityScore.STRENGTH],
            dexterity=self.scores[AbilityScore.DEXTERITY],
            constitution=self.scores[AbilityScore.CONSTITUTION],
            intelligence=self.scores[AbilityScore.INTELLIGENCE],
            wisdom=self.scores[AbilityScore.WISDOM],
            charisma=self.scores[AbilityScore.CHARISMA],
        )


@dataclass
class CharacterCreationState:
    """State for character creation wizard."""

    user_id: int
    campaign_id: str
    step: int = 0

    # Character data being built
    name: Optional[str] = None
    race_index: Optional[str] = None
    subrace_index: Optional[str] = None
    class_index: Optional[str] = None
    background_index: Optional[str] = None

    # Ability scores
    ability_method: Optional[AbilityScoreMethod] = None
    ability_rolls: Optional[AbilityScoreRolls] = None
    point_buy_state: Optional[PointBuyState] = None
    ability_assignments: Optional[dict[AbilityScore, int]] = None
    final_abilities: Optional[AbilityScores] = None

    # Proficiency choices
    skill_choices: list[Skill] = field(default_factory=list)

    # Equipment choices (indices into equipment options)
    equipment_choices: list[int] = field(default_factory=list)

    def reset(self):
        """Reset the creation state."""
        self.step = 0
        self.name = None
        self.race_index = None
        self.subrace_index = None
        self.class_index = None
        self.background_index = None
        self.ability_method = None
        self.ability_rolls = None
        self.point_buy_state = None
        self.ability_assignments = None
        self.final_abilities = None
        self.skill_choices = []
        self.equipment_choices = []


class CharacterCreator:
    """Handles character creation logic."""

    def __init__(self):
        self.roller = get_roller()
        self.srd = get_srd()

    def roll_ability_scores(self) -> AbilityScoreRolls:
        """Roll 6 ability scores using 4d6 drop lowest."""
        rolls = self.roller.roll_ability_scores_4d6()
        totals = [r.total for r in rolls]
        return AbilityScoreRolls(rolls=rolls, totals=totals)

    def get_standard_array(self) -> list[int]:
        """Get the standard array values."""
        return STANDARD_ARRAY.copy()

    def create_point_buy_state(self) -> PointBuyState:
        """Create a new point buy state."""
        return PointBuyState()

    def apply_racial_bonuses(
        self, base_scores: AbilityScores, race_index: str, subrace_index: Optional[str] = None
    ) -> AbilityScores:
        """Apply racial ability score bonuses."""
        race_data = self.srd.get_race(race_index)
        if not race_data:
            return base_scores

        # Start with base scores as dict for modification
        scores = {
            AbilityScore.STRENGTH: base_scores.strength,
            AbilityScore.DEXTERITY: base_scores.dexterity,
            AbilityScore.CONSTITUTION: base_scores.constitution,
            AbilityScore.INTELLIGENCE: base_scores.intelligence,
            AbilityScore.WISDOM: base_scores.wisdom,
            AbilityScore.CHARISMA: base_scores.charisma,
        }

        # Apply race bonuses
        for bonus in race_data.get("ability_bonuses", []):
            ability_index = bonus.get("ability_score", {}).get("index", "")
            ability_map = {
                "str": AbilityScore.STRENGTH,
                "dex": AbilityScore.DEXTERITY,
                "con": AbilityScore.CONSTITUTION,
                "int": AbilityScore.INTELLIGENCE,
                "wis": AbilityScore.WISDOM,
                "cha": AbilityScore.CHARISMA,
            }
            if ability_index in ability_map:
                scores[ability_map[ability_index]] += bonus.get("bonus", 0)

        # Apply subrace bonuses if applicable
        if subrace_index:
            subrace_data = self.srd.get_subrace(subrace_index)
            if subrace_data:
                for bonus in subrace_data.get("ability_bonuses", []):
                    ability_index = bonus.get("ability_score", {}).get("index", "")
                    ability_map = {
                        "str": AbilityScore.STRENGTH,
                        "dex": AbilityScore.DEXTERITY,
                        "con": AbilityScore.CONSTITUTION,
                        "int": AbilityScore.INTELLIGENCE,
                        "wis": AbilityScore.WISDOM,
                        "cha": AbilityScore.CHARISMA,
                    }
                    if ability_index in ability_map:
                        scores[ability_map[ability_index]] += bonus.get("bonus", 0)

        return AbilityScores(
            strength=min(20, scores[AbilityScore.STRENGTH]),
            dexterity=min(20, scores[AbilityScore.DEXTERITY]),
            constitution=min(20, scores[AbilityScore.CONSTITUTION]),
            intelligence=min(20, scores[AbilityScore.INTELLIGENCE]),
            wisdom=min(20, scores[AbilityScore.WISDOM]),
            charisma=min(20, scores[AbilityScore.CHARISMA]),
        )

    def calculate_starting_hp(self, class_index: str, con_modifier: int) -> int:
        """Calculate starting HP for a level 1 character."""
        class_data = self.srd.get_class(class_index)
        if not class_data:
            return 8 + con_modifier  # Default to d8

        hit_die = class_data.get("hit_die", 8)
        return max(1, hit_die + con_modifier)

    def get_hit_die(self, class_index: str) -> int:
        """Get the hit die size for a class."""
        class_data = self.srd.get_class(class_index)
        if not class_data:
            return 8
        return class_data.get("hit_die", 8)

    def get_saving_throw_proficiencies(self, class_index: str) -> list[AbilityScore]:
        """Get saving throw proficiencies for a class."""
        class_data = self.srd.get_class(class_index)
        if not class_data:
            return []

        saves = []
        for prof in class_data.get("saving_throws", []):
            ability_map = {
                "str": AbilityScore.STRENGTH,
                "dex": AbilityScore.DEXTERITY,
                "con": AbilityScore.CONSTITUTION,
                "int": AbilityScore.INTELLIGENCE,
                "wis": AbilityScore.WISDOM,
                "cha": AbilityScore.CHARISMA,
            }
            index = prof.get("index", "")
            if index in ability_map:
                saves.append(ability_map[index])
        return saves

    def get_skill_choices(self, class_index: str) -> tuple[list[Skill], int]:
        """Get available skill choices and number to choose for a class."""
        class_data = self.srd.get_class(class_index)
        if not class_data:
            return [], 0

        prof_choices = class_data.get("proficiency_choices", [])
        for choice in prof_choices:
            # Find skill proficiency choices
            options = choice.get("from", {}).get("options", [])
            if options and "skill" in options[0].get("item", {}).get("index", ""):
                skill_indices = [
                    opt.get("item", {}).get("index", "").replace("skill-", "")
                    for opt in options
                ]
                # Map to Skill enum
                skill_map = {s.value: s for s in Skill}
                available_skills = [
                    skill_map[idx] for idx in skill_indices if idx in skill_map
                ]
                choose_count = choice.get("choose", 2)
                return available_skills, choose_count

        return [], 0

    def get_spellcasting_ability(self, class_index: str) -> Optional[AbilityScore]:
        """Get the spellcasting ability for a class."""
        class_data = self.srd.get_class(class_index)
        if not class_data:
            return None

        spellcasting = class_data.get("spellcasting", {})
        ability_index = spellcasting.get("spellcasting_ability", {}).get("index", "")

        ability_map = {
            "int": AbilityScore.INTELLIGENCE,
            "wis": AbilityScore.WISDOM,
            "cha": AbilityScore.CHARISMA,
        }
        return ability_map.get(ability_index)

    def calculate_armor_class(self, abilities: AbilityScores, class_index: str) -> int:
        """Calculate base armor class (unarmored)."""
        # Base unarmored AC = 10 + DEX modifier
        base_ac = 10 + abilities.dex_mod

        # Some classes have special unarmored AC
        if class_index == "barbarian":
            # Unarmored Defense: 10 + DEX + CON
            base_ac = 10 + abilities.dex_mod + abilities.con_mod
        elif class_index == "monk":
            # Unarmored Defense: 10 + DEX + WIS
            base_ac = 10 + abilities.dex_mod + abilities.wis_mod

        return base_ac

    def get_speed(self, race_index: str) -> int:
        """Get base walking speed for a race."""
        race_data = self.srd.get_race(race_index)
        if not race_data:
            return 30
        return race_data.get("speed", 30)

    def build_character(self, state: CharacterCreationState) -> Character:
        """Build a Character from the creation state."""
        if not all([state.name, state.race_index, state.class_index, state.final_abilities]):
            raise ValueError("Incomplete character creation state")

        abilities = state.final_abilities
        class_index = state.class_index
        race_index = state.race_index

        # Calculate HP
        hit_die = self.get_hit_die(class_index)
        starting_hp = self.calculate_starting_hp(class_index, abilities.con_mod)

        # Get proficiencies
        saving_throw_profs = self.get_saving_throw_proficiencies(class_index)
        spellcasting_ability = self.get_spellcasting_ability(class_index)

        # Calculate AC and speed
        ac = self.calculate_armor_class(abilities, class_index)
        speed = self.get_speed(race_index)

        return Character(
            id=str(uuid.uuid4()),
            discord_user_id=state.user_id,
            campaign_id=state.campaign_id,
            name=state.name,
            race_index=race_index,
            class_index=class_index,
            subclass_index=None,  # Subclasses come at level 3 typically
            level=1,
            experience=0,
            background_index=state.background_index,
            abilities=abilities,
            armor_class=ac,
            speed=speed,
            initiative_bonus=abilities.dex_mod,
            hp=HitPoints(maximum=starting_hp, current=starting_hp, temporary=0),
            hit_dice=HitDice(die_type=hit_die, total=1, remaining=1),
            death_saves=DeathSaves(),
            spellcasting_ability=spellcasting_ability,
            spell_slots=SpellSlots(),  # Will be populated based on class
            saving_throw_proficiencies=saving_throw_profs,
            skill_proficiencies=state.skill_choices,
        )


# Global character creator instance
_creator: Optional[CharacterCreator] = None


def get_creator() -> CharacterCreator:
    """Get the global character creator instance."""
    global _creator
    if _creator is None:
        _creator = CharacterCreator()
    return _creator
