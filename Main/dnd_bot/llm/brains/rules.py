"""Rules Brain - Mechanical authority with tool-based state changes."""

from dataclasses import dataclass
from typing import Any, Optional
import json

from ..client import OllamaClient, get_llm_client
from .base import Brain, BrainContext, BrainResult
from ...config import get_settings


RULES_SYSTEM_PROMPT = """You are the D&D 5e rules arbiter. Your ONLY role is mechanical resolution - determining what rules apply and executing them through tools.

## CRITICAL REQUIREMENTS:

1. **ALWAYS use tools for random outcomes** - NEVER generate dice results yourself. Call roll_dice() for ANY roll.

2. **ALWAYS use tools for state changes** - Use update_hp(), apply_condition(), etc. for ALL game state modifications.

3. **Return structured outcomes** - Your output should be a JSON object describing the mechanical result for the Narrator.

4. **Be deterministic** - Given the same situation, apply the same rules consistently.

## PROCESS FOR RESOLVING ACTIONS:

1. Identify what D&D 5e rules apply to the attempted action
2. Determine required rolls and modifiers from character data provided
3. Call roll_dice() with appropriate notation (e.g., "1d20+5")
4. Compare results to target numbers (AC, DC, etc.)
5. Call state-mutation tools as needed (update_hp, apply_condition, etc.)
6. Return structured JSON outcome for the Narrator

## OUTPUT FORMAT:

After resolving mechanics, output a JSON block:
```json
{
    "action_type": "attack|spell|check|save|other",
    "success": true|false,
    "description": "Brief description of what happened mechanically",
    "details": {
        "roll": 17,
        "total": 22,
        "target": 15,
        "damage": 12,
        "damage_type": "slashing",
        "critical": false,
        "conditions_applied": [],
        "resources_used": []
    }
}
```

## D&D 5E RULES REFERENCE:

- Attack Roll: 1d20 + ability mod + proficiency (if proficient) + bonuses vs AC
- Saving Throw: 1d20 + ability mod + proficiency (if proficient) vs DC
- Ability Check: 1d20 + ability mod + proficiency (if proficient in skill)
- Advantage: Roll 2d20, take higher
- Disadvantage: Roll 2d20, take lower
- Critical Hit (natural 20): Auto-hit, double all damage dice
- Critical Miss (natural 1): Auto-miss
- Spell Save DC: 8 + proficiency + spellcasting ability mod
- Concentration Check: DC = max(10, damage/2)

## IMPORTANT:

You do NOT narrate or describe - that's the Narrator's job. You just determine and report the mechanical outcome."""


# Tool definitions for Ollama function calling
RULES_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "roll_dice",
            "description": "Roll dice using standard D&D notation. ALWAYS use this for any random outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notation": {
                        "type": "string",
                        "description": "Dice notation like '1d20+5', '2d6', '8d6', '4d6kh3'"
                    },
                    "advantage": {
                        "type": "boolean",
                        "description": "Roll 2d20, take higher (d20 only)"
                    },
                    "disadvantage": {
                        "type": "boolean",
                        "description": "Roll 2d20, take lower (d20 only)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "What this roll is for (attack, damage, save, etc.)"
                    }
                },
                "required": ["notation", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_hp",
            "description": "Modify a creature's HP. Use negative delta for damage, positive for healing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "string",
                        "description": "ID of the character or combatant"
                    },
                    "delta": {
                        "type": "integer",
                        "description": "HP change (negative for damage, positive for healing)"
                    },
                    "damage_type": {
                        "type": "string",
                        "description": "Type of damage (slashing, fire, etc.) for resistance checks"
                    },
                    "source": {
                        "type": "string",
                        "description": "What caused this HP change"
                    }
                },
                "required": ["target_id", "delta", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_condition",
            "description": "Apply a D&D 5e condition to a creature",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target"
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["blinded", "charmed", "deafened", "frightened",
                                 "grappled", "incapacitated", "invisible", "paralyzed",
                                 "petrified", "poisoned", "prone", "restrained",
                                 "stunned", "unconscious", "exhaustion"],
                        "description": "The condition to apply"
                    },
                    "source": {
                        "type": "string",
                        "description": "What caused this condition"
                    },
                    "duration_rounds": {
                        "type": "integer",
                        "description": "How many rounds the condition lasts (null for indefinite)"
                    }
                },
                "required": ["target_id", "condition", "source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_condition",
            "description": "Remove a condition from a creature",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "string",
                        "description": "ID of the target"
                    },
                    "condition": {
                        "type": "string",
                        "description": "The condition to remove"
                    }
                },
                "required": ["target_id", "condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "expend_spell_slot",
            "description": "Expend a spell slot when casting a leveled spell",
            "parameters": {
                "type": "object",
                "properties": {
                    "caster_id": {
                        "type": "string",
                        "description": "ID of the caster"
                    },
                    "slot_level": {
                        "type": "integer",
                        "description": "Level of the spell slot to expend"
                    }
                },
                "required": ["caster_id", "slot_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_rule",
            "description": "Look up a D&D 5e rule from the SRD when uncertain",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The rule question or topic to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "use_action",
            "description": "Mark an action resource as used (action, bonus action, reaction)",
            "parameters": {
                "type": "object",
                "properties": {
                    "combatant_id": {
                        "type": "string",
                        "description": "ID of the combatant"
                    },
                    "action_type": {
                        "type": "string",
                        "enum": ["action", "bonus_action", "reaction"],
                        "description": "Type of action to use"
                    }
                },
                "required": ["combatant_id", "action_type"]
            }
        }
    },
    # Commerce and Inventory Tools
    {
        "type": "function",
        "function": {
            "name": "purchase_item",
            "description": "Purchase an item from a vendor. Checks if character has enough gold, deducts the cost, and adds the item to inventory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "buyer_id": {
                        "type": "string",
                        "description": "ID or name of the character buying the item"
                    },
                    "item_index": {
                        "type": "string",
                        "description": "SRD item index (e.g., 'longsword', 'chain-mail', 'potion-of-healing')"
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Display name of the item"
                    },
                    "cost_gold": {
                        "type": "integer",
                        "description": "Cost in gold pieces"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to purchase (default 1)"
                    }
                },
                "required": ["buyer_id", "item_index", "item_name", "cost_gold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sell_item",
            "description": "Sell an item to a vendor. Removes item from inventory and adds gold (typically half item value).",
            "parameters": {
                "type": "object",
                "properties": {
                    "seller_id": {
                        "type": "string",
                        "description": "ID or name of the character selling"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "ID of the inventory item to sell"
                    },
                    "sale_price_gold": {
                        "type": "integer",
                        "description": "Gold received from sale"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number to sell (default 1)"
                    }
                },
                "required": ["seller_id", "item_id", "sale_price_gold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_gold",
            "description": "Add or remove gold from a character (rewards, fines, quest rewards, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "ID or name of the character"
                    },
                    "delta": {
                        "type": "integer",
                        "description": "Gold change (positive to add, negative to remove)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the gold change"
                    }
                },
                "required": ["character_id", "delta", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": "Add an item to a character's inventory (loot, gifts, quest rewards)",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "ID or name of the receiving character"
                    },
                    "item_index": {
                        "type": "string",
                        "description": "SRD item index"
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Display name of the item"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items (default 1)"
                    },
                    "source": {
                        "type": "string",
                        "description": "Where the item came from (loot, reward, gift)"
                    }
                },
                "required": ["character_id", "item_index", "item_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_item",
            "description": "Remove an item from inventory (lost, destroyed, consumed, given away)",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "ID or name of the character"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "ID of the inventory item"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number to remove (default 1)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for removal"
                    }
                },
                "required": ["character_id", "item_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "equip_item",
            "description": "Equip an item from inventory",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "ID or name of the character"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "ID of the inventory item to equip"
                    }
                },
                "required": ["character_id", "item_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unequip_item",
            "description": "Unequip an item",
            "parameters": {
                "type": "object",
                "properties": {
                    "character_id": {
                        "type": "string",
                        "description": "ID or name of the character"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "ID of the inventory item to unequip"
                    }
                },
                "required": ["character_id", "item_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_item",
            "description": "Transfer an item from one character to another (give, trade)",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_character_id": {
                        "type": "string",
                        "description": "ID or name of the giving character"
                    },
                    "to_character_id": {
                        "type": "string",
                        "description": "ID or name of the receiving character"
                    },
                    "item_id": {
                        "type": "string",
                        "description": "ID of the inventory item to transfer"
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number to transfer (default 1)"
                    }
                },
                "required": ["from_character_id", "to_character_id", "item_id"]
            }
        }
    }
]


@dataclass
class MechanicalResult:
    """Structured result from rules resolution."""

    action_type: str
    success: bool
    description: str
    details: dict

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "success": self.success,
            "description": self.description,
            "details": self.details,
        }


class RulesBrain(Brain):
    """
    The mechanical authority brain.

    Handles:
    - Determining what rules apply
    - Calling tools to roll dice and modify state
    - Returning structured mechanical outcomes

    Does NOT handle:
    - Narration or description
    - Creative storytelling
    - NPC roleplay
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        temperature: Optional[float] = None,
    ):
        settings = get_settings()
        super().__init__(
            client=client or get_llm_client(),
            temperature=temperature or settings.rules_temperature,
            system_prompt=RULES_SYSTEM_PROMPT,
        )
        self.tools = RULES_TOOLS

    async def process(self, context: BrainContext) -> BrainResult:
        """Resolve mechanics for an action."""
        messages = self._build_messages(context)

        # Add character stats if available
        if context.character_stats:
            messages.append({
                "role": "system",
                "content": f"## Character Stats\n```json\n{json.dumps(context.character_stats, indent=2)}\n```",
            })

        # Add explicit instruction for reliable output
        messages.append({
            "role": "system",
            "content": (
                "###YOUR TASK###\n"
                "Analyze the action above and determine what D&D 5e rules apply.\n"
                "1. If dice rolls are needed, call the appropriate tool(s)\n"
                "2. If state changes occur, call the appropriate tool(s)\n"
                "3. Return a JSON object describing the mechanical outcome\n\n"
                "###OUTPUT FORMAT###\n"
                "After any tool calls, output a JSON block with the result."
            ),
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            tools=self.tools,
            max_tokens=500,
            think=False,  # Disable Qwen3 thinking mode
        )

        return self._parse_response(response)

    async def resolve_attack(
        self,
        attacker_stats: dict,
        target_ac: int,
        weapon: dict,
        advantage: bool = False,
        disadvantage: bool = False,
        context: Optional[BrainContext] = None,
    ) -> BrainResult:
        """Resolve an attack action."""
        attack_context = BrainContext(
            player_action=f"Attack with {weapon.get('name', 'weapon')}",
            player_name=attacker_stats.get("name", "Attacker"),
            character_stats=attacker_stats,
            in_combat=True,
        )
        if context:
            attack_context.current_scene = context.current_scene
            attack_context.initiative_order = context.initiative_order

        messages = self._build_messages(attack_context)
        messages.append({
            "role": "system",
            "content": (
                f"Resolve this attack:\n"
                f"- Target AC: {target_ac}\n"
                f"- Advantage: {advantage}\n"
                f"- Disadvantage: {disadvantage}\n"
                f"- Weapon: {json.dumps(weapon)}\n\n"
                "First call roll_dice for the attack roll, then if it hits, roll damage."
            ),
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            tools=self.tools,
            think=False,  # Disable Qwen3 thinking mode
        )

        return self._parse_response(response)

    async def resolve_save(
        self,
        target_stats: dict,
        save_ability: str,
        dc: int,
        context: Optional[BrainContext] = None,
    ) -> BrainResult:
        """Resolve a saving throw."""
        save_context = BrainContext(
            player_action=f"Make a {save_ability.upper()} saving throw vs DC {dc}",
            player_name=target_stats.get("name", "Target"),
            character_stats=target_stats,
        )

        messages = self._build_messages(save_context)
        messages.append({
            "role": "system",
            "content": f"Resolve this {save_ability} saving throw against DC {dc}.",
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            tools=self.tools,
            think=False,  # Disable Qwen3 thinking mode
        )

        return self._parse_response(response)

    async def resolve_check(
        self,
        character_stats: dict,
        skill_or_ability: str,
        dc: int,
        advantage: bool = False,
        disadvantage: bool = False,
        context: Optional[BrainContext] = None,
    ) -> BrainResult:
        """Resolve an ability check."""
        check_context = BrainContext(
            player_action=f"Make a {skill_or_ability} check vs DC {dc}",
            player_name=character_stats.get("name", "Character"),
            character_stats=character_stats,
        )

        messages = self._build_messages(check_context)
        messages.append({
            "role": "system",
            "content": (
                f"Resolve this {skill_or_ability} check against DC {dc}.\n"
                f"Advantage: {advantage}, Disadvantage: {disadvantage}"
            ),
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            tools=self.tools,
            think=False,  # Disable Qwen3 thinking mode
        )

        return self._parse_response(response)

    def parse_mechanical_result(self, content: str) -> Optional[MechanicalResult]:
        """Parse a mechanical result from the LLM's response."""
        try:
            # Try to find JSON in the response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                return MechanicalResult(
                    action_type=data.get("action_type", "other"),
                    success=data.get("success", False),
                    description=data.get("description", ""),
                    details=data.get("details", {}),
                )
        except json.JSONDecodeError:
            pass
        return None


# Global rules brain instance
_rules: Optional[RulesBrain] = None


def get_rules_brain() -> RulesBrain:
    """Get the global rules brain."""
    global _rules
    if _rules is None:
        _rules = RulesBrain()
    return _rules
