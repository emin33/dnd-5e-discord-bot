"""Single narration path: NarrationSpec (data) + NarrationStrategy (one code path).

REFACTOR_PLAN.md Step 2: the orchestrator's three narration paths
(``_narrate_mechanical_result`` / ``_narrate_action`` / ``_narrate_outcome``)
shared one skeleton — tier-client selection → BrainContext rebuild →
bookend/basic message build → per-path prompt → tool reminder → chat →
prose+effects extraction → tool followup — but hand-copied it with drifted
BrainContext field lists (AUDIT_QUALITY_2026_06_09, Duplication P1). Here the
skeleton exists ONCE and everything that legitimately varies per path is DATA
on :class:`NarrationSpec`.

Context-field policy: the strategy derives the narrator's context via
``dataclasses.replace(context, ...)``, overriding only the per-turn
actor/action — so every field the upstream pipeline computed is carried and
a field can no longer silently drift out of one path's hand-copied rebuild.
Nothing in the spec selects context fields; if a future path genuinely must
blind the narrator to a field, add an explicit spec knob then (data, not a
re-typed constructor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Optional

import structlog

from .brains.base import Brain, BrainContext
from .effects import ProposedEffect
from .narrator_tools import tool_calls_to_effects

logger = structlog.get_logger()

# Anti-repetition penalties for narrator calls (research: 0.3-0.8 / 0.2-0.6)
NARRATOR_FREQUENCY_PENALTY = 0.4  # Penalize tokens proportional to frequency
NARRATOR_PRESENCE_PENALTY = 0.3   # Penalize any already-used token

# Prose budget shared by every narration turn (streaming and not).
NARRATOR_MAX_TOKENS = 1500

OnToken = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class NarrationSpec:
    """Describes one narration turn as data.

    A call site builds a spec — its per-path prompt text, player_action
    decoration, and output policy — and hands it to
    :meth:`NarrationStrategy.run`. No call site owns prompt-assembly code.
    """

    # Raw player action: drives tier selection and telemetry/log slices.
    action: str
    player_name: str

    # What the narrator sees as the acting player's input — the raw action
    # plus any per-path decoration ("[NARRATIVE DIRECTION: …]" for plain
    # actions, "[RESOLUTION: …]" for roll outcomes, undecorated for
    # mechanical results whose outcome rides in the prompt instead).
    player_action: str

    # The per-path prompt appended after the built messages.
    # role "user": the mechanical-outcome prompt; role "system": the
    # "###INSTRUCTION###" directive of the action/outcome paths.
    prompt: str
    prompt_role: str  # "user" | "system"

    # Delivery: only the plain-action path streams (intent — chat_stream
    # carries no tools kwargs, so tool recovery on a streamed turn rides
    # entirely on the followup leg; see the streaming pin test).
    allow_streaming: bool = False

    # Output handling when the narrator returns empty prose:
    # - continue_on_empty_prose=False (action/outcome): bail with the
    #   fallback prose and NO effects.
    # - continue_on_empty_prose=True (mechanical result): substitute the
    #   fallback (the mech narrative_hint) and keep going — the tool
    #   followup still runs against it.
    empty_prose_fallback: str = ""
    empty_prose_warn_event: Optional[str] = None
    continue_on_empty_prose: bool = False


class NarrationStrategy:
    """The single narration code path; consumes a :class:`NarrationSpec`.

    Collaborators are injected as callables so the orchestrator binds its own
    seams — tier selection via the Step-0 ``_narrator_client_factory`` (which
    also keeps ``narrator.client`` in sync), the tool reminder that reads the
    live session, the telemetry-recording prose/effects extractor — and unit
    tests bind fakes.
    """

    def __init__(
        self,
        *,
        get_narrator: Callable[[], Brain],
        select_client: Callable[[str, Any, BrainContext], Any],
        get_tools: Callable[[], list[dict]],
        append_tool_reminder: Callable[[list[dict]], None],
        extract_prose_and_effects: Callable[[Any, str], tuple[str, list[ProposedEffect]]],
        get_on_token: Callable[[], Optional[OnToken]],
    ) -> None:
        self._get_narrator = get_narrator
        self._select_client = select_client
        self._get_tools = get_tools
        self._append_tool_reminder = append_tool_reminder
        self._extract_prose_and_effects = extract_prose_and_effects
        self._get_on_token = get_on_token

    async def run(
        self,
        spec: NarrationSpec,
        context: BrainContext,
        triage: Any,
    ) -> tuple[str, list[ProposedEffect]]:
        """Run one narration turn; returns (prose, proposed_effects)."""
        # Tier-aware narrator client selection (Phase B). The injected
        # selector mutates narrator.client, so it must run before we read it.
        self._select_client(spec.action, triage, context)
        narrator = self._get_narrator()

        # Carry EVERY upstream context field; override only the per-turn
        # actor/action. replace() means a field cannot silently drift out.
        enhanced_context = replace(
            context,
            player_action=spec.player_action,
            player_name=spec.player_name,
        )

        # Bookend layout when world state is available (better grounding).
        if enhanced_context.world_state_yaml:
            messages = narrator._build_bookend_messages(enhanced_context)
        else:
            messages = narrator._build_messages(enhanced_context)

        messages.append({"role": spec.prompt_role, "content": spec.prompt})
        self._append_tool_reminder(messages)

        # Stream when the spec allows it, a token callback is wired, and the
        # client supports it. The streaming call carries NO tools kwargs —
        # the followup leg is the tool recovery for streamed turns.
        on_token = self._get_on_token() if spec.allow_streaming else None
        if on_token and hasattr(narrator.client, "chat_stream"):
            logger.debug("narrator_streaming_enabled")
            response = await narrator.client.chat_stream(
                messages=messages,
                temperature=narrator.temperature,
                max_tokens=NARRATOR_MAX_TOKENS,
                on_token=on_token,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
            )
        else:
            response = await narrator.client.chat(
                messages=messages,
                temperature=narrator.temperature,
                max_tokens=NARRATOR_MAX_TOKENS,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                tools=self._get_tools(),
                tool_choice="auto",
            )

        prose, proposed_effects = self._extract_prose_and_effects(response, spec.action)

        if not prose:
            if spec.empty_prose_warn_event:
                logger.warning(spec.empty_prose_warn_event, action=spec.action[:50])
            if not spec.continue_on_empty_prose:
                return spec.empty_prose_fallback, []
            prose = spec.empty_prose_fallback

        # Two-turn followup: if the narrator didn't call tools, ask again.
        if not proposed_effects and prose:
            proposed_effects = await self._tool_followup(prose, messages)

        # If prose seems truncated (ends mid-sentence), add ellipsis.
        if prose and prose[-1] not in '.!?"\'':
            prose += "..."

        return prose, proposed_effects

    async def _tool_followup(
        self,
        prose: str,
        messages: list[dict],
    ) -> list[ProposedEffect]:
        """Second pass: force tool calls after narration.

        Audit #20: previously this built a fresh 2-message prompt with just
        the prose, throwing away the roster, world-state YAML, and `[id: ...]`
        tags from the original messages. The model then couldn't resolve any
        roster IDs and would invent new NPCs instead of using `ref_entity`.

        Now we reuse the full original message stack, append the assistant's
        prose as an assistant turn, and add a user turn instructing the model
        to declare tool calls. This preserves all the entity context.
        """
        # Reuse the original messages — they contain the system prompt with
        # roster IDs, world state YAML, and entity context the model needs.
        followup_messages = list(messages) + [
            {"role": "assistant", "content": prose[:2000]},
            {
                "role": "user",
                "content": (
                    "Now call a tool for everything you narrated above, using only "
                    "the tools available to you:\n"
                    "- ref_entity for each roster entity you referenced (use the roster IDs)\n"
                    "- add_npc / spawn_object for any new NPC or object you introduced\n"
                    "- update_player for any player damage, healing, loot, currency, or condition change\n"
                    "- change_location if the party moved; start_combat if a fight began\n"
                    "Do NOT respond with prose — only tool calls."
                ),
            },
        ]

        try:
            response = await self._get_narrator().client.chat(
                messages=followup_messages,
                temperature=0,
                max_tokens=500,
                think=False,
                # Tier-aware (audit #2/N2): the no-tool fallback previously
                # hardcoded CORE, so on core_plus/full profiles the streaming
                # path could never recover update_player/change_location/
                # start_combat. Use the profile's tier like the primary calls do;
                # core-tier gaps are still backfilled by the state/entity extractors.
                tools=self._get_tools(),
                tool_choice="required",
            )

            if response.tool_calls:
                effects = tool_calls_to_effects(response.tool_calls)
                logger.info(
                    "narrator_tool_followup",
                    tool_count=len(response.tool_calls),
                    effects_count=len(effects),
                )
                return effects
        except Exception as e:
            logger.warning("narrator_tool_followup_failed", error=str(e), exc_info=True)

        return []
