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

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from .brains.base import Brain, BrainContext
from .continuity import ContinuityViolation, NarrativeGovernance
from .effect_obligations import (
    EffectObligationSet,
    infer_effect_coherence_obligations,
    infer_effect_obligations,
    infer_narration_effect_obligations,
)
from .effects import EffectType, EffectValidator, ProposedEffect
from .narrator_tools import tool_calls_to_effects

if TYPE_CHECKING:
    from .state_followup import StateFollowupSignal

logger = structlog.get_logger()

# Anti-repetition penalties for narrator calls (research: 0.3-0.8 / 0.2-0.6)
NARRATOR_FREQUENCY_PENALTY = 0.4  # Penalize tokens proportional to frequency
NARRATOR_PRESENCE_PENALTY = 0.3   # Penalize any already-used token

# Prose budget shared by every narration turn (streaming and not).
NARRATOR_MAX_TOKENS = 1500

# Rough token overhead of the serialized tool schemas + tool-call plumbing
# that rides alongside the messages (full tier measures ~5.1k tokens).
TOOL_SCHEMA_TOKEN_OVERHEAD = 5000

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
    empty_prose_warn_event: str | None = None
    continue_on_empty_prose: bool = False

    # Tool surface. The orchestrator paths expose the narrator tools and
    # run the followup recovery leg; combat-outcome narration has NO tool
    # surface (combat effects are owned by the combat engine, not narrator
    # tool calls), so False also skips the tool reminder and the followup.
    enable_tools: bool = True

    # Pass-through to the client's ``think`` kwarg (None = provider
    # default, i.e. the kwarg is not sent). The combat-outcome path pins
    # think=False — Qwen3 thinking mode truncates its narration.
    think: bool | None = None


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
        get_on_token: Callable[[], OnToken | None],
        get_governance: Callable[[], NarrativeGovernance] | None = None,
    ) -> None:
        self._get_narrator = get_narrator
        self._select_client = select_client
        self._get_tools = get_tools
        self._append_tool_reminder = append_tool_reminder
        self._extract_prose_and_effects = extract_prose_and_effects
        self._get_on_token = get_on_token
        self._get_governance = get_governance
        self.last_diagnostics: dict[str, Any] = {}

    async def run(
        self,
        spec: NarrationSpec,
        context: BrainContext,
        triage: Any,
    ) -> tuple[str, list[ProposedEffect]]:
        """Run one narration turn; returns (prose, proposed_effects)."""
        self.last_diagnostics = {
            "streaming_buffered": False,
            "continuity_violations": 0,
            "continuity_repair_attempted": False,
            "continuity_repair_succeeded": False,
            "continuity_failed_closed": False,
            "primary_effects": 0,
            "primary_structural_errors": 0,
            "primary_structural_error_details": [],
            "tool_followup_attempted": False,
            "tool_followup_for_mutation": False,
            "tool_followup_skipped_memory_only": False,
            "tool_repair_attempted": False,
            "tool_followup_structural_errors": 0,
            "tool_followup_structural_error_details": [],
            "tool_repair_structural_errors": 0,
            "tool_repair_structural_error_details": [],
            "tool_invalid_effects_dropped": 0,
            "tool_policy_suppressed_effects": 0,
            "tool_unknown_roster_refs_dropped": 0,
            "tool_ref_alias_mismatches_removed": 0,
            "tool_repair_failed_closed": False,
            "tool_ref_deterministic_recoveries": [],
            "tool_duplicate_creations_collapsed": 0,
            "tool_followup_effects": 0,
            "effect_obligations": [],
            "effect_obligation_reasons": [],
            "effect_obligation_missing_initial": [],
            "effect_obligation_missing_final": [],
            "effect_obligation_repair_attempted": False,
            "effect_obligation_repair_succeeded": False,
            "effect_obligation_terminal_repair_attempted": False,
            "effect_obligation_terminal_repair_succeeded": False,
            "effect_obligation_terminal_structural_errors": 0,
            "resolved_outcome_contradictions": [],
            "resolved_outcome_repair_attempted": False,
            "resolved_outcome_repair_succeeded": False,
            "resolved_outcome_failed_closed": False,
            "final_effects": 0,
        }
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
        roster_refs = self._roster_refs_from_context(enhanced_context)

        # Bookend layout when world state is available (better grounding).
        if enhanced_context.world_state_yaml:
            messages = narrator.build_bookend_messages(enhanced_context)
        else:
            messages = narrator.build_messages(enhanced_context)

        obligations = infer_effect_obligations(spec.action)
        self.last_diagnostics["effect_obligations"] = sorted(
            effect_type.value for effect_type in obligations.required_types
        )
        self.last_diagnostics["effect_obligation_reasons"] = [
            item.reason for item in obligations.obligations
        ]
        if obligations.required_types:
            messages.append(
                {
                    "role": "system",
                    "content": obligations.primary_instruction(spec.action),
                }
            )
        freshness_hint = self._prose_freshness_hint(enhanced_context)
        if freshness_hint:
            messages.append({"role": "system", "content": freshness_hint})
        messages.append({"role": spec.prompt_role, "content": spec.prompt})
        if spec.enable_tools:
            self._append_tool_reminder(messages)
            roster_refs = self._merge_roster_refs(
                roster_refs,
                self._roster_refs_from_text("\n".join(
                    str(message.get("content") or "")
                    for message in messages
                )),
            )

        # Soft context-budget check (chars/4 ≈ tokens). Local Ollama
        # silently truncates the prompt HEAD (system persona +
        # <world_state>) on overflow — see llm/client.py num_ctx notes —
        # so warn while the budget is merely tight. Only clients that
        # declare num_ctx (OllamaClient) have a known hard cap; cloud
        # narrators (128k+ contexts, no num_ctx attribute) skip the check.
        num_ctx = getattr(narrator.client, "num_ctx", None)
        if num_ctx:
            est_tokens = sum(len(m.get("content") or "") for m in messages) // 4
            # Tool schemas only ride along when the spec sends tools — a
            # no-tool-surface turn (combat outcome) has ~5k more headroom.
            overhead = TOOL_SCHEMA_TOKEN_OVERHEAD if spec.enable_tools else 0
            token_budget = num_ctx - NARRATOR_MAX_TOKENS - overhead
            if est_tokens > token_budget:
                logger.warning(
                    "narration_context_near_cap",
                    estimated_prompt_tokens=est_tokens,
                    num_ctx=num_ctx,
                    token_budget=token_budget,
                )

        # think rides only when the spec sets it — None must mean "kwarg
        # not sent" so the orchestrator paths' pinned kwargs stay exact.
        think_kwargs: dict[str, Any] = (
            {"think": spec.think} if spec.think is not None else {}
        )

        governance = (
            self._get_governance()
            if self._get_governance is not None
            else NarrativeGovernance()
        )

        # Stream when the spec allows it, a token callback is wired, and the
        # client supports it. The streaming call carries NO tools kwargs —
        # the followup leg is the tool recovery for streamed turns.
        on_token = self._get_on_token() if spec.allow_streaming else None
        if on_token and governance.requires_buffering:
            # Once a token reaches Discord it cannot be recalled. Buffer only
            # when an immutable fact actually needs post-generation checking.
            logger.info(
                "narrator_stream_buffered_for_continuity",
                dead_npcs=list(governance.dead_names),
            )
            self.last_diagnostics["streaming_buffered"] = True
            on_token = None
        if on_token and hasattr(narrator.client, "chat_stream"):
            logger.debug("narrator_streaming_enabled")
            response = await narrator.client.chat_stream(
                messages=messages,
                temperature=narrator.temperature,
                max_tokens=NARRATOR_MAX_TOKENS,
                on_token=on_token,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                **think_kwargs,
            )
        else:
            tool_kwargs: dict[str, Any] = (
                {"tools": self._get_tools(), "tool_choice": "auto"}
                if spec.enable_tools
                else {}
            )
            response = await narrator.client.chat(
                messages=messages,
                temperature=narrator.temperature,
                max_tokens=NARRATOR_MAX_TOKENS,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                **tool_kwargs,
                **think_kwargs,
            )

        prose, proposed_effects = self._extract_prose_and_effects(response, spec.action)
        self.last_diagnostics["primary_effects"] = len(proposed_effects)

        if not prose:
            if spec.empty_prose_warn_event:
                logger.warning(spec.empty_prose_warn_event, action=spec.action[:50])
            if not spec.continue_on_empty_prose:
                return spec.empty_prose_fallback, []
            prose = spec.empty_prose_fallback

        # Discard both prose and tool proposals from a contradictory draft,
        # repair once against the original grounded stack, then fail closed.
        continuity_failed_closed = False
        violations = governance.validate(prose)
        self.last_diagnostics["continuity_violations"] = len(violations)
        if violations:
            self.last_diagnostics["continuity_repair_attempted"] = True
            logger.warning(
                "narrator_continuity_violation_repairing",
                rules=[v.rule for v in violations],
                entities=[v.entity_name for v in violations],
            )
            prose, proposed_effects, repair_succeeded = await self._repair_continuity(
                prose=prose,
                messages=messages,
                spec=spec,
                governance=governance,
                violations=violations,
            )
            if not repair_succeeded:
                continuity_failed_closed = True
            else:
                self.last_diagnostics["continuity_repair_succeeded"] = True
            remaining = governance.validate(prose)
            if remaining:
                logger.error(
                    "narrator_continuity_repair_rejected",
                    rules=[v.rule for v in remaining],
                    entities=[v.entity_name for v in remaining],
                )
                prose = governance.safe_fallback(remaining)
                proposed_effects = []
                continuity_failed_closed = True
            self.last_diagnostics["continuity_failed_closed"] = (
                continuity_failed_closed
            )

        # Some authoritative corrections are visible only after prose exists
        # (for example, the player claims an item is in hand and the narrator
        # grounds it in an NPC's possession). Fold those narrow obligations
        # into the action-derived contract before any tool-repair decision.
        obligations = obligations.merged(
            infer_narration_effect_obligations(spec.action, prose)
        )
        self.last_diagnostics["effect_obligations"] = sorted(
            effect_type.value for effect_type in obligations.required_types
        )
        self.last_diagnostics["effect_obligation_reasons"] = [
            item.reason for item in obligations.obligations
        ]

        # A narrow class of no-roll actions explicitly declares its outcome
        # (accepted transfers, completed travel, terminal object destruction).
        # If the narrator negated such an outcome, prose and tool calls must be
        # repaired together; a tool-only followup would otherwise commit state
        # that contradicts the visible story.
        initial_missing = obligations.missing_from(proposed_effects)
        self.last_diagnostics["effect_obligation_missing_initial"] = sorted(
            effect_type.value for effect_type in initial_missing
        )
        contradictions = obligations.contradiction_reasons(prose)
        self.last_diagnostics["resolved_outcome_contradictions"] = list(
            contradictions
        )
        resolved_outcome_failed_closed = False
        transfer_needs_combined_repair = bool(
            "item_transfer" in obligations.outcome_kinds and initial_missing
        )
        if (
            obligations.required_types
            and (contradictions or transfer_needs_combined_repair)
            and not continuity_failed_closed
        ):
            self.last_diagnostics["resolved_outcome_repair_attempted"] = True
            repaired_prose, repaired_effects, repaired = (
                await self._repair_resolved_outcome(
                    prose=prose,
                    messages=messages,
                    spec=spec,
                    governance=governance,
                    obligations=obligations,
                    contradictions=contradictions,
                )
            )
            if repaired:
                prose = repaired_prose
                proposed_effects = repaired_effects
                self.last_diagnostics["resolved_outcome_repair_succeeded"] = True
            else:
                prose = (
                    "*The declared outcome could not be resolved consistently; "
                    "no world-state change is committed.*"
                )
                proposed_effects = []
                resolved_outcome_failed_closed = True
                self.last_diagnostics["resolved_outcome_failed_closed"] = True

        # Tool recovery: absence and structurally invalid arguments both enter
        # a bounded deterministic repair workflow. DeepSeek's non-strict API does
        # not guarantee JSON-schema adherence, so `required` in the schema is
        # guidance rather than an execution-safety boundary. Contextual checks
        # still happen later in the orchestrator against the live scene.
        proposed_effects = self._normalize_grounded_ref_aliases(
            proposed_effects,
            prose,
        )
        proposed_effects, unknown_ref_drops = self._drop_unknown_roster_refs(
            proposed_effects,
            roster_refs,
        )
        if unknown_ref_drops:
            self.last_diagnostics["tool_invalid_effects_dropped"] += (
                unknown_ref_drops
            )
            self.last_diagnostics["tool_unknown_roster_refs_dropped"] += (
                unknown_ref_drops
            )
            recovered_refs = self._recover_roster_references(
                prose,
                roster_refs,
                proposed_effects,
            )
            for recovered in recovered_refs:
                if recovered not in proposed_effects:
                    proposed_effects.append(recovered)
            if recovered_refs:
                self.last_diagnostics["tool_ref_deterministic_recoveries"] = [
                    effect.ref_entity_id for effect in recovered_refs
                ]
        proposed_effects, alias_corrections, alias_drops = (
            self._reconcile_roster_ref_aliases(
                proposed_effects,
                prose,
                roster_refs,
            )
        )
        self.last_diagnostics["tool_ref_alias_mismatches_removed"] += (
            alias_corrections
        )
        self.last_diagnostics["tool_invalid_effects_dropped"] += alias_drops
        validation_errors = self._effect_errors(proposed_effects, prose, spec.action)
        self.last_diagnostics["primary_structural_errors"] = len(validation_errors)
        self.last_diagnostics["primary_structural_error_details"] = list(
            validation_errors
        )
        # Invalid proposals do not satisfy a semantic obligation. Otherwise a
        # malformed add_npc/update call can suppress the one repair pass that
        # has the required tool family available.
        obligation_candidates = self._valid_effects_for_prose(
            proposed_effects,
            prose,
            spec.action,
        )
        missing_obligations = obligations.missing_from(obligation_candidates)
        mutation_followup = bool(missing_obligations) or self._needs_mutation_followup(
            spec.action,
            prose,
            proposed_effects,
        )
        memory_only_recollection = self._is_memory_only_recollection(
            spec.action, prose
        )
        self.last_diagnostics["tool_followup_skipped_memory_only"] = bool(
            memory_only_recollection and not proposed_effects
        )
        self.last_diagnostics["effect_obligation_repair_attempted"] = bool(
            missing_obligations
        )
        self.last_diagnostics["tool_followup_for_mutation"] = mutation_followup
        if (
            spec.enable_tools
            and prose
            and not continuity_failed_closed
            and not resolved_outcome_failed_closed
            and (
                (not proposed_effects and not memory_only_recollection)
                or validation_errors
                or mutation_followup
            )
        ):
            proposed_effects = await self._tool_followup(
                prose,
                messages,
                validation_errors=validation_errors,
                mutation_recovery=mutation_followup,
                action=spec.action,
                required_effect_types=missing_obligations,
                existing_effects=proposed_effects,
                roster_refs=roster_refs,
            )

        obligations = obligations.merged(
            infer_effect_coherence_obligations(prose, proposed_effects)
        )
        self.last_diagnostics["effect_obligations"] = sorted(
            effect_type.value for effect_type in obligations.required_types
        )
        self.last_diagnostics["effect_obligation_reasons"] = [
            item.reason for item in obligations.obligations
        ]
        final_missing = obligations.missing_from(proposed_effects)
        if (
            final_missing
            and spec.enable_tools
            and prose
            and not continuity_failed_closed
            and not resolved_outcome_failed_closed
        ):
            proposed_effects = await self._repair_missing_effect_obligations(
                prose=prose,
                messages=messages,
                action=spec.action,
                missing_effect_types=final_missing,
                existing_effects=proposed_effects,
            )
            final_missing = obligations.missing_from(proposed_effects)
        self.last_diagnostics["effect_obligation_missing_final"] = sorted(
            effect_type.value for effect_type in final_missing
        )
        if obligations.required_types and not final_missing:
            self.last_diagnostics["effect_obligation_repair_succeeded"] = True
        elif final_missing:
            logger.error(
                "narrator_effect_obligations_unmet",
                required=sorted(item.value for item in obligations.required_types),
                missing=sorted(item.value for item in final_missing),
                action=spec.action[:160],
            )

        proposed_effects, collapsed = self._collapse_duplicate_creations(
            proposed_effects
        )
        self.last_diagnostics["tool_duplicate_creations_collapsed"] = collapsed

        # If prose seems truncated (ends mid-sentence), add ellipsis.
        if prose and prose[-1] not in '.!?"\'':
            prose += "..."

        self.last_diagnostics["final_effects"] = len(proposed_effects)
        return prose, proposed_effects

    async def _repair_missing_effect_obligations(
        self,
        *,
        prose: str,
        messages: list[dict],
        action: str,
        missing_effect_types: frozenset[EffectType],
        existing_effects: list[ProposedEffect],
    ) -> list[ProposedEffect]:
        """Make one terminal repair with only the missing tool families visible."""
        self.last_diagnostics[
            "effect_obligation_terminal_repair_attempted"
        ] = True
        missing_names = {effect_type.value for effect_type in missing_effect_types}
        narrowed_tools = [
            tool
            for tool in self._get_tools()
            if tool.get("function", {}).get("name") in missing_names
        ]
        if not narrowed_tools:
            logger.error(
                "narrator_effect_obligation_tool_unavailable",
                missing=sorted(missing_names),
            )
            return existing_effects

        repair_messages = list(messages) + [
            {"role": "assistant", "content": prose[:3000]},
            {
                "role": "user",
                "content": (
                    "FINAL TOOL REPAIR. The validated, resolved player action "
                    "and the visible narration require these missing effect "
                    f"families: {', '.join(sorted(missing_names))}. Return "
                    "ONLY the missing tool calls now. Use exact roster IDs and "
                    "exact item/entity names from the action, narration, and "
                    "roster. Include every required argument. Do not return "
                    "prose, repeat existing calls, or invent unrelated effects.\n"
                    f"Resolved player action: {action}"
                ),
            },
        ]
        try:
            response = await self._request_tool_followup(
                repair_messages,
                tools=narrowed_tools,
            )
            repair_effects = (
                tool_calls_to_effects(response.tool_calls)
                if response.tool_calls
                else []
            )
            repair_effects = self._normalize_grounded_ref_aliases(
                repair_effects,
                prose,
            )
            repair_effects = [
                effect
                for effect in repair_effects
                if effect.effect_type in missing_effect_types
            ]
            errors = self._effect_errors(repair_effects, prose, action)
            self.last_diagnostics[
                "effect_obligation_terminal_structural_errors"
            ] = len(errors)
            valid_repairs = self._valid_effects_for_prose(
                repair_effects, prose, action
            )
            self.last_diagnostics["tool_invalid_effects_dropped"] += (
                len(repair_effects) - len(valid_repairs)
            )
            merged = list(existing_effects)
            for effect in valid_repairs:
                if effect not in merged:
                    merged.append(effect)
            remaining = missing_effect_types - {
                effect.effect_type for effect in merged
            }
            if not remaining:
                self.last_diagnostics[
                    "effect_obligation_terminal_repair_succeeded"
                ] = True
            else:
                logger.error(
                    "narrator_terminal_effect_obligation_repair_unmet",
                    missing=sorted(item.value for item in remaining),
                    errors=errors,
                )
            return merged
        except Exception as exc:
            logger.warning(
                "narrator_terminal_effect_obligation_repair_failed",
                error=str(exc),
                exc_info=True,
            )
            return existing_effects

    async def _repair_resolved_outcome(
        self,
        *,
        prose: str,
        messages: list[dict],
        spec: NarrationSpec,
        governance: NarrativeGovernance,
        obligations: EffectObligationSet,
        contradictions: tuple[str, ...],
    ) -> tuple[str, list[ProposedEffect], bool]:
        """Repair prose that negated a validated, explicitly resolved action."""
        required = ", ".join(
            sorted(item.value for item in obligations.required_types)
        )
        issues = contradictions or (
            "the draft omitted one or both sides of the completed item transfer",
        )
        repair_messages = list(messages) + [
            {"role": "assistant", "content": prose[:6000]},
            {
                "role": "user",
                "content": (
                    "RESOLVED OUTCOME REPAIR REQUIRED. Your draft contradicted "
                    "the validated player action: "
                    + "; ".join(issues)
                    + ". Rewrite the complete visible narration so the outcome "
                    "in the player action occurs exactly as stated. Do not turn "
                    "it into a refusal, missing item, failed attempt, or alternate "
                    "event. Return the corrected prose AND the complete tool set. "
                    f"The required effect families are: {required}."
                ),
            },
        ]
        try:
            response = await self._get_narrator().client.chat(
                messages=repair_messages,
                temperature=0,
                max_tokens=NARRATOR_MAX_TOKENS,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                tools=self._get_tools(),
                tool_choice="auto",
                think=False,
            )
            repaired_prose, repaired_effects = self._extract_prose_and_effects(
                response, spec.action
            )
            if (
                repaired_prose
                and not governance.validate(repaired_prose)
                and not obligations.contradiction_reasons(repaired_prose)
            ):
                return repaired_prose, repaired_effects, True
        except Exception as exc:
            logger.warning(
                "narrator_resolved_outcome_repair_failed",
                error=str(exc),
                exc_info=True,
            )
        return prose, [], False

    async def _repair_continuity(
        self,
        *,
        prose: str,
        messages: list[dict],
        spec: NarrationSpec,
        governance: NarrativeGovernance,
        violations: list[ContinuityViolation],
    ) -> tuple[str, list[ProposedEffect], bool]:
        """Make one grounded, deterministic rewrite of contradictory prose."""
        repair_messages = list(messages) + [
            {"role": "assistant", "content": prose[:6000]},
            {"role": "user", "content": governance.repair_instruction(violations)},
        ]
        tool_kwargs: dict[str, Any] = (
            {"tools": self._get_tools(), "tool_choice": "auto"}
            if spec.enable_tools
            else {}
        )
        think_kwargs: dict[str, Any] = (
            {"think": spec.think} if spec.think is not None else {}
        )
        try:
            response = await self._get_narrator().client.chat(
                messages=repair_messages,
                temperature=0,
                max_tokens=NARRATOR_MAX_TOKENS,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                **tool_kwargs,
                **think_kwargs,
            )
            repaired_prose, repaired_effects = self._extract_prose_and_effects(
                response, spec.action
            )
            if repaired_prose:
                return repaired_prose, repaired_effects, True
        except Exception as exc:
            logger.warning(
                "narrator_continuity_repair_failed",
                error=str(exc),
                exc_info=True,
            )
        return governance.safe_fallback(violations), [], False

    @staticmethod
    def _roster_refs_from_context(
        context: BrainContext,
    ) -> list[tuple[str, str, tuple[str, ...]]]:
        """Extract authoritative ``(id, name, aliases)`` rows from YAML context."""
        found: dict[str, tuple[str, str, tuple[str, ...]]] = {}

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                entity_id = str(value.get("id") or value.get("node_id") or "").strip()
                name = str(value.get("name") or "").strip()
                entity_type = str(value.get("type") or value.get("entity_type") or "")
                if entity_id and name and entity_type not in {"player", "character"}:
                    aliases = tuple(
                        str(alias).strip()
                        for alias in (value.get("aliases") or [])
                        if str(alias).strip()
                    )
                    found.setdefault(entity_id, (entity_id, name, aliases))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        def slugify_label(value: str) -> str:
            slug = re.sub(r"[^a-z0-9\s-]", "", value.casefold().strip())
            slug = re.sub(r"\s+", "-", slug)
            return re.sub(r"-+", "-", slug).strip("-")

        for raw_yaml, is_world_state in (
            (context.world_state_yaml, True),
            (context.kg_context_yaml, False),
        ):
            if not raw_yaml:
                continue
            try:
                document = yaml.safe_load(raw_yaml)
                walk(document)
                if is_world_state and isinstance(document, dict):
                    # WorldState's current location and scene items use a
                    # compact scalar/list representation rather than entity
                    # dictionaries. They are still authoritative roster
                    # references and must survive the unknown-ID filter.
                    location = str(document.get("location") or "").strip()
                    location_id = slugify_label(location)
                    if location_id and location:
                        found.setdefault(
                            location_id,
                            (location_id, location, ()),
                        )
                    for raw_item in document.get("scene_items") or []:
                        item_label = str(raw_item or "").split(":", 1)[0].strip()
                        item_id = slugify_label(item_label)
                        if item_id and item_label:
                            found.setdefault(
                                item_id,
                                (item_id, item_label, ()),
                            )
            except Exception:
                continue
        # SceneEntityRegistry exposes additional live entities through its
        # authoritative ``[id: slug]`` roster. They may not yet have a world/KG
        # projection, so include them in the same deterministic reference set.
        for match in re.finditer(
            r"-\s+(?:\*\*)?([^\n\[]+?)(?:\*\*)?\s+"
            r"\[id:\s*([^\]]+)\]",
            context.current_scene or "",
            re.IGNORECASE,
        ):
            name = match.group(1).strip().strip("*")
            entity_id = match.group(2).strip()
            if entity_id and name:
                found.setdefault(entity_id, (entity_id, name, ()))
        return list(found.values())

    @staticmethod
    def _roster_refs_from_text(
        text: str,
    ) -> list[tuple[str, str, tuple[str, ...]]]:
        """Extract authoritative ``name [id: value]`` rows from prompts."""
        rows: list[tuple[str, str, tuple[str, ...]]] = []
        for match in re.finditer(
            r"-\s+(?:\*\*)?([^\n\[]+?)(?:\*\*)?\s+"
            r"\[id:\s*([^\]]+)\]",
            text or "",
            re.IGNORECASE,
        ):
            name = match.group(1).strip().strip("*")
            entity_id = match.group(2).strip()
            if entity_id and name:
                rows.append((entity_id, name, ()))
        return rows

    @staticmethod
    def _merge_roster_refs(
        *groups: list[tuple[str, str, tuple[str, ...]]],
    ) -> list[tuple[str, str, tuple[str, ...]]]:
        """Merge repeated authoritative rows without fuzzy identity guesses."""
        merged: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        for group in groups:
            for entity_id, name, aliases in group:
                if entity_id not in merged:
                    merged[entity_id] = (entity_id, name, tuple(aliases))
                    continue
                prior_id, prior_name, prior_aliases = merged[entity_id]
                merged[entity_id] = (
                    prior_id,
                    prior_name or name,
                    tuple(dict.fromkeys((*prior_aliases, *aliases))),
                )
        return list(merged.values())

    @classmethod
    def _drop_unknown_roster_refs(
        cls,
        effects: list[ProposedEffect],
        roster_refs: list[tuple[str, str, tuple[str, ...]]],
    ) -> tuple[list[ProposedEffect], int]:
        """Drop references that have no exact authoritative roster owner."""
        known_labels = {
            cls._normalized_phrase(label)
            for entity_id, name, aliases in roster_refs
            for label in (entity_id, name, *aliases)
            if label
        }
        kept: list[ProposedEffect] = []
        dropped = 0
        for effect in effects:
            if (
                effect.effect_type == EffectType.REF_ENTITY
                and effect.ref_entity_id
                and cls._normalized_phrase(effect.ref_entity_id)
                not in known_labels
            ):
                dropped += 1
                continue
            kept.append(effect)
        return kept, dropped

    @classmethod
    def _normalize_grounded_ref_aliases(
        cls,
        effects: list[ProposedEffect],
        prose: str,
    ) -> list[ProposedEffect]:
        """Shorten a hallucinated full alias to the exact name visible in prose.

        The entity id remains untouched and is still checked against live
        state.  This only handles conservative contiguous reductions such as
        ``Sorin Vex`` -> ``Sorin`` when the latter appears verbatim.
        """
        normalized: list[ProposedEffect] = []
        for effect in effects:
            alias = str(effect.ref_alias_used or "").strip()
            if effect.effect_type != EffectType.REF_ENTITY or not alias:
                normalized.append(effect)
                continue
            if (
                cls._usable_ref_alias(alias)
                and f" {cls._normalized_phrase(alias)} "
                in f" {cls._normalized_phrase(prose)} "
            ):
                normalized.append(effect)
                continue
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", alias)
            replacement = ""
            for size in range(len(tokens) - 1, 0, -1):
                for start in range(0, len(tokens) - size + 1):
                    candidate = " ".join(tokens[start : start + size])
                    if not cls._usable_ref_alias(candidate):
                        continue
                    match = re.search(
                        rf"(?<!\w){re.escape(candidate)}(?!\w)",
                        prose,
                        re.IGNORECASE,
                    )
                    if match:
                        replacement = match.group(0)
                        break
                if replacement:
                    break
            normalized.append(
                effect.model_copy(update={"ref_alias_used": replacement})
                if replacement
                else effect
            )
        return normalized

    @classmethod
    def _reconcile_roster_ref_aliases(
        cls,
        effects: list[ProposedEffect],
        prose: str,
        roster_refs: list[tuple[str, str, tuple[str, ...]]],
    ) -> tuple[list[ProposedEffect], int, int]:
        """Prevent a grounded alias from being attached to the wrong roster ID.

        Presence in prose is necessary but not sufficient: a turn can mention
        both Elena and the Tollman while a model emits Elena's UUID with
        ``alias_used='the Tollman'``.  Existing canonical labels may use a
        conservative partial (``Market`` for ``Market Ring south tier``).
        Generic NPC roles may acquire a proper name only when the prose makes
        the naming relationship explicit.

        Returns ``(effects, corrected_count, dropped_count)``.  A mismatched
        alias is stripped when the canonical entity is independently visible;
        otherwise the whole reference is removed rather than recording a
        false identity assertion.
        """
        if not roster_refs:
            return effects, 0, 0

        normalized_rows = [
            (
                entity_id,
                name,
                aliases,
                {
                    cls._normalized_phrase(label)
                    for label in (entity_id, name, *aliases)
                    if cls._normalized_phrase(label)
                },
            )
            for entity_id, name, aliases in roster_refs
        ]
        visibly_recovered_ids = {
            effect.ref_entity_id
            for effect in cls._recover_roster_references(
                prose,
                roster_refs,
                [],
            )
            if effect.ref_entity_id
        }

        reconciled: list[ProposedEffect] = []
        corrected = 0
        dropped = 0
        for effect in effects:
            alias = str(effect.ref_alias_used or "").strip()
            ref_id = str(effect.ref_entity_id or "").strip()
            if effect.effect_type != EffectType.REF_ENTITY or not alias or not ref_id:
                reconciled.append(effect)
                continue
            if not cls._usable_ref_alias(alias):
                # Preserve malformed aliases for the ordinary grounding
                # validator so diagnostics explain the actual defect.
                reconciled.append(effect)
                continue

            query = cls._normalized_phrase(ref_id)
            matches = [row for row in normalized_rows if query in row[3]]
            if len(matches) != 1:
                # Unknown/ambiguous IDs are handled by the live-state and
                # authoritative-roster checks; do not guess here.
                reconciled.append(effect)
                continue

            entity_id, name, aliases, _ = matches[0]
            if any(
                cls._labels_overlap(alias, canonical_label)
                for canonical_label in (name, *aliases)
            ):
                reconciled.append(effect)
                continue
            if (
                cls._is_generic_npc_label(name)
                and cls._explicit_generic_naming_link(prose, name, alias)
            ):
                reconciled.append(effect)
                continue

            corrected += 1
            if entity_id in visibly_recovered_ids:
                reconciled.append(
                    effect.model_copy(update={"ref_alias_used": None})
                )
            else:
                dropped += 1

        return reconciled, corrected, dropped

    @classmethod
    def _labels_overlap(cls, left: str, right: str) -> bool:
        left_tokens = set(cls._normalized_phrase(left).split())
        right_tokens = set(cls._normalized_phrase(right).split())
        return bool(left_tokens and right_tokens) and (
            left_tokens.issubset(right_tokens)
            or right_tokens.issubset(left_tokens)
        )

    @classmethod
    def _explicit_generic_naming_link(
        cls,
        prose: str,
        generic_name: str,
        alias: str,
    ) -> bool:
        """Accept generic-role promotion only with an explicit naming cue."""
        # Lazy imports avoid the package cycle ``game.__init__ -> combat ->
        # llm.narration`` during a cold standalone import of this module.
        from ..game.identity import explicit_npc_naming_link

        return explicit_npc_naming_link(prose, generic_name, alias)

    @staticmethod
    def _is_generic_npc_label(value: str) -> bool:
        """Lazy shared generic-role classifier; see cycle note above."""
        from ..game.identity import is_generic_npc_label

        return is_generic_npc_label(value)

    @classmethod
    def _recover_roster_references(
        cls,
        prose: str,
        roster_refs: list[tuple[str, str, tuple[str, ...]]],
        existing_effects: list[ProposedEffect],
    ) -> list[ProposedEffect]:
        """Deterministically recover roster refs when the model emitted ``{}``."""
        normalized_prose = f" {cls._normalized_phrase(prose)} "
        existing_ids = {
            effect.ref_entity_id
            for effect in existing_effects
            if effect.effect_type == EffectType.REF_ENTITY
            and effect.ref_entity_id
        }
        partial_owners: dict[str, set[str]] = {}
        partial_labels: dict[tuple[str, str], str] = {}
        for entity_id, name, aliases in roster_refs:
            for label in (name, *aliases):
                for token in re.findall(r"[A-Za-z][A-Za-z'-]+", label):
                    normalized_token = cls._normalized_phrase(token)
                    if not cls._usable_ref_alias(token) or len(normalized_token) < 3:
                        continue
                    partial_owners.setdefault(normalized_token, set()).add(entity_id)
                    partial_labels[(entity_id, normalized_token)] = token
        recovered: list[ProposedEffect] = []
        for entity_id, name, aliases in roster_refs:
            if entity_id in existing_ids:
                continue
            matched_label = ""
            for label in sorted((name, *aliases), key=len, reverse=True):
                normalized_label = cls._normalized_phrase(label)
                if normalized_label and f" {normalized_label} " in normalized_prose:
                    matched_label = label
                    break
            if not matched_label:
                unique_partials = [
                    partial_labels[(entity_id, token)]
                    for token, owners in partial_owners.items()
                    if owners == {entity_id}
                    and f" {token} " in normalized_prose
                ]
                if unique_partials:
                    matched_label = max(unique_partials, key=len)
            if not matched_label:
                continue
            recovered.append(ProposedEffect(
                effect_type=EffectType.REF_ENTITY,
                ref_entity_id=entity_id,
                ref_alias_used=(
                    matched_label
                    if cls._normalized_phrase(matched_label)
                    != cls._normalized_phrase(name)
                    else None
                ),
            ))
        return recovered

    async def _tool_followup(
        self,
        prose: str,
        messages: list[dict],
        validation_errors: list[str] | None = None,
        mutation_recovery: bool = False,
        action: str = "",
        required_effect_types: frozenset[EffectType] = frozenset(),
        existing_effects: list[ProposedEffect] | None = None,
        roster_refs: list[tuple[str, str, tuple[str, ...]]] | None = None,
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
        self.last_diagnostics["tool_followup_attempted"] = True
        preserved_initial = self._valid_effects_for_prose(
            list(existing_effects or []), prose, action
        )
        correction = ""
        if required_effect_types:
            required = ", ".join(
                sorted(effect_type.value for effect_type in required_effect_types)
            )
            correction += (
                "\nThe validated player action creates REQUIRED effect "
                f"obligations: {required}. Valid calls from the primary "
                "response are already preserved; return ONLY calls for these "
                "missing families with complete arguments. Do not reinterpret "
                "or negate the resolved action."
            )
        if validation_errors:
            correction += (
                "\nYour previous tool arguments were invalid: "
                + "; ".join(validation_errors)
                + ". Re-issue the complete corrected tool set. If an add_npc "
                "name is absent from the prose, delete that call rather than "
                "renaming the person. Omit any call whose required value "
                "cannot be copied exactly from the prose or roster."
            )
        elif mutation_recovery:
            correction += (
                "\nYour previous calls only referenced entities, but the "
                "player action and your narration describe a durable state "
                "transition. Re-issue the complete tool set, including the "
                "required mutation tool (for example update_entity, "
                "update_player, remove_entity, or change_location)."
            )
        if EffectType.ADD_NPC in required_effect_types:
            creation_instruction = (
                "- add_npc for each properly named new NPC physically on stage; "
                "its exact name MUST appear in the prose above; never invent or "
                "rename an anonymous, dead, background, or off-screen person\n"
            )
        else:
            creation_instruction = (
                "- NPC identity creation is unavailable in this recovery. Do not "
                "emit an NPC-creation call; anonymous or newly named people are "
                "handled by the independent state extractor\n"
            )
        followup_messages = list(messages) + [
            {"role": "assistant", "content": prose[:2000]},
            {
                "role": "user",
                "content": (
                    "Now call a tool for everything you narrated above, using only "
                    "the tools available to you:\n"
                    "- ref_entity for each roster entity you referenced (use the roster IDs)\n"
                    + creation_instruction
                    + "- spawn_object only for a new interactable object\n"
                    "- update_entity when a tracked NPC's status, disposition, description, importance, or inventory changed\n"
                    "- update_player for any player damage, healing, loot, currency, or condition change\n"
                    "- remove_entity when a tracked object was destroyed or otherwise permanently removed from the scene\n"
                    "- change_location if the party moved; start_combat if a fight began\n"
                    "Every call MUST include all required arguments; never send an empty arguments object.\n"
                    "Do NOT respond with prose — only tool calls."
                ) + correction,
            },
        ]

        # Creation is the highest-risk narrator tool: when it is advertised on
        # every recovery turn, weaker models invent names for anonymous roles.
        # A high-confidence ADD_NPC obligation (for example a direct proper-name
        # self-introduction) explicitly opens the surface; all other recovery
        # turns leave anonymous people to StateDelta.
        followup_tools = list(self._get_tools())
        if EffectType.ADD_NPC not in required_effect_types:
            followup_tools = [
                tool
                for tool in followup_tools
                if tool.get("function", {}).get("name") != EffectType.ADD_NPC.value
            ]
        add_npc_allowed = EffectType.ADD_NPC in required_effect_types

        try:
            response = await self._request_tool_followup(
                followup_messages,
                tools=followup_tools,
            )

            if response.tool_calls:
                allowed_tool_calls = [
                    call
                    for call in response.tool_calls
                    if add_npc_allowed or call.get("name") != EffectType.ADD_NPC.value
                ]
                self.last_diagnostics["tool_policy_suppressed_effects"] += (
                    len(response.tool_calls) - len(allowed_tool_calls)
                )
                effects = self._normalize_grounded_ref_aliases(
                    tool_calls_to_effects(allowed_tool_calls),
                    prose,
                )
                effects, unknown_ref_drops = self._drop_unknown_roster_refs(
                    effects,
                    list(roster_refs or []),
                )
                if unknown_ref_drops:
                    self.last_diagnostics[
                        "tool_invalid_effects_dropped"
                    ] += unknown_ref_drops
                    self.last_diagnostics[
                        "tool_unknown_roster_refs_dropped"
                    ] += unknown_ref_drops
                    recovered_refs = self._recover_roster_references(
                        prose,
                        list(roster_refs or []),
                        effects,
                    )
                    for recovered in recovered_refs:
                        if recovered not in effects:
                            effects.append(recovered)
                    if recovered_refs:
                        self.last_diagnostics[
                            "tool_ref_deterministic_recoveries"
                        ] = [
                            effect.ref_entity_id for effect in recovered_refs
                        ]
                effects, alias_corrections, alias_drops = (
                    self._reconcile_roster_ref_aliases(
                        effects,
                        prose,
                        list(roster_refs or []),
                    )
                )
                self.last_diagnostics[
                    "tool_ref_alias_mismatches_removed"
                ] += alias_corrections
                self.last_diagnostics["tool_invalid_effects_dropped"] += (
                    alias_drops
                )
                errors = self._effect_errors(effects, prose, action)
                self.last_diagnostics["tool_followup_structural_errors"] = len(errors)
                self.last_diagnostics[
                    "tool_followup_structural_error_details"
                ] = list(errors)
                if errors:
                    # Preserve every already-valid call. Asking the model to
                    # regenerate the complete set caused repair churn: good
                    # calls disappeared while unrelated malformed calls were
                    # invented. The bounded repair now owns only invalid calls.
                    preserved_effects = self._valid_effects_for_prose(
                        effects, prose, action
                    )
                    invalid_effects = [
                        effect
                        for effect in effects
                        if self._effect_errors([effect], prose, action)
                    ]
                    invalid_refs = [
                        effect
                        for effect in invalid_effects
                        if effect.effect_type == EffectType.REF_ENTITY
                    ]
                    if invalid_refs:
                        recovered_refs = self._recover_roster_references(
                            prose,
                            list(roster_refs or []),
                            preserved_effects,
                        )
                        if recovered_refs:
                            for recovered in recovered_refs:
                                if recovered not in preserved_effects:
                                    preserved_effects.append(recovered)
                            self.last_diagnostics[
                                "tool_ref_deterministic_recoveries"
                            ] = [
                                effect.ref_entity_id for effect in recovered_refs
                            ]
                        else:
                            # A malformed/ungrounded reference cannot mutate
                            # state. If no roster identity is visibly recoverable,
                            # delete it instead of paying a model to guess.
                            self.last_diagnostics[
                                "tool_invalid_effects_dropped"
                            ] += len(invalid_refs)
                        invalid_effects = [
                            effect
                            for effect in invalid_effects
                            if effect.effect_type != EffectType.REF_ENTITY
                        ]

                    invalid_adds = [
                        effect
                        for effect in invalid_effects
                        if effect.effect_type == EffectType.ADD_NPC
                    ]
                    if invalid_adds:
                        # Invalid identity creation is never safely repairable
                        # by inference: generic roles belong to StateDelta and
                        # absent proper names must not be invented. A genuine
                        # required add_npc obligation still gets the separate,
                        # narrowed terminal obligation repair below.
                        self.last_diagnostics[
                            "tool_invalid_effects_dropped"
                        ] += len(invalid_adds)
                        invalid_effects = [
                            effect
                            for effect in invalid_effects
                            if effect.effect_type != EffectType.ADD_NPC
                        ]

                    remaining_errors = [
                        error
                        for effect in invalid_effects
                        for error in self._effect_errors([effect], prose, action)
                    ]
                    if not remaining_errors:
                        effects = preserved_effects
                    else:
                        self.last_diagnostics["tool_repair_attempted"] = True
                        repair_messages = list(followup_messages)
                        repair_messages[-1] = {
                            "role": "user",
                            "content": (
                                followup_messages[-1]["content"]
                                + "\nREPAIR REQUIRED: "
                                + "; ".join(remaining_errors)
                                + ". Valid calls from the prior attempt are already preserved; return ONLY corrected replacements for invalid calls, not the full set. "
                                "If an add_npc name does not appear exactly in the prose, DELETE that call; do not rename the person. "
                                "If any required value cannot be copied exactly from the prose or roster, OMIT that call instead of guessing or sending empty arguments."
                            ),
                        }
                        logger.warning(
                            "narrator_tool_arguments_invalid_repairing",
                            errors=remaining_errors,
                        )
                        invalid_type_names = {
                            effect.effect_type.value for effect in invalid_effects
                        }
                        repair_tools = [
                            tool
                            for tool in self._get_tools()
                            if tool.get("function", {}).get("name")
                            in invalid_type_names
                        ]
                        repaired = await self._request_tool_followup(
                            repair_messages,
                            tool_choice="auto",
                            tools=repair_tools or self._get_tools(),
                        )
                        raw_repair_effects = (
                            tool_calls_to_effects(repaired.tool_calls)
                            if repaired.tool_calls else []
                        )
                        repair_effects = [
                            effect
                            for effect in raw_repair_effects
                            if effect.effect_type.value in invalid_type_names
                        ]
                        self.last_diagnostics["tool_invalid_effects_dropped"] += (
                            len(raw_repair_effects) - len(repair_effects)
                        )
                        repair_effects = self._normalize_grounded_ref_aliases(
                            repair_effects,
                            prose,
                        )
                        repair_effects, unknown_ref_drops = (
                            self._drop_unknown_roster_refs(
                                repair_effects,
                                list(roster_refs or []),
                            )
                        )
                        self.last_diagnostics[
                            "tool_invalid_effects_dropped"
                        ] += unknown_ref_drops
                        self.last_diagnostics[
                            "tool_unknown_roster_refs_dropped"
                        ] += unknown_ref_drops
                        repair_effects, alias_corrections, alias_drops = (
                            self._reconcile_roster_ref_aliases(
                                repair_effects,
                                prose,
                                list(roster_refs or []),
                            )
                        )
                        self.last_diagnostics[
                            "tool_ref_alias_mismatches_removed"
                        ] += alias_corrections
                        self.last_diagnostics[
                            "tool_invalid_effects_dropped"
                        ] += alias_drops
                        repair_errors = self._effect_errors(
                            repair_effects, prose, action
                        )
                        self.last_diagnostics[
                            "tool_repair_structural_errors"
                        ] = len(repair_errors)
                        self.last_diagnostics[
                            "tool_repair_structural_error_details"
                        ] = list(repair_errors)
                        valid_repairs = self._valid_effects_for_prose(
                            repair_effects, prose, action
                        )
                        self.last_diagnostics["tool_invalid_effects_dropped"] += (
                            len(repair_effects) - len(valid_repairs)
                        )
                        effects = list(preserved_effects)
                        for repaired_effect in valid_repairs:
                            if repaired_effect not in effects:
                                effects.append(repaired_effect)
                        if repair_errors:
                            logger.error(
                                "narrator_tool_repair_rejected",
                                errors=repair_errors,
                                retained_effects=len(effects),
                            )
                        if not effects:
                            self.last_diagnostics[
                                "tool_repair_failed_closed"
                            ] = True

                # Schemas and ``tool_choice=required`` are provider guidance,
                # not a trust boundary. Validate the terminal candidate too:
                # a model can repeat the same malformed call on repair.
                terminal_errors = self._effect_errors(effects, prose, action)
                if terminal_errors:
                    self.last_diagnostics["tool_repair_structural_errors"] = len(
                        terminal_errors
                    )
                    self.last_diagnostics[
                        "tool_repair_structural_error_details"
                    ] = list(terminal_errors)
                    valid_effects = self._valid_effects_for_prose(effects, prose, action)
                    self.last_diagnostics["tool_invalid_effects_dropped"] += (
                        len(effects) - len(valid_effects)
                    )
                    effects = valid_effects
                    if not effects:
                        self.last_diagnostics["tool_repair_failed_closed"] = True
                    logger.error(
                        "narrator_tool_repair_rejected",
                        errors=terminal_errors,
                        retained_effects=len(effects),
                    )
                logger.info(
                    "narrator_tool_followup",
                    tool_count=len(response.tool_calls),
                    effects_count=len(effects),
                )
                merged_effects = list(preserved_initial)
                for effect in effects:
                    if effect not in merged_effects:
                        merged_effects.append(effect)
                self.last_diagnostics["tool_followup_effects"] = len(
                    merged_effects
                )
                return merged_effects
        except Exception as e:
            logger.warning("narrator_tool_followup_failed", error=str(e), exc_info=True)

        return preserved_initial

    @staticmethod
    def _is_memory_only_recollection(action: str, prose: str) -> bool:
        """Return True when past-tense mutations are only being remembered."""
        memory_only_cues = re.compile(
            r"\b(?:recall|remember|recollect|think\s+back|last\s+warning|"
            r"last\s+time\s+I\s+saw)\b",
            re.IGNORECASE,
        )
        return bool(
            memory_only_cues.search(action) and memory_only_cues.search(prose)
        )

    @staticmethod
    def _needs_mutation_followup(
        action: str,
        prose: str,
        effects: list[ProposedEffect],
    ) -> bool:
        """Recover when a narrator omits a state write (with refs or no tools).

        This is deliberately a high-precision trigger, not prose-to-state
        extraction. The followup still receives the grounded prompt and must
        propose typed tools that pass normal validation.
        """
        if effects and any(
            effect.effect_type != EffectType.REF_ENTITY for effect in effects
        ):
            return False
        # Recollection can repeat past mutation verbs in both the action and
        # prose without describing a new state transition.  A followup here
        # encourages the model to replay stale tools from earlier turns.
        if NarrationStrategy._is_memory_only_recollection(action, prose):
            return False
        mutation_cues = re.compile(
            r"\b(?:"
            r"hands?|gives?|gave|returns?|returned|pays?|paid|buys?|bought|"
            r"takes?|took|picks?\s+up|drops?|destroy(?:s|ed)?|breaks?|broke|"
            r"kills?|killed|dies?|died|flees?|fled|leaves?|left|travels?|"
            r"arrives?|arrived|moves?|moved|walks?|walked|cross(?:es|ed)?|"
            r"enters?|entered|follows?|followed|heads?|headed|go(?:es|ing)?|"
            r"becomes?|became|all(?:y|ied)|"
            r"wounds?|wounded|heals?|healed|gains?|gained|loses?|lost|"
            r"spends?|spent|sets?|placed?|puts?|stows?"
            r")\b",
            re.IGNORECASE,
        )
        return bool(mutation_cues.search(action) and mutation_cues.search(prose))

    async def _request_tool_followup(
        self,
        messages: list[dict],
        *,
        tool_choice: str = "required",
        tools: list[dict] | None = None,
    ):
        """Issue one deterministic tool-only request using the active tier."""
        return await self._get_narrator().client.chat(
            messages=messages,
            temperature=0,
            max_tokens=500,
            think=False,
            tools=tools if tools is not None else self._get_tools(),
            tool_choice=tool_choice,
        )

    async def targeted_state_followup(
        self,
        prose: str,
        signals: list["StateFollowupSignal"],
    ) -> list[ProposedEffect]:
        """Request exactly the tool calls an applied StateDelta proved missing.

        Unlike :meth:`_tool_followup`, this leg runs after state extraction
        and receives the specific omissions, so the narrator is never asked
        to guess what it forgot. Only the named tool families are advertised;
        results still pass structural validation here and live-state
        validation in the orchestrator's normal effect pipeline (Step 4).
        """
        if not signals:
            return []
        self.last_diagnostics["state_followup_attempted"] = True
        self.last_diagnostics["state_followup_signals"] = [
            signal.instruction for signal in signals
        ]
        allowed = {
            name for signal in signals for name in signal.tool_names
        }
        tools = [
            tool
            for tool in self._get_tools()
            if tool.get("function", {}).get("name") in allowed
        ]
        if not tools:
            return []
        instructions = "\n".join(f"- {signal.instruction}" for signal in signals)
        messages = [
            {
                "role": "system",
                "content": (
                    "You translate a Dungeon Master's already-written "
                    "narration into its missing state tool calls. Use exactly "
                    "the ids and names provided; never invent, rename, or "
                    "add anything else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Narration:\n{prose[:2000]}\n\n"
                    "These tracked state changes from the narration are "
                    f"missing tool declarations:\n{instructions}\n\n"
                    "Emit exactly these tool calls with complete arguments. "
                    "If a listed change did not actually happen in the "
                    "narration, omit that call. Do not add any other calls. "
                    "Do not respond with prose."
                ),
            },
        ]
        try:
            response = await self._request_tool_followup(messages, tools=tools)
        except Exception as e:
            logger.warning(
                "targeted_state_followup_failed", error=str(e), exc_info=True
            )
            return []
        if not getattr(response, "tool_calls", None):
            return []
        allowed_calls = [
            call for call in response.tool_calls if call.get("name") in allowed
        ]
        effects = self._normalize_grounded_ref_aliases(
            tool_calls_to_effects(allowed_calls), prose
        )
        errors = self._structural_effect_errors(effects)
        if errors:
            self.last_diagnostics["state_followup_structural_errors"] = errors
        valid = self._structurally_valid_effects(effects)
        self.last_diagnostics["state_followup_effects"] = len(valid)
        return valid

    @staticmethod
    def _structural_effect_errors(effects: list[ProposedEffect]) -> list[str]:
        """Validate tool argument shape without requiring live scene state."""
        validator = EffectValidator()
        errors = []
        for effect in effects:
            result = validator.validate(effect)
            if not result.valid:
                errors.append(
                    f"{effect.effect_type.value}: {result.rejection_reason}"
                )
        return errors

    @staticmethod
    def _structurally_valid_effects(
        effects: list[ProposedEffect],
    ) -> list[ProposedEffect]:
        """Return only effects safe to hand to the live-state validator."""
        validator = EffectValidator()
        return [effect for effect in effects if validator.validate(effect).valid]

    @staticmethod
    def _prose_freshness_hint(context: BrainContext) -> str:
        """One line telling the narrator how its recent replies opened.

        Frequency/presence penalties act within a single generation and
        cannot see across turns; the narrative grader consistently scores
        prose_freshness lowest, with recycled openers ("Elara's lips" x6
        in one soak) as the measured symptom. Listing the recent openings
        is deterministic, ~40 tokens, and sits at the prompt tail so the
        cached prefix is untouched.
        """
        history = context.message_history or context.recent_messages or []
        openings: list[str] = []
        for message in reversed(history):
            if len(openings) >= 3:
                break
            if str(message.get("role") or "") != "assistant":
                continue
            words = re.findall(r"\S+", str(message.get("content") or ""))
            if words:
                openings.append(" ".join(words[:5]))
        if len(openings) < 2:
            return ""
        listed = " / ".join(f'"{opening}…"' for opening in openings)
        return (
            f"Prose freshness: your recent replies opened with {listed}. "
            "Open this reply with a different subject and sentence shape, "
            "and avoid reusing imagery or stock phrases from recent turns."
        )

    @staticmethod
    def _normalized_phrase(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))

    @classmethod
    def _collapse_duplicate_creations(
        cls,
        effects: list[ProposedEffect],
    ) -> tuple[list[ProposedEffect], int]:
        """Collapse exact same-name ADD_NPC/SPAWN_OBJECT calls in one turn.

        Different call indexes are not different world identities. Keep the
        richer declaration so duplicate model calls cannot create two scene
        entities while the graph silently collapses them to one slug.
        """
        collapsed = 0
        output: list[ProposedEffect] = []
        positions: dict[tuple[EffectType, str], int] = {}
        for effect in effects:
            if effect.effect_type == EffectType.ADD_NPC:
                label = cls._normalized_phrase(effect.npc_name or "")
                richness = len(effect.npc_description or "") + int(
                    bool(effect.npc_disposition)
                )
            elif effect.effect_type == EffectType.SPAWN_OBJECT:
                label = cls._normalized_phrase(effect.object_name or "")
                richness = len(effect.object_description or "") + len(
                    effect.object_properties or {}
                )
            else:
                output.append(effect)
                continue
            if not label:
                output.append(effect)
                continue
            key = (effect.effect_type, label)
            if key not in positions:
                positions[key] = len(output)
                output.append(effect)
                continue
            collapsed += 1
            prior_index = positions[key]
            prior = output[prior_index]
            if prior.effect_type == EffectType.ADD_NPC:
                prior_richness = len(prior.npc_description or "") + int(
                    bool(prior.npc_disposition)
                )
            else:
                prior_richness = len(prior.object_description or "") + len(
                    prior.object_properties or {}
                )
            if richness > prior_richness:
                output[prior_index] = effect
        return output, collapsed

    @classmethod
    def _usable_ref_alias(cls, value: str) -> bool:
        """Reject aliases that are only articles, pronouns, or glue words."""
        tokens = cls._normalized_phrase(value).split()
        noise = {
            "a", "an", "the", "this", "that", "these", "those", "it",
            "he", "her", "him", "she", "them", "they", "we", "you",
            "here", "there", "of", "to", "in", "on", "at",
        }
        return any(len(token) >= 2 and token not in noise for token in tokens)

    @classmethod
    def _grounding_effect_errors(
        cls,
        effects: list[ProposedEffect],
        prose: str,
        action: str = "",
    ) -> list[str]:
        """Reject canonical identity writes/references absent from the fiction."""
        normalized_prose = f" {cls._normalized_phrase(prose)} "
        normalized_identity_surface = f" {cls._normalized_phrase(prose + ' ' + action)} "
        errors: list[str] = []
        for effect in effects:
            if effect.effect_type == EffectType.ADD_NPC and effect.npc_name:
                normalized_name = cls._normalized_phrase(effect.npc_name)
                if (
                    normalized_name
                    and f" {normalized_name} " not in normalized_identity_surface
                ):
                    errors.append(
                        "add_npc: npc_name must appear exactly in narrator prose "
                        "or the current player action "
                        f"(got {effect.npc_name!r})"
                    )
            if effect.effect_type == EffectType.REF_ENTITY and effect.ref_alias_used:
                normalized_alias = cls._normalized_phrase(effect.ref_alias_used)
                if not cls._usable_ref_alias(effect.ref_alias_used):
                    errors.append(
                        "ref_entity: alias_used is too generic to ground an identity "
                        f"(got {effect.ref_alias_used!r})"
                    )
                elif normalized_alias and f" {normalized_alias} " not in normalized_prose:
                    errors.append(
                        "ref_entity: alias_used must appear exactly in narrator prose "
                        f"(got {effect.ref_alias_used!r})"
                    )
        return errors

    @classmethod
    def _effect_errors(
        cls,
        effects: list[ProposedEffect],
        prose: str,
        action: str = "",
    ) -> list[str]:
        return (
            cls._structural_effect_errors(effects)
            + cls._grounding_effect_errors(effects, prose, action)
        )

    @classmethod
    def _valid_effects_for_prose(
        cls,
        effects: list[ProposedEffect],
        prose: str,
        action: str = "",
    ) -> list[ProposedEffect]:
        structurally_valid = cls._structurally_valid_effects(effects)
        return [
            effect
            for effect in structurally_valid
            if not cls._grounding_effect_errors([effect], prose, action)
        ]
