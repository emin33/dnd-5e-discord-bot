"""Structured Turn Logger — full pipeline observability.

Writes one JSON record per turn to data/turn_logs/{session_id}.jsonl.
Each record captures the complete pipeline state: what went in, what came
out, and what happened at every stage. Enables post-mortem debugging when
the narrator collapses (e.g., text looping at turn 61 of a 75-turn session).

Records are append-only JSONL (one JSON object per line) for efficient
streaming writes and easy grep/jq analysis.
"""

import json
import time
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger()

TURN_LOG_DIR = Path("data/turn_logs")


class TurnRecord:
    """Accumulates data for a single turn, then flushes to disk."""

    def __init__(self, session_id: str, turn: int):
        self.data: dict[str, Any] = {
            "session_id": session_id,
            "turn": turn,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "timings": {},
        }
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None

    def set(self, key: str, value: Any) -> None:
        """Set a top-level field."""
        self.data[key] = value

    def start_stage(self, stage: str) -> None:
        """Start timing a pipeline stage."""
        self._flush_stage()
        self._current_stage = stage
        self._stage_start = time.monotonic()

    def end_stage(self, stage: Optional[str] = None) -> None:
        """End timing the current (or named) stage."""
        if stage and stage != self._current_stage:
            # Mismatched stage name — just record what we have
            pass
        self._flush_stage()

    def _flush_stage(self) -> None:
        """Write the current stage timing."""
        if self._current_stage and self._stage_start:
            elapsed_ms = round((time.monotonic() - self._stage_start) * 1000)
            self.data["timings"][self._current_stage] = elapsed_ms
        self._current_stage = None
        self._stage_start = None

    def record_prompt(self, messages: list[dict]) -> None:
        """Record the full prompt sent to the narrator.

        Stores message roles and content lengths for debugging without
        bloating the log with full prompt text. Full content stored
        only for the last user message (the action).
        """
        summary = []
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
            entry = {
                "role": msg["role"],
                "chars": len(content),
            }
            # Store first 200 chars for context
            if len(content) <= 500:
                entry["content"] = content
            else:
                entry["preview"] = content[:200] + "..."
            summary.append(entry)

        self.data["prompt"] = {
            "message_count": len(messages),
            "total_chars": total_chars,
            "est_tokens": total_chars // 4,
            "messages": summary,
        }

    def record_narrator_response(self, raw: str, format_type: str = "",
                                  reprompted: bool = False) -> None:
        """Record the raw narrator response."""
        self.data["narrator_response"] = {
            "raw_length": len(raw),
            "format": format_type,
            "reprompted": reprompted,
            "preview": raw[:300] if raw else "(empty)",
            "raw": raw,  # Full content for debugging loops
        }

    def record_triage(self, action_type: str, needs_roll: bool,
                       skill: str = "", dc: int = 0) -> None:
        """Record triage decision."""
        self.data["triage"] = {
            "action_type": action_type,
            "needs_roll": needs_roll,
            "skill": skill,
            "dc": dc,
        }

    def record_state_delta(self, delta_dict: dict, rejections: list[str]) -> None:
        """Record extracted StateDelta and any rejections."""
        self.data["state_delta"] = {
            "delta": delta_dict,
            "rejections": rejections,
        }

    def record_nli(self, pairs_checked: int, contradictions: list[dict],
                    ambiguous_count: int = 0, tiebreaker_results: int = 0) -> None:
        """Record NLI validation results."""
        self.data["nli"] = {
            "pairs_checked": pairs_checked,
            "contradictions": contradictions,
            "ambiguous_count": ambiguous_count,
            "tiebreaker_confirmed": tiebreaker_results,
        }

    def record_effects(self, proposed: list[dict], executed: list[dict],
                        rejected: list[dict]) -> None:
        """Record proposed, executed, and rejected effects."""
        self.data["effects"] = {
            "proposed": proposed,
            "executed": executed,
            "rejected": rejected,
        }

    def record_world_state(self, before_yaml: str, after_yaml: str) -> None:
        """Record WorldState snapshots before and after the turn."""
        self.data["world_state"] = {
            "before": before_yaml,
            "after": after_yaml,
        }

    def record_memory_state(self, buffer_size: int, overflow_size: int,
                             pinned_facts_count: int, has_summary: bool) -> None:
        """Record memory system state."""
        self.data["memory"] = {
            "buffer_size": buffer_size,
            "overflow_size": overflow_size,
            "pinned_facts": pinned_facts_count,
            "has_summary": has_summary,
        }

    def record_knowledge_graph(
        self,
        nodes_total: int,
        edges_total: int,
        seed_entities: list[str],
        context_injected: bool,
        ops_applied: int = 0,
        ops_rejected: int = 0,
        narrative_chunk_stored: bool = False,
        vector_matches: int = 0,
        narrative_chunks_recalled: int = 0,
    ) -> None:
        """Record knowledge graph activity for this turn."""
        self.data["knowledge_graph"] = {
            "nodes_total": nodes_total,
            "edges_total": edges_total,
            "seed_entities": seed_entities,
            "context_injected": context_injected,
            "ops_applied": ops_applied,
            "ops_rejected": ops_rejected,
            "narrative_chunk_stored": narrative_chunk_stored,
            "vector_matches": vector_matches,
            "narrative_chunks_recalled": narrative_chunks_recalled,
        }

    def record_error(self, stage: str, error: str) -> None:
        """Record an error at any pipeline stage."""
        if "errors" not in self.data:
            self.data["errors"] = []
        self.data["errors"].append({"stage": stage, "error": error})


class TurnLogger:
    """Manages per-session turn log files."""

    def __init__(self):
        TURN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, Path] = {}

    def new_turn(self, session_id: str, turn: int) -> TurnRecord:
        """Create a new turn record."""
        return TurnRecord(session_id, turn)

    def flush(self, record: TurnRecord) -> None:
        """Write a completed turn record to disk."""
        session_id = record.data.get("session_id", "unknown")

        # Finalize any open timing stage
        record._flush_stage()

        # Get or create file path
        if session_id not in self._files:
            self._files[session_id] = TURN_LOG_DIR / f"{session_id}.jsonl"

        path = self._files[session_id]

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.data, default=str) + "\n")
        except Exception as e:
            logger.warning("turn_log_write_failed", error=str(e))


# Singleton
_turn_logger: Optional[TurnLogger] = None


def get_turn_logger() -> TurnLogger:
    """Get or create the turn logger singleton."""
    global _turn_logger
    if _turn_logger is None:
        _turn_logger = TurnLogger()
    return _turn_logger
