"""Context-budget caps (Step-2 review).

Step 2 restored kg context / narrative memory / char stats to the narrator
prompt, pushing realistic end-game local prompts to ~14-15.5k of a 16k
num_ctx — and on overflow Ollama silently truncates the prompt HEAD
(system persona + <world_state>). These tests pin the three growth-vector
caps:

- ``_cap_kg_context_yaml``: entity count was capped (15), rendered size
  was not — cap at ~4000 chars on an entity-entry boundary.
- ``MemoryBlock.max_tokens``: declared since day one, never enforced —
  now enforced at render time (chars ~= tokens*4), keeping the tail
  (newest, since ``append()`` grows the tail).
- History splice: non-final spliced messages capped at 1500 chars; the
  final (most recent) message stays uncapped.
"""

from __future__ import annotations

from dnd_bot.llm.brains.base import (
    HISTORY_MESSAGE_MAX_CHARS,
    Brain,
    BrainContext,
    _cap_history_messages,
)
from dnd_bot.llm.orchestrator import KG_CONTEXT_MAX_CHARS, _cap_kg_context_yaml
from dnd_bot.memory.blocks import MemoryBlock, MemoryBlockType


# ── KG context YAML cap ───────────────────────────────────────────────────────

def _kg_yaml(entries: int, desc_len: int = 200) -> str:
    """Shape-faithful to_context_yaml output: known_entities list items."""
    lines = ["known_entities:"]
    for i in range(entries):
        lines.append(f"- name: Entity {i}")
        lines.append("  type: npc")
        lines.append(f"  description: {'d' * desc_len}")
        lines.append("  relationships:")
        lines.append(f"  - knows Entity {i + 1}")
    return "\n".join(lines) + "\n"


def test_kg_yaml_under_cap_unchanged():
    yaml_text = _kg_yaml(entries=3)
    assert len(yaml_text) < KG_CONTEXT_MAX_CHARS
    assert _cap_kg_context_yaml(yaml_text) is yaml_text


def test_kg_yaml_over_cap_cuts_at_entity_boundary():
    yaml_text = _kg_yaml(entries=30)
    assert len(yaml_text) > KG_CONTEXT_MAX_CHARS

    capped = _cap_kg_context_yaml(yaml_text)

    assert len(capped) <= KG_CONTEXT_MAX_CHARS + len("\n# truncated\n")
    assert capped.endswith("\n# truncated\n")
    # The cut lands at an entry boundary: strip the marker and the kept
    # text is a prefix ending with a COMPLETE entry (its relationships
    # line survived, and the next entry's "- name:" starts the cut).
    kept = capped[: -len("\n# truncated\n")]
    assert yaml_text.startswith(kept + "\n- name:")
    assert kept.splitlines()[-1].startswith("  - knows Entity")


def test_kg_yaml_single_oversized_entry_falls_back_to_line_boundary():
    yaml_text = _kg_yaml(entries=1, desc_len=6000)

    capped = _cap_kg_context_yaml(yaml_text)

    assert capped.endswith("\n# truncated\n")
    assert len(capped) <= KG_CONTEXT_MAX_CHARS + len("\n# truncated\n")
    # No mid-line cut: everything kept is whole lines of the original.
    kept_lines = capped[: -len("\n# truncated\n")].splitlines()
    original_lines = yaml_text.splitlines()
    assert kept_lines == original_lines[: len(kept_lines)]
    # The fallback keeps the entry's leading lines, not just the header.
    assert len(kept_lines) >= 2


# ── MemoryBlock.max_tokens enforcement ────────────────────────────────────────

def _block(content: str, max_tokens: int = 10) -> MemoryBlock:
    return MemoryBlock(
        block_type=MemoryBlockType.QUESTS,
        name="Active Quests",
        content=content,
        max_tokens=max_tokens,
    )


def test_memory_block_under_cap_renders_verbatim():
    block = _block("short quest list", max_tokens=10)  # cap = 40 chars
    assert block.to_context_string() == (
        "<Active Quests>\nshort quest list\n</Active Quests>"
    )


def test_memory_block_over_cap_keeps_tail_and_marks_truncation():
    # append() grows the tail, so the NEWEST content is at the end and
    # must survive; the oldest (head) is dropped.
    block = _block("OLD " * 30 + "NEWEST QUEST", max_tokens=10)  # cap = 40 chars

    rendered = block.to_context_string()

    assert "[...older entries truncated...]" in rendered
    assert "NEWEST QUEST" in rendered
    body = rendered.split("\n", 1)[1].rsplit("\n", 1)[0]
    kept = body.split("]\n", 1)[1]
    assert len(kept) == 40
    # Stored content is untouched — enforcement is render-only.
    assert block.content.startswith("OLD OLD")


# ── History splice cap ────────────────────────────────────────────────────────

def test_cap_history_caps_non_final_only_without_mutating():
    long = "x" * (HISTORY_MESSAGE_MAX_CHARS + 500)
    history = [
        {"role": "user", "content": long},
        {"role": "assistant", "content": "short"},
        {"role": "user", "content": long},  # final: uncapped
    ]

    capped = _cap_history_messages(history)

    assert capped[0]["content"] == long[:HISTORY_MESSAGE_MAX_CHARS] + "…"
    assert capped[1] is history[1]  # under-cap entries pass through as-is
    assert capped[2] is history[2]  # final message uncapped, same object
    assert history[0]["content"] == long  # source dict not mutated


def test_cap_history_empty():
    assert _cap_history_messages([]) == []


def test_build_messages_splices_capped_history():
    # Through the real builder: the capped copy reaches the prompt, the
    # final message stays verbatim.
    long = "y" * (HISTORY_MESSAGE_MAX_CHARS + 100)
    brain = Brain(client=None, system_prompt="SYS")
    context = BrainContext(
        message_history=[
            {"role": "user", "content": long},
            {"role": "user", "content": "latest"},
        ],
    )

    messages = brain.build_messages(context)

    spliced = [m for m in messages if m["role"] == "user"]
    assert spliced[0]["content"] == long[:HISTORY_MESSAGE_MAX_CHARS] + "…"
    assert spliced[1]["content"] == "latest"
