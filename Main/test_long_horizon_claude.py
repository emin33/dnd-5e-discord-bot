"""Long-horizon test where Claude is the player.

A long-running process that owns the test_harness session, polls
``data/claude_actions.jsonl`` for player actions, and writes narrator
responses to ``data/claude_responses.jsonl``. Claude (the agent) drives
the session turn-by-turn through Bash tool calls — appending actions to
the input file and reading responses from the output file.

This lets the agent identify a seed naturally from emergent prose and
deliberately reference it many turns later — a stronger long-horizon
memory test than scripted or Gemini-Flash play.

## Protocol

Input file (`claude_actions.jsonl`) — one JSON object per line, agent appends:
    {"type": "action", "text": "I do X"}
    {"type": "stop"}

Output file (`claude_responses.jsonl`) — one JSON object per line, process appends:
    {"type": "ready", "session_id": "<uuid>"}
    {"type": "response", "turn": N, "narrative": "...", "elapsed_s": 9.2,
     "effects_summary": [...], "tool_count": K}
    {"type": "stopped", "session_id": "<uuid>"}
    {"type": "error", "stage": "...", "message": "..."}

## Usage

Start (in background):
    python test_long_horizon_claude.py --profile deepseek_v4_flash

Then write actions one at a time from another shell or via Bash tool calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)
from dotenv import load_dotenv
load_dotenv()


ACTIONS_FILE = Path("data/claude_actions.jsonl")
RESPONSES_FILE = Path("data/claude_responses.jsonl")
POLL_INTERVAL_S = 0.5


def _emit(record: dict):
    """Append one event to the responses file."""
    with open(RESPONSES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _reset_files():
    """Truncate both files at session start so old runs don't leak in."""
    ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIONS_FILE.write_text("", encoding="utf-8")
    RESPONSES_FILE.write_text("", encoding="utf-8")


async def _read_new_actions(last_pos: int) -> tuple[list[dict], int]:
    """Read any new lines from ACTIONS_FILE since last_pos.

    Returns (commands, new_pos).
    """
    if not ACTIONS_FILE.exists():
        return [], last_pos

    with open(ACTIONS_FILE, "r", encoding="utf-8") as f:
        f.seek(last_pos)
        chunk = f.read()
        new_pos = f.tell()

    if not chunk:
        return [], new_pos

    cmds: list[dict] = []
    for line in chunk.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            cmds.append(json.loads(line))
        except json.JSONDecodeError as e:
            _emit({
                "type": "error",
                "stage": "parse",
                "message": f"Bad JSON line: {line!r} ({e})",
            })
    return cmds, new_pos


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="Profile to use (overrides ACTIVE_PROFILE)")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Hard cap on turn count (safety net)")
    parser.add_argument("--idle-timeout", type=int, default=600,
                        help="Stop if no action received for this many seconds")
    args = parser.parse_args()

    if args.profile:
        os.environ["ACTIVE_PROFILE"] = args.profile

    # Reset both files. Anything in them from prior runs is stale.
    _reset_files()

    # Lazy imports so ACTIVE_PROFILE is set before profile loading
    from test_harness import TestSession

    print(f"[claude_play] Booting session (profile={os.environ.get('ACTIVE_PROFILE', '(default)')})", flush=True)

    session = TestSession()
    await session.setup()
    # GameSessionManager stores sessions keyed by ``f"discord:{channel_id}"``
    # (the session_key). The ``get_session`` helper builds the right key.
    ws_session = session.manager.get_session(session.channel_id)
    session_id = (
        getattr(ws_session, "id", None)
        or getattr(ws_session, "session_id", None)
        or getattr(getattr(ws_session, "campaign", None), "id", None)
        or "unknown"
    )

    _emit({
        "type": "ready",
        "session_id": str(session_id),
        "channel_id": session.channel_id,
        "max_turns": args.max_turns,
        "profile": os.environ.get("ACTIVE_PROFILE", "(default)"),
    })
    print(f"[claude_play] Ready — session_id={session_id}", flush=True)

    last_pos = 0
    last_action_at = time.monotonic()
    turn_count = 0

    try:
        while True:
            # Idle timeout
            if time.monotonic() - last_action_at > args.idle_timeout:
                _emit({"type": "error", "stage": "idle_timeout",
                       "message": f"No action for {args.idle_timeout}s, stopping"})
                break

            # Hard cap
            if turn_count >= args.max_turns:
                _emit({"type": "error", "stage": "max_turns",
                       "message": f"Hit max turn cap of {args.max_turns}"})
                break

            cmds, last_pos = await _read_new_actions(last_pos)
            if not cmds:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue

            for cmd in cmds:
                ctype = cmd.get("type", "")
                if ctype == "stop":
                    _emit({"type": "stopped", "session_id": str(session_id),
                           "turns_played": turn_count})
                    print(f"[claude_play] Stopping (after {turn_count} turns)", flush=True)
                    return
                if ctype != "action":
                    _emit({"type": "error", "stage": "unknown_command",
                           "message": f"Unknown type: {ctype!r}"})
                    continue

                action = (cmd.get("text") or "").strip()
                if not action:
                    _emit({"type": "error", "stage": "empty_action",
                           "message": "Action text was empty"})
                    continue

                turn_count += 1
                last_action_at = time.monotonic()
                start = time.time()
                try:
                    response = await session.send_action(action)
                    elapsed = time.time() - start
                except Exception as e:
                    _emit({
                        "type": "error",
                        "stage": "send_action",
                        "message": f"{type(e).__name__}: {e}",
                        "turn": turn_count,
                    })
                    continue

                narrative = (response.narrative or "") if response else ""

                # Pull effects + KG snapshot from the freshly-written turn log
                effects_summary: list[dict] = []
                kg_summary: dict = {}
                world_npcs: list[dict] = []
                try:
                    log_path = Path(f"data/turn_logs/{session_id}.jsonl")
                    if log_path.exists():
                        # Read the LAST line — most recent turn record
                        last_line = ""
                        with open(log_path, "rb") as f:
                            f.seek(0, 2)
                            file_size = f.tell()
                            # Read a chunk from the end big enough to contain one line
                            chunk_size = min(8192, file_size)
                            f.seek(max(0, file_size - chunk_size))
                            tail = f.read().decode("utf-8", errors="replace")
                            for ln in reversed(tail.split("\n")):
                                if ln.strip():
                                    last_line = ln
                                    break
                        if last_line:
                            tr = json.loads(last_line)
                            effects = tr.get("effects") or []
                            if isinstance(effects, list):
                                effects_summary = [
                                    {k: v for k, v in e.items() if k in ("type", "item", "npc_name", "ref_id")}
                                    for e in effects
                                ]
                            kg = tr.get("knowledge_graph") or {}
                            kg_summary = {
                                "nodes_total": kg.get("nodes_total"),
                                "edges_total": kg.get("edges_total"),
                                "seed_entities": kg.get("seed_entities", [])[:8],
                                "context_injected": kg.get("context_injected"),
                            }
                            ws = (tr.get("world_state") or {}).get("after", "")
                            # Grab NPC names + inventories at a glance
                            import re
                            for m in re.finditer(r"^- name: (.+)$", ws, re.M):
                                world_npcs.append({"name": m.group(1).strip()})
                except Exception:
                    pass

                _emit({
                    "type": "response",
                    "turn": turn_count,
                    "action": action,
                    "narrative": narrative,
                    "elapsed_s": round(elapsed, 1),
                    "effects": effects_summary,
                    "kg": kg_summary,
                    "npcs_present": world_npcs[:8],
                })
                print(f"[claude_play] Turn {turn_count} OK ({elapsed:.1f}s)", flush=True)
    finally:
        try:
            await session.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
