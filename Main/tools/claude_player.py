"""Helper for the Claude-in-the-loop long-horizon test.

Usage:
    # Send an action and wait for the response (default 240s timeout):
    python tools/claude_player.py act "I look around carefully."

    # Wait for a specific event type (e.g. ready) at startup:
    python tools/claude_player.py wait ready

    # Stop the long-running session:
    python tools/claude_player.py stop

    # Show the last N response events:
    python tools/claude_player.py tail 5

    # Show the most recent narrator response only:
    python tools/claude_player.py last

The companion long-running process is ``test_long_horizon_claude.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Run from repo root regardless of where invoked from
os.chdir(Path(__file__).resolve().parent.parent)

ACTIONS_FILE = Path("data/claude_actions.jsonl")
RESPONSES_FILE = Path("data/claude_responses.jsonl")


def _append_action(cmd: dict) -> None:
    ACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(cmd) + "\n")
        f.flush()


def _read_all_responses() -> list[dict]:
    if not RESPONSES_FILE.exists():
        return []
    out: list[dict] = []
    with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _wait_for_new_event(start_count: int, timeout: int = 240,
                       want_type: str | None = None) -> dict | None:
    """Poll responses file until a new event lands. Returns the event dict
    or None on timeout. If ``want_type`` is given, skips events of other
    types until one matches."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        events = _read_all_responses()
        if len(events) > start_count:
            for ev in events[start_count:]:
                if want_type is None or ev.get("type") == want_type:
                    return ev
                # Surface other events but keep waiting
                _print_event(ev, prefix="(skipping)")
            start_count = len(events)
        time.sleep(0.5)
    return None


def _print_event(ev: dict, prefix: str = "") -> None:
    et = ev.get("type", "?")
    tag = f"{prefix} " if prefix else ""
    if et == "response":
        narr = (ev.get("narrative") or "")[:600]
        print(f"{tag}[turn {ev.get('turn')}] elapsed={ev.get('elapsed_s')}s tools={len(ev.get('effects') or [])}")
        print(f"  narrative: {narr}")
        if ev.get("effects"):
            print(f"  effects: {json.dumps(ev['effects'])[:400]}")
        if ev.get("kg", {}).get("seed_entities"):
            print(f"  kg.seed_entities: {ev['kg']['seed_entities']}")
        if ev.get("npcs_present"):
            names = [n["name"] for n in ev["npcs_present"]]
            print(f"  npcs_present: {names}")
    elif et == "ready":
        print(f"{tag}READY: session_id={ev.get('session_id')} channel={ev.get('channel_id')} profile={ev.get('profile')}")
    elif et == "stopped":
        print(f"{tag}STOPPED: session_id={ev.get('session_id')} turns_played={ev.get('turns_played')}")
    elif et == "error":
        print(f"{tag}ERROR [{ev.get('stage')}]: {ev.get('message')}")
    else:
        print(f"{tag}{json.dumps(ev)[:400]}")


def cmd_wait(args):
    start = len(_read_all_responses())
    ev = _wait_for_new_event(start - 1, timeout=args.timeout, want_type=args.event)
    if ev is None:
        print(f"timed out after {args.timeout}s waiting for {args.event!r}")
        sys.exit(2)
    _print_event(ev)


def cmd_act(args):
    start = len(_read_all_responses())
    _append_action({"type": "action", "text": args.text})
    ev = _wait_for_new_event(start, timeout=args.timeout)
    if ev is None:
        print(f"timed out after {args.timeout}s waiting for response")
        sys.exit(2)
    _print_event(ev)


def cmd_stop(args):
    _append_action({"type": "stop"})
    start = len(_read_all_responses())
    ev = _wait_for_new_event(start, timeout=args.timeout, want_type="stopped")
    if ev:
        _print_event(ev)
    else:
        print("no stopped event observed within timeout")


def cmd_tail(args):
    events = _read_all_responses()
    for ev in events[-args.n:]:
        _print_event(ev)


def cmd_last(args):
    events = _read_all_responses()
    for ev in reversed(events):
        if ev.get("type") == "response":
            _print_event(ev)
            return
    print("no response event yet")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_act = sub.add_parser("act", help="Send a player action and wait for the narrator response")
    p_act.add_argument("text")
    p_act.add_argument("--timeout", type=int, default=240)

    p_wait = sub.add_parser("wait", help="Wait for an event of a specific type")
    p_wait.add_argument("event", choices=["ready", "response", "stopped", "error"])
    p_wait.add_argument("--timeout", type=int, default=120)

    p_stop = sub.add_parser("stop", help="Stop the long-running session")
    p_stop.add_argument("--timeout", type=int, default=30)

    p_tail = sub.add_parser("tail", help="Show the last N events")
    p_tail.add_argument("n", type=int, nargs="?", default=10)

    sub.add_parser("last", help="Show the most recent narrator response")

    args = parser.parse_args()
    {
        "act": cmd_act,
        "wait": cmd_wait,
        "stop": cmd_stop,
        "tail": cmd_tail,
        "last": cmd_last,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
