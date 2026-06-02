"""Audit a long-horizon session log for paraphrase-fragmentation dedup metrics.

Run after a long-horizon test session ends. Counts add_npc / ref_entity
calls per character, summarizes aliases recorded on each NPC, and shows
extractor-side new_npcs that the extractor itself produced.

Usage:
    python tools/audit_dedup.py <path/to/session.jsonl>
    python tools/audit_dedup.py --latest        # newest log in data/turn_logs/

This is the script referenced in docs/handoff/phase_7_dedup_validation_run.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


def _load_records(log_path: Path) -> list[dict]:
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warning: skipping unparseable line: {e}")
    return records


def _audit(log_path: Path) -> None:
    records = _load_records(log_path)
    print(f"Session log: {log_path}")
    print(f"Total turn records: {len(records)}")
    if not records:
        print("(empty log)")
        return

    add_npc_calls: list[tuple[int, str]] = []
    ref_entity_calls: list[tuple[int, str, str | None]] = []
    extractor_new_npcs: list[tuple[int, str]] = []

    for rec in records:
        turn = rec.get("turn", "?")
        # Narrator-side proposed effects
        for e in rec.get("effects", []) or []:
            t = e.get("type") or e.get("effect_type") or ""
            if t == "add_npc":
                add_npc_calls.append((turn, e.get("npc_name") or ""))
            elif t == "ref_entity":
                ref_entity_calls.append((
                    turn,
                    e.get("ref_entity_id") or e.get("ref_id") or "",
                    e.get("ref_alias_used") or e.get("ref_alias"),
                ))

        # State-extractor delta: new_npcs the extractor proposed (post-dedup-pass)
        delta = rec.get("state_delta") or {}
        for npc in (delta.get("new_npcs") or []):
            if isinstance(npc, dict):
                extractor_new_npcs.append((turn, npc.get("name") or ""))
            elif isinstance(npc, str):
                extractor_new_npcs.append((turn, npc))

    print()
    print(f"add_npc count       : {len(add_npc_calls)}")
    print(f"ref_entity count    : {len(ref_entity_calls)}")
    print(f"extractor new_npcs  : {len(extractor_new_npcs)}")

    print()
    print("add_npc per name (narrator-side):")
    for name, count in Counter(name for _, name in add_npc_calls).most_common():
        marker = "  !! " if count >= 3 else "    "
        print(f"{marker}{count}x  {name!r}")

    print()
    print("ref_entity per id (top 8):")
    for rid, count in Counter(rid for _, rid, _ in ref_entity_calls).most_common(8):
        print(f"  {count}x  {rid}")

    print()
    print("extractor new_npcs per name (post-dedup-pass — should be ~0 for known entities):")
    for name, count in Counter(name for _, name in extractor_new_npcs).most_common():
        print(f"  {count}x  {name!r}")

    # Track all NPC ids ever seen across the run by scanning every YAML
    # snapshot. Final-snapshot YAML only includes npcs at the current
    # location plus important=True elsewhere, so it under-reports the
    # full registry. Scanning all turns gives the registry's reach.
    npc_history: dict[str, dict] = {}  # id -> latest known {name, aliases, last_seen_turn}
    for rec in records:
        ws_section = rec.get("world_state") or {}
        ws_yaml = ws_section.get("after") or ""
        if not ws_yaml:
            continue
        try:
            data = yaml.safe_load(ws_yaml) or {}
        except yaml.YAMLError:
            continue
        turn = rec.get("turn", 0)
        for entry in (data.get("npcs_here") or []) + (data.get("key_npcs_elsewhere") or []):
            if not isinstance(entry, dict):
                continue
            nid = entry.get("id", "")
            if not nid:
                continue
            npc_history[nid] = {
                "name": entry.get("name", ""),
                "aliases": list(entry.get("aliases") or []),
                "disposition": entry.get("disposition") or "",
                "last_seen_turn": turn,
            }

    print()
    print(f"Distinct NPC ids observed across all turns: {len(npc_history)}")
    for nid, info in sorted(npc_history.items(), key=lambda kv: -kv[1]["last_seen_turn"]):
        print(f"  [{nid[:8]}] {info['name']!r:30}  aliases={info['aliases']}  last_seen=t{info['last_seen_turn']}")
    all_npcs = list(npc_history.values())

    # Quick-glance summary
    print()
    print("-" * 60)
    print("Summary")
    print("-" * 60)
    add_per_name = Counter(name for _, name in add_npc_calls)
    high = [(n, c) for n, c in add_per_name.items() if c >= 3]
    print(f"  add_npc=>=3 per character : {len(high)}  (target: 0)")
    if high:
        for n, c in high:
            print(f"     !! {n!r} got {c}x add_npc")
    aliases_per_npc = sum(len(e.get("aliases") or []) for e in all_npcs)
    print(f"  total aliases recorded   : {aliases_per_npc}  (target: > 0 across run)")
    print(f"  distinct NPC ids         : {len(all_npcs)}  (vs add_npc {len(add_npc_calls)})")
    if len(add_npc_calls) > 0:
        rewrite_ratio = 1 - (len(all_npcs) / len(add_npc_calls))
        print(f"  implied dedup rate       : {rewrite_ratio:.0%}  (every rewrite avoided one record)")
    print(f"  add_npc total            : {len(add_npc_calls)}")
    print(f"  ref_entity total         : {len(ref_entity_calls)}")
    if (len(add_npc_calls) + len(ref_entity_calls)) > 0:
        ref_share = len(ref_entity_calls) / (len(add_npc_calls) + len(ref_entity_calls))
        print(f"  ref_entity share         : {ref_share:.0%}  (paraphrases routed to ref instead of add)")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", help="Path to session jsonl log")
    p.add_argument("--latest", action="store_true",
                   help="Use most recent log in data/turn_logs/")
    args = p.parse_args()

    if args.latest:
        candidates = sorted(
            Path("data/turn_logs").glob("*.jsonl"),
            key=lambda p_: p_.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print("No session logs found in data/turn_logs/")
            sys.exit(1)
        log_path = candidates[0]
    elif args.log:
        log_path = Path(args.log)
    else:
        p.print_help()
        sys.exit(1)

    if not log_path.exists():
        print(f"Log not found: {log_path}")
        sys.exit(1)

    _audit(log_path)


if __name__ == "__main__":
    _main()
