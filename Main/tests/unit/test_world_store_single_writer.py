"""Single-writer enforcement (REFACTOR_PLAN Step 4, anti-re-flag rule).

Every WorldState mutation goes through ``WorldStateStore``. This guard
AST-scans the production package for the mutation idioms the Step-4
writer inventory catalogued and fails loudly if one reappears outside the
store — the same source-inspection style as the tool-registry
exhaustiveness tests, standing in for the import-linter contract the plan
wants until that lands. AST (not regex) so docstrings and comments can
never trip it.

Scope: the production package (``dnd_bot/``) plus the Main-root harness
scripts (``Main/*.py``) — the Step-4 review found the harnesses
hand-rolling phase writes, and prior waves' real breakage hid in exactly
those files (pytest never collects them). Tests may mutate WorldState
directly to arrange fixtures. Reads are unrestricted — the read-only view
is later-step work; this guard is about WRITES.

Heuristic: production receivers for the state are always named or
attribute-accessed as ``world_state`` (``world_state`` locals,
``session.world_state``, ``self.world_state``). The store's own body
aliases through ``self._state``, so it never matches; ambiguous method
names on OTHER receivers (e.g. the inventory repo's ``remove_item``)
don't either.
"""

import ast
from pathlib import Path

DND_BOT = Path(__file__).resolve().parents[2] / "dnd_bot"

# The store itself and the state class it wraps are the write authority.
ALLOWED = {
    DND_BOT / "game" / "world_store.py",
    DND_BOT / "game" / "world_state.py",
}

# WorldState's mutating methods (from the Step-4 writer inventory).
MUTATING_METHODS = {
    "increment_turn",
    "sync_player",
    "spawn_item",
    "remove_item",
    "record_transfer",
    "apply_delta",
}

# Fields whose direct assignment (or container mutation) is a write.
ASSIGNED_FIELDS = {
    "phase",
    "current_location",
    "location_description",
    "time_of_day",
    "turn",
}
MUTATED_CONTAINERS = {
    "npcs",
    "quests",
    "players",
    "scene_items",
    "recent_transfers",
    "recent_events",
    "established_facts",
    "connected_locations",
    "active_effects",
    "global_flags",
}


def _is_world_state(node: ast.AST) -> bool:
    """True when the expression is a ``world_state`` receiver."""
    if isinstance(node, ast.Name):
        return node.id == "world_state"
    if isinstance(node, ast.Attribute):
        return node.attr == "world_state"
    return False


def _violations_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(DND_BOT.parent)
    found: list[str] = []

    def flag(node: ast.AST, what: str) -> None:
        found.append(f"{rel}:{node.lineno}: {what}")

    for node in ast.walk(tree):
        # world_state.<mutating_method>(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr in MUTATING_METHODS and _is_world_state(func.value):
                flag(node, f"world_state.{func.attr}(...)")
            # world_state.<container>.append/pop/update/clear(...)
            if (
                func.attr in {"append", "pop", "update", "clear", "extend", "insert", "remove"}
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in MUTATED_CONTAINERS
                and _is_world_state(func.value.value)
            ):
                flag(node, f"world_state.{func.value.attr}.{func.attr}(...)")

        # world_state.<field> = ... / augmented assignment
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in (ASSIGNED_FIELDS | MUTATED_CONTAINERS)
                and _is_world_state(target.value)
            ):
                flag(node, f"world_state.{target.attr} = ...")
            # world_state.<container>[key] = ...
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr in MUTATED_CONTAINERS
                and _is_world_state(target.value.value)
            ):
                flag(node, f"world_state.{target.value.attr}[...] = ...")

    return found


def test_no_world_state_writes_outside_the_store():
    scan_paths = list(DND_BOT.rglob("*.py")) + list(DND_BOT.parent.glob("*.py"))
    violations: list[str] = []
    for path in sorted(scan_paths):
        if path in ALLOWED:
            continue
        violations.extend(_violations_in(path))
    assert not violations, (
        "WorldState mutation outside WorldStateStore (Step-4 single-writer "
        "rule) — route it through a store apply method instead:\n"
        + "\n".join(violations)
    )
