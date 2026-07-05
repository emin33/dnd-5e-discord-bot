"""Shared JSON extraction for LLM responses.

LLMs asked for JSON routinely wrap it in markdown code fences or surround
it with prose. This module is the single implementation of the recovery
routine (fence strip -> brace extract -> parse) that was previously
copy-pasted across the triage parser, extractors, and dedup judge with
drifted instrumentation.
"""

import json
from typing import Any, Optional


def extract_json_object(content: str) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Extract and parse the first JSON object from LLM output.

    Recovery steps, in order:
    1. Strip a ```json ... ``` or bare ``` ... ``` code fence.
    2. Brace-extract the outermost {...} span from surrounding text.
    3. json.loads; reject anything that isn't a JSON object.

    Returns (data, warnings): ``data`` is the parsed dict, or None if no
    valid JSON object could be recovered. ``warnings`` lists every recovery
    step taken and any failure reason, so callers can record them in the
    turn log for post-mortem observability.
    """
    warnings: list[str] = []

    if not content or not content.strip():
        warnings.append("empty_content")
        return None, warnings

    content = content.strip()

    # Strip markdown code fences
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()
            warnings.append("stripped_markdown_fence")
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        if end > start:
            content = content[start:end].strip()
            warnings.append("stripped_code_fence")

    # Extract the outermost JSON object from surrounding text
    if "{" not in content:
        warnings.append("no_json_object_found")
        return None, warnings

    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start > 0 or json_end < len(content):
        warnings.append("extracted_json_from_text")
    if json_end > json_start:
        content = content[json_start:json_end]

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        warnings.append(f"json_parse_failed: {e}")
        return None, warnings

    if not isinstance(data, dict):
        warnings.append("json_not_an_object")
        return None, warnings

    return data, warnings
