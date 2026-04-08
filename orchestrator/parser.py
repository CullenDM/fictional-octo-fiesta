"""
Robust JSON extraction for LLM outputs.
Handles <think> tags, markdown fences, and conversational prose.
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger("great_sage.parser")


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """
    Greedily extract the largest JSON object from a string.
    Finds the first '{' and the last '}'.
    """
    if not text:
        return None

    # 1. Look for <think> tags and strip them (or log them)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
        if thought:
            logger.debug(f"Extracted thought: {thought[:100]}...")
            # We don't strip it yet, we just note it.
            # The greedy matcher below will jump past it anyway.

    # 2. Greedy match for a JSON object
    # From first '{' to last '}'
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in text")
        return None

    json_str = match.group(1).strip()

    # 3. Handle common markdown pollution inside the greedy match
    # Sometimes models do { ... } ```json { ... } ```
    # or wrap parts of the object. We'll try to just parse the whole block first.
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 4. Fallback: try to strip common markdown fences if they got caught inside
        # (Rare, but happens with messy models)
        cleaned = re.sub(r"```(?:json)?\n?", "", json_str)
        cleaned = re.sub(r"```", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extracted JSON: {e}\nRaw segment: {json_str[:200]}...")
            return None


def extract_thought(text: str) -> Optional[str]:
    """Extract content inside <think> tags if present."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
