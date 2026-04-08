"""
Tool execution for GREAT SAGE Phase 3.

Handles:
- CodeRun: subprocess execution with pytest output parsing
- Search: DuckDuckGo free web search
- Visit: URL fetch with DOM tree decomposition
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse
import ipaddress
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("great_sage.tools")
REPO_ROOT = Path(".").resolve()
DISALLOWED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bsudo\b",
    r"\bchmod\s+777\b",
    r"\bchown\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bcurl\b.*\|\s*(bash|sh)",
    r"\bwget\b.*\|\s*(bash|sh)",
    r":\(\)\s*\{",
]


# ---------------------------------------------------------------------------
# CodeRun — subprocess with pytest parser
# ---------------------------------------------------------------------------

@dataclass
class CodeRunResult:
    stdout: str
    stderr: str
    exit_code: int
    tests_passed: list[str]
    tests_failed: list[str]
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "CodeRun",
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "duration_ms": self.duration_ms,
        }

    @property
    def all_tests_pass(self) -> bool:
        return (
            self.exit_code == 0
            and len(self.tests_failed) == 0
            and len(self.tests_passed) > 0
        )


def execute_code_run(
    command: str,
    working_dir: str = ".",
    timeout_secs: int = 30,
) -> CodeRunResult:
    """
    Execute a shell command in subprocess with timeout.
    Parses pytest output for test pass/fail counts.
    """
    start = time.monotonic()
    safe, reason = _validate_command(command)
    if not safe:
        return CodeRunResult("", f"Blocked unsafe command: {reason}", -3, [], [], 0)

    safe_working_dir = _safe_working_dir(working_dir)
    if safe_working_dir is None:
        return CodeRunResult(
            "",
            f"Blocked working directory outside repository: {working_dir}",
            -4,
            [],
            [],
            0,
        )

    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            cwd=str(safe_working_dir),
            env=_safe_env(),
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        tests_passed, tests_failed = parse_test_output(stdout + "\n" + stderr)

        return CodeRunResult(
            stdout=stdout[-5000:],  # cap at 5KB
            stderr=stderr[-2000:],  # cap at 2KB
            exit_code=proc.returncode,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired:
        duration_ms = int((time.monotonic() - start) * 1000)
        return CodeRunResult(
            stdout="",
            stderr=f"TIMEOUT after {timeout_secs}s",
            exit_code=-1,
            tests_passed=[],
            tests_failed=[],
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return CodeRunResult(
            stdout="",
            stderr=str(e),
            exit_code=-2,
            tests_passed=[],
            tests_failed=[],
            duration_ms=duration_ms,
        )


def _safe_env() -> dict[str, str]:
    keep_keys = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONPATH"}
    env = {k: v for k, v in os.environ.items() if k in keep_keys}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _safe_working_dir(working_dir: str) -> Path | None:
    resolved = (REPO_ROOT / working_dir).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
        return resolved
    except ValueError:
        return None


def _validate_command(command: str) -> tuple[bool, str]:
    if not command.strip():
        return False, "empty command"
    if len(command) > 500:
        return False, "command too long"
    if "\n" in command or "\r" in command:
        return False, "multiline command not allowed"
    for pat in DISALLOWED_COMMAND_PATTERNS:
        if re.search(pat, command, re.IGNORECASE):
            return False, f"disallowed pattern matched: {pat}"
    return True, ""


def _is_private_or_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return normalized.endswith(".local")


def parse_test_output(output: str) -> tuple[list[str], list[str]]:
    """
    Parse test output to extract passed/failed test names.
    Supports pytest, cargo test, go test, and jest-like summaries.
    """
    passed: list[str] = []
    failed: list[str] = []

    # Match individual test results: "test_file.py::test_name PASSED/FAILED"
    for match in re.finditer(
        r"([\w/]+\.py::[\w\[\]-]+)\s+(PASSED|FAILED|ERROR)", output
    ):
        test_name = match.group(1)
        status = match.group(2)
        if status == "PASSED":
            passed.append(test_name)
        elif status in ("FAILED", "ERROR"):
            failed.append(test_name)

    # cargo test lines: "test module::name ... ok/FAILED"
    for match in re.finditer(
        r"test\s+([A-Za-z0-9_:]+)\s+\.\.\.\s+(ok|FAILED)", output
    ):
        test_name = match.group(1)
        status = match.group(2)
        if status == "ok":
            passed.append(test_name)
        else:
            failed.append(test_name)

    # go test lines: "--- PASS|FAIL: TestName"
    for match in re.finditer(r"---\s+(PASS|FAIL):\s+([A-Za-z0-9_./-]+)", output):
        status = match.group(1)
        test_name = match.group(2)
        if status == "PASS":
            passed.append(test_name)
        else:
            failed.append(test_name)

    # Jest/Vitest style line items: "✓ test name" or "✕ test name"
    for match in re.finditer(r"^\s*([✓✕])\s+(.+)$", output, re.MULTILINE):
        status = match.group(1)
        test_name = match.group(2).strip()
        if status == "✓":
            passed.append(test_name)
        else:
            failed.append(test_name)

    # Fallback summary extraction from common runners.
    if not passed and not failed:
        pass_count = _extract_count(
            output,
            [
                r"(\d+)\s+passed",
                r"(\d+)\s+tests?\s+passed",
                r"test result: ok\.\s+(\d+)\s+passed",
            ],
        )
        fail_count = _extract_count(
            output,
            [
                r"(\d+)\s+failed",
                r"(\d+)\s+tests?\s+failed",
                r"test result: .*?(\d+)\s+failed",
            ],
        )
        passed = [f"test_{i}" for i in range(pass_count)]
        failed = [f"test_fail_{i}" for i in range(fail_count)]

    return passed, failed


def _extract_count(output: str, patterns: list[str]) -> int:
    for pattern in patterns:
        summary = re.search(pattern, output, flags=re.IGNORECASE)
        if not summary:
            continue
        value = summary.group(1)
        if value.isdigit():
            return int(value)
    return 0


def parse_pytest_output(output: str) -> tuple[list[str], list[str]]:
    """
    Backward-compatible alias; now delegates to the multi-runner parser.
    """
    return parse_test_output(output)


# ---------------------------------------------------------------------------
# Search — DuckDuckGo (free, no API key)
# ---------------------------------------------------------------------------

async def execute_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    Free web search via DuckDuckGo.
    Returns context-dense snippets from relevant sites.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", ""),
                })

        return {
            "type": "Search",
            "snippets": [
                f"[{r['title']}]({r['url']})\n{r['snippet']}"
                for r in results
            ],
            "raw_results": results,
        }

    except ImportError:
        logger.warning("duckduckgo-search not installed, falling back to empty results")
        return {"type": "Search", "snippets": [], "raw_results": []}

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"type": "Search", "snippets": [str(e)], "raw_results": []}


# ---------------------------------------------------------------------------
# Visit — URL fetch + DOM tree decomposition
# ---------------------------------------------------------------------------

@dataclass
class PageElement:
    """Structured element from a web page DOM tree."""
    tag: str
    text: str | None
    href: str | None
    children: list["PageElement"]
    is_interactable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "text": self.text,
            "href": self.href,
            "children": [c.to_dict() for c in self.children],
            "is_interactable": self.is_interactable,
        }


INTERACTABLE_TAGS = {"a", "button", "input", "select", "textarea", "form"}
STRUCTURAL_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "td", "th",
    "code", "pre", "blockquote",
    "nav", "main", "article", "section", "aside",
}


def decompose_dom(soup: Any, max_depth: int = 4, depth: int = 0) -> list[PageElement]:
    """
    Decompose HTML into a structured tree of headings, text blocks,
    and interactable elements.  Non-structural/non-interactable tags
    are flattened.
    """
    elements = []

    if depth >= max_depth:
        return elements

    for child in soup.children:
        if not hasattr(child, "name") or child.name is None:
            # Text node
            text = child.strip() if isinstance(child, str) else ""
            if text and len(text) > 2:
                elements.append(PageElement(
                    tag="text",
                    text=text[:500],
                    href=None,
                    children=[],
                    is_interactable=False,
                ))
            continue

        tag = child.name.lower()

        if tag in ("script", "style", "noscript", "svg", "path"):
            continue

        is_interactable = tag in INTERACTABLE_TAGS
        is_structural = tag in STRUCTURAL_TAGS

        if is_interactable or is_structural:
            text = child.get_text(strip=True)[:500] if child.string or child.get_text(strip=True) else None
            href = child.get("href") if tag == "a" else None

            elem = PageElement(
                tag=tag,
                text=text,
                href=str(href) if href else None,
                children=decompose_dom(child, max_depth, depth + 1),
                is_interactable=is_interactable,
            )
            elements.append(elem)
        else:
            # Flatten: recurse into children
            elements.extend(decompose_dom(child, max_depth, depth))

    return elements


async def execute_visit(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """
    Fetch URL, convert to markdown + structured DOM tree.
    Returns both human-readable content and machine-parseable tree.
    """
    try:
        from bs4 import BeautifulSoup
        import html2text

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http/https URLs are allowed")
        if not parsed.hostname:
            raise ValueError("Missing hostname")
        if _is_private_or_loopback_host(parsed.hostname):
            raise ValueError("Private and loopback hosts are blocked")

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, headers={
                "User-Agent": "GreatSage/0.1 (Research Agent)"
            })
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "text/html" in content_type:
            soup = BeautifulSoup(resp.text, "lxml")

            # Remove noise
            for tag in soup.find_all(["script", "style", "noscript"]):
                tag.decompose()

            # Convert to markdown for context density
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0
            markdown = h.handle(str(soup))

            # Build DOM tree
            tree = decompose_dom(soup)

            return {
                "type": "Visit",
                "content": markdown[:10000],  # cap at 10KB
                "tree": [e.to_dict() for e in tree[:50]],  # cap tree size
                "url": url,
                "status_code": resp.status_code,
            }
        else:
            # Non-HTML — return raw text
            return {
                "type": "Visit",
                "content": resp.text[:10000],
                "tree": None,
                "url": url,
                "status_code": resp.status_code,
            }

    except Exception as e:
        logger.error(f"Visit failed for {url}: {e}")
        return {
            "type": "Visit",
            "content": f"Error fetching {url}: {e}",
            "tree": None,
            "url": url,
            "status_code": -1,
        }


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

async def execute_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a tool call to the appropriate executor.
    Returns the tool result dict.
    """
    tc_type = tool_call.get("type", "")

    if tc_type == "CodeRun":
        result = execute_code_run(
            command=tool_call["command"],
            working_dir=tool_call.get("working_dir", "."),
            timeout_secs=tool_call.get("timeout_secs", 30),
        )
        return result.to_dict()

    elif tc_type == "Search":
        return await execute_search(
            query=tool_call.get("query", ""),
            max_results=tool_call.get("max_results", 5),
        )

    elif tc_type == "Visit":
        return await execute_visit(
            url=tool_call.get("url", ""),
            timeout=tool_call.get("timeout", 15.0),
        )

    else:
        return {"type": "Error", "message": f"Unknown tool type: {tc_type}"}
