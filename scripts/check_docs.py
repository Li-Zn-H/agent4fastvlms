"""Check maintained Markdown without model packages, local assets or network I/O."""

from collections import Counter
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ONLY = {"models", "data", "artifacts", "logs"}
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LINK = re.compile(r"\[[^\]\n]+\]\(([^)\n]+)\)")


def maintained_docs():
    paths = set(ROOT.glob("*.md"))
    for directory in ("docs", "configs", "search", "skills", "adapters", "scripts", "examples", ".github"):
        paths.update((ROOT / directory).rglob("*.md"))
    paths.add(ROOT / "results/README.md")
    return sorted(paths)


def prose_lines(text, label, errors):
    """Remove fenced blocks, checking their closing delimiter along the way."""
    fence = None
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = None
            continue
        if fence is None:
            result.append((number, line))
    if fence:
        errors.append(f"{label}: unclosed code fence")
    return result


def heading_anchors(lines):
    counts = Counter()
    anchors = set()
    for _, line in lines:
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", "", match[1]).lower()
        slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
        index = counts[slug]
        counts[slug] += 1
        anchors.add(f"{slug}-{index}" if index else slug)
    return anchors


def check_goal(errors):
    text = (ROOT / "GOAL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if len(blocks) != 1:
        errors.append("GOAL.md: expected one JSON input block")
        return
    try:
        goal = json.loads(blocks[0])
        assert set(goal) == {"model", "hardware"}
        assert isinstance(goal["model"], str) and goal["model"].strip()
        hw = goal["hardware"]
        assert set(hw) == {"platform", "device"}
        assert isinstance(hw["platform"], str) and hw["platform"].strip()
        assert type(hw["device"]) is int and hw["device"] >= 0
    except (AssertionError, KeyError, TypeError, ValueError):
        errors.append("GOAL.md: inputs must be model plus hardware platform/device")


def main():
    errors = []
    docs = maintained_docs()
    parsed = {}
    anchors = {}
    for path in docs:
        label = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if HAN.search(line):
                errors.append(f"{label}:{number}: untranslated Han text")
            if re.search(r"/(?:home|mnt|Users)/", line):
                errors.append(f"{label}:{number}: machine-specific absolute path")
        parsed[path] = prose_lines(text, label, errors)
        anchors[path] = heading_anchors(parsed[path])

    for path, lines in parsed.items():
        label = path.relative_to(ROOT).as_posix()
        for number, line in lines:
            for match in LINK.finditer(line):
                target = match[1].strip()
                # Optional link titles and angle-bracket paths are supported.
                target = target[1:target.index(">")] if target.startswith("<") else target.split()[0]
                link = urlsplit(target)
                if link.scheme or link.netloc:
                    continue  # Network and mail links are not fetched by this check.
                destination = (path.parent / unquote(link.path)).resolve() if link.path else path
                where = f"{label}:{number}"
                if not destination.is_relative_to(ROOT):
                    errors.append(f"{where}: link escapes source tree: {target}")
                    continue
                relative = destination.relative_to(ROOT)
                parts = relative.parts
                if parts and (parts[0] in LOCAL_ONLY or parts[0].startswith(".venv")
                              or (parts[0] == "results" and relative.as_posix() != "results/README.md")):
                    errors.append(f"{where}: link to excluded local asset; use a code path: {target}")
                elif not destination.exists():
                    errors.append(f"{where}: missing local link: {target}")
                elif link.fragment and destination.suffix == ".md":
                    if unquote(link.fragment) not in anchors.get(destination, set()):
                        errors.append(f"{where}: missing Markdown anchor: {target}")

    check_goal(errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {len(docs)} Markdown files: English coverage, local links, fences and GOAL inputs passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
