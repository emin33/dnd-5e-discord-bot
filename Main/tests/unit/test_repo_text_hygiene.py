"""No stray control characters in tracked source, docs or config.

This exists because a documented command silently became unrunnable. A README
was written containing ``Main`` + backslash + ``run_tests.bat``; the backslash-r
was consumed as an escape, and what landed on disk was ``Main<CR>un_tests.bat``
-- a real carriage return sitting mid-line. It renders as something plausible,
survives review by eye, and instructs the reader to run a path that does not
exist. A near-identical near-miss followed hours later on a Windows path in
``.env.example`` (backslash-t, backslash-f).

Nothing else in the project would have caught either one: they are not syntax
errors, not test failures, and not type errors. They are bytes. So this checks
the bytes.

Legal here: TAB (0x09) and LF (0x0A), plus CR (0x0D) *only* as the first half of
a CRLF line ending -- this repo is Windows-authored and most tracked files are
CRLF. Anything else below 0x20 is a control character that got into a text file
by accident.
"""

import subprocess
from pathlib import Path

import pytest

# tests/unit/ -> tests/ -> Main/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]

_TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".bat",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
    ".sql",
    ".cfg",
    ".ini",
    ".txt",
    ".json",
    ".example",
}

# Generated capture logs -- arbitrary model output, not authored text.
_SKIP_PREFIXES = ("Main/data/",)


def _tracked_text_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"`git ls-files` failed in {_REPO_ROOT}: {result.stderr.strip()}. "
            "This check reads the tracked file list; it deliberately does not "
            "skip when it cannot, because a silent skip is how the defect it "
            "guards against shipped in the first place."
        )
    return [
        path
        for path in (line.strip() for line in result.stdout.splitlines())
        if path
        and Path(path).suffix in _TEXT_SUFFIXES
        and not path.startswith(_SKIP_PREFIXES)
    ]


def _control_characters(data: bytes) -> list[tuple[int, int]]:
    """Return (offset, byte) for every disallowed control character."""
    offenders = []
    for index, byte in enumerate(data):
        if byte >= 0x20 or byte in (0x09, 0x0A):
            continue
        if byte == 0x0D and data[index + 1 : index + 2] == b"\n":
            continue  # CRLF line ending, which this repo uses throughout
        offenders.append((index, byte))
    return offenders


class TestNoStrayControlCharacters:
    def test_tracked_text_files_have_no_stray_control_characters(self):
        tracked = _tracked_text_files()
        assert tracked, "no tracked text files found -- the scan is not looking at anything"

        report: list[str] = []
        for relative_path in tracked:
            data = (_REPO_ROOT / relative_path).read_bytes()
            for offset, byte in _control_characters(data):
                context = data[max(0, offset - 30) : offset + 30]
                report.append(
                    f"{relative_path}: {hex(byte)} at byte {offset} -- {context!r}"
                )

        assert not report, (
            "Control characters in tracked text files. Almost always a "
            "backslash escape that was consumed when the file was written "
            r"(\r, \t, \f in a Windows path):" + "\n  " + "\n  ".join(report)
        )

    def test_the_scan_actually_detects_a_planted_control_character(self, tmp_path):
        """Positive control.

        An all-clear from a scanner nobody has ever seen fire is not evidence.
        This plants the exact byte sequence that shipped -- ``Main`` + CR +
        ``un_tests.bat`` -- and asserts the detector reports it.
        """
        planted = b"see Main" + bytes([0x0D]) + b"un_tests.bat\r\nfor details\r\n"
        offenders = _control_characters(planted)

        assert len(offenders) == 1, f"expected exactly one offender, got {offenders}"
        offset, byte = offenders[0]
        assert byte == 0x0D
        assert planted[offset + 1 : offset + 2] == b"u", (
            "the planted CR should be the one followed by 'u' (mid-line), not "
            "one of the CRLF line endings"
        )

    def test_crlf_line_endings_are_not_reported(self):
        """Negative control: the common case must stay silent."""
        assert _control_characters(b"first line\r\nsecond line\r\n") == []
        assert _control_characters(b"unix line\nunix line\n") == []
        assert _control_characters(b"tabs\tare\tfine\n") == []
