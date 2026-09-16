#!/usr/bin/env python3
"""Offline repository checks; use --install-scanner explicitly before first use.

Only Git objects are inspected. No application code is imported or executed.
Secret scanner output is captured, and only redacted finding locations are shown.
Python 3.11+, Git, and the pinned Gitleaks binary are required on Linux ARM64/x64.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import warnings


VERSION = "8.30.1"
# Official release archives; executable hashes verified from those same archives.
HASHES = {
    "arm64": (
        "e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080",
        "00e91bbe655bd7c47753e8cfe61cb76ea1a5d7e7702fe161ee40102b46b3823b",
    ),
    "x64": (
        "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb",
        "88f91962aa2f93ac6ab281d553b9e125f5197bbbce38f9f2437f7299c32e5509",
    ),
}
CONFLICT = re.compile(rb"^(?:<{7}|>{7}|\|{7})(?:\s|$)", re.MULTILINE)
OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
EXCEPTIONS = Path(__file__).with_name("safeguards_exceptions.json")


class CheckError(Exception):
    """An expected failure, with a message safe for logs."""


def git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise CheckError("Git could not read the requested repository state.")
    return completed.stdout


def resolve(repo: Path, revision: str) -> str:
    # Do not pass user-provided log options or revision expressions to Gitleaks.
    if revision.startswith("-"):
        raise CheckError("Invalid revision.")
    value = git(repo, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}").decode().strip()
    if not OID.fullmatch(value):
        raise CheckError("Git returned an invalid commit identifier.")
    return value


def scanner_location() -> tuple[Path, str]:
    if platform.system() != "Linux":
        raise CheckError("This scanner setup supports Linux ARM64 and x64 only.")
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x64", "amd64": "x64"}.get(platform.machine().lower())
    if not arch:
        raise CheckError("This scanner setup supports Linux ARM64 and x64 only.")
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return cache / "homecode-safeguards" / VERSION / arch / "gitleaks", arch


def scanner(install: bool = False) -> Path:
    target, arch = scanner_location()
    archive_hash, binary_hash = HASHES[arch]
    if target.is_file() and not target.is_symlink():
        if hashlib.sha256(target.read_bytes()).hexdigest() == binary_hash:
            return target
        if not install:
            raise CheckError("Scanner checksum mismatch; run --install-scanner to restore it.")
    if not install:
        raise CheckError("Pinned scanner unavailable. Run: python3 -B tools/repository_safeguards.py --install-scanner")
    url = f"https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}/gitleaks_{VERSION}_linux_{arch}.tar.gz"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = response.read(50 * 1024 * 1024 + 1)
        if hashlib.sha256(archive).hexdigest() != archive_hash:
            raise CheckError("Downloaded scanner archive failed checksum verification.")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
            member = package.getmember("gitleaks")
            if not member.isfile():
                raise CheckError("Scanner archive has no regular executable.")
            stream = package.extractfile(member)
            if stream is None:
                raise CheckError("Scanner archive could not be read.")
            binary = stream.read()
        if hashlib.sha256(binary).hexdigest() != binary_hash:
            raise CheckError("Downloaded scanner executable failed checksum verification.")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(binary)
        temporary.chmod(0o700)
        temporary.replace(target)
    except CheckError:
        raise
    except (OSError, KeyError, tarfile.TarError) as error:
        raise CheckError("Scanner installation failed; no checks were skipped.") from error
    return target


def entries(repo: Path, head: str | None) -> list[tuple[str, str, str]]:
    args = ("ls-tree", "-r", "-z", head) if head else ("ls-files", "--stage", "-z")
    found = []
    for record in git(repo, *args).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        path = os.fsdecode(raw_path)
        fields = metadata.decode("ascii").split()
        mode, object_id = fields[0], fields[2] if head else fields[1]
        if not head and fields[2] != "0":
            raise CheckError("Unmerged index entries must be resolved first.")
        if mode == "160000":
            raise CheckError("Submodules need a separate verified scan before use.")
        if mode not in {"100644", "100755", "120000"}:
            raise CheckError("Unsupported Git object mode.")
        parts = PurePosixPath(path).parts
        if not parts or parts[0] == "/" or any(part in {"..", ".git"} for part in parts):
            raise CheckError("Unsafe tracked pathname.")
        if not OID.fullmatch(object_id):
            raise CheckError("Invalid Git blob identifier.")
        found.append((path, object_id, mode))
    return found


def materialize(repo: Path, tree: Path, objects: list[tuple[str, str, str]]) -> None:
    """Export exact blobs, treating symlinks as text rather than following them."""
    process = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert process.stdin is not None and process.stdout is not None
        for path, object_id, _mode in objects:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[0].decode() != object_id or header[1] != b"blob":
                raise CheckError("Git blob export failed.")
            remaining = int(header[2])
            destination = tree / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                while remaining:
                    chunk = process.stdout.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise CheckError("Git blob export ended early.")
                    output.write(chunk)
                    remaining -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise CheckError("Git blob export returned invalid framing.")
        process.stdin.close()
        if process.wait(timeout=30):
            raise CheckError("Git blob export failed.")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()


def credential_path(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    if name.endswith((".example", ".sample", ".template")) or name in {".env.example", ".env.sample", ".env.template"}:
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {".netrc", "_netrc", ".npmrc", ".pypirc", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials.json", "credentials", "secrets.json"}:
        return True
    return name.endswith((".p12", ".pfx", ".key")) or bool(re.fullmatch(r"service[-_]account.*\.json", name))


def structural_checks(tree: Path, objects: list[tuple[str, str, str]], changed: set[str]) -> int:
    errors = 0
    for path, _object_id, mode in objects:
        display = json.dumps(path, ensure_ascii=True)
        if credential_path(path):
            print(f"BLOCKED: credential filename {display}")
            errors += 1
        if path not in changed or mode == "120000":
            continue
        data = (tree / path).read_bytes()
        if b"\0" not in data and CONFLICT.search(data):
            print(f"BLOCKED: conflict marker in {display}")
            errors += 1
        try:
            suffix = PurePosixPath(path).suffix.lower()
            if suffix == ".py":
                # compile() warnings may include a source line through linecache.
                # Treat only syntax errors as failures and never emit source text.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compile(data, path, "exec", dont_inherit=True)
            elif suffix == ".json":
                json.loads(data.decode("utf-8-sig"))
            elif suffix == ".toml":
                tomllib.loads(data.decode("utf-8"))
        except (SyntaxError, ValueError, UnicodeError) as error:
            # Exception text/source snippets can contain credentials; never print them.
            line = getattr(error, "lineno", None)
            location = f" near line {line}" if isinstance(line, int) else ""
            print(f"BLOCKED: invalid {suffix} syntax in {display}{location}")
            errors += 1
    return errors


def audited_exception(repo: Path, tree: Path | None, finding: dict) -> str | None:
    """Filter only unchanged, independently reviewed fixture findings after scanning.

    A scanner failure or an unrecognized finding can never be exempted. Public
    repositories ship no exception policy. Historical exemptions also require an
    exact commit/path/rule/line fingerprint, including historical path locations.
    """
    if not EXCEPTIONS.is_file():
        return None
    policy = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))
    if policy.get("version") != 1 or not isinstance(policy.get("exceptions"), list):
        raise CheckError("Invalid audited exception policy.")
    for entry in policy["exceptions"]:
        line = entry["line"]
        if finding.get("RuleID") != entry["rule_id"] or finding.get("StartLine") != line or finding.get("EndLine") != line:
            continue
        path = finding.get("File", "")
        if tree is not None:
            candidate = Path(path)
            # Gitleaks dir reports absolute paths; permit relative names too.
            if candidate.is_absolute():
                try:
                    path = candidate.relative_to(tree).as_posix()
                except ValueError:
                    continue
            if path != entry["path"]:
                continue
            source = (tree / path).read_bytes()
        else:
            if finding.get("Fingerprint") not in entry["historical_fingerprints"]:
                continue
            commit = finding.get("Commit", "")
            if not OID.fullmatch(commit):
                continue
            source = git(repo, "show", f"{commit}:{path}")
        if hashlib.sha256(source).hexdigest() != entry["source_sha256"]:
            continue
        lines = source.splitlines()
        start, end = entry["context_start"], entry["context_end"]
        if not (1 <= start <= line <= end <= len(lines)):
            continue
        if hashlib.sha256(lines[line - 1]).hexdigest() != entry["line_sha256"]:
            continue
        if hashlib.sha256(b"\n".join(lines[start - 1:end])).hexdigest() != entry["context_sha256"]:
            continue
        return entry["id"]
    return None


def scan(binary: Path, repo: Path, scratch: Path, label: str, args: list[str], *, tree: Path | None = None) -> int:
    report = scratch / f"{label}.json"
    config = scratch / "default-rules.toml"
    config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    ignore = scratch / "empty.gitleaksignore"
    ignore.write_text("", encoding="utf-8")
    command = [str(binary), *args, "--config", str(config), "--gitleaks-ignore-path", str(ignore),
               "--ignore-gitleaks-allow", "--redact=100", "--no-banner", "--no-color",
               "--max-archive-depth=2", "--max-decode-depth=5", "--max-target-megabytes=0",
               "--report-format=json", "--report-path", str(report)]
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GITLEAKS_")}
    try:
        result = subprocess.run(command, cwd=repo, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=420, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckError(f"Secret scanner unavailable or timed out during {label}; check failed.") from error
    if result.returncode not in (0, 1):
        print(f"BLOCKED: secret scan {label} returned exit {result.returncode}.")
        return 1
    if not report.is_file():
        raise CheckError(f"Secret scanner produced no completion report during {label}.")
    try:
        findings = json.loads(report.read_text(encoding="utf-8"))
    except ValueError as error:
        raise CheckError(f"Secret scanner produced an invalid report during {label}.") from error
    if not isinstance(findings, list):
        raise CheckError(f"Secret scanner produced an invalid report during {label}.")
    if result.returncode == 0 and findings:
        raise CheckError(f"Secret scanner reported findings despite success during {label}.")
    if result.returncode == 1 and not findings:
        raise CheckError(f"Secret scanner failed without findings during {label}.")
    remaining = []
    for finding in findings:
        if not isinstance(finding, dict):
            raise CheckError(f"Secret scanner produced an invalid finding during {label}.")
        exception_id = audited_exception(repo, tree, finding)
        if exception_id:
            print(f"Reviewed synthetic fixture exception: {exception_id} ({label}).")
        else:
            remaining.append(finding)
    if remaining:
        print(f"BLOCKED: secret scan {label} returned exit 1.")
        for finding in remaining[:100]:
            safe = {key: finding.get(key) for key in ("RuleID", "File", "StartLine", "Commit")}
            print("Finding location: " + json.dumps(safe, ensure_ascii=True))
        return 1
    print(f"PASS: secret scan {label}.")
    return 0


def check(repo: Path, *, staged: bool = False, base: str | None = None, head: str = "HEAD") -> int:
    repo = Path(os.fsdecode(git(repo, "rev-parse", "--show-toplevel")).strip())
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise CheckError("Full history is required; fetch with depth 0 before checking.")
    binary = scanner()
    resolved_head = None if staged else resolve(repo, head)
    resolved_base = resolve(repo, base) if base else None
    if staged:
        changed_raw = git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")
    elif resolved_base:
        changed_raw = git(repo, "diff", "--name-only", "--diff-filter=ACMRT", "-z", resolved_base, resolved_head)
    else:
        # Without a published base, scan all reachable history for secrets but
        # check syntax only in tip changes, preserving unrelated legacy archives.
        changed_raw = git(repo, "diff-tree", "--root", "--no-commit-id", "--name-only", "--diff-filter=ACMRT", "-r", "-m", "-z", resolved_head)
    changed = {os.fsdecode(path) for path in changed_raw.split(b"\0") if path}
    objects = entries(repo, resolved_head)
    with tempfile.TemporaryDirectory(prefix="homecode-safeguards-") as temporary:
        scratch = Path(temporary)
        tree = scratch / "tree"
        tree.mkdir()
        materialize(repo, tree, objects)
        failures = structural_checks(tree, objects, changed)
        if not staged:
            revision_range = f"{resolved_base}..{resolved_head}" if resolved_base else str(resolved_head)
            failures += scan(binary, repo, scratch, "submitted-history", ["git", str(repo), f"--log-opts=--full-history --no-renames -m {revision_range}"])
        failures += scan(binary, repo, scratch, "tracked-tree", ["dir", str(tree)], tree=tree)
    if failures:
        print("Repository safeguards FAILED; no bypass was applied.")
        return 1
    print(f"Repository safeguards PASS ({len(objects)} tracked files; {len(changed)} changed paths).")
    return 0


def event_range(repo: Path, event_path: Path) -> tuple[str | None, str]:
    event = json.loads(event_path.read_text(encoding="utf-8"))
    head = resolve(repo, "HEAD")
    if "pull_request" in event:
        base = event["pull_request"]["base"]["sha"]
    elif "before" in event and "after" in event:
        if event["after"] != head:
            raise CheckError("Checkout does not match the pushed commit.")
        base = event["before"]
    elif os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        base = None
    else:
        raise CheckError("Unsupported workflow event; no scan was skipped.")
    if base is not None and not OID.fullmatch(base):
        raise CheckError("Workflow event has an invalid base commit.")
    return (None if not base or set(base) == {"0"} else base), head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install-scanner", action="store_true")
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--base", help="Previously published commit; scans base..head and the head tree")
    mode.add_argument("--all-history", action="store_true", help="Scan all commits reachable from head and the head tree")
    mode.add_argument("--event", type=Path, help="GitHub event JSON (PR, main push, or manual run)")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.install_scanner:
            scanner(install=True)
            print(f"Pinned Gitleaks {VERSION} installed and checksum verified.")
            return 0
        if args.event:
            base, head = event_range(args.repo, args.event)
            return check(args.repo, base=base, head=head)
        return check(args.repo, staged=args.staged, base=args.base, head=args.head)
    except (CheckError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        # Only our own carefully bounded messages are safe to expose.
        detail = str(error) if isinstance(error, CheckError) else "A required input or tool failed; no checks were skipped."
        print("Repository safeguards FAILED: " + detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
