#!/usr/bin/env python3
"""Install repository-local safeguards without replacing existing Git hooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


# Cover Git's documented hooks, including hooks that are added to the original
# directory after installation. Existing custom hook names are delegated too.
HOOK_NAMES = {
    "applypatch-msg", "pre-applypatch", "post-applypatch", "pre-commit",
    "pre-merge-commit", "prepare-commit-msg", "commit-msg", "post-commit",
    "pre-rebase", "post-checkout", "post-merge", "pre-push", "pre-receive",
    "update", "proc-receive", "post-receive", "post-update", "reference-transaction",
    "push-to-checkout", "pre-auto-gc", "post-rewrite", "sendemail-validate",
    "fsmonitor-watchman", "p4-changelist", "p4-prepare-changelist", "p4-post-changelist",
    "p4-pre-submit", "post-index-change",
}

DISPATCHER = '''#!/usr/bin/env python3
"""Generated Git hook: scan locally, then preserve the previous hook."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def run():
    hook = Path(sys.argv[0]).name
    state = json.loads((Path(__file__).resolve().parent.parent / "state.json").read_text())
    previous = Path(state["previous_hooks_dir"]) / hook
    if hook == "pre-push":
        payload = sys.stdin.buffer.read()
    else:
        payload = None
    if hook in ("pre-commit", "pre-push"):
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        scanner = Path(root) / "tools" / "repository_safeguards.py"
        if not scanner.is_file():
            print("Safeguards blocked this operation: tools/repository_safeguards.py is missing.", file=sys.stderr)
            return 1
        commands = []
        if hook == "pre-commit":
            commands.append([sys.executable, str(scanner), "--staged"])
        else:
            for line in payload.splitlines():
                fields = line.split()
                if len(fields) != 4 or not all(re.fullmatch(rb"[0-9a-f]{40}|[0-9a-f]{64}", fields[index]) for index in (1, 3)):
                    raise ValueError("Invalid pre-push input")
                local_sha, remote_sha = (fields[index].decode("ascii") for index in (1, 3))
                if set(local_sha) == {"0"}:
                    continue
                command = [sys.executable, str(scanner), "--head", local_sha]
                known = set(remote_sha) != {"0"} and subprocess.run(
                    ["git", "cat-file", "-e", remote_sha + "^{commit}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                command += ["--base", remote_sha] if known else ["--all-history"]
                commands.append(command)
        for command in commands:
            result = subprocess.run(command, stdin=subprocess.DEVNULL)
            if result.returncode:
                print("Safeguards blocked this operation. Fix the reported issue and retry.", file=sys.stderr)
                return result.returncode
    if previous.is_file() and os.access(previous, os.X_OK):
        return subprocess.run([str(previous), *sys.argv[1:]], input=payload).returncode
    return 0

try:
    sys.exit(run())
except (OSError, ValueError, subprocess.SubprocessError):
    print("Safeguards hook failed; operation blocked. Run the installer status command.", file=sys.stderr)
    sys.exit(1)
'''


def git(root: Path, *args: str, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode not in allowed:
        raise RuntimeError(f"Git command failed: {' '.join(args)}")
    return result.stdout.rstrip("\n")


def repository(path: Path) -> tuple[Path, Path]:
    root = Path(git(path, "rev-parse", "--show-toplevel")).resolve()
    git_dir = Path(git(root, "rev-parse", "--absolute-git-dir")).resolve()
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")).resolve()
    if git_dir != common or git(root, "worktree", "list", "--porcelain").count("worktree ") != 1:
        raise RuntimeError("Install from a regular clone with no linked worktrees; shared worktree hook configuration is unsupported.")
    return root, git_dir


def local_values(root: Path) -> list[str]:
    result = subprocess.run(["git", "-C", str(root), "config", "--local", "--null", "--get-all", "core.hooksPath"],
                            stdout=subprocess.PIPE, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot read local hooks configuration.")
    return [item.decode() for item in result.stdout.split(b"\0") if item]


def effective_hooks(root: Path) -> Path:
    value = Path(git(root, "rev-parse", "--git-path", "hooks")).expanduser()
    return (root / value).resolve() if not value.is_absolute() else value.resolve()


def install(root: Path, git_dir: Path) -> None:
    area = git_dir / "repository-safeguards"
    hooks = area / "hooks"
    state_path = area / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("installed") and effective_hooks(root) == hooks.resolve():
            print("Repository safeguards hooks are already installed.")
            return
        if state.get("installed"):
            raise RuntimeError("Previous safeguards installation exists; inspect status before changing its configuration.")
    if not (root / "tools" / "repository_safeguards.py").is_file():
        raise RuntimeError("Check out tools/repository_safeguards.py before installing hooks.")
    previous = effective_hooks(root)
    if previous == hooks.resolve():
        raise RuntimeError("Hook path already points at the installation directory without valid metadata.")
    names = HOOK_NAMES | ({item.name for item in previous.iterdir() if item.is_file()} if previous.is_dir() else set())
    area.mkdir(mode=0o700, exist_ok=state_path.exists())
    hooks.mkdir(mode=0o700, exist_ok=state_path.exists())
    state = {
        "version": 1, "installed": False, "repository": str(root),
        "previous_hooks_dir": str(previous), "previous_local_values": local_values(root),
        "installed_hooks_dir": str(hooks.resolve()),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    state_path.chmod(0o600)
    for name in sorted(names):
        destination = hooks / name
        destination.write_text(DISPATCHER)
        destination.chmod(0o700)
    git(root, "config", "--local", "--replace-all", "core.hooksPath", str(hooks.resolve()))
    state["installed"] = True
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("Installed local safeguard hooks; original hooks remain unchanged at:", previous)
    print("Hook execution performs no downloads. Install the pinned scanner separately before committing or pushing.")


def uninstall(root: Path, git_dir: Path) -> None:
    state_path = git_dir / "repository-safeguards" / "state.json"
    state = json.loads(state_path.read_text())
    if not state.get("installed"):
        raise RuntimeError("Safeguards hooks are already uninstalled; original hook files remain untouched.")
    if effective_hooks(root) != Path(state["installed_hooks_dir"]):
        raise RuntimeError("core.hooksPath changed after installation; refusing to overwrite the newer setting.")
    git(root, "config", "--local", "--unset-all", "core.hooksPath", allowed=(0, 5))
    for value in state["previous_local_values"]:
        git(root, "config", "--local", "--add", "core.hooksPath", value)
    state["installed"] = False
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("Restored the previous local core.hooksPath setting. Original hook files were not modified.")
    print("Installation files and rollback metadata were retained at:", state_path.parent)


def status(root: Path, git_dir: Path) -> None:
    state_path = git_dir / "repository-safeguards" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    value = {"repository": str(root), "effective_hooks_dir": str(effective_hooks(root)),
             "saved_installation": state, "active": bool(state and state.get("installed") and
                                                          effective_hooks(root) == Path(state["installed_hooks_dir"]))}
    print(json.dumps(value, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        root, git_dir = repository(args.repo)
        {"install": install, "uninstall": uninstall, "status": status}[args.action](root, git_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Hook installer: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
