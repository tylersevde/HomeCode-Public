"""Exercise local-hook installation and preservation using isolated Git repos."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "tools" / "install_safeguard_hooks.py"
SCANNER = '''import json, os, sys
with open(os.environ["TEST_SCAN_LOG"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("TEST_SCAN_EXIT", "0")))
'''
PREVIOUS = '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["TEST_HOOK_LOG"], "a") as out:
    out.write(json.dumps({"name": os.path.basename(sys.argv[0]), "args": sys.argv[1:], "stdin": sys.stdin.read()}) + "\\n")
sys.exit(int(os.environ.get("TEST_PREVIOUS_EXIT", "0")))
'''


class HookInstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir()
        self.env = os.environ.copy()
        self.env.update({"HOME": self.tmp.name, "GIT_CONFIG_NOSYSTEM": "1",
                         "GIT_CONFIG_GLOBAL": str(Path(self.tmp.name) / "global-config"),
                         "TEST_SCAN_LOG": str(Path(self.tmp.name) / "scans.jsonl"),
                         "TEST_HOOK_LOG": str(Path(self.tmp.name) / "previous.jsonl")})
        self.git("init", "--quiet")
        self.git("config", "user.name", "Hook Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.root / "tools").mkdir()
        (self.root / "tools" / "repository_safeguards.py").write_text(SCANNER)
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "initial")
        self.initial = self.git("rev-parse", "HEAD").stdout.strip()

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.root, env=self.env,
                              text=True, capture_output=True, check=check)

    def installer(self, action, check=True):
        return subprocess.run([sys.executable, str(INSTALLER), action, "--repo", str(self.root)],
                              env=self.env, text=True, capture_output=True, check=check)

    def previous(self, directory, name):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_text(PREVIOUS)
        path.chmod(0o755)
        return path

    def hook(self, name, *args, data=""):
        hooks = Path(self.git("config", "--get", "core.hooksPath").stdout.strip())
        return subprocess.run([str(hooks / name), *args], cwd=self.root, env=self.env,
                              input=data, text=True, capture_output=True)

    def records(self, variable):
        path = Path(self.env[variable])
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_existing_hooks_unchanged_and_pre_push_stdin_replayed(self):
        original = self.previous(self.root / ".git/hooks", "pre-push")
        before = original.read_bytes()
        self.installer("install")
        payload = f"refs/heads/main {self.initial} refs/heads/main {self.initial}\n"
        result = self.hook("pre-push", "other-remote", "https://example.invalid/repo.git", data=payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(self.records("TEST_SCAN_LOG"), [["--head", self.initial, "--base", self.initial]])
        self.assertEqual(self.records("TEST_HOOK_LOG")[0]["stdin"], payload)
        self.assertEqual(self.records("TEST_HOOK_LOG")[0]["args"], ["other-remote", "https://example.invalid/repo.git"])

    def test_every_ref_and_unknown_remote_history_are_checked(self):
        self.installer("install")
        zeros = "0" * 40
        unknown = "1" * 40
        payload = (f"refs/heads/one {self.initial} refs/heads/one {zeros}\n"
                   f"refs/heads/two {self.initial} refs/heads/two {unknown}\n"
                   f"(delete) {zeros} refs/heads/gone {self.initial}\n")
        self.assertEqual(self.hook("pre-push", "backup", "/tmp/remote", data=payload).returncode, 0)
        self.assertEqual(self.records("TEST_SCAN_LOG"), [["--head", self.initial, "--all-history"]] * 2)

    def test_staged_mode_and_existing_hook_failure_propagated(self):
        self.previous(self.root / ".git/hooks", "pre-commit")
        self.installer("install")
        self.env["TEST_PREVIOUS_EXIT"] = "7"
        self.assertEqual(self.hook("pre-commit").returncode, 7)
        self.assertEqual(self.records("TEST_SCAN_LOG"), [["--staged"]])

    def test_scanner_failure_blocks_before_existing_hook(self):
        self.previous(self.root / ".git/hooks", "pre-commit")
        self.installer("install")
        self.env["TEST_SCAN_EXIT"] = "9"
        self.assertEqual(self.hook("pre-commit").returncode, 9)
        self.assertEqual(self.records("TEST_HOOK_LOG"), [])

    def test_missing_scanner_and_malformed_push_fail_closed(self):
        self.installer("install")
        self.assertNotEqual(self.hook("pre-push", "origin", "unused", data="invalid\n").returncode, 0)
        (self.root / "tools/repository_safeguards.py").unlink()
        self.assertNotEqual(self.hook("pre-commit").returncode, 0)

    def test_inherited_hooks_path_and_other_hooks_preserved_with_rollback(self):
        global_hooks = Path(self.tmp.name) / "global hooks"
        self.previous(global_hooks, "post-checkout")
        self.git("config", "--global", "core.hooksPath", str(global_hooks))
        self.installer("install")
        self.assertEqual(self.hook("post-checkout", "a", "b", "1").returncode, 0)
        self.assertEqual(self.records("TEST_HOOK_LOG")[0]["args"], ["a", "b", "1"])
        self.installer("uninstall")
        self.assertEqual(self.git("config", "--get", "core.hooksPath").stdout.strip(), str(global_hooks))
        self.assertEqual(self.git("config", "--local", "--get", "core.hooksPath", check=False).returncode, 1)

    def test_relative_path_restored_and_install_is_idempotent(self):
        self.previous(self.root / "custom-hooks", "pre-commit")
        self.git("config", "core.hooksPath", "custom-hooks")
        self.installer("install")
        state_file = self.root / ".git/repository-safeguards/state.json"
        before = state_file.read_bytes()
        self.installer("install")
        self.assertEqual(state_file.read_bytes(), before)
        self.installer("uninstall")
        self.assertEqual(self.git("config", "--local", "--get", "core.hooksPath").stdout.strip(), "custom-hooks")
        self.installer("install")
        self.assertTrue(json.loads(self.installer("status").stdout)["active"])

    def test_rollback_refuses_to_overwrite_new_configuration(self):
        self.installer("install")
        self.git("config", "core.hooksPath", "new-hooks")
        self.assertNotEqual(self.installer("uninstall", check=False).returncode, 0)
        self.assertEqual(self.git("config", "--get", "core.hooksPath").stdout.strip(), "new-hooks")


if __name__ == "__main__":
    unittest.main()
