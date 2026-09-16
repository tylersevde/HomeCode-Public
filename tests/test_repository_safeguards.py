"""Temporary Git fixtures; real pinned scanner, no network or application builds."""

from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "repository_safeguards", Path(__file__).resolve().parents[1] / "tools/repository_safeguards.py"
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class RepositorySafeguardsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="safeguards-test-")
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repository"
        self.repo.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Safeguards test")
        self.git("config", "user.email", "safeguards@example.invalid")
        self.git("config", "core.hooksPath", "/dev/null")
        self.write("README.md", "Fixture repository.\n")
        self.commit()
        self.initial = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL, text=True)

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self):
        self.git("add", "--all")
        self.git("commit", "-qm", "Synthetic test state")

    def check(self, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            status = guard.check(self.repo, **kwargs)
        return status, output.getvalue()

    @staticmethod
    def token():
        # Created only inside disposable tests, never a real GitHub credential.
        return "ghp_" + secrets.token_hex(18)

    @staticmethod
    def synthetic_hex():
        # A generated permutation keeps entropy deterministic for generic rules.
        return "".join(format((index * 7 + 3) % 16, "x") for index in range(32))

    def test_clean_changed_syntax_and_private_archives_are_allowed(self):
        self.write("private/archives/valid.py", "answer = 42\n")
        self.write("settings.json", '{"enabled": true}\n')
        self.write("project.toml", '[project]\nname="fixture"\n')
        self.commit()
        self.assertEqual(self.check(base=self.initial)[0], 0)

    def test_staged_index_is_checked_even_if_working_copy_is_fixed(self):
        self.write("module.py", "if True\n    pass\n")
        self.git("add", "module.py")
        self.write("module.py", "if True:\n    pass\n")
        status, output = self.check(staged=True)
        self.assertEqual(status, 1)
        self.assertIn("invalid .py syntax", output)

    def test_invalid_json_and_toml_and_conflict_markers_block(self):
        self.write("bad.json", '{"broken": }\n')
        self.write("bad.toml", "item=1\nitem=2\n")
        self.write("conflict.txt", "before\n<<<<<<< branch\nours\n=======\ntheirs\n>>>>>>> main\n")
        self.git("add", "--all")
        status, output = self.check(staged=True)
        self.assertEqual(status, 1)
        self.assertIn("invalid .json syntax", output)
        self.assertIn("invalid .toml syntax", output)
        self.assertIn("conflict marker", output)

    def test_existing_legacy_syntax_is_not_reparsed(self):
        self.write("legacy.py", "old intentionally incomplete source\n")
        self.commit()
        base = self.git("rev-parse", "HEAD").strip()
        self.write("README.md", "Document-only edit.\n")
        self.commit()
        self.assertEqual(self.check(base=base)[0], 0)
        self.assertEqual(self.check()[0], 0)

    def test_credential_filename_blocks_but_empty_template_is_allowed(self):
        self.write("config/.env.example", "TOKEN=\n")
        self.git("add", "--all")
        self.assertEqual(self.check(staged=True)[0], 0)
        self.write("config/.env.production", "TOKEN=\n")
        self.git("add", "--all")
        self.assertEqual(self.check(staged=True)[0], 1)

    def test_added_then_deleted_secret_is_found_in_submitted_history(self):
        token = self.token()
        self.write("temporary.txt", "token=" + token + "\n")
        self.commit()
        (self.repo / "temporary.txt").unlink()
        self.commit()
        status, output = self.check(base=self.initial)
        self.assertEqual(status, 1)
        self.assertIn("submitted-history returned exit 1", output)
        self.assertNotIn(token, output)

    def test_unchanged_secret_is_found_in_resulting_tree(self):
        token = self.token()
        self.write("existing.txt", "token=" + token + "\n")
        self.commit()
        base = self.git("rev-parse", "HEAD").strip()
        self.write("README.md", "Only the README changed.\n")
        self.commit()
        status, output = self.check(base=base)
        self.assertEqual(status, 1)
        self.assertIn("tracked-tree returned exit 1", output)
        self.assertNotIn(token, output)

    def test_inline_allow_and_repository_config_cannot_hide_secret(self):
        token = self.token()
        self.write(".gitleaks.toml", '[allowlist]\npaths=[".*"]\n')
        self.write("example.txt", "token=" + token + " # gitleaks:allow\n")
        self.git("add", "--all")
        status, output = self.check(staged=True)
        self.assertEqual(status, 1)
        self.assertNotIn(token, output)

    def test_symlink_does_not_read_files_outside_repository(self):
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_text(self.token(), encoding="utf-8")
        (self.repo / "reference.txt").symlink_to(outside)
        self.git("add", "--all")
        self.assertEqual(self.check(staged=True)[0], 0)

    def test_missing_scanner_fails_without_network(self):
        with patch.object(guard, "scanner_location", return_value=(self.repo / "absent", "arm64")):
            with patch.object(guard.urllib.request, "urlopen", side_effect=AssertionError("network used")):
                with self.assertRaisesRegex(guard.CheckError, "unavailable"):
                    self.check(staged=True)

    def test_modified_scanner_fails_checksum(self):
        location = self.repo / "wrong-scanner"
        location.write_bytes(b"unexpected executable")
        with patch.object(guard, "scanner_location", return_value=(location, "arm64")):
            with self.assertRaisesRegex(guard.CheckError, "checksum mismatch"):
                guard.scanner()

    def test_wrong_download_checksum_cannot_install(self):
        with patch.object(guard, "scanner_location", return_value=(self.repo / "absent", "arm64")):
            with patch.object(guard.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong archive")):
                with self.assertRaisesRegex(guard.CheckError, "archive failed checksum"):
                    guard.scanner(install=True)

    def test_scanner_runtime_failure_is_not_success(self):
        completion = subprocess.CompletedProcess([], 2, b"sensitive stdout", b"sensitive stderr")
        scratch = Path(self.temporary.name)
        output = io.StringIO()
        with patch.object(guard.subprocess, "run", return_value=completion), redirect_stdout(output):
            self.assertEqual(guard.scan(Path("unused"), self.repo, scratch, "fixture", ["dir", str(self.repo)]), 1)
        self.assertNotIn("sensitive", output.getvalue())

    def test_scanner_zero_without_report_is_not_success(self):
        completion = subprocess.CompletedProcess([], 0, b"", b"")
        with patch.object(guard.subprocess, "run", return_value=completion):
            with self.assertRaisesRegex(guard.CheckError, "no completion report"):
                guard.scan(Path("unused"), self.repo, Path(self.temporary.name), "fixture", ["dir", str(self.repo)])

    def test_new_branch_event_uses_all_history_and_push_requires_exact_head(self):
        event_path = Path(self.temporary.name) / "event.json"
        event_path.write_text(json.dumps({"before": "0" * 40, "after": self.initial}), encoding="utf-8")
        self.assertEqual(guard.event_range(self.repo, event_path), (None, self.initial))
        event_path.write_text(json.dumps({"before": self.initial, "after": "f" * 40}), encoding="utf-8")
        with self.assertRaisesRegex(guard.CheckError, "does not match"):
            guard.event_range(self.repo, event_path)

    def test_audited_exception_is_exact_value_path_line_context_and_rule(self):
        token = self.synthetic_hex()
        relative = "tests/synthetic.py"
        source = "# reviewed fixture\nvalue = 1\ntoken = '" + token + "'\nassert value\n# end\n"
        self.write(relative, source)
        lines = source.encode().splitlines()
        policy_path = Path(self.temporary.name) / "exceptions.json"
        entry = {
            "id": "test-only-reviewed-fixture", "rule_id": "generic-api-key",
            "path": relative, "line": 3, "context_start": 1, "context_end": 5,
            "line_sha256": hashlib.sha256(lines[2]).hexdigest(),
            "context_sha256": hashlib.sha256(b"\n".join(lines)).hexdigest(),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "historical_fingerprints": [],
        }
        policy_path.write_text(json.dumps({"version": 1, "exceptions": [entry]}), encoding="utf-8")
        self.git("add", "--all")
        with patch.object(guard, "EXCEPTIONS", policy_path):
            self.assertEqual(self.check(staged=True)[0], 0)
            self.write(relative, source.replace(token, token[::-1]))
            self.git("add", "--all")
            self.assertEqual(self.check(staged=True)[0], 1)
            self.write(relative, source)
            self.write("other.py", source)
            self.git("add", "--all")
            self.assertEqual(self.check(staged=True)[0], 1)
            (self.repo / "other.py").unlink()
            self.write(relative, "# moved line\n" + source)
            self.git("add", "--all")
            self.assertEqual(self.check(staged=True)[0], 1)
            self.write(relative, source.replace("value = 1", "value = 2"))
            self.git("add", "--all")
            self.assertEqual(self.check(staged=True)[0], 1)
            fake_finding = {"RuleID": "another-rule", "StartLine": 3, "EndLine": 3, "File": relative}
            self.assertIsNone(guard.audited_exception(self.repo, self.repo, fake_finding))

    def test_history_exception_requires_exact_reviewed_commit(self):
        policy_path = Path(self.temporary.name) / "exceptions.json"
        self.write("fixture.txt", "token = '" + self.synthetic_hex() + "'\n")
        self.commit()
        source = (self.repo / "fixture.txt").read_bytes().splitlines()[0]
        digest = hashlib.sha256(source).hexdigest()
        entry = {
            "id": "test-only-reviewed-fixture", "rule_id": "generic-api-key",
            "path": "fixture.txt", "line": 1, "context_start": 1, "context_end": 1,
            "line_sha256": digest, "context_sha256": digest, "historical_fingerprints": [],
            "source_sha256": hashlib.sha256((self.repo / "fixture.txt").read_bytes()).hexdigest(),
        }
        policy_path.write_text(json.dumps({"version": 1, "exceptions": [entry]}), encoding="utf-8")
        with patch.object(guard, "EXCEPTIONS", policy_path):
            self.assertEqual(self.check(base=self.initial)[0], 1)
            commit = self.git("rev-parse", "HEAD").strip()
            entry["historical_fingerprints"] = [f"{commit}:fixture.txt:generic-api-key:1"]
            policy_path.write_text(json.dumps({"version": 1, "exceptions": [entry]}), encoding="utf-8")
            self.assertEqual(self.check(base=self.initial)[0], 0)

    def test_compile_warning_never_prints_secret_source(self):
        token = self.token()
        self.write("warning.py", "token = '" + token + "'; assert 'x' is 1\n")
        self.git("add", "--all")
        stderr = io.StringIO()
        previous = Path.cwd()
        try:
            os.chdir(self.repo)
            with redirect_stderr(stderr):
                status, stdout = self.check(staged=True)
        finally:
            os.chdir(previous)
        self.assertEqual(status, 1)
        self.assertNotIn(token, stdout + stderr.getvalue())
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
