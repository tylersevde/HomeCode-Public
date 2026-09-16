# GitHub safeguards

## Operating policy and rollout status

Use a branch, open a pull request, review the diff, wait for **Repository
safeguards**, then merge. Zero outside approvals are required for the owner's
own pull requests. Resolve conversations and bring the branch up to date when
required. Automatic merging is disabled. Do not use CI-skipping commit messages
or push backup snapshots directly to `main`.

The rollout targets default-branch rules that require pull requests and the
successful GitHub Actions check, block force pushes and deletion, and provide
no routine administrator bypass. Existing private `history/*` and
`checkpoints/*` branches are preservation references, not backup upload targets.
They receive deletion and force-push protection. Repository files alone do not
activate server-side rules; the owner's saved rollout report records verified
settings, checks, and any remaining account-owner steps.

On 2026-09-16, the account owner confirmed account-wide $0 paid-usage budgets
with **Stop usage when budget limit is reached** for Actions, Codespaces,
Packages, and Git LFS. This is owner confirmation; the current connection cannot
independently inspect personal-account billing settings. Recheck these limits
before changing billing or adding hosted services, including any metered AI
product. A notification-only budget does not block spending. Keep new automation
disabled if the relevant stopping control cannot be verified.
Review existing subscriptions, usage, and storage separately; a new limit does
not erase earlier charges. Do not add paid runners, security trials, packages,
Codespaces, or subscriptions as part of this rollout. See GitHub's
[budgets documentation](https://docs.github.com/en/billing/how-tos/set-up-budgets)
and [limitations](https://docs.github.com/en/billing/concepts/budgets-and-alerts).

## Local checks before any upload

Install Git LFS first when the repository uses it, then explicitly install the
pinned Gitleaks scanner and local hooks in each regular clone:

```bash
python3 tools/repository_safeguards.py --install-scanner
python3 tools/install_safeguard_hooks.py install
python3 tools/install_safeguard_hooks.py status
```

Scanner installation downloads the pinned release and checks its checksum.
Commits and pushes perform no scanner downloads. Missing scanners, checksum
mismatches, scanner errors, malformed hook input, or missing check code block
the operation. Fix the problem and retry; do not bypass checks to finish a
backup. A new clone does not inherit installed Git hooks, so repeat setup.

The owner's legacy Development working clone retains its existing
publication-blocking pre-push hook unchanged. Use the dedicated private and
public safeguards checkouts for the new pull-request workflow. Installing a
dispatcher over a clone with that stronger block would preserve the block;
it does not authorize removing or bypassing it.

The pre-commit hook scans the exact staged contents, including changes that
differ from the working copy. The pre-push hook reads every ref Git proposes to
send to every remote, scans all outgoing commits plus the proposed resulting
tree, and checks the full reachable history when the remote's starting commit
is not available locally. Deleting a ref adds no content to scan. A secret
introduced and removed in an intermediate outgoing commit still blocks a push.

The installer stores its dispatchers and rollback metadata under
`.git/repository-safeguards/`, uses a local `core.hooksPath`, and leaves the
previous hook files byte-for-byte intact. It delegates the previous hooks,
including Git LFS hooks, with their original arguments and pre-push input.
Other Git hooks continue to delegate to the original directory. Inherited and
relative hook-path settings are remembered. Regular clones only are supported;
linked-worktree installation is rejected to avoid changing shared settings.
Run this installer after any tool that installs its own Git hooks and inspect
`status` if another tool later changes `core.hooksPath`.

For a deliberate local rollback:

```bash
python3 tools/install_safeguard_hooks.py uninstall
```

This restores the previous local hook-path setting, including inheritance when
there was no local setting. Original hook files and metadata remain. If another
tool has since changed the hook path, rollback refuses to overwrite it; inspect
the saved metadata and current setting first. Uninstalling disables these local
checks, so use the explicit checks below until hooks are restored.

For a manual preflight, or before uploading through an API or browser, prepare
the exact intended files in a local branch and check them **before** upload:

```bash
python3 tools/repository_safeguards.py --staged
python3 tools/repository_safeguards.py --base BASE_COMMIT --head HEAD_COMMIT
```

Replace the placeholders with the actual outgoing range. For a new history or
an unavailable base, run:

```bash
python3 tools/repository_safeguards.py --all-history --head HEAD_COMMIT
```

After checking, upload only that checked content. Git hooks do not run for
browser or API uploads. Local hooks can be bypassed; they are an additional
layer, not a GitHub-enforced upload boundary. Never print secret findings in
issue comments or workflow logs. Scanner output is fully redacted. A confirmed
credential must be revoked or rotated and its history handled deliberately;
deleting its current file does not remove it from earlier commits. False
positives need narrow, documented exceptions, not broad exclusions.

## GitHub checks and private-repository limits

GitHub Pro provides private branch rules, required checks, and CODEOWNERS.
CODEOWNERS identifies ownership; it does not add an outside-review requirement
to this solo workflow. Pro alone does **not** provide native secret scanning
and push protection for these private repositories. Keep existing public
secret scanning and push protection enabled. See GitHub's
[plan features](https://docs.github.com/en/get-started/learning-about-github/githubs-plans)
and [secret-scanning availability](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning).

The **Repository safeguards** check uses Gitleaks 8.30.1, blocks credential file
names and merge-conflict markers, and validates changed Python, JSON, and TOML
syntax. With a known base, syntax checks cover paths changed across the outgoing
commit range. With no base or `--all-history`, syntax checks cover paths changed
by the tip commit, avoiding unrelated legacy syntax errors; secret checks still
cover the full selected history and current tracked tree.

Pull-request checks run **after content has been uploaded**. Requiring
them protects merging to the default branch; it does not prevent a private
secret from reaching GitHub. API and browser edits need the same local
preflight as a Git push.

The workflow and checker execute candidate pull-request code. A pull request
can weaken the checker itself and still produce a green check, so the owner
must review changes to workflows, safeguard scripts, and scanner exceptions
before merging. Zero outside approvals and disabled automatic merging remain
the solo-owner policy; a green result does not replace that review.

Use standard Linux runners with a ten-minute job limit, read-only workflow
tokens, full-commit-SHA action pins, and cancellation of superseded runs. Add no
artifact uploads, caches, or package publishing. Public contributions require
the configured workflow approval and receive no secrets or write credentials.
Workflow pull-request approval remains disabled. The legacy private engineering
pipeline is disabled in GitHub and manual-only in its workflow file until
separately repaired; it is not a required check.

Enable Dependabot alerts and security updates where GitHub supports the
repository's manifests. GitHub Actions updates run weekly, with a grouped update
and a five-open-PR limit; review them normally without automatic merging.
Dependabot does not cover this repository's custom Debian dependency inventory
or all C++ system libraries. Those need separate maintenance.

## Private and public boundaries

`HomeCode` is private. `HomeCode-Public` contains only deliberately approved
public material. A successful scan does not authorize publication or prove
that every sensitive document has been detected. Do not mirror private project
contents into the public repository, change visibility, or enable Pages as
part of safeguards work. Backup destinations, logs, and recovery material must
respect the same boundary.

The account owner should verify two-factor authentication, a passkey or
security key where available, and recovery readiness; review apps, tokens, SSH
keys, and sessions before removing obsolete access. Keep recovery codes outside
repositories. Do not paste credentials or recovery material into check output,
issues, pull requests, or this document.

## Validation and recovery

Local tests use harmless synthetic findings and isolated temporary repositories:

```bash
python3 -m unittest discover -s tests -p 'test_*safeguard*.py' -v
```

Check clean changes, findings in intermediate commits, staged/working-copy
differences, scanner failures, existing-hook preservation, all remote ref
updates, and hook-path rollback. Test server-side rejection only on a temporary
protected branch. Confirm that the owner can merge a passing pull request and
that branch visibility remains unchanged.

Keep timestamped settings snapshots and the final rollout report privately on
the Development SSD. If a required check becomes unavailable, repair it on a
branch; do not weaken protection merely to upload a backup. A deliberate
settings rollback uses the saved prior values and verifies them afterward.
Changes to billing, repository rules, and local hooks are separate controls;
rolling back one does not restore the others.
