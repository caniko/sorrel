# Code signing policy

Free code signing provided by SignPath.io, certificate by SignPath Foundation.
([SignPath.io](https://about.signpath.io/),
[SignPath Foundation](https://signpath.org/))

Signing is not active yet. The Windows executables of current releases are
unsigned, so Windows SmartScreen may show an "unknown publisher" warning. This
page states the policy that applies once signing is enabled. Until then, check
a download against the release's `SHA256SUMS.txt`, which is signed with
minisign (key in [`keys/minisign.pub`](https://github.com/caniko/sorrel/blob/trunk/keys/minisign.pub)).
Each asset also has a Sigstore `.cosign.bundle` and `.intoto.bundle`.

## What is signed

Only artifacts built by the public GitHub Actions release workflow are signed.
These are the Windows executables (`sorrel.exe`, `sorrelw.exe`), and the zip
and standalone files that hold them. The
[release workflow](https://github.com/caniko/sorrel/blob/trunk/.github/workflows/release.yml)
builds them from a signed release tag on the `trunk` branch of
[caniko/sorrel](https://github.com/caniko/sorrel), from the repository's source,
with Nix and a pinned `flake.lock`. It then submits the build to SignPath.
Nothing built on a developer machine is signed, and neither is any artifact
from another repository or fork. Third-party dependencies are compiled from
source as part of the build; none of them is signed separately.

The signed files carry the product name "Sorrel" and the release version in
their version information.

## Team roles

| Role                     | Who                                 | What they do                                                        |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------------- |
| Committers and reviewers | [caniko](https://github.com/caniko) | Change the source; review every change from non-committers          |
| Approvers                | [caniko](https://github.com/caniko) | Approve each signing request, one per release, after the release CI |

The project has a single maintainer, who holds all roles. Each release signing
request is approved by hand in SignPath before anything is signed. The
maintainer uses multi-factor authentication on GitHub and SignPath.

## Privacy

This program will not transfer any information to other networked systems
unless specifically requested by the user or the person installing or
operating it.

Sorrel opens network connections only when you ask it to:

- `sorrel share` and `sorrel join` talk to the peer and relay you name.
- `sorrel auto --llm` sends cluster summaries to the agent CLI you are logged
  in to (`claude`, `codex` or `opencode`), or to the Anthropic API with
  `--llm=api`.
- `sorrel auto` lets [uv](https://docs.astral.sh/uv/) download its Python
  packages on first use.

Those services apply their own privacy policies. Signing transmits no user
data: SignPath receives only the build artifacts of a release.
