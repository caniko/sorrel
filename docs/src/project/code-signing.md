# Code signing policy

Free code signing provided by [SignPath.io](https://about.signpath.io/),
certificate by [SignPath Foundation](https://signpath.org/).

Signing is not active yet: the Windows executables of current releases are
unsigned, and Windows SmartScreen may show an "unknown publisher" warning. This
page is the policy that applies once signing is enabled. Until then you can
verify downloads with the checksums and signatures described under
[Windows](../getting-started/installation.md#windows).

## What is signed

Only the Windows executables (`sorrel.exe`, `sorrelw.exe`, and the zip and
standalone files that hold them) built by this project's public
[GitHub Actions release workflow](https://github.com/caniko/sorrel/blob/trunk/.github/workflows/release.yml)
from a signed release tag on the `trunk` branch of
[caniko/sorrel](https://github.com/caniko/sorrel). The workflow builds from
the repository's source with Nix and a pinned `flake.lock`, then submits the
build to SignPath. Nothing built on a developer machine, and no artifact from
another repository or fork, is signed. Third-party dependencies are compiled
from source as part of the build; none is signed separately.

The signed files carry the product name "Sorrel" and the release version in
their version information, the same for every build of a release.

## Team roles

| Role                     | Who                                 | What they do                                                        |
| ------------------------ | ----------------------------------- | ------------------------------------------------------------------- |
| Committers and reviewers | [caniko](https://github.com/caniko) | Change the source; review every change from non-committers          |
| Approvers                | [caniko](https://github.com/caniko) | Approve each signing request, one per release, after the release CI |

The project currently has a single maintainer, who holds all roles. Every
release needs a manual approval in SignPath before anything is signed. The
maintainer uses multi-factor authentication for GitHub and SignPath.

## Privacy

This program will not transfer any information to other networked systems
unless specifically requested by the user or the person installing or
operating it.

Sorrel opens connections only when you ask it to: `sorrel share` and
`sorrel join` talk to the peer and relay you name, and `sorrel auto` with
`--llm` sends cluster summaries to the Claude or OpenCode service you are
logged in to, or to the Anthropic API. `sorrel auto` also lets
[uv](https://docs.astral.sh/uv/) download its Python packages on first use.
Those services apply their own privacy policies. Signing transmits no user
data: SignPath receives only the build artifacts of a release.
