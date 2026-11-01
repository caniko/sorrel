# Installation

Sorrel is a Rust workspace and a Nix flake. The default build is pure Rust;
the NWB and Kilosort 4 `rez.mat` backends are gated behind an `hdf5`
feature that links against `libhdf5`.

Prebuilt binaries are on the
[GitHub releases page](https://github.com/caniko/sorrel/releases), and are
also published to winget, Chocolatey, PyPI and crates.io. Release binaries
follow the [code signing policy](../project/code-signing.md).

## With Cargo

```bash
cargo build --release -p sorrel
```

The binary is written to `target/release/sorrel`. A second binary,
`sorrelw`, is the console-free launcher for Windows (see [Windows](#windows)). Shell completions come from
`sorrel completions <shell>`; see the [CLI reference](cli.md#completions).

To enable the HDF5-backed readers (NWB, KS4 `rez.mat`):

```bash
cargo build --release -p sorrel --features hdf5
```

Full phy export (`sorrel export-phy`, the `export_phy` MCP tool, the
**Export phy** button and `Session.export_phy` in `sorrel-py`) is the
`phy-export` feature, on by default. Leave it out with:

```bash
cargo build --release -p sorrel --no-default-features
```

## Python

Wheels for Linux x86_64 / aarch64 (glibc 2.34+), Windows x86_64 and macOS
arm64 (Python 3.8+) hold the Python API and the GUI binary. They are on PyPI
as `sorrel-ephys`, since PyPI's `sorrel` is an unrelated project. The import
is still `import sorrel`:

```bash
pip install sorrel-ephys
```

Each GitHub release attaches the same wheels
(`sorrel_ephys-<VERSION>-cp38-abi3-<PLATFORM>.whl`).

`nix build .#release-bundle.wheels.x86_64-linux` builds one locally
(`aarch64-linux`, `windows`, `darwin-aarch64` likewise). See
[Integrations](../usage/integrations.md#python-jupyter) for the API.

## Windows

Stable releases go to winget and Chocolatey. Both install `sorrel.exe` and
`sorrelw.exe` from the release's `sorrel-<VERSION>-x86_64-windows.zip` and put
`sorrel` (and `sorrelw`) on your `PATH`:

```powershell
winget install Caniko.Sorrel
choco install sorrel
```

`sorrel.exe` is a console program. `sorrelw.exe` is the same program without a
console window: it starts `sorrel.exe` with the same arguments (bare, it opens
the launcher) and exits. Use it from shortcuts, Win+R and Explorer.

**Start menu.** The Chocolatey package adds a _Sorrel_ entry that runs
`sorrelw.exe`. With winget, or the zip unpacked by hand, run this once:

```powershell
sorrel install-shortcut            # creates the Start-menu entry
sorrel install-shortcut --remove   # deletes it
```

It writes `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Sorrel.lnk`, pointing at
the `sorrelw.exe` next to `sorrel.exe`. Run `--remove` before you uninstall:
neither winget nor Chocolatey knows about a shortcut this command made.

Both packages also install [uv](https://docs.astral.sh/uv/), which
`sorrel auto` uses to run its SpikeInterface pipeline. `sorrel mcp` works
from the same `PATH`, so an agent can launch it by name (see
[Agents](../usage/agents.md#connecting-an-agent)).

New versions show up once the winget-pkgs PR is merged and Chocolatey
moderation passes, which can take a few days after the GitHub release.

**Completions.** Add this line to your `$PROFILE` (`notepad $PROFILE`) to get
PowerShell completion in every session:

```powershell
sorrel completions powershell | Out-String | Invoke-Expression
```

**Standalone files.** Each release page also lists `sorrel-<VERSION>-x86_64-windows.exe`
(`sorrel.exe`) and `sorrel-<VERSION>-x86_64-windows-gui.exe` (`sorrelw.exe`)
for use without a package manager. `sorrelw.exe` looks for `sorrel.exe` next
to itself, so rename the downloads back to `sorrel.exe` and `sorrelw.exe` and
keep them in one folder.

**SmartScreen.** The executables are not code-signed yet, so Windows
SmartScreen may warn about an _unknown publisher_ when you run a download
(click _More info_, then _Run anyway_). Signing is planned through
SignPath Foundation (see the [Code signing policy](../project/code-signing.md)). Until then, check a
download before you run it. Every release lists `SHA256SUMS.txt`, signed
with minisign (`SHA256SUMS.txt.minisig`; the public key is
[`keys/minisign.pub`](https://github.com/caniko/sorrel/blob/trunk/keys/minisign.pub)),
and each file has Sigstore bundles (`<file>.cosign.bundle`, plus an SLSA
provenance `<file>.intoto.bundle`):

```powershell
(Get-FileHash .\sorrel-<VERSION>-x86_64-windows.zip).Hash   # compare with SHA256SUMS.txt
minisign -V -p minisign.pub -m SHA256SUMS.txt
cosign verify-blob --bundle sorrel-<VERSION>-x86_64-windows.zip.cosign.bundle `
  --certificate-identity-regexp 'https://github.com/caniko/sorrel/' `
  --certificate-oidc-issuer https://token.actions.githubusercontent.com `
  sorrel-<VERSION>-x86_64-windows.zip
```

**Uninstall.** `winget uninstall Caniko.Sorrel` or `choco uninstall sorrel`
removes the programs. Sorrel's per-user state (the GUI instance file, the
host registry, `gui.log`, `sorrelw.log`) stays in `%LOCALAPPDATA%\sorrel`, and
the derived-data cache and your curation journals stay next to your data.
Delete the folder when you no longer need it:

```powershell
Remove-Item -Recurse -Force $env:LOCALAPPDATA\sorrel
```

The window opens on the first backend wgpu likes (Vulkan or DX12). If it
crashes at startup with a shader error, or stays blank, pick the other one:

```powershell
$env:WGPU_BACKEND = "vulkan"   # or "dx12", or "gl"
sorrel open <DATA_DIR>
```

`sorrel gpu-check` shows what each backend can do. Trace filtering falls back
to the CPU on its own when a backend rejects the compute kernels.

## With Nix

```bash
nix build .#sorrel        # default build, no HDF5
nix build .#sorrelHdf5    # with HDF5 features
```

`nix build` builds the default `sorrel` package. The flake also exposes:

- `.#website` — the Zola landing page.
- `.#docs` — this mdBook documentation.
- `.#site` — combined static site, with the docs mounted under `/docs/`.

The Nix packages ship a `sorrel.desktop` menu entry with the app icon. They
also install `sorrelw`, which on Linux and macOS only runs the sibling
`sorrel`; the release archives, AppImage and `cargo install` users can ignore
it.

### Home Manager

Add the flake as an input and enable the module:

```nix
{
  inputs.sorrel.url = "github:caniko/sorrel";
  # in your Home Manager configuration:
  imports = [inputs.sorrel.homeManagerModules.default];
  programs.sorrel.enable = true;
  # NWB/HDF5 support:
  # programs.sorrel.package = inputs.sorrel.packages.${system}.sorrelHdf5;
}
```

## GPU drivers

Sorrel draws through wgpu, so it runs on Intel, AMD and NVIDIA GPUs alike.
It uses Vulkan (Linux, Windows), DX12 (Windows) or Metal (macOS), and falls back to
OpenGL. Trace CMR and high-pass filtering run on the GPU when the adapter
supports compute shaders, and on the CPU otherwise.

- **Linux, AMD and Intel:** Mesa (`radv` / `anv` Vulkan drivers; distro
  packages `mesa-vulkan-drivers` and `libvulkan1` on Debian/Ubuntu).
- **Linux, NVIDIA:** the proprietary driver with its Vulkan ICD. On nouveau,
  use Mesa's NVK.
- **NixOS:** `hardware.graphics.enable = true;`, plus
  `services.xserver.videoDrivers = [ "nvidia" ];` for NVIDIA. A Nix-built
  `sorrel` on another distribution needs
  [nixGL](https://github.com/nix-community/nixGL) to find the host driver.
- **Release tarballs and AppImage:** need glibc 2.31 or newer (Ubuntu 20.04,
  Debian 11, Fedora 32, RHEL 9 and later) and use the distribution's own
  Vulkan/GL loader and drivers.
- **Windows and macOS:** no extra setup.

Run `sorrel gpu-check` to see every adapter, which one the GUI prefers, and
whether the compute kernels match the CPU on each. The usual wgpu variables
override the choice, for example on hybrid laptops:

```bash
WGPU_POWER_PREF=low sorrel ...   # integrated instead of discrete GPU
WGPU_BACKEND=gl sorrel ...       # OpenGL if Vulkan misbehaves
```

## Development Shell

```bash
nix develop
```

The shell pulls in the Rust toolchain, native GUI dependencies (wgpu /
egui), `mdbook` for the docs, and `zola` for the website.

To run the docs and site locally:

```bash
cd docs    && mdbook serve
cd website && zola serve
```
