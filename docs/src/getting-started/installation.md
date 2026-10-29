# Installation

Sorrel is a Rust workspace and a Nix flake. The default build is pure Rust;
the NWB and Kilosort 4 `rez.mat` backends are gated behind an `hdf5`
feature that links against `libhdf5`.

## With Cargo

```bash
cargo build --release -p sorrel
```

The binary is written to `target/release/sorrel`.

To enable the HDF5-backed readers (NWB, KS4 `rez.mat`):

```bash
cargo build --release -p sorrel --features hdf5
```

Full phy export (`sorrel export-phy`, the `export_phy` serve tool, the
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

Stable releases go to winget and Chocolatey. Both install `sorrel.exe` from
the release's `sorrel-<VERSION>-x86_64-windows.zip` and put `sorrel` on your
`PATH`:

```powershell
winget install Caniko.Sorrel
choco install sorrel
```

Both packages also install [uv](https://docs.astral.sh/uv/), which
`sorrel auto` uses to run its SpikeInterface pipeline. `sorrel mcp` works
from the same `PATH`, so an agent can launch it by name (see
[Agents](../usage/agents.md#connecting-an-agent)).

New versions show up once the winget-pkgs PR is merged and Chocolatey
moderation passes, which can take a few days after the GitHub release.

The window opens on the first backend wgpu likes (Vulkan or DX12). If it
crashes at startup with a shader error, or stays blank, pick the other one:

```powershell
$env:WGPU_BACKEND = "vulkan"   # or "dx12", or "gl"
sorrel <DATA_DIR>
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

The Nix packages ship a `sorrel.desktop` menu entry.

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
