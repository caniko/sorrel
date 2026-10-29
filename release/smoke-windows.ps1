<#
Real-Windows smoke test for the release binary and wheel. Run from a repo
checkout on a windows runner, after the Nix output `.#sorrel-windows-smoke`
(sorrel.exe + the win_amd64 wheel) was downloaded to $env:SIMIT_TARGET_SMOKE_DIR.

  pwsh release/smoke-windows.ps1 [-BuildDir DIR] [-OutDir smoke-out] [-GuiSeconds 20]

Steps: version, gpu-check, headless qc + report on release/fixtures/tiny-phy,
GUI start + screenshot, wheel install in a venv, Microsoft Defender scan of
BuildDir. Any failure exits non-zero;
logs and gui.png land in OutDir for the runner to upload.
#>
param(
    [string]$BuildDir = $env:SIMIT_TARGET_SMOKE_DIR,
    [string]$OutDir = 'smoke-out',
    [int]$GuiSeconds = 20
)
$ErrorActionPreference = 'Stop'

if (-not $BuildDir) { throw 'pass -BuildDir or set SIMIT_TARGET_SMOKE_DIR' }
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$exe = Join-Path $BuildDir 'sorrel.exe'
if (-not (Test-Path $exe)) { throw "missing $exe" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

# Workspace version = the first `version = "..."` in Cargo.toml.
$version = (Select-String -Path (Join-Path $repo 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' |
    Select-Object -First 1).Matches[0].Groups[1].Value
if (-not $version) { throw 'could not read the crate version from Cargo.toml' }

# Run a native command with stdout/stderr captured to files (PowerShell 5.1
# turns native stderr into terminating errors under 'Stop'). Returns the exit
# code; both streams are echoed so they show up in the runner log too.
function Invoke-Native([string]$Name, [string]$File, [string[]]$Arguments) {
    $o = Join-Path $OutDir "$Name.out.txt"
    $e = Join-Path $OutDir "$Name.err.txt"
    $argList = @{ FilePath = $File; Wait = $true; PassThru = $true; NoNewWindow = $true;
        RedirectStandardOutput = $o; RedirectStandardError = $e }
    # Start-Process joins ArgumentList with spaces unquoted, so quote any
    # argument with whitespace or quotes (MSVCRT command-line rules).
    if ($Arguments) {
        $argList.ArgumentList = $Arguments | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' } else { $_ }
        }
    }
    $p = Start-Process @argList
    Write-Host "--- $Name (exit $($p.ExitCode))"
    Get-Content $o, $e | Write-Host
    return $p.ExitCode
}
function Assert-Ok([string]$Name, [int]$Code) {
    if ($Code -ne 0) { throw "$Name failed with exit code $Code" }
}

# 1. version
Assert-Ok 'version' (Invoke-Native 'version' $exe @('--version'))
$got = (Get-Content (Join-Path $OutDir 'version.out.txt') -Raw).Trim()
if ($got -ne "sorrel $version") { throw "sorrel.exe --version is '$got', expected 'sorrel $version'" }

# 2. gpu-check: real FXC/DX12 (WARP on a runner without a GPU) must build the
#    compute kernels and match the CPU. Exit 0 is the contract.
Assert-Ok 'gpu-check' (Invoke-Native 'gpu-check' $exe @('gpu-check'))

# 3. headless qc + report on a throwaway copy (sorrel writes next to the data)
$data = Join-Path ([IO.Path]::GetTempPath()) "sorrel-smoke-$PID"
Copy-Item (Join-Path $repo 'release\fixtures\tiny-phy') $data -Recurse
$qc = Join-Path $OutDir 'qc'
Assert-Ok 'qc' (Invoke-Native 'qc' $exe @('qc', $data, '--out', $qc))
foreach ($f in 'cluster_qc.tsv', 'cluster_qc.json') {
    if (-not (Get-Item (Join-Path $qc $f)).Length) { throw "qc wrote an empty $f" }
}
$report = Join-Path $OutDir 'report.html'
Assert-Ok 'report' (Invoke-Native 'report' $exe @('report', $data, '--out', $report))
if (-not (Get-Item $report).Length) { throw 'report.html is empty' }

# 4. GUI: default backend selection, must survive $GuiSeconds
Remove-Item Env:WGPU_BACKEND -ErrorAction SilentlyContinue
$env:RUST_LOG = 'info'
$guiLog = Join-Path $OutDir 'gui.log'
$gui = Start-Process -FilePath $exe -ArgumentList @($data) -PassThru `
    -RedirectStandardOutput (Join-Path $OutDir 'gui.out.txt') -RedirectStandardError $guiLog
try {
    Start-Sleep -Seconds $GuiSeconds
    $gui.Refresh()
    if ($gui.HasExited) { throw "GUI exited after <$GuiSeconds s with code $($gui.ExitCode)" }
    Add-Type -AssemblyName System.Windows.Forms, System.Drawing
    $s = [System.Windows.Forms.SystemInformation]::VirtualScreen
    $bmp = New-Object System.Drawing.Bitmap $s.Width, $s.Height
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($s.Left, $s.Top, 0, 0, $bmp.Size)
    $bmp.Save((Join-Path $OutDir 'gui.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}
finally {
    if (-not $gui.HasExited) { Stop-Process -Id $gui.Id -Force }
}
Get-Content $guiLog | Write-Host
# A panic on a worker thread leaves the process alive; still a failure.
if (Select-String -Path $guiLog -Pattern 'panicked at' -Quiet) { throw 'GUI log contains a panic' }

# 5. Python wheel in a fresh venv
$wheel = Get-ChildItem $BuildDir -Filter 'sorrel_ephys-*-win_amd64.whl' | Select-Object -First 1
if (-not $wheel) { throw "no sorrel_ephys wheel in $BuildDir" }
$venv = Join-Path ([IO.Path]::GetTempPath()) "sorrel-venv-$PID"
$py = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
Assert-Ok 'venv' (Invoke-Native 'venv' $py @('-m', 'venv', $venv))
$vpy = Join-Path $venv 'Scripts\python.exe'
Assert-Ok 'pip' (Invoke-Native 'pip' $vpy @('-m', 'pip', 'install', $wheel.FullName))
Assert-Ok 'import' (Invoke-Native 'import' $vpy @('-c',
    "import sorrel; print(sorrel.__version__ if hasattr(sorrel,'__version__') else 'ok')"))
Assert-Ok 'wheel-exe' (Invoke-Native 'wheel-exe' (Join-Path $venv 'Scripts\sorrel.exe') @('--version'))
$got = (Get-Content (Join-Path $OutDir 'wheel-exe.out.txt') -Raw).Trim()
if ($got -ne "sorrel $version") { throw "wheel sorrel.exe --version is '$got', expected 'sorrel $version'" }

# 6. Microsoft Defender: winget-pkgs validation scans the installed package
#    (Validation-Defender-Error), so reproduce that here on fresh signatures.
#    -DisableRemediation reports without quarantining; exit 2 = threat found.
$mp = Join-Path $env:ProgramFiles 'Windows Defender\MpCmdRun.exe'
if (Test-Path $mp) {
    Invoke-Native 'defender-update' $mp @('-SignatureUpdate') | Out-Null
    $code = Invoke-Native 'defender-scan' $mp @('-Scan', '-ScanType', '3', '-File', $BuildDir, '-DisableRemediation')
    if ($code -eq 2) { throw "Microsoft Defender flagged files in $BuildDir (see defender-scan.out.txt)" }
    Assert-Ok 'defender-scan' $code
}
else { Write-Warning "no $mp on this runner; Defender scan skipped" }

Write-Host "Windows smoke passed for Sorrel $version; evidence in $OutDir"
