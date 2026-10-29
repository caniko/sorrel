<#
Real-Windows smoke test for the release binary and wheel. Run from a repo
checkout on a windows runner, after the Nix output `.#sorrel-windows-smoke`
(sorrel.exe + the win_amd64 wheel) was downloaded to $env:SIMIT_TARGET_SMOKE_DIR.

  pwsh release/smoke-windows.ps1 [-BuildDir DIR] [-OutDir smoke-out] [-GuiSeconds 20]

Steps: version, gpu-check, headless qc + report on release/fixtures/tiny-phy,
MCP over stdio (`sorrel mcp`), MCP attached to a `--headless` host through the
%LOCALAPPDATA% token file, `sorrel auto` launching its sidecar through uv.exe,
GUI start + screenshot, wheel install in a venv, Microsoft Defender scan of
BuildDir. (`sorrel rescue` is not run: the fixture has no amplitudes.npy or
spike_templates.npy, which it needs.) Any failure exits non-zero;
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
function Invoke-Native([string]$Name, [string]$File, [string[]]$Arguments,
    [string]$StdinFile, [int]$TimeoutSec = 900) {
    $o = Join-Path $OutDir "$Name.out.txt"
    $e = Join-Path $OutDir "$Name.err.txt"
    $argList = @{ FilePath = $File; PassThru = $true; NoNewWindow = $true;
        RedirectStandardOutput = $o; RedirectStandardError = $e }
    if ($StdinFile) { $argList.RedirectStandardInput = $StdinFile }
    # Start-Process joins ArgumentList with spaces unquoted, so quote any
    # argument with whitespace or quotes (MSVCRT command-line rules).
    if ($Arguments) {
        $argList.ArgumentList = $Arguments | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' } else { $_ }
        }
    }
    $p = Start-Process @argList
    $null = $p.Handle # without it ExitCode can read $null after a no-Wait start
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        Stop-Process -Id $p.Id -Force
        throw "$Name did not finish within $TimeoutSec s"
    }
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

# Python launcher for the uv install and the wheel venv.
$py = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }

# MCP helpers: newline-delimited JSON-RPC in from a file (so stdin closes and
# the server exits), replies parsed from stdout. The request file, the raw
# replies and stderr stay in OutDir as <Name>.in.jsonl / .out.txt / .err.txt.
function New-DataCopy([string]$Tag) {
    $d = Join-Path ([IO.Path]::GetTempPath()) "sorrel-smoke-$Tag-$PID"
    Copy-Item (Join-Path $repo 'release\fixtures\tiny-phy') $d -Recurse
    return $d
}
function Rpc([int]$Id, [string]$Method, $Params = @{}) {
    @{ jsonrpc = '2.0'; id = $Id; method = $Method; params = $Params }
}
function Rpc-Init([int]$Id) {
    Rpc $Id 'initialize' @{ protocolVersion = '2025-06-18'; capabilities = @{};
        clientInfo = @{ name = 'smoke'; version = '1' } }
}
$rpcInitialized = @{ jsonrpc = '2.0'; method = 'notifications/initialized' }
function Rpc-Call([int]$Id, [string]$Tool, $Arguments = @{}) {
    Rpc $Id 'tools/call' @{ name = $Tool; arguments = $Arguments }
}
function Invoke-Mcp([string]$Name, [string]$Dir, $Requests) {
    $in = Join-Path $OutDir "$Name.in.jsonl"
    $text = (($Requests | ForEach-Object { $_ | ConvertTo-Json -Depth 10 -Compress }) -join "`n") + "`n"
    [IO.File]::WriteAllText($in, $text, [Text.UTF8Encoding]::new($false))
    Assert-Ok $Name (Invoke-Native $Name $exe @('mcp', $Dir) -StdinFile $in -TimeoutSec 120)
    @(Get-Content (Join-Path $OutDir "$Name.out.txt") | Where-Object { $_.Trim() } |
        ForEach-Object { $_ | ConvertFrom-Json })
}
# The `result` of the reply with this id; a JSON-RPC error or isError fails.
function Get-Result($Replies, [int]$Id) {
    $r = @($Replies | Where-Object { $_.id -eq $Id }) | Select-Object -First 1
    if (-not $r) { throw "no MCP reply with id $Id" }
    if ($r.error) { throw "MCP error for id ${Id}: $($r.error | ConvertTo-Json -Compress)" }
    if ($r.result.isError) { throw "MCP tool error for id ${Id}: $($r.result.content[0].text)" }
    return $r.result
}
# A tool's JSON payload (a text content block holding JSON).
function Get-Tool($Replies, [int]$Id) {
    ,@((Get-Result $Replies $Id).content[0].text | ConvertFrom-Json)
}
function Get-Label($Rows, [int]$Cluster) {
    (@($Rows) | Where-Object { $_.cluster -eq $Cluster }).label
}

# 4. MCP over stdio: no host is running, so `sorrel mcp` serves the dataset itself.
$nClusters = @(Get-Content (Join-Path $qc 'cluster_qc.tsv') | Select-Object -Skip 1 |
    Where-Object { $_.Trim() }).Count
if ($nClusters -lt 1) { throw 'qc found no clusters in tiny-phy' }
$dataA = New-DataCopy 'mcp'
$r = Invoke-Mcp 'mcp-stdio' $dataA @(
    (Rpc-Init 1), $rpcInitialized, (Rpc 2 'tools/list'), (Rpc-Call 3 'list_clusters'))
$init = Get-Result $r 1
if ($init.serverInfo.name -ne 'sorrel' -or $init.serverInfo.version -ne $version) {
    throw "MCP serverInfo is $($init.serverInfo | ConvertTo-Json -Compress), expected sorrel $version"
}
$tools = @((Get-Result $r 2).tools | ForEach-Object name)
foreach ($t in 'list_clusters', 'relabel', 'export_phy') {
    if ($tools -notcontains $t) { throw "MCP tools/list lacks $t (got: $($tools -join ', '))" }
}
$rows = Get-Tool $r 3
if ($rows.Count -ne $nClusters) { throw "list_clusters returned $($rows.Count) clusters, qc found $nClusters" }
Write-Host "MCP stdio: sorrel $($init.serverInfo.version), $($tools.Count) tools, $($rows.Count) clusters"

# 5. MCP through a host: `sorrel --headless` publishes its port and token under
#    %LOCALAPPDATA%\sorrel\mcp\ (never in the dataset directory on Windows) and
#    `sorrel mcp` attaches to it. While the host holds the journal, a `sorrel mcp`
#    that served in-process would fail with "already open", so success proves the
#    attach; the label surviving into a second session shows the host owns the state.
$dataB = New-DataCopy 'host'
$mcpDir = Join-Path $env:LOCALAPPDATA 'sorrel\mcp'
$known = @(Get-ChildItem $mcpDir -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object Name)
$hostProc = Start-Process -FilePath $exe -ArgumentList @('--headless', $dataB) -PassThru `
    -RedirectStandardOutput (Join-Path $OutDir 'headless.out.txt') `
    -RedirectStandardError (Join-Path $OutDir 'headless.err.txt')
$portFile = $null
try {
    $deadline = (Get-Date).AddSeconds(30)
    while (-not $portFile -and (Get-Date) -lt $deadline) {
        if ($hostProc.HasExited) { throw "headless host exited with code $($hostProc.ExitCode)" }
        $portFile = Get-ChildItem $mcpDir -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -eq '.json' -and $known -notcontains $_.Name } |
            Select-Object -First 1
        if (-not $portFile) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $portFile) { throw "headless host wrote no token file under $mcpDir within 30 s" }
    if (Test-Path (Join-Path $dataB '.sorrel\mcp.json')) { throw 'token file was written inside the dataset directory' }
    # Evidence without the token.
    Get-Content $portFile.FullName -Raw | ConvertFrom-Json | Select-Object port, pid |
        ConvertTo-Json | Set-Content (Join-Path $OutDir 'headless-listener.json')

    $r = Invoke-Mcp 'mcp-attach' $dataB @(
        (Rpc-Init 1), $rpcInitialized, (Rpc-Call 2 'list_clusters'))
    $before = Get-Label (Get-Tool $r 2) 0
    $target = if ($before -eq 'noise') { 'good' } else { 'noise' }
    $r = Invoke-Mcp 'mcp-attach-relabel' $dataB @(
        (Rpc-Init 1), $rpcInitialized,
        (Rpc-Call 2 'relabel' @{ cluster = 0; label = $target }), (Rpc-Call 3 'list_clusters'))
    Get-Result $r 2 | Out-Null
    $after = Get-Label (Get-Tool $r 3) 0
    if ($after -ne $target) { throw "relabel via the host left cluster 0 '$after', expected '$target'" }
    $r = Invoke-Mcp 'mcp-attach-second' $dataB @((Rpc-Init 1), (Rpc-Call 2 'list_clusters'))
    $second = Get-Label (Get-Tool $r 2) 0
    if ($second -ne $target) { throw "a second session sees cluster 0 '$second', expected '$target'" }
    Write-Host "MCP attach: cluster 0 $before -> $target through the headless host"
}
finally {
    if (-not $hostProc.HasExited) { Stop-Process -Id $hostProc.Id -Force }
    # A killed host cannot delete its own token file.
    if ($portFile) { Remove-Item $portFile.FullName -ErrorAction SilentlyContinue }
    Get-Content (Join-Path $OutDir 'headless.out.txt'), (Join-Path $OutDir 'headless.err.txt') -ErrorAction SilentlyContinue | Write-Host
}

# 6. `sorrel auto` must launch its SpikeInterface sidecar through uv.exe. The
#    fixture has no channel_positions.npy, so the sidecar cannot finish; the
#    proof is that it ran far enough to import SpikeInterface (uv resolved the
#    PEP 723 dependencies), read the raw file and log, and only then died on
#    the missing file. A plain `python` has no SpikeInterface, so it cannot get
#    that far. `sorrel rescue` is not run: it needs amplitudes.npy and
#    spike_templates.npy, which the fixture lacks.
Assert-Ok 'pip-uv' (Invoke-Native 'pip-uv' $py @('-m', 'pip', 'install', 'uv'))
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $scripts = (& $py -c 'import sysconfig; print(sysconfig.get_path("scripts"))').Trim()
    $env:PATH = "$scripts$([IO.Path]::PathSeparator)$env:PATH"
}
Write-Host "uv: $((Get-Command uv).Source)"
Remove-Item Env:SORREL_PYTHON -ErrorAction SilentlyContinue
$dataC = New-DataCopy 'auto'
$autoCode = Invoke-Native 'auto' $exe @('auto', $dataC, '--dry-run', '--no-unitrefine') -TimeoutSec 600
$autoLog = (Get-Content (Join-Path $OutDir 'auto.out.txt'), (Join-Path $OutDir 'auto.err.txt') -Raw) -join "`n"
if ($autoLog -match 'spawn the sorrel_auto\.py sidecar') { throw 'sorrel auto could not launch uv or python' }
if (-not (Test-Path (Join-Path $dataC 'sorrel_auto\sorrel_auto.py'))) { throw 'sorrel auto did not write the sidecar script' }
if ($autoLog -notmatch '\[sorrel-auto\]') { throw 'the sidecar never logged: uv did not run it (see auto.err.txt)' }
if ($autoCode -ne 0 -and $autoLog -notmatch 'channel_positions\.npy') {
    throw "sorrel auto failed with exit code $autoCode for a reason other than the fixture's missing channel_positions.npy"
}

# 7. GUI: default backend selection, must survive $GuiSeconds
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

# 8. Python wheel in a fresh venv
$wheel = Get-ChildItem $BuildDir -Filter 'sorrel_ephys-*-win_amd64.whl' | Select-Object -First 1
if (-not $wheel) { throw "no sorrel_ephys wheel in $BuildDir" }
$venv = Join-Path ([IO.Path]::GetTempPath()) "sorrel-venv-$PID"
Assert-Ok 'venv' (Invoke-Native 'venv' $py @('-m', 'venv', $venv))
$vpy = Join-Path $venv 'Scripts\python.exe'
Assert-Ok 'pip' (Invoke-Native 'pip' $vpy @('-m', 'pip', 'install', $wheel.FullName))
Assert-Ok 'import' (Invoke-Native 'import' $vpy @('-c',
    "import sorrel; print(sorrel.__version__ if hasattr(sorrel,'__version__') else 'ok')"))
Assert-Ok 'wheel-exe' (Invoke-Native 'wheel-exe' (Join-Path $venv 'Scripts\sorrel.exe') @('--version'))
$got = (Get-Content (Join-Path $OutDir 'wheel-exe.out.txt') -Raw).Trim()
if ($got -ne "sorrel $version") { throw "wheel sorrel.exe --version is '$got', expected 'sorrel $version'" }

# 9. Microsoft Defender: winget-pkgs validation scans the installed package
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
