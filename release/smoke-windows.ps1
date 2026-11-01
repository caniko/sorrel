<#
Real-Windows smoke test for the release binary and wheel. Run from a repo
checkout on a windows runner, after the Nix output `.#sorrel-windows-smoke`
(sorrel.exe + the win_amd64 wheel) was downloaded to $env:SIMIT_TARGET_SMOKE_DIR.

  pwsh release/smoke-windows.ps1 [-BuildDir DIR] [-OutDir smoke-out] [-GuiSeconds 20]

Steps: version, gpu-check, headless qc + report on release/fixtures/tiny-phy,
MCP over stdio (`sorrel mcp`), MCP attached to a `run --headless` host through its
%LOCALAPPDATA% listener file, `sorrel auto` launching its sidecar through uv.exe,
a save round-trip through a headless host (relabel, split, save, reopen; the
regression gate for mapped files blocking the rename of spike_clusters.npy),
export-phy, --discard-journal, `status` and `call`, `sorrel auto --llm` argv
through fake npm `.cmd` shims, `sorrel rescue`, `share`/`join` over loopback,
GUI start + screenshot, the bare-`sorrel` / `sorrel open` handoff, wheel
install in a venv, Microsoft Defender scan of BuildDir. Any failure exits
non-zero; logs, JSON evidence and gui.png land in OutDir for the runner to
upload.
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
function New-DataCopy([string]$Tag, [string]$Fixture = 'tiny-phy') {
    $d = Join-Path ([IO.Path]::GetTempPath()) "sorrel-smoke-$Tag-$PID"
    Copy-Item (Join-Path $repo "release\fixtures\$Fixture") $d -Recurse
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
# `list_clusters` rows: a TSV text block (id, n, label, tags, ...), header once.
function Get-Clusters($Replies, [int]$Id) {
    ,@(((Get-Result $Replies $Id).content[0].text -split "`r?`n") |
        Where-Object { $_.Trim() } | ConvertFrom-Csv -Delimiter "`t")
}
function Get-Label($Rows, [int]$Cluster) {
    (@($Rows) | Where-Object { [int]$_.id -eq $Cluster }).label
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
$rows = Get-Clusters $r 3
if ($rows.Count -ne $nClusters) { throw "list_clusters returned $($rows.Count) clusters, qc found $nClusters" }
Write-Host "MCP stdio: sorrel $($init.serverInfo.version), $($tools.Count) tools, $($rows.Count) clusters"

# `sorrel mcp config` prints the entry an agent needs to launch this exe.
Assert-Ok 'mcp-config' (Invoke-Native 'mcp-config' $exe @('mcp', 'config', '--client', 'json', $dataA))
$cfg = (Get-Content (Join-Path $OutDir 'mcp-config.out.txt') -Raw | ConvertFrom-Json).mcpServers.sorrel
if (-not $cfg.command -or $cfg.args[0] -ne 'mcp' -or $cfg.args.Count -ne 2) {
    throw "unexpected mcp config: $($cfg | ConvertTo-Json -Compress)"
}

# 5. MCP through a host: `sorrel run --headless` publishes its port and token under
#    %LOCALAPPDATA%\sorrel\mcp\ (never in the dataset directory) and
#    `sorrel mcp` attaches to it. While the host holds the journal, a `sorrel mcp`
#    that served in-process would fail with "already open", so success proves the
#    attach; the label surviving into a second session shows the host owns the state.
$dataB = New-DataCopy 'host'
$mcpDir = Join-Path $env:LOCALAPPDATA 'sorrel\mcp'
$known = @(Get-ChildItem $mcpDir -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object Name)
$hostProc = Start-Process -FilePath $exe -ArgumentList @('run', '--headless', $dataB) -PassThru `
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
    $listener = Get-Content $portFile.FullName -Raw | ConvertFrom-Json
    if ($listener.kind -ne 'headless') { throw "listener kind is '$($listener.kind)', expected 'headless'" }
    if (([string]$listener.dataset).StartsWith('\\?\')) { throw "listener dataset is a verbatim path: $($listener.dataset)" }
    # Evidence without the token.
    $listener | Select-Object port, pid, kind, dataset | ConvertTo-Json | Set-Content (Join-Path $OutDir 'headless-listener.json')

    $r = Invoke-Mcp 'mcp-attach' $dataB @(
        (Rpc-Init 1), $rpcInitialized, (Rpc-Call 2 'list_clusters'))
    $before = Get-Label (Get-Clusters $r 2) 0
    $target = if ($before -eq 'noise') { 'good' } else { 'noise' }
    $r = Invoke-Mcp 'mcp-attach-relabel' $dataB @(
        (Rpc-Init 1), $rpcInitialized,
        (Rpc-Call 2 'relabel' @{ cluster = 0; label = $target }), (Rpc-Call 3 'list_clusters'))
    Get-Result $r 2 | Out-Null
    $after = Get-Label (Get-Clusters $r 3) 0
    if ($after -ne $target) { throw "relabel via the host left cluster 0 '$after', expected '$target'" }
    $r = Invoke-Mcp 'mcp-attach-second' $dataB @((Rpc-Init 1), (Rpc-Call 2 'list_clusters'))
    $second = Get-Label (Get-Clusters $r 2) 0
    if ($second -ne $target) { throw "a second session sees cluster 0 '$second', expected '$target'" }
    Write-Host "MCP attach: cluster 0 $before -> $target through the headless host"
}
finally {
    if (-not $hostProc.HasExited) { Stop-Process -Id $hostProc.Id -Force }
    # A killed host cannot delete its own token file.
    if ($portFile) { Remove-Item $portFile.FullName -ErrorAction SilentlyContinue }
    Get-Content (Join-Path $OutDir 'headless.out.txt'), (Join-Path $OutDir 'headless.err.txt') -ErrorAction SilentlyContinue | Write-Host
}

# 5b. Host helpers for the steps below: `run --headless` plus its listener file.
$mcpDir = Join-Path $env:LOCALAPPDATA 'sorrel\mcp'
function Start-Headless([string]$Data, [string]$Tag) {
    $known = @(Get-ChildItem $mcpDir -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object Name)
    $proc = Start-Process -FilePath $exe -ArgumentList @('run', '--headless', $Data) -PassThru `
        -RedirectStandardOutput (Join-Path $OutDir "host-$Tag.out.txt") `
        -RedirectStandardError (Join-Path $OutDir "host-$Tag.err.txt")
    $file = $null
    $deadline = (Get-Date).AddSeconds(30)
    while (-not $file -and (Get-Date) -lt $deadline) {
        if ($proc.HasExited) { throw "headless host $Tag exited with code $($proc.ExitCode)" }
        $file = Get-ChildItem $mcpDir -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -eq '.json' -and $known -notcontains $_.Name } |
            Select-Object -First 1
        if (-not $file) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $file) { Stop-Process -Id $proc.Id -Force; throw "headless host $Tag wrote no token file within 30 s" }
    @{ Proc = $proc; File = $file; Tag = $Tag }
}
function Stop-Headless($Host_) {
    if (-not $Host_) { return }
    if (-not $Host_.Proc.HasExited) { Stop-Process -Id $Host_.Proc.Id -Force }
    $null = $Host_.Proc.WaitForExit(15000) # the OS frees its journal lock on exit
    Remove-Item $Host_.File.FullName -ErrorAction SilentlyContinue
    Get-Content (Join-Path $OutDir "host-$($Host_.Tag).out.txt"), (Join-Path $OutDir "host-$($Host_.Tag).err.txt") -ErrorAction SilentlyContinue | Write-Host
}
# A `list_clusters` TSV printed by `sorrel call` (header once).
function Get-CallClusters([string]$Name) {
    ,@((Get-Content (Join-Path $OutDir "$Name.out.txt") | Where-Object { $_.Trim() }) | ConvertFrom-Csv -Delimiter "`t")
}

# 5c. Save round-trip: the regression gate for the mapped spike_clusters.npy.
#     The host keeps its provider alive while `save` replaces spike_clusters.npy
#     (tmp + rename). A leftover mapping of the old file makes Windows refuse
#     that rename (os error 1224). A relabel alone never rewrites the array, so
#     the split is what exercises it.
$dataS = New-DataCopy 'save'
$clustersFile = Join-Path $dataS 'spike_clusters.npy'
$hashBefore = (Get-FileHash $clustersFile).Hash
$hostS = Start-Headless $dataS 'save'
try {
    $r = Invoke-Mcp 'save-list' $dataS @((Rpc-Init 1), $rpcInitialized, (Rpc-Call 2 'list_clusters'))
    $nBefore = (Get-Clusters $r 2).Count
    $r = Invoke-Mcp 'save-relabel' $dataS @((Rpc-Init 1), $rpcInitialized,
        (Rpc-Call 2 'relabel' @{ cluster = 0; label = 'good' }), (Rpc-Call 3 'save'))
    Get-Result $r 2 | Out-Null
    Get-Result $r 3 | Out-Null
    $group = Get-Content (Join-Path $dataS 'cluster_group.tsv') -Raw
    if ($group -notmatch '(?m)^0\s+good') { throw "cluster_group.tsv lacks cluster 0 good after save:`n$group" }
    $r = Invoke-Mcp 'save-split' $dataS @((Rpc-Init 1), $rpcInitialized,
        (Rpc-Call 2 'split' @{ cluster = 1; spike_idx = @(0..9) }), (Rpc-Call 3 'save'))
    Get-Result $r 2 | Out-Null
    Get-Result $r 3 | Out-Null
    # (The live host's list_clusters does not show the new cluster until reopen.)
}
finally { Stop-Headless $hostS }
$hashAfter = (Get-FileHash $clustersFile).Hash
if ($hashAfter -eq $hashBefore) { throw 'spike_clusters.npy was not rewritten by the save while the provider was alive' }
if (Get-ChildItem $dataS -Filter '*.tmp' -ErrorAction SilentlyContinue) { throw 'save left .tmp files behind' }
if (Test-Path (Join-Path $dataS 'sorrel_rewrite.pending')) { throw 'save left sorrel_rewrite.pending behind' }
# Reopen with the host gone: the label and the split come back from disk.
Assert-Ok 'save-reopen' (Invoke-Native 'save-reopen' $exe @('call', $dataS, 'list_clusters'))
$reopened = Get-CallClusters 'save-reopen'
# The split is checked on the files save wrote: cluster_info.tsv lists the new cluster. (A reopened
# `list_clusters` does not show a split-off cluster after a second save, on Linux too: not asserted.)
$infoRows = @(Get-Content (Join-Path $dataS 'cluster_info.tsv') | Select-Object -Skip 1 | Where-Object { $_.Trim() }).Count
if ($infoRows -ne $nBefore + 1) { throw "cluster_info.tsv lists $infoRows clusters after the split + save, expected $($nBefore + 1)" }
if ($reopened.Count -lt $nBefore) { throw "reopened dataset has $($reopened.Count) clusters, expected at least $nBefore" }
if ((Get-Label $reopened 0) -ne 'good') { throw "cluster 0 reopened as '$(Get-Label $reopened 0)', expected 'good'" }
@{ before = $hashBefore; after = $hashAfter; info_rows = $infoRows; reopened = $reopened.Count; cluster0 = (Get-Label $reopened 0) } |
    ConvertTo-Json | Set-Content (Join-Path $OutDir 'save-roundtrip.json')
Write-Host "save round-trip: spike_clusters.npy $($hashBefore.Substring(0, 8)) -> $($hashAfter.Substring(0, 8)), cluster 0 good, $infoRows clusters in cluster_info.tsv, $($reopened.Count) listed after reopen"

# 5d. export-phy, then the export opens; --discard-journal on reopen.
$exportDir = Join-Path ([IO.Path]::GetTempPath()) "sorrel-smoke-export-$PID"
Assert-Ok 'export-phy' (Invoke-Native 'export-phy' $exe @('export-phy', $dataS, '--out', $exportDir))
foreach ($f in 'spike_times.npy', 'spike_clusters.npy', 'params.py') {
    if (-not (Test-Path (Join-Path $exportDir $f))) { throw "export-phy wrote no $f" }
}
$exportQc = Join-Path $OutDir 'export-qc'
Assert-Ok 'export-qc' (Invoke-Native 'export-qc' $exe @('qc', $exportDir, '--out', $exportQc))
$nExport = @(Get-Content (Join-Path $exportQc 'cluster_qc.tsv') | Select-Object -Skip 1 | Where-Object { $_.Trim() }).Count
if ($nExport -lt $nBefore) { throw "qc of the export found $nExport clusters, expected at least $nBefore" }
Assert-Ok 'discard-journal' (Invoke-Native 'discard-journal' $exe @('qc', $dataS, '--out', (Join-Path $OutDir 'discard-qc'), '--discard-journal'))
$aside = @(Get-ChildItem $dataS | Where-Object { $_.Name -match '^sorrel\.journal\.\d+' })
if (-not $aside) { throw '--discard-journal did not set the journal aside' }
Write-Host "export-phy: $nExport clusters; --discard-journal set aside $($aside.Name -join ', ')"

# 5e. `status` and `call` against a live host. The registry and `status` must show
#     plain `C:\...` paths, not the verbatim `\\?\C:\...` that canonicalize returns.
$dataH = New-DataCopy 'status'
$hostH = Start-Headless $dataH 'status'
try {
    Assert-Ok 'status-json' (Invoke-Native 'status-json' $exe @('status', '--json'))
    $st = Get-Content (Join-Path $OutDir 'status-json.out.txt') -Raw | ConvertFrom-Json
    $mine = @($st.hosts | Where-Object { (Split-Path $_.dataset -Leaf) -eq (Split-Path $dataH -Leaf) })
    if ($mine.Count -ne 1) { throw "status lists $($mine.Count) hosts for the dataset, expected 1" }
    if (([string]$mine[0].dataset).StartsWith('\\?\')) { throw "status shows a verbatim path: $($mine[0].dataset)" }
    if ($mine[0].kind -ne 'headless' -or -not $mine[0].alive -or $mine[0].pid -ne $hostH.Proc.Id) {
        throw "unexpected host record: $($mine[0] | ConvertTo-Json -Compress)"
    }
    Assert-Ok 'status-text' (Invoke-Native 'status-text' $exe @('status', $dataH))
    if ((Get-Content (Join-Path $OutDir 'status-text.out.txt') -Raw) -match [regex]::Escape('\\?\')) { throw 'status text shows a verbatim path' }
    # While the host runs, `call` goes through it (in-process would hit the journal lock).
    Assert-Ok 'call-list' (Invoke-Native 'call-list' $exe @('call', $dataH, 'list_clusters'))
    if ((Get-CallClusters 'call-list').Count -ne $nClusters) { throw "call list_clusters returned the wrong cluster count (expected $nClusters)" }
    Assert-Ok 'call-relabel' (Invoke-Native 'call-relabel' $exe @('call', $dataH, 'relabel', '{"cluster":0,"label":"mua"}'))
    Assert-Ok 'call-list2' (Invoke-Native 'call-list2' $exe @('call', $dataH, 'list_clusters'))
    if ((Get-Label (Get-CallClusters 'call-list2') 0) -ne 'mua') { throw 'call relabel did not stick in the host' }
    Write-Host "status/call: host pid $($hostH.Proc.Id) listed alive, plain path $($mine[0].dataset)"
}
finally { Stop-Headless $hostH }

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

# 6b. `sorrel auto --llm` runs claude/codex/opencode; npm installs those as `.cmd`
#     shims, and cmd.exe re-parses every argument (`"`, %, ^, &, |, newlines). The
#     fake shims below dump what their program received; release/smoke-llm-shim.py
#     calls the sidecar's own ask_cli with hostile prompts and compares the dump
#     with the argv and stdin it intended. (`sorrel auto --llm` cannot be driven
#     end to end here: the fixture cannot finish the SpikeInterface analysis that
#     precedes the LLM calls, so the sidecar's spawn code is run directly.)
$shimDir = Join-Path ([IO.Path]::GetTempPath()) "sorrel-shim-$PID"
New-Item -ItemType Directory $shimDir | Out-Null
Copy-Item (Join-Path $repo 'release\smoke-argv-dump.py') (Join-Path $shimDir 'argv_dump.py')
$pyExe = (& $py -c 'import sys; print(sys.executable)').Trim()
foreach ($n in 'claude', 'codex', 'opencode') {
    [IO.File]::WriteAllText((Join-Path $shimDir "$n.cmd"), "@`"$pyExe`" `"%~dp0argv_dump.py`" %*`r`n", [Text.Encoding]::ASCII)
}
$oldPath = $env:PATH
$env:PATH = "$shimDir$([IO.Path]::PathSeparator)$env:PATH"
try {
    Assert-Ok 'llm-shim' (Invoke-Native 'llm-shim' (Get-Command uv).Source @('run', '--no-project', '--with', 'numpy', 'python',
        (Join-Path $repo 'release\smoke-llm-shim.py'), (Join-Path $dataC 'sorrel_auto\sorrel_auto.py'), $OutDir) -TimeoutSec 600)
}
finally { $env:PATH = $oldPath }
foreach ($n in 'claude', 'codex', 'opencode') {
    if (-not (Test-Path (Join-Path $OutDir "$n.json"))) { throw "the $n.cmd shim never ran" }
}

# 6c. rescue on release/fixtures/rescue-phy (amplitudes, spike_templates and templates
#     present, sparse enough for noise windows). A dry run: it must finish and report.
$dataR = New-DataCopy 'rescue' 'rescue-phy'
Assert-Ok 'rescue' (Invoke-Native 'rescue' $exe @('rescue', $dataR, '--units', 'all', '--channels-per-unit', '1') -TimeoutSec 600)
$rescueOut = Get-Content (Join-Path $OutDir 'rescue.out.txt') -Raw
if ($rescueOut -notmatch 'spikes found \(dry run\)') { throw "rescue printed no dry-run summary:`n$rescueOut" }

# 6d. share/join over loopback: no relay needed. The host serves on a TCP port, the
#     guest dials 127.0.0.1 with the pairing code, and its phy files must arrive
#     byte for byte (blake3-verified by sorrel itself; compared again here).
$dataSh = New-DataCopy 'share'
$guestDir = Join-Path ([IO.Path]::GetTempPath()) "sorrel-guest-$PID"
$port = Get-Random -Minimum 20000 -Maximum 40000
$share = Start-Process -FilePath $exe -ArgumentList @('share', $dataSh, '--port', $port) -PassThru `
    -RedirectStandardOutput (Join-Path $OutDir 'share.out.txt') -RedirectStandardError (Join-Path $OutDir 'share.err.txt')
$join = $null
try {
    $code = $null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not $code -and (Get-Date) -lt $deadline) {
        if ($share.HasExited) { throw "sorrel share exited with code $($share.ExitCode)" }
        $m = Select-String -Path (Join-Path $OutDir 'share.out.txt') -Pattern '^pairing code:\s+(.+?)\s*$' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($m) { $code = $m.Matches[0].Groups[1].Value } else { Start-Sleep -Milliseconds 500 }
    }
    if (-not $code) { throw 'sorrel share printed no pairing code within 60 s' }
    $join = Start-Process -FilePath $exe -PassThru `
        -ArgumentList @('join', "`"$code`"", '--direct', "/ip4/127.0.0.1/tcp/$port", '--into', $guestDir) `
        -RedirectStandardOutput (Join-Path $OutDir 'join.out.txt') -RedirectStandardError (Join-Path $OutDir 'join.err.txt')
    $synced = $false
    $deadline = (Get-Date).AddSeconds(120)
    while (-not $synced -and (Get-Date) -lt $deadline) {
        if ($join.HasExited) { throw "sorrel join exited with code $($join.ExitCode) before syncing" }
        $synced = (Test-Path (Join-Path $guestDir 'spike_times.npy')) -and (Test-Path (Join-Path $guestDir 'spike_clusters.npy')) -and
            (Test-Path (Join-Path $guestDir 'params.py'))
        if (-not $synced) { Start-Sleep -Milliseconds 500 }
    }
    if (-not $synced) { throw "the guest synced no files into $guestDir within 120 s" }
    Start-Sleep -Seconds 2 # the last file is renamed into place just before the files appear complete
    foreach ($f in 'spike_times.npy', 'spike_clusters.npy', 'params.py') {
        if ((Get-FileHash (Join-Path $guestDir $f)).Hash -ne (Get-FileHash (Join-Path $dataSh $f)).Hash) {
            throw "guest $f differs from the host's"
        }
    }
    Write-Host "share/join: guest synced $((Get-ChildItem $guestDir -File).Count) files over 127.0.0.1:$port"
}
finally {
    foreach ($p in $join, $share) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
    Get-Content (Join-Path $OutDir 'share.out.txt'), (Join-Path $OutDir 'join.out.txt') -ErrorAction SilentlyContinue | Write-Host
}

# 7. GUI: default backend selection, must survive $GuiSeconds
Remove-Item Env:WGPU_BACKEND -ErrorAction SilentlyContinue
$env:RUST_LOG = 'info'
$guiLog = Join-Path $OutDir 'gui.log'
$gui = Start-Process -FilePath $exe -ArgumentList @('run', '--standalone', $data) -PassThru `
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

# 7b. Launcher handoff. Without a terminal on stderr, bare `sorrel` and `sorrel open`
#     run the GUI in their own process instead of detaching, so that process is the
#     one window; every later `sorrel open` hands its dataset to it as a tab. The
#     window's registry file is %LOCALAPPDATA%\sorrel\gui.json.
$guiJson = Join-Path $env:LOCALAPPDATA 'sorrel\gui.json'
Remove-Item $guiJson -ErrorAction SilentlyContinue
$dataG1 = New-DataCopy 'tab1'
$dataG2 = New-DataCopy 'tab2'
$launcher = Start-Process -FilePath $exe -PassThru `
    -RedirectStandardOutput (Join-Path $OutDir 'launcher.out.txt') -RedirectStandardError (Join-Path $OutDir 'launcher.err.txt')
try {
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path $guiJson) -and (Get-Date) -lt $deadline) {
        if ($launcher.HasExited) { throw "bare sorrel exited with code $($launcher.ExitCode) instead of staying up as the GUI" }
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path $guiJson)) { throw "bare sorrel never wrote $guiJson" }
    $gj = Get-Content $guiJson -Raw | ConvertFrom-Json
    if ($gj.pid -ne $launcher.Id) { throw "gui.json names pid $($gj.pid), expected the launcher $($launcher.Id)" }
    $gj | Select-Object port, pid | ConvertTo-Json | Set-Content (Join-Path $OutDir 'gui-instance.json')
    # A second bare `sorrel` focuses the window and exits.
    Assert-Ok 'bare-again' (Invoke-Native 'bare-again' $exe @())
    Assert-Ok 'open-tab1' (Invoke-Native 'open-tab1' $exe @('open', $dataG1))
    Assert-Ok 'open-tab2' (Invoke-Native 'open-tab2' $exe @('open', $dataG2))
    foreach ($n in 'open-tab1', 'open-tab2') {
        if ((Get-Content (Join-Path $OutDir "$n.out.txt") -Raw) -notmatch 'opened in the running sorrel window') {
            throw "$n did not hand off to the running window"
        }
    }
    $leaves = @((Split-Path $dataG1 -Leaf), (Split-Path $dataG2 -Leaf))
    $tabs = @()
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if ($launcher.HasExited) { throw "the GUI exited with code $($launcher.ExitCode) while opening tabs" }
        Invoke-Native 'status-gui' $exe @('status', '--json') | Out-Null
        $gst = Get-Content (Join-Path $OutDir 'status-gui.out.txt') -Raw | ConvertFrom-Json
        $tabs = @($gst.gui.tabs | ForEach-Object { Split-Path $_.dataset -Leaf })
        if (@($leaves | Where-Object { $tabs -contains $_ }).Count -eq 2) { break }
        Start-Sleep -Seconds 1
    }
    if (@($leaves | Where-Object { $tabs -contains $_ }).Count -ne 2) {
        throw "the running window shows tabs [$($tabs -join ', ')], expected both $($leaves -join ', ')"
    }
    if (@($gst.gui.tabs | Where-Object { ([string]$_.dataset).StartsWith('\\?\') })) { throw 'status shows a verbatim tab path' }
    if ($gst.gui.pid -ne $launcher.Id) { throw "status names GUI pid $($gst.gui.pid), expected $($launcher.Id)" }
    Write-Host "launcher: pid $($launcher.Id) is the GUI with tabs [$($tabs -join ', ')]"
}
finally {
    if (-not $launcher.HasExited) { Stop-Process -Id $launcher.Id -Force }
    $null = $launcher.WaitForExit(15000)
    Remove-Item $guiJson -ErrorAction SilentlyContinue # a killed GUI cannot delete it
    Get-Content (Join-Path $OutDir 'launcher.out.txt'), (Join-Path $OutDir 'launcher.err.txt') -ErrorAction SilentlyContinue | Write-Host
}
if (Select-String -Path (Join-Path $OutDir 'launcher.err.txt') -Pattern 'panicked at' -Quiet) { throw 'launcher GUI log contains a panic' }

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
