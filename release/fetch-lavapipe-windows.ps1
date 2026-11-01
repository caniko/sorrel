<#
Fetch a pinned software Vulkan for Windows into -Dir and print the path of its
ICD file: Mesa's lavapipe (the Windows build of the dev shell's
`SORREL_LAVAPIPE_ICD`; nix/devshell.nix pins Mesa 26.2.4 too, so the UI
snapshot tests rasterise on the same Mesa version as the committed reference
PNGs) plus the Khronos Vulkan loader, which a hosted runner without a GPU
driver lacks. Put -Dir on PATH so `vulkan-1.dll` is found, and point
`VK_DRIVER_FILES` (or `SORREL_LAVAPIPE_ICD`) at the printed ICD.

  pwsh release/fetch-lavapipe-windows.ps1 [-Dir DIR]   # prints DIR\lvp_icd.x86_64.json

Mesa: pal1000/mesa-dist-win, MSVC release. Loader: LunarG Vulkan Runtime
components. Bump each version with its sha256.
#>
param([string]$Dir = (Join-Path ([IO.Path]::GetTempPath()) 'sorrel-lavapipe'))
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Mesa = '26.2.4'
$MesaSha256 = '351fc8c8b695878ffb3eaa044b3ead08672a48b1a045e3c3e3975811df0f6695'
$Loader = '1.4.363.0'
$LoaderSha256 = 'a25a927aa8b9f0371048f1861cf88ac3b9bc9b1fb332c42d897c8ab32695769a'

function Get-Pinned([string]$Url, [string]$Sha256, [string]$File) {
    Invoke-WebRequest $Url -OutFile $File -UseBasicParsing
    $got = (Get-FileHash $File -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $Sha256) { throw "$Url has sha256 $got, expected $Sha256" }
}

New-Item -ItemType Directory -Force $Dir | Out-Null
$icd = Join-Path $Dir 'lvp_icd.x86_64.json'
if (-not (Test-Path (Join-Path $Dir 'vulkan_lvp.dll'))) {
    $archive = Join-Path $Dir "mesa3d-$Mesa-release-msvc.7z"
    Get-Pinned "https://github.com/pal1000/mesa-dist-win/releases/download/$Mesa/mesa3d-$Mesa-release-msvc.7z" $MesaSha256 $archive
    $7z = (Get-Command 7z -ErrorAction SilentlyContinue).Source
    if (-not $7z) { $7z = Join-Path $env:ProgramFiles '7-Zip\7z.exe' }
    # `e` flattens x64\ so the ICD's relative library_path finds the DLL.
    & $7z e -y "-o$Dir" $archive 'x64/vulkan_lvp.dll' 'x64/lvp_icd.x86_64.json' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z failed with exit code $LASTEXITCODE" }
    Remove-Item $archive
}
if (-not (Test-Path (Join-Path $Dir 'vulkan-1.dll'))) {
    $zip = Join-Path $Dir 'vulkan-runtime.zip'
    Get-Pinned "https://sdk.lunarg.com/sdk/download/$Loader/windows/vulkan-runtime-components.zip" $LoaderSha256 $zip
    $tmp = Join-Path $Dir 'loader'
    Expand-Archive $zip -DestinationPath $tmp -Force
    Move-Item (Join-Path $tmp "VulkanRT-X64-$Loader-Components\x64\vulkan-1.dll") $Dir -Force
    Remove-Item $zip, $tmp -Recurse -Force
}
if (-not (Test-Path $icd)) { throw "$icd missing after extraction" }
$icd
