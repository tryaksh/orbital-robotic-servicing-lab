[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$Execute,
    [string]$IsaacSimRoot = 'C:\isaac-sim',
    [string]$OldIsaacLabRoot = 'C:\IsaacLab',
    [string]$CondaRoot = 'C:\Users\tryak\miniconda3'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ArtifactsRoot = Join-Path $ProjectRoot 'artifacts'
$ValidationMarker = Join-Path $ArtifactsRoot 'sim_validation.json'
$ValidationScript = Join-Path $PSScriptRoot 'validate_sim.py'
$SimPython = Join-Path $IsaacSimRoot 'python.bat'

$DocsPdfName = 'Installation using Isaac Sim Pre-built Binaries ' + [char]0x2014 + ' Isaac Lab Documentation.pdf'
$ExactTargets = @(
    [System.IO.Path]::GetFullPath($OldIsaacLabRoot),
    [System.IO.Path]::GetFullPath('C:\Users\tryak\Downloads\isaac-sim-standalone-5.1.0-windows-x86_64.zip'),
    [System.IO.Path]::GetFullPath((Join-Path 'C:\Users\tryak\Downloads' $DocsPdfName))
)
$CondaEnvPath = [System.IO.Path]::GetFullPath((Join-Path $CondaRoot 'envs\env_isaaclab'))

function Assert-SafeExactPath {
    param([Parameter(Mandatory)][string]$Path)

    $full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    $forbidden = @(
        [System.IO.Path]::GetPathRoot($full).TrimEnd('\'),
        [Environment]::GetFolderPath('UserProfile').TrimEnd('\'),
        $ProjectRoot.TrimEnd('\'),
        $IsaacSimRoot.TrimEnd('\'),
        $CondaRoot.TrimEnd('\')
    )
    if ($forbidden -contains $full) {
        throw "Refusing destructive operation against broad or retained path: $full"
    }
    return $full
}

function Get-PathSizeGiB {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) { return 0.0 }
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) {
        return [math]::Round($item.Length / 1GB, 3)
    }
    $bytes = (Get-ChildItem -LiteralPath $Path -File -Recurse -Force -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    return [math]::Round($bytes / 1GB, 3)
}

function Get-FreeGiB {
    $drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($IsaacSimRoot).Substring(0, 1))
    return [math]::Round($drive.Free / 1GB, 2)
}

function Stop-IsaacProcesses {
    $allowedRoots = @(
        [System.IO.Path]::GetFullPath($IsaacSimRoot).TrimEnd('\') + '\',
        [System.IO.Path]::GetFullPath($OldIsaacLabRoot).TrimEnd('\') + '\'
    )
    foreach ($process in Get-Process -ErrorAction SilentlyContinue) {
        try {
            $path = $process.Path
            if ([string]::IsNullOrWhiteSpace($path)) { continue }
            $fullPath = [System.IO.Path]::GetFullPath($path)
            $inScope = $false
            foreach ($root in $allowedRoots) {
                if ($fullPath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $inScope = $true
                    break
                }
            }
            if ($inScope -and $process.ProcessName -match '^(kit|isaac-sim|python|pythonw)$') {
                Write-Host "Stopping verified Isaac process $($process.Id): $fullPath"
                Stop-Process -Id $process.Id -Force
            }
        } catch {
            # Access-denied process paths are outside the verified deletion scope.
        }
    }
}

Write-Host 'Retained simulator:' $IsaacSimRoot
if (-not (Test-Path -LiteralPath $SimPython -PathType Leaf)) {
    throw "Retained Isaac Sim Python was not found at $SimPython"
}

$rows = foreach ($target in @($ExactTargets + $CondaEnvPath)) {
    [pscustomobject]@{
        Path = $target
        Exists = Test-Path -LiteralPath $target
        SizeGiB = Get-PathSizeGiB -Path $target
    }
}
$rows | Format-Table -AutoSize

Write-Host "Preserved explicitly: $IsaacSimRoot, Omniverse/NVIDIA caches, generic package caches, Docker, D:\Isaac_Robots"
if (-not $Execute) {
    Write-Host 'Inspection only. Re-run with -Execute to validate Sim and permanently delete the exact targets above.'
    return
}

New-Item -ItemType Directory -Path $ArtifactsRoot -Force | Out-Null
Write-Host 'Validating the retained simulator before deletion...'
& $SimPython $ValidationScript --output $ValidationMarker
if ($LASTEXITCODE -ne 0) {
    throw "Isaac Sim validation failed with exit code $LASTEXITCODE; nothing was deleted."
}
$validation = Get-Content -LiteralPath $ValidationMarker -Raw | ConvertFrom-Json
if ($validation.success -ne $true -or $validation.cuda_available -ne $true) {
    throw 'Isaac Sim did not produce a successful CUDA validation marker; nothing was deleted.'
}

Stop-IsaacProcesses
$freeBefore = Get-FreeGiB

$condaExe = Join-Path $CondaRoot 'Scripts\conda.exe'
if (Test-Path -LiteralPath $CondaEnvPath) {
    Assert-SafeExactPath -Path $CondaEnvPath | Out-Null
    if (Test-Path -LiteralPath $condaExe -PathType Leaf) {
        if ($PSCmdlet.ShouldProcess($CondaEnvPath, 'Remove Conda environment env_isaaclab')) {
            & $condaExe env remove --name env_isaaclab --yes
            if ($LASTEXITCODE -ne 0) {
                throw "Conda failed to remove env_isaaclab (exit $LASTEXITCODE)."
            }
        }
    } elseif ($PSCmdlet.ShouldProcess($CondaEnvPath, 'Remove verified orphan Conda environment directory')) {
        Remove-Item -LiteralPath $CondaEnvPath -Recurse -Force
    }
}

foreach ($target in $ExactTargets) {
    $safeTarget = Assert-SafeExactPath -Path $target
    if ((Test-Path -LiteralPath $safeTarget) -and $PSCmdlet.ShouldProcess($safeTarget, 'Permanently remove obsolete Isaac artifact')) {
        Remove-Item -LiteralPath $safeTarget -Recurse -Force
    }
}

$remaining = @($ExactTargets + $CondaEnvPath) | Where-Object { Test-Path -LiteralPath $_ }
if ($remaining.Count -gt 0) {
    throw "Cleanup incomplete. Remaining targets: $($remaining -join ', ')"
}

$freeAfter = Get-FreeGiB
Write-Host "Cleanup complete. C: free space before/after: $freeBefore GiB / $freeAfter GiB."
Write-Host 'The retained simulator and caches were not removed. Deleted targets are not recoverable.'
