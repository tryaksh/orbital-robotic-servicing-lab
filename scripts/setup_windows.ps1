[CmdletBinding()]
param(
    [string]$IsaacSimRoot = 'C:\isaac-sim',
    [switch]$SkipLongPaths,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$ExpectedLabCommit = '37ddf626871758333d6ed89cf64ad702aef127d0'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$DependenciesRoot = Join-Path $ProjectRoot '.deps'
$IsaacLabRoot = Join-Path $DependenciesRoot 'IsaacLab'
$IsaacSimRoot = [System.IO.Path]::GetFullPath($IsaacSimRoot)
$SimPython = Join-Path $IsaacSimRoot 'python.bat'
$SimPackagePython = Join-Path $IsaacSimRoot 'kit\python\python.exe'
$LabLauncher = Join-Path $IsaacLabRoot 'isaaclab.bat'
$SimJunction = Join-Path $IsaacLabRoot '_isaac_sim'

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Path -LiteralPath $SimPython -PathType Leaf)) {
    throw "Isaac Sim 5.1 was not found at $IsaacSimRoot"
}
if (-not (Test-Path -LiteralPath $SimPackagePython -PathType Leaf)) {
    throw "Isaac Sim package Python was not found at $SimPackagePython"
}

if (-not $SkipLongPaths) {
    $registryPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem'
    $enabled = (Get-ItemProperty -LiteralPath $registryPath -Name LongPathsEnabled).LongPathsEnabled
    if ($enabled -ne 1) {
        if (-not (Test-IsAdministrator)) {
            Write-Warning 'Windows long paths are disabled and this shell is not elevated. Continuing with Git core.longpaths; re-run elevated later if Windows reports a long-path error.'
        } else {
            Set-ItemProperty -LiteralPath $registryPath -Name LongPathsEnabled -Type DWord -Value 1
            Write-Host 'Enabled Windows long paths.'
        }
    }
}

git -c "safe.directory=$($ProjectRoot.Replace('\', '/'))" -C $ProjectRoot config --local core.longpaths true
if ($LASTEXITCODE -ne 0) { throw 'Unable to enable repository-local core.longpaths.' }

New-Item -ItemType Directory -Path $DependenciesRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $IsaacLabRoot '.git'))) {
    if (Test-Path -LiteralPath $IsaacLabRoot) {
        throw "$IsaacLabRoot exists but is not the expected Git checkout. Remove it explicitly and retry."
    }
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git $IsaacLabRoot
    if ($LASTEXITCODE -ne 0) { throw 'Isaac Lab clone failed.' }
}

$resolvedCommit = (git -c "safe.directory=$($IsaacLabRoot.Replace('\', '/'))" -C $IsaacLabRoot rev-parse HEAD).Trim()
if ($resolvedCommit -ne $ExpectedLabCommit) {
    throw "Isaac Lab commit mismatch: expected $ExpectedLabCommit, found $resolvedCommit"
}

if (Test-Path -LiteralPath $SimJunction) {
    $target = (Get-Item -LiteralPath $SimJunction -Force).Target
    if ([System.IO.Path]::GetFullPath($target) -ne $IsaacSimRoot) {
        throw "Existing _isaac_sim link points to $target rather than $IsaacSimRoot"
    }
} else {
    New-Item -ItemType Junction -Path $SimJunction -Target $IsaacSimRoot | Out-Null
}

if (-not $SkipInstall) {
    # Isaac Lab's batch helper prefers CONDA_PREFIX when one is inherited and
    # loops through optional packages (including Jupyter-heavy mimic tooling).
    # Install only the essential packages explicitly against Sim's Python. This
    # also lets us work around flatdict's broken isolated setuptools build.
    $savedCondaPrefix = $env:CONDA_PREFIX
    $savedCondaDefaultEnv = $env:CONDA_DEFAULT_ENV
    $savedPythonExe = $env:PYTHONEXE
    try {
        Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
        Remove-Item Env:PYTHONEXE -ErrorAction SilentlyContinue

        # python.bat selects a driver-optimized kit.exe name and is unreliable
        # when invoked from an elevated shell.  Package management can safely
        # use the same interpreter's python.exe while preserving Sim's paths.
        $env:CARB_APP_PATH = Join-Path $IsaacSimRoot 'kit'
        $env:ISAAC_PATH = $IsaacSimRoot
        $env:EXP_PATH = Join-Path $IsaacSimRoot 'apps'
        $env:PYTHONPATH = (Join-Path $IsaacSimRoot 'site')

        & $SimPackagePython -m pip install 'setuptools<82'
        if ($LASTEXITCODE -ne 0) { throw 'Unable to pin a pkg_resources-compatible setuptools.' }
        & $SimPackagePython -m pip install --no-build-isolation 'flatdict==4.0.1'
        if ($LASTEXITCODE -ne 0) { throw 'Unable to build flatdict with the simulator environment.' }

        $editablePackages = @(
            (Join-Path $IsaacLabRoot 'source\isaaclab'),
            (Join-Path $IsaacLabRoot 'source\isaaclab_assets'),
            (Join-Path $IsaacLabRoot 'source\isaaclab_tasks')
        )
        foreach ($package in $editablePackages) {
            & $SimPackagePython -m pip install --no-build-isolation -e $package
            if ($LASTEXITCODE -ne 0) { throw "Essential Isaac Lab package installation failed: $package" }
        }
        $rlPackage = Join-Path $IsaacLabRoot 'source\isaaclab_rl'
        & $SimPackagePython -m pip install --no-build-isolation -e $rlPackage
        if ($LASTEXITCODE -ne 0) { throw 'Isaac Lab RL package installation failed.' }

        & $SimPackagePython -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('rl_games') is None)"
        if ($LASTEXITCODE -ne 0) {
            # RL-Games uses Poetry as its PEP-517 backend. Keep isolation enabled
            # for this dependency and pin the exact commit resolved by Lab 2.3.2.
            & $SimPackagePython -m pip install 'rl-games @ git+https://github.com/isaac-sim/rl_games.git@6b3534f29568158e9e29ec8bf83cc88fce5f0cae'
            if ($LASTEXITCODE -ne 0) { throw 'Pinned RL-Games installation failed.' }
        }
    } finally {
        if ($null -ne $savedCondaPrefix) { $env:CONDA_PREFIX = $savedCondaPrefix }
        if ($null -ne $savedCondaDefaultEnv) { $env:CONDA_DEFAULT_ENV = $savedCondaDefaultEnv }
        if ($null -ne $savedPythonExe) { $env:PYTHONEXE = $savedPythonExe }
    }

    & $SimPackagePython -m pip install -e "${ProjectRoot}[dev]"
    if ($LASTEXITCODE -ne 0) { throw "Project installation failed with exit code $LASTEXITCODE" }

    & $SimPackagePython -c "import importlib.util, sys; names=('isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl','rl_games'); missing=[n for n in names if importlib.util.find_spec(n) is None]; print('resolved packages:', names); print('missing packages:', missing); sys.exit(bool(missing))"
    if ($LASTEXITCODE -ne 0) { throw 'Installation import gate failed.' }
}

& $SimPackagePython $PSScriptRoot\write_environment_lock.py
if ($LASTEXITCODE -ne 0) { throw 'Failed to write environment-lock.local.json.' }

Write-Host 'Isaac Lab setup complete.'
Write-Host "Simulator: $IsaacSimRoot"
Write-Host "Isaac Lab: $IsaacLabRoot ($resolvedCommit)"
Write-Host 'Run scripts through C:\isaac-sim\python.bat.'
