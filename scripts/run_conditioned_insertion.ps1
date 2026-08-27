# Run paired learned/guarded insertion from every reset station and optional chain handoffs.

[CmdletBinding()]
param(
    [string]$IsaacPython = "C:\isaac-sim\python.bat",
    [string]$CheckpointRoot = "logs\rl_games\zero_g_blade_insertion_contact",
    [int[]]$Stations = (0..8),
    [int[]]$Seeds = @(1070, 2070, 3070),
    [int]$NumEnvs = 64,
    [int]$Episodes = 64,
    [switch]$IncludeChainHandoffs,
    [string]$OutputRoot = "artifacts\conditioned-insertion\v1",
    [string]$EvidencePath = "evidence\insertion_conditioned_controller_v1.json"
)

$ErrorActionPreference = "Stop"
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $taskRoot

$trackedStatus = git status --porcelain=v1 --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $trackedStatus) {
    throw "Conditioned evidence must run from a clean tracked worktree. Commit the implementation first."
}

$graspCheckpoint = Join-Path $CheckpointRoot "grapple_grasp_l0_seed70_v7m130\nn\last_zero_g_blade_insertion_contact_ep_3100_rew_30.262873.pth"
$extractCheckpoint = Join-Path $CheckpointRoot "grapple_extract_l0_seed70_v18pin\nn\last_zero_g_blade_insertion_contact_ep_12600_rew_172.70488.pth"
$insertCheckpoint = Join-Path $CheckpointRoot "grapple_insert_l0_seed70_v24rack\nn\last_zero_g_blade_insertion_contact_ep_2100_rew_43.909218.pth"
$expectedInsertSha256 = "47AA9EFB60F7794BE5CDD1EBD0AD5EC0E94CE00345BCF975D83AE9418D9A1B9F"

foreach ($checkpoint in @($graspCheckpoint, $extractCheckpoint, $insertCheckpoint)) {
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Checkpoint is unavailable: $checkpoint"
    }
}
$actualInsertSha256 = (Get-FileHash -LiteralPath $insertCheckpoint -Algorithm SHA256).Hash
if ($actualInsertSha256 -ne $expectedInsertSha256) {
    throw "v24 insertion checkpoint hash is $actualInsertSha256, expected $expectedInsertSha256"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$rawRuns = [System.Collections.Generic.List[string]]::new()

function Invoke-ConditionedRun {
    param(
        [Parameter(Mandatory)][ValidateSet("guarded", "policy")][string]$Controller,
        [Parameter(Mandatory)][int]$Seed,
        [Parameter(Mandatory)][string]$Tag,
        [Parameter(Mandatory)][string[]]$ExtraArguments
    )

    $metrics = Join-Path $OutputRoot "$Tag.npz"
    $report = Join-Path $OutputRoot "${Tag}_report.json"
    $log = Join-Path $OutputRoot "$Tag.log"
    foreach ($path in @($metrics, $report, $log)) {
        if (Test-Path -LiteralPath $path) {
            throw "Refusing to overwrite preserved run artifact: $path"
        }
    }
    $runArguments = @(
        "-u", "scripts/run_workflow_demo.py", "--headless",
        "--task", "Isaac-ZeroG-Blade-GrapplePin-TwoSlotWorkflow-v0",
        "--workflow", "relocate", "--curriculum_stage", "0",
        "--grasp_checkpoint", $graspCheckpoint,
        "--extract_checkpoint", $extractCheckpoint,
        "--insert_controller", $Controller,
        "--seed", "$Seed", "--report", $report, "--episode_metrics", $metrics
    )
    if ($Controller -eq "policy") {
        $runArguments += @("--insert_checkpoint", $insertCheckpoint)
    }
    $runArguments += $ExtraArguments
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $Tag"
    & $IsaacPython @runArguments *> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac run failed ($LASTEXITCODE): see $log"
    }
    $rawRuns.Add($metrics)
}

foreach ($station in $Stations) {
    foreach ($seed in $Seeds) {
        foreach ($controller in @("guarded", "policy")) {
            $tag = "station_{0:D2}_seed{1}_{2}" -f $station, $seed, $controller
            Invoke-ConditionedRun -Controller $controller -Seed $seed -Tag $tag -ExtraArguments @(
                "--start_insert_station", "$station", "--steps", "0",
                "--num_envs", "$NumEnvs", "--episodes", "$Episodes"
            )
        }
    }
}

if ($IncludeChainHandoffs) {
    foreach ($seed in @(4070, 5070, 6070)) {
        foreach ($controller in @("guarded", "policy")) {
            $tag = "chain_handoff_seed${seed}_${controller}"
            $trace = Join-Path $OutputRoot "${tag}_trace.npz"
            if (Test-Path -LiteralPath $trace) {
                throw "Refusing to overwrite preserved run artifact: $trace"
            }
            Invoke-ConditionedRun -Controller $controller -Seed $seed -Tag $tag -ExtraArguments @(
                "--steps", "5000", "--num_envs", "32", "--episodes", "32",
                "--robot_rail_on_relocation", "--latch_on_release", "--latch_joint_mode", "fixed",
                "--latch_rated_force_n", "20000", "--latch_rated_torque_nm", "1000",
                "--latch_position_stiffness_n_per_m", "40000",
                "--latch_rotation_stiffness_nm_per_rad", "20000",
                "--destination_channel_relief_m", "0.0046125", "--mating_mode", "compliant",
                "--mating_force_cap_n", "1000", "--handoff_trace", $trace
            )
        }
    }
}

if (Test-Path -LiteralPath $EvidencePath) {
    throw "Refusing to overwrite preserved evidence: $EvidencePath"
}
& .\.venv\Scripts\python.exe scripts/report_conditioned_insertion.py `
    --runs @rawRuns `
    --expected_policy_sha256 $expectedInsertSha256 `
    --output $EvidencePath
if ($LASTEXITCODE -ne 0) {
    throw "Conditioned report aggregation failed with exit $LASTEXITCODE"
}
Write-Host "Wrote $EvidencePath from $($rawRuns.Count) preserved controller arms."
