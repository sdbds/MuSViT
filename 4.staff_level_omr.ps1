[CmdletBinding()]
param (
    [switch]$DryRun,
    [string]$DataPath,
    [string]$DatasetBundlePath,
    [string]$ExperimentName
)

$ErrorActionPreference = "Stop"

function Format-CanonicalDecimal {
    param (
        [Parameter(Mandatory = $true)]
        [double]$Value
    )

    return $Value.ToString(
        "0.0################",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Resolve-InputDirectory {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$SettingName
    )

    $Candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    }
    else {
        Join-Path $PSScriptRoot $Path
    }
    if (-not (Test-Path -LiteralPath $Candidate -PathType Container)) {
        throw "$SettingName does not exist or is not a directory: $Candidate"
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

# This launcher owns operational defaults only. Dataset identity still comes
# from a prepared v2 bundle supplied by the caller.
$Config = @{
    experiment_name        = "catedrales"
    data_path              = $null
    dataset_bundle_path    = $null
    model_name             = "musvit"
    method                 = "lora"
    patch_rows             = 8
    patch_cols             = 128
    augmentation_profile   = "staff_omr_train_v1"
    batch_size             = 8
    num_workers            = 6
    learning_rate          = 0.0003
    max_epochs             = 1000
    start_eval             = 20
    patience               = 30
    seed                   = 7
    device                 = "cuda"
    verify_image_hashes    = "always"
}

$Runtime = @{
    cuda_visible_devices = "1"
}

Set-Location $PSScriptRoot

if ($Env:PYTHONPATH) {
    $Env:PYTHONPATH = "$PSScriptRoot;$Env:PYTHONPATH"
}
else {
    $Env:PYTHONPATH = $PSScriptRoot
}
$Env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
$TokenCandidates = @(
    (Join-Path $Env:USERPROFILE ".cache\huggingface\token"),
    (Join-Path $Env:LOCALAPPDATA "huggingface\token")
)
if ([string]::IsNullOrWhiteSpace($Env:HF_TOKEN)) {
    foreach ($TokenPath in $TokenCandidates) {
        if (Test-Path -LiteralPath $TokenPath -PathType Leaf) {
            $Env:HF_TOKEN = (Get-Content -LiteralPath $TokenPath -Raw).Trim()
            if (-not [string]::IsNullOrWhiteSpace($Env:HF_TOKEN)) {
                break
            }
        }
    }
}
if ($Env:CUDA_PATH) {
    $Env:CUDA_HOME = $Env:CUDA_PATH
}
if ($Env:LOCALAPPDATA) {
    $Env:UV_CACHE_DIR = Join-Path $Env:LOCALAPPDATA "uv\cache"
}
$Env:UV_NO_CACHE = "0"
$Env:UV_LINK_MODE = "symlink"
if (-not [string]::IsNullOrWhiteSpace($Runtime.cuda_visible_devices)) {
    $Env:CUDA_VISIBLE_DEVICES = [string]$Runtime.cuda_visible_devices
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv or add it to PATH."
}

if (-not [string]::IsNullOrWhiteSpace($ExperimentName)) {
    $Config.experiment_name = $ExperimentName
}
if ($Config.experiment_name -notmatch "^[A-Za-z0-9._-]{1,64}$" -or
    $Config.experiment_name -in @(".", "..")) {
    throw "experiment_name must contain 1-64 ASCII letters, digits, '.', '_', or '-'."
}
if ($Config.model_name -notin @("musvit", "musvit_light")) {
    throw "Unsupported model_name '$($Config.model_name)'."
}
if ($Config.method -notin @("linear_probe", "lora")) {
    throw "Unsupported method '$($Config.method)'."
}

$PatchRows = 0
$PatchColumns = 0
if (-not [int]::TryParse([string]$Config.patch_rows, [ref]$PatchRows) -or
    -not [int]::TryParse([string]$Config.patch_cols, [ref]$PatchColumns) -or
    $PatchRows -lt 1 -or $PatchColumns -lt 1) {
    throw "patch_rows and patch_cols must be positive integers."
}

$BatchSize = 0
$NumWorkers = 0
$MaxEpochs = 0
$StartEval = 0
$Patience = 0
$Seed = 0
if (-not [int]::TryParse([string]$Config.batch_size, [ref]$BatchSize) -or
    $BatchSize -lt 1) {
    throw "batch_size must be a positive integer."
}
if (-not [int]::TryParse([string]$Config.num_workers, [ref]$NumWorkers) -or
    $NumWorkers -lt 0) {
    throw "num_workers must be a non-negative integer."
}
if (-not [int]::TryParse([string]$Config.max_epochs, [ref]$MaxEpochs) -or
    $MaxEpochs -lt 1) {
    throw "max_epochs must be a positive integer."
}
if (-not [int]::TryParse([string]$Config.start_eval, [ref]$StartEval) -or
    $StartEval -lt 1 -or $StartEval -gt $MaxEpochs) {
    throw "start_eval must be between 1 and max_epochs."
}
if (-not [int]::TryParse([string]$Config.patience, [ref]$Patience) -or
    $Patience -lt 1) {
    throw "patience must be a positive integer."
}
if (-not [int]::TryParse([string]$Config.seed, [ref]$Seed) -or $Seed -lt 0) {
    throw "seed must be a non-negative integer."
}

$LearningRate = 0.0
if (-not [double]::TryParse(
        [string]$Config.learning_rate,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$LearningRate
    ) -or
    [double]::IsNaN($LearningRate) -or
    [double]::IsInfinity($LearningRate) -or
    $LearningRate -le 0) {
    throw "learning_rate must be a positive finite number."
}

$RequestedDataPath = if (-not [string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath
}
else {
    $Config.data_path
}
if ([string]::IsNullOrWhiteSpace($RequestedDataPath)) {
    throw "data_path is required. Pass -DataPath or set Config.data_path."
}
$ResolvedDataPath = Resolve-InputDirectory `
    -Path $RequestedDataPath `
    -SettingName "data_path"
$HasPair = Get-ChildItem `
    -LiteralPath $ResolvedDataPath `
    -Recurse `
    -File `
    -Filter "*_region.png" |
    Where-Object {
        $GroundTruthName = $_.Name -replace "_region\.png$", "_gt.txt"
        Test-Path `
            -LiteralPath (Join-Path $_.DirectoryName $GroundTruthName) `
            -PathType Leaf
    } |
    Select-Object -First 1
if ($null -eq $HasPair) {
    throw "data_path must contain at least one paired *_region.png and *_gt.txt sample: $ResolvedDataPath"
}

$RequestedBundlePath = if (
    -not [string]::IsNullOrWhiteSpace($DatasetBundlePath)
) {
    $DatasetBundlePath
}
else {
    $Config.dataset_bundle_path
}
if ([string]::IsNullOrWhiteSpace($RequestedBundlePath)) {
    throw "dataset_bundle_path is required. Pass -DatasetBundlePath or set Config.dataset_bundle_path."
}
$ResolvedBundlePath = Resolve-InputDirectory `
    -Path $RequestedBundlePath `
    -SettingName "dataset_bundle_path"
foreach ($RequiredFile in @(
    "bundle.json",
    "split_manifest.json",
    "vocabulary.json",
    "image_verification_index.json"
)) {
    $RequiredPath = Join-Path $ResolvedBundlePath $RequiredFile
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "dataset_bundle_path is missing required file '$RequiredFile': $ResolvedBundlePath"
    }
}

$UvArgs = [System.Collections.ArrayList]::new()
foreach ($Argument in @(
    "run",
    "--frozen",
    "musvit",
    "staff-level-omr",
    "train",
    "--experiment_name=$($Config.experiment_name)",
    "--data_path=$ResolvedDataPath",
    "--dataset_bundle_path=$ResolvedBundlePath",
    "--model_name=$($Config.model_name)",
    "--method=$($Config.method)",
    "--patch_rows=$PatchRows",
    "--patch_cols=$PatchColumns",
    "--augmentation_profile=$($Config.augmentation_profile)",
    "--batch_size=$BatchSize",
    "--num_workers=$NumWorkers",
    "--learning_rate=$(Format-CanonicalDecimal $LearningRate)",
    "--max_epochs=$MaxEpochs",
    "--start_eval=$StartEval",
    "--patience=$Patience",
    "--seed=$Seed",
    "--device=$($Config.device)",
    "--verify_image_hashes=$($Config.verify_image_hashes)"
)) {
    [void]$UvArgs.Add($Argument)
}

$DisplayArgs = $UvArgs | ForEach-Object {
    if ($_ -match "\s") {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    else {
        $_
    }
}

Write-Output "Starting trusted staff-level OMR v2 training..."
Write-Output "GPU selection: CUDA_VISIBLE_DEVICES=$Env:CUDA_VISIBLE_DEVICES"
Write-Output "Dataset path: $ResolvedDataPath"
Write-Output "Dataset bundle: $ResolvedBundlePath"
Write-Output "Command: uv $($DisplayArgs -join ' ')"

if ($DryRun) {
    Write-Output "Dry run finished; training was not started."
    return
}

& uv @UvArgs
if ($LASTEXITCODE -ne 0) {
    throw "Staff-level OMR v2 training failed with exit code $LASTEXITCODE."
}
Write-Output "Staff-level OMR v2 training finished."
