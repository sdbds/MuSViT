[CmdletBinding()]
param (
    [string]$CheckpointPath = "experiments/full_page_omr/weights/polish_scores_cl_CL-step=830001-val_SER_v2=10.6498.ckpt",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-InputFile {
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
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        throw "$SettingName does not exist: $Candidate"
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

function Format-Command {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $DisplayArguments = $Arguments | ForEach-Object {
        if ($_ -match "\s") {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    }
    return "$Executable $($DisplayArguments -join ' ')"
}

$PythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Project Python does not exist: $PythonPath. Run .\1.install-uv-qinglong.ps1 first."
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$ResolvedCheckpointPath = Resolve-InputFile `
    -Path $CheckpointPath `
    -SettingName "CheckpointPath"
$EncoderConfigPath = Resolve-InputFile `
    -Path "experiments/full_page_omr/config/LSMT-MAE-Base-1024-16.json" `
    -SettingName "EncoderConfigPath"

$CheckpointItem = Get-Item -LiteralPath $ResolvedCheckpointPath
$OutputStem = [System.IO.Path]::GetFileNameWithoutExtension($CheckpointItem.Name)
$WeightsPath = Join-Path $CheckpointItem.DirectoryName "$OutputStem.safetensors"
$ModelConfigPath = Join-Path $CheckpointItem.DirectoryName "$OutputStem.config.json"
$OutputDirectory = Join-Path $CheckpointItem.DirectoryName "${OutputStem}_onnx"

$ConvertArguments = @(
    "-m",
    "experiments.full_page_omr.convert_checkpoint",
    "--checkpoint-path",
    $ResolvedCheckpointPath,
    "--weights-path",
    $WeightsPath,
    "--config-path",
    $ModelConfigPath
)
$ExportArguments = @(
    "-m",
    "experiments.full_page_omr.export_onnx",
    "--weights-path",
    $WeightsPath,
    "--model-config-path",
    $ModelConfigPath,
    "--encoder-config-path",
    $EncoderConfigPath,
    "--output-dir",
    $OutputDirectory,
    "--device",
    $Device,
    "--verify-dataset"
)

Write-Output "Checkpoint: $ResolvedCheckpointPath"
Write-Output "Inference weights: $WeightsPath"
Write-Output "Model config: $ModelConfigPath"
Write-Output "ONNX bundle: $OutputDirectory"
Write-Output "Convert command: $(Format-Command -Executable $PythonPath -Arguments $ConvertArguments)"
Write-Output "Export command: $(Format-Command -Executable $PythonPath -Arguments $ExportArguments)"

if ($DryRun) {
    Write-Output "Dry run finished; no files were written."
    return
}

& $PythonPath -c "import onnx, onnxruntime, safetensors, torch"
if ($LASTEXITCODE -ne 0) {
    throw "The .venv ONNX export dependencies are incomplete."
}

Write-Output "Extracting inference-only checkpoint assets..."
& $PythonPath @ConvertArguments
if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint conversion failed with exit code $LASTEXITCODE."
}

Write-Output "Exporting and validating the ONNX bundle..."
& $PythonPath @ExportArguments
if ($LASTEXITCODE -ne 0) {
    throw "ONNX export failed with exit code $LASTEXITCODE."
}

$MetadataPath = Join-Path $OutputDirectory "metadata.json"
if (-not (Test-Path -LiteralPath $MetadataPath -PathType Leaf)) {
    throw "ONNX export did not produce metadata: $MetadataPath"
}
$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
if ($Metadata.bundle_status -ne "complete" -or
    $Metadata.validation.status -ne "passed" -or
    $Metadata.validation.dataset_parity.status -ne "passed") {
    throw "ONNX bundle validation did not pass: $MetadataPath"
}

Write-Output "ONNX export and locked-dataset validation passed."
Write-Output "Bundle: $OutputDirectory"
