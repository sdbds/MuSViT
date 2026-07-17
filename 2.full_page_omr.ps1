[CmdletBinding()]
param (
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

#region Configuration

$Config = @{
    config_path            = "experiments/full_page_omr/config/Polish_Scores/finetuning.json"
    experiment_name        = "polish_scores_cl"
    foundation_architecture = "ViTMAEBase"
    foundation_weights     = "carlospm12/LSMT-MAE-Base-1024-16"
    finetuning             = "CL"     # Supported: CL, SR, CL1
    encoder_training_mode  = "fine_tune" # Supported: fine_tune, linear_probe
    resolution             = $null    # $null uses the experiment default (1024)
    max_steps              = -1       # -1 lets Lightning train without a step limit
    checkpoint_every_n_epochs = 100   # Periodic full checkpoint interval
    learning_rate          = $null    # $null uses the experiment default (1e-4)
    attention_backend      = "auto"   # auto: FA2 -> SDPA -> eager fallback
    from_checkpoint        = $null    # Start a fresh Trainer run
    starting_weights       = $null    # Initialize model weights from a .ckpt file
}

$Features = @{
    train = $true
}

$Runtime = @{
    cuda_visible_devices = "0"       # Use $null to leave CUDA device selection unchanged
    cairo_dll_directory  = "E:\Roaming\baidu\BaiduNetdisk\module\ImageViewer"
    windows_num_workers  = 24          # Best measured long-run throughput on this workstation
}

#endregion

#region Environment Setup

Set-Location $PSScriptRoot

if ($Env:PYTHONPATH) {
    $Env:PYTHONPATH = "$PSScriptRoot;$Env:PYTHONPATH"
}
else {
    $Env:PYTHONPATH = $PSScriptRoot
}

$Env:HF_HOME = Join-Path $PSScriptRoot ".cache\huggingface"
$TokenCandidates = @(
    (Join-Path $env:USERPROFILE ".cache\huggingface\token"),
    (Join-Path $env:LOCALAPPDATA "huggingface\token")
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

if ($null -ne $Runtime.cuda_visible_devices -and $Runtime.cuda_visible_devices -ne "") {
    $Env:CUDA_VISIBLE_DEVICES = [string]$Runtime.cuda_visible_devices
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Run .\1.install-uv-qinglong.ps1 first or add uv to PATH."
}

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

function New-WindowsRuntimeConfig {
    param (
        [Parameter(Mandatory = $true)]
        [string]$SourcePath
    )

    $Document = Get-Content -LiteralPath $SourcePath -Raw | ConvertFrom-Json
    if ($null -eq $Document.data) {
        throw "The OMR config does not contain a data section: $SourcePath"
    }

    $Document.data.num_workers = [int]$Runtime.windows_num_workers
    $RuntimeConfigDirectory = Join-Path $PSScriptRoot ".cache\full_page_omr"
    New-Item -ItemType Directory -Path $RuntimeConfigDirectory -Force | Out-Null
    $RuntimeConfigPath = Join-Path $RuntimeConfigDirectory "windows-finetuning.json"
    $Document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $RuntimeConfigPath -Encoding ASCII
    return (Resolve-Path -LiteralPath $RuntimeConfigPath).Path
}

function Find-CairoDllDirectory {
    $CandidateDirectories = [System.Collections.ArrayList]::new()

    if (-not [string]::IsNullOrWhiteSpace($Runtime.cairo_dll_directory)) {
        [void]$CandidateDirectories.Add($Runtime.cairo_dll_directory)
    }
    if ($Env:CAIROCFFI_DLL_DIRECTORIES) {
        foreach ($Directory in ($Env:CAIROCFFI_DLL_DIRECTORIES -split [System.IO.Path]::PathSeparator)) {
            [void]$CandidateDirectories.Add($Directory)
        }
    }
    if ($Env:PATH) {
        foreach ($Directory in ($Env:PATH -split [System.IO.Path]::PathSeparator)) {
            [void]$CandidateDirectories.Add($Directory)
        }
    }

    [void]$CandidateDirectories.Add("C:\Program Files\GTK3-Runtime Win64\bin")
    [void]$CandidateDirectories.Add("C:\msys64\ucrt64\bin")
    [void]$CandidateDirectories.Add("C:\msys64\mingw64\bin")

    foreach ($Directory in ($CandidateDirectories | Select-Object -Unique)) {
        if ([string]::IsNullOrWhiteSpace($Directory)) {
            continue
        }

        $CleanDirectory = $Directory.Trim().Trim('"')
        foreach ($DllName in @("libcairo-2.dll", "cairo-2.dll", "cairo.dll")) {
            if (Test-Path -LiteralPath (Join-Path $CleanDirectory $DllName) -PathType Leaf) {
                return (Resolve-Path -LiteralPath $CleanDirectory).Path
            }
        }
    }

    return $null
}

if ($Env:OS -eq "Windows_NT") {
    $CairoDllDirectory = Find-CairoDllDirectory
    if ($CairoDllDirectory) {
        $Env:CAIROCFFI_DLL_DIRECTORIES = $CairoDllDirectory
        if (($Env:PATH -split [System.IO.Path]::PathSeparator) -notcontains $CairoDllDirectory) {
            $Env:PATH = "$CairoDllDirectory$([System.IO.Path]::PathSeparator)$Env:PATH"
        }
    }
    elseif ($DryRun) {
        Write-Warning "Cairo DLL was not found. Install GTK/Cairo or set Runtime.cairo_dll_directory before training."
    }
    else {
        throw "Cairo DLL was not found. Full-page OMR requires libcairo-2.dll on Windows. Install GTK/Cairo or set Runtime.cairo_dll_directory."
    }
}

#endregion


#region Build Arguments

if ([string]::IsNullOrWhiteSpace($Config.experiment_name)) {
    throw "experiment_name must not be empty."
}
if (-not [string]::IsNullOrWhiteSpace($Config.from_checkpoint) -and
    -not [string]::IsNullOrWhiteSpace($Config.starting_weights)) {
    throw "from_checkpoint and starting_weights are mutually exclusive."
}

$CheckpointEveryNEpochs = 0
if (-not [int]::TryParse([string]$Config.checkpoint_every_n_epochs, [ref]$CheckpointEveryNEpochs) -or $CheckpointEveryNEpochs -lt 1) {
    throw "checkpoint_every_n_epochs must be a positive integer."
}
$Config.checkpoint_every_n_epochs = $CheckpointEveryNEpochs

$SupportedFinetuningModes = @("CL", "SR", "CL1")
if ($Config.finetuning -notin $SupportedFinetuningModes) {
    throw "Unsupported finetuning mode '$($Config.finetuning)'. Choose: $($SupportedFinetuningModes -join ', ')."
}
$SupportedEncoderTrainingModes = @("fine_tune", "linear_probe")
if ($Config.encoder_training_mode -notin $SupportedEncoderTrainingModes) {
    throw "Unsupported encoder training mode '$($Config.encoder_training_mode)'. Choose: $($SupportedEncoderTrainingModes -join ', ')."
}
$SupportedAttentionBackends = @("auto", "flash_attention_2", "sdpa", "eager")
if ($Config.attention_backend -notin $SupportedAttentionBackends) {
    throw "Unsupported attention backend '$($Config.attention_backend)'. Choose: $($SupportedAttentionBackends -join ', ')."
}

$ConfigPath = Resolve-InputFile -Path $Config.config_path -SettingName "config_path"
$TrainingConfigPath = $ConfigPath
if ($Env:OS -eq "Windows_NT" -and $null -ne $Runtime.windows_num_workers) {
    $TrainingConfigPath = New-WindowsRuntimeConfig -SourcePath $ConfigPath
}

$UvArgs = [System.Collections.ArrayList]::new()
[void]$UvArgs.Add("run")
[void]$UvArgs.Add("--frozen")
[void]$UvArgs.Add("musvit")
[void]$UvArgs.Add("full-page-omr")
[void]$UvArgs.Add("--config_path=$TrainingConfigPath")
[void]$UvArgs.Add("--experiment_name=$($Config.experiment_name)")
[void]$UvArgs.Add("--foundation_architecture=$($Config.foundation_architecture)")
[void]$UvArgs.Add("--foundation_weights=$($Config.foundation_weights)")
[void]$UvArgs.Add("--finetuning=$($Config.finetuning)")
[void]$UvArgs.Add("--encoder_training_mode=$($Config.encoder_training_mode)")
[void]$UvArgs.Add("--max_steps=$($Config.max_steps)")
[void]$UvArgs.Add("--checkpoint_every_n_epochs=$($Config.checkpoint_every_n_epochs)")
[void]$UvArgs.Add("--train=$($Features.train.ToString().ToLowerInvariant())")
[void]$UvArgs.Add("--attention_backend=$($Config.attention_backend)")

if ($null -ne $Config.resolution) {
    [void]$UvArgs.Add("--resolution=$($Config.resolution)")
}
if ($null -ne $Config.learning_rate) {
    [void]$UvArgs.Add("--learning_rate=$($Config.learning_rate)")
}
if (-not [string]::IsNullOrWhiteSpace($Config.from_checkpoint)) {
    $CheckpointPath = Resolve-InputFile -Path $Config.from_checkpoint -SettingName "from_checkpoint"
    [void]$UvArgs.Add("--from_checkpoint=$CheckpointPath")
}
if (-not [string]::IsNullOrWhiteSpace($Config.starting_weights)) {
    $StartingWeightsPath = Resolve-InputFile -Path $Config.starting_weights -SettingName "starting_weights"
    [void]$UvArgs.Add("--starting_weights=$StartingWeightsPath")
}

$DisplayArgs = $UvArgs | ForEach-Object {
    if ($_ -match "\s") {
        '"' + ($_ -replace '"', '\"') + '"'
    }
    else {
        $_
    }
}

#endregion

#region Execute Fine-tuning

Write-Output "Starting full-page OMR fine-tuning..."
Write-Output "GPU selection: CUDA_VISIBLE_DEVICES=$Env:CUDA_VISIBLE_DEVICES"
if ($TrainingConfigPath -ne $ConfigPath) {
    Write-Output "Windows runtime config: $TrainingConfigPath (num_workers=$($Runtime.windows_num_workers))"
}
Write-Output "Command: uv $($DisplayArgs -join ' ')"

if ($DryRun) {
    Write-Output "Dry run finished; training was not started."
    return
}

& uv @UvArgs
if ($LASTEXITCODE -ne 0) {
    throw "Full-page OMR fine-tuning failed with exit code $LASTEXITCODE."
}

Write-Output "Full-page OMR fine-tuning finished."

#endregion
