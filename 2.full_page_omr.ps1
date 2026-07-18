[CmdletBinding()]
param (
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Format-CanonicalDecimal {
    param (
        [Parameter(Mandatory = $true)]
        [double]$Value
    )

    return $Value.ToString("0.0################", [System.Globalization.CultureInfo]::InvariantCulture)
}

#region Configuration

$Config = @{
    config_path            = "experiments/full_page_omr/config/Polish_Scores/finetuning.json"
    experiment_name        = "polish_scores_cl"
    foundation_architecture = "ViTMAEBase"
    foundation_weights     = "carlospm12/LSMT-MAE-Base-1024-16"
    finetuning             = "CL"     # Supported: CL, SR, CL1
    encoder_training_mode  = "fine_tune" # Supported: fine_tune, linear_probe
    resolution             = $null    # $null uses the experiment default (1024)
    max_steps                  = 4000000 # Absolute Trainer endpoint for the AdamW/WSD protocol
    validation_every_n_epochs  = 2000    # Validate on the declared epoch cadence
    checkpoint_every_n_epochs = 100   # Periodic full checkpoint interval
    task_learning_rate        = 0.0001
    encoder_learning_rate     = 0.00001
    weight_decay              = 0.01
    wsd_warmup_steps          = 10000
    wsd_decay_steps           = 400000
    wsd_warmup_type           = "linear"
    wsd_decay_type            = "cosine"
    wsd_min_lr_ratio          = 0.0
    attention_backend      = "auto"   # auto: FA2 -> SDPA -> eager fallback
    protocol_version       = "full_page_omr_adamw_wsd_4m_v1"
    source_curriculum_step = $null    # Required with starting_weights
    source_checkpoint_sha256 = $null  # Required with starting_weights
    from_checkpoint        = $null    # Start a fresh Trainer run
    starting_weights       = $null    # Initialize model weights from a .ckpt file
}

$Features = @{
    train = $true
}

$Runtime = @{
    cuda_visible_devices = "0"       # Use $null to leave CUDA device selection unchanged
    cairo_dll_directory  = $null       # Prefer CAIROCFFI_DLL_DIRECTORIES or a Cairo DLL on PATH
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
if ([string]::IsNullOrWhiteSpace($Config.protocol_version)) {
    throw "protocol_version must not be empty."
}
if (-not [string]::IsNullOrWhiteSpace($Config.from_checkpoint) -and
    -not [string]::IsNullOrWhiteSpace($Config.starting_weights)) {
    throw "from_checkpoint and starting_weights are mutually exclusive."
}
if (-not [string]::IsNullOrWhiteSpace($Config.starting_weights)) {
    $SourceCurriculumStep = 0
    if ($null -eq $Config.source_curriculum_step -or
        -not [int]::TryParse([string]$Config.source_curriculum_step, [ref]$SourceCurriculumStep) -or
        $SourceCurriculumStep -lt 0) {
        throw "source_curriculum_step must be a non-negative integer with starting_weights."
    }
    $Config.source_curriculum_step = $SourceCurriculumStep
    if ([string]::IsNullOrWhiteSpace($Config.source_checkpoint_sha256) -or
        [string]$Config.source_checkpoint_sha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw "source_checkpoint_sha256 must be a lowercase SHA-256 digest with starting_weights."
    }
}
elseif ($null -ne $Config.source_curriculum_step -or
        $null -ne $Config.source_checkpoint_sha256) {
    throw "source_curriculum_step and source_checkpoint_sha256 require starting_weights."
}

$CheckpointEveryNEpochs = 0
if (-not [int]::TryParse([string]$Config.checkpoint_every_n_epochs, [ref]$CheckpointEveryNEpochs) -or $CheckpointEveryNEpochs -lt 1) {
    throw "checkpoint_every_n_epochs must be a positive integer."
}
$Config.checkpoint_every_n_epochs = $CheckpointEveryNEpochs

$ValidationEveryNEpochs = 0
if (-not [int]::TryParse([string]$Config.validation_every_n_epochs, [ref]$ValidationEveryNEpochs) -or $ValidationEveryNEpochs -lt 1) {
    throw "validation_every_n_epochs must be a positive integer."
}
$Config.validation_every_n_epochs = $ValidationEveryNEpochs

$MaxSteps = 0
if (-not [int]::TryParse([string]$Config.max_steps, [ref]$MaxSteps) -or $MaxSteps -lt 1) {
    throw "max_steps must be a positive integer for production training."
}
$Config.max_steps = $MaxSteps

foreach ($LearningRateName in @("task_learning_rate", "encoder_learning_rate")) {
    $LearningRate = 0.0
    if (-not [double]::TryParse([string]$Config[$LearningRateName], [ref]$LearningRate) -or
        [double]::IsNaN($LearningRate) -or [double]::IsInfinity($LearningRate) -or $LearningRate -le 0) {
        throw "$LearningRateName must be a positive finite number."
    }
    $Config[$LearningRateName] = $LearningRate
}

foreach ($NonNegativeNumberName in @("weight_decay", "wsd_min_lr_ratio")) {
    $NonNegativeNumber = 0.0
    if (-not [double]::TryParse([string]$Config[$NonNegativeNumberName], [ref]$NonNegativeNumber) -or
        [double]::IsNaN($NonNegativeNumber) -or [double]::IsInfinity($NonNegativeNumber) -or $NonNegativeNumber -lt 0) {
        throw "$NonNegativeNumberName must be a non-negative finite number."
    }
    $Config[$NonNegativeNumberName] = $NonNegativeNumber
}

$WsdWarmupSteps = 0
if (-not [int]::TryParse([string]$Config.wsd_warmup_steps, [ref]$WsdWarmupSteps) -or $WsdWarmupSteps -lt 0) {
    throw "wsd_warmup_steps must be a non-negative integer."
}
$Config.wsd_warmup_steps = $WsdWarmupSteps

$WsdDecaySteps = 0
if (-not [int]::TryParse([string]$Config.wsd_decay_steps, [ref]$WsdDecaySteps) -or $WsdDecaySteps -lt 1) {
    throw "wsd_decay_steps must be a positive integer."
}
$Config.wsd_decay_steps = $WsdDecaySteps

$SupportedWsdTypes = @("linear", "cosine", "1-sqrt")
if ($Config.wsd_warmup_type -notin $SupportedWsdTypes) {
    throw "Unsupported wsd_warmup_type '$($Config.wsd_warmup_type)'. Choose: $($SupportedWsdTypes -join ', ')."
}
if ($Config.wsd_decay_type -notin $SupportedWsdTypes) {
    throw "Unsupported wsd_decay_type '$($Config.wsd_decay_type)'. Choose: $($SupportedWsdTypes -join ', ')."
}
if (([long]$Config.wsd_warmup_steps + [long]$Config.wsd_decay_steps) -ge [long]$Config.max_steps) {
    throw "wsd_warmup_steps + wsd_decay_steps must be less than max_steps."
}

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
[void]$UvArgs.Add("--validation_every_n_epochs=$($Config.validation_every_n_epochs)")
[void]$UvArgs.Add("--checkpoint_every_n_epochs=$($Config.checkpoint_every_n_epochs)")
[void]$UvArgs.Add("--task_learning_rate=$(Format-CanonicalDecimal $Config.task_learning_rate)")
[void]$UvArgs.Add("--encoder_learning_rate=$(Format-CanonicalDecimal $Config.encoder_learning_rate)")
[void]$UvArgs.Add("--weight_decay=$(Format-CanonicalDecimal $Config.weight_decay)")
[void]$UvArgs.Add("--wsd_warmup_steps=$($Config.wsd_warmup_steps)")
[void]$UvArgs.Add("--wsd_decay_steps=$($Config.wsd_decay_steps)")
[void]$UvArgs.Add("--wsd_warmup_type=$($Config.wsd_warmup_type)")
[void]$UvArgs.Add("--wsd_decay_type=$($Config.wsd_decay_type)")
[void]$UvArgs.Add("--wsd_min_lr_ratio=$(Format-CanonicalDecimal $Config.wsd_min_lr_ratio)")
[void]$UvArgs.Add("--train=$($Features.train.ToString().ToLowerInvariant())")
[void]$UvArgs.Add("--attention_backend=$($Config.attention_backend)")
[void]$UvArgs.Add("--protocol_version=$($Config.protocol_version)")

if ($null -ne $Config.resolution) {
    [void]$UvArgs.Add("--resolution=$($Config.resolution)")
}
if (-not [string]::IsNullOrWhiteSpace($Config.from_checkpoint)) {
    $CheckpointPath = Resolve-InputFile -Path $Config.from_checkpoint -SettingName "from_checkpoint"
    [void]$UvArgs.Add("--from_checkpoint=$CheckpointPath")
}
if (-not [string]::IsNullOrWhiteSpace($Config.starting_weights)) {
    $StartingWeightsPath = Resolve-InputFile -Path $Config.starting_weights -SettingName "starting_weights"
    [void]$UvArgs.Add("--starting_weights=$StartingWeightsPath")
    [void]$UvArgs.Add("--source_curriculum_step=$($Config.source_curriculum_step)")
    [void]$UvArgs.Add("--source_checkpoint_sha256=$($Config.source_checkpoint_sha256)")
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
