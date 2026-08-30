param(
    [string]$InputDataset = "datasets\processed\bea_judge_cleaned_3400.json",
    [string]$JudgeRunDir = "datasets\judge_outputs\m_prometheus_3b_real_full_promptfix_256",
    [string]$JudgeName = "m_prometheus_3b_real_full_promptfix_256",
    [string]$ModelPath = "models\M-Prometheus-3B",
    [int]$MaxNewTokens = 256,
    [int]$CheckpointInterval = 25,
    [string]$BaseScoresPath = "",
    [switch]$SkipScoring,
    [switch]$RunSwapProbe,
    [switch]$SwapProbeDryRun,
    [int]$SwapPerDatasetLimit = 20,
    [double]$SwapLowConfidenceThreshold = 0.70
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Python = Join-Path $Root "judge\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}

Write-Host "== BEA-Judge SCI pipeline =="
Write-Host "Input dataset: $InputDataset"
Write-Host "Judge run dir: $JudgeRunDir"
Write-Host "Max new tokens: $MaxNewTokens"

if (-not $SkipScoring) {
    & $Python "src\base_judge.py" `
        --input $InputDataset `
        --backend "m_prometheus" `
        --model-path $ModelPath `
        --pairwise-only `
        --max-new-tokens $MaxNewTokens `
        --judge-name $JudgeName `
        --run-dir $JudgeRunDir `
        --checkpoint-interval $CheckpointInterval `
        --resume
    if ($LASTEXITCODE -ne 0) {
        throw "Base judge scoring failed with exit code $LASTEXITCODE"
    }

    $SummaryPath = Join-Path $Root (Join-Path $JudgeRunDir "summary.json")
    if (-not (Test-Path $SummaryPath)) {
        throw "Missing base judge summary: $SummaryPath"
    }
    $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
    $Coverage = $Summary.coverage
    if ($null -eq $Coverage) {
        throw "Base judge summary has no coverage block."
    }
    if (
        [int]$Coverage.parsed_rows -ne [int]$Coverage.total_pairwise_samples -or
        [int]$Coverage.failed_rows -ne 0 -or
        [int]$Coverage.backend_error_rows -ne 0 -or
        [double]$Coverage.parse_success_rate -lt 0.95
    ) {
        throw "Coverage gate failed: parsed=$($Coverage.parsed_rows), total=$($Coverage.total_pairwise_samples), failed=$($Coverage.failed_rows), backend_errors=$($Coverage.backend_error_rows), rate=$($Coverage.parse_success_rate)"
    }
}

if ($BaseScoresPath) {
    $BaseScores = $BaseScoresPath
} else {
    $BaseScores = Join-Path $JudgeRunDir "base_scores.json"
}
& $Python "src\bea_judge_train.py" --input $InputDataset --judge-output $BaseScores
if ($LASTEXITCODE -ne 0) {
    throw "BEA-Judge training failed with exit code $LASTEXITCODE"
}

$TrainingConfigPath = Join-Path $Root "configs\experiment.json"
$TrainingConfig = Get-Content -Raw -LiteralPath $TrainingConfigPath | ConvertFrom-Json
$CalibratedResults = $TrainingConfig.latest_outputs.calibrated_results
if (-not $CalibratedResults) {
    throw "Training config did not record latest calibrated_results."
}

& $Python "scripts\bias_awareness_audit.py" --calibrated-results $CalibratedResults
if ($LASTEXITCODE -ne 0) {
    throw "Bias audit failed with exit code $LASTEXITCODE"
}

& $Python "scripts\evidence_fact_audit.py"
if ($LASTEXITCODE -ne 0) {
    throw "Evidence audit failed with exit code $LASTEXITCODE"
}

if ($RunSwapProbe) {
    $SwapArgs = @(
        "scripts\order_swap_probe.py",
        "--input", $InputDataset,
        "--base-scores", $BaseScores,
        "--calibrated-results", $CalibratedResults,
        "--per-dataset-limit", "$SwapPerDatasetLimit",
        "--low-confidence-threshold", "$SwapLowConfidenceThreshold",
        "--model-path", $ModelPath,
        "--max-new-tokens", "$MaxNewTokens"
    )
    if ($SwapProbeDryRun) {
        $SwapArgs += "--dry-run"
    }
    & $Python @SwapArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Order-swap probe failed with exit code $LASTEXITCODE"
    }
}

& $Python "scripts\bea_judge_ablation.py" --input $InputDataset --judge-output $BaseScores
if ($LASTEXITCODE -ne 0) {
    throw "Ablation failed with exit code $LASTEXITCODE"
}

& $Python "scripts\generate_sci_results_tables.py" --base-scores $BaseScores
if ($LASTEXITCODE -ne 0) {
    throw "SCI result table generation failed with exit code $LASTEXITCODE"
}

Write-Host "== BEA-Judge SCI pipeline completed =="
Write-Host "Base scores: $BaseScores"
Write-Host "Calibrated results: $CalibratedResults"
Write-Host "Bias report: datasets\bias_awareness_report.md"
Write-Host "Evidence report: datasets\evidence_fact_report.md"
if ($RunSwapProbe) {
    Write-Host "Order-swap probe: datasets\judge_outputs\order_swap_probe\swap_probe_report.json"
}
Write-Host "Ablation report: datasets\model_outputs\latest_ablation_report.md"
Write-Host "SCI results: datasets\model_outputs\sci_tables\sci_results_report.md"
