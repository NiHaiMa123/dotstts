[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$SmokeOnly
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$composePath = Join-Path $repoRoot 'docker\slice12\compose.yaml'

docker info --format '{{json .ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Build) {
    docker compose -f $composePath build trainer
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

docker compose -f $composePath run --rm trainer `
    python scripts/build_slice12_container_manifests.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SmokeOnly) {
    docker compose -f $composePath run --rm trainer `
        accelerate launch --num_processes 1 --num_machines 1 `
        --dynamo_backend no --mixed_precision bf16 scripts/train_dots_tts.py `
        --config configs/lab/slice12/soar_lora_train_v1.yaml `
        --max-train-steps 1 `
        --skip-validation `
        --output-dir /workspace/data/work/slice12/soar_lora_smoke_metrics_v1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    docker compose -f $composePath run --rm trainer `
        accelerate launch --num_processes 1 --num_machines 1 `
        --dynamo_backend no --mixed_precision bf16 scripts/train_dots_tts.py `
        --config configs/lab/slice12/soar_lora_train_v1.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
