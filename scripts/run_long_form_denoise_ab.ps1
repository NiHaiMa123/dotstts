param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
if (-not $docker) {
    $docker = @(
        "C:\Program Files\Docker\Docker\resources\bin\docker.exe",
        "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
$compose = Join-Path $PSScriptRoot "..\docker\long_form_denoise\compose.yaml"

if (-not $docker) {
    throw "Docker CLI not found on PATH or standard install locations"
}
if ($Build) {
    & $docker compose -f $compose build enhancer
}
& $docker compose -f $compose run --rm enhancer `
    python scripts/build_long_form_denoise_ab.py

