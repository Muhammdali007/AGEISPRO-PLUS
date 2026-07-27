[CmdletBinding()]
param(
    [switch]$Detached,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Write-Info {
    param([string]$Message)
    Write-Host "[AegisPro] $Message"
}

function Invoke-DockerPullWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Image,
        [int]$Attempts = 3
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        Write-Info "Checking Docker image $Image (attempt $attempt of $Attempts)..."
        docker image inspect $Image *> $null
        if ($LASTEXITCODE -eq 0) {
            return
        }

        docker pull $Image
        if ($LASTEXITCODE -eq 0) {
            return
        }

        if ($attempt -lt $Attempts) {
            Start-Sleep -Seconds (10 * $attempt)
        }
    }

    Write-Error "Docker could not download $Image. Check your internet connection or Docker Desktop, then run .\start-local.cmd again."
}

function Backup-PostgresIfRunning {
    $containerName = "aegispro-postgres"
    $running = docker ps --filter "name=^/$containerName$" --filter "status=running" --format "{{.Names}}"
    if ($LASTEXITCODE -ne 0 -or $running -ne $containerName) {
        Write-Info "No running PostgreSQL container found; skipping pre-start database backup."
        return
    }

    $backupDir = Join-Path $root "storage\backups"
    New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $containerDumpPath = "/tmp/aegispro-pre-start-$timestamp.dump"
    $hostDumpPath = Join-Path $backupDir "aegispro-pre-start-$timestamp.dump"

    Write-Info "Creating PostgreSQL backup before container recreate..."
    docker exec $containerName pg_dump -U aegispro -d aegispro -Fc -f $containerDumpPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pre-start PostgreSQL backup failed; leaving the stack unchanged."
    }

    docker cp "$containerName`:$containerDumpPath" $hostDumpPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not copy PostgreSQL backup to $hostDumpPath; leaving the stack unchanged."
    }

    docker exec $containerName rm -f $containerDumpPath *> $null
    Write-Info "Backup saved to $hostDumpPath"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker was not found. Install and start Docker Desktop, then run this command again."
}

try {
    docker compose version | Out-Null
}
catch {
    Write-Error "Docker Compose is not available. Install/update Docker Desktop, then run this command again."
}

if (-not (Test-Path ".env.production")) {
    Write-Error ".env.production was not found. Copy .env.production.example to .env.production first."
}

$localDefaults = @{
    API_CONTINUOUS_DETECTION_BATCH_SIZE = "4"
    API_CONTINUOUS_DETECTION_MAX_PENDING_PER_CAMERA = "1"
    API_CONTINUOUS_DETECTION_SCHEDULER_INTERVAL_MS = "250"
    API_CONTINUOUS_DETECTION_RECOGNITION_INTERVAL_SECONDS = "4.0"
    AI_MODEL_BATCH_SIZE = "8"
    AI_MODEL_PRELOAD_ON_STARTUP = "true"
    AI_MODEL_RUNTIME_AUTOINSTALL = "false"
    AI_RUNTIME_GATE_REPORT_PATH = "/app/storage/runtime/runtime-gates.json"
}

foreach ($name in $localDefaults.Keys) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        [Environment]::SetEnvironmentVariable($name, $localDefaults[$name], "Process")
    }
}

$composeArgs = @(
    "compose",
    "--env-file", ".env.production",
    "-f", "docker-compose.yml",
    "-f", "docker-compose.local.yml"
)

if ($Stop) {
    Write-Info "Stopping local stack..."
    docker @composeArgs down
    exit $LASTEXITCODE
}

Write-Info "Starting Postgres, Redis, API, AI service, frontend, and Nginx..."
Write-Info "Open http://localhost:8080 when the containers finish starting."
Write-Info "Default login: admin@aegispro.local / AegisProAdmin!9vT4xQ2L"

Invoke-DockerPullWithRetry -Image "python:3.11-slim"
Backup-PostgresIfRunning

$upArgs = $composeArgs + @("up", "--build", "--force-recreate")
if ($Detached) {
    $upArgs += "-d"
}

docker @upArgs
