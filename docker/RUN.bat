@echo off
REM Check if Docker Desktop is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is not running. Please start Docker Desktop and try again.
    pause
    exit /b
)

REM Run docker compose with the specified options
docker compose -f docker-compose.yml --project-name gaussians build
docker compose -f docker-compose.yml --project-name gaussians run --name gaussians -d --remove-orphans --rm dev

echo ready
pause
