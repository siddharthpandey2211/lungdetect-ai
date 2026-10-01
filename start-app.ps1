$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if (Test-Path "$root\.venv-2\Scripts\python.exe") {
  "$root\.venv-2\Scripts\python.exe"
} elseif (Test-Path "$root\.venv\Scripts\python.exe") {
  "$root\.venv\Scripts\python.exe"
} else {
  throw "No Python virtual environment found. Expected .venv-2 or .venv."
}

foreach ($port in @(5050, 3000)) {
  $listeners = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
  foreach ($listener in $listeners) {
    Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$root'; & '$python' -m uvicorn --app-dir Backend app:app --host 127.0.0.1 --port 5050"
)

Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$root\frontend'; npm run dev -- --host 0.0.0.0 --port 3000"
)

Write-Host "Backend:  http://127.0.0.1:5050"
Write-Host "Frontend: http://localhost:3000"
