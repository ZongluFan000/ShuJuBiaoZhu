$root = Split-Path -Parent $PSScriptRoot

$paths = @(
  "data\output",
  "data\logs",
  "data\checkpoint"
)

foreach ($rel in $paths) {
  $path = Join-Path $root $rel
  if (Test-Path $path) {
    Get-ChildItem -LiteralPath $path -Force | Remove-Item -Recurse -Force
  } else {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
  }
}

Write-Host "Cleaned output, logs, and checkpoint directories."
