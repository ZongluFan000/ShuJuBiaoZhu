$root = Split-Path -Parent $PSScriptRoot
$data = Join-Path $root "data"

if (-not (Test-Path $data)) {
  Write-Host "No data directory found."
  exit 0
}

$targets = Get-ChildItem -LiteralPath $data -Force | Where-Object {
  $_.Name -like "output*" -or $_.Name -like "logs*" -or $_.Name -like "checkpoint*"
}

foreach ($target in $targets) {
  $full = (Resolve-Path -LiteralPath $target.FullName).Path
  if ($full.StartsWith((Resolve-Path -LiteralPath $data).Path)) {
    Remove-Item -LiteralPath $full -Recurse -Force
    Write-Host "Removed $full"
  }
}

Get-ChildItem -LiteralPath (Join-Path $data "input") -Force -Filter "rules_representative_*.xlsx" -ErrorAction SilentlyContinue |
  ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
    Write-Host "Removed $($_.FullName)"
  }

Write-Host "Clean complete."
