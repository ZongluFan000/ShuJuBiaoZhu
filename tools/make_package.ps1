param(
  [string]$Output = "annotation_sandbox_package.zip"
)

$root = Split-Path -Parent $PSScriptRoot
$parent = Split-Path -Parent $root
$dest = Join-Path $parent $Output

if (Test-Path $dest) {
  Remove-Item -LiteralPath $dest -Force
}

Compress-Archive -Path $root -DestinationPath $dest -Force
Write-Host "Package created: $dest"
