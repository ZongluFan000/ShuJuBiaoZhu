$HostName = "127.0.0.1"
$Port = 8765
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Send-TextResponse {
  param (
    [Parameter(Mandatory = $true)] $Context,
    [Parameter(Mandatory = $true)] [string] $Text,
    [string] $ContentType = "text/plain; charset=utf-8",
    [int] $StatusCode = 200
  )

  $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
  $Context.Response.StatusCode = $StatusCode
  $Context.Response.ContentType = $ContentType
  $Context.Response.ContentLength64 = $bytes.Length
  $Context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
  $Context.Response.OutputStream.Close()
}

function Send-JsonResponse {
  param (
    [Parameter(Mandatory = $true)] $Context,
    [Parameter(Mandatory = $true)] $Data,
    [int] $StatusCode = 200
  )

  $json = $Data | ConvertTo-Json -Depth 40 -Compress
  Send-TextResponse -Context $Context -Text $json -ContentType "application/json; charset=utf-8" -StatusCode $StatusCode
}

function Read-RequestJson {
  param ([Parameter(Mandatory = $true)] $Request)

  $reader = New-Object System.IO.StreamReader($Request.InputStream, [System.Text.Encoding]::UTF8)
  $text = $reader.ReadToEnd()
  if ([string]::IsNullOrWhiteSpace($text)) {
    return @{}
  }
  return $text | ConvertFrom-Json
}

function Get-AssistantText {
  param ($Data)

  if ($Data -is [array]) {
    foreach ($item in $Data) {
      $text = Get-AssistantText -Data $item
      if (-not [string]::IsNullOrWhiteSpace($text)) {
        return $text
      }
    }
    return ""
  }

  if ($Data.choices -and $Data.choices.Count -gt 0) {
    foreach ($choice in @($Data.choices)) {
      foreach ($value in @($choice.message.content, $choice.delta.content, $choice.text, $choice.content)) {
        $text = Convert-ContentToText -Value $value
        if (-not [string]::IsNullOrWhiteSpace($text)) {
          return $text
        }
      }
    }
  }

  if ($Data.message -and $Data.message.content) {
    $text = Convert-ContentToText -Value $Data.message.content
    if (-not [string]::IsNullOrWhiteSpace($text)) {
      return $text
    }
  }

  foreach ($nestedKey in @("data", "reply", "message", "result")) {
    if ($Data.PSObject.Properties.Name -contains $nestedKey) {
      $nested = $Data.$nestedKey
      if ($nested -and -not ($nested -is [string])) {
        $text = Get-AssistantText -Data $nested
        if (-not [string]::IsNullOrWhiteSpace($text)) {
          return $text
        }
      }
    }
  }

  if ($Data.candidates -and $Data.candidates.Count -gt 0) {
    foreach ($candidate in @($Data.candidates)) {
      if ($candidate.content -and $candidate.content.parts) {
        $text = Convert-ContentToText -Value $candidate.content.parts
        if (-not [string]::IsNullOrWhiteSpace($text)) {
          return $text
        }
      }
    }
  }

  foreach ($collectionName in @("outputs", "output", "generations")) {
    if ($Data.PSObject.Properties.Name -contains $collectionName) {
      foreach ($item in @($Data.$collectionName)) {
        $text = Convert-ContentToText -Value $item
        if (-not [string]::IsNullOrWhiteSpace($text)) {
          return $text
        }
      }
    }
  }

  foreach ($key in @("response", "text", "content", "completion", "generated_text", "answer", "result", "output_text")) {
    if ($Data.PSObject.Properties.Name -contains $key) {
      $text = Convert-ContentToText -Value $Data.$key
      if (-not [string]::IsNullOrWhiteSpace($text)) {
        return $text
      }
    }
  }

  return ""
}

function Convert-ContentToText {
  param ($Value)

  if ($null -eq $Value) {
    return ""
  }

  if ($Value -is [string]) {
    return $Value
  }

  if ($Value -is [array]) {
    $parts = @()
    foreach ($item in $Value) {
      $text = Convert-ContentToText -Value $item
      if (-not [string]::IsNullOrWhiteSpace($text)) {
        $parts += $text
      }
    }
    return ($parts -join "`n")
  }

  if ($Value.PSObject.Properties.Name -contains "text") {
    return [string]$Value.text
  }

  if ($Value.PSObject.Properties.Name -contains "content") {
    return Convert-ContentToText -Value $Value.content
  }

  if ($Value.PSObject.Properties.Name -contains "generated_text") {
    return [string]$Value.generated_text
  }

  if ($Value.PSObject.Properties.Name -contains "message" -and $Value.message.content) {
    return Convert-ContentToText -Value $Value.message.content
  }

  return [string]$Value
}

function Invoke-UpstreamChat {
  param ($Payload)

  $messages = @()
  if (-not [string]::IsNullOrWhiteSpace([string]$Payload.systemPrompt)) {
    $messages += [pscustomobject]@{
      role = "system"
      content = [string]$Payload.systemPrompt
    }
  }

  foreach ($message in @($Payload.messages)) {
    $messages += $message
  }

  if ([string]$Payload.apiUrl -like "*/api/chat*") {
    $body = [pscustomobject]@{
      model = [string]$Payload.model
      messages = $messages
      stream = $false
      options = [pscustomobject]@{
        temperature = [double]$Payload.temperature
        num_predict = [int]$Payload.maxTokens
      }
    }
  } elseif ([string]$Payload.apiUrl -like "*/api/generate*") {
    $body = [pscustomobject]@{
      model = [string]$Payload.model
      prompt = Convert-MessagesToPrompt -Messages $messages
      stream = $false
      options = [pscustomobject]@{
        temperature = [double]$Payload.temperature
        num_predict = [int]$Payload.maxTokens
      }
    }
  } elseif ([string]$Payload.apiUrl -like "*/completion") {
    $body = [pscustomobject]@{
      prompt = Convert-MessagesToPrompt -Messages $messages
      temperature = [double]$Payload.temperature
      n_predict = [int]$Payload.maxTokens
      stream = $false
    }
  } else {
    $body = [pscustomobject]@{
      model = [string]$Payload.model
      messages = $messages
      temperature = [double]$Payload.temperature
      max_tokens = [int]$Payload.maxTokens
      stream = $false
    }
  }

  $headers = @{
    "Content-Type" = "application/json"
  }

  if (-not [string]::IsNullOrWhiteSpace([string]$Payload.apiKey)) {
    $headers["Authorization"] = "Bearer $($Payload.apiKey)"
  }

  $jsonBody = $body | ConvertTo-Json -Depth 40 -Compress
  $response = Invoke-WebRequest -Uri ([string]$Payload.apiUrl) -Method Post -Headers $headers -Body $jsonBody -UseBasicParsing -TimeoutSec 120
  return Convert-ResponseTextToJson -Text $response.Content
}

function Convert-ResponseTextToJson {
  param ([string] $Text)

  try {
    return $Text | ConvertFrom-Json
  } catch {
  }

  $items = @()
  foreach ($line in $Text -split "`r?`n") {
    $current = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($current)) {
      continue
    }
    if ($current.StartsWith("data:")) {
      $current = $current.Substring(5).Trim()
    }
    if ($current -eq "[DONE]") {
      continue
    }
    try {
      $items += ($current | ConvertFrom-Json)
    } catch {
    }
  }

  if ($items.Count -eq 0) {
    return [pscustomobject]@{ text = $Text }
  }

  $pieces = @()
  foreach ($item in $items) {
    $piece = Get-AssistantText -Data $item
    if (-not [string]::IsNullOrWhiteSpace($piece)) {
      $pieces += $piece
    }
  }

  if ($pieces.Count -gt 0) {
    return [pscustomobject]@{
      content = ($pieces -join "")
      chunks = $items
    }
  }

  return $items[-1]
}

function Convert-MessagesToPrompt {
  param ($Messages)

  $lines = @()
  foreach ($message in @($Messages)) {
    $content = Convert-ContentToText -Value $message.content
    if ($message.role -eq "system") {
      $lines += "System: $content"
    } elseif ($message.role -eq "assistant") {
      $lines += "Assistant: $content"
    } else {
      $lines += "User: $content"
    }
  }
  $lines += "Assistant:"
  return ($lines -join "`n")
}

$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://${HostName}:${Port}/")

try {
  $listener.Start()
} catch {
  Write-Host "Failed to start local proxy on http://${HostName}:${Port}/"
  Write-Host $_.Exception.Message
  Write-Host "Try running this script as Administrator, or change the port."
  pause
  exit 1
}

Write-Host "Local proxy started: http://${HostName}:${Port}/"
Write-Host "Close this window to stop it."
Start-Process "http://${HostName}:${Port}/"

while ($listener.IsListening) {
  $context = $listener.GetContext()
  $path = $context.Request.Url.AbsolutePath

  try {
    if ($context.Request.HttpMethod -eq "GET" -and ($path -eq "/" -or $path -eq "/index.html")) {
      $htmlPath = Join-Path $Root "proxy-chat.html"
      $html = Get-Content -LiteralPath $htmlPath -Raw -Encoding UTF8
      Send-TextResponse -Context $context -Text $html -ContentType "text/html; charset=utf-8"
      continue
    }

    if ($context.Request.HttpMethod -eq "POST" -and $path -eq "/api/chat") {
      $payload = Read-RequestJson -Request $context.Request
      if ([string]::IsNullOrWhiteSpace([string]$payload.apiUrl)) {
        Send-JsonResponse -Context $context -Data @{ error = "Missing model API URL" } -StatusCode 400
        continue
      }

      $data = Invoke-UpstreamChat -Payload $payload
      $content = Get-AssistantText -Data $data
      Send-JsonResponse -Context $context -Data @{ content = $content; raw = $data }
      continue
    }

    Send-JsonResponse -Context $context -Data @{ error = "Not found" } -StatusCode 404
  } catch {
    Send-JsonResponse -Context $context -Data @{ error = $_.Exception.Message } -StatusCode 500
  }
}
