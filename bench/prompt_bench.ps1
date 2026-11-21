# One real task on every model: sends the same chat prompt through zero-server (each model with its own sampling
# defaults, reasoning off, a fixed seed), streams the answer and records time to first token, answer tokens,
# generation speed and draft acceptance. Saves the raw answer and the HTML page it contains.
# The PowerShell port of the former prompt_bench.py (2026-11-21, the user: no Python in the repository): the same
# requests, files and summary records. Windows PowerShell 5.1 or 7.
#   powershell -File bench\prompt_bench.ps1 [-Models 35b,g12,9b,4b,e4b,27b] [-Out bench\results\landing] [-Port 8090] [-MaxTokens 6144]
param([string]$Models = "35b,g12,9b,4b,e4b,27b", [string]$Out = "", [int]$Port = 8090, [int]$MaxTokens = 6144)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http
$prompt = "buat html landing page professional toko coklat enak di pondok gede dalam satu file yang ringkas, " +
  "interaksi dengan animasi, vanilla css, dan vanilla js dengan struktur mvvm"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Out) { $Out = Join-Path $root "bench\results\landing" }
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
New-Item -ItemType Directory -Force $Out | Out-Null
$base = "http://127.0.0.1:$Port"
$serverPs1 = Join-Path $root "scripts\zero-server.ps1"
$utf8 = New-Object Text.UTF8Encoding $false
# (zero-server through files, waiting for its own process only: it starts serverz in the background, which would
#  inherit a captured pipe and keep the read waiting for good; the former .py hung there)
function Run-Server([string[]]$a) {
  $fo = [IO.Path]::GetTempFileName(); $fe = [IO.Path]::GetTempFileName()
  try {
    $p = Start-Process -FilePath "powershell" -ArgumentList (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$serverPs1`"") + $a) `
      -RedirectStandardOutput $fo -RedirectStandardError $fe -WindowStyle Hidden -PassThru
    $null = $p.Handle  # (without this the exit code stays empty once the process has exited)
    $p.WaitForExit()
    # (read with sharing: in Windows PowerShell the serverz it started still holds these files open)
    $share = [IO.FileShare]([int][IO.FileShare]::ReadWrite -bor [int][IO.FileShare]::Delete)
    $read = { param($f) $s = New-Object IO.FileStream -ArgumentList $f, ([IO.FileMode]::Open), ([IO.FileAccess]::Read), $share
      try { (New-Object IO.StreamReader($s)).ReadToEnd() } finally { $s.Dispose() } }
    return @{ code = $p.ExitCode; out = ((& $read $fo) + (& $read $fe)) }
  } finally { Remove-Item $fo, $fe -ErrorAction SilentlyContinue }
}
function Extract-Html([string]$t) {
  $m = [regex]::Match($t, '```html\s*\n(.*?)```', 'Singleline, IgnoreCase')
  if ($m.Success) { return $m.Groups[1].Value.Trim() }
  $m = [regex]::Match($t, '(<!DOCTYPE html.*?</html>)', 'Singleline, IgnoreCase')
  if ($m.Success) { return $m.Groups[1].Value.Trim() }
  return $null
}
function Get-P($o, $n) { if ($null -eq $o) { return $null }; $p = $o.PSObject.Properties[$n]; if ($p) { $p.Value } else { $null } }
$http = New-Object System.Net.Http.HttpClient
$http.Timeout = [TimeSpan]::FromSeconds(7200)
foreach ($model in $Models.Split(",")) {
  $r = Run-Server @("start", "-Model", $model, "-Quiet", "-Port", "$Port")
  if ($r.code -ne 0) { "${model}: server failed to start`n$($r.out)"; continue }
  $alias = (Invoke-RestMethod "$base/v1/models").data[0].id
  Start-Sleep -Seconds 10  # a short idle period before the measurement
  $body = [ordered]@{ model = $alias; stream = $true; stream_options = @{ include_usage = $true }; max_tokens = $MaxTokens; seed = 1234
    reasoning_effort = "none"; messages = @(@{ role = "user"; content = $prompt }) } | ConvertTo-Json -Depth 6 -Compress
  $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Post, "$base/v1/chat/completions")
  $req.Content = New-Object System.Net.Http.StringContent($body, [Text.Encoding]::UTF8, "application/json")
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $tFirst = $null; $tAnswer = $null
  $reasoning = New-Object Text.StringBuilder; $content = New-Object Text.StringBuilder
  $timings = $null; $usage = $null; $finish = $null
  $resp = $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
  $rd = New-Object IO.StreamReader($resp.Content.ReadAsStreamAsync().Result, $utf8)
  while ($null -ne ($line = $rd.ReadLine())) {
    $line = $line.Trim()
    if (-not $line.StartsWith("data:") -or $line -eq "data: [DONE]") { continue }
    $ev = $line.Substring(5) | ConvertFrom-Json
    if (Get-P $ev "usage") { $usage = $ev.usage }
    if (Get-P $ev "timings") { $timings = $ev.timings }
    foreach ($ch in @(Get-P $ev "choices")) {
      if ($null -eq $ch) { continue }
      $d = Get-P $ch "delta"
      $rc = Get-P $d "reasoning_content"
      if ($rc) { [void]$reasoning.Append($rc); if ($null -eq $tFirst) { $tFirst = $sw.Elapsed.TotalSeconds } }
      $cc = Get-P $d "content"
      if ($cc) {
        [void]$content.Append($cc)
        if ($null -eq $tFirst) { $tFirst = $sw.Elapsed.TotalSeconds }
        if ($null -eq $tAnswer) { $tAnswer = $sw.Elapsed.TotalSeconds }
      }
      $fr = Get-P $ch "finish_reason"; if ($fr) { $finish = $fr }
    }
  }
  $rd.Dispose()
  $t1 = $sw.Elapsed.TotalSeconds
  $text = $content.ToString(); $think = $reasoning.ToString()
  $page = Extract-Html $text
  [IO.File]::WriteAllText((Join-Path $Out "$model.md"), $(if ($think) { "<!-- reasoning -->`n$think`n<!-- answer -->`n" } else { "" }) + $text, $utf8)
  if ($page) { [IO.File]::WriteAllText((Join-Path $Out "$model.html"), $page, $utf8) }
  $rec = [ordered]@{ model = $model; alias = $alias; time = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss"); finish = $finish
    prompt_tokens = Get-P $usage "prompt_tokens"; completion_tokens = Get-P $usage "completion_tokens"
    ttft_s = [math]::Round($(if ($null -ne $tFirst) { $tFirst } else { $t1 }), 2); answer_start_s = [math]::Round($(if ($null -ne $tAnswer) { $tAnswer } else { $t1 }), 2)
    total_s = [math]::Round($t1, 1); gen_tps = [math]::Round([double]$(if (Get-P $timings "predicted_per_second") { $timings.predicted_per_second } else { 0 }), 2)
    draft_n = Get-P $timings "draft_n"; draft_accepted = Get-P $timings "draft_n_accepted"
    reasoning_chars = $think.Length; answer_chars = $text.Length; html_chars = $(if ($page) { $page.Length } else { 0 }) }
  $j = $rec | ConvertTo-Json -Compress
  [IO.File]::AppendAllText((Join-Path $Out "summary.jsonl"), $j + "`n", $utf8)
  $j
}
[void](Run-Server @("stop", "-Port", "$Port"))
