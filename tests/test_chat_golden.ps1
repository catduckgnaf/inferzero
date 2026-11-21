# Checks bin\chat_render.exe against committed golden renderings of the chat templates, byte for byte:
#   render.cases  the native Qwen rendering (src/chat.h), each case with q38 off and on   (14 renderings)
#   jinja.cases   minja (src/chat_jinja.h) on the Gemma 4 templates in tests\data         (9 cases x 3 templates)
# A case is one line: its name, a tab, the request as chat_render reads it ({messages, tools?, enable_thinking?,
# reasoning_effort?}); the golden of each rendering is tests\chat_golden\<suite>\<template>.<case>.txt.
#
# Where the goldens come from (2026-11-21, the user: no Python in the repository): until then, tests\test_chat_render.py
# and tests\test_chat_jinja.py rendered the same cases with Python's Jinja2 and compared. Both passed on 2026-11-10
# (every case identical). Since then nothing they render has changed: the templates in tests\data date from October,
# src\json.h and the minja sources are unchanged, and the two later changes to src\chat.h / tools\chat_render.cpp are a
# conversation that opens with a tool message (no case does) and an error message moved to the log. So chat_render's
# output on these cases, written here with -Update, is the output Jinja2 gave. A change to the renderer or a template
# that changes a golden needs its new rendering checked by reading it against the template, then -Update.
#
#   powershell -File tests\test_chat_golden.ps1 [-Exe bin\chat_render.exe] [-Data tests\data] [-Update]
# Prints "ALL IDENTICAL" when every rendering matches. Windows PowerShell 5.1 or 7.
param([string]$Exe = "", [string]$Data = "", [switch]$Update)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Exe) { $Exe = Join-Path $root "bin\chat_render.exe" }
if (-not $Data) { $Data = Join-Path $root "tests\data" }
$gold = Join-Path $PSScriptRoot "chat_golden"
$utf8 = New-Object Text.UTF8Encoding $false

# chat_render's stdout as bytes (no console decoding)
function Invoke-Render([string]$caseJson, [string[]]$extra) {
  $tmp = [IO.Path]::GetTempFileName()
  try {
    [IO.File]::WriteAllText($tmp, $caseJson, $utf8)
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.Arguments = (@("`"$tmp`"") + ($extra | ForEach-Object { "`"$_`"" })) -join " "
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    $p = [Diagnostics.Process]::Start($psi)
    $errTask = $p.StandardError.ReadToEndAsync()
    $ms = New-Object IO.MemoryStream
    $p.StandardOutput.BaseStream.CopyTo($ms)
    $p.WaitForExit()
    $err = $errTask.Result
    return @{ code = $p.ExitCode; bytes = $ms.ToArray(); err = $err }
  } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
}
function Read-Cases([string]$path) {
  $list = @()
  foreach ($line in [IO.File]::ReadAllLines($path, $utf8)) {
    if (-not $line.Trim()) { continue }
    $tab = $line.IndexOf("`t")
    $list += , @($line.Substring(0, $tab), $line.Substring($tab + 1))
  }
  return $list
}
function First-Diff([byte[]]$a, [byte[]]$b) {
  $n = [Math]::Min($a.Length, $b.Length)
  for ($i = 0; $i -lt $n; $i++) { if ($a[$i] -ne $b[$i]) { return $i } }
  return $n
}
function Show-Around([byte[]]$x, [int]$i) {
  $s = [Math]::Max(0, $i - 60); $len = [Math]::Min($x.Length - $s, 140)
  if ($len -le 0) { return "" }
  return ($utf8.GetString($x, $s, $len)) -replace "`r", "\r" -replace "`n", "\n"
}

if (-not (Test-Path $Exe)) { "test_chat_golden: $Exe not found (build it: .\build.ps1 tools\chat_render.cpp)"; exit 2 }
$runs = @()
foreach ($c in (Read-Cases (Join-Path $gold "render.cases"))) {
  foreach ($style in "q35", "q38") {
    $flag = if ($style -eq "q38") { "true" } else { "false" }
    $j = $c[1].Trim()
    $j = "{`"q38`": $flag, " + $j.Substring(1)
    $runs += , @("render", $style, $c[0], $j, @())
  }
}
foreach ($t in "gemma4_template", "gemma4_e4b_template", "gemma4_google_template") {
  $tp = Join-Path $Data "$t.jinja"
  if (-not (Test-Path $tp)) { "SKIP jinja / ${t}: $tp missing"; continue }
  foreach ($c in (Read-Cases (Join-Path $gold "jinja.cases"))) { $runs += , @("jinja", $t, $c[0], $c[1].Trim(), @("--jinja", $tp)) }
}

$ok = $true; $n = 0
foreach ($r in $runs) {
  $suite, $tmpl, $name, $json, $extra = $r
  $file = Join-Path $gold "$suite\$tmpl.$name.txt"
  $o = Invoke-Render $json $extra
  if ($o.code -ne 0) { "FAIL $suite / $tmpl / ${name}: chat_render exit $($o.code) $($o.err.Trim())"; $ok = $false; continue }
  $n++
  if ($Update) {
    New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
    [IO.File]::WriteAllBytes($file, $o.bytes)
    "wrote $suite\$tmpl.$name.txt ($($o.bytes.Length) bytes)"
    continue
  }
  if (-not (Test-Path $file)) { "FAIL $suite / $tmpl / ${name}: no golden (run with -Update after checking it)"; $ok = $false; continue }
  $want = [IO.File]::ReadAllBytes($file)
  $i = First-Diff $o.bytes $want
  if ($i -eq $want.Length -and $i -eq $o.bytes.Length) { "OK   $suite / $tmpl / $name"; continue }
  $ok = $false
  "DIFF $suite / $tmpl / $name, first difference at byte $i"
  "   golden: " + (Show-Around $want $i)
  "   got   : " + (Show-Around $o.bytes $i)
}
if ($Update) { "$n goldens written"; exit 0 }
if ($ok) { "ALL IDENTICAL ($n renderings)" } else { "DIFFERENCES FOUND"; exit 1 }
