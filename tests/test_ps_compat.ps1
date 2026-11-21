# Every PowerShell script of the repository works in Windows PowerShell 5.1 and PowerShell 7 (the user, 2026-11-21:
# "make all tools compatible with ps5.1 and ps7"):
#   1. each tracked .ps1 parses in both (powershell.exe and pwsh, each that is installed);
#   2. no Get-Content without -Encoding: 5.1 reads a UTF-8 file without a BOM as the ANSI code page, 7 as UTF-8, so a
#      prompt, a path or a JSON value with a non-ASCII character came out different in the two.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tests\test_ps_compat.ps1
# Prints "test_ps_compat: all passed" or each problem, and exits 1 then.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$files = @(git ls-files "*.ps1" 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if (-not $files.Count) { $files = @(Get-ChildItem -Recurse -Filter *.ps1 | ForEach-Object { $_.FullName.Substring($root.Length + 1) }) }
$bad = @()
# (1) in each host: one process per host checks every file (about a second)
#     (the check as a script file of its own: a -Command string lost its quotes on the way to the other host)
$checkPs = Join-Path ([IO.Path]::GetTempPath()) ("inferzero-pscheck-" + [Guid]::NewGuid().ToString("N").Substring(0, 8) + ".ps1")
[IO.File]::WriteAllText($checkPs, 'foreach ($f in $args) { $err = $null; [void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $f).Path, [ref]$null, [ref]$err); foreach ($x in $err) { "$f line $($x.Extent.StartLineNumber): $($x.Message)" } }')
$hosts = @(foreach ($h in "powershell", "pwsh") { if (Get-Command $h -ErrorAction SilentlyContinue) { $h } })
try {
  foreach ($h in $hosts) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { $out = @(& $h -NoProfile -ExecutionPolicy Bypass -File $checkPs @files 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ }) }
    finally { $ErrorActionPreference = $old }
    foreach ($o in $out) { $bad += "[$h] $o" }
  }
} finally { Remove-Item $checkPs -ErrorAction SilentlyContinue }
# (2) Get-Content without -Encoding (comment lines left out)
foreach ($f in $files) {
  $n = 0
  foreach ($l in [IO.File]::ReadAllLines((Join-Path $root $f))) {
    $n++
    if ($l -match 'Get-Content' -and $l -notmatch '-Encoding' -and $l -notmatch '^\s*#') { $bad += "$f line ${n}: Get-Content without -Encoding UTF8" }
  }
}
Write-Host ("{0} scripts, hosts: {1}" -f $files.Count, ($hosts -join ", "))
if ($bad.Count) { $bad | ForEach-Object { Write-Host "  $_" }; Write-Host "test_ps_compat: $($bad.Count) problem(s)"; exit 1 }
Write-Host "test_ps_compat: all passed"
