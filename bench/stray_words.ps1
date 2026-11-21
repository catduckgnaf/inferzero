# stray_words: counts the words from other languages in Japanese answers (2026-11-21, for issue #4's stray words: a
# Japanese answer with a Korean, Hebrew or Swedish word in it), so runs compare by number, not by eye.
#   .\bench\stray_words.ps1 -Path answers\*.txt          one answer per file (UTF-8)
#   .\bench\stray_words.ps1 -Path a.txt,b.txt -Show       and each stray piece with the text around it
# Counted per answer and per 1,000 characters:
#   script  characters of scripts no Japanese text uses: Hangul, Hebrew, Cyrillic, Greek, Arabic, Thai, Devanagari;
#   latin   words of 4+ Latin letters inside a Japanese sentence (one with kana), such as "Swedish" in the middle of
#           a Japanese line (code, names and English answers are not Japanese sentences and are not counted).
# Not counted: Chinese written with characters Japanese also uses (the two share them; telling them apart needs a
# model, not a character table).
param(
  [Parameter(Mandatory = $true)][string[]]$Path,
  [switch]$Show
)
$ErrorActionPreference = "Stop"
$ranges = @(
  @{ n = "Hangul"; r = @(@(0xAC00, 0xD7A3), @(0x1100, 0x11FF), @(0x3130, 0x318F)) },
  @{ n = "Hebrew"; r = @(, @(0x0590, 0x05FF)) },
  @{ n = "Cyrillic"; r = @(, @(0x0400, 0x04FF)) },
  @{ n = "Greek"; r = @(, @(0x0370, 0x03FF)) },
  @{ n = "Arabic"; r = @(, @(0x0600, 0x06FF)) },
  @{ n = "Thai"; r = @(, @(0x0E00, 0x0E7F)) },
  @{ n = "Devanagari"; r = @(, @(0x0900, 0x097F)) }
)
function Script-Of([int]$c) {
  foreach ($s in $ranges) { foreach ($r in $s.r) { if ($c -ge $r[0] -and $c -le $r[1]) { return $s.n } } }
  return $null
}
$kana = [regex]'[\u3040-\u30FF]'
$latin = [regex]'[A-Za-z\u00C0-\u024F]{4,}'
$files = @(foreach ($p in $Path) { Get-ChildItem -Path $p -File -ErrorAction Stop })
if (-not $files.Count) { throw "no files: $Path" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$rows = @(); $tc = 0; $ts = 0; $tl = 0
foreach ($f in $files) {
  $t = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8)
  $chars = 0; $script = 0; $by = @{}; $pieces = @()
  for ($i = 0; $i -lt $t.Length; $i++) {
    if ([char]::IsWhiteSpace($t[$i])) { continue }
    $chars++
    $s = Script-Of ([int]$t[$i])
    if ($s) {
      $script++; $by[$s] = 1 + [int]$by[$s]
      if ($Show -and ($i -eq 0 -or -not (Script-Of ([int]$t[$i - 1])))) {
        $a = [Math]::Max(0, $i - 12); $pieces += "    [$s] ..." + $t.Substring($a, [Math]::Min(36, $t.Length - $a)).Replace("`n", " ") + "..."
      }
    }
  }
  $lat = 0
  foreach ($sent in ($t -split '[\u3002\uFF01\uFF1F!?\n]')) {  # (escapes: Windows PowerShell 5.1 reads this file as ANSI)
    if (-not $kana.IsMatch($sent)) { continue }
    foreach ($m in $latin.Matches($sent)) {
      $lat++
      if ($Show) { $pieces += "    [latin] " + $m.Value + " in: " + $sent.Substring(0, [Math]::Min(60, $sent.Length)).Trim() }
    }
  }
  $per = if ($chars) { 1000.0 * ($script + $lat) / $chars } else { 0 }
  $rows += [pscustomobject]@{ answer = $f.Name; chars = $chars; script = $script; latin = $lat
    per1000 = [string]::Format($inv, "{0:F2}", $per); which = (($by.Keys | Sort-Object | ForEach-Object { "$_ $($by[$_])" }) -join ", ") }
  $tc += $chars; $ts += $script; $tl += $lat
  if ($Show -and $pieces.Count) { Write-Output ("{0}:" -f $f.Name); $pieces | Select-Object -First 20 | ForEach-Object { Write-Output $_ } }
}
$rows | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
$clean = @($rows | Where-Object { $_.script -eq 0 -and $_.latin -eq 0 }).Count
Write-Output ([string]::Format($inv, "all {0} answers: {1} characters, {2} of other scripts, {3} Latin words in Japanese sentences: {4:F2} per 1,000 characters; {5} of {0} answers clean",
  $rows.Count, $tc, $ts, $tl, $(if ($tc) { 1000.0 * ($ts + $tl) / $tc } else { 0 }), $clean))
