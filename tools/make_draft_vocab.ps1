# The ranked token list for drafting over a reduced vocabulary (ARC_MTP_VOCAB=N, QUEUE D1; the idea from FR-Spec, Zhao
# et al., ACL 2025): a model's token ids, most frequent first, counted over a mixed corpus of English prose, Markdown,
# C++, OpenCL C, PowerShell, bash, HTML / JavaScript and JSON (this repository's own files), tokenized with the
# model's vocabulary by `runz --dump-ids`. The engine takes the first N ids of the list (and every control token) as
# the rows its drafts are made from; verification keeps the whole vocabulary, so the text never changes, only how
# often a draft is accepted.
#   powershell -File tools\make_draft_vocab.ps1 -Model <a .gguf of that vocabulary> [-Vocab 248320] [-Out FILE]
# (-Vocab: the model's vocabulary size, the file's name scripts\draft-vocab\vocab-<Vocab>.txt; Qwen3.5 / 3.6 / 3.8
#  share 248,320)
# The file: '#' lines, then one "id count" a line, most frequent first (ties: the lower id first). Windows PowerShell
# 5.1 or 7.
# Also the model's own answers in Indonesian, Chinese and Spanish (scripts\draft-vocab\corpus\*.ids: the greedy token ids
# of runz --mtp on chat prompts in those languages, thinking off), each id counted -AnswerWeight times, so that a few
# thousand tokens of other languages rank beside millions of the repository's English and code (FR-Spec counts the
# model's own generations too)
param([Parameter(Mandatory = $true)][string]$Model, [int]$Vocab = 248320, [string]$Out = "", [string]$Runz = "",
  [int]$AnswerWeight = 100)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not $Runz) { $Runz = Join-Path $root "bin\runz.exe" }
$utf8 = New-Object Text.UTF8Encoding $false
$pats = @("README.md", "USAGE.md", "CHANGELOG.md", "docs\*.md", "tools\*.cpp", "tools\*.h", "zero\*.h", "zero\*.cpp",
  "zero\xe2\*.h", "src\*.h", "zero\kernels\*.cl", "scripts\*.ps1", "*.sh", "bench\*.ps1", "bench\*.sh", "webui\index.html",
  "tests\data\*.jinja", "tests\chat_golden\*.cases", "site\*.html")
$files = @(foreach ($p in $pats) { Get-ChildItem -Path (Join-Path $root $p) -File -ErrorAction SilentlyContinue })
if (-not $files.Count) { throw "no corpus files under $root" }
$counts = New-Object 'System.Collections.Generic.Dictionary[int,long]'
$tmp = [IO.Path]::GetTempFileName()
$n = 0; $total = 0
try {
  foreach ($f in $files) {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Runz $Model --text-file $f.FullName --dump-ids $tmp 2>&1 | Out-Null
    $rc = $LASTEXITCODE; $ErrorActionPreference = $old
    if ($rc -ne 0) { Write-Host "skipped (runz exit $rc): $($f.FullName)"; continue }
    foreach ($s in ([IO.File]::ReadAllText($tmp, $utf8) -split ',')) {
      $t = $s.Trim()
      if (-not $t) { continue }
      $id = [int]$t
      if ($counts.ContainsKey($id)) { $counts[$id]++ } else { $counts[$id] = 1 }
      $total++
    }
    $n++
  }
} finally { Remove-Item $tmp -ErrorAction SilentlyContinue }
$na = 0
foreach ($f in @(Get-ChildItem -Path (Join-Path $root "scripts\draft-vocab\corpus\*.ids") -File -ErrorAction SilentlyContinue)) {
  foreach ($s in ([IO.File]::ReadAllText($f.FullName, $utf8) -split ',')) {
    $t = $s.Trim()
    if (-not $t) { continue }
    $id = [int]$t
    if ($counts.ContainsKey($id)) { $counts[$id] += $AnswerWeight } else { $counts[$id] = $AnswerWeight }
    $na++
  }
  $n++
}
$ranked = @($counts.GetEnumerator() | Sort-Object @{ Expression = { $_.Value }; Descending = $true }, @{ Expression = { $_.Key }; Ascending = $true })
if (-not $Out) { $Out = Join-Path $root "scripts\draft-vocab\vocab-$Vocab.txt" }  # (the engine finds it by the model's vocabulary size)
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$sb = New-Object Text.StringBuilder
[void]$sb.Append("# Token ids for drafting over a reduced vocabulary (ARC_MTP_VOCAB=N), most frequent first: tools\make_draft_vocab.ps1`n")
[void]$sb.Append("# $($ranked.Count) distinct ids: $total tokens of this repository's files (prose, Markdown, code, HTML, JSON) and $na of the model's own answers (Indonesian, Chinese, Spanish; x$AnswerWeight), $n files`n")
foreach ($kv in $ranked) { [void]$sb.Append("$($kv.Key) $($kv.Value)`n") }
[IO.File]::WriteAllText($Out, $sb.ToString(), $utf8)
"$Out : $($ranked.Count) distinct ids, $total tokens, $n files"
