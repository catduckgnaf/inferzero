# Writes the llama.cpp comparison tables of docs/review.md (the block between <!-- DATA-BEGIN --> and
# <!-- DATA-END -->) from bench/results/results.jsonl: per model / engine / test / prompt the best of the repeated
# runs (highest tok/s, shortest time to first token), as the review's method says.
# The PowerShell port of the former review_data.py (2026-11-21, the user: no Python in the repository); the same page
# byte for byte. Windows PowerShell 5.1 or 7.
#   powershell -File bench\review_data.ps1 [results.jsonl] [review.md]
param([string]$Results = "", [string]$Page = "")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Numerics
$root = Split-Path $PSScriptRoot -Parent
if (-not $Results) { $Results = Join-Path $root "bench\results\results.jsonl" }
if (-not $Page) { $Page = Join-Path $root "docs\review.md" }
$utf8 = New-Object Text.UTF8Encoding $false
$inv = [Globalization.CultureInfo]::InvariantCulture

# a number as Python's f"{v:,.{d}f}" writes it: the double's exact value rounded half to even, thousands with commas
function Num([double]$v, [int]$d) {
  $bits = [BitConverter]::DoubleToInt64Bits($v)
  $neg = $bits -lt 0
  $ex = [int](($bits -shr 52) -band 0x7FF)
  $man = [System.Numerics.BigInteger]($bits -band 0xFFFFFFFFFFFFF)
  if ($ex -eq 0) { $ex = 1 } else { $man += [System.Numerics.BigInteger]::Pow(2, 52) }
  $ex -= 1075  # (value = man * 2^ex)
  $p10 = [System.Numerics.BigInteger]::Pow(10, $d)
  if ($ex -ge 0) { $q = $man * $p10 * [System.Numerics.BigInteger]::Pow(2, $ex) }
  else {
    $den = [System.Numerics.BigInteger]::Pow(2, -$ex)
    $num = $man * $p10
    $r = [System.Numerics.BigInteger]::Zero
    $q = [System.Numerics.BigInteger]::DivRem($num, $den, [ref]$r)
    $c = ($r * 2).CompareTo($den)
    if ($c -gt 0 -or ($c -eq 0 -and -not $q.IsEven)) { $q += 1 }
  }
  $ip = [System.Numerics.BigInteger]::Divide($q, $p10)
  $fp = [System.Numerics.BigInteger]::Remainder($q, $p10)
  $s = $ip.ToString("N0", $inv)
  if ($d -gt 0) { $s += "." + $fp.ToString($inv).PadLeft($d, "0") }
  if ($neg -and -not $q.IsZero) { $s = "-" + $s }
  return $s
}
function Get-F($r, [string]$k) { if ($null -eq $r) { return $null }; $p = $r.PSObject.Properties[$k]; if ($p) { $p.Value } else { $null } }

# (rows recorded before the rename, 2026-11-10, name the engine "arcinferzero": read as "inferzero")
$oldEngine = @{ "arcinferzero" = "inferzero"; "arcinferzero-npu" = "inferzero-npu" }
$best = [ordered]@{}
foreach ($line in [IO.File]::ReadAllLines((Resolve-Path -LiteralPath $Results).Path, $utf8)) {
  $line = $line.Trim([char]0xFEFF).Trim()
  if (-not $line) { continue }
  $r = $line | ConvertFrom-Json
  # (PowerShell 7's ConvertFrom-Json turns an ISO date into a DateTime, written back in the local format; 5.1 keeps
  #  the text: the text taken from the line itself in both)
  $tm = [regex]::Match($line, '"time"\s*:\s*"([^"]*)"')
  if ($tm.Success) { $r.time = $tm.Groups[1].Value }
  $e = Get-F $r "engine"
  if ($oldEngine.ContainsKey("$e")) { $r.engine = $oldEngine["$e"] }
  $key = "{0}|{1}|{2}|{3}" -f (Get-F $r "model"), (Get-F $r "engine"), (Get-F $r "test"), (Get-F $r "prompt")
  $old = $best[$key]
  if ($null -eq $old) { $best[$key] = $r }
  elseif ($r.PSObject.Properties["ttft_s"]) { if ([double]$r.ttft_s -lt [double]$old.ttft_s) { $best[$key] = $r } }
  elseif ($r.PSObject.Properties["tps"]) { if ([double]$r.tps -gt [double]$old.tps) { $best[$key] = $r } }
  elseif ([string]::CompareOrdinal("$(Get-F $r 'time')", "$(Get-F $old 'time')") -ge 0) { $best[$key] = $r }
}
function Get-Run($model, $engine, $test, $prompt = $null) { $best["$model|$engine|$test|$prompt"] }
$engines = @(@("inferzero", "inferzero"), @("vulkan", "llama.cpp Vulkan"), @("sycl", "llama.cpp SYCL"))
$models = @(@("4b", "Qwen3.5 4B"), @("9b", "Qwen3.5 9B"), @("27b", "Qwen3.8 27B"), @("g12", "Gemma 4 12B"), @("35b", "Qwen3.6 35B-A3B (GSQ 2-bit)"))
$prompts = @(@("code-11k", "code 11K"), @("code-34k", "code 34K"))
$notRun = "not run"
function Cell($r, [string]$field, [int]$d) { $v = Get-F $r $field; if ($null -eq $v) { return $notRun }; return (Num ([double]$v) $d) }

$out = New-Object System.Collections.Generic.List[string]
# (the latest date: as Python's max() of the strings, compared ordinally)
$mx = ""; foreach ($v in $best.Values) { $t = "$(Get-F $v 'time')"; $t = $t.Substring(0, [Math]::Min(10, $t.Length)); if ([string]::CompareOrdinal($t, $mx) -gt 0) { $mx = $t } }; $date = $mx
$out.Add("Measured $date with the OpenCL engine; best of the repeated runs per engine. llama.cpp cannot load the GSQ 2-bit Qwen3.6 35B-A3B, so its row has inferzero only.")
$out.Add("")
$out.Add("Time to first token (seconds) on the code prompts, and generation speed right after the prompt (tok/s):")
$out.Add("")
$out.Add("| Model | Prompt | Tokens | " + (($engines | ForEach-Object { "$($_[1]) TTFT" }) -join " | ") + " | " + (($engines | ForEach-Object { "$($_[1]) gen" }) -join " | ") + " |")
$out.Add("|---|---|---:|" + ("---:|" * (2 * $engines.Count)))
foreach ($mm in $models) {
  foreach ($pp in $prompts) {
    $rs = @($engines | ForEach-Object { , (Get-Run $mm[0] $_[0] "ttft" $pp[0]) })
    $n = $null
    foreach ($r in $rs) { $v = Get-F $r "n_prompt"; if ($v) { $n = $v; break } }
    if ($null -eq $n) { $n = if ($mm[0] -ne "g12") { @{ "code-11k" = 11240; "code-34k" = 33937 }[$pp[0]] } else { @{ "code-11k" = 13921; "code-34k" = 35957 }[$pp[0]] } }
    $out.Add("| $($mm[1]) | $($pp[1]) | " + ([long]$n).ToString("N0", $inv) + " | " + (($rs | ForEach-Object { Cell $_ "ttft_s" 1 }) -join " | ") + " | " +
      (($rs | ForEach-Object { Cell $_ "tg_tps" 2 }) -join " | ") + " |")
  }
}
$out.Add("")
$out.Add("Synthetic throughput (llama-bench style; tok/s): prompt processing over 512 and 2,048 tokens, generation of 64 tokens after a short prompt:")
$out.Add("")
$out.Add("| Model | " + (($engines | ForEach-Object { "$($_[1]) pp512" }) -join " | ") + " | " + (($engines | ForEach-Object { "$($_[1]) pp2048" }) -join " | ") + " | " +
  (($engines | ForEach-Object { "$($_[1]) tg64" }) -join " | ") + " |")
$out.Add("|---|" + ("---:|" * (3 * $engines.Count)))
foreach ($mm in $models) {
  $row = @()
  foreach ($t in "pp512", "pp2048", "tg64") { foreach ($e in $engines) { $row += Cell (Get-Run $mm[0] $e[0] $t) "tps" 1 } }
  $out.Add("| $($mm[1]) | " + ($row -join " | ") + " |")
}
$out.Add("")
$out.Add("Perplexity (40 chunks of 512 tokens of llama.cpp documentation, second half of each chunk scored; lower is better). Gemma 4's raw-text perplexity is not a usable accuracy metric on this corpus (see Accuracy); the 35B row is the GSQ file, which llama.cpp cannot load:")
$out.Add("")
$out.Add("| Model | " + (($engines | ForEach-Object { $_[1] }) -join " | ") + " |")
$out.Add("|---|" + ("---:|" * $engines.Count))
foreach ($mm in $models) { $out.Add("| $($mm[1]) | " + (($engines | ForEach-Object { Cell (Get-Run $mm[0] $_[0] "ppl") "ppl" 4 }) -join " | ") + " |") }
# speculative decoding runs of the benchmark script, when there are any
$spec = @(foreach ($mm in ($models + , @("e4b", "Gemma 4 E4B"))) { $r = Get-Run $mm[0] "inferzero" "mtp" "code-11k"; if ($r) { , @($mm[0], $mm[1], $r) } })
if ($spec.Count) {
  $out.Add(""); $out.Add("Speculative decoding after the 11K code prompt (tok/s):"); $out.Add("")
  $out.Add("| Model | plain | drafting | drafts accepted | tokens per pass |"); $out.Add("|---|---:|---:|---:|---:|")
  foreach ($s in $spec) {
    $r = $s[2]
    $out.Add("| $($s[1]) | $(Cell $r 'plain_tps' 2) | $(Cell $r 'mtp_tps' 2) | $(Cell $r 'accept_pct' 1) % | $(Cell $r 'tokens_per_pass' 2) |")
  }
}
# one task on every model (bench/prompt_bench.ps1): the latest run per model, linked to labelled copies of the pages
$landDir = Join-Path $root "bench\results\landing"
$pubDir = Join-Path (Split-Path $Page -Parent) "landing"
$names = [ordered]@{ "27b" = "Qwen3.8 27B"; "35b" = "Qwen3.6 35B-A3B"; "9b" = "Qwen3.5 9B"; "4b" = "Qwen3.5 4B"; "g12" = "Gemma 4 12B"; "e4b" = "Gemma 4 E4B" }
$sumFile = Join-Path $landDir "summary.jsonl"
if (Test-Path -LiteralPath $sumFile) {
  $runs = @{}
  foreach ($line in [IO.File]::ReadAllLines($sumFile, $utf8)) { if ($line.Trim()) { $r = $line | ConvertFrom-Json; $runs[$r.model] = $r } }
  $rows = @()
  foreach ($m in "27b", "35b", "9b", "4b", "g12", "e4b") {
    $r = $runs[$m]
    if (-not $r) { continue }
    $src = Join-Path $landDir "$m.html"
    $link = $names[$m]
    if ((Test-Path -LiteralPath $src) -and (Get-F $r "html_chars")) {
      New-Item -ItemType Directory -Force $pubDir | Out-Null
      $html = [IO.File]::ReadAllText($src, $utf8)
      $label = '<div style="position:fixed;left:8px;bottom:8px;z-index:2147483647;font:12px/1.4 system-ui,sans-serif;' +
        'background:#111;color:#fff;padding:6px 10px;border-radius:6px;opacity:.85">' + "Output of $($names[$m]) on inferzero (benchmark prompt, fictional shop)</div>"
      $rx = New-Object Text.RegularExpressions.Regex('(<body[^>]*>)', 'IgnoreCase')
      $html = if ($rx.IsMatch($html)) { $rx.Replace($html, { param($x) $x.Groups[1].Value + $label }, 1) } else { $label + $html }
      [IO.File]::WriteAllText((Join-Path $pubDir "$m.html"), $html, $utf8)
      $link = "[$($names[$m])](landing/$m.html)"
    }
    $f = { param($k, $d) $v = Get-F $r $k; if ($null -eq $v) { $notRun } else { Num ([double]$v) $d } }
    $rows += "| $link | $(& $f 'ttft_s' 1) | $(& $f 'total_s' 1) | $(& $f 'completion_tokens' 0) | $(& $f 'gen_tps' 1) |"
  }
  if ($rows.Count) {
    $out.Add(""); $out.Add("One task on every model through the server (bench/prompt_bench.ps1):"); $out.Add("")
    $out.Add("| Model | first token (s) | total (s) | tokens | tok/s |"); $out.Add("|---|---:|---:|---:|---:|")
    foreach ($x in $rows) { $out.Add($x) }
  }
}
$blob = $out -join "`n"
$s = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $Page).Path, $utf8)
$rx = New-Object Text.RegularExpressions.Regex('<!-- DATA-BEGIN -->.*?<!-- DATA-END -->', 'Singleline')
if ($rx.Matches($s).Count -ne 1) { throw "data markers not found" }
$s2 = $rx.Replace($s, { param($x) "<!-- DATA-BEGIN -->`n" + $blob + "`n<!-- DATA-END -->" }, 1)
[IO.File]::WriteAllText((Resolve-Path -LiteralPath $Page).Path, $s2.Replace("`r`n", "`n"), $utf8)
"filled $Page ($($out.Count) lines of tables)"
