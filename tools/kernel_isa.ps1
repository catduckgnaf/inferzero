# Compiles one OpenCL kernel for an Intel GPU with Intel's offline compiler (ocloc, from the oneAPI DPC++/C++ compiler
# install) and summarizes its GPU assembly: instruction count, DP4a, DPAS (XMX), memory messages (of them: shared
# local memory, 2D block loads), scratch (spill) traffic. -Device: ocloc's name of the GPU, mtl (Meteor Lake, the
# default) or e.g. dg2-g10 (A770 / A750), bmg-g21 (B580 / B60), bmg-g31 (B70), ptl-h (Panther Lake's B390), lnl-m
# (Lunar Lake), arl-h (the 140T); "ocloc compile --help" lists them all.
# The PowerShell port of the former kernel_isa.py (2026-11-21, the user: no Python in the repository); the same
# summary. Windows PowerShell 5.1 or 7.
#   powershell -File tools\kernel_isa.ps1 FILE.cl KERNEL -Options "BUILD OPTIONS" [-Device mtl] [-OutDir out_dir] [-Loops]
# (-Loops: each loop's body counted too, the biggest first, with its instructions a DPAS / DP4a: docs/xe2-tuning.md 16)
# (-Options named: build options start with a dash, which PowerShell would otherwise take for a parameter)
# The .asm is kept in out_dir (default: a new folder in %TEMP%) for reading. $env:OCLOC: ocloc's path if not the one
# below. ocloc carries its own compiler version, which can differ slightly from the one in the graphics driver.
param([Parameter(Mandatory = $true)][string]$File, [Parameter(Mandatory = $true)][string]$Kernel,
  [Parameter(Mandatory = $true)][string]$Options, [string]$Device = "mtl", [string]$OutDir = "", [switch]$Loops)
$ErrorActionPreference = "Stop"
$ocloc = if ($env:OCLOC) { $env:OCLOC } else { 'C:\Program Files (x86)\Intel\oneAPI\ocloc\2026.1\bin\ocloc.exe' }
$src = (Resolve-Path -LiteralPath $File).Path
$out = if ($OutDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir) } else { Join-Path ([IO.Path]::GetTempPath()) ("kisa_" + [Guid]::NewGuid().ToString("N").Substring(0, 8)) }
New-Item -ItemType Directory -Force $out | Out-Null
$old = $ErrorActionPreference; $ErrorActionPreference = "Continue"  # (5.1: a native program's stderr lines are not errors here)
try {
  $c = @(& $ocloc compile -file $src -device $Device -options $Options -output k -out_dir $out 2>&1 | ForEach-Object { "$_" })
  $rc = $LASTEXITCODE
} finally { $ErrorActionPreference = $old }
$notes = @($c | Where-Object { $_.Contains($Kernel) -or $_.ToLower().Contains("error") })
if ($rc -ne 0) {
  if ($notes.Count) { $notes } else { $c | Select-Object -Last 20 }
  exit 1
}
$dis = Join-Path $out "dis"
New-Item -ItemType Directory -Force $dis | Out-Null
Push-Location $dis
try {
  $ErrorActionPreference = "Continue"
  $bin = Get-ChildItem -LiteralPath $out -Filter "k*.bin" | Select-Object -First 1  # (k_<device>.bin)
  & $ocloc disasm -file $bin.FullName -device $Device -dump . 2>&1 | Out-Null
} finally { $ErrorActionPreference = $old; Pop-Location }
$asm = Join-Path $dis ".text.$Kernel.asm"
if (-not (Test-Path -LiteralPath $asm)) {
  "kernel not found in the binary: " + ((Get-ChildItem -LiteralPath $dis -Force | Where-Object { $_.Name.StartsWith(".text.") } | ForEach-Object { $_.Name }) -join ", ")
  exit 1
}
$lines = @([IO.File]::ReadAllLines($asm, (New-Object Text.UTF8Encoding $false)) | Where-Object { $_.Trim() -and -not $_.Trim().EndsWith(":") })
$instr = @($lines | Where-Object { -not $_.Trim().StartsWith("//") })
function Count-Of([string]$pat) { @($instr | Where-Object { $_ -match $pat }).Count }
"$Kernel on $Device [$Options]"
foreach ($n in $notes) { "  ocloc: " + $n.Trim() }
"  instructions {0} | dp4a {1} | dpas {2} | send (memory) {3}, of them slm {4}, 2D block {5} | scratch (spill) {6} | nop/sync {7}" -f `
  $instr.Count, (Count-Of '\bdp4a\b'), (Count-Of '\bdpas'), (Count-Of '\bsend'), (Count-Of '\bsend.*\bslm\b'),
  (Count-Of '\bsend.*(block2d|2d_block|\.2d\b)'), (Count-Of 'scratch'), (Count-Of '\bnop\b|sync\.')
"  asm: $asm"
if ($Loops) {
  # each loop (a branch back to an earlier label) with its body's counts, the biggest first: what one iteration costs
  $all = [IO.File]::ReadAllLines($asm, (New-Object Text.UTF8Encoding $false))
  $labelAt = @{}
  for ($i = 0; $i -lt $all.Count; $i++) { if ($all[$i] -match '^\s*(L\d+):') { $labelAt[$matches[1]] = $i } }
  $found = @{}
  $rows = @()
  for ($i = 0; $i -lt $all.Count; $i++) {
    if ($all[$i] -notmatch '\b(goto|jmpi|while)\b') { continue }
    foreach ($m in [regex]::Matches($all[$i], '\b(L\d+)\b')) {
      $t = $m.Groups[1].Value
      if (-not $labelAt.ContainsKey($t) -or $labelAt[$t] -ge $i) { continue }
      $s = $labelAt[$t]
      if ($found.ContainsKey("$s-$i")) { continue }
      $found["$s-$i"] = 1
      $body = @($all[$s..$i] | Where-Object { $_.Trim() -and -not $_.Trim().EndsWith(":") -and -not $_.Trim().StartsWith("//") })
      $n = { param($p) @($body | Where-Object { $_ -match $p }).Count }
      $dp = (& $n '\bdpas') + (& $n '\bdp4a\b')
      $line = "  loop (asm lines {0}-{1}): instructions {2} | dpas {3} | dp4a {4} | send {5} (slm {6}) | sync/nop {7} | mov {8} | integer {9} | branches {10}{11}" -f `
        ($s + 1), ($i + 1), $body.Count, (& $n '\bdpas'), (& $n '\bdp4a\b'), (& $n '\bsend'), (& $n '\bsend.*\bslm\b'),
        (& $n '\bsync\.|\bnop\b'), (& $n '^\s*(\([^)]*\)\s*)?mov'),
        (& $n '^\s*(\([^)]*\)\s*)?(add|addc|add3|mul|mad|shl|shr|asr|and|or|xor|bfn)\b'),
        (& $n '\b(goto|jmpi|while|join|endif|else|if)\b'), $(if ($dp) { " | " + ($body.Count / $dp).ToString("0.0", [Globalization.CultureInfo]::InvariantCulture) + " a dpas / dp4a" } else { "" })
      $rows += , @($body.Count, $line)
    }
  }
  if (-not $rows.Count) { "  no loops" }
  $rows | Sort-Object { -$_[0] } | ForEach-Object { $_[1] }
}
