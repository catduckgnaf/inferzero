# Which kinds of PC this one is, from Windows' display adapters: the words of scripts\options.tsv's best_for column.
# Dot-sourced by zero-server.ps1 (zero-server options) and zero-launcher.ps1 (the options section), so the two agree.
# Windows PowerShell 5.1 or 7.
#   igpu-xe2      Lunar Lake / Panther Lake iGPUs (Arc 130V / 140V, B370 / B390: Xe2 / Xe3, 16-lane XMX)
#   igpu-140t     the Arc 140T (Xe-LPG Plus, 8-lane XMX: the 140T preset)
#   igpu-xe-lpg   other Core Ultra iGPUs (Meteor Lake's Arc Graphics)
#   alchemist     discrete Arc A-series (A770, A750, A580, ...)
#   b-series      discrete Arc B-series (B580, B570, Pro B50 / B60 / B65 / B70)
#   multi-gpu     two or more discrete Arc cards
# (the iGPU names first: Panther Lake's B370 / B390 have a B and three digits too, and are no discrete card; the
#  Researcher's review, 2026-11-21)
function Get-PcKinds {
  $k = @(); $discrete = 0
  $gs = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Intel" } | ForEach-Object { $_.Name })
  foreach ($g in $gs) {
    if ($g -match 'B3[0-9]0|140V|130V|Xe2') { $k += "igpu-xe2" }
    elseif ($g -match '140T') { $k += "igpu-140t" }
    elseif ($g -match 'Arc\S* A\d{3}') { $k += "alchemist"; $discrete++ }
    elseif ($g -match 'Arc\S* (Pro )?B\d{2,3}\b') { $k += "b-series"; $discrete++ }
    elseif ($g -match 'Arc|Graphics') { $k += "igpu-xe-lpg" }
  }
  if ($discrete -gt 1) { $k += "multi-gpu" }
  return @($k | Select-Object -Unique)
}
