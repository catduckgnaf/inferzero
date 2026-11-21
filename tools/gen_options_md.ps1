# docs/options.md from scripts/options.tsv (the launchers' catalog of opt-in switches, QUEUE O0 / O6). A tool for
# whoever edits the catalog (users never need it: the launchers read the .tsv themselves). Windows PowerShell 5.1 or 7.
#   powershell -File tools\gen_options_md.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$utf8 = New-Object Text.UTF8Encoding $false
$lines = [IO.File]::ReadAllText((Join-Path $root "scripts\options.tsv"), $utf8) -split "\r?\n" | Where-Object { $_ -ne "" }  # (LF or CRLF)
$cols = $lines[0] -split "`t"
$rows = foreach ($l in $lines[1..($lines.Count - 1)]) {
  $v = $l -split "`t"
  $d = @{}
  for ($i = 0; $i -lt $cols.Count; $i++) { $d[$cols[$i]] = $v[$i] }
  $d
}
$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Options you can turn on"); $md.Add("")
$md.Add('The switches below are off unless the table says otherwise. Each changes speed or memory; the ones marked ' +
  '"same output" give the same text bit for bit, the others are checked against an f32 reference (KL). The ' +
  'launchers read this list from `scripts/options.tsv`; this page is made from that file ' +
  '(`powershell -File tools\gen_options_md.ps1`).')
$md.Add("")
$md.Add('On Windows: `zero-server start -Opt <id>` turns one on (kept for later starts), `-NoOpt <id>` off, and ' +
  '`zero-server options` lists them with whether each is for your PC. On Linux: `./zero-launcher.sh --opt <id>` ' +
  '(`--opt <id>=<value>` for one that takes a value), `--no-opt <id>`, `--default-opt <id>`, `--options` to list them, ' +
  'or `o` in its menu; the same choices file as on Windows (`~/.inferzero/options.json`), also written into the ' +
  'systemd unit with `--install-service`. In Docker: `OPTIONS` in compose.yaml or kubernetes.yaml, e.g. ' +
  '`"copy_draft,xcpu=off,budget=12"` (exactly these at each start). By hand: set the environment variable before ' +
  'starting the server (PowerShell: `$env:NAME = "value"`, bash: `export NAME=value`), or add the flag to serverz''s ' +
  'command line.')
$md.Add("")
foreach ($d in $rows) {
  $how = $d.name + $(if ($d.kind -eq "env") { "=" + $d.on } elseif ($d.on) { " " + $d.on } else { "" })
  $link = if ($d.docs.StartsWith("docs/")) { $d.docs.Substring(5) } else { "../" + $d.docs }
  $md.Add("## " + $d.title + " (``" + $how + "``)"); $md.Add("")
  $md.Add("- **What it does:** " + $d.what + ".")
  $md.Add("- **Default:** " + $d.default + ".")
  $md.Add("- **Best for:** " + $d.best_for.Replace(";", ", ") + ".")
  $md.Add("- **Cost:** " + $d.cost + ".")
  $md.Add("- **Output:** " + $(if ($d.output -eq "same") { "the same, bit for bit" } else { "slightly different, checked against f32 (KL)" }) + ".")
  $md.Add("- **Measured:** " + $d.measured + ".")
  if ($d.conflicts) { $md.Add("- **Note:** " + $d.conflicts + ".") }
  $md.Add("- **More:** [" + $d.docs + "](" + $link + "), " + $d.status + "."); $md.Add("")
}
[IO.File]::WriteAllText((Join-Path $root "docs\options.md"), ($md -join "`n"), $utf8)
"docs/options.md: $(@($rows).Count) options"
