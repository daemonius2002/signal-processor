# Closes the Signal Processor instances that run from the given install folder.
# Only processes whose executable lives in that folder are touched.
param([string]$Dir)
$prefix = (Join-Path $Dir 'python').TrimEnd('\') + '\'
$procs = @(Get-Process -Name python, pythonw -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase) })
if ($procs.Count -eq 0) { exit 0 }
foreach ($p in $procs) { try { [void]$p.CloseMainWindow() } catch {} }
$deadline = (Get-Date).AddSeconds(5)
while ((Get-Date) -lt $deadline) {
    if (@($procs | Where-Object { -not $_.HasExited }).Count -eq 0) { break }
    Start-Sleep -Milliseconds 200
}
foreach ($p in $procs) { if (-not $p.HasExited) { try { $p.Kill() } catch {} } }
foreach ($p in $procs) { try { [void]$p.WaitForExit(3000) } catch {} }
exit 0
