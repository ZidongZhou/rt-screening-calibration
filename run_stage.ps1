param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("primary", "robustness", "publication", "additional", "selective", "checks")]
    [string]$Stage,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$arguments = @("scripts/run_reproduction_stage.py", $Stage)
if ($Force) {
    $arguments += "--force"
}
python @arguments
