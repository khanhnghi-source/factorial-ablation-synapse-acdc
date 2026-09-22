# ============================================================================
#  audit_log_staleness.ps1 -- detect test logs that are STALE relative to the
#  checkpoint they were produced from
# ============================================================================
#
#  WHY THIS SCRIPT EXISTS
#  ----------------------
#  11/09/2026: Synapse +CBAM+DS seed 3456 was found to have a test log generated
#  back in January FROM A DIFFERENT CHECKPOINT than the best_model.pth currently
#  on disk. The evidence is the CBAM alpha_raw fingerprint printed inside the
#  test log itself:
#
#      January log : cbam1 = -2.2610 , cbam2 = -2.1888 , cbam3 = -2.0835
#      11/09 run   : cbam1 = -2.2605 , cbam2 = -2.1886 , cbam3 = -2.0813
#
#  Same path, different weights => the model was retrained and the test log was
#  never regenerated. Result: Mean Dice 0.7973 -> 0.8062 (+0.89 points).
#
#  For any run whose best_model.pth is NEWER than its test log, the numbers in
#  KetQua_v2.xlsx belong to a model that no longer exists.
#
#  USAGE
#  -----
#      .\audit_log_staleness.ps1 -Repo <repo path> -LogRoots <log directories>
#      .\audit_log_staleness.ps1 -Csv audit.csv      # export to a file
#
#  By default the paths are taken from the environment variables SEQATT_REPO /
#  SEQATT_LOG_ROOTS when those are set, otherwise they are resolved relative to
#  the script location. No machine-specific absolute path is hard-coded: this
#  script is released together with the paper.
#
#  The script is READ-ONLY; it modifies nothing and deletes nothing.
# ============================================================================

param(
    [string]$Repo     = $null,
    [string[]]$LogRoots = $null,
    [string]$Csv      = ""
)

$ErrorActionPreference = "Stop"

# Resolve the paths AFTER param(): $PSScriptRoot is not guaranteed to be usable
# inside the param() block for every way the script can be invoked. Priority
# order:
#   1. arguments passed in
#   2. environment variables SEQATT_REPO / SEQATT_LOG_ROOTS (several paths,
#      separated by ';')
#   3. paths relative to the script location
$Here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }

if (-not $Repo) {
    $Repo = if ($env:SEQATT_REPO) { $env:SEQATT_REPO } else { Split-Path -Parent $Here }
}
if (-not $LogRoots) {
    $LogRoots = if ($env:SEQATT_LOG_ROOTS) { $env:SEQATT_LOG_ROOTS -split ';' }
                else { @((Join-Path $Here 'test_log'),
                         (Join-Path (Split-Path -Parent $Here) 'results\test_log')) }
}

$ModelRoot = Join-Path $Repo "model"   # may be a junction pointing at another drive
if (-not (Test-Path $ModelRoot)) { Write-Error "Not found: $ModelRoot"; exit 1 }

# ---------------------------------------------------------------------------
# Build the log INDEX from ALL roots.
#
# Lesson from 11/09: the script originally searched only on E:. The 20 factorial
# logs produced on Colab live on H:, so the script reported "NO LOG" for 20
# perfectly healthy runs -- and reported it silently. An integrity checker that
# raises false alarms is more dangerous than no checker at all. The script now
# PRINTS which roots were scanned and how many logs were indexed; if that count
# is zero, stop right there.
#
# Index key = "<parent directory name>|<file name>", because the file name
# contains NEITHER the dataset NOR the config:
# TU_pretrain_..._s1234_tta_simple.txt appears identically under both
# test_log_TU_ACDC224_Phase3_DS and test_log_TU_Synapse224_Phase3_DS.
# ---------------------------------------------------------------------------
$logIndex = @{}
$rootsUsed = New-Object System.Collections.Generic.List[string]

Write-Host ""
Write-Host "Model root : $ModelRoot"
foreach ($r in $LogRoots) {
    if (-not (Test-Path $r)) {
        Write-Host ("Log root   : {0}   [DOES NOT EXIST - skipped]" -f $r) -ForegroundColor DarkGray
        continue
    }
    $found = @(Get-ChildItem $r -Recurse -Filter "*_tta_simple.txt" -File -ErrorAction SilentlyContinue)
    foreach ($f in $found) {
        $key = $f.Directory.Name + "|" + $f.Name
        if (-not $logIndex.ContainsKey($key)) { $logIndex[$key] = $f.FullName }
    }
    $rootsUsed.Add($r)
    Write-Host ("Log root   : {0}   [{1} logs]" -f $r, $found.Count) -ForegroundColor Green
}

if ($logIndex.Count -eq 0) {
    Write-Error "No log index could be built. Pass the correct -LogRoots and run again."
    exit 1
}
Write-Host ("Total logs indexed: {0}" -f $logIndex.Count)
Write-Host ""

$rows = New-Object System.Collections.Generic.List[object]

Get-ChildItem $ModelRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "TU_*" } | ForEach-Object {

    $cfgDir = $_.Name

    Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue | ForEach-Object {

        $runName = $_.Name
        $ckpt    = Join-Path $_.FullName "best_model.pth"
        if (-not (Test-Path $ckpt)) { return }

        # Skip the batch_size=24 runs, which have been quarantined
        if ($runName -match "_bs24_") { return }

        $ckptTime = (Get-Item $ckpt).LastWriteTime

        $logKey  = "test_log_" + $cfgDir + "|" + $runName + "_tta_simple.txt"
        $logPath = $null
        if ($logIndex.ContainsKey($logKey)) { $logPath = $logIndex[$logKey] }

        # --- compute into variables first, build the object afterwards --------
        # Do NOT place "-replace 'a',''" or "if (...) {..} else {..}" directly
        # inside a hash literal: PowerShell 5.1 reads the comma as an element
        # separator and raises ParserError "The hash literal was incomplete".
        $logTime = $null
        $gapH    = $null
        $status  = "NO LOG"

        if ($logPath -and (Test-Path $logPath)) {
            $logTime = (Get-Item $logPath).LastWriteTime
            $gapH    = [math]::Round((New-TimeSpan -Start $logTime -End $ckptTime).TotalHours, 1)

            # Testing always runs AFTER training finishes, so the log has to be
            # newer than the checkpoint. A log older than the checkpoint means it
            # was produced from a model that has since been overwritten.
            # A 1-hour margin absorbs clock skew and file copying.
            if ($ckptTime -gt $logTime.AddHours(1)) { $status = "STALE LOG" }
            else                                    { $status = "ok" }

            # Old format (no HD95_vox=) makes aggregate_results.py skip per-patient rows
            $hasVox = Select-String -Path $logPath -Pattern "HD95_vox=" -Quiet
            if (-not $hasVox) { $status = "OLD FORMAT" }
        }

        $cfgShort = $cfgDir -replace '^TU_', ''

        $seed = '?'
        if ($runName -match '_s(\d+)$') { $seed = $Matches[1] }

        $obj = New-Object psobject
        $obj | Add-Member NoteProperty Status     $status
        $obj | Add-Member NoteProperty Config     $cfgShort
        $obj | Add-Member NoteProperty Seed       $seed
        $obj | Add-Member NoteProperty CkptTime   $ckptTime
        $obj | Add-Member NoteProperty LogTime    $logTime
        $obj | Add-Member NoteProperty CkptNewerH $gapH
        $obj | Add-Member NoteProperty Run        $runName
        $obj | Add-Member NoteProperty LogPath    $logPath
        $rows.Add($obj)
    }
}

if ($rows.Count -eq 0) { Write-Warning "No checkpoint found."; exit 0 }

# @() keeps .Count correct even when there are only 0 or 1 elements
$bad = @($rows | Where-Object { $_.Status -ne "ok" })

Write-Host "=================== NEEDS ATTENTION ===================" -ForegroundColor Yellow
if ($bad.Count -eq 0) {
    Write-Host "  No abnormal runs." -ForegroundColor Green
} else {
    $bad | Sort-Object Status, Config, Seed |
        Format-Table Status, Config, Seed, CkptTime, LogTime, CkptNewerH -AutoSize
}

Write-Host ""
Write-Host "=================== SUMMARY ===================" -ForegroundColor Cyan
$rows | Group-Object Status | Sort-Object Name | ForEach-Object {
    "{0,-12} {1,3}" -f $_.Name, $_.Count
}

Write-Host "Total runs (bs24 excluded): $($rows.Count)"
Write-Host ""
Write-Host "MEANING:"
Write-Host "  ok          - test log is newer than the checkpoint; the numbers can be trusted"
Write-Host "  STALE LOG   - checkpoint NEWER than the log => the log came from a model that has been overwritten. THE TEST MUST BE RE-RUN."
Write-Host "  OLD FORMAT  - log has no HD95_vox field => aggregate_results.py skips per-patient rows. THE TEST MUST BE RE-RUN."
Write-Host "  NO LOG      - trained but not tested yet."
Write-Host ""

if ($Csv) {
    $rows | Sort-Object Status, Config, Seed | Export-Csv -Path $Csv -NoTypeInformation -Encoding UTF8
    Write-Host "Wrote: $Csv"
}
