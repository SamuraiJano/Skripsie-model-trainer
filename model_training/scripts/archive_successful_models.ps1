param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'

$trainingRoot = Split-Path -Parent $PSScriptRoot
$runsRoot = Join-Path $trainingRoot 'runs'
$archiveRoot = Join-Path $trainingRoot 'models\completed'
$modelExtensions = @('.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.tflite', '.engine', '.torchscript', '.h5', '.keras', '.pb')
$completedRuns = @()

foreach ($family in @('dfine', 'dfine-seg', 'yolo26')) {
    $familyRoot = Join-Path $runsRoot $family
    if (-not (Test-Path -LiteralPath $familyRoot)) { continue }

    foreach ($run in Get-ChildItem -LiteralPath $familyRoot -Directory) {
        if ($run.Name -match 'smoke') { continue }

        $completionMarker = $null
        if ($family -eq 'dfine') {
            $metadataPath = Join-Path $run.FullName 'run_metadata.json'
            $logPath = Join-Path $run.FullName 'log.txt'
            if (-not ((Test-Path -LiteralPath $metadataPath) -and (Test-Path -LiteralPath $logPath))) { continue }
            $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
            $targetEpochs = [int]$metadata.options.epochs
            try { $lastEpoch = [int]((Get-Content -LiteralPath $logPath -Tail 1 | ConvertFrom-Json).epoch) }
            catch { continue }
            if ($targetEpochs -lt 1 -or $lastEpoch -lt ($targetEpochs - 1)) { continue }
            $completionMarker = Get-Item -LiteralPath $logPath
        }
        elseif ($family -eq 'dfine-seg') {
            $trainLogPath = Join-Path $run.FullName 'model\train_log.txt'
            $testMetricsPath = Join-Path $run.FullName 'test_metrics.json'
            if (-not ((Test-Path -LiteralPath $trainLogPath) -and (Test-Path -LiteralPath $testMetricsPath))) { continue }
            if (-not (Select-String -LiteralPath $trainLogPath -Pattern 'Full training time:' -Quiet)) { continue }
            $completionMarker = Get-Item -LiteralPath $trainLogPath
        }
        else {
            $metricsPath = Join-Path $run.FullName 'final_metrics.json'
            if (-not (Test-Path -LiteralPath $metricsPath)) { continue }
            $completionMarker = Get-Item -LiteralPath $metricsPath
        }

        $files = @(Get-ChildItem -LiteralPath $run.FullName -Recurse -File | Where-Object {
            $relative = $_.FullName.Substring($run.FullName.Length).TrimStart('\')
            $excluded = $relative -match '^(prepared_dataset|eval_preds|preview_images|debug_images|images|labels)\\' -or
                $relative -match '\\(prepared_dataset|eval_preds|preview_images|debug_images|images|labels)\\'
            $isGraph = ($_.Extension -in @('.png', '.jpg', '.jpeg', '.svg', '.pdf')) -and (
                $relative -match '(^|\\)plots\\' -or
                (($relative -notmatch '\\') -and $_.Name -match '^(results\.|labels\.|confusion_matrix|.*_curve\.)')
            )
            (-not $excluded) -and (
                ($modelExtensions -contains $_.Extension.ToLowerInvariant()) -or
                $relative.StartsWith('exports\', [StringComparison]::OrdinalIgnoreCase) -or
                ($_.Extension -in @('.json', '.jsonl', '.csv', '.txt', '.log')) -or
                ($_.Name -like 'events.out.tfevents*') -or
                $isGraph -or
                ($relative -in @('model\config.yaml', 'model\config.yml')) -or
                (($relative -notmatch '\\') -and ($_.Extension -in @('.yml', '.yaml')))
            )
        })
        if ($files.Count -eq 0) { continue }

        $completionDate = $completionMarker.LastWriteTime.ToString('yyyy-MM-dd')
        $destinationRoot = Join-Path $archiveRoot (Join-Path $completionDate (Join-Path $family $run.Name))
        $completedRuns += [pscustomobject]@{
            Family = $family
            Run = $run.Name
            Completed = $completionMarker.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
            Destination = $destinationRoot
            Files = $files
            Source = $run.FullName
        }
    }
}

if (-not $Apply) {
    $completedRuns | ForEach-Object {
        [pscustomobject]@{ Completed = $_.Completed; Family = $_.Family; Run = $_.Run; Files = $_.Files.Count; Destination = $_.Destination }
    } | Format-Table -AutoSize
    Write-Output "Dry run: $($completedRuns.Count) completed runs. Use -Apply to copy and verify the files."
    return
}

$manifest = @()
foreach ($completed in $completedRuns) {
    $manifestFiles = @()
    foreach ($sourceFile in $completed.Files) {
        $relative = $sourceFile.FullName.Substring($completed.Source.Length).TrimStart('\')
        $destination = Join-Path $completed.Destination $relative
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null

        $sourceHash = (Get-FileHash -LiteralPath $sourceFile.FullName -Algorithm SHA256).Hash
        if (-not (Test-Path -LiteralPath $destination)) {
            Copy-Item -LiteralPath $sourceFile.FullName -Destination $destination -ErrorAction Stop
        }
        $destinationHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($sourceHash -ne $destinationHash) {
            throw "Archive verification failed; existing file differs from source: $destination"
        }

        $manifestFiles += [pscustomobject]@{
            RelativePath = $relative
            Bytes = $sourceFile.Length
            SHA256 = $sourceHash
        }
    }

    $manifest += [pscustomobject]@{
        Family = $completed.Family
        Run = $completed.Run
        Completed = $completed.Completed
        Destination = $completed.Destination
        Files = $manifestFiles
    }
    Write-Output "Verified $($manifestFiles.Count) files: $($completed.Run)"
}

$manifestPath = Join-Path $archiveRoot 'manifest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output "Archive complete: $($manifest.Count) runs. Manifest: $manifestPath"
