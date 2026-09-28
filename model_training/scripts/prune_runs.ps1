param(
    [switch] $Apply
)

$ErrorActionPreference = 'Stop'
$runRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\runs')).Path.TrimEnd('\')
$modelExtensions = @(
    '.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.engine',
    '.tflite', '.torchscript', '.mlmodel', '.pb', '.h5', '.keras',
    '.bin', '.param', '.xml'
)
$imageExtensions = @('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')
$stagingNames = @('prepared_dataset', 'debug_images', 'eval_preds', 'bench_images', 'preview_images')
$protectedRunNames = @('dfine-seg-s-segment-20260928-134825')
$planned = [System.Collections.Generic.List[object]]::new()
$protectedRuns = [System.Collections.Generic.List[string]]::new()

function Assert-InRunRoot([string] $Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($runRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to touch a path outside the run root: $full"
    }
    if ($full -eq $runRoot) {
        throw 'Refusing to touch the run root itself.'
    }
}

function Get-RunFiles([string] $Path) {
    @(Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction Stop)
}

function Test-ModelFile([IO.FileInfo] $File) {
    $modelExtensions -contains $File.Extension.ToLowerInvariant()
}

function Test-ActiveRun([string] $Path) {
    $processes = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction Stop)
    foreach ($process in $processes) {
        if ($process.CommandLine -and
            $process.CommandLine.IndexOf($Path, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }
    return $false
}

function Add-Plan([string] $RunName, [string] $Path, [bool] $IsDirectory) {
    Assert-InRunRoot $Path
    if (-not (Test-Path -LiteralPath $Path)) { return }
    if ($IsDirectory) {
        $files = Get-RunFiles $Path
        if (@($files | Where-Object { Test-ModelFile $_ }).Count -ne 0) {
            throw "A proposed deletion contains a model file: $Path"
        }
        $bytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
    } else {
        $file = Get-Item -LiteralPath $Path -Force
        if (Test-ModelFile $file) { throw "A proposed deletion is a model file: $Path" }
        $bytes = [long]$file.Length
    }
    $planned.Add([pscustomobject]@{
        Run = $RunName
        Path = [IO.Path]::GetFullPath($Path)
        Directory = $IsDirectory
        Bytes = $bytes
    })
}

$runs = [System.Collections.Generic.List[object]]::new()
foreach ($family in @('dfine', 'dfine-seg', 'yolo26')) {
    $familyPath = Join-Path $runRoot $family
    if (-not (Test-Path -LiteralPath $familyPath)) { continue }
    foreach ($run in (Get-ChildItem -LiteralPath $familyPath -Directory -Force)) {
        if ($run.Name -ne '_dataset_configs') { $runs.Add($run) }
    }
}
$smokeDataset = Join-Path $runRoot '_dfine_seg_smoke_dataset'
if (Test-Path -LiteralPath $smokeDataset) {
    $runs.Add((Get-Item -LiteralPath $smokeDataset))
}

foreach ($run in $runs) {
    Assert-InRunRoot $run.FullName
    if ($protectedRunNames -contains $run.Name -or (Test-ActiveRun $run.FullName)) {
        $protectedRuns.Add($run.FullName)
        continue
    }
    $files = Get-RunFiles $run.FullName
    $models = @($files | Where-Object { Test-ModelFile $_ })
    if ($models.Count -eq 0) {
        Add-Plan $run.Name $run.FullName $true
        continue
    }

    foreach ($folderName in $stagingNames) {
        Add-Plan $run.Name (Join-Path $run.FullName $folderName) $true
    }
    $testEvaluation = Join-Path $run.FullName 'test_evaluation'
    if (Test-Path -LiteralPath $testEvaluation) {
        foreach ($evaluation in (Get-ChildItem -LiteralPath $testEvaluation -Directory -Force)) {
            foreach ($folderName in $stagingNames) {
                Add-Plan $run.Name (Join-Path $evaluation.FullName $folderName) $true
            }
        }
    }

    foreach ($file in $files) {
        if ($imageExtensions -notcontains $file.Extension.ToLowerInvariant()) { continue }
        $relative = $file.FullName.Substring($run.FullName.Length + 1)
        if ($relative -match '^(model|weights|exports)[\\/]') { continue }
        if ($planned | Where-Object {
            $_.Directory -and $file.FullName.StartsWith($_.Path + '\', [StringComparison]::OrdinalIgnoreCase)
        }) { continue }
        Add-Plan $run.Name $file.FullName $false
    }
}

$totalBytes = [long](($planned | Measure-Object -Property Bytes -Sum).Sum)
$wholeRuns = @($planned | Where-Object {
    $_.Directory -and $_.Run -eq (Split-Path -Leaf $_.Path)
})
Write-Output "Run root: $runRoot"
Write-Output "Protected runs kept intact: $($protectedRuns.Count)"
$protectedRuns | ForEach-Object { Write-Output "  $_" }
Write-Output "Folders with no model files to remove: $($wholeRuns.Count)"
Write-Output "Planned deletions: $($planned.Count); nominal size: $([math]::Round($totalBytes / 1GB, 2)) GiB"
$planned | Group-Object Run | Sort-Object Name | ForEach-Object {
    $size = [math]::Round((($_.Group | Measure-Object -Property Bytes -Sum).Sum) / 1MB, 1)
    Write-Output "  $($_.Name): $($_.Count) paths, $size MiB"
}
Write-Output 'Model files, exports, run configs, metrics, and logs are retained.'

if (-not $Apply) {
    Write-Output 'Dry run only. Pass -Apply to execute this reviewed plan.'
    return
}

foreach ($item in $planned) {
    Assert-InRunRoot $item.Path
    $run = $runs | Where-Object { $_.Name -eq $item.Run } | Select-Object -First 1
    if (Test-ActiveRun $run.FullName) {
        Write-Warning "Run became active; skipping $($run.FullName)"
        continue
    }
    if (-not (Test-Path -LiteralPath $item.Path)) { continue }
    if ($item.Directory) {
        $currentFiles = Get-RunFiles $item.Path
        if (@($currentFiles | Where-Object { Test-ModelFile $_ }).Count -ne 0) {
            throw "Model file appeared in deletion target; stopping: $($item.Path)"
        }
        Remove-Item -LiteralPath $item.Path -Recurse -Force
    } else {
        $current = Get-Item -LiteralPath $item.Path -Force
        if (Test-ModelFile $current) { throw "Model file appeared at target: $($item.Path)" }
        Remove-Item -LiteralPath $item.Path -Force
    }
}
Write-Output 'Run cleanup complete.'
