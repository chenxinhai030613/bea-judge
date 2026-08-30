param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalPath,
    [Parameter(Mandatory = $true)]
    [string]$UpdatedPath
)

$Original = (Resolve-Path -LiteralPath $OriginalPath).Path
$Updated = (Resolve-Path -LiteralPath $UpdatedPath).Path
$Directory = Split-Path -Parent $Original
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($Original)
$Backup = Join-Path $Directory ($BaseName + "_pre_1000_update_backup.docx")

try {
    $Word = [Runtime.InteropServices.Marshal]::GetActiveObject("Word.Application")
    foreach ($OpenDoc in @($Word.Documents)) {
        if ($OpenDoc.FullName -ieq $Original) {
            $OpenDoc.Close(0)
            break
        }
    }
}
catch {
    # Word is not running or cannot be reached; continue with filesystem replace.
}

Copy-Item -LiteralPath $Original -Destination $Backup -Force
Copy-Item -LiteralPath $Updated -Destination $Original -Force

Write-Output "original=$Original"
Write-Output "updated=$Updated"
Write-Output "backup=$Backup"
