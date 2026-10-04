# Export the dasanyuan database to one file you can send to a teammate (Windows).
#
#   powershell -ExecutionPolicy Bypass -File db\export_db.ps1
#   powershell -ExecutionPolicy Bypass -File db\export_db.ps1 -HandsOnly     # small file, no decisions table
#
# Writes dasanyuan.dump in the repo root. The file works on Mac, Windows and Linux;
# the teammate loads it with db/import_db.sh (Mac) or db\import_db.ps1 (Windows).

param(
    [string]$Out = "dasanyuan.dump",
    [string]$Database = "dasanyuan",
    [string]$User = "dsy",
    [switch]$HandsOnly
)

# Find pg_dump: on PATH, or in the newest PostgreSQL install folder.
$pgDump = (Get-Command pg_dump -ErrorAction SilentlyContinue).Source
if (-not $pgDump) {
    $pgDump = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\pg_dump.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [int]$_.Directory.Parent.Name } -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $pgDump) { Write-Error "pg_dump not found. Is PostgreSQL installed?"; exit 1 }

if (-not $env:PGPASSWORD) { $env:PGPASSWORD = "dsy" }

$dumpArgs = @("-h", "localhost", "-U", $User, "-d", $Database,
          "--format=custom", "--no-owner", "--no-privileges", "--file=$Out")
if ($HandsOnly) { $dumpArgs += "--exclude-table-data=decisions" }

Write-Host "Using $pgDump"
& $pgDump @dumpArgs
if ($LASTEXITCODE -ne 0) { Write-Error "pg_dump failed"; exit 1 }

$size = "{0:N0} MB" -f ((Get-Item $Out).Length / 1MB)
Write-Host "Wrote $Out ($size). Send it to your teammate (Google Drive, OneDrive, Teams...)."
