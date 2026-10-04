# Load a dasanyuan.dump from a teammate (Windows).
#
#   powershell -ExecutionPolicy Bypass -File db\import_db.ps1 path\to\dasanyuan.dump
#
# Creates the dsy user (password dsy) and the dasanyuan database if they are missing,
# then loads the dump. Asks once for the "postgres" password you chose when installing.

param([string]$Dump = "dasanyuan.dump")

if (-not (Test-Path $Dump)) { Write-Error "File not found: $Dump"; exit 1 }

$bin = Split-Path (Get-Command psql -ErrorAction SilentlyContinue).Source -ErrorAction SilentlyContinue
if (-not $bin) {
    $bin = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [int]$_.Directory.Parent.Name } -Descending | Select-Object -First 1 -ExpandProperty DirectoryName
}
if (-not $bin) { Write-Error "PostgreSQL not found."; exit 1 }
$psql = Join-Path $bin "psql.exe"; $restore = Join-Path $bin "pg_restore.exe"

$sec = Read-Host "Password for the postgres admin user" -AsSecureString
$env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
& $psql -h localhost -U postgres -d postgres -q -c "DO `$`$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='dsy') THEN CREATE ROLE dsy LOGIN PASSWORD 'dsy'; END IF; END `$`$;"
$exists = & $psql -h localhost -U postgres -d postgres -Atc "select 1 from pg_database where datname='dasanyuan'"
if ($exists -ne "1") { & $psql -h localhost -U postgres -d postgres -q -c "CREATE DATABASE dasanyuan OWNER dsy" }

$env:PGPASSWORD = "dsy"
Write-Host "Loading $Dump into dasanyuan..."
& $restore -h localhost -U dsy -d dasanyuan --clean --if-exists --no-owner --no-privileges $Dump
& $psql -h localhost -U dsy -d dasanyuan -c "select r.name as run, count(*) as hands from runs r join hands h using (run_id) group by r.name order by r.name"
