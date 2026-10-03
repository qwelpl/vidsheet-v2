<#
vidsheet - local Synthesia-to-sheet-music tool (Windows launcher).

  vidsheet                       start the app, open it in your browser
  vidsheet <url|file>            start the app attached to that source + analyze
  vidsheet analyze <url|file>    headless: write MIDI/CSV/MusicXML to --out
  vidsheet -h | --help

Flags: --preset fast|balanced|maximum  --out DIR (analyze)
       --engine-port N  --web-port N  --no-open

Everything runs locally, so YouTube downloads use your own residential IP and
dodge the datacenter-IP block that cloud hosts hit.
#>
$ErrorActionPreference = 'Stop'

# SRC = pristine source. Scoop sets VIDSHEET_SRC to its (shared) install dir;
# a git clone just uses this script's own directory.
$SRC = if ($env:VIDSHEET_SRC) { $env:VIDSHEET_SRC } else { $PSScriptRoot }

# WORK = where we actually run. A packaged install (VIDSHEET_SRC set) must not
# write into its install dir, so mirror into a per-user data dir and build the
# venv / node_modules there. A clone works in place.
$DATA = if ($env:VIDSHEET_HOME) { $env:VIDSHEET_HOME } else { Join-Path $env:LOCALAPPDATA 'vidsheet' }
$WORK = if ($env:VIDSHEET_SRC) { $DATA } else { $SRC }

$EnginePort = if ($env:ENGINE_PORT) { $env:ENGINE_PORT } else { '8000' }
$WebPort    = if ($env:WEB_PORT)    { $env:WEB_PORT }    else { '3000' }
$Preset = 'balanced'
$Out = 'out'
$Open = $true

function Show-Usage {
    $inHelp = $false
    foreach ($line in Get-Content -LiteralPath $PSCommandPath) {
        if ($line -eq '<#') { $inHelp = $true; continue }
        if ($line -eq '#>') { break }
        if ($inHelp) { $line }
    }
}

function Test-Url($s) { return $s -match '^https?://' }

function Sync-Source {
    if ($WORK -eq $SRC) { return }
    New-Item -ItemType Directory -Force -Path $WORK | Out-Null
    $stamp = Join-Path $WORK '.src'
    if ((Test-Path $stamp) -and ((Get-Content -Raw $stamp).Trim() -eq $SRC)) { return }
    Write-Host "Preparing vidsheet in $WORK ..."
    # robocopy mirrors, skipping the generated / heavy dirs. Exit codes 0-7 are
    # success for robocopy; anything >=8 is a real error.
    $rc = Start-Process robocopy -ArgumentList @(
        "`"$SRC`"", "`"$WORK`"", '/MIR',
        '/XD', '.git', '.venv', 'node_modules', '.next',
        '/NFL', '/NDL', '/NJH', '/NJS', '/NP'
    ) -NoNewWindow -Wait -PassThru
    if ($rc.ExitCode -ge 8) { throw "robocopy failed ($($rc.ExitCode))" }
    Set-Content -LiteralPath $stamp -Value $SRC
}

function Test-Prereqs {
    $missing = @()
    function need($name, $hint) {
        if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
            Write-Host "  [x] $name - $hint"; $script:missing += $name
        }
    }
    need python 'install Python 3.12+ (scoop install python)'
    need node   'install Node.js (scoop install nodejs) - also yt-dlp''s JS runtime'
    need npm    'ships with Node.js'
    need ffmpeg 'scoop install ffmpeg'
    need yt-dlp 'scoop install yt-dlp'
    if ($missing.Count -gt 0) {
        Write-Error 'Install the missing tools above, then re-run vidsheet.'
        exit 1
    }
}

function Get-VenvPython { Join-Path $WORK 'engine\.venv\Scripts\python.exe' }

function Initialize-Engine {
    $venv = Get-VenvPython
    if (-not (Test-Path $venv)) {
        Write-Host 'Setting up engine venv...'
        Push-Location (Join-Path $WORK 'engine')
        try {
            & python -m venv .venv
            & $venv -m pip install -q -r requirements.txt yt-dlp
        } finally { Pop-Location }
    }
}

function Initialize-Web {
    if (-not (Test-Path (Join-Path $WORK 'web\node_modules'))) {
        Write-Host 'Installing web deps...'
        Push-Location (Join-Path $WORK 'web')
        try { & npm.cmd install } finally { Pop-Location }
    }
}

# Create a job on the running engine, return its id. $src = url|file.
function New-Job($src) {
    $base = "http://localhost:$EnginePort"
    if (Test-Url $src) {
        $json = & curl.exe -fsS -X POST "$base/api/analyze/youtube" -F "url=$src" -F "preset=$Preset"
    } else {
        if (-not (Test-Path $src)) { throw "No such file: $src" }
        $json = & curl.exe -fsS -X POST "$base/api/analyze/upload" -F "file=@$src" -F "preset=$Preset"
    }
    return ($json | ConvertFrom-Json).id
}

function Open-Browser($url) {
    if ($Open) { Start-Process $url }
}

function Invoke-Launch($src) {
    Test-Prereqs
    Sync-Source
    Initialize-Engine
    Initialize-Web
    $venv = Get-VenvPython
    Write-Host "Starting engine on :$EnginePort and web UI on :$WebPort ..."
    $api = Start-Process -FilePath $venv `
        -ArgumentList @('-m', 'uvicorn', 'server:app', '--port', $EnginePort) `
        -WorkingDirectory (Join-Path $WORK 'engine') -NoNewWindow -PassThru
    $env:NEXT_PUBLIC_API = "http://localhost:$EnginePort"
    $web = Start-Process -FilePath 'npm.cmd' `
        -ArgumentList @('run', 'dev', '--', '--port', $WebPort) `
        -WorkingDirectory (Join-Path $WORK 'web') -NoNewWindow -PassThru
    try {
        $ready = $false
        foreach ($i in 1..60) {
            try {
                Invoke-WebRequest -UseBasicParsing "http://localhost:$EnginePort/api/health" -TimeoutSec 2 | Out-Null
                $ready = $true; break
            } catch { Start-Sleep -Seconds 1 }
        }
        if (-not $ready) { throw 'Engine did not come up.' }

        $url = "http://localhost:$WebPort"
        if ($src) {
            Write-Host "Submitting $src ..."
            $id = New-Job $src
            $url = "$url/?job=$id"
            Write-Host "Job $id started."
        }
        Write-Host "Opening $url"
        Open-Browser $url
        Write-Host 'Press Ctrl-C to stop.'
        Wait-Process -Id $api.Id, $web.Id
    } finally {
        foreach ($p in @($api, $web)) {
            if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        }
    }
}

function Invoke-Analyze($src) {
    if (-not $src) { Write-Error 'usage: vidsheet analyze <url|file>'; exit 1 }
    Test-Prereqs
    Sync-Source
    Initialize-Engine
    $venv = Get-VenvPython
    Push-Location (Join-Path $WORK 'engine')
    try { & $venv cli.py analyze $src --preset $Preset --out $Out }
    finally { Pop-Location }
}

# --- arg parsing -----------------------------------------------------------
$first = $null; $second = $null; $pcount = 0
$i = 0
while ($i -lt $args.Count) {
    $a = $args[$i]
    switch -Regex ($a) {
        '^--preset$'      { $Preset = $args[++$i] }
        '^--preset=(.+)$' { $Preset = $Matches[1] }
        '^--out$'         { $Out = $args[++$i] }
        '^--out=(.+)$'    { $Out = $Matches[1] }
        '^--engine-port$' { $EnginePort = $args[++$i] }
        '^--web-port$'    { $WebPort = $args[++$i] }
        '^--no-open$'     { $Open = $false }
        '^(-h|--help)$'   { Show-Usage; exit 0 }
        '^-'              { Write-Error "unknown flag: $a"; Show-Usage; exit 1 }
        default {
            if ($pcount -eq 0) { $first = $a } else { $second = $a }
            $pcount++
        }
    }
    $i++
}

switch ($first) {
    $null      { Invoke-Launch $null }
    'analyze'  { Invoke-Analyze $second }
    'help'     { Show-Usage }
    default    { Invoke-Launch $first }   # positional treated as a source
}
