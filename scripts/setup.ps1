$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    Push-Location $projectRoot
    try {
        python -m venv .venv
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Could not create .venv. Install Python 3.11 or newer and retry.'
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
& $venvPython -m ipykernel install --user --name returnwise --display-name 'Python (ReturnWise)'
Write-Host 'ReturnWise environment and Jupyter kernel are ready.'
