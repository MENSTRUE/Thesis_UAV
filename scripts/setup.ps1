# Setup Windows (PowerShell). Jalankan dari root proyek.
#   .\scripts\setup.ps1            -> default: GPU nVidia (CUDA)
#   .\scripts\setup.ps1 -CPU       -> ONNX Runtime CPU saja
#   .\scripts\setup.ps1 -DML       -> ONNX Runtime DirectML (DX12 / GPU apa pun)
#   .\scripts\setup.ps1 -Pip       -> pakai pip, bukan uv
param(
    [switch]$CPU,
    [switch]$DML,
    [switch]$Pip
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot\..

if (-not (Test-Path .python-version)) {
    throw ".python-version tidak ditemukan"
}

# 1. Install uv (kecuali -Pip)
if (-not $Pip) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "[uv] belum ada, menginstall..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    $python = "3.11"
} else {
    $python = "3.11"
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "python 3.11 belum terpasang"
    }
}

# 2. Create venv
if ($Pip) {
    python -m venv .venv
    $venv = ".venv\Scripts"
} else {
    uv venv --python $python .venv
    $venv = ".venv\Scripts"
}

# 3. Base dependencies (HAR + Tello, tanpa ONNX Runtime)
Write-Host "[deps] Base (HAR + Tello)..."
if ($Pip) {
    & $venv\python -m pip install --upgrade pip
    & $venv\python -m pip install -r requirements\requirements-har.txt
} else {
    & uv pip install -p .venv -r requirements\requirements-har.txt
}

# 4. GPU / CPU / DML specific (ONNX Runtime variant + face + torch)
$torchCu124 = "https://download.pytorch.org/whl/cu124"
if ($DML) {
    Write-Host "[deps] DirectML..."
    if ($Pip) {
        & $venv\python -m pip install -r requirements\requirements-dml.txt -r requirements\requirements-full.txt
    } else {
        uv pip install -p .venv -r requirements\requirements-dml.txt -r requirements\requirements-full.txt
    }
} elseif ($CPU) {
    Write-Host "[deps] CPU..."
    if ($Pip) {
        & $venv\python -m pip install torch torchvision --index-url $torchCu124.Replace("cu124","cpu")
        & $venv\python -m pip install -r requirements\requirements-full.txt -r requirements\requirements-cpu.txt
    } else {
        uv pip install -p .venv torch torchvision --index-url $torchCu124.Replace("cu124","cpu")
        uv pip install -p .venv -r requirements\requirements-full.txt -r requirements\requirements-cpu.txt
    }
} else {
    Write-Host "[deps] GPU (nvidia) - torch cu124 dulu..."
    if ($Pip) {
        & $venv\python -m pip install torch torchvision --index-url $torchCu124
        & $venv\python -m pip install -r requirements\requirements-full.txt -r requirements\requirements-gpu.txt
    } else {
        uv pip install -p .venv torch torchvision --index-url $torchCu124
        uv pip install -p .venv -r requirements\requirements-full.txt -r requirements\requirements-gpu.txt
    }
}

Write-Host ""
Write-Host "Selesai. Jalankan:"
Write-Host "  $venv\python main.py"